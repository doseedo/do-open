import { useState, useCallback, useRef } from 'react';
import { useApp } from '../context/AppContext';
import * as generationAPI from '../services/generationAPI';
import { parseMIDI } from '../utils/midiParser';

/**
 * Try to fetch MIDI file from multiple possible paths
 * @returns {Promise<{path: string, arrayBuffer: ArrayBuffer}|null>}
 */
async function tryFetchMidi(voiceInputFiles, params) {
  // Try explicit midi_path first
  if (voiceInputFiles.midi_path) {
    try {
      const response = await fetch(voiceInputFiles.midi_path);
      if (response.ok) {
        return { path: voiceInputFiles.midi_path, arrayBuffer: await response.arrayBuffer() };
      }
    } catch (e) {
      console.warn(`⚠️ Failed to fetch explicit midi_path: ${voiceInputFiles.midi_path}`);
    }
  }

  // If no explicit path or it failed, try to construct from render_path
  if (!voiceInputFiles.render_path || !params.enableMidiExport) {
    return null;
  }

  const basePath = voiceInputFiles.render_path.replace(/\.(wav|mp3)$/i, '');
  const possiblePaths = [
    `${basePath}.mid`,                          // /download/task_id/0_input.mid
    `${basePath}.midi`,                         // /download/task_id/0_input.midi
    basePath.replace(/_input$/i, '.mid'),       // /download/task_id/0.mid
    basePath.replace(/_input$/i, '.midi'),      // /download/task_id/0.midi
  ];

  console.log(`🔍 Trying to find MIDI file, possible paths:`, possiblePaths);

  // Try each possible path
  for (const path of possiblePaths) {
    try {
      const response = await fetch(path);
      if (response.ok) {
        console.log(`✅ Found MIDI file at: ${path}`);
        return { path, arrayBuffer: await response.arrayBuffer() };
      }
    } catch (e) {
      // Continue to next path
    }
  }

  console.warn(`⚠️ Could not find MIDI file at any of the tried paths`);
  return null;
}

/**
 * Custom hook for managing audio generation lifecycle
 * Handles: starting generation, polling, progress tracking, and adding results to timeline
 */
export function useGeneration() {
  const { state, dispatch } = useApp();
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationProgress, setGenerationProgress] = useState(null);
  const [generationError, setGenerationError] = useState(null);
  const [elapsedTime, setElapsedTime] = useState(0);

  const timerRef = useRef(null);
  const startTimeRef = useRef(null);
  const addedTracksRef = useRef(new Set()); // Track URLs added during current generation

  /**
   * Start generation timer
   */
  const startTimer = useCallback(() => {
    startTimeRef.current = Date.now();
    setElapsedTime(0);

    if (timerRef.current) {
      clearInterval(timerRef.current);
    }

    timerRef.current = setInterval(() => {
      if (startTimeRef.current) {
        const elapsed = Math.floor((Date.now() - startTimeRef.current) / 1000);
        setElapsedTime(elapsed);
      }
    }, 1000);
  }, []);

  /**
   * Stop generation timer
   */
  const stopTimer = useCallback(() => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    startTimeRef.current = null;
  }, []);

  /**
   * Format elapsed time as MM:SS
   */
  const formatElapsedTime = useCallback(() => {
    const minutes = Math.floor(elapsedTime / 60);
    const seconds = elapsedTime % 60;
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  }, [elapsedTime]);

  /**
   * Add a track to the timeline, or replace sourceTrack if specified
   */
  const addTrackToTimeline = useCallback((busId, track, sourceTrack = null, forceAdd = false) => {
    // Check if we should replace an existing track
    const actualSourceTrack = sourceTrack || state.sourceTrack;

    // For composite multi-track MIDI, replace the corresponding MIDI track
    let targetTrackId = null;
    if (actualSourceTrack?.isComposite && actualSourceTrack?.midiTracks) {
      const voiceNumber = track.metadata?.voiceNumber;
      // Voice 1 -> midiTracks[0], Voice 2 -> midiTracks[1], etc.
      if (voiceNumber && voiceNumber >= 1 && voiceNumber <= actualSourceTrack.midiTracks.length) {
        targetTrackId = actualSourceTrack.midiTracks[voiceNumber - 1].id;
        console.log(`🔄 Voice ${voiceNumber} will replace MIDI track: ${actualSourceTrack.midiTracks[voiceNumber - 1].name} (${targetTrackId})`);
      }
    }

    // Determine if we should replace
    const shouldReplace = targetTrackId || (actualSourceTrack && actualSourceTrack.trackId && !forceAdd && track.metadata?.voiceNumber === 0);

    if (shouldReplace) {
      const replaceTrackId = targetTrackId || actualSourceTrack.trackId;
      console.log(`🔄 Replacing track with generated audio:`, {
        targetTrackId: replaceTrackId,
        newTrackName: track.name
      });

      // Find the actual MIDI track to get its data
      let midiTrackData = null;
      let sourceBusId = busId;

      if (targetTrackId) {
        // Find the MIDI track in the bus
        const bus = state.buses.find(b => b.id === actualSourceTrack.busId);
        const midiTrack = bus?.tracks.find(t => t.id === targetTrackId);
        if (midiTrack) {
          midiTrackData = midiTrack.midiData;
          sourceBusId = actualSourceTrack.busId;
          console.log(`📋 Found MIDI track data for replacement:`, { trackId: targetTrackId, notesCount: midiTrackData?.notes?.length });
        }
      } else {
        midiTrackData = actualSourceTrack.midiData;
        sourceBusId = actualSourceTrack.busId;
      }

      // Add MIDI data to the new track's metadata
      // IMPORTANT: Remove type: 'midi' so it's treated as audio track!
      const { type: _unusedType, ...trackWithoutType } = track;
      const trackWithMidi = {
        ...trackWithoutType,
        type: undefined, // Ensure it's NOT a MIDI track
        metadata: {
          ...track.metadata,
          midiData: midiTrackData, // Preserve original MIDI data for editing
          sourceName: targetTrackId ? actualSourceTrack.midiTracks.find(t => t.id === targetTrackId)?.name : actualSourceTrack.name
        }
      };

      // Use the source track's bus instead of creating new one
      dispatch({
        type: 'REPLACE_TRACK',
        payload: {
          busId: sourceBusId,
          trackId: replaceTrackId,
          newTrack: trackWithMidi
        }
      });

      console.log('✅ REPLACE_TRACK dispatched');
    } else {
      console.log(`➕ Adding track to timeline:`, { busId, trackName: track.name, trackUrl: track.audioUrl });
      console.log(`📊 Full track object:`, track);
      dispatch({
        type: 'ADD_TRACK',
        payload: { busId, track }
      });
      console.log('✅ ADD_TRACK dispatched, React should re-render now');
    }
  }, [dispatch, state.sourceTrack]);

  /**
   * Create a new bus for generation results
   */
  const createBusForGeneration = useCallback((type) => {
    const timestamp = Date.now();
    const busId = `${type.toLowerCase()}-${timestamp}`;
    console.log(`🆕 Creating new bus:`, busId);

    dispatch({
      type: 'CREATE_BUS',
      payload: {
        id: busId,  // Pass the busId so reducer uses it
        type,
        name: `${type} ${state.buses.filter(b => b.type === type).length + 1}`
      }
    });

    return busId;
  }, [dispatch, state.buses]);

  /**
   * Create a new bus for each generation
   */
  const getBusForGeneration = useCallback((type) => {
    // Always create a new bus for each generation
    return createBusForGeneration(type);
  }, [createBusForGeneration]);

  /**
   * Handle partial result (incremental track)
   */
  const handlePartialResult = useCallback(async (partialData, busId, params) => {
    console.log('📥 Handling partial result:', partialData);

    const { completed_voices, input_files } = partialData;

    if (!completed_voices || completed_voices.length === 0) {
      return;
    }

    // Process each completed voice
    for (const voiceFile of completed_voices) {
      // Skip if already added (check ref, not state)
      if (addedTracksRef.current.has(voiceFile)) {
        console.log(`⏭️  Skipping duplicate track: ${voiceFile}`);
        continue;
      }

      // Extract voice number from path (e.g., "/download/task_id/1.wav" -> 1)
      const match = voiceFile.match(/\/(\d+)\.wav$/);
      if (!match) {
        console.error(`❌ Could not extract voice number from: ${voiceFile}`);
        continue;
      }

      const voiceNumber = parseInt(match[1], 10);

      // Process all voices including voice 0 (for non-monophonic tracks)
      console.log(`✅ Processing voice ${voiceNumber}: ${voiceFile}`);

      // Extract input files for this voice from backend response
      const voiceInputFiles = input_files?.[voiceNumber.toString()] || {};
      console.log(`   Input files for voice ${voiceNumber}:`, voiceInputFiles);

      // Create track object
      const track = {
        id: `track-${Date.now()}-${voiceNumber}`,
        name: `Voice ${voiceNumber}`,
        audioUrl: voiceFile,
        duration: 0, // Will be calculated when audio loads
        startPosition: 0,
        gain: 1.0,
        isMuted: false,
        isSolo: false,
        cropStart: 0,
        cropEnd: 0,
        fx: {
          reverb: 0,
          fadeIn: 0.2,
          fadeOut: 1.0
        },
        // Store generation metadata for regeneration
        metadata: {
          params: params,
          type: 'generated',
          voiceNumber: voiceNumber,
          // Store all input file paths from backend
          inputFiles: {
            type: voiceInputFiles.type || (params.fastModeVariant ? 'midi' : 'wav'),
            midiPath: voiceInputFiles.midi_path || null,
            renderPath: voiceInputFiles.render_path || null,
            basicPitchMidiPath: voiceInputFiles.basicpitch_midi_path || null,
            masterMidiPath: voiceInputFiles.master_midi_path || null
          },
          tempo: null // Will be extracted from MIDI file if available
        },
        instrumentGroup: params.instrumentGroup,
        instrumentSubgroup: params.instrumentSubgroup
      };

      // Extract tempo from MIDI file if available
      const midiResult = await tryFetchMidi(voiceInputFiles, params);

      if (midiResult) {
        try {
          console.log(`🎵 Parsing MIDI file from: ${midiResult.path}`);
          const parsedMidi = parseMIDI(midiResult.arrayBuffer);
          if (parsedMidi) {
            track.metadata.tempo = parsedMidi.tempo || 120;

            // CRITICAL FIX: Add full MIDI data to track so MIDI window can display it!
            track.midiData = {
              notes: parsedMidi.notes || [],
              tempo: parsedMidi.tempo || 120,
              duration: parsedMidi.duration || 0
            };

            console.log(`✅ Extracted MIDI data: ${parsedMidi.notes?.length || 0} notes, ${parsedMidi.tempo || 120} BPM`);
          } else {
            console.warn('⚠️ Could not parse MIDI file, using default 120 BPM');
            track.metadata.tempo = 120;
          }
        } catch (error) {
          console.error('❌ Error parsing MIDI:', error);
          track.metadata.tempo = 120;
        }
      } else {
        // No MIDI file found
        track.metadata.tempo = 120;
      }

      // Track that we've added this URL
      addedTracksRef.current.add(voiceFile);

      // Load audio duration before adding to timeline
      try {
        const audio = new Audio();
        audio.src = voiceFile;
        await new Promise((resolve, reject) => {
          audio.addEventListener('loadedmetadata', () => {
            track.duration = audio.duration;
            console.log(`✅ Loaded audio duration: ${audio.duration.toFixed(2)}s`);
            resolve();
          });
          audio.addEventListener('error', (error) => {
            console.warn('⚠️ Failed to load audio duration, using default 10s:', error);
            track.duration = 10; // Fallback duration
            resolve(); // Still resolve to continue
          });
          // Timeout after 5 seconds
          setTimeout(() => {
            if (track.duration === 0) {
              console.warn('⚠️ Audio duration load timeout, using default 10s');
              track.duration = 10;
              resolve();
            }
          }, 5000);
        });
      } catch (error) {
        console.error('❌ Error loading audio duration:', error);
        track.duration = 10; // Fallback duration
      }

      // Add to timeline
      addTrackToTimeline(busId, track);
    }
  }, [addTrackToTimeline]);

  /**
   * Handle final generation results
   */
  const handleFinalResults = useCallback(async (result, busId, params, placeholderTrackId) => {
    console.log('📦 Handling final results:', result);
    console.log('📋 Current addedTracksRef contents:', Array.from(addedTracksRef.current));

    // Set placeholder to settling state if it exists
    // Don't remove it yet - wait until new tracks are loaded
    if (placeholderTrackId) {
      console.log('🎨 Setting placeholder to settling state:', placeholderTrackId);
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId: busId,
          trackId: placeholderTrackId,
          updates: {
            isSettling: true
          }
        }
      });
    }

    const { file_paths, input_files } = result;

    if (!file_paths || file_paths.length === 0) {
      console.warn('⚠️  No file paths in result');
      return;
    }

    let addedCount = 0;

    // Process each file path
    for (const filePath of file_paths) {
      // Skip if already added (check ref, not stale state)
      if (addedTracksRef.current.has(filePath)) {
        console.log(`⏭️  Skipping duplicate track: ${filePath} (already in ref)`);
        continue;
      }

      // Extract voice number from filename
      const match = filePath.match(/\/(\d+)\.wav$/);

      // In monophonic mode, skip voice 0 (it's the mix, individual voices are 1, 2, 3...)
      // In non-monophonic mode, voice 0 is the only track
      if (params.monophonicMode && match && parseInt(match[1], 10) === 0) {
        console.log(`⏭️  Skipping voice 0 (mix track) in monophonic mode`);
        continue;
      }

      // Extract voice number if available
      const voiceNumber = match ? parseInt(match[1], 10) : null;
      const trackName = voiceNumber !== null ? `Voice ${voiceNumber}` : 'Generated Track';

      // Extract input files for this voice from backend response
      const voiceInputFiles = input_files?.[voiceNumber?.toString()] || {};
      console.log(`✅ Adding track: ${trackName} (${filePath})`);
      console.log(`   Input files:`, voiceInputFiles);

      // Create track object
      const track = {
        id: `track-${Date.now()}-${Math.random()}`,
        name: trackName,
        audioUrl: filePath,
        duration: 0, // Will be calculated when audio loads
        startPosition: 0,
        gain: 1.0,
        isMuted: false,
        isSolo: false,
        cropStart: 0,
        cropEnd: 0,
        fx: {
          reverb: 0,
          fadeIn: 0.2,
          fadeOut: 1.0
        },
        // Store generation metadata for regeneration
        metadata: {
          params: params,
          type: 'generated',
          voiceNumber: voiceNumber,
          // Store all input file paths from backend
          inputFiles: {
            type: voiceInputFiles.type || (params.fastModeVariant ? 'midi' : 'wav'),
            midiPath: voiceInputFiles.midi_path || null,
            renderPath: voiceInputFiles.render_path || null,
            basicPitchMidiPath: voiceInputFiles.basicpitch_midi_path || null,
            masterMidiPath: voiceInputFiles.master_midi_path || null
          },
          tempo: null // Will be extracted from MIDI file if available
        },
        instrumentGroup: params.instrumentGroup,
        instrumentSubgroup: params.instrumentSubgroup
      };

      // Extract tempo from MIDI file if available
      const midiResult = await tryFetchMidi(voiceInputFiles, params);

      if (midiResult) {
        try {
          console.log(`🎵 Parsing MIDI file from: ${midiResult.path}`);
          const parsedMidi = parseMIDI(midiResult.arrayBuffer);
          if (parsedMidi) {
            track.metadata.tempo = parsedMidi.tempo || 120;

            // CRITICAL FIX: Add full MIDI data to track so MIDI window can display it!
            track.midiData = {
              notes: parsedMidi.notes || [],
              tempo: parsedMidi.tempo || 120,
              duration: parsedMidi.duration || 0
            };

            console.log(`✅ Extracted MIDI data: ${parsedMidi.notes?.length || 0} notes, ${parsedMidi.tempo || 120} BPM`);
          } else {
            console.warn('⚠️ Could not parse MIDI file, using default 120 BPM');
            track.metadata.tempo = 120;
          }
        } catch (error) {
          console.error('❌ Error parsing MIDI:', error);
          track.metadata.tempo = 120;
        }
      } else {
        // No MIDI file found
        track.metadata.tempo = 120;
      }

      // Track that we've added this URL
      addedTracksRef.current.add(filePath);

      // Load audio duration before adding to timeline
      try {
        const audio = new Audio();
        audio.src = filePath;
        await new Promise((resolve, reject) => {
          audio.addEventListener('loadedmetadata', () => {
            track.duration = audio.duration;
            console.log(`✅ Loaded audio duration: ${audio.duration.toFixed(2)}s`);
            resolve();
          });
          audio.addEventListener('error', (error) => {
            console.warn('⚠️ Failed to load audio duration, using default 10s:', error);
            track.duration = 10; // Fallback duration
            resolve(); // Still resolve to continue
          });
          // Timeout after 5 seconds
          setTimeout(() => {
            if (track.duration === 0) {
              console.warn('⚠️ Audio duration load timeout, using default 10s');
              track.duration = 10;
              resolve();
            }
          }, 5000);
        });
      } catch (error) {
        console.error('❌ Error loading audio duration:', error);
        track.duration = 10; // Fallback duration
      }

      // Add to timeline
      addTrackToTimeline(busId, track);
      addedCount++;
    }

    console.log(`✅ Added ${addedCount} new track(s) to bus ${busId}`);

    // Update bus metadata with instrument information
    if (addedCount > 0 && params.instrumentGroup && params.instrumentSubgroup) {
      dispatch({
        type: 'UPDATE_BUS_METADATA',
        payload: {
          busId,
          metadata: {
            instrumentGroup: params.instrumentGroup,
            instrumentSubgroup: params.instrumentSubgroup
          }
        }
      });
      console.log(`✅ Updated bus ${busId} metadata with instrument: ${params.instrumentSubgroup}`);
    }

    // NOW remove the placeholder after new tracks are loaded and settling animation completes
    if (placeholderTrackId && addedCount > 0) {
      // Wait for 0.5s settling animation to complete
      await new Promise(resolve => setTimeout(resolve, 500));

      console.log('🗑️ Removing placeholder track after settling and new tracks loaded:', placeholderTrackId);
      dispatch({
        type: 'REMOVE_TRACK',
        payload: {
          busId: busId,
          trackId: placeholderTrackId
        }
      });
    }
  }, [addTrackToTimeline, dispatch]);

  /**
   * Main generation function
   * @param {Object} params - Generation parameters
   * @param {File} audioFile - Optional audio conditioning file
   * @returns {Promise<void>}
   */
  const generate = useCallback(async (params, audioFile = null) => {
    console.log('🎵 Starting generation with params:', params);

    // Reset state
    setIsGenerating(true);
    setGenerationError(null);
    setGenerationProgress({
      status: 'starting',
      completedVoices: 0,
      totalVoices: 0,
      progress: 0
    });

    // Clear tracked URLs from previous generation
    addedTracksRef.current.clear();

    // Start timer
    startTimer();

    // Declare variables outside try block for error handler access
    let busId = null;
    let placeholderTrackId = null;

    try {
      // Start generation
      const startResult = await generationAPI.startGeneration(params, audioFile);
      console.log('✅ Generation started:', startResult);

      if (!startResult.task_id) {
        throw new Error('No task_id received from server');
      }

      const { task_id, expected_voices = 0 } = startResult;

      // Determine which bus to use
      // If sourceTrack exists (MIDI track loaded as input), check if instrument has changed
      if (state.sourceTrack?.busId) {
        const sourceBus = state.buses.find(b => b.id === state.sourceTrack.busId);
        const previousInstrument = sourceBus?.metadata?.instrumentSubgroup;
        const currentInstrument = params.instrumentSubgroup;

        // If instrument changed, create a new bus
        if (previousInstrument && previousInstrument !== currentInstrument) {
          console.log(`🎸 Instrument changed from ${previousInstrument} to ${currentInstrument}, creating new bus`);
          busId = getBusForGeneration(params.selectedMode || 'Music');
        } else {
          console.log(`🎯 Using existing bus from source track: ${state.sourceTrack.busId}`);
          busId = state.sourceTrack.busId;
        }
      } else {
        busId = getBusForGeneration(params.selectedMode || 'Music');
        console.log(`🆕 Created new bus for generation: ${busId}`);
      }

      // Create placeholder track immediately
      placeholderTrackId = `placeholder-${Date.now()}`;
      dispatch({
        type: 'ADD_TRACK',
        payload: {
          busId: busId,
          track: {
            id: placeholderTrackId,
            name: 'Generating...',
            audioUrl: null,
            duration: 0,
            isPlaceholder: true,
            isGenerating: true,
            gain: 1.0,
            pan: 0,
            isMuted: false,
            startTime: 0,
            offset: 0
          }
        }
      });

      // Update progress
      setGenerationProgress({
        status: 'generating',
        completedVoices: 0,
        totalVoices: expected_voices,
        progress: 0,
        placeholderTrackId: placeholderTrackId
      });

      // Poll until complete
      console.log(`🔍 Polling setup: enableVoiceSeparation=${params.enableVoiceSeparation}, monophonicMode=${params.monophonicMode}`);

      // Enable partial results if voice separation OR monophonic mode is enabled
      // (backend returns completed_voices array in both cases)
      const partialResultCallback = (params.enableVoiceSeparation || params.monophonicMode)
        ? (partialData) => {
            console.log('🔔 Partial result callback triggered!');
            handlePartialResult(partialData, busId, params);
          }
        : null;

      console.log(`🔍 Partial result callback: ${partialResultCallback ? 'ENABLED' : 'DISABLED'}`);

      const result = await generationAPI.pollUntilComplete(
        task_id,
        // Progress callback
        (progressData) => {
          setGenerationProgress(prev => ({
            ...prev,
            completedVoices: progressData.completedVoices || 0,
            totalVoices: progressData.totalVoices || expected_voices,
            progress: progressData.progress || 0,
            status: progressData.status || 'generating'
          }));
        },
        // Partial result callback (for incremental updates)
        partialResultCallback
      );

      console.log('✅ Generation completed:', result);

      // Handle final results
      await handleFinalResults(result, busId, params, placeholderTrackId);

      // Update progress to completed
      setGenerationProgress({
        status: 'completed',
        completedVoices: expected_voices,
        totalVoices: expected_voices,
        progress: 1.0
      });

    } catch (error) {
      console.error('❌ Generation error:', error);

      // Remove placeholder track on error
      if (placeholderTrackId && busId) {
        console.log('🗑️ Removing placeholder track due to error:', placeholderTrackId);
        dispatch({
          type: 'REMOVE_TRACK',
          payload: {
            busId: busId,
            trackId: placeholderTrackId
          }
        });
      }

      setGenerationError(error.message || 'Generation failed');
      setGenerationProgress({
        status: 'failed',
        completedVoices: 0,
        totalVoices: 0,
        progress: 0
      });
    } finally {
      setIsGenerating(false);
      stopTimer();
    }
  }, [
    startTimer,
    stopTimer,
    getBusForGeneration,
    handlePartialResult,
    handleFinalResults,
    state.sourceTrack,  // Need to track sourceTrack for bus selection
    state.buses  // Need to check bus metadata for instrument changes
  ]);

  /**
   * Cancel ongoing generation (if supported)
   */
  const cancelGeneration = useCallback(() => {
    // TODO: Implement cancellation if backend supports it
    console.warn('⚠️  Generation cancellation not yet implemented');
    setIsGenerating(false);
    stopTimer();
  }, [stopTimer]);

  return {
    // State
    isGenerating,
    generationProgress,
    generationError,
    elapsedTime: formatElapsedTime(),

    // Actions
    generate,
    cancelGeneration,

    // Exposed state setters for custom generation flows (e.g., ACE-Step)
    setIsGenerating,
    setGenerationProgress,
    setGenerationError,
    startTimer,
    stopTimer
  };
}
