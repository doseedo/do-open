import React, { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useThemeColor } from '../../hooks/useThemeColor';
import midiPlayer from '../../utils/midiPlayer';
import { parseMIDI } from '../../utils/midiParser';
import styles from './MIDIChart.module.css';

/**
 * MIDIChart - Interactive piano roll for placing MIDI notes
 * Can work standalone or display/edit a selected MIDI track
 */
const MIDIChart = () => {
  const { state, dispatch } = useApp();

  // Get theme colors for MIDI notes
  const primaryBlue = useThemeColor('--color-primary-blue', '#667eea');
  const primaryPurpleDark = useThemeColor('--color-primary-purple-dark', '#764ba2');
  const primaryBlueLight = useThemeColor('--color-primary-blue-light', '#88a3f7');

  const canvasRef = useRef(null);
  const canvasWrapperRef = useRef(null);
  const [notes, setNotes] = useState([]); // Array of {note, time, duration, velocity}
  const [originalNotes, setOriginalNotes] = useState([]); // Original unedited MIDI for ghost display
  const [dragState, setDragState] = useState(null); // { type: 'move' | 'resize' | 'select', noteIndex, startX, startTime, startDuration }
  const [hoveredNote, setHoveredNote] = useState(null);
  const [selectedNote, setSelectedNote] = useState(null); // Index of selected note for deletion (deprecated - use selectedNotes)
  const [selectedNotes, setSelectedNotes] = useState([]); // Array of selected note indices
  const [selectionBox, setSelectionBox] = useState(null); // { startX, startY, endX, endY }
  const [toolMode, setToolMode] = useState('draw'); // 'draw' or 'select'
  const [cmdKeyHeld, setCmdKeyHeld] = useState(false); // Track Cmd/Ctrl key state
  const [isPlaying, setIsPlaying] = useState(false);
  const [playerInitialized, setPlayerInitialized] = useState(false);
  const [zoomX, setZoomX] = useState(1.0); // Horizontal zoom (0.5 to 5.0)
  const [zoomY, setZoomY] = useState(1.0); // Vertical zoom (0.5 to 3.0)
  const [zoomMode, setZoomMode] = useState('horizontal'); // 'horizontal' or 'vertical'
  const [chartMode, setChartMode] = useState('midi'); // 'midi', 'f0', or 'score'
  const [f0Contour, setF0Contour] = useState([]); // Array of {time, frequency} for F0 mode
  const [isDrawingF0, setIsDrawingF0] = useState(false); // Track if currently drawing F0 contour
  const [midiTempo, setMidiTempo] = useState(120); // Track MIDI tempo for grid display
  const [lastNoteDuration, setLastNoteDuration] = useState(1.0); // Remember last note duration for placing new notes

  // Generate MIDI settings
  const [showGenerateSettings, setShowGenerateSettings] = useState(false);
  const [generateSettings, setGenerateSettings] = useState({
    mode: 'basic',      // 'basic', 'genre', 'context', 'chords'
    key: 'C minor',
    chords: 'Cm7,G7,Cm7,G7',
    bars: 4,
    minNote: 60,
    maxNote: 76,
    chromatic: 0.2,
    tempo: 120,
    seed: null,
    // Genre mode settings
    genre: 'jazz',
    style: 'bebop',
    // Context mode settings
    role: 'melody',
    matchDensity: true,
    matchStyle: true,
    // Chord mode settings
    voicing: 'random',
    rhythm: 'random',
    chordStyle: 'random'
  });

  // Determine if we're in select mode based on tool mode and Cmd key
  // Draw mode: Cmd key enables select mode
  // Select mode: Cmd key enables draw mode (inverted)
  // F0 mode: always in draw mode
  const isSelectMode = chartMode === 'f0' ? false : (toolMode === 'draw' ? cmdKeyHeld : !cmdKeyHeld);

  // Get selected track from context
  const selectedTrack = state.selectedTrack;
  const isMidiTrack = selectedTrack && selectedTrack.type === 'midi';

  // Check if selected track has MIDI data (either from generation or stored directly)
  const hasGeneratedMidi = selectedTrack &&
                           selectedTrack.metadata &&
                           (selectedTrack.metadata.inputFiles || selectedTrack.metadata.midiData);

  // Auto-select first MIDI track if none is selected (on first interaction with MIDI window)
  useEffect(() => {
    // Only auto-select if:
    // 1. No track is currently selected
    // 2. There are buses available
    // 3. There's at least one MIDI track available
    if (!selectedTrack && state.buses.length > 0) {
      // Find the first MIDI bus with tracks
      for (const bus of state.buses) {
        if (bus.type === 'MIDI' && bus.tracks.length > 0) {
          const firstMidiTrack = bus.tracks[0];
          console.log('🎹 Auto-selecting first MIDI track:', firstMidiTrack.name);
          dispatch({
            type: 'SELECT_TRACK',
            payload: { trackId: firstMidiTrack.id }
          });
          break;
        }
      }
    }
  }, []); // Run only once on mount

  // Calculate duration from notes (or use default)
  const calculatedDuration = useMemo(() => {
    if (notes.length === 0) {
      return 16; // Default ~4 bars worth of seconds at 120 BPM
    }
    const maxTime = Math.max(...notes.map(n => n.time + n.duration));
    const duration = Math.max(maxTime + 4, 16); // Add 4 seconds padding, minimum 16 seconds
    return duration;
  }, [notes]);

  // Calculate note range from notes (or use default)
  const noteRange = useMemo(() => {
    const defaultRange = { min: 36, max: 84 }; // Default C2-C6 (48 notes)
    if (notes.length === 0) return defaultRange;

    const noteNumbers = notes.map(n => n.note);
    const minNote = Math.min(...noteNumbers);
    const maxNote = Math.max(...noteNumbers);

    // Calculate range with padding
    let min = Math.max(0, minNote - 2);
    let max = Math.min(127, maxNote + 2);

    // Ensure minimum range of 2 octaves (24 notes) to prevent canvas from becoming too small
    const currentRange = max - min + 1;
    const minRange = 24; // 2 octaves

    if (currentRange < minRange) {
      const expansion = minRange - currentRange;
      const expandTop = Math.floor(expansion / 2);
      const expandBottom = expansion - expandTop;

      min = Math.max(0, min - expandBottom);
      max = Math.min(127, max + expandTop);

      // If we hit the boundary, expand the other side
      if (min === 0 && max < minRange - 1) {
        max = Math.min(127, minRange - 1);
      } else if (max === 127 && min > 127 - minRange + 1) {
        min = Math.max(0, 127 - minRange + 1);
      }
    }

    return { min, max };
  }, [notes]);

  // Configuration - memoized to prevent re-renders, scales with zoom
  const config = useMemo(() => ({
    minNote: noteRange.min,
    maxNote: noteRange.max,
    gridWidth: 40 * zoomX, // pixels per SECOND (scaled by horizontal zoom) - notes are stored in seconds
    noteHeight: 20 * zoomY, // pixels per note (scaled by vertical zoom)
    duration: calculatedDuration,
    leftMargin: 60, // space for piano keys (not scaled)
    topMargin: 30,
    bottomMargin: 20,
    resizeHandleWidth: 8 // Width of resize handle in pixels
  }), [zoomX, zoomY, calculatedDuration, noteRange]);

  const totalNotes = config.maxNote - config.minNote + 1;
  const canvasWidth = config.leftMargin + (config.gridWidth * config.duration) + 20;
  const canvasHeight = config.topMargin + (config.noteHeight * totalNotes) + config.bottomMargin;

  // Calculate snap resolution based on zoom level (matches visible grid subdivisions)
  const getSnapSubdivision = useCallback((zoom) => {
    // Returns the snap factor: how many subdivisions per quarter note
    // Higher zoom = finer snapping
    if (zoom < 1.5) return 1;      // Quarter notes (no subdivision)
    if (zoom < 3.0) return 4;      // 16th notes
    if (zoom < 4.5) return 8;      // 32nd notes
    return 16;                      // 64th notes (maximum detail)
  }, []);

  // Snap a time value to the current grid subdivision
  const snapTime = useCallback((timeInSeconds) => {
    const snapFactor = getSnapSubdivision(zoomX);
    return Math.round(timeInSeconds * snapFactor) / snapFactor;
  }, [zoomX, getSnapSubdivision]);

  // Initialize MIDI player on mount
  useEffect(() => {
    const initPlayer = async () => {
      try {
        await midiPlayer.initialize();
        setPlayerInitialized(true);
      } catch (error) {
        console.error('Failed to initialize MIDI player:', error);
      }
    };
    initPlayer();

    return () => {
      midiPlayer.stopAll();
    };
  }, []);

  // Sync MIDI tempo with timeline BPM and convert note times
  useEffect(() => {
    if (state.bpm && state.bpm !== midiTempo && notes.length > 0) {
      console.log(`🎵 MIDI Chart: Converting notes from ${midiTempo} BPM to ${state.bpm} BPM`);

      // Calculate conversion ratio (old tempo / new tempo)
      const conversionRatio = midiTempo / state.bpm;

      // Convert all note times and durations
      const convertedNotes = notes.map(note => ({
        ...note,
        time: note.time * conversionRatio,
        duration: note.duration * conversionRatio
      }));

      console.log(`   Converted ${convertedNotes.length} notes (ratio: ${conversionRatio.toFixed(3)}x)`);
      setNotes(convertedNotes);
      setMidiTempo(state.bpm);
    } else if (state.bpm && state.bpm !== midiTempo) {
      // No notes yet, just sync tempo
      console.log(`🎵 MIDI Chart: Syncing tempo to ${state.bpm} BPM (was ${midiTempo} BPM)`);
      setMidiTempo(state.bpm);
    }
  }, [state.bpm, midiTempo, notes]);

  // Attach native wheel event listener to prevent page zoom
  useEffect(() => {
    const wrapper = canvasWrapperRef.current;
    if (!wrapper) return;

    const handleNativeWheel = (e) => {
      // Prevent page zoom when Ctrl/Cmd + scroll
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        e.stopPropagation();
      }
    };

    const preventDefault = (e) => {
      if (e.touches.length > 1) {
        e.preventDefault();
      }
    };

    const preventGestureZoom = (e) => {
      e.preventDefault();
    };

    // Use { passive: false } to allow preventDefault()
    wrapper.addEventListener('wheel', handleNativeWheel, { passive: false });

    // Prevent pinch zoom on touch devices
    wrapper.addEventListener('touchmove', preventDefault, { passive: false });
    wrapper.addEventListener('gesturestart', preventGestureZoom, { passive: false });
    wrapper.addEventListener('gesturechange', preventGestureZoom, { passive: false });
    wrapper.addEventListener('gestureend', preventGestureZoom, { passive: false });

    return () => {
      wrapper.removeEventListener('wheel', handleNativeWheel);
      wrapper.removeEventListener('touchmove', preventDefault);
      wrapper.removeEventListener('gesturestart', preventGestureZoom);
      wrapper.removeEventListener('gesturechange', preventGestureZoom);
      wrapper.removeEventListener('gestureend', preventGestureZoom);
    };
  }, []);

  // Cmd key handler - toggle select mode
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.metaKey || e.ctrlKey) {
        setCmdKeyHeld(true);
      }
    };

    const handleKeyUp = (e) => {
      if (!e.metaKey && !e.ctrlKey) {
        setCmdKeyHeld(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);

    // Clear select mode when window loses focus
    const handleBlur = () => setCmdKeyHeld(false);
    window.addEventListener('blur', handleBlur);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleBlur);
    };
  }, []);

  // Delete key handler - delete selected notes
  useEffect(() => {
    const handleKeyDown = (e) => {
      // CRITICAL: Check if we're in MIDI Chart context by checking if the component is mounted
      // Only intercept Delete/Backspace if we have selected notes in the MIDI chart
      if ((e.key === 'Delete' || e.key === 'Backspace')) {
        // Only handle if we have selected notes (don't interfere with empty state)
        if (selectedNotes.length > 0 || selectedNote !== null) {
          console.log('🎹 MIDIChart intercepting delete key for note deletion');

          // Prevent default behavior
          e.preventDefault();
          // CRITICAL: Stop immediate propagation to prevent global track delete handler from firing
          // This stops other listeners on the same element (window) from receiving the event
          e.stopImmediatePropagation();

          if (selectedNotes.length > 0) {
            console.log(`🗑️ Deleting ${selectedNotes.length} notes`);

            // Remove all selected notes
            setNotes(prevNotes => {
              const newNotes = prevNotes.filter((_, index) => !selectedNotes.includes(index));
              console.log(`   Notes before: ${prevNotes.length}, after: ${newNotes.length}`);
              return newNotes;
            });

            // Clear selection
            setSelectedNotes([]);
            setSelectedNote(null);
          } else if (selectedNote !== null) {
            console.log(`🗑️ Deleting note ${selectedNote}`);

            // Remove the selected note
            setNotes(prevNotes => {
              const newNotes = prevNotes.filter((_, index) => index !== selectedNote);
              console.log(`   Notes before: ${prevNotes.length}, after: ${newNotes.length}`);
              return newNotes;
            });

            // Clear selection
            setSelectedNote(null);
          }
        } else {
          console.log('🎹 MIDIChart: No notes selected, allowing event to propagate');
        }
      }
    };

    // Use capture phase to ensure this handler runs before the global track delete handler
    // Capture phase runs from window down to target, bubble phase runs from target up to window
    window.addEventListener('keydown', handleKeyDown, { capture: true });
    return () => window.removeEventListener('keydown', handleKeyDown, { capture: true });
  }, [selectedNote, selectedNotes, notes]);

  // Load MIDI data from selected track
  useEffect(() => {
    if (isMidiTrack && selectedTrack.midiData && selectedTrack.midiData.notes) {
      // Convert track MIDI data to piano roll format
      const trackNotes = selectedTrack.midiData.notes
        .map(note => ({
          note: note.midi || note.note, // Support both field names
          time: note.time,
          duration: note.duration,
          velocity: note.velocity || 100
        }))
        .filter(note =>
          Number.isFinite(note.note) &&
          Number.isFinite(note.time) &&
          Number.isFinite(note.duration) &&
          note.duration > 0
        );

      setNotes(trackNotes);
      setOriginalNotes(JSON.parse(JSON.stringify(trackNotes))); // Deep copy for ghost display
      const trackTempo = selectedTrack.midiData.tempo || 120;
      setMidiTempo(trackTempo);

      // CRITICAL FIX: Update timeline BPM to match MIDI track tempo
      // This prevents the auto-conversion from compressing the notes!
      if (trackTempo !== state.bpm) {
        console.log(`🎵 Setting timeline BPM to match MIDI track: ${trackTempo} BPM`);
        dispatch({ type: 'SET_BPM', payload: trackTempo });
      }
    } else if (hasGeneratedMidi) {
      // Selected track is a generated audio track with MIDI data
      const storedTempo = selectedTrack.metadata.tempo || 120;

      // Priority 1: Check if there's directly stored MIDI data (from replaced MIDI track)
      if (selectedTrack.metadata.midiData && selectedTrack.metadata.midiData.notes) {
        console.log('🎹 Using stored MIDI data from replaced track');
        const trackNotes = selectedTrack.metadata.midiData.notes
          .map(note => ({
            note: note.midi || note.note,
            time: note.time,
            duration: note.duration,
            velocity: note.velocity || 100
          }))
          .filter(note =>
            Number.isFinite(note.note) &&
            Number.isFinite(note.time) &&
            Number.isFinite(note.duration) &&
            note.duration > 0
          );

        setNotes(trackNotes);
        setOriginalNotes(JSON.parse(JSON.stringify(trackNotes)));
        setMidiTempo(selectedTrack.metadata.midiData.tempo || storedTempo);
      }
      // Priority 2: Check if there's edited MIDI in metadata
      else if (selectedTrack.metadata.editedMidi) {
        console.log('🎹 Using edited MIDI from metadata');
        const editedMidi = selectedTrack.metadata.editedMidi;
        setNotes(editedMidi.notes);
        setMidiTempo(editedMidi.tempo || storedTempo);

        // For ghost display, fetch original from backend
        const inputFiles = selectedTrack.metadata.inputFiles;
        const midiUrl = inputFiles.midiPath ||
                        inputFiles.basicPitchMidiPath ||
                        inputFiles.masterMidiPath;

        if (midiUrl) {
          fetch(midiUrl)
            .then(response => response.arrayBuffer())
            .then(arrayBuffer => {
              const parsedData = parseMIDI(arrayBuffer);
              if (parsedData && parsedData.notes) {
                setOriginalNotes(parsedData.notes);
              }
            })
            .catch(error => {
              console.error('Failed to load original MIDI for ghost display:', error);
              setOriginalNotes([]);
            });
        }
      } else {
        // No edits yet, load original MIDI from backend
        const inputFiles = selectedTrack.metadata.inputFiles;

        // Determine which MIDI file to load (priority order):
        // 1. midiPath - direct MIDI input
        // 2. basicPitchMidiPath - extracted from audio
        // 3. masterMidiPath - concatenated scene MIDI
        const midiUrl = inputFiles.midiPath ||
                        inputFiles.basicPitchMidiPath ||
                        inputFiles.masterMidiPath;

        if (midiUrl) {
          console.log(`📥 Fetching MIDI file: ${midiUrl}`);

          // Fetch and parse the MIDI file
          fetch(midiUrl)
            .then(response => {
              if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
              }
              return response.arrayBuffer();
            })
            .then(arrayBuffer => {
              // Parse MIDI file
              const parsedData = parseMIDI(arrayBuffer);

              if (parsedData && parsedData.notes) {
                const minNoteTime = parsedData.notes.length > 0 ? Math.min(...parsedData.notes.map(n => n.time)) : 0;
                const maxNoteTime = parsedData.notes.length > 0 ? Math.max(...parsedData.notes.map(n => n.time + n.duration)) : 0;

                console.log(`✅ Loaded ${parsedData.notes.length} notes from MIDI`);
                console.log(`   Parsed tempo: ${parsedData.tempo} BPM`);
                console.log(`   Stored tempo: ${storedTempo} BPM`);
                console.log(`   Parsed duration: ${parsedData.duration?.toFixed(2)}s`);
                console.log(`   Note time range: ${minNoteTime.toFixed(2)}s to ${maxNoteTime.toFixed(2)}s`);
                console.log(`   At ${parsedData.tempo} BPM: ${(maxNoteTime / (60/parsedData.tempo)).toFixed(1)} beats = ${(maxNoteTime / (60/parsedData.tempo) / 4).toFixed(1)} bars`);

                // Rescale note times if there's a tempo mismatch
                // The MIDI parser used the embedded tempo to convert ticks to seconds
                // But we need to use the stored tempo from when the track was generated
                let processedNotes = parsedData.notes;

                if (storedTempo && parsedData.tempo && storedTempo !== parsedData.tempo) {
                  console.log(`⚠️ TEMPO MISMATCH! Rescaling from ${parsedData.tempo} BPM to ${storedTempo} BPM`);

                  // Time at tempo1 = beats * (60 / tempo1)
                  // Time at tempo2 = beats * (60 / tempo2)
                  // Therefore: time2 = time1 * (tempo1 / tempo2)
                  const tempoRatio = parsedData.tempo / storedTempo;

                  processedNotes = parsedData.notes.map(note => ({
                    ...note,
                    time: note.time * tempoRatio,
                    duration: note.duration * tempoRatio
                  }));

                  const newMaxTime = Math.max(...processedNotes.map(n => n.time + n.duration));
                  console.log(`   Tempo scaling factor: ${tempoRatio.toFixed(4)}`);
                  console.log(`   NEW time range: ${Math.min(...processedNotes.map(n => n.time)).toFixed(2)}s to ${newMaxTime.toFixed(2)}s`);
                  console.log(`   At ${storedTempo} BPM: ${(newMaxTime / (60/storedTempo)).toFixed(1)} beats = ${(newMaxTime / (60/storedTempo) / 4).toFixed(1)} bars`);
                  console.log(`   First 3 rescaled notes:`, processedNotes.slice(0, 3).map(n =>
                    `note=${n.note} time=${n.time.toFixed(3)}s dur=${n.duration.toFixed(3)}s`
                  ));
                } else {
                  console.log(`✓ No tempo rescaling (both at ${storedTempo || parsedData.tempo || 120} BPM)`);
                  console.log(`   First 3 notes:`, parsedData.notes.slice(0, 3).map(n =>
                    `note=${n.note} time=${n.time.toFixed(3)}s dur=${n.duration.toFixed(3)}s`
                  ));
                }

                setNotes(processedNotes);
                setOriginalNotes(JSON.parse(JSON.stringify(processedNotes))); // Deep copy for ghost display
                setMidiTempo(storedTempo || parsedData.tempo || 120);
              } else {
                console.warn('⚠️ MIDI file loaded but no notes found');
                setNotes([]);
                setOriginalNotes([]);
                setMidiTempo(storedTempo);
              }
            })
            .catch(error => {
              console.error('❌ Failed to load MIDI file:', error);
              setNotes([]);
              setOriginalNotes([]);
            });
        } else {
          console.log('   No MIDI file available for this track');
          setNotes([]);
          setOriginalNotes([]);
        }
      }
    } else {
      // No MIDI data, clear notes
      setNotes([]);
      setOriginalNotes([]);
    }
  }, [isMidiTrack, selectedTrack, hasGeneratedMidi, dispatch, state.bpm]);

  // Save edited notes back to track when notes change (debounced)
  useEffect(() => {
    // Only save if we have a selected track and notes have been loaded
    if (!selectedTrack || notes.length === 0) return;
    if (!isMidiTrack && !hasGeneratedMidi) return;

    // Debounce to avoid too many updates
    const timeoutId = setTimeout(() => {
      // Find which bus this track belongs to
      let trackBusId = null;
      for (const bus of state.buses) {
        if (bus.tracks.find(t => t.id === selectedTrack.id)) {
          trackBusId = bus.id;
          break;
        }
      }

      if (!trackBusId) return;

      // Convert piano roll notes back to MIDI format
      const updatedNotes = notes.map(note => ({
        note: note.note,  // Use 'note' field to match midiParser output
        time: note.time,
        duration: note.duration,
        velocity: note.velocity || 100,
        name: '', // Not used in piano roll
        ticks: 0 // Will be recalculated if needed
      }));

      if (isMidiTrack) {
        // For MIDI tracks, sync to timeline BPM
        const updatedMidiData = {
          ...selectedTrack.midiData,
          notes: updatedNotes,
          duration: calculatedDuration,
          tempo: midiTempo // Sync to current timeline BPM
        };

        dispatch({
          type: 'UPDATE_TRACK_MIDI_DATA',
          payload: {
            busId: trackBusId,
            trackId: selectedTrack.id,
            midiData: updatedMidiData
          }
        });
      } else if (hasGeneratedMidi) {
        // For generated audio tracks with stored MIDI data (from replaced track)
        // Update metadata.midiData if it exists, otherwise use editedMidi
        const midiDataKey = selectedTrack.metadata.midiData ? 'midiData' : 'editedMidi';

        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId: trackBusId,
            trackId: selectedTrack.id,
            updates: {
              metadata: {
                ...selectedTrack.metadata,
                [midiDataKey]: {
                  ...selectedTrack.metadata[midiDataKey],
                  notes: updatedNotes,
                  duration: calculatedDuration,
                  tempo: midiTempo // Sync to current timeline BPM
                }
              }
            }
          }
        });
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [notes, isMidiTrack, hasGeneratedMidi, selectedTrack, calculatedDuration, midiTempo, state.buses, dispatch]);

  // Note names for display
  const getNoteNameFromMidi = (midiNote) => {
    const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
    const octave = Math.floor(midiNote / 12) - 1;
    const noteName = noteNames[midiNote % 12];
    return `${noteName}${octave}`;
  };

  // Check if note is a black key
  const isBlackKey = (midiNote) => {
    const noteInOctave = midiNote % 12;
    return [1, 3, 6, 8, 10].includes(noteInOctave); // C#, D#, F#, G#, A#
  };

  // Draw the piano roll grid
  const drawGrid = useCallback((ctx) => {
    // Clear canvas
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw piano keys on the left
    for (let i = 0; i < totalNotes; i++) {
      const midiNote = config.maxNote - i;
      const y = config.topMargin + (i * config.noteHeight);
      const isBlack = isBlackKey(midiNote);

      // Draw key background
      if (isBlack) {
        // Black key
        ctx.fillStyle = '#1a1a2e';
        ctx.fillRect(0, y, config.leftMargin, config.noteHeight);
      } else {
        // White key
        ctx.fillStyle = '#e8e8f0';
        ctx.fillRect(0, y, config.leftMargin, config.noteHeight);
      }

      // Draw key border
      ctx.strokeStyle = '#0a0a0f';
      ctx.lineWidth = 1;
      ctx.strokeRect(0, y, config.leftMargin, config.noteHeight);

      // Draw note label - only for C notes (C1, C2, C3, etc.)
      if (midiNote % 12 === 0) {
        const noteName = getNoteNameFromMidi(midiNote);
        ctx.fillStyle = '#888888'; // Light grey
        ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(noteName, config.leftMargin / 2, y + config.noteHeight / 2);
      }
    }

    // Draw horizontal grid lines (note separators) - only in grid area
    ctx.strokeStyle = '#2a2a3e';
    ctx.lineWidth = 1;
    for (let i = 0; i <= totalNotes; i++) {
      const y = config.topMargin + (i * config.noteHeight);
      const midiNote = config.maxNote - i;

      ctx.beginPath();
      ctx.moveTo(config.leftMargin, y);
      ctx.lineTo(canvasWidth, y);

      // Highlight C notes
      if (midiNote % 12 === 0) {
        ctx.strokeStyle = '#4a4a6e';
        ctx.lineWidth = 2;
      } else {
        ctx.strokeStyle = '#2a2a3e';
        ctx.lineWidth = 1;
      }
      ctx.stroke();
    }

    // Draw vertical grid lines (beat separators) with subdivisions when zoomed in
    // Notes are stored in SECONDS, but grid should show BEATS
    // Calculate seconds per beat using tempo
    const secondsPerBeat = 60 / midiTempo; // e.g., at 120 BPM, 1 beat = 0.5 seconds
    const totalBeats = config.duration / secondsPerBeat; // Total beats in the duration

    // Subdivisions scale with zoom level
    const subdivisions = getSnapSubdivision(zoomX); // 1 = quarter, 4 = 16th, 8 = 32nd, 16 = 64th

    for (let i = 0; i <= totalBeats * subdivisions; i++) {
      const beatPosition = i / subdivisions; // Position in beats (quarter notes)
      const timeInSeconds = beatPosition * secondsPerBeat; // Convert beats to seconds
      const x = config.leftMargin + (timeInSeconds * config.gridWidth); // gridWidth is pixels per second
      const isBarLine = i % (4 * subdivisions) === 0; // Every 4 beats (1 bar)
      const isBeatLine = i % subdivisions === 0; // Every beat
      const isSubbeatLine = !isBeatLine; // Subdivision lines (16th notes)

      // Set line style based on position
      if (isBarLine) {
        ctx.strokeStyle = '#5a5a6e'; // Grey instead of blue
        ctx.lineWidth = 2;
      } else if (isBeatLine) {
        ctx.strokeStyle = '#2a2a3e';
        ctx.lineWidth = 1;
      } else if (isSubbeatLine) {
        ctx.strokeStyle = '#1a1a2e';
        ctx.lineWidth = 1;
      }

      ctx.beginPath();
      ctx.moveTo(x, config.topMargin);
      ctx.lineTo(x, canvasHeight - config.bottomMargin);
      ctx.stroke();

      // Draw bar numbers
      if (isBarLine) {
        ctx.fillStyle = '#888888'; // Grey instead of blue
        ctx.font = '12px -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(`${Math.floor(beatPosition / 4) + 1}`, x + 3, config.topMargin - 10);
      }
    }
  }, [canvasWidth, canvasHeight, config, totalNotes, zoomX, midiTempo]);

  // Draw F0 contour (pitch line)
  const drawF0Contour = useCallback((ctx) => {
    if (f0Contour.length === 0) return;

    // Draw the pitch contour line
    ctx.strokeStyle = primaryBlue;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    ctx.beginPath();
    f0Contour.forEach((point, index) => {
      const x = config.leftMargin + (point.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - point.note) * config.noteHeight);

      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();

    // Draw control points
    ctx.fillStyle = primaryBlueLight;
    f0Contour.forEach((point) => {
      const x = config.leftMargin + (point.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - point.note) * config.noteHeight);

      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [f0Contour, config, primaryBlue, primaryBlueLight]);

  // Draw notes
  const drawNotes = useCallback((ctx) => {
    notes.forEach((note, index) => {
      // Validate note data before drawing
      if (!Number.isFinite(note.time) || !Number.isFinite(note.duration) || !Number.isFinite(note.note)) {
        console.warn('⚠️ Skipping invalid note:', note);
        return;
      }

      const x = config.leftMargin + (note.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const width = note.duration * config.gridWidth;
      const height = config.noteHeight - 2;

      // Skip notes with invalid calculated positions
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(width) || width <= 0) {
        console.warn('⚠️ Skipping note with invalid position:', { x, y, width, note });
        return;
      }

      // Highlight hovered and selected notes
      const isHovered = hoveredNote === index;
      const isSelected = selectedNote === index || selectedNotes.includes(index);

      // Draw note rectangle with gradient (no border)
      const gradient = ctx.createLinearGradient(x, y, x + width, y);
      if (isSelected) {
        // Selected note - brighter theme gradient
        gradient.addColorStop(0, primaryBlueLight);
        gradient.addColorStop(1, primaryBlue);
      } else if (isHovered) {
        // Hovered note - lighter theme gradient
        gradient.addColorStop(0, primaryBlueLight);
        gradient.addColorStop(1, primaryPurpleDark);
      } else {
        // Normal note - theme gradient
        gradient.addColorStop(0, primaryBlue);
        gradient.addColorStop(1, primaryPurpleDark);
      }
      ctx.fillStyle = gradient;
      ctx.fillRect(x + 1, y + 1, width - 2, height);

      // Draw resize handles on both edges (thinner, gradient)
      if (isHovered || isSelected) {
        const handleWidth = 3; // Thinner resize handle
        const handleGradient = ctx.createLinearGradient(x, y, x, y + height);
        if (isSelected) {
          // Brighter gradient for selected
          handleGradient.addColorStop(0, '#c5d0ff');
          handleGradient.addColorStop(1, '#ddaaff');
        } else {
          // Lighter gradient for hovered
          handleGradient.addColorStop(0, '#aabbff');
          handleGradient.addColorStop(1, '#cc99ee');
        }
        ctx.fillStyle = handleGradient;

        // Left resize handle
        ctx.fillRect(x + 1, y + 1, handleWidth, height);
        // Right resize handle
        ctx.fillRect(x + width - handleWidth - 1, y + 1, handleWidth, height);
      }
    });
  }, [notes, config, hoveredNote, selectedNote, selectedNotes, primaryBlue, primaryPurpleDark, primaryBlueLight]);

  // Draw ghost notes (original unedited MIDI in semi-transparent grey)
  const drawGhostNotes = useCallback((ctx) => {
    if (originalNotes.length === 0) return; // No ghost notes to draw

    originalNotes.forEach((note) => {
      // Validate note data before drawing
      if (!Number.isFinite(note.time) || !Number.isFinite(note.duration) || !Number.isFinite(note.note)) {
        return;
      }

      const x = config.leftMargin + (note.time * config.gridWidth);
      const y = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const width = note.duration * config.gridWidth;
      const height = config.noteHeight - 2;

      // Skip notes with invalid calculated positions
      if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(width) || width <= 0) {
        return;
      }

      // Draw ghost note as very light semi-transparent grey
      ctx.fillStyle = 'rgba(120, 120, 120, 0.12)'; // Light grey, 12% opacity (was 25%)
      ctx.fillRect(x + 1, y + 1, width - 2, height);

      // Draw subtle border
      ctx.strokeStyle = 'rgba(100, 100, 100, 0.15)'; // Darker grey, 15% opacity (was 30%)
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 1, y + 1, width - 2, height);
    });
  }, [originalNotes, config]);

  // Draw playhead position indicator
  const drawPlayhead = useCallback((ctx) => {
    const playheadPosition = state.playheadPosition; // In seconds

    // MIDI notes are stored in seconds, so use seconds directly (not beats)
    // gridWidth is actually "pixels per second" for MIDI display
    const x = config.leftMargin + (playheadPosition * config.gridWidth);

    // Only draw if playhead is within visible range
    if (x >= config.leftMargin && x <= canvasWidth) {
      // Draw playhead line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.9)'; // White instead of red
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(x, config.topMargin);
      ctx.lineTo(x, canvasHeight - config.bottomMargin);
      ctx.stroke();

      // Draw playhead triangle at top
      ctx.fillStyle = 'rgba(255, 255, 255, 0.9)'; // White instead of red
      ctx.beginPath();
      ctx.moveTo(x, config.topMargin - 8);
      ctx.lineTo(x - 6, config.topMargin);
      ctx.lineTo(x + 6, config.topMargin);
      ctx.closePath();
      ctx.fill();
    }
  }, [state.playheadPosition, config, canvasWidth, canvasHeight]);

  // Draw selection box
  const drawSelectionBox = useCallback((ctx) => {
    if (!selectionBox) return;

    const { startX, startY, endX, endY } = selectionBox;
    const x = Math.min(startX, endX);
    const y = Math.min(startY, endY);
    const width = Math.abs(endX - startX);
    const height = Math.abs(endY - startY);

    // Draw selection box with dashed border
    ctx.strokeStyle = 'rgba(100, 200, 255, 0.8)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.strokeRect(x, y, width, height);
    ctx.setLineDash([]); // Reset

    // Draw semi-transparent fill
    ctx.fillStyle = 'rgba(100, 200, 255, 0.15)';
    ctx.fillRect(x, y, width, height);
  }, [selectionBox]);

  // Render canvas
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    drawGrid(ctx);

    if (chartMode === 'f0') {
      // F0 mode: draw pitch contour
      drawF0Contour(ctx);
    } else {
      // MIDI mode: draw notes
      drawGhostNotes(ctx); // Draw ghost MIDI first (as background layer)
      drawNotes(ctx);       // Draw editable notes on top
      drawSelectionBox(ctx); // Draw selection box on top
    }

    drawPlayhead(ctx);
  }, [drawGrid, drawGhostNotes, drawNotes, drawPlayhead, drawSelectionBox, drawF0Contour, chartMode]);

  // Auto-scroll to follow playhead when it moves
  useEffect(() => {
    if (!state.isPlaying) return;

    const wrapper = canvasWrapperRef.current;
    if (!wrapper) return;

    const playheadPosition = state.playheadPosition;
    // MIDI notes are in seconds, so use seconds directly
    const playheadX = config.leftMargin + (playheadPosition * config.gridWidth);

    // Get wrapper scroll position and dimensions
    const scrollLeft = wrapper.scrollLeft;
    const wrapperWidth = wrapper.clientWidth;

    // Auto-scroll if playhead is near the right edge (within 20% of viewport)
    const rightThreshold = scrollLeft + wrapperWidth * 0.8;
    const leftThreshold = scrollLeft + wrapperWidth * 0.2;

    if (playheadX > rightThreshold) {
      // Scroll right to keep playhead in view
      wrapper.scrollLeft = playheadX - wrapperWidth * 0.5;
    } else if (playheadX < leftThreshold && scrollLeft > 0) {
      // Scroll left if playhead moves backwards
      wrapper.scrollLeft = Math.max(0, playheadX - wrapperWidth * 0.5);
    }
  }, [state.playheadPosition, state.isPlaying, config]);

  // Helper to get mouse position relative to canvas
  const getMousePos = useCallback((e) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();

    // Account for canvas scaling (CSS size vs actual canvas size)
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    return { x, y };
  }, []);

  // Helper to find note at position
  const findNoteAtPos = useCallback((x, y) => {
    for (let i = notes.length - 1; i >= 0; i--) {
      const note = notes[i];
      const noteX = config.leftMargin + (note.time * config.gridWidth);
      const noteY = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
      const noteWidth = note.duration * config.gridWidth;
      const noteHeight = config.noteHeight - 2;

      if (x >= noteX && x <= noteX + noteWidth &&
          y >= noteY && y <= noteY + noteHeight) {
        // Check if clicking on resize handles (left or right)
        const handleWidth = 3; // Match the thinner handle width
        const isLeftResizeHandle = x <= noteX + handleWidth;
        const isRightResizeHandle = x >= noteX + noteWidth - handleWidth;
        const isResizeHandle = isLeftResizeHandle || isRightResizeHandle;
        return { index: i, isResizeHandle, isLeftHandle: isLeftResizeHandle };
      }
    }
    return null;
  }, [notes, config]);

  // Mouse down - start drag or resize (or selection in select mode)
  const handleMouseDown = useCallback(async (e) => {
    const pos = getMousePos(e);
    const result = findNoteAtPos(pos.x, pos.y);

    // F0 MODE - Start drawing pitch contour
    if (chartMode === 'f0') {
      setIsDrawingF0(true);

      // Convert mouse position to continuous pitch and time (no quantization)
      if (pos.x >= config.leftMargin && pos.x <= canvasWidth - 20 &&
          pos.y >= config.topMargin && pos.y <= canvasHeight - config.bottomMargin) {

        const time = (pos.x - config.leftMargin) / config.gridWidth;
        // Use continuous pitch value (not quantized to MIDI notes)
        const continuousPitch = config.maxNote - ((pos.y - config.topMargin) / config.noteHeight);

        // Start new contour
        setF0Contour([{ time, note: continuousPitch }]);
      }
      e.preventDefault();
      return;
    }

    if (isSelectMode) {
      // SELECT MODE - select notes or start selection box
      if (result) {
        // Clicking on a note
        if (selectedNotes.includes(result.index)) {
          // Clicking on already selected note
          if (result.isResizeHandle) {
            // Start resizing all selected notes - store as object keyed by index
            const initialDurations = {};
            selectedNotes.forEach(idx => {
              initialDurations[idx] = notes[idx].duration;
            });
            setDragState({
              type: 'resize-multiple',
              selectedIndices: [...selectedNotes],
              startX: pos.x,
              initialDurations
            });
            console.log(`🎵 [SELECT MODE] Starting resize for ${selectedNotes.length} selected notes`);
          } else {
            // Start dragging all selected notes - store as object keyed by index
            const initialPositions = {};
            selectedNotes.forEach(idx => {
              initialPositions[idx] = { time: notes[idx].time, note: notes[idx].note };
            });
            setDragState({
              type: 'move-multiple',
              selectedIndices: [...selectedNotes],
              startX: pos.x,
              startY: pos.y,
              initialPositions
            });
            console.log(`🎵 [SELECT MODE] Starting move for ${selectedNotes.length} selected notes`);
          }
        } else {
          // Clicking on unselected note - select it and prepare to drag
          let newSelection;
          if (e.shiftKey) {
            // Shift adds to selection
            newSelection = [...selectedNotes, result.index];
            setSelectedNotes(newSelection);
          } else {
            // Replace selection with this note
            newSelection = [result.index];
            setSelectedNotes(newSelection);
          }

          // Set up drag state for the new selection - store as object keyed by index
          const initialPositions = {};
          newSelection.forEach(idx => {
            initialPositions[idx] = { time: notes[idx].time, note: notes[idx].note };
          });
          setDragState({
            type: 'move-multiple',
            selectedIndices: newSelection,
            startX: pos.x,
            startY: pos.y,
            initialPositions
          });
        }
      } else {
        // Clicking on empty space - start selection box
        setSelectedNotes([]); // Clear selection
        setSelectionBox({
          startX: pos.x,
          startY: pos.y,
          endX: pos.x,
          endY: pos.y
        });
        setDragState({
          type: 'select',
          startX: pos.x,
          startY: pos.y
        });
      }
      e.preventDefault();
    } else {
      // NORMAL MODE - place/move/resize notes
      if (result) {
        const note = notes[result.index];

        // Check if clicked note is part of current selection
        const isPartOfSelection = selectedNotes.includes(result.index);
        let currentSelection = selectedNotes;

        // Handle selection based on modifier keys
        if (!isPartOfSelection) {
          if (e.shiftKey) {
            // Shift-click: add to selection (don't start drag on shift-add)
            const newSelection = [...selectedNotes, result.index];
            setSelectedNote(result.index);
            setSelectedNotes(newSelection);
            console.log(`🎵 Added note to selection (now ${newSelection.length} notes selected)`);
            e.preventDefault();
            return; // Don't start drag when adding to selection
          } else {
            // Normal click: select only this note
            setSelectedNote(result.index);
            setSelectedNotes([result.index]);
            currentSelection = [result.index];
          }
        }

        const hasMultipleSelected = currentSelection.length > 1;

        // Play the note sound (short preview)
        if (playerInitialized) {
          await midiPlayer.resume(); // Resume audio context if needed
          midiPlayer.playNote(note.note, 0.7, 0.3); // velocity 0.7, duration 0.3s
          console.log(`🔊 Playing note: ${getNoteNameFromMidi(note.note)}`);
        }

        if (result.isResizeHandle) {
          // Start resize - if multiple notes selected, resize all
          if (isPartOfSelection && hasMultipleSelected) {
            // Store initial durations and times for all selected notes
            const initialDurations = {};
            const initialTimes = {};
            currentSelection.forEach(idx => {
              initialDurations[idx] = notes[idx].duration;
              initialTimes[idx] = notes[idx].time;
            });
            setDragState({
              type: 'resize-multiple',
              selectedIndices: [...currentSelection],
              startX: pos.x,
              initialDurations,
              initialTimes,
              isLeftHandle: result.isLeftHandle
            });
            console.log(`🎵 [NORMAL MODE] Starting ${result.isLeftHandle ? 'left' : 'right'} resize for ${currentSelection.length} selected notes`);
          } else {
            // Single note resize
            setDragState({
              type: 'resize',
              noteIndex: result.index,
              startX: pos.x,
              startDuration: note.duration,
              startTime: note.time,
              isLeftHandle: result.isLeftHandle
            });
          }
        } else {
          // Start move - if multiple notes selected, move all
          if (isPartOfSelection && hasMultipleSelected) {
            // Store initial positions for all selected notes
            const initialPositions = {};
            currentSelection.forEach(idx => {
              initialPositions[idx] = { time: notes[idx].time, note: notes[idx].note };
            });
            setDragState({
              type: 'move-multiple',
              selectedIndices: [...currentSelection],
              startX: pos.x,
              startY: pos.y,
              initialPositions
            });
            console.log(`🎵 [NORMAL MODE] Starting move for ${currentSelection.length} selected notes`);
          } else {
            // Single note move
            setDragState({
              type: 'move',
              noteIndex: result.index,
              startX: pos.x,
              startY: pos.y,
              startTime: note.time,
              startNote: note.note
            });
          }
        }
        e.preventDefault();
      } else {
        // Click on empty space - clear selection and add new note
        setSelectedNote(null);

        if (pos.x >= config.leftMargin && pos.x <= canvasWidth - 20 &&
            pos.y >= config.topMargin && pos.y <= canvasHeight - config.bottomMargin) {

          const rawTime = (pos.x - config.leftMargin) / config.gridWidth; // Time in seconds
          const time = snapTime(rawTime); // Snap to current grid subdivision
          const noteIndex = Math.floor((pos.y - config.topMargin) / config.noteHeight);
          const midiNote = config.maxNote - noteIndex;

          const newNote = {
            note: midiNote,
            time: time,
            duration: lastNoteDuration, // Use last note duration
            velocity: 100
          };
          setNotes(prevNotes => [...prevNotes, newNote]);

          // Play the newly placed note sound
          if (playerInitialized) {
            await midiPlayer.resume(); // Resume audio context if needed
            midiPlayer.playNote(midiNote, 0.7, 0.3); // velocity 0.7, duration 0.3s
            console.log(`🔊 Playing new note: ${getNoteNameFromMidi(midiNote)}`);
          }

          console.log(`🎵 Added note: ${getNoteNameFromMidi(midiNote)} at beat ${time} with duration ${lastNoteDuration.toFixed(2)}`);
        }
      }
    }
  }, [notes, config, canvasWidth, canvasHeight, getMousePos, findNoteAtPos, playerInitialized, lastNoteDuration, isSelectMode, selectedNotes, chartMode, snapTime]);

  // Mouse move - handle drag or resize
  const handleMouseMove = useCallback((e) => {
    const pos = getMousePos(e);

    // F0 MODE - Continue drawing pitch contour
    if (isDrawingF0 && chartMode === 'f0') {
      if (pos.x >= config.leftMargin && pos.x <= canvasWidth - 20 &&
          pos.y >= config.topMargin && pos.y <= canvasHeight - config.bottomMargin) {

        const time = (pos.x - config.leftMargin) / config.gridWidth;
        // Use continuous pitch value (not quantized to MIDI notes)
        const continuousPitch = config.maxNote - ((pos.y - config.topMargin) / config.noteHeight);

        // Add point to contour (throttle to avoid too many points)
        setF0Contour(prev => {
          // Only add point if it's different enough from the last point
          const lastPoint = prev[prev.length - 1];
          if (!lastPoint || Math.abs(lastPoint.time - time) > 0.005 || Math.abs(lastPoint.note - continuousPitch) > 0.1) {
            return [...prev, { time, note: continuousPitch }];
          }
          return prev;
        });
      }
      return;
    }

    if (dragState) {
      if (dragState.type === 'select') {
        // Update selection box
        setSelectionBox(prev => ({
          ...prev,
          endX: pos.x,
          endY: pos.y
        }));
      } else if (dragState.type === 'move-multiple') {
        // Move all selected notes together
        const deltaX = pos.x - dragState.startX;
        const deltaY = pos.y - dragState.startY;
        const deltaTime = deltaX / config.gridWidth;
        const deltaNoteIndex = Math.round(deltaY / config.noteHeight);

        setNotes(prevNotes => {
          const newNotes = [...prevNotes];
          let movedCount = 0;
          dragState.selectedIndices.forEach(idx => {
            const startPos = dragState.initialPositions[idx];
            if (!startPos) {
              console.error(`❌ Missing initial position for note index ${idx}`);
              return;
            }
            const rawTime = startPos.time + deltaTime;
            const minSnapInterval = 1 / getSnapSubdivision(zoomX); // Minimum snap interval
            const newTime = Math.max(0, Math.min(config.duration - minSnapInterval, rawTime));
            const newNote = Math.max(config.minNote, Math.min(config.maxNote, startPos.note - deltaNoteIndex));
            const snappedTime = snapTime(newTime);

            newNotes[idx] = {
              ...newNotes[idx],
              time: snappedTime,
              note: newNote
            };
            movedCount++;
          });
          if (movedCount > 0) {
            console.log(`🎵 Moving ${movedCount} notes (deltaTime: ${deltaTime.toFixed(2)}, deltaNoteIndex: ${deltaNoteIndex})`);
          }
          return newNotes;
        });
      } else if (dragState.type === 'move') {
        // Move note
        const deltaX = pos.x - dragState.startX;
        const deltaY = pos.y - dragState.startY;
        const deltaTime = deltaX / config.gridWidth;
        const deltaNoteIndex = Math.round(deltaY / config.noteHeight);

        const rawTime = dragState.startTime + deltaTime;
        const minSnapInterval = 1 / getSnapSubdivision(zoomX); // Minimum snap interval
        const newTime = Math.max(0, Math.min(config.duration - minSnapInterval, rawTime));
        const newNote = Math.max(config.minNote, Math.min(config.maxNote, dragState.startNote - deltaNoteIndex));

        // Snap to current grid subdivision based on zoom level
        const snappedTime = snapTime(newTime);

        // Play note sound when pitch changes
        if (newNote !== dragState.startNote && playerInitialized) {
          // Update the last played note to avoid playing repeatedly
          if (!dragState.lastPlayedNote || dragState.lastPlayedNote !== newNote) {
            midiPlayer.resume().then(() => {
              midiPlayer.playNote(newNote, 0.5, 0.2); // Quieter and shorter during drag
            });
            // Store the last played note in drag state
            dragState.lastPlayedNote = newNote;
          }
        }

        setNotes(prevNotes => {
          const newNotes = [...prevNotes];
          newNotes[dragState.noteIndex] = {
            ...newNotes[dragState.noteIndex],
            time: snappedTime,
            note: newNote
          };
          return newNotes;
        });
      } else if (dragState.type === 'resize') {
        // Resize note (left or right handle)
        const deltaX = pos.x - dragState.startX;
        const deltaDuration = deltaX / config.gridWidth;
        const minSnapInterval = 1 / getSnapSubdivision(zoomX); // Minimum snap interval

        if (dragState.isLeftHandle) {
          // Left handle: adjust start time and duration
          const deltaTime = deltaX / config.gridWidth;
          const newTime = snapTime(dragState.startTime + deltaTime);
          const newDuration = snapTime(Math.max(minSnapInterval, dragState.startDuration - deltaTime));

          setNotes(prevNotes => {
            const newNotes = [...prevNotes];
            newNotes[dragState.noteIndex] = {
              ...newNotes[dragState.noteIndex],
              time: Math.max(0, newTime), // Don't allow negative time
              duration: newDuration
            };
            return newNotes;
          });
        } else {
          // Right handle: adjust duration only
          const rawDuration = dragState.startDuration + deltaDuration;
          const newDuration = Math.max(minSnapInterval, rawDuration);
          const snappedDuration = snapTime(newDuration);

          setNotes(prevNotes => {
            const newNotes = [...prevNotes];
            newNotes[dragState.noteIndex] = {
              ...newNotes[dragState.noteIndex],
              duration: snappedDuration
            };
            return newNotes;
          });
        }
      } else if (dragState.type === 'resize-multiple') {
        // Resize all selected notes
        const deltaX = pos.x - dragState.startX;
        const deltaDuration = deltaX / config.gridWidth;
        const minSnapInterval = 1 / getSnapSubdivision(zoomX); // Minimum snap interval

        setNotes(prevNotes => {
          const newNotes = [...prevNotes];
          let resizedCount = 0;
          dragState.selectedIndices.forEach(idx => {
            const startDuration = dragState.initialDurations[idx];
            if (!startDuration) {
              console.error(`❌ Missing initial duration for note index ${idx}`);
              return;
            }
            const rawDuration = startDuration + deltaDuration;
            const newDuration = Math.max(minSnapInterval, rawDuration);
            const snappedDuration = snapTime(newDuration);

            newNotes[idx] = {
              ...newNotes[idx],
              duration: snappedDuration
            };
            resizedCount++;
          });
          if (resizedCount > 0) {
            console.log(`🎵 Resizing ${resizedCount} notes (deltaDuration: ${deltaDuration.toFixed(2)})`);
          }
          return newNotes;
        });
      }
    } else {
      // Update hover state
      const result = findNoteAtPos(pos.x, pos.y);
      setHoveredNote(result ? result.index : null);

      // Update cursor
      const canvas = canvasRef.current;
      if (canvas) {
        if (chartMode === 'f0') {
          // F0 mode - always pencil cursor
          canvas.style.cursor = 'crosshair';
        } else if (isSelectMode) {
          // Select mode cursor
          canvas.style.cursor = result ? 'pointer' : 'crosshair';
        } else {
          // Normal mode cursor
          if (result) {
            canvas.style.cursor = result.isResizeHandle ? 'ew-resize' : 'move';
          } else {
            canvas.style.cursor = 'crosshair';
          }
        }
      }
    }
  }, [dragState, config, getMousePos, findNoteAtPos, playerInitialized, isSelectMode, chartMode, isDrawingF0, canvasWidth, canvasHeight]);

  // Convert AudioBuffer to WAV Blob
  const audioBufferToWav = useCallback((audioBuffer) => {
    const numberOfChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numberOfChannels * bytesPerSample;

    const data = [];
    for (let i = 0; i < audioBuffer.numberOfChannels; i++) {
      data.push(audioBuffer.getChannelData(i));
    }

    const dataLength = audioBuffer.length * numberOfChannels * bytesPerSample;
    const buffer = new ArrayBuffer(44 + dataLength);
    const view = new DataView(buffer);

    // Write WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + dataLength, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true); // Subchunk1Size
    view.setUint16(20, format, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, dataLength, true);

    // Write interleaved audio data
    let offset = 44;
    for (let i = 0; i < audioBuffer.length; i++) {
      for (let channel = 0; channel < numberOfChannels; channel++) {
        const sample = Math.max(-1, Math.min(1, data[channel][i]));
        view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7FFF, true);
        offset += 2;
      }
    }

    return new Blob([buffer], { type: 'audio/wav' });
  }, []);

  // Convert F0 contour to sine wave audio
  const convertF0ToSineWave = useCallback(async (contour) => {
    if (!contour || contour.length === 0) return null;

    console.log(`🎵 Converting F0 contour (${contour.length} points) to sine wave...`);

    // Calculate duration from last point
    const duration = Math.max(...contour.map(p => p.time));
    const sampleRate = 44100;
    const numberOfSamples = Math.ceil(duration * sampleRate);

    // Create offline context
    const offlineContext = new OfflineAudioContext(1, numberOfSamples, sampleRate);

    // Create oscillator
    const oscillator = offlineContext.createOscillator();
    oscillator.type = 'sine';

    // Create gain for amplitude envelope
    const oscGain = offlineContext.createGain();
    oscGain.gain.setValueAtTime(0.3, 0); // Constant amplitude
    oscillator.connect(oscGain);
    oscGain.connect(offlineContext.destination);

    // Set initial frequency (from first point)
    const midiToFreq = (midi) => 440 * Math.pow(2, (midi - 69) / 12);
    const initialFreq = midiToFreq(contour[0].note);
    oscillator.frequency.setValueAtTime(initialFreq, 0);

    // Schedule frequency changes for each point
    for (let i = 0; i < contour.length; i++) {
      const point = contour[i];
      const freq = midiToFreq(point.note);
      const time = point.time;

      // Use linear ramp for smooth pitch glides
      oscillator.frequency.linearRampToValueAtTime(freq, time);
    }

    // Start and stop oscillator
    oscillator.start(0);
    oscillator.stop(duration);

    // Render to audio buffer
    const renderedBuffer = await offlineContext.startRendering();
    console.log(`✅ F0 sine wave generated: ${duration.toFixed(2)}s`);

    return renderedBuffer;
  }, []);

  // Mouse up - end drag or resize
  const handleMouseUp = useCallback((e) => {
    // F0 MODE - Finish drawing pitch contour
    if (isDrawingF0 && chartMode === 'f0') {
      setIsDrawingF0(false);
      console.log(`✅ F0 contour drawn with ${f0Contour.length} points`);

      // Convert to sine wave and add to track
      if (f0Contour.length > 1 && selectedTrack) {
        convertF0ToSineWave(f0Contour).then(audioBuffer => {
          if (!audioBuffer) return;

          // Convert AudioBuffer to Blob
          const wavBlob = audioBufferToWav(audioBuffer);
          const audioUrl = URL.createObjectURL(wavBlob);

          // Add F0 audio to the track
          dispatch({
            type: 'UPDATE_TRACK',
            payload: {
              busId: state.buses.find(b => b.tracks.some(t => t.id === selectedTrack.id))?.id,
              trackId: selectedTrack.id,
              updates: {
                f0Audio: audioUrl,
                f0Contour: f0Contour,
                duration: Math.max(selectedTrack.duration || 0, audioBuffer.duration)
              }
            }
          });

          console.log(`✅ F0 audio added to track: ${audioBuffer.duration.toFixed(2)}s`);
        }).catch(err => {
          console.error('❌ Failed to convert F0 to sine wave:', err);
        });
      }

      return;
    }

    if (dragState) {
      // If we just resized a note, save its duration for next note placement
      if (dragState.type === 'resize' && dragState.noteIndex !== undefined) {
        const resizedNote = notes[dragState.noteIndex];
        if (resizedNote) {
          setLastNoteDuration(resizedNote.duration);
          console.log(`✅ Note resized - saved duration ${resizedNote.duration.toFixed(2)} for next note`);
        }
      } else if (dragState.type === 'resize-multiple' && dragState.selectedIndices && dragState.selectedIndices.length > 0) {
        // Multiple notes resized - save duration from first note
        const firstResizedNote = notes[dragState.selectedIndices[0]];
        if (firstResizedNote) {
          setLastNoteDuration(firstResizedNote.duration);
          console.log(`✅ ${dragState.selectedIndices.length} notes resized - saved duration ${firstResizedNote.duration.toFixed(2)} for next note`);
        }
      } else if (dragState.type === 'select' && selectionBox) {
        // Complete selection box - find all notes inside the box
        const { startX, startY, endX, endY } = selectionBox;
        const boxLeft = Math.min(startX, endX);
        const boxRight = Math.max(startX, endX);
        const boxTop = Math.min(startY, endY);
        const boxBottom = Math.max(startY, endY);

        const selectedIndices = [];
        notes.forEach((note, index) => {
          const noteX = config.leftMargin + (note.time * config.gridWidth);
          const noteY = config.topMargin + ((config.maxNote - note.note) * config.noteHeight);
          const noteWidth = note.duration * config.gridWidth;
          const noteHeight = config.noteHeight - 2;

          // Check if note intersects with selection box
          if (noteX + noteWidth >= boxLeft && noteX <= boxRight &&
              noteY + noteHeight >= boxTop && noteY <= boxBottom) {
            selectedIndices.push(index);
          }
        });

        setSelectedNotes(selectedIndices);
        setSelectionBox(null);
        console.log(`✅ Selected ${selectedIndices.length} notes`);
      }
      setDragState(null);
    }
  }, [dragState, notes, selectionBox, config, isDrawingF0, chartMode, f0Contour, selectedTrack, convertF0ToSineWave, audioBufferToWav, dispatch, state.buses]);

  // Double click to delete note
  const handleDoubleClick = useCallback((e) => {
    const pos = getMousePos(e);
    const result = findNoteAtPos(pos.x, pos.y);

    if (result) {
      const note = notes[result.index];
      setNotes(prevNotes => prevNotes.filter((_, i) => i !== result.index));
      console.log(`🗑️ Removed note: ${getNoteNameFromMidi(note.note)} at beat ${note.time}`);
    }
  }, [notes, getMousePos, findNoteAtPos]);

  // Export MIDI file
  const handleExportMIDI = useCallback(() => {
    if (notes.length === 0) {
      alert('Please add some notes before exporting!');
      return;
    }

    // Create a simple MIDI file format
    // This is a basic implementation - you may want to use a library like 'midi-writer-js' for production
    const midiData = createMIDIFile(notes);

    // Create blob and download
    const blob = new Blob([midiData], { type: 'audio/midi' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `composition-${Date.now()}.mid`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`✅ Exported ${notes.length} notes to MIDI file`);
  }, [notes]);

  // Play MIDI notes
  const handlePlayback = useCallback(async () => {
    if (notes.length === 0) {
      alert('Please add some notes before playing!');
      return;
    }

    if (!playerInitialized) {
      alert('MIDI player is still loading...');
      return;
    }

    if (isPlaying) {
      // Stop playback
      midiPlayer.stopAll();
      setIsPlaying(false);
      console.log('⏹️ Stopped playback');
      return;
    }

    // Start playback
    setIsPlaying(true);

    // Use timeline BPM (state.bpm) instead of track's internal tempo
    console.log('🔍 DEBUG - state.bpm:', state.bpm);
    console.log('🔍 DEBUG - full state:', state);
    const playbackBpm = state.bpm || 120;
    console.log(`▶️ Playing ${notes.length} notes at ${playbackBpm} BPM (timeline tempo)`);

    // Resume audio context (required for user interaction)
    await midiPlayer.resume();

    // Schedule all notes (times are already in seconds from MIDI parser)
    midiPlayer.scheduleNotes(notes, playbackBpm);

    // Calculate total duration in seconds (note times are already in seconds!)
    const maxEndTime = Math.max(...notes.map(n => n.time + n.duration));

    // Auto-stop after playback finishes
    setTimeout(() => {
      setIsPlaying(false);
      console.log('✅ Playback finished');
    }, (maxEndTime + 0.5) * 1000);
  }, [notes, playerInitialized, isPlaying, state.bpm]);

  // Clear all notes
  const handleClear = useCallback(() => {
    if (notes.length === 0) return;
    if (window.confirm('Clear all notes?')) {
      setNotes([]);
      console.log('🗑️ Cleared all notes');
    }
  }, [notes]);

  // Toggle zoom mode
  const handleToggleZoomMode = useCallback(() => {
    setZoomMode(prev => prev === 'horizontal' ? 'vertical' : 'horizontal');
  }, []);

  // Zoom in based on current mode
  const handleZoomIn = useCallback(() => {
    if (zoomMode === 'horizontal') {
      setZoomX(prev => Math.min(prev + 0.25, 5.0));
    } else {
      setZoomY(prev => Math.min(prev + 0.25, 3.0));
    }
  }, [zoomMode]);

  // Zoom out based on current mode
  const handleZoomOut = useCallback(() => {
    if (zoomMode === 'horizontal') {
      setZoomX(prev => Math.max(prev - 0.25, 0.5));
    } else {
      setZoomY(prev => Math.max(prev - 0.25, 0.5));
    }
  }, [zoomMode]);

  // Mouse wheel zoom (two-finger scroll for horizontal only)
  const handleWheel = useCallback((e) => {
    // Only handle two-finger scroll (Ctrl key is set on trackpad pinch/scroll)
    if (!e.ctrlKey) return;

    e.preventDefault();
    e.stopPropagation(); // Prevent event from bubbling to page
    const delta = e.deltaY > 0 ? -0.1 : 0.1;

    // Two-finger scroll = Horizontal zoom only
    setZoomX(prev => Math.max(0.5, Math.min(5.0, prev + delta)));
  }, []);

  // Transpose selected notes up by one octave (12 semitones)
  const handleTransposeUp = useCallback(() => {
    if (selectedNotes.length === 0) {
      console.log('⚠️ No notes selected for transpose');
      return;
    }

    setNotes(prevNotes => {
      const newNotes = [...prevNotes];
      selectedNotes.forEach(idx => {
        const newNote = Math.min(127, newNotes[idx].note + 12);
        newNotes[idx] = {
          ...newNotes[idx],
          note: newNote
        };
      });
      return newNotes;
    });

    console.log(`🎵 Transposed ${selectedNotes.length} notes up by 1 octave`);
  }, [selectedNotes]);

  // Transpose selected notes down by one octave (12 semitones)
  const handleTransposeDown = useCallback(() => {
    if (selectedNotes.length === 0) {
      console.log('⚠️ No notes selected for transpose');
      return;
    }

    setNotes(prevNotes => {
      const newNotes = [...prevNotes];
      selectedNotes.forEach(idx => {
        const newNote = Math.max(0, newNotes[idx].note - 12);
        newNotes[idx] = {
          ...newNotes[idx],
          note: newNote
        };
      });
      return newNotes;
    });

    console.log(`🎵 Transposed ${selectedNotes.length} notes down by 1 octave`);
  }, [selectedNotes]);

  // Generate MIDI using melody generator (supports multiple modes)
  const handleGenerateMIDI = useCallback(async () => {
    console.log('🎹 Generating MIDI with settings:', generateSettings);

    try {
      let response;
      let data;

      if (generateSettings.mode === 'chords') {
        // Chord rendering mode - uses /api/render-chords endpoint
        // Convert chord string to beat map (one chord per bar)
        const chordsList = generateSettings.chords.split(',').map(c => c.trim());
        const beatsPerBar = 4;
        const chordsMap = {};
        for (let i = 0; i < generateSettings.bars; i++) {
          const chordIdx = i % chordsList.length;
          chordsMap[i * beatsPerBar] = chordsList[chordIdx];
        }

        response = await fetch('/api/render-chords', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            chords: chordsMap,
            bpm: generateSettings.tempo,
            duration: generateSettings.bars * beatsPerBar,
            voicing: generateSettings.voicing,
            rhythm: generateSettings.rhythm,
            style: generateSettings.chordStyle
          })
        });

        if (!response.ok) {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        data = await response.json();

        // Chord rendering returns a file path - notify user
        if (data.file_path) {
          console.log(`✅ Chord MIDI generated: ${data.file_path}`);
          alert(`Chord MIDI generated!\nDownload: ${data.file_path}\n\nNote: Chord mode generates a MIDI file for download rather than notes to display.`);
        }
        return;

      } else {
        // Melody generation modes (basic, genre, context)
        response = await fetch('/api/generate-melody', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            mode: generateSettings.mode,
            key: generateSettings.key,
            chords: generateSettings.chords,
            bars: generateSettings.bars,
            minNote: generateSettings.minNote,
            maxNote: generateSettings.maxNote,
            chromatic: generateSettings.chromatic,
            tempo: generateSettings.tempo,
            seed: generateSettings.seed,
            // Genre mode
            genre: generateSettings.genre,
            style: generateSettings.style,
            // Context mode
            role: generateSettings.role,
            matchDensity: generateSettings.matchDensity,
            matchStyle: generateSettings.matchStyle
          })
        });
      }

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
      }

      data = await response.json();

      if (data.notes && data.notes.length > 0) {
        console.log(`✅ Generated ${data.notes.length} notes (mode: ${data.mode})`);
        console.log(`   Algorithm: ${data.metadata?.algorithm || 'unknown'}`);

        // Convert backend format to MIDIChart format
        // Backend: { pitch, start (ticks), duration (ticks), velocity }
        // MIDIChart: { note (pitch), time (seconds), duration (seconds), velocity }
        const ticksPerBeat = 480;
        const secondsPerBeat = 60 / generateSettings.tempo;
        const ticksToSeconds = (ticks) => (ticks / ticksPerBeat) * secondsPerBeat;

        const convertedNotes = data.notes.map(n => ({
          note: n.pitch,
          time: ticksToSeconds(n.start),
          duration: ticksToSeconds(n.duration),
          velocity: n.velocity || 80
        }));

        setNotes(convertedNotes);
        setOriginalNotes(JSON.parse(JSON.stringify(convertedNotes)));

        if (data.tempo) {
          setMidiTempo(data.tempo);
          dispatch({ type: 'SET_BPM', payload: data.tempo });
        }
      } else {
        alert('No notes were generated. Please check your settings.');
      }
    } catch (error) {
      console.error('❌ Failed to generate MIDI:', error);
      alert(`Failed to generate MIDI: ${error.message}`);
    }
  }, [generateSettings, dispatch]);

  return (
    <div className={styles.midiChartContainer}>
      <div className={styles.header}>
        {/* Left side: Chart Mode Toggle */}
        <div className={styles.leftControls}>
          <div className={styles.chartModeToggle}>
            <button
              className={`${styles.chartModeButton} ${chartMode === 'midi' ? styles.active : ''}`}
              onClick={() => setChartMode('midi')}
              title="MIDI Note Mode"
            >
              <i className="fa-solid fa-keyboard"></i>
            </button>
            <button
              className={`${styles.chartModeButton} ${chartMode === 'f0' ? styles.active : ''}`}
              onClick={() => setChartMode('f0')}
              title="F0 Pitch Contour Mode"
            >
              <i className="fa-solid fa-wave-square"></i>
            </button>
            <button
              className={`${styles.chartModeButton} ${chartMode === 'score' ? styles.active : ''}`}
              onClick={() => setChartMode('score')}
              title="Score View (Coming Soon)"
              disabled
            >
              <i className="fa-solid fa-music"></i>
            </button>
          </div>

          {chartMode === 'midi' && (
            <>
              <button
                className={styles.generateButton}
                onClick={handleGenerateMIDI}
                title="Generate MIDI from settings"
              >
                <i className="fa-solid fa-wand-magic-sparkles"></i>
                Generate MIDI
              </button>
              <button
                className={`${styles.settingsToggle} ${showGenerateSettings ? styles.active : ''}`}
                onClick={() => setShowGenerateSettings(!showGenerateSettings)}
                title="Toggle generation settings"
              >
                <i className="fa-solid fa-gear"></i>
              </button>
            </>
          )}
        </div>

        {/* Center: Main controls */}
        <div className={styles.controls}>
          {/* Tool Mode Toggle - only show in MIDI mode */}
          {chartMode === 'midi' && (
            <>
              <div className={styles.toolModeToggle}>
                <button
                  className={`${styles.toolButton} ${toolMode === 'draw' ? styles.active : ''}`}
                  onClick={() => setToolMode('draw')}
                  title="Draw Mode (Place and edit notes)"
                >
                  <i className="fa-solid fa-pencil"></i>
                </button>
                <button
                  className={`${styles.toolButton} ${toolMode === 'select' ? styles.active : ''}`}
                  onClick={() => setToolMode('select')}
                  title="Select Mode (Select and move notes)"
                >
                  <i className="fa-solid fa-arrow-pointer"></i>
                </button>
              </div>
              <span className={styles.controlSeparator}>|</span>
            </>
          )}

          {/* F0 Mode Info */}
          {chartMode === 'f0' && (
            <div className={styles.f0Info}>
              <i className="fa-solid fa-pencil"></i>
              <span>Draw pitch contour</span>
            </div>
          )}

          {chartMode === 'midi' && (
            <>
              <button
                className={styles.exportButton}
                onClick={handleExportMIDI}
                disabled={notes.length === 0}
                title="Download MIDI file"
              >
                <i className="fa-solid fa-download"></i>
              </button>
              <span className={styles.controlSeparator}>|</span>
            </>
          )}

          {chartMode === 'f0' && (
            <>
              <button
                className={styles.exportButton}
                onClick={() => {
                  // TODO: Export F0 contour as audio
                  console.log('Export F0 contour as audio');
                  alert('F0 audio export coming soon!');
                }}
                disabled={f0Contour.length === 0}
                title="Export F0 as Audio"
              >
                <i className="fa-solid fa-download"></i>
              </button>
              <span className={styles.controlSeparator}>|</span>
            </>
          )}
          <span className={styles.controlSeparator}>|</span>
          <div className={styles.zoomControls}>
            <button
              className={styles.zoomModeButton}
              onClick={handleToggleZoomMode}
              title={`Switch to ${zoomMode === 'horizontal' ? 'Vertical' : 'Horizontal'} Zoom`}
            >
              <i className={`fa-solid ${zoomMode === 'horizontal' ? 'fa-left-right' : 'fa-up-down'}`} aria-hidden="true"></i>
            </button>
            <button
              className={styles.zoomButton}
              onClick={handleZoomOut}
              disabled={(zoomMode === 'horizontal' && zoomX <= 0.5) || (zoomMode === 'vertical' && zoomY <= 0.5)}
              title={`Zoom Out (${zoomMode === 'horizontal' ? 'Horizontal' : 'Vertical'})`}
            >
              <i className="fa-solid fa-minus" aria-hidden="true"></i>
            </button>
            <button
              className={styles.zoomButton}
              onClick={handleZoomIn}
              disabled={(zoomMode === 'horizontal' && zoomX >= 5.0) || (zoomMode === 'vertical' && zoomY >= 3.0)}
              title={`Zoom In (${zoomMode === 'horizontal' ? 'Horizontal' : 'Vertical'})`}
            >
              <i className="fa-solid fa-plus" aria-hidden="true"></i>
            </button>
          </div>
          {chartMode === 'midi' && (
            <>
              <span className={styles.controlSeparator}>|</span>
              <div className={styles.transposeControls}>
            <span className={styles.transposeLabel}>Octave:</span>
            <button
              className={styles.transposeButton}
              onClick={handleTransposeDown}
              disabled={selectedNotes.length === 0}
              title="Transpose selected notes down by 1 octave"
            >
              <i className="fa-solid fa-arrow-down"></i>
            </button>
            <button
              className={styles.transposeButton}
              onClick={handleTransposeUp}
              disabled={selectedNotes.length === 0}
              title="Transpose selected notes up by 1 octave"
            >
              <i className="fa-solid fa-arrow-up"></i>
            </button>
          </div>
            </>
          )}
        </div>
      </div>

      {/* Expandable Generate Settings Panel */}
      {showGenerateSettings && (
        <div className={styles.settingsPanel}>
          {/* Mode Selector */}
          <div className={styles.modeSelector}>
            <button
              className={`${styles.modeButton} ${generateSettings.mode === 'basic' ? styles.active : ''}`}
              onClick={() => setGenerateSettings({ ...generateSettings, mode: 'basic' })}
              title="Basic melody generation using target-note technique"
            >
              <i className="fa-solid fa-music"></i> Basic
            </button>
            <button
              className={`${styles.modeButton} ${generateSettings.mode === 'genre' ? styles.active : ''}`}
              onClick={() => setGenerateSettings({ ...generateSettings, mode: 'genre' })}
              title="Genre-specific melody generation (Jazz, Blues, Funk, etc.)"
            >
              <i className="fa-solid fa-guitar"></i> Genre
            </button>
            <button
              className={`${styles.modeButton} ${generateSettings.mode === 'context' ? styles.active : ''}`}
              onClick={() => setGenerateSettings({ ...generateSettings, mode: 'context' })}
              title="Context-aware generation (matches existing arrangement)"
            >
              <i className="fa-solid fa-layer-group"></i> Context
            </button>
            <button
              className={`${styles.modeButton} ${generateSettings.mode === 'chords' ? styles.active : ''}`}
              onClick={() => setGenerateSettings({ ...generateSettings, mode: 'chords' })}
              title="Chord progression rendering with voicings"
            >
              <i className="fa-solid fa-cubes"></i> Chords
            </button>
          </div>

          <div className={styles.settingsGrid}>
            {/* Common Settings */}
            <div className={styles.settingGroup}>
              <label className={styles.settingLabel}>Key</label>
              <input
                type="text"
                className={styles.settingInput}
                value={generateSettings.key}
                onChange={(e) => setGenerateSettings({ ...generateSettings, key: e.target.value })}
                placeholder="C minor"
              />
            </div>

            <div className={styles.settingGroup}>
              <label className={styles.settingLabel}>Chords</label>
              <input
                type="text"
                className={styles.settingInput}
                value={generateSettings.chords}
                onChange={(e) => setGenerateSettings({ ...generateSettings, chords: e.target.value })}
                placeholder="Cm7,G7,Cm7,G7"
              />
            </div>

            <div className={styles.settingGroup}>
              <label className={styles.settingLabel}>Bars</label>
              <input
                type="number"
                className={styles.settingInput}
                value={generateSettings.bars}
                onChange={(e) => setGenerateSettings({ ...generateSettings, bars: parseInt(e.target.value) })}
                min="1"
                max="32"
              />
            </div>

            <div className={styles.settingGroup}>
              <label className={styles.settingLabel}>Tempo (BPM)</label>
              <input
                type="number"
                className={styles.settingInput}
                value={generateSettings.tempo}
                onChange={(e) => setGenerateSettings({ ...generateSettings, tempo: parseInt(e.target.value) })}
                min="40"
                max="240"
              />
            </div>

            {/* Melody-specific settings (Basic, Genre, Context modes) */}
            {generateSettings.mode !== 'chords' && (
              <>
                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Min Note</label>
                  <input
                    type="number"
                    className={styles.settingInput}
                    value={generateSettings.minNote}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, minNote: parseInt(e.target.value) })}
                    min="0"
                    max="127"
                  />
                </div>

                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Max Note</label>
                  <input
                    type="number"
                    className={styles.settingInput}
                    value={generateSettings.maxNote}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, maxNote: parseInt(e.target.value) })}
                    min="0"
                    max="127"
                  />
                </div>
              </>
            )}

            {/* Basic mode: Chromatic setting */}
            {generateSettings.mode === 'basic' && (
              <div className={styles.settingGroup}>
                <label className={styles.settingLabel}>Chromatic %</label>
                <input
                  type="number"
                  className={styles.settingInput}
                  value={generateSettings.chromatic * 100}
                  onChange={(e) => setGenerateSettings({ ...generateSettings, chromatic: parseFloat(e.target.value) / 100 })}
                  min="0"
                  max="100"
                  step="5"
                />
              </div>
            )}

            {/* Genre mode: Genre and Style selectors */}
            {generateSettings.mode === 'genre' && (
              <>
                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Genre</label>
                  <select
                    className={styles.settingSelect}
                    value={generateSettings.genre}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, genre: e.target.value })}
                  >
                    <option value="jazz">Jazz</option>
                    <option value="blues">Blues</option>
                    <option value="funk">Funk</option>
                    <option value="pop">Pop</option>
                    <option value="rock">Rock</option>
                    <option value="electronic">Electronic</option>
                    <option value="classical">Classical</option>
                    <option value="hip-hop">Hip-Hop</option>
                    <option value="r&b">R&B</option>
                    <option value="country">Country</option>
                    <option value="reggae">Reggae</option>
                    <option value="gospel">Gospel</option>
                    <option value="latin">Latin</option>
                    <option value="arabic">Arabic</option>
                    <option value="indian">Indian</option>
                  </select>
                </div>

                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Style</label>
                  <select
                    className={styles.settingSelect}
                    value={generateSettings.style}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, style: e.target.value })}
                  >
                    <option value="bebop">Bebop</option>
                    <option value="swing">Swing</option>
                    <option value="ballad">Ballad</option>
                    <option value="funk">Funk</option>
                    <option value="blues">Blues</option>
                    <option value="modal">Modal</option>
                    <option value="fusion">Fusion</option>
                  </select>
                </div>
              </>
            )}

            {/* Context mode: Role selector */}
            {generateSettings.mode === 'context' && (
              <div className={styles.settingGroup}>
                <label className={styles.settingLabel}>Role</label>
                <select
                  className={styles.settingSelect}
                  value={generateSettings.role}
                  onChange={(e) => setGenerateSettings({ ...generateSettings, role: e.target.value })}
                >
                  <option value="melody">Melody</option>
                  <option value="bass">Bass</option>
                  <option value="harmony">Harmony</option>
                  <option value="countermelody">Counter-melody</option>
                </select>
              </div>
            )}

            {/* Chords mode: Voicing, Rhythm, Style */}
            {generateSettings.mode === 'chords' && (
              <>
                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Voicing</label>
                  <select
                    className={styles.settingSelect}
                    value={generateSettings.voicing}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, voicing: e.target.value })}
                  >
                    <option value="random">Random</option>
                    <option value="close">Close</option>
                    <option value="open">Open</option>
                    <option value="drop2">Drop 2</option>
                    <option value="drop3">Drop 3</option>
                    <option value="shell">Shell</option>
                    <option value="spread">Spread</option>
                  </select>
                </div>

                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Rhythm</label>
                  <select
                    className={styles.settingSelect}
                    value={generateSettings.rhythm}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, rhythm: e.target.value })}
                  >
                    <option value="random">Random</option>
                    <option value="whole">Whole Notes</option>
                    <option value="half">Half Notes</option>
                    <option value="quarter">Quarter Notes</option>
                    <option value="eighth">Eighth Notes</option>
                    <option value="syncopated">Syncopated</option>
                    <option value="arpeggio">Arpeggio</option>
                    <option value="dotted">Dotted</option>
                  </select>
                </div>

                <div className={styles.settingGroup}>
                  <label className={styles.settingLabel}>Style</label>
                  <select
                    className={styles.settingSelect}
                    value={generateSettings.chordStyle}
                    onChange={(e) => setGenerateSettings({ ...generateSettings, chordStyle: e.target.value })}
                  >
                    <option value="random">Random</option>
                    <option value="block">Block</option>
                    <option value="arpeggio">Arpeggio</option>
                  </select>
                </div>
              </>
            )}

            {/* Seed (all modes) */}
            <div className={styles.settingGroup}>
              <label className={styles.settingLabel}>Seed</label>
              <input
                type="number"
                className={styles.settingInput}
                value={generateSettings.seed || ''}
                onChange={(e) => setGenerateSettings({ ...generateSettings, seed: e.target.value ? parseInt(e.target.value) : null })}
                placeholder="Random"
              />
            </div>
          </div>

          {/* Mode-specific help text */}
          <div className={styles.settingsHelp}>
            <i className="fa-solid fa-circle-info"></i>
            <span>
              {generateSettings.mode === 'basic' &&
                'Basic mode uses target-note technique with Berklee chord-scale theory. Great for jazz improvisation lines.'}
              {generateSettings.mode === 'genre' &&
                'Genre mode generates melodies with style-specific characteristics. Choose a genre and style for authentic results.'}
              {generateSettings.mode === 'context' &&
                'Context mode analyzes existing arrangement and generates complementary parts. Choose a role (melody, bass, harmony).'}
              {generateSettings.mode === 'chords' &&
                'Chords mode renders full chord progressions with various voicings and rhythmic patterns. Downloads as MIDI file.'}
            </span>
          </div>
        </div>
      )}

      <div ref={canvasWrapperRef} className={styles.canvasWrapper} onWheel={handleWheel}>
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
          onDoubleClick={handleDoubleClick}
          className={styles.canvas}
        />
      </div>
    </div>
  );
};

/**
 * Create a simple MIDI file from notes
 * This is a basic MIDI file writer - for production, consider using a library
 */
function createMIDIFile(notes) {
  // MIDI file structure: Header + Track
  const header = new Uint8Array([
    // MThd (MIDI header chunk)
    0x4D, 0x54, 0x68, 0x64, // "MThd"
    0x00, 0x00, 0x00, 0x06, // Header length (6 bytes)
    0x00, 0x00, // Format type (0 = single track)
    0x00, 0x01, // Number of tracks (1)
    0x00, 0x60  // Ticks per quarter note (96)
  ]);

  // Build track events
  const events = [];
  const ticksPerBeat = 96;

  // Sort notes by time
  const sortedNotes = [...notes].sort((a, b) => a.time - b.time);

  // Add note on/off events
  sortedNotes.forEach(note => {
    const startTick = Math.round(note.time * ticksPerBeat);
    const endTick = Math.round((note.time + note.duration) * ticksPerBeat);

    // Note ON event
    events.push({
      tick: startTick,
      data: [0x90, note.note, note.velocity] // Note ON, channel 0
    });

    // Note OFF event
    events.push({
      tick: endTick,
      data: [0x80, note.note, 0x40] // Note OFF, channel 0
    });
  });

  // Sort events by tick
  events.sort((a, b) => a.tick - b.tick);

  // Build track data with delta times
  const trackData = [];
  let lastTick = 0;

  events.forEach(event => {
    const delta = event.tick - lastTick;
    trackData.push(...encodeVariableLength(delta));
    trackData.push(...event.data);
    lastTick = event.tick;
  });

  // End of track
  trackData.push(0x00, 0xFF, 0x2F, 0x00);

  // MTrk chunk
  const track = new Uint8Array([
    // MTrk (MIDI track chunk)
    0x4D, 0x54, 0x72, 0x6B, // "MTrk"
    ...intToBytes(trackData.length, 4), // Track length
    ...trackData
  ]);

  // Combine header and track
  const midiFile = new Uint8Array(header.length + track.length);
  midiFile.set(header, 0);
  midiFile.set(track, header.length);

  return midiFile;
}

// Helper: Encode variable-length quantity (for MIDI delta times)
function encodeVariableLength(value) {
  const bytes = [];
  bytes.push(value & 0x7F);
  value >>= 7;
  while (value > 0) {
    bytes.unshift((value & 0x7F) | 0x80);
    value >>= 7;
  }
  return bytes;
}

// Helper: Convert integer to byte array
function intToBytes(value, numBytes) {
  const bytes = [];
  for (let i = numBytes - 1; i >= 0; i--) {
    bytes.push((value >> (i * 8)) & 0xFF);
  }
  return bytes;
}

export default MIDIChart;
