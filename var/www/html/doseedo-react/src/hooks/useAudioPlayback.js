import { useEffect, useRef, useCallback } from 'react';
import midiPlayer from '../utils/midiPlayer';
import tunaFX from '../services/tunaFX';
import { isMaskPlaybackReady, setGains as setMaskGains, setActiveStem, seek as maskSeek, play as maskPlay, stop as maskStop } from '../services/maskPlayback';

// Global audio buffer cache (shared across all instances)
const audioBufferCache = new Map();

/**
 * Custom hook for managing audio playback across multiple tracks
 * Integrates with global state via dispatch
 */
export function useAudioPlayback(tracks, isPlaying, dispatch, totalDuration = 10, currentPlayheadPosition = 0, bpm = 120, masterGain = 0.8) {
  const audioContextRef = useRef(null);
  const masterGainNodeRef = useRef(null); // Master gain node for overall volume control
  const sourceNodesRef = useRef([]);
  const gainNodesRef = useRef(new Map()); // Map of trackId -> gainNode for real-time updates
  const startTimeRef = useRef(0);
  const pauseTimeRef = useRef(0);
  const animationFrameRef = useRef(null);
  const isSeekingRef = useRef(false);
  const lastKnownPlayheadRef = useRef(0);
  const isPlayingRef = useRef(isPlaying); // Track isPlaying in ref for animation loop
  const tracksRef = useRef(tracks); // Store tracks in ref to avoid recreating play callback
  const hasStartedPlaybackRef = useRef(false); // Track if playback has actually started

  // Initialize audio context, Tuna FX, and MIDI player
  useEffect(() => {
    // Ensure any previous Tuna FX is destroyed before creating new context
    tunaFX.destroy();

    audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)();
    // Expose globally so mask playback can use the same context
    window.__doseedo_audioCtx = audioContextRef.current;

    // Initialize Tuna FX chain with new context
    tunaFX.initialize(audioContextRef.current);

    // Create master gain node and connect FX output to it
    masterGainNodeRef.current = audioContextRef.current.createGain();
    masterGainNodeRef.current.gain.value = masterGain;

    // Signal chain: tracks → FX bus input → [effects] → FX bus output → master gain → destination
    const fxOutput = tunaFX.getFXBusOutput();
    if (fxOutput) {
      fxOutput.connect(masterGainNodeRef.current);
    }
    masterGainNodeRef.current.connect(audioContextRef.current.destination);

    console.log(`🎚️ Master gain initialized: ${(masterGain * 100).toFixed(0)}%`);
    console.log('🎛️ Signal chain: Tracks → FX Bus → Reverb → Delay → Chorus → Compressor → Filter → Phaser → Master → Output');

    // Initialize MIDI player
    midiPlayer.initialize().catch(err => {
      console.error('Failed to initialize MIDI player:', err);
    });

    return () => {
      tunaFX.destroy();
      midiPlayer.stopAll();
      if (audioContextRef.current) {
        audioContextRef.current.close();
        audioContextRef.current = null;
      }
    };
  }, []);

  // Keep refs in sync with props
  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    tracksRef.current = tracks;
  }, [tracks]);

  // Update gain nodes in real-time when track settings change (without restarting playback)
  useEffect(() => {
    if (!isPlaying || gainNodesRef.current.size === 0) return;

    const voTracks = tracks.vo || [];
    const musicTracks = tracks.music || [];
    const sfxTracks = tracks.sfx || [];
    const midiTracks = tracks.midi || [];
    const audioTracks = tracks.audio || [];

    // Include MIDI bus audio tracks (not MIDI type tracks)
    const midiAudioTracks = midiTracks.filter(t => t.type !== 'midi' && t.audioUrl);

    const allTracks = [...voTracks, ...musicTracks, ...sfxTracks, ...midiAudioTracks, ...audioTracks];

    // Check if any track has solo enabled
    const hasSoloTracks = allTracks.some(track => track.isSolo);
    const hasBusSolo = allTracks.some(t => t._busSolo);

    allTracks.forEach(track => {
      const gainNode = gainNodesRef.current.get(track.id);
      if (!gainNode) return;

      // Apply bus-level mute/solo
      const busMuted = track._busMuted || false;
      const busSolo = track._busSolo || false;
      const busGain = track._busGain || 1.0;

      // Determine if this track should play
      let shouldPlay = false;
      if (busMuted) {
        shouldPlay = false;
      } else if (hasBusSolo) {
        shouldPlay = busSolo;
      } else if (hasSoloTracks) {
        shouldPlay = track.isSolo;
      } else {
        shouldPlay = !track.isMuted;
      }

      // Calculate final gain
      const finalGain = shouldPlay ? (track.gain || 1.0) * busGain : 0;

      // Update gain node in real-time (smooth ramp to avoid clicks)
      gainNode.gain.setTargetAtTime(finalGain, audioContextRef.current.currentTime, 0.01);
    });

    // Also update mask playback gains for stem tracks
    if (isMaskPlaybackReady()) {
      const stemGains = {};
      let activeSolo = null;
      const stemTracks = allTracks.filter(t => t.metadata?.type === 'stem');
      const hasStemSolo = stemTracks.some(t => t.isSolo);
      for (const t of stemTracks) {
        const stemName = t.metadata?.stemType || t.metadata?.instrument;
        if (!stemName) continue;
        const busMuted = t._busMuted || false;
        const busGain = t._busGain || 1.0;
        if (busMuted || t.isMuted) {
          stemGains[stemName] = 0;
        } else if (hasStemSolo && !t.isSolo) {
          stemGains[stemName] = 0;
        } else {
          stemGains[stemName] = (t.gain || 1.0) * busGain;
          if (t.isSolo) activeSolo = stemName;
        }
      }
      setMaskGains(stemGains);
      setActiveStem(activeSolo);
    }
  }, [tracks, isPlaying]);

  // Sync internal pause time with external playhead position
  useEffect(() => {
    if (!isSeekingRef.current && !isPlaying) {
      // Only update when not seeking and not playing
      pauseTimeRef.current = currentPlayheadPosition;
      lastKnownPlayheadRef.current = currentPlayheadPosition;
      console.log('🔄 Synced pauseTime to external playhead:', currentPlayheadPosition.toFixed(2), 's');
    }
  }, [currentPlayheadPosition, isPlaying]);

  // Pre-load and cache all audio buffers when tracks change
  useEffect(() => {
    if (!audioContextRef.current) return;

    const audioContext = audioContextRef.current;
    const voTracks = tracks.vo || [];
    const musicTracks = tracks.music || [];
    const sfxTracks = tracks.sfx || [];
    const audioTracks = tracks.audio || [];
    const allTracks = [...voTracks, ...musicTracks, ...sfxTracks, ...audioTracks];

    // Pre-load all unique audio URLs (including drum hits)
    const audioUrls = [];
    allTracks.forEach(track => {
      if (track.isDrumTrack && track.drumHits) {
        // Add all drum hit audio URLs
        track.drumHits.forEach(hit => {
          if (hit.audioUrl) audioUrls.push(hit.audioUrl);
        });
      } else if (track.audioUrl) {
        audioUrls.push(track.audioUrl);
      }
      // Also preload F0 audio if available
      if (track.f0Audio) {
        audioUrls.push(track.f0Audio);
      }
    });
    const uniqueUrls = [...new Set(audioUrls.filter(Boolean))];

    uniqueUrls.forEach(async (url) => {
      if (!audioBufferCache.has(url)) {
        try {
          // Reuse the shared audioCacheService blob (in-flight Promise
          // dedupe + IndexedDB cache + memory cache). This avoids the
          // dual-fetch where useAudioPlayback and useWaveform each
          // hit the network for the same stem URL.
          const { fetchAudioWithCache } = await import('../services/audioCacheService');
          const { blob } = await fetchAudioWithCache(url);
          const arrayBuffer = await blob.arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          audioBufferCache.set(url, audioBuffer);
          console.log(`✅ Pre-loaded audio: ${url}, duration: ${audioBuffer.duration.toFixed(2)}s`);

          // Update track duration if it's 0
          const trackWithThisUrl = allTracks.find(t => t.audioUrl === url);
          if (trackWithThisUrl && (trackWithThisUrl.duration === 0 || !trackWithThisUrl.duration)) {
            console.log(`📏 Updating track duration for ${trackWithThisUrl.name}: ${audioBuffer.duration.toFixed(2)}s`);

            // Use the busId attached to the track
            const busId = trackWithThisUrl._busId;
            if (busId) {
              dispatch({
                type: 'UPDATE_TRACK',
                payload: {
                  busId,
                  trackId: trackWithThisUrl.id,
                  updates: { duration: audioBuffer.duration }
                }
              });
              console.log(`✅ Dispatched duration update: ${audioBuffer.duration.toFixed(2)}s for track ${trackWithThisUrl.id} in bus ${busId}`);
            } else {
              console.warn(`⚠️ No busId found for track ${trackWithThisUrl.id}`);
            }
          }
        } catch (error) {
          console.error(`❌ Failed to pre-load audio: ${url}`, error);
        }
      }
    });
  }, [tracks, dispatch]);

  /**
   * Update playhead position during playback
   */
  const updatePlayhead = useCallback(() => {
    // Use ref instead of prop to get current value in animation loop
    if (!isPlayingRef.current || !audioContextRef.current) {
      // Cancel animation frame if stopped
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      return;
    }

    const currentTime = audioContextRef.current.currentTime - startTimeRef.current;
    dispatch({ type: 'UPDATE_PLAYHEAD', payload: currentTime });

    if (currentTime >= totalDuration) {
      // End of playback
      dispatch({ type: 'SET_PLAYING', payload: false });
      dispatch({ type: 'RESET_PLAYHEAD' });
      return;
    }

    animationFrameRef.current = requestAnimationFrame(updatePlayhead);
  }, [dispatch, totalDuration]);

  /**
   * Start playback from current position
   * Respects each track's startPosition on the timeline
   */
  const play = useCallback(async () => {
    if (!audioContextRef.current) return;

    const audioContext = audioContextRef.current;

    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    try {
      // Stop any existing playback
      sourceNodesRef.current.forEach(source => {
        try {
          source.stop();
        } catch (e) {
          // Ignore errors from already stopped sources
        }
      });
      sourceNodesRef.current = [];
      gainNodesRef.current.clear(); // Clear old gain nodes

      // Get all tracks from ref (not prop) to avoid recreating this callback
      const currentTracks = tracksRef.current;
      const voTracks = currentTracks.vo || [];
      const musicTracks = currentTracks.music || [];
      const sfxTracks = currentTracks.sfx || [];
      const drumTracks = currentTracks.drums || [];
      const midiTracks = currentTracks.midi || [];
      const audioTracks = currentTracks.audio || [];

      // Include MIDI bus tracks - include MIDI tracks with audioUrl OR f0Audio
      const midiAudioTracks = midiTracks.filter(t =>
        (t.type !== 'midi' && t.audioUrl) || // Audio tracks in MIDI bus
        (t.type === 'midi' && (t.audioUrl || t.f0Audio || t.midiData)) // MIDI tracks with playable content
      );

      const allTracks = [...voTracks, ...musicTracks, ...sfxTracks, ...drumTracks, ...midiAudioTracks, ...audioTracks];

      console.log('🎵 Starting playback with', allTracks.length, 'tracks at playhead position', pauseTimeRef.current.toFixed(2), 's');

      // Check if any track has solo enabled
      const hasSoloTracks = allTracks.some(track => track.isSolo);
      console.log(`🎚️ Solo mode: ${hasSoloTracks ? 'ACTIVE' : 'INACTIVE'}`);

      // Current playhead position in timeline (seconds)
      const currentPlayheadTime = pauseTimeRef.current;

      // Capture audioContext.currentTime ONCE to use as reference for all scheduling
      const schedulingStartTime = audioContext.currentTime;

      console.log('🎬 Starting playback from position:', currentPlayheadTime.toFixed(2), 's');
      console.log('⏱️  AudioContext time at scheduling start:', schedulingStartTime.toFixed(3), 's');

      // Resume MIDI player audio context once (if there are MIDI tracks)
      const hasMidiTracks = allTracks.some(t => t.type === 'midi' && t.midiData);
      if (hasMidiTracks) {
        await midiPlayer.resume();
        console.log('🎹 MIDI player audio context resumed');
      }

      // Pre-load all audio buffers before playback (to avoid missing buffers)
      console.log('📦 Pre-loading audio buffers...');
      const tracksWithAudio = allTracks.filter(t => t.audioUrl && !t.type);
      for (const track of tracksWithAudio) {
        if (!audioBufferCache.has(track.audioUrl)) {
          console.log(`  📥 Loading buffer: ${track.audioUrl}`);
          try {
            const response = await fetch(track.audioUrl);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            audioBufferCache.set(track.audioUrl, audioBuffer);
            console.log(`  ✅ Buffer loaded: ${track.audioUrl}`);
          } catch (error) {
            console.error(`  ❌ Failed to load buffer: ${track.audioUrl}`, error);
          }
        }
      }

      // Pre-load F0 audio from MIDI tracks
      const tracksWithF0 = allTracks.filter(t => t.type === 'midi' && t.f0Audio);
      for (const track of tracksWithF0) {
        if (!audioBufferCache.has(track.f0Audio)) {
          console.log(`  📥 Loading F0 buffer: ${track.f0Audio}`);
          try {
            const response = await fetch(track.f0Audio);
            const arrayBuffer = await response.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            audioBufferCache.set(track.f0Audio, audioBuffer);
            console.log(`  ✅ F0 buffer loaded: ${track.f0Audio}`);
          } catch (error) {
            console.error(`  ❌ Failed to load F0 buffer: ${track.f0Audio}`, error);
          }
        }
      }

      // ── Mask-based playback for stem tracks ──────────────────────
      // If mask playback is ready, stem tracks play through the worklet
      // (master audio + spectral masks) instead of individual decoded buffers.
      // This gives master-quality audio with no decoder overhead.
      const useMaskPlayback = isMaskPlaybackReady();
      if (useMaskPlayback) {
        // Collect stem track gains and solo/mute state
        const stemGains = {};
        const stemTracks = allTracks.filter(t => t.metadata?.type === 'stem');
        const nonStemTracks = allTracks.filter(t => t.metadata?.type !== 'stem');
        const hasStemSolo = stemTracks.some(t => t.isSolo);

        let activeSolo = null;
        for (const t of stemTracks) {
          const stemName = t.metadata?.stemType || t.metadata?.instrument;
          if (!stemName) continue;
          const busGain = t._busGain || 1.0;
          const busMuted = t._busMuted || false;

          if (busMuted || t.isMuted) {
            stemGains[stemName] = 0;
          } else if (hasStemSolo && !t.isSolo) {
            stemGains[stemName] = 0;
          } else {
            stemGains[stemName] = (t.gain || 1.0) * busGain;
            if (t.isSolo) activeSolo = stemName;
          }
        }

        // Send gains to worklet — instant, no re-render
        setMaskGains(stemGains);
        setActiveStem(activeSolo);
        maskSeek(currentPlayheadTime);
        maskPlay();
        console.log(`🎭 Mask playback: ${Object.keys(stemGains).length} stems, solo=${activeSolo || 'none'}`);

        // Only schedule non-stem tracks normally below
        // (parent track is muted when stems exist, so it won't double-play)
        var tracksToSchedule = nonStemTracks;
      } else {
        var tracksToSchedule = allTracks;
      }

      // Load and schedule tracks (non-stem when mask playback active, all otherwise)
      for (const track of tracksToSchedule) {
        // Apply bus-level mute/solo first
        const busMuted = track._busMuted || false;
        const busSolo = track._busSolo || false;
        const busGain = track._busGain || 1.0;

        // Check if ANY bus has solo enabled
        const hasBusSolo = allTracks.some(t => t._busSolo);

        // Determine if this track should play
        // Priority: bus solo > bus mute > track solo > track mute
        let shouldPlay = false;
        if (busMuted) {
          // Bus is muted - don't play any tracks from this bus
          shouldPlay = false;
        } else if (hasBusSolo) {
          // Some bus has solo - only play tracks from solo'd buses
          shouldPlay = busSolo;
        } else if (hasSoloTracks) {
          // Some track has solo - only play solo'd tracks
          shouldPlay = track.isSolo;
        } else {
          // Normal mode - play non-muted tracks
          shouldPlay = !track.isMuted;
        }

        // Calculate final gain (bus gain * track gain)
        const finalGain = (track.gain || 1.0) * busGain;

        console.log(`  🎛️ Track: ${track.name} | TrackGain: ${track.gain || 1.0} | BusGain: ${busGain} | FinalGain: ${finalGain.toFixed(2)} | Muted: ${track.isMuted}/${busMuted} | Solo: ${track.isSolo}/${busSolo} | ShouldPlay: ${shouldPlay}`);

        if (shouldPlay) {
          // Check if this is a drum track with multiple hits
          if (track.isDrumTrack && track.drumHits && track.drumHits.length > 0) {
            console.log(`  🥁 Scheduling drum track with ${track.drumHits.length} hits`);

            // Schedule each drum hit individually
            for (const hit of track.drumHits) {
              const hitTime = hit.startTime || 0;
              const hitGain = (hit.velocity || 1.0) * finalGain;

              // Only schedule hits that haven't been played yet
              if (currentPlayheadTime <= hitTime + 1) { // +1 for drum hit duration
                console.log(`    🥁 Hit at ${hitTime.toFixed(2)}s, velocity: ${hit.velocity.toFixed(2)}, gain: ${hitGain.toFixed(2)}`);

                const { source, gainNode } = scheduleTrack(
                  audioContext,
                  hit.audioUrl,
                  hitGain,
                  hitTime, // Hit time on timeline
                  0, // No crop for drum hits
                  currentPlayheadTime,
                  schedulingStartTime,
                  masterGainNodeRef.current // Master gain node
                );

                if (source && gainNode) {
                  sourceNodesRef.current.push(source);
                  // Store gain node for drum hit (use unique ID)
                  gainNodesRef.current.set(`${track.id}-hit-${hitTime}`, gainNode);
                }
              }
            }
          } else if (track.audioUrl) {
            // Regular track (non-drum)
            const trackStartTime = track.startPosition || 0; // When track starts on timeline
            const trackCropStart = track.cropStart || 0; // Cropped portion at start
            const trackDuration = track.duration || track.length || 10;
            const trackEndTime = trackStartTime + trackDuration;

            // Only play if playhead is within or before the track's time range
            if (currentPlayheadTime <= trackEndTime) {
              console.log(`  ▶ Scheduling track: ${track.name || track.audioUrl}
                 Track starts at: ${trackStartTime.toFixed(2)}s on timeline
                 Track ends at: ${trackEndTime.toFixed(2)}s
                 Playhead currently at: ${currentPlayheadTime.toFixed(2)}s
                 Track is ${currentPlayheadTime >= trackStartTime && currentPlayheadTime <= trackEndTime ? 'ACTIVE' : 'upcoming'}`);

              const { source, gainNode } = scheduleTrack(
                audioContext,
                track.audioUrl,
                finalGain,
                trackStartTime,
                trackCropStart,
                currentPlayheadTime,
                schedulingStartTime,
                masterGainNodeRef.current // Master gain node
              );

              if (source && gainNode) {
                sourceNodesRef.current.push(source);
                gainNodesRef.current.set(track.id, gainNode);
              }
            } else {
              console.log(`  ⏭ Skipping track (playhead already past): ${track.name || track.audioUrl}`);
            }
          } else if (track.type === 'midi') {
            // MIDI track - check if it has F0 audio or MIDI notes
            const trackStartTime = track.startPosition || 0;
            const trackDuration = track.duration || 10;
            const trackEndTime = trackStartTime + trackDuration;

            // Play F0 audio if available
            if (track.f0Audio && currentPlayheadTime <= trackEndTime) {
              console.log(`  ▶ Scheduling F0 audio for MIDI track: ${track.name || 'Untitled'}`);
              const { source, gainNode } = scheduleTrack(
                audioContext,
                track.f0Audio,
                finalGain,
                trackStartTime,
                0, // No crop for F0
                currentPlayheadTime,
                schedulingStartTime,
                masterGainNodeRef.current
              );

              if (source && gainNode) {
                sourceNodesRef.current.push(source);
                gainNodesRef.current.set(`${track.id}-f0`, gainNode);
              }
            }

            // Only play MIDI notes if playhead is within or before the track's time range
            if (track.midiData && track.midiData.notes && currentPlayheadTime <= trackEndTime) {
              console.log(`  🎹 Scheduling MIDI track: ${track.name}
                 Track starts at: ${trackStartTime.toFixed(2)}s on timeline
                 Track ends at: ${trackEndTime.toFixed(2)}s
                 Playhead currently at: ${currentPlayheadTime.toFixed(2)}s
                 Notes: ${track.midiData.notes.length}
                 Track duration: ${trackDuration.toFixed(2)}s`);

              // Filter notes to only play those that haven't been played yet
              // NOTE: MIDI note times are already in SECONDS (from midiParser.js)
              // No tempo scaling needed - just use the times directly!
              const notesToPlay = track.midiData.notes
                .filter(note => {
                  const noteAbsoluteTime = trackStartTime + note.time;
                  return noteAbsoluteTime >= currentPlayheadTime;
                })
                .map(note => ({
                  note: note.note || note.midi,
                  time: note.time, // Already in seconds!
                  duration: note.duration, // Already in seconds!
                  velocity: (note.velocity || 100) / 127 // Normalize to 0-1
                }));

              console.log(`    🎹 Playing ${notesToPlay.length} MIDI notes (${track.midiData.notes.length - notesToPlay.length} already played)`);

              // Schedule MIDI notes
              // Calculate when to start playing notes in AudioContext time
              const midiStartTime = schedulingStartTime + (trackStartTime - currentPlayheadTime);

              notesToPlay.forEach(note => {
                const notePlayTime = midiStartTime + note.time;
                midiPlayer.playNote(
                  note.note,
                  note.velocity * finalGain, // Apply track/bus gain to velocity
                  note.duration,
                  notePlayTime
                );
              });
            } else {
              console.log(`  ⏭ Skipping MIDI track (playhead already past): ${track.name}`);
            }
          }
        }
      }

      // Record when playback starts in AudioContext time
      // Use the SAME time reference that was used for scheduling
      startTimeRef.current = schedulingStartTime - currentPlayheadTime;
      hasStartedPlaybackRef.current = true; // Mark that playback has actually started

      console.log('⏱️  Playback timing:', {
        schedulingStartTime: schedulingStartTime.toFixed(3),
        startTimeRef: startTimeRef.current.toFixed(3),
        playheadPosition: currentPlayheadTime.toFixed(3)
      });

      // Start playhead animation
      updatePlayhead();
    } catch (error) {
      console.error('❌ Error starting playback:', error);
      dispatch({ type: 'SET_PLAYING', payload: false });
    }
  }, [updatePlayhead, dispatch, bpm]); // Include bpm so playback updates when tempo changes

  /**
   * Pause playback
   * @param {boolean} storePosition - Whether to store current position (default true)
   */
  const pause = useCallback((storePosition = true) => {
    sourceNodesRef.current.forEach(source => {
      try {
        source.stop();
      } catch (e) {
        // Ignore
      }
    });
    sourceNodesRef.current = [];

    // Stop mask playback worklet
    if (isMaskPlaybackReady()) {
      maskStop();
    }

    // Stop all MIDI playback immediately
    midiPlayer.stopAll();
    console.log('🎹 Stopped all MIDI notes');

    // Store current playhead position for resume (unless seeking)
    // ONLY calculate from AudioContext if playback has actually started
    // Otherwise, keep the existing pauseTimeRef value (which was synced from external playhead)
    if (storePosition && audioContextRef.current && hasStartedPlaybackRef.current) {
      const calculatedTime = audioContextRef.current.currentTime - startTimeRef.current;
      // Sanity check: only use calculated time if it's within reasonable bounds
      if (calculatedTime >= 0 && calculatedTime <= totalDuration + 1) {
        pauseTimeRef.current = calculatedTime;
        // Also update global state so visual playhead stays in place
        dispatch({ type: 'UPDATE_PLAYHEAD', payload: pauseTimeRef.current });
      }
    }

    // Reset the playback started flag
    hasStartedPlaybackRef.current = false;

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    console.log('⏸ Playback paused at', pauseTimeRef.current.toFixed(2), 's');
  }, [dispatch, totalDuration]);

  /**
   * Stop playback and reset
   */
  const stop = useCallback(() => {
    pause();
    pauseTimeRef.current = 0;
    dispatch({ type: 'RESET_PLAYHEAD' });
    console.log('⏹ Playback stopped');
  }, [pause, dispatch]);

  /**
   * Seek to a specific time position
   */
  const seek = useCallback((timeInSeconds) => {
    // Check if we're currently playing using the ref (more reliable than source nodes)
    const wasPlaying = isPlayingRef.current;

    // Set seeking flag to prevent useEffect interference
    isSeekingRef.current = true;

    // Stop current playback WITHOUT storing position (we're setting a new one)
    pause(false);

    // Update playhead position (both ref and state)
    pauseTimeRef.current = timeInSeconds;
    dispatch({ type: 'SEEK_TO', payload: timeInSeconds });

    console.log('⏩ Seeked to', timeInSeconds.toFixed(2), 's', wasPlaying ? '(will resume playing)' : '(will stay paused)');

    // If was playing, restart immediately from new position
    if (wasPlaying) {
      // Use requestAnimationFrame for smooth restart
      requestAnimationFrame(() => {
        isSeekingRef.current = false;
        play();
      });
    } else {
      isSeekingRef.current = false;
    }
  }, [pause, dispatch, play]);

  /**
   * Handle play/pause based on isPlaying prop
   */
  useEffect(() => {
    // Don't interfere if we're currently seeking
    if (isSeekingRef.current) return;

    if (isPlaying) {
      // Only start playback if not already playing
      if (sourceNodesRef.current.length === 0) {
        play();
      }
    } else {
      // Always call pause when stopping - it handles both audio nodes and MIDI
      // (MIDI doesn't create sourceNodes, so we can't rely on length check)
      pause();
    }
  }, [isPlaying, play, pause]);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      sourceNodesRef.current.forEach(source => {
        try {
          source.stop();
        } catch (e) {
          // Ignore
        }
      });
    };
  }, []);

  return {
    play,
    pause,
    stop,
    seek,
    audioContext: audioContextRef.current,
    gainNodes: gainNodesRef.current
  };
}


/**
 * Schedule playback using cached audio buffer
 * NO network fetch - uses pre-loaded buffer from cache
 *
 * @param {AudioContext} audioContext - Web Audio API context
 * @param {string} audioUrl - URL to audio file (used as cache key)
 * @param {number} gain - Volume level (0-1)
 * @param {number} trackStartTime - When track starts on timeline (seconds)
 * @param {number} trackCropStart - Cropped portion at start of track (seconds)
 * @param {number} currentPlayheadTime - Current playhead position on timeline (seconds)
 * @param {number} schedulingStartTime - AudioContext.currentTime at start of scheduling (seconds)
 */
function scheduleTrack(
  audioContext,
  audioUrl,
  gain,
  trackStartTime,
  trackCropStart,
  currentPlayheadTime,
  schedulingStartTime,
  masterGainNode
) {
  try {
    // Get cached audio buffer (should already be loaded)
    const audioBuffer = audioBufferCache.get(audioUrl);

    if (!audioBuffer) {
      console.warn(`⚠️ Audio buffer not cached yet: ${audioUrl}`);
      return { source: null, gainNode: null };
    }

    const source = audioContext.createBufferSource();
    const gainNode = audioContext.createGain();

    source.buffer = audioBuffer;
    gainNode.gain.value = gain;

    source.connect(gainNode);
    // TEMPORARY: Bypass FX chain for testing - connect directly to master
    // Signal chain: source → gainNode → master gain → destination
    gainNode.connect(masterGainNode);
    // Original FX routing (disabled for testing):
    // gainNode.connect(tunaFX.getFXBusInput());

    // Calculate when to start playback in AudioContext time
    // If playhead is before track start, schedule for future
    // If playhead is after track start, start immediately but offset into the audio
    // Use schedulingStartTime as reference to ensure consistency

    let when = 0; // When to start in AudioContext time
    let offset = trackCropStart; // Where to start within the audio buffer

    if (currentPlayheadTime < trackStartTime) {
      // Playhead hasn't reached this track yet - schedule for future
      const delayUntilTrackStart = trackStartTime - currentPlayheadTime;
      when = schedulingStartTime + delayUntilTrackStart;
      offset = trackCropStart; // Start from beginning (plus crop)
    } else {
      // Playhead is already past track start - start immediately but offset
      when = schedulingStartTime;
      const timeIntoTrack = currentPlayheadTime - trackStartTime;
      offset = trackCropStart + timeIntoTrack; // Skip ahead to current position

      // Don't play if we're past the end of the audio
      if (offset >= audioBuffer.duration) {
        console.log('    ⏭ Track already finished');
        return { source: null, gainNode: null };
      }
    }

    console.log(`    🎵 Starting at AudioContext time: ${when.toFixed(2)}s, offset: ${offset.toFixed(2)}s`);

    source.start(when, offset);

    return { source, gainNode };
  } catch (error) {
    console.error('❌ Error scheduling track:', audioUrl, error);
    return { source: null, gainNode: null };
  }
}
