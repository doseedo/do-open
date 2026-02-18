import React, { useCallback, useState, useEffect, useMemo, useRef } from 'react';
import { useApp } from '../../context/AppContext';
import PlayheadCursor from './PlayheadCursor';
import styles from './DAW.module.css';

/**
 * Timeline Component - Complete Rewrite with Modern React Patterns
 *
 * Features:
 * - GPU-accelerated cursor animations
 * - Responsive width measurement
 * - Adaptive tick spacing
 * - Memoized calculations
 * - Click-to-seek interaction
 * - Clean, performant rendering
 */
const Timeline = React.memo(({
  totalDuration,
  zoomLevel,
  onSeek,
  timelineRef,
  playheadPosition = 0,
  isBPMMode = false,
  bpm = 120,
  sceneTempos = [],
  sceneChanges = [],
  playheadOnly = false, // New prop: if true, only render playhead without ticks
  skipWidthMeasurement = false, // New prop: if true, don't measure/dispatch width (for duplicate timelines)
  onZoomIn = null,
  onZoomOut = null,
  trackHeight = null,
  onZoomYIn = null,
  onZoomYOut = null
}) => {
  const { state, dispatch } = useApp();
  const [containerWidth, setContainerWidth] = useState(700);
  const [actualGridWidth, setActualGridWidth] = useState(700);
  const [isClickEnabled, setIsClickEnabled] = useState(true);
  const measurementRef = useRef(null);
  const lastWidthRef = useRef(null);
  const measureInProgressRef = useRef(false);

  // For duplicate timelines (skipWidthMeasurement), use global state width
  const effectiveContainerWidth = skipWidthMeasurement ? (state.timelineWidth || 700) : containerWidth;

  // Measure the grid column's actual width (not the content)
  // ONLY measure on initial mount - DO NOT use ResizeObserver to avoid feedback loop
  useEffect(() => {
    // Skip measurement for duplicate timelines (playhead-only)
    if (skipWidthMeasurement || !timelineRef?.current) return;

    const measure = () => {
      if (!timelineRef?.current) return;

      // Get the parent timelineRow to measure the grid column space
      const timelineRow = timelineRef.current.closest('[class*="timelineRow"]');
      if (timelineRow) {
        const rect = timelineRow.getBoundingClientRect();
        // Grid is: 60px + label-width + timeline-column
        // Subtract the first two columns (60px + ~340px = 400px)
        const availableWidth = rect.width - 350;
        const width = Math.max(100, availableWidth); // Minimum 100px to prevent collapse to 0

        console.log('📏 Timeline initial width:', width.toFixed(1));
        lastWidthRef.current = width;
        setContainerWidth(width);
        setActualGridWidth(width);

        // Update global state so tracks can sync their width
        dispatch({ type: 'UPDATE_TIMELINE_WIDTH', payload: width });
      }
    };

    // Single measurement on mount with delay to let grid layout complete
    const timer = setTimeout(measure, 150);

    return () => {
      clearTimeout(timer);
    };
  }, [timelineRef, dispatch, skipWidthMeasurement]);

  // Calculate pixels per second
  const pixelsPerSecond = useMemo(() => {
    const width = effectiveContainerWidth * zoomLevel;
    return totalDuration > 0 ? width / totalDuration : 0;
  }, [effectiveContainerWidth, zoomLevel, totalDuration]);

  // Format time label (seconds or minutes:seconds) - Must be defined before use
  const formatTimeLabel = useCallback((seconds) => {
    if (seconds >= 60) {
      const mins = Math.floor(seconds / 60);
      const secs = seconds % 60;
      return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    return `${seconds}s`;
  }, []);

  // Determine optimal tick interval based on zoom level - NO FLOATS
  const tickInterval = useMemo(() => {
    if (pixelsPerSecond < 10) return 10;  // Very zoomed out - 10s intervals
    if (pixelsPerSecond < 20) return 5;   // Zoomed out - 5s intervals
    if (pixelsPerSecond < 40) return 2;   // Medium zoom - 2s intervals
    return 1;  // Zoomed in - 1s intervals
  }, [pixelsPerSecond]);

  // Determine subdivision level based on zoom
  // 1 = quarter notes only, 2 = 8th notes, 4 = 16th notes
  const subdivisionLevel = useMemo(() => {
    if (!isBPMMode) return 1;

    // Base on how many pixels per beat
    const avgBPM = sceneTempos.length > 0 ? sceneTempos.reduce((a, b) => a + b) / sceneTempos.length : bpm;
    const secondsPerBeat = 60 / avgBPM;
    const pixelsPerBeat = pixelsPerSecond * secondsPerBeat;

    // Show subdivisions based on pixel spacing
    if (pixelsPerBeat < 20) return 1;  // Too tight, only show quarter notes
    if (pixelsPerBeat < 60) return 2;  // Show 8th notes
    return 4; // Show 16th notes
  }, [isBPMMode, bpm, sceneTempos, pixelsPerSecond]);

  // Update global state with subdivision level for snapping
  useEffect(() => {
    if (isBPMMode) {
      dispatch({ type: 'UPDATE_SUBDIVISION_LEVEL', payload: subdivisionLevel });
    }
  }, [isBPMMode, subdivisionLevel, dispatch]);

  // Generate BPM tick marks (bars, beats, and sub-beats)
  const bpmTicks = useMemo(() => {
    if (!isBPMMode) return null;

    const tickArray = [];
    const width = effectiveContainerWidth * zoomLevel;
    const useSceneTempos = sceneTempos.length > 0 && sceneChanges.length > 1;

    if (useSceneTempos) {
      // Render beats/bars with tempo changes per scene
      console.log('🎵 Rendering BPM timeline with scene tempos:', sceneTempos, 'subdivision:', subdivisionLevel);

      let accumulatedBeats = 0;
      let barNumber = 1;

      for (let sceneIdx = 0; sceneIdx < sceneTempos.length; sceneIdx++) {
        const sceneBPM = sceneTempos[sceneIdx];
        const sceneStart = sceneChanges[sceneIdx];
        const sceneEnd = sceneChanges[sceneIdx + 1] || totalDuration;
        const secondsPerBeat = 60 / sceneBPM;
        const secondsPerBar = secondsPerBeat * 4; // 4/4 time signature

        // Calculate where the next bar should start based on accumulated beats
        const beatsIntoFirstBar = accumulatedBeats % 4;
        const beatsUntilNextBar = beatsIntoFirstBar === 0 ? 0 : (4 - beatsIntoFirstBar);
        const firstBarTime = sceneStart + (beatsUntilNextBar * secondsPerBeat);

        // Render bars starting from first bar boundary in this scene
        let barTime = firstBarTime;
        while (barTime < sceneEnd && barTime <= totalDuration) {
          if (barTime >= sceneStart) {
            const barPosition = (barTime / totalDuration) * width;

            // Add bar marker
            tickArray.push({
              id: `bar-${barNumber}`,
              time: barTime,
              position: barPosition,
              label: `${barNumber}`,
              isMajor: true,
              isBar: true,
              subdivision: 1
            });

            // Add beat and sub-beat subdivisions
            const totalSubdivisions = 4 * subdivisionLevel; // 4 beats per bar * subdivision
            for (let sub = 1; sub < totalSubdivisions; sub++) {
              const subTime = barTime + (sub * secondsPerBeat / subdivisionLevel);
              if (subTime >= sceneEnd || subTime > totalDuration) break;

              const subPosition = (subTime / totalDuration) * width;
              const isBeat = (sub % subdivisionLevel) === 0; // Every Nth subdivision is a beat

              tickArray.push({
                id: `bar-${barNumber}-sub-${sub}`,
                time: subTime,
                position: subPosition,
                isMajor: false,
                isBeat: isBeat,
                isSubBeat: !isBeat,
                subdivision: subdivisionLevel
              });
            }

            barNumber++;
          }
          barTime += secondsPerBar;
        }

        // Update accumulated beats for next scene
        const sceneDuration = sceneEnd - sceneStart;
        accumulatedBeats += sceneDuration / secondsPerBeat;
      }

    } else {
      // Render with constant BPM
      const secondsPerBeat = 60 / bpm;
      const secondsPerBar = secondsPerBeat * 4; // 4/4 time signature
      let barNumber = 1;

      for (let time = 0; time <= totalDuration; time += secondsPerBar) {
        const barPosition = (time / totalDuration) * width;

        // Add bar marker
        tickArray.push({
          id: `bar-${barNumber}`,
          time,
          position: barPosition,
          label: `${barNumber}`,
          isMajor: true,
          isBar: true,
          subdivision: 1
        });

        // Add beat and sub-beat subdivisions
        const totalSubdivisions = 4 * subdivisionLevel; // 4 beats per bar * subdivision
        for (let sub = 1; sub < totalSubdivisions; sub++) {
          const subTime = time + (sub * secondsPerBeat / subdivisionLevel);
          if (subTime > totalDuration) break;

          const subPosition = (subTime / totalDuration) * width;
          const isBeat = (sub % subdivisionLevel) === 0; // Every Nth subdivision is a beat

          tickArray.push({
            id: `bar-${barNumber}-sub-${sub}`,
            time: subTime,
            position: subPosition,
            isMajor: false,
            isBeat: isBeat,
            isSubBeat: !isBeat,
            subdivision: subdivisionLevel
          });
        }

        barNumber++;
      }

    }

    return tickArray;
  }, [isBPMMode, bpm, sceneTempos, sceneChanges, totalDuration, effectiveContainerWidth, zoomLevel, subdivisionLevel]);

  // Generate time tick marks (seconds)
  const timeTicks = useMemo(() => {
    if (isBPMMode) return null;

    const tickArray = [];
    const width = effectiveContainerWidth * zoomLevel;

    for (let time = 0; time <= totalDuration; time += tickInterval) {
      const position = (time / totalDuration) * width;

      tickArray.push({
        id: `tick-${time}`,
        time,
        position,
        label: formatTimeLabel(time),
        isMajor: time % (tickInterval * 5) === 0 || tickInterval >= 5
      });
    }

    return tickArray;
  }, [isBPMMode, totalDuration, tickInterval, effectiveContainerWidth, zoomLevel, formatTimeLabel]);

  // Select which ticks to use
  const ticks = isBPMMode ? bpmTicks : timeTicks;

  // Handle timeline click with debouncing
  const handleClick = useCallback((e) => {
    if (!timelineRef?.current || !isClickEnabled) return;

    // Check if click originated from a track, bus, or interactive element
    // This prevents playhead seeking when clicking on tracks
    const target = e.target;
    const clickedElement = target.closest('[data-track-id], [data-bus-id], [class*="trackLabel"], [class*="busRow"], [class*="track"], button, input, select, [role="button"]');

    if (clickedElement) {
      console.log('🚫 Timeline click ignored - clicked on interactive element:', clickedElement.className || clickedElement.tagName);
      return;
    }

    const contentElement = timelineRef.current.querySelector('[class*="timelineContent"]');
    if (!contentElement) return;

    // Get click position relative to the actual rendered timeline content
    const rect = contentElement.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const timelineWidth = rect.width;

    // Calculate time from position
    const clickedTime = (clickX / timelineWidth) * totalDuration;
    const clampedTime = Math.max(0, Math.min(clickedTime, totalDuration));

    console.log('🖱️ Timeline clicked:', {
      clickX: clickX.toFixed(1) + 'px',
      width: timelineWidth.toFixed(1) + 'px',
      time: clampedTime.toFixed(2) + 's'
    });

    // Temporarily disable clicks to prevent double-firing
    setIsClickEnabled(false);
    onSeek(clampedTime);

    setTimeout(() => setIsClickEnabled(true), 100);
  }, [totalDuration, onSeek, timelineRef, isClickEnabled]);

  // Handle click on timeline spacer to seek to 0
  const handleSpacerClick = useCallback((e) => {
    // Only seek if clicking directly on the spacer (not on buttons)
    if (e.target === e.currentTarget) {
      console.log('🎯 Timeline spacer clicked - seeking to 0');
      onSeek(0);
    }
  }, [onSeek]);

  // Calculate timeline content width
  const timelineContentWidth = effectiveContainerWidth * zoomLevel;

  // Drag and drop handlers for audio files
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragOver(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Check if it's a MIDI file
      const isMidiFile = file.name.toLowerCase().endsWith('.mid') ||
                        file.name.toLowerCase().endsWith('.midi') ||
                        file.type === 'audio/midi' ||
                        file.type === 'audio/mid';

      // Check if it's an audio file
      if (!file.type.startsWith('audio/') && !isMidiFile) {
        alert('Please drop an audio or MIDI file');
        return;
      }

      console.log(`🎵 ${isMidiFile ? 'MIDI' : 'Audio'} file dropped on timeline:`, file.name);

      // Calculate drop position in timeline
      const contentElement = timelineRef.current.querySelector('[class*="timelineContent"]');
      if (!contentElement) return;

      const rect = contentElement.getBoundingClientRect();
      const dropX = e.clientX - rect.left;
      const timelineWidth = rect.width;
      const dropTime = Math.max(0, (dropX / timelineWidth) * totalDuration);

      console.log(`📍 Drop position: ${dropTime.toFixed(2)}s`);

      if (isMidiFile) {
        // Handle MIDI file
        const reader = new FileReader();
        reader.onload = async (readerEvent) => {
          try {
            // Parse MIDI file using Tone.js
            const Midi = (await import('@tonejs/midi')).Midi;
            const midi = new Midi(readerEvent.target.result);

            console.log(`🎹 Parsed MIDI: ${midi.tracks.length} tracks, duration: ${midi.duration}s`);

            // Create a new MIDI bus for the dropped MIDI file
            const busId = `midi-${Date.now()}`;

            dispatch({
              type: 'CREATE_BUS',
              payload: {
                id: busId,
                type: 'MIDI',
                name: file.name.replace(/\.mid$/i, ''),
                expanded: true
              }
            });

            // Filter out empty tracks
            const validTracks = midi.tracks.filter(track => track.notes.length > 0);

            if (validTracks.length === 0) {
              alert('MIDI file has no notes');
              return;
            }

            // If this is a multi-track MIDI, store the original file in bus metadata
            if (validTracks.length > 1) {
              const originalMidiBlob = new Blob([await file.arrayBuffer()], { type: 'audio/midi' });
              dispatch({
                type: 'UPDATE_BUS_METADATA',
                payload: {
                  busId,
                  metadata: {
                    originalMultitrackMidi: originalMidiBlob,
                    originalMidiFilename: file.name
                  }
                }
              });
              console.log(`💾 Stored original multi-track MIDI file for bus ${busId}`);
            }

            // Create separate track for each MIDI track
            validTracks.forEach(async (midiTrack, trackIndex) => {
              const trackId = `track-${Date.now()}-${trackIndex}`;

              // Create a single-track MIDI file for this track
              const { Midi: ToneMidi } = await import('@tonejs/midi');
              const singleTrackMidi = new ToneMidi();
              singleTrackMidi.header.setTempo(midi.header.tempos[0]?.bpm || 120);
              // Note: ppq is readonly in Tone.js, it defaults to 480
              midi.header.timeSignatures.forEach(ts => {
                singleTrackMidi.header.timeSignatures.push(ts);
              });
              const newTrack = singleTrackMidi.addTrack();
              newTrack.name = midiTrack.name;
              midiTrack.notes.forEach(note => {
                newTrack.addNote({
                  midi: note.midi,
                  time: note.time,
                  duration: note.duration,
                  velocity: note.velocity
                });
              });

              // Convert to binary and create blob
              const singleTrackBlob = new Blob([singleTrackMidi.toArray()], { type: 'audio/midi' });

              const track = {
                id: trackId,
                name: midiTrack.name || `Track ${trackIndex + 1}`,
                type: 'midi',
                midiData: {
                  duration: midi.duration,
                  notes: midiTrack.notes.map(note => ({
                    midi: note.midi,
                    note: note.midi,
                    time: note.time,
                    duration: note.duration,
                    velocity: note.velocity,
                    name: note.name
                  })),
                  tempo: midi.header.tempos[0]?.bpm || 120,
                  tempos: midi.header.tempos.map(t => ({ time: t.time, bpm: t.bpm })),
                  timeSignatures: midi.header.timeSignatures.map(ts => ({
                    time: ts.time,
                    numerator: ts.timeSignature[0],
                    denominator: ts.timeSignature[1]
                  })),
                  ppq: midi.header.ppq,
                  isMultitrack: validTracks.length > 1
                },
                duration: midi.duration,
                startPosition: dropTime,
                gain: 1.0,
                isMuted: false,
                isSolo: false,
                fx: {
                  reverb: 0,
                  fadeIn: 0,
                  fadeOut: 0
                },
                metadata: {
                  type: 'uploaded',
                  midiBlob: singleTrackBlob,
                  midiFilename: `${file.name.replace(/\.mid$/i, '')}_track_${trackIndex + 1}.mid`,
                  instrument: midiTrack.instrument?.name,
                  isMultitrackSource: validTracks.length > 1
                }
              };

              // Add track to the new bus
              dispatch({
                type: 'ADD_TRACK',
                payload: { busId, track }
              });
            });

            console.log(`✅ MIDI file added to timeline with ${validTracks.length} tracks at ${dropTime.toFixed(2)}s`);
          } catch (error) {
            console.error('❌ Error parsing MIDI file:', error);
            alert(`Failed to load MIDI file: ${error.message}`);
          }
        };
        reader.readAsArrayBuffer(file);
        return;
      }

      // Handle audio file
      // Create blob URL for instant display
      const blobUrl = URL.createObjectURL(file);

      // Get audio duration immediately
      const audio = new Audio();
      audio.src = blobUrl;

      audio.addEventListener('loadedmetadata', () => {
        const duration = audio.duration;

        // Create a new SFX bus for the dropped audio
        const busId = `sfx-${Date.now()}`;
        const trackId = `track-${Date.now()}`;

        dispatch({
          type: 'CREATE_BUS',
          payload: {
            id: busId,
            type: 'SFX',
            name: `SFX ${file.name}`,
            expanded: true
          }
        });

        // Create the track with blob URL for instant display
        const track = {
          id: trackId,
          name: file.name,
          audioUrl: blobUrl, // Blob URL for instant playback
          duration: duration,
          startPosition: dropTime,
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
          metadata: {
            type: 'uploaded',
            originalFilename: file.name
          }
        };

        // Add track to the new bus immediately
        dispatch({
          type: 'ADD_TRACK',
          payload: { busId, track }
        });

        console.log('✅ Audio file added to timeline with blob URL at', dropTime.toFixed(2) + 's');

        // Upload file to server in the background
        const uploadFile = async () => {
          try {
            const formData = new FormData();
            formData.append('audioFile', file);

            console.log('📤 Uploading file to server in background...');
            const response = await fetch('/api/upload-audio', {
              method: 'POST',
              body: formData
            });

            if (!response.ok) {
              throw new Error(`Upload failed: ${response.status}`);
            }

            const result = await response.json();
            console.log('✅ File uploaded to server:', result);

            // Update the track with the server URL
            dispatch({
              type: 'UPDATE_TRACK',
              payload: {
                busId: busId,
                trackId: trackId,
                updates: {
                  audioUrl: result.url
                }
              }
            });

            console.log('✅ Track updated with server URL');
          } catch (error) {
            console.error('❌ Background upload failed:', error);
            // Track still works with blob URL, just won't persist across refreshes
          }
        };

        uploadFile();
      });
    }
  }, [timelineRef, totalDuration, dispatch]);

  // Handler for adding a new bus
  const handleAddBus = useCallback(() => {
    const busId = `sfx-${Date.now()}`;
    dispatch({
      type: 'CREATE_BUS',
      payload: {
        id: busId,
        type: 'SFX',
        name: `SFX Bus`,
        expanded: true
      }
    });
    console.log('✅ Created new empty SFX bus:', busId);
  }, [dispatch]);

  // If playheadOnly mode, render simplified version
  if (playheadOnly) {
    return (
      <>
        {/* Empty spacers for columns 1 and 2 - must match dawGrid columns */}
        <div style={{ gridColumn: 1, height: '0px', overflow: 'visible' }}></div>
        <div style={{ gridColumn: 2, height: '0px', overflow: 'visible' }}></div>

        {/* Playhead line in column 3, aligned with tracks */}
        <div
          ref={timelineRef}
          style={{
            gridColumn: 3,
            height: '0px',
            border: 'none',
            background: 'transparent',
            pointerEvents: 'none',
            overflow: 'visible',
            position: 'relative'
          }}
        >
          <div
            className={styles.timelineContent}
            style={{
              width: `${timelineContentWidth}px`,
              height: '0px',
              overflow: 'visible',
              position: 'relative'
            }}
          >
            {/* Only playhead cursor - no triangle, only line */}
            <PlayheadCursor
              position={playheadPosition}
              totalDuration={totalDuration}
              width={effectiveContainerWidth}
              zoomLevel={zoomLevel}
              showTriangle={false}
              showLine={true}
              showTimeDisplay={false}
            />
          </div>
        </div>
      </>
    );
  }

  // Full timeline with ticks and interaction
  return (
    <div className={styles.timelineRow}>
      {/* Combined spacer for columns 1 and 2 - just add bus button */}
      <div
        className={styles.timelineSpacer1}
        onClick={handleSpacerClick}
        style={{ cursor: 'pointer' }}
      >
        <button
          onClick={handleAddBus}
          className={styles.addTrackButton}
          title="Add new bus"
        >
          <span style={{ fontSize: '16px', marginRight: '4px' }}>+</span>
          <span style={{ fontSize: '11px', fontWeight: '500' }}>Add Track</span>
        </button>

        {/* Zoom Controls */}
        <div style={{ display: 'flex', gap: '4px', alignItems: 'center', marginRight: '10px' }}>
          {/* Zoom Mode Toggle */}
          <button
            className={styles.zoomModeButton}
            onClick={() => {
              const newMode = state.zoomMode === 'x' ? 'y' : 'x';
              dispatch({ type: 'SET_ZOOM_MODE', payload: newMode });
            }}
            title={state.zoomMode === 'x' ? 'Switch to Vertical Zoom' : 'Switch to Horizontal Zoom'}
          >
            <i className={`fa-solid ${state.zoomMode === 'y' ? 'fa-up-down' : 'fa-left-right'}`}></i>
          </button>

          {/* Zoom Out */}
          <button
            className={styles.zoomButton}
            onClick={() => {
              if (state.zoomMode === 'x') {
                onZoomOut();
              } else {
                onZoomYOut();
              }
            }}
            title={state.zoomMode === 'x' ? 'Zoom Out (Horizontal)' : 'Decrease Track Height'}
          >
            <i className="fa-solid fa-minus"></i>
          </button>

          {/* Zoom In */}
          <button
            className={styles.zoomButton}
            onClick={() => {
              if (state.zoomMode === 'x') {
                onZoomIn();
              } else {
                onZoomYIn();
              }
            }}
            title={state.zoomMode === 'x' ? 'Zoom In (Horizontal)' : 'Increase Track Height'}
          >
            <i className="fa-solid fa-plus"></i>
          </button>
        </div>
      </div>

      <div
        ref={timelineRef}
        className={`${styles.timeline} ${isDragOver ? styles.dragOver : ''}`}
        onClick={handleClick}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          cursor: isClickEnabled ? 'pointer' : 'wait'
        }}
      >
          {/* Inner content with dynamic width */}
          <div
            ref={measurementRef}
            className={styles.timelineContent}
            style={{
              width: `${timelineContentWidth}px`
            }}
          >
            {/* Render tick marks */}
            {ticks && ticks.map((tick) => (
              <React.Fragment key={tick.id}>
                {/* Tick line */}
                <div
                  className={styles.tick}
                  style={{
                    position: 'absolute',
                    left: `${tick.position}px`,
                    top: tick.isSubBeat ? '15px' : (tick.isMajor ? '0' : '10px'),
                    width: tick.isBar ? '2px' : '1px',
                    height: tick.isSubBeat ? '30%' : (tick.isMajor ? '100%' : '50%'),
                    backgroundColor: tick.isBar ? '#888' : (tick.isBeat ? '#555' : (tick.isSubBeat ? '#333' : (tick.isMajor ? '#666' : '#444'))),
                    bottom: tick.isBeat ? '0' : 'auto',
                    pointerEvents: 'none',
                    opacity: tick.isSubBeat ? 0.5 : 1
                  }}
                />

                {/* Tick label (only for major ticks) */}
                {tick.isMajor && tick.label && (
                  <div
                    className={styles.tickLabel}
                    style={{
                      position: 'absolute',
                      left: tick.position === 0 ? '2px' : `${tick.position + 3}px`, // Special case for position 0
                      top: '5px',
                      fontSize: '10px',
                      color: '#ccc',
                      fontWeight: tick.isBar ? 'bold' : 'normal',
                      whiteSpace: 'nowrap',
                      userSelect: 'none',
                      pointerEvents: 'none'
                    }}
                  >
                    {tick.label}
                  </div>
                )}
              </React.Fragment>
            ))}

            {/* Playhead cursor - only triangle, no line */}
            <PlayheadCursor
              position={playheadPosition}
              totalDuration={totalDuration}
              width={effectiveContainerWidth}
              zoomLevel={zoomLevel}
              showTriangle={true}
              showLine={false}
              showTimeDisplay={true}
            />
        </div>
      </div>
    </div>
  );
});

Timeline.displayName = 'Timeline';

export default Timeline;
