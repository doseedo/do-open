import React, { useCallback, useRef, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import { useAudioPlayback } from '../../hooks/useAudioPlayback';
import { useKeyboardControls } from '../../hooks/useKeyboardControls';
import { useMetronome } from '../../hooks/useMetronome';
import { parseMIDIFile } from '../../utils/midiParser';
import TransportControls from './TransportControls';
import Timeline from './Timeline';
import TimelineGrid from './TimelineGrid';
import SceneMarkers from './SceneMarkers';
import ChordTrack from './ChordTrack';
import BusRow from './BusRow';
import Button from '../common/Button';
import AutomationWindow from '../AutomationWindow/AutomationWindow';
import styles from './DAW.module.css';

/**
 * DAWOptimized - Fully optimized DAW using CSS Grid and CSS Modules
 *
 * Key Optimizations:
 * 1. CSS Grid for layout (no absolute positioning hacks)
 * 2. CSS Modules for scoped styles (no !important needed)
 * 3. Flatter DOM hierarchy (fewer nested divs)
 * 4. Transform-based animations (GPU accelerated)
 * 5. Extracted reusable components for better organization
 */

// Main DAW Component
const DAWOptimized = React.memo(({ maxTracksHeight = 600, panelWidth = 400, pluginMode = false }) => {
  const { state, dispatch } = useApp();
  const timelineRef = useRef(null);
  const scrollableContentRef = useRef(null);
  const dawGridRef = useRef(null);
  const dragIndexRef = useRef(null);
  const [dragOverIndex, setDragOverIndex] = React.useState(null);

  // Audio playback hook - convert buses array to tracks object with bus info
  const tracksForPlayback = useMemo(() => {
    const tracks = { vo: [], music: [], sfx: [], drums: [], midi: [], audio: [] };
    state.buses.forEach(bus => {
      const typeKey = bus.type.toLowerCase();
      // Add bus info to each track so playback can apply bus-level controls
      const tracksWithBusInfo = bus.tracks.map(track => ({
        ...track,
        _busId: bus.id,      // Add busId so playback can update track
        _busGain: bus.gain,
        _busPan: bus.pan,
        _busReverbSend: bus.reverbSend,
        _busMuted: bus.mute,
        _busSolo: bus.solo
      }));
      tracks[typeKey] = [...(tracks[typeKey] || []), ...tracksWithBusInfo];
    });
    return tracks;
  }, [state.buses]);

  const { seek, gainNodes } = useAudioPlayback(
    tracksForPlayback,
    state.isPlaying,
    dispatch,
    state.totalDuration || 10,
    state.playheadPosition || 0,
    state.bpm || 120,
    state.masterGain || 0.8  // Master gain (default 80% to prevent clipping)
  );

  // Enable keyboard controls (spacebar for play/pause, etc.)
  useKeyboardControls(dispatch, state.isPlaying);

  // Enable metronome with tempo sync
  useMetronome(
    state.isPlaying,
    state.playheadPosition || 0,
    state.isMetronomeOn,
    state.bpm || 120,
    state.video?.sceneTempos || [],
    state.video?.sceneChanges || []
  );

  // Drag and drop handlers for bus reordering
  const handleBusDragStart = useCallback((index) => (e) => {
    dragIndexRef.current = index;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/html', e.target);
  }, []);

  const handleBusDragOver = useCallback((index) => (e) => {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';
    if (dragIndexRef.current !== null && dragIndexRef.current !== index) {
      setDragOverIndex(index);
    }
  }, []);

  const handleBusDragLeave = useCallback(() => {
    setDragOverIndex(null);
  }, []);

  const handleBusDrop = useCallback((index) => (e) => {
    e.preventDefault();
    const fromIndex = dragIndexRef.current;
    if (fromIndex !== null && fromIndex !== index) {
      dispatch({
        type: 'REORDER_BUSES',
        payload: { fromIndex, toIndex: index }
      });
    }
    dragIndexRef.current = null;
    setDragOverIndex(null);
  }, [dispatch]);

  const handleBusDragEnd = useCallback(() => {
    dragIndexRef.current = null;
    setDragOverIndex(null);
  }, []);

  // Calculate timeline metrics (not used here - Timeline.js does its own calculations)
  useTimeline(
    state.totalDuration || 10,
    state.zoomLevel || 1.0,
    800
  );

  // Debug: Log when totalDuration changes
  React.useEffect(() => {
    console.log('🎬 Total duration:', state.totalDuration + 's');
  }, [state.totalDuration]);

  const timelineContainerRef = useRef(null);

  // Reset scroll position to 0 when zoom level changes
  React.useEffect(() => {
    if (timelineContainerRef.current) {
      timelineContainerRef.current.scrollLeft = 0;
    }
  }, [state.zoomLevel]);

  // Copy/Paste keyboard shortcuts (Cmd+C / Cmd+V)
  React.useEffect(() => {
    const handleKeyDown = (e) => {
      // Check for Cmd (Mac) or Ctrl (Windows/Linux)
      const isCmdOrCtrl = e.metaKey || e.ctrlKey;

      // Ignore if user is typing in an input field
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') {
        return;
      }

      if (isCmdOrCtrl && e.key === 'c') {
        // Copy selected track
        if (state.selectedTrack) {
          e.preventDefault();
          dispatch({
            type: 'COPY_TRACK',
            payload: { track: state.selectedTrack }
          });
          console.log('📋 Copied track:', state.selectedTrack.name);
        }
      } else if (isCmdOrCtrl && e.key === 'v') {
        // Paste track at playhead position
        if (state.copiedTrack) {
          e.preventDefault();

          // Determine target bus
          let targetBusId;
          if (state.selectedTrack) {
            // Find bus containing selected track
            const trackBus = state.buses.find(bus =>
              bus.tracks.some(t => t.id === state.selectedTrack.id)
            );
            targetBusId = trackBus?.id;
          } else if (state.selectedBus) {
            // Use selected bus
            targetBusId = state.selectedBus.id;
          } else if (state.buses.length > 0) {
            // Default to first bus
            targetBusId = state.buses[0].id;
          }

          if (targetBusId) {
            dispatch({
              type: 'PASTE_TRACK',
              payload: {
                targetBusId,
                playheadPosition: state.playheadPosition || 0
              }
            });
            console.log('📌 Pasted track at', state.playheadPosition, 's');
          }
        }
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.selectedTrack, state.selectedBus, state.copiedTrack, state.playheadPosition, state.buses, dispatch]);

  // Transport control handlers
  const handlePlayPause = useCallback(() => {
    dispatch({ type: 'TOGGLE_PLAY' });
  }, [dispatch]);

  const handleStop = useCallback(() => {
    dispatch({ type: 'SET_PLAYING', payload: false });
    dispatch({ type: 'RESET_PLAYHEAD' });
  }, [dispatch]);

  // Zoom control handlers
  const handleZoomIn = useCallback(() => {
    const newZoom = Math.min(state.zoomLevel * 1.2, 5);
    console.log('🔍 Zoom In:', state.zoomLevel.toFixed(2), '→', newZoom.toFixed(2));
    dispatch({ type: 'UPDATE_ZOOM_LEVEL', payload: newZoom });
  }, [dispatch, state.zoomLevel]);

  const handleZoomOut = useCallback(() => {
    const newZoom = Math.max(state.zoomLevel / 1.2, 0.2);
    console.log('🔍 Zoom Out:', state.zoomLevel.toFixed(2), '→', newZoom.toFixed(2));
    dispatch({ type: 'UPDATE_ZOOM_LEVEL', payload: newZoom });
  }, [dispatch, state.zoomLevel]);

  // Vertical zoom control handlers
  const handleZoomYIn = useCallback(() => {
    dispatch({ type: 'UPDATE_TRACK_HEIGHT', payload: Math.min(state.trackHeight * 1.2, 200) });
  }, [dispatch, state.trackHeight]);

  const handleZoomYOut = useCallback(() => {
    dispatch({ type: 'UPDATE_TRACK_HEIGHT', payload: Math.max(state.trackHeight / 1.2, 30) });
  }, [dispatch, state.trackHeight]);

  // Mouse wheel zoom handler (two-finger scroll for horizontal only)
  const handleWheel = useCallback((e) => {
    // Only handle two-finger scroll (Ctrl key is set on trackpad pinch/scroll)
    if (!e.ctrlKey) return;

    e.preventDefault();
    e.stopPropagation();

    const zoomFactor = e.deltaY > 0 ? 1 / 1.1 : 1.1;

    // Two-finger scroll = Horizontal zoom only
    const newZoom = Math.max(0.2, Math.min(5, state.zoomLevel * zoomFactor));
    dispatch({ type: 'UPDATE_ZOOM_LEVEL', payload: newZoom });
  }, [dispatch, state.zoomLevel]);

  // Automation control handlers
  const toggleAutomation = useCallback(() => {
    dispatch({ type: 'TOGGLE_AUTOMATION_WINDOW' });
  }, [dispatch]);

  const clearAutomation = useCallback(() => {
    if (!window.confirm('Clear all automation points?')) {
      return;
    }

    // Keep only edge points
    const edgePoints = state.automationWindow.points.filter(p => p.isEdge);
    dispatch({
      type: 'UPDATE_AUTOMATION_POINTS',
      payload: edgePoints
    });
  }, [dispatch, state.automationWindow.points]);

  const restoreSceneAutomation = useCallback(() => {
    if (!state.video.sceneChanges || state.video.sceneChanges.length === 0) {
      alert('No scene changes detected. Upload a video with scene detection first.');
      return;
    }

    console.log('🔄 Restoring scene automation points...');

    // Keep edge points
    const edgePoints = state.automationWindow.points.filter(p => p.isEdge);

    // Get the default volume (50% / 0.5)
    const defaultVolume = 0.5;
    const midVolume = edgePoints.find(p => p.time === 0)?.volume || defaultVolume;

    // Create new points at each scene change
    const scenePoints = state.video.sceneChanges
      .filter(time => time > 0 && time < state.totalDuration)
      .map(time => ({
        time,
        volume: midVolume,
        isEdge: false
      }));

    // Combine edge points and scene points
    const updatedPoints = [...edgePoints, ...scenePoints].sort((a, b) => a.time - b.time);

    console.log(`✅ Restored ${scenePoints.length} scene points`);

    dispatch({
      type: 'UPDATE_AUTOMATION_POINTS',
      payload: updatedPoints
    });
  }, [dispatch, state.automationWindow.points, state.video.sceneChanges, state.totalDuration]);

  // BPM mode toggle handler
  const toggleBPMMode = useCallback(() => {
    dispatch({ type: 'TOGGLE_BPM_MODE' });
    console.log('🎵 BPM Mode:', !state.isBPMMode ? 'ON' : 'OFF');
  }, [dispatch, state.isBPMMode]);

  // Metronome toggle handler
  const toggleMetronome = useCallback(() => {
    dispatch({ type: 'TOGGLE_METRONOME' });
    console.log('🥁 Metronome:', !state.isMetronomeOn ? 'ON' : 'OFF');
  }, [dispatch, state.isMetronomeOn]);

  // Drag and drop state for entire DAW area
  const [isDragOver, setIsDragOver] = React.useState(false);

  // Drag and drop handlers for audio files on entire DAW area
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
      const isMidi = file.name.toLowerCase().endsWith('.mid') ||
                     file.name.toLowerCase().endsWith('.midi') ||
                     file.type === 'audio/midi' ||
                     file.type === 'audio/x-midi';

      if (isMidi) {
        console.log(`🎹 MIDI file dropped on DAW:`, file.name);

        try {
          // Parse MIDI file
          const midiData = await parseMIDIFile(file);

          // Find Music bus or use a consistent ID for auto-creation
          const musicBus = state.buses.find(bus => bus.type.toLowerCase() === 'music');
          const busId = musicBus ? musicBus.id : `music-${Date.now()}`;

          const baseName = file.name.replace(/\.(mid|midi)$/i, '');

          // Filter tracks that have notes
          const tracksWithNotes = midiData.tracks.filter(track => track.notes && track.notes.length > 0);

          if (tracksWithNotes.length > 1) {
            // Multitrack MIDI - create one track per MIDI track
            console.log(`🎹 Creating ${tracksWithNotes.length} tracks for multitrack MIDI file`);

            tracksWithNotes.forEach((midiTrack, index) => {
              const trackId = `track-${Date.now()}-${index}`;
              const newTrack = {
                id: trackId,
                name: `${baseName} - Track ${index + 1}`,
                type: 'midi',
                midiData: {
                  ...midiData,
                  notes: midiTrack.notes, // Only this track's notes
                  trackIndex: index,
                  isMultitrack: true,
                  allTracks: midiData.tracks
                },
                file: file,
                duration: midiData.duration,
                startPosition: 0,
                color: `hsl(${(index * 360 / tracksWithNotes.length)}, 70%, 60%)`,
                solo: false,
                mute: false,
                volume: 1.0,
                pan: 0,
                cropStart: 0,
                cropEnd: 0
              };

              dispatch({
                type: 'ADD_TRACK',
                payload: {
                  busId: busId,
                  track: newTrack
                }
              });
            });

            console.log(`✅ MIDI file loaded with ${tracksWithNotes.length} tracks: ${file.name}`);
          } else {
            // Single track MIDI
            const trackId = `track-${Date.now()}`;
            const newTrack = {
              id: trackId,
              name: baseName,
              type: 'midi',
              midiData: midiData,
              file: file,
              duration: midiData.duration,
              startPosition: 0,
              color: `hsl(${Math.random() * 360}, 70%, 60%)`,
              solo: false,
              mute: false,
              volume: 1.0,
              pan: 0,
              cropStart: 0,
              cropEnd: 0
            };

            dispatch({
              type: 'ADD_TRACK',
              payload: {
                busId: busId,
                track: newTrack
              }
            });

            console.log(`✅ MIDI track created: ${file.name}`);
          }
        } catch (error) {
          console.error('Error parsing MIDI file:', error);
          alert(`Failed to load MIDI file: ${error.message}`);
        }
        return;
      }

      // Check if it's an audio file
      if (!file.type.startsWith('audio/')) {
        alert('Please drop an audio or MIDI file');
        return;
      }

      console.log(`🎵 Audio file dropped on DAW:`, file.name);

      // Create a blob URL for the audio file
      const audioUrl = URL.createObjectURL(file);

      // Get audio duration
      const audio = new Audio();
      audio.src = audioUrl;

      audio.addEventListener('loadedmetadata', () => {
        const duration = audio.duration;

        // Create a new SFX bus for this track
        const busId = `sfx-${Date.now()}`;
        dispatch({
          type: 'CREATE_BUS',
          payload: {
            id: busId,
            type: 'SFX',
            name: `SFX ${state.buses.filter(b => b.type === 'SFX').length + 1}`,
            expanded: true
          }
        });

        // Create the track
        const track = {
          id: `track-${Date.now()}`,
          name: file.name,
          audioUrl: audioUrl,
          duration: duration,
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
          }
        };

        // Add track to the new bus
        dispatch({
          type: 'ADD_TRACK',
          payload: { busId, track }
        });

        console.log(`✅ Audio file added to new SFX bus`);
      });
    }
  }, [dispatch, state.buses]);

  return (
    <div ref={dawGridRef} className={`${styles.dawGrid} ${pluginMode ? styles.pluginMode : ''}`}>
      {/* Zoom controls moved to timeline spacer */}

      {/* Controls Row - spans all columns with 2-column grid (NO scroll) */}

      {/* Scrollable Wrapper - Single scroll for timeline and tracks */}
      <div
        ref={timelineContainerRef}
        className={styles.scrollableWrapper}
        style={{
          height: `calc(100vh - 280px)` /* Fill viewport minus header/controls */
        }}
        onWheel={handleWheel}
      >
        {/* Timeline Row with ChordTrack/SceneMarkers - Hidden in plugin mode */}
        {!pluginMode && (
          <div className={styles.timelineContainer}>
          {/* Combined spacer - spans columns 1 & 2 with transport controls */}
          <div className={styles.timelineSpacer1} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 8px 4px 8px', justifyContent: 'space-between', paddingRight: '18px' }}>
            {/* Tempo Group: BPM Mode + Metronome + BPM Input (LEFT) */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '2px',
              background: 'rgba(30, 30, 30, 0.6)',
              padding: '4px',
              borderRadius: '6px',
              border: '1px solid rgba(102, 126, 234, 0.2)',
              flex: '0 0 auto'
            }}>
              <Button
                id="bpm-mode-btn"
                icon="fa-solid fa-music"
                onClick={toggleBPMMode}
                isActive={state.isBPMMode}
                title="Toggle BPM Mode"
                style={{ padding: '6px 10px' }}
              />

              <Button
                id="metronome-btn"
                icon="fa-solid fa-drum"
                onClick={toggleMetronome}
                isActive={state.isMetronomeOn}
                title="Toggle Metronome"
                style={{ padding: '6px 10px' }}
              />

              {/* BPM Input */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '3px' }}>
                <label htmlFor="bpm-input" style={{ color: '#c5cae9', fontSize: '11px', fontWeight: '600' }}>
                  BPM
                </label>
                <input
                  type="number"
                  id="bpm-input"
                  min="40"
                  max="240"
                  value={state.bpm}
                  onChange={(e) => dispatch({ type: 'UPDATE_BPM', payload: parseInt(e.target.value, 10) })}
                  style={{
                    width: '48px',
                    padding: '4px 5px',
                    borderRadius: '4px',
                    border: '1px solid rgba(102, 126, 234, 0.3)',
                    background: 'rgba(20, 20, 20, 0.8)',
                    color: 'white',
                    fontSize: '12px',
                    textAlign: 'center'
                  }}
                />
              </div>
            </div>

            {/* Automation + Transport Group (RIGHT) */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Button
                id="autobtn"
                icon="fa-solid fa-chart-simple"
                onClick={toggleAutomation}
                isActive={state.automationWindow.isVisible}
                title="Toggle automation"
              />

              {/* Transport Group: Play/Pause/Stop */}
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0px',
                background: 'rgba(30, 30, 30, 0.6)',
                padding: '4px',
                borderRadius: '6px',
                border: '1px solid rgba(102, 126, 234, 0.2)'
              }}>
                <TransportControls
                  isPlaying={state.isPlaying}
                  playheadPosition={state.playheadPosition}
                  onPlayPause={handlePlayPause}
                  onStop={handleStop}
                />
              </div>
            </div>
          </div>

          {/* Column 3: ChordTrack/SceneMarkers/AutomationWindow above Timeline */}
          <div style={{
            gridColumn: 3,
            position: 'relative',
            overflow: 'visible',
            height: state.automationWindow.isVisible ? '100px' : '30px',
            width: '100%',
            zIndex: 2,
            marginBottom: '4px'
          }}>
            {state.automationWindow.isVisible ? (
              <div style={{ width: '100%', height: '100%', position: 'relative', display: 'flex', alignItems: 'center' }}>
                <AutomationWindow />
              </div>
            ) : (state.video?.sceneChanges?.length > 0) ? (
              <div style={{
                width: '100%',
                height: '100%',
                position: 'relative',
                minHeight: '30px',
                display: 'flex',
                alignItems: 'center'
              }}>
                {/* Video timeline preview as bottom layer */}
                {state.video?.videoFrames && state.video.videoFrames.length > 0 && (
                  <div style={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    width: '100%',
                    height: '100%',
                    overflowX: 'auto',
                    overflowY: 'hidden',
                    zIndex: 1
                  }}>
                    <div style={{
                      display: 'flex',
                      height: '100%',
                      position: 'relative',
                      width: `${(state.timelineWidth || 700) * (state.zoomLevel || 1)}px`,
                      minWidth: '100%'
                    }}>
                      {state.video.videoFrames.map((frame, index) => {
                        const frameWidth = ((state.timelineWidth || 700) * (state.zoomLevel || 1)) / state.video.videoFrames.length;
                        return (
                          <div
                            key={index}
                            style={{
                              position: 'relative',
                              height: '30px',
                              width: `${frameWidth}px`,
                              minWidth: '80px',
                              flexShrink: 0,
                              borderRight: '1px solid rgba(255, 255, 255, 0.05)',
                              backgroundColor: 'rgba(0, 0, 0, 0.2)',
                              backgroundImage: `url(${frame.dataUrl})`,
                              backgroundSize: 'cover',
                              backgroundPosition: 'center',
                              opacity: 0.3
                            }}
                          />
                        );
                      })}
                    </div>
                  </div>
                )}
                {/* Chord track layer - underneath scene markers */}
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  zIndex: 2
                }}>
                  <ChordTrack
                    totalDuration={state.totalDuration || 10}
                    zoomLevel={state.zoomLevel || 1.0}
                    onBeatSelect={(beatNumber) => {
                      const beatIndex = beatNumber - 1;
                      dispatch({ type: 'SET_CHORD_WINDOW_BEAT', payload: beatIndex });
                    }}
                  />
                </div>
                {/* Scene markers overlay on top */}
                <div style={{
                  position: 'relative',
                  width: '100%',
                  height: '100%',
                  zIndex: 3
                }}>
                  <SceneMarkers
                    totalDuration={state.totalDuration || 10}
                    width={(state.timelineWidth || 700) * (state.zoomLevel || 1.0)}
                  />
                </div>
              </div>
            ) : (
              <ChordTrack
                totalDuration={state.totalDuration || 10}
                zoomLevel={state.zoomLevel || 1.0}
                onBeatSelect={(beatNumber) => {
                  // Convert beat number (1-based) to beat index (0-based)
                  const beatIndex = beatNumber - 1;
                  dispatch({ type: 'SET_CHORD_WINDOW_BEAT', payload: beatIndex });
                }}
              />
            )}
          </div>

          {/* Spacer row for timeline - dummy elements for grid alignment */}
          <div></div>
          <div></div>

          {/* Timeline in column 3 */}
          <Timeline
            totalDuration={state.totalDuration || 10}
            zoomLevel={state.zoomLevel || 1.0}
            onSeek={seek}
            timelineRef={timelineRef}
            playheadPosition={state.playheadPosition || 0}
            isBPMMode={state.isBPMMode}
            bpm={state.bpm || 120}
            sceneTempos={state.video?.sceneTempos || []}
            sceneChanges={state.video?.sceneChanges || []}
            onZoomIn={handleZoomIn}
            onZoomOut={handleZoomOut}
            trackHeight={state.trackHeight || 72}
            onZoomYIn={handleZoomYIn}
            onZoomYOut={handleZoomYOut}
          />
          </div>
        )}

        {/* Scrollable Content Area - Buses with Playhead Overlay */}
        <div
          ref={scrollableContentRef}
          className={`${styles.scrollableContent} ${isDragOver ? styles.dragOver : ''}`}
          style={{
            minHeight: `${maxTracksHeight}px`
          }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
        {/* Duplicate Timeline - Playhead Only (invisible ticks, visible playhead) - Hidden in plugin mode */}
        {!pluginMode && (
          <Timeline
            totalDuration={state.totalDuration || 10}
            zoomLevel={state.zoomLevel || 1.0}
            onSeek={() => {}} // No-op, interaction handled by main timeline
            timelineRef={React.createRef()} // Separate ref
            playheadPosition={state.playheadPosition || 0}
            isBPMMode={state.isBPMMode}
            bpm={state.bpm || 120}
            sceneTempos={state.video?.sceneTempos || []}
            sceneChanges={state.video?.sceneChanges || []}
            playheadOnly={true}
            skipWidthMeasurement={true} // Don't measure width - use global state from main timeline
          />
        )}

        {/* Timeline Grid - Vertical and horizontal grid lines matching timeline ticks and track heights */}
        {!pluginMode && (
          <TimelineGrid
            totalDuration={state.totalDuration || 10}
            zoomLevel={state.zoomLevel || 1.0}
            containerWidth={state.timelineWidth || 700}
            isBPMMode={state.isBPMMode}
            bpm={state.bpm || 120}
            sceneTempos={state.video?.sceneTempos || []}
            sceneChanges={state.video?.sceneChanges || []}
            buses={state.buses}
            trackHeight={state.trackHeight || 72}
            onSeek={seek}
          />
        )}

        {/* Bus Rows - dynamically rendered from buses array */}
        {state.buses.map((bus, index) => {
          // Determine icon based on bus type and instrument metadata
          const getIcon = (bus) => {
            // Check if bus has instrument metadata
            if (bus.metadata?.instrumentSubgroup) {
              const instrument = bus.metadata.instrumentSubgroup;
              // Map instruments to specific icons
              if (instrument.includes('piano')) return 'fa-solid fa-piano';
              if (instrument.includes('guitar')) return 'fa-solid fa-guitar';
              if (instrument.includes('bass')) return 'fa-solid fa-guitar-electric';
              if (instrument.includes('violin') || instrument.includes('strings')) return 'fa-solid fa-violin';
              if (instrument.includes('cello')) return 'fa-solid fa-cello';
              if (instrument.includes('trumpet') || instrument.includes('brass')) return 'fa-solid fa-trumpet';
              if (instrument.includes('trombone')) return 'fa-solid fa-trombone';
              if (instrument.includes('sax') || instrument.includes('winds') || instrument.includes('flute')) return 'fa-solid fa-saxophone';
              if (instrument.includes('drum')) return 'fa-solid fa-drum';
            }

            // Fallback to bus type
            switch (bus.type.toLowerCase()) {
              case 'vo': return 'fa-video';
              case 'music': return 'fa-music';
              case 'midi': return 'fa-music';
              case 'sfx': return 'fa-volume-high';
              default: return 'fa-music';
            }
          };

          return (
            <BusRow
              key={bus.id}
              bus={bus}
              icon={getIcon(bus)}
              trackHeight={state.trackHeight || 72}
              gainNodes={gainNodes}
              draggable={true}
              onDragStart={handleBusDragStart(index)}
              onDragOver={handleBusDragOver(index)}
              onDragLeave={handleBusDragLeave}
              onDrop={handleBusDrop(index)}
              onDragEnd={handleBusDragEnd}
              isDragOver={dragOverIndex === index}
              pluginMode={pluginMode}
            />
          );
        })}
        </div>
      </div>

    </div>
  );
});

DAWOptimized.displayName = 'DAWOptimized';

export default DAWOptimized;
