import React, { useCallback, useRef, useMemo, useState, useEffect } from 'react';
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

// Genre to instruments mapping - 4 instruments per genre for wand mode
// Genres with audio files: jazz (bb), electronic (house), reggae (rga)
const genreInstruments = {
  cinematic: [
    { name: 'Strings', group: 'strings', subgroup: 'ensemble_strings', icon: '/assets/icons/stringens.png' },
    { name: 'Brass', group: 'brass', subgroup: 'ensemble_brass', icon: '/assets/icons/tpt.png' },
    { name: 'Piano', group: 'piano', subgroup: 'acoustic_piano', icon: '/assets/icons/piano.png' },
    { name: 'Percussion', group: 'drums', subgroup: 'orchestral', icon: '/assets/icons/drumkit.png' }
  ],
  electronic: [
    { name: 'Synth Lead', group: 'keys', subgroup: 'keys', icon: '/assets/icons/keyboard.png', audioUrl: '/assets/audio/electronic/imagineleadd.wav' },
    { name: 'Bass', group: 'bass', subgroup: 'electric_bass', icon: '/assets/icons/elecbass.png', audioUrl: '/assets/audio/electronic/imaginebass.wav' },
    { name: 'Drums', group: 'drums', subgroup: 'electronic', icon: '/assets/icons/elecdrums.png', audioUrl: '/assets/audio/electronic/imaginedrums.wav' },
    { name: 'Synth', group: 'keys', subgroup: 'keys', icon: '/assets/icons/keyboard.png', audioUrl: '/assets/audio/electronic/imaginesynth.wav' }
  ],
  orchestral: [
    { name: 'Violin', group: 'strings', subgroup: 'violin', icon: '/assets/icons/violin.png' },
    { name: 'Cello', group: 'strings', subgroup: 'cello', icon: '/assets/icons/cello.png' },
    { name: 'French Horn', group: 'brass', subgroup: 'ensemble_brass', icon: '/assets/icons/tpt.png' },
    { name: 'Flute', group: 'winds', subgroup: 'flute', icon: '/assets/icons/flute.png' }
  ],
  ambient: [
    { name: 'Pad', group: 'keys', subgroup: 'keys', icon: '/assets/icons/keyboard.png' },
    { name: 'Guitar', group: 'guitar', subgroup: 'acoustic_guitar', icon: '/assets/icons/acguitar.png' },
    { name: 'Strings', group: 'strings', subgroup: 'ensemble_strings', icon: '/assets/icons/stringens.png' },
    { name: 'Winds', group: 'winds', subgroup: 'ensemble_winds', icon: '/assets/icons/windens.png' }
  ],
  rock: [
    { name: 'Electric Guitar', group: 'guitar', subgroup: 'electric_guitar', icon: '/assets/icons/elecgtr.png' },
    { name: 'Bass', group: 'bass', subgroup: 'electric_bass', icon: '/assets/icons/elecbass.png' },
    { name: 'Drums', group: 'drums', subgroup: 'rock', icon: '/assets/icons/drumkit.png' },
    { name: 'Keys', group: 'keys', subgroup: 'keys', icon: '/assets/icons/keyboard.png' }
  ],
  jazz: [
    { name: 'Drums', group: 'drums', subgroup: 'jazz', icon: '/assets/icons/drumkit.png', audioUrl: '/assets/audio/jazz/imaginationdrums.wav' },
    { name: 'Strings', group: 'strings', subgroup: 'ensemble_strings', icon: '/assets/icons/stringens.png', audioUrl: '/assets/audio/jazz/imaginationstrings.wav' },
    { name: 'Brass', group: 'brass', subgroup: 'brass', icon: '/assets/icons/tpt.png', audioUrl: '/assets/audio/jazz/imaginationbrass.wav' },
    { name: 'Piano', group: 'piano', subgroup: 'acoustic_piano', icon: '/assets/icons/piano.png', audioUrl: '/assets/audio/jazz/imaginationpno.wav' },
    { name: 'Trumpet', group: 'brass', subgroup: 'trumpet', icon: '/assets/icons/tpt.png', audioUrl: '/assets/audio/jazz/imaginationtp.wav' },
    { name: 'Trombone', group: 'brass', subgroup: 'trombone', icon: '/assets/icons/tbn.png', audioUrl: '/assets/audio/jazz/imaginationbrass.wav' }
  ],
  reggae: [
    { name: 'Guitar', group: 'guitar', subgroup: 'electric_guitar', icon: '/assets/icons/elecgtr.png', audioUrl: '/assets/audio/reggae/imaginegtr_1.wav' },
    { name: 'Bass', group: 'bass', subgroup: 'electric_bass', icon: '/assets/icons/elecbass.png', audioUrl: '/assets/audio/reggae/imaginebass_1.wav' },
    { name: 'Drums', group: 'drums', subgroup: 'reggae', icon: '/assets/icons/drumkit.png', audioUrl: '/assets/audio/reggae/imaginedrums_1.wav' },
    { name: 'Sax', group: 'winds', subgroup: 'sax', icon: '/assets/icons/sax.png' },
    { name: 'Trumpet', group: 'brass', subgroup: 'trumpet', icon: '/assets/icons/tpt.png' },
    { name: 'Synth', group: 'keys', subgroup: 'keys', icon: '/assets/icons/keyboard.png', audioUrl: '/assets/audio/reggae/imaginesynth_1.wav' }
  ],
  hiphop: [
    { name: 'Drums', group: 'drums', subgroup: 'hiphop', icon: '/assets/icons/elecdrums.png' },
    { name: 'Bass', group: 'bass', subgroup: 'electric_bass', icon: '/assets/icons/elecbass.png' },
    { name: 'Keys', group: 'keys', subgroup: 'keys', icon: '/assets/icons/keyboard.png' },
    { name: 'Guitar', group: 'guitar', subgroup: 'electric_guitar', icon: '/assets/icons/elecgtr.png' }
  ],
  world: [
    { name: 'Acoustic Guitar', group: 'guitar', subgroup: 'acoustic_guitar', icon: '/assets/icons/acguitar.png' },
    { name: 'Percussion', group: 'drums', subgroup: 'world', icon: '/assets/icons/drumkit.png' },
    { name: 'Flute', group: 'winds', subgroup: 'flute', icon: '/assets/icons/flute.png' },
    { name: 'Strings', group: 'strings', subgroup: 'ensemble_strings', icon: '/assets/icons/stringens.png' }
  ]
};

// Main DAW Component
const DAWOptimized = React.memo(({ maxTracksHeight = 600, panelWidth = 400, pluginMode = false }) => {
  const { state, dispatch } = useApp();
  const timelineRef = useRef(null);
  const scrollableContentRef = useRef(null);
  const dawGridRef = useRef(null);
  const dragIndexRef = useRef(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);

  // Marquee selection state
  const [isMarqueeSelecting, setIsMarqueeSelecting] = useState(false);
  const [marqueeStart, setMarqueeStart] = useState({ x: 0, y: 0 });
  const [marqueeEnd, setMarqueeEnd] = useState({ x: 0, y: 0 });
  const marqueeRef = useRef({ isSelecting: false, startX: 0, startY: 0 });

  // Wand mode state - creates placeholder tracks when dragging
  const [wandMode, setWandMode] = useState(true); // Wand mode on by default

  // Demo animation - triggered when play is pressed for the first time
  useEffect(() => {
    // Only run demo animation once, when play is pressed
    // Use module-level flag for synchronous check (dispatch is async)
    if (!state.isPlaying || state.demoAnimationPlayed || window.__demoAnimationStarted) {
      return;
    }

    // Mark as started immediately with synchronous flag (prevents race condition)
    window.__demoAnimationStarted = true;
    dispatch({ type: 'SET_DEMO_ANIMATION_PLAYED', payload: true });
    console.log('🎬 Starting demo animation!');

    // Step 1: After 0.5 seconds, add the main track
    const timer1 = setTimeout(() => {
      console.log('🎬 Adding main track...');
      const mainTrackId = `demo-main-${Date.now()}`;
      dispatch({
        type: 'ADD_BUS',
        payload: {
          bus: {
            id: 'demo-bus-main',
            type: 'audio',
            name: 'Generating Stems',
            animateIn: true,  // Flag for slide-in animation
            isGeneratingStems: true,  // Flag to hide controls and show loading
            tracks: [{
              id: mainTrackId,
              name: 'Generating Stems',
              type: 'audio',
              audioUrl: '/assets/audio/demo/full.wav',
              duration: 30,
              startPosition: 0,
              gain: 1.0,
              isMuted: false,
              isSolo: false,
              isGeneratingStems: true,  // Flag to hide controls and show loading animation
              cropStart: 0,
              cropEnd: 0,
              fx: { reverb: 0, fadeIn: 0.1, fadeOut: 0.5 }
            }],
            gain: 1.0,
            pan: 0,
            reverbSend: 0,
            mute: false,
            solo: false,
            expanded: false
          }
        }
      });

      // Step 2: After 1.5 seconds, transform to Vocals and add other stems
      const timer2 = setTimeout(() => {
        // Update first track to become Vocals
        dispatch({
          type: 'UPDATE_TRACK_PROPS',
          payload: {
            trackId: mainTrackId,
            name: 'Vocals',
            isGeneratingStems: false,
            instrumentGroup: 'vocals',
            instrumentIcon: '/assets/icons/microphone.png',
            audioUrl: '/assets/audio/demo/vox.wav'
          }
        });

        // Update bus name and remove loading state
        dispatch({
          type: 'UPDATE_BUS_NAME',
          payload: {
            busId: 'demo-bus-main',
            name: 'Vocals',
            isGeneratingStems: false,
            animateIn: false
          }
        });

        // Add remaining stem tracks (Strings, Celeste, Winds)
        const stemData = [
          { name: 'Strings', icon: '/assets/icons/stringens.png', group: 'strings', audioUrl: '/assets/audio/demo/strings_1.wav' },
          { name: 'Celeste', icon: '/assets/icons/keyboard.png', group: 'keys', audioUrl: '/assets/audio/demo/celeste_1.wav' },
          { name: 'Winds', icon: '/assets/icons/windens.png', group: 'winds', audioUrl: '/assets/audio/demo/winds_1.wav' }
        ];

        stemData.forEach((stem, index) => {
          setTimeout(() => {
            dispatch({
              type: 'ADD_BUS',
              payload: {
                bus: {
                  id: `demo-bus-stem-${index}`,
                  type: 'audio',
                  name: stem.name,
                  animateIn: true,  // Flag for slide-in animation
                  tracks: [{
                    id: `demo-stem-${index}-${Date.now()}`,
                    name: stem.name,
                    type: 'audio',
                    audioUrl: stem.audioUrl,
                    duration: 30,
                    startPosition: 0,
                    gain: 1.0,
                    isMuted: false,
                    isSolo: false,
                    instrumentGroup: stem.group,
                    instrumentIcon: stem.icon,
                    metadata: { type: 'stem', stemType: stem.group },
                    cropStart: 0,
                    cropEnd: 0,
                    fx: { reverb: 0, fadeIn: 0.1, fadeOut: 0.5 }
                  }],
                  gain: 1.0,
                  pan: 0,
                  reverbSend: 0,
                  mute: false,
                  solo: false,
                  expanded: false
                }
              }
            });
          }, index * 150); // Stagger each stem by 150ms
        });
      }, 1500);

      return () => clearTimeout(timer2);
    }, 500);

    return () => clearTimeout(timer1);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.isPlaying, dispatch]); // Intentionally exclude demoAnimationPlayed - we only read it, don't react to it

  // Audio playback hook - convert buses array to tracks object with bus info
  const tracksForPlayback = useMemo(() => {
    const tracks = { vo: [], music: [], sfx: [], drums: [], midi: [], audio: [] };
    state.buses.forEach(bus => {
      const typeKey = bus.type.toLowerCase();
      // Add bus info to each track so playback can apply bus-level controls
      const tracksWithBusInfo = bus.tracks
        .filter(track => !track.metadata?.isBusMaster)
        .map(track => ({
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

  // Handle form section generation (when genre clicked with section selected)
  // Replaces existing tracks like wand mode instead of adding new buses
  useEffect(() => {
    if (!state.formSectionGeneration) return;

    const { genre, section, timestamp } = state.formSectionGeneration;
    if (!genre || !section) return;

    // Clear the generation request
    dispatch({ type: 'CLEAR_FORM_SECTION_GENERATION' });
    dispatch({ type: 'SET_SELECTED_FORM_SECTION', payload: null });

    const bpm = state.bpm || 120;
    const secondsPerBeat = 60 / bpm;
    const secondsPerBar = secondsPerBeat * 4;

    // Form section definitions (in bars)
    const sectionDefs = {
      intro: { startBar: 0, endBar: 4 },
      verse: { startBar: 4, endBar: 12 },
      chorus: { startBar: 12, endBar: 20 },
      outro: { startBar: 20, endBar: Math.ceil((state.totalDuration || 36) / secondsPerBar) }
    };

    const sectionDef = sectionDefs[section];
    if (!sectionDef) return;

    const startTime = sectionDef.startBar * secondsPerBar;
    const endTime = sectionDef.endBar * secondsPerBar;
    const duration = endTime - startTime;

    // Get instruments for this genre (with audio)
    const allGenreInsts = genreInstruments[genre] || genreInstruments.cinematic;
    const genreInsts = allGenreInsts.filter(inst => inst.audioUrl);

    if (genreInsts.length === 0) {
      console.log(`⚠️ No audio instruments for genre: ${genre}`);
      return;
    }

    // Genre colors
    const genreColors = {
      jazz: '#a855f7', reggae: '#2dd4bf', electronic: '#22d3ee',
      cinematic: '#667eea', orchestral: '#818cf8', ambient: '#60a5fa',
      rock: '#f97316', hiphop: '#ec4899', world: '#fbbf24'
    };
    const genreColor = genreColors[genre] || '#667eea';

    // Get existing buses to replace
    const existingBuses = state.buses || [];
    const numToReplace = Math.min(genreInsts.length, Math.max(existingBuses.length, genreInsts.length));

    console.log(`🎵 Generating ${genreInsts.length} tracks for ${section} (bars ${sectionDef.startBar + 1}-${sectionDef.endBar}), replacing ${Math.min(existingBuses.length, genreInsts.length)} existing`);

    const newTrackIds = [];

    genreInsts.forEach((instrument, index) => {
      const newTrackId = `track-form-${timestamp}-${index}`;

      if (index < existingBuses.length) {
        // Replace existing bus's track
        const existingBus = existingBuses[index];
        const existingTrack = existingBus.tracks?.[0];

        newTrackIds.push({ trackId: newTrackId, busId: existingBus.id, instrument });

        // Crop the existing track to end at section start (if it extends past it)
        if (existingTrack) {
          const trackStart = existingTrack.startPosition || 0;
          const trackDuration = existingTrack.duration || 30;
          const trackEnd = trackStart + trackDuration - (existingTrack.cropStart || 0) - (existingTrack.cropEnd || 0);

          if (startTime > trackStart && startTime < trackEnd) {
            const newCropEnd = trackDuration - (existingTrack.cropStart || 0) - (startTime - trackStart);
            dispatch({
              type: 'UPDATE_TRACK',
              payload: {
                busId: existingBus.id,
                trackId: existingTrack.id,
                updates: { cropEnd: Math.max(0, newCropEnd) }
              }
            });
          }
        }

        // Add new track to existing bus
        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId: existingBus.id,
            track: {
              id: newTrackId,
              name: instrument.name,
              type: 'audio',
              audioUrl: null,
              duration: duration,
              startPosition: startTime,
              gain: 1.0,
              isMuted: false,
              isSolo: false,
              isPlaceholder: true,
              isLoading: true,
              instrumentGroup: instrument.group,
              instrumentSubgroup: instrument.subgroup,
              instrumentIcon: instrument.icon,
              waveformColor: genreColor,
              cropStart: 0,
              cropEnd: 0,
              fx: { reverb: 0, fadeIn: 0.1, fadeOut: 0.5 }
            }
          }
        });

        // Update bus name and icon
        dispatch({
          type: 'UPDATE_BUS_NAME',
          payload: {
            busId: existingBus.id,
            name: instrument.name,
            instrumentIcon: instrument.icon,
            instrumentGroup: instrument.group,
            instrumentSubgroup: instrument.subgroup
          }
        });
      } else {
        // Add new bus if we have more instruments than existing buses
        const newBusId = `bus-form-${timestamp}-${index}`;
        newTrackIds.push({ trackId: newTrackId, busId: newBusId, instrument });

        dispatch({
          type: 'ADD_BUS',
          payload: {
            bus: {
              id: newBusId,
              type: 'audio',
              name: instrument.name,
              instrumentIcon: instrument.icon,
              instrumentGroup: instrument.group,
              instrumentSubgroup: instrument.subgroup,
              animateIn: true,
              tracks: [{
                id: newTrackId,
                name: instrument.name,
                type: 'audio',
                audioUrl: null,
                duration: duration,
                startPosition: startTime,
                gain: 1.0,
                isMuted: false,
                isSolo: false,
                isPlaceholder: true,
                isLoading: true,
                instrumentGroup: instrument.group,
                instrumentSubgroup: instrument.subgroup,
                instrumentIcon: instrument.icon,
                waveformColor: genreColor,
                cropStart: 0,
                cropEnd: 0,
                fx: { reverb: 0, fadeIn: 0.1, fadeOut: 0.5 }
              }],
              gain: 1.0,
              pan: 0,
              reverbSend: 0,
              mute: false,
              solo: false,
              expanded: false
            }
          }
        });
      }
    });

    // Staggered loading
    newTrackIds.forEach(({ trackId, busId, instrument }, index) => {
      const baseDelay = 1500;
      const staggerDelay = (Math.random() * 0.2 + 0.4) * index * 1000;
      const totalDelay = baseDelay + staggerDelay;

      setTimeout(() => {
        dispatch({
          type: 'UPDATE_TRACK_PROPS',
          payload: {
            trackId,
            audioUrl: instrument.audioUrl || null,
            isPlaceholder: !instrument.audioUrl,
            isSettling: false,
            isLoading: false
          }
        });
        console.log(`✅ Form track ${index + 1}/${newTrackIds.length} loaded`);
      }, totalDelay);
    });

  }, [state.formSectionGeneration, dispatch, state.bpm, state.totalDuration, state.buses]);

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

  // ========== MARQUEE SELECTION HANDLERS ==========

  // Helper to get all track elements and their bounding rects
  const getTrackElements = useCallback(() => {
    if (!scrollableContentRef.current) return [];

    const trackElements = [];
    const container = scrollableContentRef.current;
    const containerRect = container.getBoundingClientRect();

    // Find all bus rows and their tracks
    state.buses.forEach((bus) => {
      if (!bus.expanded) {
        // Collapsed bus - the entire bus row is selectable
        const busElement = container.querySelector(`[data-bus-id="${bus.id}"]`);
        if (busElement) {
          const busTracksElement = busElement.querySelector(`.${styles.busTracks}`);
          if (busTracksElement) {
            const rect = busTracksElement.getBoundingClientRect();
            // All tracks in this bus share the same visual position
            bus.tracks.forEach((track) => {
              trackElements.push({
                trackId: track.id,
                busId: bus.id,
                rect: {
                  left: rect.left - containerRect.left + container.scrollLeft,
                  top: rect.top - containerRect.top + container.scrollTop,
                  right: rect.right - containerRect.left + container.scrollLeft,
                  bottom: rect.bottom - containerRect.top + container.scrollTop,
                  width: rect.width,
                  height: rect.height
                }
              });
            });
          }
        }
      } else {
        // Expanded bus - each track is individually selectable
        bus.tracks.forEach((track, trackIndex) => {
          const trackElement = container.querySelector(`[data-track-id="${track.id}"]`);
          if (trackElement) {
            const rect = trackElement.getBoundingClientRect();
            trackElements.push({
              trackId: track.id,
              busId: bus.id,
              rect: {
                left: rect.left - containerRect.left + container.scrollLeft,
                top: rect.top - containerRect.top + container.scrollTop,
                right: rect.right - containerRect.left + container.scrollLeft,
                bottom: rect.bottom - containerRect.top + container.scrollTop,
                width: rect.width,
                height: rect.height
              }
            });
          }
        });
      }
    });

    return trackElements;
  }, [state.buses]);

  // Check if two rectangles intersect
  const rectsIntersect = useCallback((rect1, rect2) => {
    return !(rect1.right < rect2.left ||
             rect1.left > rect2.right ||
             rect1.bottom < rect2.top ||
             rect1.top > rect2.bottom);
  }, []);

  // Handle marquee mouse down
  const handleMarqueeMouseDown = useCallback((e) => {
    // Only start marquee on left click and on empty space (not on tracks/controls)
    if (e.button !== 0) return;

    // Don't start marquee if clicking on a track, button, or other interactive element
    const target = e.target;

    // Check for interactive elements using data attributes and tag names
    // (CSS Module class names can have special characters that break closest())
    // In wand mode, allow clicking in bus rows (but not on actual track waveforms)
    const clickedOnTrack = target.closest('[data-track-id]');
    const clickedOnBus = target.closest('[data-bus-id]');

    // When shift is held, allow marquee to start on tracks (for replace mode)
    const isShiftReplace = state.shiftHeld;

    if (!isShiftReplace && clickedOnTrack) {
      return;
    }

    if (target.closest('button') ||
        target.closest('input') ||
        target.closest('canvas') ||
        target.tagName === 'BUTTON' ||
        target.tagName === 'INPUT') {
      return;
    }

    // In normal mode (not shift replace, not wand), also block clicks on bus rows
    if (!wandMode && !isShiftReplace && clickedOnBus) {
      return;
    }

    const container = scrollableContentRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const x = e.clientX - containerRect.left + container.scrollLeft;
    const y = e.clientY - containerRect.top + container.scrollTop;

    // Store which bus row we started in (for wand mode)
    const busElement = clickedOnBus;
    const startBusId = busElement ? busElement.getAttribute('data-bus-id') : null;

    marqueeRef.current = { isSelecting: true, startX: x, startY: y, startBusId, isShiftReplace };
    setMarqueeStart({ x, y });
    setMarqueeEnd({ x, y });
    setIsMarqueeSelecting(true);

    // Prevent text selection
    e.preventDefault();
  }, [wandMode, state.shiftHeld]);

  // Handle marquee mouse move
  const handleMarqueeMouseMove = useCallback((e) => {
    if (!marqueeRef.current.isSelecting) return;

    const container = scrollableContentRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const x = e.clientX - containerRect.left + container.scrollLeft;
    const y = e.clientY - containerRect.top + container.scrollTop;

    setMarqueeEnd({ x, y });
  }, []);

  // Handle marquee mouse up
  const handleMarqueeMouseUp = useCallback((e) => {
    if (!marqueeRef.current.isSelecting) return;

    marqueeRef.current.isSelecting = false;
    setIsMarqueeSelecting(false);

    // Calculate selection rectangle
    const selectionRect = {
      left: Math.min(marqueeStart.x, marqueeEnd.x),
      right: Math.max(marqueeStart.x, marqueeEnd.x),
      top: Math.min(marqueeStart.y, marqueeEnd.y),
      bottom: Math.max(marqueeStart.y, marqueeEnd.y)
    };

    // Only select if the marquee has some size (not just a click)
    const width = selectionRect.right - selectionRect.left;
    const height = selectionRect.bottom - selectionRect.top;

    if (width < 5 && height < 5) {
      // This was just a click, clear selection
      dispatch({ type: 'CLEAR_SELECTION' });
      return;
    }

    // SHIFT REPLACE MODE: Find tracks in selection and replace them with wand tracks
    if (marqueeRef.current.isShiftReplace) {
      const trackElements = getTrackElements();
      const tracksToReplaceMap = new Map(); // Use Map to dedupe by busId - only one track per bus

      trackElements.forEach(({ trackId, busId, rect }) => {
        if (rectsIntersect(selectionRect, rect)) {
          // Find the track data from state
          const bus = state.buses.find(b => b.id === busId);
          const track = bus?.tracks.find(t => t.id === trackId);
          if (track && bus) {
            // Only keep one track per bus (skip if we already have a track for this bus)
            // Prefer original tracks over replacement tracks (skip track-replace-* IDs)
            if (!tracksToReplaceMap.has(busId)) {
              tracksToReplaceMap.set(busId, { trackId, busId, track, bus });
            } else if (!trackId.startsWith('track-replace-') && tracksToReplaceMap.get(busId).trackId.startsWith('track-replace-')) {
              // Replace the replacement track entry with the original track
              tracksToReplaceMap.set(busId, { trackId, busId, track, bus });
            }
          }
        }
      });

      const tracksToReplace = Array.from(tracksToReplaceMap.values());

      if (tracksToReplace.length > 0) {
        // Calculate position from marquee (same as wand mode)
        const zoomLevel = state.zoomLevel || 1.0;
        const totalDuration = state.totalDuration || 36;
        const timelineWidth = state.timelineWidth || 950;
        const pixelsPerSecond = (timelineWidth * zoomLevel) / totalDuration;
        const bpm = state.bpm || 120;
        const secondsPerBar = (60 / bpm) * 4;

        // Account for bus label columns offset
        const busLabelOffset = 320;

        // Convert pixel positions to time, snap to NEAREST bar boundaries
        const startTime = Math.max(0, (selectionRect.left - busLabelOffset)) / pixelsPerSecond;
        const endTime = Math.max(0, (selectionRect.right - busLabelOffset)) / pixelsPerSecond;
        const startBar = Math.round(startTime / secondsPerBar);  // Round to nearest bar
        const endBar = Math.max(startBar + 1, Math.round(endTime / secondsPerBar));  // At least 1 bar duration
        const snappedStartTime = startBar * secondsPerBar;
        const snappedEndTime = endBar * secondsPerBar;
        const duration = snappedEndTime - snappedStartTime;

        console.log(`🔄 Shift+Replace: Replacing ${tracksToReplace.length} track(s) from bar ${startBar + 1} to bar ${endBar}`);

        // Get instrument info from generation params or genre
        const selectedInstrumentGroup = state.generationParams?.instrumentGroup;
        const selectedInstrumentSubgroup = state.generationParams?.instrumentSubgroup;
        const selectedGenre = state.selectedGenre || 'cinematic';
        const allGenreInsts = genreInstruments[selectedGenre] || genreInstruments.cinematic;
        // Filter to only instruments with audio URLs for replacement
        const genreInsts = allGenreInsts.filter(inst => inst.audioUrl);

        // Genre-specific waveform colors
        const genreColors = {
          jazz: '#a855f7',      // Violet/light purple
          reggae: '#2dd4bf',    // Aquamarine
          electronic: '#22d3ee', // Medium cyan
          cinematic: '#667eea',  // Default purple-blue
          orchestral: '#818cf8', // Indigo
          ambient: '#60a5fa',    // Light blue
          rock: '#f97316',       // Orange
          hiphop: '#ec4899',     // Pink
          world: '#fbbf24'       // Amber
        };
        const genreColor = genreColors[selectedGenre] || genreColors.cinematic;

        // Map subgroups to icons
        const subgroupIconMap = {
          'acoustic_piano': '/assets/icons/piano.png',
          'keys': '/assets/icons/keyboard.png',
          'acoustic_guitar': '/assets/icons/acguitar.png',
          'electric_guitar': '/assets/icons/elecgtr.png',
          'electric_bass': '/assets/icons/elecbass.png',
          'upright_bass': '/assets/icons/elecbass.png',
          'violin': '/assets/icons/violin.png',
          'cello': '/assets/icons/cello.png',
          'ensemble_strings': '/assets/icons/stringens.png',
          'trumpet': '/assets/icons/tpt.png',
          'trombone': '/assets/icons/tbn.png',
          'ensemble_brass': '/assets/icons/tpt.png',
          'flute': '/assets/icons/flute.png',
          'sax': '/assets/icons/sax.png',
          'ensemble_winds': '/assets/icons/windens.png'
        };

        const groupIconMap = {
          'piano': '/assets/icons/piano.png',
          'guitar': '/assets/icons/acguitar.png',
          'bass': '/assets/icons/elecbass.png',
          'strings': '/assets/icons/violin.png',
          'brass': '/assets/icons/tpt.png',
          'winds': '/assets/icons/sax.png'
        };

        // Map subgroups to audio URLs (genre-aware) for shift replace
        const subgroupAudioMap = {
          'trumpet': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationtp.wav' : null,
          'trombone': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationbrass.wav' : null,
          'brass': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationbrass.wav' : null,
          'ensemble_brass': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationbrass.wav' : null,
          'acoustic_piano': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationpno.wav' : null,
          'ensemble_strings': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationstrings.wav' : null,
          'jazz': selectedGenre === 'jazz' ? '/assets/audio/jazz/imaginationdrums.wav' : null,
          'electric_guitar': selectedGenre === 'reggae' ? '/assets/audio/reggae/imaginegtr_1.wav' : null,
          'electric_bass': selectedGenre === 'reggae' ? '/assets/audio/reggae/imaginebass_1.wav' : null,
          'reggae': selectedGenre === 'reggae' ? '/assets/audio/reggae/imaginedrums_1.wav' : null,
          'keys': selectedGenre === 'reggae' ? '/assets/audio/reggae/imaginesynth_1.wav' : null
        };

        const formatSubgroupName = (subgroup) => {
          if (!subgroup) return 'Track';
          return subgroup.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
        };

        const timestamp = Date.now();

        const newTrackIds = [];

        tracksToReplace.forEach(({ trackId, busId, track, bus }, index) => {
          // Get instrument for replacement
          let instrument;
          if (selectedInstrumentGroup) {
            const subgroup = selectedInstrumentSubgroup || selectedInstrumentGroup;
            const audioUrl = subgroupAudioMap[subgroup] || subgroupAudioMap[selectedInstrumentGroup] || null;

            // If selected instrument has no audio URL for current genre, fall back to genre instruments
            if (!audioUrl && genreInsts.length > 0) {
              instrument = genreInsts[index % genreInsts.length];
            } else {
              instrument = {
                name: formatSubgroupName(subgroup),
                group: selectedInstrumentGroup,
                subgroup: subgroup,
                icon: subgroupIconMap[subgroup] || groupIconMap[selectedInstrumentGroup] || '/assets/icons/piano.png',
                audioUrl: audioUrl
              };
            }
          } else {
            instrument = genreInsts[index % genreInsts.length];
          }

          // Calculate where the original track ends on timeline
          const originalStart = track.startPosition || 0;
          const originalDuration = track.duration || 30;
          const originalCropStart = track.cropStart || 0;
          const originalCropEnd = track.cropEnd || 0;
          const originalVisibleEnd = originalStart + originalDuration - originalCropStart - originalCropEnd;

          // Trim original track to end at snappedStartTime (if it extends past it)
          if (snappedStartTime > originalStart && snappedStartTime < originalVisibleEnd) {
            // Calculate new cropEnd to make track end at snappedStartTime
            const newVisibleDuration = snappedStartTime - originalStart;
            const newCropEnd = originalDuration - originalCropStart - newVisibleDuration;

            dispatch({
              type: 'UPDATE_TRACK',
              payload: {
                busId,
                trackId,
                updates: { cropEnd: Math.max(0, newCropEnd) }
              }
            });
          }

          // Create new track starting at snappedStartTime - with loading animation
          const newTrackId = `track-replace-${timestamp}-${index}`;
          newTrackIds.push({ trackId: newTrackId, instrument });

          const newTrack = {
            id: newTrackId,
            name: instrument.name,
            type: 'audio',
            audioUrl: null,  // Start with no audio - will be set after loading delay
            duration: duration,
            startPosition: snappedStartTime,
            gain: 1.0,
            isMuted: false,
            isSolo: false,
            cropStart: 0,
            cropEnd: 0,
            isPlaceholder: true,  // Always start as placeholder to show loading animation
            isSettling: false,  // Not settling yet - pure noise animation
            isLoading: true,  // Show loading animation
            instrumentGroup: instrument.group,
            instrumentSubgroup: instrument.subgroup,
            instrumentIcon: instrument.icon,
            waveformColor: genreColor,  // Genre-specific waveform color
            fx: { reverb: 0, fadeIn: 0.1, fadeOut: 0.5 }
          };

          // Add new track to same bus
          dispatch({
            type: 'ADD_TRACK',
            payload: { busId, track: newTrack }
          });

          // Update bus name and icon to match the new instrument
          dispatch({
            type: 'UPDATE_BUS_NAME',
            payload: {
              busId,
              name: instrument.name,
              instrumentIcon: instrument.icon,
              instrumentGroup: instrument.group,
              instrumentSubgroup: instrument.subgroup
            }
          });
        });

        // Staggered loading: 1.5 seconds base + (0.4-0.6 seconds) * index
        newTrackIds.forEach(({ trackId: newTrackId, instrument }, index) => {
          const baseDelay = 1500; // 1.5 seconds base
          const staggerDelay = (Math.random() * 0.2 + 0.4) * index * 1000; // 0.4-0.6s per track index
          const totalDelay = baseDelay + staggerDelay;

          setTimeout(() => {
            dispatch({
              type: 'UPDATE_TRACK_PROPS',
              payload: {
                trackId: newTrackId,
                audioUrl: instrument.audioUrl || null,
                isPlaceholder: !instrument.audioUrl,  // Stay as placeholder if no audio
                isSettling: false,
                isLoading: false
              }
            });
            console.log(`✅ Replace track ${index + 1}/${newTrackIds.length} loaded after ${(totalDelay / 1000).toFixed(2)}s`);
          }, totalDelay);
        });
      }

      return;
    }

    // WAND MODE: Create placeholder track instead of selecting
    if (wandMode) {
      const zoomLevel = state.zoomLevel || 1.0;
      const totalDuration = state.totalDuration || 36;
      const timelineWidth = state.timelineWidth || 950;
      // Use same formula as useTimeline hook: (containerWidth * zoomLevel) / totalDuration
      const pixelsPerSecond = (timelineWidth * zoomLevel) / totalDuration;
      const bpm = state.bpm || 120;
      const secondsPerBar = (60 / bpm) * 4; // 4 beats per bar

      // Account for bus label columns offset (320px = icon column + label column)
      // The marquee X is relative to scrollableContent, but tracks are in column 3
      const busLabelOffset = 320; // --bus-label-width CSS variable value

      // Convert pixel positions to time (seconds), accounting for label offset
      const startTime = Math.max(0, (selectionRect.left - busLabelOffset)) / pixelsPerSecond;
      const endTime = Math.max(0, (selectionRect.right - busLabelOffset)) / pixelsPerSecond;

      // Snap to bar boundaries
      const startBar = Math.floor(startTime / secondsPerBar);
      const endBar = Math.ceil(endTime / secondsPerBar);
      const snappedStartTime = startBar * secondsPerBar;
      const snappedEndTime = endBar * secondsPerBar;
      const duration = snappedEndTime - snappedStartTime;

      // Check if we started in an existing bus row
      const existingBusId = marqueeRef.current.startBusId;

      // Calculate how many tracks to create based on selection height
      const trackHeight = state.trackHeight || 72;
      const selectionHeight = selectionRect.bottom - selectionRect.top;
      const numTracks = Math.max(1, Math.round(selectionHeight / trackHeight));

      // Check if a specific instrument is selected in generation params
      const selectedInstrumentGroup = state.generationParams?.instrumentGroup;
      const selectedInstrumentSubgroup = state.generationParams?.instrumentSubgroup;

      // Map subgroups to icons
      const subgroupIconMap = {
        'acoustic_piano': '/assets/icons/piano.png',
        'keys': '/assets/icons/keyboard.png',
        'acoustic_guitar': '/assets/icons/acguitar.png',
        'electric_guitar': '/assets/icons/elecgtr.png',
        'electric_bass': '/assets/icons/elecbass.png',
        'upright_bass': '/assets/icons/elecbass.png',
        'violin': '/assets/icons/violin.png',
        'cello': '/assets/icons/cello.png',
        'ensemble_strings': '/assets/icons/stringens.png',
        'trumpet': '/assets/icons/tpt.png',
        'trombone': '/assets/icons/tbn.png',
        'ensemble_brass': '/assets/icons/tpt.png',
        'flute': '/assets/icons/flute.png',
        'sax': '/assets/icons/sax.png',
        'ensemble_winds': '/assets/icons/windens.png'
      };

      // Map groups to icons (fallback)
      const groupIconMap = {
        'piano': '/assets/icons/piano.png',
        'guitar': '/assets/icons/acguitar.png',
        'bass': '/assets/icons/elecbass.png',
        'strings': '/assets/icons/violin.png',
        'brass': '/assets/icons/brass.png',
        'winds': '/assets/icons/sax.png'
      };

      // Map subgroups to audio URLs (genre-aware)
      const activeGenre = state.selectedGenre || 'cinematic';
      const subgroupAudioMap = {
        // Jazz-specific
        'trumpet': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationtp.wav' : null,
        'trombone': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationbrass.wav' : null,
        'brass': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationbrass.wav' : null,
        'ensemble_brass': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationbrass.wav' : null,
        'acoustic_piano': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationpno.wav' : null,
        'ensemble_strings': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationstrings.wav' : null,
        'jazz': activeGenre === 'jazz' ? '/assets/audio/jazz/imaginationdrums.wav' : null,  // jazz drums
        // Reggae-specific
        'electric_guitar': activeGenre === 'reggae' ? '/assets/audio/reggae/imaginegtr_1.wav' : null,
        'electric_bass': activeGenre === 'reggae' ? '/assets/audio/reggae/imaginebass_1.wav' : null,
        'reggae': activeGenre === 'reggae' ? '/assets/audio/reggae/imaginedrums_1.wav' : null,  // reggae drums
        'keys': activeGenre === 'reggae' ? '/assets/audio/reggae/imaginesynth_1.wav' : null
      };

      // Format subgroup name for display
      const formatSubgroupName = (subgroup) => {
        if (!subgroup) return 'Track';
        return subgroup.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
      };

      // Genre-specific waveform colors
      const genreColors = {
        jazz: '#a855f7',      // Violet/light purple
        reggae: '#2dd4bf',    // Aquamarine
        electronic: '#22d3ee', // Medium cyan
        cinematic: '#667eea',  // Default purple-blue
        orchestral: '#818cf8', // Indigo
        ambient: '#60a5fa',    // Light blue
        rock: '#f97316',       // Orange
        hiphop: '#ec4899',     // Pink
        world: '#fbbf24'       // Amber
      };

      // If instrument is selected, use it; otherwise use genre
      let getInstrumentForTrack;
      let logMessage;
      let genreColor;

      if (selectedInstrumentGroup) {
        // Check if selected instrument has audio for current genre
        const instrumentAudioUrl = subgroupAudioMap[selectedInstrumentSubgroup] || subgroupAudioMap[selectedInstrumentGroup] || null;
        const genreInsts = genreInstruments[activeGenre] || genreInstruments.cinematic;

        if (instrumentAudioUrl) {
          // Use the selected instrument (has audio)
          const selectedInstrument = {
            name: formatSubgroupName(selectedInstrumentSubgroup || selectedInstrumentGroup),
            group: selectedInstrumentGroup,
            subgroup: selectedInstrumentSubgroup || selectedInstrumentGroup,
            icon: subgroupIconMap[selectedInstrumentSubgroup] || groupIconMap[selectedInstrumentGroup] || '/assets/icons/piano.png',
            audioUrl: instrumentAudioUrl
          };
          getInstrumentForTrack = () => selectedInstrument;
          genreColor = genreColors[activeGenre] || genreColors.cinematic;
          logMessage = `🪄 Wand mode: Creating ${numTracks} track(s) from bar ${startBar + 1} to bar ${endBar} (${duration.toFixed(2)}s) with ${selectedInstrument.name}${existingBusId ? ` in bus ${existingBusId}` : ''}`;
        } else {
          // Selected instrument has no audio for current genre, fall back to genre instruments
          genreColor = genreColors[activeGenre] || genreColors.cinematic;
          getInstrumentForTrack = (index) => genreInsts[index % genreInsts.length];
          logMessage = `🪄 Wand mode: Creating ${numTracks} track(s) from bar ${startBar + 1} to bar ${endBar} (${duration.toFixed(2)}s) with ${activeGenre} genre (instrument fallback)${existingBusId ? ` in bus ${existingBusId}` : ''}`;
        }
      } else {
        // Use genre-based instruments
        const selectedGenre = state.selectedGenre || 'cinematic';
        const genreInsts = genreInstruments[selectedGenre] || genreInstruments.cinematic;
        genreColor = genreColors[selectedGenre] || genreColors.cinematic;
        getInstrumentForTrack = (index) => genreInsts[index % genreInsts.length];
        logMessage = `🪄 Wand mode: Creating ${numTracks} track(s) from bar ${startBar + 1} to bar ${endBar} (${duration.toFixed(2)}s) with ${selectedGenre} genre${existingBusId ? ` in bus ${existingBusId}` : ''}`;
      }

      console.log(logMessage);

      const timestamp = Date.now();
      const trackIds = [];
      const affectedBusIds = []; // Track which buses need instrument updates

      // Find the starting bus index if we started in an existing bus
      const startingBusIndex = existingBusId
        ? state.buses.findIndex(b => b.id === existingBusId)
        : -1;

      // Create placeholder tracks
      for (let i = 0; i < numTracks; i++) {
        const trackId = `track-wand-${timestamp}-${i}`;
        trackIds.push(trackId);

        // Get instrument for this track (either selected instrument or genre-based)
        const instrument = getInstrumentForTrack(i);

        const placeholderTrack = {
          id: trackId,
          name: instrument.name,
          type: 'audio',
          audioUrl: null,  // Start with no audio - will be set after loading delay
          duration: duration,
          startPosition: snappedStartTime,
          gain: 1.0,
          isMuted: false,
          isSolo: false,
          cropStart: 0,
          cropEnd: 0,
          isPlaceholder: true,  // Always start as placeholder to show loading animation
          isSettling: false,  // Not settling yet - pure noise animation
          isLoading: true,  // Show loading animation
          instrumentGroup: instrument.group,
          instrumentSubgroup: instrument.subgroup,
          instrumentIcon: instrument.icon,
          waveformColor: genreColor,  // Genre-specific waveform color
          fx: {
            reverb: 0,
            fadeIn: 0.1,
            fadeOut: 0.5
          }
        };

        // Determine which bus to add to
        const targetBusIndex = startingBusIndex >= 0 ? startingBusIndex + i : -1;
        const targetBus = targetBusIndex >= 0 && targetBusIndex < state.buses.length
          ? state.buses[targetBusIndex]
          : null;

        if (targetBus) {
          // Add track to existing bus (starting bus or one below it)
          affectedBusIds.push(targetBus.id);
          dispatch({
            type: 'ADD_TRACK',
            payload: {
              busId: targetBus.id,
              track: placeholderTrack
            }
          });

          // Update the bus name and icon to match the new genre's instrument
          dispatch({
            type: 'UPDATE_BUS_NAME',
            payload: {
              busId: targetBus.id,
              name: instrument.name,
              instrumentIcon: instrument.icon,
              instrumentGroup: instrument.group,
              instrumentSubgroup: instrument.subgroup
            }
          });

          // Update all existing tracks in this bus with new genre's instrument
          targetBus.tracks.forEach(existingTrack => {
            dispatch({
              type: 'UPDATE_TRACK_PROPS',
              payload: {
                trackId: existingTrack.id,
                name: instrument.name,
                instrumentGroup: instrument.group,
                instrumentSubgroup: instrument.subgroup,
                instrumentIcon: instrument.icon
              }
            });
          });
        } else {
          // Create new bus for this track
          const busId = `bus-wand-${timestamp}-${i}`;
          dispatch({
            type: 'ADD_BUS',
            payload: {
              bus: {
                id: busId,
                type: 'music',
                name: instrument.name,
                tracks: [placeholderTrack],
                gain: 1.0,
                pan: 0,
                reverbSend: 0,
                mute: false,
                solo: false,
                expanded: false
              }
            }
          });
        }
      }

      // Staggered loading: 1.5 seconds base + (0.4-0.6 seconds) * index
      trackIds.forEach((trackId, index) => {
        // Get the instrument for this track to use its audioUrl
        const instrument = getInstrumentForTrack(index);
        const baseDelay = 1500; // 1.5 seconds base
        const staggerDelay = (Math.random() * 0.2 + 0.4) * index * 1000; // 0.4-0.6s per track index
        const totalDelay = baseDelay + staggerDelay;

        setTimeout(() => {
          dispatch({
            type: 'UPDATE_TRACK_PROPS',
            payload: {
              trackId: trackId,
              audioUrl: instrument.audioUrl || null,
              isPlaceholder: !instrument.audioUrl,  // Stay as placeholder if no audio
              isSettling: false,
              isLoading: false
            }
          });
          console.log(`✅ Track ${index + 1}/${trackIds.length} loaded after ${(totalDelay / 1000).toFixed(2)}s`);
        }, totalDelay);
      });

      return;
    }

    // Normal mode: Find all tracks that intersect with the selection rectangle
    const trackElements = getTrackElements();
    const selectedTrackIds = [];

    trackElements.forEach(({ trackId, rect }) => {
      if (rectsIntersect(selectionRect, rect)) {
        selectedTrackIds.push(trackId);
      }
    });

    // Dispatch multi-select action
    if (selectedTrackIds.length > 0) {
      dispatch({ type: 'SELECT_TRACKS', payload: { trackIds: selectedTrackIds } });
      console.log(`🎯 Selected ${selectedTrackIds.length} tracks via marquee`);
    } else {
      dispatch({ type: 'CLEAR_SELECTION' });
    }
  }, [marqueeStart, marqueeEnd, getTrackElements, rectsIntersect, dispatch, wandMode, state.zoomLevel, state.bpm, state.totalDuration, state.timelineWidth, state.trackHeight, state.buses, state.generationParams, state.selectedGenre]);

  // Add global mouse event listeners for marquee
  useEffect(() => {
    const handleGlobalMouseMove = (e) => {
      if (marqueeRef.current.isSelecting) {
        handleMarqueeMouseMove(e);
      }
    };

    const handleGlobalMouseUp = (e) => {
      if (marqueeRef.current.isSelecting) {
        handleMarqueeMouseUp(e);
      }
    };

    window.addEventListener('mousemove', handleGlobalMouseMove);
    window.addEventListener('mouseup', handleGlobalMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleGlobalMouseMove);
      window.removeEventListener('mouseup', handleGlobalMouseUp);
    };
  }, [handleMarqueeMouseMove, handleMarqueeMouseUp]);

  // Calculate marquee box dimensions
  const marqueeBox = useMemo(() => {
    if (!isMarqueeSelecting) return null;

    return {
      left: Math.min(marqueeStart.x, marqueeEnd.x),
      top: Math.min(marqueeStart.y, marqueeEnd.y),
      width: Math.abs(marqueeEnd.x - marqueeStart.x),
      height: Math.abs(marqueeEnd.y - marqueeStart.y)
    };
  }, [isMarqueeSelecting, marqueeStart, marqueeEnd]);

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

              {/* Wand Mode Toggle */}
              <Button
                id="wand-mode-btn"
                icon="fa-solid fa-wand-magic-sparkles"
                onClick={() => setWandMode(!wandMode)}
                isActive={wandMode}
                title={wandMode ? "Wand Mode: ON - Drag to create tracks" : "Wand Mode: OFF"}
                style={{
                  padding: '6px 10px',
                  background: wandMode ? 'linear-gradient(135deg, #9c27b0 0%, #673ab7 100%)' : undefined,
                  boxShadow: wandMode ? '0 0 12px rgba(156, 39, 176, 0.5)' : undefined
                }}
              />

              {/* Scissor / Split Tool */}
              <Button
                id="scissor-btn"
                icon="fa-solid fa-scissors"
                onClick={() => {/* TODO: implement split tool */}}
                title="Split Tool"
                style={{ padding: '6px 10px' }}
              />
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
          onMouseDown={handleMarqueeMouseDown}
        >
        {/* Marquee Selection Box */}
        {marqueeBox && (
          <div
            className={styles.marqueeSelection}
            style={{
              left: `${marqueeBox.left}px`,
              top: `${marqueeBox.top}px`,
              width: `${marqueeBox.width}px`,
              height: `${marqueeBox.height}px`
            }}
          />
        )}

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
