import React, { createContext, useContext, useReducer, useEffect, useRef } from 'react';
import * as sessionService from '../services/sessionService';

const AppContext = createContext();

// Initial state based on doseedo2.html
const initialState = {
  // Project state
  projectName: 'Untitled Session',
  isAuthenticated: true,

  // Audio state
  audioTracks: [],
  currentTrack: null,
  isPlaying: false,
  playheadPosition: 0,  // In seconds

  // Generation parameters
  generationParams: {
    instrumentGroup: null,  // No instrument selected by default
    instrumentSubgroup: null,
    generationKey: 'C',
    midiTarget: 'melody',
    drumSubgroup: 'orchestral',  // For MIDI drums: orchestral or riser
    drumTiming: 'scene_changes',  // 'scene_changes' or 'bar_pattern'
    drumPattern: 1,  // For MIDI drums: 1, 2, or 4 bars between hits (when using bar_pattern)
    activeSamples: ['bass_drum'],  // Active ORCH samples: bass_drum, timpani, cymbals, percussion
    riserTiming: 'scene_changes',  // 'scene_changes' or 'bar_pattern'
    riserPattern: 4,  // For MIDI risers: 1, 2, or 4 bars between risers (when using bar_pattern)
    seed: 0,
    steps: 20,
    adapterScale: 1.0,
    cfgWeight: 1.0,
    t0: 0.95,
    noiseLevel: 1.0,
    midiMode: false,
    renderAndExtract: false,
    renderExtractMono: false,
    tempoOverride: 120,
    tapeSpeed: 1.0,
    slowdownMethod: 'stretch',
    upsampleMode: false,
    upsampleNoiseLevel: 0.3,
    upsampleSteps: 20,
    monophonicMode: false,
    arrangeMode: false,
    fattenMode: false,
    fattenType: 'real',
    useBestOfN: false,
    nCandidates: 12,
    useSelfConsistency: false,
    consistencySamples: 3,
    extractFormats: ['midi'], // Default: extract only MIDI
    stemphonicCkpt: 'stage2d-130k', // Always default to 130k, never DoPerformer
    coverNoiseStrength: 0.2,         // UI 0-1 → backend ×7.5; 0.2 = 1.5 to model
    audioCoverStrength: 0.5,         // Fraction of steps using cover-mode encoder
  },

  // Automation state
  automationWindow: {
    isVisible: false,
    points: [],
    resolution: { width: 0, height: 100 }
  },

  // UI state
  sidebar: {
    isExpanded: false
  },
  presetsPanel: {
    isVisible: false
  },
  pluginMode: false, // Plugin mode toggle for compact DAW view - default to false
  cinemaMode: false, // Cinema mode - fullscreen video with auto-hiding panels
  cinemaPanels: { left: false, right: false, bottom: false }, // Which panels are revealed in cinema mode
  shiftHeld: false, // Track if shift key is being held (for marquee replace mode)

  // File upload state
  uploadedFile: null,
  fileType: null,
  previewUrl: null,
  sourceTrack: null, // Track info for replacement after generation

  // DAW state - New bus-based architecture supporting multiple buses per type
  // Start with empty buses for demo animation
  buses: [],
  demoAnimationPlayed: false,  // Track if demo animation has played
  selectedTrack: null,
  selectedTracks: [],  // For multi-select (marquee selection)
  selectedBus: null,  // For when a collapsed bus is clicked
  copiedTrack: null,  // Track copied with Cmd+C for paste operation
  bpm: 120,
  beatsPerBar: 4,
  meterDenominator: 4,
  // Detected per-beat tempo map: [{ t: seconds, pos: 1..beatsPerBar }, ...]
  // Where present, all timeline/chord/metronome renderers should use
  // these times directly rather than assuming a constant BPM.
  beatMap: null,
  // Detected per-bar tempo + meter map: [{ bar, t, bpm, meter: [n,d], grouping? }, ...]
  // Populated from /api/analyze-rhythm at audio upload. Each entry holds
  // the LOCAL tempo and meter for bars starting at `bar` (1-indexed) or
  // `t` seconds. In-song tempo/meter changes show up as multiple entries.
  // Consumed by Timeline, Transport, Metronome, and virtualTrackEdit —
  // never displayed via automation (user requirement).
  tempoMap: null,
  // Pre-roll seconds before bar 1 (the "pickup"). When an audio import
  // is detected to start with silence + a downbeat at t=N seconds, set
  // timelineOffset = N so bar 1 lands on the actual downbeat.
  timelineOffset: 0,
  // (tempoAutomation demo removed — tempoMap above is the source of truth
  // for in-song tempo/meter changes, populated by /api/analyze-rhythm.)
  isBPMMode: true,
  isMetronomeOn: false,
  subdivisionLevel: 1,  // 1 = quarter notes, 2 = 8th notes, 4 = 16th notes
  masterGain: 0.8,  // 80% to prevent clipping (20% reduction)
  masterFX: {
    showReverb: false,
    showEQ: false,
    reverbMix: 0.5,
    eqBands: {
      '60Hz': 0,
      '250Hz': 0,
      '1kHz': 0,
      '4kHz': 0,
      '12kHz': 0
    }
  },
  // Tuna FX Parameters
  reverbDecay: 2.5,
  reverbPreDelay: 0,
  reverbRoomSize: 0.5,
  reverbDamping: 0.5,
  delayTime: 100,
  delayFeedback: 0.45,
  delayCutoff: 20000,
  chorusRate: 1.5,
  chorusDepth: 0.7,
  chorusFeedback: 0.4,
  compressorThreshold: -20,
  compressorRatio: 4,
  compressorAttack: 0.003,
  filterFrequency: 800,
  filterResonance: 1,
  filterGain: 0,
  phaserRate: 0.1,
  phaserDepth: 0.6,
  phaserFeedback: 0.7,
  stemsSidebar: {
    isCollapsed: true
  },
  inpaintMode: {
    enabled: false,
    trackId: null
  },
  inpaintSelection: null, // { trackId, startTime, endTime }
  userInfoPage: {
    isVisible: false
  },
  chordWindow: {
    isVisible: false,
    beatIndex: null  // Which beat is being edited
  },
  chordTrack: {
    chords: {}
  },
  zoomLevel: 1.3,  // Default zoom level (130%)
  trackHeight: 48,  // Default track height in pixels (vertical zoom) - zoomed out 2 levels from 72
  zoomMode: 'x',  // Default zoom mode: 'x' for horizontal, 'y' for vertical
  totalDuration: 60,  // Default timeline duration in seconds (30 bars at 120 BPM, +6 bars outro)
  timelineWidth: 950,  // Actual timeline container width in pixels
  selectedGenre: null,  // No genre selected by default

  // Video state
  video: {
    videoId: null,
    fileName: null,
    videoFile: null,   // Original video file object for export
    videoPreviewUrl: null,  // Object URL for video preview (shared across instances)
    duration: null,
    sceneChanges: [],  // Array of scene change timestamps
    sceneTempos: [],   // Array of optimal tempo (BPM) for each scene
    audioUrl: null     // Extracted audio from video
  },

  // Chord progression
  chords: {},  // { beatIndex: chordSymbol }
  chordWindowBeat: null,  // Currently selected beat for chord editing (null = closed)

  // Undo/Redo history
  history: {
    past: [],
    future: []
  }
};

// Helper: Save current state to history
function saveToHistory(state) {
  const MAX_HISTORY = 50; // Limit history size
  const snapshot = {
    buses: state.buses,
    selectedTrack: state.selectedTrack,
    selectedTracks: state.selectedTracks,
    selectedBus: state.selectedBus,
    bpm: state.bpm,
    totalDuration: state.totalDuration
  };

  const newPast = [...state.history.past, snapshot].slice(-MAX_HISTORY);

  return {
    ...state,
    history: {
      past: newPast,
      future: [] // Clear future when new action is performed
    }
  };
}

// Reducer function
function appReducer(state, action) {
  // Handle undo/redo first
  if (action.type === 'UNDO') {
    if (state.history.past.length === 0) return state;

    const previous = state.history.past[state.history.past.length - 1];
    const newPast = state.history.past.slice(0, -1);
    const current = {
      buses: state.buses,
      selectedTrack: state.selectedTrack,
      selectedTracks: state.selectedTracks,
      selectedBus: state.selectedBus,
      bpm: state.bpm,
      totalDuration: state.totalDuration
    };

    return {
      ...state,
      ...previous,
      history: {
        past: newPast,
        future: [current, ...state.history.future]
      }
    };
  }

  if (action.type === 'REDO') {
    if (state.history.future.length === 0) return state;

    const next = state.history.future[0];
    const newFuture = state.history.future.slice(1);
    const current = {
      buses: state.buses,
      selectedTrack: state.selectedTrack,
      selectedTracks: state.selectedTracks,
      selectedBus: state.selectedBus,
      bpm: state.bpm,
      totalDuration: state.totalDuration
    };

    return {
      ...state,
      ...next,
      history: {
        past: [...state.history.past, current],
        future: newFuture
      }
    };
  }

  switch (action.type) {
    case 'SAVE_HISTORY_SNAPSHOT':
      // Explicitly save current state to history (for drag/resize start)
      return saveToHistory(state);

    case 'SET_PROJECT_NAME':
      return { ...state, projectName: action.payload };

    case 'UPDATE_GENERATION_PARAMS':
      return {
        ...state,
        generationParams: { ...state.generationParams, ...action.payload }
      };

    case 'ADD_AUDIO_TRACK':
      return {
        ...state,
        audioTracks: [...state.audioTracks, action.payload]
      };

    case 'REMOVE_AUDIO_TRACK':
      return {
        ...state,
        audioTracks: state.audioTracks.filter(track => track.id !== action.payload)
      };

    case 'SET_CURRENT_TRACK':
      return { ...state, currentTrack: action.payload };

    case 'TOGGLE_PLAY':
      return { ...state, isPlaying: !state.isPlaying };

    case 'SET_PLAYING':
      return { ...state, isPlaying: action.payload };

    case 'UPDATE_PLAYHEAD':
      return { ...state, playheadPosition: action.payload };

    case 'RESET_PLAYHEAD':
      return { ...state, playheadPosition: 0 };

    case 'SEEK_TO':
      return { ...state, playheadPosition: action.payload };

    case 'TOGGLE_AUTOMATION_WINDOW':
      return {
        ...state,
        automationWindow: {
          ...state.automationWindow,
          isVisible: !state.automationWindow.isVisible
        }
      };

    case 'UPDATE_AUTOMATION_POINTS':
      return {
        ...state,
        automationWindow: {
          ...state.automationWindow,
          points: action.payload
        }
      };

    case 'TOGGLE_SIDEBAR':
      return {
        ...state,
        sidebar: {
          ...state.sidebar,
          isExpanded: !state.sidebar.isExpanded
        }
      };

    case 'TOGGLE_PRESETS_PANEL':
      return {
        ...state,
        presetsPanel: {
          ...state.presetsPanel,
          isVisible: !state.presetsPanel.isVisible
        }
      };

    case 'TOGGLE_PLUGIN_MODE':
      return {
        ...state,
        pluginMode: !state.pluginMode
      };

    case 'TOGGLE_CINEMA_MODE':
      return {
        ...state,
        cinemaMode: !state.cinemaMode,
        cinemaPanels: { left: false, right: false, bottom: false } // Reset panels when toggling
      };

    case 'SET_CINEMA_PANEL':
      return {
        ...state,
        cinemaPanels: {
          ...state.cinemaPanels,
          [action.payload.panel]: action.payload.visible
        }
      };

    case 'SET_SHIFT_HELD':
      return {
        ...state,
        shiftHeld: action.payload
      };

    case 'SET_UPLOADED_FILE':
      return {
        ...state,
        uploadedFile: action.payload.file,
        fileType: action.payload.fileType,
        previewUrl: action.payload.previewUrl || null,
        sourceTrack: action.payload.sourceTrack || null // Store source track info for replacement
      };

    case 'RESET_STATE':
      return initialState;

    // DAW actions
    case 'CREATE_BUS':
      const { type, name, id, expanded } = action.payload;
      const busNumber = state.buses.filter(b => b.type === type).length + 1;
      const newBus = {
        id: id || `${type.toLowerCase()}-${Date.now()}`,  // Use provided id or generate new one
        type,
        name: name || `${type} ${busNumber}`,
        tracks: [],
        gain: 1.0,
        pan: 0,
        reverbSend: 0.15,
        mute: false,
        solo: false,
        expanded: expanded !== undefined ? expanded : false,  // Default to collapsed unless specified
        metadata: {}  // Initialize metadata object
      };
      return {
        ...state,
        buses: [...state.buses, newBus]
      };

    case 'UPDATE_BUS_METADATA':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, metadata: { ...(bus.metadata || {}), ...action.payload.metadata } }
            : bus
        )
      };

    case 'REORDER_BUSES':
      const { fromIndex, toIndex } = action.payload;
      const newBusesOrder = [...state.buses];
      const [movedBus] = newBusesOrder.splice(fromIndex, 1);
      newBusesOrder.splice(toIndex, 0, movedBus);
      return {
        ...state,
        buses: newBusesOrder
      };

    case 'ADD_TRACK':
      const { busId, track } = action.payload;

      // Check if bus exists, if not create it
      let busExists = state.buses.some(bus => bus.id === busId);
      let updatedBuses;

      if (!busExists) {
        console.log(`🔄 ADD_TRACK: Bus ${busId} doesn't exist, creating it...`);
        // Create the bus first
        const newBus = {
          id: busId,
          type: 'Music',
          name: 'Music',
          tracks: [track],
          gain: 1.0,
          pan: 0,
          reverbSend: 0.15,
          mute: false,
          solo: false,
          expanded: false  // Default to collapsed
        };
        updatedBuses = [...state.buses, newBus];
      } else {
        // Bus exists, add track to it
        updatedBuses = state.buses.map(bus =>
          bus.id === busId
            ? { ...bus, tracks: [...bus.tracks, track] }
            : bus
        );
      }

      const updatedBus = updatedBuses.find(b => b.id === busId);

      // Calculate the longest track duration across all buses
      let maxTrackDuration = 10; // Default minimum
      updatedBuses.forEach(bus => {
        bus.tracks.forEach(t => {
          // Track duration can be stored in different properties
          const trackDuration = t.duration || t.length || 0;
          const trackEnd = (t.startPosition || 0) + trackDuration;
          if (trackEnd > maxTrackDuration) {
            maxTrackDuration = trackEnd;
          }
        });
      });


      return {
        ...state,
        buses: updatedBuses,
        totalDuration: Math.max(state.totalDuration, maxTrackDuration)
      };

    case 'ADD_TRACKS_BULK': {
      // Adds multiple tracks to a bus in ONE state update so the
      // tree only re-renders once instead of N times. Used by the
      // auto-stem-separation flow which produces 6 stems at once.
      const { busId: bulkBusId, tracks: bulkTracks } = action.payload;
      let bulkBuses;
      const bulkBusExists = state.buses.some(bus => bus.id === bulkBusId);
      if (!bulkBusExists) {
        bulkBuses = [...state.buses, {
          id: bulkBusId, type: 'Music', name: 'Music',
          tracks: [...bulkTracks],
          gain: 1.0, pan: 0, reverbSend: 0, mute: false, solo: false, expanded: false,
        }];
      } else {
        bulkBuses = state.buses.map(bus =>
          bus.id === bulkBusId
            ? { ...bus, tracks: [...bus.tracks, ...bulkTracks] }
            : bus
        );
      }
      let bulkMaxDur = state.totalDuration || 10;
      bulkBuses.forEach(bus => {
        bus.tracks.forEach(t => {
          const td = t.duration || t.length || 0;
          const te = (t.startPosition || 0) + td;
          if (te > bulkMaxDur) bulkMaxDur = te;
        });
      });
      return { ...state, buses: bulkBuses, totalDuration: bulkMaxDur };
    }

    case 'REPLACE_TRACK':
      const { busId: replaceBusId, trackId, newTrack } = action.payload;
      console.log(`🔄 REPLACE_TRACK: Replacing track ${trackId} in bus ${replaceBusId}`);

      const busesAfterReplace = state.buses.map(bus =>
        bus.id === replaceBusId
          ? {
              ...bus,
              tracks: bus.tracks.map(t =>
                t.id === trackId ? { ...newTrack, id: trackId } : t
              )
            }
          : bus
      );

      // Recalculate timeline duration after track replacement
      let maxDurationAfterReplace = 10; // Default minimum
      busesAfterReplace.forEach(bus => {
        bus.tracks.forEach(t => {
          const trackDuration = t.duration || t.length || 0;
          const trackEnd = (t.startPosition || 0) + trackDuration;
          if (trackEnd > maxDurationAfterReplace) {
            maxDurationAfterReplace = trackEnd;
          }
        });
      });

      return {
        ...state,
        buses: busesAfterReplace,
        totalDuration: Math.max(state.totalDuration, maxDurationAfterReplace),
        sourceTrack: null // Clear sourceTrack after replacement
      };

    case 'REMOVE_TRACK':
      const busesAfterRemoval = state.buses.map(bus =>
        bus.id === action.payload.busId
          ? { ...bus, tracks: bus.tracks.filter(t => t.id !== action.payload.trackId) }
          : bus
      );

      // Recalculate timeline duration after track removal
      let maxDurationAfterRemoval = 10; // Default minimum
      busesAfterRemoval.forEach(bus => {
        bus.tracks.forEach(t => {
          const trackDuration = t.duration || t.length || 0;
          const trackEnd = (t.startPosition || 0) + trackDuration;
          if (trackEnd > maxDurationAfterRemoval) {
            maxDurationAfterRemoval = trackEnd;
          }
        });
      });

      console.log(`📏 Timeline duration after removal: ${maxDurationAfterRemoval.toFixed(2)}s`);

      return {
        ...state,
        buses: busesAfterRemoval,
        totalDuration: maxDurationAfterRemoval
      };

    case 'UPDATE_TRACK':
      // Save to history for undoable actions (position, duration, crop)
      // BUT skip if skipHistory flag is set (during drag/resize)
      const shouldSaveHistory = !action.payload.skipHistory && (
        action.payload.updates.startPosition !== undefined ||
        action.payload.updates.duration !== undefined ||
        action.payload.updates.cropStart !== undefined ||
        action.payload.updates.cropEnd !== undefined
      );

      const stateBeforeTrackUpdate = shouldSaveHistory ? saveToHistory(state) : state;

      // Clamp track position to timeline max - prevent dragging past timeline end
      const clampedUpdates = { ...action.payload.updates };
      if (clampedUpdates.startPosition !== undefined) {
        // Get track duration to calculate end position
        const track = stateBeforeTrackUpdate.buses
          .find(bus => bus.id === action.payload.busId)
          ?.tracks.find(t => t.id === action.payload.trackId);
        if (track) {
          const trackDuration = track.duration || track.length || 0;
          const visibleDuration = trackDuration - (track.cropStart || 0) - (track.cropEnd || 0);
          const maxStartPosition = Math.max(0, stateBeforeTrackUpdate.totalDuration - visibleDuration);
          clampedUpdates.startPosition = Math.max(0, Math.min(clampedUpdates.startPosition, maxStartPosition));
        }
      }

      const busesAfterUpdate = stateBeforeTrackUpdate.buses.map(bus =>
        bus.id === action.payload.busId
          ? {
              ...bus,
              tracks: bus.tracks.map(t =>
                t.id === action.payload.trackId
                  ? {
                      ...t,
                      ...clampedUpdates,
                      // Deep-merge metadata so envelopeData / stemType / etc are never
                      // wiped by a partial update like { playbackReady: true }.
                      ...(clampedUpdates.metadata
                        ? { metadata: { ...t.metadata, ...clampedUpdates.metadata } }
                        : {}),
                    }
                  : t
              )
            }
          : bus
      );

      // Update selectedTrack if it's the one being modified
      const updatedSelectedTrack = stateBeforeTrackUpdate.selectedTrack?.id === action.payload.trackId
        ? busesAfterUpdate
            .find(bus => bus.id === action.payload.busId)
            ?.tracks.find(t => t.id === action.payload.trackId)
        : stateBeforeTrackUpdate.selectedTrack;

      // Don't auto-extend timeline - keep current totalDuration
      return {
        ...stateBeforeTrackUpdate,
        buses: busesAfterUpdate,
        selectedTrack: updatedSelectedTrack
      };

    case 'UPDATE_ALL_TRACKS_POSITION':
      // Update all tracks in a bus by the same delta (for master view dragging)
      // Clamp all tracks to timeline bounds
      const busesAfterBulkUpdate = state.buses.map(bus =>
        bus.id === action.payload.busId
          ? {
              ...bus,
              tracks: bus.tracks.map(t => {
                const trackDuration = t.duration || t.length || 0;
                const visibleDuration = trackDuration - (t.cropStart || 0) - (t.cropEnd || 0);
                const maxStartPosition = Math.max(0, state.totalDuration - visibleDuration);
                const newPosition = Math.max(0, (t.startPosition || 0) + action.payload.deltaPosition);
                return {
                  ...t,
                  startPosition: Math.min(newPosition, maxStartPosition)
                };
              })
            }
          : bus
      );

      // Don't auto-extend timeline - keep current totalDuration
      return {
        ...state,
        buses: busesAfterBulkUpdate
      };

    case 'SELECT_TRACK':
      let selectedTrack = null;
      let selectedBusForTrack = null;

      // Check if this is a composite track passed directly
      if (action.payload.compositeTrack) {
        selectedTrack = action.payload.compositeTrack;
        // If busId is provided, also select the bus (for showing Bus Info alongside MIDI window)
        if (action.payload.busId) {
          selectedBusForTrack = state.buses.find(bus => bus.id === action.payload.busId);
        }
      } else {
        // Find track in buses
        for (const bus of state.buses) {
          const track = bus.tracks.find(t => t.id === action.payload.trackId);
          if (track) {
            selectedTrack = track;
            break;
          }
        }
      }
      return { ...state, selectedTrack, selectedTracks: [], selectedBus: selectedBusForTrack };

    case 'SELECT_TRACKS':
      // Multi-select tracks (from marquee selection)
      const trackIds = action.payload.trackIds || [];
      const tracksToSelect = [];
      for (const bus of state.buses) {
        for (const track of bus.tracks) {
          if (trackIds.includes(track.id)) {
            tracksToSelect.push({ ...track, _busId: bus.id });
          }
        }
      }
      return {
        ...state,
        selectedTracks: tracksToSelect,
        selectedTrack: tracksToSelect.length === 1 ? tracksToSelect[0] : null,
        selectedBus: null
      };

    case 'CLEAR_SELECTION':
      return { ...state, selectedTrack: null, selectedTracks: [], selectedBus: null };

    case 'SELECT_BUS':
      const selectedBus = state.buses.find(bus => bus.id === action.payload.busId);
      return { ...state, selectedBus, selectedTrack: null, selectedTracks: [] };

    case 'UPDATE_BPM':
      return { ...state, bpm: action.payload };

    case 'TOGGLE_BPM_MODE':
      return { ...state, isBPMMode: !state.isBPMMode };

    case 'TOGGLE_METRONOME':
      return { ...state, isMetronomeOn: !state.isMetronomeOn };

    case 'UPDATE_SUBDIVISION_LEVEL':
      return { ...state, subdivisionLevel: action.payload };

    case 'UPDATE_TIMELINE_WIDTH':
      return { ...state, timelineWidth: action.payload };

    case 'UPDATE_MASTER_GAIN':
      return { ...state, masterGain: action.payload };

    case 'TOGGLE_MASTER_REVERB_PANEL':
      return {
        ...state,
        masterFX: {
          ...state.masterFX,
          showReverb: !state.masterFX.showReverb,
          showEQ: false
        }
      };

    case 'TOGGLE_MASTER_EQ_PANEL':
      return {
        ...state,
        masterFX: {
          ...state.masterFX,
          showEQ: !state.masterFX.showEQ,
          showReverb: false
        }
      };

    case 'UPDATE_BUS_GAIN':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, gain: action.payload.gain }
            : bus
        )
      };

    case 'UPDATE_BUS_PAN':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, pan: action.payload.pan }
            : bus
        )
      };

    case 'TOGGLE_BUS_MUTE':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, mute: !bus.mute }
            : bus
        )
      };

    case 'TOGGLE_BUS_SOLO':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, solo: !bus.solo }
            : bus
        )
      };

    case 'TOGGLE_BUS_EXPANDED':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, expanded: !bus.expanded }
            : bus
        )
      };

    case 'SET_BUS_EXPANDED':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, expanded: !!action.payload.expanded }
            : bus
        )
      };

    case 'TOGGLE_TRACK_EXPANDED':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? {
                ...bus,
                tracks: bus.tracks.map(track =>
                  track.id === action.payload.trackId
                    ? {
                        ...track,
                        // undefined or true -> false (collapse), false -> true (expand)
                        expanded: track.expanded === false ? true : false
                      }
                    : track
                )
              }
            : bus
        )
      };

    case 'ADD_BUS':
      // Add a new bus (used for wand mode track creation)
      const busToAdd = action.payload.bus;
      // Calculate new total duration based on new tracks
      let durationAfterAddBus = state.totalDuration;
      busToAdd.tracks.forEach(track => {
        const trackEnd = (track.startPosition || 0) + (track.duration || 0);
        if (trackEnd > durationAfterAddBus) {
          durationAfterAddBus = trackEnd;
        }
      });
      return {
        ...state,
        buses: [...state.buses, busToAdd],
        totalDuration: durationAfterAddBus
      };

    case 'UPDATE_TRACK_GAIN':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? {
                ...bus,
                tracks: bus.tracks.map(t =>
                  t.id === action.payload.trackId
                    ? { ...t, gain: action.payload.gain }
                    : t
                )
              }
            : bus
        )
      };

    case 'UPDATE_TRACK_PAN':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? {
                ...bus,
                tracks: bus.tracks.map(t =>
                  t.id === action.payload.trackId
                    ? { ...t, pan: action.payload.pan }
                    : t
                )
              }
            : bus
        )
      };

    case 'UPDATE_BUS_REVERB':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? { ...bus, reverbSend: action.payload.reverbSend }
            : bus
        )
      };

    case 'UPDATE_TRACK_REVERB':
      console.log('🔧 UPDATE_TRACK_REVERB reducer called:', action.payload);
      console.log('🔧 Current buses:', state.buses.map(b => ({ id: b.id, tracks: b.tracks.map(t => ({ id: t.id, reverbSend: t.reverbSend })) })));

      const busesAfterReverbUpdate = state.buses.map(bus =>
        bus.id === action.payload.busId
          ? {
              ...bus,
              tracks: bus.tracks.map(t =>
                t.id === action.payload.trackId
                  ? { ...t, reverbSend: action.payload.reverbSend }
                  : t
              )
            }
          : bus
      );

      console.log('🔧 Updated buses:', busesAfterReverbUpdate.map(b => ({ id: b.id, tracks: b.tracks.map(t => ({ id: t.id, reverbSend: t.reverbSend })) })));

      // Also update selectedTrack if it's the one being modified
      const selectedTrackAfterReverb = state.selectedTrack?.id === action.payload.trackId
        ? { ...state.selectedTrack, reverbSend: action.payload.reverbSend }
        : state.selectedTrack;

      return {
        ...state,
        buses: busesAfterReverbUpdate,
        selectedTrack: selectedTrackAfterReverb
      };

    case 'UPDATE_TRACK_MIDI_DATA':
      const stateBeforeMidiUpdate = saveToHistory(state);
      return {
        ...stateBeforeMidiUpdate,
        buses: stateBeforeMidiUpdate.buses.map(bus =>
          bus.id === action.payload.busId
            ? {
                ...bus,
                tracks: bus.tracks.map(t =>
                  t.id === action.payload.trackId
                    ? {
                        ...t,
                        midiData: action.payload.midiData,
                        duration: action.payload.midiData.duration || t.duration // Update duration from MIDI data
                      }
                    : t
                )
              }
            : bus
        ),
        // Also update selectedTrack if it's the one being modified
        selectedTrack: stateBeforeMidiUpdate.selectedTrack?.id === action.payload.trackId
          ? {
              ...stateBeforeMidiUpdate.selectedTrack,
              midiData: action.payload.midiData,
              duration: action.payload.midiData.duration || stateBeforeMidiUpdate.selectedTrack.duration
            }
          : stateBeforeMidiUpdate.selectedTrack
      };

    case 'UPDATE_TRACK_AUDIO':
      // Update track's audio URL and save previous version to version history
      const { trackId: audioTrackId, newAudioUrl, versionName } = action.payload;

      return {
        ...state,
        buses: state.buses.map(bus => ({
          ...bus,
          tracks: bus.tracks.map(track => {
            if (track.id !== audioTrackId) return track;

            // Get current version info
            const currentVersions = track.metadata?.versions || [
              {
                name: 'Original',
                audioUrl: track.audioUrl,
                timestamp: new Date().toISOString()
              }
            ];
            const currentVersionIndex = track.metadata?.currentVersionIndex ?? 0;

            // Add current audio to version history (before replacing)
            const updatedVersions = [
              ...currentVersions,
              {
                name: versionName || `Version ${currentVersions.length + 1}`,
                audioUrl: newAudioUrl,
                timestamp: new Date().toISOString()
              }
            ];

            return {
              ...track,
              audioUrl: newAudioUrl,
              metadata: {
                ...track.metadata,
                versions: updatedVersions,
                currentVersionIndex: updatedVersions.length - 1
              }
            };
          })
        })),
        // Also update selectedTrack if it's the one being modified
        selectedTrack: state.selectedTrack?.id === audioTrackId
          ? {
              ...state.selectedTrack,
              audioUrl: newAudioUrl,
              metadata: {
                ...state.selectedTrack.metadata,
                versions: [
                  ...(state.selectedTrack.metadata?.versions || [
                    {
                      name: 'Original',
                      audioUrl: state.selectedTrack.audioUrl,
                      timestamp: new Date().toISOString()
                    }
                  ]),
                  {
                    name: versionName || `Version ${(state.selectedTrack.metadata?.versions?.length || 0) + 1}`,
                    audioUrl: newAudioUrl,
                    timestamp: new Date().toISOString()
                  }
                ],
                currentVersionIndex: (state.selectedTrack.metadata?.versions?.length || 1)
              }
            }
          : state.selectedTrack
      };

    case 'DELETE_TRACK':
      const stateBeforeDelete = saveToHistory(state);
      const busesAfterDelete = stateBeforeDelete.buses.map(bus =>
        bus.id === action.payload.busId
          ? {
              ...bus,
              tracks: bus.tracks.filter(t => t.id !== action.payload.trackId)
            }
          : bus
      );

      // Recalculate timeline duration after deletion
      let maxDurationAfterDelete = 10;
      busesAfterDelete.forEach(bus => {
        bus.tracks.forEach(t => {
          const trackDuration = t.duration || t.length || 0;
          const trackEnd = (t.startPosition || 0) + trackDuration;
          if (trackEnd > maxDurationAfterDelete) {
            maxDurationAfterDelete = trackEnd;
          }
        });
      });

      return {
        ...stateBeforeDelete,
        buses: busesAfterDelete,
        totalDuration: maxDurationAfterDelete,
        selectedTrack: stateBeforeDelete.selectedTrack?.id === action.payload.trackId ? null : stateBeforeDelete.selectedTrack
      };

    case 'UPDATE_TRACK_PROPS':
      // Update any track properties (used for wand mode placeholder -> loaded transition)
      const { trackId: propsTrackId, ...trackProps } = action.payload;
      return {
        ...state,
        buses: state.buses.map(bus => ({
          ...bus,
          tracks: bus.tracks.map(track =>
            track.id === propsTrackId
              ? { ...track, ...trackProps }
              : track
          )
        }))
      };

    case 'TOGGLE_TRACK_MUTE':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? {
                ...bus,
                tracks: bus.tracks.map(t =>
                  t.id === action.payload.trackId
                    ? { ...t, isMuted: !t.isMuted }
                    : t
                )
              }
            : bus
        )
      };

    case 'TOGGLE_TRACK_SOLO':
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === action.payload.busId
            ? {
                ...bus,
                tracks: bus.tracks.map(t =>
                  t.id === action.payload.trackId
                    ? { ...t, isSolo: !t.isSolo }
                    : t
                )
              }
            : bus
        )
      };

    case 'TOGGLE_STEMS_SIDEBAR':
      return {
        ...state,
        stemsSidebar: {
          ...state.stemsSidebar,
          isCollapsed: !state.stemsSidebar.isCollapsed
        }
      };

    case 'TOGGLE_USER_INFO_PAGE':
      return {
        ...state,
        userInfoPage: {
          ...state.userInfoPage,
          isVisible: !state.userInfoPage.isVisible
        }
      };

    case 'DOWNLOAD_TRACK':
      // Foundation for track download functionality
      console.log('Download track:', action.payload.trackId);
      return state;

    case 'REGENERATE_TRACK':
      // Foundation for track regeneration functionality
      console.log('Regenerate track:', action.payload.trackId);
      return state;

    case 'SEPARATE_STEMS':
      // Foundation for stem separation functionality
      console.log('Separate stems for track:', action.payload.trackId);
      return state;

    case 'UPDATE_ZOOM_LEVEL':
      return { ...state, zoomLevel: action.payload };

    case 'UPDATE_TRACK_HEIGHT':
      return { ...state, trackHeight: action.payload };

    case 'SET_ZOOM_MODE':
      return { ...state, zoomMode: action.payload };

    case 'SET_SELECTED_GENRE':
      return { ...state, selectedGenre: action.payload };

    case 'SET_SELECTED_FORM_SECTION':
      return { ...state, selectedFormSection: action.payload };

    case 'SET_DRUM_AUTOMATION_MODE':
      // payload: { enabled: boolean, parameter: string (cutoff/fx/volume/attack/decay/tone) }
      return { ...state, drumAutomationMode: action.payload };

    case 'SET_DRUM_AUTOMATION_SHAPE':
      // payload: { section: string, parameter: string, shape: 'ramp-up' | 'ramp-down' | 'flat' }
      return {
        ...state,
        drumAutomation: {
          ...state.drumAutomation,
          [action.payload.section]: {
            ...(state.drumAutomation?.[action.payload.section] || {}),
            [action.payload.parameter]: action.payload.shape
          }
        }
      };

    case 'GENERATE_FOR_FORM_SECTION':
      // Store the generation request - DAWOptimized will handle it
      return {
        ...state,
        formSectionGeneration: {
          ...action.payload,
          timestamp: Date.now()
        }
      };

    case 'CLEAR_FORM_SECTION_GENERATION':
      return { ...state, formSectionGeneration: null };

    case 'SET_DEMO_ANIMATION_PLAYED':
      return { ...state, demoAnimationPlayed: action.payload };

    case 'UPDATE_BUS_NAME':
      // Update bus name and optionally other properties
      const { busId: updateBusId, ...busUpdates } = action.payload;
      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === updateBusId
            ? { ...bus, ...busUpdates }
            : bus
        )
      };

    case 'CLEAR_BUS':
      const busesAfterClear = state.buses.map(bus =>
        bus.id === action.payload.busId
          ? { ...bus, tracks: [] }
          : bus
      );

      // Recalculate timeline duration after clearing bus
      let maxDurationAfterClear = 10; // Default minimum
      busesAfterClear.forEach(bus => {
        bus.tracks.forEach(t => {
          const trackDuration = t.duration || t.length || 0;
          const trackEnd = (t.startPosition || 0) + trackDuration;
          if (trackEnd > maxDurationAfterClear) {
            maxDurationAfterClear = trackEnd;
          }
        });
      });

      console.log(`📏 Timeline duration after clearing bus: ${maxDurationAfterClear.toFixed(2)}s`);

      return {
        ...state,
        buses: busesAfterClear,
        totalDuration: maxDurationAfterClear
      };

    case 'UPDATE_TOTAL_DURATION':
      return { ...state, totalDuration: action.payload };

    case 'UPDATE_TIMELINE_WIDTH':
      return { ...state, timelineWidth: action.payload };

    // Chord actions
    case 'UPDATE_CHORD':
      return {
        ...state,
        chords: {
          ...state.chords,
          [action.payload.beatIndex]: action.payload.chord
        }
      };

    case 'DELETE_CHORD':
      const newChords = { ...state.chords };
      delete newChords[action.payload.beatIndex];
      return {
        ...state,
        chords: newChords
      };

    case 'SET_CHORD_WINDOW_BEAT':
      return {
        ...state,
        chordWindow: {
          isVisible: true,
          beatIndex: action.payload
        }
      };

    case 'CLOSE_CHORD_WINDOW':
      return {
        ...state,
        chordWindow: {
          isVisible: false,
          beatIndex: null
        }
      };

    case 'SET_BEATS_PER_BAR':
      return { ...state, beatsPerBar: Math.max(1, parseInt(action.payload, 10) || 4) };
    case 'SET_BEAT_MAP':
      return { ...state, beatMap: action.payload || null };
    case 'SET_PROJECT_TEMPO_MAP':
      // Project-level tempo/meter map — [{bar, t, bpm, meter: [n,d], grouping?}].
      // Populated when an uploaded file is analyzed; consumed by Timeline,
      // Transport, Metronome, and virtualTrackEdit to drive per-bar local
      // tempo + meter. Independent of automation (user requirement).
      return { ...state, tempoMap: action.payload || null };
    case 'SET_METER': {
      // payload: "N/D" string e.g. "7/8"
      const [n, d] = String(action.payload).split('/').map((x) => parseInt(x, 10));
      if (!n || !d) return state;
      // No-op when the value is unchanged. Auto-detect dispatches
      // SET_METER after rhythm analysis even when the detected meter
      // equals the current default (4/4 → 4/4), and without this guard
      // every detection caused downstream effects (live-reschedule,
      // meter-log fan-out) to fire as if the user had flipped the
      // meter dropdown.
      if (n === state.beatsPerBar && d === state.meterDenominator) return state;
      return { ...state, beatsPerBar: n, meterDenominator: d };
    }

    case 'SET_TIMELINE_OFFSET':
      return { ...state, timelineOffset: Math.max(0, parseFloat(action.payload) || 0) };

    case 'SET_CHORDS':
      return {
        ...state,
        chordTrack: {
          ...state.chordTrack,
          chords: action.payload || {}
        }
      };

    case 'CLEAR_CHORDS':
      return {
        ...state,
        chordTrack: { ...state.chordTrack, chords: {} }
      };

    case 'SET_CHORD_FOR_BEAT': {
      // Writing a falsy chord (null / '' / undefined) deletes the cell —
      // lets UIs use one action for both set and clear.
      const next = { ...(state.chordTrack?.chords || {}) };
      if (action.payload.chord) {
        next[action.payload.beatIndex] = action.payload.chord;
      } else {
        delete next[action.payload.beatIndex];
      }
      return {
        ...state,
        chordTrack: { ...state.chordTrack, chords: next },
      };
    }

    // Video actions
    case 'SET_VIDEO_INFO':
      return {
        ...state,
        video: {
          ...state.video,
          videoId: action.payload.videoId,
          fileName: action.payload.fileName,
          videoFile: action.payload.videoFile || state.video.videoFile,  // Store video file if provided
          videoPreviewUrl: action.payload.videoPreviewUrl || state.video.videoPreviewUrl  // Store preview URL if provided
        }
      };

    case 'SET_SCENE_CHANGES':
      return {
        ...state,
        video: {
          ...state.video,
          sceneChanges: action.payload.sceneChanges,
          sceneTempos: action.payload.sceneTempos,
          duration: action.payload.videoDuration
        }
      };

    case 'ADD_VIDEO_AUDIO':
      return {
        ...state,
        video: {
          ...state.video,
          audioUrl: action.payload.audioUrl
        }
      };

    case 'SET_VIDEO_FRAMES':
      return {
        ...state,
        video: {
          ...state.video,
          videoFrames: action.payload
        }
      };

    case 'CLEAR_VIDEO':
      return {
        ...state,
        video: {
          videoId: null,
          fileName: null,
          videoFile: null,
          videoPreviewUrl: null,
          duration: null,
          sceneChanges: [],
          sceneTempos: [],
          audioUrl: null
        }
      };

    case 'SET_INPAINT_MODE':
      return {
        ...state,
        inpaintMode: {
          enabled: action.payload.enabled,
          trackId: action.payload.trackId
        }
      };

    case 'SET_INPAINT_SELECTION':
      return {
        ...state,
        inpaintSelection: action.payload
      };

    case 'CLEAR_INPAINT_SELECTION':
      return {
        ...state,
        inpaintSelection: null
      };

    case 'LOAD_SESSION':
      // Load full session state from saved data
      // Migrate old tracks to ensure they have reverbSend property
      const migratedPayload = {
        ...action.payload,
        buses: action.payload.buses?.map(bus => ({
          ...bus,
          reverbSend: bus.reverbSend !== undefined ? bus.reverbSend : 0, // 0% default for buses
          tracks: bus.tracks?.map(track => ({
            ...track,
            reverbSend: track.reverbSend !== undefined ? track.reverbSend : 0.15, // 15% default for tracks
            pan: track.pan !== undefined ? track.pan : 0
          })) || []
        })) || []
      };

      return {
        ...state,
        ...migratedPayload
      };

    case 'RESET_SESSION':
      // Reset to initial state (for new projects)
      return {
        ...initialState,
        projectName: action.payload?.projectName || initialState.projectName
      };

    case 'REORDER_TRACK_IN_BUS': {
      // Move a track within its bus. payload = { busId, trackId, toIndex }
      const { busId: rBusId, trackId: rTrackId, toIndex } = action.payload;
      return {
        ...state,
        buses: state.buses.map((bus) => {
          if (bus.id !== rBusId) return bus;
          const idx = (bus.tracks || []).findIndex((t) => t.id === rTrackId);
          if (idx < 0) return bus;
          const next = bus.tracks.slice();
          const [moved] = next.splice(idx, 1);
          const dest = Math.max(0, Math.min(next.length, toIndex));
          next.splice(dest, 0, moved);
          return { ...bus, tracks: next };
        }),
      };
    }

    case 'COPY_TRACK':
      // Store track for copying (Cmd+C)
      return {
        ...state,
        copiedTrack: action.payload.track
      };

    case 'PASTE_TRACK':
      // Paste track at playhead position (Cmd+V)
      const { targetBusId, playheadPosition } = action.payload;
      if (!state.copiedTrack) return state;

      // Clone the track with new ID and position
      const pastedTrack = {
        ...state.copiedTrack,
        id: `track-${Date.now()}`,
        startPosition: playheadPosition,
        name: `${state.copiedTrack.name} (Copy)`
      };

      // If it's an audio track with a blob URL, we need to keep the same URL
      // (it will still reference the same audio data in memory)

      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === targetBusId
            ? {
                ...bus,
                tracks: [...bus.tracks, pastedTrack]
              }
            : bus
        )
      };

    case 'DUPLICATE_TRACK':
      // Duplicate selected track in same bus
      const dupTrackPayload = action.payload;
      const duplicatedTrack = {
        ...dupTrackPayload.track,
        id: `track-${Date.now()}`,
        name: `${dupTrackPayload.track.name} (Copy)`,
        startPosition: dupTrackPayload.track.startPosition + 0.5 // Offset by 0.5s
      };

      return {
        ...state,
        buses: state.buses.map(bus =>
          bus.id === dupTrackPayload.busId
            ? {
                ...bus,
                tracks: [...bus.tracks, duplicatedTrack]
              }
            : bus
        ),
        selectedTrack: duplicatedTrack
      };

    case 'DUPLICATE_BUS':
      // Duplicate entire bus with all tracks
      const srcBus = state.buses.find(b => b.id === action.payload.busId);
      if (!srcBus) return state;

      const newBusId = `${srcBus.type.toLowerCase()}-${Date.now()}`;
      const duplicatedBus = {
        ...srcBus,
        id: newBusId,
        name: `${srcBus.name} (Copy)`,
        tracks: srcBus.tracks.map(track => ({
          ...track,
          id: `track-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
        }))
      };

      return {
        ...state,
        buses: [...state.buses, duplicatedBus],
        selectedBus: duplicatedBus
      };

    // Tuna FX Parameter Updates
    case 'SET_REVERB_DECAY':
      return { ...state, reverbDecay: action.payload };
    case 'SET_REVERB_PREDELAY':
      return { ...state, reverbPreDelay: action.payload };
    case 'SET_REVERB_ROOM_SIZE':
      return { ...state, reverbRoomSize: action.payload };
    case 'SET_REVERB_DAMPING':
      return { ...state, reverbDamping: action.payload };
    case 'SET_REVERB_MIX':
      return { ...state, masterFX: { ...state.masterFX, reverbMix: action.payload } };

    case 'SET_DELAY_TIME':
      return { ...state, delayTime: action.payload };
    case 'SET_DELAY_FEEDBACK':
      return { ...state, delayFeedback: action.payload };
    case 'SET_DELAY_CUTOFF':
      return { ...state, delayCutoff: action.payload };

    case 'SET_CHORUS_RATE':
      return { ...state, chorusRate: action.payload };
    case 'SET_CHORUS_DEPTH':
      return { ...state, chorusDepth: action.payload };
    case 'SET_CHORUS_FEEDBACK':
      return { ...state, chorusFeedback: action.payload };

    case 'SET_COMPRESSOR_THRESHOLD':
      return { ...state, compressorThreshold: action.payload };
    case 'SET_COMPRESSOR_RATIO':
      return { ...state, compressorRatio: action.payload };
    case 'SET_COMPRESSOR_ATTACK':
      return { ...state, compressorAttack: action.payload };

    case 'SET_FILTER_FREQUENCY':
      return { ...state, filterFrequency: action.payload };
    case 'SET_FILTER_RESONANCE':
      return { ...state, filterResonance: action.payload };
    case 'SET_FILTER_GAIN':
      return { ...state, filterGain: action.payload };

    case 'SET_PHASER_RATE':
      return { ...state, phaserRate: action.payload };
    case 'SET_PHASER_DEPTH':
      return { ...state, phaserDepth: action.payload };
    case 'SET_PHASER_FEEDBACK':
      return { ...state, phaserFeedback: action.payload };

    default:
      return state;
  }
}

// Provider component
export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);
  const autoSaveTimeoutRef = useRef(null);
  const isInitialMount = useRef(true);

  // Persist project name to localStorage
  useEffect(() => {
    if (state.projectName !== 'Untitled Session') {
      localStorage.setItem('projectName', state.projectName);
    }
  }, [state.projectName]);

  // Load project name from localStorage on mount
  useEffect(() => {
    const savedProjectName = localStorage.getItem('projectName');
    if (savedProjectName) {
      dispatch({ type: 'SET_PROJECT_NAME', payload: savedProjectName });
    }
  }, []);

  // One-time migration: Set all bus reverb sends to 0 if they're still at old default
  const migrationRan = useRef(false);
  useEffect(() => {
    // Only run once, and only if we have buses loaded
    if (migrationRan.current || state.buses.length === 0) return;

    const needsMigration = state.buses.some(bus => bus.reverbSend === 0.15);

    if (needsMigration) {
      migrationRan.current = true;

      // Migrate each bus individually to trigger proper updates
      state.buses.forEach((bus, index) => {
        if (bus.reverbSend === 0.15) {
          dispatch({
            type: 'UPDATE_BUS_REVERB',
            payload: { busId: bus.id, reverbSend: 0 }
          });
        }
      });
    }
  }, [state.buses, state.buses.length]); // Run when buses change

  // Auto-save session to localStorage (debounced)
  useEffect(() => {
    // Skip autosave on initial mount
    if (isInitialMount.current) {
      isInitialMount.current = false;
      return;
    }

    // Only autosave if there's an active project
    const activeProject = sessionService.getActiveProject();
    if (!activeProject) {
      return;
    }

    // Clear existing timeout
    if (autoSaveTimeoutRef.current) {
      clearTimeout(autoSaveTimeoutRef.current);
    }

    // Set new timeout for autosave (3 seconds after last change)
    autoSaveTimeoutRef.current = setTimeout(() => {
      sessionService.saveSession(activeProject, state);
    }, 3000);

    // Cleanup
    return () => {
      if (autoSaveTimeoutRef.current) {
        clearTimeout(autoSaveTimeoutRef.current);
      }
    };
  }, [state.buses, state.video, state.generationParams, state.bpm, state.masterGain, state.masterFX, state.zoomLevel, state.totalDuration]);

  // Keyboard shortcuts for undo/redo and delete
  useEffect(() => {
    const handleKeyDown = (e) => {
      const isMac = navigator.platform.toUpperCase().indexOf('MAC') >= 0;
      const cmdOrCtrl = isMac ? e.metaKey : e.ctrlKey;

      // Delete key - delete selected track
      if (e.key === 'Delete' || e.key === 'Backspace') {
        if (state.selectedTrack && !e.target.matches('input, textarea')) {
          e.preventDefault();

          // Find which bus contains the selected track
          for (const bus of state.buses) {
            const track = bus.tracks.find(t => t.id === state.selectedTrack.id);
            if (track) {
              dispatch({
                type: 'DELETE_TRACK',
                payload: {
                  busId: bus.id,
                  trackId: state.selectedTrack.id
                }
              });
              break;
            }
          }
        }
      }

      // Cmd+Z / Ctrl+Z - Undo
      if (cmdOrCtrl && e.key === 'z' && !e.shiftKey && !e.target.matches('input, textarea')) {
        e.preventDefault();
        dispatch({ type: 'UNDO' });
      }

      // Cmd+Shift+Z / Ctrl+Shift+Z - Redo
      if (cmdOrCtrl && e.key === 'z' && e.shiftKey && !e.target.matches('input, textarea')) {
        e.preventDefault();
        dispatch({ type: 'REDO' });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [state.selectedTrack, state.buses, dispatch]);

  // Track shift key state for marquee replace mode
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Shift' && !state.shiftHeld) {
        dispatch({ type: 'SET_SHIFT_HELD', payload: true });
      }
    };

    const handleKeyUp = (e) => {
      if (e.key === 'Shift' && state.shiftHeld) {
        dispatch({ type: 'SET_SHIFT_HELD', payload: false });
      }
    };

    // Also handle blur to reset shift when window loses focus
    const handleBlur = () => {
      if (state.shiftHeld) {
        dispatch({ type: 'SET_SHIFT_HELD', payload: false });
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    window.addEventListener('keyup', handleKeyUp);
    window.addEventListener('blur', handleBlur);

    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      window.removeEventListener('keyup', handleKeyUp);
      window.removeEventListener('blur', handleBlur);
    };
  }, [state.shiftHeld, dispatch]);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

// Custom hook to use the context
export function useApp() {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
}
