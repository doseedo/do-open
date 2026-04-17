import React, { useCallback, useRef, useMemo, useState, useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import { useAudioPlayback } from '../../hooks/useAudioPlayback';
import { useKeyboardControls } from '../../hooks/useKeyboardControls';
import { useMetronome } from '../../hooks/useMetronome';
import { parseMIDIFile } from '../../utils/midiParser';
import { analyzeAudio, iconForType, separateStemsAuto, repaintMeter, encodeLatentsBulk, detectChordsAndTempo } from '../../services/trackAnalysisAPI';
import { meterChangeAllStems, initLatentEditor } from '../../services/latentMeterChange';
import { initDrumSep, splitDrumLatent } from '../../services/latentDrumSep';
import { initMaskPlayback, createMaskPlaybackNode, setMaster, computeAndSetMasks, setRefinedMasks, precomputeStemAudio } from '../../services/maskPlayback';
import {
  latentIdToBlobUrl,
  initLatentDecoder,
  decodeLatentFrameRange,
  decodeLatentFrameRangeBatched,
  concatChannelsFirstStereo,
  audioDataToBlobUrl,
  parseDoae,
  buildDoae,
} from '../../services/latentDecoder';
import {
  audioFileToStereo48k,
  uploadLatent,
} from '../../services/latentDemucs';
import {
  initLatentDemucsV4,
  separate6Stems,
  STEM_NAMES_V4COND_6,
} from '../../services/latentDemucsV4';
import { analyzeRms } from '../../services/rmsDemucs';
import { initLatentVisual, envelopeFromLatent, buildFakeBufferFromEnvelope } from '../../services/latentVisual';
import { audioBufferCache } from '../../hooks/useWaveform';
import ImportAudioModal from './ImportAudioModal';
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
const DAWOptimized = React.memo(({ maxTracksHeight = 600, busLabelWidth = 300, pluginMode = false }) => {
  const { state, dispatch } = useApp();
  const timelineRef = useRef(null);
  const scrollableContentRef = useRef(null);
  const dawGridRef = useRef(null);
  const dragIndexRef = useRef(null);
  const [dragOverIndex, setDragOverIndex] = useState(null);

  // Progressive visibility driven by the current bus-label column width.
  // Each breakpoint is the smallest width at which that element still fits
  // inside its 50%-of-bus-label cell without clipping the dropdown or
  // pushing the right cluster out of view. Labels ("BPM", "Meter") drop
  // out first — the dropdowns themselves always stay.
  const showTempoLabels  = busLabelWidth >= 380;
  const showMetronome    = busLabelWidth >= 320;
  const showAutomation   = busLabelWidth >= 280;
  const showTimeDisplay  = busLabelWidth >= 280;

  // Marquee selection state
  const [isMarqueeSelecting, setIsMarqueeSelecting] = useState(false);
  const [marqueeStart, setMarqueeStart] = useState({ x: 0, y: 0 });
  const [marqueeEnd, setMarqueeEnd] = useState({ x: 0, y: 0 });
  const marqueeRef = useRef({ isSelecting: false, startX: 0, startY: 0 });

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
    state.video?.sceneChanges || [],
    state.beatsPerBar || 4,
    state.meterDenominator || 4,
    state.timelineOffset || 0,
    state.beatMap || null
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

  // ── PREWARM on studio mount ──────────────────────────────────────
  // 1. Ping /health to wake Modal backend (cold start takes ~40-50s;
  //    by the time the user clicks Generate it should be warm).
  // 2. Start downloading the ONNX models for latent_demucs + decoder
  //    so they're ready when the user drops audio. Both are ~325 MB
  //    and cache in the browser after the first download.
  React.useEffect(() => {
    // Backend prewarm — fire and forget
    fetch('/health').catch(() => {});

    // Prewarm the lightweight rmsDemucs model (577 KB) so it's ready
    // when the user drops their first audio file.
    import('../../services/rmsDemucs').then(({ initRmsDemucs }) =>
      initRmsDemucs().catch(() => {})
    ).catch(() => {});

    console.log('[prewarm] studio opened — warming Modal backend + rmsDemucs model');
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  // Handler for adding a new bus. Lives here so the Add Track button in
  // the shared 2×2 .spacerArea can call it. Named "Track N" counting ALL
  // existing buses so the label matches the user's mental model of a flat
  // track list. Bus is created collapsed — double-clicking it later adds
  // a MIDI region inline (see BusRow.handleDoubleClick).
  const handleAddBus = useCallback(() => {
    const trackNumber = state.buses.length + 1;
    dispatch({
      type: 'CREATE_BUS',
      payload: {
        id: `track-${Date.now()}`,
        type: 'SFX',
        name: `Track ${trackNumber}`,
        expanded: false,
      },
    });
  }, [dispatch, state.buses.length]);

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

  // Metronome toggle handler
  const toggleMetronome = useCallback(() => {
    dispatch({ type: 'TOGGLE_METRONOME' });
    console.log('🥁 Metronome:', !state.isMetronomeOn ? 'ON' : 'OFF');
  }, [dispatch, state.isMetronomeOn]);

  // ─── Auto-repaint on BPM/meter change ──────────────────────────────
  // Snapshot what we last successfully repainted from. The next change
  // remaps every track's cached latent from the snapshot to the current
  // values via stemphonic stage2d-130k.
  const [repaintApplying, setRepaintApplying] = React.useState(false);
  const repaintLastRef = React.useRef({ bpm: state.bpm, beatsPerBar: state.beatsPerBar || 4, meterDen: state.meterDenominator || 4 });
  const repaintDebounceRef = React.useRef(null);
  const repaintInFlightRef = React.useRef(false);

  const runRepaintNow = useCallback(async () => {
    if (repaintInFlightRef.current) return;
    const srcBpm = repaintLastRef.current.bpm;
    const srcBeats = repaintLastRef.current.beatsPerBar;
    const srcDen = repaintLastRef.current.meterDen || 4;
    const tgtBpm = state.bpm;
    const tgtBeats = state.beatsPerBar || 4;
    const tgtDen = state.meterDenominator || 4;
    if (srcBpm === tgtBpm && srcBeats === tgtBeats && srcDen === tgtDen) return;

    // Collect STEM tracks. Stem tracks may NOT have a cached latent
    // yet (we deferred their encoding to save upfront cost on upload).
    // We'll lazy-encode any missing latents in one bulk call below.
    const stemTracksToProcess = []; // {bus, track}
    const parentsToMute = [];
    state.buses?.forEach((bus) => {
      const busStems = (bus.tracks || []).filter((t) => t.metadata?.type === 'stem');
      const busParents = (bus.tracks || []).filter(
        (t) => t.metadata?.type !== 'stem' && t.metadata?.latentId
      );
      busStems.forEach((track) => {
        stemTracksToProcess.push({ bus, track });
      });
      // Mute parents only if THIS bus has stems (so we don't hear master + stems doubled)
      if (busStems.length > 0) {
        busParents.forEach((p) => parentsToMute.push({ busId: bus.id, trackId: p.id }));
      }
    });

    // Lazy-encode any stem tracks that don't have a latent yet
    const needsEncode = stemTracksToProcess
      .filter(({ track }) => !track.metadata?.latentId && track.audioUrl)
      .map(({ track }) => track.audioUrl);

    if (needsEncode.length > 0) {
      console.log(`🎚️ lazy-encoding ${needsEncode.length} stem latents before repaint…`);
      try {
        const enc = await encodeLatentsBulk(needsEncode);
        // Map URL → latent_id and dispatch UPDATE_TRACK for each
        const urlToLatent = {};
        (enc.results || []).forEach((r) => {
          if (r.latent_id) urlToLatent[r.url] = { id: r.latent_id, url: r.latent_url };
        });
        stemTracksToProcess.forEach(({ bus, track }) => {
          const lat = urlToLatent[track.audioUrl];
          if (lat) {
            track.metadata = { ...track.metadata, latentId: lat.id, latent: lat.url };
            dispatch({
              type: 'UPDATE_TRACK',
              payload: {
                busId: bus.id, trackId: track.id,
                updates: { metadata: track.metadata },
              },
            });
          }
        });
      } catch (err) {
        console.error('lazy stem-latent encode failed:', err);
      }
    }

    // Now build the stems array from tracks that have a latent.
    // CRITICAL: pull the SOURCE BPM from the parent track's
    // metadata.detectedBpm. The latent_remap_meter math depends on
    // frames_per_beat = (60/bpm)*25 — if we pass the timeline default
    // instead of the song's actual BPM, the bar slicing falls on the
    // wrong latent frames and the diffusion model just smears the
    // result into "everything sounds slowed down".
    const stems = [];
    const trackRefs = [];
    let detectedSrcBpm = null;
    let detectedDownbeatOffset = 0;
    stemTracksToProcess.forEach(({ bus, track }) => {
      const latentId = track.metadata?.latentId;
      if (!latentId || latentId === 'undefined' || latentId === 'null') return;
      stems.push({
        latent_id: latentId,
        stem_type: track.metadata?.stemType || track.metadata?.instrument || 'other',
      });
      trackRefs.push({ busId: bus.id, trackId: track.id });
      // Find the parent (un-stemmed) track for this stem to read the
      // detected BPM applied during ImportAudioModal flow.
      if (!detectedSrcBpm) {
        const parent = bus.tracks.find(
          (t) => t.id === track.metadata?.parentTrackId && t.metadata?.detectedBpm
        );
        if (parent) {
          detectedSrcBpm = parent.metadata.detectedBpm;
          if (typeof parent.metadata.downbeatOffset === 'number') {
            detectedDownbeatOffset = parent.metadata.downbeatOffset;
          }
        }
      }
    });

    // Fallback: if there are no stem tracks at all, repaint any track with a latent
    if (stems.length === 0) {
      state.buses?.forEach((bus) => {
        bus.tracks?.forEach((track) => {
          const latentId = track.metadata?.latentId;
          if (!latentId || latentId === 'undefined' || latentId === 'null') return;
          stems.push({
            latent_id: latentId,
            stem_type: track.metadata?.instrument || track.metadata?.stemType || 'other',
          });
          trackRefs.push({ busId: bus.id, trackId: track.id });
        });
      });
    }

    if (stems.length === 0) {
      repaintLastRef.current = { bpm: tgtBpm, beatsPerBar: tgtBeats, meterDen: tgtDen };
      return;
    }

    // Mute parents now so playback only carries the stems
    parentsToMute.forEach((p) => {
      dispatch({
        type: 'UPDATE_TRACK',
        payload: { busId: p.busId, trackId: p.trackId, updates: { isMuted: true } },
      });
    });

    repaintInFlightRef.current = true;
    setRepaintApplying(true);
    try {
      const effectiveSrcBpm = detectedSrcBpm || srcBpm;

      // ── Try local WebGPU latent path first (instant, no backend) ──
      // Collect cached latent data for each stem via /api/latent (DOAE format)
      const stemLatents = {};
      let hasAllLatents = true;
      let cachedVaeHash = '';
      for (let si = 0; si < stems.length; si++) {
        const lid = stems[si].latent_id;
        const sname = stems[si].stem_type || `stem${si}`;
        console.log(`[meterChange] stem ${si}: ${sname}, latent_id=${lid}`);
        if (!lid || lid === 'undefined' || lid === 'null') { console.warn(`[meterChange] stem ${sname} has no latent_id (${lid}), skipping WebGPU path`); hasAllLatents = false; break; }
        try {
          const resp = await fetch(`/api/latent/${lid}`);
          if (!resp.ok) { hasAllLatents = false; break; }
          const buf = await resp.arrayBuffer();
          const { T, D, flatTD, vaeHash } = parseDoae(buf);
          stemLatents[sname] = { data: flatTD, T };
          if (vaeHash) cachedVaeHash = vaeHash;
          console.log(`[meterChange] fetched ${sname} (${lid}): ${T} frames`);
        } catch (e) {
          console.warn(`[meterChange] failed to fetch latent ${lid}:`, e);
          hasAllLatents = false;
          break;
        }
      }

      if (hasAllLatents && Object.keys(stemLatents).length > 0) {
        console.log(`🎚️ local WebGPU meter change: ${srcBeats}/${srcDen} ${effectiveSrcBpm} → ${tgtBeats}/${tgtDen} ${tgtBpm}`);

        const changed = await meterChangeAllStems(
          stemLatents, effectiveSrcBpm, tgtBpm,
          [srcBeats, srcDen], [tgtBeats, tgtDen]
        );

        // Upload new latents and decode via existing WebGPU decoder
        for (let i = 0; i < stems.length; i++) {
          const stemName = stems[i].stem_type || `stem${i}`;
          const ref = trackRefs[i];
          const { latent: newLatent, T: newT } = changed[stemName] || {};
          if (!newLatent) continue;

          // Upload the new latent to backend for caching (DOAE format)
          const doaeBody = buildDoae(newLatent, newT, 64, 25, cachedVaeHash);
          const uploadResp = await fetch('/api/upload-latent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-doae' },
            body: doaeBody,
          });
          if (!uploadResp.ok) continue;
          const { latent_id: newLatentId } = await uploadResp.json();

          // Decode via existing WebGPU decoder (already in browser)
          const decoded = await decodeLatentFrameRangeBatched(
            newLatent, newT, 0, newT, 64
          );
          const blobUrl = audioDataToBlobUrl(decoded, 48000);

          dispatch({
            type: 'UPDATE_TRACK',
            payload: {
              busId: ref.busId,
              trackId: ref.trackId,
              updates: {
                audioUrl: blobUrl,
                metadata: { latentId: newLatentId },
              },
            },
          });
        }

        console.log(`🎚️ local meter change done (all WebGPU, no backend processing)`);
      } else {
        // ── Fallback: backend repaint (for stems without cached latents) ──
        console.log(`🎚️ falling back to backend repaint-meter`);
        const result = await repaintMeter({
          stems,
          srcMeter: [srcBeats, srcDen],
          tgtMeter: [tgtBeats, tgtDen],
          srcBpm: effectiveSrcBpm,
          tgtBpm,
          coverNoise: 0.55,
          prompt: 'preserve original style and instrument timbre',
          downbeatOffset: detectedDownbeatOffset,
        });
        console.log(`🎚️ auto-repaint ${srcBeats}/${srcDen} ${srcBpm} → ${tgtBeats}/${tgtDen} ${tgtBpm} for ${stems.length} stems`, result);

        (result.results || []).forEach((r, i) => {
          if (r.error || !r.task_id) return;
          const ref = trackRefs[i];
          const taskId = r.task_id;
          (async () => {
            for (let p = 0; p < 120; p++) {
              await new Promise((res) => setTimeout(res, 2500));
              const tr = await fetch(`/api/generate-stemphonic/task/${taskId}`);
              if (!tr.ok) continue;
              const td = await tr.json();
              if (td.state === 'SUCCESS' && td.result?.file_paths?.[0]) {
                dispatch({
                  type: 'UPDATE_TRACK',
                  payload: {
                    busId: ref.busId,
                    trackId: ref.trackId,
                    updates: {
                      audioUrl: td.result.file_paths[0],
                      metadata: { latentId: r.new_latent_id },
                    },
                  },
                });
                return;
              }
              if (td.state === 'FAILURE') return;
            }
          })();
        });
      }

      repaintLastRef.current = { bpm: tgtBpm, beatsPerBar: tgtBeats, meterDen: tgtDen };
    } catch (err) {
      console.error('auto-repaint failed', err);
    } finally {
      repaintInFlightRef.current = false;
      setRepaintApplying(false);
    }
  }, [state.bpm, state.beatsPerBar, state.meterDenominator, state.buses, dispatch]);

  // Debounce: schedule a repaint 1.2s after the last BPM/meter change
  useEffect(() => {
    if (repaintDebounceRef.current) clearTimeout(repaintDebounceRef.current);
    const srcBpm = repaintLastRef.current.bpm;
    const srcBeats = repaintLastRef.current.beatsPerBar;
    const srcDen = repaintLastRef.current.meterDen || 4;
    if (state.bpm === srcBpm && (state.beatsPerBar || 4) === srcBeats && (state.meterDenominator || 4) === srcDen) return;
    repaintDebounceRef.current = setTimeout(runRepaintNow, 1200);
    return () => {
      if (repaintDebounceRef.current) clearTimeout(repaintDebounceRef.current);
    };
  }, [state.bpm, state.beatsPerBar, state.meterDenominator, runRepaintNow]);

  // Drag and drop state for entire DAW area
  const [isDragOver, setIsDragOver] = React.useState(false);

  // Import audio modal — asks the user whether to detect tempo/key on
  // each new audio drop, then runs detect-chords + applies to timeline.
  const [importModal, setImportModal] = React.useState(null); // { file, busId, trackId } | null

  // Listen for the window event Timeline.js fires when an audio file is
  // dropped onto the timeline (separate from the whole-DAW drop handler).
  // Both code paths now route through the same import modal.
  useEffect(() => {
    const onAudioImported = (e) => {
      const detail = e.detail || {};
      if (detail.file) setImportModal({
        file: detail.file,
        busId: detail.busId,
        trackId: detail.trackId,
      });
    };
    window.addEventListener('doseedo-audio-imported', onAudioImported);
    return () => window.removeEventListener('doseedo-audio-imported', onAudioImported);
  }, []);

  // Apply detected tempo + meter + downbeat offset to the timeline.
  // The downbeat offset becomes the timeline pre-roll so bar 1 lands
  // exactly on the song's first downbeat (silence + pickup → bar 0).
  const applyDetection = useCallback(async (file, trackId, busId) => {
    try {
      const det = await detectChordsAndTempo(file);
      if (det.bpm) dispatch({ type: 'UPDATE_BPM', payload: Math.round(det.bpm) });
      if (det.beats_per_bar) dispatch({ type: 'SET_BEATS_PER_BAR', payload: det.beats_per_bar });
      if (det.beat_map) dispatch({ type: 'SET_BEAT_MAP', payload: det.beat_map });
      // Detection-driven changes are NOT user meter changes — they
      // establish the baseline. Bump the repaint snapshot so the debounced
      // auto-repaint sees no diff and doesn't fire a meter conversion.
      if (repaintDebounceRef.current) clearTimeout(repaintDebounceRef.current);
      repaintLastRef.current = {
        bpm: det.bpm ? Math.round(det.bpm) : repaintLastRef.current.bpm,
        beatsPerBar: det.beats_per_bar || repaintLastRef.current.beatsPerBar,
        meterDen: repaintLastRef.current.meterDen || 4,
      };
      if (typeof det.downbeat_offset === 'number') {
        dispatch({ type: 'SET_TIMELINE_OFFSET', payload: det.downbeat_offset });
      }
      // Populate chord row
      if (det.chords) {
        const chordsNum = {};
        Object.entries(det.chords).forEach(([k, v]) => { chordsNum[parseInt(k, 10)] = v; });
        dispatch({ type: 'SET_CHORDS', payload: chordsNum });
      }
      // Stash the source BPM on the track metadata so the repaint flow
      // uses the actual song BPM (not the timeline default) when
      // computing latent bar boundaries.
      dispatch({
        type: 'UPDATE_TRACK',
        payload: {
          busId, trackId,
          updates: {
            metadata: {
              detectedBpm: det.bpm,
              detectedMeter: det.beats_per_bar,
              downbeatOffset: det.downbeat_offset,
              detected: true,
            },
          },
        },
      });
      console.log(`🎵 detected: ${Math.round(det.bpm)} BPM, ${det.beats_per_bar}/4, downbeat at ${det.downbeat_offset?.toFixed(2)}s`);
    } catch (err) {
      console.warn('detection failed:', err.message);
    }
  }, [dispatch]);

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

        // Create a new SFX bus for this track.
        // Start COLLAPSED: the bus row renders the uploaded track as
        // its master waveform immediately, and stays locked in that
        // state until stem separation completes (see BusRow.js guard
        // on handleBusClick/handleExpandToggle/etc — it checks for an
        // uploaded master with no stem siblings). Once stems land the
        // user can expand to see the composite-above-stems layout.
        const busId = `sfx-${Date.now()}`;
        dispatch({
          type: 'CREATE_BUS',
          payload: {
            id: busId,
            type: 'SFX',
            name: `SFX ${state.buses.filter(b => b.type === 'SFX').length + 1}`,
            expanded: false
          }
        });

        // Create the track (analysis pending — runs in background below)
        const trackId = `track-${Date.now()}`;
        const track = {
          id: trackId,
          name: file.name,
          audioUrl: audioUrl,
          duration: duration,
          startPosition: 0,
          gain: 1.0,
          isMuted: false,
          isSolo: false,
          cropStart: 0,
          cropEnd: 0,
          fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
          metadata: {
            type: 'uploaded',
            instrument: null,         // filled in by classifier
            instrumentLabel: null,
            instrumentScore: null,
            icon: 'fa-spinner fa-spin',
            midi: null,               // filled in by basic-pitch
            inputFiles: {},
            analysisStatus: 'pending',
          },
        };

        dispatch({ type: 'ADD_TRACK', payload: { busId, track } });
        console.log(`✅ Audio file added to new SFX bus, analyzing…`);

        // Import modal disabled — skip tempo/key detection prompt

        // Background analysis: basic-pitch + PANNs + VAE latent in parallel
        analyzeAudio(file).then((res) => {
          const cls = res.classification;
          const midi = res.midi;
          const latent = res.latent;
          const instType = cls?.type || 'other';
          dispatch({
            type: 'UPDATE_TRACK',
            payload: {
              busId,
              trackId,
              updates: {
                metadata: {
                  ...track.metadata,
                  instrument: instType,
                  instrumentLabel: cls?.label || null,
                  instrumentScore: cls?.score || null,
                  icon: iconForType(instType),
                  midi: midi?.midi_url || null,
                  latent: latent?.latent_url || null,
                  latentId: latent?.latent_id || null,
                  inputFiles: midi?.midi_url ? { midiPath: midi.midi_url } : {},
                  analysisStatus: 'done',
                  analysisErrors: {
                    midi: res.midiError,
                    classify: res.classifyError,
                    latent: res.latentError,
                  },
                },
              },
            },
          });
          console.log(`🎯 Analysis done for ${file.name}: ${instType} (${cls?.label}), midi=${midi?.n_notes}n, latent=${latent?.n_frames}f`);
        }).catch((err) => {
          console.error('❌ Track analysis failed:', err);
          dispatch({
            type: 'UPDATE_TRACK',
            payload: { busId, trackId, updates: { metadata: { ...track.metadata, analysisStatus: 'failed', analysisError: err.message } } },
          });
        });

        // Background stem separation: rmsDemucs paints instant envelopes,
        // then server-side htdemucs_6s provides real WAV audio.
        (async () => {
          const RMS_STEM_NAMES = ['drums', 'bass', 'vocals', 'other'];

          // -- INSTANT 4-STEM VISUAL via rmsDemucs (577 KB) ---------------
          // Paint stem envelopes immediately from the lightweight RMS model.
          // No backend wait -- this runs in < 500ms on any device.
          let rmsPainted = false;
          try {
            const { flat, numFrames } = await audioFileToStereo48k(file);
            const rmsResult = await analyzeRms(flat, numFrames);
            // Convert amplitude[T] -> [2*T] (min, max) pairs for useWaveform
            const instantStems = RMS_STEM_NAMES.map((stemName, idx) => {
              const amp = rmsResult.stemEnvelopes[idx];
              const T = amp.length;
              const env = new Float32Array(2 * T);
              for (let t = 0; t < T; t++) { env[t] = -amp[t]; env[T + t] = amp[t]; }
              return {
                id: `stem-${trackId}-${stemName}`,
                name: `${file.name.replace(/\.[^.]+$/, '')} -- ${stemName}`,
                audioUrl: null,
                duration,
                startPosition: 0,
                gain: 1.0, isMuted: false, isSolo: false, cropStart: 0, cropEnd: 0,
                fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
                metadata: {
                  type: 'stem', stemType: stemName, parentTrackId: trackId,
                  instrument: stemName, icon: iconForType(stemName),
                  envelopeData: env, envelopeFps: rmsResult.stemEnvelopesFps,
                  playbackReady: false,
                },
              };
            });
            dispatch({ type: 'ADD_TRACKS_BULK', payload: { busId, tracks: instantStems } });
            dispatch({ type: 'SET_BUS_EXPANDED', payload: { busId, expanded: true } });
            rmsPainted = true;
            console.log(`[rmsDemucs] painted ${RMS_STEM_NAMES.length} instant stem envelopes`);
          } catch (rmsErr) {
            console.warn('[rmsDemucs] instant envelopes failed (non-fatal):', rmsErr?.message || rmsErr);
          }

          // -- SERVER-SIDE SEPARATION via /separate-stems ------------------
          // htdemucs_6s on Modal returns WAV download URLs; use these
          // directly for playback (no ONNX decoder needed in browser).
          try {
            const sep = await separateStemsAuto(file);
            const wavUrls = sep?.stems || {};
            const stemNames = Object.keys(wavUrls);
            if (stemNames.length === 0) {
              console.warn('[separate-stems] no WAV URLs in response');
              return;
            }
            if (rmsPainted) {
              // Update existing rms stem tracks with real audio
              for (const stemName of stemNames) {
                const wavUrl = wavUrls[stemName];
                if (!wavUrl) continue;
                dispatch({
                  type: 'UPDATE_TRACK',
                  payload: {
                    busId,
                    trackId: `stem-${trackId}-${stemName}`,
                    updates: { audioUrl: wavUrl, metadata: { playbackReady: true } },
                  },
                });
              }
              // Add any server stems not covered by rmsDemucs (guitar, piano)
              const rmsSet = new Set(RMS_STEM_NAMES.map(n => `stem-${trackId}-${n}`));
              for (const stemName of stemNames) {
                if (rmsSet.has(`stem-${trackId}-${stemName}`)) continue;
                dispatch({
                  type: 'ADD_TRACK',
                  payload: {
                    busId,
                    track: {
                      id: `stem-${trackId}-${stemName}`,
                      name: `${file.name.replace(/\.[^.]+$/, '')} -- ${stemName}`,
                      audioUrl: wavUrls[stemName],
                      duration,
                      startPosition: 0,
                      gain: 1.0, isMuted: false, isSolo: false, cropStart: 0, cropEnd: 0,
                      fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
                      metadata: {
                        type: 'stem', stemType: stemName, parentTrackId: trackId,
                        instrument: stemName, icon: iconForType(stemName),
                        playbackReady: true,
                      },
                    },
                  },
                });
              }
            } else {
              // rmsDemucs failed -- add all stems fresh with WAV audio
              const tracks = stemNames.map((stemName) => ({
                id: `stem-${trackId}-${stemName}`,
                name: `${file.name.replace(/\.[^.]+$/, '')} -- ${stemName}`,
                audioUrl: wavUrls[stemName] || null,
                duration,
                startPosition: 0,
                gain: 1.0, isMuted: false, isSolo: false, cropStart: 0, cropEnd: 0,
                fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
                metadata: {
                  type: 'stem', stemType: stemName, parentTrackId: trackId,
                  instrument: stemName, icon: iconForType(stemName),
                  playbackReady: !!wavUrls[stemName],
                },
              }));
              dispatch({ type: 'ADD_TRACKS_BULK', payload: { busId, tracks } });
              dispatch({ type: 'SET_BUS_EXPANDED', payload: { busId, expanded: true } });
            }
            console.log(`Separation done: ${stemNames.join(', ')} (${file.name})`);
          } catch (sepErr) {
            console.warn('auto-separation failed:', sepErr?.message || sepErr);
          }
        })();
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
    if (target.closest('[data-track-id]') ||
        target.closest('[data-bus-id]') ||
        target.closest('button') ||
        target.closest('input') ||
        target.closest('canvas') ||
        target.tagName === 'BUTTON' ||
        target.tagName === 'INPUT') {
      return;
    }

    const container = scrollableContentRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const x = e.clientX - containerRect.left + container.scrollLeft;
    const y = e.clientY - containerRect.top + container.scrollTop;

    marqueeRef.current = { isSelecting: true, startX: x, startY: y };
    setMarqueeStart({ x, y });
    setMarqueeEnd({ x, y });
    setIsMarqueeSelecting(true);

    // Prevent text selection
    e.preventDefault();
  }, []);

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

    // Find all tracks that intersect with the selection rectangle
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
  }, [marqueeStart, marqueeEnd, getTrackElements, rectsIntersect, dispatch]);

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
      {/* Import audio detection modal */}
      {importModal && (
        <ImportAudioModal
          filename={importModal.file?.name || 'audio'}
          onYes={() => {
            const m = importModal;
            setImportModal(null);
            applyDetection(m.file, m.trackId, m.busId);
          }}
          onNo={() => setImportModal(null)}
        />
      )}

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
          {/* Shared 2×2 spacer grid. All four cells live in ONE nested grid
              so their column widths are pixel-locked (col 1 = max of BPM
              cluster & Add Track; col 2 = max of Transport & Zoom). The
              right cluster sits immediately after the left cluster with a
              fixed 8px gap instead of an open 50/50 void. */}
          <div className={styles.spacerArea}>
            {/* Top-left: Metronome / BPM / Meter */}
            <div className={styles.spacerCellTopLeft}>
              {showMetronome && (
                <Button
                  id="metronome-btn"
                  icon="fa-solid fa-drum"
                  onClick={toggleMetronome}
                  isActive={state.isMetronomeOn}
                  title="Toggle Metronome"
                  style={{ padding: '4px 8px' }}
                />
              )}
              {showTempoLabels && <span style={{ color: '#c5cae9', fontSize: '11px', fontWeight: 600 }}>BPM</span>}
              <input
                type="number" min="40" max="240"
                value={state.bpm}
                onChange={(e) => dispatch({ type: 'UPDATE_BPM', payload: parseInt(e.target.value, 10) })}
                style={{ width: '46px', padding: '3px 5px', borderRadius: '4px', border: '1px solid rgba(102,126,234,0.3)', background: 'rgba(20,20,20,0.8)', color: 'white', fontSize: '12px', textAlign: 'center' }}
              />
              {showTempoLabels && <span style={{ color: '#c5cae9', fontSize: '11px', fontWeight: 600 }}>Meter</span>}
              <select
                value={`${state.beatsPerBar || 4}/${state.meterDenominator || 4}`}
                onChange={(e) => dispatch({ type: 'SET_METER', payload: e.target.value })}
                style={{ width: '58px', padding: '3px 5px', borderRadius: '4px', border: '1px solid rgba(102,126,234,0.3)', background: 'rgba(20,20,20,0.8)', color: 'white', fontSize: '12px' }}
              >
                <option value="3/4">3/4</option>
                <option value="4/4">4/4</option>
                <option value="5/4">5/4</option>
                <option value="6/8">6/8</option>
                <option value="7/8">7/8</option>
              </select>
              {repaintApplying && (
                <i className="fa-solid fa-wand-magic-sparkles fa-spin" style={{ color: '#8B7FF0', fontSize: 13 }} title="Repainting tracks" />
              )}
            </div>

            {/* Top-right: Automation toggle + Transport */}
            <div className={styles.spacerCellTopRight}>
              {showAutomation && (
                <Button
                  id="autobtn"
                  icon="fa-solid fa-chart-simple"
                  onClick={toggleAutomation}
                  isActive={state.automationWindow.isVisible}
                  title="Toggle automation"
                />
              )}
              <div style={{ display: 'flex', alignItems: 'center', gap: '0px', background: 'rgba(30,30,30,0.6)', padding: '4px', borderRadius: '6px', border: '1px solid rgba(102,126,234,0.2)' }}>
                <TransportControls
                  isPlaying={state.isPlaying}
                  playheadPosition={state.playheadPosition}
                  onPlayPause={handlePlayPause}
                  onStop={handleStop}
                  showTime={showTimeDisplay}
                />
              </div>
            </div>

            {/* Bottom-left: Add Track */}
            <div className={styles.spacerCellBottomLeft}>
              <button
                onClick={handleAddBus}
                className={styles.addTrackButton}
                title="Add new bus"
              >
                <span style={{ fontSize: '16px', marginRight: '4px' }}>+</span>
                <span style={{ fontSize: '11px', fontWeight: 500 }}>Add Track</span>
              </button>
            </div>

            {/* Bottom-right: Zoom controls */}
            <div className={styles.spacerCellBottomRight}>
              <button
                className={styles.zoomModeButton}
                onClick={() => dispatch({ type: 'SET_ZOOM_MODE', payload: state.zoomMode === 'x' ? 'y' : 'x' })}
                title={state.zoomMode === 'x' ? 'Switch to Vertical Zoom' : 'Switch to Horizontal Zoom'}
              >
                <i className={`fa-solid ${state.zoomMode === 'y' ? 'fa-up-down' : 'fa-left-right'}`}></i>
              </button>
              <button
                className={styles.zoomButton}
                onClick={() => { if (state.zoomMode === 'x') handleZoomOut(); else handleZoomYOut(); }}
                title={state.zoomMode === 'x' ? 'Zoom Out' : 'Decrease Track Height'}
              >
                <i className="fa-solid fa-minus"></i>
              </button>
              <button
                className={styles.zoomButton}
                onClick={() => { if (state.zoomMode === 'x') handleZoomIn(); else handleZoomYIn(); }}
                title={state.zoomMode === 'x' ? 'Zoom In' : 'Increase Track Height'}
              >
                <i className="fa-solid fa-plus"></i>
              </button>
            </div>
          </div>

          {/* Col 2 Row 1: ChordTrack / SceneMarkers / AutomationWindow above Timeline */}
          <div style={{
            gridColumn: 2,
            gridRow: 1,
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
