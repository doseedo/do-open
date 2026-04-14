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

    // ONNX model preload — decoder first (63 MB, needed for playback),
    // then demucs (325 MB, needed for separation). Decoder is now small
    // enough to load in parallel but sequential avoids bandwidth
    // contention on slower connections.
    (async () => {
      try {
        // semDemucsV4 (9 MB) first — it runs on EVERY upload before demucs
        // to classify mix-vs-solo + populate instant stem envelopes.
        // Tiny and fast, loads in parallel with the decoder below.
        import('../../services/semDemucsV4').then(({ initSemDemucsV4 }) =>
          initSemDemucsV4().catch(() => {})
        ).catch(() => {});

        console.log('[prewarm] loading Oobleck VAE decoder (63 MB)…');
        await initLatentDecoder();
        console.log('[prewarm] decoder ready, now loading latentDemucsV4 (98 MB fp16)…');
        await initLatentDemucsV4();
        // After demucs is ready, preload the Oobleck VAE encoder (~337 MB).
        console.log('[prewarm] demucs ready, now loading latentEncoder (337 MB)…');
        const { initLatentEncoder } = await import('../../services/latentEncoder');
        initLatentEncoder().catch((e) => {
          console.warn('[prewarm] latentEncoder preload failed (non-fatal):', e?.message || e);
        });
      } catch (e) {
        // If decoder or demucs fails, still try to start them + encoder.
        initLatentDemucsV4().catch(() => {});
        import('../../services/latentEncoder').then(({ initLatentEncoder }) =>
          initLatentEncoder().catch(() => {})
        ).catch(() => {});
        import('../../services/semDemucsV4').then(({ initSemDemucsV4 }) =>
          initSemDemucsV4().catch(() => {})
        ).catch(() => {});
      }
    })();

    console.log('[prewarm] studio opened — warming Modal backend + preloading ONNX models');
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

        // Pop the import modal — user can detect tempo/key/meter
        setImportModal({ file, busId, trackId });

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

        // Background stem separation: turn this single track into a
        // collapsed bus stack. Demucs can take ~30s; we kick it off in
        // parallel and add all 6 stems in ONE bulk dispatch so the
        // track tree re-renders once instead of 6× (which previously
        // caused all existing tracks to re-fetch their audio on every
        // ADD_TRACK).
        // === STRICT-LATENT SEPARATION ===============================
        // Primary path: run distill_demucs (WebGPU ONNX) in the browser.
        //   audio → stereo 48k Float32 → 4 stem latents → upload each
        //   latent to /api/upload-latent (backend never sees audio).
        // Fallback path: backend /separate-stems (POST audio, get latent_ids).
        //   Only hit if WebGPU ONNX runtime can't load the distill model.
        // =============================================================
        (async () => {
          const ACtx = window.AudioContext || window.webkitAudioContext;
          const audioCtx = new ACtx({ sampleRate: 48000 });

          // Start downloading the VAE decoder in the background so it's
          // ready by the time we need to decode stems for playback.
          const _loadT0 = performance.now();
          let decoderLastPct = -20;
          initLatentDecoder((p) => {
            if (!p?.bytesTotal) return;
            const pct = Math.floor((p.bytesLoaded / p.bytesTotal) * 100);
            if (pct >= decoderLastPct + 20) {
              decoderLastPct = pct;
              const elap = ((performance.now() - _loadT0) / 1000).toFixed(1);
              const mb = (p.bytesLoaded / 1e6).toFixed(0);
              console.log(`[latentDecoder] loading ${pct}% (${mb} MB, ${elap}s)`);
            }
          }).catch((err) => {
            const emsg = err?.message || err?.toString?.() || JSON.stringify(err) || 'unknown';
            console.warn('[latentDecoder] init failed (non-fatal):', emsg);
          });

          // ── PRE-DEMUCS: v4-small-6 classifier (9 MB, WebGPU) ─────────
          // Runs on every upload BEFORE latent demucs. Produces:
          //   - 6 stem RMS envelopes → instant stem waveforms (no wait
          //     for the 325 MB demucs)
          //   - mix-vs-solo classification via per-stem mask energy
          // If the file is a solo stem (e.g., drum loop, isolated vocal),
          // we skip demucs entirely and run the oobleck encoder in the
          // background to cache the single latent instead.
          let v4Result = null;
          try {
            const { analyze: semAnalyze } = await import('../../services/semDemucsV4');
            const { flat: preFlat, numFrames: preN } = await audioFileToStereo48k(file);
            v4Result = await semAnalyze(preFlat, preN);
            const c = v4Result.classification;
            console.log(`[semDemucsV4] ${c.rationale} (confidence=${c.confidence.toFixed(2)})`);
            // perStemEnergy is a Float32Array; its .map returns another
            // typed array (numbers, not entries), so Object.fromEntries
            // throws "Iterator value NaN is not an entry object".
            // Array.from() gives a plain Array whose .map can return
            // arbitrary [k, v] pairs.
            console.log('[semDemucsV4] per-stem energy:',
              Object.fromEntries(Array.from(v4Result.perStemEnergy).map((e, i) => [
                ['drums','bass','vocals','other','guitar','piano'][i],
                (e * 100).toFixed(1) + '%',
              ])));

            if (!c.isMix) {
              console.log(`[flow] detected solo stem (${c.soloStemName}) — skipping demucs, caching encoder latent in background`);
              // Attach solo-stem metadata to the uploaded track so later
              // UI can key off it (icon, name, etc.).
              dispatch({
                type: 'UPDATE_TRACK',
                payload: {
                  busId, trackId,
                  updates: {
                    metadata: {
                      ...track.metadata,
                      soloStemClassification: {
                        stemName: c.soloStemName,
                        stemIndex: c.soloStemIndex,
                        confidence: c.confidence,
                        perStemEnergy: Array.from(v4Result.perStemEnergy),
                      },
                    },
                  },
                },
              });
              // Background oobleck-encoder latent caching (fire-and-forget).
              (async () => {
                try {
                  const { initLatentEncoder, encodeToLatent } = await import('../../services/latentEncoder');
                  await initLatentEncoder();
                  const latent = await encodeToLatent(preFlat, preN);
                  const T = latent.length / 64;
                  const { uploadLatent } = await import('../../services/latentDemucs');
                  const meta = await uploadLatent(latent, T, 64, 25);
                  console.log(`[flow] solo stem encoder cached latent ${meta.latent_id} (${T} frames)`);
                  dispatch({
                    type: 'UPDATE_TRACK',
                    payload: { busId, trackId,
                      updates: { metadata: {
                        ...track.metadata,
                        soloStemLatentId: meta.latent_id,
                        soloStemNFrames: T,
                      } } },
                  });
                } catch (err) {
                  console.warn('[flow] solo-stem encoder caching failed (non-fatal):', err?.message || err);
                }
              })();
              return; // bail out of the demucs pipeline — keep the single uploaded track
            }
          } catch (v4Err) {
            console.warn('[semDemucsV4] failed (non-fatal, continuing with demucs):', v4Err?.message || v4Err);
          }

          // ── INSTANT 6-STEM VISUALIZER (mix branch) ───────────────────
          // If v4 classified as a mix, paint 6 stem placeholder tracks
          // right now using its per-stem RMS envelopes. demucs will
          // later UPDATE_TRACK 4 of these with real latent_ids +
          // refined envelopes (guitar/piano stay v4-only since the
          // current demucs is 4-stem). This makes the DAW show 6 stem
          // waveforms within ~200ms of upload instead of waiting ~30s
          // for the 325 MB demucs model.
          let v4PaintedStems = false;
          if (v4Result && v4Result.classification.isMix && v4Result.stemEnvelopes?.length === 6) {
            const v4StemNames = ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'];
            v4PaintedStems = true;
            const instantStems = v4StemNames.map((stemName, idx) => ({
              id: `stem-${trackId}-${stemName}`,
              name: `${file.name.replace(/\.[^.]+$/, '')} — ${stemName}`,
              audioUrl: null,  // envelope-only render until demucs/decoder populates audio
              duration,
              startPosition: 0,
              gain: 1.0, isMuted: false, isSolo: false, cropStart: 0, cropEnd: 0,
              fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
              metadata: {
                type: 'stem',
                stemType: stemName,
                parentTrackId: trackId,
                instrument: stemName,
                icon: iconForType(stemName),
                envelopeData: v4Result.stemEnvelopes[idx],
                envelopeFps: v4Result.rmsFps,
                fromV4Small: true,   // will be set false once latent-demucs completes
                playbackReady: false,
              },
            }));
            dispatch({ type: 'ADD_TRACKS_BULK', payload: { busId, tracks: instantStems } });
            console.log(`[semDemucsV4] painted 6 instant stem waveforms (fps=${v4Result.rmsFps.toFixed(1)})`);
          }

          // Try the browser-local latent_demucs path first.
          let stemLatentIds = null;
          let localFlatTDs = null;  // keep the raw stem latents around for latent_visual
          try {
            console.log('[latentDemucsV4] trying browser-local 6-stem separation (conditioned on v4-small)...');
            let demucsLastPct = -20;
            const _demucsT0 = performance.now();
            await initLatentDemucsV4((p) => {
              if (!p?.bytesTotal) return;
              const pct = Math.floor((p.bytesLoaded / p.bytesTotal) * 100);
              if (pct >= demucsLastPct + 20) {
                demucsLastPct = pct;
                const elap = ((performance.now() - _demucsT0) / 1000).toFixed(1);
                const mb = (p.bytesLoaded / 1e6).toFixed(0);
                console.log(`[latentDemucsV4] loading ${pct}% (${mb} MB, ${elap}s)`);
              }
            });
            const { flat, numFrames } = await audioFileToStereo48k(file);
            console.log(`[latentDemucsV4] decoded audio in browser: ${numFrames} samples @ 48kHz`);
            // v4Result was computed earlier (the same run that gated
            // mix-vs-solo). We reuse its masks + embedding as
            // conditioning here — no re-inference needed.
            if (!v4Result) throw new Error('v4Result unavailable — semDemucsV4 must run before latentDemucsV4');
            const stems = await separate6Stems(flat, numFrames, v4Result);
            console.log(`[latentDemucsV4] separated into ${Object.keys(stems).length} stem latents`);
            // Stash flatTD so we can run latent_visual without re-fetching.
            localFlatTDs = {};
            for (const name of Object.keys(stems)) {
              localFlatTDs[name] = { flatTD: stems[name].flatTD, T: stems[name].T };
            }
            stemLatentIds = {};
            for (const name of STEM_NAMES_V4COND_6) {
              const s = stems[name];
              if (!s) continue;
              const meta = await uploadLatent(s.flatTD, s.T, s.D, 25);
              stemLatentIds[name] = {
                latent_id: meta.latent_id,
                n_frames: s.T,
              };
              console.log(`[latentDemucsV4] uploaded ${name}: ${meta.latent_id} (${s.T} frames)`);
            }

            // ─── Refiner pass: clean latent + v4-small noisy mask →
            //     refined per-stem STFT mask. +25.2% SI-SDR per the
            //     training run. Stash in the module-level map so the
            //     (follow-up) maskPlayback integration can pull them
            //     in. We DON'T UPDATE_TRACK with them here because the
            //     stem tracks already carry envelopeData in metadata,
            //     and the reducer replaces metadata wholesale.
            if (v4Result && stems) {
              try {
                const { refine6StemMasks, initMaskRefiner } = await import('../../services/maskRefiner');
                await initMaskRefiner();
                const _rfT0 = performance.now();
                const refined = await refine6StemMasks(stems, v4Result);
                const rfMs = (performance.now() - _rfT0).toFixed(0);
                console.log(`[maskRefiner] refined ${Object.keys(refined).length} masks in ${rfMs}ms (F=${v4Result.maskF} T=${v4Result.maskT})`);
                // Stash where the mask-playback block below can pick
                // them up (same-scope async ordering is fragile across
                // try/catch boundaries; a local variable is cleaner).
                var _v4RefinedMasks = refined;
                var _v4RefinedF = v4Result.maskF;
                var _v4RefinedT = v4Result.maskT;
              } catch (rfErr) {
                console.warn('[maskRefiner] skipped (non-fatal):', rfErr?.message || rfErr);
              }
            }
          } catch (localErr) {
            const emsg = localErr?.message || localErr?.toString?.() || JSON.stringify(localErr) || 'unknown';
            console.warn('[latentDemucsV4] browser path failed, falling back to backend /separate-stems:', emsg);
            stemLatentIds = null;
            localFlatTDs = null;
          }

          // Backend fallback: POST audio, backend runs latent_demucs server-side
          if (!stemLatentIds) {
            try {
              const sep = await separateStemsAuto(file);
              const backendStems = sep?.stem_latents || {};
              if (Object.keys(backendStems).length === 0) {
                console.warn('backend fallback returned no stem latents');
                return;
              }
              stemLatentIds = backendStems;
              console.log(`[separate-stems backend fallback] got ${Object.keys(backendStems).length} stems`);
            } catch (err) {
              const emsg = err?.message || err?.toString?.() || JSON.stringify(err) || 'unknown';
              console.warn('auto-separation failed (both paths, non-fatal):', emsg);
              return;
            }
          }

          // =============================================================
          // STEP 1: INSTANT visual previews via latent_visual (tiny model).
          // We create stem tracks IMMEDIATELY with:
          //   - placeholder blob URL (44-byte header; swapped on decode)
          //   - fake AudioBuffer built from the envelope, pre-populated in
          //     useWaveform's in-memory cache so the waveform strip
          //     renders correctly
          // Stems appear in the timeline in ~milliseconds after separation.
          // =============================================================
          const stemNames = Object.keys(stemLatentIds);

          // If we took the backend fallback, we need to fetch each DOAE to
          // get the raw latent bytes for latent_visual. Parallel fetches.
          let flatTDs = localFlatTDs;
          if (!flatTDs) {
            try {
              const entries = await Promise.all(stemNames.map(async (name) => {
                const id = stemLatentIds[name].latent_id;
                const resp = await fetch(`/api/latent/${id}`);
                if (!resp.ok) throw new Error(`fetch /api/latent/${id} HTTP ${resp.status}`);
                const buf = await resp.arrayBuffer();
                const { flatTD, T } = parseDoae(buf);
                return [name, { flatTD, T }];
              }));
              flatTDs = Object.fromEntries(entries);
            } catch (fetchErr) {
              console.warn('[latentVisual] failed to fetch DOAEs for preview:', fetchErr.message);
              flatTDs = null;
            }
          }

          // Load the tiny latent_visual envelope model on the MAIN thread
          // (244 KB WASM, <100ms load, <1ms per call). This is separate
          // from the heavyweight decoder worker because it needs to be
          // available IMMEDIATELY after separation — before the 320MB
          // decoder model finishes loading in the worker (~49s).
          try { await initLatentVisual(); } catch (e) { /* non-fatal */ }

          const placeholderTracks = [];
          for (const stemName of stemNames) {
            const meta = stemLatentIds[stemName];
            const latentId = meta?.latent_id;
            if (!latentId) continue;

            let fakeBuf = null;
            let env = null;
            if (flatTDs?.[stemName]) {
              try {
                const { flatTD, T } = flatTDs[stemName];
                env = await envelopeFromLatent(flatTD, T);
                fakeBuf = buildFakeBufferFromEnvelope(env, T, audioCtx);
                console.log(`[latentVisual] ${stemName}: T=${T}, buf.dur=${fakeBuf.duration.toFixed(2)}s`);
              } catch (e) {
                console.warn(`[latentVisual] preview gen failed for ${stemName}:`, e?.message || e);
              }
            }

            // Placeholder blob URL that useWaveform's cache is keyed on.
            // We pre-populate the cache with the fake envelope buffer so the
            // waveform strip renders the right shape immediately.
            // Valid minimal WAV: 44-byte header + 480 stereo float32 samples
            // = 0.01s silence at 48kHz. The old 44-byte stub had NO samples
            // and caused decodeAudioData → EncodingError which killed the
            // decode loop. This blob decodes to a valid 0.01s AudioBuffer.
            const _phSz = 44 + 480 * 2 * 4;
            const _ph = new ArrayBuffer(_phSz);
            const _pv = new DataView(_ph);
            const _pw = (o, s) => { for (let i = 0; i < s.length; i++) _pv.setUint8(o + i, s.charCodeAt(i)); };
            _pw(0,'RIFF'); _pv.setUint32(4,_phSz-8,true); _pw(8,'WAVE'); _pw(12,'fmt ');
            _pv.setUint32(16,16,true); _pv.setUint16(20,3,true); _pv.setUint16(22,2,true);
            _pv.setUint32(24,48000,true); _pv.setUint32(28,48000*8,true);
            _pv.setUint16(32,8,true); _pv.setUint16(34,32,true); _pw(36,'data');
            _pv.setUint32(40,480*2*4,true);
            const placeholderBlob = new Blob([new Uint8Array(_ph)], { type: 'audio/wav' });
            const placeholderUrl = URL.createObjectURL(placeholderBlob);
            if (fakeBuf) audioBufferCache.set(placeholderUrl, fakeBuf);

            placeholderTracks.push({
              id: `stem-${trackId}-${stemName}`,
              name: `${file.name.replace(/\.[^.]+$/, '')} — ${stemName}`,
              audioUrl: placeholderUrl,
              duration,
              startPosition: 0,
              gain: 1.0, isMuted: false, isSolo: false, cropStart: 0, cropEnd: 0,
              fx: { reverb: 0, fadeIn: 0.2, fadeOut: 1.0 },
              metadata: {
                type: 'stem',
                stemType: stemName,
                parentTrackId: trackId,
                instrument: stemName,
                icon: iconForType(stemName),
                latentId,
                nFrames: meta.n_frames,
                playbackReady: false,
                // Pre-computed envelope from latent_visual for instant
                // waveform rendering — useWaveform reads this and draws
                // bars directly without needing decoded audio.
                envelopeData: env || null,
              },
            });
          }

          if (placeholderTracks.length === 0) {
            console.warn('auto-separation: no previews built');
            return;
          }
          if (v4PaintedStems) {
            // v4 already added these 6 stems (4 of which match demucs).
            // UPDATE_TRACK each one with the real latent_id, refined
            // envelope from latent_visual, and placeholder audio URL
            // that points at the cached fakeBuf. Non-matching v4 stems
            // (guitar, piano) remain v4-only previews.
            for (const t of placeholderTracks) {
              dispatch({
                type: 'UPDATE_TRACK',
                payload: {
                  busId,
                  trackId: t.id,
                  updates: {
                    audioUrl: t.audioUrl,
                    duration: t.duration,
                    metadata: { ...t.metadata, fromV4Small: false },
                  },
                },
              });
            }
            console.log(`🎚️ latent-demucs refined ${placeholderTracks.length} v4 stems (${file.name})`);
          } else {
            dispatch({
              type: 'ADD_TRACKS_BULK',
              payload: { busId, tracks: placeholderTracks },
            });
            console.log(`🎚️ Auto-separated ${file.name} → ${placeholderTracks.length} stems (instant previews, decoding audio…)`);
          }
          dispatch({ type: 'SET_BUS_EXPANDED', payload: { busId, expanded: false } });

          // ── STEP 1b: Set up mask-based playback (instant, no decode) ──
          // Compute spectral masks from stem latents and set up the
          // AudioWorklet so stems can play immediately at master quality.
          if (localFlatTDs && Object.keys(localFlatTDs).length >= 4) {
            try {
              // Use the SAME AudioContext as useAudioPlayback (shared via global)
              const maskCtx = window.__doseedo_audioCtx;
              if (!maskCtx) throw new Error('AudioContext not ready');
              if (maskCtx.state === 'suspended') await maskCtx.resume();

              const ok = await initMaskPlayback(maskCtx);
              if (ok) {
                // Create the worklet node FIRST so setMaster/computeAndSetMasks
                // can actually post messages to it (previously they no-op'd
                // because workletNode was still null).
                createMaskPlaybackNode(maskCtx.destination);

                // Send master audio to the worklet
                const masterResp = await fetch(audioUrl);
                const masterArrayBuf = await masterResp.arrayBuffer();
                const masterAudioBuf = await maskCtx.decodeAudioData(masterArrayBuf);
                setMaster(masterAudioBuf);

                // Compute masks. Prefer refined masks from the
                // latent_mask_refiner (post-v4cond-pred) if they were
                // produced successfully; otherwise fall back to the
                // old latent_mask_e2e path via computeAndSetMasks.
                const T = localFlatTDs[Object.keys(localFlatTDs)[0]].T;
                if (typeof _v4RefinedMasks !== 'undefined' && _v4RefinedMasks) {
                  setRefinedMasks(_v4RefinedMasks, _v4RefinedF, _v4RefinedT, T);
                  console.log('🎭 Mask playback ready — using refined masks (+25.2% SI-SDR over noisy)');
                } else {
                  await computeAndSetMasks(
                    Object.fromEntries(
                      Object.entries(localFlatTDs).map(([name, { flatTD }]) => [name, flatTD])
                    ), T
                  );
                  console.log('🎭 Mask playback ready — worklet connected to main AudioContext');
                }
              }
            } catch (maskErr) {
              console.warn('🎭 Mask playback setup failed, falling back to decode:', maskErr?.message || maskErr);
            }
          }

          // =============================================================
          // STEP 2: round-robin streaming VAE decode.
          // DISABLED: mask playback handles audio from master directly.
          // Decode loop only needed for waveform visuals — re-enable
          // after mask playback is verified working.
          // =============================================================
          console.log('🎭 Decode loop DISABLED — mask playback active, no decode needed');

          if (false) { // DECODE LOOP DISABLED FOR MASK PLAYBACK TESTING
          const FIRST_CHUNK = 64;
          const CHUNK = 64;

          const canStream = !!flatTDs;

          (async () => {
            if (!canStream) {
              // Fallback: sequential one-shot decodes (old behavior).
              for (const stemName of stemNames) {
                const meta = stemLatentIds[stemName];
                const latentId = meta?.latent_id;
                if (!latentId) continue;
                try {
                  const realUrl = await latentIdToBlobUrl(latentId, audioCtx);
                  dispatch({
                    type: 'UPDATE_TRACK',
                    payload: {
                      busId,
                      trackId: `stem-${trackId}-${stemName}`,
                      updates: { audioUrl: realUrl },
                    },
                  });
                  console.log(`[latentDecoder] ${stemName} playable (one-shot)`);
                } catch (decErr) {
                  const emsg = decErr?.message || decErr?.toString?.() || JSON.stringify(decErr) || 'unknown';
                  console.warn(`[latentDecoder] stem "${stemName}" decode failed:`, emsg);
                }
              }
              return;
            }

            // Per-stem accumulator: growing Float32Array of decoded audio
            // in channels-first layout [L0..Ln, R0..Rn].
            const stemAccum = {};
            for (const name of stemNames) {
              stemAccum[name] = {
                data: new Float32Array(0),
                samplesPerCh: 0,
                framesDone: 0,
                totalFrames: flatTDs[name].T,
              };
            }

            // Append a newly-decoded chunk to a stem's accumulator.
            // Only builds a WAV blob + dispatches UPDATE_TRACK when
            // `shouldDispatch` is true (first chunk + final chunk) to
            // avoid the massive JS overhead of WAV encoding + React
            // re-renders on every intermediate round.
            function appendChunk(stemName, chunk, newSamplesPerCh, endFrame) {
              const acc = stemAccum[stemName];
              const merged = concatChannelsFirstStereo(
                acc.data, acc.samplesPerCh,
                chunk,    newSamplesPerCh,
              );
              acc.data = merged;
              acc.samplesPerCh += newSamplesPerCh;
              acc.framesDone = endFrame;
            }

            // Look up the envelope data that was stored on the original
            // placeholder track — we need to preserve it across updates
            // so useWaveform keeps rendering the envelope until full audio.
            const stemEnvelopes = {};

            function dispatchStemAudio(stemName) {
              const acc = stemAccum[stemName];
              const { blobUrl, audioBuffer } = audioDataToBlobUrl(
                acc.data, acc.samplesPerCh, audioCtx,
              );
              audioBufferCache.set(blobUrl, audioBuffer);
              // Find the envelope from the original placeholder track
              if (!stemEnvelopes[stemName]) {
                const buses = state.buses || [];
                for (const bus of buses) {
                  const t = (bus.tracks || []).find((t2) => t2.id === `stem-${trackId}-${stemName}`);
                  if (t?.metadata?.envelopeData) {
                    stemEnvelopes[stemName] = t.metadata.envelopeData;
                    break;
                  }
                }
              }
              dispatch({
                type: 'UPDATE_TRACK',
                payload: {
                  busId,
                  trackId: `stem-${trackId}-${stemName}`,
                  updates: {
                    audioUrl: blobUrl,
                    metadata: {
                      type: 'stem',
                      stemType: stemName,
                      parentTrackId: trackId,
                      instrument: stemName,
                      icon: iconForType(stemName),
                      latentId: stemLatentIds[stemName].latent_id,
                      nFrames: stemLatentIds[stemName].n_frames,
                      playbackReady: true,
                      decodeFramesDone: acc.framesDone,
                      decodeTotalFrames: acc.totalFrames,
                      // Preserve envelope data so useWaveform keeps showing
                      // the envelope until the FULL audio replaces it.
                      envelopeData: stemEnvelopes[stemName] || null,
                    },
                  },
                },
              });
            }

            // Run a BATCHED decode covering [startFrame..endFrame] across
            // all stems in a single ONNX pass. Then dispatch each stem's
            // growing blob. This replaces 4 sequential batch=1 calls with
            // 1 batch=4 call — typically 1.5-3× faster on WebGPU.
            async function decodeRoundBatched(startFrame, chunkFrames) {
              const active = stemNames.filter((n) => stemAccum[n].framesDone === startFrame
                                                    && stemAccum[n].totalFrames > startFrame);
              if (active.length === 0) return;
              const refT = flatTDs[active[0]].T;
              const endFrame = Math.min(startFrame + chunkFrames, refT);
              const newSamplesPerCh = (endFrame - startFrame) * 1920;
              const stemsArg = active.map((n) => ({
                flatTD: flatTDs[n].flatTD,
                totalT: flatTDs[n].T,
              }));
              const outs = await decodeLatentFrameRangeBatched(stemsArg, startFrame, endFrame);
              for (let i = 0; i < active.length; i++) {
                appendChunk(active[i], outs[i], newSamplesPerCh, endFrame);
              }
            }

            const t0 = performance.now();

            // Round A: FIRST_CHUNK across all stems → fast time-to-first-audio.
            // This is the ONLY intermediate dispatch — so user can play immediately.
            try {
              await decodeRoundBatched(0, FIRST_CHUNK);
              // Dispatch first-chunk audio for all stems
              for (const name of stemNames) dispatchStemAudio(name);
              console.log(`[decode] batched first chunk (${FIRST_CHUNK}f × ${stemNames.length}) at ${((performance.now() - t0)/1000).toFixed(2)}s — all stems have ${(FIRST_CHUNK * 1920 / 48000).toFixed(1)}s playable`);
            } catch (e) {
              console.warn('[decode] batched first chunk failed, falling back to serial:', e?.message || e);
              for (const name of stemNames) {
                try {
                  const { flatTD, T } = flatTDs[name];
                  const end = Math.min(FIRST_CHUNK, T);
                  const chunk = await decodeLatentFrameRange(flatTD, T, 0, end);
                  appendChunk(name, chunk, end * 1920, end);
                  dispatchStemAudio(name);
                } catch (e2) { console.warn(`[decode] ${name} serial fallback failed:`, e2?.message || e2); }
              }
            }

            // For long files (>30s), use backend L4 decode instead of WebGPU
            // — the L4 GPU does ~60× realtime vs WebGPU's ~1× realtime.
            const refT = flatTDs[stemNames[0]].T;
            const audioDurS = refT * 1920 / 48000;
            const USE_BACKEND_DECODE = false; // all WebGPU, no backend decode

            if (USE_BACKEND_DECODE) {
              console.log(`[decode] audio is ${audioDurS.toFixed(0)}s (>${30}s) — using backend L4 for full decode`);
              for (const name of stemNames) {
                const meta = stemLatentIds[name];
                if (!meta?.latent_id) continue;
                try {
                  const realUrl = await latentIdToBlobUrl(meta.latent_id, audioCtx);
                  dispatchStemAudio(name);
                  // Also update with real audio URL
                  const acc = stemAccum[name];
                  // We don't have the raw audio data in the accum for backend path,
                  // so build it from the blob URL directly.
                  dispatch({
                    type: 'UPDATE_TRACK',
                    payload: {
                      busId,
                      trackId: `stem-${trackId}-${name}`,
                      updates: { audioUrl: realUrl },
                    },
                  });
                  console.log(`[decode] ${name} decoded via backend (${((performance.now() - t0)/1000).toFixed(1)}s)`);
                } catch (e) {
                  console.warn(`[decode] backend decode ${name} failed:`, e?.message || e);
                }
              }
              console.log(`[decode] all stems fully decoded at ${((performance.now() - t0)/1000).toFixed(2)}s (backend path)`);
            } else {
              // Rounds B..N: CHUNK frames in batched passes until all stems complete.
              let cursor = FIRST_CHUNK;
              let roundNum = 1;
              const totalRounds = Math.ceil((refT - FIRST_CHUNK) / CHUNK) + 1;
              while (true) {
                if (cursor >= refT) break;
                const roundEnd = Math.min(cursor + CHUNK, refT);
                roundNum++;
                try {
                  await decodeRoundBatched(cursor, CHUNK);
                } catch (e) {
                  console.warn(`[decode] batched round at ${cursor} failed:`, e?.message || e);
                  for (const name of stemNames) {
                    const acc = stemAccum[name];
                    if (acc.framesDone !== cursor || acc.framesDone >= acc.totalFrames) continue;
                    try {
                      const end = Math.min(cursor + CHUNK, acc.totalFrames);
                      const chunk = await decodeLatentFrameRange(flatTDs[name].flatTD, acc.totalFrames, cursor, end);
                      appendChunk(name, chunk, (end - cursor) * 1920, end);
                    } catch (e2) { console.warn(`[decode] ${name} serial fallback failed:`, e2?.message || e2); }
                  }
                }
                const decodedSec = (roundEnd * 1920 / 48000).toFixed(1);
                const elap = ((performance.now() - t0) / 1000).toFixed(1);
                console.log(`[decode] round ${roundNum}/${totalRounds}: ${decodedSec}s decoded (${elap}s elapsed)`);
                cursor += CHUNK;
                // Yield to the GPU compositor between rounds so the OS can
                // render cursor + desktop frames. Without this, back-to-back
                // WebGPU compute starves the compositor and the whole system
                // feels frozen. 50ms = 3 compositor frames at 60fps.
                await new Promise((r) => setTimeout(r, 50));
              }

              // Final dispatch: swap all stems to their full-length audio in one shot.
              for (const name of stemNames) dispatchStemAudio(name);
              console.log(`[decode] all stems fully decoded at ${((performance.now() - t0)/1000).toFixed(2)}s (WebGPU batched)`);
            }
          })();
          } // END DECODE LOOP DISABLED
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
          {/* Combined spacer - spans columns 1 & 2 with transport controls */}
          <div className={styles.timelineSpacer1} style={{ display: 'flex', alignItems: 'center', gap: '8px', padding: '4px 8px 4px 8px', justifyContent: 'space-between', paddingRight: '18px' }}>
            {/* Tempo + Meter Group: Metronome + BPM + Meter (LEFT) */}
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              background: 'rgba(30, 30, 30, 0.6)',
              padding: '4px',
              borderRadius: '6px',
              border: '1px solid rgba(102, 126, 234, 0.2)',
              flex: '0 0 auto'
            }}>
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

              {/* Meter Select */}
              <div style={{ display: 'flex', alignItems: 'center', gap: '3px' }} title="Time signature">
                <label htmlFor="meter-input" style={{ color: '#c5cae9', fontSize: '11px', fontWeight: '600' }}>
                  Meter
                </label>
                <select
                  id="meter-input"
                  value={`${state.beatsPerBar || 4}/${state.meterDenominator || 4}`}
                  onChange={(e) => dispatch({ type: 'SET_METER', payload: e.target.value })}
                  style={{
                    width: '60px',
                    padding: '4px 5px',
                    borderRadius: '4px',
                    border: '1px solid rgba(102, 126, 234, 0.3)',
                    background: 'rgba(20, 20, 20, 0.8)',
                    color: 'white',
                    fontSize: '12px',
                    textAlign: 'center'
                  }}
                >
                  <option value="3/4">3/4</option>
                  <option value="4/4">4/4</option>
                  <option value="5/4">5/4</option>
                  <option value="6/8">6/8</option>
                  <option value="7/8">7/8</option>
                </select>
              </div>

              {repaintApplying && (
                <i className="fa-solid fa-wand-magic-sparkles fa-spin"
                   style={{ color: '#8B7FF0', marginLeft: 4, fontSize: 14 }}
                   title="Repainting tracks via stemphonic stage2d-130k" />
              )}
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
