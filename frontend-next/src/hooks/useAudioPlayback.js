import { useEffect, useRef, useCallback } from 'react';
import midiPlayer from '../utils/midiPlayer';
import tunaFX from '../services/tunaFX';
import { isMaskPlaybackReady, setGains as setMaskGains, setActiveStem, seek as maskSeek, play as maskPlay, stop as maskStop } from '../services/maskPlayback';
import { dispatchStrategy, getTrackSubstemSchedules, resumeFromPlayhead } from '../services/virtualTrackEdit';
import { getStretchedBuffer } from '../services/wsolaStretch';
import { LRUBufferCache } from '../utils/lruBufferCache';
import { computeClipPlayback } from '../utils/clipScheduling';
import { scheduleAutomation, clearAutomation } from '../services/automationPlayback';
import PluginAdapter from '../lib/PluginAdapter';
import liveTrackChainRegistry from '../lib/liveTrackChainRegistry';
import { useApp } from '../context/AppContext';
import {
  enqueueSetPluginParam,
  enqueueSetPluginBypass,
} from '../services/sessionEditsAPI';

// R11 splice — feature flag for live web-DSP plugin chains. When unset/empty
// (the default), the entire PluginAdapter path is bypassed and behavior is
// byte-identical to the bounce-cache pipeline. Set NEXT_PUBLIC_LIVE_PLUGINS=1
// to opt in. Comparison is strict against '1' (Vercel ships some envs as the
// literal string 'true'/'false', so we lock to '1' for unambiguous gating).
const LIVE_PLUGINS_ENABLED =
  typeof process !== 'undefined' &&
  process.env &&
  process.env.NEXT_PUBLIC_LIVE_PLUGINS === '1';

// Global audio buffer cache — byte-capped LRU to keep a long session's worth
// of synced projects from OOM-ing the tab. At 48kHz stereo float32, the cap
// below holds ~22 minutes of audio before the oldest buffer gets bumped.
const audioBufferCache = new LRUBufferCache({
  maxBytes: 512 * 1024 * 1024,
  name: 'audioBufferCache',
});
const RESUME_ATTACK_SECONDS = 0.005;

function isRenderablePlaybackTrack(track) {
  // `isBusMaster` is the hidden uploaded full-mix parent kept in state for
  // analysis/regen after stem separation. It must never be scheduled for
  // playback once stems exist or it will mask stem edits and double the mix.
  return !track?.metadata?.isBusMaster;
}

// MIDI-type detection. Tracks added through the StudioDev "add instrument"
// flow stamp `metadata.type = 'midi'` but leave the top-level `type` field
// undefined. Older paths (DAW BusRow, MIDI browser, useAudioRecorder, etc.)
// stamp `type: 'midi'` at the top level. Either form should be treated as
// a MIDI track for scheduling — without this helper, the studio piano-roll
// tracks fall through every MIDI branch and produce silence on playback.
function isMidiTypeTrack(track) {
  return track?.type === 'midi' || track?.metadata?.type === 'midi';
}

function normalizeStemForMaskPlayback(track) {
  const raw = (
    track?.metadata?.stemType ||
    track?.metadata?.instrument ||
    track?.metadata?.instrumentGroup ||
    ''
  ).toLowerCase();

  if (!raw) return null;
  if (raw === 'drums' || raw === 'drum_kit' || raw === 'percussion') return 'drums';
  if (raw === 'bass' || raw === 'electric_bass' || raw === 'upright_bass') return 'bass';
  if (
    raw === 'vocals' ||
    raw === 'lead_vox' ||
    raw === 'bg_vox' ||
    raw === 'choir' ||
    raw === 'voice'
  ) return 'vocals';
  return 'other';
}

/**
 * Custom hook for managing audio playback across multiple tracks
 * Integrates with global state via dispatch
 */
export function useAudioPlayback(tracks, isPlaying, dispatch, totalDuration = 10, currentPlayheadPosition = 0, bpm = 120, masterGain = 0.8, beatsPerBar = 4, meterDenominator = 4, tempoMap = null, beatMap = null, timelineOffset = 0, cycleRegion = null) {
  const audioContextRef = useRef(null);
  const masterGainNodeRef = useRef(null); // Master gain node for overall volume control
  const sourceNodesRef = useRef([]);
  const gainNodesRef = useRef(new Map()); // Map of trackId -> gainNode for real-time updates
  // Per-bus GainNode keyed by busId. Topology:
  //   track source → trackGain → busGain[busId] → masterGain → destination
  // Bus gain holds bus.gain × bus-level mute/solo state. Track gain holds
  // track.gain × track-level mute/solo. Multiplicative. Persists across
  // pause/play cycles so fader automation stays put.
  const busGainNodesRef = useRef(new Map());
  // Per-track aux send GainNodes — Map<trackId, GainNode[]>. A track can
  // tap its postfader signal into N destination bus inputs at custom
  // levels. Stored separately from gainNodesRef so the gain-update effect
  // can reach send levels without iterating sources.
  const sendGainNodesRef = useRef(new Map());
  const startTimeRef = useRef(0);
  const pauseTimeRef = useRef(0);
  const animationFrameRef = useRef(null);
  const isSeekingRef = useRef(false);
  const lastKnownPlayheadRef = useRef(0);
  const isPlayingRef = useRef(isPlaying); // Track isPlaying in ref for animation loop
  const tracksRef = useRef(tracks); // Store tracks in ref to avoid recreating play callback
  const hasStartedPlaybackRef = useRef(false); // Track if playback has actually started
  const playbackSessionRef = useRef(0); // Invalidate stale late-load scheduling across pause/play cycles
  const playRef = useRef(null); // Latest play() fn — called by the meter-change live-reschedule effect
  const seekRef = useRef(null); // Latest seek() fn — called by the cycle-loop wrap in updatePlayhead
  const cycleLoopGuardRef = useRef(0); // debounce loop wraparound to avoid double-fires

  // R11 PluginAdapter splice — lazy singleton bound to the AudioContext.
  // `pluginAdapterRef` holds the adapter; `trackChainsRef` tracks live
  // chains keyed by trackId so dispose runs on stop/seek/unmount. Both
  // remain dormant when the feature flag is off.
  const pluginAdapterRef = useRef(null);
  const trackChainsRef = useRef(new Map()); // Map<trackId, {input, output, dispose, slots}>

  // activeSessionId is read inside the adapter's editCallbacks so the
  // closure stays stable across navigations. We pull it from AppContext
  // and mirror into a ref the callback can dereference on every fire.
  // This avoids re-creating the adapter every time the session changes.
  const { state: appState } = useApp();
  const activeSessionIdRef = useRef(appState?.activeSessionId || null);
  useEffect(() => {
    activeSessionIdRef.current = appState?.activeSessionId || null;
  }, [appState?.activeSessionId]);

  // Mid-playback chain mutation: when a track's logicPlugins length /
  // identity changes (e.g. peer adds Tape Delay), call the chain's
  // appendSlot / removeSlot so the running engine reflects the change
  // without stopping playback. We track per-track signatures and
  // diff them; appendSlot/removeSlot keep the chain.input/output
  // nodes stable so external connections survive.
  const trackPluginSigsRef = useRef(new Map()); // trackId → signature string
  useEffect(() => {
    if (!LIVE_PLUGINS_ENABLED) return;
    const buses = tracksRef.current?.byBus || tracks; // tracks shape varies
    const list = Array.isArray(tracks) ? tracks : [];
    for (const t of list) {
      const tid = t?.id;
      if (!tid) continue;
      const chain = trackChainsRef.current.get(tid);
      if (!chain || !chain.appendSlot) continue;
      const cur = (t.logicPlugins || []).map(
        (p, i) => `${p?.plugin_id || 0}:${p?.plugin_name || ''}:${i}`
      ).join('|');
      const prior = trackPluginSigsRef.current.get(tid) || '';
      if (cur === prior) continue;

      // Identity-aware diff: walk the prior and current signature
      // tokens in lockstep, removing any slot that changed identity
      // and appending replacements at the right positions. Handles
      // (a) tail append, (b) tail remove, (c) middle remove + tail
      // append (peer reorder), and (d) replace-in-place (peer
      // dropped one plugin and added another at the same slot). The
      // mutators chain.removeSlot / chain.appendSlot keep slot
      // indices contiguous so the post-diff state matches `cur`.
      //
      // Reorder of unchanged slots (e.g. peer drag-reordered three
      // plugins without changing identities) still falls through to
      // the next play() rebuild — the audio path handles that case
      // correctly enough that re-routing mid-playback isn't worth
      // the complexity. Identity-changes — the common collab edits —
      // are now handled live.
      const priorTokens = prior ? prior.split('|').filter(Boolean) : [];
      const curTokens = cur ? cur.split('|').filter(Boolean) : [];

      // Build a map of identity-change positions. The signature is
      // `pluginId:pluginName:index`; we strip the trailing `:index`
      // before comparing so a slot moving stays "same identity".
      const identityOf = (tok) => tok.split(':').slice(0, -1).join(':');
      const priorIds = priorTokens.map(identityOf);
      const curIds = curTokens.map(identityOf);

      // Walk forward, finding the first divergent index.
      let div = 0;
      while (div < priorIds.length && div < curIds.length
             && priorIds[div] === curIds[div]) div++;

      // From `div` onward, drop everything that differs (tail
      // remove or replace), then append the new tail. removeSlot
      // shifts indices, so iterate from the end. Appendix runs
      // sequentially after the tail is gone.
      const removeFrom = div;
      for (let i = priorIds.length - 1; i >= removeFrom; i--) {
        try { chain.removeSlot(i); } catch (_) { /* noop */ }
      }
      for (let i = removeFrom; i < curIds.length; i++) {
        const lp = t.logicPlugins[i];
        if (!lp) continue;
        // appendSlot is async — fire-and-forget, audio splice
        // happens when the new slot is ready.
        Promise.resolve(chain.appendSlot(lp)).catch((err) => {
          console.warn(
            `[R11] appendSlot failed for ${tid}: ${err?.message || err}`,
          );
        });
      }
      trackPluginSigsRef.current.set(tid, cur);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tracks]);

  // Lazily create the bus GainNode for a given busId and return it. Called
  // at schedule time and from the real-time gain-update effect. If busId is
  // falsy, falls back to masterGain so tracks that bypass the bus model
  // (legacy / test) still play.
  const ensureBusGain = (busId) => {
    const master = masterGainNodeRef.current;
    if (!busId || !master || !audioContextRef.current) return master;
    let node = busGainNodesRef.current.get(busId);
    if (!node) {
      node = audioContextRef.current.createGain();
      node.gain.value = 1.0;   // real-time effect will clamp per mute/solo
      node.connect(master);
      busGainNodesRef.current.set(busId, node);
    }
    return node;
  };

  // R11 splice — try to build a live web-DSP plugin chain for this track.
  // Returns null when:
  //   - feature flag NEXT_PUBLIC_LIVE_PLUGINS !== '1' (default-disabled path)
  //   - adapter hasn't initialized (no AudioContext yet)
  //   - track has no logicPlugins metadata
  //   - any plugin on the track lacks a registered mapping (chain.fallback)
  // When the return is null, the caller MUST take the unchanged bounce-cache
  // path. When non-null, the returned chain has shape {input, output,
  // dispose, slots} and the caller is responsible for splicing it into the
  // graph and registering dispose for cleanup.
  const maybeBuildPluginChain = useCallback(async (track) => {
    if (!LIVE_PLUGINS_ENABLED) return null;
    const adapter = pluginAdapterRef.current;
    if (!adapter) return null;
    // sessionSyncAPI._adaptTrack surfaces logicPlugins on metadata, so
    // accept either layout. New code prefers the top-level field.
    const logicPlugins = track?.logicPlugins ?? track?.metadata?.logicPlugins;
    if (!Array.isArray(logicPlugins) || logicPlugins.length === 0) return null;
    try {
      // PluginAdapter expects logicPlugins at the top level — pass a
      // shallow override so a metadata-nested record still resolves.
      const trackForAdapter = track?.logicPlugins
        ? track
        : { ...track, logicPlugins };
      const chain = await adapter.buildTrackChain(trackForAdapter);
      if (!chain || chain.fallback) return null;
      return chain;
    } catch (err) {
      // Any error → bounce-cache fallback. Never let a plugin-mapping bug
      // silence playback for a track.
      console.warn(`[R11] buildTrackChain failed for ${track?.id || track?.name || '?'}; using bounce path:`, err?.message || err);
      return null;
    }
  }, []);

  // A5 — register/unregister live PluginAdapter chains in the singleton
  // registry so useLiveParamDeltas can look slots up by ginstid/trackIndex
  // when doo_hook fires a param_delta. Falls back gracefully when the
  // sync record lacks those identifiers — A5's lookup tries every key.
  const _registerChain = (track, chain) => {
    if (!chain || !track) return;
    // Seed the per-track signature so the mutation watcher knows the
    // chain reflects this exact logicPlugins shape. Without seeding,
    // the next state-tick would diff against an empty prior and
    // append every plugin a second time.
    try {
      const sig = (track.logicPlugins || []).map(
        (p, i) => `${p?.plugin_id || 0}:${p?.plugin_name || ''}:${i}`
      ).join('|');
      trackPluginSigsRef.current.set(track.id, sig);
    } catch (_) { /* noop */ }
    try {
      const meta = track.metadata || {};
      const ginstid =
        typeof track.logicGinstid === 'number' ? track.logicGinstid
        : typeof meta.logicGinstid === 'number' ? meta.logicGinstid
        : null;
      const trackIndex =
        typeof track.logicTrackIndex === 'number' ? track.logicTrackIndex
        : typeof meta.logicTrackIndex === 'number' ? meta.logicTrackIndex
        : null;
      const trackUuid = track.uuid || meta.uuid || null;
      liveTrackChainRegistry.register(track.id, {
        trackIndex,
        ginstid,
        // trackUuid lets useEditStream's inbound plugin handlers match
        // a peer edit (which addresses by uuid) onto the local chain
        // (which is keyed by track.id). Registry never indexes by uuid
        // itself — the snapshot walk in _applyToLiveSlot is bounded by
        // O(tracks-with-chains), which is small.
        trackUuid: trackUuid ? String(trackUuid).toLowerCase() : null,
        slots: chain.slots || [],
        // A2 owns disposal; the registry never invokes dispose itself.
        // We mirror it here as a no-op so unit tests can opt-in.
        dispose: undefined,
      });
    } catch (err) {
      console.warn(`[A5] registry.register failed for ${track?.id || '?'}:`, err?.message || err);
    }
  };
  const _unregisterChain = (trackId) => {
    if (trackId == null) return;
    try { liveTrackChainRegistry.unregister(trackId); } catch (_) {}
  };
  const _clearRegistry = () => {
    try { liveTrackChainRegistry.clear(); } catch (_) {}
  };

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


    // Attach the MIDI player to the main graph. One shared AudioContext
    // (no drift against audio tracks) and all MIDI output flows through
    // masterGain (master fader + any future FX apply uniformly). For
    // bus-aware routing, playback passes the track's busGainNode as
    // opts.destination per playNote call — see the MIDI notes loop below.
    midiPlayer.attachToContext(audioContextRef.current, masterGainNodeRef.current);

    // R11 PluginAdapter — lazy singleton bound to this AudioContext.
    // Constructed unconditionally (cheap) but only USED when the feature
    // flag is on. `load()` is fire-and-forget: kicks off the index.json
    // fetch so a later maybeBuildPluginChain() call can resolve fast.
    // When the flag is off, nothing ever calls into the adapter.
    let pluginIndexRefreshTimer = null;
    if (LIVE_PLUGINS_ENABLED) {
      try {
        // Edit callbacks broadcast plugin param/bypass changes through
        // the session edit-log so the desktop replays them into Logic
        // (and peers see them too). activeSessionIdRef lets the
        // callback read the latest session id without re-instantiating
        // the adapter every time the user navigates between sessions.
        const editCallbacks = {
          onParamEdit: ({ trackUuid, slot, paramId, value }) => {
            const sid = activeSessionIdRef.current;
            if (!sid || !trackUuid) return;
            enqueueSetPluginParam(sid, trackUuid, slot, paramId, value);
          },
          onBypassChange: ({ trackUuid, slot, bypassed }) => {
            const sid = activeSessionIdRef.current;
            if (!sid || !trackUuid) return;
            enqueueSetPluginBypass(sid, trackUuid, slot, bypassed);
          },
        };
        pluginAdapterRef.current = new PluginAdapter(
          audioContextRef.current,
          { editCallbacks },
        );
        // Fire-and-forget: failures here just mean every chain attempt
        // falls back to the bounce-cache path (the safe default).
        Promise.resolve(pluginAdapterRef.current.load()).catch((err) => {
          console.warn('[R11] PluginAdapter load failed; falling back to bounce path:', err?.message || err);
        });
        // Mid-session mapping refresh: poll index.json every 60s so
        // mappings the desktop publishes during the session (via
        // auto_driver.publish --zero-shot) become available without a
        // tab reload. Cheap — one HTTP request per minute.
        pluginIndexRefreshTimer = setInterval(() => {
          const adapter = pluginAdapterRef.current;
          if (!adapter || typeof adapter.refreshIndex !== 'function') return;
          Promise.resolve(adapter.refreshIndex()).catch(() => { /* swallow */ });
        }, 60_000);
      } catch (err) {
        console.warn('[R11] PluginAdapter init failed; falling back to bounce path:', err?.message || err);
        pluginAdapterRef.current = null;
      }
    }

    return () => {
      tunaFX.destroy();
      midiPlayer.stopAll();
      // R11 — dispose any live plugin chains and drop adapter caches so the
      // next AudioContext gets a fresh singleton (engines hold ctx-bound
      // AudioNodes; reusing across contexts would dangle them).
      try {
        for (const chain of trackChainsRef.current.values()) {
          try { chain.dispose && chain.dispose(); } catch (_) {}
        }
        trackChainsRef.current.clear();
        _clearRegistry();
      } catch (_) {}
      if (pluginIndexRefreshTimer != null) {
        try { clearInterval(pluginIndexRefreshTimer); } catch (_) {}
        pluginIndexRefreshTimer = null;
      }
      if (pluginAdapterRef.current) {
        try { pluginAdapterRef.current.clearCache && pluginAdapterRef.current.clearCache(); } catch (_) {}
        pluginAdapterRef.current = null;
      }
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

  // Update gain nodes in real-time when track settings change (without
  // restarting playback). Split model: per-bus GainNode holds bus.gain +
  // bus-level mute/solo; per-track GainNode holds track.gain + track-level
  // mute/solo. The two multiply through the signal chain so net audible
  // gain matches the old single-node precomputed `finalGain` exactly.
  //
  // Old precedence (kept for parity): bus-mute zeros everything; bus-solo
  // overrides track mute/solo entirely (solo'd bus plays even if its
  // tracks are muted); otherwise track solo/mute applies.
  useEffect(() => {
    if (!isPlaying) return;
    const ctx = audioContextRef.current;
    if (!ctx) return;

    const voTracks = (tracks.vo || []).filter(isRenderablePlaybackTrack);
    const musicTracks = (tracks.music || []).filter(isRenderablePlaybackTrack);
    const sfxTracks = (tracks.sfx || []).filter(isRenderablePlaybackTrack);
    const midiTracks = (tracks.midi || []).filter(isRenderablePlaybackTrack);
    const audioTracks = (tracks.audio || []).filter(isRenderablePlaybackTrack);

    // Include MIDI bus audio tracks (not MIDI type tracks)
    const midiAudioTracks = midiTracks.filter(t => !isMidiTypeTrack(t) && t.audioUrl);

    const allTracks = [...voTracks, ...musicTracks, ...sfxTracks, ...midiAudioTracks, ...audioTracks];

    const hasSoloTracks = allTracks.some(track => track.isSolo);
    const hasBusSolo = allTracks.some(t => t._busSolo);

    // Per-bus pass: collect unique bus state (all tracks on a bus carry
    // the same snapshot of bus.gain/mute/solo, so first seen wins).
    const busInfo = new Map();   // busId → { gain, muted, solo }
    for (const t of allTracks) {
      if (!t._busId || busInfo.has(t._busId)) continue;
      busInfo.set(t._busId, {
        gain:  typeof t._busGain === 'number' ? t._busGain : 1.0,
        muted: !!t._busMuted,
        solo:  !!t._busSolo,
      });
    }
    const now = ctx.currentTime;
    for (const [busId, info] of busInfo) {
      const node = busGainNodesRef.current.get(busId);
      if (!node) continue;
      let g;
      if (info.muted)                  g = 0;
      else if (hasBusSolo && !info.solo) g = 0;
      else                             g = info.gain;
      node.gain.setTargetAtTime(g, now, 0.01);
    }

    // Per-track pass: track node holds track.gain × (track mute/solo).
    // When bus-solo is active, track mute/solo is IGNORED (parity with
    // pre-refactor behavior); the bus node has already zeroed non-solo
    // buses, so this just lets solo'd-bus tracks pass through cleanly.
    //
    // Automation interaction: if the track has a volume lane and the
    // gate is open, re-schedule the lane against the current transport
    // position rather than stomping the param with a static value.
    // setValueAtTime would override the in-flight ramp; scheduleAutomation
    // cancels and re-anchors so live edits to the lane are heard.
    if (gainNodesRef.current.size === 0) return;
    allTracks.forEach(track => {
      const gainNode = gainNodesRef.current.get(track.id);
      if (!gainNode) return;
      let trackGate;
      if (hasBusSolo)                       trackGate = 1;
      else if (hasSoloTracks)               trackGate = track.isSolo ? 1 : 0;
      else                                  trackGate = track.isMuted ? 0 : 1;
      const baseGain = (track.gain || 1.0) * trackGate;
      const volLane = track.automation?.volume;
      const hasVolAuto = trackGate > 0 && Array.isArray(volLane) && volLane.length > 0;
      if (hasVolAuto) {
        // Re-anchor automation at the live transport position so edits
        // are heard immediately. startTimeRef is the AudioContext anchor
        // for transport time 0, so currentTransport = ctx.currentTime -
        // startTimeRef.current.
        const transportTime = audioContextRef.current.currentTime - (startTimeRef.current || 0);
        scheduleAutomation(gainNode.gain, volLane, {
          audioContextCurrentTime: audioContextRef.current.currentTime,
          schedulingStartTime: audioContextRef.current.currentTime,
          anchorTimelineTime: transportTime,
          fallbackValue: baseGain,
        });
      } else {
        gainNode.gain.setTargetAtTime(baseGain, now, 0.01);
      }
    });

    // Also update mask playback gains for stem tracks
    if (isMaskPlaybackReady()) {
      const stemGains = {};
      let activeSolo = null;
      const stemTracks = allTracks.filter(t => t.metadata?.type === 'stem');
      const hasStemSolo = stemTracks.some(t => t.isSolo);
      for (const t of stemTracks) {
        const stemName = normalizeStemForMaskPlayback(t);
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
    }
  }, [currentPlayheadPosition, isPlaying]);

  // Pre-load and cache all audio buffers when tracks change
  useEffect(() => {
    if (!audioContextRef.current) return;

    const audioContext = audioContextRef.current;
    const voTracks = (tracks.vo || []).filter(isRenderablePlaybackTrack);
    const musicTracks = (tracks.music || []).filter(isRenderablePlaybackTrack);
    const sfxTracks = (tracks.sfx || []).filter(isRenderablePlaybackTrack);
    const audioTracks = (tracks.audio || []).filter(isRenderablePlaybackTrack);
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
      // Drum substem WAVs (from backend MDX23C teacher). Without this,
      // they're only fetched inside scheduleOrDefer's late-load path,
      // which races user seeks — if a seek lands before fetch+decode
      // finishes, the captured session becomes stale and the substem
      // is silently dropped. Pre-warming here caches the buffers so
      // every subsequent play() schedules them immediately.
      const subUrls = track.metadata?.drumSubstems;
      if (subUrls && typeof subUrls === 'object') {
        for (const u of Object.values(subUrls)) {
          if (u) audioUrls.push(u);
        }
      }
      // Also preload F0 audio if available
      if (track.f0Audio) {
        audioUrls.push(track.f0Audio);
      }
      // Per-clip variants: when the desktop emits metadata.clips (Logic
      // regions with per-clip source offsets + optional root/stretch),
      // pre-warm each clip's actual playback buffer. Skip rootUrl unless
      // playbackPath='root_stretch' — in 'variant' (SPEED mode) rootUrl
      // is informational and would waste bandwidth on a file we never
      // play. Legacy clips without `playbackPath` get both pre-warmed
      // since the scheduler's heuristic could pick either.
      const clips = track.metadata?.clips;
      if (Array.isArray(clips)) {
        for (const c of clips) {
          if (c?.url) audioUrls.push(c.url);
          if (c?.rootUrl && c.playbackPath !== 'variant') {
            audioUrls.push(c.rootUrl);
          }
        }
      }
    });
    const uniqueUrls = [...new Set(audioUrls.filter(Boolean))];

    // Bail entirely when nothing new — the effect re-runs on every
    // state.buses change (each MIDI commit during the Tier 1/2/3
    // transcription cascade), but the URL set rarely changes. Without
    // this gate, fetchAudioWithCache fired a fresh IndexedDB lookup +
    // network race per midi commit on every stem URL — and polypitch's
    // own loadAudioBuffer (which now shares fetchAudioWithCache via
    // the in-flight dedupe) was queueing behind that storm and hung
    // at "loading audio…" for 20+ seconds per stem.
    const uncachedUrls = uniqueUrls.filter((u) => !audioBufferCache.has(u));
    if (uncachedUrls.length === 0) return;
    console.log(`[prewarm] iterating ${uncachedUrls.length} uncached url(s):`, uncachedUrls.map((u) => u.slice(0, 80)));
    uniqueUrls.forEach(async (url) => {
      if (!audioBufferCache.has(url)) {
        const tStart = performance.now();
        console.log(`[prewarm] ⏳ fetching ${url.slice(0, 80)}`);
        try {
          // Reuse the shared audioCacheService blob (in-flight Promise
          // dedupe + IndexedDB cache + memory cache). This avoids the
          // dual-fetch where useAudioPlayback and useWaveform each
          // hit the network for the same stem URL.
          const { fetchAudioWithCache } = await import('../services/audioCacheService');
          const { blob } = await fetchAudioWithCache(url);
          console.log(`[prewarm] ✓ fetched ${url.slice(0, 80)} in ${(performance.now() - tStart).toFixed(0)}ms (${(blob.size / 1024).toFixed(0)}KB)`);
          const arrayBuffer = await blob.arrayBuffer();
          const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
          console.log(`[prewarm] ✓ decoded ${url.slice(0, 80)} — ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.numberOfChannels}ch`);
          audioBufferCache.set(url, audioBuffer);

          // Update track duration if it's 0
          const trackWithThisUrl = allTracks.find(t => t.audioUrl === url);
          if (trackWithThisUrl && (trackWithThisUrl.duration === 0 || !trackWithThisUrl.duration)) {

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

    // ── Cycle / loop region ─────────────────────────────────────────
    // Logic emits cycleRegion as { enabled, start_ticks, end_ticks }. When
    // active and the playhead crosses end_ticks, snap back to start_ticks
    // and re-schedule. Linear-bpm tick→sec conversion is good enough for
    // SPEED-mode projects (no tempo ramp); a tempoMap-aware version would
    // walk the map for projects with mid-cycle tempo changes.
    if (cycleRegion && cycleRegion.enabled
        && Number.isFinite(Number(cycleRegion.start_ticks))
        && Number.isFinite(Number(cycleRegion.end_ticks))
        && (bpm > 0)) {
      const tickToSec = 60 / (960 * bpm);
      const cycStart = Number(cycleRegion.start_ticks) * tickToSec;
      const cycEnd   = Number(cycleRegion.end_ticks)   * tickToSec;
      if (cycEnd > cycStart && currentTime >= cycEnd) {
        const ctxNow = audioContextRef.current.currentTime;
        // 50ms guard so a slow seek doesn't double-fire on the next frame.
        if (ctxNow - cycleLoopGuardRef.current > 0.05) {
          cycleLoopGuardRef.current = ctxNow;
          const fn = seekRef.current;
          if (typeof fn === 'function') {
            fn(cycStart);
            // Don't fall through to the totalDuration check; seek owns the
            // next frame.
            animationFrameRef.current = requestAnimationFrame(updatePlayhead);
            return;
          }
        }
      }
    }

    if (currentTime >= totalDuration) {
      // End of playback
      dispatch({ type: 'SET_PLAYING', payload: false });
      dispatch({ type: 'RESET_PLAYHEAD' });
      return;
    }

    animationFrameRef.current = requestAnimationFrame(updatePlayhead);
  }, [dispatch, totalDuration, cycleRegion, bpm]);

  /**
   * Start playback from current position
   * Respects each track's startPosition on the timeline
   */
  const play = useCallback(async () => {
    if (!audioContextRef.current) return;

    const audioContext = audioContextRef.current;
    const sessionId = ++playbackSessionRef.current;
    const isSessionActive = () =>
      playbackSessionRef.current === sessionId && isPlayingRef.current;

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

      // R11 — dispose any live plugin chains from the previous play() pass.
      // Each chain owns engines holding context-bound AudioNodes; if we
      // skip this, every play/seek leaks a chain into the graph.
      if (LIVE_PLUGINS_ENABLED && trackChainsRef.current.size > 0) {
        for (const chain of trackChainsRef.current.values()) {
          try { chain.dispose && chain.dispose(); } catch (_) {}
        }
        trackChainsRef.current.clear();
        _clearRegistry();
      }

      // Get all tracks from ref (not prop) to avoid recreating this callback
      const currentTracks = tracksRef.current;
      const voTracks = (currentTracks.vo || []).filter(isRenderablePlaybackTrack);
      const musicTracks = (currentTracks.music || []).filter(isRenderablePlaybackTrack);
      const sfxTracks = (currentTracks.sfx || []).filter(isRenderablePlaybackTrack);
      const drumTracks = (currentTracks.drums || []).filter(isRenderablePlaybackTrack);
      const midiTracks = (currentTracks.midi || []).filter(isRenderablePlaybackTrack);
      const audioTracks = (currentTracks.audio || []).filter(isRenderablePlaybackTrack);

      // Include MIDI bus tracks - include MIDI tracks with audioUrl OR f0Audio
      const midiAudioTracks = midiTracks.filter(t =>
        (!isMidiTypeTrack(t) && t.audioUrl) || // Audio tracks in MIDI bus
        (isMidiTypeTrack(t) && (t.audioUrl || t.f0Audio || t.midiData)) // MIDI tracks with playable content
      );

      const allTracks = [...voTracks, ...musicTracks, ...sfxTracks, ...drumTracks, ...midiAudioTracks, ...audioTracks];

      // Bus arrival order — used to resolve aux sends' bus_sub_index
      // (1-based) against actual bus IDs in this session.
      const allBusIds = [];
      for (const t of allTracks) {
        if (t._busId && !allBusIds.includes(t._busId)) allBusIds.push(t._busId);
      }

      // Check if any track has solo enabled
      const hasSoloTracks = allTracks.some(track => track.isSolo);

      // Current playhead position in timeline (seconds)
      const currentPlayheadTime = pauseTimeRef.current;

      // Capture audioContext.currentTime ONCE to use as reference for all scheduling
      const schedulingStartTime = audioContext.currentTime;

      // Establish the timeline origin BEFORE any async work. This ensures
      // the playhead advances the instant the user hits play even if no
      // buffers are cached yet. Tracks whose buffers arrive later schedule
      // themselves retroactively at the live playhead (see scheduleOrDefer
      // below).
      startTimeRef.current = schedulingStartTime - currentPlayheadTime;
      hasStartedPlaybackRef.current = true;
      updatePlayhead();

      // Resume MIDI player audio context — fire and forget. If it misses
      // the first few ms of notes, the notes are still scheduled against
      // AudioContext time so they catch up.
      const hasMidiTracks = allTracks.some(t => isMidiTypeTrack(t) && t.midiData);
      if (hasMidiTracks) {
        midiPlayer.resume().then(() => console.log('🎹 MIDI player audio context resumed'))
          .catch((e) => console.warn('MIDI resume failed:', e));
      }

      // Non-blocking preload — kicks off fetches + decodes for any URLs
      // not yet in the cache. The actual scheduling happens in
      // scheduleOrDefer on a per-track basis and doesn't wait for this.
      const urlsToWarm = [];
      for (const t of allTracks) {
        if (t.audioUrl && !audioBufferCache.has(t.audioUrl)) urlsToWarm.push(t.audioUrl);
        if (isMidiTypeTrack(t) && t.f0Audio && !audioBufferCache.has(t.f0Audio)) urlsToWarm.push(t.f0Audio);
      }
      if (urlsToWarm.length) {
        console.log(`📦 Warming ${urlsToWarm.length} uncached buffer(s) in background`);
      }

      // Every regular-audio track plays through a Schedule — a list of
      // segments mapping src-buffer windows to dst-timeline windows with
      // rate 1 (rearrange-only) and short fades at cut points. Identity
      // case collapses to a single full-buffer segment, so this path is
      // safe for every track — no separate "simple" vs "edited" branch.
      //
      // When project meter != track meter, getTrackSchedule returns a
      // keep/drop/duplicate schedule (see services/virtualTrackEdit.js).
      // NO pitch shift — playbackRate is never used for meter math.
      //
      // DRUM SUBSTEMS: if the track has metadata.drumSubstems (the per-
      // substem WAV URLs from MDX23C-DrumSep on the backend, attached by
      // the onDrumTeacher callback), we ignore the bar-level mix and
      // schedule each substem independently. Percussive substems get
      // hit-snap to the new meter grid; sustain substems stay on bar
      // rearrange. All substems mix into ONE shared per-track gain so
      // solo/mute on the parent drum track keeps working.
      // Derive project-level barStarts from the analyze-rhythm beatMap
      // so drum stem tracks that don't carry their own barStarts/downbeat
      // can still align to real detected downbeats. Without this, the
      // substem scheduler falls back to synthesized bars (constant-BPM
      // grid starting at t=0), which misaligns triplet-snap / 4+3 warps.
      const projectBarStarts = (Array.isArray(beatMap) && beatMap.length > 0)
        ? beatMap.filter((b) => b && b.pos === 1).map((b) => b.t)
        : null;
      const projectMeter = {
        bpm, beatsPerBar, meterDenominator, tempoMap,
        barStarts: projectBarStarts,
        downbeatOffset: typeof timelineOffset === 'number' ? timelineOffset : 0,
      };

      // Helper: ensure a buffer is decoded + cached, then run the cb.
      const ensureBuffer = async (url) => {
        if (audioBufferCache.has(url)) return audioBufferCache.get(url);
        const { fetchAudioWithCache } = await import('../services/audioCacheService');
        const { blob } = await fetchAudioWithCache(url);
        const arrayBuffer = await blob.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        audioBufferCache.set(url, audioBuffer);
        return audioBuffer;
      };

      const scheduleOrDefer = (track, trackLocalGain) => {
        const url = track.audioUrl;
        if (!url) return;
        const trackStartTime = track.startPosition || 0;

        // Per-bus GainNode. Track gain connects here instead of masterGain;
        // bus handles its own gain/mute/solo independently.
        const busGainNode = ensureBusGain(track._busId);

        // R11 splice — pre-resolved live plugin chain for this track (or
        // null if the feature flag is off / no mappings / fallback). When
        // present, the chain is wired between the per-track sink (trackGain
        // or trackPan) and busGainNode below.
        const pluginChain = prebuiltChains.get(track.id) || null;

        // ── Per-track sink: gain + (optional) pan + automation ──────────
        // Topology when pan exists/automated:
        //   sources → trackGain → trackPan → busGain → master
        // Otherwise:
        //   sources → trackGain → busGain → master
        // With R11 plugin chain spliced in:
        //   ... → (trackGain | trackPan) → chain.input → ... → chain.output → busGain
        // Sub-schedulers receive `trackGain` as existingTrackGain so the
        // sink is created exactly once per play() per track. Volume and
        // pan automation lanes are scheduled here against the live
        // transport anchor.
        const volLane = track.automation?.volume;
        const panLane = track.automation?.pan;
        const hasVolAuto = Array.isArray(volLane) && volLane.length > 0;
        const hasPanAuto = Array.isArray(panLane) && panLane.length > 0;
        const staticPan = Number(track.pan) || 0;
        const wantPanNode = hasPanAuto || Math.abs(staticPan) > 1e-3;

        const trackGain = audioContext.createGain();
        trackGain.gain.value = trackLocalGain;

        // The "sink" — last per-track node before either busGainNode or the
        // plugin chain. Pan node sits between trackGain and the sink path
        // when present; otherwise trackGain itself is the sink.
        let trackPan = null;
        let busFeed; // node whose output should reach busGainNode (directly or via chain)
        if (wantPanNode && typeof audioContext.createStereoPanner === 'function') {
          trackPan = audioContext.createStereoPanner();
          trackPan.pan.value = Math.max(-1, Math.min(1, staticPan));
          trackGain.connect(trackPan);
          busFeed = trackPan;
        } else {
          busFeed = trackGain;
        }
        if (pluginChain) {
          // Splice the live plugin chain between the per-track sink and
          // the bus. `pluginChain.input/output` are GainNodes the adapter
          // owns; dispose() on the chain disconnects them and tears the
          // engines down.
          busFeed.connect(pluginChain.input);
          pluginChain.output.connect(busGainNode);
        } else {
          busFeed.connect(busGainNode);
        }
        gainNodesRef.current.set(track.id, trackGain);

        if (hasVolAuto) {
          const r = scheduleAutomation(trackGain.gain, volLane, {
            audioContextCurrentTime: audioContext.currentTime,
            schedulingStartTime,
            anchorTimelineTime: currentPlayheadTime,
            fallbackValue: trackLocalGain,
          });
          console.log(`  📈 vol-auto ${track.name || track.id}: ${r.scheduledCount} pts ahead, anchor=${r.paramAtAnchor.toFixed(3)}`);
        }
        if (trackPan && hasPanAuto) {
          const r = scheduleAutomation(trackPan.pan, panLane, {
            audioContextCurrentTime: audioContext.currentTime,
            schedulingStartTime,
            anchorTimelineTime: currentPlayheadTime,
            fallbackValue: staticPan,
          });
          console.log(`  📈 pan-auto ${track.name || track.id}: ${r.scheduledCount} pts ahead, anchor=${r.paramAtAnchor.toFixed(3)}`);
        }

        // ── Aux sends ───────────────────────────────────────────────────
        // Each enabled send taps trackGain (postfader) and feeds another
        // bus's GainNode at the send's level. Multi-output: trackGain
        // connects to its primary bus (above) AND to one sendGain per
        // active send. Mute/solo at the SOURCE track propagates because
        // the tap is post-trackGain — silencing the track silences sends.
        //
        // Bus resolution: Logic ships `bus_sub_index` (1-based, by mixer
        // slot). We map it to allBusIds[bus_sub_index - 1] using the bus
        // arrival order from this play() pass. When the index doesn't
        // resolve to a known bus (e.g. send to a Logic bus the web hasn't
        // seen as a track's _busId yet), we log and skip.
        const sendDescriptors = Array.isArray(track.metadata?.sends) ? track.metadata.sends : [];
        const trackSendNodes = [];
        for (const send of sendDescriptors) {
          if (!send || !send.enabled) continue;
          const level = Number(send.level);
          if (!Number.isFinite(level) || level <= 0) continue;
          const idx = Number(send.bus_sub_index);
          let destBusId = null;
          if (typeof send.busId === 'string' && send.busId) {
            destBusId = send.busId;
          } else if (Number.isFinite(idx) && idx > 0 && idx <= allBusIds.length) {
            destBusId = allBusIds[idx - 1];
          }
          if (!destBusId) {
            console.warn(
              `[send] ${track.name || track.id} slot ${send.slot}: ` +
              `bus_sub_index ${idx} unresolved (have ${allBusIds.length} buses); skipping`,
            );
            continue;
          }
          const destNode = ensureBusGain(destBusId);
          if (!destNode) continue;
          const sendGain = audioContext.createGain();
          sendGain.gain.value = level;
          trackGain.connect(sendGain);
          sendGain.connect(destNode);
          trackSendNodes.push({ slot: send.slot, busId: destBusId, gainNode: sendGain });
        }
        if (trackSendNodes.length > 0) {
          sendGainNodesRef.current.set(track.id, trackSendNodes);
          const summary = trackSendNodes.map((s) => `${s.busId}@${s.gainNode.gain.value.toFixed(2)}`).join(', ');
          console.log(`  📤 sends ${track.name || track.id}: ${summary}`);
        }

        // Logic flex-time region shift (Slicing algorithm). Applied to
        // the track's wall-clock schedule, not its UI startPosition —
        // the clip stays visually on-grid, only audio timing moves.
        const flexOffset = Number(track.metadata?.flexTime?.offsetSeconds) || 0;
        if (flexOffset !== 0) {
          const tCount = track.metadata?.flexTime?.transients?.length || 0;
          console.log(
            `  ⏱ flex ${track.name || track.id}: ${(flexOffset * 1000).toFixed(2)}ms ` +
            `(${tCount} transient${tCount === 1 ? '' : 's'})`,
          );
        }

        // ── Drum substem fan-out ───────────────────────────────────────
        const substems = getTrackSubstemSchedules(track, projectMeter);
        if (substems) {
          // getTrackSubstemSchedules already logs the per-substem schedule
          // breakdown (kind + segment counts). Mark the fan-out here so
          // the order is clear when reading the console: substem log →
          // fan-out log → late-scheduled confirmations per substem.
          const cachedCount = Object.values(substems)
            .filter(({ audioUrl }) => audioBufferCache.has(audioUrl)).length;
          const total = Object.keys(substems).length;
          console.log(`🥁 fan-out ${track.name || track.id}: ${total} drum substems (${cachedCount} cached, ${total - cachedCount} late-loading)`);
          // The shared per-track gain (and optional pan/automation) was
          // created and registered at the top of scheduleOrDefer; substem
          // sources just sum into it. Don't create a second one.
          for (const [name, { audioUrl, schedule, kind }] of Object.entries(substems)) {
            const scheduleSubstem = (when, anchor) => {
              const { sources } = scheduleTrackWithSchedule(
                audioContext, audioUrl, schedule, trackLocalGain, trackStartTime,
                when, anchor, busGainNode,
                trackGain,    // <- share the parent's gain node
                flexOffset,
              );
              sourceNodesRef.current.push(...sources);
              return sources.length;
            };
            if (audioBufferCache.has(audioUrl)) {
              scheduleSubstem(currentPlayheadTime, schedulingStartTime);
              continue;
            }
            // Late-load: fetch + decode in parallel with other substems,
            // then schedule when ready (if still playing).
            (async () => {
              try {
                await ensureBuffer(audioUrl);
                if (!isSessionActive()) {
                  // Session advanced (seek or pause) while we were fetching.
                  // Buffer is now cached — next play() will pick it up.
                  console.log(`  🥁 ${name} late-load: session stale, cached for next play`);
                  return;
                }
                const livePlayhead = audioContext.currentTime - startTimeRef.current;
                const n = scheduleSubstem(livePlayhead, audioContext.currentTime);
                console.log(`  🥁 Late-scheduled drum substem ${name} (${kind}, ${n} seg)`);
              } catch (e) {
                console.warn(`  ❌ Substem ${name} late-load failed:`, e?.message || e);
              }
            })();
          }
          return;
        }

        // ── Per-clip path (Logic regions with source offsets + stretch) ──
        // Each clip is an independent AudioBufferSourceNode reading its own
        // slice of either the pre-rendered variant wav (rate=1) or the
        // original root wav (playbackRate=stretchRatio). One trackGain is
        // shared across all clips so mute/solo/fader still operate on the
        // track as a whole.
        //
        // Gate: per-clip only when it adds value (multi-clip, flex-variant
        // stretch, or a non-trivial source offset). Single-clip tracks
        // with a zero offset fall through to the single-buffer path, which
        // keeps its meter-change + vocal/bass schedule strategies intact.
        // Meter-change rearrangement isn't implemented per-clip yet; a
        // future step can port virtualTrackEdit into this branch.
        const clipList = track.metadata?.clips;
        const clipPathUseful = Array.isArray(clipList) && clipList.length > 0 && (
          clipList.length > 1 ||
          clipList.some((c) => c && c.rootUrl && Number(c.stretchRatio) > 0 && c.stretchRatio !== 1) ||
          clipList.some((c) => c && Math.abs(Number(c.sourceOffsetSeconds) || 0) > 1e-4)
        );
        if (clipPathUseful) {
          const schedClips = () => {
            const { sources } = scheduleTrackClips(
              audioContext, clipList, trackLocalGain,
              currentPlayheadTime, schedulingStartTime, busGainNode,
              trackGain,   // reuse the per-track sink (gain + automation)
            );
            if (sources.length > 0) sourceNodesRef.current.push(...sources);
            return sources.length;
          };

          // Fast path: if every buffer we need is already in cache, schedule
          // synchronously (same as scheduleTrackWithSchedule's cached path).
          const needsLate = clipList.some((c) => {
            if (!c) return false;
            const primary = c.rootUrl || c.url;
            return primary && !audioBufferCache.has(primary);
          });
          if (!needsLate) {
            schedClips();
            return;
          }

          // Slow path: one or more clip buffers haven't arrived yet. Kick
          // off fetches in parallel, then schedule once everything's in.
          const urlsToLoad = new Set();
          for (const c of clipList) {
            if (!c) continue;
            if (c.url && !audioBufferCache.has(c.url)) urlsToLoad.add(c.url);
            if (c.rootUrl && !audioBufferCache.has(c.rootUrl)) urlsToLoad.add(c.rootUrl);
          }
          (async () => {
            try {
              await Promise.all([...urlsToLoad].map((u) => ensureBuffer(u)));
              if (!isSessionActive()) {
                console.log(`  🎬 ${track.name || track.id} clip late-load: session stale`);
                return;
              }
              const livePlayhead = audioContext.currentTime - startTimeRef.current;
              const n = schedClips();
              console.log(`  🎬 Late-scheduled ${n} clip source(s) for ${track.name || track.id}`);
            } catch (err) {
              console.warn(`  ❌ Clip late-load failed for ${track.name || track.id}:`, err?.message || err);
            }
          })();
          return;
        }

        // ── Regular single-buffer path ────────────────────────────────
        // Dispatcher picks per-stem-type strategy: vocals get word-
        // protected schedule, bass gets melodic-protected, others fall
        // through to bar rearrange.
        const schedule = dispatchStrategy(track, projectMeter);
        // Visibility: only log when the meter actually changed. A
        // tempoMap alone can produce multi-segment schedules at the
        // same meter (identity-stretch); those aren't meter changes
        // and were spamming the console on every play/seek.
        const srcN = track.metadata?.detectedMeter;
        const srcD = track.metadata?.detectedMeterDenominator;
        const tgtN = projectMeter.beatsPerBar;
        const tgtD = projectMeter.meterDenominator;
        const meterChanged = srcN && srcD && (srcN !== tgtN || srcD !== tgtD);
        if (schedule && schedule.length > 1 && meterChanged) {
          console.log(`🎚️ [meter] ${track.name || track.id}: ${srcN}/${srcD} → ${tgtN}/${tgtD} — ${schedule.length} segments`);
          // Diagnostic: if a drum track is meter-changing but using the
          // bar-rearrange path, the drumSubstems metadata hasn't landed
          // (backend _run_drum_teacher didn't finish yet, or MDX23C-DrumSep
          // unavailable). Make that state loud so it's obvious why the
          // per-substem snap path didn't fire.
          const stemType = (track.metadata?.stemType || track.metadata?.instrument || '').toLowerCase();
          if (stemType === 'drums' || stemType === 'drum_kit' || stemType === 'percussion') {
            console.warn(`  ⚠️ drums meter-change on bar-rearrange path — no drumSubstems metadata yet (backend MDX23C-DrumSep still running or unavailable)`);
          }
        }
        if (audioBufferCache.has(url)) {
          const { sources } = scheduleTrackWithSchedule(
            audioContext, url, schedule, trackLocalGain, trackStartTime,
            currentPlayheadTime, schedulingStartTime, busGainNode,
            trackGain, flexOffset,
          );
          if (track.metadata?.polypitchRendered) {
            const buf = audioBufferCache.get(url);
            const ch = buf.getChannelData(0);
            let sum = 0, peak = 0;
            const N = Math.min(ch.length, 96000); // sample first 2s at 48k
            for (let i = 0; i < N; i++) { const v = ch[i]; sum += v*v; const a = Math.abs(v); if (a > peak) peak = a; }
            const rms = Math.sqrt(sum / Math.max(1, N));
            console.log(`  🎹 scheduled polypitch ${track.metadata?.stemType || '?'} dur=${buf.duration.toFixed(2)}s rms(2s)=${rms.toFixed(5)} peak(2s)=${peak.toFixed(4)} gain(track)=${trackLocalGain.toFixed(2)} segs=${sources.length} from=${url.slice(0, 40)}…`);
            // Post-decode click scan: measure sample discontinuities on the
            // decoded AudioBuffer (after WAV→AudioBuffer conversion by
            // audioContext.decodeAudioData). This confirms the blob content
            // at the point it's handed to the graph. A clean algorithm output
            // should decode to a clean AudioBuffer — any click here would be
            // from the WAV encoder or decoder.
            let calmMax = 0;
            for (let i = 1; i < N; i++) {
              const d = Math.abs(ch[i] - ch[i - 1]);
              if (d > calmMax) calmMax = d;
            }
            const clickThresh = Math.max(calmMax * 6, 0.1);
            let clickCount = 0;
            let firstClickT = -1;
            const scanLen = ch.length;
            for (let i = 1; i < scanLen; i++) {
              const d = Math.abs(ch[i] - ch[i - 1]);
              if (d > clickThresh) {
                clickCount++;
                if (firstClickT < 0) firstClickT = i / buf.sampleRate;
              }
            }
            if (clickCount > 0) {
              console.warn(`  🎹 post-decode click scan: ${clickCount} click(s) in AudioBuffer, first @ ${firstClickT.toFixed(3)}s (calmMax=${calmMax.toExponential(2)} threshold=${clickThresh.toExponential(2)})`);
            } else {
              console.log(`  🎹 post-decode click scan: clean (calmMax=${calmMax.toExponential(2)})`);
            }
          }
          sourceNodesRef.current.push(...sources);
          return;
        }
        // Not cached — fire async, schedule on arrival if still playing.
        // Closure-capture trackGain so the late-scheduled sources sum into
        // the same per-track sink (already wired to bus + automation).
        (async () => {
          try {
            const audioBuffer = await ensureBuffer(url);
            if (!isSessionActive()) return;
            const livePlayhead = audioContext.currentTime - startTimeRef.current;
            const trackDuration = track.duration || audioBuffer.duration;
            if (livePlayhead > trackStartTime + trackDuration) return;  // already past
            const { sources } = scheduleTrackWithSchedule(
              audioContext, url, schedule, trackLocalGain, trackStartTime,
              livePlayhead, audioContext.currentTime, busGainNode,
              trackGain, flexOffset,
            );
            if (sources.length > 0) {
              sourceNodesRef.current.push(...sources);
              const tag = track.metadata?.polypitchRendered ? '🎹 Late-scheduled polypitch' : '🔈 Late-scheduled';
              console.log(`  ${tag}: ${track.metadata?.stemType || track.name || url} at live playhead ${livePlayhead.toFixed(2)}s (${sources.length} seg(s))`);
            }
          } catch (error) {
            console.warn(`  ❌ Late-load failed for ${url}:`, error?.message || error);
          }
        })();
      };

      const hasPolypitchStem = allTracks.some(
        (t) => t.metadata?.type === 'stem' && t.metadata?.polypitchRendered
      );

      // ── Mask-based playback for stem tracks ──────────────────────
      // If mask playback is ready, stem tracks play through the worklet
      // (master audio + spectral masks) instead of individual decoded buffers.
      // This gives master-quality audio with no decoder overhead.
      //
      // IMPORTANT: when ANY stem has been re-rendered by polypitch, disable
      // mask playback for the whole pass and route everything through the
      // regular audioUrl scheduler. Even with per-stem gains at 0, the worklet
      // can still color or leak the original content enough to mask the
      // edited stem. For polypitch correctness, plain decoded-buffer playback
      // wins over master-mask fidelity.
      const useMaskPlayback = isMaskPlaybackReady() && !hasPolypitchStem;
      if (!useMaskPlayback && isMaskPlaybackReady() && hasPolypitchStem) {
        console.log('🎭 Mask playback bypassed: polypitch-rendered stem present; scheduling all tracks from audioUrl');
        maskStop();
      }
      if (useMaskPlayback) {
        // Collect stem track gains and solo/mute state.
        // Stems that polypitch has re-rendered carry metadata.polypitchRendered:
        // route them through the regular audioUrl scheduling path (so their
        // new blob WAV is actually played), AND force their mask gain to 0
        // so the worklet doesn't ALSO emit the original version of that stem
        // from the master × mask path. Net: (master × non-edited-stem masks)
        // from the worklet + polypitch blob from the regular graph = edited mix.
        const stemGains = {};
        const maskStemTracks = allTracks.filter(t =>
          t.metadata?.type === 'stem' && !t.metadata?.polypitchRendered
        );
        const polypitchStemTracks = allTracks.filter(t =>
          t.metadata?.type === 'stem' && t.metadata?.polypitchRendered
        );
        const nonStemTracks = allTracks.filter(t => t.metadata?.type !== 'stem');
        const hasStemSolo = [...maskStemTracks, ...polypitchStemTracks].some(t => t.isSolo);

        let activeSolo = null;
        for (const t of maskStemTracks) {
          const stemName = normalizeStemForMaskPlayback(t);
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

        // Polypitch-edited stems: 0 gain in the mask path (they'll play via
        // the regular scheduler below from their new audioUrl).
        for (const t of polypitchStemTracks) {
          const stemName = normalizeStemForMaskPlayback(t);
          if (stemName) stemGains[stemName] = 0;
        }

        // Send gains to worklet — instant, no re-render
        setMaskGains(stemGains);
        setActiveStem(activeSolo);
        maskSeek(currentPlayheadTime);
        maskPlay();
        console.log(`🎭 Mask playback: ${Object.keys(stemGains).length} stems`
          + (polypitchStemTracks.length ? ` (${polypitchStemTracks.length} routed to audioUrl)` : '')
          + `, solo=${activeSolo || 'none'}`);
        if (polypitchStemTracks.length > 0) {
          for (const t of polypitchStemTracks) {
            const url = (t.audioUrl || '').slice(0, 60);
            const cached = audioBufferCache.has(t.audioUrl);
            const stem = t.metadata?.stemType || t.metadata?.instrument || '?';
            const maskStem = normalizeStemForMaskPlayback(t) || '?';
            console.log(`  🎹 polypitch stem: id=${t.id} stem=${stem} maskStem=${maskStem} gain(mask)=0 url=${url}… cached=${cached}`);
          }
        }

        // Schedule non-stem tracks + polypitch-rendered stems via audioUrl.
        // (Parent track is muted when stems exist, so it won't double-play.)
        var tracksToSchedule = [...nonStemTracks, ...polypitchStemTracks];
      } else {
        var tracksToSchedule = allTracks;
      }

      // Load and schedule tracks (non-stem when mask playback active, all otherwise).
      // Bus-level gating (mute/solo and bus.gain) now lives on a per-bus
      // GainNode (ensureBusGain). Per-track gating is on the track's
      // GainNode. We still decide whether to spin up source nodes at all
      // based on the combined shouldPlay — scheduling a 100%-muted track
      // wastes decode/buffer work for zero signal.
      const hasBusSolo = allTracks.some(t => t._busSolo);

      // Prime each bus node's initial gain synchronously BEFORE any source
      // starts. The real-time gain-update effect also applies it (via
      // setTargetAtTime), but effects run after React commits, which races
      // source.start() and can leak a few ms of 1.0-gain audio on a muted
      // bus. Write .value directly here so the first sample is silent.
      const seenBuses = new Set();
      for (const t of allTracks) {
        if (!t._busId || seenBuses.has(t._busId)) continue;
        seenBuses.add(t._busId);
        const node = ensureBusGain(t._busId);
        const bGain = typeof t._busGain === 'number' ? t._busGain : 1.0;
        let g;
        if (t._busMuted)                     g = 0;
        else if (hasBusSolo && !t._busSolo)  g = 0;
        else                                 g = bGain;
        node.gain.value = g;
      }

      // R11 splice — pre-resolve PluginAdapter chains in parallel before
      // the schedule loop. Building each chain is async (mapping fetch +
      // engine instantiation), so resolving sequentially inside the loop
      // would serialize play(). Resolved chains are stashed in
      // `prebuiltChains` keyed by track.id; scheduleOrDefer reads from the
      // map without awaiting. Chains for tracks that lack mappings (or
      // when the feature flag is off) resolve to null and the bounce-cache
      // path takes over for that track. Sessionguard: if play() got
      // cancelled while we were resolving, dispose anything we built and
      // bail before scheduling.
      const prebuiltChains = new Map();
      if (LIVE_PLUGINS_ENABLED) {
        const chainCandidates = tracksToSchedule.filter(
          (t) => Array.isArray(t?.logicPlugins) && t.logicPlugins.length > 0,
        );
        if (chainCandidates.length > 0) {
          const built = await Promise.all(
            chainCandidates.map(async (t) => {
              const chain = await maybeBuildPluginChain(t);
              return [t.id, chain];
            }),
          );
          if (!isSessionActive()) {
            // Session cancelled during resolution — dispose and bail.
            for (const [, chain] of built) {
              if (chain) { try { chain.dispose && chain.dispose(); } catch (_) {} }
            }
            return;
          }
          // Build a quick map so we can register against the originating
          // track record (the registry needs trackIndex / ginstid, which
          // live on the synced track object — not on the chain itself).
          const candidatesById = new Map(chainCandidates.map((t) => [t.id, t]));
          for (const [trackId, chain] of built) {
            if (chain) {
              prebuiltChains.set(trackId, chain);
              trackChainsRef.current.set(trackId, chain);
              _registerChain(candidatesById.get(trackId), chain);
            }
          }
          if (prebuiltChains.size > 0) {
            console.log(`🎛 [R11] live plugin chains active for ${prebuiltChains.size}/${chainCandidates.length} candidate track(s)`);
          }
        }
      }

      for (const track of tracksToSchedule) {
        const busMuted = track._busMuted || false;
        const busSolo  = track._busSolo  || false;

        // shouldPlay — same precedence as before: bus mute blocks the whole
        // bus; bus-solo overrides per-track solo/mute; track solo/mute apply
        // only when no bus is solo'd.
        let shouldPlay;
        if (busMuted) {
          shouldPlay = false;
        } else if (hasBusSolo) {
          shouldPlay = busSolo;
        } else if (hasSoloTracks) {
          shouldPlay = !!track.isSolo;
        } else {
          shouldPlay = !track.isMuted;
        }

        // Track-local gain only; bus.gain lives on the bus node.
        const trackLocalGain = track.gain || 1.0;
        const busGainNode = ensureBusGain(track._busId);

        if (shouldPlay) {
          // R11 — pre-resolved live plugin chain (or null when feature flag
          // is off / no mappings / fallback). Passed into helper paths that
          // build their own per-source gain (drum hits, F0). For
          // scheduleOrDefer-routed tracks the chain is wired at the
          // shared trackGain layer (see scheduleOrDefer).
          const trackPluginChain = prebuiltChains.get(track.id) || null;

          // Check if this is a drum track with multiple hits
          if (track.isDrumTrack && track.drumHits && track.drumHits.length > 0) {
            console.log(`  🥁 Scheduling drum track with ${track.drumHits.length} hits`);

            // Schedule each drum hit individually
            for (const hit of track.drumHits) {
              const hitTime = hit.startTime || 0;
              const hitGain = (hit.velocity || 1.0) * trackLocalGain;

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
                  busGainNode, // per-bus gain, feeds into master
                  0,           // flexOffsetSeconds (default)
                  trackPluginChain, // R11 splice
                );

                if (source && gainNode) {
                  sourceNodesRef.current.push(source);
                  // Store gain node for drum hit (use unique ID)
                  gainNodesRef.current.set(`${track.id}-hit-${hitTime}`, gainNode);
                }
              }
            }
          } else if (track.audioUrl) {
            // Regular track (non-drum). scheduleOrDefer handles both
            // cached-immediately and arrive-later buffers — neither path
            // blocks the play-click.
            const trackStartTime = track.startPosition || 0;
            const trackDuration = track.duration || track.length || 10;
            const trackEndTime = trackStartTime + trackDuration;
            if (currentPlayheadTime <= trackEndTime) {
              scheduleOrDefer(track, trackLocalGain);
            } else {
              console.log(`  ⏭ Skipping track (playhead already past): ${track.name || track.audioUrl}`);
            }
          } else if (isMidiTypeTrack(track)) {
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
                trackLocalGain,
                trackStartTime,
                0, // No crop for F0
                currentPlayheadTime,
                schedulingStartTime,
                busGainNode,
                0,                  // flexOffsetSeconds (default)
                trackPluginChain,   // R11 splice
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
                // Track-local velocity only; bus.gain is applied by the
                // bus GainNode we pass as opts.destination so the MIDI
                // track respects bus mute/solo/fader exactly like audio.
                midiPlayer.playNote(
                  note.note,
                  note.velocity * trackLocalGain,
                  note.duration,
                  notePlayTime,
                  { destination: busGainNode }
                );
              });
            } else {
              console.log(`  ⏭ Skipping MIDI track (playhead already past): ${track.name}`);
            }
          }
        }
      }

      // startTimeRef + hasStartedPlaybackRef + updatePlayhead() were set
      // above, before any async work, so the timeline has already been
      // advancing while we iterated tracks.
    } catch (error) {
      console.error('❌ Error starting playback:', error);
      dispatch({ type: 'SET_PLAYING', payload: false });
    }
  }, [updatePlayhead, dispatch, bpm, beatsPerBar, meterDenominator]); // Include bpm + meter so next play rebuilds schedules

  // Keep a ref to the latest play() fn so the meter-change effect below
  // can re-invoke it without bloating its own dep list.
  useEffect(() => { playRef.current = play; }, [play]);
  // seekRef is read by the cycle-loop wrap inside updatePlayhead. Same
  // forward-declaration trick as playRef — keeps useCallback deps from
  // exploding while letting the RAF loop reach the latest seek().

  // Live reschedule on bpm / meter change while playing. Schedules are
  // derived from (project bpm, project meter, track metadata), so when the
  // user flips meter mid-playback we tear down in-flight sources and
  // start over from the current playhead. Instant, pure rearrange, no
  // backend. ~5ms dropout at the cut.
  const lastMeterRef = useRef({ bpm, beatsPerBar, meterDenominator });
  useEffect(() => {
    const prev = lastMeterRef.current;
    const changed = prev.bpm !== bpm || prev.beatsPerBar !== beatsPerBar || prev.meterDenominator !== meterDenominator;
    lastMeterRef.current = { bpm, beatsPerBar, meterDenominator };
    if (!changed) return;
    if (!isPlayingRef.current || !audioContextRef.current || !hasStartedPlaybackRef.current) return;

    const t = audioContextRef.current.currentTime - startTimeRef.current;
    if (t >= 0 && t <= totalDuration + 1) {
      pauseTimeRef.current = t;
    }
    playbackSessionRef.current += 1;
    sourceNodesRef.current.forEach((s) => { try { s.stop(); } catch (_) {} });
    sourceNodesRef.current = [];
    gainNodesRef.current.clear();
    hasStartedPlaybackRef.current = false;

    const fn = playRef.current;
    if (fn) {
      console.log(`🎚️ live reschedule: bpm=${bpm}, meter=${beatsPerBar}/${meterDenominator} at t=${pauseTimeRef.current.toFixed(2)}s`);
      Promise.resolve().then(() => { if (isPlayingRef.current) fn(); });
    }
  }, [bpm, beatsPerBar, meterDenominator, totalDuration]);

  /**
   * Pause playback
   * @param {boolean} storePosition - Whether to store current position (default true)
   */
  const pause = useCallback((storePosition = true) => {
    playbackSessionRef.current += 1;
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

  // Bind seekRef so updatePlayhead's cycle-loop branch can call the latest
  // seek without taking it as a useCallback dep (would re-create the RAF
  // loop every render). Same forward-decl pattern as playRef.
  useEffect(() => { seekRef.current = seek; }, [seek]);

  /**
   * Handle play/pause based on isPlaying prop
   */
  useEffect(() => {
    // Don't interfere if we're currently seeking
    if (isSeekingRef.current) return;

    if (isPlaying) {
      // Only start playback if not already playing
      if (!hasStartedPlaybackRef.current) {
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
      playbackSessionRef.current += 1;
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
      // R11 — dispose any live plugin chains. The audio-context useEffect's
      // cleanup also covers this (chains hold ctx-bound nodes), but
      // belt-and-suspenders here avoids relying on cleanup ordering.
      if (trackChainsRef.current.size > 0) {
        for (const chain of trackChainsRef.current.values()) {
          try { chain.dispose && chain.dispose(); } catch (_) {}
        }
        trackChainsRef.current.clear();
        _clearRegistry();
      }
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
 * @param {number} gain - Track-local gain (track.gain × track-level gate).
 *   Bus.gain and bus mute/solo live on `outputNode` (the bus GainNode).
 * @param {number} trackStartTime - When track starts on timeline (seconds)
 * @param {number} trackCropStart - Cropped portion at start of track (seconds)
 * @param {number} currentPlayheadTime - Current playhead position on timeline (seconds)
 * @param {number} schedulingStartTime - AudioContext.currentTime at start of scheduling (seconds)
 * @param {AudioNode} outputNode - Where this track sums into. Typically the
 *   per-bus GainNode returned by ensureBusGain(track._busId). Falls back
 *   to masterGainNode for trackless test paths.
 * @param {number} [flexOffsetSeconds=0] - Logic-style flex-time shift on the
 *   region: negative pulls content earlier, positive pushes later. The UI
 *   position (trackStartTime) is untouched — only scheduling moves.
 */
function scheduleTrack(
  audioContext,
  audioUrl,
  gain,
  trackStartTime,
  trackCropStart,
  currentPlayheadTime,
  schedulingStartTime,
  outputNode,
  flexOffsetSeconds = 0,
  pluginChain = null,
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

    source.connect(gainNode);
    // Signal chain: source → trackGain → outputNode (bus gain) → master → destination
    // (FX chain on outputNode is a future hook; per-bus inserts go here.)
    // R11 splice: when a live plugin chain is supplied, splice between
    // trackGain and outputNode. Default (pluginChain=null) is byte-identical
    // to today's bounce-cache path.
    if (pluginChain) {
      gainNode.connect(pluginChain.input);
      pluginChain.output.connect(outputNode);
    } else {
      gainNode.connect(outputNode);
    }

    // Calculate when to start playback in AudioContext time
    // If playhead is before track start, schedule for future
    // If playhead is after track start, start immediately but offset into the audio
    // Use schedulingStartTime as reference to ensure consistency

    // Flex-time: the region's playback clock is nominal minus flexOffset.
    // A -16ms flex makes the audio play 16ms earlier than its UI position.
    const effectiveStart = trackStartTime + (flexOffsetSeconds || 0);

    let when = 0; // When to start in AudioContext time
    let offset = trackCropStart; // Where to start within the audio buffer

    if (currentPlayheadTime < effectiveStart) {
      // Playhead hasn't reached this track yet - schedule for future
      const delayUntilTrackStart = effectiveStart - currentPlayheadTime;
      when = schedulingStartTime + delayUntilTrackStart;
      offset = trackCropStart; // Start from beginning (plus crop)
    } else {
      // Playhead is already past track start - start immediately but offset
      when = schedulingStartTime;
      const timeIntoTrack = currentPlayheadTime - effectiveStart;
      offset = trackCropStart + timeIntoTrack; // Skip ahead to current position

      // Don't play if we're past the end of the audio
      if (offset >= audioBuffer.duration) {
        console.log('    ⏭ Track already finished');
        return { source: null, gainNode: null };
      }
    }

    console.log(`    🎵 Starting at AudioContext time: ${when.toFixed(2)}s, offset: ${offset.toFixed(2)}s`);

    const immediateStart = when <= audioContext.currentTime + 1e-4;
    const attack = immediateStart ? Math.min(RESUME_ATTACK_SECONDS, 0.02) : 0;
    if (attack > 0) {
      gainNode.gain.setValueAtTime(0, when);
      gainNode.gain.linearRampToValueAtTime(gain, when + attack);
    } else {
      gainNode.gain.setValueAtTime(gain, when);
    }

    source.start(when, offset);

    return { source, gainNode };
  } catch (error) {
    console.error('❌ Error scheduling track:', audioUrl, error);
    return { source: null, gainNode: null };
  }
}

/**
 * Schedule-aware playback for a regular audio track.
 *
 * The schedule is an array of Segments (services/virtualTrackEdit.js).
 * Identity schedules collapse to a single full-buffer Segment so this
 * path is safe for every track — no separate "simple" codepath.
 *
 * Signal chain per track:
 *
 *   [ seg0 source → seg0 fadeGain ─┐
 *   [ seg1 source → seg1 fadeGain ─┼── trackGain ── outputNode (bus gain) ── master
 *   [ segN source → segN fadeGain ─┘
 *
 * Most segments play at rate 1 (rearrange-only schedules). When a
 * `stretch` segment appears (rate !== 1, e.g. tempo change or future
 * non-meter stretch features), the engine pre-renders the slice via
 * WSOLA (services/wsolaStretch.js) into a new AudioBuffer at the target
 * duration and plays THAT at rate 1. Pitch is preserved — we never set
 * AudioBufferSourceNode.playbackRate to anything other than 1.
 *
 * source.start(when, offset, srcDuration):
 *   - `when`         AudioContext time to start this segment
 *   - `offset`       offset into the (possibly pre-stretched) buffer
 *   - `srcDuration`  buffer content to play; for rate=1 segments equals
 *                    dst duration; for rate≠1 the buffer IS the dst
 *                    content already, so offset/duration are dst-space.
 *
 * Returns { sources: [...], gainNode }. Callers push every element of
 * `sources` into sourceNodesRef so pause()/stop() tears them all down.
 *
 * `gain` is track-local (track.gain × track-level gate); bus.gain and
 * bus-level mute/solo live on `outputNode`. `outputNode` is the per-bus
 * GainNode from ensureBusGain(track._busId).
 */
function scheduleTrackWithSchedule(
  audioContext, audioUrl, schedule, gain, trackStartTime,
  currentPlayheadTime, schedulingStartTime, outputNode,
  existingTrackGain = null,
  flexOffsetSeconds = 0,
  pluginChain = null,
) {
  try {
    const audioBuffer = audioBufferCache.get(audioUrl);
    if (!audioBuffer) {
      console.warn(`⚠️ Audio buffer not cached yet: ${audioUrl}`);
      return { sources: [], gainNode: existingTrackGain };
    }
    if (!schedule || schedule.length === 0) {
      return { sources: [], gainNode: existingTrackGain };
    }

    // Reuse the caller's track gain if provided (substem fan-out path —
    // every substem sums into one per-track gain so solo/mute on the
    // parent still works). Otherwise create a fresh one.
    // R11 splice: when a fresh trackGain is created and a live plugin chain
    // is supplied, route the trackGain through the chain before reaching
    // outputNode. When existingTrackGain is provided, the caller
    // (scheduleOrDefer) already wired the chain at the trackGain layer.
    const trackGain = existingTrackGain || (() => {
      const g = audioContext.createGain();
      g.gain.value = gain;
      if (pluginChain) {
        g.connect(pluginChain.input);
        pluginChain.output.connect(outputNode);
      } else {
        g.connect(outputNode);
      }
      return g;
    })();

    // Flex-time shifts the whole schedule's anchor on the timeline. The
    // schedule's internal dst times stay in clip-local coordinates; only
    // the wall-clock placement moves. UI position (trackStartTime) is
    // untouched — this is a playback-only shift, same as Logic.
    const effectiveStart = trackStartTime + (flexOffsetSeconds || 0);

    // Clip-local playhead (0 == track start on the timeline).
    const clipPlayhead = currentPlayheadTime - effectiveStart;
    const live = resumeFromPlayhead(schedule, Math.max(0, clipPlayhead));

    const sources = [];
    for (const entry of live) {
      const { seg, dstOffsetIntoSeg, srcOffsetIntoSeg, srcDuration } = entry;

      // When does this segment start in AudioContext wall-clock time?
      //   - Already in flight (dstOffsetIntoSeg > 0): start now.
      //   - Future: trackStart + seg.dstStart relative to the wall-clock
      //     anchor (schedulingStartTime - currentPlayheadTime).
      let when;
      if (dstOffsetIntoSeg > 0) {
        when = schedulingStartTime;
      } else {
        when = schedulingStartTime + (effectiveStart + seg.dstStart - currentPlayheadTime);
      }
      if (when < schedulingStartTime - 1e-3) continue;

      // ── Pick playback buffer + offset/duration based on seg.rate ──
      // rate === 1: slice the original audioBuffer at srcStart+offset.
      // rate !== 1: pre-render the slice via WSOLA into a new buffer
      //   at the target dst length (pitch preserved). The pre-rendered
      //   buffer plays at rate 1 — NEVER use playbackRate, which would
      //   pitch-shift. Cached per (url, srcStart, srcEnd, ratio).
      let bufferToPlay, offset, srcPlayable;
      const dstDurFromHere = (seg.dstEnd - seg.dstStart) - dstOffsetIntoSeg;
      if (seg.rate === 1) {
        bufferToPlay = audioBuffer;
        offset = seg.srcStart + srcOffsetIntoSeg;
        if (offset >= audioBuffer.duration) continue;
        srcPlayable = Math.min(srcDuration, audioBuffer.duration - offset);
      } else {
        // ratio = output_length / input_length = 1 / seg.rate.
        //   seg.rate > 1 (compress): ratio < 1, output shorter than input
        //   seg.rate < 1 (stretch):  ratio > 1, output longer than input
        const ratio = 1 / seg.rate;
        bufferToPlay = getStretchedBuffer(
          audioContext, audioBuffer, audioUrl,
          seg.srcStart, seg.srcEnd, ratio,
        );
        // The stretched buffer's full length corresponds to the segment's
        // dst window. Mid-segment resume offsets directly into dst space.
        offset = dstOffsetIntoSeg;
        if (offset >= bufferToPlay.duration) continue;
        srcPlayable = Math.min(dstDurFromHere, bufferToPlay.duration - offset);
      }
      if (srcPlayable <= 0) continue;

      const src = audioContext.createBufferSource();
      src.buffer = bufferToPlay;
      // playbackRate stays at 1 — pitch preservation. WSOLA already
      // bent the time axis; the source just plays the result back.

      const segGain = audioContext.createGain();
      const immediateStart = Math.abs(when - schedulingStartTime) < 1e-4;
      const attackFade = immediateStart
        ? Math.min(RESUME_ATTACK_SECONDS, dstDurFromHere * 0.5)
        : 0;
      const fadeInBase = dstOffsetIntoSeg > 0 ? 0 : (seg.fadeIn || 0);
      const fadeIn  = Math.min(Math.max(fadeInBase, attackFade), dstDurFromHere * 0.5);
      const fadeOut = Math.min(seg.fadeOut || 0, dstDurFromHere * 0.5);
      const segEnd = when + dstDurFromHere;
      if (fadeIn > 0) {
        segGain.gain.setValueAtTime(0, when);
        segGain.gain.linearRampToValueAtTime(1, when + fadeIn);
      } else {
        segGain.gain.setValueAtTime(1, when);
      }
      if (fadeOut > 0) {
        segGain.gain.setValueAtTime(1, segEnd - fadeOut);
        segGain.gain.linearRampToValueAtTime(0, segEnd);
      }

      src.connect(segGain);
      segGain.connect(trackGain);

      try {
        src.start(when, offset, srcPlayable);
      } catch (err) {
        console.warn('  ⚠️ segment start failed:', err?.message || err);
        continue;
      }
      sources.push(src);
    }

    return { sources, gainNode: trackGain };
  } catch (error) {
    console.error('❌ Error scheduling track with schedule:', audioUrl, error);
    return { sources: [], gainNode: null };
  }
}

/**
 * Per-clip scheduler — Logic's real model.
 *
 * Each Clip has:
 *   auflGid, subIndex, startPosition, duration,
 *   sourceDuration, sourceOffsetSamples, sourceOffsetSeconds,
 *   url,                     // bit-exact variant wav (always present)
 *   rootAuflGid?, rootUrl?,  // only on flex variants
 *   stretchRatio?,           // feed to source.playbackRate
 *   flexDeltaSeconds?,       // per-clip Logic flex onset shift
 *
 * Two playback modes:
 *   (a) Variant:   rate=1, buffer=url,     offset=sourceOffsetSeconds.
 *       Bit-exact to Logic; used when rootUrl is absent or rootUrl's
 *       buffer didn't decode.
 *   (b) Root+stretch: rate=stretchRatio, buffer=rootUrl, offset=
 *       sourceOffsetSeconds. Logic's actual internal model — saves
 *       download bandwidth (6 roots vs 15 rendered variants) and lets
 *       stretch-ratio changes be runtime-only.
 *
 * Buffer-time math for source.start(when, offset, durationBuf):
 *   - `offset` is in buffer-seconds (rate=1 content indexing).
 *   - `durationBuf` is in buffer-seconds (how much buffer to consume).
 *   - With playbackRate r, durationBuf=D*r gives wall-clock D seconds.
 *
 * Mid-clip resume: if playhead is already inside a clip, we skip
 * `timeIntoClip` wall-clock seconds, which means `timeIntoClip * r`
 * buffer-seconds of content are already "behind" us in the source.
 *
 * All clips on the track share one `trackGain` (track-local mute/solo
 * + fader). The track gain connects to `outputNode` (the bus gain),
 * so bus fader/mute/solo apply.
 */
function scheduleTrackClips(
  audioContext,
  clips,
  trackLocalGain,
  currentPlayheadTime,
  schedulingStartTime,
  outputNode,
  existingTrackGain = null,
  sessionSampleRate = null,
  pluginChain = null,
) {
  const result = { sources: [], gainNode: existingTrackGain };
  if (!Array.isArray(clips) || clips.length === 0) return result;

  // R11 splice: when a fresh trackGain is created and a live plugin chain
  // is supplied, route the trackGain through the chain before reaching
  // outputNode. When existingTrackGain is provided, the caller
  // (scheduleOrDefer) already wired the chain at the trackGain layer.
  const trackGain = existingTrackGain || (() => {
    const g = audioContext.createGain();
    g.gain.value = trackLocalGain;
    if (pluginChain) {
      g.connect(pluginChain.input);
      pluginChain.output.connect(outputNode);
    } else {
      g.connect(outputNode);
    }
    return g;
  })();
  result.gainNode = trackGain;

  let rootCount = 0;
  let variantCount = 0;
  let sliceCount = 0;
  let fadedCount = 0;
  const lookup = (url) => audioBufferCache.get(url) || null;
  const opts = sessionSampleRate ? { sampleRate: sessionSampleRate } : undefined;

  for (const clip of clips) {
    const plan = computeClipPlayback(clip, currentPlayheadTime, schedulingStartTime, lookup, opts);
    if (!plan || plan.sources.length === 0) {
      const primary = clip?.rootUrl || clip?.url;
      if (primary && !audioBufferCache.has(primary)) {
        console.warn(`[clip] buffer not cached: ${primary.slice(0, 60)}…`);
      }
      continue;
    }

    // Per-clip gain wraps every source so fade-in/out applies uniformly
    // across all slices of a sliced clip. clipGain → trackGain → outputNode.
    const needsFade = plan.fadeInSec > 0 || plan.fadeOutSec > 0;
    const clipGain = needsFade ? audioContext.createGain() : null;
    if (clipGain) {
      clipGain.connect(trackGain);
      const clipStartCtx = schedulingStartTime + (plan.clipStart - currentPlayheadTime);
      const clipEndCtx   = schedulingStartTime + (plan.clipEnd   - currentPlayheadTime);
      // Linear ramps until we map Logic's curve enum precisely (gap #7 in
      // playback-model). Curves <=50ms are imperceptibly different.
      if (plan.fadeInSec > 0) {
        const fadeStart = Math.max(audioContext.currentTime, clipStartCtx);
        const fadeEnd   = clipStartCtx + plan.fadeInSec;
        clipGain.gain.setValueAtTime(0, fadeStart);
        clipGain.gain.linearRampToValueAtTime(1, fadeEnd);
      } else {
        clipGain.gain.value = 1;
      }
      if (plan.fadeOutSec > 0) {
        const fadeStart = clipEndCtx - plan.fadeOutSec;
        clipGain.gain.setValueAtTime(1, fadeStart);
        clipGain.gain.linearRampToValueAtTime(0, clipEndCtx);
      }
      fadedCount++;
    }
    const target = clipGain || trackGain;

    // Flex-pitch hook. When a clip carries metadata indicating Logic's
    // per-hop flex-pitch is active, the audible result is a phase-vocoded
    // pitch shift the browser must reproduce — polypitchService is the
    // intended render path. The data we need to drive it (per-hop detune
    // values bound to specific clips) isn't fully threaded yet:
    //  - session.flexPitchNotes lacks a gid/auflGid linkage to bind notes
    //    to clips (gap noted in playback-model §10 item 6).
    //  - polypitchService.renderWithNewPitches expects basic-pitch-style
    //    notes with original pitches; mapping flexPitchNotes onto those
    //    requires either basic-pitch on the clip's source first, or new
    //    backend fields carrying original+edited pitch pairs per hop.
    // Until that lands, log an explicit warning so an audibly-shifted clip
    // playing un-shifted is obvious in the console rather than silent.
    if (clip.flexPitch && (clip.flexPitch.editedFrameCount > 0 || clip.flexPitch.frameCount > 0)) {
      const fp = clip.flexPitch;
      console.warn(
        `[flexPitch] ${clip.regionName || clip.auflGid || '?'} has ${fp.editedFrameCount}/${fp.frameCount} edited frames `
        + `(hopFactor=${fp.hopFactor}); per-hop detune not yet rendered — playing unshifted`,
      );
    }

    for (const spec of plan.sources) {
      const buf = lookup(spec.playUrl);
      if (!buf) continue; // lookup raced with eviction between plan + apply
      const src = audioContext.createBufferSource();
      src.buffer = buf;
      if (spec.playbackRate !== 1) src.playbackRate.value = spec.playbackRate;
      src.connect(target);
      try {
        src.start(spec.when, spec.bufOffset, spec.bufDur);
      } catch (err) {
        console.warn(`[clip] start failed: ${err?.message || err}`);
        continue;
      }
      result.sources.push(src);
      if (spec.isSlice) sliceCount++;
      else if (spec.useRoot) rootCount++;
      else variantCount++;
    }
  }

  const totalSources = rootCount + variantCount + sliceCount;
  if (totalSources > 0) {
    const parts = [];
    if (variantCount) parts.push(`${variantCount} variant`);
    if (rootCount)    parts.push(`${rootCount} root+stretch`);
    if (sliceCount)   parts.push(`${sliceCount} slice`);
    if (fadedCount)   parts.push(`${fadedCount} faded`);
    console.log(`  🎬 clips: ${parts.join(', ')} (of ${clips.length} clip(s))`);
  }
  return result;
}
