import { useEffect, useRef, useCallback } from 'react';
import midiPlayer from '../utils/midiPlayer';
import tunaFX from '../services/tunaFX';
import { isMaskPlaybackReady, setGains as setMaskGains, setActiveStem, seek as maskSeek, play as maskPlay, stop as maskStop } from '../services/maskPlayback';
import { dispatchStrategy, getTrackSubstemSchedules, resumeFromPlayhead } from '../services/virtualTrackEdit';
import { getStretchedBuffer } from '../services/wsolaStretch';

// Global audio buffer cache (shared across all instances)
const audioBufferCache = new Map();
const RESUME_ATTACK_SECONDS = 0.005;

function isRenderablePlaybackTrack(track) {
  // `isBusMaster` is the hidden uploaded full-mix parent kept in state for
  // analysis/regen after stem separation. It must never be scheduled for
  // playback once stems exist or it will mask stem edits and double the mix.
  return !track?.metadata?.isBusMaster;
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
export function useAudioPlayback(tracks, isPlaying, dispatch, totalDuration = 10, currentPlayheadPosition = 0, bpm = 120, masterGain = 0.8, beatsPerBar = 4, meterDenominator = 4, tempoMap = null) {
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
  const playbackSessionRef = useRef(0); // Invalidate stale late-load scheduling across pause/play cycles
  const playRef = useRef(null); // Latest play() fn — called by the meter-change live-reschedule effect

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

    const voTracks = (tracks.vo || []).filter(isRenderablePlaybackTrack);
    const musicTracks = (tracks.music || []).filter(isRenderablePlaybackTrack);
    const sfxTracks = (tracks.sfx || []).filter(isRenderablePlaybackTrack);
    const midiTracks = (tracks.midi || []).filter(isRenderablePlaybackTrack);
    const audioTracks = (tracks.audio || []).filter(isRenderablePlaybackTrack);

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
        (t.type !== 'midi' && t.audioUrl) || // Audio tracks in MIDI bus
        (t.type === 'midi' && (t.audioUrl || t.f0Audio || t.midiData)) // MIDI tracks with playable content
      );

      const allTracks = [...voTracks, ...musicTracks, ...sfxTracks, ...drumTracks, ...midiAudioTracks, ...audioTracks];

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
      const hasMidiTracks = allTracks.some(t => t.type === 'midi' && t.midiData);
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
        if (t.type === 'midi' && t.f0Audio && !audioBufferCache.has(t.f0Audio)) urlsToWarm.push(t.f0Audio);
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
      const projectMeter = { bpm, beatsPerBar, meterDenominator, tempoMap };

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

      const scheduleOrDefer = (track, finalGain) => {
        const url = track.audioUrl;
        if (!url) return;
        const trackStartTime = track.startPosition || 0;

        // ── Drum substem fan-out ───────────────────────────────────────
        const substems = getTrackSubstemSchedules(track, projectMeter);
        if (substems) {
          // Create the shared per-track gain once and register it under
          // the parent track id so the gain-update loop reaches it.
          const trackGain = audioContext.createGain();
          trackGain.gain.value = finalGain;
          trackGain.connect(masterGainNodeRef.current);
          gainNodesRef.current.set(track.id, trackGain);

          for (const [name, { audioUrl, schedule, kind }] of Object.entries(substems)) {
            const scheduleSubstem = (when, anchor) => {
              const { sources } = scheduleTrackWithSchedule(
                audioContext, audioUrl, schedule, finalGain, trackStartTime,
                when, anchor, masterGainNodeRef.current,
                trackGain,    // <- share the parent's gain node
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
                if (!isSessionActive()) return;
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
        }
        if (audioBufferCache.has(url)) {
          const { sources, gainNode } = scheduleTrackWithSchedule(
            audioContext, url, schedule, finalGain, trackStartTime,
            currentPlayheadTime, schedulingStartTime, masterGainNodeRef.current,
          );
          if (track.metadata?.polypitchRendered) {
            const buf = audioBufferCache.get(url);
            const ch = buf.getChannelData(0);
            let sum = 0, peak = 0;
            const N = Math.min(ch.length, 96000); // sample first 2s at 48k
            for (let i = 0; i < N; i++) { const v = ch[i]; sum += v*v; const a = Math.abs(v); if (a > peak) peak = a; }
            const rms = Math.sqrt(sum / Math.max(1, N));
            console.log(`  🎹 scheduled polypitch ${track.metadata?.stemType || '?'} dur=${buf.duration.toFixed(2)}s rms(2s)=${rms.toFixed(5)} peak(2s)=${peak.toFixed(4)} gain=${finalGain.toFixed(2)} segs=${sources.length} from=${url.slice(0, 40)}…`);
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
          if (gainNode) {
            sourceNodesRef.current.push(...sources);
            gainNodesRef.current.set(track.id, gainNode);
          }
          return;
        }
        // Not cached — fire async, schedule on arrival if still playing.
        (async () => {
          try {
            const audioBuffer = await ensureBuffer(url);
            if (!isSessionActive()) return;
            const livePlayhead = audioContext.currentTime - startTimeRef.current;
            const trackDuration = track.duration || audioBuffer.duration;
            if (livePlayhead > trackStartTime + trackDuration) return;  // already past
            const { sources, gainNode } = scheduleTrackWithSchedule(
              audioContext, url, schedule, finalGain, trackStartTime,
              livePlayhead, audioContext.currentTime, masterGainNodeRef.current,
            );
            if (gainNode) {
              sourceNodesRef.current.push(...sources);
              gainNodesRef.current.set(track.id, gainNode);
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
            // Regular track (non-drum). scheduleOrDefer handles both
            // cached-immediately and arrive-later buffers — neither path
            // blocks the play-click.
            const trackStartTime = track.startPosition || 0;
            const trackDuration = track.duration || track.length || 10;
            const trackEndTime = trackStartTime + trackDuration;
            if (currentPlayheadTime <= trackEndTime) {
              scheduleOrDefer(track, finalGain);
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
 *   [ seg1 source → seg1 fadeGain ─┼── track gain ── master
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
 */
function scheduleTrackWithSchedule(
  audioContext, audioUrl, schedule, gain, trackStartTime,
  currentPlayheadTime, schedulingStartTime, masterGainNode,
  existingTrackGain = null,
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
    const trackGain = existingTrackGain || (() => {
      const g = audioContext.createGain();
      g.gain.value = gain;
      g.connect(masterGainNode);
      return g;
    })();

    // Clip-local playhead (0 == track start on the timeline).
    const clipPlayhead = currentPlayheadTime - trackStartTime;
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
        when = schedulingStartTime + (trackStartTime + seg.dstStart - currentPlayheadTime);
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
