import React, { useEffect, useRef } from 'react';
import { fetchAudioWithCache } from '../services/audioCacheService';
import { LRUBufferCache } from '../utils/lruBufferCache';

// Global in-memory cache for decoded audio buffers (faster than re-decoding).
// EXPORTED so other services (e.g. latent-based waveform previews) can
// pre-populate it before a track's real audio is ready. Byte-capped LRU to
// avoid OOM on long sessions — see `src/utils/lruBufferCache.js`.
export const audioBufferCache = new LRUBufferCache({
  maxBytes: 256 * 1024 * 1024,
  name: 'waveformBufferCache',
});

// ── Bar geometry constants (shared across all tracks) ─────────────────
const BAR_WIDTH = 2;
const BAR_SPACING = 2;
const BAR_STRIDE = BAR_WIDTH + BAR_SPACING;

// ── Target-computation helpers (pure) ─────────────────────────────────
// Each returns a Float32Array[numBars] of pixel heights the animator should
// ease toward. All self-normalised so the loudest bar fills maxBarHeight.

function _noiseTarget(numBars, maxBarHeight) {
  const baseline = Math.max(1, maxBarHeight * 0.08);
  const out = new Float32Array(numBars);
  out.fill(baseline);
  return out;
}

function _targetFromEnvelope(envelopeData, numBars, maxBarHeight, gain) {
  const T = envelopeData.length / 2;
  const mins = envelopeData.subarray(0, T);
  const maxs = envelopeData.subarray(T, 2 * T);
  const ptp = new Float32Array(numBars);
  let maxPtp = 0;
  for (let i = 0; i < numBars; i++) {
    const s = Math.floor((i * T) / numBars);
    const e = Math.max(s + 1, Math.floor(((i + 1) * T) / numBars));
    let mx = -Infinity, mn = Infinity;
    for (let j = s; j < e && j < T; j++) {
      if (maxs[j] > mx) mx = maxs[j];
      if (mins[j] < mn) mn = mins[j];
    }
    ptp[i] = mx - mn;
    if (ptp[i] > maxPtp) maxPtp = ptp[i];
  }
  const normFactor = maxPtp > 0.001 ? maxBarHeight / maxPtp : 1.0;
  const noiseFloor = maxPtp * 0.02;
  const out = new Float32Array(numBars);
  const g = Math.max(0, gain);
  for (let i = 0; i < numBars; i++) {
    out[i] = (ptp[i] < noiseFloor || g === 0)
      ? 0.5
      : Math.min(ptp[i] * normFactor * g, maxBarHeight);
  }
  return out;
}

function _targetFromBuffer(buffer, numBars, maxBarHeight, gain, cropStart, cropEnd) {
  const data = buffer.getChannelData(0);
  const duration = buffer.duration;
  const sr = buffer.sampleRate;
  const startSample = Math.floor(cropStart * sr);
  const endSample = cropEnd > 0 ? Math.floor((duration - cropEnd) * sr) : data.length;
  const visible = Math.max(1, endSample - startSample);
  const samplesPerBar = Math.floor(visible / numBars);
  const ptp = new Float32Array(numBars);
  let maxPtp = 0;
  if (samplesPerBar > 0) {
    for (let i = 0; i < numBars; i++) {
      const s = startSample + i * samplesPerBar;
      const e = Math.min(s + samplesPerBar, endSample);
      let mx = -Infinity, mn = Infinity;
      for (let j = s; j < e && j < data.length; j++) {
        const v = data[j];
        if (v > mx) mx = v;
        if (v < mn) mn = v;
      }
      ptp[i] = mx === -Infinity ? 0 : (mx - mn);
      if (ptp[i] > maxPtp) maxPtp = ptp[i];
    }
  }
  const norm = maxPtp > 0.001 ? maxBarHeight / maxPtp : 1.0;
  const noiseFloor = maxPtp * 0.02;
  const out = new Float32Array(numBars);
  const g = Math.max(0, gain);
  for (let i = 0; i < numBars; i++) {
    out[i] = (ptp[i] < noiseFloor || g === 0)
      ? 0.5
      : Math.min(ptp[i] * norm * g, maxBarHeight);
  }
  return out;
}

/**
 * useWaveform
 *
 * Single-writer waveform renderer. One persistent requestAnimationFrame
 * loop (lifetime-bound to the hook instance) is the ONLY code that ever
 * clears or writes to the canvas. External state changes — new envelope,
 * decoded audio buffer, gain tweak, width resize, wobble-on/off — only
 * update a target array held in stateRef. Every frame, the animator
 * eases current bar heights toward target via exponential smoothing and
 * adds an optional sine wobble overlay.
 *
 * This guarantees:
 *   • no paint path outside the animator can blank the canvas
 *   • canvas.width/height are assigned only on a real dimension change
 *   • every target swap (noise → rms → mask → backend WAV) is a smooth
 *     per-bar morph with no snap, no redraw flash, no rAF gap
 *
 * @param {string|null}       audioUrl
 * @param {number}            width
 * @param {number}            height
 * @param {string}            color
 * @param {number}            cropStart      seconds
 * @param {number}            cropEnd        seconds
 * @param {Float32Array|null} envelopeData   [2*T] min/max per frame
 * @param {number}            envelopeFps
 * @param {number}            gain           0..N
 * @param {boolean}           showNoise      force noise baseline (no audio)
 * @param {boolean}           loadingWobble  idle wobble overlay while loading
 * @param {number}            revealDelayMs  hold noise target for this long
 *                                           on first envelope paint
 */
export function useWaveform(
  audioUrl,
  width = 800,
  height = 60,
  color = '#f5f5f5',
  cropStart = 0,
  cropEnd = 0,
  envelopeData = null,
  envelopeFps = 25,
  gain = 1.0,
  showNoise = false,
  loadingWobble = false,
  revealDelayMs = 0,
) {
  const canvasRef = useRef(null);
  const rafIdRef = useRef(null);
  const audioBufferRef = useRef(null);
  const loadedUrlRef = useRef(null);
  const revealedRef = useRef(false);
  const revealTimerRef = useRef(null);

  const [isLoaded, setIsLoaded] = React.useState(false);
  const [duration, setDuration] = React.useState(null);
  const [decodedAudioUrl, setDecodedAudioUrl] = React.useState(null);

  // Initialise the animator state on first render. Subsequent renders
  // reuse the same object — the animator reads from it every frame.
  const stateRef = useRef(null);
  if (!stateRef.current) {
    stateRef.current = {
      // current display config (synced from props each render below)
      width, height, color, gain,
      cropStart, cropEnd,
      // target + current bar arrays (pixel heights, eased per frame)
      current: null,
      target: null,
      // per-frame smoothing factor for current → target
      //  0.12 ≈ ~16 frames to 86% converge ≈ 265 ms at 60 Hz.
      lerpRate: 0.12,
      // wobble overlay (sine oscillation). Amp eases with smoothstep
      // between 0 and wobbleTargetAmp when enabled/disabled.
      wobble: {
        amp: 0,
        targetAmp: 0,
        fadeStartMs: 0,
        fadeStartAmp: 0,
        phases: null,
        freqs: null,
      },
      animStartMs: 0,
      // whether target needs recompute (set by target-source effects)
      targetKind: 'noise',   // 'noise' | 'envelope' | 'buffer'
      // Has a REAL (non-noise) target been applied yet? The first time it
      // is, we snap `current` to match so the waveform appears at full
      // shape instantly — no ease-in from the noise baseline. Subsequent
      // target swaps still ease normally for smooth rms → mask → WAV.
      hasRealTarget: false,
    };
  }
  const envelopeDataRef = useRef(envelopeData);
  envelopeDataRef.current = envelopeData;

  // Keep state in sync with current props (cheap; read each frame).
  stateRef.current.width = width;
  stateRef.current.height = height;
  stateRef.current.color = color;
  stateRef.current.gain = gain;
  stateRef.current.cropStart = cropStart;
  stateRef.current.cropEnd = cropEnd;

  // ─────────────────────────────────────────────────────────────────
  // Persistent rAF animator — the ONLY code that touches the canvas.
  // Runs from mount to unmount; no cancel/restart per envelope change.
  // ─────────────────────────────────────────────────────────────────
  useEffect(() => {
    stateRef.current.animStartMs = performance.now();

    const tick = (now) => {
      const s = stateRef.current;
      const canvas = canvasRef.current;
      if (!canvas) {
        rafIdRef.current = requestAnimationFrame(tick);
        return;
      }

      const numBars = Math.max(1, Math.floor(s.width / BAR_STRIDE));
      const maxBarHeight = s.height * 0.45;
      const midY = s.height / 2;

      // Resize canvas pixel buffer ONLY on a real dimension change.
      // Any canvas.width = x assignment clears the canvas, so gate it.
      if (canvas.width !== s.width)   canvas.width = s.width;
      if (canvas.height !== s.height) canvas.height = s.height;

      // Ensure target exists. If no upstream source has set one, render
      // a quiet noise baseline so the canvas is never blank.
      if (!s.target || s.target.length !== numBars) {
        s.target = _noiseTarget(numBars, maxBarHeight);
      }
      // Ensure current. On numBars changes, snap to target (resize is
      // rare and shouldn't animate).
      if (!s.current || s.current.length !== numBars) {
        s.current = new Float32Array(s.target);
      }
      // Ensure wobble phase/freq arrays match numBars.
      if (!s.wobble.phases || s.wobble.phases.length !== numBars) {
        s.wobble.phases = new Float32Array(numBars);
        s.wobble.freqs = new Float32Array(numBars);
        for (let i = 0; i < numBars; i++) {
          s.wobble.phases[i] = Math.random() * Math.PI * 2;
          s.wobble.freqs[i] = 0.8 + Math.random() * 1.0;  // 0.8–1.8 Hz
        }
      }

      // Ease current toward target, per-bar exponential smoothing.
      // This is THE morph — every target swap flows through here.
      const r = s.lerpRate;
      for (let i = 0; i < numBars; i++) {
        s.current[i] += (s.target[i] - s.current[i]) * r;
      }

      // Wobble amp smoothstep between fadeStartAmp and targetAmp over 500 ms.
      const fadeDur = 500;
      const wElapsed = now - s.wobble.fadeStartMs;
      const wFrac = Math.min(Math.max(wElapsed / fadeDur, 0), 1);
      const wEase = wFrac * wFrac * (3 - 2 * wFrac);
      s.wobble.amp = s.wobble.fadeStartAmp
        + (s.wobble.targetAmp - s.wobble.fadeStartAmp) * wEase;

      // Draw. clearRect is the only way the canvas ever gets blanked,
      // and it's immediately followed by the redraw in the same frame —
      // no chance of a blank-canvas flash.
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, s.width, s.height);
      ctx.strokeStyle = s.color;
      ctx.lineWidth = BAR_WIDTH;
      ctx.lineCap = 'round';
      ctx.globalAlpha = 0.7;
      const wobbleActive = s.wobble.amp > 0.05;
      const wobbleDt = (now - s.animStartMs) / 1000;
      for (let i = 0; i < numBars; i++) {
        let h = s.current[i];
        if (wobbleActive) {
          h += s.wobble.amp * Math.sin(2 * Math.PI * s.wobble.freqs[i] * wobbleDt + s.wobble.phases[i]);
        }
        h = Math.max(0.5, h);
        const x = i * BAR_STRIDE + BAR_WIDTH / 2;
        ctx.beginPath();
        ctx.moveTo(x, midY - h);
        ctx.lineTo(x, midY + h);
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      rafIdRef.current = requestAnimationFrame(tick);
    };

    rafIdRef.current = requestAnimationFrame(tick);

    return () => {
      if (rafIdRef.current) cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
      if (revealTimerRef.current) clearTimeout(revealTimerRef.current);
      revealTimerRef.current = null;
    };
  // Animator is lifetime-bound. Props updates feed in via stateRef above.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ─────────────────────────────────────────────────────────────────
  // Target sources. Each effect only updates stateRef.current.target;
  // none of them draws. Priority: envelopeData > audioBuffer > noise.
  // ─────────────────────────────────────────────────────────────────

  // Envelope-driven target (instant visual from rms, mask, WAV-derived envs).
  useEffect(() => {
    if (!envelopeData) return;
    const s = stateRef.current;
    const numBars = Math.max(1, Math.floor(width / BAR_STRIDE));
    const maxBar = height * 0.45;
    const newTarget = _targetFromEnvelope(envelopeData, numBars, maxBar, gain);

    const apply = () => {
      s.target = newTarget;
      s.targetKind = 'envelope';
      if (!s.hasRealTarget) {
        s.current = new Float32Array(newTarget);  // snap on first real target
        s.hasRealTarget = true;
      }
    };

    // First envelope paint may be gated by a reveal delay (e.g. stems
    // hold noise for 1 s after mount). Subsequent envelope swaps always
    // morph instantly through the animator.
    if (revealDelayMs > 0 && !revealedRef.current) {
      // Ensure the animator shows a noise target during the delay.
      s.target = _noiseTarget(numBars, maxBar);
      s.targetKind = 'noise';
      if (revealTimerRef.current) clearTimeout(revealTimerRef.current);
      revealTimerRef.current = setTimeout(() => {
        revealedRef.current = true;
        revealTimerRef.current = null;
        if (envelopeDataRef.current) {
          apply();
        }
      }, revealDelayMs);
    } else {
      revealedRef.current = true;
      apply();
    }

    const T = envelopeData.length / 2;
    if (!isLoaded) {
      setDuration(T / envelopeFps);
      setIsLoaded(true);
    }
  }, [envelopeData, width, height, gain, envelopeFps, revealDelayMs, isLoaded]);

  // Noise-only target (MIDI tracks pre-generation, no envelope + no audio).
  useEffect(() => {
    const s = stateRef.current;
    if (!showNoise) return;
    if (envelopeData) return;      // envelope wins
    if (audioBufferRef.current) return;  // decoded audio wins
    const numBars = Math.max(1, Math.floor(width / BAR_STRIDE));
    const maxBar = height * 0.45;
    s.target = _noiseTarget(numBars, maxBar);
    s.targetKind = 'noise';
  }, [showNoise, envelopeData, width, height]);

  // Audio-buffer-driven target (used when no envelopeData — master track
  // pre-separation, or any audio track without a latent envelope).
  useEffect(() => {
    if (!audioUrl) return;
    if (loadedUrlRef.current === audioUrl) {
      // Re-apply buffer target on dim / crop / gain changes.
      const buf = audioBufferRef.current;
      if (buf && !envelopeDataRef.current) {
        const s = stateRef.current;
        const numBars = Math.max(1, Math.floor(width / BAR_STRIDE));
        const maxBar = height * 0.45;
        const newTarget = _targetFromBuffer(buf, numBars, maxBar, gain, cropStart, cropEnd);
        s.target = newTarget;
        s.targetKind = 'buffer';
        if (!s.hasRealTarget) {
          s.current = new Float32Array(newTarget);
          s.hasRealTarget = true;
        }
      }
      return;
    }

    let cancelled = false;
    const loadAudio = async () => {
      try {
        let audioBuffer;
        if (audioBufferCache.has(audioUrl)) {
          audioBuffer = audioBufferCache.get(audioUrl);
        } else {
          const { blob } = await fetchAudioWithCache(audioUrl);
          if (cancelled) return;
          const ctx = new (window.AudioContext || window.webkitAudioContext)();
          const ab = await blob.arrayBuffer();
          audioBuffer = await ctx.decodeAudioData(ab);
          if (cancelled) return;
          audioBufferCache.set(audioUrl, audioBuffer);
        }
        if (cancelled) return;
        audioBufferRef.current = audioBuffer;
        loadedUrlRef.current = audioUrl;
        setDuration(audioBuffer.duration);
        setIsLoaded(true);
        setDecodedAudioUrl(audioUrl);

        // Only update the target from the buffer if no envelope is
        // currently active — envelope is the higher-priority source.
        const s = stateRef.current;
        if (!envelopeDataRef.current) {
          const numBars = Math.max(1, Math.floor(width / BAR_STRIDE));
          const maxBar = height * 0.45;
          const newTarget = _targetFromBuffer(audioBuffer, numBars, maxBar, gain, cropStart, cropEnd);
          s.target = newTarget;
          s.targetKind = 'buffer';
          if (!s.hasRealTarget) {
            s.current = new Float32Array(newTarget);  // snap on first real target
            s.hasRealTarget = true;
          }
        }
      } catch (err) {
        console.error('Error loading audio:', err);
      }
    };
    loadAudio();
    return () => { cancelled = true; };
  }, [audioUrl, width, height, gain, cropStart, cropEnd]);

  // Loading wobble toggle — just flips the amp target; the animator
  // smoothsteps amp → new target over 500 ms, and the overlay sine rides
  // on top of whatever bars the ease-toward-target loop is producing.
  useEffect(() => {
    const s = stateRef.current;
    const maxBar = height * 0.45;
    const targetAmp = loadingWobble ? Math.max(1.6, maxBar * 0.09) : 0;
    if (s.wobble.targetAmp !== targetAmp) {
      s.wobble.fadeStartAmp = s.wobble.amp;
      s.wobble.targetAmp = targetAmp;
      s.wobble.fadeStartMs = performance.now();
    }
  }, [loadingWobble, height]);

  return {
    canvasRef,
    audioBuffer: audioBufferRef.current,
    duration,
    isLoaded,
    decodedAudioUrl,
  };
}

export function clearWaveformCache() {
  const n = audioBufferCache.size;
  audioBufferCache.clear();
  console.log(`🗑️ Cleared ${n} cached audio buffers`);
  return n;
}

export function getWaveformCacheStats() {
  return {
    size: audioBufferCache.size,
    bytes: audioBufferCache.bytes,
    maxBytes: audioBufferCache.maxBytes,
    pressure: audioBufferCache.pressure,
    urls: Array.from(audioBufferCache.keys()),
  };
}
