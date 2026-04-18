import React, { useEffect, useRef, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import { audioBufferCache } from '../../hooks/useWaveform';

/**
 * CompositeBusWaveform — single rAF-driven canvas that renders the master
 * waveform in every phase of the lifecycle:
 *
 *   1. Drop → bus appears immediately with a wobbling noise baseline.
 *   2. Master AudioBuffer lands in audioBufferCache (from background
 *      decode) → bars ease from noise to the real ptp-per-bar shape.
 *   3. Stems arrive (rms → backend WAV) → stem gains can modulate
 *      per-bar amplitude via a ratio when the user touches a slider;
 *      all-gains-=1 leaves the master shape unchanged.
 *
 * Design rules (from user):
 *   • There is NEVER an empty frame. If nothing is loaded, draw noise.
 *   • Phase transitions always FADE via exponential easing — no snaps.
 *   • Master shape is only modulated by user slider movement, not by
 *     envelope refinements (rms → mask → WAV).
 */
const CompositeBusWaveform = React.memo(({
  bus,
  height,
  envelopeFps = 25,
  color = '#667eea',
}) => {
  const { state } = useApp();
  const canvasRef = useRef(null);
  const rafIdRef = useRef(null);

  // Width sizing — match OptimizedTrack so a collapsed composite spans
  // the same horizontal footprint as an expanded stem would.
  const { pixelsPerSecond } = useTimeline(
    state.totalDuration || 10,
    state.zoomLevel || 1.0,
    state.timelineWidth || 950
  );
  const busDuration = useMemo(() => {
    let d = 0;
    for (const t of bus.tracks) {
      const td = t.duration || 0;
      if (td > d) d = td;
    }
    return d || 10;
  }, [bus.tracks]);
  const width = Math.max(1, Math.floor(busDuration * pixelsPerSecond));

  // Reactive per-stem inputs — re-memoized on every bus.tracks dispatch.
  const stemInputs = useMemo(() => {
    const hasStems = bus.tracks.some(t => t.metadata?.type === 'stem');
    const anySolo = bus.tracks.some(t => t.isSolo);
    const pool = hasStems
      ? bus.tracks.filter(t => t.metadata?.type === 'stem' && !t.isPlaceholder)
      : bus.tracks.filter(t => t.type !== 'midi' && !t.isPlaceholder);
    return pool.map(t => {
      let eff = 1.0;
      if (t.isMuted) eff = 0;
      else if (anySolo && !t.isSolo) eff = 0;
      else eff = t.gain ?? 1.0;
      return {
        id: t.id,
        gain: eff,
        envelope: t.metadata?.envelopeData || null,
        audioBuffer: t.audioUrl ? audioBufferCache.get(t.audioUrl) || null : null,
        audioUrl: t.audioUrl || null,
      };
    });
  }, [bus.tracks]);

  // Master-track audioUrl — the key we use to probe audioBufferCache
  // from inside the rAF loop (no re-render needed once the cache fills).
  const masterAudioUrl = useMemo(() => {
    const m = bus.tracks.find(t => t.metadata?.type === 'uploaded');
    return m?.audioUrl || null;
  }, [bus.tracks]);

  // Self-decode: if nothing else seeded the cache within the first ~50 ms
  // of mount, decode the master ourselves. Happens in the background; the
  // animator keeps rendering noise until the buffer lands.
  useEffect(() => {
    if (!masterAudioUrl || audioBufferCache.has(masterAudioUrl)) return;
    let cancelled = false;
    (async () => {
      try {
        const { fetchAudioWithCache } = await import('../../services/audioCacheService');
        const { blob } = await fetchAudioWithCache(masterAudioUrl);
        if (cancelled) return;
        const ctx = new (window.AudioContext || window.webkitAudioContext)();
        const ab = await blob.arrayBuffer();
        const buf = await ctx.decodeAudioData(ab);
        if (cancelled) return;
        audioBufferCache.set(masterAudioUrl, buf);
        // No re-render trigger needed — the rAF loop polls the cache.
      } catch (e) {
        console.warn('[composite] master decode failed:', e?.message || e);
      }
    })();
    return () => { cancelled = true; };
  }, [masterAudioUrl]);

  const busGain = bus.mute ? 0 : (bus.gain ?? 1.0);

  // ── Animator state ─────────────────────────────────────────────────
  // Persisted across renders; the rAF tick reads from here each frame.
  const stateRef = useRef(null);
  if (!stateRef.current) {
    stateRef.current = {
      width, height, color, busGain,
      stemInputs,
      masterAudioUrl,
      envelopeFps,
      // Per-bar heights (pixels). current eases toward target.
      current: null,
      target: null,
      lerpRate: 0.12,            // ≈265 ms to 86% convergence at 60 Hz
      // Cached master ptp computation (only recomputed when buffer ref changes).
      cachedMasterBuf: null,
      cachedMasterPtp: null,
      cachedNumBars: 0,
      // Wobble overlay — active while we're showing the noise baseline.
      wobble: {
        amp: 0,
        targetAmp: 0,
        fadeStartMs: 0,
        fadeStartAmp: 0,
        phases: null,
        freqs: null,
      },
      animStartMs: 0,
    };
  }
  // Keep sync'd (cheap).
  stateRef.current.width = width;
  stateRef.current.height = height;
  stateRef.current.color = color;
  stateRef.current.busGain = busGain;
  stateRef.current.stemInputs = stemInputs;
  stateRef.current.masterAudioUrl = masterAudioUrl;
  stateRef.current.envelopeFps = envelopeFps;

  // ── Persistent rAF animator ────────────────────────────────────────
  useEffect(() => {
    stateRef.current.animStartMs = performance.now();

    const tick = (now) => {
      const s = stateRef.current;
      const canvas = canvasRef.current;
      if (!canvas) {
        rafIdRef.current = requestAnimationFrame(tick);
        return;
      }

      const barWidth = 2;
      const barSpacing = 2;
      const barStride = barWidth + barSpacing;
      const numBars = Math.max(1, Math.floor(s.width / barStride));
      const maxBarHeight = s.height * 0.45;
      const midY = s.height / 2;

      // Resize canvas only on a real dimension change (canvas.width = x
      // clears the pixel buffer, so gate it).
      if (canvas.width !== s.width) canvas.width = s.width;
      if (canvas.height !== s.height) canvas.height = s.height;

      // Ensure wobble phase/freq arrays.
      if (!s.wobble.phases || s.wobble.phases.length !== numBars) {
        s.wobble.phases = new Float32Array(numBars);
        s.wobble.freqs = new Float32Array(numBars);
        for (let i = 0; i < numBars; i++) {
          s.wobble.phases[i] = Math.random() * Math.PI * 2;
          s.wobble.freqs[i] = 0.8 + Math.random() * 1.0;  // 0.8–1.8 Hz
        }
      }

      // Compute the master's ptp-per-bar, cached by (buffer ref, numBars).
      const masterBuf = s.masterAudioUrl ? audioBufferCache.get(s.masterAudioUrl) || null : null;
      if (masterBuf && (masterBuf !== s.cachedMasterBuf || s.cachedNumBars !== numBars)) {
        s.cachedMasterBuf = masterBuf;
        s.cachedNumBars = numBars;
        s.cachedMasterPtp = ptpPerBarFromBuffer(masterBuf, numBars);
      }
      if (!masterBuf) {
        s.cachedMasterBuf = null;
        s.cachedMasterPtp = null;
      }

      // Reveal delay: hold the noise baseline for REVEAL_DELAY_MS after
      // mount regardless of when the master buffer lands. Matches the
      // stem-reveal behaviour so the master + stems feel consistent —
      // user always sees noise first, then the real shape fades in.
      const elapsed = now - s.animStartMs;
      const REVEAL_DELAY_MS = 1000;
      const forceNoise = elapsed < REVEAL_DELAY_MS;
      const effectiveMasterPtp = forceNoise ? null : s.cachedMasterPtp;

      // Compute target bars for this frame.
      const newTarget = computeTarget({
        numBars,
        maxBarHeight,
        masterPtp: effectiveMasterPtp,
        stemInputs: s.stemInputs,
        busGain: s.busGain,
        envelopeFps: s.envelopeFps,
      });
      s.target = newTarget;

      // Wobble visibility: on during the reveal-delay noise phase,
      // off once the real target takes over. The wobble is initialised
      // at full amp on the first tick so motion is visible from frame 1
      // (a 500 ms fade-in would bury it if the master decodes quickly).
      const loading = forceNoise || !s.cachedMasterPtp;
      const wobbleTargetAmp = loading ? Math.max(1.6, maxBarHeight * 0.09) : 0;
      if (!s.wobbleInitialized) {
        s.wobbleInitialized = true;
        s.wobble.amp = wobbleTargetAmp;
        s.wobble.targetAmp = wobbleTargetAmp;
        s.wobble.fadeStartAmp = wobbleTargetAmp;
        s.wobble.fadeStartMs = now;
      }
      if (s.wobble.targetAmp !== wobbleTargetAmp) {
        s.wobble.fadeStartAmp = s.wobble.amp;
        s.wobble.targetAmp = wobbleTargetAmp;
        s.wobble.fadeStartMs = now;
      }
      const fadeDur = 500;
      const wElapsed = now - s.wobble.fadeStartMs;
      const wFrac = Math.min(Math.max(wElapsed / fadeDur, 0), 1);
      const wEase = wFrac * wFrac * (3 - 2 * wFrac);
      s.wobble.amp = s.wobble.fadeStartAmp
        + (s.wobble.targetAmp - s.wobble.fadeStartAmp) * wEase;

      // Ease current → target (per-bar exponential smoothing).
      if (!s.current || s.current.length !== numBars) {
        s.current = new Float32Array(newTarget);
      }
      const r = s.lerpRate;
      for (let i = 0; i < numBars; i++) {
        s.current[i] += (newTarget[i] - s.current[i]) * r;
      }

      // Draw.
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, s.width, s.height);
      ctx.strokeStyle = s.color;
      ctx.lineWidth = barWidth;
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
        const x = i * barStride + barWidth / 2;
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
    };
  // Animator is lifetime-bound. All dynamic state flows via stateRef.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        width: `${width}px`,
        height: `${height}px`,
        display: 'block',
      }}
    />
  );
});

CompositeBusWaveform.displayName = 'CompositeBusWaveform';

// ── Target computation (pure) ────────────────────────────────────────

function _noiseBars(numBars, maxBarHeight) {
  const baseline = Math.max(1, maxBarHeight * 0.08);
  const out = new Float32Array(numBars);
  out.fill(baseline);
  return out;
}

/**
 * Compute the per-bar target heights based on the current inputs.
 *   • No master buffer yet → noise baseline (wobble rides on top).
 *   • Master buffer present → ptp-per-bar of the master, normalised to
 *     maxBarHeight. Stem gains apply a per-bar ratio ONLY when the user
 *     has moved at least one slider (anyUserModulation); otherwise the
 *     master shape is untouched by envelope refinements.
 */
function computeTarget({ numBars, maxBarHeight, masterPtp, stemInputs, busGain, envelopeFps }) {
  if (!masterPtp) {
    return _noiseBars(numBars, maxBarHeight);
  }

  let maxMasterPtp = 0;
  for (let i = 0; i < numBars; i++) {
    if (masterPtp[i] > maxMasterPtp) maxMasterPtp = masterPtp[i];
  }
  if (maxMasterPtp <= 0.001) {
    return _noiseBars(numBars, maxBarHeight);
  }

  const norm = maxBarHeight / maxMasterPtp;
  const noiseFloor = maxMasterPtp * 0.02;

  // Stem-ratio modulation only engages when a slider has moved. Before
  // then, the shape is pure master-ptp × norm — stable through every
  // envelope refinement (rms → mask → WAV).
  const anyUserModulation = stemInputs.some((s) => s.gain !== 1.0);

  let baselineSummed = null;
  let summed = null;
  if (anyUserModulation) {
    const perStemBars = stemInputs.map(s => peakPerBar(s, numBars, envelopeFps));
    baselineSummed = new Float32Array(numBars);
    summed = new Float32Array(numBars);
    for (let s = 0; s < perStemBars.length; s++) {
      const peaks = perStemBars[s];
      if (!peaks) continue;
      const g = stemInputs[s].gain;
      for (let i = 0; i < numBars; i++) {
        baselineSummed[i] += peaks[i];
        if (g !== 0) summed[i] += peaks[i] * g;
      }
    }
  }

  const out = new Float32Array(numBars);
  for (let i = 0; i < numBars; i++) {
    const mv = masterPtp[i];
    let ratio = 1;
    if (anyUserModulation) {
      const bs = baselineSummed[i];
      ratio = bs > 1e-6 ? summed[i] / bs : 0;
    }
    if (mv < noiseFloor || busGain === 0 || ratio === 0) {
      out[i] = 0.5;
      continue;
    }
    out[i] = Math.min(mv * ratio * norm * busGain, maxBarHeight);
  }
  return out;
}

/** Per-stem ptp-per-bar — envelope preferred, decoded RMS fallback. */
function peakPerBar(stem, numBars, envelopeFps) {
  const out = new Float32Array(numBars);
  if (stem.envelope && stem.envelope.length >= 2) {
    const T = stem.envelope.length / 2;
    const mins = stem.envelope.subarray(0, T);
    const maxs = stem.envelope.subarray(T, 2 * T);
    for (let i = 0; i < numBars; i++) {
      const start = Math.floor((i * T) / numBars);
      const end = Math.max(start + 1, Math.floor(((i + 1) * T) / numBars));
      let mx = -Infinity, mn = Infinity;
      for (let j = start; j < end && j < T; j++) {
        if (maxs[j] > mx) mx = maxs[j];
        if (mins[j] < mn) mn = mins[j];
      }
      out[i] = mx === -Infinity ? 0 : (mx - mn);
    }
    return out;
  }
  if (stem.audioBuffer) {
    const data = stem.audioBuffer.getChannelData(0);
    const samplesPerBar = Math.floor(data.length / numBars);
    if (samplesPerBar <= 0) return out;
    for (let i = 0; i < numBars; i++) {
      const s0 = i * samplesPerBar;
      const s1 = Math.min(s0 + samplesPerBar, data.length);
      let sumSq = 0, n = 0;
      for (let j = s0; j < s1; j++) { sumSq += data[j] * data[j]; n++; }
      out[i] = n > 0 ? Math.sqrt(sumSq / n) * 2 : 0;
    }
    return out;
  }
  return out;
}

/** Direct ptp per bar from a decoded AudioBuffer (channel 0). */
function ptpPerBarFromBuffer(buffer, numBars) {
  const out = new Float32Array(numBars);
  const data = buffer.getChannelData(0);
  if (!data.length || numBars <= 0) return out;
  const samplesPerBar = Math.floor(data.length / numBars);
  if (samplesPerBar <= 0) return out;
  for (let i = 0; i < numBars; i++) {
    const s0 = i * samplesPerBar;
    const s1 = Math.min(s0 + samplesPerBar, data.length);
    let mx = -Infinity, mn = Infinity;
    for (let j = s0; j < s1; j++) {
      const v = data[j];
      if (v > mx) mx = v;
      if (v < mn) mn = v;
    }
    out[i] = mx === -Infinity ? 0 : (mx - mn);
  }
  return out;
}

export default CompositeBusWaveform;
