import React, { useEffect, useRef, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import { audioBufferCache } from '../../hooks/useWaveform';

/**
 * CompositeBusWaveform - a single canvas rendering of a bus's "master"
 * waveform when the bus is collapsed, computed as the volume-weighted sum
 * of every stem's envelope (or decoded RMS). When any stem slider moves,
 * React re-renders this component with fresh per-stem gains and the canvas
 * repaints in real time.
 *
 * Aggregation math (per-bar, time-aligned across stems):
 *   effGain(stem)  = isMuted ? 0
 *                  : (anySolo && !stem.isSolo) ? 0
 *                  : stem.gain
 *   masterPeak[t]  = (busMute ? 0 : busGain) * Σ effGain(stem) * stem.peak[t]
 *
 * Stems contribute via `track.metadata.envelopeData` (from latent_visual) at
 * `envelopeFps`. If a stem has no envelope yet but its audio is decoded,
 * we fall back to the cached AudioBuffer RMS. If neither is available
 * (rare — pre-latent), the stem contributes 0 to the sum.
 */
const CompositeBusWaveform = React.memo(({
  bus,
  height,
  envelopeFps = 25,
  color = '#667eea',
}) => {
  const { state } = useApp();
  const canvasRef = useRef(null);

  // Match OptimizedTrack's sizing exactly so the composite spans the same
  // horizontal footprint a single expanded stem would occupy.
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

  // Derive the reactive inputs (per-stem gain/mute/solo + peak source).
  // This runs on every dispatch, which is exactly when we want a repaint.
  // We explicitly take ONLY stems here — the original uploaded audio
  // (metadata.type === 'uploaded') is kept in bus.tracks to preserve
  // analysis data + mask-playback audio but MUST NOT contribute to the
  // composite, or it would double-count on top of the stem sum.
  // If no stems exist yet we fall back to any non-MIDI, non-placeholder
  // track (the single-file case BEFORE separation completes).
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

  const busGain = bus.mute ? 0 : (bus.gain ?? 1.0);

  // Paint whenever gain/envelope/width/height changes.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    if (stemInputs.length === 0) return;

    // Match the stem waveform style exactly (useWaveform.renderEnvelope):
    // 2px bars, 2px spacing, rounded caps, alpha 0.7.
    const barSpacing = 2;
    const barWidth = 2;
    const numBars = Math.floor(width / (barWidth + barSpacing));
    if (numBars <= 0) return;
    const midY = height / 2;
    const maxBarHeight = height * 0.45;

    // Per-stem peaks at native resolution (no gain applied). Envelope
    // preferred; decoded RMS fallback; zero if neither.
    const perStemBars = stemInputs.map(s => peakPerBar(s, numBars, envelopeFps));

    // CONSTANT baseline: Σ peak[i] if every stem were at gain=1.0. We
    // normalize to THIS max, not to the current live sum. That's the
    // only way gain-down-the-loudest-stem correctly shrinks the master:
    // otherwise the renormalization pumps the remaining (quieter) stems
    // UP to fill the container, and the master looks bigger when you
    // turn the dominant stem down. Ask me how I know.
    const baselineSummed = new Float32Array(numBars);
    for (let s = 0; s < perStemBars.length; s++) {
      const peaks = perStemBars[s];
      if (!peaks) continue;
      for (let i = 0; i < numBars; i++) baselineSummed[i] += peaks[i];
    }
    let baselineMax = 0;
    for (let i = 0; i < numBars; i++) {
      if (baselineSummed[i] > baselineMax) baselineMax = baselineSummed[i];
    }
    // Peak bars fill the full track height at the busGain=1 baseline.
    // Matches the single-track view (renderWaveform RMS) so the
    // composite doesn't visibly shrink when stem separation lands.
    const norm = baselineMax > 0.001 ? maxBarHeight / baselineMax : 1.0;
    const noiseFloor = baselineMax * 0.02;

    // Live sum = Σ peak[i] × stem.gain (respecting mute/solo). This is
    // what we actually draw. The constant `norm` means reducing a stem's
    // gain linearly shrinks its contribution to the displayed bar height
    // without pumping the others.
    const summed = new Float32Array(numBars);
    for (let s = 0; s < perStemBars.length; s++) {
      const peaks = perStemBars[s];
      const g = stemInputs[s].gain;
      if (!peaks || g === 0) continue;
      for (let i = 0; i < numBars; i++) summed[i] += peaks[i] * g;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = barWidth;
    ctx.lineCap = 'round';
    ctx.globalAlpha = 0.7;

    for (let i = 0; i < numBars; i++) {
      const x = i * (barWidth + barSpacing);
      const v = summed[i];
      if (v < noiseFloor || busGain === 0) {
        ctx.beginPath();
        ctx.moveTo(x + barWidth / 2, midY - 0.5);
        ctx.lineTo(x + barWidth / 2, midY + 0.5);
        ctx.stroke();
        continue;
      }
      const h = Math.min(v * norm * busGain, maxBarHeight);
      ctx.beginPath();
      ctx.moveTo(x + barWidth / 2, midY - h);
      ctx.lineTo(x + barWidth / 2, midY + h);
      ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
  }, [stemInputs, busGain, width, height, envelopeFps, color]);

  // Render the canvas at EXACT pixel size (no CSS stretching) so the
  // bar resolution matches individual stem waveforms 1:1. The width
  // in pixels equals busDuration × pixelsPerSecond, same math that
  // OptimizedTrack uses for stem tracks.
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

/**
 * Return a Float32Array[numBars] of peak-to-peak magnitudes for one stem,
 * time-aligned across the bar grid. Uses envelope if present, else
 * decoded AudioBuffer RMS, else an all-zero array.
 */
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
      out[i] = n > 0 ? Math.sqrt(sumSq / n) * 2 : 0; // ×2 to approximate ptp
    }
    return out;
  }
  return out;
}

export default CompositeBusWaveform;
