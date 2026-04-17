/**
 * Latent-domain meter change — runs entirely in WebGPU.
 *
 * Replaces the backend /api/repaint-meter call for all stem types.
 * The pipeline:
 *   1. Parse bar boundaries from BPM + meter
 *   2. For each bar, compute target bar length in the new meter
 *   3. Stretch/compress latent frames per bar (linear interp)
 *   4. Concatenate → new latent with changed meter
 *   5. (Optional) Run latent editor for boundary repair
 *
 * No backend call needed. All latent manipulation + decode happens in browser.
 */

import * as ort from 'onnxruntime-web';
import { initDrumSep, splitDrumLatent } from './latentDrumSep';

const EDITOR_URL = '/models/latent_editor.onnx';
const EDITOR_DATA_URL = '/models/latent_editor.onnx.data';
const LATENT_DIM = 64;
const FPS = 25; // latent frames per second

let editorSession = null;

/**
 * Load the latent editor ONNX model for boundary repair.
 * Fetches the external .onnx.data weights file and passes it
 * via the externalData option (same pattern as latentDemucs).
 */
export async function initLatentEditor() {
  if (editorSession) return;
  try {
    const dataResp = await fetch(EDITOR_DATA_URL);
    if (!dataResp.ok) throw new Error(`latent_editor.onnx.data HTTP ${dataResp.status}`);
    const dataBytes = new Uint8Array(await dataResp.arrayBuffer());
    const externalData = [{ path: 'latent_editor.onnx.data', data: dataBytes.buffer }];

    editorSession = await ort.InferenceSession.create(EDITOR_URL, {
      executionProviders: ['webgpu', 'wasm'],
      externalData,
    });
    console.log('[meterChange] editor ready');
  } catch (err) {
    console.warn('[meterChange] editor load failed, splice-only mode:', err);
  }
}

/**
 * Compute bar start frames from BPM and meter.
 *
 * @param {number} bpm
 * @param {number} beatsPerBar - numerator (e.g. 4 for 4/4, 7 for 7/8)
 * @param {number} beatUnit - denominator (e.g. 4 for quarter, 8 for eighth)
 * @param {number} totalFrames - total latent frames
 * @returns {number[]} - frame indices where each bar starts
 */
function computeBarStarts(bpm, beatsPerBar, beatUnit, totalFrames) {
  // One beat = 60/bpm seconds. One bar = beatsPerBar * (4/beatUnit) beats
  const beatSec = 60.0 / bpm;
  const barSec = beatsPerBar * (4.0 / beatUnit) * beatSec;
  const barFrames = barSec * FPS;

  const starts = [0];
  let pos = barFrames;
  while (pos < totalFrames) {
    starts.push(Math.round(pos));
    pos += barFrames;
  }
  starts.push(totalFrames);
  return starts;
}

/**
 * Linear interpolation stretch of a latent segment.
 *
 * @param {Float32Array} segment - [srcFrames * 64] flattened
 * @param {number} srcFrames
 * @param {number} tgtFrames
 * @returns {Float32Array} - [tgtFrames * 64] flattened
 */
function stretchSegment(segment, srcFrames, tgtFrames) {
  if (srcFrames === tgtFrames) return segment;
  if (srcFrames <= 0 || tgtFrames <= 0) return new Float32Array(tgtFrames * LATENT_DIM);

  const out = new Float32Array(tgtFrames * LATENT_DIM);
  for (let t = 0; t < tgtFrames; t++) {
    const srcPos = (t / Math.max(tgtFrames - 1, 1)) * (srcFrames - 1);
    const lo = Math.floor(srcPos);
    const hi = Math.min(lo + 1, srcFrames - 1);
    const frac = srcPos - lo;
    for (let d = 0; d < LATENT_DIM; d++) {
      out[t * LATENT_DIM + d] =
        segment[lo * LATENT_DIM + d] * (1 - frac) +
        segment[hi * LATENT_DIM + d] * frac;
    }
  }
  return out;
}

/**
 * Apply latent editor boundary repair at splice points.
 *
 * @param {Float32Array} latent - [T * 64] full spliced latent
 * @param {number} T - total frames
 * @param {number[]} spliceFrames - frame indices where splices occurred
 * @param {number} radius - frames around each splice to repair (default 4)
 * @returns {Float32Array} - repaired latent
 */
async function repairBoundaries(latent, T, spliceFrames, radius = 4) {
  if (!editorSession || spliceFrames.length === 0) return latent;

  // Build boundary mask [T]
  const mask = new Float32Array(T);
  for (const sf of spliceFrames) {
    for (let r = -radius; r <= radius; r++) {
      const idx = sf + r;
      if (idx >= 0 && idx < T) {
        mask[idx] = 1.0;
      }
    }
  }

  const latentTensor = new ort.Tensor('float32', latent, [1, T, LATENT_DIM]);
  const maskTensor = new ort.Tensor('float32', mask, [1, T]);
  const phaseTensor = new ort.Tensor('float32', new Float32Array([0.0]), [1]);

  const results = await editorSession.run({
    latent: latentTensor,
    boundary_mask: maskTensor,
    phase: phaseTensor,
  });

  return results.edited_latent.data;
}

/**
 * Snap a percussive substem to the new meter grid instead of smearing via lerp.
 *
 * For each source bar, identifies beat positions in the source meter, maps them
 * to the closest beat in the target meter, and places the hit latent frames at
 * the new positions.  Frames between hits are zeroed (silence) rather than
 * interpolated, which preserves transient sharpness.
 *
 * @param {Float32Array} segment - [srcBarFrames * 64] one bar of substem latent
 * @param {number} srcBarFrames
 * @param {number} tgtBarFrames
 * @param {number} srcBeatsPerBar
 * @param {number} tgtBeatsPerBar
 * @param {number} hitRadius - frames around each beat center to copy (default 2)
 * @returns {Float32Array} - [tgtBarFrames * 64]
 */
function snapBeatsInBar(segment, srcBarFrames, tgtBarFrames, srcBeatsPerBar, tgtBeatsPerBar, hitRadius = 2) {
  const out = new Float32Array(tgtBarFrames * LATENT_DIM);

  for (let srcBeat = 0; srcBeat < srcBeatsPerBar; srcBeat++) {
    // Source beat position (frame index within bar)
    const srcCenter = Math.round((srcBeat / srcBeatsPerBar) * srcBarFrames);

    // Map to nearest target beat
    const tgtBeat = Math.round((srcBeat / srcBeatsPerBar) * tgtBeatsPerBar);
    if (tgtBeat >= tgtBeatsPerBar) continue;
    const tgtCenter = Math.round((tgtBeat / tgtBeatsPerBar) * tgtBarFrames);

    // Copy a small window around the hit
    for (let r = -hitRadius; r <= hitRadius; r++) {
      const si = srcCenter + r;
      const ti = tgtCenter + r;
      if (si < 0 || si >= srcBarFrames || ti < 0 || ti >= tgtBarFrames) continue;
      for (let d = 0; d < LATENT_DIM; d++) {
        out[ti * LATENT_DIM + d] = segment[si * LATENT_DIM + d];
      }
    }
  }
  return out;
}

/**
 * Meter change specifically for drum stems:
 *   1. Split drum latent → 6 substems (kick, snare, toms, hh, ride, crash)
 *   2. Percussive substems (kick, snare, toms) get beat-snapped per bar
 *   3. Sustain substems (hh, ride, crash) get linear-stretched (like pads)
 *   4. Recombine by summing all substems
 *   5. Boundary repair via latent editor
 *
 * @param {Float32Array} drumLatent - [T_src * 64]
 * @param {number} T_src
 * @param {number} srcBpm
 * @param {number} tgtBpm
 * @param {number[]} srcMeter - [beatsPerBar, beatUnit]
 * @param {number[]} tgtMeter - [beatsPerBar, beatUnit]
 * @returns {{ latent: Float32Array, T: number }}
 */
async function meterChangeDrumStem(drumLatent, T_src, srcBpm, tgtBpm, srcMeter, tgtMeter) {
  const [srcBeats, srcDen] = srcMeter;
  const [tgtBeats, tgtDen] = tgtMeter;
  const t0 = performance.now();

  // 1. Split into 6 substems
  await initDrumSep();
  const substems = await splitDrumLatent(drumLatent, T_src);
  console.log(`[meterChange] drum split → ${Object.keys(substems).length} substems in ${(performance.now() - t0).toFixed(0)}ms`);

  // Percussive vs sustain classification
  const PERCUSSIVE = new Set(['kick', 'snare', 'toms']);

  // 2. Compute bar structure
  const srcBarStarts = computeBarStarts(srcBpm, srcBeats, srcDen, T_src);
  const nBars = srcBarStarts.length - 1;

  const tgtBeatSec = 60.0 / tgtBpm;
  const tgtBarSec = tgtBeats * (4.0 / tgtDen) * tgtBeatSec;
  const tgtBarFrames = Math.round(tgtBarSec * FPS);

  // Source beats per bar (in quarter-note equivalent)
  const srcBeatsActual = srcBeats * (4.0 / srcDen);
  const tgtBeatsActual = tgtBeats * (4.0 / tgtDen);

  // 3. Process each substem
  const processedSubstems = {};
  const T_tgt = nBars * tgtBarFrames;

  for (const [name, subLatent] of Object.entries(substems)) {
    const bars = [];
    const isPercussive = PERCUSSIVE.has(name);

    for (let b = 0; b < nBars; b++) {
      const srcStart = srcBarStarts[b];
      const srcEnd = srcBarStarts[b + 1];
      const srcBarLen = srcEnd - srcStart;
      const segment = subLatent.slice(srcStart * LATENT_DIM, srcEnd * LATENT_DIM);

      if (isPercussive) {
        // Beat-snap: place hits on nearest target grid positions
        bars.push(snapBeatsInBar(segment, srcBarLen, tgtBarFrames, srcBeatsActual, tgtBeatsActual));
      } else {
        // Sustain (hh/ride/crash): linear stretch like any other stem
        bars.push(stretchSegment(segment, srcBarLen, tgtBarFrames));
      }
    }

    // Concatenate bars
    const full = new Float32Array(T_tgt * LATENT_DIM);
    let pos = 0;
    for (const bar of bars) {
      full.set(bar, pos);
      pos += bar.length;
    }
    processedSubstems[name] = full;
  }

  // 4. Recombine by summing all substems
  const combined = new Float32Array(T_tgt * LATENT_DIM);
  for (const sub of Object.values(processedSubstems)) {
    for (let i = 0; i < combined.length; i++) {
      combined[i] += sub[i];
    }
  }

  // 5. Boundary repair at bar joints
  const spliceFrames = [];
  for (let b = 1; b < nBars; b++) {
    spliceFrames.push(b * tgtBarFrames);
  }
  const repaired = await repairBoundaries(combined, T_tgt, spliceFrames);

  const elapsed = performance.now() - t0;
  console.log(`[meterChange] drum: ${T_src}→${T_tgt} frames, ${nBars} bars, 6 substems (3 snapped + 3 stretched), ${elapsed.toFixed(0)}ms`);
  return { latent: repaired, T: T_tgt };
}

/**
 * Perform meter change on a single stem latent — entirely in browser.
 *
 * @param {Float32Array} srcLatent - [T_src * 64] source latent
 * @param {number} T_src - source frame count
 * @param {number} srcBpm
 * @param {number} tgtBpm
 * @param {number[]} srcMeter - [beatsPerBar, beatUnit] e.g. [4, 4]
 * @param {number[]} tgtMeter - [beatsPerBar, beatUnit] e.g. [7, 8]
 * @returns {{ latent: Float32Array, T: number }}
 */
export async function meterChangeStem(srcLatent, T_src, srcBpm, tgtBpm, srcMeter, tgtMeter) {
  const [srcBeats, srcDen] = srcMeter;
  const [tgtBeats, tgtDen] = tgtMeter;
  const t0 = performance.now();

  // Compute source bar boundaries
  const srcBarStarts = computeBarStarts(srcBpm, srcBeats, srcDen, T_src);
  const nBars = srcBarStarts.length - 1;

  // Compute target bar length in frames
  const tgtBeatSec = 60.0 / tgtBpm;
  const tgtBarSec = tgtBeats * (4.0 / tgtDen) * tgtBeatSec;
  const tgtBarFrames = Math.round(tgtBarSec * FPS);

  // Process each bar: extract → stretch to target length
  const barOutputs = [];
  const spliceFrames = [];
  let outOffset = 0;

  for (let b = 0; b < nBars; b++) {
    const srcStart = srcBarStarts[b];
    const srcEnd = srcBarStarts[b + 1];
    const srcBarLen = srcEnd - srcStart;

    // Extract bar segment
    const segment = srcLatent.slice(
      srcStart * LATENT_DIM,
      srcEnd * LATENT_DIM
    );

    // Stretch to target bar length
    const stretched = stretchSegment(segment, srcBarLen, tgtBarFrames);
    barOutputs.push(stretched);

    if (b > 0) {
      spliceFrames.push(outOffset);
    }
    outOffset += tgtBarFrames;
  }

  // Concatenate all bars
  const T_tgt = outOffset;
  const result = new Float32Array(T_tgt * LATENT_DIM);
  let writePos = 0;
  for (const bar of barOutputs) {
    result.set(bar, writePos);
    writePos += bar.length;
  }

  // Repair splice boundaries with latent editor
  const repaired = await repairBoundaries(result, T_tgt, spliceFrames);

  const elapsed = performance.now() - t0;
  console.log(`[meterChange] stem: ${T_src}→${T_tgt} frames, ${nBars} bars, ${spliceFrames.length} splices, ${elapsed.toFixed(0)}ms`);
  return { latent: repaired, T: T_tgt };
}

/**
 * Meter change for all stems at once.
 * Each stem is processed independently but shares the same bar structure.
 *
 * @param {Object} stemLatents - { drums: { data: Float32Array, T: number }, bass: ..., ... }
 * @param {number} srcBpm
 * @param {number} tgtBpm
 * @param {number[]} srcMeter
 * @param {number[]} tgtMeter
 * @returns {Object} - { drums: { latent: Float32Array, T: number }, ... }
 */
export async function meterChangeAllStems(stemLatents, srcBpm, tgtBpm, srcMeter, tgtMeter) {
  await initLatentEditor();

  const t0 = performance.now();
  const results = {};
  for (const [stemName, { data, T }] of Object.entries(stemLatents)) {
    const isDrum = stemName === 'drums' || stemName === 'drum_kit';
    console.log(`[meterChange] processing ${stemName} (${T} frames)${isDrum ? ' [drum-aware]' : ''}...`);

    if (isDrum) {
      // Drum-aware path: split → beat-snap percussive / stretch sustain → recombine
      try {
        results[stemName] = await meterChangeDrumStem(data, T, srcBpm, tgtBpm, srcMeter, tgtMeter);
      } catch (err) {
        console.warn(`[meterChange] drum-aware path failed, falling back to linear stretch:`, err?.message || err);
        results[stemName] = await meterChangeStem(data, T, srcBpm, tgtBpm, srcMeter, tgtMeter);
      }
    } else {
      results[stemName] = await meterChangeStem(data, T, srcBpm, tgtBpm, srcMeter, tgtMeter);
    }
  }
  console.log(`[meterChange] all stems done in ${(performance.now() - t0).toFixed(0)}ms`);
  return results;
}
