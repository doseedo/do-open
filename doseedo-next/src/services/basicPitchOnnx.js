/*
 * basicPitchOnnx — Spotify BasicPitch as a WebGPU ONNX service.
 *
 * Drop nmp.onnx (from the basic-pitch pip package, under
 * `basic_pitch/saved_models/icassp_2022/nmp.onnx`) into
 * `public/static/models/basic_pitch.onnx`. The model is ~17 MB float32,
 * fully convolutional (harmonic stacking + 2D conv blocks — no
 * attention), so it runs comfortably on WebGPU at typical stem sizes.
 *
 * Unlike latentPitch (which consumes 25-fps VAE latents), BasicPitch
 * takes RAW AUDIO at 22050 Hz mono and emits onset / contour / frame
 * probability maps at ~86 fps. CQT + harmonic stacking are baked into
 * the exported graph — no client-side CQT needed.
 *
 * API:
 *   await initBasicPitch();
 *   const { notes, duration } = await transcribeAudio(audioFloat32, sr);
 *   // notes: [{note, time, duration, velocity}, ...] — same shape as
 *   //        latentPitch + @tonejs/midi output so the piano roll and
 *   //        chord detector consume it without change.
 */

import * as ort from 'onnxruntime-web';

const MODEL_URL         = '/static/models/basic_pitch.onnx';
const MODEL_DATA_URL    = '/static/models/basic_pitch.onnx.data';
const TARGET_SR         = 22050;
const CHUNK_SAMPLES     = 43844;                          // ≈2s chunks
const CHUNK_FRAMES      = 172;                            // model's time dim
const FPS               = CHUNK_FRAMES / (CHUNK_SAMPLES / TARGET_SR);  // ≈86.13
const OVERLAP_SAMPLES   = 4096;                           // ≈186ms
const MIDI_OFFSET       = 21;                             // A0
const N_PITCH           = 88;

// Paper defaults; tune via opts.onsetThresh / opts.frameThresh when calling.
const ONSET_THRESH          = 0.5;
const FRAME_THRESH          = 0.3;
const MIN_NOTE_LEN_FRAMES   = 11;                         // ≈128ms

let session = null;
let outputKeys = null;
let initFailed = false;

/**
 * Lazy-load the ONNX session. Safe to call repeatedly. Returns null if
 * the model file is missing — the pipeline falls back to latent_pitch
 * gracefully in that case.
 */
export async function initBasicPitch() {
  if (session) return session;
  if (initFailed) return null;
  try {
    // Optional external-data sibling: basic_pitch.onnx.data. If present
    // the loader passes it in; if 404 we load the model alone (assumes
    // weights are inline, which is the default for nmp.onnx at 17 MB).
    let externalData;
    try {
      const dataResp = await fetch(MODEL_DATA_URL);
      if (dataResp.ok) {
        const dataBytes = new Uint8Array(await dataResp.arrayBuffer());
        externalData = [{ path: 'basic_pitch.onnx.data', data: dataBytes.buffer }];
      }
    } catch (_) { /* inline weights — fine */ }

    const modelResp = await fetch(MODEL_URL);
    if (!modelResp.ok) {
      initFailed = true;
      console.warn(
        `[basicPitch] model missing at ${MODEL_URL} (HTTP ${modelResp.status}). ` +
        'Client-side BasicPitch is disabled. To enable, download nmp.onnx from ' +
        'the basic-pitch pip package (basic_pitch/saved_models/icassp_2022/nmp.onnx) ' +
        'and place it at public/static/models/basic_pitch.onnx.'
      );
      return null;
    }
    const modelBytes = new Uint8Array(await modelResp.arrayBuffer());
    session = await ort.InferenceSession.create(modelBytes, {
      executionProviders: ['webgpu', 'wasm'],
      ...(externalData ? { externalData } : {}),
    });
    outputKeys = mapOutputs(session);
    if (outputKeys.needsShapeProbe) {
      outputKeys = await probeShapes(session);
    }
    console.log(
      `[basicPitch] ready (${(modelBytes.byteLength / 1024).toFixed(0)} KB). ` +
      `onset=${outputKeys.onset} frame=${outputKeys.frame} contour=${outputKeys.contour}`
    );
    return session;
  } catch (err) {
    initFailed = true;
    console.warn('[basicPitch] init failed:', err?.message || err);
    return null;
  }
}

/**
 * Map session.outputNames to semantic roles.
 *
 * nmp.onnx (TF SavedModel exported via tf2onnx, confirmed with onnx.load()):
 *   StatefulPartitionedCall:0  [B, 172, 264]  contour  (3 bins per semitone)
 *   StatefulPartitionedCall:1  [B, 172,  88]  note     (frame activity)
 *   StatefulPartitionedCall:2  [B, 172,  88]  onset    (onset probabilities)
 *
 * All three are POST-SIGMOID in the graph (the model has 3 Sigmoid ops
 * at the tail), so outputs are already probabilities in [0,1]. Do NOT
 * apply an extra sigmoid in the decoder.
 *
 * We pin by name to avoid any ordering ambiguity, with a shape-based
 * fallback in case someone re-exports the model with different names.
 */
function mapOutputs(sess) {
  const names = sess.outputNames;
  const haveStandard =
    names.includes('StatefulPartitionedCall:0') &&
    names.includes('StatefulPartitionedCall:1') &&
    names.includes('StatefulPartitionedCall:2');
  if (haveStandard) {
    return {
      contour: 'StatefulPartitionedCall:0',
      frame:   'StatefulPartitionedCall:1',
      onset:   'StatefulPartitionedCall:2',
      needsShapeProbe: false,
    };
  }
  // Unknown export: probe with zero input to inspect output widths.
  return { needsShapeProbe: true, names };
}

async function probeShapes(sess) {
  const zeroIn = new ort.Tensor('float32', new Float32Array(CHUNK_SAMPLES), [1, CHUNK_SAMPLES, 1]);
  const out = await sess.run({ [sess.inputNames[0]]: zeroIn });
  let onset = null, frame = null, contour = null;
  for (const n of sess.outputNames) {
    const w = out[n].dims[out[n].dims.length - 1];
    if (w === 264 && !contour) contour = n;
    else if (w === 88 && !onset) onset = n;
    else if (w === 88 && !frame) frame = n;
  }
  return { onset, frame, contour, needsShapeProbe: false };
}

/**
 * Resample to 22050 mono via OfflineAudioContext (browser-native
 * anti-aliased). Accepts Float32Array at any sample rate; returns
 * mono Float32Array at TARGET_SR.
 */
async function toMono22050(audio, sr) {
  if (!(audio instanceof Float32Array)) audio = Float32Array.from(audio);
  if (Math.abs(sr - TARGET_SR) < 1) return audio;
  const OfflineCtx = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const outLen = Math.ceil(audio.length * TARGET_SR / sr);
  const ctx = new OfflineCtx(1, outLen, TARGET_SR);
  const buf = ctx.createBuffer(1, audio.length, sr);
  buf.copyToChannel(audio, 0);
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(ctx.destination);
  src.start();
  const rendered = await ctx.startRendering();
  return rendered.getChannelData(0).slice();
}

/**
 * Transcribe audio to MIDI notes via BasicPitch.
 *
 * Chunks input into 2s windows with ~200ms overlap, runs inference per
 * chunk, averages overlapping frames, decodes onset/frame probability
 * maps to note events via a standard greedy algorithm (onset peak →
 * extend while frame above threshold → velocity = mean of frame probs).
 *
 * @param {Float32Array|Array} audio - mono PCM
 * @param {number} sr - input sample rate
 * @param {Object} [opts]
 * @param {number} [opts.onsetThresh=0.5]
 * @param {number} [opts.frameThresh=0.3]
 * @param {number} [opts.minNoteLenFrames=11]
 * @returns {Promise<{notes: Array, duration: number}>}
 */
export async function transcribeAudio(audio, sr, opts = {}) {
  const sess = await initBasicPitch();
  if (!sess) return { notes: [], duration: 0 };
  const audio22k = await toMono22050(audio, sr);
  const duration = audio22k.length / TARGET_SR;

  // Total frames at ~86 fps, with overlap averaging.
  const totalFrames = Math.ceil(duration * FPS) + CHUNK_FRAMES;
  const onsetAcc = new Float32Array(totalFrames * N_PITCH);
  const frameAcc = new Float32Array(totalFrames * N_PITCH);
  const countAcc = new Float32Array(totalFrames);

  const HOP = CHUNK_SAMPLES - OVERLAP_SAMPLES;

  for (let start = 0; start < audio22k.length; start += HOP) {
    const chunk = new Float32Array(CHUNK_SAMPLES);
    const end = Math.min(start + CHUNK_SAMPLES, audio22k.length);
    chunk.set(audio22k.subarray(start, end));
    const inputTensor = new ort.Tensor('float32', chunk, [1, CHUNK_SAMPLES, 1]);
    const out = await sess.run({ [sess.inputNames[0]]: inputTensor });
    // Outputs are already sigmoid'd inside the graph — copy straight in.
    const onset = out[outputKeys.onset].data;
    const frame = out[outputKeys.frame].data;
    const T = out[outputKeys.onset].dims[1];
    const baseFrame = Math.round((start / TARGET_SR) * FPS);
    for (let t = 0; t < T; t++) {
      const dt = baseFrame + t;
      if (dt >= totalFrames) break;
      countAcc[dt] += 1;
      for (let p = 0; p < N_PITCH; p++) {
        const srcIdx = t * N_PITCH + p;
        const dstIdx = dt * N_PITCH + p;
        onsetAcc[dstIdx] += onset[srcIdx];
        frameAcc[dstIdx] += frame[srcIdx];
      }
    }
  }

  // Normalize overlap regions so averaged probs are comparable to
  // single-chunk probs at the thresholds below.
  for (let t = 0; t < totalFrames; t++) {
    const c = countAcc[t] || 1;
    if (c === 1) continue;
    for (let p = 0; p < N_PITCH; p++) {
      const idx = t * N_PITCH + p;
      onsetAcc[idx] /= c;
      frameAcc[idx] /= c;
    }
  }

  const notes = decodeNotes(onsetAcc, frameAcc, totalFrames, opts);
  return { notes, duration };
}

/**
 * Greedy onset→frame decoder.
 *
 * For each frame × pitch:
 *   - If pitch is inactive and onsetProb > ONSET_THRESH and it's a
 *     local peak (vs t-1, t+1) → start a note.
 *   - While active and frameProb > FRAME_THRESH, accumulate velocity.
 *   - When frameProb drops, close the note. Keep if duration ≥ MIN_LEN.
 *
 * Velocity = mean frameProb × 127. The contour head is ignored here —
 * it carries pitch-bend / vibrato info we don't use for chord detection
 * or piano-roll display. If we later add microtonal readouts, we'd
 * consume out[outputKeys.contour] instead of dropping it.
 */
function decodeNotes(onsetMap, frameMap, totalFrames, opts = {}) {
  const ONSET_T = opts.onsetThresh ?? ONSET_THRESH;
  const FRAME_T = opts.frameThresh ?? FRAME_THRESH;
  const MIN_LEN = opts.minNoteLenFrames ?? MIN_NOTE_LEN_FRAMES;

  const notes = [];
  const active = new Map(); // pitch → {startFrame, velSum, velCount}

  const close = (p, endFrame) => {
    const n = active.get(p);
    const dur = endFrame - n.startFrame;
    if (dur >= MIN_LEN) {
      notes.push({
        note: p + MIDI_OFFSET,
        time: n.startFrame / FPS,
        duration: dur / FPS,
        velocity: Math.max(1, Math.min(127,
          Math.round((n.velSum / Math.max(1, n.velCount)) * 127))),
      });
    }
    active.delete(p);
  };

  for (let t = 0; t < totalFrames; t++) {
    for (let p = 0; p < N_PITCH; p++) {
      const idx = t * N_PITCH + p;
      const on = onsetMap[idx];
      const fr = frameMap[idx];
      if (!active.has(p)) {
        if (on > ONSET_T) {
          const prev = t > 0 ? onsetMap[(t - 1) * N_PITCH + p] : 0;
          const next = t + 1 < totalFrames ? onsetMap[(t + 1) * N_PITCH + p] : 0;
          if (on >= prev && on >= next) {
            active.set(p, { startFrame: t, velSum: fr, velCount: 1 });
          }
        }
      } else {
        if (fr > FRAME_T) {
          const cur = active.get(p);
          cur.velSum += fr;
          cur.velCount += 1;
        } else {
          close(p, t);
        }
      }
    }
  }
  for (const p of Array.from(active.keys())) close(p, totalFrames);

  notes.sort((a, b) => a.time - b.time);
  return notes;
}
