/**
 * latent_pitch — stem latent → MIDI notes.
 *
 * Runs the LatentBasicPitchStudent ONNX (drop-in BasicPitch replacement)
 * on a [T, 64] VAE latent and produces a sorted array of
 * `{ note, time, duration, velocity }` events, compatible with the shape
 * produced by `utils/midiParser.js` so the DAW's existing MIDI window
 * renders it with no extra work.
 *
 * Model metadata (baked into pitch_054000.pt):
 *   • input:  `latent`  [B=1, T, 64]   time-major
 *   • outputs (all [B, T, 128]):
 *       - onset_logits    pre-sigmoid
 *       - frame_logits    pre-sigmoid
 *       - velocity        already sigmoid'd, [0,1]
 *       - onset_offset    already sigmoid'd, [0,1) sub-frame timing
 *   • max_len = 256 frames ≈ 10.24 s @ 25 Hz (positional encoding cap)
 *
 * We chunk into ≤CHUNK_FRAMES windows to stay inside that cap. The
 * transcription post-process mirrors `latent_pitch/infer.py`:
 *   onset NMS → extend while frame>thresh → min duration → velocity mean.
 */

import * as ort from 'onnxruntime-web';

const MODEL_URL     = '/static/models/latent_pitch.onnx';
const LATENT_CHANS  = 64;
const VAE_HZ        = 25;
const N_PITCH       = 128;
const CHUNK_FRAMES  = 256;  // ≤ max_len in the checkpoint's pos encoding
const ONSET_THRESH  = 0.7;
const FRAME_THRESH  = 0.5;
const MIN_NOTE_FRAMES = 2;
const NMS_RADIUS    = 2;

let _session = null;
let _sessionPromise = null;

/** Load the model (IndexedDB-cached, then session). Safe to call many times. */
export async function initLatentPitch() {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      ort.env.wasm.numThreads = Math.min(2, navigator.hardwareConcurrency || 1);
      ort.env.wasm.simd = true;
    }
    const t0 = performance.now();
    const { fetchModelWithCache } = await import('./modelCacheService');
    const bytes = await fetchModelWithCache(MODEL_URL);
    _session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    const ms = (performance.now() - t0).toFixed(0);
    console.log(`[latentPitch] model loaded in ${ms}ms (${(bytes.byteLength / (1 << 20)).toFixed(1)} MB)`);
    return _session;
  })();

  return _sessionPromise;
}

/**
 * Run the pitch model on one stem latent → MIDI notes.
 *
 * @param {Float32Array} latentCT   length LATENT_CHANS*T, channels-first [64,T]
 *                                  (this is what sem4Decoder emits per stem)
 * @param {number} T                number of latent frames
 * @returns {{notes: Array<{note,time,duration,velocity}>, duration: number}}
 */
export async function extractPitchFromLatent(latentCT, T) {
  if (!T || T < MIN_NOTE_FRAMES) return { notes: [], duration: 0 };
  const sess = await initLatentPitch();

  // Aggregate the full-length posteriors across chunks.
  const onset  = new Float32Array(T * N_PITCH);
  const frame  = new Float32Array(T * N_PITCH);
  const velocity = new Float32Array(T * N_PITCH);
  const offset = new Float32Array(T * N_PITCH);

  for (let start = 0; start < T; start += CHUNK_FRAMES) {
    const end = Math.min(T, start + CHUNK_FRAMES);
    const Tc = end - start;
    // Build [1, Tc, 64] TIME-MAJOR from channels-first [64, T] source.
    const chunk = new Float32Array(Tc * LATENT_CHANS);
    for (let t = 0; t < Tc; t++) {
      const gt = start + t;
      for (let d = 0; d < LATENT_CHANS; d++) {
        chunk[t * LATENT_CHANS + d] = latentCT[d * T + gt];
      }
    }
    const input = new ort.Tensor('float32', chunk, [1, Tc, LATENT_CHANS]);
    const out = await sess.run({ latent: input });

    const onLog = out.onset_logits.data;
    const frLog = out.frame_logits.data;
    const vel   = out.velocity.data;
    const off   = out.onset_offset.data;
    for (let t = 0; t < Tc; t++) {
      const dst = (start + t) * N_PITCH;
      const src = t * N_PITCH;
      for (let p = 0; p < N_PITCH; p++) {
        onset[dst + p]    = _sigmoid(onLog[src + p]);
        frame[dst + p]    = _sigmoid(frLog[src + p]);
        velocity[dst + p] = vel[src + p];
        offset[dst + p]   = off[src + p];
      }
    }
  }

  // Post-process posteriors → note events (mirrors infer.py:transcribe).
  const dt = 1.0 / VAE_HZ;
  const notes = [];

  for (let pitch = 0; pitch < N_PITCH; pitch++) {
    for (let t = 0; t < T; t++) {
      const on = onset[t * N_PITCH + pitch];
      if (on <= ONSET_THRESH) continue;
      // Local max within ±NMS_RADIUS frames for this pitch.
      let peak = true;
      const lo = Math.max(0, t - NMS_RADIUS);
      const hi = Math.min(T, t + NMS_RADIUS + 1);
      for (let k = lo; k < hi; k++) {
        if (k === t) continue;
        if (onset[k * N_PITCH + pitch] > on + 1e-9) { peak = false; break; }
      }
      if (!peak) continue;

      // Extend while frame probability stays above threshold.
      let endFrame = t + 1;
      while (endFrame < T && frame[endFrame * N_PITCH + pitch] > FRAME_THRESH) {
        endFrame++;
      }
      const nFrames = endFrame - t;
      if (nFrames < MIN_NOTE_FRAMES) continue;

      // Mean velocity across the sustained region (matches infer.py).
      let vSum = 0;
      for (let k = t; k < endFrame; k++) vSum += velocity[k * N_PITCH + pitch];
      const vMean = vSum / nFrames;
      const sub = offset[t * N_PITCH + pitch];  // [0,1) within frame

      notes.push({
        note: pitch,
        time: (t + sub) * dt,
        duration: nFrames * dt,
        velocity: Math.max(1, Math.min(127, Math.round(vMean * 127))),
      });
    }
  }

  notes.sort((a, b) => a.time - b.time);
  const duration = T * dt;
  return { notes, duration };
}

function _sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
