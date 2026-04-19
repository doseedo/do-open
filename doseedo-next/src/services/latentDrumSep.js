/**
 * Latent drum sub-separator — runs entirely in WebGPU via ONNX Runtime.
 *
 * Takes the drum stem latent from latent_demucs and splits into 6 sub-stems:
 * kick, snare, toms, hh, ride, crash.
 *
 * ONNX graph (inspected at runtime 2026-04-18):
 *   input  L_drum   float32 [1, 64, 64]        channels-first, FIXED T=64
 *   output L_stems  float32 [1, 6, 64, 64]     [B, N_stems, C, T] CT
 *
 * The graph's time axis is static at 64 frames (2.56 s @ 25 Hz). For any
 * longer drum latent we run inference in 64-frame non-overlapping chunks
 * and stitch the per-sub-stem outputs back together. Tail shorter than 64
 * is zero-padded and the pad is discarded on the read-back.
 *
 * Usage:
 *   import { initDrumSep, splitDrumLatent } from './latentDrumSep';
 *   await initDrumSep();
 *   const substems = await splitDrumLatent(drumLatentTD, T);
 *   // substems = { kick: Float32Array[T*64], snare: ..., ... }  // time-major
 */

import * as ort from 'onnxruntime-web';

// latent_drumsep.onnx is a self-contained 57 MB graph on R2 — no external
// .data sidecar. Served via the /static/models/* rewrite in next.config.js.
const MODEL_URL = '/static/models/latent_drumsep.onnx';
const STEMS = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash'];
const LATENT_DIM = 64;
const CHUNK = 64;             // graph's hard-coded T dimension

let session = null;

/** Load the drum sub-separator ONNX model. */
export async function initDrumSep() {
  if (session) return;
  try {
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu', 'wasm'],
    });
    console.log('[drumSep] ready');
  } catch (err) {
    console.error('[drumSep] failed to load:', err);
  }
}

/**
 * Split a drum latent into 6 sub-stem latents via chunked inference.
 *
 * @param {Float32Array} drumLatentTD  [T * 64] time-major drum stem latent
 *                                     (layout used by latentDrumTranscribe)
 * @param {number} T                   number of latent frames
 * @returns {Promise<Object>}          { kick, snare, toms, hh, ride, crash }
 *                                     each a Float32Array[T*64] time-major
 */
export async function splitDrumLatent(drumLatentTD, T) {
  if (!session) await initDrumSep();
  if (!session) throw new Error('drumSep not initialized');

  // Allocate per-stem time-major output buffers.
  const substems = {};
  for (const name of STEMS) substems[name] = new Float32Array(T * LATENT_DIM);

  const stemStride = LATENT_DIM * CHUNK;   // floats per sub-stem in the graph output

  for (let start = 0; start < T; start += CHUNK) {
    const Tc = Math.min(CHUNK, T - start);

    // Build chunk input as channels-first [1, 64, CHUNK]. Frames past Tc
    // stay zero — silence padding the model was trained to tolerate.
    const chunkCT = new Float32Array(LATENT_DIM * CHUNK);
    for (let t = 0; t < Tc; t++) {
      const srcRow = (start + t) * LATENT_DIM;
      for (let d = 0; d < LATENT_DIM; d++) {
        chunkCT[d * CHUNK + t] = drumLatentTD[srcRow + d];
      }
    }
    const input = new ort.Tensor('float32', chunkCT, [1, LATENT_DIM, CHUNK]);
    // Serialize against every other WebGPU ORT session on the page — ORT's
    // WebGPU EP shares one GPUDevice and throws "Session mismatch" on
    // overlapping .run() calls across sessions (latent_pitch, sem4Decoder,
    // latentEncoder all ride the same queue).
    const { ortWebGPURun } = await import('./webgpuOrtQueue');
    const results = await ortWebGPURun(() => session.run({ L_drum: input }));
    const out = results.L_stems.data;   // Float32Array length 1*6*64*64, CT per stem

    // Per sub-stem: read [64, CHUNK] CT → write back as [T_glob, 64] TD.
    for (let i = 0; i < STEMS.length; i++) {
      const stemBase = i * stemStride;
      const dst = substems[STEMS[i]];
      for (let t = 0; t < Tc; t++) {
        const dstRow = (start + t) * LATENT_DIM;
        for (let d = 0; d < LATENT_DIM; d++) {
          dst[dstRow + d] = out[stemBase + d * CHUNK + t];
        }
      }
    }
  }

  return substems;
}

/** Get the list of sub-stem names. */
export function getDrumStemNames() {
  return [...STEMS];
}
