/**
 * Latent drum sub-separator — runs entirely in WebGPU via ONNX Runtime.
 *
 * Takes the drum stem latent from latentDemucs and splits into
 * 6 sub-stems: kick, snare, toms, hh, ride, crash.
 *
 * Usage:
 *   import { initDrumSep, splitDrumLatent } from './latentDrumSep';
 *   await initDrumSep();
 *   const substems = await splitDrumLatent(drumLatentFloat32, T);
 *   // substems = { kick: Float32Array, snare: Float32Array, ... }
 */

import * as ort from 'onnxruntime-web';

// latent_drumsep.onnx is a self-contained 57 MB graph on R2 — no external
// .data sidecar exists. Served via the /static/models/* rewrite in
// next.config.js (same path convention as the other ONNX models).
const MODEL_URL = '/static/models/latent_drumsep.onnx';
const STEMS = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash'];
const LATENT_DIM = 64;

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
 * Split a drum latent into 6 sub-stem latents.
 *
 * @param {Float32Array} drumLatent - [T * 64] flattened drum latent
 * @param {number} T - number of latent frames
 * @returns {Object} { kick: Float32Array, snare: ..., toms: ..., hh: ..., ride: ..., crash: ... }
 *          Each value is [T * 64] flattened latent for that sub-stem.
 */
export async function splitDrumLatent(drumLatent, T) {
  if (!session) await initDrumSep();
  if (!session) throw new Error('drumSep not initialized');

  // Input: [1, T, 64]
  const input = new ort.Tensor('float32', drumLatent, [1, T, LATENT_DIM]);
  const results = await session.run({ drum_latent: input });
  const output = results.substem_latents; // [1, 6, T, 64]

  // Split into per-stem arrays
  const substems = {};
  const stride = T * LATENT_DIM;
  for (let i = 0; i < STEMS.length; i++) {
    substems[STEMS[i]] = output.data.slice(i * stride, (i + 1) * stride);
  }

  return substems;
}

/**
 * Get the list of sub-stem names.
 */
export function getDrumStemNames() {
  return [...STEMS];
}
