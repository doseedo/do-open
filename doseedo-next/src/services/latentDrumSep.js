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

const MODEL_URL = '/models/latent_drumsep.onnx';
const MODEL_DATA_URL = '/models/latent_drumsep.onnx.data';
const STEMS = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash'];
const LATENT_DIM = 64;

let session = null;

/**
 * Load the drum sub-separator ONNX model.
 * Fetches the external .onnx.data weights file and passes it
 * via the externalData option.
 */
export async function initDrumSep() {
  if (session) return;
  try {
    const dataResp = await fetch(MODEL_DATA_URL);
    if (!dataResp.ok) throw new Error(`latent_drumsep.onnx.data HTTP ${dataResp.status}`);
    const dataBytes = new Uint8Array(await dataResp.arrayBuffer());
    const externalData = [{ path: 'latent_drumsep.onnx.data', data: dataBytes.buffer }];

    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ['webgpu', 'wasm'],
      externalData,
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
