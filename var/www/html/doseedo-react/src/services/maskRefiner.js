/**
 * Client-side LatentMaskRefiner — final step in the upload pipeline.
 *
 * Takes the clean stem latent from v4cond-pred and the noisy per-stem
 * STFT mask from v4-small, produces a refined mask:
 *
 *   latent      [1, 64, T]         clean, from latentDemucsV4
 *   noisy_mask  [1, 1025, T_stft]  noisy, from semDemucsV4.stft_masks
 * →
 *   refined_mask[1, 1025, T_stft]  +25.2% SI-SDR per the training run
 *
 * Refined masks feed maskPlayback's AudioWorklet — real-time solo/
 * mute on the master buffer uses them instead of the v4-small noisy
 * ones. The 1.5 MB fp16 ONNX is tiny enough to preload eagerly.
 */

const MODEL_URL = '/static/models/mask_refiner_fp16.onnx';
const N_FREQ = 1025;
const LATENT_CHANNELS = 64;

let _ort = null;
let _session = null;
let _sessionPromise = null;
let _runQueue = Promise.resolve();

export async function initMaskRefiner() {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ort.env.wasm.simd = true;
    }

    const graphResp = await fetch(MODEL_URL, { cache: 'force-cache' });
    if (!graphResp.ok) throw new Error(`mask_refiner HTTP ${graphResp.status}`);
    const graphBytes = new Uint8Array(await graphResp.arrayBuffer());

    const backends = [];
    if (ort.env?.webgpu) backends.push('webgpu');
    backends.push('wasm');
    let lastErr = null;
    for (const ep of backends) {
      try {
        const sess = await ort.InferenceSession.create(graphBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
        });
        _session = sess;
        console.log(`[maskRefiner] ready on ${ep}`);
        return sess;
      } catch (err) {
        console.warn(`[maskRefiner] ${ep} init failed:`, err?.message || err);
        lastErr = err;
      }
    }
    throw lastErr || new Error('no ORT backend available');
  })();

  return _sessionPromise;
}

export function isMaskRefinerReady() { return _session != null; }

async function pullTensor(t) {
  if (!t) return new Float32Array(0);
  try {
    if (typeof t.getData === 'function') return await t.getData();
  } catch (_) { /* fall through */ }
  return t.data || new Float32Array(0);
}

/**
 * Refine a single stem's mask.
 *
 * @param {Float32Array} latentTC  flat [T*64] time-major latent from
 *                                  latentDemucsV4 output (one stem).
 * @param {number} T                number of latent frames.
 * @param {Float32Array} noisyMask  flat [F*T_stft] mask values (v4-small
 *                                  softmax output, one stem).
 * @param {number} F                1025 (matches v4-small).
 * @param {number} T_stft           STFT frame count.
 * @returns {Promise<Float32Array>} refined mask, same [F*T_stft] shape.
 */
export async function refineMask(latentTC, T, noisyMask, F, T_stft) {
  const sess = await initMaskRefiner();
  const ort = _ort;
  const work = _runQueue.then(() => _refine(sess, ort, latentTC, T, noisyMask, F, T_stft));
  _runQueue = work.catch(() => {});
  return work;
}

async function _refine(sess, ort, latentTC, T, noisyMask, F, T_stft) {
  // latentTC is time-major [T, 64]; model expects channels-first
  // [B, 64, T]. Transpose once.
  const latCF = new Float32Array(LATENT_CHANNELS * T);
  for (let t = 0; t < T; t++) {
    for (let c = 0; c < LATENT_CHANNELS; c++) {
      latCF[c * T + t] = latentTC[t * LATENT_CHANNELS + c];
    }
  }
  const latentInput = new ort.Tensor('float32', latCF, [1, LATENT_CHANNELS, T]);
  const maskInput = new ort.Tensor('float32', noisyMask, [1, F, T_stft]);

  let res;
  try {
    res = await sess.run({ latent: latentInput, noisy_mask: maskInput });
  } catch (err) {
    const msg = err?.message || String(err);
    if (/already started|backend is still in use|webgpu/i.test(msg)) {
      await new Promise(r => setTimeout(r, 120));
      res = await sess.run({ latent: latentInput, noisy_mask: maskInput });
    } else {
      throw err;
    }
  }
  return await pullTensor(res.refined_mask);
}

/**
 * Refine all six stem masks in one call. Typical usage after v4cond-
 * pred completes in the upload flow.
 *
 * @param {Object} stems        { drums, bass, vocals, other, guitar, piano }
 *                              as returned by latentDemucsV4.separate6Stems.
 *                              Each entry has { flatTD, T, D }.
 * @param {Object} v4Result     semDemucsV4.analyze() return value
 *                              (for .masks, .maskF, .maskT).
 * @returns {Promise<Object>}   { stemName: Float32Array(F*T_stft) }
 */
export async function refine6StemMasks(stems, v4Result) {
  const { masks, maskF, maskT } = v4Result || {};
  if (!masks || !maskF || !maskT) throw new Error('refine6StemMasks: v4Result.masks missing');

  const stemNames = ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'];
  const stemStride = maskF * maskT;
  const out = {};
  for (let s = 0; s < stemNames.length; s++) {
    const name = stemNames[s];
    const stem = stems[name];
    if (!stem) continue;
    // Slice this stem's noisy mask.
    const noisy = masks.subarray(s * stemStride, (s + 1) * stemStride);
    const refined = await refineMask(stem.flatTD, stem.T, noisy, maskF, maskT);
    out[name] = refined;
  }
  return out;
}
