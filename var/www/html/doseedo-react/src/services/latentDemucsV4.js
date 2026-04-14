/**
 * Client-side v4cond-pred 6-stem separator (WebGPU/WASM).
 *
 * Replaces distill_demucs. Takes the v4-small (semDemucsV4) outputs
 * as conditioning — the upload flow already computes them — and
 * produces 6 stem latents that feed the existing uploadLatent +
 * Oobleck decoder pipeline.
 *
 * I/O:
 *   waveform     [1, 2, N]            stereo 48 kHz float32
 *   sem_emb      [1, 6, 128]          from semDemucsV4.analyze().embedding
 *   stft_masks   [1, 6, 1025, T_stft] from semDemucsV4.analyze().masks
 * →
 *   stem_latents [1, 6, 64, T_enc]    feeds uploadLatent as DOAE binary
 *
 * Weights are fp16 internal (~98 MB), inputs/outputs stay fp32 so
 * existing browser tensor code is untouched.
 *
 * Stem order (DistillDataset6.STEMS_6, matches semDemucsV4.STEM_NAMES_6):
 *   0 drums  1 bass  2 vocals  3 other  4 guitar  5 piano
 */

const MODEL_URL = '/static/models/v4cond_pred_6s_fp16.onnx';
const MODEL_DATA_URL = '/static/models/v4cond_pred_6s_fp16.onnx.data';
const TARGET_SR = 48000;
const FRAME_SAMPLES = 1920;      // Oobleck encoder ratio
const LATENT_CHANNELS = 64;
const N_STEMS = 6;

// Order matches DistillDataset6.STEMS_6 in the training code (vocals/other
// were swapped in an earlier version — see semDemucsV4.STEM_NAMES_6).
export const STEM_NAMES_V4COND_6 = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];

let _ort = null;
let _session = null;
let _sessionPromise = null;
let _loadProgress = null;
let _runQueue = Promise.resolve();

export async function initLatentDemucsV4(onProgress = null) {
  if (_session) return _session;
  if (_sessionPromise) return _sessionPromise;

  _sessionPromise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;
    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ort.env.wasm.simd = true;
    }

    const graphResp = await fetch(MODEL_URL, { cache: 'force-cache' });
    if (!graphResp.ok) throw new Error(`v4cond_pred HTTP ${graphResp.status}`);
    const graphBytes = new Uint8Array(await graphResp.arrayBuffer());

    const dataResp = await fetch(MODEL_DATA_URL, { cache: 'force-cache' });
    if (!dataResp.ok) throw new Error(`v4cond_pred.data HTTP ${dataResp.status}`);
    const total = parseInt(dataResp.headers.get('content-length') || '0', 10);
    const reader = dataResp.body.getReader();
    const chunks = [];
    let loaded = 0;
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.byteLength;
      _loadProgress = { bytesLoaded: loaded, bytesTotal: total };
      if (onProgress) onProgress(_loadProgress);
    }
    const dataBytes = new Uint8Array(loaded);
    let off = 0;
    for (const c of chunks) { dataBytes.set(c, off); off += c.byteLength; }

    const externalData = [
      { path: 'v4cond_pred_6s_fp16.onnx.data', data: dataBytes.buffer },
    ];

    const backends = [];
    if (ort.env?.webgpu) backends.push('webgpu');
    backends.push('wasm');
    let lastErr = null;
    for (const ep of backends) {
      try {
        const sess = await ort.InferenceSession.create(graphBytes, {
          executionProviders: [ep],
          graphOptimizationLevel: 'all',
          externalData,
        });
        _session = sess;
        console.log(`[latentDemucsV4] ready on ${ep}`);
        return sess;
      } catch (err) {
        console.warn(`[latentDemucsV4] ${ep} init failed:`, err?.message || err);
        if (ep === 'webgpu') {
          try {
            const { trackEvent, PRODUCT_EVENTS, platformString } = await import('../lib/telemetry');
            trackEvent(PRODUCT_EVENTS.WEBGPU_INIT_FAILED, {
              model: 'latentDemucsV4',
              reason: (err?.message || String(err)).slice(0, 300),
              platform: platformString(),
            });
          } catch (_) { /* best-effort */ }
        }
        lastErr = err;
      }
    }
    throw lastErr || new Error('no ORT backend available');
  })();

  return _sessionPromise;
}

export function getLoadProgress() { return _loadProgress; }
export function isLatentDemucsV4Ready() { return _session != null; }

/** GPU→CPU pull helper — ORT-Web JSEP returns empty .data until explicit. */
async function pullTensor(t) {
  if (!t) return new Float32Array(0);
  try {
    if (typeof t.getData === 'function') return await t.getData();
  } catch (_) { /* fall through */ }
  return t.data || new Float32Array(0);
}

/**
 * Run v4cond-pred on a full-length waveform using v4-small outputs as
 * conditioning. Returns an object keyed by stem name, each entry a
 * time-major flat Float32Array [T_enc * 64] ready to be DOAE-serialized
 * and uploaded via uploadLatent().
 *
 * @param {Float32Array} flat       [2 * N] channels-first (L;R) waveform
 * @param {number} numFrames        N samples @ 48 kHz
 * @param {Object} v4Result         semDemucsV4.analyze() return value
 * @param {number} chunkSamples     max per-inference window; defaults
 *                                   to 30 s to keep GPU memory bounded.
 */
export async function separate6Stems(flat, numFrames, v4Result, chunkSamples = 48000 * 30) {
  const sess = await initLatentDemucsV4();
  const ort = _ort;
  const work = _runQueue.then(() => _separate(sess, ort, flat, numFrames, v4Result, chunkSamples));
  _runQueue = work.catch(() => {});
  return work;
}

async function _separate(sess, ort, flat, numFrames, v4Result, chunkSamples) {
  // We only have ONE pass of v4-small conditioning (computed over the
  // first ~30s). If the upload is longer than that, we reuse the same
  // sem_emb + masks across chunks — conditioning is global-ish for
  // stem identity, so cross-chunk drift is small. Longer-than-chunk
  // handling for truly long uploads is a follow-up; typical mixes are
  // under 5 min and this chunk boundary doesn't show artifacts there.
  const { embedding, masks, maskF, maskT } = v4Result || {};
  if (!embedding || embedding.length !== N_STEMS * 128) {
    throw new Error('latentDemucsV4: v4Result.embedding missing/wrong size');
  }
  if (!masks || !maskF || !maskT) {
    throw new Error('latentDemucsV4: v4Result.masks missing');
  }

  // Build constant conditioning tensors once per call.
  const semTensor = new ort.Tensor('float32', embedding, [1, N_STEMS, 128]);
  const masksTensor = new ort.Tensor('float32', masks, [1, N_STEMS, maskF, maskT]);

  const stepSamples = Math.floor(chunkSamples / FRAME_SAMPLES) * FRAME_SAMPLES;
  const totalOutFrames = Math.floor(numFrames / FRAME_SAMPLES);
  if (totalOutFrames <= 0) {
    return Object.fromEntries(STEM_NAMES_V4COND_6.map(n => [n, { flatTD: new Float32Array(0), T: 0, D: LATENT_CHANNELS }]));
  }

  // Accumulate per-stem latents in time-major layout [T, 64].
  const perStem = STEM_NAMES_V4COND_6.map(() => new Float32Array(totalOutFrames * LATENT_CHANNELS));

  let consumed = 0;
  let tWritten = 0;
  while (consumed < numFrames) {
    const thisN = Math.min(stepSamples, numFrames - consumed);
    if (thisN < FRAME_SAMPLES) break;
    const useN = Math.floor(thisN / FRAME_SAMPLES) * FRAME_SAMPLES;
    const chunkFlat = new Float32Array(2 * useN);
    chunkFlat.set(flat.subarray(consumed, consumed + useN), 0);
    chunkFlat.set(flat.subarray(numFrames + consumed, numFrames + consumed + useN), useN);
    const wavTensor = new ort.Tensor('float32', chunkFlat, [1, 2, useN]);

    let res;
    try {
      res = await sess.run({ waveform: wavTensor, sem_emb: semTensor, stft_masks: masksTensor });
    } catch (err) {
      const msg = err?.message || String(err);
      if (/already started|backend is still in use|webgpu/i.test(msg)) {
        await new Promise(r => setTimeout(r, 120));
        res = await sess.run({ waveform: wavTensor, sem_emb: semTensor, stft_masks: masksTensor });
      } else {
        throw err;
      }
    }

    const latents = await pullTensor(res.stem_latents);
    // latents: [1, 6, 64, T_chunk] flattened. Slice per-stem and
    // transpose to time-major for DOAE.
    const Tchunk = useN / FRAME_SAMPLES;
    const stride = LATENT_CHANNELS * Tchunk;
    for (let s = 0; s < N_STEMS; s++) {
      const src = latents.subarray(s * stride, (s + 1) * stride); // [64, T_chunk]
      const dst = perStem[s];
      const base = tWritten * LATENT_CHANNELS;
      for (let t = 0; t < Tchunk; t++) {
        for (let c = 0; c < LATENT_CHANNELS; c++) {
          dst[base + t * LATENT_CHANNELS + c] = src[c * Tchunk + t];
        }
      }
    }
    tWritten += Tchunk;
    consumed += useN;
  }

  const out = {};
  for (let s = 0; s < N_STEMS; s++) {
    out[STEM_NAMES_V4COND_6[s]] = {
      flatTD: perStem[s].subarray(0, tWritten * LATENT_CHANNELS),
      T: tWritten,
      D: LATENT_CHANNELS,
    };
  }
  return out;
}
