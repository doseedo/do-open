/**
 * Client-side SemDemucs v4-small-6 — fast pre-upload analysis.
 *
 * This ONNX runs on every uploaded audio file BEFORE latent demucs, and
 * produces the data the DAW needs to decide the upload flow:
 *
 *   1. rms [1, 6, T, 2]   — per-stem amplitude envelope used immediately
 *                           as the instant stem waveform visualizer, so
 *                           the DAW draws 6 stem rows within ~50ms of
 *                           upload (no wait for the 325 MB demucs).
 *
 *   2. stft_masks [1, 6, F, T_stft] — per-stem softmax masks (sum to 1
 *                           across stems at each bin). Integrated per
 *                           stem, they classify the file as "mix" vs
 *                           "solo stem":
 *                             drums energy high         → mix
 *                             vocals+other high, drums ~0 → synth/solo
 *                             one stem dominates        → solo stem
 *                           This gates whether we run latent demucs.
 *
 *   3. embedding [1, 6, 128] — per-stem global embedding (unused so far,
 *                           reserved for future stem-conditional work).
 *
 * Flow, wired by DAWOptimized.js on file drop:
 *   fetchBlob → audioFileToStereo48k → semDemucsV4.analyze(flat, N)
 *   → if isMix: latentDemucs (webGPU)
 *     else:    skip demucs; run oobleck encoder in background to cache
 *              the single stem's latent
 */

const MODEL_URL = '/static/models/sem_demucs_v4_6s_packed.onnx';
const MODEL_DATA_URL = '/static/models/sem_demucs_v4_6s_packed.onnx.data';
const TARGET_SR = 48000;
// STEMS_6 order must match DistillDataset6.STEMS_6 in the training code:
//   0 drums  1 bass  2 vocals  3 other  4 guitar  5 piano
export const STEM_NAMES_6 = ['drums', 'bass', 'vocals', 'other', 'guitar', 'piano'];
const N_STEMS = 6;
const RMS_FPS = 48000 / 1536;   // ≈ 31.25 Hz (matches rms time axis)

let _ort = null;
let _session = null;
let _sessionPromise = null;
let _loadProgress = null;
let _runQueue = Promise.resolve();

export async function initSemDemucsV4(onProgress = null) {
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
    if (!graphResp.ok) throw new Error(`sem_demucs_v4_6s.onnx HTTP ${graphResp.status}`);
    const graphBytes = new Uint8Array(await graphResp.arrayBuffer());

    const dataResp = await fetch(MODEL_DATA_URL, { cache: 'force-cache' });
    if (!dataResp.ok) throw new Error(`sem_demucs_v4_6s.onnx.data HTTP ${dataResp.status}`);
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
      { path: 'sem_demucs_v4_6s_packed.onnx.data', data: dataBytes.buffer },
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
        console.log(`[semDemucsV4] ready on ${ep}`);
        return sess;
      } catch (err) {
        const msg = err?.message || String(err);
        console.warn(`[semDemucsV4] ${ep} init failed:`, msg);
        if (ep === 'webgpu') {
          try {
            const { trackEvent, PRODUCT_EVENTS, platformString } = await import('../lib/telemetry');
            trackEvent(PRODUCT_EVENTS.WEBGPU_INIT_FAILED, {
              model: 'semDemucsV4',
              reason: msg.slice(0, 300),
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

export function isSemDemucsV4Ready() { return _session != null; }
export function getSemDemucsV4LoadProgress() { return _loadProgress; }

/**
 * Run v4-small-6 on a stereo waveform.
 *
 * Returns:
 *   {
 *     rms:        Float32Array[6 * T * 2]   // [6, T, 2] flattened
 *     rmsFrames:  T
 *     rmsFps:     ≈ 31.25
 *     masks:      Float32Array[6 * F * T_stft]  // [6, F, T_stft] flattened
 *     maskF:      1025
 *     maskT:      T_stft
 *     embedding:  Float32Array[6 * 128]
 *     perStemEnergy: Float32Array[6]  // integrated mask energy per stem
 *     classification: {
 *       isMix: boolean
 *       soloStemIndex: number | null   // which stem if not a mix
 *       soloStemName:  string | null
 *       confidence: number              // 0..1
 *       rationale: string
 *     }
 *   }
 */
export async function analyze(flat, numFrames) {
  const sess = await initSemDemucsV4();
  const ort = _ort;
  const work = _runQueue.then(() => _analyze(sess, ort, flat, numFrames));
  _runQueue = work.catch(() => {});
  return work;
}

async function _analyze(sess, ort, flat, numFrames) {
  // Model was traced at 4s. Inference supports dynamic length but we cap
  // it to the first ~30s — enough to classify + draw a visualizer — and
  // the user's full-length latent demucs does the real per-sample work.
  const maxAnalysisSamples = 48000 * 30;
  const useN = Math.min(numFrames, maxAnalysisSamples);
  const chunkFlat = new Float32Array(2 * useN);
  chunkFlat.set(flat.subarray(0, useN), 0);
  chunkFlat.set(flat.subarray(numFrames, numFrames + useN), useN);

  const input = new ort.Tensor('float32', chunkFlat, [1, 2, useN]);
  // ORT-Web's JSEP (WebGPU) backend occasionally rejects concurrent
  // sessions sharing the device queue with "Session already started"
  // or similar transient state errors. All of our big ONNX sessions
  // (decoder, demucs, encoder, semDemucsV4) eventually resolve to the
  // same GPU adapter; the prewarm chain creates them in order but the
  // first .run() can still race with a background init that just
  // finished. Retry once after a short backoff — reliably clears in
  // practice. If it fails twice, the caller catches and falls through
  // to the demucs path (the v4 mix-detect is non-critical).
  let res;
  try {
    res = await sess.run({ waveform: input });
  } catch (err) {
    const msg = err?.message || String(err);
    if (/already started|backend is still in use|webgpu/i.test(msg)) {
      await new Promise(r => setTimeout(r, 120));
      res = await sess.run({ waveform: input });
    } else {
      throw err;
    }
  }

  const rms = res.rms?.data || new Float32Array(0);
  const rmsDims = res.rms?.dims || [1, N_STEMS, 0, 2];
  const rmsT = rmsDims[2] | 0;

  const masks = res.stft_masks?.data || new Float32Array(0);
  const maskDims = res.stft_masks?.dims || [1, N_STEMS, 0, 0];
  const maskF = maskDims[2] | 0;
  const maskT = maskDims[3] | 0;

  const embedding = res.embedding?.data || new Float32Array(0);

  const perStemEnergy = computePerStemEnergy(masks, N_STEMS, maskF, maskT);
  const classification = classifyMixVsSolo(perStemEnergy);

  return {
    rms, rmsFrames: rmsT, rmsFps: RMS_FPS,
    masks, maskF, maskT,
    embedding,
    perStemEnergy,
    classification,
    // derived: per-stem envelope (ch0+ch1 avg) for direct canvas use
    stemEnvelopes: buildStemEnvelopes(rms, N_STEMS, rmsT),
  };
}

/**
 * Integrate each stem's mask energy across all (f, t) bins, then
 * normalize so the 6 values sum to 1. Since stft_masks are softmax'd
 * across stems at each bin, this is the fraction of total spectral
 * energy attributed to each stem by the model.
 */
function computePerStemEnergy(masks, nStems, F, T) {
  if (!masks.length || !F || !T) return new Float32Array(nStems);
  const out = new Float32Array(nStems);
  const stemStride = F * T;
  for (let s = 0; s < nStems; s++) {
    let sum = 0;
    const base = s * stemStride;
    for (let i = 0; i < stemStride; i++) sum += masks[base + i];
    out[s] = sum;
  }
  // Normalize
  let total = 0;
  for (let s = 0; s < nStems; s++) total += out[s];
  if (total > 0) for (let s = 0; s < nStems; s++) out[s] /= total;
  return out;
}

/**
 * Mix-vs-solo classifier from per-stem mask energy fractions.
 *
 * Rules (calibrated empirically — tune as we learn):
 *   - drums ≥ 0.08                     → definitely a mix (drums only
 *                                        show up strongly in full mixes;
 *                                        an isolated drum track will
 *                                        still classify as mix, which is
 *                                        fine — we'd want it separated
 *                                        into kick/snare/hh anyway).
 *   - any single stem ≥ 0.60           → solo of that stem.
 *   - entropy >= log(3) (i.e. energy
 *     spread across ≥ ~3 stems)         → mix.
 *   - otherwise default to solo at argmax.
 */
function classifyMixVsSolo(energy) {
  const stems = STEM_NAMES_6;
  const drumsIdx = 0;
  if (energy[drumsIdx] >= 0.08) {
    return {
      isMix: true,
      soloStemIndex: null,
      soloStemName: null,
      confidence: Math.min(1, energy[drumsIdx] / 0.25),
      rationale: `drums energy ${(energy[drumsIdx] * 100).toFixed(1)}% ≥ 8% — mix`,
    };
  }
  let maxIdx = 0, maxVal = 0;
  for (let i = 0; i < energy.length; i++) if (energy[i] > maxVal) { maxVal = energy[i]; maxIdx = i; }
  if (maxVal >= 0.60) {
    return {
      isMix: false,
      soloStemIndex: maxIdx,
      soloStemName: stems[maxIdx],
      confidence: Math.min(1, (maxVal - 0.5) * 2),
      rationale: `${stems[maxIdx]} ${(maxVal * 100).toFixed(1)}% ≥ 60% — solo stem`,
    };
  }
  // Compute Shannon entropy (nats). High entropy → energy spread → mix.
  let H = 0;
  for (let i = 0; i < energy.length; i++) {
    const p = energy[i];
    if (p > 1e-6) H -= p * Math.log(p);
  }
  const threshold = Math.log(3); // ≈ 1.098 — energy concentrated in > 2 stems
  if (H >= threshold) {
    return {
      isMix: true,
      soloStemIndex: null,
      soloStemName: null,
      confidence: Math.min(1, (H - threshold) / Math.log(2)),
      rationale: `entropy ${H.toFixed(2)} ≥ ${threshold.toFixed(2)} (energy spread across ≥3 stems) — mix`,
    };
  }
  return {
    isMix: false,
    soloStemIndex: maxIdx,
    soloStemName: stems[maxIdx],
    confidence: 0.5,
    rationale: `${stems[maxIdx]} ${(maxVal * 100).toFixed(1)}% argmax (fallback) — probably solo`,
  };
}

/**
 * Convert the rms [6, T, 2] output into a per-stem envelope
 * Float32Array[T*2] matching the (min, max) layout useWaveform expects:
 * first T values are the min envelope (negative), next T are max.
 * For rms (always ≥ 0) we mirror around 0 to synthesize a symmetric
 * envelope that draws like a normal stereo waveform.
 */
function buildStemEnvelopes(rms, nStems, T) {
  const out = [];
  if (!T) { for (let s = 0; s < nStems; s++) out.push(new Float32Array(0)); return out; }
  for (let s = 0; s < nStems; s++) {
    const env = new Float32Array(2 * T);
    const base = s * T * 2;
    for (let t = 0; t < T; t++) {
      const a = Math.max(rms[base + t * 2], rms[base + t * 2 + 1]);
      env[t]     = -a;   // min
      env[T + t] =  a;   // max
    }
    out.push(env);
  }
  return out;
}
