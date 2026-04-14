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
//   0 drums  1 bass  2 other  3 vocals  4 guitar  5 piano
// (Earlier this file had vocals/other swapped, so the row labeled "vocals"
//  was painting the "other" stem and vice-versa. All 8 training files
//  in latent_demucs_student/ — distill_dataset_6.py, train_*, eval_*,
//  build_stem6_targets.py, prep_moisesdb.py — agree on this order.)
export const STEM_NAMES_6 = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano'];
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

  // ORT-Web WebGPU (JSEP) keeps output tensors on GPU by default —
  // `.data` is a zero-length view until the buffer is copied to CPU.
  // That's what was producing the "drums 0.0% argmax" classification
  // on real mixes: every stem summed to zero because the masks
  // Float32Array was empty, so argmax defaulted to index 0. Use
  // await getData() which triggers the GPU→CPU copy on both backends
  // (no-op on WASM since the data is already CPU-resident).
  const pullTensor = async (t) => {
    if (!t) return new Float32Array(0);
    try {
      if (typeof t.getData === 'function') return await t.getData();
    } catch (_) { /* fall through */ }
    return t.data || new Float32Array(0);
  };

  const rms = await pullTensor(res.rms);
  const rmsDims = res.rms?.dims || [1, N_STEMS, 0, 2];
  const rmsT = rmsDims[2] | 0;

  const masks = await pullTensor(res.stft_masks);
  const maskDims = res.stft_masks?.dims || [1, N_STEMS, 0, 0];
  const maskF = maskDims[2] | 0;
  const maskT = maskDims[3] | 0;

  const embedding = await pullTensor(res.embedding);

  // Sniff the actual mask values — length-correct doesn't mean
  // value-correct on WebGPU. The 69 MB stft_masks buffer has been
  // observed to arrive at the right length but full of zeros on certain
  // WebGPU configs (the GPU→CPU copy completes but the destination
  // buffer was never written). Sample 4 indices across the buffer and
  // check for non-zero/non-NaN content; if dead, fall back.
  const sniff = (buf) => {
    const N = buf.length;
    if (!N) return { ok: false, reason: 'empty', sample: [] };
    const ix = [0, (N >> 2), (N >> 1), N - 1];
    const sample = ix.map(i => buf[i]);
    let nonZero = 0, nan = 0;
    for (const v of sample) {
      if (Number.isNaN(v)) nan++;
      else if (v !== 0) nonZero++;
    }
    return { ok: nonZero > 0 && nan === 0, reason: nan ? 'nan' : (nonZero ? 'ok' : 'all-zero'), sample };
  };
  const maskSniff = sniff(masks);
  const rmsSniff  = sniff(rms);
  console.log(
    `[semDemucsV4] tensor sizes — rms=${rms.length} (${rmsSniff.reason}, sample=[${Array.from(rmsSniff.sample).map(v => v.toExponential(2)).join(',')}])`
    + ` masks=${masks.length} (${maskSniff.reason}, sample=[${Array.from(maskSniff.sample).map(v => v.toExponential(2)).join(',')}])`
    + ` emb=${embedding.length}`
  );

  // Prefer masks (model softmax sums to 1.0 across stems — accurate
  // per-stem energy). Fall back to RMS only when masks fail to transfer
  // (length OK but values dead) — RMS head is known miscalibrated but
  // it's better than all-zero classification.
  let perStemEnergy;
  let stemEnvelopes;
  let envelopeSource;
  if (masks.length === N_STEMS * maskF * maskT && maskSniff.ok) {
    perStemEnergy = computePerStemEnergyFromMasks(masks, N_STEMS, maskF, maskT);
    stemEnvelopes = buildStemEnvelopesFromMasks(masks, N_STEMS, maskF, maskT, rmsT || ENV_FALLBACK_T);
    envelopeSource = 'masks';
  } else if (rms.length === N_STEMS * rmsT * 2 && rmsSniff.ok) {
    perStemEnergy = computePerStemEnergyFromRms(rms, N_STEMS, rmsT);
    stemEnvelopes = buildStemEnvelopesFromRms(rms, N_STEMS, rmsT);
    envelopeSource = 'rms';
    console.warn(`[semDemucsV4] masks dead (${maskSniff.reason}) — fallback to RMS head (known miscalibrated)`);
  } else {
    console.warn(`[semDemucsV4] both masks and rms dead — cannot classify (mask=${maskSniff.reason} rms=${rmsSniff.reason})`);
    perStemEnergy = new Float32Array(N_STEMS);
    stemEnvelopes = Array.from({ length: N_STEMS }, () => new Float32Array(0));
    envelopeSource = 'empty';
  }
  const classification = classifyMixVsSolo(perStemEnergy);

  return {
    rms, rmsFrames: rmsT, rmsFps: RMS_FPS,
    masks, maskF, maskT,
    embedding,
    perStemEnergy,
    classification,
    stemEnvelopes,
    envelopeSource,
  };
}

// Used by the empty-input branch of _analyze when neither masks nor rms
// arrived — pick a reasonable display length (~30s at RMS_FPS).
const ENV_FALLBACK_T = 938;

/**
 * Integrate each stem's mask energy across all (f, t) bins, then
 * normalize so the 6 values sum to 1. Since stft_masks are softmax'd
 * across stems at each bin, this is the fraction of total spectral
 * energy attributed to each stem by the model.
 */
function computePerStemEnergyFromMasks(masks, nStems, F, T) {
  if (!masks.length || !F || !T) return new Float32Array(nStems);
  const out = new Float32Array(nStems);
  const stemStride = F * T;
  for (let s = 0; s < nStems; s++) {
    let sum = 0;
    const base = s * stemStride;
    for (let i = 0; i < stemStride; i++) sum += masks[base + i];
    out[s] = sum;
  }
  let total = 0;
  for (let s = 0; s < nStems; s++) total += out[s];
  if (total > 0) for (let s = 0; s < nStems; s++) out[s] /= total;
  return out;
}

/**
 * RMS-based energy — primary classifier input.
 *
 * rms shape: [1, nStems, T, 2] (stereo RMS per latent frame). Summing
 * rms² across time + channels gives each stem's total energy in the
 * input mix; normalizing across stems yields the same "fraction of
 * audible energy per stem" signal as the mask-based version but with
 * a 1000× smaller tensor (12 KB vs 69 MB for 30 s clips). That tiny
 * size is key — the large mask buffer occasionally fails to transfer
 * from GPU to CPU on WebGPU configs, silently returning zeros.
 * rms is robust.
 */
function computePerStemEnergyFromRms(rms, nStems, T) {
  const out = new Float32Array(nStems);
  if (!rms.length || !T) return out;
  // rms layout is [1, nStems, T, 2] flattened; stride per stem = T*2.
  for (let s = 0; s < nStems; s++) {
    let e = 0;
    const base = s * T * 2;
    for (let i = 0; i < T * 2; i++) {
      const v = rms[base + i];
      e += v * v;
    }
    out[s] = e;
  }
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
 * Build per-stem envelopes from stft_masks [6, F, T_stft] by integrating
 * mask probability over frequency at each time step. Output is mirrored
 * (-a, +a) to match the (min, max) layout useWaveform expects.
 *
 * Masks are softmax'd across stems at each (f, t) bin and sum to 1.0,
 * so `sum_f masks[s, f, t]` is the spectral energy fraction this stem
 * holds at time t. Multiplying by a fixed display amplitude keeps the
 * waveform readable. We downsample T_stft → Tout by box-averaging so
 * the placeholder waveform aligns with the same per-frame timeline the
 * RMS path used (≈31.25 fps).
 */
function buildStemEnvelopesFromMasks(masks, nStems, F, T_stft, Tout) {
  const out = [];
  if (!F || !T_stft || !Tout) {
    for (let s = 0; s < nStems; s++) out.push(new Float32Array(0));
    return out;
  }
  const stemStride = F * T_stft;
  const DISPLAY_AMP = 0.9;  // 0..1 mask fraction → ±0.9 envelope range
  for (let s = 0; s < nStems; s++) {
    const env = new Float32Array(2 * Tout);
    const base = s * stemStride;
    // For each output frame, average mask across F over the corresponding
    // T_stft window, then scale to display range.
    for (let to = 0; to < Tout; to++) {
      const tStart = Math.floor(to * T_stft / Tout);
      const tEnd   = Math.max(tStart + 1, Math.floor((to + 1) * T_stft / Tout));
      let sum = 0;
      let count = 0;
      for (let ti = tStart; ti < tEnd; ti++) {
        for (let f = 0; f < F; f++) sum += masks[base + f * T_stft + ti];
        count += F;
      }
      const v = (sum / count) * DISPLAY_AMP;
      env[to]        = -v;
      env[Tout + to] =  v;
    }
    out.push(env);
  }
  return out;
}

/**
 * Convert the rms [6, T, 2] output into a per-stem envelope
 * Float32Array[T*2] matching the (min, max) layout useWaveform expects.
 * Kept as a last-resort fallback only — see _analyze for why masks are
 * the primary source.
 */
function buildStemEnvelopesFromRms(rms, nStems, T) {
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
