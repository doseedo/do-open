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

// _v3 adds a `masks_envelope` output (stft_masks integrated over freq, ~17K floats
// per 30 s clip) so the browser doesn't need to read back the broken 17M-float
// stft_masks tensor for display/classification. v2 was Conv1d-STFT (same as v3
// internally). v1 was native STFT op. Bump suffix on every re-export to bust
// the `cache: 'force-cache'` reads below.
const MODEL_URL = '/static/models/sem_demucs_v4_6s_packed_v3.onnx';
const MODEL_DATA_URL = '/static/models/sem_demucs_v4_6s_packed_v3.onnx.data';
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
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
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
      // Path must match the external_data location string baked into the ONNX
      // graph (set at export time from the output filename). v3 was exported as
      // sem_demucs_v4_6s_packed_v3.onnx so the internal reference is the _v3
      // version. Bumping the file URL above also requires updating this key.
      { path: 'sem_demucs_v4_6s_packed_v3.onnx.data', data: dataBytes.buffer },
    ];

    // WebGPU first, WASM fallback. The earlier WASM-only restriction was
    // for the v1 ONNX (native STFT op zeroed masks on WebGPU). The v2
    // ONNX uses Conv1d-based |STFT| — plain ops that WebGPU handles, and
    // WASM happens to NaN-poison this model's spec head regardless of
    // ORT-Web version (1.22 + 1.24 both broken). So WebGPU is now the
    // primary path. WASM is kept as fallback for browsers without
    // WebGPU; sniff-and-fallback in _analyze still flips to RMS if the
    // mask buffer comes back dead.
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
  // Run the model in 4-second chunks and concatenate outputs along time.
  //
  // The model was traced at 4 s. At full 30 s input the stft_masks output
  // is 17 M floats (~69 MB), which ORT-Web mishandles on every backend
  // we've tested — WebGPU silently zeros it, WASM NaN-poisons it, both
  // 1.22 and 1.24, both ONNX STFT op (v1) and Conv1d STFT (v2/v3). The
  // hypothesis is that the failure is size-related (intermediate buffer
  // / GPU→CPU copy of the mask), so chunking back to the trace size
  // keeps each call's mask at ~9 MB and should survive. We then stitch
  // the per-chunk outputs into the same shape the rest of the pipeline
  // expects (full-length rms, full-length masks, full-length envelope).
  // embedding is per-stem global with no time axis — average across
  // chunks. Cap total analysis at 30 s for compute budget.
  const CHUNK_SAMPLES = 48000 * 4;
  const MAX_TOTAL = 48000 * 30;
  const totalN = Math.min(numFrames, MAX_TOTAL);
  const numChunks = Math.ceil(totalN / CHUNK_SAMPLES);

  // Run each 4 s chunk through the model.
  const chunks = [];
  for (let i = 0; i < numChunks; i++) {
    const start = i * CHUNK_SAMPLES;
    const end = Math.min(start + CHUNK_SAMPLES, totalN);
    const chunkLen = end - start;

    const chunkFlat = new Float32Array(2 * chunkLen);
    chunkFlat.set(flat.subarray(start, end), 0);
    chunkFlat.set(flat.subarray(numFrames + start, numFrames + end), chunkLen);

    const input = new ort.Tensor('float32', chunkFlat, [1, 2, chunkLen]);
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
    chunks.push(res);
  }

  // ORT-Web WebGPU (JSEP) keeps output tensors on GPU by default —
  // `.data` is a zero-length view until the buffer is copied to CPU.
  const pullTensor = async (t) => {
    if (!t) return new Float32Array(0);
    try {
      if (typeof t.getData === 'function') return await t.getData();
    } catch (_) { /* fall through */ }
    return t.data || new Float32Array(0);
  };

  // Pull per-chunk tensors + dims.
  const perChunk = [];
  for (const res of chunks) {
    perChunk.push({
      rms:      await pullTensor(res.rms),
      rmsT:     (res.rms?.dims || [1, N_STEMS, 0, 2])[2] | 0,
      masks:    await pullTensor(res.stft_masks),
      maskF:    (res.stft_masks?.dims || [1, N_STEMS, 0, 0])[2] | 0,
      maskT:    (res.stft_masks?.dims || [1, N_STEMS, 0, 0])[3] | 0,
      envelope: await pullTensor(res.masks_envelope),
      envT:     (res.masks_envelope?.dims || [1, N_STEMS, 0])[2] | 0,
      embedding:await pullTensor(res.embedding),
    });
  }

  // Concatenate across chunks. The model's output dims for a given input
  // length are deterministic, so summing per-chunk T values gives the
  // total T we'd have gotten from a single full-length call.
  const rmsT  = perChunk.reduce((a, c) => a + c.rmsT, 0);
  const maskF = perChunk[0]?.maskF | 0;
  const maskT = perChunk.reduce((a, c) => a + c.maskT, 0);
  const envT  = perChunk.reduce((a, c) => a + c.envT, 0);

  const rms      = concatStemsTime(perChunk.map(c => c.rms),      N_STEMS, perChunk.map(c => c.rmsT),  /*perFrame=*/2);
  const envelope = concatStemsTime(perChunk.map(c => c.envelope), N_STEMS, perChunk.map(c => c.envT),  /*perFrame=*/1);
  const masks    = concatStemsFreqTime(perChunk.map(c => c.masks), N_STEMS, maskF, perChunk.map(c => c.maskT));
  const embedding = averageEmbeddings(perChunk.map(c => c.embedding), N_STEMS, /*embDim=*/128);

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
  const envSniff  = sniff(envelope);
  const maskSniff = sniff(masks);
  const rmsSniff  = sniff(rms);
  console.log(
    `[semDemucsV4] ${numChunks}×4s chunks — rms=${rms.length} (${rmsSniff.reason})`
    + ` envelope=${envelope.length} (${envSniff.reason}, sample=[${Array.from(envSniff.sample).map(v => v.toExponential(2)).join(',')}])`
    + ` masks=${masks.length} (${maskSniff.reason}) emb=${embedding.length}`
  );

  // Envelope is the primary signal — small enough to always survive readback.
  // Masks-from-full-tensor and RMS are kept as fallbacks for backwards-compat
  // with old ONNX builds and for browsers where envelope readback also fails.
  let perStemEnergy;
  let stemEnvelopes;
  let envelopeSource;
  if (envelope.length === N_STEMS * envT && envSniff.ok) {
    perStemEnergy = computePerStemEnergyFromEnvelope(envelope, N_STEMS, envT);
    stemEnvelopes = buildStemEnvelopesFromEnvelope(envelope, N_STEMS, envT, rmsT || ENV_FALLBACK_T);
    envelopeSource = 'envelope';
  } else if (masks.length === N_STEMS * maskF * maskT && maskSniff.ok) {
    perStemEnergy = computePerStemEnergyFromMasks(masks, N_STEMS, maskF, maskT);
    stemEnvelopes = buildStemEnvelopesFromMasks(masks, N_STEMS, maskF, maskT, rmsT || ENV_FALLBACK_T);
    envelopeSource = 'masks';
  } else if (rms.length === N_STEMS * rmsT * 2 && rmsSniff.ok) {
    perStemEnergy = computePerStemEnergyFromRms(rms, N_STEMS, rmsT);
    stemEnvelopes = buildStemEnvelopesFromRms(rms, N_STEMS, rmsT);
    envelopeSource = 'rms';
    console.warn(`[semDemucsV4] envelope+masks dead (env=${envSniff.reason} masks=${maskSniff.reason}) — fallback to RMS head (known miscalibrated)`);
  } else {
    console.warn(`[semDemucsV4] all signals dead — cannot classify (env=${envSniff.reason} mask=${maskSniff.reason} rms=${rmsSniff.reason})`);
    perStemEnergy = new Float32Array(N_STEMS);
    stemEnvelopes = Array.from({ length: N_STEMS }, () => new Float32Array(0));
    envelopeSource = 'empty';
  }
  const classification = classifyMixVsSolo(perStemEnergy);

  return {
    rms, rmsFrames: rmsT, rmsFps: RMS_FPS,
    masks, maskF, maskT,
    envelope, envT,
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
 * Concatenate per-stem buffers across chunks along the time axis.
 *
 * Layout assumption: each chunk's buffer is `[stem, t, perFrame]` row-major,
 * i.e. for a given stem the time axis is contiguous. Output preserves the
 * same row-major layout with summed T.
 *
 * Used for `rms` (perFrame=2) and `envelope` (perFrame=1).
 */
function concatStemsTime(chunkBufs, nStems, chunkTs, perFrame) {
  const totalT = chunkTs.reduce((a, t) => a + t, 0);
  if (!totalT) return new Float32Array(0);
  const out = new Float32Array(nStems * totalT * perFrame);
  for (let s = 0; s < nStems; s++) {
    let dstT = 0;
    for (let i = 0; i < chunkBufs.length; i++) {
      const buf = chunkBufs[i];
      const T = chunkTs[i];
      if (!T || !buf || buf.length < (s + 1) * T * perFrame) continue;
      const src = buf.subarray(s * T * perFrame, (s + 1) * T * perFrame);
      out.set(src, s * totalT * perFrame + dstT * perFrame);
      dstT += T;
    }
  }
  return out;
}

/**
 * Concatenate per-stem masks across chunks along the time axis.
 * Each chunk's mask layout is `[stem, freq, t]` row-major; output preserves
 * the same shape with summed T. Inner loop is per (stem, freq) which means
 * S*F = 6*1025 = 6150 small `set()` calls per call — negligible vs the
 * 17M-float total move.
 */
function concatStemsFreqTime(chunkBufs, nStems, F, chunkTs) {
  const totalT = chunkTs.reduce((a, t) => a + t, 0);
  if (!totalT || !F) return new Float32Array(0);
  const out = new Float32Array(nStems * F * totalT);
  for (let s = 0; s < nStems; s++) {
    for (let f = 0; f < F; f++) {
      let dstT = 0;
      for (let i = 0; i < chunkBufs.length; i++) {
        const buf = chunkBufs[i];
        const T = chunkTs[i];
        if (!T || !buf || buf.length < nStems * F * T) continue;
        const srcOff = s * F * T + f * T;
        const src = buf.subarray(srcOff, srcOff + T);
        out.set(src, s * F * totalT + f * totalT + dstT);
        dstT += T;
      }
    }
  }
  return out;
}

/**
 * Average per-stem embeddings across chunks. Each chunk's embedding is a
 * full [B=1, S, embDim] tensor (the model returns one global embedding per
 * stem per call), so we mean-pool across chunks. If a chunk's embedding is
 * empty/short, skip it.
 */
function averageEmbeddings(chunkBufs, nStems, embDim) {
  const out = new Float32Array(nStems * embDim);
  let count = 0;
  for (const buf of chunkBufs) {
    if (!buf || buf.length < nStems * embDim) continue;
    for (let i = 0; i < nStems * embDim; i++) out[i] += buf[i];
    count++;
  }
  if (count > 1) for (let i = 0; i < nStems * embDim; i++) out[i] /= count;
  return out;
}

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
 * Per-stem envelope from the tiny `masks_envelope` output (shape [6, T_stft],
 * already integrated over freq inside the ONNX). Each value is `sum_f
 * mask[s,f,t]` ∈ [0, F] — divide by F to get a 0..1 fraction. Then mirror
 * around 0 and downsample T_stft → Tout for the (min, max) layout
 * useWaveform expects.
 */
function buildStemEnvelopesFromEnvelope(envelope, nStems, T_stft, Tout) {
  const out = [];
  if (!T_stft || !Tout) {
    for (let s = 0; s < nStems; s++) out.push(new Float32Array(0));
    return out;
  }
  const F_NORM = 1025;            // matches mask freq-axis size — amounts to /F
  const DISPLAY_AMP = 0.9;
  for (let s = 0; s < nStems; s++) {
    const env = new Float32Array(2 * Tout);
    const base = s * T_stft;
    for (let to = 0; to < Tout; to++) {
      const tStart = Math.floor(to * T_stft / Tout);
      const tEnd   = Math.max(tStart + 1, Math.floor((to + 1) * T_stft / Tout));
      let sum = 0;
      for (let ti = tStart; ti < tEnd; ti++) sum += envelope[base + ti];
      const v = (sum / ((tEnd - tStart) * F_NORM)) * DISPLAY_AMP;
      env[to]        = -v;
      env[Tout + to] =  v;
    }
    out.push(env);
  }
  return out;
}

/**
 * Per-stem energy fraction from the envelope output: integrate over T,
 * normalize so the 6 fractions sum to 1. Equivalent to integrating the
 * full mask over both F and T but cheaper because the F sum already
 * happened inside the ONNX.
 */
function computePerStemEnergyFromEnvelope(envelope, nStems, T_stft) {
  const out = new Float32Array(nStems);
  if (!T_stft || !envelope.length) return out;
  for (let s = 0; s < nStems; s++) {
    let e = 0;
    const base = s * T_stft;
    for (let t = 0; t < T_stft; t++) e += envelope[base + t];
    out[s] = e;
  }
  let total = 0;
  for (let s = 0; s < nStems; s++) total += out[s];
  if (total > 0) for (let s = 0; s < nStems; s++) out[s] /= total;
  return out;
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
 *
 * Used only when the new `masks_envelope` output is unavailable (older
 * ONNX builds). v3 ONNX has the integration done inside the model.
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
