/**
 * Lightweight mix detector using demucs_rms_int8_v2.onnx (577 KB).
 *
 * Model I/O:
 *   input:  logmel  [B, 1, 64, T]   — raw log-mel spectrogram (no normalization)
 *   output: logits  [B, 4, T]       — per-stem log-amplitude envelopes
 *                                     stems: [drums, bass, vocals, other]
 *
 * Outputs of analyze():
 *   stemEnvelopes   Float32Array[4][T]  per-stem amplitude over time (linear)
 *   classification  { isMix, soloStemIndex, soloStemName, confidence }
 *   fps             ≈ 93.75 (48000/512)
 *
 * This runs on EVERY uploaded file before any backend work. For mixes it
 * paints instant 4-stem waveforms; for solo stems it skips demucs entirely.
 */

const MODEL_URL = '/static/models/demucs_rms_int8_v2.onnx?v=1';
const TARGET_SR  = 48000;
const N_FFT      = 1024;
const HOP        = 512;
const N_MELS     = 64;
const FMIN       = 20;
const N_STEMS    = 4;

export const STEM_NAMES = ['drums', 'bass', 'vocals', 'other'];
export const RMS_FPS    = TARGET_SR / HOP;   // 93.75 Hz

// Solo threshold: if dominant stem is more than MIX_GAP_DB above the
// median of the others, classify as solo.
const MIX_GAP_DB      = 15;   // dB above second-highest → solo
const SOLO_MIN_DB     = -30;  // dominant must be at least this loud

let _ort     = null;
let _session = null;
let _promise = null;

// ── Mel filterbank (built once, cached) ─────────────────────────────────────
let _melFb = null;

function buildMelFb(sr, nFft, nMels, fMin, fMax) {
  if (_melFb) return _melFb;
  const hz2mel = (h) => 2595 * Math.log10(1 + h / 700);
  const mel2hz = (m) => 700 * (10 ** (m / 2595) - 1);
  const melMin = hz2mel(fMin);
  const melMax = hz2mel(fMax);
  const melPts = Array.from({ length: nMels + 2 }, (_, i) =>
    mel2hz(melMin + (melMax - melMin) * i / (nMels + 1))
  );
  const bins = melPts.map((f) => Math.floor((nFft + 1) * f / sr));
  const nFreq = nFft / 2 + 1;
  const fb    = new Float32Array(nMels * nFreq);   // [nMels, nFreq] row-major

  for (let m = 1; m <= nMels; m++) {
    for (let k = bins[m - 1]; k < bins[m]; k++) {
      if (k >= 0 && k < nFreq) fb[(m - 1) * nFreq + k] = (k - bins[m - 1]) / (bins[m] - bins[m - 1] + 1e-10);
    }
    for (let k = bins[m]; k < bins[m + 1]; k++) {
      if (k >= 0 && k < nFreq) fb[(m - 1) * nFreq + k] = (bins[m + 1] - k) / (bins[m + 1] - bins[m] + 1e-10);
    }
  }
  _melFb = { fb, nFreq };
  return _melFb;
}

// ── Bit-reversal permutation for Cooley-Tukey FFT ────────────────────────────
function fft(re, im) {
  const n = re.length;
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len;
    const wRe = Math.cos(ang), wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let curRe = 1, curIm = 0;
      for (let j = 0; j < len / 2; j++) {
        const uRe = re[i + j], uIm = im[i + j];
        const vRe = re[i + j + len/2] * curRe - im[i + j + len/2] * curIm;
        const vIm = re[i + j + len/2] * curIm + im[i + j + len/2] * curRe;
        re[i + j]        = uRe + vRe; im[i + j]        = uIm + vIm;
        re[i + j + len/2] = uRe - vRe; im[i + j + len/2] = uIm - vIm;
        const nr = curRe * wRe - curIm * wIm;
        curIm = curRe * wIm + curIm * wRe;
        curRe = nr;
      }
    }
  }
}

/** Compute log-mel spectrogram from mono Float32Array (48 kHz). */
function computeLogMel(mono, nFft, hop, nMels, sr, fMin) {
  const fMax  = sr / 2;
  const { fb, nFreq } = buildMelFb(sr, nFft, nMels, fMin, fMax);
  const pad   = nFft / 2;
  const total = mono.length + 2 * pad;
  const padded = new Float32Array(total);
  // reflect padding
  for (let i = 0; i < pad; i++) padded[pad - 1 - i] = mono[i] || 0;
  padded.set(mono, pad);
  for (let i = 0; i < pad; i++) padded[pad + mono.length + i] = mono[mono.length - 1 - i] || 0;

  const T = Math.floor((padded.length - nFft) / hop) + 1;
  const logMel = new Float32Array(nMels * T);   // [nMels, T] row-major

  // Hann window
  const win = Float32Array.from({ length: nFft }, (_, i) =>
    0.5 * (1 - Math.cos(2 * Math.PI * i / (nFft - 1)))
  );

  const reArr = new Float32Array(nFft);
  const imArr = new Float32Array(nFft);
  const power = new Float32Array(nFreq);

  for (let t = 0; t < T; t++) {
    const off = t * hop;
    for (let k = 0; k < nFft; k++) { reArr[k] = padded[off + k] * win[k]; imArr[k] = 0; }
    fft(reArr, imArr);
    for (let k = 0; k < nFreq; k++) power[k] = reArr[k] * reArr[k] + imArr[k] * imArr[k];

    // Mel filterbank dot product + log
    for (let m = 0; m < nMels; m++) {
      let s = 0;
      const base = m * nFreq;
      for (let k = 0; k < nFreq; k++) s += fb[base + k] * power[k];
      logMel[m * T + t] = 10 * Math.log10(Math.max(s, 1e-10));  // raw dB — NO normalization
    }
  }
  return { logMel, T };
}

// ── Classification ────────────────────────────────────────────────────────────
function classifyMixVsSolo(meanLogitPerStem) {
  // meanLogitPerStem: Float32Array[4] of mean log-amplitude values per stem
  let maxIdx = 0, secIdx = 1;
  for (let i = 1; i < N_STEMS; i++) {
    if (meanLogitPerStem[i] > meanLogitPerStem[maxIdx]) { secIdx = maxIdx; maxIdx = i; }
    else if (meanLogitPerStem[i] > meanLogitPerStem[secIdx] && i !== maxIdx) secIdx = i;
  }
  const dominant = meanLogitPerStem[maxIdx];
  const second   = meanLogitPerStem[secIdx];
  const gap      = dominant - second;   // both in dB

  if (dominant < SOLO_MIN_DB) {
    // Everything is near silence — treat as solo with low confidence
    return { isMix: false, soloStemIndex: maxIdx, soloStemName: STEM_NAMES[maxIdx],
             confidence: 0.3, rationale: 'near-silence, defaulting to solo' };
  }
  if (gap > MIX_GAP_DB) {
    return { isMix: false, soloStemIndex: maxIdx, soloStemName: STEM_NAMES[maxIdx],
             confidence: Math.min(0.99, gap / 30),
             rationale: `solo ${STEM_NAMES[maxIdx]} (gap=${gap.toFixed(1)} dB)` };
  }
  return { isMix: true, soloStemIndex: null, soloStemName: null,
           confidence: Math.min(0.99, 1 - gap / MIX_GAP_DB),
           rationale: `mix (gap=${gap.toFixed(1)} dB, max=${dominant.toFixed(1)} dB)` };
}

// ── Public API ────────────────────────────────────────────────────────────────
export async function initRmsDemucs() {
  if (_session) return _session;
  if (_promise)  return _promise;

  _promise = (async () => {
    const ort = await import('onnxruntime-web');
    _ort = ort;

    if (ort.env?.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';
      ort.env.wasm.numThreads = Math.min(4, navigator.hardwareConcurrency || 2);
      ort.env.wasm.simd = true;
    }

    const resp = await fetch(MODEL_URL, { cache: 'default' });
    if (!resp.ok) throw new Error(`demucs_rms_int8_v2.onnx HTTP ${resp.status}`);
    const graphBytes = new Uint8Array(await resp.arrayBuffer());

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
        console.log(`[rmsDemucs] ready on ${ep}`);
        return sess;
      } catch (err) {
        console.warn(`[rmsDemucs] ${ep} init failed:`, err?.message || err);
        lastErr = err;
      }
    }
    throw lastErr || new Error('no ORT backend available for rmsDemucs');
  })();

  return _promise;
}

export function isRmsDemucsReady() { return _session != null; }

/**
 * Analyze a stereo 48 kHz waveform.
 *
 * @param {Float32Array} flat  channels-first stereo [L0..L_{N-1}, R0..R_{N-1}]
 * @param {number}       numFrames
 * @returns {{
 *   stemEnvelopes:   Float32Array[]   length-4 array, each is amplitude over time
 *   stemEnvelopesFps: number          ≈ 93.75
 *   classification:  object
 *   perStemMeanDb:   Float32Array[4]  mean log-amplitude per stem
 * }}
 */
export async function analyzeRms(flat, numFrames) {
  const sess = await initRmsDemucs();
  const ort  = _ort;

  // Cap to 30 s for fast response (model is fast but we want sub-second)
  const maxSamples = TARGET_SR * 30;
  const useN       = Math.min(numFrames, maxSamples);

  // Mix to mono
  const mono = new Float32Array(useN);
  for (let i = 0; i < useN; i++) mono[i] = (flat[i] + flat[numFrames + i]) * 0.5;

  const { logMel, T } = computeLogMel(mono, N_FFT, HOP, N_MELS, TARGET_SR, FMIN);

  // ONNX input: [1, 1, N_MELS, T] — already in dB, no additional normalization
  const inp = new ort.Tensor('float32', logMel, [1, 1, N_MELS, T]);
  const res = await sess.run({ logmel: inp });
  const logits = res.logits.data;   // Float32Array [1, 4, T] flattened

  // Separate per-stem time-series (stored as [4, T] interleaved by ort)
  const stemEnvelopes = [];
  const perStemMeanDb = new Float32Array(N_STEMS);

  for (let s = 0; s < N_STEMS; s++) {
    const env = new Float32Array(T);
    let sumDb = 0;
    for (let t = 0; t < T; t++) {
      const db    = logits[s * T + t];
      env[t]      = Math.pow(10, db / 20);  // dB → linear amplitude
      sumDb      += db;
    }
    stemEnvelopes.push(env);
    perStemMeanDb[s] = sumDb / T;
  }

  const classification = classifyMixVsSolo(perStemMeanDb);

  return {
    stemEnvelopes,
    stemEnvelopesFps: RMS_FPS,
    classification,
    perStemMeanDb,
  };
}
