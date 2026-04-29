/**
 * dspMetrics.js — pure-JS DSP comparison metrics.
 *
 * All public functions accept two `Float32Array` inputs of the same length
 * (truncated/zero-padded to the shorter of the two if they differ) and a
 * `sampleRate` where relevant. None of these functions touch the DOM or
 * the Web Audio API, so they are safe to import from any module (worker,
 * Node, etc.) and trivial to unit-test.
 *
 * Conventions
 *   - Amplitudes are linear (e.g. 1.0 = digital full-scale).
 *   - dB outputs use 20·log10(amp) and are clamped to a floor of -240 dB
 *     to avoid -Infinity propagating through downstream code.
 *
 * Functions
 *   rmsDb(buf)                              → number  (dB FS)
 *   peakDb(buf)                             → number  (dB FS)
 *   rmsDiffDb(refBuf, webBuf)               → number  (dB; lower is better)
 *   peakDiffDb(refBuf, webBuf)              → number  (dB; lower is better)
 *   nullDiff(refBuf, webBuf)                → Float32Array  (refBuf-webBuf)
 *   thirdOctaveSpectralDiff(refBuf, webBuf, sampleRate, fftSize?) →
 *                                              [{ freq, diffDb }]
 *   downsampleForDisplay(buf, points)       → Float32Array  (max-pooled)
 *   poolMaxAbs(buf, points)                 → Float32Array  (peak-magnitude pool)
 *   alignBuffers(a, b)                      → [Float32Array, Float32Array]
 *
 * Acceptance:
 *   - identical buffers → all *Diff metrics return 0 (rms/peak) and an empty
 *     null-diff (within float-epsilon of zero per sample).
 *   - ref==zeros, web==zeros → rmsDiffDb / peakDiffDb return DB_FLOOR
 *     (clamped −∞).
 */

export const DB_FLOOR = -240;

// ── Helpers ───────────────────────────────────────────────────────────────

function safeDb(linear) {
  if (!isFinite(linear) || linear <= 0) return DB_FLOOR;
  return Math.max(DB_FLOOR, 20 * Math.log10(linear));
}

function asF32(buf) {
  if (buf instanceof Float32Array) return buf;
  if (Array.isArray(buf)) return Float32Array.from(buf);
  if (buf && typeof buf.length === 'number') return Float32Array.from(buf);
  return new Float32Array(0);
}

/** Truncate both inputs to the common length so we can subtract them. */
export function alignBuffers(a, b) {
  const aa = asF32(a);
  const bb = asF32(b);
  const n = Math.min(aa.length, bb.length);
  if (aa.length === n && bb.length === n) return [aa, bb];
  return [aa.subarray(0, n), bb.subarray(0, n)];
}

// ── Single-buffer level metrics ───────────────────────────────────────────

export function rmsLinear(buf) {
  const b = asF32(buf);
  const n = b.length;
  if (n === 0) return 0;
  let acc = 0;
  for (let i = 0; i < n; i++) acc += b[i] * b[i];
  return Math.sqrt(acc / n);
}

export function peakLinear(buf) {
  const b = asF32(buf);
  let p = 0;
  for (let i = 0; i < b.length; i++) {
    const v = Math.abs(b[i]);
    if (v > p) p = v;
  }
  return p;
}

export function rmsDb(buf)  { return safeDb(rmsLinear(buf)); }
export function peakDb(buf) { return safeDb(peakLinear(buf)); }

// ── Pairwise diff metrics (the core "is web faithful?" answers) ──────────

/**
 * `rmsDiffDb(ref, web)` — the RMS amplitude of the sample-wise difference
 * between two buffers, expressed in dB. This is the standard "null test":
 * if web exactly reproduces ref, this returns DB_FLOOR (≈ -240 dB).
 */
export function rmsDiffDb(refBuf, webBuf) {
  const [a, b] = alignBuffers(refBuf, webBuf);
  if (a.length === 0) return DB_FLOOR;
  let acc = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    acc += d * d;
  }
  return safeDb(Math.sqrt(acc / a.length));
}

export function peakDiffDb(refBuf, webBuf) {
  const [a, b] = alignBuffers(refBuf, webBuf);
  let p = 0;
  for (let i = 0; i < a.length; i++) {
    const d = Math.abs(a[i] - b[i]);
    if (d > p) p = d;
  }
  return safeDb(p);
}

export function nullDiff(refBuf, webBuf) {
  const [a, b] = alignBuffers(refBuf, webBuf);
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

// ── 1/3-octave spectral diff ──────────────────────────────────────────────

/**
 * Vanilla in-place radix-2 FFT. Operates on real `re` and imag `im` arrays
 * of identical length `n` where `n` is a power of two. We don't import a
 * 3rd-party FFT just so this stays a pure ESM JS file with zero deps.
 */
function fftInPlace(re, im) {
  const n = re.length;
  // Bit-reversal permutation
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      let tr = re[i]; re[i] = re[j]; re[j] = tr;
      let ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) { j -= m; m >>= 1; }
    j += m;
  }
  // Cooley-Tukey
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const tableStep = (-2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < half; k++) {
        const angle = tableStep * k;
        const cos = Math.cos(angle);
        const sin = Math.sin(angle);
        const tre =  cos * re[i + k + half] - sin * im[i + k + half];
        const tim =  sin * re[i + k + half] + cos * im[i + k + half];
        re[i + k + half] = re[i + k] - tre;
        im[i + k + half] = im[i + k] - tim;
        re[i + k] += tre;
        im[i + k] += tim;
      }
    }
  }
}

function nextPow2Below(n) {
  let p = 1;
  while ((p << 1) <= n) p <<= 1;
  return p;
}

/** Hann window applied in-place to `buf`. */
function applyHann(buf) {
  const n = buf.length;
  for (let i = 0; i < n; i++) {
    buf[i] *= 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1 || 1)));
  }
}

function magSpectrum(buf, fftSize) {
  const n = Math.min(buf.length, fftSize);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  for (let i = 0; i < n; i++) re[i] = buf[i];
  applyHann(re);
  fftInPlace(re, im);
  const half = fftSize >> 1;
  const mag = new Float32Array(half);
  for (let i = 0; i < half; i++) {
    mag[i] = Math.hypot(re[i], im[i]);
  }
  return mag;
}

// ISO 1/3-octave centre frequencies, audio band.
const THIRD_OCT_CENTRES = [
  20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630,
  800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000,
  12500, 16000, 20000,
];
const THIRD_OCT_RATIO = Math.pow(2, 1 / 6); // half a 1/3-octave span

/**
 * Compute |FFT(ref)| - |FFT(web)| binned into 1/3-octave bands, returned
 * as an array of `{ freq, diffDb }` pairs (the diff is band-averaged
 * magnitude, expressed in dB).
 *
 * Identical inputs → diffDb is DB_FLOOR for every band.
 */
export function thirdOctaveSpectralDiff(refBuf, webBuf, sampleRate, fftSize) {
  const [a, b] = alignBuffers(refBuf, webBuf);
  if (a.length === 0 || !sampleRate) {
    return THIRD_OCT_CENTRES.map((freq) => ({ freq, diffDb: DB_FLOOR }));
  }
  const N = fftSize || nextPow2Below(a.length) || 1024;
  const refMag = magSpectrum(a, N);
  const webMag = magSpectrum(b, N);
  const half = N >> 1;
  const binHz = sampleRate / N;

  return THIRD_OCT_CENTRES.map((fc) => {
    if (fc > sampleRate / 2) return { freq: fc, diffDb: DB_FLOOR };
    const lo = fc / THIRD_OCT_RATIO;
    const hi = fc * THIRD_OCT_RATIO;
    const i0 = Math.max(1, Math.floor(lo / binHz));
    const i1 = Math.min(half - 1, Math.ceil(hi / binHz));
    if (i1 < i0) return { freq: fc, diffDb: DB_FLOOR };

    let refSum = 0, webSum = 0;
    const span = i1 - i0 + 1;
    for (let i = i0; i <= i1; i++) {
      refSum += refMag[i];
      webSum += webMag[i];
    }
    const refAvg = refSum / span;
    const webAvg = webSum / span;
    const diffLin = Math.abs(refAvg - webAvg);
    return { freq: fc, diffDb: safeDb(diffLin) };
  });
}

// ── Display-side helpers ──────────────────────────────────────────────────

/**
 * Max-pool `buf` to `points` samples. Each output sample is the absolute
 * extremum (positive or negative) of its source bucket — preserves
 * transient peaks at any zoom level. Sign of the extremum is retained.
 */
export function poolMaxAbs(buf, points) {
  const b = asF32(buf);
  if (b.length === 0 || points <= 0) return new Float32Array(0);
  if (b.length <= points) return b.slice();
  const out = new Float32Array(points);
  const bucket = b.length / points;
  for (let i = 0; i < points; i++) {
    const start = Math.floor(i * bucket);
    const end = Math.min(b.length, Math.floor((i + 1) * bucket));
    let extremum = 0, absMax = 0;
    for (let j = start; j < end; j++) {
      const a = Math.abs(b[j]);
      if (a > absMax) { absMax = a; extremum = b[j]; }
    }
    out[i] = extremum;
  }
  return out;
}

/** Alias kept for callers that just want a "downsampled-for-display" array. */
export const downsampleForDisplay = poolMaxAbs;

// ── Default export (handy for `import metrics from '@/lib/dspMetrics'`) ──

const dspMetrics = {
  DB_FLOOR,
  rmsLinear, peakLinear, rmsDb, peakDb,
  rmsDiffDb, peakDiffDb, nullDiff,
  thirdOctaveSpectralDiff,
  poolMaxAbs, downsampleForDisplay,
  alignBuffers,
};

export default dspMetrics;
