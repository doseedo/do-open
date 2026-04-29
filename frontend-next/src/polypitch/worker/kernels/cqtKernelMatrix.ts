/**
 * cqtKernelMatrix — pure TypeScript sparse CQT kernel matrix builder.
 *
 * Follows the Brown/Puckette 1992 constant-Q design: for each cent-bin k
 * with center frequency f_k, build an analysis window of length
 *     N_k = ceil(Q * sr / f_k),      Q = 1 / (2^(1/bins_per_octave) - 1)
 * containing a Hann-windowed complex exponential exp(+j 2π f_k n / sr).
 *
 * The dense CQT kernel row is the FFT of that time-domain window,
 * truncated to the analysis FFT size. Following Schörkhuber & Klapuri
 * 2010, we throw away bins below a magnitude threshold to make the
 * resulting matrix very sparse (~1% non-zero for typical configs).
 *
 * The resulting CSR sparse matrix is multiplied at runtime against each
 * frame's complex STFT (see hcqt.wgsl). For HCQT we stack `nHarmonics`
 * independent kernel blocks into one big matrix; the row index encodes
 * (harmonic_idx * nCentBins + cent_idx).
 *
 * This file deliberately has no WebGPU / DOM dependencies — it's a pure
 * builder so it can be CPU-tested in vitest with no GPU needed.
 */

export interface CQTKernelMatrixConfig {
  readonly binsPerOctave: number;
  readonly nOctaves: number;
  readonly nHarmonics: number;
  readonly harmonicScales: readonly number[];  // length === nHarmonics
  readonly fminHz: number;
  readonly sampleRate: number;
  readonly nFft: number;
  readonly magnitudeThreshold?: number;  // default 5e-3 relative to max per row
}

export interface SparseCQTKernel {
  readonly valuesRe: Float32Array;
  readonly valuesIm: Float32Array;
  readonly colIndices: Uint32Array;
  readonly rowStart: Uint32Array;   // length = nRows + 1
  readonly nRows: number;
  readonly nCols: number;           // = nFft / 2 + 1 (rfft bins) OR nFft (full)
  readonly nnz: number;
  readonly binsPerHarmonic: number;
}

function hann(n: number, N: number): number {
  // Symmetric Hann for the analysis window; Brown's paper uses symmetric.
  return 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / Math.max(N - 1, 1));
}

/**
 * In-place radix-2 FFT used for kernel-matrix construction only (not
 * the hot path). nFft must be a power of two.
 */
function fftInPlace(re: Float64Array, im: Float64Array): void {
  const N = re.length;
  // Bit reversal.
  let j = 0;
  for (let i = 0; i < N; i++) {
    if (i < j) {
      const tr = re[i]; re[i] = re[j]; re[j] = tr;
      const ti = im[i]; im[i] = im[j]; im[j] = ti;
    }
    let m = N >>> 1;
    while (m >= 1 && j >= m) { j -= m; m >>>= 1; }
    j += m;
  }
  for (let len = 2; len <= N; len <<= 1) {
    const half = len >>> 1;
    const theta = (-2 * Math.PI) / len;
    const wR = Math.cos(theta);
    const wI = Math.sin(theta);
    for (let i = 0; i < N; i += len) {
      let wr = 1, wi = 0;
      for (let k = 0; k < half; k++) {
        const a = i + k;
        const b = a + half;
        const tr = wr * re[b] - wi * im[b];
        const ti = wr * im[b] + wi * re[b];
        re[b] = re[a] - tr; im[b] = im[a] - ti;
        re[a] = re[a] + tr; im[a] = im[a] + ti;
        const nwr = wr * wR - wi * wI;
        const nwi = wr * wI + wi * wR;
        wr = nwr; wi = nwi;
      }
    }
  }
}

export function buildCQTKernelMatrix(cfg: CQTKernelMatrixConfig): SparseCQTKernel {
  if (cfg.harmonicScales.length !== cfg.nHarmonics) {
    throw new Error("cqtKernelMatrix: harmonicScales.length must equal nHarmonics");
  }

  const binsPerHarmonic = cfg.nOctaves * cfg.binsPerOctave;
  const nRows = cfg.nHarmonics * binsPerHarmonic;
  const nCols = Math.floor(cfg.nFft / 2) + 1;  // rfft layout (we ignore mirror)
  const thresh = cfg.magnitudeThreshold ?? 5e-3;

  const Q = 1 / (Math.pow(2, 1 / cfg.binsPerOctave) - 1);

  // Build CSR incrementally.
  const rowStart = new Uint32Array(nRows + 1);
  const valuesReBuf: number[] = [];
  const valuesImBuf: number[] = [];
  const colIdxBuf: number[] = [];

  const nyquist = cfg.sampleRate * 0.5;

  for (let h = 0; h < cfg.nHarmonics; h++) {
    const scale = cfg.harmonicScales[h];
    for (let b = 0; b < binsPerHarmonic; b++) {
      const rowIdx = h * binsPerHarmonic + b;
      rowStart[rowIdx] = valuesReBuf.length;

      // Center frequency for this cent-bin (scaled by harmonic factor).
      const f_k = cfg.fminHz * Math.pow(2, b / cfg.binsPerOctave) * scale;
      // Skip bins outside the representable range; emit an empty row.
      if (f_k <= 0 || f_k >= nyquist * 0.98) {
        continue;
      }

      // Analysis window length N_k = ceil(Q * sr / f_k). Cap at nFft.
      let N_k = Math.ceil((Q * cfg.sampleRate) / f_k);
      if (N_k < 4) N_k = 4;
      if (N_k > cfg.nFft) N_k = cfg.nFft;

      // Build the complex time-domain kernel centered in an nFft window.
      // Zero-pad so the FFT matches STFT bin alignment.
      const sigRe = new Float64Array(cfg.nFft);
      const sigIm = new Float64Array(cfg.nFft);
      // Center the window at n = nFft/2 so the FFT has zero phase at the
      // center frequency (standard Schörkhuber trick).
      const centerOffset = Math.floor((cfg.nFft - N_k) / 2);
      let windowEnergy = 0;
      for (let n = 0; n < N_k; n++) {
        const w = hann(n, N_k);
        windowEnergy += w;
        const t = n; // sample index within the window
        const ang = (2 * Math.PI * f_k * t) / cfg.sampleRate;
        sigRe[centerOffset + n] = (w * Math.cos(ang)) / N_k;
        sigIm[centerOffset + n] = (w * Math.sin(ang)) / N_k;
      }
      // Normalize so the FFT of this kernel has unit magnitude at f_k.
      // Hann sum ≈ N_k / 2; we already scaled by 1/N_k above, so the
      // FFT magnitude peak will be ~0.5. Compensate:
      if (windowEnergy > 0) {
        const scale = (2 * N_k) / Math.max(windowEnergy, 1e-12);
        for (let n = 0; n < N_k; n++) {
          sigRe[centerOffset + n] *= scale / N_k;
          sigIm[centerOffset + n] *= scale / N_k;
        }
      }

      // FFT the time-domain kernel and keep only the first nCols rfft bins.
      // Copy into working buffers since fftInPlace mutates.
      const re = new Float64Array(sigRe);
      const im = new Float64Array(sigIm);
      fftInPlace(re, im);

      // Peak magnitude across kept bins → threshold.
      let peak = 0;
      for (let k = 0; k < nCols; k++) {
        const m = Math.hypot(re[k], im[k]);
        if (m > peak) peak = m;
      }
      const cutoff = peak * thresh;

      for (let k = 0; k < nCols; k++) {
        const m = Math.hypot(re[k], im[k]);
        if (m >= cutoff && m > 0) {
          // We want the matrix-vector product y[k_row] = sum_j K[k_row, j] * X[j]
          // to recover the CQT coefficient. Given our construction (kernel is
          // time-domain prototype FFT'd to freq), the analysis is the inner
          // product of the STFT with the CONJUGATE of the frequency-domain
          // kernel:
          //   y = <X, K> = sum_j X[j] * conj(K_freq[j])
          // So we store conj(K_freq) in the sparse values. Equivalent to
          // negating the imaginary part.
          valuesReBuf.push(re[k]);
          valuesImBuf.push(-im[k]);
          colIdxBuf.push(k);
        }
      }
    }
  }
  rowStart[nRows] = valuesReBuf.length;

  return {
    valuesRe: new Float32Array(valuesReBuf),
    valuesIm: new Float32Array(valuesImBuf),
    colIndices: new Uint32Array(colIdxBuf),
    rowStart,
    nRows,
    nCols,
    nnz: valuesReBuf.length,
    binsPerHarmonic,
  };
}

/**
 * Dense-reference matmul for tests: y[row] = sum_j K[row,j] * x[j] where
 * K is the sparse CSR stored in `kernel` and x is a complex input of
 * length `nCols` (interleaved re/im). Returns complex output per row.
 */
export function sparseMatvecComplex(
  kernel: SparseCQTKernel,
  x: Float32Array  // [nCols, 2] interleaved
): { re: Float32Array; im: Float32Array } {
  const re = new Float32Array(kernel.nRows);
  const im = new Float32Array(kernel.nRows);
  for (let r = 0; r < kernel.nRows; r++) {
    let ar = 0;
    let ai = 0;
    const start = kernel.rowStart[r];
    const end = kernel.rowStart[r + 1];
    for (let j = start; j < end; j++) {
      const col = kernel.colIndices[j];
      const kr = kernel.valuesRe[j];
      const ki = kernel.valuesIm[j];
      const xr = x[col * 2];
      const xi = x[col * 2 + 1];
      ar += kr * xr - ki * xi;
      ai += kr * xi + ki * xr;
    }
    re[r] = ar;
    im[r] = ai;
  }
  return { re, im };
}
