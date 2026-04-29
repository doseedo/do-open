/**
 * r13-match-eq-processor — Match-EQ analyzer + applier in a single worklet.
 *
 * Mirrors Logic Pro's "Match EQ":
 *   1. ANALYZE the long-term magnitude spectrum of a "target" reference
 *      (mode='analyze_target') and a "source" current signal
 *      (mode='analyze_source'). Both modes pass audio through unchanged
 *      and accumulate magnitude averages frame by frame.
 *   2. Once both spectra are captured (host-side, by reading the
 *      `spectrum_average` messages this worklet emits), the host computes
 *      a match curve (target / source), smooths it, and posts it back via
 *      port.postMessage({ type: 'set_curve', curve: Float32Array(N/2) }).
 *      The host can also pre-load `target_curve` / `source_curve` via
 *      `processorOptions` and request `recompute` to derive `match_curve`
 *      inside the worklet.
 *   3. APPLY the curve as an FFT-domain magnitude multiplier
 *      (mode='apply'). Implementation is fast-convolution-style: window,
 *      FFT, multiply each bin's complex value by the curve gain, IFFT,
 *      overlap-add. Identical OLA scaffolding as r5-spectral-filter-processor.
 *
 * Smoothing is performed host-side ahead of `set_curve`. The worklet
 * additionally clamps the applied curve via [low_bin, high_bin] so the
 * matched range can be limited to the mid spectrum (stock Logic exposes
 * Low/High Cut frequencies for the same purpose).
 *
 * Param surface (k-rate AudioParams + port messages):
 *   AudioParams: `mode` (0=analyze_target, 1=analyze_source, 2=apply),
 *                `amount` (0..1), `lowBin` (0..1), `highBin` (0..1),
 *                `gainMakeup` (linear, default 1.0)
 *   port:        { type: 'set_curve',     curve: Float32Array }
 *                { type: 'set_target',    curve: Float32Array }   // analyzer override
 *                { type: 'set_source',    curve: Float32Array }
 *                { type: 'recompute' }                            // derive match=target/source internally
 *                { type: 'reset_analysis' }
 *                { type: 'request_spectrum' }                     // emit current accumulated avg
 *                { type: 'reset' }
 *   port emits:  { type: 'spectrum_average', mode, framesAccumulated, magnitudes }
 *                { type: 'curve_ready', length }
 *
 * Latency: fftSize samples (≈ 92 ms at 4096 / 44.1 kHz) when in apply mode.
 * Analyze mode is true pass-through (zero added latency in practice; FFT
 * runs in parallel with the audio passing through).
 *
 * Self-contained: inlines the same radix-2 FFT used by R5.
 *
 * @author Doseedo R13
 */

class R13FFT {
  constructor(size) {
    this.size = size;
    this.half = size >> 1;
    this.cosT = new Float32Array(this.half);
    this.sinT = new Float32Array(this.half);
    for (let i = 0; i < this.half; i++) {
      const a = (-2 * Math.PI * i) / size;
      this.cosT[i] = Math.cos(a);
      this.sinT[i] = Math.sin(a);
    }
    const bits = Math.log2(size) | 0;
    this.rev = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let r = 0;
      for (let j = 0; j < bits; j++) r = (r << 1) | ((i >> j) & 1);
      this.rev[i] = r;
    }
  }
  _bitReverse(re, im) {
    const rev = this.rev;
    for (let i = 0; i < this.size; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
  }
  forward(re, im) {
    this._bitReverse(re, im);
    const N = this.size;
    for (let block = 2; block <= N; block <<= 1) {
      const halfBlock = block >> 1;
      const step = N / block;
      for (let i = 0; i < N; i += block) {
        for (let j = i, k = 0; j < i + halfBlock; j++, k += step) {
          const l = j + halfBlock;
          const wr = this.cosT[k];
          const wi = this.sinT[k];
          const tr = re[l] * wr - im[l] * wi;
          const ti = re[l] * wi + im[l] * wr;
          re[l] = re[j] - tr;
          im[l] = im[j] - ti;
          re[j] += tr;
          im[j] += ti;
        }
      }
    }
  }
  inverse(re, im) {
    for (let i = 0; i < this.size; i++) im[i] = -im[i];
    this.forward(re, im);
    const inv = 1 / this.size;
    for (let i = 0; i < this.size; i++) {
      re[i] *= inv;
      im[i] = -im[i] * inv;
    }
  }
}

function r13HannWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  return w;
}

// Mode enum (numeric — matches AudioParam values)
const MODE_ANALYZE_TARGET = 0;
const MODE_ANALYZE_SOURCE = 1;
const MODE_APPLY          = 2;

class R13MatchEQProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'mode',       defaultValue: 0,   minValue: 0, maxValue: 2, automationRate: 'k-rate' },
      { name: 'amount',     defaultValue: 1,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'lowBin',     defaultValue: 0,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'highBin',    defaultValue: 1,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'gainMakeup', defaultValue: 1,   minValue: 0, maxValue: 8, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.fftSize  = opts.fftSize || 4096;
    this.hopSize  = this.fftSize >> 2;
    this.halfSize = this.fftSize >> 1;

    this.fft    = new R13FFT(this.fftSize);
    this.window = r13HannWindow(this.fftSize);
    // Hann + 75% overlap → analysis-window²-sum normalising factor
    this.windowNorm = 2 / 3;

    // Ring buffer for input frames
    this.inputRing       = new Float32Array(this.fftSize);
    this.inputWritePos   = 0;
    this.samplesUntilFFT = this.fftSize;

    // Output OLA buffer (apply mode only)
    this.outputOLA      = new Float32Array(this.fftSize);
    this.outputReadable = 0;

    // FFT scratch
    this.re = new Float32Array(this.fftSize);
    this.im = new Float32Array(this.fftSize);

    // Magnitude accumulators (size = halfSize+1 to include Nyquist)
    this.targetMag    = new Float32Array(this.halfSize + 1);
    this.sourceMag    = new Float32Array(this.halfSize + 1);
    this.targetFrames = 0;
    this.sourceFrames = 0;

    // Active match curve (linear gain per bin, halfSize+1 entries).
    // Defaults to flat 1.0 (no change).
    this.matchCurve = new Float32Array(this.halfSize + 1).fill(1);

    // Optional pre-loaded curves via processorOptions
    if (opts.targetCurve && opts.targetCurve.length === this.halfSize + 1) {
      this.targetMag.set(opts.targetCurve);
      this.targetFrames = 1;
    }
    if (opts.sourceCurve && opts.sourceCurve.length === this.halfSize + 1) {
      this.sourceMag.set(opts.sourceCurve);
      this.sourceFrames = 1;
    }
    if (opts.matchCurve && opts.matchCurve.length === this.halfSize + 1) {
      this.matchCurve.set(opts.matchCurve);
    }

    this._modeOverride       = null;
    this._amountOverride     = null;
    this._lowOverride        = null;
    this._highOverride       = null;
    this._gainMakeupOverride = null;

    this.port.onmessage = (e) => {
      const m = e.data;
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'mode': {
          if (typeof m.value === 'number') this._modeOverride = m.value | 0;
          else if (typeof m.value === 'string') {
            const s = m.value.toLowerCase();
            if (s === 'analyze_target') this._modeOverride = MODE_ANALYZE_TARGET;
            else if (s === 'analyze_source') this._modeOverride = MODE_ANALYZE_SOURCE;
            else if (s === 'apply') this._modeOverride = MODE_APPLY;
          }
          break;
        }
        case 'amount':      this._amountOverride     = +m.value; break;
        case 'low_bin':     this._lowOverride        = +m.value; break;
        case 'high_bin':    this._highOverride       = +m.value; break;
        case 'gain_makeup': this._gainMakeupOverride = +m.value; break;
        case 'set_curve':
          if (m.curve && m.curve.length) {
            this._installCurve(this.matchCurve, m.curve);
            this.port.postMessage({ type: 'curve_ready', length: this.matchCurve.length });
          }
          break;
        case 'set_target':
          if (m.curve && m.curve.length) {
            this._installCurve(this.targetMag, m.curve);
            this.targetFrames = 1;
          }
          break;
        case 'set_source':
          if (m.curve && m.curve.length) {
            this._installCurve(this.sourceMag, m.curve);
            this.sourceFrames = 1;
          }
          break;
        case 'recompute':
          this._recomputeMatchCurve();
          this.port.postMessage({ type: 'curve_ready', length: this.matchCurve.length });
          break;
        case 'reset_analysis':
          this.targetMag.fill(0);
          this.sourceMag.fill(0);
          this.targetFrames = 0;
          this.sourceFrames = 0;
          break;
        case 'request_spectrum':
          this.port.postMessage({
            type: 'spectrum_average',
            mode: m.which === 'source' ? 'source' : 'target',
            framesAccumulated: m.which === 'source' ? this.sourceFrames : this.targetFrames,
            magnitudes: m.which === 'source' ? Float32Array.from(this.sourceMag) : Float32Array.from(this.targetMag),
          });
          break;
        case 'reset':
          this.outputOLA.fill(0);
          this.outputReadable = 0;
          this.samplesUntilFFT = this.fftSize;
          break;
      }
    };
  }

  // Copy `src` into `dst` of size halfSize+1, linearly resampling if lengths differ.
  _installCurve(dst, src) {
    const N = dst.length;
    const M = src.length;
    if (N === M) {
      for (let i = 0; i < N; i++) dst[i] = +src[i];
      return;
    }
    for (let i = 0; i < N; i++) {
      const t = (i * (M - 1)) / (N - 1);
      const lo = Math.floor(t);
      const hi = Math.min(M - 1, lo + 1);
      const frac = t - lo;
      dst[i] = src[lo] * (1 - frac) + src[hi] * frac;
    }
  }

  // match = target / source, with epsilon to avoid div-by-zero. Stays in
  // linear-magnitude domain. Smoothing is host-side.
  _recomputeMatchCurve() {
    const eps = 1e-9;
    const N = this.matchCurve.length;
    const targetScale = this.targetFrames > 0 ? 1 / this.targetFrames : 1;
    const sourceScale = this.sourceFrames > 0 ? 1 / this.sourceFrames : 1;
    for (let k = 0; k < N; k++) {
      const t = this.targetMag[k] * targetScale;
      const s = this.sourceMag[k] * sourceScale;
      this.matchCurve[k] = t / Math.max(eps, s);
    }
  }

  process(inputs, outputs, parameters) {
    const input  = inputs[0];
    const output = outputs[0];
    if (!output || output.length < 1) return true;

    const mode = (this._modeOverride != null
      ? this._modeOverride
      : Math.round(parameters.mode[0])) | 0;

    let lo = this._lowOverride  != null ? this._lowOverride  : parameters.lowBin[0];
    let hi = this._highOverride != null ? this._highOverride : parameters.highBin[0];
    const amount     = this._amountOverride     != null ? this._amountOverride     : parameters.amount[0];
    const gainMakeup = this._gainMakeupOverride != null ? this._gainMakeupOverride : parameters.gainMakeup[0];

    if (lo < 0) lo = 0; if (lo > 1) lo = 1;
    if (hi < 0) hi = 0; if (hi > 1) hi = 1;
    if (lo > hi) { const t = lo; lo = hi; hi = t; }

    const halfN = this.halfSize;
    const loBin = Math.floor(lo * halfN);
    const hiBin = Math.ceil(hi * halfN);

    const inL  = (input && input[0]) ? input[0] : null;
    const outL = output[0];
    const outR = output[1] || null;

    const block = outL.length;
    const N = this.fftSize;
    const H = this.hopSize;

    if (mode !== MODE_APPLY) {
      // Analyze mode — pass through audio, accumulate spectra in the
      // background (only on FFT cadence so cost is amortized).
      for (let i = 0; i < block; i++) {
        const dry = inL ? inL[i] : 0;

        this.inputRing[this.inputWritePos] = dry;
        this.inputWritePos = (this.inputWritePos + 1) % N;
        this.samplesUntilFFT--;

        if (this.samplesUntilFFT <= 0) {
          this._accumulateFrame(mode);
          this.samplesUntilFFT = H;
        }

        outL[i] = dry;
        if (outR) outR[i] = dry;
      }
      return true;
    }

    // Apply mode — FFT, multiply by match curve, IFFT, OLA.
    for (let i = 0; i < block; i++) {
      const dry = inL ? inL[i] : 0;

      this.inputRing[this.inputWritePos] = dry;
      this.inputWritePos = (this.inputWritePos + 1) % N;
      this.samplesUntilFFT--;

      if (this.samplesUntilFFT <= 0) {
        this._processApplyFrame(loBin, hiBin, amount, gainMakeup);
        this.samplesUntilFFT = H;
      }

      let wet = 0;
      if (this.outputReadable > 0) {
        wet = this.outputOLA[0];
        for (let s = 0; s < N - 1; s++) this.outputOLA[s] = this.outputOLA[s + 1];
        this.outputOLA[N - 1] = 0;
        this.outputReadable--;
      }

      outL[i] = wet;
      if (outR) outR[i] = wet;
    }

    return true;
  }

  _accumulateFrame(mode) {
    const N = this.fftSize;
    const halfN = this.halfSize;

    let p = this.inputWritePos;
    for (let i = 0; i < N; i++) {
      this.re[i] = this.inputRing[p] * this.window[i];
      this.im[i] = 0;
      p++;
      if (p >= N) p = 0;
    }

    this.fft.forward(this.re, this.im);

    const mag = (mode === MODE_ANALYZE_SOURCE) ? this.sourceMag : this.targetMag;
    for (let k = 0; k <= halfN; k++) {
      const m = Math.sqrt(this.re[k] * this.re[k] + this.im[k] * this.im[k]);
      mag[k] += m;
    }
    if (mode === MODE_ANALYZE_SOURCE) this.sourceFrames++;
    else this.targetFrames++;
  }

  _processApplyFrame(loBin, hiBin, amount, gainMakeup) {
    const N = this.fftSize;
    const halfN = this.halfSize;

    let p = this.inputWritePos;
    for (let i = 0; i < N; i++) {
      this.re[i] = this.inputRing[p] * this.window[i];
      this.im[i] = 0;
      p++;
      if (p >= N) p = 0;
    }

    this.fft.forward(this.re, this.im);

    // Per-bin magnitude scaling. amount=0 → identity (curve_gain=1).
    // amount=1 → full curve. We blend in linear domain.
    for (let k = 0; k <= halfN; k++) {
      let g;
      if (k < loBin || k >= hiBin) {
        g = 1; // outside user-selected range → leave bin untouched
      } else {
        const target = this.matchCurve[k];
        // Guard against absurd boosts (unstable when a bin's source ≈ 0)
        const clamped = Math.min(64, Math.max(1 / 64, target));
        g = 1 + amount * (clamped - 1);
      }
      g *= gainMakeup;

      this.re[k] *= g;
      this.im[k] *= g;
      if (k > 0 && k < halfN) {
        this.re[N - k] =  this.re[k];
        this.im[N - k] = -this.im[k];
      }
    }

    this.fft.inverse(this.re, this.im);

    const norm = this.windowNorm;
    for (let i = 0; i < N; i++) {
      this.outputOLA[i] += this.re[i] * this.window[i] * norm;
    }
    this.outputReadable = Math.min(N, this.outputReadable + this.hopSize);
  }
}

registerProcessor('r13-match-eq-processor', R13MatchEQProcessor);
