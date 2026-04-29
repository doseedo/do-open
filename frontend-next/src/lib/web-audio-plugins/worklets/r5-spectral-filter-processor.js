/**
 * r5-spectral-filter-processor — FFT-bin gating between [low_bin, high_bin].
 *
 * Both bounds are normalized 0..1 mapping linearly to bins 0..(N/2). Bins
 * outside the [low, high] window are zeroed in the magnitude spectrum
 * before resynthesis. The transition is hard (rectangular), but the
 * Hann analysis/synthesis windows give an effective ~hop-cycle smoothing
 * across time, so audible artifacts are minor unless cutoffs are swept
 * very fast.
 *
 * AudioWorklet message protocol:
 *   port.postMessage({ type: 'low_bin',  value: <0..1> })
 *   port.postMessage({ type: 'high_bin', value: <0..1> })
 *   port.postMessage({ type: 'mix',      value: <0..1> })
 *
 * Latency: fftSize samples (≈ 46 ms at 2048 / 44.1 kHz).
 *
 * Self-contained: inlines the same Cooley–Tukey radix-2 FFT used by the
 * R5 pitch-shift worklet, so this file is independently registrable.
 *
 * @author Doseedo R5
 */

class R5FFT {
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

function r5HannWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  return w;
}

class R5SpectralFilterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'lowBin',  defaultValue: 0,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'highBin', defaultValue: 1,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'mix',     defaultValue: 1,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.fftSize  = opts.fftSize || 2048;
    this.hopSize  = this.fftSize >> 2;
    this.halfSize = this.fftSize >> 1;

    this.fft    = new R5FFT(this.fftSize);
    this.window = r5HannWindow(this.fftSize);
    this.windowNorm = 2 / 3;

    this.inputRing      = new Float32Array(this.fftSize);
    this.inputWritePos  = 0;
    this.samplesUntilFFT = this.fftSize;

    this.outputOLA      = new Float32Array(this.fftSize);
    this.outputReadable = 0;

    this.re = new Float32Array(this.fftSize);
    this.im = new Float32Array(this.fftSize);

    this._lowOverride  = null;
    this._highOverride = null;
    this._mixOverride  = null;

    this.port.onmessage = (e) => {
      const m = e.data;
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'low_bin':  this._lowOverride  = +m.value; break;
        case 'high_bin': this._highOverride = +m.value; break;
        case 'mix':      this._mixOverride  = +m.value; break;
        case 'reset':
          this.outputOLA.fill(0);
          this.outputReadable = 0;
          break;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input  = inputs[0];
    const output = outputs[0];
    if (!output || output.length < 1) return true;

    let lo = this._lowOverride  != null ? this._lowOverride  : parameters.lowBin[0];
    let hi = this._highOverride != null ? this._highOverride : parameters.highBin[0];
    const mixK = this._mixOverride != null ? this._mixOverride : parameters.mix[0];

    if (lo < 0) lo = 0; if (lo > 1) lo = 1;
    if (hi < 0) hi = 0; if (hi > 1) hi = 1;
    if (lo > hi) { const t = lo; lo = hi; hi = t; }

    const halfN = this.halfSize;
    const loBin = Math.floor(lo * halfN);
    const hiBin = Math.ceil(hi * halfN);

    const inL = (input && input[0]) ? input[0] : null;
    const outL = output[0];
    const outR = output[1] || null;

    const block = outL.length;
    const N = this.fftSize;
    const H = this.hopSize;

    for (let i = 0; i < block; i++) {
      const dry = inL ? inL[i] : 0;

      this.inputRing[this.inputWritePos] = dry;
      this.inputWritePos = (this.inputWritePos + 1) % N;
      this.samplesUntilFFT--;

      if (this.samplesUntilFFT <= 0) {
        this._processFrame(loBin, hiBin);
        this.samplesUntilFFT = H;
      }

      let wet = 0;
      if (this.outputReadable > 0) {
        wet = this.outputOLA[0];
        for (let s = 0; s < N - 1; s++) this.outputOLA[s] = this.outputOLA[s + 1];
        this.outputOLA[N - 1] = 0;
        this.outputReadable--;
      }

      const y = dry * (1 - mixK) + wet * mixK;
      outL[i] = y;
      if (outR) outR[i] = y;
    }

    return true;
  }

  _processFrame(loBin, hiBin) {
    const N = this.fftSize;
    const halfN = this.halfSize;

    // ring → window → re
    let p = this.inputWritePos;
    for (let i = 0; i < N; i++) {
      this.re[i] = this.inputRing[p] * this.window[i];
      this.im[i] = 0;
      p++;
      if (p >= N) p = 0;
    }

    this.fft.forward(this.re, this.im);

    // Bin gate: zero everything outside [loBin, hiBin], maintain Hermitian symmetry
    for (let k = 0; k < halfN; k++) {
      const inside = (k >= loBin && k < hiBin);
      if (!inside) {
        this.re[k] = 0;
        this.im[k] = 0;
        if (k > 0) {
          this.re[N - k] = 0;
          this.im[N - k] = 0;
        }
      } else {
        // Hermitian conjugate must match for real-output ifft
        if (k > 0) {
          this.re[N - k] =  this.re[k];
          this.im[N - k] = -this.im[k];
        }
      }
    }
    // Nyquist bin
    if (halfN >= loBin && halfN < hiBin) {
      // keep
    } else {
      this.re[halfN] = 0;
      this.im[halfN] = 0;
    }

    this.fft.inverse(this.re, this.im);

    const norm = this.windowNorm;
    for (let i = 0; i < N; i++) {
      this.outputOLA[i] += this.re[i] * this.window[i] * norm;
    }
    this.outputReadable = Math.min(N, this.outputReadable + this.hopSize);
  }
}

registerProcessor('r5-spectral-filter-processor', R5SpectralFilterProcessor);
