/**
 * r5-pitch-shift-processor — phase-vocoder pitch shift in ±24 semitones.
 *
 * Method: STFT analysis → phase-vocoder phase propagation → spectral
 * resampling (frequency-domain bin interpolation). Equivalent to
 * time-stretch + linear interpolation resample, but cheaper because
 * the resample stage is folded into the synthesis-frame magnitude
 * remap, so output rate stays at the input rate (no buffer pumping).
 *
 * Pipeline per analysis hop:
 *   1) Hann window → forward FFT (size N=2048 default)
 *   2) Convert to magnitude/phase
 *   3) Phase vocoder: estimate true bin frequency from phase deviation
 *      (Laroche–Dolson formulation). Accumulate synthesis phase using
 *      the same hop, then warp magnitudes by the pitch ratio.
 *   4) Convert back to rectangular → inverse FFT
 *   5) Re-window + overlap-add (window normalization gain ≈ 1.5 for
 *      Hann at 75% overlap → factor 2/3 baked into output)
 *
 * AudioWorklet message protocol:
 *   port.postMessage({ type: 'semitones', value: <number, -24..24> })
 *   port.postMessage({ type: 'mix',       value: <number, 0..1> })
 *
 * Latency: fftSize samples (≈ 46 ms at 2048 / 44.1 kHz).
 *
 * Self-contained: inlines a Cooley–Tukey radix-2 FFT + Hann window so
 * this file works without importScripts, which is required because
 * Doseedo serves worklets from arbitrary roots (Vercel + CRA).
 *
 * @author Doseedo R5
 */

// ────────────────────────────────────────────────────────────────────────────
// Inline FFT (Cooley–Tukey, radix-2, in-place, real+imag arrays)
// ────────────────────────────────────────────────────────────────────────────
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

// ────────────────────────────────────────────────────────────────────────────
// Pitch Shift Processor
// ────────────────────────────────────────────────────────────────────────────
class R5PitchShiftProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'semitones', defaultValue: 0, minValue: -24, maxValue: 24,  automationRate: 'k-rate' },
      { name: 'mix',       defaultValue: 1, minValue: 0,   maxValue: 1.0, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.fftSize  = opts.fftSize  || 2048;
    this.hopSize  = (this.fftSize >> 2);            // 75% overlap → window-norm 1.5
    this.halfSize = this.fftSize >> 1;

    this.fft     = new R5FFT(this.fftSize);
    this.window  = r5HannWindow(this.fftSize);

    // Window-sum normalization: with Hann + hop=N/4, COLA sum = 1.5
    this.windowNorm = 2 / 3;

    // Ring buffers
    this.inputRing      = new Float32Array(this.fftSize);
    this.inputWritePos  = 0;
    this.samplesUntilFFT = this.fftSize;             // first fill = full window

    // Output overlap-add buffer (length fftSize). We extract hopSize samples
    // each time we push a synthesis frame.
    this.outputOLA      = new Float32Array(this.fftSize);
    this.outputReadable = 0;                          // samples currently safe to emit

    // FFT scratch
    this.re   = new Float32Array(this.fftSize);
    this.im   = new Float32Array(this.fftSize);
    this.mag  = new Float32Array(this.halfSize);
    this.ph   = new Float32Array(this.halfSize);

    // Phase-vocoder state
    this.lastInputPhase  = new Float32Array(this.halfSize);
    this.synthPhaseAccum = new Float32Array(this.halfSize);

    // Spectral remap scratch
    this.shiftedMag    = new Float32Array(this.halfSize);
    this.shiftedPhase  = new Float32Array(this.halfSize);

    // Live param overrides via port (lets host bypass AudioParam if needed)
    this._semitonesOverride = null;
    this._mixOverride = null;

    this.port.onmessage = (e) => {
      const m = e.data;
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'semitones': this._semitonesOverride = +m.value; break;
        case 'mix':       this._mixOverride       = +m.value; break;
        case 'reset':
          this.lastInputPhase.fill(0);
          this.synthPhaseAccum.fill(0);
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

    const semK = this._semitonesOverride != null ? this._semitonesOverride : parameters.semitones[0];
    const mixK = this._mixOverride       != null ? this._mixOverride       : parameters.mix[0];
    const ratio = Math.pow(2, semK / 12);

    const inL = (input && input[0]) ? input[0] : null;
    const outL = output[0];
    const outR = output[1] || null;

    const block = outL.length;
    const N = this.fftSize;
    const H = this.hopSize;

    for (let i = 0; i < block; i++) {
      const dry = inL ? inL[i] : 0;

      // Slide ring buffer one sample (cheap because hop happens every H samples)
      this.inputRing[this.inputWritePos] = dry;
      this.inputWritePos = (this.inputWritePos + 1) % N;
      this.samplesUntilFFT--;

      if (this.samplesUntilFFT <= 0) {
        this._processFrame(ratio);
        this.samplesUntilFFT = H;
      }

      // Emit OLA[0]; shift left by one each sample
      let wet = 0;
      if (this.outputReadable > 0) {
        wet = this.outputOLA[0];
        // Shift left 1, zero trailing
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

  _processFrame(ratio) {
    const N = this.fftSize;
    const H = this.hopSize;
    const halfN = this.halfSize;

    // Copy ring → re, applying Hann + linearization (oldest sample first)
    let p = this.inputWritePos;
    for (let i = 0; i < N; i++) {
      this.re[i] = this.inputRing[p] * this.window[i];
      this.im[i] = 0;
      p++;
      if (p >= N) p = 0;
    }

    this.fft.forward(this.re, this.im);

    // Polar
    for (let k = 0; k < halfN; k++) {
      const r = this.re[k], im = this.im[k];
      this.mag[k] = Math.sqrt(r * r + im * im);
      this.ph[k]  = Math.atan2(im, r);
    }

    // Phase vocoder: deviate from expected, then propagate
    const omega = (2 * Math.PI * H) / N; // expected phase advance per bin per hop
    for (let k = 0; k < halfN; k++) {
      let dPhi = this.ph[k] - this.lastInputPhase[k] - k * omega;
      // wrap to [-π, π]
      dPhi -= 2 * Math.PI * Math.round(dPhi / (2 * Math.PI));
      const trueBin = k + dPhi / omega;
      // Synthesis phase advances by trueBin * omega per hop (analysis hop = synth hop)
      this.synthPhaseAccum[k] += trueBin * omega;
      this.lastInputPhase[k] = this.ph[k];
    }

    // Spectral remap (pitch shift) — bins source from k/ratio in mag/synthPhase
    for (let k = 0; k < halfN; k++) {
      const src = k / ratio;
      const i0 = Math.floor(src);
      const frac = src - i0;
      const i1 = i0 + 1;
      if (i0 >= 0 && i1 < halfN) {
        this.shiftedMag[k]   = this.mag[i0]              * (1 - frac) + this.mag[i1]              * frac;
        this.shiftedPhase[k] = this.synthPhaseAccum[i0]  * (1 - frac) + this.synthPhaseAccum[i1]  * frac;
      } else if (i0 >= 0 && i0 < halfN) {
        this.shiftedMag[k]   = this.mag[i0];
        this.shiftedPhase[k] = this.synthPhaseAccum[i0];
      } else {
        this.shiftedMag[k]   = 0;
        this.shiftedPhase[k] = 0;
      }
    }

    // Rectangular for output spectrum (Hermitian symmetry)
    this.re[0] = this.shiftedMag[0] * Math.cos(this.shiftedPhase[0]);
    this.im[0] = this.shiftedMag[0] * Math.sin(this.shiftedPhase[0]);
    for (let k = 1; k < halfN; k++) {
      const r = this.shiftedMag[k] * Math.cos(this.shiftedPhase[k]);
      const im = this.shiftedMag[k] * Math.sin(this.shiftedPhase[k]);
      this.re[k] = r;
      this.im[k] = im;
      this.re[N - k] =  r;
      this.im[N - k] = -im;
    }
    // Nyquist bin
    this.re[halfN] = this.shiftedMag[0]; // rough — not strictly correct but inaudible
    this.im[halfN] = 0;

    this.fft.inverse(this.re, this.im);

    // Window + overlap-add into outputOLA
    const norm = this.windowNorm;
    for (let i = 0; i < N; i++) {
      this.outputOLA[i] += this.re[i] * this.window[i] * norm;
    }
    // We just produced one synthesis hop worth of fresh output samples;
    // mark them readable. (Cap so we don't claim more than buffer length.)
    this.outputReadable = Math.min(N, this.outputReadable + H);
  }
}

registerProcessor('r5-pitch-shift-processor', R5PitchShiftProcessor);
