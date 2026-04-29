/**
 * r5-spectral-freeze-processor — magnitude-latch spectral freeze.
 *
 * When the `freeze` parameter rises above 0.5, the worklet captures the
 * current FFT magnitude spectrum and replays it on every subsequent
 * synthesis frame. Phase is *advanced* per hop (not held) so we don't
 * collapse to a buzzy comb of fixed phases. Two phase modes are
 * supported via the constructor / port:
 *   - 'advance' (default): each frozen bin's phase advances by its
 *     measured "true bin" frequency at the moment of freeze, giving
 *     a tonal, harmonic-stable pad. This is the classic phase-vocoder
 *     freeze (Laroche-Dolson).
 *   - 'random': each frozen bin's phase is randomized each hop, giving
 *     a noisy, lush pad with phase incoherence (good for ambient).
 *
 * The `freeze` parameter is also a continuous crossfade between live
 * and frozen magnitude spectra, so values between 0 and 1 produce a
 * blend (live spectrum gradually replaced by frozen).
 *
 * AudioWorklet message protocol:
 *   port.postMessage({ type: 'freeze',     value: <0..1> })
 *   port.postMessage({ type: 'mix',        value: <0..1> })
 *   port.postMessage({ type: 'phase_mode', value: 'advance' | 'random' })
 *
 * Latency: fftSize samples (≈ 46 ms at 2048 / 44.1 kHz).
 *
 * Self-contained: inlines the same Cooley–Tukey radix-2 FFT used by the
 * R5 pitch-shift / spectral-filter worklets.
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

class R5SpectralFreezeProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'freeze', defaultValue: 0,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'mix',    defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.fftSize  = opts.fftSize  || 2048;
    this.hopSize  = this.fftSize >> 2;
    this.halfSize = this.fftSize >> 1;
    this.phaseMode = opts.phaseMode || 'advance'; // 'advance' | 'random'

    this.fft    = new R5FFT(this.fftSize);
    this.window = r5HannWindow(this.fftSize);
    this.windowNorm = 2 / 3;

    this.inputRing      = new Float32Array(this.fftSize);
    this.inputWritePos  = 0;
    this.samplesUntilFFT = this.fftSize;

    this.outputOLA      = new Float32Array(this.fftSize);
    this.outputReadable = 0;

    this.re   = new Float32Array(this.fftSize);
    this.im   = new Float32Array(this.fftSize);
    this.mag  = new Float32Array(this.halfSize);
    this.ph   = new Float32Array(this.halfSize);

    // Frozen-state buffers
    this.frozenMag       = new Float32Array(this.halfSize);
    this.frozenTrueBin   = new Float32Array(this.halfSize); // estimated bin freq at freeze time
    this.synthPhaseAccum = new Float32Array(this.halfSize);

    // Phase-vocoder analysis state for "true bin" estimation
    this.lastInputPhase = new Float32Array(this.halfSize);

    this.frozen = false;        // true once a snapshot has been captured
    this.freezeAmount = 0;      // 0..1, mirrors freeze param

    this._freezeOverride = null;
    this._mixOverride    = null;

    this.port.onmessage = (e) => {
      const m = e.data;
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'freeze':     this._freezeOverride = +m.value; break;
        case 'mix':        this._mixOverride    = +m.value; break;
        case 'phase_mode':
          if (m.value === 'advance' || m.value === 'random') this.phaseMode = m.value;
          break;
        case 'unfreeze':
          this.frozen = false;
          this.frozenMag.fill(0);
          this.frozenTrueBin.fill(0);
          break;
        case 'reset':
          this.frozen = false;
          this.frozenMag.fill(0);
          this.frozenTrueBin.fill(0);
          this.synthPhaseAccum.fill(0);
          this.lastInputPhase.fill(0);
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

    const freezeK = this._freezeOverride != null ? this._freezeOverride : parameters.freeze[0];
    const mixK    = this._mixOverride    != null ? this._mixOverride    : parameters.mix[0];

    this.freezeAmount = freezeK < 0 ? 0 : (freezeK > 1 ? 1 : freezeK);

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
        this._processFrame();
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

  _processFrame() {
    const N = this.fftSize;
    const H = this.hopSize;
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

    // Polar
    for (let k = 0; k < halfN; k++) {
      const r = this.re[k], im = this.im[k];
      this.mag[k] = Math.sqrt(r * r + im * im);
      this.ph[k]  = Math.atan2(im, r);
    }

    // Estimate true bin frequencies via phase vocoder (used for freeze capture
    // and for advance-phase mode while frozen).
    const omega = (2 * Math.PI * H) / N;
    const liveTrueBin = new Float32Array(halfN);
    for (let k = 0; k < halfN; k++) {
      let dPhi = this.ph[k] - this.lastInputPhase[k] - k * omega;
      dPhi -= 2 * Math.PI * Math.round(dPhi / (2 * Math.PI));
      liveTrueBin[k] = k + dPhi / omega;
      this.lastInputPhase[k] = this.ph[k];
    }

    // Decide whether to (re)capture the snapshot. We capture once per
    // rising edge above 0.5 to avoid re-snapping every hop.
    const wantFrozen = this.freezeAmount >= 0.5;
    if (wantFrozen && !this.frozen) {
      for (let k = 0; k < halfN; k++) {
        this.frozenMag[k]     = this.mag[k];
        this.frozenTrueBin[k] = liveTrueBin[k];
        // initialize synth phase from current input phase so we don't
        // jump-cut on capture
        this.synthPhaseAccum[k] = this.ph[k];
      }
      this.frozen = true;
    } else if (!wantFrozen && this.frozen) {
      // partial release: keep the snapshot but freezeAmount blends it down
      // (we leave this.frozen true so user can re-freeze without recapture
      //  on the way up — acts as a sticky latch around the 0.5 threshold)
    }

    // Build the synthesized magnitude: blend live and frozen by freezeAmount
    const fa = this.freezeAmount;
    const useFrozen = this.frozen;
    const outMag = new Float32Array(halfN);
    if (useFrozen) {
      for (let k = 0; k < halfN; k++) {
        outMag[k] = this.mag[k] * (1 - fa) + this.frozenMag[k] * fa;
      }
    } else {
      for (let k = 0; k < halfN; k++) outMag[k] = this.mag[k];
    }

    // Phase: depends on whether we're frozen and which mode
    const outPhase = new Float32Array(halfN);
    if (useFrozen && fa > 0) {
      if (this.phaseMode === 'random') {
        for (let k = 0; k < halfN; k++) {
          // Blend live phase (when fa < 1) with random phase
          const live = this.ph[k];
          const rand = (Math.random() * 2 - 1) * Math.PI;
          outPhase[k] = live * (1 - fa) + rand * fa;
        }
      } else {
        // 'advance' mode (default): per-frame phase accumulation by the
        // estimated true bin frequency captured at freeze time
        for (let k = 0; k < halfN; k++) {
          this.synthPhaseAccum[k] += this.frozenTrueBin[k] * omega;
          // Wrap to [-π, π] to avoid float blow-up over long freezes
          if (this.synthPhaseAccum[k] > Math.PI || this.synthPhaseAccum[k] < -Math.PI) {
            this.synthPhaseAccum[k] -= 2 * Math.PI * Math.round(this.synthPhaseAccum[k] / (2 * Math.PI));
          }
          outPhase[k] = this.ph[k] * (1 - fa) + this.synthPhaseAccum[k] * fa;
        }
      }
    } else {
      // Live passthrough phase
      for (let k = 0; k < halfN; k++) outPhase[k] = this.ph[k];
    }

    // Rebuild Hermitian-symmetric spectrum
    this.re[0] = outMag[0] * Math.cos(outPhase[0]);
    this.im[0] = outMag[0] * Math.sin(outPhase[0]);
    for (let k = 1; k < halfN; k++) {
      const r  = outMag[k] * Math.cos(outPhase[k]);
      const im = outMag[k] * Math.sin(outPhase[k]);
      this.re[k] = r;
      this.im[k] = im;
      this.re[N - k] =  r;
      this.im[N - k] = -im;
    }
    this.re[halfN] = 0;
    this.im[halfN] = 0;

    this.fft.inverse(this.re, this.im);

    const norm = this.windowNorm;
    for (let i = 0; i < N; i++) {
      this.outputOLA[i] += this.re[i] * this.window[i] * norm;
    }
    this.outputReadable = Math.min(N, this.outputReadable + H);
  }
}

registerProcessor('r5-spectral-freeze-processor', R5SpectralFreezeProcessor);
