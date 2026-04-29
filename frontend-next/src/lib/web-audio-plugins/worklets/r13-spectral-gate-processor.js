/**
 * r13-spectral-gate-processor — frequency-domain noise gate.
 *
 * Pipeline per FFT frame (Hann window, 75% overlap):
 *   1. Read the latest fftSize samples from the input ring buffer.
 *   2. Apply Hann analysis window, FFT.
 *   3. For every bin k in [0..N/2]:
 *        magdB(k)   = 20·log10(max(|X[k]|, 1e-12))
 *        thresh(k)  = thresholdDb + tilt_curve(k)
 *        below      = magdB(k) < thresh(k) and bin in [low_cut_bin, high_cut_bin]
 *        target_g(k)= below ? 10^(reduction_db/20) : 1.0
 *      Smooth target_g with a per-bin one-pole low-pass envelope:
 *        coef = below ? attackCoef : releaseCoef
 *        env(k) ← env(k) + coef * (target_g - env(k))
 *      X'[k] = X[k] * env(k)
 *   4. Maintain Hermitian symmetry on the upper half so the IFFT is real.
 *   5. IFFT, Hann synthesis window, OLA into the output ring.
 *   6. Per output sample: y = dry * (1-mix) + wet * mix.
 *
 * Out-of-band bins (k < lowCutBin or k >= highCutBin) are excluded from the
 * gate (env forced to 1) so bass bins don't get gated out by a global threshold.
 *
 * AudioParams (k-rate):
 *   thresholdDb      -60..0     dB
 *   reductionDb      -60..0     dB (how much to attenuate gated bins)
 *   attackMs           1..100   ms
 *   releaseMs         10..1000  ms
 *   lowCut            0..1      normalized → bin 0..N/2
 *   highCut           0..1      normalized → bin 0..N/2
 *   tiltDb           -12..+12   dB tilt across frequency (− = stricter on highs)
 *   mix               0..1      wet/dry blend
 *
 * Optional port messages mirror the same names with snake_case for parity
 * with the spectral_filter / spectral_freeze worklets.
 *
 * Self-contained: inlines the same Cooley–Tukey radix-2 FFT used elsewhere
 * in R5, so the file can register independently.
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

class R13SpectralGateProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'thresholdDb', defaultValue: -40, minValue: -60,  maxValue:    0,  automationRate: 'k-rate' },
      { name: 'reductionDb', defaultValue: -40, minValue: -60,  maxValue:    0,  automationRate: 'k-rate' },
      { name: 'attackMs',    defaultValue:  10, minValue:   1,  maxValue:  100,  automationRate: 'k-rate' },
      { name: 'releaseMs',   defaultValue: 100, minValue:  10,  maxValue: 1000,  automationRate: 'k-rate' },
      { name: 'lowCut',      defaultValue:   0, minValue:   0,  maxValue:    1,  automationRate: 'k-rate' },
      { name: 'highCut',     defaultValue:   1, minValue:   0,  maxValue:    1,  automationRate: 'k-rate' },
      { name: 'tiltDb',      defaultValue:   0, minValue: -12,  maxValue:   12,  automationRate: 'k-rate' },
      { name: 'mix',         defaultValue:   1, minValue:   0,  maxValue:    1,  automationRate: 'k-rate' },
    ];
  }

  constructor(options) {
    super();

    const opts = (options && options.processorOptions) || {};
    this.fftSize  = opts.fftSize || 2048;
    this.hopSize  = this.fftSize >> 2;        // 75% overlap
    this.halfSize = this.fftSize >> 1;

    this.fft        = new R13FFT(this.fftSize);
    this.window     = r13HannWindow(this.fftSize);
    this.windowNorm = 2 / 3;                  // Hann + 75% overlap COLA

    // Per-bin gate envelope state. One slot per [0..halfSize], inclusive of Nyquist.
    // Initialized to 1.0 so a fresh node passes audio cleanly through until the
    // first below-threshold frame arrives.
    this.binEnv = new Float32Array(this.halfSize + 1);
    for (let i = 0; i <= this.halfSize; i++) this.binEnv[i] = 1.0;

    // Input ring + frame timer
    this.inputRing       = new Float32Array(this.fftSize);
    this.inputWritePos   = 0;
    this.samplesUntilFFT = this.fftSize;

    // Output OLA
    this.outputOLA      = new Float32Array(this.fftSize);
    this.outputReadable = 0;

    // FFT scratch
    this.re = new Float32Array(this.fftSize);
    this.im = new Float32Array(this.fftSize);

    // Port-side overrides (snake_case mirrors of the AudioParam k-rates)
    this._override = {
      threshold_db: null,
      reduction_db: null,
      attack_ms:    null,
      release_ms:   null,
      low_cut:      null,
      high_cut:     null,
      tilt_db:      null,
      mix:          null,
    };

    this.port.onmessage = (e) => {
      const m = e.data;
      if (!m || typeof m !== 'object') return;
      switch (m.type) {
        case 'threshold_db': this._override.threshold_db = +m.value; break;
        case 'reduction_db': this._override.reduction_db = +m.value; break;
        case 'attack_ms':    this._override.attack_ms    = +m.value; break;
        case 'release_ms':   this._override.release_ms   = +m.value; break;
        case 'low_cut':      this._override.low_cut      = +m.value; break;
        case 'high_cut':     this._override.high_cut     = +m.value; break;
        case 'tilt_db':      this._override.tilt_db      = +m.value; break;
        case 'mix':          this._override.mix          = +m.value; break;
        case 'reset':
          this.outputOLA.fill(0);
          this.outputReadable = 0;
          for (let i = 0; i <= this.halfSize; i++) this.binEnv[i] = 1.0;
          break;
        default: break;
      }
    };
  }

  process(inputs, outputs, parameters) {
    const input  = inputs[0];
    const output = outputs[0];
    if (!output || output.length < 1) return true;

    const o = this._override;
    const thresholdDb = o.threshold_db != null ? o.threshold_db : parameters.thresholdDb[0];
    const reductionDb = o.reduction_db != null ? o.reduction_db : parameters.reductionDb[0];
    const attackMs    = o.attack_ms    != null ? o.attack_ms    : parameters.attackMs[0];
    const releaseMs   = o.release_ms   != null ? o.release_ms   : parameters.releaseMs[0];
    let   lowCut      = o.low_cut      != null ? o.low_cut      : parameters.lowCut[0];
    let   highCut     = o.high_cut     != null ? o.high_cut     : parameters.highCut[0];
    const tiltDb      = o.tilt_db      != null ? o.tilt_db      : parameters.tiltDb[0];
    const mixK        = o.mix          != null ? o.mix          : parameters.mix[0];

    if (lowCut  < 0) lowCut  = 0; if (lowCut  > 1) lowCut  = 1;
    if (highCut < 0) highCut = 0; if (highCut > 1) highCut = 1;
    if (lowCut > highCut) { const t = lowCut; lowCut = highCut; highCut = t; }

    const halfN     = this.halfSize;
    const lowCutBin  = Math.floor(lowCut  * halfN);
    const highCutBin = Math.ceil (highCut * halfN);

    // Convert attack/release ms → per-frame one-pole coefficient. The gate
    // envelope updates once per analysis hop, so the time constant operates
    // on hop-rate samples. coef = 1 - exp(-hopSamples / (timeMs * sr / 1000))
    const sr = sampleRate; // global in AudioWorkletGlobalScope
    const hopT = this.hopSize / sr; // seconds per hop
    const attTau = Math.max(1e-3, attackMs  / 1000);
    const relTau = Math.max(1e-3, releaseMs / 1000);
    const attackCoef  = 1 - Math.exp(-hopT / attTau);
    const releaseCoef = 1 - Math.exp(-hopT / relTau);

    const reductionGain = Math.pow(10, reductionDb / 20);

    const inL  = (input && input[0]) ? input[0] : null;
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
        this._processFrame(
          thresholdDb, reductionGain, tiltDb,
          lowCutBin, highCutBin,
          attackCoef, releaseCoef,
        );
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

  _processFrame(thresholdDb, reductionGain, tiltDb,
                lowCutBin, highCutBin, attackCoef, releaseCoef) {
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

    // Per-bin gate decision + envelope smoothing on bins 0..halfN.
    // Tilt: linear-in-bin curve. tiltDb is the *total* boost/cut applied at
    // Nyquist relative to DC, so per-bin offset = tiltDb * (k / halfN).
    // Subtracting from threshold makes the gate stricter on highs (negative
    // tilt = highs need to be louder than lows to pass).
    for (let k = 0; k <= halfN; k++) {
      const reK = this.re[k];
      const imK = this.im[k];
      const mag = Math.sqrt(reK * reK + imK * imK);
      const magDb = 20 * Math.log10(Math.max(mag, 1e-12));

      // Bins outside [low_cut_bin, high_cut_bin] are exempt — pull env back to 1.
      const inBand = (k >= lowCutBin && k < highCutBin);

      let targetGain;
      let coef;
      if (!inBand) {
        targetGain = 1.0;
        coef = releaseCoef; // release out-of-band bins back to unity
      } else {
        const tiltAtK = tiltDb * (k / halfN);
        const threshK = thresholdDb + tiltAtK;
        const below = magDb < threshK;
        targetGain = below ? reductionGain : 1.0;
        coef = below ? attackCoef : releaseCoef;
      }

      // One-pole LP envelope smoothing (per-bin "gate envelope").
      const env = this.binEnv[k] + coef * (targetGain - this.binEnv[k]);
      this.binEnv[k] = env;

      this.re[k] = reK * env;
      this.im[k] = imK * env;
    }

    // Hermitian conjugate mirror so IFFT yields a real signal.
    for (let k = 1; k < halfN; k++) {
      this.re[N - k] =  this.re[k];
      this.im[N - k] = -this.im[k];
    }
    // Nyquist already real (im=0 from the forward FFT); leave as scaled.

    this.fft.inverse(this.re, this.im);

    const norm = this.windowNorm;
    for (let i = 0; i < N; i++) {
      this.outputOLA[i] += this.re[i] * this.window[i] * norm;
    }
    this.outputReadable = Math.min(N, this.outputReadable + this.hopSize);
  }
}

registerProcessor('r13-spectral-gate-processor', R13SpectralGateProcessor);
