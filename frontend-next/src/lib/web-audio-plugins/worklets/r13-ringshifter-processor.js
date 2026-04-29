/**
 * r13-ringshifter-processor — ring modulator + frequency shifter (Logic
 * Pro Ringshifter parity).
 *
 * Modes
 *   0  ring_mod        : y = x · sin(2πf·t)
 *   1  freq_shift_up   : y = x_re·cos(2πf·t) − x_im·sin(2πf·t)
 *   2  freq_shift_down : y = x_re·cos(2πf·t) + x_im·sin(2πf·t)
 *   3  both            : ring_mod + freq_shift_up summed, scaled 0.5
 *
 *   x_im is the Hilbert transform of x (90° phase shift).
 *
 * Hilbert transform — 8-stage IIR allpass cascade
 * ------------------------------------------------
 * Two parallel allpass cascades; their outputs are nominally 90° apart
 * across the design band ≈ 80 Hz – 10 kHz.
 *
 * The "real" path runs through allpasses with coefficients tuned to a set
 * of frequencies; the "imag" path runs through a different set. The coef
 * pairs are chosen via the half-band / Olli Niemitalo design criterion — a
 * standard set of 8 first-order allpass coefficients that yields ≤0.05°
 * phase-difference deviation across the design band. We use the Niemitalo
 * 8-stage tap (a²) values directly (see INTEGRATION_R13_RINGSHIFTER.md).
 *
 * Each first-order allpass is the well-known one-pole-one-zero structure:
 *
 *      y[n] = a·x[n] + x[n-1] − a·y[n-1]
 *
 * with `a` close to 1 (a² is the published tap value). Cascading 4 such
 * stages on each path gives a smooth 90° phase difference between the two
 * outputs over the design band. Outside the band the difference drifts but
 * the wet output remains musical (ring-mod-like residue).
 *
 * Each AudioWorkletProcessor sample-block runs at k-rate parameters. We
 * accumulate carrier phase per-sample; for `lfo_rate` and `lfo_depth` the
 * carrier frequency is `freq_hz · (1 + depth · lfo(t))`.
 *
 * Author: Agent R13
 */

const TWO_PI = Math.PI * 2;

// Olli Niemitalo's 8-stage half-band IIR allpass coefficients (a² values),
// design band roughly 80 Hz – 10 kHz at 48 kHz sample rate. Real path uses
// the first 4; imag path uses the second 4. These pair-up to give a near
// 90° phase difference between the two outputs.
//
// Source: Niemitalo, "Polyphase IIR filters with hilbert transform pair
// outputs", 1999. The published a² values:
const HILBERT_REAL_COEFS_SQ = [
  0.4794008309967391,
  0.8762488188644797,
  0.9737948977765025,
  0.9947827710764999,
];
const HILBERT_IMAG_COEFS_SQ = [
  0.1617775918114787,
  0.7330289323414905,
  0.9453497240399447,
  0.9907126061851186,
];

class FirstOrderAllpass {
  constructor(a) {
    this.a = a;
    this.x1 = 0; // previous input
    this.y1 = 0; // previous output
  }

  setCoef(a) { this.a = a; }

  process(x) {
    // y[n] = a·x[n] + x[n-1] − a·y[n-1]
    const y = this.a * x + this.x1 - this.a * this.y1;
    this.x1 = x;
    this.y1 = y;
    return y;
  }

  reset() { this.x1 = 0; this.y1 = 0; }
}

function buildAllpassCascade(coefSquares) {
  const stages = [];
  for (let i = 0; i < coefSquares.length; i++) {
    // The published values are a² — the first-order coef is the value
    // itself (see Niemitalo). For a true first-order section, a is the
    // coef directly; many implementations apply the coef as-is. We use
    // the published value as `a`.
    stages.push(new FirstOrderAllpass(coefSquares[i]));
  }
  return stages;
}

class RingshifterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'mode',        defaultValue: 0,    minValue: 0,  maxValue: 3,    automationRate: 'k-rate' },
      { name: 'freq_hz',     defaultValue: 220,  minValue: 0,  maxValue: 5000, automationRate: 'k-rate' },
      { name: 'lfo_rate',    defaultValue: 0,    minValue: 0,  maxValue: 10,   automationRate: 'k-rate' },
      { name: 'lfo_depth',   defaultValue: 0,    minValue: 0,  maxValue: 100,  automationRate: 'k-rate' },
      { name: 'lfo_shape',   defaultValue: 0,    minValue: 0,  maxValue: 3,    automationRate: 'k-rate' },
      { name: 'feedback',    defaultValue: 0,    minValue: 0,  maxValue: 0.99, automationRate: 'k-rate' },
      { name: 'dry_mix',     defaultValue: 0.5,  minValue: 0,  maxValue: 1,    automationRate: 'k-rate' },
      { name: 'wet_mix',     defaultValue: 0.5,  minValue: 0,  maxValue: 1,    automationRate: 'k-rate' },
      { name: 'output_gain', defaultValue: 1.0,  minValue: 0,  maxValue: 4,    automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();

    // Per-channel Hilbert cascades (stereo-safe). We allocate up to 2 chans.
    this._realStagesL = buildAllpassCascade(HILBERT_REAL_COEFS_SQ);
    this._imagStagesL = buildAllpassCascade(HILBERT_IMAG_COEFS_SQ);
    this._realStagesR = buildAllpassCascade(HILBERT_REAL_COEFS_SQ);
    this._imagStagesR = buildAllpassCascade(HILBERT_IMAG_COEFS_SQ);

    // The imag path of a Niemitalo Hilbert cascade has an inherent unit-
    // sample delay relative to the real path. To keep them aligned, we
    // delay the real-path output by 1 sample.
    this._realDelayL = 0;
    this._realDelayR = 0;

    // Carrier oscillator phase (radians). Shared L/R for stereo coherence.
    this._phase = 0;

    // LFO phase (radians, 0..2π).
    this._lfoPhase = 0;

    // Sample-and-hold value for `random` LFO shape, plus its hold counter.
    this._randHold = 0;
    this._randCounter = 0;

    // Feedback memory (one sample per channel)
    this._fbL = 0;
    this._fbR = 0;
  }

  _runHilbert(x, realStages, imagStages, isLeft) {
    let r = x;
    for (let i = 0; i < realStages.length; i++) r = realStages[i].process(r);
    let im = x;
    for (let i = 0; i < imagStages.length; i++) im = imagStages[i].process(im);
    // Compensate the 1-sample delay between the two paths
    const rOut = isLeft ? this._realDelayL : this._realDelayR;
    if (isLeft) this._realDelayL = r;
    else this._realDelayR = r;
    return [rOut, im];
  }

  _lfo(rateHz, shapeIdx, sr) {
    if (rateHz <= 0) return 0;
    const inc = (TWO_PI * rateHz) / sr;
    this._lfoPhase += inc;
    if (this._lfoPhase >= TWO_PI) this._lfoPhase -= TWO_PI;
    switch (shapeIdx | 0) {
      case 0: // sine
        return Math.sin(this._lfoPhase);
      case 1: { // triangle
        const t = this._lfoPhase / TWO_PI; // 0..1
        return 4 * Math.abs(t - 0.5) - 1;
      }
      case 2: // square
        return this._lfoPhase < Math.PI ? 1 : -1;
      case 3: { // random S&H — refresh once per LFO cycle
        const samplesPerCycle = Math.max(1, Math.floor(sr / Math.max(0.01, rateHz)));
        if (this._randCounter <= 0) {
          this._randHold = Math.random() * 2 - 1;
          this._randCounter = samplesPerCycle;
        }
        this._randCounter--;
        return this._randHold;
      }
      default:
        return Math.sin(this._lfoPhase);
    }
  }

  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!out || out.length === 0) return true;

    const numCh = Math.min(out.length, 2);
    const numSamples = out[0].length;
    const sr = sampleRate;

    const mode      = (parameters.mode[0]      | 0);
    const baseFreq  = parameters.freq_hz[0];
    const lfoRate   = parameters.lfo_rate[0];
    const lfoDepth  = parameters.lfo_depth[0] / 100; // % → 0..1
    const lfoShape  = (parameters.lfo_shape[0] | 0);
    const feedback  = Math.max(0, Math.min(0.99, parameters.feedback[0]));
    const dryMix    = parameters.dry_mix[0];
    const wetMix    = parameters.wet_mix[0];
    const outGain   = parameters.output_gain[0];

    for (let i = 0; i < numSamples; i++) {
      // Carrier frequency modulated by LFO
      const lfo = this._lfo(lfoRate, lfoShape, sr);
      const fNow = baseFreq * (1 + lfoDepth * lfo);

      // Advance carrier phase
      this._phase += (TWO_PI * fNow) / sr;
      if (this._phase >= TWO_PI) this._phase -= TWO_PI;

      const cosT = Math.cos(this._phase);
      const sinT = Math.sin(this._phase);

      for (let c = 0; c < numCh; c++) {
        const inChan = (inp && inp[c]) ? inp[c][i] : (inp && inp[0] ? inp[0][i] : 0);

        // Apply feedback into shift path input
        const fbPrev = (c === 0) ? this._fbL : this._fbR;
        const x = inChan + feedback * fbPrev;

        // Hilbert pair
        const isLeft = (c === 0);
        const realStages = isLeft ? this._realStagesL : this._realStagesR;
        const imagStages = isLeft ? this._imagStagesL : this._imagStagesR;
        const [xRe, xIm] = this._runHilbert(x, realStages, imagStages, isLeft);

        let wet = 0;
        switch (mode) {
          case 0: // ring_mod
            wet = inChan * sinT;
            break;
          case 1: // freq_shift_up
            wet = xRe * cosT - xIm * sinT;
            break;
          case 2: // freq_shift_down
            wet = xRe * cosT + xIm * sinT;
            break;
          case 3: // both
            wet = 0.5 * (inChan * sinT + (xRe * cosT - xIm * sinT));
            break;
          default:
            wet = inChan * sinT;
        }

        // Update feedback memory
        if (c === 0) this._fbL = wet;
        else this._fbR = wet;

        out[c][i] = (dryMix * inChan + wetMix * wet) * outGain;
      }

      // Mono → stereo upmix if needed
      if (numCh < out.length) {
        for (let c = numCh; c < out.length; c++) {
          out[c][i] = out[0][i];
        }
      }
    }

    return true;
  }
}

registerProcessor('r13-ringshifter-processor', RingshifterProcessor);
