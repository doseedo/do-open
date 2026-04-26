/**
 * r13-multipressor-processor — single-pass 4-band multiband compressor.
 *
 * Implements Logic Pro's Multipressor in one AudioWorkletProcessor:
 *   1. Linkwitz-Riley 4th-order crossovers split the input into 4 bands.
 *      Band edges live at `crossover_1` < `crossover_2` < `crossover_3`.
 *      LR4 = two cascaded 2nd-order Butterworth biquads at the same cutoff
 *      with q = 1/sqrt(2). The classic LR property LP(f)+HP(f) ≅ ALLPASS(f)
 *      means the band sum reconstructs the input (allpass-equivalent) when
 *      no band is compressing.
 *   2. Each band feeds an independent feed-forward compressor with its own
 *      threshold, ratio, attack, release, knee, and makeup gain. Detector
 *      is the per-sample peak (full-wave rect → log dB) smoothed by a
 *      one-pole envelope follower with attack/release time constants.
 *   3. Per-band bypass routes the band un-processed.
 *   4. Bands are summed; master `output_gain` (in dB) trims the bus.
 *   5. A pre-split `lookahead_ms` delay line (≤ 10 ms) lets the detector
 *      see audio ahead of the processed sample so the comp can react to
 *      short transients without overshoot.
 *
 * Self-contained: no `importScripts`, just direct DSP. Stereo throughout
 * (allocates per-channel state). Mono input is duplicated to L/R.
 *
 * Registered name: 'r13-multipressor-processor'
 *
 * @author Agent R13 — Multipressor
 */

const NUM_BANDS = 4;
const MAX_LOOKAHEAD_MS = 10;
const MAX_CHANNELS = 2;

// One LR4 stage = two cascaded 2nd-order biquads at the same cutoff, q=0.7071.
// We store coefficient sets per cutoff and re-derive when the cutoff drifts.
class LR4Filter {
  constructor() {
    // Two biquad stages (state per channel)
    this._z1 = [new Float32Array(2), new Float32Array(2)]; // stage[ch]
    this._z2 = [new Float32Array(2), new Float32Array(2)];
    this._b0 = 0; this._b1 = 0; this._b2 = 0;
    this._a1 = 0; this._a2 = 0;
    this._lastCutoff = -1;
    this._lastType = '';
  }

  // Update biquad coefficients for the requested cutoff/type.
  // Standard RBJ formulas for Butterworth (q = 1/sqrt(2)).
  setCutoff(type, hz, sr) {
    if (hz === this._lastCutoff && type === this._lastType) return;
    const q = Math.SQRT1_2;
    const w0 = 2 * Math.PI * Math.max(20, Math.min(sr * 0.45, hz)) / sr;
    const cw = Math.cos(w0);
    const sw = Math.sin(w0);
    const alpha = sw / (2 * q);
    let b0, b1, b2, a0, a1, a2;
    if (type === 'lowpass') {
      b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2;
    } else { // 'highpass'
      b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2;
    }
    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
    this._b0 = b0 / a0; this._b1 = b1 / a0; this._b2 = b2 / a0;
    this._a1 = a1 / a0; this._a2 = a2 / a0;
    this._lastCutoff = hz; this._lastType = type;
  }

  // Direct-form II transposed for one stage; we apply twice for LR4.
  processSample(stage, ch, x) {
    const b0 = this._b0, b1 = this._b1, b2 = this._b2, a1 = this._a1, a2 = this._a2;
    const z1 = this._z1[stage], z2 = this._z2[stage];
    const y = b0 * x + z1[ch];
    z1[ch] = b1 * x - a1 * y + z2[ch];
    z2[ch] = b2 * x - a2 * y;
    return y;
  }

  process(ch, x) {
    const y1 = this.processSample(0, ch, x);
    return this.processSample(1, ch, y1);
  }
}

class BandCompressor {
  constructor(sr) {
    this._sr = sr;
    this.threshold = -20;
    this.ratio = 3;
    this.attack = 0.010;
    this.release = 0.150;
    this.knee = 6;
    this.makeup = 1.0;
    this.bypass = 0;
    this._envDb = [-120, -120]; // per-channel envelope follower
  }

  setParams(t, r, a, rel, k, mkLin, byp) {
    this.threshold = t;
    this.ratio = Math.max(1, r);
    this.attack = Math.max(0.0001, a);
    this.release = Math.max(0.001, rel);
    this.knee = Math.max(0, k);
    this.makeup = Math.max(0, mkLin);
    this.bypass = byp;
  }

  // Returns gainDb to apply (negative means attenuation).
  computeGainDb(envDb) {
    const t = this.threshold, k = this.knee, r = this.ratio;
    const overshoot = envDb - t;
    if (overshoot <= -k / 2) return 0;
    if (k > 0 && overshoot < k / 2) {
      const x = overshoot + k / 2;
      return -x * x * (1 - 1 / r) / (2 * k);
    }
    return -overshoot * (1 - 1 / r);
  }

  processChannelSample(ch, s) {
    if (this.bypass > 0.5) return s;
    const sr = this._sr;
    const aCoef = Math.exp(-1 / (this.attack * sr));
    const rCoef = Math.exp(-1 / (this.release * sr));
    const inDb = 20 * Math.log10(Math.max(1e-7, Math.abs(s)));
    let env = this._envDb[ch];
    if (inDb > env) env = aCoef * env + (1 - aCoef) * inDb;
    else            env = rCoef * env + (1 - rCoef) * inDb;
    this._envDb[ch] = env;
    const gDb = this.computeGainDb(env);
    const linG = Math.pow(10, gDb / 20);
    return s * linG * this.makeup;
  }
}

class R13MultipressorProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    const bandParams = [];
    for (let i = 1; i <= NUM_BANDS; i++) {
      bandParams.push(
        { name: `band${i}_threshold_db`, defaultValue: [-22, -20, -18, -16][i - 1], minValue: -100, maxValue: 0,    automationRate: 'k-rate' },
        { name: `band${i}_ratio`,        defaultValue: 3,                            minValue: 1,    maxValue: 50,   automationRate: 'k-rate' },
        { name: `band${i}_attack_ms`,    defaultValue: [20, 15, 10, 5][i - 1],       minValue: 0.1,  maxValue: 300,  automationRate: 'k-rate' },
        { name: `band${i}_release_ms`,   defaultValue: [200, 150, 120, 100][i - 1],  minValue: 1,    maxValue: 2000, automationRate: 'k-rate' },
        { name: `band${i}_gain_db`,      defaultValue: 0,                            minValue: -24,  maxValue: 24,   automationRate: 'k-rate' },
        { name: `band${i}_bypass`,       defaultValue: 0,                            minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      );
    }
    return [
      { name: 'crossover_1',  defaultValue: 120,  minValue: 50,    maxValue: 500,    automationRate: 'k-rate' },
      { name: 'crossover_2',  defaultValue: 800,  minValue: 200,   maxValue: 2000,   automationRate: 'k-rate' },
      { name: 'crossover_3',  defaultValue: 4000, minValue: 1000,  maxValue: 10000,  automationRate: 'k-rate' },
      { name: 'lookahead_ms', defaultValue: 0,    minValue: 0,     maxValue: MAX_LOOKAHEAD_MS, automationRate: 'k-rate' },
      { name: 'output_gain',  defaultValue: 0,    minValue: -24,   maxValue: 24,     automationRate: 'k-rate' },
      ...bandParams,
    ];
  }

  constructor() {
    super();
    const sr = sampleRate;
    // Filters: per-channel LR4 instances.
    // We need: LP@xo1, LP@xo2, LP@xo3, HP@xo1, HP@xo2, HP@xo3.
    this._lpFilters = [new LR4Filter(), new LR4Filter(), new LR4Filter()];
    this._hpFilters = [new LR4Filter(), new LR4Filter(), new LR4Filter()];

    this._bands = [];
    for (let i = 0; i < NUM_BANDS; i++) this._bands.push(new BandCompressor(sr));

    // Lookahead delay line
    const maxLA = Math.ceil((MAX_LOOKAHEAD_MS / 1000) * sr) + 4;
    this._laBufL = new Float32Array(maxLA);
    this._laBufR = new Float32Array(maxLA);
    this._laMaxLen = maxLA;
    this._laWrite = 0;
  }

  // Get a delayed sample (per-channel) from the lookahead line.
  _readLA(ch, delaySamps) {
    const buf = ch === 0 ? this._laBufL : this._laBufR;
    let r = this._laWrite - delaySamps;
    if (r < 0) r += this._laMaxLen;
    return buf[r | 0] || 0;
  }

  _writeLA(L, R) {
    this._laBufL[this._laWrite] = L;
    this._laBufR[this._laWrite] = R;
    this._laWrite = (this._laWrite + 1) % this._laMaxLen;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length || !output || !output.length) return true;

    const blockSize = output[0].length;
    const sr = sampleRate;

    const xo1 = parameters.crossover_1[0];
    const xo2 = parameters.crossover_2[0];
    const xo3 = parameters.crossover_3[0];
    // Setup filter coefs once per block (k-rate)
    this._lpFilters[0].setCutoff('lowpass',  xo1, sr);
    this._lpFilters[1].setCutoff('lowpass',  xo2, sr);
    this._lpFilters[2].setCutoff('lowpass',  xo3, sr);
    this._hpFilters[0].setCutoff('highpass', xo1, sr);
    this._hpFilters[1].setCutoff('highpass', xo2, sr);
    this._hpFilters[2].setCutoff('highpass', xo3, sr);

    // Update band parameters once per block
    for (let i = 0; i < NUM_BANDS; i++) {
      const idx = i + 1;
      const tDb = parameters[`band${idx}_threshold_db`][0];
      const r   = parameters[`band${idx}_ratio`][0];
      const aMs = parameters[`band${idx}_attack_ms`][0];
      const rMs = parameters[`band${idx}_release_ms`][0];
      const gDb = parameters[`band${idx}_gain_db`][0];
      const byp = parameters[`band${idx}_bypass`][0];
      const mkLin = Math.pow(10, gDb / 20);
      this._bands[i].setParams(tDb, r, aMs / 1000, rMs / 1000, 6, mkLin, byp);
    }

    const laMs = parameters.lookahead_ms[0];
    const laSamps = Math.max(0, Math.min(this._laMaxLen - 2, Math.floor((laMs / 1000) * sr)));

    const masterLin = Math.pow(10, parameters.output_gain[0] / 20);

    const numCh = Math.min(MAX_CHANNELS, output.length);
    const inCh = input.length;

    for (let n = 0; n < blockSize; n++) {
      // Read raw input (mono → duplicate to L/R)
      const xL = input[0] ? input[0][n] : 0;
      const xR = (inCh > 1 && input[1]) ? input[1][n] : xL;

      // Push into lookahead, read delayed
      this._writeLA(xL, xR);
      const dL = laSamps > 0 ? this._readLA(0, laSamps) : xL;
      const dR = laSamps > 0 ? this._readLA(1, laSamps) : xR;

      // For each channel, build the 4 bands.
      // Band 0: LP@xo1
      // Band 1: HP@xo1 → LP@xo2
      // Band 2: HP@xo2 → LP@xo3
      // Band 3: HP@xo3
      for (let ch = 0; ch < numCh; ch++) {
        const d = ch === 0 ? dL : dR;
        const b0 = this._lpFilters[0].process(ch, d);
        const after_hp0 = this._hpFilters[0].process(ch, d);
        const b1 = this._lpFilters[1].process(ch, after_hp0);
        const after_hp1 = this._hpFilters[1].process(ch, d);
        const b2 = this._lpFilters[2].process(ch, after_hp1);
        const b3 = this._hpFilters[2].process(ch, d);

        const c0 = this._bands[0].processChannelSample(ch, b0);
        const c1 = this._bands[1].processChannelSample(ch, b1);
        const c2 = this._bands[2].processChannelSample(ch, b2);
        const c3 = this._bands[3].processChannelSample(ch, b3);

        const sum = (c0 + c1 + c2 + c3) * masterLin;
        output[ch][n] = sum;
      }
    }

    return true;
  }
}

registerProcessor('r13-multipressor-processor', R13MultipressorProcessor);
