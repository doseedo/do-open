/**
 * r13-deesser-processor — DeEsser 2 (Logic Pro stock parity)
 *
 * Pipeline (single fused worklet, sample-accurate):
 *
 *   x ─┬─► detection path: bandpass(freq_low..freq_high) → |x| → env follower
 *      │                                                    └► env (CV)
 *      └─► signal   path: peaking biquad(centerHz, Q, gain_db) ─► y
 *                                                       ▲
 *                          gain_db = -range_db · σ((env_dB - threshold_dB)/3)
 *
 * The detection path's bandpass is implemented as the cascade of a 2nd-order
 * highpass at `freq_low` and a 2nd-order lowpass at `freq_high`. The
 * envelope follower is asymmetric one-pole (peak-tracking) with separate
 * attack/release time-constants. Its instantaneous level is converted to dB
 * (clamped at -120 dB to avoid -Inf), and the amount-of-overshoot above
 * threshold is mapped through a soft-knee sigmoid into [0..1]. That fraction
 * scales `range_db` down (negative gain) and is applied to a peaking biquad
 * centred at `(freq_low * freq_high)^0.5` with the configured `q`.
 *
 * The peaking biquad is a standard RBJ cookbook `peakingEQ` so behaviour
 * matches BiquadFilterNode(type='peaking') if a graph elsewhere uses it.
 * Coefficients are recomputed each block (or whenever centre/Q/gain changes
 * by ≥ 0.1 dB or ≥ 0.5 Hz to avoid arithmetic in tight inner loops on every
 * sample for static settings).
 *
 * Modes:
 *   monitor=0 → output the processed signal y (default)
 *   monitor=1 → output the detection bandpass tap (so the user can hear
 *               the sibilant range and tune freq_low / freq_high)
 *
 * AudioWorkletParams (all a-rate where it makes sense — frequency,
 * threshold, range — and k-rate where smoothing already happens internally):
 *   - freq_low      [1500..10000]  Hz   default 4000
 *   - freq_high     [5000..15000]  Hz   default 9000
 *   - threshold_db  [-60..0]       dB   default -28
 *   - range_db      [0..24]        dB   default 12   (cut amount — applied negatively)
 *   - attack_ms     [0.1..10]      ms   default 1.5
 *   - release_ms    [10..200]      ms   default 40
 *   - q             [0.5..10]            default 2.0
 *   - monitor       [0..1]               default 0
 *
 * @author Doseedo R13
 */

class DeEsserProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'freq_low',     defaultValue: 4000,  minValue: 1500, maxValue: 10000, automationRate: 'a-rate' },
      { name: 'freq_high',    defaultValue: 9000,  minValue: 5000, maxValue: 15000, automationRate: 'a-rate' },
      { name: 'threshold_db', defaultValue: -28,   minValue: -60,  maxValue: 0,     automationRate: 'a-rate' },
      { name: 'range_db',     defaultValue: 12,    minValue: 0,    maxValue: 24,    automationRate: 'a-rate' },
      { name: 'attack_ms',    defaultValue: 1.5,   minValue: 0.1,  maxValue: 10,    automationRate: 'k-rate' },
      { name: 'release_ms',   defaultValue: 40,    minValue: 10,   maxValue: 200,   automationRate: 'k-rate' },
      { name: 'q',            defaultValue: 2.0,   minValue: 0.5,  maxValue: 10,    automationRate: 'a-rate' },
      { name: 'monitor',      defaultValue: 0,     minValue: 0,    maxValue: 1,     automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();

    // Detection path filter state (per channel — we keep stereo-summed so 1 set)
    this._hp = { x1: 0, x2: 0, y1: 0, y2: 0, b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 };
    this._lp = { x1: 0, x2: 0, y1: 0, y2: 0, b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 };
    this._lastFlow = -1;
    this._lastFhigh = -1;

    // Peaking biquad state — independent per channel for stereo output
    this._peakL = { x1: 0, x2: 0, y1: 0, y2: 0 };
    this._peakR = { x1: 0, x2: 0, y1: 0, y2: 0 };
    this._peakCoef = { b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 };
    this._lastFc = -1;
    this._lastQ  = -1;
    this._lastGdb = 0;

    // Envelope follower state
    this._env = 0;
    this._coefA = 0.01;
    this._coefR = 0.001;
    this._lastAttackMs = -1;
    this._lastReleaseMs = -1;

    // Periodic main-thread report ~8 ms (matches r1-envelope-follower)
    this._reportSamples = Math.floor(sampleRate * 0.008);
    this._reportCounter = 0;
    this._lastReportedReductionDb = 0;
  }

  // ── Coefficient computation ────────────────────────────────────────────

  _coefFromMs(ms) {
    const samples = Math.max(1, (ms / 1000) * sampleRate);
    return 1 - Math.exp(-1 / samples);
  }

  /**
   * RBJ cookbook 2nd-order Butterworth-ish highpass (Q=0.7071). Sets the
   * coefficients on the supplied state object (b0/b1/b2/a1/a2 — assuming
   * a0 normalised to 1).
   */
  _setHighpass(state, fc) {
    const f = Math.max(20, Math.min(0.49 * sampleRate, fc));
    const w0 = 2 * Math.PI * f / sampleRate;
    const cos_w0 = Math.cos(w0);
    const sin_w0 = Math.sin(w0);
    const Q = Math.SQRT1_2;
    const alpha = sin_w0 / (2 * Q);
    const b0 =  (1 + cos_w0) / 2;
    const b1 = -(1 + cos_w0);
    const b2 =  (1 + cos_w0) / 2;
    const a0 =   1 + alpha;
    const a1 =  -2 * cos_w0;
    const a2 =   1 - alpha;
    state.b0 = b0 / a0;
    state.b1 = b1 / a0;
    state.b2 = b2 / a0;
    state.a1 = a1 / a0;
    state.a2 = a2 / a0;
  }

  _setLowpass(state, fc) {
    const f = Math.max(20, Math.min(0.49 * sampleRate, fc));
    const w0 = 2 * Math.PI * f / sampleRate;
    const cos_w0 = Math.cos(w0);
    const sin_w0 = Math.sin(w0);
    const Q = Math.SQRT1_2;
    const alpha = sin_w0 / (2 * Q);
    const b0 = (1 - cos_w0) / 2;
    const b1 =  1 - cos_w0;
    const b2 = (1 - cos_w0) / 2;
    const a0 =  1 + alpha;
    const a1 = -2 * cos_w0;
    const a2 =  1 - alpha;
    state.b0 = b0 / a0;
    state.b1 = b1 / a0;
    state.b2 = b2 / a0;
    state.a1 = a1 / a0;
    state.a2 = a2 / a0;
  }

  /**
   * RBJ cookbook peakingEQ — same as BiquadFilterNode(type='peaking').
   * gainDb < 0 cuts. fc is centre frequency; Q is filter Q.
   */
  _setPeaking(coef, fc, Q, gainDb) {
    const f = Math.max(20, Math.min(0.49 * sampleRate, fc));
    const A = Math.pow(10, gainDb / 40);
    const w0 = 2 * Math.PI * f / sampleRate;
    const cos_w0 = Math.cos(w0);
    const sin_w0 = Math.sin(w0);
    const QQ = Math.max(0.05, Q);
    const alpha = sin_w0 / (2 * QQ);
    const b0 = 1 + alpha * A;
    const b1 = -2 * cos_w0;
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * cos_w0;
    const a2 = 1 - alpha / A;
    coef.b0 = b0 / a0;
    coef.b1 = b1 / a0;
    coef.b2 = b2 / a0;
    coef.a1 = a1 / a0;
    coef.a2 = a2 / a0;
  }

  _processBiquad(state, coef, x) {
    const y = coef.b0 * x + coef.b1 * state.x1 + coef.b2 * state.x2
              - coef.a1 * state.y1 - coef.a2 * state.y2;
    state.x2 = state.x1;
    state.x1 = x;
    state.y2 = state.y1;
    state.y1 = y;
    return y;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || !output.length) return true;

    const blockSize = output[0].length;
    const inL = input && input[0] ? input[0] : null;
    const inR = input && input[1] ? input[1] : inL;
    const outL = output[0];
    const outR = output.length > 1 ? output[1] : null;

    // Resolve k-rate-cached time constants
    const attackMs  = parameters.attack_ms[0];
    const releaseMs = parameters.release_ms[0];
    if (attackMs !== this._lastAttackMs) {
      this._coefA = this._coefFromMs(attackMs);
      this._lastAttackMs = attackMs;
    }
    if (releaseMs !== this._lastReleaseMs) {
      this._coefR = this._coefFromMs(releaseMs);
      this._lastReleaseMs = releaseMs;
    }
    const monitor = parameters.monitor[0] >= 0.5;

    // a-rate → use [0] when length 1 (parameter stable across block)
    const fLowArr  = parameters.freq_low;
    const fHighArr = parameters.freq_high;
    const thrArr   = parameters.threshold_db;
    const rngArr   = parameters.range_db;
    const qArr     = parameters.q;

    let env = this._env;
    const coefA = this._coefA;
    const coefR = this._coefR;

    // We refresh peaking biquad coefficients up to once per sample but only
    // run the trig math when fc / Q / gain materially changes. Detection
    // bandpass is recomputed only when fLow/fHigh change (per-block check
    // before the inner loop).
    const fLow0  = fLowArr.length === 1  ? fLowArr[0]  : fLowArr[0];
    const fHigh0 = fHighArr.length === 1 ? fHighArr[0] : fHighArr[0];
    if (fLow0 !== this._lastFlow) {
      this._setHighpass(this._hp, fLow0);
      this._lastFlow = fLow0;
    }
    if (fHigh0 !== this._lastFhigh) {
      this._setLowpass(this._lp, fHigh0);
      this._lastFhigh = fHigh0;
    }

    let lastReductionDb = 0;

    for (let i = 0; i < blockSize; i++) {
      const fLow  = fLowArr.length  > 1 ? fLowArr[i]  : fLow0;
      const fHigh = fHighArr.length > 1 ? fHighArr[i] : fHigh0;
      const thr   = thrArr.length   > 1 ? thrArr[i]   : thrArr[0];
      const rng   = rngArr.length   > 1 ? rngArr[i]   : rngArr[0];
      const qVal  = qArr.length     > 1 ? qArr[i]     : qArr[0];

      // Update detection bandpass coefs when params change mid-block (a-rate)
      if (fLowArr.length > 1 && fLow !== this._lastFlow) {
        this._setHighpass(this._hp, fLow);
        this._lastFlow = fLow;
      }
      if (fHighArr.length > 1 && fHigh !== this._lastFhigh) {
        this._setLowpass(this._lp, fHigh);
        this._lastFhigh = fHigh;
      }

      const l = inL ? inL[i] : 0;
      const r = inR ? inR[i] : l;
      const xMono = (l + r) * 0.5;

      // Detection path
      const hp_out = this._processBiquad(this._hp, this._hp, xMono);
      const lp_out = this._processBiquad(this._lp, this._lp, hp_out);
      const detect = lp_out;
      const rect = Math.abs(detect);

      // Asymmetric envelope follower
      const coef = (rect > env) ? coefA : coefR;
      env = (1 - coef) * env + coef * rect;
      // Convert env (linear) to dB; floor at -120 dB
      const envDb = (env > 1e-6) ? 20 * Math.log10(env) : -120;

      // Soft-knee sigmoid mapping of overshoot into [0..1]. Knee width ≈ 6 dB:
      //   below thr by knee/2 → 0
      //   above thr by knee/2 → 1
      const knee = 6;
      const overshoot = envDb - thr;
      let amount;
      if (overshoot <= -knee / 2) amount = 0;
      else if (overshoot >=  knee / 2) amount = 1;
      else amount = (overshoot + knee / 2) / knee; // linear knee — cheap, smooth-enough

      const cutDb = -rng * amount;          // negative gain in dB
      lastReductionDb = cutDb;

      // Peaking biquad — geometric centre between fLow and fHigh
      const fc = Math.sqrt(Math.max(20, fLow * fHigh));
      // Recompute coefs only when fc / Q / gain changes meaningfully
      if (Math.abs(fc - this._lastFc) > 0.5
          || Math.abs(qVal - this._lastQ) > 0.001
          || Math.abs(cutDb - this._lastGdb) > 0.05) {
        this._setPeaking(this._peakCoef, fc, qVal, cutDb);
        this._lastFc = fc;
        this._lastQ = qVal;
        this._lastGdb = cutDb;
      }

      let yL, yR;
      if (monitor) {
        // Monitor mode: bypass dynamic biquad, output the bandpass detection tap
        yL = detect;
        yR = detect;
      } else {
        yL = this._processBiquad(this._peakL, this._peakCoef, l);
        yR = this._processBiquad(this._peakR, this._peakCoef, r);
      }

      outL[i] = yL;
      if (outR) outR[i] = yR;
    }

    this._env = env;

    this._reportCounter += blockSize;
    if (this._reportCounter >= this._reportSamples) {
      this._reportCounter = 0;
      this._lastReportedReductionDb = lastReductionDb;
      this.port.postMessage({ type: 'deesser', env, reductionDb: lastReductionDb });
    }

    return true;
  }
}

registerProcessor('r13-deesser-processor', DeEsserProcessor);
