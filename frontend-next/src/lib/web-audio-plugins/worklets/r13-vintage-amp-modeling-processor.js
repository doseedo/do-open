/**
 * r13-vintage-amp-modeling-processor — optional fused worklet for the
 * `vintage_amp_modeling` composite node.
 *
 * The composite builder (src/audio/builders/r13_vintage_amp_modeling.js)
 * already produces working audio out of native Web Audio nodes + R2/R3
 * worklets. This worklet exists as an *optional* fused path: if the host
 * registers it via `audioWorklet.addModule(...)`, the builder can route
 * audio through this single processor instead of the multi-node chain to
 * cut node-graph overhead for low-latency live use.
 *
 * The fused chain implements the same topology end-to-end:
 *
 *   x → preamp_shaper(harmonic_sig + drive)
 *     → tone_stack(bass, mid, treble)
 *     → phase_inverter(tanh)
 *     → power_amp_shaper(power_tube_type + bias_drift)
 *     → output_transformer(saturation + LP)
 *     → psu_sag(env-follow * sag_amount)
 *     → output
 *
 * Cabinet IR + mic preEQ live OUTSIDE the worklet so the same convolver
 * buffer + biquad cascade can be reused / swapped without reconstructing
 * this processor.
 *
 * Per-sample cost: ~2 LUT lookups (waveshapers) + 3 biquads + 1-pole LP +
 * envelope follower = ~30 mults/sample. Trivial for an AudioWorklet thread.
 *
 * Author: Doseedo R13-VintageAmp
 */

// Mirror of AMP_MODEL_PRESETS in the builder (kept here so the worklet is
// fully self-contained — no importScripts in this file).
const AMP_MODEL_PRESETS = {
  tweed_5e3:       { ts: '6V6', sig: [0.85, 0.45, 0.25, 0.10], lp: 4500, sat: 0.55, bias: 0.30, nfb: 0.15, voicing: [90, 700, 4500],   pres: 3500, mk: 1.10 },
  tweed_5f6:       { ts: '6L6', sig: [0.75, 0.55, 0.30, 0.15], lp: 5500, sat: 0.45, bias: 0.05, nfb: 0.30, voicing: [80, 500, 5000],   pres: 3000, mk: 1.00 },
  vox_ac30:        { ts: 'EL84',sig: [0.70, 0.65, 0.40, 0.25], lp: 6500, sat: 0.60, bias: 0.45, nfb: 0.05, voicing: [100, 1000, 6500], pres: 4500, mk: 1.20 },
  marshall_plexi:  { ts: 'EL34',sig: [0.55, 0.75, 0.45, 0.30], lp: 6000, sat: 0.50, bias: 0.10, nfb: 0.55, voicing: [110, 650, 5500],  pres: 3000, mk: 0.95 },
  marshall_jcm800: { ts: 'EL34',sig: [0.45, 0.85, 0.55, 0.35], lp: 5500, sat: 0.55, bias: 0.15, nfb: 0.65, voicing: [110, 600, 6000],  pres: 3200, mk: 0.90 },
  hiwatt:          { ts: 'EL34',sig: [0.65, 0.50, 0.20, 0.10], lp: 8000, sat: 0.30, bias: 0.00, nfb: 0.75, voicing: [90, 800, 5500],   pres: 4000, mk: 0.95 },
  orange_or120:    { ts: 'EL34',sig: [0.55, 0.70, 0.40, 0.20], lp: 4500, sat: 0.55, bias: 0.20, nfb: 0.45, voicing: [100, 900, 4800],  pres: 3200, mk: 1.00 },
  silvertone:      { ts: '6L6', sig: [0.80, 0.50, 0.25, 0.15], lp: 3800, sat: 0.65, bias: 0.35, nfb: 0.20, voicing: [95, 750, 4000],   pres: 2800, mk: 1.15 },
};

const MODEL_KEYS = Object.keys(AMP_MODEL_PRESETS);

// ── Curve generators (run once per model switch on the worklet thread) ───
function makeHarmonicCurve(sig, asym, drive, N = 4096) {
  const [h2, h3, h4, h5] = sig;
  const c = new Float32Array(N);
  const baseAt = (x) => {
    const t2 = 2 * x * x - 1;
    const t3 = 4 * x * x * x - 3 * x;
    const t4 = 8 * x * x * x * x - 8 * x * x + 1;
    const t5 = 16 * x * x * x * x * x - 20 * x * x * x + 5 * x;
    return Math.tanh(x * drive)
         + 0.15 * h2 * t2
         + 0.20 * h3 * t3
         + 0.10 * h4 * t4
         + 0.12 * h5 * t5
         - 0.15 * h2;
  };
  const yOff = baseAt(-asym);
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    let y = baseAt(x + asym) - yOff;
    if (y > 1.0) y = 1.0;
    if (y < -1.0) y = -1.0;
    c[i] = y;
  }
  return c;
}

function makePowerTubeCurve(tubeType, sig, biasDrift, N = 4096) {
  const [h2, h3, h4, h5] = sig;
  const k = ({ EL84: 1.6, EL34: 1.4, '6V6': 1.2, '6L6': 1.1 })[tubeType] || 1.3;
  const c = new Float32Array(N);
  const xOff = biasDrift * 0.4;
  const baseAt = (x) => {
    const xs = x * k;
    const t2 = 2 * xs * xs - 1;
    const t3 = 4 * xs * xs * xs - 3 * xs;
    return Math.tanh(xs)
         + 0.08 * h2 * t2
         + 0.15 * h3 * t3
         + 0.05 * (h4 + h5) * Math.sin(xs * 2);
  };
  const yOff = baseAt(-xOff);
  for (let i = 0; i < N; i++) {
    const x = (i / (N - 1)) * 2 - 1;
    let y = baseAt(x + xOff) - yOff;
    if (y > 1.0) y = 1.0;
    if (y < -1.0) y = -1.0;
    c[i] = y;
  }
  return c;
}

// LUT lookup with linear interpolation, x in [-1, +1]
function lookup(curve, x) {
  if (x <= -1) return curve[0];
  if (x >= 1)  return curve[curve.length - 1];
  const N = curve.length - 1;
  const f = (x + 1) * 0.5 * N;
  const i = f | 0;
  const t = f - i;
  return curve[i] * (1 - t) + curve[i + 1] * t;
}

// ── Processor ─────────────────────────────────────────────────────────────

class VintageAmpModelingProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'gain',     defaultValue: 5.0, minValue: 0,  maxValue: 10 },
      { name: 'bass',     defaultValue: 0.5, minValue: 0,  maxValue: 1 },
      { name: 'mid',      defaultValue: 0.5, minValue: 0,  maxValue: 1 },
      { name: 'treble',   defaultValue: 0.5, minValue: 0,  maxValue: 1 },
      { name: 'master',   defaultValue: 0.5, minValue: 0,  maxValue: 1 },
      { name: 'bias',     defaultValue: 0.5, minValue: 0,  maxValue: 1 },
      { name: 'nfb',      defaultValue: 0.5, minValue: 0,  maxValue: 1 },
      { name: 'output_level', defaultValue: 1.0, minValue: 0, maxValue: 4 },
      // amp_model is k-rate as an integer index 0..7
      { name: 'amp_model_idx', defaultValue: 0, minValue: 0, maxValue: 7 },
    ];
  }

  constructor() {
    super();
    this._modelKey = MODEL_KEYS[0];
    this._preset = AMP_MODEL_PRESETS[this._modelKey];
    this._preampCurve = makeHarmonicCurve(this._preset.sig, 0, 2.0);
    this._powerCurve  = makePowerTubeCurve(this._preset.ts, this._preset.sig, this._preset.bias);
    // Tone-stack biquad state (3 bands, stereo) — Direct Form I transposed
    this._biquads = [];
    for (let ch = 0; ch < 2; ch++) {
      this._biquads[ch] = [
        { z1: 0, z2: 0 }, { z1: 0, z2: 0 }, { z1: 0, z2: 0 },
        { z1: 0, z2: 0 }, // transformer LP
      ];
    }
    // PSU sag envelope state (shared across channels)
    this._env = 0;
    this._lastModelIdx = -1;
    // Allow host to set the model by string via port message
    this.port.onmessage = (e) => {
      if (!e || !e.data) return;
      if (e.data.type === 'setAmpModel' && AMP_MODEL_PRESETS[e.data.value]) {
        this._setModel(e.data.value);
      }
    };
  }

  _setModel(modelKey) {
    this._modelKey = modelKey;
    this._preset = AMP_MODEL_PRESETS[modelKey];
    this._preampCurve = makeHarmonicCurve(this._preset.sig, 0, 2.0);
    this._powerCurve  = makePowerTubeCurve(this._preset.ts, this._preset.sig, this._preset.bias);
  }

  // RBJ peaking biquad — stateless coefficient generator. `which` ∈ 0..2
  // (low shelf / mid peak / high shelf); chosen freq pulled from preset.voicing.
  _shelfCoeffs(which, gainDb, sr) {
    const f0 = this._preset.voicing[which];
    const A = Math.pow(10, gainDb / 40);
    const w0 = 2 * Math.PI * f0 / sr;
    const cosw = Math.cos(w0);
    const sinw = Math.sin(w0);
    const Q = which === 1 ? 0.7 : 0.7071;
    const alpha = sinw / (2 * Q);
    let b0, b1, b2, a0, a1, a2;
    if (which === 1) {
      // peaking
      b0 = 1 + alpha * A;
      b1 = -2 * cosw;
      b2 = 1 - alpha * A;
      a0 = 1 + alpha / A;
      a1 = -2 * cosw;
      a2 = 1 - alpha / A;
    } else if (which === 0) {
      // low shelf
      const sqrtA2alpha = 2 * Math.sqrt(A) * alpha;
      b0 = A * ((A + 1) - (A - 1) * cosw + sqrtA2alpha);
      b1 = 2 * A * ((A - 1) - (A + 1) * cosw);
      b2 = A * ((A + 1) - (A - 1) * cosw - sqrtA2alpha);
      a0 = (A + 1) + (A - 1) * cosw + sqrtA2alpha;
      a1 = -2 * ((A - 1) + (A + 1) * cosw);
      a2 = (A + 1) + (A - 1) * cosw - sqrtA2alpha;
    } else {
      // high shelf
      const sqrtA2alpha = 2 * Math.sqrt(A) * alpha;
      b0 = A * ((A + 1) + (A - 1) * cosw + sqrtA2alpha);
      b1 = -2 * A * ((A - 1) + (A + 1) * cosw);
      b2 = A * ((A + 1) + (A - 1) * cosw - sqrtA2alpha);
      a0 = (A + 1) - (A - 1) * cosw + sqrtA2alpha;
      a1 = 2 * ((A - 1) - (A + 1) * cosw);
      a2 = (A + 1) - (A - 1) * cosw - sqrtA2alpha;
    }
    return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
  }

  _lpCoeffs(fc, sr) {
    const w0 = 2 * Math.PI * Math.min(fc, 0.45 * sr) / sr;
    const cosw = Math.cos(w0);
    const sinw = Math.sin(w0);
    const Q = 0.7071;
    const alpha = sinw / (2 * Q);
    const b0 = (1 - cosw) / 2;
    const b1 = 1 - cosw;
    const b2 = (1 - cosw) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * cosw;
    const a2 = 1 - alpha;
    return [b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0];
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0) return true;

    // Read k-rate params (use [0] if a-rate doesn't matter for this composite)
    const gain   = parameters.gain[0]   ?? 5.0;
    const bassDb = ((parameters.bass[0]   ?? 0.5) - 0.5) * 24;
    const midDb  = ((parameters.mid[0]    ?? 0.5) - 0.5) * 18;
    const trebDb = ((parameters.treble[0] ?? 0.5) - 0.5) * 24;
    const master = parameters.master[0] ?? 0.5;
    const bias   = parameters.bias[0]   ?? 0.5;
    const nfb    = parameters.nfb[0]    ?? 0.5;
    const outLvl = parameters.output_level[0] ?? 1.0;
    const modelIdx = Math.round(parameters.amp_model_idx[0] ?? 0);

    if (modelIdx !== this._lastModelIdx
        && modelIdx >= 0 && modelIdx < MODEL_KEYS.length) {
      this._setModel(MODEL_KEYS[modelIdx]);
      this._lastModelIdx = modelIdx;
    }

    const sr = sampleRate;
    const lpFc = this._preset.lp * (1 + 0.5 * nfb);
    const [bb0, bb1, bb2, ba1, ba2] = this._shelfCoeffs(0, bassDb, sr);
    const [mb0, mb1, mb2, ma1, ma2] = this._shelfCoeffs(1, midDb,  sr);
    const [tb0, tb1, tb2, ta1, ta2] = this._shelfCoeffs(2, trebDb, sr);
    const [lb0, lb1, lb2, la1, la2] = this._lpCoeffs(lpFc, sr);

    const sagAmt = 0.3 + 0.2 * (this._preset.bias - 0.5 + bias - 0.5);
    const releaseCoeff = Math.exp(-1 / (0.05 * sr));
    const attackCoeff  = Math.exp(-1 / (0.005 * sr));
    const drive = 0.1 + gain * 0.3;          // map 0..10 → 0.1..3.1
    const masterDrive = 0.5 + master * 1.5;  // map 0..1  → 0.5..2.0

    for (let ch = 0; ch < input.length; ch++) {
      const inCh  = input[ch];
      const outCh = output[ch];
      const bq = this._biquads[ch] || this._biquads[0];
      if (!inCh) continue;
      const N = inCh.length;
      for (let i = 0; i < N; i++) {
        let x = inCh[i];

        // 1. Preamp drive + waveshaper
        x = lookup(this._preampCurve, Math.max(-1, Math.min(1, x * drive)));

        // 2. Tone stack (3 cascaded biquads)
        // bass shelf
        let y = bb0 * x + bb1 * 0 + bb2 * 0 - ba1 * bq[0].z1 - ba2 * bq[0].z2;
        // shift state via Direct Form I
        const xb = x;
        x = bb0 * x + bq[0].z1;
        bq[0].z1 = bb1 * xb - ba1 * x + bq[0].z2;
        bq[0].z2 = bb2 * xb - ba2 * x;

        // mid peak
        const xm = x;
        x = mb0 * x + bq[1].z1;
        bq[1].z1 = mb1 * xm - ma1 * x + bq[1].z2;
        bq[1].z2 = mb2 * xm - ma2 * x;

        // treble shelf
        const xt = x;
        x = tb0 * x + bq[2].z1;
        bq[2].z1 = tb1 * xt - ta1 * x + bq[2].z2;
        bq[2].z2 = tb2 * xt - ta2 * x;

        // 3. Phase inverter (gentle tanh)
        x = Math.tanh(x * 1.2);

        // 4. Power amp drive + LUT
        x = lookup(this._powerCurve,
          Math.max(-1, Math.min(1, x * masterDrive)));

        // 5. Output transformer LP
        const xl = x;
        x = lb0 * x + bq[3].z1;
        bq[3].z1 = lb1 * xl - la1 * x + bq[3].z2;
        bq[3].z2 = lb2 * xl - la2 * x;

        // 6. PSU sag — envelope-based gain reduction (shared env across channels)
        const lev = Math.abs(x);
        if (lev > this._env) this._env = attackCoeff * this._env + (1 - attackCoeff) * lev;
        else                 this._env = releaseCoeff * this._env + (1 - releaseCoeff) * lev;
        let gr = 1 - sagAmt * this._env;
        if (gr < 0.5) gr = 0.5;
        if (gr > 1)   gr = 1;
        x *= gr;

        // 7. Output gain + makeup
        outCh[i] = x * outLvl * this._preset.mk;
      }
    }
    return true;
  }
}

registerProcessor('r13-vintage-amp-modeling-processor', VintageAmpModelingProcessor);
