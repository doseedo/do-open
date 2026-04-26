/**
 * R13 BassAmpDesignerProcessor — single-block-DSP fallback for the
 * bass_amp_designer composite builder.
 *
 * The primary `bass_amp_designer` builder composes native AudioNodes
 * (waveshapers, biquads, dynamics-compressor, convolver, gains) plus the
 * existing R2 `wdf_tone_stack` worklet, so a bespoke worklet is NOT required
 * for the audio path to work. This processor exists for two reasons:
 *
 *   1. A diagnostic / "all-in-one" mode where the composite is collapsed into
 *      a single sample-loop — useful for offline benches and CI null-diff.
 *   2. A self-contained future path if/when we move the whole bass-amp signal
 *      chain into a single worklet for tighter latency / lower per-sample
 *      AudioNode overhead in heavy sessions.
 *
 * It implements:
 *   - Pre-gain
 *   - Tube + solid-state shaping in parallel, blended via tube_blend
 *   - 4-band tone stack (bass shelf / low-mid peak / hi-mid peak / treble shelf)
 *     tuned for bass-amp frequency centres
 *   - Mild built-in compressor (envelope follower → static compression curve)
 *   - DI tap (HPF + slight low-shelf) blended via direct_out_mix
 *   - Post / output gain
 *
 * It does NOT implement the cabinet IR (run a ConvolverNode upstream) nor
 * the 5-band graphic EQ (cheap to do as biquads in the builder). Those are
 * still handled by the host-side composite when a complete chain is needed.
 *
 * Author: Agent R13 (Bass Amp Designer)
 */

class BassAmpDesignerProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'pre_gain',        defaultValue: 1.0, minValue: 0,    maxValue: 4,   automationRate: 'k-rate' },
      { name: 'tube_drive',      defaultValue: 1.4, minValue: 0,    maxValue: 3,   automationRate: 'k-rate' },
      { name: 'ss_drive',        defaultValue: 1.0, minValue: 0,    maxValue: 3,   automationRate: 'k-rate' },
      { name: 'tube_blend',      defaultValue: 0.75, minValue: 0,   maxValue: 1,   automationRate: 'k-rate' },
      { name: 'bass',            defaultValue: 0.5, minValue: 0,    maxValue: 1,   automationRate: 'k-rate' },
      { name: 'mid_low',         defaultValue: 0.5, minValue: 0,    maxValue: 1,   automationRate: 'k-rate' },
      { name: 'mid_hi',          defaultValue: 0.5, minValue: 0,    maxValue: 1,   automationRate: 'k-rate' },
      { name: 'treble',          defaultValue: 0.5, minValue: 0,    maxValue: 1,   automationRate: 'k-rate' },
      { name: 'compression',     defaultValue: 0.3, minValue: 0,    maxValue: 1,   automationRate: 'k-rate' },
      { name: 'direct_out_mix',  defaultValue: 0.0, minValue: 0,    maxValue: 1,   automationRate: 'k-rate' },
      { name: 'post_gain',       defaultValue: 1.0, minValue: 0,    maxValue: 4,   automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    // Per-channel biquad state (Direct-Form-II Transposed): [z1, z2] per band
    // 4 bands: bassShelf, midLowPeak, midHiPeak, trebleShelf
    // Plus DI HPF + DI low-shelf
    this._toneState = [0, 1].map(() => Array(4).fill(0).map(() => [0, 0]));
    this._diState = [0, 1].map(() => Array(2).fill(0).map(() => [0, 0]));

    // Compressor envelope follower (per-block, mono link)
    this._envDb = -120;

    // Tone-stack coefficient cache — recomputed when params drift
    this._toneCoefs = null;
    this._toneKey = '';
    this._diCoefs = null;
  }

  // ── Biquad coef helpers (RBJ cookbook) ───────────────────────────────────
  static _peaking(freq, Q, dbGain, sr) {
    const A = Math.pow(10, dbGain / 40);
    const w0 = 2 * Math.PI * freq / sr;
    const alpha = Math.sin(w0) / (2 * Q);
    const cosw = Math.cos(w0);
    const b0 = 1 + alpha * A;
    const b1 = -2 * cosw;
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * cosw;
    const a2 = 1 - alpha / A;
    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0];
  }

  static _lowshelf(freq, Q, dbGain, sr) {
    const A = Math.pow(10, dbGain / 40);
    const w0 = 2 * Math.PI * freq / sr;
    const cosw = Math.cos(w0);
    const sinw = Math.sin(w0);
    const alpha = sinw / 2 * Math.sqrt((A + 1/A) * (1/Q - 1) + 2);
    const b0 =    A * ((A + 1) - (A - 1) * cosw + 2 * Math.sqrt(A) * alpha);
    const b1 =  2*A * ((A - 1) - (A + 1) * cosw);
    const b2 =    A * ((A + 1) - (A - 1) * cosw - 2 * Math.sqrt(A) * alpha);
    const a0 =        (A + 1) + (A - 1) * cosw + 2 * Math.sqrt(A) * alpha;
    const a1 =   -2 * ((A - 1) + (A + 1) * cosw);
    const a2 =        (A + 1) + (A - 1) * cosw - 2 * Math.sqrt(A) * alpha;
    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0];
  }

  static _highshelf(freq, Q, dbGain, sr) {
    const A = Math.pow(10, dbGain / 40);
    const w0 = 2 * Math.PI * freq / sr;
    const cosw = Math.cos(w0);
    const sinw = Math.sin(w0);
    const alpha = sinw / 2 * Math.sqrt((A + 1/A) * (1/Q - 1) + 2);
    const b0 =    A * ((A + 1) + (A - 1) * cosw + 2 * Math.sqrt(A) * alpha);
    const b1 = -2*A * ((A - 1) + (A + 1) * cosw);
    const b2 =    A * ((A + 1) + (A - 1) * cosw - 2 * Math.sqrt(A) * alpha);
    const a0 =        (A + 1) - (A - 1) * cosw + 2 * Math.sqrt(A) * alpha;
    const a1 =    2 * ((A - 1) - (A + 1) * cosw);
    const a2 =        (A + 1) - (A - 1) * cosw - 2 * Math.sqrt(A) * alpha;
    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0];
  }

  static _highpass(freq, Q, sr) {
    const w0 = 2 * Math.PI * freq / sr;
    const cosw = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * Q);
    const b0 =  (1 + cosw) / 2;
    const b1 = -(1 + cosw);
    const b2 =  (1 + cosw) / 2;
    const a0 =   1 + alpha;
    const a1 =  -2 * cosw;
    const a2 =   1 - alpha;
    return [b0/a0, b1/a0, b2/a0, a1/a0, a2/a0];
  }

  // Process a single biquad sample. coefs = [b0, b1, b2, a1, a2], state=[z1, z2]
  static _biquadStep(x, coefs, state) {
    const y = coefs[0] * x + state[0];
    state[0] = coefs[1] * x - coefs[3] * y + state[1];
    state[1] = coefs[2] * x - coefs[4] * y;
    return y;
  }

  // ── Saturation curves ────────────────────────────────────────────────────
  static _tubeShape(x, drive) {
    // Asymmetric soft clip (positive lobe compresses harder)
    const a = drive * (1 + (x > 0 ? 0.18 : 0));
    return Math.tanh(x * (1 + a)) / Math.tanh(1 + a);
  }

  static _ssShape(x, drive) {
    const k = 1 + drive * 3;
    const xs = x * k;
    return xs / (1 + Math.abs(xs));
  }

  // ── Param → coefficient recompute ────────────────────────────────────────
  _updateToneCoefs(bass, midLow, midHi, treble, sr) {
    const key = `${bass.toFixed(3)}|${midLow.toFixed(3)}|${midHi.toFixed(3)}|${treble.toFixed(3)}|${sr}`;
    if (key === this._toneKey && this._toneCoefs) return;
    this._toneKey = key;
    const bassDb   = (bass   - 0.5) * 24;  // ±12 dB
    const midLowDb = (midLow - 0.5) * 18;  // ±9 dB
    const midHiDb  = (midHi  - 0.5) * 18;  // ±9 dB
    const trebDb   = (treble - 0.5) * 24;  // ±12 dB
    this._toneCoefs = [
      BassAmpDesignerProcessor._lowshelf(80,    0.7, bassDb,   sr),
      BassAmpDesignerProcessor._peaking(250,   0.9, midLowDb, sr),
      BassAmpDesignerProcessor._peaking(900,   1.1, midHiDb,  sr),
      BassAmpDesignerProcessor._highshelf(3000, 0.7, trebDb,   sr),
    ];
  }

  _updateDiCoefs(sr) {
    if (this._diCoefs) return;
    this._diCoefs = [
      BassAmpDesignerProcessor._highpass(35,  0.7, sr),
      BassAmpDesignerProcessor._lowshelf(90, 0.7, 1.5, sr),
    ];
  }

  // ── Compressor (one-pole envelope follower + soft-knee gain reduction) ──
  _compress(x, threshold, ratio, knee, sr) {
    const inDb = 20 * Math.log10(Math.max(1e-6, Math.abs(x)));
    const aCoef = Math.exp(-1 / (0.005 * sr));
    const rCoef = Math.exp(-1 / (0.18  * sr));
    if (inDb > this._envDb) this._envDb = aCoef * this._envDb + (1 - aCoef) * inDb;
    else                    this._envDb = rCoef * this._envDb + (1 - rCoef) * inDb;
    const overshoot = this._envDb - threshold;
    let gainDb = 0;
    if (overshoot >= knee / 2) {
      gainDb = -overshoot * (1 - 1 / ratio);
    } else if (overshoot > -knee / 2 && knee > 0) {
      const u = overshoot + knee / 2;
      gainDb = -u * u * (1 - 1 / ratio) / (2 * knee);
    }
    return x * Math.pow(10, gainDb / 20);
  }

  process(inputs, outputs, parameters) {
    const input  = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const sr = sampleRate;
    const preGain  = parameters.pre_gain[0];
    const tubeDr   = parameters.tube_drive[0];
    const ssDr     = parameters.ss_drive[0];
    const tubeMix  = parameters.tube_blend[0];
    const bass     = parameters.bass[0];
    const midLow   = parameters.mid_low[0];
    const midHi    = parameters.mid_hi[0];
    const treble   = parameters.treble[0];
    const compAmt  = parameters.compression[0];
    const diMix    = parameters.direct_out_mix[0];
    const postGain = parameters.post_gain[0];

    this._updateToneCoefs(bass, midLow, midHi, treble, sr);
    this._updateDiCoefs(sr);

    // 0 → -6 dB threshold, 1.5 ratio (mild)
    // 1 → -30 dB threshold, 6 ratio    (assertive)
    const compThresh = -6  - compAmt * 24;
    const compRatio  = 1.5 + compAmt * 4.5;
    const compKnee   = 8;

    const numChannels = Math.min(input.length, 2);
    const blockSize = input[0].length;

    for (let ch = 0; ch < numChannels; ch++) {
      const inCh  = input[ch];
      const outCh = output[ch];
      const toneState = this._toneState[ch];
      const diState   = this._diState[ch];

      for (let i = 0; i < blockSize; i++) {
        const xRaw = (inCh[i] || 0) * preGain;

        // --- Stage paths ---
        const tube = BassAmpDesignerProcessor._tubeShape(xRaw, tubeDr);
        const ss   = BassAmpDesignerProcessor._ssShape(xRaw, ssDr);
        let stage = tube * tubeMix + ss * (1 - tubeMix);

        // --- 4-band tone stack ---
        for (let b = 0; b < 4; b++) {
          stage = BassAmpDesignerProcessor._biquadStep(stage, this._toneCoefs[b], toneState[b]);
        }

        // --- Compression ---
        stage = this._compress(stage, compThresh, compRatio, compKnee, sr);

        // --- DI tap (from xRaw, the post-pre-gain signal) ---
        let di = xRaw;
        di = BassAmpDesignerProcessor._biquadStep(di, this._diCoefs[0], diState[0]);
        di = BassAmpDesignerProcessor._biquadStep(di, this._diCoefs[1], diState[1]);

        // --- Mic + DI sum, post gain ---
        const blended = stage * (1 - diMix) + di * diMix;
        outCh[i] = blended * postGain;
      }
    }

    return true;
  }
}

registerProcessor('r13-bass-amp-designer-processor', BassAmpDesignerProcessor);
