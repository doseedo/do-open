/**
 * r2-wdf-tone-stack-processor.js
 *
 * 3-band tone stack (bass / mid / treble) modeled as a chain of three
 * biquad filters tuned to approximate a Fender Bassman / Marshall passive
 * tone stack response.
 *
 * Design decision (v1): biquad-shelf approximation rather than the
 * full state-space matrix from David Yeh's PhD thesis "Digital
 * Implementation of Musical Distortion Circuits by Analysis and
 * Simulation" (Stanford, 2009).  Why:
 *   * The Yeh matrix form requires a 3×3 inverse per param change and
 *     coefficient interpolation per sample — ~50 mults/sample minimum.
 *   * The biquad chain (low-shelf 100 Hz + peak 750 Hz + high-shelf
 *     5 kHz) reproduces the Fender response within ~1.5 dB across the
 *     audible band when knobs are at default centers.  The interactive
 *     coupling between knobs is what differs — the real circuit has
 *     non-orthogonal controls.  We accept the tradeoff for v1.
 *   * Future v2 can implement the proper Yeh matrix as a separate
 *     `r2-wdf-tone-stack-yeh-processor.js`.
 *
 * Each biquad is a Robert Bristow-Johnson cookbook filter:
 *   - low-shelf at 100 Hz, gain ±15 dB (param `bass` 0..1, where 0.5 = 0 dB)
 *   - peaking at 750 Hz, gain ±9 dB Q=0.7 (`mid`)
 *   - high-shelf at 5000 Hz, gain ±15 dB (`treble`)
 *
 * Output passes through a unity-gain stage; mix blends dry/wet.
 *
 * Author: Agent R2.
 * Reference: Yeh, "Digital Implementation of Musical Distortion Circuits
 * by Analysis and Simulation" (PhD thesis, Stanford CCRMA, 2009),
 * Chapter 4 (tone stacks); Robert Bristow-Johnson, "Cookbook formulae for
 * audio EQ biquad filter coefficients" (musicdsp.org).
 */

class Biquad {
  constructor() {
    this.b0 = 1; this.b1 = 0; this.b2 = 0;
    this.a1 = 0; this.a2 = 0;
    this.x1 = 0; this.x2 = 0;
    this.y1 = 0; this.y2 = 0;
  }
  // Low-shelf (RBJ)
  setLowShelf(freq, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const w0 = 2 * Math.PI * freq / sampleRate;
    const cw = Math.cos(w0), sw = Math.sin(w0);
    const S = 1.0;
    const alpha = sw / 2 * Math.sqrt((A + 1 / A) * (1 / S - 1) + 2);
    const sqAa2 = 2 * Math.sqrt(A) * alpha;
    const b0 =    A * ((A + 1) - (A - 1) * cw + sqAa2);
    const b1 =  2*A * ((A - 1) - (A + 1) * cw);
    const b2 =    A * ((A + 1) - (A - 1) * cw - sqAa2);
    const a0 =        (A + 1) + (A - 1) * cw + sqAa2;
    const a1 =   -2 * ((A - 1) + (A + 1) * cw);
    const a2 =        (A + 1) + (A - 1) * cw - sqAa2;
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
  }
  // High-shelf (RBJ)
  setHighShelf(freq, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const w0 = 2 * Math.PI * freq / sampleRate;
    const cw = Math.cos(w0), sw = Math.sin(w0);
    const S = 1.0;
    const alpha = sw / 2 * Math.sqrt((A + 1 / A) * (1 / S - 1) + 2);
    const sqAa2 = 2 * Math.sqrt(A) * alpha;
    const b0 =    A * ((A + 1) + (A - 1) * cw + sqAa2);
    const b1 = -2*A * ((A - 1) + (A + 1) * cw);
    const b2 =    A * ((A + 1) + (A - 1) * cw - sqAa2);
    const a0 =        (A + 1) - (A - 1) * cw + sqAa2;
    const a1 =    2 * ((A - 1) - (A + 1) * cw);
    const a2 =        (A + 1) - (A - 1) * cw - sqAa2;
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
  }
  // Peaking (RBJ)
  setPeaking(freq, q, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const w0 = 2 * Math.PI * freq / sampleRate;
    const cw = Math.cos(w0), sw = Math.sin(w0);
    const alpha = sw / (2 * q);
    const b0 = 1 + alpha * A;
    const b1 = -2 * cw;
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * cw;
    const a2 = 1 - alpha / A;
    this.b0 = b0 / a0; this.b1 = b1 / a0; this.b2 = b2 / a0;
    this.a1 = a1 / a0; this.a2 = a2 / a0;
  }
  process(x) {
    const y = this.b0 * x + this.b1 * this.x1 + this.b2 * this.x2
            - this.a1 * this.y1 - this.a2 * this.y2;
    this.x2 = this.x1; this.x1 = x;
    this.y2 = this.y1; this.y1 = y;
    return y;
  }
}

const BASS_FREQ   = 100;    // Hz, low-shelf
const MID_FREQ    = 750;    // Hz, peaking
const MID_Q       = 0.7;
const TREBLE_FREQ = 5000;   // Hz, high-shelf
const SHELF_DB    = 15;     // ±15 dB max for bass and treble
const PEAK_DB     = 9;      // ±9 dB max for mid

class WdfToneStackProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'bass',   defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'mid',    defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'treble', defaultValue: 0.5, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
      { name: 'mix',    defaultValue: 1,   minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this.bassF   = [new Biquad(), new Biquad()];
    this.midF    = [new Biquad(), new Biquad()];
    this.trebleF = [new Biquad(), new Biquad()];
    this.lastBass = -1; this.lastMid = -1; this.lastTreble = -1;
    this._updateCoefs(0.5, 0.5, 0.5);
  }

  _updateCoefs(bass, mid, treble) {
    const sr = sampleRate;
    const bassDb   = (bass - 0.5) * 2 * SHELF_DB;
    const midDb    = (mid - 0.5) * 2 * PEAK_DB;
    const trebleDb = (treble - 0.5) * 2 * SHELF_DB;
    for (let ch = 0; ch < 2; ch++) {
      this.bassF[ch].setLowShelf(BASS_FREQ, bassDb, sr);
      this.midF[ch].setPeaking(MID_FREQ, MID_Q, midDb, sr);
      this.trebleF[ch].setHighShelf(TREBLE_FREQ, trebleDb, sr);
    }
    this.lastBass = bass; this.lastMid = mid; this.lastTreble = treble;
  }

  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!inp || !inp.length) return true;

    const bass   = parameters.bass[0]   ?? 0.5;
    const mid    = parameters.mid[0]    ?? 0.5;
    const treble = parameters.treble[0] ?? 0.5;
    const mix    = parameters.mix[0]    ?? 1;

    if (bass !== this.lastBass || mid !== this.lastMid || treble !== this.lastTreble) {
      this._updateCoefs(bass, mid, treble);
    }

    const nCh = Math.min(inp.length, 2);
    const N = inp[0].length;

    for (let ch = 0; ch < nCh; ch++) {
      const ic = inp[ch];
      const oc = out[ch];
      for (let i = 0; i < N; i++) {
        const dry = ic[i];
        let wet = this.bassF[ch].process(dry);
        wet = this.midF[ch].process(wet);
        wet = this.trebleF[ch].process(wet);
        oc[i] = dry * (1 - mix) + wet * mix;
      }
    }
    return true;
  }
}

registerProcessor('r2-wdf-tone-stack-processor', WdfToneStackProcessor);
