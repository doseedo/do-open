/**
 * r3-wdf-rlc-filter-processor — series-RLC second-order filter from physical
 * R, L, C values. Treated as a band-pass on the resistor (i.e. damping
 * controls Q, fc set by L+C).
 *
 * Continuous-time series-RLC band-pass (taken across R):
 *   H(s) = (s · R/L) / (s² + s·R/L + 1/(LC))
 *   ω0 = 1/√(LC),  Q = (1/R) · √(L/C)
 *
 * Implementation: convert to a BiquadFilter band-pass via cookbook RBJ formula
 *   (the JS builder uses a native BiquadFilterNode of type 'bandpass'); this
 *   worklet implements it directly with Direct-Form-I biquad coefficients.
 *
 * Params:
 *   - resistance  10..1e5 Ω
 *   - inductance  1e-6..1 H
 *   - capacitance 1e-12..1e-4 F
 *   - mix         0..1
 *
 * @author Doseedo R3
 */

class WdfRlcFilterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'resistance',  defaultValue: 1000, minValue: 10,    maxValue: 1e5,  automationRate: 'k-rate' },
      { name: 'inductance',  defaultValue: 0.01, minValue: 1e-6,  maxValue: 1,    automationRate: 'k-rate' },
      { name: 'capacitance', defaultValue: 1e-7, minValue: 1e-12, maxValue: 1e-4, automationRate: 'k-rate' },
      { name: 'mix',         defaultValue: 1,    minValue: 0,     maxValue: 1,    automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    const C = 2;
    this._x1 = new Float32Array(C);
    this._x2 = new Float32Array(C);
    this._y1 = new Float32Array(C);
    this._y2 = new Float32Array(C);

    this._lastR = -1; this._lastL = -1; this._lastC = -1;
    this._b0 = 0; this._b1 = 0; this._b2 = 0;
    this._a1 = 0; this._a2 = 0;
  }

  _updateCoeffs(R, L, C) {
    // Compute resonant frequency and Q from physical values
    const w0Cont = 1 / Math.sqrt(L * C);     // rad/s
    const f0 = w0Cont / (2 * Math.PI);
    // Clamp to safe audio range
    const f0Safe = Math.max(20, Math.min(f0, 0.45 * sampleRate));
    const Q = Math.max(0.1, Math.min((1 / R) * Math.sqrt(L / C), 100));

    // RBJ band-pass (constant 0 dB peak gain) — gives a constant-Q BP
    const w0 = 2 * Math.PI * f0Safe / sampleRate;
    const cosw0 = Math.cos(w0);
    const sinw0 = Math.sin(w0);
    const alpha = sinw0 / (2 * Q);

    const b0 =  alpha;
    const b1 =  0;
    const b2 = -alpha;
    const a0 =  1 + alpha;
    const a1 = -2 * cosw0;
    const a2 =  1 - alpha;

    // Pre-divide by a0
    const inv = 1 / a0;
    this._b0 = b0 * inv;
    this._b1 = b1 * inv;
    this._b2 = b2 * inv;
    this._a1 = a1 * inv;
    this._a2 = a2 * inv;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const R = parameters.resistance[0];
    const L = parameters.inductance[0];
    const C = parameters.capacitance[0];
    const mxArr = parameters.mix;
    const mxA = mxArr.length > 1;

    if (R !== this._lastR || L !== this._lastL || C !== this._lastC) {
      this._updateCoeffs(R, L, C);
      this._lastR = R; this._lastL = L; this._lastC = C;
    }

    const b0 = this._b0, b1 = this._b1, b2 = this._b2;
    const a1 = this._a1, a2 = this._a2;

    const channels = Math.min(input.length, output.length, 2);
    const blockSize = input[0].length;

    for (let ch = 0; ch < channels; ch++) {
      const ic = input[ch];
      const oc = output[ch];
      let x1 = this._x1[ch], x2 = this._x2[ch];
      let y1 = this._y1[ch], y2 = this._y2[ch];

      for (let i = 0; i < blockSize; i++) {
        const x0 = ic[i];
        const y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
        x2 = x1; x1 = x0;
        y2 = y1; y1 = y0;

        const mix = mxA ? mxArr[i] : mxArr[0];
        oc[i] = x0 * (1 - mix) + y0 * mix;
      }

      this._x1[ch] = x1; this._x2[ch] = x2;
      this._y1[ch] = y1; this._y2[ch] = y2;
    }
    return true;
  }
}

registerProcessor('r3-wdf-rlc-filter-processor', WdfRlcFilterProcessor);
