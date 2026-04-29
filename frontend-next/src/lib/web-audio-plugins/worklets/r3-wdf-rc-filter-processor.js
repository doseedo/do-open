/**
 * r3-wdf-rc-filter-processor — RC lowpass derived from physical R, L, C values.
 *
 * Continuous-time TF:  H(s) = 1 / (1 + sRC)
 * Bilinear transform with prewarped fc = 1/(2π·R·C):
 *   ωc = 2π·fc
 *   K  = tan(ωc·T/2),  with T = 1/sampleRate
 *   y[n] = (K*(x[n] + x[n-1]) - (K-1)*y[n-1]) / (K+1)
 *
 * This gives a one-pole LP with the cutoff exactly at 1/(2π·R·C).
 *
 * Params:
 *   - resistance   100..1e6 Ω
 *   - capacitance  1e-12..1e-5 F
 *   - mix          0..1
 *
 * For audio-graph efficiency the JS builder usually maps this onto a
 * BiquadFilterNode (see r3.js); this worklet exists as a reference / fallback.
 *
 * @author Doseedo R3
 */

class WdfRcFilterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'resistance',  defaultValue: 10000, minValue: 100,   maxValue: 1e6,  automationRate: 'k-rate' },
      { name: 'capacitance', defaultValue: 1e-8,  minValue: 1e-12, maxValue: 1e-5, automationRate: 'k-rate' },
      { name: 'mix',         defaultValue: 1,     minValue: 0,     maxValue: 1,    automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    const C = 2;
    this._x1 = new Float32Array(C);
    this._y1 = new Float32Array(C);

    this._lastR = -1;
    this._lastC = -1;
    this._a = 0; // K / (K+1)
    this._b = 0; // (K-1) / (K+1)
  }

  _updateCoeffs(R, C) {
    const T = 1 / sampleRate;
    // fc = 1 / (2π R C)
    const fc = 1 / (2 * Math.PI * R * C);
    // Clamp against Nyquist
    const fcSafe = Math.max(1, Math.min(fc, 0.49 * sampleRate));
    const wc = 2 * Math.PI * fcSafe;
    const K = Math.tan(wc * T * 0.5);
    const denom = K + 1;
    this._a = K / denom;
    this._b = (K - 1) / denom;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const R = parameters.resistance[0];
    const C = parameters.capacitance[0];
    const mxArr = parameters.mix;
    const mxA = mxArr.length > 1;

    if (R !== this._lastR || C !== this._lastC) {
      this._updateCoeffs(R, C);
      this._lastR = R;
      this._lastC = C;
    }

    const a = this._a;
    const b = this._b;

    const channels = Math.min(input.length, output.length, 2);
    const blockSize = input[0].length;

    for (let ch = 0; ch < channels; ch++) {
      const ic = input[ch];
      const oc = output[ch];
      let x1 = this._x1[ch];
      let y1 = this._y1[ch];

      for (let i = 0; i < blockSize; i++) {
        const x0 = ic[i];
        // y = a*(x + x1) - b*y1
        const y0 = a * (x0 + x1) - b * y1;
        x1 = x0;
        y1 = y0;

        const mix = mxA ? mxArr[i] : mxArr[0];
        oc[i] = x0 * (1 - mix) + y0 * mix;
      }

      this._x1[ch] = x1;
      this._y1[ch] = y1;
    }
    return true;
  }
}

registerProcessor('r3-wdf-rc-filter-processor', WdfRcFilterProcessor);
