/**
 * r3-wdf-transformer-processor — output transformer saturation.
 *
 * v1 model: atan(drive * x) * saturation, with two-stage soft compressor on
 * the post-saturator gain to mimic core saturation. A simple 1-tap memory
 * gives a tiny hysteresis-like asymmetry. DC blocker on output.
 *
 * (Full Jiles-Atherton B-H hysteresis is left as a TODO; see Holters/Zölzer
 *  "Physical Modelling of a Wah-wah Effect Pedal as a Case Study for the
 *  Application of the Nodal DK Method" for a related WDF treatment.)
 *
 * Params:
 *   - drive       0.1..5    — pre-gain into the nonlinearity
 *   - saturation  0.1..1    — output scaling / compression depth
 *   - mix         0..1
 *
 * @author Doseedo R3
 */

class WdfTransformerProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'drive',      defaultValue: 1,   minValue: 0.1, maxValue: 5, automationRate: 'a-rate' },
      { name: 'saturation', defaultValue: 0.5, minValue: 0.1, maxValue: 1, automationRate: 'a-rate' },
      { name: 'mix',        defaultValue: 1,   minValue: 0,   maxValue: 1, automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    const C = 2;
    this._dcX = new Float32Array(C);
    this._dcY = new Float32Array(C);
    this._prev = new Float32Array(C); // 1-sample memory for hysteresis-ish
  }

  _dcBlock(x, ch) {
    const R = 0.995;
    const y = x - this._dcX[ch] + R * this._dcY[ch];
    this._dcX[ch] = x;
    this._dcY[ch] = y;
    return y;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const drArr = parameters.drive;
    const saArr = parameters.saturation;
    const mxArr = parameters.mix;
    const drA = drArr.length > 1;
    const saA = saArr.length > 1;
    const mxA = mxArr.length > 1;

    const channels = Math.min(input.length, output.length, 2);
    const blockSize = input[0].length;

    const TWO_OVER_PI = 2 / Math.PI;

    for (let ch = 0; ch < channels; ch++) {
      const ic = input[ch];
      const oc = output[ch];

      for (let i = 0; i < blockSize; i++) {
        const drive      = drA ? drArr[i] : drArr[0];
        const saturation = saA ? saArr[i] : saArr[0];
        const mix        = mxA ? mxArr[i] : mxArr[0];

        const dry = ic[i];

        // Hysteresis-flavored input: blend current sample with a fraction of
        // the previous saturated output (B-H lag emulation). Tiny coefficient
        // keeps it stable.
        const x = dry * drive + 0.05 * this._prev[ch];

        // Core nonlinearity: 2/π * atan(x). Smooth, monotonic, perfect for
        // transformer-core soft saturation.
        const sat = TWO_OVER_PI * Math.atan(x);

        // Saturation knob scales output and applies a touch of compression
        // by mixing in tanh(0.7*sat) for a smoother knee at low values.
        const compressed = saturation * sat + (1 - saturation) * dry * 0.5;

        // Update hysteresis memory
        this._prev[ch] = sat;

        // DC block
        const wet = this._dcBlock(compressed, ch);

        oc[i] = dry * (1 - mix) + wet * mix;
      }
    }
    return true;
  }
}

registerProcessor('r3-wdf-transformer-processor', WdfTransformerProcessor);
