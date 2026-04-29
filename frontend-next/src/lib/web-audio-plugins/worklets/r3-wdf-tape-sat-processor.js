/**
 * r3-wdf-tape-sat-processor — Wave Digital Filter style tape saturation.
 *
 * Topology:
 *   x → input gain → DC-blocking HP → asymmetric soft saturator (with bias)
 *      → low-shelf "head bump" boost (≈80–150 Hz)
 *      → speed-dependent low-pass (slow tape = darker high end)
 *      → DC blocker → wet/dry mix → out
 *
 * Refs:
 *   - Karjalainen & Pakarinen, "Modeling of Audio Distortion Circuits", DAFx-06
 *   - Eichas / Möller / Zölzer, "Block-oriented modeling of distortion audio
 *     effects using iterative minimization", DAFx-15
 *
 * a-rate parameters keep modulation buttery for input_level / head_bump / mix.
 *
 * @author Doseedo R3
 */

class WdfTapeSatProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'inputLevel', defaultValue: 1.5, minValue: 0.5, maxValue: 5,  automationRate: 'a-rate' },
      { name: 'bias',       defaultValue: 0.5, minValue: 0,   maxValue: 1,  automationRate: 'k-rate' },
      { name: 'speed',      defaultValue: 0.5, minValue: 0,   maxValue: 1,  automationRate: 'k-rate' },
      { name: 'headBump',   defaultValue: 0.5, minValue: 0,   maxValue: 1,  automationRate: 'a-rate' },
      { name: 'mix',        defaultValue: 1,   minValue: 0,   maxValue: 1,  automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    const C = 2;

    // DC blockers (1-pole HP, R≈0.995 → ~35 Hz @ 48k)
    this._dcInX = new Float32Array(C);
    this._dcInY = new Float32Array(C);
    this._dcOutX = new Float32Array(C);
    this._dcOutY = new Float32Array(C);

    // Low-shelf "head bump" 1-pole (centered ~110 Hz) — implemented as
    // input + alpha * lowpass(input). lowpass state per channel.
    this._bumpLP = new Float32Array(C);

    // Speed-dependent 1-pole LP. Coeff updated per block.
    this._speedLP = new Float32Array(C);
    this._speedAlpha = 1.0;

    // Cache the last bias / speed so we only rebuild when they move
    this._lastBias = -1;
    this._lastSpeed = -1;
    this._biasOffset = 0;
  }

  // 1-pole HP DC blocker: y[n] = x[n] - x[n-1] + R*y[n-1]
  _dcBlock(x, ch, xs, ys) {
    const R = 0.995;
    const y = x - xs[ch] + R * ys[ch];
    xs[ch] = x;
    ys[ch] = y;
    return y;
  }

  // Asymmetric soft saturator — odd-dominant tanh + small even-harmonic bias.
  // Returns f(x); f'(0)≈1 so unity small-signal gain.
  _saturate(x, biasOffset) {
    // Add bias to introduce 2nd-harmonic content, then re-center
    const xb = x + biasOffset;
    // Cubic-tanh hybrid: tanh-like but cheaper, well-behaved past |x|=4
    // y = xb / (1 + |xb| + 0.28*xb*xb) gives ~tanh shape with asymmetry
    const ax = Math.abs(xb);
    const y = xb / (1 + ax + 0.28 * xb * xb);
    // Subtract DC offset created by biasing — offset = bias / (1+|bias|+0.28*b^2)
    const ab = Math.abs(biasOffset);
    const yOffset = biasOffset / (1 + ab + 0.28 * biasOffset * biasOffset);
    return y - yOffset;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const ilArr = parameters.inputLevel;
    const biArr = parameters.bias;
    const spArr = parameters.speed;
    const hbArr = parameters.headBump;
    const mxArr = parameters.mix;

    const ilA = ilArr.length > 1;
    const hbA = hbArr.length > 1;
    const mxA = mxArr.length > 1;

    const bias = biArr[0];
    const speed = spArr[0];

    // Update bias offset (asymmetry knob 0..1 → -0.4..+0.4)
    if (bias !== this._lastBias) {
      this._biasOffset = (bias - 0.5) * 0.8;
      this._lastBias = bias;
    }

    // Speed: 0=slow/warm/dark, 1=fast/bright. Map to 1-pole LP cutoff.
    // alpha = exp(-2π*fc/fs). fc ranges ~2.5 kHz (slow) → 18 kHz (fast).
    if (speed !== this._lastSpeed) {
      const fc = 2500 + speed * (18000 - 2500);
      this._speedAlpha = Math.exp(-2 * Math.PI * fc / sampleRate);
      this._lastSpeed = speed;
    }
    const alpha = this._speedAlpha;
    const oneMinusAlpha = 1 - alpha;

    const channels = Math.min(input.length, output.length, 2);
    const blockSize = input[0].length;

    for (let ch = 0; ch < channels; ch++) {
      const ic = input[ch];
      const oc = output[ch];

      for (let i = 0; i < blockSize; i++) {
        const drive = ilA ? ilArr[i] : ilArr[0];
        const bump  = hbA ? hbArr[i] : hbArr[0];
        const mix   = mxA ? mxArr[i] : mxArr[0];

        const dry = ic[i];

        // 1) Pre-saturation DC block
        let x = this._dcBlock(dry, ch, this._dcInX, this._dcInY);

        // 2) Drive
        x = x * drive;

        // 3) Asymmetric saturation
        x = this._saturate(x, this._biasOffset);

        // 4) Head bump — 1-pole LP @ ~110 Hz, then add alpha*LP to signal
        // alphaB = exp(-2π*110/fs)
        const alphaB = Math.exp(-2 * Math.PI * 110 / sampleRate);
        this._bumpLP[ch] = alphaB * this._bumpLP[ch] + (1 - alphaB) * x;
        // bump 0..1 → 0..+6 dB shelf gain (linear ≈ 0..1.0 added)
        x = x + bump * this._bumpLP[ch];

        // 5) Speed-dependent low-pass
        this._speedLP[ch] = alpha * this._speedLP[ch] + oneMinusAlpha * x;
        x = this._speedLP[ch];

        // 6) Output DC block
        x = this._dcBlock(x, ch, this._dcOutX, this._dcOutY);

        // 7) Mix
        oc[i] = dry * (1 - mix) + x * mix;
      }
    }
    return true;
  }
}

registerProcessor('r3-wdf-tape-sat-processor', WdfTapeSatProcessor);
