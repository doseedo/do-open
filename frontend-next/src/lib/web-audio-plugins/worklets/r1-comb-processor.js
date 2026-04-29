/**
 * r1-comb-processor — feedback comb filter (positive + negative modes)
 *
 *   Positive comb (mode=0):  y[n] = x[n] + g * y[n - D]    (resonant peaks at k/D)
 *   Negative comb (mode=1):  y[n] = x[n] - g * y[n - D]    (notches at k/D)
 *
 * Where D = delay_ms * sampleRate / 1000 (with linear-interpolated fractional delay).
 *
 * Stable for |g| < 1. Clamped to ±0.99 internally.
 *
 * AudioWorkletParams:
 *   - delayMs   [0.1..100]
 *   - feedback  [-0.99..0.99]   sign + magnitude both honored
 *   - mode      0=positive, 1=negative   (k-rate; positive uses +feedback, negative flips sign)
 *   - mix       [0..1]
 *
 * @author Doseedo R1
 */

const MAX_DELAY_S = 0.2;   // 200 ms is plenty for comb (covers 5 Hz fundamental)

class CombProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'delayMs',  defaultValue: 5,    minValue: 0.05, maxValue: 100,  automationRate: 'a-rate' },
      { name: 'feedback', defaultValue: 0.5,  minValue: -0.99, maxValue: 0.99, automationRate: 'a-rate' },
      { name: 'mode',     defaultValue: 0,    minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      { name: 'mix',      defaultValue: 1.0,  minValue: 0,    maxValue: 1.0,  automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this._bufLen = Math.ceil(MAX_DELAY_S * sampleRate) + 4;
    this._buf = [new Float32Array(this._bufLen), new Float32Array(this._bufLen)];
    this._writeIdx = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || !output.length) return true;

    const delayArr  = parameters.delayMs;
    const fbArr     = parameters.feedback;
    const mode      = parameters.mode[0] >= 0.5 ? 1 : 0;
    const mix       = parameters.mix[0];
    const delayIsA  = delayArr.length > 1;
    const fbIsA     = fbArr.length > 1;

    const channels = Math.min(2, output.length);
    const blockSize = output[0].length;
    const bufLen = this._bufLen;

    let widx = this._writeIdx;

    for (let i = 0; i < blockSize; i++) {
      const dms = delayIsA ? delayArr[i] : delayArr[0];
      let fb = fbIsA ? fbArr[i] : fbArr[0];
      // Mode flips sign — and combine with feedback's own sign
      if (mode === 1) fb = -fb;
      // Hard-clamp for stability
      if (fb > 0.99) fb = 0.99;
      else if (fb < -0.99) fb = -0.99;

      let dSamps = (dms / 1000) * sampleRate;
      if (dSamps < 1) dSamps = 1;
      if (dSamps > bufLen - 2) dSamps = bufLen - 2;

      // Per-channel processing — independent comb per channel
      for (let ch = 0; ch < channels; ch++) {
        const buf = this._buf[ch];
        const inSrc = (input && input[ch]) ? input[ch] : (input && input[0] ? input[0] : null);
        const dry = inSrc ? inSrc[i] : 0;

        // Read with fractional delay (linear interp)
        let ridx = widx - dSamps;
        while (ridx < 0) ridx += bufLen;
        const i0 = Math.floor(ridx);
        const i1 = (i0 + 1) % bufLen;
        const frac = ridx - i0;
        const delayed = buf[i0] * (1 - frac) + buf[i1] * frac;

        // Comb output (feedback form)
        const y = dry + fb * delayed;
        buf[widx] = y;

        // Output: mix wet (y) with dry — wet is the comb output, dry passes
        output[ch][i] = dry * (1 - mix) + y * mix;
      }

      widx += 1;
      if (widx >= bufLen) widx = 0;
    }

    this._writeIdx = widx;
    return true;
  }
}

registerProcessor('r1-comb-processor', CombProcessor);
