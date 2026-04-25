/**
 * r1-bitcrusher-processor — sample-rate reduction + bit-depth quantization
 *
 * Implements true digital decimation:
 *   1. Sample-and-hold downsampling (sample_rate_div=N → keep 1 sample every N)
 *   2. Bit-depth quantization (bit_depth=B → 2^B discrete levels in [-1, 1])
 *   3. Wet/dry mix
 *
 * AudioWorkletParams (a-rate where useful):
 *   - bitDepth      [1..24]    fractional bits OK (int part used for steps)
 *   - sampleRateDiv [1..64]    integer; div=1 = no downsampling
 *   - mix           [0..1]     wet amount
 *
 * @author Doseedo R1
 */

class BitcrusherProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'bitDepth',      defaultValue: 8,   minValue: 1, maxValue: 24,  automationRate: 'a-rate' },
      { name: 'sampleRateDiv', defaultValue: 4,   minValue: 1, maxValue: 64,  automationRate: 'a-rate' },
      { name: 'mix',           defaultValue: 0.5, minValue: 0, maxValue: 1.0, automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    // Sample-and-hold state (per channel, up to 8)
    this._held = new Float32Array(8);
    this._holdCounter = new Float32Array(8);
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !input.length) return true;

    const bitArr  = parameters.bitDepth;
    const divArr  = parameters.sampleRateDiv;
    const mixArr  = parameters.mix;
    const bitIsA  = bitArr.length > 1;
    const divIsA  = divArr.length > 1;
    const mixIsA  = mixArr.length > 1;

    const channels = Math.min(input.length, output.length);
    const blockSize = input[0].length;

    for (let ch = 0; ch < channels; ch++) {
      const ic = input[ch];
      const oc = output[ch];
      let held = this._held[ch];
      let counter = this._holdCounter[ch];

      for (let i = 0; i < blockSize; i++) {
        const bits = bitIsA ? bitArr[i] : bitArr[0];
        const div  = Math.max(1, divIsA ? divArr[i] : divArr[0]);
        const mix  = mixIsA ? mixArr[i] : mixArr[0];

        const dry = ic[i];

        // Sample-and-hold: refresh held value every `div` samples
        if (counter <= 0) {
          held = dry;
          counter = div;
        }
        counter -= 1;

        // Bit quantization: levels = 2^bits, so step = 2/levels in [-1,1]
        const levels = Math.pow(2, Math.max(1, bits));
        const step = 2 / levels;
        // Round to nearest grid point — symmetric around 0
        const quantized = Math.round(held / step) * step;

        oc[i] = dry * (1 - mix) + quantized * mix;
      }

      this._held[ch] = held;
      this._holdCounter[ch] = counter;
    }

    return true;
  }
}

registerProcessor('r1-bitcrusher-processor', BitcrusherProcessor);
