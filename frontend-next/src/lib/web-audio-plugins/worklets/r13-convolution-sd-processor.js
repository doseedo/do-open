/**
 * r13-convolution-sd-processor — Space Designer (convolution_sd) worklet stub
 *
 * RESERVED for a future partitioned-FFT real-time convolution implementation
 * that would replace the ConvolverNode-based primary path in
 * `src/audio/builders/r13_space_designer.js` for very long IRs (> 5 s).
 *
 * Today this file is a passthrough — it registers the processor name so
 * `audioWorklet.addModule(...)` succeeds on cold engines, but the R13 builder
 * never instantiates it (it always uses the ConvolverNode path). Loading this
 * module is therefore optional; absence does not break the builder.
 *
 * When the partitioned convolver is implemented, the contract should be:
 *
 *   parameters:
 *     mix          (a-rate, 0..1, default 0.3)
 *     low_cut_hz   (k-rate, 20..20000, default 20)      // pre-IR HPF
 *     high_cut_hz  (k-rate, 20..20000, default 20000)   // pre-IR LPF
 *     predelay_ms  (k-rate, 0..500,    default 0)
 *
 *   port messages (main → worklet):
 *     { type: 'setIR', channels: Float32Array[][] }       // raw source IR
 *     { type: 'setShape',
 *         length, attack_time, decay_time, density, reverse }
 *
 *   port messages (worklet → main):
 *     { type: 'irShaped', frames }                        // ack IR rebuild
 *
 * The shaping math itself lives in `r13_space_designer.js:shapeIR(...)` and is
 * portable into the worklet wholesale (no AudioContext-only APIs in there).
 *
 * Author: Agent R13
 */

class R13ConvolutionSDProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'mix',         defaultValue: 0.3,   minValue: 0,  maxValue: 1,     automationRate: 'a-rate' },
      { name: 'low_cut_hz',  defaultValue: 20,    minValue: 20, maxValue: 20000, automationRate: 'k-rate' },
      { name: 'high_cut_hz', defaultValue: 20000, minValue: 20, maxValue: 20000, automationRate: 'k-rate' },
      { name: 'predelay_ms', defaultValue: 0,     minValue: 0,  maxValue: 500,   automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    // Stub — no IR partitioning yet. Builder never instantiates this; the
    // class exists so addModule() resolves and a future R13 worklet upgrade
    // is a drop-in replacement.
    this.port.onmessage = (_e) => { /* swallow until real impl lands */ };
  }

  process(inputs, outputs /*, parameters */) {
    // Passthrough. Real implementation would:
    //   1. apply HPF + LPF (state-variable or biquad in-place)
    //   2. zero-pad input frame, FFT, multiply with the partitioned IR
    //      spectrum, IFFT, overlap-add into the output ring
    //   3. apply pre-delay via a small ring read offset
    //   4. crossfade dry/wet via the `mix` a-rate param
    const input = inputs[0];
    const output = outputs[0];
    if (!input || !output) return true;
    const channels = Math.min(input.length, output.length);
    for (let ch = 0; ch < channels; ch++) {
      const inCh = input[ch];
      const outCh = output[ch];
      if (!inCh || !outCh) continue;
      for (let i = 0; i < outCh.length; i++) {
        outCh[i] = inCh[i] || 0;
      }
    }
    return true;
  }
}

try {
  registerProcessor('r13-convolution-sd-processor', R13ConvolutionSDProcessor);
} catch (e) {
  // Already registered (idempotent module load) — fine.
}
