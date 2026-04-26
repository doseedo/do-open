/**
 * r13-amp-designer-processor — placeholder AudioWorkletProcessor.
 *
 * The Amp Designer composite (`r13_amp_designer.js`) is built almost entirely
 * out of existing R2/R3 worklets + a ConvolverNode, so it does NOT need a
 * dedicated DSP worklet. This file exists for two reasons:
 *
 *   1. Spec-compliance with the round naming convention (every R-round may
 *      register at most one new processor file).
 *   2. Headroom — if a future revision wants to fold the entire chain into
 *      one worklet for tighter latency or shared SAB state, this file is
 *      where that processor will live. Until then it's a passthrough.
 *
 * The composite builder works perfectly well without this module being
 * loaded; `_safeWorklet()` will skip any `new AudioWorkletNode(ctx,
 * 'r13-amp-designer-processor')` call that isn't backed by a registered
 * processor and proceed via the R2/R3 substrate.
 */

class R13AmpDesignerProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      // No params — see r13_amp_designer.js builder for the public surface.
    ];
  }

  process(inputs, outputs /* , parameters */) {
    const input = inputs[0];
    const output = outputs[0];
    if (!input || input.length === 0) return true;
    for (let ch = 0; ch < output.length; ch++) {
      const inCh  = input[ch] || input[0];
      const outCh = output[ch];
      if (!inCh) {
        outCh.fill(0);
        continue;
      }
      for (let i = 0; i < outCh.length; i++) outCh[i] = inCh[i];
    }
    return true;
  }
}

registerProcessor('r13-amp-designer-processor', R13AmpDesignerProcessor);
