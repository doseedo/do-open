/**
 * r13-vintage-1073-processor — Vintage 1073-style inductor saturation worklet.
 *
 * The R13 builder owns the linear EQ via BiquadFilterNodes (highpass, low
 * shelf, mid bell, high shelf). This processor is the optional non-linear
 * "inductor" stage — a per-sample 3rd-order soft clip with a small
 * even-harmonic bias and a `ind_drive` k-rate parameter.
 *
 * Splitting the linear curve and the saturation across two stages lets
 * Web Audio's BiquadFilterNode handle the heavy lifting (it's coded
 * natively) while the worklet only runs the cheap waveshaper. If the
 * worklet fails to register the builder skips it and uses a static
 * WaveShaperNode fallback — audibly similar at moderate drive.
 *
 * No allocations inside process(). Stereo / mono channel-agnostic.
 *
 * Author: Doseedo R13
 */

class R13Vintage1073Processor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'ind_drive', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' },
    ];
  }

  process(inputs, outputs, parameters) {
    const inp = inputs[0];
    const out = outputs[0];
    if (!inp || !out || !inp.length) return true;

    const drive = parameters.ind_drive[0] || 0;
    const k     = 1 + drive * 4;     // pre-gain 1..5
    const evenBias = 0.04 * drive;
    const post = 1 / Math.max(1, k * 0.6);

    const channels = Math.min(inp.length, out.length);
    for (let ch = 0; ch < channels; ch++) {
      const xCh = inp[ch];
      const yCh = out[ch];
      if (!xCh) continue;
      for (let i = 0; i < xCh.length; i++) {
        const sx = xCh[i] * k;
        // 3rd-order soft clip + even-harmonic bias
        let y = (sx - (sx * sx * sx) / 3) * (1 - drive * 0.05) + (sx * sx) * evenBias;
        y *= post;
        // Hard limit ±1
        if (y > 1) y = 1;
        else if (y < -1) y = -1;
        yCh[i] = y;
      }
    }
    return true;
  }
}

registerProcessor('r13-vintage-1073-processor', R13Vintage1073Processor);
