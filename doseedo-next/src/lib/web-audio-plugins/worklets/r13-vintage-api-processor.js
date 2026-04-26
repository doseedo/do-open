/**
 * r13-vintage-api-processor — Vintage API-style inductor saturation worklet.
 *
 * Functionally identical to the 1073 version but registered under a separate
 * processor name so the two builders can be swapped/extended independently
 * (e.g. the API model historically uses a slightly more aggressive curve;
 * leaving room for that here without forcing a tag change on the other
 * circuit).
 *
 * Per-sample 3rd-order soft clip with a touch of even-harmonic bias.
 * `ind_drive` 0..1 → pre-gain 1..6 (slightly hotter than 1073).
 *
 * No allocations inside process(). Stereo / mono channel-agnostic.
 *
 * Author: Doseedo R13
 */

class R13VintageAPIProcessor extends AudioWorkletProcessor {
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
    const k     = 1 + drive * 5;     // pre-gain 1..6 (hotter than 1073)
    const evenBias = 0.06 * drive;
    const post = 1 / Math.max(1, k * 0.65);

    const channels = Math.min(inp.length, out.length);
    for (let ch = 0; ch < channels; ch++) {
      const xCh = inp[ch];
      const yCh = out[ch];
      if (!xCh) continue;
      for (let i = 0; i < xCh.length; i++) {
        const sx = xCh[i] * k;
        let y = (sx - (sx * sx * sx) / 3) * (1 - drive * 0.07) + (sx * sx) * evenBias;
        y *= post;
        if (y > 1) y = 1;
        else if (y < -1) y = -1;
        yCh[i] = y;
      }
    }
    return true;
  }
}

registerProcessor('r13-vintage-api-processor', R13VintageAPIProcessor);
