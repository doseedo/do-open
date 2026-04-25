/**
 * r3-wdf-power-supply-sag-processor — dynamic gain reduction simulating PSU
 * sag under load (rectifier model + capacitor drain).
 *
 * Topology:
 *   1. Full-wave rectify input → "current draw" envelope
 *   2. Fast-attack / slow-recovery 1-pole follower → smoothed "load"
 *   3. Convert load (0..1+) to gain-reduction ratio:
 *        gr = 1 - sag_amount * load        (clamped to [1-sag, 1])
 *   4. Apply gr to input. DC blocker on output for safety.
 *
 * This emulates a B+ rail dipping when the output stage pulls more current,
 * with `recovery` sec being the RC time constant of the reservoir cap.
 *
 * Params:
 *   - sag       0..1     amount of GR per unit load (1 = full kill possible)
 *   - recovery  0.01..0.5 s  cap discharge / recharge time
 *   - mix       0..1
 *
 * @author Doseedo R3
 */

class WdfPowerSupplySagProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'sag',      defaultValue: 0.5,  minValue: 0,    maxValue: 1,   automationRate: 'a-rate' },
      { name: 'recovery', defaultValue: 0.05, minValue: 0.01, maxValue: 0.5, automationRate: 'k-rate' },
      { name: 'mix',      defaultValue: 1,    minValue: 0,    maxValue: 1,   automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    const C = 2;
    // Single shared envelope (a real PSU is shared between channels)
    this._env = 0;
    // Fast attack ≈ 5 ms for the rectifier "rush"
    this._attackCoeff = Math.exp(-1 / (0.005 * sampleRate));

    // Output DC blocker
    this._dcX = new Float32Array(C);
    this._dcY = new Float32Array(C);

    // Cached release coeff
    this._lastRecovery = -1;
    this._releaseCoeff = 0;
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

    const sgArr = parameters.sag;
    const rcArr = parameters.recovery;
    const mxArr = parameters.mix;
    const sgA = sgArr.length > 1;
    const mxA = mxArr.length > 1;

    const recovery = rcArr[0];
    if (recovery !== this._lastRecovery) {
      // 1-pole release: alpha = exp(-1 / (tau * fs))
      this._releaseCoeff = Math.exp(-1 / (Math.max(0.001, recovery) * sampleRate));
      this._lastRecovery = recovery;
    }

    const aAttack  = this._attackCoeff;
    const aRelease = this._releaseCoeff;

    const channels = Math.min(input.length, output.length, 2);
    const blockSize = input[0].length;

    for (let i = 0; i < blockSize; i++) {
      // Sum-to-mono rectified "current draw"
      let load = 0;
      for (let ch = 0; ch < channels; ch++) {
        load += Math.abs(input[ch][i]);
      }
      load /= channels;

      // Asymmetric 1-pole envelope follower
      // (fast charge when load rises, slow release when it falls)
      if (load > this._env) {
        this._env = aAttack * this._env + (1 - aAttack) * load;
      } else {
        this._env = aRelease * this._env + (1 - aRelease) * load;
      }

      const sag = sgA ? sgArr[i] : sgArr[0];
      const mix = mxA ? mxArr[i] : mxArr[0];

      // Map env (≈0..1+ for hot signals) to gain reduction
      // gr = 1 - sag * env, clamped to [1-sag, 1]
      let gr = 1 - sag * this._env;
      const floor = 1 - sag;
      if (gr < floor) gr = floor;
      if (gr > 1)     gr = 1;

      for (let ch = 0; ch < channels; ch++) {
        const dry = input[ch][i];
        const wet = this._dcBlock(dry * gr, ch);
        output[ch][i] = dry * (1 - mix) + wet * mix;
      }
    }
    return true;
  }
}

registerProcessor('r3-wdf-power-supply-sag-processor', WdfPowerSupplySagProcessor);
