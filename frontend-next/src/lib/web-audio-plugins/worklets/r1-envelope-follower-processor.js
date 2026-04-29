/**
 * r1-envelope-follower-processor — RMS / peak envelope follower with
 * separate attack/release smoothing.
 *
 * Implements the classic "ballistic" follower:
 *   1. Per-sample rectification (peak = |x|, rms = sqrt(lowpass(x^2)))
 *   2. Asymmetric one-pole smoothing:
 *        env <- (1 - coef_attack) * env + coef_attack * |x|   if |x| > env
 *        env <- (1 - coef_release) * env + coef_release * |x| else
 *
 * Output is a CV-rate signal in [0, 1] connected as a normal audio output —
 * downstream consumers (LFO mod targets, filter cutoff, etc.) can listen to
 * its current value either via the audio output or via port messages.
 *
 * The processor also pushes its current value to main thread roughly every
 * 8 ms via port.postMessage so UI meters can read it without sample blocks.
 *
 * AudioWorkletParams:
 *   - attackMs   [0.1..500]
 *   - releaseMs  [1..2000]
 *   - mode       0=peak, 1=rms (k-rate, integer)
 *
 * @author Doseedo R1
 */

class EnvelopeFollowerProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'attackMs',  defaultValue: 10,  minValue: 0.1, maxValue: 500,  automationRate: 'k-rate' },
      { name: 'releaseMs', defaultValue: 100, minValue: 1,   maxValue: 2000, automationRate: 'k-rate' },
      { name: 'mode',      defaultValue: 1,   minValue: 0,   maxValue: 1,    automationRate: 'k-rate' },
    ];
  }

  constructor() {
    super();
    this._env = 0;
    this._rmsState = 0;
    // Cached coefficients — recomputed when ms params change
    this._coefA = 0.01;
    this._coefR = 0.001;
    this._lastAttackMs = -1;
    this._lastReleaseMs = -1;

    this._reportSamples = Math.floor(sampleRate * 0.008); // ~8ms
    this._reportCounter = 0;
  }

  _coefFromMs(ms) {
    // One-pole tau formula: coef = 1 - exp(-1 / (ms * sr / 1000))
    const samples = Math.max(1, (ms / 1000) * sampleRate);
    return 1 - Math.exp(-1 / samples);
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];
    if (!output || !output.length) return true;

    const attackMs  = parameters.attackMs[0];
    const releaseMs = parameters.releaseMs[0];
    const mode      = Math.round(parameters.mode[0]); // 0 peak, 1 rms

    if (attackMs !== this._lastAttackMs) {
      this._coefA = this._coefFromMs(attackMs);
      this._lastAttackMs = attackMs;
    }
    if (releaseMs !== this._lastReleaseMs) {
      this._coefR = this._coefFromMs(releaseMs);
      this._lastReleaseMs = releaseMs;
    }

    const coefA = this._coefA;
    const coefR = this._coefR;

    const inL = input && input[0] ? input[0] : null;
    const inR = input && input[1] ? input[1] : inL;
    const outChannels = output.length;
    const blockSize = output[0].length;

    let env = this._env;
    let rmsState = this._rmsState;

    // RMS smoothing tau ~= 5ms
    const rmsCoef = 1 - Math.exp(-1 / Math.max(1, sampleRate * 0.005));

    for (let i = 0; i < blockSize; i++) {
      const l = inL ? inL[i] : 0;
      const r = inR ? inR[i] : l;
      // Stereo sum/2 then rectify
      const x = (l + r) * 0.5;

      let target;
      if (mode === 0) {
        target = Math.abs(x);
      } else {
        rmsState = (1 - rmsCoef) * rmsState + rmsCoef * (x * x);
        target = Math.sqrt(rmsState);
      }

      // Asymmetric smoothing
      const coef = (target > env) ? coefA : coefR;
      env = (1 - coef) * env + coef * target;
      // Clamp output to safe CV range
      const v = env > 1 ? 1 : env;

      for (let ch = 0; ch < outChannels; ch++) {
        output[ch][i] = v;
      }
    }

    this._env = env;
    this._rmsState = rmsState;

    this._reportCounter += blockSize;
    if (this._reportCounter >= this._reportSamples) {
      this._reportCounter = 0;
      this.port.postMessage({ type: 'env', value: env });
    }

    return true;
  }
}

registerProcessor('r1-envelope-follower-processor', EnvelopeFollowerProcessor);
