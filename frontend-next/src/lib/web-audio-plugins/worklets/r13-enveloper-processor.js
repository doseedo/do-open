/**
 * r13-enveloper-processor — Logic Pro Enveloper (transient shaper) port.
 *
 * Two parallel asymmetric one-pole envelope followers:
 *   - Fast follower: ~1 ms attack / ~10 ms release. Tracks transients.
 *   - Slow follower: ~30 ms attack / ~300 ms release. Tracks the body.
 *
 * Envelope decomposition:
 *   attack_env  = max(0, fast - slow)               // spikes at transients
 *   sustain_env = max(0, slow - 0.6 * fast)         // body between transients
 *
 * Both envelopes are auto-normalised by a slow-tracking input peak so the
 * gain math is independent of input level — a -20 dBFS drum hit and a
 * 0 dBFS drum hit produce the same shape factor (just different peak heights).
 *
 * Gain calculation (per-sample, identical L/R):
 *   attack_gain  = 1 + (attack  / 100) * attack_env_norm
 *   sustain_gain = 1 + (sustain / 100) * sustain_env_norm
 *   total_gain   = attack_gain * sustain_gain
 *   y            = mix * (x * total_gain * 10^(output_gain/20))
 *                + (1 - mix) * x
 *
 * AudioWorkletParams (k-rate where listed):
 *   attack             [-100..+100]   percent (boost (+) / soften (-) attack)
 *   sustain            [-100..+100]   percent (boost (+) / cut    (-) sustain)
 *   attack_time_ms     [0.1..10]   k   fast follower attack
 *   attack_release_ms  [1..50]     k   fast follower release
 *   sustain_time_ms    [10..200]   k   slow follower attack
 *   sustain_release_ms [100..2000] k   slow follower release
 *   output_gain        [-12..+12]      dB makeup
 *   mix                [0..1]          dry/wet
 *
 * @author Doseedo R13
 */

class EnveloperProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'attack',             defaultValue: 0,    minValue: -100, maxValue:  100, automationRate: 'a-rate' },
      { name: 'sustain',            defaultValue: 0,    minValue: -100, maxValue:  100, automationRate: 'a-rate' },
      { name: 'attack_time_ms',     defaultValue: 1.0,  minValue: 0.1,  maxValue:  10,  automationRate: 'k-rate' },
      { name: 'attack_release_ms',  defaultValue: 10,   minValue: 1,    maxValue:  50,  automationRate: 'k-rate' },
      { name: 'sustain_time_ms',    defaultValue: 30,   minValue: 10,   maxValue:  200, automationRate: 'k-rate' },
      { name: 'sustain_release_ms', defaultValue: 300,  minValue: 100,  maxValue:  2000,automationRate: 'k-rate' },
      { name: 'output_gain',        defaultValue: 0,    minValue: -12,  maxValue:  12,  automationRate: 'a-rate' },
      { name: 'mix',                defaultValue: 1,    minValue: 0,    maxValue:  1,   automationRate: 'a-rate' },
    ];
  }

  constructor() {
    super();
    // Follower state
    this._fast = 0;
    this._slow = 0;
    // Slow-tracking peak used to normalise the (fast - slow) shape factor
    // so the gain math doesn't depend on absolute input level.
    this._peak = 1e-6;

    // Cached one-pole coefficients — recomputed when the corresponding ms
    // params change.
    this._lastFa = -1; this._coefFa = 0;
    this._lastFr = -1; this._coefFr = 0;
    this._lastSa = -1; this._coefSa = 0;
    this._lastSr = -1; this._coefSr = 0;

    // Peak follower coefficients (fixed: 5 ms attack, 1 s release)
    this._coefPa = this._coefFromMs(5);
    this._coefPr = this._coefFromMs(1000);
  }

  _coefFromMs(ms) {
    const samples = Math.max(1, (ms / 1000) * sampleRate);
    return 1 - Math.exp(-1 / samples);
  }

  process(inputs, outputs, parameters) {
    const input  = inputs[0];
    const output = outputs[0];
    if (!output || !output.length) return true;

    // Recompute follower coefficients on param changes (k-rate)
    const fa = parameters.attack_time_ms[0];
    const fr = parameters.attack_release_ms[0];
    const sa = parameters.sustain_time_ms[0];
    const sr = parameters.sustain_release_ms[0];
    if (fa !== this._lastFa) { this._coefFa = this._coefFromMs(fa); this._lastFa = fa; }
    if (fr !== this._lastFr) { this._coefFr = this._coefFromMs(fr); this._lastFr = fr; }
    if (sa !== this._lastSa) { this._coefSa = this._coefFromMs(sa); this._lastSa = sa; }
    if (sr !== this._lastSr) { this._coefSr = this._coefFromMs(sr); this._lastSr = sr; }

    const cFa = this._coefFa, cFr = this._coefFr;
    const cSa = this._coefSa, cSr = this._coefSr;
    const cPa = this._coefPa, cPr = this._coefPr;

    // a-rate params: may be Float32Array of blockSize OR length-1 (constant)
    const pAttack  = parameters.attack;
    const pSustain = parameters.sustain;
    const pOutDb   = parameters.output_gain;
    const pMix     = parameters.mix;

    const inL = input && input[0] ? input[0] : null;
    const inR = input && input[1] ? input[1] : inL;
    const outChannels = output.length;
    const blockSize = output[0].length;

    let fast = this._fast;
    let slow = this._slow;
    let peak = this._peak;

    for (let i = 0; i < blockSize; i++) {
      const l = inL ? inL[i] : 0;
      const r = inR ? inR[i] : l;
      // Use mono sum/2 for envelope detection; gain is applied per-channel
      const xMono = (l + r) * 0.5;
      const ax = Math.abs(xMono);

      // Fast follower (asymmetric)
      {
        const c = (ax > fast) ? cFa : cFr;
        fast = (1 - c) * fast + c * ax;
      }
      // Slow follower (asymmetric)
      {
        const c = (ax > slow) ? cSa : cSr;
        slow = (1 - c) * slow + c * ax;
      }
      // Slow-tracking peak (always toward max) for normalisation
      {
        const c = (ax > peak) ? cPa : cPr;
        peak = (1 - c) * peak + c * ax;
      }

      const denom = peak > 1e-5 ? peak : 1e-5;

      // Decompose
      let attackEnv  = (fast - slow) / denom;
      if (attackEnv < 0) attackEnv = 0;
      else if (attackEnv > 1) attackEnv = 1;

      let sustainEnv = (slow - 0.6 * fast) / denom;
      if (sustainEnv < 0) sustainEnv = 0;
      else if (sustainEnv > 1) sustainEnv = 1;

      // Per-sample params (a-rate guarded for length-1 case)
      const aPct  = (pAttack.length  > 1) ? pAttack[i]  : pAttack[0];
      const sPct  = (pSustain.length > 1) ? pSustain[i] : pSustain[0];
      const odb   = (pOutDb.length   > 1) ? pOutDb[i]   : pOutDb[0];
      const mix   = (pMix.length     > 1) ? pMix[i]     : pMix[0];

      const aGain = 1 + (aPct / 100) * attackEnv;
      const sGain = 1 + (sPct / 100) * sustainEnv;
      let totalG = aGain * sGain;
      // Bound total gain — Logic's Enveloper soft-clips here. We just clamp.
      if (totalG < 0) totalG = 0;
      else if (totalG > 16) totalG = 16;

      const outDbLin = Math.pow(10, odb / 20);
      const wet      = totalG * outDbLin;
      const m        = mix < 0 ? 0 : (mix > 1 ? 1 : mix);

      // Output L/R
      const yL = m * (l * wet) + (1 - m) * l;
      const yR = m * (r * wet) + (1 - m) * r;
      output[0][i] = yL;
      if (outChannels > 1) output[1][i] = yR;
      for (let ch = 2; ch < outChannels; ch++) output[ch][i] = yL;
    }

    this._fast = fast;
    this._slow = slow;
    this._peak = peak;

    return true;
  }
}

registerProcessor('r13-enveloper-processor', EnveloperProcessor);
