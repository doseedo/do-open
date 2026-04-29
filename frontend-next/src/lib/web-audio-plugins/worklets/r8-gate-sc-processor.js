/**
 * r8-gate-sc-processor.js
 *
 * SIDECHAIN-CAPABLE noise gate variant. Created by Agent R8.
 *
 * Why this exists (vs editing gate-processor.js in-place):
 *   Stock gate-processor.js detects level on inputs[0] only. Logic's Noise
 *   Gate has a "Side Chain" key input — and the entire Enveloper effect
 *   relies on a key signal driving the gating envelope of a different
 *   audio source. A non-destructive new processor was added so existing
 *   gates referencing 'gate-processor' keep working.
 *
 *   inputs[0] = main audio (the signal being gated)
 *   inputs[1] = key input (the detector signal)
 *
 *   When `sidechain_active` < 0.5 OR inputs[1] is silent/missing, this
 *   processor behaves identically to the stock gate-processor.
 *
 * Registered name: 'r8-gate-sc-processor'
 *
 * @author Agent R8 — Sidechain Routing
 * @version 1.0.0
 */

importScripts('../core/dsp-utils.js');

class R8GateSCProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'threshold', defaultValue: -40, minValue: -60, maxValue: 0,    automationRate: 'k-rate' },
      { name: 'attack',    defaultValue: 0.010, minValue: 0.0001, maxValue: 0.100, automationRate: 'k-rate' },
      { name: 'release',   defaultValue: 0.100, minValue: 0.010, maxValue: 2.0,  automationRate: 'k-rate' },
      { name: 'range',     defaultValue: -60, minValue: -80, maxValue: 0,    automationRate: 'k-rate' },
      { name: 'sidechain_active', defaultValue: 0, minValue: 0, maxValue: 1, automationRate: 'k-rate' }
    ];
  }

  constructor(options) {
    super();

    this.envelopes = [
      new EnvelopeFollower(0.010, 0.100, sampleRate),
      new EnvelopeFollower(0.010, 0.100, sampleRate)
    ];
    this.gainSmoothers = [
      new OnePoleFilter(10, sampleRate),
      new OnePoleFilter(10, sampleRate)
    ];

    this.threshold = -40;
    this.attack = 0.010;
    this.release = 0.100;
    this.range = -60;
    this.isGateOpen = false;
  }

  calculateGainReduction(levelDb) {
    if (levelDb >= this.threshold) {
      this.isGateOpen = true;
      return 0;
    }
    this.isGateOpen = false;
    return -this.range; // range is negative; negate for attenuation magnitude
  }

  process(inputs, outputs, parameters) {
    const audio = inputs[0];
    const sidechain = inputs[1];
    const output = outputs[0];

    if (!audio || !audio.length) return true;

    const threshold = parameters.threshold;
    const attack = parameters.attack;
    const release = parameters.release;
    const range = parameters.range;
    const sidechainActive = parameters.sidechain_active;

    const isThresholdArray = threshold.length > 1;
    const blockSize = audio[0].length;

    const scLive =
      sidechainActive[0] > 0.5 &&
      sidechain && sidechain.length > 0 && sidechain[0] && sidechain[0].length > 0;

    for (let channel = 0; channel < audio.length; channel++) {
      const inputChannel = audio[channel];
      const outputChannel = output[channel];
      const envelope = this.envelopes[channel];
      const gainSmoother = this.gainSmoothers[channel];

      let detectorChannel = inputChannel;
      if (scLive) {
        detectorChannel = sidechain[channel] || sidechain[0] || inputChannel;
      }

      for (let i = 0; i < blockSize; i++) {
        this.threshold = isThresholdArray ? threshold[i] : threshold[0];
        this.attack    = isThresholdArray ? attack[i]    : attack[0];
        this.release   = isThresholdArray ? release[i]   : release[0];
        this.range     = isThresholdArray ? range[i]     : range[0];

        envelope.setAttack(this.attack);
        envelope.setRelease(this.release);

        const audioSample = inputChannel[i];
        const detectorSample = detectorChannel[i];

        // Detection on key signal
        const envelopeLevel = envelope.process(detectorSample);
        const levelDb = gainToDb(envelopeLevel);

        // Gain reduction from key
        const gainReductionDb = this.calculateGainReduction(levelDb);
        const gainReduction = dbToGain(-gainReductionDb);
        const smoothedGain = gainSmoother.process(gainReduction);

        // Applied to audio
        outputChannel[i] = audioSample * smoothedGain;
      }
    }

    this.port.postMessage({
      type: 'gateState',
      value: this.isGateOpen,
      sidechainActive: scLive
    });

    return true;
  }
}

registerProcessor('r8-gate-sc-processor', R8GateSCProcessor);
