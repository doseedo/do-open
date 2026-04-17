/**
 * GateProcessor - AudioWorklet implementation of a noise gate
 *
 * Features:
 * - Noise gate for removing low-level signals
 * - Configurable threshold
 * - Attack/release/hold times
 * - Range parameter (how much to attenuate)
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class GateProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'threshold',
        defaultValue: -40, // -40 dB default
        minValue: -60,
        maxValue: 0,
        automationRate: 'k-rate'
      },
      {
        name: 'attack',
        defaultValue: 0.010, // 10ms
        minValue: 0.0001, // 0.1ms
        maxValue: 0.100, // 100ms
        automationRate: 'k-rate'
      },
      {
        name: 'release',
        defaultValue: 0.100, // 100ms
        minValue: 0.010, // 10ms
        maxValue: 2.0, // 2 seconds
        automationRate: 'k-rate'
      },
      {
        name: 'range',
        defaultValue: -60, // Attenuate by 60 dB when gated
        minValue: -80,
        maxValue: 0,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Initialize envelope followers (one per channel)
    this.envelopes = [
      new EnvelopeFollower(0.010, 0.100, sampleRate),
      new EnvelopeFollower(0.010, 0.100, sampleRate)
    ];

    // Gain smoothing filters (one per channel)
    this.gainSmoothers = [
      new OnePoleFilter(10, sampleRate),
      new OnePoleFilter(10, sampleRate)
    ];

    // Current parameter values
    this.threshold = -40;
    this.attack = 0.010;
    this.release = 0.100;
    this.range = -60;

    // Gate state tracking
    this.isGateOpen = false;
  }

  /**
   * Calculate gain reduction for gate
   * Below threshold: attenuate by range amount
   * Above threshold: no attenuation
   */
  calculateGainReduction(levelDb) {
    if (levelDb >= this.threshold) {
      // Above threshold = gate open, no attenuation
      this.isGateOpen = true;
      return 0;
    }

    // Below threshold = gate closed, attenuate by range
    this.isGateOpen = false;
    return -this.range; // range is negative, so negate it for attenuation
  }

  /**
   * Process audio block
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input.length) {
      return true;
    }

    // Get parameter values
    const threshold = parameters.threshold;
    const attack = parameters.attack;
    const release = parameters.release;
    const range = parameters.range;

    const isThresholdArray = threshold.length > 1;
    const blockSize = input[0].length;

    // Process each channel
    for (let channel = 0; channel < input.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];
      const envelope = this.envelopes[channel];
      const gainSmoother = this.gainSmoothers[channel];

      for (let i = 0; i < blockSize; i++) {
        // Update parameters
        this.threshold = isThresholdArray ? threshold[i] : threshold[0];
        this.attack = isThresholdArray ? attack[i] : attack[0];
        this.release = isThresholdArray ? release[i] : release[0];
        this.range = isThresholdArray ? range[i] : range[0];

        // Update envelope follower
        envelope.setAttack(this.attack);
        envelope.setRelease(this.release);

        // Get input sample
        const sample = inputChannel[i];

        // 1. Detect level
        const envelopeLevel = envelope.process(sample);
        const levelDb = gainToDb(envelopeLevel);

        // 2. Calculate gain reduction
        const gainReductionDb = this.calculateGainReduction(levelDb);

        // 3. Convert to linear and smooth
        const gainReduction = dbToGain(-gainReductionDb);
        const smoothedGain = gainSmoother.process(gainReduction);

        // 4. Apply gating
        outputChannel[i] = sample * smoothedGain;
      }
    }

    // Send gate state to main thread
    this.port.postMessage({
      type: 'gateState',
      value: this.isGateOpen
    });

    return true;
  }
}

registerProcessor('gate-processor', GateProcessor);
