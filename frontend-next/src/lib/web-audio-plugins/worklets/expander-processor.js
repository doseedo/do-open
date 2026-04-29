/**
 * ExpanderProcessor - AudioWorklet implementation of a downward expander
 *
 * Features:
 * - Downward expansion (increases dynamic range)
 * - Configurable ratio
 * - Attack/release times
 * - More subtle than a gate
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class ExpanderProcessor extends AudioWorkletProcessor {
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
        name: 'ratio',
        defaultValue: 2, // 1:2 expansion
        minValue: 1,
        maxValue: 10,
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
    this.ratio = 2;
    this.attack = 0.010;
    this.release = 0.100;

    // Expansion tracking
    this.currentExpansion = 0;
  }

  /**
   * Calculate gain reduction for expander
   * Below threshold: expand downward (make quieter signals even quieter)
   * Above threshold: no change
   */
  calculateGainReduction(levelDb) {
    if (levelDb >= this.threshold) {
      // Above threshold = no expansion
      return 0;
    }

    // Below threshold = expand downward
    // Expansion formula: gain = (levelDb - threshold) * (1 - ratio)
    // For example, with ratio = 2, a signal 10 dB below threshold gets attenuated by an additional 10 dB
    return (levelDb - this.threshold) * (1 - this.ratio);
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
    const ratio = parameters.ratio;
    const attack = parameters.attack;
    const release = parameters.release;

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
        this.ratio = isThresholdArray ? ratio[i] : ratio[0];
        this.attack = isThresholdArray ? attack[i] : attack[0];
        this.release = isThresholdArray ? release[i] : release[0];

        // Update envelope follower
        envelope.setAttack(this.attack);
        envelope.setRelease(this.release);

        // Get input sample
        const sample = inputChannel[i];

        // 1. Detect level
        const envelopeLevel = envelope.process(sample);
        const levelDb = gainToDb(envelopeLevel);

        // 2. Calculate gain reduction (expansion)
        const gainReductionDb = this.calculateGainReduction(levelDb);
        this.currentExpansion = gainReductionDb;

        // 3. Convert to linear and smooth
        const gainReduction = dbToGain(-gainReductionDb);
        const smoothedGain = gainSmoother.process(gainReduction);

        // 4. Apply expansion
        outputChannel[i] = sample * smoothedGain;
      }
    }

    // Send expansion amount to main thread
    this.port.postMessage({
      type: 'expansion',
      value: this.currentExpansion
    });

    return true;
  }
}

registerProcessor('expander-processor', ExpanderProcessor);
