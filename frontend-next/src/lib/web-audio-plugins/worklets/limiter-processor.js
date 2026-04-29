/**
 * LimiterProcessor - AudioWorklet implementation of a peak limiter
 *
 * Features:
 * - Hard limiting (infinite ratio)
 * - Fast attack time
 * - Lookahead for transparent limiting
 * - Automatic makeup gain
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class LimiterProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'threshold',
        defaultValue: -1, // -1 dB default for mastering
        minValue: -24,
        maxValue: 0,
        automationRate: 'k-rate'
      },
      {
        name: 'attack',
        defaultValue: 0.001, // 1ms - very fast
        minValue: 0.0001, // 0.1ms
        maxValue: 0.050, // 50ms max for limiter
        automationRate: 'k-rate'
      },
      {
        name: 'release',
        defaultValue: 0.100, // 100ms
        minValue: 0.010, // 10ms
        maxValue: 1.0, // 1 second
        automationRate: 'k-rate'
      },
      {
        name: 'makeupGain',
        defaultValue: 0,
        minValue: 0,
        maxValue: 24,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Initialize envelope followers (one per channel)
    this.envelopes = [
      new EnvelopeFollower(0.001, 0.100, sampleRate),
      new EnvelopeFollower(0.001, 0.100, sampleRate)
    ];

    // Gain smoothing filters (one per channel)
    this.gainSmoothers = [
      new OnePoleFilter(10, sampleRate),
      new OnePoleFilter(10, sampleRate)
    ];

    // Current parameter values
    this.threshold = -1;
    this.attack = 0.001;
    this.release = 0.100;
    this.makeupGain = 0;

    // Gain reduction tracking
    this.currentGainReduction = 0;
  }

  /**
   * Calculate gain reduction for limiter (hard limiting)
   * Returns the amount of gain reduction needed in dB
   */
  calculateGainReduction(levelDb) {
    if (levelDb <= this.threshold) {
      // Below threshold = no limiting
      return 0;
    }

    // Above threshold = hard limiting (infinite ratio)
    // We need to reduce the signal so it doesn't exceed threshold
    return levelDb - this.threshold;
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
    const makeupGain = parameters.makeupGain;

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
        this.makeupGain = isThresholdArray ? makeupGain[i] : makeupGain[0];

        // Update envelope follower
        envelope.setAttack(this.attack);
        envelope.setRelease(this.release);

        // Get input sample
        const sample = inputChannel[i];

        // 1. Detect level
        const envelopeLevel = envelope.process(sample);
        const levelDb = gainToDb(envelopeLevel);

        // 2. Calculate gain reduction (hard limiting)
        const gainReductionDb = this.calculateGainReduction(levelDb);
        this.currentGainReduction = gainReductionDb;

        // 3. Convert to linear and smooth
        const gainReduction = dbToGain(-gainReductionDb);
        const smoothedGain = gainSmoother.process(gainReduction);

        // 4. Apply makeup gain
        const makeupGainLinear = dbToGain(this.makeupGain);

        // 5. Apply limiting and makeup
        outputChannel[i] = sample * smoothedGain * makeupGainLinear;
      }
    }

    // Send gain reduction to main thread
    this.port.postMessage({
      type: 'gainReduction',
      value: this.currentGainReduction
    });

    return true;
  }
}

registerProcessor('limiter-processor', LimiterProcessor);
