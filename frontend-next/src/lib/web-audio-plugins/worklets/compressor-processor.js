/**
 * CompressorProcessor - AudioWorklet implementation of a dynamics compressor
 *
 * Features:
 * - Envelope follower for level detection
 * - Soft knee compression
 * - Configurable attack/release
 * - Makeup gain
 * - Wet/dry mix for parallel compression
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class CompressorProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'threshold',
        defaultValue: -24,
        minValue: -60,
        maxValue: 0,
        automationRate: 'k-rate'
      },
      {
        name: 'ratio',
        defaultValue: 4,
        minValue: 1,
        maxValue: 20,
        automationRate: 'k-rate'
      },
      {
        name: 'attack',
        defaultValue: 0.010, // 10ms
        minValue: 0.0001, // 0.1ms
        maxValue: 0.5, // 500ms
        automationRate: 'k-rate'
      },
      {
        name: 'release',
        defaultValue: 0.100, // 100ms
        minValue: 0.001, // 1ms
        maxValue: 2.0, // 2 seconds
        automationRate: 'k-rate'
      },
      {
        name: 'knee',
        defaultValue: 0,
        minValue: 0,
        maxValue: 12,
        automationRate: 'k-rate'
      },
      {
        name: 'makeupGain',
        defaultValue: 0,
        minValue: 0,
        maxValue: 24,
        automationRate: 'k-rate'
      },
      {
        name: 'mix',
        defaultValue: 1.0, // 100% wet
        minValue: 0,
        maxValue: 1.0,
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
    this.threshold = -24;
    this.ratio = 4;
    this.attack = 0.010;
    this.release = 0.100;
    this.knee = 0;
    this.makeupGain = 0;
    this.mix = 1.0;

    // Gain reduction tracking (for metering)
    this.currentGainReduction = 0;
  }

  /**
   * Calculate gain reduction in dB based on input level
   * Implements soft-knee compression characteristic
   */
  calculateGainReduction(levelDb) {
    if (levelDb <= this.threshold - this.knee / 2) {
      // Below threshold - no compression
      return 0;
    }

    if (this.knee > 0 && levelDb < this.threshold + this.knee / 2) {
      // Soft knee region - gradual compression
      const delta = levelDb - this.threshold;
      const slope = 1 / this.ratio - 1;
      return slope * delta * delta / (2 * this.knee);
    }

    // Above knee - full compression
    return (levelDb - this.threshold) * (1 - 1 / this.ratio);
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

    // Get parameter values (k-rate or a-rate)
    const threshold = parameters.threshold;
    const ratio = parameters.ratio;
    const attack = parameters.attack;
    const release = parameters.release;
    const knee = parameters.knee;
    const makeupGain = parameters.makeupGain;
    const mix = parameters.mix;

    // Check if parameters are arrays (a-rate) or single values (k-rate)
    const isThresholdArray = threshold.length > 1;

    const blockSize = input[0].length;

    // Process each channel
    for (let channel = 0; channel < input.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];
      const envelope = this.envelopes[channel];
      const gainSmoother = this.gainSmoothers[channel];

      for (let i = 0; i < blockSize; i++) {
        // Update parameters if they've changed
        this.threshold = isThresholdArray ? threshold[i] : threshold[0];
        this.ratio = isThresholdArray ? ratio[i] : ratio[0];
        this.attack = isThresholdArray ? attack[i] : attack[0];
        this.release = isThresholdArray ? release[i] : release[0];
        this.knee = isThresholdArray ? knee[i] : knee[0];
        this.makeupGain = isThresholdArray ? makeupGain[i] : makeupGain[0];
        this.mix = isThresholdArray ? mix[i] : mix[0];

        // Update envelope follower time constants if changed
        envelope.setAttack(this.attack);
        envelope.setRelease(this.release);

        // Get input sample
        const sample = inputChannel[i];

        // 1. Detect level using envelope follower
        const envelopeLevel = envelope.process(sample);
        const levelDb = gainToDb(envelopeLevel);

        // 2. Calculate gain reduction
        const gainReductionDb = this.calculateGainReduction(levelDb);
        this.currentGainReduction = gainReductionDb;

        // 3. Convert to linear gain and smooth
        const gainReduction = dbToGain(-gainReductionDb);
        const smoothedGain = gainSmoother.process(gainReduction);

        // 4. Apply makeup gain
        const makeupGainLinear = dbToGain(this.makeupGain);

        // 5. Apply gain reduction and makeup
        const compressed = sample * smoothedGain * makeupGainLinear;

        // 6. Mix dry and wet signals
        outputChannel[i] = sample * (1 - this.mix) + compressed * this.mix;
      }
    }

    // Send gain reduction to main thread for metering
    this.port.postMessage({
      type: 'gainReduction',
      value: this.currentGainReduction
    });

    return true;
  }
}

registerProcessor('compressor-processor', CompressorProcessor);
