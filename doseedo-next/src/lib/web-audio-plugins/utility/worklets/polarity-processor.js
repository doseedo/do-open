/**
 * Polarity AudioWorklet Processor
 * Phase inversion per channel
 *
 * Allows independent phase inversion for left and right channels
 * Useful for fixing phase issues and creating special effects
 */

class PolarityProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.invertLeft = false;
    this.invertRight = false;

    // Listen for parameter changes from main thread
    this.port.onmessage = (event) => {
      const { type, params } = event.data;
      if (type === 'setParams') {
        this.updateParams(params);
      }
    };
  }

  /**
   * Update parameters from main thread
   */
  updateParams(params) {
    if (params.invertLeft !== undefined) {
      this.invertLeft = Boolean(params.invertLeft);
    }
    if (params.invertRight !== undefined) {
      this.invertRight = Boolean(params.invertRight);
    }
  }

  /**
   * Process audio sample with polarity inversion
   * @param {number} sample - Input sample
   * @param {number} channel - Channel index (0 = left, 1 = right)
   * @returns {number} Processed sample
   */
  processSample(sample, channel) {
    if (channel === 0 && this.invertLeft) {
      return -sample;
    }
    if (channel === 1 && this.invertRight) {
      return -sample;
    }
    return sample;
  }

  /**
   * Process audio (called by Web Audio API)
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || input.length === 0) {
      return true;
    }

    // Process each channel
    for (let channel = 0; channel < input.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        outputChannel[i] = this.processSample(inputChannel[i], channel);
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('polarity-processor', PolarityProcessor);
