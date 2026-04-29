/**
 * Gain AudioWorklet Processor
 * Simple gain/volume control processor
 *
 * Provides precise gain control with both linear and dB modes
 * Runs on a separate audio thread for optimal performance
 */

class GainProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.gain = 1.0; // Linear gain
    this.gainDb = 0; // dB gain

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
    if (params.gain !== undefined) {
      this.gain = Math.max(0, params.gain); // No upper limit on linear gain
    }
    if (params.gainDb !== undefined) {
      this.gainDb = params.gainDb;
      // Convert dB to linear gain
      this.gain = this.dbToGain(this.gainDb);
    }
  }

  /**
   * Convert dB to linear gain
   */
  dbToGain(db) {
    if (db === -Infinity || db < -100) return 0;
    return Math.pow(10, db / 20);
  }

  /**
   * Process audio samples
   */
  processSample(sample) {
    return sample * this.gain;
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
        outputChannel[i] = this.processSample(inputChannel[i]);
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('gain-processor', GainProcessor);
