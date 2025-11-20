/**
 * Pan AudioWorklet Processor
 * Stereo panning with constant power curve
 *
 * Provides smooth panning from left to right using constant power panning
 * to maintain perceived loudness across the stereo field
 */

class PanProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.pan = 0; // -1 (left) to +1 (right)

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
    if (params.pan !== undefined) {
      // Clamp to -1 to +1
      this.pan = Math.max(-1, Math.min(1, params.pan));
    }
  }

  /**
   * Process audio sample with constant power panning
   * @param {number} sample - Input sample
   * @param {number} channel - Channel index (0 = left, 1 = right)
   * @returns {number} Panned sample
   */
  processSample(sample, channel) {
    // Constant power panning (uses ±45 degree rotation)
    // This maintains constant perceived loudness across the stereo field
    const panRadians = this.pan * Math.PI / 4; // ±45 degrees

    if (channel === 0) { // Left channel
      // Left gain decreases as we pan right
      return sample * Math.cos(panRadians);
    } else { // Right channel
      // Right gain increases as we pan right
      return sample * Math.sin(panRadians);
    }
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

    // Ensure we have stereo output
    if (output.length < 2) {
      return true;
    }

    // Get input channels (mono input will be duplicated)
    const inputL = input[0];
    const inputR = input.length > 1 ? input[1] : input[0];

    // Get output channels
    const outputL = output[0];
    const outputR = output[1];

    // Process each sample
    for (let i = 0; i < inputL.length; i++) {
      // Average both input channels for panning
      const monoSample = (inputL[i] + inputR[i]) / 2;

      // Apply panning to both output channels
      outputL[i] = this.processSample(monoSample, 0);
      outputR[i] = this.processSample(monoSample, 1);
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('pan-processor', PanProcessor);
