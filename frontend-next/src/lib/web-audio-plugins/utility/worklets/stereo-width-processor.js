/**
 * Stereo Width AudioWorklet Processor
 * Mid/Side stereo width control
 *
 * Provides stereo width control from mono (0%) to extra wide (200%)
 * Uses Mid/Side processing for transparent stereo image adjustment
 */

class StereoWidthProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.width = 1.0; // 0 = mono, 1 = normal stereo, 2 = extra wide

    // State buffers for M/S processing
    this.leftBuffer = 0;
    this.rightBuffer = 0;
    this.leftOutput = 0;
    this.rightOutput = 0;

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
    if (params.width !== undefined) {
      // Clamp to 0 to 2 (0% to 200%)
      this.width = Math.max(0, Math.min(2, params.width));
    }
  }

  /**
   * Process audio sample with Mid/Side width adjustment
   * @param {number} sample - Input sample
   * @param {number} channel - Channel index (0 = left, 1 = right)
   * @returns {number} Processed sample
   */
  processSample(sample, channel) {
    if (channel === 0) { // Left channel
      this.leftBuffer = sample;
    } else { // Right channel
      this.rightBuffer = sample;

      // Convert to Mid/Side
      const mid = (this.leftBuffer + this.rightBuffer) * 0.5;
      const side = (this.leftBuffer - this.rightBuffer) * 0.5;

      // Adjust width by scaling the side signal
      const wideSide = side * this.width;

      // Convert back to Left/Right
      this.leftOutput = mid + wideSide;
      this.rightOutput = mid - wideSide;
    }

    return channel === 0 ? this.leftOutput : this.rightOutput;
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

    // Ensure we have stereo input/output
    if (input.length < 2 || output.length < 2) {
      // If mono input, just pass through
      if (input.length === 1 && output.length >= 1) {
        output[0].set(input[0]);
        if (output.length >= 2) {
          output[1].set(input[0]);
        }
      }
      return true;
    }

    // Get input channels
    const inputL = input[0];
    const inputR = input[1];

    // Get output channels
    const outputL = output[0];
    const outputR = output[1];

    // Process each sample
    for (let i = 0; i < inputL.length; i++) {
      // Convert to Mid/Side
      const mid = (inputL[i] + inputR[i]) * 0.5;
      const side = (inputL[i] - inputR[i]) * 0.5;

      // Adjust width by scaling the side signal
      const wideSide = side * this.width;

      // Convert back to Left/Right
      outputL[i] = mid + wideSide;
      outputR[i] = mid - wideSide;
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('stereo-width-processor', StereoWidthProcessor);
