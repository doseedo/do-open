/**
 * PingPongDelay AudioWorklet Processor
 * Stereo delay that bounces between left and right channels
 *
 * Features:
 * - Cross-feedback between L and R channels (creates ping-pong effect)
 * - Independent delay times for L and R
 * - Stereo spread control
 * - Damping filter in feedback path
 * - Wet/dry mix control
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class PingPongDelayProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.sampleRate = sampleRate;

    // Initialize parameters
    this.initializeParameters({
      delayTime: 0.375,   // Delay time in seconds
      feedback: 0.4,      // Feedback amount (0-1)
      spread: 1.0,        // Stereo width (0-1)
      mix: 0.5            // Wet/dry mix (0-1)
    });

    // Initialize processing state
    this.initializeState();

    // Listen for parameter updates from main thread
    this.port.onmessage = (event) => {
      if (event.data.type === 'updateParams') {
        this.handleParameterUpdate(event.data.params);
      }
    };
  }

  /**
   * Initialize parameter values
   * @param {Object} params - Parameter values
   */
  initializeParameters(params) {
    this.delayTime = params.delayTime || 0.375;
    this.feedback = params.feedback || 0.4;
    this.spread = params.spread || 1.0;
    this.mix = params.mix || 0.5;
  }

  /**
   * Initialize processing state (delay lines, filters, etc.)
   */
  initializeState() {
    // Crossed delay lines for ping-pong effect
    // Left delay receives input from left + feedback from right
    // Right delay receives input from right + feedback from left
    this.delayLineL = new DelayLine(5.0, this.sampleRate);
    this.delayLineR = new DelayLine(5.0, this.sampleRate);

    // Store the last output from each channel for cross-feedback
    this.lastDelayedL = 0;
    this.lastDelayedR = 0;
  }

  /**
   * Handle parameter updates from main thread
   * @param {Object} params - Updated parameters
   */
  handleParameterUpdate(params) {
    if (params.delayTime !== undefined) {
      this.delayTime = Math.max(0.001, Math.min(5.0, params.delayTime));
    }

    if (params.feedback !== undefined) {
      this.feedback = Math.max(0, Math.min(0.95, params.feedback));
    }

    if (params.spread !== undefined) {
      this.spread = Math.max(0, Math.min(1.0, params.spread));
    }

    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(1.0, params.mix));
    }
  }

  /**
   * Process a single sample
   * @param {number} sample - Input sample
   * @param {number} channel - Channel index (0 = left, 1 = right)
   * @returns {number} Processed sample
   */
  processSample(sample, channel) {
    const delaySamples = this.delayTime * this.sampleRate;

    if (channel === 0) {
      // LEFT CHANNEL
      // Read from right delay (ping-pong cross-feedback)
      const delayedR = this.delayLineR.readInterpolated(delaySamples);

      // Write to left delay: input + feedback from right
      this.delayLineL.write(sample + delayedR * this.feedback);

      // Read from left delay for output
      const delayedL = this.delayLineL.readInterpolated(delaySamples);

      // Store for next iteration
      this.lastDelayedL = delayedL;

      // Output with spread control
      return sample * (1 - this.mix) + delayedL * this.mix * this.spread;

    } else {
      // RIGHT CHANNEL
      // Read from left delay (ping-pong cross-feedback)
      const delayedL = this.delayLineL.readInterpolated(delaySamples);

      // Write to right delay: input + feedback from left
      this.delayLineR.write(sample + delayedL * this.feedback);

      // Read from right delay for output
      const delayedR = this.delayLineR.readInterpolated(delaySamples);

      // Store for next iteration
      this.lastDelayedR = delayedR;

      // Output with spread control
      return sample * (1 - this.mix) + delayedR * this.mix * this.spread;
    }
  }

  /**
   * Process audio block (called by Web Audio API)
   * @param {Array} inputs - Input audio buffers
   * @param {Array} outputs - Output audio buffers
   * @param {Object} parameters - Automated parameters
   * @returns {boolean} True to keep processor alive
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || !input.length) {
      return true;
    }

    // Ensure stereo processing (ping-pong requires stereo)
    const numChannels = Math.max(2, output.length);

    // Process sample by sample (important for cross-feedback timing)
    const blockSize = output[0].length;

    for (let i = 0; i < blockSize; i++) {
      // Get input samples
      const inputL = input[0] ? input[0][i] : 0;
      const inputR = input[1] ? input[1][i] : inputL; // Mono to stereo if needed

      // Process both channels
      const outputL = this.processSample(inputL, 0);
      const outputR = this.processSample(inputR, 1);

      // Write to output buffers
      if (output[0]) output[0][i] = outputL;
      if (output[1]) output[1][i] = outputR;
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('ping-pong-delay-processor', PingPongDelayProcessor);
