/**
 * Echo AudioWorklet Processor
 * Multi-tap echo effect with rhythmic delays
 *
 * Features:
 * - Multiple delay taps with independent spacing
 * - Exponential decay for each tap
 * - Feedback loop for sustained echoes
 * - Wet/dry mix control
 * - Stereo processing
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class EchoProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.sampleRate = sampleRate;

    // Initialize parameters
    this.initializeParameters({
      baseDelay: 0.25,    // Base delay time in seconds
      numTaps: 4,         // Number of echo taps
      feedback: 0.3,      // Overall feedback (0-1)
      tapDecay: 0.7,      // Gain reduction per tap (0-1)
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
    this.baseDelay = params.baseDelay || 0.25;
    this.numTaps = params.numTaps || 4;
    this.feedback = params.feedback || 0.3;
    this.tapDecay = params.tapDecay || 0.7;
    this.mix = params.mix || 0.5;
  }

  /**
   * Initialize processing state (delay lines, filters, etc.)
   */
  initializeState() {
    // Create delay lines for stereo
    this.delayLines = [
      new DelayLine(5.0, this.sampleRate),  // Left - max 5 seconds
      new DelayLine(5.0, this.sampleRate)   // Right - max 5 seconds
    ];
  }

  /**
   * Handle parameter updates from main thread
   * @param {Object} params - Updated parameters
   */
  handleParameterUpdate(params) {
    if (params.baseDelay !== undefined) {
      this.baseDelay = Math.max(0.001, Math.min(5.0, params.baseDelay));
    }

    if (params.numTaps !== undefined) {
      this.numTaps = Math.max(1, Math.min(16, Math.floor(params.numTaps)));
    }

    if (params.feedback !== undefined) {
      this.feedback = Math.max(0, Math.min(0.95, params.feedback));
    }

    if (params.tapDecay !== undefined) {
      this.tapDecay = Math.max(0, Math.min(1.0, params.tapDecay));
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
    const delayLine = this.delayLines[channel];

    // Read multiple taps from the delay line
    let wetSignal = 0;

    for (let tap = 1; tap <= this.numTaps; tap++) {
      // Calculate delay time for this tap
      const tapDelay = this.baseDelay * tap;
      const tapDelaySamples = tapDelay * this.sampleRate;

      // Calculate gain for this tap (exponential decay)
      const tapGain = Math.pow(this.tapDecay, tap - 1);

      // Read from delay line with interpolation
      const tapSample = delayLine.readInterpolated(tapDelaySamples);

      // Accumulate tap
      wetSignal += tapSample * tapGain;
    }

    // Normalize by number of taps to maintain consistent level
    wetSignal /= Math.max(1, this.numTaps);

    // Write input + feedback to delay line
    delayLine.write(sample + wetSignal * this.feedback);

    // Mix dry and wet signals
    return sample * (1 - this.mix) + wetSignal * this.mix;
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

    // Process each channel
    for (let channel = 0; channel < output.length; channel++) {
      const inputChannel = input[channel] || input[0]; // Fallback to mono
      const outputChannel = output[channel];

      // Process each sample in the block
      for (let i = 0; i < outputChannel.length; i++) {
        outputChannel[i] = this.processSample(inputChannel[i], channel);
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('echo-processor', EchoProcessor);
