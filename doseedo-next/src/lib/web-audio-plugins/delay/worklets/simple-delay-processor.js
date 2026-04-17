/**
 * SimpleDelay AudioWorklet Processor
 * Basic delay with feedback and damping filter
 *
 * Features:
 * - Variable delay time with smooth interpolation
 * - Feedback loop with damping filter
 * - Wet/dry mix control
 * - Stereo processing
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class SimpleDelayProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.sampleRate = sampleRate;

    // Initialize parameters
    this.initializeParameters({
      delayTime: 0.25,    // seconds
      feedback: 0.3,      // 0-1
      mix: 0.5,           // 0-1
      damping: 0.5        // 0-1 (controls feedback filter)
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
    this.delayTime = params.delayTime || 0.25;
    this.feedback = params.feedback || 0.3;
    this.mix = params.mix || 0.5;
    this.damping = params.damping || 0.5;
  }

  /**
   * Initialize processing state (delay lines, filters, etc.)
   */
  initializeState() {
    // Create delay lines for stereo (left and right channels)
    this.delayLines = [
      new DelayLine(5.0, this.sampleRate),  // Left - max 5 seconds
      new DelayLine(5.0, this.sampleRate)   // Right - max 5 seconds
    ];

    // Damping filters (lowpass in feedback path)
    this.dampingFilters = [
      new OnePoleFilter(this.damping * 10000, this.sampleRate, 'lowpass'),
      new OnePoleFilter(this.damping * 10000, this.sampleRate, 'lowpass')
    ];
  }

  /**
   * Handle parameter updates from main thread
   * @param {Object} params - Updated parameters
   */
  handleParameterUpdate(params) {
    if (params.delayTime !== undefined) {
      this.delayTime = Math.max(0, Math.min(5.0, params.delayTime));
    }

    if (params.feedback !== undefined) {
      this.feedback = Math.max(0, Math.min(0.95, params.feedback));
    }

    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(1.0, params.mix));
    }

    if (params.damping !== undefined) {
      this.damping = Math.max(0, Math.min(1.0, params.damping));
      // Update damping filter cutoff
      const cutoffFreq = this.damping * 10000 + 20; // 20Hz to 10kHz
      this.dampingFilters[0].setCutoff(cutoffFreq);
      this.dampingFilters[1].setCutoff(cutoffFreq);
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
    const dampingFilter = this.dampingFilters[channel];

    // Calculate delay in samples
    const delaySamples = this.delayTime * this.sampleRate;

    // Read delayed signal with interpolation
    const delayed = delayLine.readInterpolated(delaySamples);

    // Apply damping to feedback signal (lowpass filter)
    const dampedFeedback = dampingFilter.process(delayed);

    // Write to delay line with feedback
    delayLine.write(sample + dampedFeedback * this.feedback);

    // Mix dry and wet signals
    return sample * (1 - this.mix) + delayed * this.mix;
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
registerProcessor('simple-delay-processor', SimpleDelayProcessor);
