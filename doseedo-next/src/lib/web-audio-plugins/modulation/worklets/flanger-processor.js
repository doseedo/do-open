/**
 * Flanger AudioWorklet Processor
 * Creates sweeping comb filter effect with feedback
 *
 * This is the modern AudioWorklet version for high-performance offline and real-time processing.
 * Runs on a separate audio thread for better performance.
 *
 * @author Agent 4 (Modulation Plugins)
 * @version 1.0.0
 */

importScripts('dsp-utils.js');

class FlangerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.rate = 0.3;        // LFO rate in Hz
    this.depth = 1.0;       // Modulation depth (0-1)
    this.feedback = 0.5;    // Feedback amount (0-1)
    this.delay = 0.002;     // Base delay (2ms)
    this.mix = 0.5;         // Dry/wet mix (0-1)

    // Will be initialized after receiving sample rate
    this.delayLines = null;
    this.lfo = null;
    this.initialized = false;

    // Listen for parameter changes from main thread
    this.port.onmessage = (event) => {
      const { type, params } = event.data;
      if (type === 'setParams') {
        this.updateParams(params);
      } else if (type === 'init') {
        this.initializeState(params.sampleRate);
      }
    };
  }

  /**
   * Initialize processing state
   */
  initializeState(sampleRate) {
    if (this.initialized) return;

    // Delay lines for modulated delays (per channel)
    this.delayLines = [
      new DelayLine(0.05, sampleRate),
      new DelayLine(0.05, sampleRate)
    ];

    // Single LFO for flanger effect
    this.lfo = new LFO(this.rate, this.depth);

    this.initialized = true;
  }

  /**
   * Update parameters from main thread
   */
  updateParams(params) {
    if (params.rate !== undefined) {
      this.rate = Math.max(0.01, Math.min(10, params.rate));
      if (this.lfo) {
        this.lfo.setRate(this.rate);
      }
    }
    if (params.depth !== undefined) {
      this.depth = Math.max(0, Math.min(1, params.depth));
      if (this.lfo) {
        this.lfo.setDepth(this.depth);
      }
    }
    if (params.feedback !== undefined) {
      this.feedback = Math.max(0, Math.min(0.95, params.feedback));
    }
    if (params.delay !== undefined) {
      this.delay = Math.max(0.001, Math.min(0.010, params.delay));
    }
    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(1, params.mix));
    }
  }

  /**
   * Process a single sample through the flanger effect
   */
  processSample(sample, channel, sampleRate) {
    if (!this.initialized) {
      this.initializeState(sampleRate);
    }

    const delayLine = this.delayLines[channel];

    // Get modulated delay time
    const modulation = this.lfo.process(sampleRate);
    // ±5ms modulation
    const modulatedDelay = this.delay + modulation * 0.005;
    const delaySamples = modulatedDelay * sampleRate;

    // Read delayed signal
    const delayed = delayLine.readInterpolated(delaySamples);

    // Write with feedback (creates resonances)
    delayLine.write(sample + delayed * this.feedback);

    // Mix dry and wet
    return sample * (1 - this.mix) + delayed * this.mix;
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

    const numChannels = Math.min(input.length, output.length);

    for (let channel = 0; channel < numChannels; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        outputChannel[i] = this.processSample(
          inputChannel[i],
          channel,
          sampleRate
        );
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('flanger-processor', FlangerProcessor);
