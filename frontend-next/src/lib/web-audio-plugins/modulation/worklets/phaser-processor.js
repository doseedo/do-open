/**
 * Phaser AudioWorklet Processor
 * Creates phase-shifting effect using cascaded allpass filters
 *
 * This is the modern AudioWorklet version for high-performance offline and real-time processing.
 * Runs on a separate audio thread for better performance.
 *
 * @author Agent 4 (Modulation Plugins)
 * @version 1.0.0
 */

importScripts('dsp-utils.js');

class PhaserProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.rate = 0.5;        // LFO rate in Hz
    this.depth = 1.0;       // Modulation depth (0-1)
    this.feedback = 0.5;    // Feedback amount (0-1)
    this.stages = 4;        // Number of allpass stages (2, 4, 6, 8)
    this.mix = 0.5;         // Dry/wet mix (0-1)

    // Will be initialized after receiving sample rate
    this.allpassFilters = null;
    this.lfo = null;
    this.feedbackSample = [0, 0]; // Per channel
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

    // Create allpass filters for each channel
    // Max 8 stages, initialize all
    this.allpassFilters = [
      [], // Channel 0
      []  // Channel 1
    ];

    for (let ch = 0; ch < 2; ch++) {
      for (let i = 0; i < 8; i++) {
        this.allpassFilters[ch].push({
          z1: 0
        });
      }
    }

    // Single LFO for phaser effect
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
    if (params.stages !== undefined) {
      this.stages = Math.max(2, Math.min(8, Math.floor(params.stages)));
      // Ensure even number of stages
      if (this.stages % 2 !== 0) this.stages++;
    }
    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(1, params.mix));
    }
  }

  /**
   * Process a single sample through the phaser effect
   */
  processSample(sample, channel, sampleRate) {
    if (!this.initialized) {
      this.initializeState(sampleRate);
    }

    // LFO modulates allpass frequency
    const modulation = this.lfo.process(sampleRate);
    const centerFreq = 440 + modulation * 2000; // 440Hz ± 2kHz
    const coefficient = (Math.tan(Math.PI * centerFreq / sampleRate) - 1) /
                       (Math.tan(Math.PI * centerFreq / sampleRate) + 1);

    // Input with feedback
    let output = sample + this.feedbackSample[channel] * this.feedback;

    // Process through allpass cascade
    const filters = this.allpassFilters[channel];
    for (let i = 0; i < this.stages; i++) {
      const filter = filters[i];
      const temp = output;
      output = filter.z1 + temp * -coefficient;
      filter.z1 = temp + output * coefficient;
    }

    // Store for feedback
    this.feedbackSample[channel] = output;

    // Mix dry and wet
    return sample * (1 - this.mix) + output * this.mix;
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

    const numChannels = Math.min(input.length, output.length, 2);

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
registerProcessor('phaser-processor', PhaserProcessor);
