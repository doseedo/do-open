/**
 * Tremolo AudioWorklet Processor
 * Creates amplitude modulation effect
 *
 * This is the modern AudioWorklet version for high-performance offline and real-time processing.
 * Runs on a separate audio thread for better performance.
 *
 * @author Agent 4 (Modulation Plugins)
 * @version 1.0.0
 */

importScripts('dsp-utils.js');

class TremoloProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.rate = 5.0;         // LFO rate in Hz
    this.depth = 0.5;        // Modulation depth (0-1)
    this.waveform = 'sine';  // LFO waveform type

    // Will be initialized after receiving sample rate
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

    // Single LFO for tremolo effect
    this.lfo = new LFO(this.rate, this.depth, this.waveform);

    this.initialized = true;
  }

  /**
   * Update parameters from main thread
   */
  updateParams(params) {
    if (params.rate !== undefined) {
      this.rate = Math.max(0.1, Math.min(20, params.rate));
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
    if (params.waveform !== undefined) {
      this.waveform = params.waveform;
      if (this.lfo) {
        this.lfo.setWaveform(this.waveform);
      }
    }
  }

  /**
   * Process a single sample through the tremolo effect
   */
  processSample(sample, channel, sampleRate) {
    if (!this.initialized) {
      this.initializeState(sampleRate);
    }

    // LFO modulates amplitude
    // Note: Only process LFO once per sample (not per channel)
    // to keep stereo channels in sync
    const modulation = this.lfo.process(sampleRate);

    // Map -1 to +1 range to 1-depth to 1+depth range
    const gain = 1 + modulation * this.depth;

    return sample * gain;
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

    // Process LFO once per sample to keep channels synchronized
    for (let i = 0; i < input[0].length; i++) {
      if (!this.initialized) {
        this.initializeState(sampleRate);
      }

      // Get LFO modulation for this sample
      const modulation = this.lfo.process(sampleRate);
      const gain = 1 + modulation * this.depth;

      // Apply same modulation to all channels
      for (let channel = 0; channel < numChannels; channel++) {
        const inputChannel = input[channel];
        const outputChannel = output[channel];
        outputChannel[i] = inputChannel[i] * gain;
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('tremolo-processor', TremoloProcessor);
