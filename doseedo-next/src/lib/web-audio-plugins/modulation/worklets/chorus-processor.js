/**
 * Chorus AudioWorklet Processor
 * Creates the illusion of multiple voices/instruments by layering slightly detuned delays
 *
 * This is the modern AudioWorklet version for high-performance offline and real-time processing.
 * Runs on a separate audio thread for better performance.
 *
 * @author Agent 4 (Modulation Plugins)
 * @version 1.0.0
 */

importScripts('dsp-utils.js');

class ChorusProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.rate = 0.5;        // LFO rate in Hz
    this.depth = 0.5;       // Modulation depth (0-1)
    this.delay = 0.020;     // Base delay (20ms)
    this.voices = 2;        // Number of chorus voices
    this.mix = 0.5;         // Dry/wet mix (0-1)

    // Will be initialized after receiving sample rate
    this.delayLines = null;
    this.lfos = null;
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
      new DelayLine(0.1, sampleRate),
      new DelayLine(0.1, sampleRate)
    ];

    // Multiple LFOs for multiple voices
    this.lfos = [];
    for (let i = 0; i < 8; i++) { // Max 8 voices
      const lfo = new LFO(this.rate, this.depth);
      // Phase offset for richness
      lfo.phase = i / 8;
      this.lfos.push(lfo);
    }

    this.initialized = true;
  }

  /**
   * Update parameters from main thread
   */
  updateParams(params) {
    if (params.rate !== undefined) {
      this.rate = Math.max(0.01, Math.min(10, params.rate));
      if (this.lfos) {
        this.lfos.forEach(lfo => lfo.setRate(this.rate));
      }
    }
    if (params.depth !== undefined) {
      this.depth = Math.max(0, Math.min(1, params.depth));
      if (this.lfos) {
        this.lfos.forEach(lfo => lfo.setDepth(this.depth));
      }
    }
    if (params.delay !== undefined) {
      this.delay = Math.max(0.005, Math.min(0.050, params.delay));
    }
    if (params.voices !== undefined) {
      this.voices = Math.max(1, Math.min(8, Math.floor(params.voices)));
    }
    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(1, params.mix));
    }
  }

  /**
   * Process a single sample through the chorus effect
   */
  processSample(sample, channel, sampleRate) {
    if (!this.initialized) {
      this.initializeState(sampleRate);
    }

    const delayLine = this.delayLines[channel];

    // Write input to delay line
    delayLine.write(sample);

    // Sum all modulated delay taps (voices)
    let chorus = 0;
    for (let i = 0; i < this.voices; i++) {
      const lfo = this.lfos[i];
      const modulation = lfo.process(sampleRate);
      // ±10ms modulation
      const modulatedDelay = this.delay + modulation * 0.010;
      const delaySamples = modulatedDelay * sampleRate;
      chorus += delayLine.readInterpolated(delaySamples);
    }

    // Normalize by number of voices
    chorus /= this.voices;

    // Mix dry and wet
    return sample * (1 - this.mix) + chorus * this.mix;
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
registerProcessor('chorus-processor', ChorusProcessor);
