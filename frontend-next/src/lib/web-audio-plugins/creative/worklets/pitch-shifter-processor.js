/**
 * Pitch Shifter AudioWorklet Processor
 * Time-domain pitch shifting using overlap-add method
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class PitchShifterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.pitchShift = 0; // semitones (-12 to +12)
    this.mix = 1.0; // 0-1
    this.windowSize = 0.1 * sampleRate; // 100ms windows

    // Initialize state
    this.initializeState();

    // Listen for parameter changes
    this.port.onmessage = (event) => {
      const { type, value } = event.data;
      this.handleParameterUpdate(type, value);
    };
  }

  /**
   * Initialize processing state
   */
  initializeState() {
    // Two delay lines for crossfading windows
    this.delayLine1 = new DelayLine(1.0, sampleRate);
    this.delayLine2 = new DelayLine(1.0, sampleRate);

    // Window read positions
    this.readPos1 = 0;
    this.readPos2 = this.windowSize / 2;
    this.writePos = 0;
  }

  /**
   * Handle parameter updates
   */
  handleParameterUpdate(param, value) {
    switch (param) {
      case 'pitchShift':
        this.pitchShift = Math.max(-12, Math.min(12, value));
        break;
      case 'mix':
        this.mix = Math.max(0, Math.min(1, value));
        break;
      case 'windowSize':
        this.windowSize = Math.max(0.05, Math.min(0.2, value)) * sampleRate;
        break;
    }
  }

  /**
   * Calculate window function (Hann window)
   */
  getWindowValue(phase) {
    return 0.5 - 0.5 * Math.cos(phase * 2 * Math.PI);
  }

  /**
   * Process a single sample
   */
  processSample(sample, channel) {
    // Write input to delay lines
    this.delayLine1.write(sample);
    this.delayLine2.write(sample);

    // Calculate playback rate from pitch shift (semitones)
    const ratio = Math.pow(2, this.pitchShift / 12);

    // Read from both windows with time-stretching
    const sample1 = this.delayLine1.readInterpolated(this.readPos1);
    const sample2 = this.delayLine2.readInterpolated(this.readPos2);

    // Calculate crossfade weights using Hann window
    const phase1 = this.readPos1 / this.windowSize;
    const phase2 = this.readPos2 / this.windowSize;

    const window1 = this.getWindowValue(phase1);
    const window2 = this.getWindowValue(phase2);

    // Crossfade between windows
    const shifted = sample1 * window1 + sample2 * window2;

    // Advance read positions at modified rate
    this.readPos1 += ratio;
    this.readPos2 += ratio;

    // Wrap read positions
    if (this.readPos1 >= this.windowSize) {
      this.readPos1 = 0;
    }
    if (this.readPos2 >= this.windowSize) {
      this.readPos2 = 0;
    }

    this.writePos++;

    // Mix dry and wet
    return sample * (1 - this.mix) + shifted * this.mix;
  }

  /**
   * Process audio
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0]) {
      return true;
    }

    for (let channel = 0; channel < output.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        outputChannel[i] = this.processSample(inputChannel[i], channel);
      }
    }

    return true;
  }
}

registerProcessor('pitch-shifter-processor', PitchShifterProcessor);
