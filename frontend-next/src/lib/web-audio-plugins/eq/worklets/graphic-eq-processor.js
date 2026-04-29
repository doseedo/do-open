/**
 * Graphic EQ Processor - 10-Band Graphic Equalizer
 * AudioWorklet processor for professional graphic EQ
 *
 * Features:
 * - 10 bands at standard ISO frequencies
 * - ±15dB gain per band
 * - Fixed Q for musical response
 * - Optimized for real-time and offline processing
 *
 * @author Agent 2 (EQ Plugins)
 */

importScripts('dsp-utils.js');

class GraphicEQProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Initialize parameters
    this.initializeParameters({});

    // Initialize state
    this.initializeState();

    // Handle parameter updates from main thread
    this.port.onmessage = (event) => {
      this.handleMessage(event.data);
    };
  }

  /**
   * Initialize parameters with defaults
   */
  initializeParameters(params) {
    // Standard ISO graphic EQ frequencies
    this.frequencies = [31.25, 62.5, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];

    // Gain for each band (in dB)
    this.gains = [
      params.gain31 || 0,
      params.gain62 || 0,
      params.gain125 || 0,
      params.gain250 || 0,
      params.gain500 || 0,
      params.gain1k || 0,
      params.gain2k || 0,
      params.gain4k || 0,
      params.gain8k || 0,
      params.gain16k || 0
    ];

    // Fixed Q for graphic EQ (typically 1.0 for 1/3 octave spacing)
    this.q = params.q || 1.0;

    // Global output gain
    this.outputGain = params.outputGain || 1.0;
  }

  /**
   * Initialize filter state
   */
  initializeState() {
    // Create 10 bands, each with 2 filters (stereo)
    // bands[bandIndex][channel]
    this.bands = [];

    for (let i = 0; i < 10; i++) {
      this.bands.push([new BiquadFilter(), new BiquadFilter()]);
    }

    // Update filter coefficients
    this.updateFilters();
  }

  /**
   * Update all filter coefficients
   */
  updateFilters() {
    for (let i = 0; i < 10; i++) {
      const freq = this.frequencies[i];
      const gain = this.gains[i];

      for (let ch = 0; ch < 2; ch++) {
        this.bands[i][ch].setPeaking(freq, this.q, gain, sampleRate);
      }
    }
  }

  /**
   * Handle messages from main thread
   */
  handleMessage(data) {
    if (data.type === 'update') {
      this.handleParameterUpdate(data.param, data.value);
    } else if (data.type === 'init') {
      this.initializeParameters(data.params);
      this.updateFilters();
    } else if (data.type === 'updateBand') {
      // Update a specific band
      const bandIndex = data.bandIndex;
      const gain = data.gain;

      if (bandIndex >= 0 && bandIndex < 10) {
        this.gains[bandIndex] = gain;
        const freq = this.frequencies[bandIndex];

        for (let ch = 0; ch < 2; ch++) {
          this.bands[bandIndex][ch].setPeaking(freq, this.q, gain, sampleRate);
        }
      }
    }
  }

  /**
   * Handle parameter update
   */
  handleParameterUpdate(param, value) {
    // Map parameter names to band indices
    const bandMap = {
      'gain31': 0,
      'gain62': 1,
      'gain125': 2,
      'gain250': 3,
      'gain500': 4,
      'gain1k': 5,
      'gain2k': 6,
      'gain4k': 7,
      'gain8k': 8,
      'gain16k': 9
    };

    if (param in bandMap) {
      const bandIndex = bandMap[param];
      this.gains[bandIndex] = value;

      // Update filter for this band
      const freq = this.frequencies[bandIndex];
      for (let ch = 0; ch < 2; ch++) {
        this.bands[bandIndex][ch].setPeaking(freq, this.q, value, sampleRate);
      }
    } else if (param === 'q') {
      this.q = value;
      this.updateFilters();
    } else if (param === 'outputGain') {
      this.outputGain = value;
    }
  }

  /**
   * Process audio samples
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const numChannels = Math.min(input.length, output.length, 2);
    const bufferLength = input[0].length;

    // Process each channel
    for (let ch = 0; ch < numChannels; ch++) {
      const inputChannel = input[ch];
      const outputChannel = output[ch];

      // Process each sample
      for (let i = 0; i < bufferLength; i++) {
        outputChannel[i] = this.processSample(inputChannel[i], ch);
      }
    }

    return true;
  }

  /**
   * Process a single sample through all bands
   */
  processSample(sample, channel) {
    let output = sample;

    // Process through all 10 bands in series
    for (let i = 0; i < 10; i++) {
      output = this.bands[i][channel].process(output);
    }

    // Apply output gain
    return output * this.outputGain;
  }

  static get parameterDescriptors() {
    return [
      { name: 'gain31', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain62', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain125', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain250', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain500', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain1k', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain2k', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain4k', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain8k', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'gain16k', defaultValue: 0, minValue: -15, maxValue: 15 },
      { name: 'q', defaultValue: 1.0, minValue: 0.5, maxValue: 5.0 },
      { name: 'outputGain', defaultValue: 1.0, minValue: 0, maxValue: 2.0 }
    ];
  }
}

registerProcessor('graphic-eq-processor', GraphicEQProcessor);
