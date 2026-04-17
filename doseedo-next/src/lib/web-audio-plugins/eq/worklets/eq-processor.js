/**
 * EQ Processor - 3-Band Parametric Equalizer
 * AudioWorklet processor for high-performance EQ processing
 *
 * Features:
 * - 3 independent peaking filters (Low, Mid, High)
 * - Adjustable frequency, gain, and Q per band
 * - Smooth parameter changes
 * - Optimized for offline rendering
 *
 * @author Agent 2 (EQ Plugins)
 */

importScripts('dsp-utils.js');

class EQProcessor extends AudioWorkletProcessor {
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
    // Band 1 (Low)
    this.freq1 = params.freq1 || 100;
    this.gain1 = params.gain1 || 0;  // dB
    this.q1 = params.q1 || 1.0;

    // Band 2 (Mid)
    this.freq2 = params.freq2 || 1000;
    this.gain2 = params.gain2 || 0;  // dB
    this.q2 = params.q2 || 1.0;

    // Band 3 (High)
    this.freq3 = params.freq3 || 10000;
    this.gain3 = params.gain3 || 0;  // dB
    this.q3 = params.q3 || 1.0;

    // Global output gain
    this.outputGain = params.outputGain || 1.0;
  }

  /**
   * Initialize filter state
   */
  initializeState() {
    // Create biquad filters for each band and channel
    // [band][channel]
    this.band1 = [new BiquadFilter(), new BiquadFilter()];
    this.band2 = [new BiquadFilter(), new BiquadFilter()];
    this.band3 = [new BiquadFilter(), new BiquadFilter()];

    // Update filter coefficients
    this.updateFilters();
  }

  /**
   * Update all filter coefficients based on current parameters
   */
  updateFilters() {
    // Update band 1 (both channels)
    for (let ch = 0; ch < 2; ch++) {
      this.band1[ch].setPeaking(this.freq1, this.q1, this.gain1, sampleRate);
      this.band2[ch].setPeaking(this.freq2, this.q2, this.gain2, sampleRate);
      this.band3[ch].setPeaking(this.freq3, this.q3, this.gain3, sampleRate);
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
    }
  }

  /**
   * Handle parameter update
   */
  handleParameterUpdate(param, value) {
    this[param] = value;

    // Recalculate filters when any EQ param changes
    if (param.startsWith('freq') || param.startsWith('gain') || param.startsWith('q')) {
      this.updateFilters();
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
        // Process through all bands in series
        let sample = inputChannel[i];
        sample = this.band1[ch].process(sample);
        sample = this.band2[ch].process(sample);
        sample = this.band3[ch].process(sample);

        // Apply output gain
        outputChannel[i] = sample * this.outputGain;
      }
    }

    return true;
  }

  /**
   * Process a single sample through all bands
   */
  processSample(sample, channel) {
    // Process through all bands in series
    let output = sample;
    output = this.band1[channel].process(output);
    output = this.band2[channel].process(output);
    output = this.band3[channel].process(output);
    return output * this.outputGain;
  }

  static get parameterDescriptors() {
    return [
      // Band 1
      {
        name: 'freq1',
        defaultValue: 100,
        minValue: 20,
        maxValue: 20000
      },
      {
        name: 'gain1',
        defaultValue: 0,
        minValue: -15,
        maxValue: 15
      },
      {
        name: 'q1',
        defaultValue: 1.0,
        minValue: 0.1,
        maxValue: 10.0
      },
      // Band 2
      {
        name: 'freq2',
        defaultValue: 1000,
        minValue: 20,
        maxValue: 20000
      },
      {
        name: 'gain2',
        defaultValue: 0,
        minValue: -15,
        maxValue: 15
      },
      {
        name: 'q2',
        defaultValue: 1.0,
        minValue: 0.1,
        maxValue: 10.0
      },
      // Band 3
      {
        name: 'freq3',
        defaultValue: 10000,
        minValue: 20,
        maxValue: 20000
      },
      {
        name: 'gain3',
        defaultValue: 0,
        minValue: -15,
        maxValue: 15
      },
      {
        name: 'q3',
        defaultValue: 1.0,
        minValue: 0.1,
        maxValue: 10.0
      },
      // Output
      {
        name: 'outputGain',
        defaultValue: 1.0,
        minValue: 0,
        maxValue: 2.0
      }
    ];
  }
}

registerProcessor('eq-processor', EQProcessor);
