/**
 * Filter Processor - Single Versatile Filter
 * AudioWorklet processor for multi-mode filter
 *
 * Features:
 * - Multiple filter types (lowpass, highpass, bandpass, notch, peaking, lowshelf, highshelf, allpass)
 * - Adjustable frequency, Q, and gain
 * - Smooth parameter changes
 * - Optimized for real-time and offline processing
 *
 * @author Agent 2 (EQ Plugins)
 */

importScripts('dsp-utils.js');

class FilterProcessor extends AudioWorkletProcessor {
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
    this.type = params.type || 'lowpass';  // Filter type
    this.frequency = params.frequency || 1000;  // Hz
    this.q = params.q || 1.0;  // Resonance/bandwidth
    this.gain = params.gain || 0;  // dB (for peaking/shelving)
    this.mix = params.mix || 1.0;  // Dry/wet mix (0-1)
    this.outputGain = params.outputGain || 1.0;  // Output level
  }

  /**
   * Initialize filter state
   */
  initializeState() {
    // Create filters for stereo (2 channels)
    this.filters = [new BiquadFilter(), new BiquadFilter()];

    // Update filter coefficients
    this.updateFilter();
  }

  /**
   * Update filter coefficients based on current parameters
   */
  updateFilter() {
    for (let ch = 0; ch < 2; ch++) {
      switch (this.type) {
        case 'lowpass':
          this.filters[ch].setLowpass(this.frequency, this.q, sampleRate);
          break;

        case 'highpass':
          this.filters[ch].setHighpass(this.frequency, this.q, sampleRate);
          break;

        case 'bandpass':
          this.filters[ch].setBandpass(this.frequency, this.q, sampleRate);
          break;

        case 'notch':
          this.filters[ch].setNotch(this.frequency, this.q, sampleRate);
          break;

        case 'peaking':
          this.filters[ch].setPeaking(this.frequency, this.q, this.gain, sampleRate);
          break;

        case 'lowshelf':
          this.filters[ch].setLowShelf(this.frequency, this.q, this.gain, sampleRate);
          break;

        case 'highshelf':
          this.filters[ch].setHighShelf(this.frequency, this.q, this.gain, sampleRate);
          break;

        case 'allpass':
          this.filters[ch].setAllpass(this.frequency, this.q, sampleRate);
          break;

        default:
          console.warn(`Unknown filter type: ${this.type}`);
          this.filters[ch].setLowpass(this.frequency, this.q, sampleRate);
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
      this.updateFilter();
    }
  }

  /**
   * Handle parameter update
   */
  handleParameterUpdate(param, value) {
    const oldValue = this[param];
    this[param] = value;

    // Recalculate filter if relevant parameter changed
    if (param === 'type' || param === 'frequency' || param === 'q' || param === 'gain') {
      this.updateFilter();
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
   * Process a single sample
   */
  processSample(sample, channel) {
    // Apply filter
    const wet = this.filters[channel].process(sample);

    // Mix dry/wet
    const mixed = sample * (1 - this.mix) + wet * this.mix;

    // Apply output gain
    return mixed * this.outputGain;
  }

  static get parameterDescriptors() {
    return [
      {
        name: 'frequency',
        defaultValue: 1000,
        minValue: 20,
        maxValue: 20000
      },
      {
        name: 'q',
        defaultValue: 1.0,
        minValue: 0.1,
        maxValue: 20.0
      },
      {
        name: 'gain',
        defaultValue: 0,
        minValue: -15,
        maxValue: 15
      },
      {
        name: 'mix',
        defaultValue: 1.0,
        minValue: 0,
        maxValue: 1.0
      },
      {
        name: 'outputGain',
        defaultValue: 1.0,
        minValue: 0,
        maxValue: 2.0
      }
    ];
  }
}

registerProcessor('filter-processor', FilterProcessor);
