/**
 * OverdriveProcessor - AudioWorklet implementation of tube-style overdrive
 *
 * Features:
 * - Soft clipping with multiple curve types (tanh, atan, softClip)
 * - Asymmetric distortion for even harmonics (bias)
 * - Tone stack (post-distortion lowshelf EQ)
 * - Auto gain compensation
 * - Dry/wet mix
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class OverdriveProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'drive',
        defaultValue: 30,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'tone',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'bias',
        defaultValue: 0,
        minValue: -100,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'curveType',
        defaultValue: 0, // 0=tanh, 1=atan, 2=softClip
        minValue: 0,
        maxValue: 2,
        automationRate: 'k-rate'
      },
      {
        name: 'output',
        defaultValue: 0,
        minValue: -24,
        maxValue: 24,
        automationRate: 'k-rate'
      },
      {
        name: 'mix',
        defaultValue: 100,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Tone filters (lowshelf, one per channel)
    this.toneFilters = [
      new BiquadFilter(),
      new BiquadFilter()
    ];

    // Current parameter values
    this.drive = 30;
    this.tone = 50;
    this.bias = 0;
    this.curveType = 0;
    this.output = 0;
    this.mix = 100;

    // Initialize filters
    this.updateFilters();
  }

  /**
   * Update filter coefficients based on tone parameter
   */
  updateFilters() {
    // Tone control: 200 Hz to 5000 Hz lowshelf
    const freq = 200 + (this.tone / 100) * 4800;
    // Gain: -6 dB to +6 dB
    const gain = (this.tone / 100) * 12 - 6;

    this.toneFilters.forEach(filter => {
      filter.setLowShelf(freq, 0.7071, gain, sampleRate);
    });
  }

  /**
   * Apply soft clipping based on curve type
   */
  applySoftClipping(input, driveGain, bias) {
    // Apply bias for asymmetric distortion
    const biased = input + bias;
    const curveTypeInt = Math.floor(this.curveType);

    switch (curveTypeInt) {
      case 0: // Tanh - hyperbolic tangent, smooth warm saturation
        return Math.tanh(biased * driveGain);

      case 1: // Atan - arctangent, softer than tanh
        return (2 / Math.PI) * Math.atan(biased * driveGain * Math.PI / 2);

      case 2: // Soft clip - algebraic soft clipping
        {
          const val = biased * driveGain;
          if (Math.abs(val) < 1) {
            return val;
          } else {
            return val / (1 + Math.abs(val));
          }
        }

      default:
        return Math.tanh(biased * driveGain);
    }
  }

  /**
   * Process a single sample through the overdrive chain
   */
  processSample(sample, channel) {
    // Calculate drive gain (1 to 20)
    const driveGain = 1 + (this.drive / 100) * 19;

    // Calculate bias (-1 to 1)
    const biasAmount = this.bias / 100;

    // Calculate output gain from dB
    const outputGain = dbToGain(this.output);

    // Store dry signal for mixing
    const dry = sample;

    // Apply soft clipping
    let wet = this.applySoftClipping(sample, driveGain, biasAmount);

    // Apply tone filter (post-distortion)
    wet = this.toneFilters[channel].process(wet);

    // Apply output gain
    wet = wet * outputGain;

    // Mix dry and wet
    const mixAmount = this.mix / 100;
    return dry * (1 - mixAmount) + wet * mixAmount;
  }

  /**
   * Process audio block
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input.length) {
      return true;
    }

    // Get parameter values (k-rate)
    const drive = parameters.drive;
    const tone = parameters.tone;
    const bias = parameters.bias;
    const curveType = parameters.curveType;
    const outputParam = parameters.output;
    const mix = parameters.mix;

    // Update internal state
    const driveVal = drive[0];
    const toneVal = tone[0];
    const biasVal = bias[0];
    const curveTypeVal = curveType[0];
    const outputVal = outputParam[0];
    const mixVal = mix[0];

    // Check if tone filter needs update
    const filtersNeedUpdate = this.tone !== toneVal;

    this.drive = driveVal;
    this.tone = toneVal;
    this.bias = biasVal;
    this.curveType = curveTypeVal;
    this.output = outputVal;
    this.mix = mixVal;

    if (filtersNeedUpdate) {
      this.updateFilters();
    }

    const blockSize = input[0].length;

    // Process each channel
    for (let channel = 0; channel < Math.min(input.length, 2); channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < blockSize; i++) {
        outputChannel[i] = this.processSample(inputChannel[i], channel);
      }
    }

    return true;
  }
}

registerProcessor('overdrive-processor', OverdriveProcessor);
