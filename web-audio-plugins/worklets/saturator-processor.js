/**
 * SaturatorProcessor - AudioWorklet implementation of multi-mode saturation
 *
 * Features:
 * - Multiple saturation algorithms (warm, digital, analog, clip, foldback, sine-fold)
 * - Harmonic emphasis with color filter
 * - DC offset removal
 * - Depth control for saturation intensity
 * - Dry/wet mix
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class SaturatorProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'drive',
        defaultValue: 0,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'type',
        defaultValue: 0, // 0=warm, 1=digital, 2=analog, 3=clip, 4=foldback, 5=sine-fold
        minValue: 0,
        maxValue: 5,
        automationRate: 'k-rate'
      },
      {
        name: 'color',
        defaultValue: 0,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'depth',
        defaultValue: 100,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'dcFilter',
        defaultValue: 1, // 1=enabled, 0=disabled
        minValue: 0,
        maxValue: 1,
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

    // Color filters (peaking, one per channel)
    this.colorFilters = [
      new BiquadFilter(),
      new BiquadFilter()
    ];

    // DC blocking filters (highpass at 5 Hz, one per channel)
    this.dcFilters = [
      new BiquadFilter(),
      new BiquadFilter()
    ];

    // Current parameter values
    this.drive = 0;
    this.type = 0;
    this.color = 0;
    this.depth = 100;
    this.dcFilterEnabled = 1;
    this.output = 0;
    this.mix = 100;

    // Initialize filters
    this.updateFilters();
  }

  /**
   * Update filter coefficients based on color parameter
   */
  updateFilters() {
    // Color filter: 2000 Hz to 8000 Hz peaking
    const freq = 2000 + (this.color / 100) * 6000;
    // Gain: 0 dB to +6 dB
    const gain = (this.color / 100) * 6;

    this.colorFilters.forEach(filter => {
      filter.setPeaking(freq, 2, gain, sampleRate);
    });

    // DC filter: highpass at 5 Hz (or very low if disabled)
    const dcFreq = this.dcFilterEnabled > 0.5 ? 5 : 0.1;
    this.dcFilters.forEach(filter => {
      filter.setHighpass(dcFreq, 0.7071, sampleRate);
    });
  }

  /**
   * Apply saturation based on type
   */
  applySaturation(input, driveGain) {
    const typeInt = Math.floor(this.type);

    switch (typeInt) {
      case 0: // Warm - soft tanh saturation
        return Math.tanh(input);

      case 1: // Digital - hard clipping
        return Math.max(-1, Math.min(1, input));

      case 2: // Analog - asymmetric soft clip (simulates analog circuits)
        {
          const biasedInput = input + 0.1; // Slight asymmetry
          return Math.tanh(biasedInput);
        }

      case 3: // Clip - very hard clip with quick transition
        {
          if (input > 0.1) return 1;
          if (input < -0.1) return -1;
          return input * 10;
        }

      case 4: // Foldback - complex harmonics
        {
          const folded = input;
          return Math.abs((folded + 1) % 4 - 2) - 1;
        }

      case 5: // Sine-fold - musical harmonics
        return Math.sin(input * Math.PI);

      default:
        return Math.tanh(input);
    }
  }

  /**
   * Process a single sample through the saturation chain
   */
  processSample(sample, channel) {
    // Calculate drive gain (1 to 10)
    const driveGain = 1 + (this.drive / 100) * 9;

    // Calculate depth factor (0 to 1)
    const depthFactor = this.depth / 100;

    // Calculate output gain from dB
    const outputGain = dbToGain(this.output);

    // Store dry signal for mixing
    const dry = sample;

    // Apply drive
    let wet = sample * driveGain;

    // Apply saturation
    const saturated = this.applySaturation(wet, driveGain);

    // Apply depth (blend between dry and saturated)
    wet = sample + (saturated - sample) * depthFactor;

    // Apply DC filter
    wet = this.dcFilters[channel].process(wet);

    // Apply color filter
    wet = this.colorFilters[channel].process(wet);

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
    const type = parameters.type;
    const color = parameters.color;
    const depth = parameters.depth;
    const dcFilter = parameters.dcFilter;
    const outputParam = parameters.output;
    const mix = parameters.mix;

    // Update internal state
    const driveVal = drive[0];
    const typeVal = type[0];
    const colorVal = color[0];
    const depthVal = depth[0];
    const dcFilterVal = dcFilter[0];
    const outputVal = outputParam[0];
    const mixVal = mix[0];

    // Check if filters need update
    const filtersNeedUpdate =
      this.color !== colorVal ||
      this.dcFilterEnabled !== dcFilterVal;

    this.drive = driveVal;
    this.type = typeVal;
    this.color = colorVal;
    this.depth = depthVal;
    this.dcFilterEnabled = dcFilterVal;
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

registerProcessor('saturator-processor', SaturatorProcessor);
