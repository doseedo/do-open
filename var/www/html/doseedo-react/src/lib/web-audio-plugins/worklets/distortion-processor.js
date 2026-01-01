/**
 * DistortionProcessor - AudioWorklet implementation of hard clipping distortion
 *
 * Features:
 * - Multiple waveshaping algorithms (hard, soft, asymmetric, foldback)
 * - Pre/post filtering with tone control
 * - High gain capability
 * - DC blocking
 * - Dry/wet mix
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class DistortionProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'drive',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'tone',
        defaultValue: 1000,
        minValue: 20,
        maxValue: 20000,
        automationRate: 'k-rate'
      },
      {
        name: 'toneWidth',
        defaultValue: 1,
        minValue: 0.1,
        maxValue: 10,
        automationRate: 'k-rate'
      },
      {
        name: 'clipType',
        defaultValue: 0, // 0=hard, 1=soft, 2=asymmetric, 3=foldback
        minValue: 0,
        maxValue: 3,
        automationRate: 'k-rate'
      },
      {
        name: 'filterPosition',
        defaultValue: 0, // 0=post, 1=pre
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

    // Tone filters (one per channel for pre-filtering)
    this.preFilters = [
      new BiquadFilter(),
      new BiquadFilter()
    ];

    // Tone filters (one per channel for post-filtering)
    this.postFilters = [
      new BiquadFilter(),
      new BiquadFilter()
    ];

    // DC blocking filters (highpass at 5 Hz)
    this.dcFilters = [
      new BiquadFilter(),
      new BiquadFilter()
    ];

    // Initialize DC filters
    this.dcFilters.forEach(filter => {
      filter.setHighpass(5, 0.7071, sampleRate);
    });

    // Current parameter values
    this.drive = 50;
    this.tone = 1000;
    this.toneWidth = 1;
    this.clipType = 0;
    this.filterPosition = 0;
    this.output = 0;
    this.mix = 100;

    // Initialize filters
    this.updateFilters();
  }

  /**
   * Update filter coefficients based on tone parameters
   */
  updateFilters() {
    const filters = this.filterPosition === 1 ? this.preFilters : this.postFilters;
    filters.forEach(filter => {
      filter.setPeaking(this.tone, this.toneWidth, 0, sampleRate);
    });
  }

  /**
   * Apply waveshaping based on clip type
   */
  applyWaveshaping(input, drive) {
    const clipTypeInt = Math.floor(this.clipType);

    switch (clipTypeInt) {
      case 0: // Hard clipping
        return Math.max(-1, Math.min(1, input));

      case 1: // Soft clipping (tanh)
        return Math.tanh(input);

      case 2: // Asymmetric clipping
        if (input > 0) {
          return Math.min(1, input * 1.5);
        } else {
          return Math.max(-1, input * 0.8);
        }

      case 3: // Foldback distortion
        {
          const threshold = 1.0;
          if (Math.abs(input) > threshold) {
            const excess = Math.abs(input) - threshold;
            const folded = threshold - (excess % (2 * threshold));
            return input > 0 ? folded : -folded;
          }
          return input;
        }

      default:
        return Math.max(-1, Math.min(1, input));
    }
  }

  /**
   * Process a single sample through the distortion chain
   */
  processSample(sample, channel) {
    // Calculate drive gain (1 to 50)
    const driveGain = 1 + (this.drive / 100) * 49;

    // Calculate output gain from dB
    const outputGain = 0.5 * dbToGain(this.output); // Base gain 0.5 to compensate

    // Store dry signal for mixing
    const dry = sample;

    let wet = sample;

    // Pre-filtering if enabled
    if (this.filterPosition === 1) {
      wet = this.preFilters[channel].process(wet);
    }

    // Apply drive
    wet = wet * driveGain;

    // Apply waveshaping
    wet = this.applyWaveshaping(wet, driveGain);

    // Post-filtering if enabled
    if (this.filterPosition === 0) {
      wet = this.postFilters[channel].process(wet);
    }

    // DC blocking
    wet = this.dcFilters[channel].process(wet);

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
    const toneWidth = parameters.toneWidth;
    const clipType = parameters.clipType;
    const filterPosition = parameters.filterPosition;
    const outputParam = parameters.output;
    const mix = parameters.mix;

    // Check if parameters changed
    const driveVal = drive[0];
    const toneVal = tone[0];
    const toneWidthVal = toneWidth[0];
    const clipTypeVal = clipType[0];
    const filterPositionVal = filterPosition[0];
    const outputVal = outputParam[0];
    const mixVal = mix[0];

    // Update internal state
    const filtersNeedUpdate =
      this.tone !== toneVal ||
      this.toneWidth !== toneWidthVal ||
      this.filterPosition !== filterPositionVal;

    this.drive = driveVal;
    this.tone = toneVal;
    this.toneWidth = toneWidthVal;
    this.clipType = clipTypeVal;
    this.filterPosition = filterPositionVal;
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

registerProcessor('distortion-processor', DistortionProcessor);
