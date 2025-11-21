/**
 * PhaserProcessor - AudioWorklet implementation of phaser effect
 *
 * Features:
 * - Cascade of allpass filters (4, 6, 8, or 12 stages)
 * - LFO modulates filter frequencies for sweeping notches
 * - Feedback for resonance
 * - Frequency spread control
 * - Multiple LFO waveforms
 *
 * @author Agent 4 - Modulation Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

/**
 * LFO - Low Frequency Oscillator
 * Supports sine, triangle, square, sawtooth waveforms
 */
class LFO {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.phase = 0;
    this.frequency = 1.0;
    this.waveform = 0; // 0=sine, 1=triangle, 2=square, 3=sawtooth
  }

  setFrequency(hz) {
    this.frequency = hz;
  }

  setWaveform(type) {
    this.waveform = type;
  }

  process() {
    // Increment phase
    this.phase += this.frequency / this.sampleRate;
    if (this.phase >= 1.0) {
      this.phase -= 1.0;
    }

    // Generate waveform
    let output = 0;
    switch (this.waveform) {
      case 0: // Sine
        output = Math.sin(this.phase * 2 * Math.PI);
        break;
      case 1: // Triangle
        output = this.phase < 0.5
          ? 4 * this.phase - 1
          : -4 * this.phase + 3;
        break;
      case 2: // Square
        output = this.phase < 0.5 ? 1 : -1;
        break;
      case 3: // Sawtooth
        output = 2 * this.phase - 1;
        break;
    }

    return output;
  }

  reset() {
    this.phase = 0;
  }
}

/**
 * AllpassFilter - First-order allpass filter for phaser
 * Creates phase shift without affecting magnitude
 */
class AllpassFilter {
  constructor() {
    this.x1 = 0; // Previous input
    this.y1 = 0; // Previous output
    this.coefficient = 0;
  }

  /**
   * Set filter coefficient based on frequency
   * @param {number} frequency - Center frequency in Hz
   * @param {number} sampleRate - Sample rate in Hz
   */
  setFrequency(frequency, sampleRate) {
    // Calculate allpass coefficient
    const tan = Math.tan(Math.PI * frequency / sampleRate);
    this.coefficient = (tan - 1) / (tan + 1);
  }

  /**
   * Process one sample through allpass filter
   * Formula: y = -x + x[n-1] + coefficient * y[n-1]
   */
  process(input) {
    const output = -input + this.x1 + this.coefficient * this.y1;

    // Update state
    this.x1 = input;
    this.y1 = output;

    return output;
  }

  reset() {
    this.x1 = 0;
    this.y1 = 0;
  }
}

class PhaserProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'rate',
        defaultValue: 0.5,
        minValue: 0.01,
        maxValue: 10,
        automationRate: 'k-rate'
      },
      {
        name: 'depth',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'feedback',
        defaultValue: 0,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'stages',
        defaultValue: 6,
        minValue: 2,
        maxValue: 12,
        automationRate: 'k-rate'
      },
      {
        name: 'frequency',
        defaultValue: 1000,
        minValue: 200,
        maxValue: 8000,
        automationRate: 'k-rate'
      },
      {
        name: 'spread',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'waveform',
        defaultValue: 0, // 0=sine, 1=triangle, 2=square, 3=sawtooth
        minValue: 0,
        maxValue: 3,
        automationRate: 'k-rate'
      },
      {
        name: 'mix',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Create maximum number of allpass stages (12)
    this.maxStages = 12;
    this.allpassFiltersLeft = [];
    this.allpassFiltersRight = [];

    for (let i = 0; i < this.maxStages; i++) {
      this.allpassFiltersLeft.push(new AllpassFilter());
      this.allpassFiltersRight.push(new AllpassFilter());
    }

    // Create LFO
    this.lfo = new LFO(sampleRate);

    // Feedback buffers (store output of last stage)
    this.feedbackLeft = 0;
    this.feedbackRight = 0;

    // Current parameters
    this.rate = 0.5;
    this.depth = 50;
    this.feedbackAmount = 0;
    this.numStages = 6;
    this.centerFrequency = 1000;
    this.spread = 50;
    this.waveform = 0;
    this.mix = 50;

    // Update initial filter frequencies
    this.updateFilterFrequencies(0);
  }

  /**
   * Update all filter frequencies based on LFO, center frequency, and spread
   */
  updateFilterFrequencies(lfoValue) {
    const depthAmount = this.depth / 100;
    const spreadAmount = this.spread / 100;

    // LFO modulates center frequency (±2000 Hz at 100% depth)
    const modulation = lfoValue * depthAmount * 2000;
    const baseFreq = this.centerFrequency + modulation;

    // Update active stages with exponential frequency spacing
    for (let i = 0; i < this.numStages; i++) {
      // Exponential spacing for more musical notches
      const spreadFactor = Math.pow(2, (i / this.numStages) * spreadAmount);
      const frequency = clamp(baseFreq * spreadFactor, 200, 8000);

      this.allpassFiltersLeft[i].setFrequency(frequency, sampleRate);
      this.allpassFiltersRight[i].setFrequency(frequency, sampleRate);
    }
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

    // Get parameter values
    const rate = parameters.rate[0];
    const depth = parameters.depth[0];
    const feedback = parameters.feedback[0];
    const stages = Math.floor(clamp(parameters.stages[0], 2, this.maxStages));
    const frequency = parameters.frequency[0];
    const spread = parameters.spread[0];
    const waveform = Math.floor(parameters.waveform[0]);
    const mix = parameters.mix[0];

    // Update parameters if changed
    if (rate !== this.rate) {
      this.rate = rate;
      this.lfo.setFrequency(rate);
    }

    if (waveform !== this.waveform) {
      this.waveform = waveform;
      this.lfo.setWaveform(waveform);
    }

    this.depth = depth;
    this.feedbackAmount = feedback;
    this.numStages = stages;
    this.centerFrequency = frequency;
    this.spread = spread;
    this.mix = mix;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, output.length);

    // Convert feedback to linear (max 0.9 to prevent runaway)
    const feedbackGain = (this.feedbackAmount / 100) * 0.9;

    // Process each sample
    for (let i = 0; i < blockSize; i++) {
      // Get LFO value (-1 to 1)
      const lfoValue = this.lfo.process();

      // Update filter frequencies based on LFO
      this.updateFilterFrequencies(lfoValue);

      // Process left channel
      const inputL = input[0] ? input[0][i] : 0;
      let wetL = inputL + this.feedbackLeft * feedbackGain;

      // Cascade allpass filters
      for (let stage = 0; stage < this.numStages; stage++) {
        wetL = this.allpassFiltersLeft[stage].process(wetL);
      }

      // Store feedback
      this.feedbackLeft = wetL;

      // Mix dry and wet (equal power crossfade)
      const mixAmount = this.mix / 100;
      const wetGain = Math.sin(mixAmount * Math.PI / 2);
      const dryGain = Math.cos(mixAmount * Math.PI / 2);

      output[0][i] = inputL * dryGain + wetL * wetGain;

      // Process right channel if stereo
      if (numChannels > 1) {
        const inputR = input[1] ? input[1][i] : inputL;
        let wetR = inputR + this.feedbackRight * feedbackGain;

        // Cascade allpass filters
        for (let stage = 0; stage < this.numStages; stage++) {
          wetR = this.allpassFiltersRight[stage].process(wetR);
        }

        // Store feedback
        this.feedbackRight = wetR;

        output[1][i] = inputR * dryGain + wetR * wetGain;
      }
    }

    return true;
  }
}

registerProcessor('phaser-processor', PhaserProcessor);
