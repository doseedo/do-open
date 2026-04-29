/**
 * FlangerProcessor - AudioWorklet implementation of flanger effect
 *
 * Features:
 * - Short delay with LFO modulation (0.5-10ms)
 * - High feedback for jet-plane whoosh effect
 * - Manual offset control
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

class FlangerProcessor extends AudioWorkletProcessor {
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
        defaultValue: 50,
        minValue: -100,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'delay',
        defaultValue: 3,
        minValue: 0.5,
        maxValue: 10,
        automationRate: 'k-rate'
      },
      {
        name: 'manual',
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

    // Create delay lines for stereo
    this.delayLeft = new DelayLine(0.02, sampleRate); // 20ms max
    this.delayRight = new DelayLine(0.02, sampleRate);

    // Create LFO
    this.lfo = new LFO(sampleRate);

    // Current parameters
    this.rate = 0.5;
    this.depth = 50;
    this.feedback = 50;
    this.baseDelay = 3;
    this.manual = 50;
    this.waveform = 0;
    this.mix = 50;
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
    const delay = parameters.delay[0];
    const manual = parameters.manual[0];
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
    this.feedback = feedback;
    this.baseDelay = delay;
    this.manual = manual;
    this.mix = mix;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, output.length);

    // Process each sample
    for (let i = 0; i < blockSize; i++) {
      // Get LFO value (-1 to 1)
      const lfoValue = this.lfo.process();

      // Calculate modulated delay time
      const depthSeconds = (this.depth / 100) * 0.003; // Max 3ms modulation for flanger
      const baseDelaySeconds = this.baseDelay / 1000;
      const manualOffsetSeconds = (this.manual / 100) * 0.005; // Max 5ms manual offset
      const modulation = lfoValue * depthSeconds;
      const delayTime = baseDelaySeconds + manualOffsetSeconds + modulation;
      const delaySamples = delayTime * sampleRate;

      // Process left channel
      const inputL = input[0] ? input[0][i] : 0;
      const delayedL = this.delayLeft.readInterpolated(delaySamples);

      // Apply feedback (can be positive or negative)
      const feedbackAmount = (this.feedback / 100) * 0.95; // Max 0.95 to prevent runaway
      const feedbackL = inputL + delayedL * feedbackAmount;
      this.delayLeft.write(feedbackL);

      // Mix dry and wet (equal power crossfade)
      const mixAmount = this.mix / 100;
      const wetGain = Math.sin(mixAmount * Math.PI / 2);
      const dryGain = Math.cos(mixAmount * Math.PI / 2);

      output[0][i] = inputL * dryGain + delayedL * wetGain;

      // Process right channel if stereo
      if (numChannels > 1) {
        const inputR = input[1] ? input[1][i] : inputL;
        const delayedR = this.delayRight.readInterpolated(delaySamples);

        const feedbackR = inputR + delayedR * feedbackAmount;
        this.delayRight.write(feedbackR);

        output[1][i] = inputR * dryGain + delayedR * wetGain;
      }
    }

    return true;
  }
}

registerProcessor('flanger-processor', FlangerProcessor);
