/**
 * EchoProcessor - AudioWorklet implementation of multi-tap delay/echo
 *
 * Features:
 * - Multiple delay taps with decay
 * - Stereo delays with independent timing
 * - Feedback control
 * - Highpass/lowpass filtering
 * - Tempo sync capability
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class EchoProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'delayTimeL',
        defaultValue: 250,
        minValue: 0,
        maxValue: 2000,
        automationRate: 'k-rate'
      },
      {
        name: 'delayTimeR',
        defaultValue: 375,
        minValue: 0,
        maxValue: 2000,
        automationRate: 'k-rate'
      },
      {
        name: 'feedback',
        defaultValue: 40,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'numTaps',
        defaultValue: 4,
        minValue: 1,
        maxValue: 8,
        automationRate: 'k-rate'
      },
      {
        name: 'tapDecay',
        defaultValue: 0.7,
        minValue: 0.1,
        maxValue: 1.0,
        automationRate: 'k-rate'
      },
      {
        name: 'highpass',
        defaultValue: 20,
        minValue: 20,
        maxValue: 1000,
        automationRate: 'k-rate'
      },
      {
        name: 'lowpass',
        defaultValue: 20000,
        minValue: 1000,
        maxValue: 20000,
        automationRate: 'k-rate'
      },
      {
        name: 'stereoOffset',
        defaultValue: 0,
        minValue: -50,
        maxValue: 50,
        automationRate: 'k-rate'
      },
      {
        name: 'mix',
        defaultValue: 30,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Delay lines (stereo)
    this.delayL = new DelayLine(2.0, sampleRate);
    this.delayR = new DelayLine(2.0, sampleRate);

    // Feedback delay lines (separate for cleaner feedback)
    this.feedbackDelayL = new DelayLine(2.0, sampleRate);
    this.feedbackDelayR = new DelayLine(2.0, sampleRate);

    // Filters
    this.highpassL = new BiquadFilter();
    this.highpassR = new BiquadFilter();
    this.lowpassL = new BiquadFilter();
    this.lowpassR = new BiquadFilter();

    // Current parameter values
    this.delayTimeL = 250;
    this.delayTimeR = 375;
    this.feedback = 40;
    this.numTaps = 4;
    this.tapDecay = 0.7;
    this.highpass = 20;
    this.lowpass = 20000;
    this.stereoOffset = 0;
    this.mix = 30;

    // Update filters
    this.updateFilters();
  }

  /**
   * Update filter coefficients
   */
  updateFilters() {
    this.highpassL.setHighpass(this.highpass, 0.707, sampleRate);
    this.highpassR.setHighpass(this.highpass, 0.707, sampleRate);
    this.lowpassL.setLowpass(this.lowpass, 0.707, sampleRate);
    this.lowpassR.setLowpass(this.lowpass, 0.707, sampleRate);
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
    const delayTimeL = parameters.delayTimeL[0];
    const delayTimeR = parameters.delayTimeR[0];
    const feedback = parameters.feedback[0];
    const numTaps = Math.floor(parameters.numTaps[0]);
    const tapDecay = parameters.tapDecay[0];
    const highpass = parameters.highpass[0];
    const lowpass = parameters.lowpass[0];
    const stereoOffset = parameters.stereoOffset[0];
    const mix = parameters.mix[0];

    // Check if parameters changed
    if (this.delayTimeL !== delayTimeL) {
      this.delayTimeL = delayTimeL;
    }

    if (this.delayTimeR !== delayTimeR) {
      this.delayTimeR = delayTimeR;
    }

    if (this.feedback !== feedback) {
      this.feedback = feedback;
    }

    if (this.numTaps !== numTaps) {
      this.numTaps = numTaps;
    }

    if (this.tapDecay !== tapDecay) {
      this.tapDecay = tapDecay;
    }

    if (this.highpass !== highpass || this.lowpass !== lowpass) {
      this.highpass = highpass;
      this.lowpass = lowpass;
      this.updateFilters();
    }

    if (this.stereoOffset !== stereoOffset) {
      this.stereoOffset = stereoOffset;
    }

    this.mix = mix;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, 2);

    // Convert delay times to samples
    const delayTimeL_samples = (this.delayTimeL / 1000) * sampleRate;
    const delayTimeR_samples = (this.delayTimeR / 1000) * sampleRate;
    const offsetSamples = (this.stereoOffset / 1000) * sampleRate;

    // Feedback gain (max 0.95 to prevent runaway)
    const feedbackGain = (this.feedback / 100) * 0.95;

    // Process audio
    for (let channel = 0; channel < numChannels; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      const delayLine = channel === 0 ? this.delayL : this.delayR;
      const feedbackDelay = channel === 0 ? this.feedbackDelayL : this.feedbackDelayR;
      const highpassFilter = channel === 0 ? this.highpassL : this.highpassR;
      const lowpassFilter = channel === 0 ? this.lowpassL : this.lowpassR;

      let baseDelayTime = channel === 0 ? delayTimeL_samples : delayTimeR_samples;

      // Apply stereo offset
      if (channel === 1 && offsetSamples !== 0) {
        baseDelayTime += offsetSamples;
      }

      for (let i = 0; i < blockSize; i++) {
        const drySample = inputChannel[i];

        // Read feedback
        const feedbackSample = feedbackDelay.readInterpolated(baseDelayTime);

        // Filter feedback
        const filtered = lowpassFilter.process(highpassFilter.process(feedbackSample));

        // Write input + filtered feedback to delay line
        delayLine.write(drySample + filtered * feedbackGain);
        feedbackDelay.write(drySample + filtered * feedbackGain);

        // Generate multi-tap echo output
        let wetSample = 0;

        for (let tap = 1; tap <= this.numTaps; tap++) {
          const tapDelay = baseDelayTime * tap;
          const tapGain = Math.pow(this.tapDecay, tap);
          const tapSample = delayLine.readInterpolated(tapDelay);

          wetSample += tapSample * tapGain;
        }

        // Normalize by number of taps
        wetSample /= this.numTaps;

        // Mix dry and wet
        const wetGain = this.mix / 100;
        const dryGain = 1 - wetGain;

        outputChannel[i] = drySample * dryGain + wetSample * wetGain;
      }
    }

    return true;
  }
}

registerProcessor('echo-processor', EchoProcessor);
