/**
 * Convolution Reverb AudioWorklet Processor
 *
 * High-performance convolution reverb using impulse responses.
 * NOTE: This is a simplified convolution implementation for real-time processing.
 * For production, consider using native ConvolverNode or FFT-based convolution.
 *
 * @author Agent 6: Reverb Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class ConvolutionReverbProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.mix = 0.3;
    this.predelay = 0; // milliseconds
    this.impulseResponse = null;
    this.irLength = 0;

    // Convolution buffers
    this.inputBuffer = null;
    this.inputIndex = 0;

    // Pre-delay line
    this.predelayLine = null;
    this.sampleRate = sampleRate;

    // Listen for parameter updates and IR data from main thread
    this.port.onmessage = (event) => {
      const { type, params, impulseResponse } = event.data;

      if (type === 'setParams') {
        this.updateParams(params);
      } else if (type === 'setImpulseResponse') {
        this.setImpulseResponse(impulseResponse);
      }
    };
  }

  /**
   * Update parameters from main thread
   * @param {Object} params - Parameter values
   */
  updateParams(params) {
    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(1, params.mix / 100));
    }

    if (params.predelay !== undefined) {
      this.predelay = Math.max(0, Math.min(250, params.predelay));

      // Recreate predelay line if needed
      if (this.predelay > 0) {
        const predelaySeconds = this.predelay / 1000;
        this.predelayLine = new DelayLine(predelaySeconds + 0.01, this.sampleRate);
      } else {
        this.predelayLine = null;
      }
    }
  }

  /**
   * Set impulse response
   * @param {Float32Array} impulseResponse - IR samples
   */
  setImpulseResponse(impulseResponse) {
    if (!impulseResponse || impulseResponse.length === 0) {
      console.warn('Invalid impulse response');
      return;
    }

    this.impulseResponse = impulseResponse;
    this.irLength = impulseResponse.length;

    // Initialize input buffer for convolution
    // Using overlap-save method for efficiency
    const bufferSize = Math.max(2048, this.irLength);
    this.inputBuffer = new Float32Array(bufferSize);
    this.inputIndex = 0;

    this.port.postMessage({
      type: 'irLoaded',
      length: this.irLength,
      duration: this.irLength / this.sampleRate
    });
  }

  /**
   * Perform direct convolution (time-domain)
   * NOTE: This is simple but CPU-intensive. For production,
   * use FFT-based convolution or native ConvolverNode.
   *
   * @param {number} input - Input sample
   * @param {number} channel - Channel index
   * @returns {number} Convolved output
   */
  convolve(input, channel) {
    if (!this.impulseResponse || this.irLength === 0) {
      return input;
    }

    // Write input to circular buffer
    this.inputBuffer[this.inputIndex] = input;

    // Perform convolution
    let output = 0;
    const maxTaps = Math.min(this.irLength, this.inputIndex + 1);

    for (let i = 0; i < maxTaps; i++) {
      const bufferIdx = (this.inputIndex - i + this.inputBuffer.length) % this.inputBuffer.length;
      output += this.inputBuffer[bufferIdx] * this.impulseResponse[i];
    }

    return output;
  }

  /**
   * Process audio (called by Web Audio API)
   * @param {Float32Array[][]} inputs - Input buffers
   * @param {Float32Array[][]} outputs - Output buffers
   * @param {Object} parameters - AudioParam values
   * @returns {boolean} Keep processor alive
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || input.length === 0) {
      return true;
    }

    const inputL = input[0];
    const inputR = input.length > 1 ? input[1] : input[0];
    const outputL = output[0];
    const outputR = output.length > 1 ? output[1] : output[0];

    // Process each sample
    for (let i = 0; i < inputL.length; i++) {
      // Mono input for convolution (sum stereo to mono)
      const monoInput = input.length > 1 ? (inputL[i] + inputR[i]) * 0.5 : inputL[i];

      // Apply pre-delay if configured
      let delayedInput = monoInput;
      if (this.predelayLine && this.predelay > 0) {
        this.predelayLine.write(monoInput);
        const predelaySamples = (this.predelay / 1000) * this.sampleRate;
        delayedInput = this.predelayLine.readInterpolated(predelaySamples);
      }

      // Perform convolution
      const wet = this.convolve(delayedInput, 0);

      // Advance input buffer index
      this.inputIndex = (this.inputIndex + 1) % this.inputBuffer.length;

      // Mix dry/wet for both channels
      const dryGain = 1 - this.mix;
      const wetGain = this.mix;

      outputL[i] = inputL[i] * dryGain + wet * wetGain;

      if (outputR) {
        outputR[i] = inputR[i] * dryGain + wet * wetGain;
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('convolution-reverb-processor', ConvolutionReverbProcessor);
