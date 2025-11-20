/**
 * Redux AudioWorklet Processor
 * High-performance bit crushing and sample rate reduction
 *
 * This is the modern AudioWorklet version for use in production.
 * Runs on a separate audio thread for better performance.
 */

class ReduxProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Processing state
    this.sampleCounter = 0;
    this.lastSample = [0, 0]; // For stereo

    // Default parameters
    this.bitDepth = 8;
    this.sampleRate = 22050;
    this.hardness = 0.5;
    this.dither = 0.0;
    this.jitter = 0.0;

    // Listen for parameter changes from main thread
    this.port.onmessage = (event) => {
      const { type, params } = event.data;
      if (type === 'setParams') {
        this.updateParams(params);
      }
    };
  }

  /**
   * Update parameters from main thread
   */
  updateParams(params) {
    if (params.bitDepth !== undefined) {
      this.bitDepth = Math.max(1, Math.min(16, params.bitDepth));
    }
    if (params.sampleRate !== undefined) {
      this.sampleRate = Math.max(50, Math.min(sampleRate, params.sampleRate));
    }
    if (params.hardness !== undefined) {
      this.hardness = Math.max(0, Math.min(1, params.hardness / 100));
    }
    if (params.dither !== undefined) {
      this.dither = Math.max(0, Math.min(1, params.dither / 100));
    }
    if (params.jitter !== undefined) {
      this.jitter = Math.max(0, Math.min(1, params.jitter / 100));
    }
  }

  /**
   * Process a single sample with bit crushing
   */
  processSample(sample, levels, ditherAmount, hardness) {
    // Add dither (TPDF - Triangular Probability Density Function)
    const dither1 = Math.random();
    const dither2 = Math.random();
    const tpdf = (dither1 + dither2 - 1) * ditherAmount / levels;
    const ditheredSample = sample + tpdf;

    // Quantization with hardness curve
    let quantized;
    if (hardness === 1) {
      // Hard quantization (standard)
      quantized = Math.round(ditheredSample * levels) / levels;
    } else {
      // Soft quantization (blend between original and quantized)
      const hard = Math.round(ditheredSample * levels) / levels;
      quantized = sample + (hard - sample) * hardness;
    }

    // Clamp to -1 to 1
    return Math.max(-1, Math.min(1, quantized));
  }

  /**
   * Process audio (called by Web Audio API)
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

    const reduction = sampleRate / this.sampleRate;
    const levels = Math.pow(2, this.bitDepth);
    const ditherAmount = this.dither;
    const hardness = this.hardness;
    const jitterAmount = this.jitter;

    for (let i = 0; i < inputL.length; i++) {
      this.sampleCounter++;

      // Calculate jitter (random timing variation)
      const jitter = (Math.random() * 2 - 1) * jitterAmount * reduction;
      const threshold = reduction + jitter;

      // Sample and hold (sample rate reduction)
      if (this.sampleCounter >= threshold) {
        // Process left channel
        outputL[i] = this.processSample(inputL[i], levels, ditherAmount, hardness);
        this.lastSample[0] = outputL[i];

        // Process right channel
        if (outputR) {
          outputR[i] = this.processSample(inputR[i], levels, ditherAmount, hardness);
          this.lastSample[1] = outputR[i];
        }

        this.sampleCounter = 0;
      } else {
        // Hold previous sample
        outputL[i] = this.lastSample[0];
        if (outputR) {
          outputR[i] = this.lastSample[1];
        }
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('redux-processor', ReduxProcessor);
