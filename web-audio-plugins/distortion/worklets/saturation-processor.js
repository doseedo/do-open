/**
 * Saturation AudioWorklet Processor
 * Multi-mode saturation from subtle warmth to heavy distortion
 *
 * Features:
 * - Multiple saturation algorithms (warm, digital, analog, clip, foldback, sine-fold)
 * - Color filter for harmonic emphasis
 * - DC offset removal
 * - Depth parameter for wet/dry character
 */

/**
 * Simple Biquad Filter implementation for AudioWorklet
 */
class BiquadFilter {
  constructor() {
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;
    this.a1 = 0;
    this.a2 = 0;

    // State variables
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }

  /**
   * Set as peaking filter
   */
  setPeaking(frequency, Q, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sn = Math.sin(omega);
    const cs = Math.cos(omega);
    const alpha = sn / (2 * Q);

    const a0 = 1 + alpha / A;
    this.b0 = (1 + alpha * A) / a0;
    this.b1 = (-2 * cs) / a0;
    this.b2 = (1 - alpha * A) / a0;
    this.a1 = (-2 * cs) / a0;
    this.a2 = (1 - alpha / A) / a0;
  }

  /**
   * Set as highpass filter (for DC removal)
   */
  setHighpass(frequency, Q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sn = Math.sin(omega);
    const cs = Math.cos(omega);
    const alpha = sn / (2 * Q);

    const a0 = 1 + alpha;
    this.b0 = ((1 + cs) / 2) / a0;
    this.b1 = (-(1 + cs)) / a0;
    this.b2 = ((1 + cs) / 2) / a0;
    this.a1 = (-2 * cs) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Process a single sample through the filter
   */
  process(input) {
    const output = this.b0 * input + this.b1 * this.x1 + this.b2 * this.x2
                 - this.a1 * this.y1 - this.a2 * this.y2;

    // Update state
    this.x2 = this.x1;
    this.x1 = input;
    this.y2 = this.y1;
    this.y1 = output;

    return output;
  }

  reset() {
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }
}

class SaturationProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.drive = 1.0;
    this.type = 'warm';
    this.colorFreq = 2000;
    this.colorGain = 0;
    this.depth = 1.0;
    this.dcFilterEnabled = true;
    this.output = 1.0;
    this.mix = 1.0;

    // Create filters for each channel
    this.dcFilters = [new BiquadFilter(), new BiquadFilter()];
    this.colorFilters = [new BiquadFilter(), new BiquadFilter()];

    // Initialize filters
    this.updateFilters();

    // Listen for parameter updates
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
    let needsFilterUpdate = false;

    if (params.drive !== undefined) {
      // Drive range: 1 to 10
      const drivePercent = Math.max(0, Math.min(100, params.drive));
      this.drive = 1 + (drivePercent / 100) * 9;
    }

    if (params.type !== undefined) {
      this.type = params.type;
    }

    if (params.color !== undefined) {
      const colorPercent = Math.max(0, Math.min(100, params.color));
      // Frequency range: 2000 Hz to 8000 Hz
      this.colorFreq = 2000 + (colorPercent / 100) * 6000;
      // Gain: 0 dB to +6 dB
      this.colorGain = (colorPercent / 100) * 6;
      needsFilterUpdate = true;
    }

    if (params.depth !== undefined) {
      this.depth = Math.max(0, Math.min(100, params.depth)) / 100;
    }

    if (params.dcFilter !== undefined) {
      this.dcFilterEnabled = params.dcFilter;
      needsFilterUpdate = true;
    }

    if (params.output !== undefined) {
      // Output in dB: -24 to +24
      const outputDb = Math.max(-24, Math.min(24, params.output));
      this.output = Math.pow(10, outputDb / 20);
    }

    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(100, params.mix)) / 100;
    }

    if (needsFilterUpdate) {
      this.updateFilters();
    }
  }

  /**
   * Update filter coefficients
   */
  updateFilters() {
    for (let ch = 0; ch < 2; ch++) {
      // DC filter (5Hz highpass or very low for bypass)
      if (this.dcFilterEnabled) {
        this.dcFilters[ch].setHighpass(5, 0.7071, sampleRate);
      } else {
        this.dcFilters[ch].setHighpass(0.1, 0.7071, sampleRate);
      }

      // Color filter (peaking for harmonic emphasis)
      this.colorFilters[ch].setPeaking(this.colorFreq, 2.0, this.colorGain, sampleRate);
    }
  }

  /**
   * Saturation algorithms
   */
  warmSaturation(x) {
    // Soft tanh saturation - warm, musical
    return Math.tanh(x);
  }

  digitalSaturation(x) {
    // Hard clipping - aggressive digital sound
    return Math.max(-1, Math.min(1, x));
  }

  analogSaturation(x) {
    // Asymmetric soft clip - simulates analog circuits
    const biasedX = x + 0.1; // Slight asymmetry
    return Math.tanh(biasedX);
  }

  clipSaturation(x) {
    // Very hard clip - extreme distortion
    if (x > 0.1) return 1;
    if (x < -0.1) return -1;
    return x * 10;
  }

  foldbackSaturation(x) {
    // Foldback distortion - complex harmonics
    return Math.abs((x + 1) % 4 - 2) - 1;
  }

  sineFoldSaturation(x) {
    // Sine folding - musical harmonics
    return Math.sin(x * Math.PI);
  }

  /**
   * Apply saturation based on type
   */
  applySaturation(input) {
    const driven = input * this.drive;
    let saturated;

    switch (this.type) {
      case 'warm':
        saturated = this.warmSaturation(driven);
        break;
      case 'digital':
        saturated = this.digitalSaturation(driven);
        break;
      case 'analog':
        saturated = this.analogSaturation(driven);
        break;
      case 'clip':
        saturated = this.clipSaturation(driven);
        break;
      case 'foldback':
        saturated = this.foldbackSaturation(driven);
        break;
      case 'sine-fold':
        saturated = this.sineFoldSaturation(driven);
        break;
      default:
        saturated = this.warmSaturation(driven);
    }

    // Apply depth (blend between dry and saturated)
    return input + (saturated - input) * this.depth;
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

    const numChannels = Math.min(input.length, output.length);

    for (let channel = 0; channel < numChannels; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      const dcFilter = this.dcFilters[channel];
      const colorFilter = this.colorFilters[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        const dry = inputChannel[i];
        let wet = dry;

        // Processing chain: saturation → DC filter → color filter → output gain
        wet = this.applySaturation(wet);
        wet = dcFilter.process(wet);
        wet = colorFilter.process(wet);
        wet *= this.output;

        // Mix dry/wet
        outputChannel[i] = dry * (1 - this.mix) + wet * this.mix;
      }
    }

    return true;
  }
}

registerProcessor('saturation-processor', SaturationProcessor);
