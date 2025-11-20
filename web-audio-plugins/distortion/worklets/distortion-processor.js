/**
 * Distortion AudioWorklet Processor
 * Hard clipping distortion with aggressive harmonic generation
 *
 * Features:
 * - Multiple clipping algorithms
 * - Pre/post filtering
 * - DC blocking
 * - High performance processing
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
   * Set as highpass filter
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

/**
 * DC Blocker (simple highpass at very low frequency)
 */
class DCBlocker {
  constructor() {
    this.x1 = 0;
    this.y1 = 0;
  }

  process(input) {
    // First-order highpass at ~5Hz (coefficient = 0.995)
    const output = input - this.x1 + 0.995 * this.y1;
    this.x1 = input;
    this.y1 = output;
    return output;
  }

  reset() {
    this.x1 = 0;
    this.y1 = 0;
  }
}

class DistortionProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.drive = 1.0;
    this.tone = 1000;
    this.toneWidth = 1.0;
    this.filterPosition = 'post';
    this.clipType = 'hard';
    this.output = 1.0;
    this.mix = 1.0;

    // Create filters for each channel
    this.preFilters = [new BiquadFilter(), new BiquadFilter()];
    this.postFilters = [new BiquadFilter(), new BiquadFilter()];
    this.dcBlockers = [new DCBlocker(), new DCBlocker()];

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
      // Drive range: 1 to 50
      const drivePercent = Math.max(0, Math.min(100, params.drive));
      this.drive = 1 + (drivePercent / 100) * 49;
    }

    if (params.tone !== undefined) {
      this.tone = Math.max(20, Math.min(20000, params.tone));
      needsFilterUpdate = true;
    }

    if (params.toneWidth !== undefined) {
      this.toneWidth = Math.max(0.1, Math.min(10, params.toneWidth));
      needsFilterUpdate = true;
    }

    if (params.filterPosition !== undefined) {
      this.filterPosition = params.filterPosition;
    }

    if (params.clipType !== undefined) {
      this.clipType = params.clipType;
    }

    if (params.output !== undefined) {
      // Output in dB: -24 to +24
      const outputDb = Math.max(-24, Math.min(24, params.output));
      // Base gain is 0.5 to compensate for drive
      this.output = 0.5 * Math.pow(10, outputDb / 20);
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
      this.preFilters[ch].setPeaking(this.tone, this.toneWidth, 0, sampleRate);
      this.postFilters[ch].setPeaking(this.tone, this.toneWidth, 0, sampleRate);
    }
  }

  /**
   * Waveshaping functions
   */
  softClip(input) {
    const x = input;
    if (x > 1) return 2/3;
    if (x < -1) return -2/3;
    return x - (x * x * x) / 3;
  }

  hardClip(input) {
    return Math.max(-1, Math.min(1, input));
  }

  tanhDistortion(input) {
    return Math.tanh(input);
  }

  asymmetricDistortion(input) {
    if (input > 0) {
      return Math.min(1, input * 1.5);
    } else {
      return Math.max(-1, input * 0.8);
    }
  }

  foldback(input) {
    const threshold = 1.0;
    if (Math.abs(input) > threshold) {
      const excess = Math.abs(input) - threshold;
      const folded = threshold - (excess % (2 * threshold));
      return input > 0 ? folded : -folded;
    }
    return input;
  }

  /**
   * Apply distortion to a sample
   */
  applyDistortion(input) {
    const driven = input * this.drive;

    switch (this.clipType) {
      case 'hard':
        return this.hardClip(driven);
      case 'soft':
        return this.softClip(driven);
      case 'tanh':
        return this.tanhDistortion(driven);
      case 'asymmetric':
        return this.asymmetricDistortion(driven);
      case 'foldback':
        return this.foldback(driven);
      default:
        return this.hardClip(driven);
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

    const numChannels = Math.min(input.length, output.length);

    for (let channel = 0; channel < numChannels; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      const preFilter = this.preFilters[channel];
      const postFilter = this.postFilters[channel];
      const dcBlocker = this.dcBlockers[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        const dry = inputChannel[i];
        let wet = dry;

        // Apply processing based on filter position
        if (this.filterPosition === 'pre') {
          // Pre-filtering: filter → drive → distortion → DC block
          wet = preFilter.process(wet);
          wet = this.applyDistortion(wet);
          wet = dcBlocker.process(wet);
        } else {
          // Post-filtering: drive → distortion → filter → DC block
          wet = this.applyDistortion(wet);
          wet = postFilter.process(wet);
          wet = dcBlocker.process(wet);
        }

        // Apply output gain
        wet *= this.output;

        // Mix dry/wet
        outputChannel[i] = dry * (1 - this.mix) + wet * this.mix;
      }
    }

    return true;
  }
}

registerProcessor('distortion-processor', DistortionProcessor);
