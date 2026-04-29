/**
 * DSP Utilities for AudioWorklet Processors
 * Shared utilities for delay, filtering, and signal processing
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 1.0.0
 */

/**
 * DelayLine - Circular buffer for audio delay with interpolation
 *
 * Provides:
 * - Variable delay with linear interpolation
 * - Circular buffer implementation for efficiency
 * - Maximum delay time specified at construction
 */
class DelayLine {
  /**
   * Create a delay line
   * @param {number} maxDelaySeconds - Maximum delay time in seconds
   * @param {number} sampleRate - Audio sample rate
   */
  constructor(maxDelaySeconds, sampleRate) {
    this.sampleRate = sampleRate;
    this.maxDelaySamples = Math.ceil(maxDelaySeconds * sampleRate);

    // Circular buffer
    this.buffer = new Float32Array(this.maxDelaySamples);
    this.writeIndex = 0;
  }

  /**
   * Write a sample to the delay line
   * @param {number} sample - Sample value to write
   */
  write(sample) {
    this.buffer[this.writeIndex] = sample;
    this.writeIndex = (this.writeIndex + 1) % this.maxDelaySamples;
  }

  /**
   * Read from delay line (no interpolation)
   * @param {number} delaySamples - Delay time in samples
   * @returns {number} Delayed sample
   */
  read(delaySamples) {
    // Clamp delay to valid range
    delaySamples = Math.max(0, Math.min(this.maxDelaySamples - 1, delaySamples));

    // Calculate read position
    let readIndex = this.writeIndex - Math.floor(delaySamples);

    // Wrap around if negative
    if (readIndex < 0) {
      readIndex += this.maxDelaySamples;
    }

    return this.buffer[readIndex];
  }

  /**
   * Read from delay line with linear interpolation
   * @param {number} delaySamples - Delay time in samples (can be fractional)
   * @returns {number} Interpolated delayed sample
   */
  readInterpolated(delaySamples) {
    // Clamp delay to valid range
    delaySamples = Math.max(0, Math.min(this.maxDelaySamples - 1, delaySamples));

    // Split into integer and fractional parts
    const delayInt = Math.floor(delaySamples);
    const delayFrac = delaySamples - delayInt;

    // Calculate read positions
    let readIndex1 = this.writeIndex - delayInt;
    let readIndex2 = readIndex1 - 1;

    // Wrap around if negative
    if (readIndex1 < 0) readIndex1 += this.maxDelaySamples;
    if (readIndex2 < 0) readIndex2 += this.maxDelaySamples;

    // Linear interpolation
    const sample1 = this.buffer[readIndex1];
    const sample2 = this.buffer[readIndex2];

    return sample1 + (sample2 - sample1) * delayFrac;
  }

  /**
   * Clear the delay line
   */
  clear() {
    this.buffer.fill(0);
    this.writeIndex = 0;
  }
}

/**
 * OnePoleFilter - Simple one-pole lowpass/highpass filter
 *
 * Used for damping in feedback paths and smoothing parameter changes
 */
class OnePoleFilter {
  /**
   * Create a one-pole filter
   * @param {number} cutoffFreq - Cutoff frequency in Hz
   * @param {number} sampleRate - Audio sample rate
   * @param {string} type - Filter type: 'lowpass' or 'highpass'
   */
  constructor(cutoffFreq, sampleRate, type = 'lowpass') {
    this.sampleRate = sampleRate;
    this.type = type;
    this.z1 = 0; // State variable

    this.setCutoff(cutoffFreq);
  }

  /**
   * Set cutoff frequency
   * @param {number} cutoffFreq - Cutoff frequency in Hz
   */
  setCutoff(cutoffFreq) {
    // Calculate coefficient from cutoff frequency
    const omega = 2 * Math.PI * cutoffFreq / this.sampleRate;
    this.b1 = Math.exp(-omega);
    this.a0 = 1 - this.b1;
  }

  /**
   * Process a single sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    if (this.type === 'lowpass') {
      // Lowpass: y[n] = a0 * x[n] + b1 * y[n-1]
      const output = this.a0 * input + this.b1 * this.z1;
      this.z1 = output;
      return output;
    } else {
      // Highpass: y[n] = x[n] - lowpass(x[n])
      const lowpass = this.a0 * input + this.b1 * this.z1;
      this.z1 = lowpass;
      return input - lowpass;
    }
  }

  /**
   * Reset filter state
   */
  reset() {
    this.z1 = 0;
  }
}

/**
 * BiquadFilter - Second-order IIR filter
 *
 * More sophisticated than one-pole, supports various filter types
 */
class BiquadFilter {
  constructor() {
    // State variables
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;

    // Coefficients
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;
    this.a1 = 0;
    this.a2 = 0;
  }

  /**
   * Set as lowpass filter
   * @param {number} freq - Cutoff frequency
   * @param {number} q - Q factor
   * @param {number} sampleRate - Sample rate
   */
  setLowpass(freq, q, sampleRate) {
    const omega = 2 * Math.PI * freq / sampleRate;
    const sin = Math.sin(omega);
    const cos = Math.cos(omega);
    const alpha = sin / (2 * q);

    const b0 = (1 - cos) / 2;
    const b1 = 1 - cos;
    const b2 = (1 - cos) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * cos;
    const a2 = 1 - alpha;

    // Normalize
    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
  }

  /**
   * Set as highpass filter
   * @param {number} freq - Cutoff frequency
   * @param {number} q - Q factor
   * @param {number} sampleRate - Sample rate
   */
  setHighpass(freq, q, sampleRate) {
    const omega = 2 * Math.PI * freq / sampleRate;
    const sin = Math.sin(omega);
    const cos = Math.cos(omega);
    const alpha = sin / (2 * q);

    const b0 = (1 + cos) / 2;
    const b1 = -(1 + cos);
    const b2 = (1 + cos) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * cos;
    const a2 = 1 - alpha;

    // Normalize
    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
  }

  /**
   * Set as peaking EQ filter
   * @param {number} freq - Center frequency
   * @param {number} q - Q factor
   * @param {number} gain - Gain in dB
   * @param {number} sampleRate - Sample rate
   */
  setPeaking(freq, q, gain, sampleRate) {
    const omega = 2 * Math.PI * freq / sampleRate;
    const sin = Math.sin(omega);
    const cos = Math.cos(omega);
    const A = Math.pow(10, gain / 40);
    const alpha = sin / (2 * q);

    const b0 = 1 + alpha * A;
    const b1 = -2 * cos;
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * cos;
    const a2 = 1 - alpha / A;

    // Normalize
    this.b0 = b0 / a0;
    this.b1 = b1 / a0;
    this.b2 = b2 / a0;
    this.a1 = a1 / a0;
    this.a2 = a2 / a0;
  }

  /**
   * Process a single sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    // Direct Form I implementation
    const output = this.b0 * input + this.b1 * this.x1 + this.b2 * this.x2
                 - this.a1 * this.y1 - this.a2 * this.y2;

    // Update state
    this.x2 = this.x1;
    this.x1 = input;
    this.y2 = this.y1;
    this.y1 = output;

    return output;
  }

  /**
   * Reset filter state
   */
  reset() {
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }
}

/**
 * LFO - Low Frequency Oscillator
 *
 * Used for modulation effects
 */
class LFO {
  /**
   * Create an LFO
   * @param {number} rate - Rate in Hz
   * @param {number} depth - Modulation depth (0-1)
   * @param {string} waveform - Waveform type: 'sine', 'triangle', 'square', 'saw'
   */
  constructor(rate, depth, waveform = 'sine') {
    this.rate = rate;
    this.depth = depth;
    this.waveform = waveform;
    this.phase = 0;
  }

  /**
   * Set rate
   * @param {number} rate - Rate in Hz
   */
  setRate(rate) {
    this.rate = rate;
  }

  /**
   * Set depth
   * @param {number} depth - Depth (0-1)
   */
  setDepth(depth) {
    this.depth = depth;
  }

  /**
   * Set waveform
   * @param {string} waveform - Waveform type
   */
  setWaveform(waveform) {
    this.waveform = waveform;
  }

  /**
   * Process and advance LFO by one sample
   * @param {number} sampleRate - Sample rate
   * @returns {number} LFO output value (-depth to +depth)
   */
  process(sampleRate) {
    const phaseIncrement = this.rate / sampleRate;
    this.phase += phaseIncrement;
    if (this.phase >= 1.0) this.phase -= 1.0;

    let value;
    switch (this.waveform) {
      case 'sine':
        value = Math.sin(this.phase * 2 * Math.PI);
        break;
      case 'triangle':
        value = this.phase < 0.5
          ? (this.phase * 4 - 1)
          : (3 - this.phase * 4);
        break;
      case 'square':
        value = this.phase < 0.5 ? 1 : -1;
        break;
      case 'saw':
        value = this.phase * 2 - 1;
        break;
      default:
        value = Math.sin(this.phase * 2 * Math.PI);
    }

    return value * this.depth;
  }

  /**
   * Reset LFO phase
   */
  reset() {
    this.phase = 0;
  }
}

// Export utilities (for worklet environment, these are accessed via importScripts)
if (typeof globalThis !== 'undefined') {
  globalThis.DelayLine = DelayLine;
  globalThis.OnePoleFilter = OnePoleFilter;
  globalThis.BiquadFilter = BiquadFilter;
  globalThis.LFO = LFO;
}
