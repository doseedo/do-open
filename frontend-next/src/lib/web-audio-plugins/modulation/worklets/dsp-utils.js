/**
 * DSP Utilities for AudioWorklet Processors
 * Shared utilities for modulation and other effects
 *
 * @author Agent 4 (Modulation Plugins)
 * @version 1.0.0
 */

/**
 * Low Frequency Oscillator (LFO)
 * Generates modulation signals at low frequencies
 */
class LFO {
  /**
   * @param {number} rate - Frequency in Hz
   * @param {number} depth - Modulation depth (0-1)
   * @param {string} waveform - Waveform type: 'sine', 'triangle', 'square', 'saw'
   */
  constructor(rate = 1.0, depth = 1.0, waveform = 'sine') {
    this.rate = rate;
    this.depth = depth;
    this.waveform = waveform;
    this.phase = 0;
  }

  /**
   * Set LFO rate
   * @param {number} rate - Frequency in Hz
   */
  setRate(rate) {
    this.rate = rate;
  }

  /**
   * Set LFO depth
   * @param {number} depth - Depth (0-1)
   */
  setDepth(depth) {
    this.depth = depth;
  }

  /**
   * Set waveform type
   * @param {string} waveform - 'sine', 'triangle', 'square', 'saw'
   */
  setWaveform(waveform) {
    this.waveform = waveform;
  }

  /**
   * Process one sample and advance phase
   * @param {number} sampleRate - Current sample rate
   * @returns {number} LFO output value (-depth to +depth)
   */
  process(sampleRate) {
    const phaseIncrement = this.rate / sampleRate;
    this.phase += phaseIncrement;
    if (this.phase >= 1.0) this.phase -= 1.0;
    if (this.phase < 0) this.phase += 1.0;

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
}

/**
 * Delay Line with interpolated reads
 * Circular buffer for delay effects
 */
class DelayLine {
  /**
   * @param {number} maxDelay - Maximum delay time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(maxDelay, sampleRate) {
    this.maxDelay = maxDelay;
    this.sampleRate = sampleRate;
    this.bufferSize = Math.ceil(maxDelay * sampleRate);
    this.buffer = new Float32Array(this.bufferSize);
    this.writeIndex = 0;
  }

  /**
   * Write a sample to the delay line
   * @param {number} sample - Input sample
   */
  write(sample) {
    this.buffer[this.writeIndex] = sample;
    this.writeIndex = (this.writeIndex + 1) % this.bufferSize;
  }

  /**
   * Read a delayed sample (no interpolation)
   * @param {number} delaySamples - Delay in samples
   * @returns {number} Delayed sample
   */
  read(delaySamples) {
    const delay = Math.max(0, Math.min(this.bufferSize - 1, delaySamples));
    let readIndex = this.writeIndex - Math.floor(delay) - 1;
    if (readIndex < 0) readIndex += this.bufferSize;
    return this.buffer[readIndex];
  }

  /**
   * Read a delayed sample with linear interpolation
   * @param {number} delaySamples - Delay in samples (can be fractional)
   * @returns {number} Interpolated delayed sample
   */
  readInterpolated(delaySamples) {
    const delay = Math.max(0, Math.min(this.bufferSize - 1, delaySamples));
    const delaySamplesInt = Math.floor(delay);
    const frac = delay - delaySamplesInt;

    let readIndex1 = this.writeIndex - delaySamplesInt - 1;
    if (readIndex1 < 0) readIndex1 += this.bufferSize;

    let readIndex2 = readIndex1 - 1;
    if (readIndex2 < 0) readIndex2 += this.bufferSize;

    const sample1 = this.buffer[readIndex1];
    const sample2 = this.buffer[readIndex2];

    // Linear interpolation
    return sample1 * (1 - frac) + sample2 * frac;
  }

  /**
   * Clear the delay buffer
   */
  clear() {
    this.buffer.fill(0);
    this.writeIndex = 0;
  }
}

/**
 * One-pole lowpass filter
 * Simple first-order filter for smoothing
 */
class OnePoleFilter {
  /**
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(frequency, sampleRate) {
    this.sampleRate = sampleRate;
    this.z1 = 0;
    this.setFrequency(frequency);
  }

  /**
   * Set filter cutoff frequency
   * @param {number} frequency - Cutoff frequency in Hz
   */
  setFrequency(frequency) {
    this.frequency = frequency;
    const omega = 2 * Math.PI * frequency / this.sampleRate;
    this.b1 = Math.exp(-omega);
    this.a0 = 1 - this.b1;
  }

  /**
   * Process one sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    this.z1 = input * this.a0 + this.z1 * this.b1;
    return this.z1;
  }

  /**
   * Reset filter state
   */
  reset() {
    this.z1 = 0;
  }
}

/**
 * Biquad Filter
 * Second-order IIR filter for EQ and filtering
 */
class BiquadFilter {
  constructor() {
    this.a0 = 1;
    this.a1 = 0;
    this.a2 = 0;
    this.b1 = 0;
    this.b2 = 0;
    this.z1 = 0;
    this.z2 = 0;
  }

  /**
   * Set as lowpass filter
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} q - Q factor
   * @param {number} sampleRate - Sample rate in Hz
   */
  setLowpass(frequency, q, sampleRate) {
    const w0 = 2 * Math.PI * frequency / sampleRate;
    const cosw0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * q);

    const b0 = (1 - cosw0) / 2;
    const b1 = 1 - cosw0;
    const b2 = (1 - cosw0) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * cosw0;
    const a2 = 1 - alpha;

    this.setCoefficients(b0, b1, b2, a0, a1, a2);
  }

  /**
   * Set as highpass filter
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} q - Q factor
   * @param {number} sampleRate - Sample rate in Hz
   */
  setHighpass(frequency, q, sampleRate) {
    const w0 = 2 * Math.PI * frequency / sampleRate;
    const cosw0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * q);

    const b0 = (1 + cosw0) / 2;
    const b1 = -(1 + cosw0);
    const b2 = (1 + cosw0) / 2;
    const a0 = 1 + alpha;
    const a1 = -2 * cosw0;
    const a2 = 1 - alpha;

    this.setCoefficients(b0, b1, b2, a0, a1, a2);
  }

  /**
   * Set as peaking EQ filter
   * @param {number} frequency - Center frequency in Hz
   * @param {number} q - Q factor
   * @param {number} gainDb - Gain in dB
   * @param {number} sampleRate - Sample rate in Hz
   */
  setPeaking(frequency, q, gainDb, sampleRate) {
    const w0 = 2 * Math.PI * frequency / sampleRate;
    const cosw0 = Math.cos(w0);
    const alpha = Math.sin(w0) / (2 * q);
    const A = Math.pow(10, gainDb / 40);

    const b0 = 1 + alpha * A;
    const b1 = -2 * cosw0;
    const b2 = 1 - alpha * A;
    const a0 = 1 + alpha / A;
    const a1 = -2 * cosw0;
    const a2 = 1 - alpha / A;

    this.setCoefficients(b0, b1, b2, a0, a1, a2);
  }

  /**
   * Set filter coefficients
   */
  setCoefficients(b0, b1, b2, a0, a1, a2) {
    this.a0 = b0 / a0;
    this.a1 = b1 / a0;
    this.a2 = b2 / a0;
    this.b1 = a1 / a0;
    this.b2 = a2 / a0;
  }

  /**
   * Process one sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    const output = this.a0 * input + this.a1 * this.z1 + this.a2 * this.z2
                 - this.b1 * this.z1 - this.b2 * this.z2;

    this.z2 = this.z1;
    this.z1 = input;

    return output;
  }

  /**
   * Reset filter state
   */
  reset() {
    this.z1 = 0;
    this.z2 = 0;
  }
}
