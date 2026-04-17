/**
 * DSP Utilities for AudioWorklet Processors
 * Provides common DSP building blocks for audio effects
 *
 * @version 1.0.0
 * @author Agent 2 (EQ Plugins)
 */

/**
 * Biquad Filter - Universal second-order IIR filter
 * Implements various filter types: lowpass, highpass, peaking, etc.
 */
class BiquadFilter {
  constructor() {
    // Filter coefficients
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;
    this.a1 = 0;
    this.a2 = 0;

    // State variables (for each channel)
    this.z1 = 0;
    this.z2 = 0;
  }

  /**
   * Process a single sample through the filter
   * @param {number} input - Input sample
   * @returns {number} Filtered output sample
   */
  process(input) {
    // Direct Form II transposed structure
    const output = input * this.b0 + this.z1;
    this.z1 = input * this.b1 - output * this.a1 + this.z2;
    this.z2 = input * this.b2 - output * this.a2;
    return output;
  }

  /**
   * Set filter to lowpass type
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} Q - Resonance (0.1 to 10)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setLowpass(frequency, Q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * Q);

    const a0 = 1 + alpha;
    this.b0 = (1 - cosOmega) / (2 * a0);
    this.b1 = (1 - cosOmega) / a0;
    this.b2 = (1 - cosOmega) / (2 * a0);
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Set filter to highpass type
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} Q - Resonance (0.1 to 10)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setHighpass(frequency, Q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * Q);

    const a0 = 1 + alpha;
    this.b0 = (1 + cosOmega) / (2 * a0);
    this.b1 = -(1 + cosOmega) / a0;
    this.b2 = (1 + cosOmega) / (2 * a0);
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Set filter to peaking (parametric) type
   * @param {number} frequency - Center frequency in Hz
   * @param {number} Q - Bandwidth (0.1 to 10)
   * @param {number} gainDb - Gain in dB (-15 to +15)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setPeaking(frequency, Q, gainDb, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const A = Math.pow(10, gainDb / 40);  // sqrt of linear gain
    const alpha = sinOmega / (2 * Q);

    const a0 = 1 + alpha / A;
    this.b0 = (1 + alpha * A) / a0;
    this.b1 = (-2 * cosOmega) / a0;
    this.b2 = (1 - alpha * A) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha / A) / a0;
  }

  /**
   * Set filter to low shelf type
   * @param {number} frequency - Corner frequency in Hz
   * @param {number} Q - Slope (0.1 to 10)
   * @param {number} gainDb - Gain in dB (-15 to +15)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setLowShelf(frequency, Q, gainDb, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const A = Math.pow(10, gainDb / 40);
    const beta = Math.sqrt(A) / Q;

    const a0 = (A + 1) + (A - 1) * cosOmega + beta * sinOmega;
    this.b0 = (A * ((A + 1) - (A - 1) * cosOmega + beta * sinOmega)) / a0;
    this.b1 = (2 * A * ((A - 1) - (A + 1) * cosOmega)) / a0;
    this.b2 = (A * ((A + 1) - (A - 1) * cosOmega - beta * sinOmega)) / a0;
    this.a1 = (-2 * ((A - 1) + (A + 1) * cosOmega)) / a0;
    this.a2 = ((A + 1) + (A - 1) * cosOmega - beta * sinOmega) / a0;
  }

  /**
   * Set filter to high shelf type
   * @param {number} frequency - Corner frequency in Hz
   * @param {number} Q - Slope (0.1 to 10)
   * @param {number} gainDb - Gain in dB (-15 to +15)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setHighShelf(frequency, Q, gainDb, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const A = Math.pow(10, gainDb / 40);
    const beta = Math.sqrt(A) / Q;

    const a0 = (A + 1) - (A - 1) * cosOmega + beta * sinOmega;
    this.b0 = (A * ((A + 1) + (A - 1) * cosOmega + beta * sinOmega)) / a0;
    this.b1 = (-2 * A * ((A - 1) + (A + 1) * cosOmega)) / a0;
    this.b2 = (A * ((A + 1) + (A - 1) * cosOmega - beta * sinOmega)) / a0;
    this.a1 = (2 * ((A - 1) - (A + 1) * cosOmega)) / a0;
    this.a2 = ((A + 1) - (A - 1) * cosOmega - beta * sinOmega) / a0;
  }

  /**
   * Set filter to bandpass type
   * @param {number} frequency - Center frequency in Hz
   * @param {number} Q - Bandwidth (0.1 to 10)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setBandpass(frequency, Q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * Q);

    const a0 = 1 + alpha;
    this.b0 = alpha / a0;
    this.b1 = 0;
    this.b2 = -alpha / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Set filter to notch type
   * @param {number} frequency - Notch frequency in Hz
   * @param {number} Q - Bandwidth (0.1 to 10)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setNotch(frequency, Q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * Q);

    const a0 = 1 + alpha;
    this.b0 = 1 / a0;
    this.b1 = (-2 * cosOmega) / a0;
    this.b2 = 1 / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Set filter to allpass type
   * @param {number} frequency - Center frequency in Hz
   * @param {number} Q - Bandwidth (0.1 to 10)
   * @param {number} sampleRate - Sample rate in Hz
   */
  setAllpass(frequency, Q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * Q);

    const a0 = 1 + alpha;
    this.b0 = (1 - alpha) / a0;
    this.b1 = (-2 * cosOmega) / a0;
    this.b2 = 1;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Reset filter state (clear delay line)
   */
  reset() {
    this.z1 = 0;
    this.z2 = 0;
  }
}

/**
 * One-Pole Filter - Simple first-order lowpass filter
 * Useful for smoothing parameter changes and envelope following
 */
class OnePoleFilter {
  constructor(cutoffFrequency, sampleRate) {
    this.state = 0;
    this.setCutoff(cutoffFrequency, sampleRate);
  }

  /**
   * Set cutoff frequency
   * @param {number} cutoffFrequency - Cutoff frequency in Hz
   * @param {number} sampleRate - Sample rate in Hz
   */
  setCutoff(cutoffFrequency, sampleRate) {
    const omega = 2 * Math.PI * cutoffFrequency / sampleRate;
    this.coefficient = 1 - Math.exp(-omega);
  }

  /**
   * Process a single sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output sample
   */
  process(input) {
    this.state += this.coefficient * (input - this.state);
    return this.state;
  }

  /**
   * Reset filter state
   */
  reset() {
    this.state = 0;
  }
}

/**
 * Delay Line - Circular buffer for delay effects
 */
class DelayLine {
  constructor(maxDelaySeconds, sampleRate) {
    this.bufferLength = Math.ceil(maxDelaySeconds * sampleRate);
    this.buffer = new Float32Array(this.bufferLength);
    this.writeIndex = 0;
  }

  /**
   * Write a sample to the delay line
   * @param {number} sample - Input sample
   */
  write(sample) {
    this.buffer[this.writeIndex] = sample;
    this.writeIndex = (this.writeIndex + 1) % this.bufferLength;
  }

  /**
   * Read a delayed sample (integer delay)
   * @param {number} delaySamples - Delay in samples
   * @returns {number} Delayed sample
   */
  read(delaySamples) {
    const readIndex = (this.writeIndex - Math.floor(delaySamples) + this.bufferLength) % this.bufferLength;
    return this.buffer[readIndex];
  }

  /**
   * Read a delayed sample with linear interpolation
   * @param {number} delaySamples - Delay in samples (can be fractional)
   * @returns {number} Interpolated delayed sample
   */
  readInterpolated(delaySamples) {
    const intDelay = Math.floor(delaySamples);
    const fracDelay = delaySamples - intDelay;

    const index1 = (this.writeIndex - intDelay + this.bufferLength - 1) % this.bufferLength;
    const index2 = (this.writeIndex - intDelay + this.bufferLength) % this.bufferLength;

    const sample1 = this.buffer[index1];
    const sample2 = this.buffer[index2];

    // Linear interpolation
    return sample1 + fracDelay * (sample2 - sample1);
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
 * Envelope Follower - Tracks amplitude envelope of a signal
 */
class EnvelopeFollower {
  constructor(attackTime, releaseTime, sampleRate) {
    this.attackCoeff = 0;
    this.releaseCoeff = 0;
    this.envelope = 0;
    this.setTimes(attackTime, releaseTime, sampleRate);
  }

  /**
   * Set attack and release times
   * @param {number} attackTime - Attack time in seconds
   * @param {number} releaseTime - Release time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  setTimes(attackTime, releaseTime, sampleRate) {
    this.attackCoeff = Math.exp(-1 / (attackTime * sampleRate));
    this.releaseCoeff = Math.exp(-1 / (releaseTime * sampleRate));
  }

  /**
   * Process a single sample
   * @param {number} input - Input sample
   * @returns {number} Envelope value (0 to 1)
   */
  process(input) {
    const inputLevel = Math.abs(input);

    if (inputLevel > this.envelope) {
      // Attack phase
      this.envelope = this.attackCoeff * this.envelope + (1 - this.attackCoeff) * inputLevel;
    } else {
      // Release phase
      this.envelope = this.releaseCoeff * this.envelope + (1 - this.releaseCoeff) * inputLevel;
    }

    return this.envelope;
  }

  /**
   * Reset envelope state
   */
  reset() {
    this.envelope = 0;
  }
}

/**
 * Utility functions
 */

// Convert dB to linear gain
function dbToGain(db) {
  return Math.pow(10, db / 20);
}

// Convert linear gain to dB
function gainToDb(gain) {
  return 20 * Math.log10(Math.max(gain, 0.00001)); // Avoid log(0)
}

// Clamp value between min and max
function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
