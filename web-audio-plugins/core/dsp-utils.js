/**
 * DSP Utilities for AudioWorklet Processors
 *
 * Provides common DSP building blocks:
 * - EnvelopeFollower: Level detection with attack/release
 * - OnePoleFilter: Simple smoothing filter
 * - BiquadFilter: Versatile IIR filter for EQ and filtering
 * - DelayLine: Circular buffer for delays and echoes
 * - Utility functions for dB/gain conversion
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

/**
 * EnvelopeFollower - Tracks the amplitude envelope of a signal
 * Used for level detection in dynamics processors (compressor, gate, etc.)
 */
class EnvelopeFollower {
  /**
   * @param {number} attackTime - Attack time in seconds
   * @param {number} releaseTime - Release time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(attackTime, releaseTime, sampleRate) {
    this.sampleRate = sampleRate;
    this.envelope = 0;

    this.setAttack(attackTime);
    this.setRelease(releaseTime);
  }

  setAttack(attackTime) {
    this.attackTime = Math.max(0.0001, attackTime);
    // Time constant for exponential response
    this.attackCoeff = Math.exp(-1.0 / (this.attackTime * this.sampleRate));
  }

  setRelease(releaseTime) {
    this.releaseTime = Math.max(0.001, releaseTime);
    this.releaseCoeff = Math.exp(-1.0 / (this.releaseTime * this.sampleRate));
  }

  /**
   * Process one sample
   * @param {number} input - Input sample
   * @returns {number} Envelope value
   */
  process(input) {
    const rectified = Math.abs(input);

    // Use attack or release coefficient depending on whether envelope is rising or falling
    if (rectified > this.envelope) {
      // Attack (rising)
      this.envelope = this.attackCoeff * this.envelope + (1 - this.attackCoeff) * rectified;
    } else {
      // Release (falling)
      this.envelope = this.releaseCoeff * this.envelope + (1 - this.releaseCoeff) * rectified;
    }

    return this.envelope;
  }

  reset() {
    this.envelope = 0;
  }
}

/**
 * OnePoleFilter - Simple first-order lowpass filter
 * Used for smoothing parameter changes and gain reduction
 */
class OnePoleFilter {
  /**
   * @param {number} cutoffFreq - Cutoff frequency in Hz
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(cutoffFreq, sampleRate) {
    this.sampleRate = sampleRate;
    this.z1 = 0;

    this.setCutoff(cutoffFreq);
  }

  setCutoff(cutoffFreq) {
    this.cutoffFreq = cutoffFreq;
    // Calculate filter coefficient
    const omega = 2.0 * Math.PI * cutoffFreq / this.sampleRate;
    this.a0 = 1.0 / (1.0 + omega);
    this.b1 = omega * this.a0;
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

  reset() {
    this.z1 = 0;
  }
}

/**
 * BiquadFilter - Second-order IIR filter
 * Supports lowpass, highpass, bandpass, notch, peaking, lowshelf, highshelf
 */
class BiquadFilter {
  constructor() {
    // Filter state
    this.z1 = 0;
    this.z2 = 0;

    // Filter coefficients (initially bypass)
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;
    this.a1 = 0;
    this.a2 = 0;
  }

  /**
   * Configure as lowpass filter
   */
  setLowpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);

    const a0 = 1 + alpha;
    this.b0 = ((1 - cosOmega) / 2) / a0;
    this.b1 = (1 - cosOmega) / a0;
    this.b2 = ((1 - cosOmega) / 2) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Configure as highpass filter
   */
  setHighpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);

    const a0 = 1 + alpha;
    this.b0 = ((1 + cosOmega) / 2) / a0;
    this.b1 = -(1 + cosOmega) / a0;
    this.b2 = ((1 + cosOmega) / 2) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Configure as peaking EQ filter
   */
  setPeaking(frequency, q, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);

    const a0 = 1 + alpha / A;
    this.b0 = (1 + alpha * A) / a0;
    this.b1 = (-2 * cosOmega) / a0;
    this.b2 = (1 - alpha * A) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha / A) / a0;
  }

  /**
   * Configure as bandpass filter
   */
  setBandpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);

    const a0 = 1 + alpha;
    this.b0 = alpha / a0;
    this.b1 = 0;
    this.b2 = -alpha / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Configure as notch filter
   */
  setNotch(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);

    const a0 = 1 + alpha;
    this.b0 = 1 / a0;
    this.b1 = (-2 * cosOmega) / a0;
    this.b2 = 1 / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Configure as low shelf filter
   */
  setLowShelf(frequency, q, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);
    const sqrtA = Math.sqrt(A);

    const a0 = (A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha;
    this.b0 = (A * ((A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha)) / a0;
    this.b1 = (2 * A * ((A - 1) - (A + 1) * cosOmega)) / a0;
    this.b2 = (A * ((A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha)) / a0;
    this.a1 = (-2 * ((A - 1) + (A + 1) * cosOmega)) / a0;
    this.a2 = ((A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha) / a0;
  }

  /**
   * Configure as high shelf filter
   */
  setHighShelf(frequency, q, gainDb, sampleRate) {
    const A = Math.pow(10, gainDb / 40);
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const alpha = sinOmega / (2 * q);
    const sqrtA = Math.sqrt(A);

    const a0 = (A + 1) - (A - 1) * cosOmega + 2 * sqrtA * alpha;
    this.b0 = (A * ((A + 1) + (A - 1) * cosOmega + 2 * sqrtA * alpha)) / a0;
    this.b1 = (-2 * A * ((A - 1) + (A + 1) * cosOmega)) / a0;
    this.b2 = (A * ((A + 1) + (A - 1) * cosOmega - 2 * sqrtA * alpha)) / a0;
    this.a1 = (2 * ((A - 1) - (A + 1) * cosOmega)) / a0;
    this.a2 = ((A + 1) - (A - 1) * cosOmega - 2 * sqrtA * alpha) / a0;
  }

  /**
   * Process one sample through the filter
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    const output = this.b0 * input + this.b1 * this.z1 + this.b2 * this.z2
                   - this.a1 * this.z1 - this.a2 * this.z2;

    // Shift delay line
    this.z2 = this.z1;
    this.z1 = output;

    return output;
  }

  reset() {
    this.z1 = 0;
    this.z2 = 0;
  }
}

/**
 * DelayLine - Circular buffer for delays
 * Supports integer and fractional delay (interpolated)
 */
class DelayLine {
  /**
   * @param {number} maxDelayTime - Maximum delay time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(maxDelayTime, sampleRate) {
    this.sampleRate = sampleRate;
    this.bufferSize = Math.ceil(maxDelayTime * sampleRate) + 1;
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
   * Read a sample from the delay line (integer delay)
   * @param {number} delaySamples - Delay in samples (integer)
   * @returns {number} Delayed sample
   */
  read(delaySamples) {
    delaySamples = Math.floor(delaySamples);
    delaySamples = Math.max(0, Math.min(this.bufferSize - 1, delaySamples));

    const readIndex = (this.writeIndex - delaySamples + this.bufferSize) % this.bufferSize;
    return this.buffer[readIndex];
  }

  /**
   * Read a sample with linear interpolation (fractional delay)
   * @param {number} delaySamples - Delay in samples (can be fractional)
   * @returns {number} Interpolated delayed sample
   */
  readInterpolated(delaySamples) {
    delaySamples = Math.max(0, Math.min(this.bufferSize - 1, delaySamples));

    const intDelay = Math.floor(delaySamples);
    const frac = delaySamples - intDelay;

    const readIndex1 = (this.writeIndex - intDelay + this.bufferSize) % this.bufferSize;
    const readIndex2 = (this.writeIndex - intDelay - 1 + this.bufferSize) % this.bufferSize;

    const sample1 = this.buffer[readIndex1];
    const sample2 = this.buffer[readIndex2];

    // Linear interpolation
    return sample1 + frac * (sample2 - sample1);
  }

  reset() {
    this.buffer.fill(0);
    this.writeIndex = 0;
  }
}

/**
 * Utility functions for dB/gain conversion
 */

function dbToGain(db) {
  return Math.pow(10, db / 20);
}

function gainToDb(gain) {
  return 20 * Math.log10(Math.max(gain, 0.00001)); // Avoid log(0)
}

function linearToDb(linear) {
  return 20 * Math.log10(Math.max(linear, 0.00001));
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// Export for use in AudioWorklet (if supported)
if (typeof globalThis !== 'undefined') {
  globalThis.EnvelopeFollower = EnvelopeFollower;
  globalThis.OnePoleFilter = OnePoleFilter;
  globalThis.BiquadFilter = BiquadFilter;
  globalThis.DelayLine = DelayLine;
  globalThis.dbToGain = dbToGain;
  globalThis.gainToDb = gainToDb;
  globalThis.linearToDb = linearToDb;
  globalThis.clamp = clamp;
}
