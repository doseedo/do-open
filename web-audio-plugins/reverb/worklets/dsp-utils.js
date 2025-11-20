/**
 * DSP Utilities for AudioWorklet Processors
 *
 * Shared digital signal processing components for reverb effects:
 * - DelayLine: Circular buffer with interpolation
 * - OnePoleFilter: Simple lowpass/highpass filter
 * - BiquadFilter: Parametric filter (peaking, lowpass, highpass, etc.)
 * - AllpassFilter: For diffusion in reverb
 *
 * @author Agent 6: Reverb Plugins
 * @version 1.0.0
 */

/**
 * DelayLine - Circular buffer for delay effects
 * Supports fractional delay with linear interpolation
 */
class DelayLine {
  /**
   * Create a delay line
   * @param {number} maxDelaySeconds - Maximum delay time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(maxDelaySeconds, sampleRate) {
    this.sampleRate = sampleRate;
    this.maxDelay = maxDelaySeconds;
    this.bufferSize = Math.ceil(maxDelaySeconds * sampleRate);
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
    const delayInt = Math.floor(delaySamples);
    const readIndex = (this.writeIndex - delayInt - 1 + this.bufferSize) % this.bufferSize;
    return this.buffer[readIndex];
  }

  /**
   * Read a delayed sample with linear interpolation
   * @param {number} delaySamples - Delay in samples (can be fractional)
   * @returns {number} Interpolated delayed sample
   */
  readInterpolated(delaySamples) {
    const delayInt = Math.floor(delaySamples);
    const delayFrac = delaySamples - delayInt;

    const readIndex1 = (this.writeIndex - delayInt - 1 + this.bufferSize) % this.bufferSize;
    const readIndex2 = (readIndex1 - 1 + this.bufferSize) % this.bufferSize;

    const sample1 = this.buffer[readIndex1];
    const sample2 = this.buffer[readIndex2];

    // Linear interpolation
    return sample1 + delayFrac * (sample2 - sample1);
  }

  /**
   * Clear the delay line buffer
   */
  clear() {
    this.buffer.fill(0);
    this.writeIndex = 0;
  }
}

/**
 * OnePoleFilter - Simple first-order lowpass filter
 * Used for smoothing and damping
 */
class OnePoleFilter {
  /**
   * Create a one-pole filter
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(frequency, sampleRate) {
    this.sampleRate = sampleRate;
    this.z1 = 0; // Previous output
    this.setFrequency(frequency);
  }

  /**
   * Set cutoff frequency
   * @param {number} frequency - Cutoff frequency in Hz
   */
  setFrequency(frequency) {
    this.frequency = frequency;
    // Calculate coefficient for lowpass
    const omega = 2 * Math.PI * frequency / this.sampleRate;
    this.a = Math.min(omega, 1); // Clamp for stability
  }

  /**
   * Process a sample (lowpass)
   * @param {number} input - Input sample
   * @returns {number} Filtered sample
   */
  process(input) {
    this.z1 = this.z1 + this.a * (input - this.z1);
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
 * BiquadFilter - Second-order IIR filter
 * Supports lowpass, highpass, bandpass, peaking, and shelving
 */
class BiquadFilter {
  constructor() {
    // Filter state
    this.x1 = 0; // input[n-1]
    this.x2 = 0; // input[n-2]
    this.y1 = 0; // output[n-1]
    this.y2 = 0; // output[n-2]

    // Filter coefficients
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;
    this.a1 = 0;
    this.a2 = 0;
  }

  /**
   * Configure as lowpass filter
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} q - Quality factor
   * @param {number} sampleRate - Sample rate in Hz
   */
  setLowpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const alpha = Math.sin(omega) / (2 * q);

    const a0 = 1 + alpha;
    this.b0 = ((1 - cosOmega) / 2) / a0;
    this.b1 = (1 - cosOmega) / a0;
    this.b2 = ((1 - cosOmega) / 2) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Configure as highpass filter
   * @param {number} frequency - Cutoff frequency in Hz
   * @param {number} q - Quality factor
   * @param {number} sampleRate - Sample rate in Hz
   */
  setHighpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const alpha = Math.sin(omega) / (2 * q);

    const a0 = 1 + alpha;
    this.b0 = ((1 + cosOmega) / 2) / a0;
    this.b1 = -(1 + cosOmega) / a0;
    this.b2 = ((1 + cosOmega) / 2) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha) / a0;
  }

  /**
   * Configure as peaking filter
   * @param {number} frequency - Center frequency in Hz
   * @param {number} q - Quality factor
   * @param {number} gainDb - Gain in dB
   * @param {number} sampleRate - Sample rate in Hz
   */
  setPeaking(frequency, q, gainDb, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const cosOmega = Math.cos(omega);
    const sinOmega = Math.sin(omega);
    const A = Math.pow(10, gainDb / 40);
    const alpha = sinOmega / (2 * q);

    const a0 = 1 + alpha / A;
    this.b0 = (1 + alpha * A) / a0;
    this.b1 = (-2 * cosOmega) / a0;
    this.b2 = (1 - alpha * A) / a0;
    this.a1 = (-2 * cosOmega) / a0;
    this.a2 = (1 - alpha / A) / a0;
  }

  /**
   * Process a sample through the filter
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    // Direct Form II implementation
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
 * AllpassFilter - For phase-based diffusion
 * Used in reverb for increasing echo density
 */
class AllpassFilter {
  /**
   * Create an allpass filter
   * @param {number} delaySeconds - Delay time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(delaySeconds, sampleRate) {
    this.delayLine = new DelayLine(delaySeconds + 0.001, sampleRate);
    this.delaySamples = delaySeconds * sampleRate;
    this.feedback = 0.5; // Default feedback coefficient
  }

  /**
   * Set feedback coefficient
   * @param {number} feedback - Feedback amount (-1 to 1)
   */
  setFeedback(feedback) {
    this.feedback = Math.max(-0.99, Math.min(0.99, feedback));
  }

  /**
   * Process a sample through the allpass filter
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    const delayed = this.delayLine.read(this.delaySamples);
    const output = -input + delayed;
    this.delayLine.write(input + delayed * this.feedback);
    return output;
  }

  /**
   * Clear the allpass filter
   */
  clear() {
    this.delayLine.clear();
  }
}

/**
 * CombFilter - Feedback delay for reverb
 * Creates resonances at harmonic frequencies
 */
class CombFilter {
  /**
   * Create a comb filter
   * @param {number} delaySeconds - Delay time in seconds
   * @param {number} sampleRate - Sample rate in Hz
   */
  constructor(delaySeconds, sampleRate) {
    this.delayLine = new DelayLine(delaySeconds + 0.001, sampleRate);
    this.delaySamples = delaySeconds * sampleRate;
    this.feedback = 0.5;
    this.dampingFilter = new OnePoleFilter(5000, sampleRate);
  }

  /**
   * Set feedback coefficient
   * @param {number} feedback - Feedback amount (0 to 1)
   */
  setFeedback(feedback) {
    this.feedback = Math.max(0, Math.min(0.98, feedback));
  }

  /**
   * Set damping frequency
   * @param {number} frequency - Damping cutoff in Hz
   */
  setDamping(frequency) {
    this.dampingFilter.setFrequency(frequency);
  }

  /**
   * Process a sample through the comb filter
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    const delayed = this.delayLine.read(this.delaySamples);
    const damped = this.dampingFilter.process(delayed);
    this.delayLine.write(input + damped * this.feedback);
    return delayed;
  }

  /**
   * Clear the comb filter
   */
  clear() {
    this.delayLine.clear();
    this.dampingFilter.reset();
  }
}

// Export classes if in module context
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    DelayLine,
    OnePoleFilter,
    BiquadFilter,
    AllpassFilter,
    CombFilter
  };
}
