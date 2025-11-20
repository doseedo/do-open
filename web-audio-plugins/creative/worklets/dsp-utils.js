/**
 * DSP Utilities for AudioWorklet Processors
 * Shared building blocks for all creative effects
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

/**
 * Low Frequency Oscillator
 * Generates modulation waveforms for effects
 */
class LFO {
  constructor(rate = 1.0, depth = 1.0, waveform = 'sine') {
    this.rate = rate; // Hz
    this.depth = depth; // 0-1
    this.waveform = waveform; // 'sine', 'triangle', 'square', 'saw'
    this.phase = 0;
  }

  setRate(rate) {
    this.rate = rate;
  }

  setDepth(depth) {
    this.depth = depth;
  }

  setWaveform(waveform) {
    this.waveform = waveform;
  }

  /**
   * Process one sample and return LFO output
   * @param {number} sampleRate - Current sample rate
   * @returns {number} LFO value in range [-depth, +depth]
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

  /**
   * Reset phase to start
   */
  reset() {
    this.phase = 0;
  }
}

/**
 * Delay Line
 * Circular buffer for delay effects
 */
class DelayLine {
  constructor(maxDelayTime, sampleRate) {
    this.maxDelayTime = maxDelayTime; // seconds
    this.sampleRate = sampleRate;
    this.bufferSize = Math.ceil(maxDelayTime * sampleRate);
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
   * Read a sample from delay line (no interpolation)
   * @param {number} delaySamples - Delay in samples
   * @returns {number} Delayed sample
   */
  read(delaySamples) {
    const delayInt = Math.floor(delaySamples);
    const readIndex = (this.writeIndex - delayInt + this.bufferSize) % this.bufferSize;
    return this.buffer[readIndex] || 0;
  }

  /**
   * Read a sample with linear interpolation
   * @param {number} delaySamples - Delay in samples (can be fractional)
   * @returns {number} Interpolated delayed sample
   */
  readInterpolated(delaySamples) {
    const delayInt = Math.floor(delaySamples);
    const delayFrac = delaySamples - delayInt;

    const index1 = (this.writeIndex - delayInt + this.bufferSize) % this.bufferSize;
    const index2 = (this.writeIndex - delayInt - 1 + this.bufferSize) % this.bufferSize;

    const sample1 = this.buffer[index1] || 0;
    const sample2 = this.buffer[index2] || 0;

    // Linear interpolation
    return sample1 + delayFrac * (sample2 - sample1);
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
 * One-Pole Filter (Simple Lowpass)
 * Used for smoothing parameter changes and gain reduction
 */
class OnePoleFilter {
  constructor(cutoffFreq, sampleRate) {
    this.sampleRate = sampleRate;
    this.state = 0;
    this.setCutoff(cutoffFreq);
  }

  setCutoff(cutoffFreq) {
    this.cutoffFreq = cutoffFreq;
    // Calculate coefficient
    const omega = 2 * Math.PI * cutoffFreq / this.sampleRate;
    this.coefficient = Math.exp(-omega);
  }

  /**
   * Process one sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output
   */
  process(input) {
    this.state = this.coefficient * this.state + (1 - this.coefficient) * input;
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
 * Biquad Filter
 * Full-featured biquad filter for EQ and filtering
 */
class BiquadFilter {
  constructor() {
    // Biquad coefficients
    this.a0 = 1;
    this.a1 = 0;
    this.a2 = 0;
    this.b0 = 1;
    this.b1 = 0;
    this.b2 = 0;

    // State variables
    this.x1 = 0;
    this.x2 = 0;
    this.y1 = 0;
    this.y2 = 0;
  }

  /**
   * Set as lowpass filter
   */
  setLowpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * q);

    this.b0 = (1 - cosOmega) / 2;
    this.b1 = 1 - cosOmega;
    this.b2 = (1 - cosOmega) / 2;
    this.a0 = 1 + alpha;
    this.a1 = -2 * cosOmega;
    this.a2 = 1 - alpha;

    this.normalizeCoefficients();
  }

  /**
   * Set as highpass filter
   */
  setHighpass(frequency, q, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * q);

    this.b0 = (1 + cosOmega) / 2;
    this.b1 = -(1 + cosOmega);
    this.b2 = (1 + cosOmega) / 2;
    this.a0 = 1 + alpha;
    this.a1 = -2 * cosOmega;
    this.a2 = 1 - alpha;

    this.normalizeCoefficients();
  }

  /**
   * Set as peaking EQ filter
   */
  setPeaking(frequency, q, gainDb, sampleRate) {
    const omega = 2 * Math.PI * frequency / sampleRate;
    const sinOmega = Math.sin(omega);
    const cosOmega = Math.cos(omega);
    const alpha = sinOmega / (2 * q);
    const A = Math.pow(10, gainDb / 40);

    this.b0 = 1 + alpha * A;
    this.b1 = -2 * cosOmega;
    this.b2 = 1 - alpha * A;
    this.a0 = 1 + alpha / A;
    this.a1 = -2 * cosOmega;
    this.a2 = 1 - alpha / A;

    this.normalizeCoefficients();
  }

  /**
   * Normalize coefficients by a0
   */
  normalizeCoefficients() {
    this.b0 /= this.a0;
    this.b1 /= this.a0;
    this.b2 /= this.a0;
    this.a1 /= this.a0;
    this.a2 /= this.a0;
    this.a0 = 1;
  }

  /**
   * Process one sample
   * @param {number} input - Input sample
   * @returns {number} Filtered output
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
 * Envelope Follower
 * Tracks amplitude envelope of a signal
 */
class EnvelopeFollower {
  constructor(attack, release, sampleRate) {
    this.sampleRate = sampleRate;
    this.envelope = 0;
    this.setAttack(attack);
    this.setRelease(release);
  }

  setAttack(attack) {
    this.attack = attack;
    this.attackCoeff = Math.exp(-1 / (attack * this.sampleRate));
  }

  setRelease(release) {
    this.release = release;
    this.releaseCoeff = Math.exp(-1 / (release * this.sampleRate));
  }

  /**
   * Process one sample
   * @param {number} input - Input sample
   * @returns {number} Envelope level
   */
  process(input) {
    const inputAbs = Math.abs(input);

    if (inputAbs > this.envelope) {
      // Attack (rising envelope)
      this.envelope = this.attackCoeff * this.envelope + (1 - this.attackCoeff) * inputAbs;
    } else {
      // Release (falling envelope)
      this.envelope = this.releaseCoeff * this.envelope + (1 - this.releaseCoeff) * inputAbs;
    }

    return this.envelope;
  }

  /**
   * Reset envelope
   */
  reset() {
    this.envelope = 0;
  }
}

// Export utilities (Note: in AudioWorklet global scope, there's no module.exports)
// These classes are available globally within the worklet
