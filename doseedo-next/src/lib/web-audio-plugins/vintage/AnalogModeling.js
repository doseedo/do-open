/**
 * AnalogModeling - Shared DSP utilities for analog hardware emulation
 *
 * @description
 * Provides common analog modeling algorithms used across vintage plugins:
 * - Saturation/distortion curves (tape, tube, transistor)
 * - Component aging simulation
 * - Noise generation (tape hiss, hum, thermal noise)
 * - Non-linear frequency response
 * - Transformer coloration
 * - Analog imperfections (drift, wow & flutter)
 *
 * Based on research from:
 * - "Virtual Analog Modeling" (Välimäki et al.)
 * - "Digital Signal Processing" (Zölzer)
 * - Analog circuit analysis
 */

export class AnalogModeling {
  /**
   * Tape saturation curve
   * Smooth, warm saturation characteristic of analog tape
   * @param {number} input - Input value (-1 to 1)
   * @param {number} drive - Drive amount (0-1)
   * @returns {number} Saturated output
   */
  static tapeSaturation(input, drive = 0.5) {
    const x = input * (1 + drive * 5);

    // Hyperbolic tangent - smooth tape-like saturation
    return Math.tanh(x);
  }

  /**
   * Tube saturation curve
   * Asymmetric, warm distortion characteristic of vacuum tubes
   * @param {number} input - Input value (-1 to 1)
   * @param {number} drive - Drive amount (0-1)
   * @returns {number} Saturated output
   */
  static tubeSaturation(input, drive = 0.5) {
    const x = input * (1 + drive * 3);

    // Asymmetric tube response
    if (x >= 0) {
      return x / (1 + Math.pow(x, 2));
    } else {
      return x / (1 + Math.pow(Math.abs(x), 1.5));
    }
  }

  /**
   * Transistor saturation (FET/BJT)
   * Hard clipping characteristic of solid-state devices
   * @param {number} input - Input value (-1 to 1)
   * @param {number} drive - Drive amount (0-1)
   * @returns {number} Saturated output
   */
  static transistorSaturation(input, drive = 0.5) {
    const x = input * (1 + drive * 4);

    // Soft knee followed by hard clip
    const threshold = 0.7;

    if (Math.abs(x) < threshold) {
      return x;
    } else {
      const sign = x >= 0 ? 1 : -1;
      const excess = Math.abs(x) - threshold;
      return sign * (threshold + excess / (1 + excess));
    }
  }

  /**
   * Soft clipping (diode-like)
   * Simulates diode clipping circuits
   * @param {number} input - Input value (-1 to 1)
   * @param {number} threshold - Clipping threshold (0-1)
   * @returns {number} Clipped output
   */
  static softClip(input, threshold = 0.7) {
    const x = input / threshold;

    if (Math.abs(x) < 1) {
      return threshold * x;
    } else if (Math.abs(x) < 2) {
      const sign = x >= 0 ? 1 : -1;
      return threshold * sign * (3 - Math.pow(2 - sign * x, 2)) / 3;
    } else {
      return threshold * (x >= 0 ? 1 : -1);
    }
  }

  /**
   * Generate tape hiss noise
   * Pink-ish noise characteristic of analog tape
   * @returns {number} Noise sample (-1 to 1)
   */
  static tapeHiss() {
    // Pink noise approximation (1/f spectrum)
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    const white = Math.random() * 2 - 1;

    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.96900 * b2 + white * 0.1538520;
    b3 = 0.86650 * b3 + white * 0.3104856;
    b4 = 0.55000 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.0168980;

    const pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362;
    b6 = white * 0.115926;

    return pink * 0.11; // Scale down
  }

  /**
   * Generate AC hum (50/60Hz)
   * Simulates power supply hum
   * @param {number} phase - Current phase (0-2π)
   * @param {number} frequency - Hum frequency (50 or 60 Hz)
   * @returns {number} Hum sample
   */
  static acHum(phase, frequency = 60) {
    // Fundamental + harmonics
    const fundamental = Math.sin(phase);
    const second = Math.sin(2 * phase) * 0.3;
    const third = Math.sin(3 * phase) * 0.1;

    return (fundamental + second + third) * 0.005; // Very subtle
  }

  /**
   * Component aging simulation
   * Simulates drift and imperfections in aged components
   * @param {number} value - Input value
   * @param {number} age - Age factor (0-1, 0=new, 1=very old)
   * @returns {number} Aged value
   */
  static componentAging(value, age = 0.5) {
    // Random drift based on age
    const drift = (Math.random() - 0.5) * age * 0.02;

    // Non-linearity increases with age
    const nonlinearity = Math.pow(value, 1 + age * 0.1);

    return value + drift + (nonlinearity - value) * age * 0.1;
  }

  /**
   * Wow & Flutter simulation
   * Simulates tape speed variations
   * @param {number} time - Current time in seconds
   * @param {number} amount - Amount of wow/flutter (0-1)
   * @returns {number} Pitch variation (-1 to 1, in cents)
   */
  static wowAndFlutter(time, amount = 0.5) {
    // Wow: slow speed variations (0.5-4 Hz)
    const wow = Math.sin(2 * Math.PI * 0.7 * time) * 0.5 +
                Math.sin(2 * Math.PI * 1.2 * time) * 0.3;

    // Flutter: fast speed variations (5-20 Hz)
    const flutter = Math.sin(2 * Math.PI * 6 * time) * 0.2 +
                    Math.sin(2 * Math.PI * 12 * time) * 0.1 +
                    Math.sin(2 * Math.PI * 18 * time) * 0.05;

    // Combine (cents)
    return (wow * 0.4 + flutter * 0.6) * amount * 50; // Up to ±50 cents
  }

  /**
   * Transformer coloration
   * Simulates frequency response of audio transformers
   * Low-end bump, slight high-end roll-off
   * @param {number} frequency - Frequency in Hz
   * @param {number} amount - Coloration amount (0-1)
   * @returns {number} Gain multiplier
   */
  static transformerResponse(frequency, amount = 0.5) {
    // Bass bump around 60-100 Hz
    const bassBump = Math.exp(-Math.pow((frequency - 80) / 40, 2)) * 0.15;

    // High-end roll-off above 10kHz
    const highRolloff = frequency > 10000 ?
      -Math.log(frequency / 10000) * 0.1 : 0;

    const totalGain = 1 + (bassBump + highRolloff) * amount;

    return Math.max(0.5, Math.min(1.5, totalGain));
  }

  /**
   * Hysteresis simulation
   * Simulates magnetic tape hysteresis
   * @param {number} input - Current input
   * @param {number} previousOutput - Previous output
   * @param {number} amount - Hysteresis amount (0-1)
   * @returns {number} Output with hysteresis
   */
  static hysteresis(input, previousOutput, amount = 0.5) {
    const difference = input - previousOutput;
    const delayed = previousOutput + difference * (1 - amount * 0.5);

    return delayed;
  }

  /**
   * Saturation detector (for VU meters, compression)
   * Returns how much signal is saturating
   * @param {number} value - Input value
   * @param {number} threshold - Saturation threshold (0-1)
   * @returns {number} Saturation amount (0-1)
   */
  static saturationAmount(value, threshold = 0.7) {
    const abs = Math.abs(value);

    if (abs < threshold) {
      return 0;
    } else {
      return (abs - threshold) / (1 - threshold);
    }
  }

  /**
   * VU meter ballistics
   * Simulates analog VU meter response
   * @param {number} input - Current input level
   * @param {number} previousLevel - Previous meter level
   * @param {number} sampleRate - Sample rate
   * @returns {number} Meter level
   */
  static vuMeterBallistics(input, previousLevel, sampleRate = 44100) {
    const abs = Math.abs(input);

    // VU meter attack: 300ms to reach 99% of final value
    const attackTime = 0.3;
    const attackCoeff = Math.exp(-1 / (sampleRate * attackTime));

    // VU meter release: similar to attack
    const releaseTime = 0.3;
    const releaseCoeff = Math.exp(-1 / (sampleRate * releaseTime));

    if (abs > previousLevel) {
      // Attack
      return attackCoeff * previousLevel + (1 - attackCoeff) * abs;
    } else {
      // Release
      return releaseCoeff * previousLevel + (1 - releaseCoeff) * abs;
    }
  }

  /**
   * Analog EQ curve (smooth, non-digital)
   * Simulates smooth analog filter response
   * @param {number} frequency - Center frequency
   * @param {number} q - Q factor
   * @param {number} gain - Gain in dB
   * @param {number} sampleFreq - Frequency to evaluate
   * @returns {number} Gain at sampleFreq
   */
  static analogEQCurve(frequency, q, gain, sampleFreq) {
    // Simplified analog bell curve
    const ratio = sampleFreq / frequency;
    const bandwidth = 1 / q;

    const response = 1 / (1 + Math.pow((ratio - 1 / ratio) / bandwidth, 2));

    const gainLinear = Math.pow(10, gain / 20);

    return 1 + (gainLinear - 1) * response;
  }

  /**
   * Generate DC offset (common in analog circuits)
   * @param {number} amount - Amount of DC offset (0-1)
   * @returns {number} DC offset value
   */
  static dcOffset(amount = 0.01) {
    return amount * (Math.random() - 0.5) * 2;
  }

  /**
   * Crosstalk simulation
   * Simulates signal bleed between channels
   * @param {number} leftChannel - Left channel signal
   * @param {number} rightChannel - Right channel signal
   * @param {number} amount - Crosstalk amount (0-1)
   * @returns {Object} {left, right} with crosstalk
   */
  static crosstalk(leftChannel, rightChannel, amount = 0.1) {
    return {
      left: leftChannel + rightChannel * amount * 0.05,
      right: rightChannel + leftChannel * amount * 0.05
    };
  }

  /**
   * Bias noise (self-noise of analog circuits)
   * @param {number} amount - Noise amount (0-1)
   * @returns {number} Noise sample
   */
  static biasNoise(amount = 0.01) {
    // White noise, very low level
    return (Math.random() * 2 - 1) * amount * 0.001;
  }

  /**
   * Phase distortion (characteristic of some analog gear)
   * @param {number} frequency - Frequency in Hz
   * @param {number} amount - Distortion amount (0-1)
   * @returns {number} Phase shift in radians
   */
  static phaseDistortion(frequency, amount = 0.5) {
    // High frequencies get more phase shift
    const shift = (frequency / 20000) * amount * Math.PI * 0.1;

    return shift;
  }
}

export default AnalogModeling;
