/**
 * FFT Library for AudioWorklet
 * Lightweight FFT implementation for real-time audio processing
 * Based on Cooley-Tukey radix-2 FFT algorithm
 */

class FFT {
  constructor(size) {
    this.size = size;
    this.halfSize = size / 2;

    // Pre-compute twiddle factors for efficiency
    this.cosTable = new Float32Array(this.halfSize);
    this.sinTable = new Float32Array(this.halfSize);

    for (let i = 0; i < this.halfSize; i++) {
      const angle = (-2 * Math.PI * i) / size;
      this.cosTable[i] = Math.cos(angle);
      this.sinTable[i] = Math.sin(angle);
    }

    // Bit reversal lookup table
    this.reverseTable = new Uint32Array(size);
    this.computeReversalTable();
  }

  computeReversalTable() {
    const bits = Math.log2(this.size);

    for (let i = 0; i < this.size; i++) {
      let reversed = 0;
      for (let j = 0; j < bits; j++) {
        reversed = (reversed << 1) | ((i >> j) & 1);
      }
      this.reverseTable[i] = reversed;
    }
  }

  /**
   * Forward FFT (time domain to frequency domain)
   * @param {Float32Array} real - Real part of input
   * @param {Float32Array} imag - Imaginary part of input
   */
  forward(real, imag) {
    // Bit-reversal permutation
    this.bitReverse(real, imag);

    // Cooley-Tukey FFT
    for (let blockSize = 2; blockSize <= this.size; blockSize *= 2) {
      const halfBlock = blockSize / 2;
      const tableStep = this.size / blockSize;

      for (let i = 0; i < this.size; i += blockSize) {
        for (let j = i, k = 0; j < i + halfBlock; j++, k += tableStep) {
          const l = j + halfBlock;
          const twiddleReal = this.cosTable[k];
          const twiddleImag = this.sinTable[k];

          const tReal = real[l] * twiddleReal - imag[l] * twiddleImag;
          const tImag = real[l] * twiddleImag + imag[l] * twiddleReal;

          real[l] = real[j] - tReal;
          imag[l] = imag[j] - tImag;
          real[j] += tReal;
          imag[j] += tImag;
        }
      }
    }
  }

  /**
   * Inverse FFT (frequency domain to time domain)
   * @param {Float32Array} real - Real part of input
   * @param {Float32Array} imag - Imaginary part of input
   */
  inverse(real, imag) {
    // Conjugate the complex numbers
    for (let i = 0; i < this.size; i++) {
      imag[i] = -imag[i];
    }

    // Forward FFT
    this.forward(real, imag);

    // Conjugate the complex numbers again and scale
    const scale = 1 / this.size;
    for (let i = 0; i < this.size; i++) {
      real[i] *= scale;
      imag[i] = -imag[i] * scale;
    }
  }

  bitReverse(real, imag) {
    for (let i = 0; i < this.size; i++) {
      const j = this.reverseTable[i];
      if (j > i) {
        // Swap real
        const tempReal = real[i];
        real[i] = real[j];
        real[j] = tempReal;

        // Swap imaginary
        const tempImag = imag[i];
        imag[i] = imag[j];
        imag[j] = tempImag;
      }
    }
  }
}

/**
 * Window functions for FFT processing
 */
class WindowFunctions {
  static hann(size) {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / (size - 1)));
    }
    return window;
  }

  static hamming(size) {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.54 - 0.46 * Math.cos(2 * Math.PI * i / (size - 1));
    }
    return window;
  }

  static blackman(size) {
    const window = new Float32Array(size);
    const a0 = 0.42;
    const a1 = 0.5;
    const a2 = 0.08;

    for (let i = 0; i < size; i++) {
      const angle = 2 * Math.PI * i / (size - 1);
      window[i] = a0 - a1 * Math.cos(angle) + a2 * Math.cos(2 * angle);
    }
    return window;
  }

  static applyWindow(signal, window) {
    for (let i = 0; i < signal.length; i++) {
      signal[i] *= window[i];
    }
  }
}

/**
 * Spectral utilities
 */
class SpectralUtils {
  /**
   * Convert rectangular (real/imag) to polar (magnitude/phase)
   */
  static toPolar(real, imag, magnitude, phase) {
    for (let i = 0; i < real.length; i++) {
      magnitude[i] = Math.sqrt(real[i] * real[i] + imag[i] * imag[i]);
      phase[i] = Math.atan2(imag[i], real[i]);
    }
  }

  /**
   * Convert polar (magnitude/phase) to rectangular (real/imag)
   */
  static toRectangular(magnitude, phase, real, imag) {
    for (let i = 0; i < magnitude.length; i++) {
      real[i] = magnitude[i] * Math.cos(phase[i]);
      imag[i] = magnitude[i] * Math.sin(phase[i]);
    }
  }

  /**
   * Phase unwrapping for continuous phase
   */
  static unwrapPhase(phase) {
    const unwrapped = new Float32Array(phase.length);
    unwrapped[0] = phase[0];

    for (let i = 1; i < phase.length; i++) {
      let delta = phase[i] - phase[i - 1];

      // Wrap delta to [-pi, pi]
      while (delta > Math.PI) delta -= 2 * Math.PI;
      while (delta < -Math.PI) delta += 2 * Math.PI;

      unwrapped[i] = unwrapped[i - 1] + delta;
    }

    return unwrapped;
  }

  /**
   * Bin frequency from bin index
   */
  static binToFrequency(bin, sampleRate, fftSize) {
    return (bin * sampleRate) / fftSize;
  }

  /**
   * Bin index from frequency
   */
  static frequencyToBin(frequency, sampleRate, fftSize) {
    return Math.round((frequency * fftSize) / sampleRate);
  }
}

/**
 * Overlap-Add processor for STFT
 */
class OverlapAdd {
  constructor(fftSize, hopSize) {
    this.fftSize = fftSize;
    this.hopSize = hopSize;
    this.outputBuffer = new Float32Array(fftSize);
    this.position = 0;
  }

  /**
   * Add a frame to the output buffer
   */
  add(frame) {
    for (let i = 0; i < this.fftSize; i++) {
      this.outputBuffer[i] += frame[i];
    }
  }

  /**
   * Extract samples from the output buffer
   */
  extract(output) {
    const samplesToExtract = Math.min(this.hopSize, output.length);

    for (let i = 0; i < samplesToExtract; i++) {
      output[i] = this.outputBuffer[i];
    }

    // Shift the buffer
    for (let i = 0; i < this.fftSize - this.hopSize; i++) {
      this.outputBuffer[i] = this.outputBuffer[i + this.hopSize];
    }

    // Clear the end
    for (let i = this.fftSize - this.hopSize; i < this.fftSize; i++) {
      this.outputBuffer[i] = 0;
    }
  }
}

/**
 * Hilbert Transform for single-sideband modulation
 * Approximation using FIR filter
 */
class HilbertTransform {
  constructor(order = 64) {
    this.order = order;
    this.coefficients = this.computeCoefficients();
    this.buffer = new Float32Array(order);
    this.position = 0;
  }

  computeCoefficients() {
    const coeffs = new Float32Array(this.order);
    const center = Math.floor(this.order / 2);

    for (let i = 0; i < this.order; i++) {
      const n = i - center;
      if (n === 0) {
        coeffs[i] = 0;
      } else if (n % 2 === 0) {
        coeffs[i] = 0;
      } else {
        coeffs[i] = 2 / (Math.PI * n);
      }
    }

    return coeffs;
  }

  process(sample) {
    // Add sample to buffer
    this.buffer[this.position] = sample;
    this.position = (this.position + 1) % this.order;

    // Convolve
    let output = 0;
    for (let i = 0; i < this.order; i++) {
      const bufferIndex = (this.position + i) % this.order;
      output += this.buffer[bufferIndex] * this.coefficients[i];
    }

    return output;
  }
}

// Export for use in AudioWorklet
if (typeof registerProcessor !== 'undefined') {
  // Running in AudioWorklet context
  globalThis.FFT = FFT;
  globalThis.WindowFunctions = WindowFunctions;
  globalThis.SpectralUtils = SpectralUtils;
  globalThis.OverlapAdd = OverlapAdd;
  globalThis.HilbertTransform = HilbertTransform;
}
