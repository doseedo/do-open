/**
 * Spectrum Analyzer Processor
 * AudioWorklet processor for real-time FFT-based frequency spectrum analysis
 *
 * Features:
 * - Real-time FFT processing (power of 2 sizes from 512 to 16384)
 * - Temporal smoothing for stable display
 * - Peak hold per frequency bin
 * - Configurable update rate
 * - Pass-through audio (no processing, just analysis)
 *
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

// Import FFT library (loaded before this processor)
importScripts('../../spectral/worklets/fft-lib.js');

class SpectrumAnalyzerProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // FFT settings
    this.fftSize = 2048;
    this.halfSize = this.fftSize / 2;

    // FFT engine
    this.fft = new FFT(this.fftSize);
    this.window = this.createHannWindow(this.fftSize);

    // Buffers
    this.inputBuffer = new Float32Array(this.fftSize);
    this.bufferIndex = 0;

    // FFT buffers
    this.real = new Float32Array(this.fftSize);
    this.imag = new Float32Array(this.fftSize);

    // Spectrum data
    this.spectrum = new Float32Array(this.halfSize);
    this.peakSpectrum = new Float32Array(this.halfSize);

    // Parameters
    this.smoothing = 0.8; // 0-1
    this.peakHold = false;
    this.peakDecay = 0.995;
    this.updateRate = 30; // updates per second

    // Update timing
    this.samplesPerUpdate = Math.floor(sampleRate / this.updateRate);
    this.sampleCounter = 0;

    // Message handling
    this.port.onmessage = (e) => {
      this.handleMessage(e.data);
    };
  }

  /**
   * Create Hann window function
   */
  createHannWindow(size) {
    const window = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (size - 1)));
    }
    return window;
  }

  handleMessage(data) {
    switch (data.type) {
      case 'fftSize':
        this.setFFTSize(data.value);
        break;

      case 'smoothing':
        this.smoothing = Math.max(0, Math.min(1, data.value));
        break;

      case 'peakHold':
        this.peakHold = data.value;
        if (!this.peakHold) {
          this.peakSpectrum.fill(0);
        }
        break;

      case 'updateRate':
        this.updateRate = Math.max(1, Math.min(120, data.value));
        this.samplesPerUpdate = Math.floor(sampleRate / this.updateRate);
        break;

      case 'reset':
        this.spectrum.fill(0);
        this.peakSpectrum.fill(0);
        this.bufferIndex = 0;
        break;
    }
  }

  /**
   * Set FFT size
   */
  setFFTSize(size) {
    const validSizes = [512, 1024, 2048, 4096, 8192, 16384];
    if (!validSizes.includes(size)) {
      console.warn(`Invalid FFT size: ${size}. Using 2048.`);
      size = 2048;
    }

    this.fftSize = size;
    this.halfSize = size / 2;

    // Recreate FFT engine and buffers
    this.fft = new FFT(size);
    this.window = this.createHannWindow(size);
    this.inputBuffer = new Float32Array(size);
    this.real = new Float32Array(size);
    this.imag = new Float32Array(size);
    this.spectrum = new Float32Array(this.halfSize);
    this.peakSpectrum = new Float32Array(this.halfSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || !input[0]) {
      return true;
    }

    // Pass through audio unmodified
    const channelCount = Math.min(input.length, output.length);
    for (let channel = 0; channel < channelCount; channel++) {
      output[channel].set(input[channel]);
    }

    // Accumulate samples for FFT (analyze first channel)
    const inputChannel = input[0];
    for (let i = 0; i < inputChannel.length; i++) {
      this.inputBuffer[this.bufferIndex] = inputChannel[i];
      this.bufferIndex = (this.bufferIndex + 1) % this.fftSize;
    }

    // Perform FFT periodically
    this.sampleCounter += inputChannel.length;
    if (this.sampleCounter >= this.samplesPerUpdate) {
      this.performFFT();
      this.sampleCounter = 0;

      // Send spectrum data to main thread
      this.port.postMessage({
        type: 'spectrum-update',
        spectrum: Array.from(this.spectrum),
        peakSpectrum: this.peakHold ? Array.from(this.peakSpectrum) : null,
        fftSize: this.fftSize,
        sampleRate: sampleRate,
        timestamp: currentTime
      });
    }

    return true;
  }

  /**
   * Perform FFT and calculate spectrum
   */
  performFFT() {
    // Copy input to FFT buffer with window function applied
    // Start from current buffer position for most recent data
    for (let i = 0; i < this.fftSize; i++) {
      const bufferPos = (this.bufferIndex + i) % this.fftSize;
      this.real[i] = this.inputBuffer[bufferPos] * this.window[i];
      this.imag[i] = 0;
    }

    // Forward FFT
    this.fft.forward(this.real, this.imag);

    // Calculate magnitude spectrum with smoothing
    for (let i = 0; i < this.halfSize; i++) {
      // Calculate magnitude
      const magnitude = Math.sqrt(
        this.real[i] * this.real[i] + this.imag[i] * this.imag[i]
      );

      // Normalize by FFT size
      const normalizedMag = magnitude / this.fftSize;

      // Apply smoothing (exponential moving average)
      this.spectrum[i] =
        this.smoothing * this.spectrum[i] + (1 - this.smoothing) * normalizedMag;

      // Update peak hold
      if (this.peakHold) {
        if (this.spectrum[i] > this.peakSpectrum[i]) {
          this.peakSpectrum[i] = this.spectrum[i];
        } else {
          // Slow decay
          this.peakSpectrum[i] *= this.peakDecay;
        }
      }
    }
  }

  static get parameterDescriptors() {
    return [];
  }
}

registerProcessor('spectrum-analyzer-processor', SpectrumAnalyzerProcessor);
