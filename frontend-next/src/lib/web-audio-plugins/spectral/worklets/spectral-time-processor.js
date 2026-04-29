/**
 * Spectral Time Processor
 * Phase vocoder implementation for time stretching, pitch shifting, and spectral freezing
 */

// Import FFT library (loaded before this processor)
importScripts('fft-lib.js');

class SpectralTimeProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // FFT settings
    this.fftSize = 4096;
    this.hopSize = this.fftSize / 4;
    this.halfSize = this.fftSize / 2;

    // FFT engine
    this.fft = new FFT(this.fftSize);
    this.window = WindowFunctions.hann(this.fftSize);
    this.overlapAdd = new OverlapAdd(this.fftSize, this.hopSize);

    // Buffers
    this.inputBuffer = new Float32Array(this.fftSize);
    this.outputBuffer = new Float32Array(this.fftSize);
    this.inputPosition = 0;
    this.outputPosition = 0;

    // FFT buffers
    this.real = new Float32Array(this.fftSize);
    this.imag = new Float32Array(this.fftSize);
    this.magnitude = new Float32Array(this.halfSize);
    this.phase = new Float32Array(this.halfSize);

    // Phase vocoder state
    this.previousPhase = new Float32Array(this.halfSize);
    this.sumPhase = new Float32Array(this.halfSize);
    this.frozenMagnitude = new Float32Array(this.halfSize);
    this.frozenPhase = new Float32Array(this.halfSize);

    // Parameters
    this.stretch = 1.0;
    this.freeze = false;
    this.blur = 0.0;
    this.pitchShift = 0; // in semitones
    this.formantShift = 0;
    this.residual = 0.0;
    this.mix = 1.0;

    // Processing state
    this.frameCount = 0;
    this.analysisHopSize = this.hopSize;
    this.synthesisHopSize = this.hopSize;

    // Message handling
    this.port.onmessage = (e) => {
      this.handleMessage(e.data);
    };
  }

  handleMessage(data) {
    switch (data.type) {
      case 'stretch':
        this.stretch = Math.max(0.1, Math.min(4.0, data.value));
        this.synthesisHopSize = Math.floor(this.hopSize / this.stretch);
        break;

      case 'freeze':
        if (data.value && !this.freeze) {
          // Capture current spectrum
          this.frozenMagnitude.set(this.magnitude);
          this.frozenPhase.set(this.phase);
        }
        this.freeze = data.value;
        break;

      case 'blur':
        this.blur = Math.max(0, Math.min(1, data.value));
        break;

      case 'shift':
        this.pitchShift = Math.max(-24, Math.min(24, data.value));
        break;

      case 'formant':
        this.formantShift = Math.max(-4, Math.min(4, data.value));
        break;

      case 'residual':
        this.residual = Math.max(0, Math.min(1, data.value));
        break;

      case 'mix':
        this.mix = Math.max(0, Math.min(1, data.value));
        break;
    }
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0]) {
      return true;
    }

    const inputChannel = input[0];
    const outputChannel = output[0];

    for (let i = 0; i < inputChannel.length; i++) {
      // Store input sample
      this.inputBuffer[this.inputPosition] = inputChannel[i];
      this.inputPosition++;

      // Process when we have enough input
      if (this.inputPosition >= this.fftSize) {
        this.processFrame();
        this.inputPosition = this.hopSize;

        // Shift input buffer
        for (let j = 0; j < this.fftSize - this.hopSize; j++) {
          this.inputBuffer[j] = this.inputBuffer[j + this.hopSize];
        }
      }

      // Output sample
      const wetSample = this.outputBuffer[this.outputPosition];
      this.outputPosition++;

      if (this.outputPosition >= this.synthesisHopSize) {
        this.outputPosition = 0;
      }

      // Mix wet/dry
      outputChannel[i] = inputChannel[i] * (1 - this.mix) + wetSample * this.mix;
    }

    return true;
  }

  processFrame() {
    // Copy input to FFT buffer and apply window
    for (let i = 0; i < this.fftSize; i++) {
      this.real[i] = this.inputBuffer[i] * this.window[i];
      this.imag[i] = 0;
    }

    // Forward FFT
    this.fft.forward(this.real, this.imag);

    // Convert to polar
    SpectralUtils.toPolar(this.real, this.imag, this.magnitude, this.phase);

    // Phase vocoder processing
    if (!this.freeze) {
      this.phaseVocoder();
    } else {
      // Use frozen spectrum
      this.magnitude.set(this.frozenMagnitude);
      this.phase.set(this.frozenPhase);
    }

    // Apply spectral blur
    if (this.blur > 0) {
      this.applyBlur();
    }

    // Apply pitch shift
    if (this.pitchShift !== 0) {
      this.applyPitchShift();
    }

    // Convert back to rectangular
    SpectralUtils.toRectangular(this.magnitude, this.sumPhase, this.real, this.imag);

    // Inverse FFT
    this.fft.inverse(this.real, this.imag);

    // Apply window and overlap-add
    const frame = new Float32Array(this.fftSize);
    for (let i = 0; i < this.fftSize; i++) {
      frame[i] = this.real[i] * this.window[i];
    }

    this.overlapAdd.add(frame);
    this.overlapAdd.extract(this.outputBuffer);

    this.frameCount++;
  }

  phaseVocoder() {
    const expectedPhaseDelta = (2 * Math.PI * this.hopSize) / this.fftSize;

    for (let i = 0; i < this.halfSize; i++) {
      // Calculate phase difference
      let phaseDiff = this.phase[i] - this.previousPhase[i];

      // Subtract expected phase advance
      let heteroPhase = phaseDiff - expectedPhaseDelta * i;

      // Wrap to [-pi, pi]
      while (heteroPhase > Math.PI) heteroPhase -= 2 * Math.PI;
      while (heteroPhase < -Math.PI) heteroPhase += 2 * Math.PI;

      // Calculate true frequency (in bins)
      const trueFrequency = i + heteroPhase / expectedPhaseDelta;

      // Accumulate phase for synthesis
      this.sumPhase[i] += (trueFrequency * expectedPhaseDelta * this.synthesisHopSize) / this.hopSize;

      // Store current phase for next frame
      this.previousPhase[i] = this.phase[i];
    }
  }

  applyBlur() {
    // Spectral blur by smoothing magnitude
    const blurred = new Float32Array(this.halfSize);
    const kernelSize = Math.floor(this.blur * 20) + 1;

    for (let i = 0; i < this.halfSize; i++) {
      let sum = 0;
      let count = 0;

      for (let j = -kernelSize; j <= kernelSize; j++) {
        const index = i + j;
        if (index >= 0 && index < this.halfSize) {
          sum += this.magnitude[index];
          count++;
        }
      }

      blurred[i] = sum / count;
    }

    // Blend with original
    for (let i = 0; i < this.halfSize; i++) {
      this.magnitude[i] = this.magnitude[i] * (1 - this.blur) + blurred[i] * this.blur;
    }
  }

  applyPitchShift() {
    // Pitch shift by resampling spectrum
    const ratio = Math.pow(2, this.pitchShift / 12);
    const shifted = new Float32Array(this.halfSize);
    const shiftedPhase = new Float32Array(this.halfSize);

    for (let i = 0; i < this.halfSize; i++) {
      const sourceIndex = i / ratio;
      const lowerIndex = Math.floor(sourceIndex);
      const upperIndex = Math.ceil(sourceIndex);
      const fraction = sourceIndex - lowerIndex;

      if (upperIndex < this.halfSize) {
        // Linear interpolation
        shifted[i] = this.magnitude[lowerIndex] * (1 - fraction) +
                     this.magnitude[upperIndex] * fraction;
        shiftedPhase[i] = this.sumPhase[lowerIndex] * (1 - fraction) +
                          this.sumPhase[upperIndex] * fraction;
      }
    }

    this.magnitude.set(shifted);
    this.sumPhase.set(shiftedPhase);
  }

  static get parameterDescriptors() {
    return [];
  }
}

registerProcessor('spectral-time-processor', SpectralTimeProcessor);
