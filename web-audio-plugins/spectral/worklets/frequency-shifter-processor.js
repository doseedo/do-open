/**
 * Frequency Shifter Processor
 * Single-sideband modulation for linear frequency shifting
 */

// Import FFT library
importScripts('fft-lib.js');

class FrequencyShifterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.frequency = 0; // Shift frequency in Hz
    this.fineFrequency = 0; // Fine tuning in Hz
    this.mode = 'up'; // 'up', 'down', 'wide'
    this.wideAmount = 0.5; // For wide mode
    this.drive = 0.0; // Harmonic saturation
    this.mix = 1.0; // Wet/dry mix

    // Hilbert transform for quadrature generation
    this.hilbertL = new HilbertTransform(64);
    this.hilbertR = new HilbertTransform(64);

    // Delay compensation (Hilbert introduces delay)
    this.delayCompensation = 32; // Half of Hilbert order
    this.delayBufferL = new Float32Array(this.delayCompensation);
    this.delayBufferR = new Float32Array(this.delayCompensation);
    this.delayPosition = 0;

    // Oscillator phase
    this.phaseL = 0;
    this.phaseR = 0;

    // Message handling
    this.port.onmessage = (e) => {
      this.handleMessage(e.data);
    };
  }

  handleMessage(data) {
    switch (data.type) {
      case 'frequency':
        this.frequency = Math.max(-5000, Math.min(5000, data.value));
        break;

      case 'fine':
        this.fineFrequency = Math.max(-100, Math.min(100, data.value));
        break;

      case 'mode':
        this.mode = data.value;
        break;

      case 'wideAmount':
        this.wideAmount = Math.max(0, Math.min(1, data.value));
        break;

      case 'drive':
        this.drive = Math.max(0, Math.min(1, data.value));
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

    const inputL = input[0];
    const inputR = input[1] || input[0]; // Mono to stereo
    const outputL = output[0];
    const outputR = output[1] || output[0];

    const totalShift = this.frequency + this.fineFrequency;

    for (let i = 0; i < inputL.length; i++) {
      // Process left channel
      const {wet: wetL, dry: dryL} = this.processChannel(
        inputL[i],
        this.hilbertL,
        this.delayBufferL,
        totalShift,
        this.phaseL,
        'left'
      );

      this.phaseL += (2 * Math.PI * totalShift) / sampleRate;
      while (this.phaseL > 2 * Math.PI) this.phaseL -= 2 * Math.PI;
      while (this.phaseL < 0) this.phaseL += 2 * Math.PI;

      // Process right channel
      const shiftR = this.mode === 'wide'
        ? totalShift * (1 + this.wideAmount)
        : totalShift;

      const {wet: wetR, dry: dryR} = this.processChannel(
        inputR[i],
        this.hilbertR,
        this.delayBufferR,
        shiftR,
        this.phaseR,
        'right'
      );

      this.phaseR += (2 * Math.PI * shiftR) / sampleRate;
      while (this.phaseR > 2 * Math.PI) this.phaseR -= 2 * Math.PI;
      while (this.phaseR < 0) this.phaseR += 2 * Math.PI;

      // Apply drive (soft clipping)
      const wetLDriven = this.applyDrive(wetL);
      const wetRDriven = this.applyDrive(wetR);

      // Mix wet/dry
      outputL[i] = dryL * (1 - this.mix) + wetLDriven * this.mix;
      outputR[i] = dryR * (1 - this.mix) + wetRDriven * this.mix;

      // Update delay buffer position
      this.delayPosition = (this.delayPosition + 1) % this.delayCompensation;
    }

    return true;
  }

  processChannel(inputSample, hilbert, delayBuffer, shiftFreq, phase, channel) {
    // Get delayed (compensated) input
    const dry = delayBuffer[this.delayPosition];

    // Store current input for delay compensation
    delayBuffer[this.delayPosition] = inputSample;

    // Generate quadrature signal (90-degree phase shift)
    const quadrature = hilbert.process(inputSample);

    // Generate modulation signals
    const cosine = Math.cos(phase);
    const sine = Math.sin(phase);

    // Single-sideband modulation
    let wet;

    if (this.mode === 'up' || this.mode === 'wide') {
      // Upper sideband
      wet = dry * cosine - quadrature * sine;
    } else if (this.mode === 'down') {
      // Lower sideband
      wet = dry * cosine + quadrature * sine;
    } else {
      wet = dry;
    }

    return { wet, dry };
  }

  applyDrive(sample) {
    if (this.drive === 0) return sample;

    // Soft clipping
    const gain = 1 + this.drive * 10;
    const driven = sample * gain;

    // Tanh soft clipping
    return Math.tanh(driven);
  }

  static get parameterDescriptors() {
    return [];
  }
}

registerProcessor('frequency-shifter-processor', FrequencyShifterProcessor);
