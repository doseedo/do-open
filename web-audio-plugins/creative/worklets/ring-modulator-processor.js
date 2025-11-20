/**
 * Ring Modulator AudioWorklet Processor
 * Creates inharmonic sidebands through amplitude modulation
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

class RingModulatorProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Default parameters
    this.frequency = 440; // Hz
    this.mix = 1.0; // 0-1
    this.phase = 0;
    this.waveform = 'sine'; // sine, triangle, square, saw

    // Listen for parameter changes
    this.port.onmessage = (event) => {
      const { type, value } = event.data;
      this.handleParameterUpdate(type, value);
    };
  }

  /**
   * Handle parameter updates from main thread
   */
  handleParameterUpdate(param, value) {
    switch (param) {
      case 'frequency':
        this.frequency = Math.max(0.1, Math.min(20000, value));
        break;
      case 'mix':
        this.mix = Math.max(0, Math.min(1, value));
        break;
      case 'waveform':
        this.waveform = value;
        break;
    }
  }

  /**
   * Generate carrier oscillator sample
   */
  generateCarrier() {
    const phaseIncrement = this.frequency / sampleRate;
    this.phase += phaseIncrement;
    if (this.phase >= 1.0) this.phase -= 1.0;

    let carrier;
    switch (this.waveform) {
      case 'sine':
        carrier = Math.sin(this.phase * 2 * Math.PI);
        break;
      case 'triangle':
        carrier = this.phase < 0.5
          ? (this.phase * 4 - 1)
          : (3 - this.phase * 4);
        break;
      case 'square':
        carrier = this.phase < 0.5 ? 1 : -1;
        break;
      case 'saw':
        carrier = this.phase * 2 - 1;
        break;
      default:
        carrier = Math.sin(this.phase * 2 * Math.PI);
    }

    return carrier;
  }

  /**
   * Process audio
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input[0]) {
      return true;
    }

    for (let channel = 0; channel < output.length; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < inputChannel.length; i++) {
        const sample = inputChannel[i];

        // Generate carrier oscillator
        const carrier = this.generateCarrier();

        // Ring modulation = multiplication
        const modulated = sample * carrier;

        // Mix dry and wet
        outputChannel[i] = sample * (1 - this.mix) + modulated * this.mix;
      }
    }

    return true;
  }
}

registerProcessor('ring-modulator-processor', RingModulatorProcessor);
