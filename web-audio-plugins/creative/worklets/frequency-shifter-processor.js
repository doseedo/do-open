/**
 * Frequency Shifter AudioWorklet Processor
 * Linear frequency shifting using single-sideband modulation
 *
 * Unlike ring modulation, frequency shifting moves ALL frequencies
 * by the same amount (linear shift), creating unique timbral effects.
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

class FrequencyShifterProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.shift = 0; // Hz to shift
    this.mix = 1.0; // 0-1

    // Initialize state
    this.initializeState();

    // Listen for parameter changes
    this.port.onmessage = (event) => {
      const { type, value } = event.data;
      this.handleParameterUpdate(type, value);
    };
  }

  /**
   * Initialize processing state
   */
  initializeState() {
    // Hilbert transform allpass filters for 90° phase shift
    // These coefficients create a wide-band 90° phase difference
    this.allpassFilters = [
      { a1: 0.6923878, b0: 0.9942116, b1: -1.0, z1: 0 },
      { a1: 0.4021921, b0: 0.9952884, b1: -1.0, z1: 0 }
    ];

    // Oscillator phase for modulation
    this.phase = 0;
  }

  /**
   * Handle parameter updates
   */
  handleParameterUpdate(param, value) {
    switch (param) {
      case 'shift':
        this.shift = Math.max(-5000, Math.min(5000, value));
        break;
      case 'mix':
        this.mix = Math.max(0, Math.min(1, value));
        break;
    }
  }

  /**
   * Process a single sample through Hilbert transform
   * Creates 90° phase-shifted version of input
   */
  hilbertTransform(sample) {
    let output = sample;

    // Process through cascaded allpass filters
    for (const filter of this.allpassFilters) {
      const temp = output;
      output = filter.b0 * temp + filter.z1;
      filter.z1 = filter.b1 * temp - filter.a1 * output;
    }

    return output;
  }

  /**
   * Process a single sample
   */
  processSample(sample, channel) {
    // Apply Hilbert transform to get 90° phase-shifted signal
    const hilbert = this.hilbertTransform(sample);

    // Generate quadrature oscillators (cosine and sine)
    const phaseRad = this.phase * 2 * Math.PI;
    const cosine = Math.cos(phaseRad);
    const sine = Math.sin(phaseRad);

    // Advance phase
    this.phase += this.shift / sampleRate;
    if (this.phase >= 1.0) this.phase -= 1.0;
    if (this.phase < 0) this.phase += 1.0;

    // Single-sideband modulation
    // This shifts all frequencies linearly
    const shifted = sample * cosine - hilbert * sine;

    // Mix dry and wet
    return sample * (1 - this.mix) + shifted * this.mix;
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
        outputChannel[i] = this.processSample(inputChannel[i], channel);
      }
    }

    return true;
  }
}

registerProcessor('frequency-shifter-processor', FrequencyShifterProcessor);
