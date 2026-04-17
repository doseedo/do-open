/**
 * ChorusProcessor - AudioWorklet implementation of chorus effect
 *
 * Features:
 * - Multiple voices with phase-offset LFOs
 * - DelayLine-based modulation
 * - Stereo spread with panning
 * - Feedback control
 * - Configurable voice count (1-8)
 *
 * @author Agent 4 - Modulation Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

/**
 * LFO - Low Frequency Oscillator
 * Supports sine, triangle, square, sawtooth waveforms
 */
class LFO {
  constructor(sampleRate) {
    this.sampleRate = sampleRate;
    this.phase = 0;
    this.frequency = 1.0;
    this.waveform = 0; // 0=sine, 1=triangle, 2=square, 3=sawtooth
  }

  setFrequency(hz) {
    this.frequency = hz;
  }

  setWaveform(type) {
    this.waveform = type;
  }

  setPhase(phase) {
    this.phase = phase;
  }

  process() {
    // Increment phase
    this.phase += this.frequency / this.sampleRate;
    if (this.phase >= 1.0) {
      this.phase -= 1.0;
    }

    // Generate waveform
    let output = 0;
    switch (this.waveform) {
      case 0: // Sine
        output = Math.sin(this.phase * 2 * Math.PI);
        break;
      case 1: // Triangle
        output = this.phase < 0.5
          ? 4 * this.phase - 1
          : -4 * this.phase + 3;
        break;
      case 2: // Square
        output = this.phase < 0.5 ? 1 : -1;
        break;
      case 3: // Sawtooth
        output = 2 * this.phase - 1;
        break;
    }

    return output;
  }

  reset() {
    this.phase = 0;
  }
}

class ChorusProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'rate',
        defaultValue: 0.5,
        minValue: 0.01,
        maxValue: 10,
        automationRate: 'k-rate'
      },
      {
        name: 'depth',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'voices',
        defaultValue: 4,
        minValue: 1,
        maxValue: 8,
        automationRate: 'k-rate'
      },
      {
        name: 'spread',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'feedback',
        defaultValue: 0,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'mix',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'delay',
        defaultValue: 20,
        minValue: 5,
        maxValue: 50,
        automationRate: 'k-rate'
      },
      {
        name: 'waveform',
        defaultValue: 0, // 0=sine, 1=triangle, 2=square, 3=sawtooth
        minValue: 0,
        maxValue: 3,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Create 8 voices (max), each with stereo delay lines and LFO
    this.maxVoices = 8;
    this.voices = [];

    for (let i = 0; i < this.maxVoices; i++) {
      const voice = {
        delayLeft: new DelayLine(0.1, sampleRate), // 100ms max delay
        delayRight: new DelayLine(0.1, sampleRate),
        lfo: new LFO(sampleRate),
        pan: 0 // -1 to 1
      };

      // Set phase offset for this voice (evenly distributed)
      const phaseOffset = i / this.maxVoices;
      voice.lfo.setPhase(phaseOffset);

      this.voices.push(voice);
    }

    // Current parameters
    this.rate = 0.5;
    this.depth = 50;
    this.numVoices = 4;
    this.spread = 50;
    this.feedback = 0;
    this.mix = 50;
    this.baseDelay = 20;
    this.waveform = 0;

    // Update initial state
    this.updateVoicePanning();
  }

  /**
   * Update panning for each voice based on spread
   */
  updateVoicePanning() {
    const spreadAmount = this.spread / 100;

    for (let i = 0; i < this.maxVoices; i++) {
      if (this.numVoices === 1) {
        this.voices[i].pan = 0;
      } else {
        // Distribute voices across stereo field
        const position = (i / (this.numVoices - 1)) * 2 - 1; // -1 to 1
        this.voices[i].pan = position * spreadAmount;
      }
    }
  }

  /**
   * Process audio block
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    if (!input || !input.length) {
      return true;
    }

    // Get parameter values
    const rate = parameters.rate[0];
    const depth = parameters.depth[0];
    const voices = Math.floor(parameters.voices[0]);
    const spread = parameters.spread[0];
    const feedback = parameters.feedback[0];
    const mix = parameters.mix[0];
    const delay = parameters.delay[0];
    const waveform = Math.floor(parameters.waveform[0]);

    // Update parameters if changed
    if (rate !== this.rate) {
      this.rate = rate;
      this.voices.forEach(voice => voice.lfo.setFrequency(rate));
    }

    if (waveform !== this.waveform) {
      this.waveform = waveform;
      this.voices.forEach(voice => voice.lfo.setWaveform(waveform));
    }

    if (voices !== this.numVoices || spread !== this.spread) {
      this.numVoices = voices;
      this.spread = spread;
      this.updateVoicePanning();
    }

    this.depth = depth;
    this.feedback = feedback;
    this.mix = mix;
    this.baseDelay = delay;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, output.length);

    // Process each sample
    for (let i = 0; i < blockSize; i++) {
      // Get input samples (mono or stereo)
      const inputL = input[0] ? input[0][i] : 0;
      const inputR = numChannels > 1 && input[1] ? input[1][i] : inputL;

      let wetL = 0;
      let wetR = 0;

      // Process active voices
      for (let v = 0; v < this.numVoices; v++) {
        const voice = this.voices[v];

        // Get LFO value (-1 to 1)
        const lfoValue = voice.lfo.process();

        // Calculate modulated delay time
        const depthSeconds = (this.depth / 100) * 0.010; // Max 10ms modulation
        const baseDelaySeconds = this.baseDelay / 1000;
        const modulation = lfoValue * depthSeconds;
        const delayTime = baseDelaySeconds + modulation;
        const delaySamples = delayTime * sampleRate;

        // Read delayed samples
        const delayedL = voice.delayLeft.readInterpolated(delaySamples);
        const delayedR = voice.delayRight.readInterpolated(delaySamples);

        // Apply feedback
        const feedbackAmount = (this.feedback / 100) * 0.7; // Max 0.7 to prevent runaway
        const feedbackL = inputL + delayedL * feedbackAmount;
        const feedbackR = inputR + delayedR * feedbackAmount;

        // Write to delay lines
        voice.delayLeft.write(feedbackL);
        voice.delayRight.write(feedbackR);

        // Apply panning (constant power)
        const pan = voice.pan; // -1 to 1
        const panRadians = (pan + 1) * 0.5 * Math.PI * 0.5; // Map to 0 to π/2
        const gainL = Math.cos(panRadians);
        const gainR = Math.sin(panRadians);

        // Accumulate wet signal with voice gain
        const voiceGain = 1.0 / this.numVoices;
        wetL += delayedL * gainL * voiceGain;
        wetR += delayedR * gainR * voiceGain;
      }

      // Mix dry and wet (equal power crossfade)
      const mixAmount = this.mix / 100;
      const wetGain = Math.sin(mixAmount * Math.PI / 2);
      const dryGain = Math.cos(mixAmount * Math.PI / 2);

      output[0][i] = inputL * dryGain + wetL * wetGain;
      if (numChannels > 1) {
        output[1][i] = inputR * dryGain + wetR * wetGain;
      }
    }

    return true;
  }
}

registerProcessor('chorus-processor', ChorusProcessor);
