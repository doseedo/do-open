/**
 * Granular Synthesis AudioWorklet Processor
 * Creates texture and atmosphere by manipulating micro-segments of audio
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class GranularProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.grainSize = 0.05; // seconds
    this.density = 10; // grains per second
    this.randomness = 0.5; // 0-1
    this.pitch = 1.0; // playback rate
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
    // Buffer to record source audio
    this.sourceBuffer = new DelayLine(5.0, sampleRate);

    // Active grains
    this.grains = [];
    this.maxGrains = 32;

    // Grain spawning
    this.samplesUntilNextGrain = 0;
  }

  /**
   * Handle parameter updates
   */
  handleParameterUpdate(param, value) {
    switch (param) {
      case 'grainSize':
        this.grainSize = Math.max(0.01, Math.min(0.5, value));
        break;
      case 'density':
        this.density = Math.max(1, Math.min(100, value));
        break;
      case 'randomness':
        this.randomness = Math.max(0, Math.min(1, value));
        break;
      case 'pitch':
        this.pitch = Math.max(0.25, Math.min(4.0, value));
        break;
      case 'mix':
        this.mix = Math.max(0, Math.min(1, value));
        break;
    }
  }

  /**
   * Spawn a new grain
   */
  spawnGrain() {
    if (this.grains.length >= this.maxGrains) {
      return;
    }

    const grainSizeSamples = this.grainSize * sampleRate;
    const randomOffset = (Math.random() - 0.5) * this.randomness * grainSizeSamples;

    this.grains.push({
      position: Math.max(0, grainSizeSamples + randomOffset),
      age: 0,
      duration: grainSizeSamples,
      amplitude: 0.5 + Math.random() * 0.5
    });
  }

  /**
   * Process a single sample
   */
  processSample(sample, channel) {
    // Write to source buffer
    this.sourceBuffer.write(sample);

    // Spawn new grains
    this.samplesUntilNextGrain--;
    if (this.samplesUntilNextGrain <= 0) {
      this.spawnGrain();
      const interval = sampleRate / this.density;
      this.samplesUntilNextGrain = interval * (1 + (Math.random() - 0.5) * this.randomness);
    }

    // Process all active grains
    let output = 0;
    for (let i = this.grains.length - 1; i >= 0; i--) {
      const grain = this.grains[i];

      // Read from source buffer
      const readPos = grain.position * this.pitch;
      const grainSample = this.sourceBuffer.readInterpolated(readPos);

      // Apply Hann window
      const windowPhase = grain.age / grain.duration;
      const window = 0.5 - 0.5 * Math.cos(windowPhase * 2 * Math.PI);

      output += grainSample * window * grain.amplitude;

      // Advance grain
      grain.age++;
      grain.position += this.pitch;

      // Remove finished grains
      if (grain.age >= grain.duration) {
        this.grains.splice(i, 1);
      }
    }

    // Normalize by max possible grains
    const normalization = Math.min(this.grains.length + 1, 8);
    if (normalization > 0) {
      output /= normalization;
    }

    // Mix dry and wet
    return sample * (1 - this.mix) + output * this.mix;
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

registerProcessor('granular-processor', GranularProcessor);
