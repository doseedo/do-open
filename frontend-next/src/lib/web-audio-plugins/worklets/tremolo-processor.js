/**
 * TremoloProcessor - AudioWorklet implementation of tremolo/auto-pan effect
 *
 * Features:
 * - Amplitude modulation (tremolo mode)
 * - Stereo panning modulation (pan mode)
 * - Stereo mode with phase offset between L/R
 * - Multiple LFO waveforms
 * - Configurable depth and rate
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

class TremoloProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'rate',
        defaultValue: 5,
        minValue: 0.01,
        maxValue: 40,
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
        name: 'waveform',
        defaultValue: 0, // 0=sine, 1=triangle, 2=square, 3=sawtooth
        minValue: 0,
        maxValue: 3,
        automationRate: 'k-rate'
      },
      {
        name: 'mode',
        defaultValue: 0, // 0=tremolo (amplitude), 1=pan
        minValue: 0,
        maxValue: 1,
        automationRate: 'k-rate'
      },
      {
        name: 'stereo',
        defaultValue: 0, // 0=mono, 1=stereo (180° phase offset)
        minValue: 0,
        maxValue: 1,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Create LFOs (one for left, one for right in stereo mode)
    this.lfoLeft = new LFO(sampleRate);
    this.lfoRight = new LFO(sampleRate);

    // Set 180° phase offset for right channel
    this.lfoRight.setPhase(0.5); // 0.5 = 180°

    // Current parameters
    this.rate = 5;
    this.depth = 50;
    this.waveform = 0;
    this.mode = 0; // 0=tremolo, 1=pan
    this.stereo = 0; // 0=mono, 1=stereo
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
    const waveform = Math.floor(parameters.waveform[0]);
    const mode = Math.floor(parameters.mode[0]);
    const stereo = Math.floor(parameters.stereo[0]);

    // Update parameters if changed
    if (rate !== this.rate) {
      this.rate = rate;
      this.lfoLeft.setFrequency(rate);
      this.lfoRight.setFrequency(rate);
    }

    if (waveform !== this.waveform) {
      this.waveform = waveform;
      this.lfoLeft.setWaveform(waveform);
      this.lfoRight.setWaveform(waveform);
    }

    this.depth = depth;
    this.mode = mode;
    this.stereo = stereo;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, output.length);

    const depthAmount = this.depth / 100;

    // Process each sample
    for (let i = 0; i < blockSize; i++) {
      // Get input samples
      const inputL = input[0] ? input[0][i] : 0;
      const inputR = numChannels > 1 && input[1] ? input[1][i] : inputL;

      if (this.mode === 0) {
        // TREMOLO MODE - Amplitude modulation

        if (this.stereo === 1 && numChannels > 1) {
          // Stereo tremolo with 180° phase offset
          const lfoL = this.lfoLeft.process();
          const lfoR = this.lfoRight.process();

          // Convert LFO (-1 to 1) to gain (centered at 1, varying by ±depth/2)
          const centerGain = 1 - (depthAmount * 0.5);
          const variationGain = depthAmount * 0.5;

          const gainL = centerGain + lfoL * variationGain;
          const gainR = centerGain + lfoR * variationGain;

          output[0][i] = inputL * gainL;
          output[1][i] = inputR * gainR;
        } else {
          // Mono tremolo
          const lfo = this.lfoLeft.process();

          const centerGain = 1 - (depthAmount * 0.5);
          const variationGain = depthAmount * 0.5;
          const gain = centerGain + lfo * variationGain;

          output[0][i] = inputL * gain;
          if (numChannels > 1) {
            output[1][i] = inputR * gain;
          }
        }
      } else {
        // PAN MODE - Stereo panning modulation
        const lfo = this.lfoLeft.process();

        // Convert LFO (-1 to 1) to pan position (-depth to +depth)
        const pan = lfo * depthAmount; // -1 to 1 scaled by depth

        // Constant power panning
        const panRadians = (pan + 1) * 0.5 * Math.PI * 0.5; // Map to 0 to π/2
        const gainL = Math.cos(panRadians);
        const gainR = Math.sin(panRadians);

        // For mono input, pan across stereo field
        // For stereo input, modulate the stereo image
        output[0][i] = inputL * gainL;
        if (numChannels > 1) {
          output[1][i] = inputR * gainR;
        } else {
          // Mono to stereo
          output[1][i] = inputL * gainR;
        }
      }
    }

    return true;
  }
}

registerProcessor('tremolo-processor', TremoloProcessor);
