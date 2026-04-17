/**
 * ReverbProcessor - AudioWorklet implementation of Schroeder algorithmic reverb
 *
 * Features:
 * - Parallel comb filters for reverb tail
 * - Series allpass filters for diffusion
 * - OnePoleFilter damping in feedback paths
 * - Pre-delay
 * - Stereo processing
 *
 * Based on Freeverb/Schroeder reverberator architecture
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class ReverbProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'preDelay',
        defaultValue: 0,
        minValue: 0,
        maxValue: 250,
        automationRate: 'k-rate'
      },
      {
        name: 'decayTime',
        defaultValue: 2.0,
        minValue: 0.1,
        maxValue: 20,
        automationRate: 'k-rate'
      },
      {
        name: 'size',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'diffusion',
        defaultValue: 70,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'damping',
        defaultValue: 50,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      },
      {
        name: 'mix',
        defaultValue: 30,
        minValue: 0,
        maxValue: 100,
        automationRate: 'k-rate'
      }
    ];
  }

  constructor(options) {
    super();

    // Base delay times for comb filters (in seconds)
    // Tuned to avoid resonances (prime-related ratios)
    this.baseCombTimes = [
      0.0297, 0.0371, 0.0411, 0.0437, // Left channel
      0.0306, 0.0379, 0.0420, 0.0445  // Right channel (slightly detuned)
    ];

    // Base delay times for allpass filters (in seconds)
    this.baseAllpassTimes = [0.0051, 0.0126, 0.0100, 0.0077];

    // Create comb filters (8 total: 4 per channel)
    this.combFilters = [];
    for (let i = 0; i < 8; i++) {
      this.combFilters.push({
        delayLine: new DelayLine(0.1, sampleRate),
        damping: new OnePoleFilter(5000, sampleRate),
        feedback: 0.84,
        channel: i < 4 ? 0 : 1 // Left or right channel
      });
    }

    // Create allpass filters (4 total, series configuration)
    this.allpassFilters = [];
    for (let i = 0; i < 4; i++) {
      this.allpassFilters.push({
        delayLineL: new DelayLine(0.05, sampleRate),
        delayLineR: new DelayLine(0.05, sampleRate),
        coefficient: 0.5
      });
    }

    // Pre-delay lines
    this.preDelayL = new DelayLine(0.25, sampleRate);
    this.preDelayR = new DelayLine(0.25, sampleRate);

    // Current parameter values
    this.preDelay = 0;
    this.decayTime = 2.0;
    this.size = 50;
    this.diffusion = 70;
    this.damping = 50;
    this.mix = 30;

    // Apply initial settings
    this.updateCombDelayTimes();
    this.updateAllpassDelayTimes();
    this.updateDamping();
    this.updateFeedback();
  }

  /**
   * Update comb filter delay times based on size parameter
   */
  updateCombDelayTimes() {
    const scale = 0.5 + (this.size / 100) * 1.5; // 0.5x to 2x

    this.combFilters.forEach((comb, index) => {
      const newTime = this.baseCombTimes[index] * scale;
      comb.delaySamples = newTime * sampleRate;
    });
  }

  /**
   * Update allpass filter delay times based on size parameter
   */
  updateAllpassDelayTimes() {
    const scale = 0.5 + (this.size / 100) * 1.5; // 0.5x to 2x

    this.allpassFilters.forEach((ap, index) => {
      const newTime = this.baseAllpassTimes[index] * scale;
      ap.delaySamplesL = newTime * sampleRate;
      ap.delaySamplesR = newTime * 1.02 * sampleRate; // Slight stereo detuning
    });
  }

  /**
   * Update damping filter cutoff based on damping parameter
   */
  updateDamping() {
    // Map to lowpass frequency: 100% damping = 1kHz, 0% = 20kHz
    const freq = 20000 * Math.pow(0.05, this.damping / 100);

    this.combFilters.forEach(comb => {
      comb.damping.setCutoff(freq);
    });
  }

  /**
   * Update feedback gains based on decay time
   */
  updateFeedback() {
    // Calculate feedback gain for desired decay time
    // Using RT60 formula: feedback = 10^(-3 * delay / RT60)
    this.combFilters.forEach((comb, index) => {
      const delayTime = this.baseCombTimes[index] * (0.5 + (this.size / 100) * 1.5);
      const feedback = Math.pow(10, (-3 * delayTime) / this.decayTime);
      comb.feedback = Math.min(0.98, feedback); // Clamp to prevent runaway
    });
  }

  /**
   * Update allpass coefficients based on diffusion parameter
   */
  updateDiffusion() {
    const amount = this.diffusion / 100 * 0.7; // 0 to 0.7

    this.allpassFilters.forEach(ap => {
      ap.coefficient = amount;
    });
  }

  /**
   * Process a sample through a comb filter
   */
  processComb(comb, input) {
    // Read delayed sample
    const delayed = comb.delayLine.read(comb.delaySamples);

    // Apply damping filter
    const damped = comb.damping.process(delayed);

    // Write input + feedback to delay line
    comb.delayLine.write(input + damped * comb.feedback);

    // Output is the damped delayed signal
    return damped;
  }

  /**
   * Process a sample through an allpass filter
   */
  processAllpass(delayLine, delaySamples, coefficient, input) {
    // Read delayed sample
    const delayed = delayLine.readInterpolated(delaySamples);

    // Allpass formula: output = -input + delayed + input * coefficient
    const output = -input + delayed;

    // Write to delay line: input + delayed * coefficient
    delayLine.write(input + delayed * coefficient);

    return output;
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

    // Get parameter values (k-rate)
    const preDelay = parameters.preDelay[0];
    const decayTime = parameters.decayTime[0];
    const size = parameters.size[0];
    const diffusion = parameters.diffusion[0];
    const damping = parameters.damping[0];
    const mix = parameters.mix[0];

    // Check if parameters changed and update accordingly
    let paramsChanged = false;

    if (this.preDelay !== preDelay) {
      this.preDelay = preDelay;
      paramsChanged = true;
    }

    if (this.decayTime !== decayTime) {
      this.decayTime = decayTime;
      this.updateFeedback();
    }

    if (this.size !== size) {
      this.size = size;
      this.updateCombDelayTimes();
      this.updateAllpassDelayTimes();
      this.updateFeedback();
    }

    if (this.diffusion !== diffusion) {
      this.diffusion = diffusion;
      this.updateDiffusion();
    }

    if (this.damping !== damping) {
      this.damping = damping;
      this.updateDamping();
    }

    this.mix = mix;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, 2);
    const preDelaySamples = (this.preDelay / 1000) * sampleRate;

    // Ensure output has same number of channels as input
    for (let channel = 0; channel < numChannels; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      for (let i = 0; i < blockSize; i++) {
        const drySample = inputChannel[i];

        // Apply pre-delay
        const preDelayLine = channel === 0 ? this.preDelayL : this.preDelayR;
        preDelayLine.write(drySample);
        const preDelayed = preDelayLine.readInterpolated(preDelaySamples);

        // Process through comb filters (parallel, 4 per channel)
        let combSum = 0;
        const startComb = channel === 0 ? 0 : 4;
        const endComb = channel === 0 ? 4 : 8;

        for (let c = startComb; c < endComb; c++) {
          combSum += this.processComb(this.combFilters[c], preDelayed);
        }

        // Average the comb outputs
        combSum *= 0.25; // 1/4 for mixing 4 combs

        // Process through allpass filters (series)
        let apOutput = combSum;
        for (let a = 0; a < this.allpassFilters.length; a++) {
          const ap = this.allpassFilters[a];
          const delayLine = channel === 0 ? ap.delayLineL : ap.delayLineR;
          const delaySamples = channel === 0 ? ap.delaySamplesL : ap.delaySamplesR;

          apOutput = this.processAllpass(delayLine, delaySamples, ap.coefficient, apOutput);
        }

        // Mix dry and wet
        const wetGain = this.mix / 100;
        const dryGain = 1 - wetGain;

        outputChannel[i] = drySample * dryGain + apOutput * wetGain;
      }
    }

    return true;
  }
}

registerProcessor('reverb-processor', ReverbProcessor);
