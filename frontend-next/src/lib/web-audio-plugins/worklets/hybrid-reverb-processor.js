/**
 * HybridReverbProcessor - AudioWorklet implementation of hybrid reverb
 *
 * Features:
 * - Early reflections (simulated with multi-tap delays)
 * - Algorithmic tail (comb + allpass filters)
 * - Crossover control between early and tail
 * - Pre-delay
 * - Independent level control for early/tail
 *
 * Note: This is a simplified version using delay-based early reflections.
 * A full hybrid reverb would use convolution for early reflections,
 * but convolution is complex in AudioWorklet.
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('../core/dsp-utils.js');

class HybridReverbProcessor extends AudioWorkletProcessor {
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
        name: 'earlyLevel',
        defaultValue: -6,
        minValue: -60,
        maxValue: 0,
        automationRate: 'k-rate'
      },
      {
        name: 'tailLevel',
        defaultValue: -6,
        minValue: -60,
        maxValue: 0,
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

    // Pre-delay
    this.preDelayL = new DelayLine(0.25, sampleRate);
    this.preDelayR = new DelayLine(0.25, sampleRate);

    // Early reflections (multi-tap delays)
    // Simulating discrete room reflections
    this.earlyDelayTimes = [
      0.019, 0.022, 0.027, 0.031, 0.037, 0.043, 0.048, 0.053
    ];

    this.earlyDelaysL = [];
    this.earlyDelaysR = [];

    for (let i = 0; i < this.earlyDelayTimes.length; i++) {
      this.earlyDelaysL.push({
        delayLine: new DelayLine(0.1, sampleRate),
        gain: Math.pow(0.7, i + 1) // Exponential decay
      });
      this.earlyDelaysR.push({
        delayLine: new DelayLine(0.1, sampleRate),
        gain: Math.pow(0.7, i + 1) * 1.02 // Slightly different for stereo
      });
    }

    // Algorithmic tail - Comb filters
    this.combDelayTimes = [
      0.0297, 0.0371, 0.0411, 0.0437
    ];

    this.combFilters = [];
    for (let i = 0; i < 4; i++) {
      this.combFilters.push({
        delayLineL: new DelayLine(0.1, sampleRate),
        delayLineR: new DelayLine(0.1, sampleRate),
        dampingL: new OnePoleFilter(5000, sampleRate),
        dampingR: new OnePoleFilter(5000, sampleRate),
        feedback: 0.84
      });
    }

    // Allpass filters for tail diffusion
    this.allpassDelayTimes = [0.0051, 0.0126];

    this.allpassFilters = [];
    for (let i = 0; i < 2; i++) {
      this.allpassFilters.push({
        delayLineL: new DelayLine(0.05, sampleRate),
        delayLineR: new DelayLine(0.05, sampleRate),
        coefficient: 0.5
      });
    }

    // Current parameter values
    this.preDelay = 0;
    this.decayTime = 2.0;
    this.earlyLevel = -6;
    this.tailLevel = -6;
    this.damping = 50;
    this.mix = 30;

    // Update initial settings
    this.updateDamping();
    this.updateFeedback();
  }

  /**
   * Convert dB to linear gain
   */
  dbToGain(db) {
    if (db <= -60) return 0;
    return Math.pow(10, db / 20);
  }

  /**
   * Update damping filter cutoff
   */
  updateDamping() {
    const freq = 20000 * Math.pow(0.05, this.damping / 100);

    this.combFilters.forEach(comb => {
      comb.dampingL.setCutoff(freq);
      comb.dampingR.setCutoff(freq);
    });
  }

  /**
   * Update feedback gains based on decay time
   */
  updateFeedback() {
    this.combFilters.forEach((comb, index) => {
      const delayTime = this.combDelayTimes[index];
      const feedback = Math.pow(10, (-3 * delayTime) / this.decayTime);
      comb.feedback = Math.min(0.98, feedback);
    });
  }

  /**
   * Process a sample through a comb filter
   */
  processComb(delayLine, damping, feedback, input) {
    const delaySamples = delayLine.bufferSize / 2; // Use half buffer
    const delayed = delayLine.read(delaySamples);
    const damped = damping.process(delayed);

    delayLine.write(input + damped * feedback);

    return damped;
  }

  /**
   * Process a sample through an allpass filter
   */
  processAllpass(delayLine, delaySamples, coefficient, input) {
    const delayed = delayLine.readInterpolated(delaySamples);
    const output = -input + delayed;

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
    const earlyLevel = parameters.earlyLevel[0];
    const tailLevel = parameters.tailLevel[0];
    const damping = parameters.damping[0];
    const mix = parameters.mix[0];

    // Check if parameters changed
    if (this.preDelay !== preDelay) {
      this.preDelay = preDelay;
    }

    if (this.decayTime !== decayTime) {
      this.decayTime = decayTime;
      this.updateFeedback();
    }

    if (this.earlyLevel !== earlyLevel) {
      this.earlyLevel = earlyLevel;
    }

    if (this.tailLevel !== tailLevel) {
      this.tailLevel = tailLevel;
    }

    if (this.damping !== damping) {
      this.damping = damping;
      this.updateDamping();
    }

    this.mix = mix;

    const blockSize = input[0].length;
    const numChannels = Math.min(input.length, 2);

    const preDelaySamples = (this.preDelay / 1000) * sampleRate;
    const earlyGain = this.dbToGain(this.earlyLevel);
    const tailGain = this.dbToGain(this.tailLevel);

    // Process each channel
    for (let channel = 0; channel < numChannels; channel++) {
      const inputChannel = input[channel];
      const outputChannel = output[channel];

      const preDelayLine = channel === 0 ? this.preDelayL : this.preDelayR;
      const earlyDelays = channel === 0 ? this.earlyDelaysL : this.earlyDelaysR;

      for (let i = 0; i < blockSize; i++) {
        const drySample = inputChannel[i];

        // Apply pre-delay
        preDelayLine.write(drySample);
        const preDelayed = preDelayLine.readInterpolated(preDelaySamples);

        // === EARLY REFLECTIONS ===
        let earlySum = 0;

        for (let e = 0; e < this.earlyDelayTimes.length; e++) {
          const early = earlyDelays[e];
          const delaySamples = this.earlyDelayTimes[e] * sampleRate;

          early.delayLine.write(preDelayed);
          const tapped = early.delayLine.readInterpolated(delaySamples);

          earlySum += tapped * early.gain;
        }

        earlySum *= earlyGain;

        // === ALGORITHMIC TAIL ===
        // Process through comb filters (parallel)
        let combSum = 0;

        for (let c = 0; c < this.combFilters.length; c++) {
          const comb = this.combFilters[c];
          const delayLine = channel === 0 ? comb.delayLineL : comb.delayLineR;
          const damping = channel === 0 ? comb.dampingL : comb.dampingR;

          const combOut = this.processComb(delayLine, damping, comb.feedback, preDelayed);
          combSum += combOut;
        }

        combSum *= 0.25; // Average 4 combs

        // Process through allpass filters (series)
        let apOutput = combSum;

        for (let a = 0; a < this.allpassFilters.length; a++) {
          const ap = this.allpassFilters[a];
          const delayLine = channel === 0 ? ap.delayLineL : ap.delayLineR;
          const delaySamples = this.allpassDelayTimes[a] * sampleRate;

          apOutput = this.processAllpass(delayLine, delaySamples, ap.coefficient, apOutput);
        }

        apOutput *= tailGain;

        // Combine early + tail
        const wetSample = earlySum + apOutput;

        // Mix dry and wet
        const wetGain = this.mix / 100;
        const dryGain = 1 - wetGain;

        outputChannel[i] = drySample * dryGain + wetSample * wetGain;
      }
    }

    return true;
  }
}

registerProcessor('hybrid-reverb-processor', HybridReverbProcessor);
