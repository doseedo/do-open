/**
 * Algorithmic Reverb AudioWorklet Processor
 *
 * Freeverb-style reverb implementation using:
 * - Parallel comb filters (8 total, 4 per channel)
 * - Series allpass filters (4 stages for diffusion)
 * - Frequency-dependent damping
 * - Stereo width control
 *
 * Based on Schroeder reverberator architecture.
 *
 * @author Agent 6: Reverb Plugins
 * @version 1.0.0
 */

// Import DSP utilities
importScripts('dsp-utils.js');

class ReverbProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    // Parameters
    this.roomSize = 0.5;
    this.damping = 0.5;
    this.width = 1.0;
    this.mix = 0.3;
    this.predelay = 0; // milliseconds

    // Sample rate
    this.sampleRate = sampleRate;

    // Will be initialized in initializeState()
    this.combFilters = null;
    this.allpassFilters = null;
    this.predelayLine = null;

    // Initialize DSP components
    this.initializeState();

    // Listen for parameter updates from main thread
    this.port.onmessage = (event) => {
      const { type, params } = event.data;
      if (type === 'setParams') {
        this.updateParams(params);
      }
    };
  }

  /**
   * Initialize reverb DSP components
   */
  initializeState() {
    // Base comb filter delay times (in seconds)
    // Tuned to avoid modal resonances (prime-related ratios)
    const baseCombTimes = [
      0.0297, 0.0371, 0.0411, 0.0437, // Left channel
      0.0306, 0.0379, 0.0420, 0.0445  // Right channel (slightly detuned)
    ];

    // Create comb filters
    this.combFilters = baseCombTimes.map((time, index) => {
      const comb = new CombFilter(time, this.sampleRate);
      comb.setFeedback(0.84);
      comb.setDamping(5000);
      return {
        filter: comb,
        channel: index < 4 ? 0 : 1, // First 4 = left, last 4 = right
        baseTime: time
      };
    });

    // Allpass filter delay times (for diffusion)
    const allpassTimes = [0.0051, 0.0126, 0.0100, 0.0077];

    // Create allpass filters (one set per channel for stereo)
    this.allpassFilters = {
      left: allpassTimes.map(time => {
        const ap = new AllpassFilter(time, this.sampleRate);
        ap.setFeedback(0.5);
        return ap;
      }),
      right: allpassTimes.map(time => {
        const ap = new AllpassFilter(time * 1.02, this.sampleRate); // Slightly detuned
        ap.setFeedback(0.5);
        return ap;
      })
    };

    // Pre-delay line (if needed)
    this.updatePredelay();
  }

  /**
   * Update pre-delay line
   */
  updatePredelay() {
    if (this.predelay > 0) {
      const predelaySeconds = this.predelay / 1000;
      this.predelayLine = new DelayLine(predelaySeconds + 0.01, this.sampleRate);
    } else {
      this.predelayLine = null;
    }
  }

  /**
   * Update parameters from main thread
   * @param {Object} params - Parameter values
   */
  updateParams(params) {
    let needsUpdate = false;

    if (params.roomSize !== undefined) {
      this.roomSize = Math.max(0, Math.min(100, params.roomSize)) / 100;
      needsUpdate = true;
    }

    if (params.damping !== undefined) {
      this.damping = Math.max(0, Math.min(100, params.damping)) / 100;
      needsUpdate = true;
    }

    if (params.width !== undefined) {
      this.width = Math.max(0, Math.min(100, params.width)) / 100;
    }

    if (params.mix !== undefined) {
      this.mix = Math.max(0, Math.min(100, params.mix)) / 100;
    }

    if (params.predelay !== undefined) {
      this.predelay = Math.max(0, Math.min(250, params.predelay));
      this.updatePredelay();
    }

    if (params.decayTime !== undefined) {
      this.updateDecayTime(params.decayTime);
    }

    if (needsUpdate) {
      this.updateFilters();
    }
  }

  /**
   * Update filter parameters based on room size and damping
   */
  updateFilters() {
    // Update comb filters
    const scale = 0.5 + this.roomSize * 1.5; // 0.5x to 2x room size

    this.combFilters.forEach(({ filter, baseTime }) => {
      // Scale delay time for room size
      const newDelay = baseTime * scale;
      filter.delaySamples = newDelay * this.sampleRate;

      // Update damping (highpass frequency based on damping parameter)
      // More damping = lower cutoff = more high frequency absorption
      const dampingFreq = 20000 * Math.pow(0.05, this.damping);
      filter.setDamping(dampingFreq);
    });
  }

  /**
   * Update decay time (affects feedback gain)
   * @param {number} decayTime - Decay time in seconds (RT60)
   */
  updateDecayTime(decayTime) {
    const seconds = Math.max(0.1, Math.min(20, decayTime));

    this.combFilters.forEach(({ filter, baseTime }) => {
      const delayTime = baseTime * (0.5 + this.roomSize * 1.5);
      // RT60 formula: feedback = 10^(-3 * delay / RT60)
      const feedback = Math.pow(10, (-3 * delayTime) / seconds);
      filter.setFeedback(Math.min(0.98, feedback)); // Clamp for stability
    });
  }

  /**
   * Process a single sample through comb filters
   * @param {number} input - Input sample
   * @param {number} channel - Channel (0=left, 1=right)
   * @returns {number} Processed sample
   */
  processCombFilters(input, channel) {
    let output = 0;

    // Sum outputs from all comb filters for this channel
    for (const { filter, channel: combChannel } of this.combFilters) {
      if (combChannel === channel) {
        output += filter.process(input);
      }
    }

    // Normalize by number of comb filters per channel
    return output * 0.25; // 1/4 for 4 comb filters
  }

  /**
   * Process a single sample through allpass filters
   * @param {number} input - Input sample
   * @param {number} channel - Channel (0=left, 1=right)
   * @returns {number} Processed sample
   */
  processAllpassFilters(input, channel) {
    const filters = channel === 0 ? this.allpassFilters.left : this.allpassFilters.right;

    let output = input;
    for (const filter of filters) {
      output = filter.process(output);
    }

    return output;
  }

  /**
   * Process audio (called by Web Audio API)
   * @param {Float32Array[][]} inputs - Input buffers
   * @param {Float32Array[][]} outputs - Output buffers
   * @param {Object} parameters - AudioParam values
   * @returns {boolean} Keep processor alive
   */
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    const output = outputs[0];

    // If no input, return silence
    if (!input || input.length === 0) {
      return true;
    }

    const inputL = input[0];
    const inputR = input.length > 1 ? input[1] : input[0];
    const outputL = output[0];
    const outputR = output.length > 1 ? output[1] : output[0];

    // Process each sample
    for (let i = 0; i < inputL.length; i++) {
      // Convert to mono for reverb input
      const monoInput = input.length > 1 ? (inputL[i] + inputR[i]) * 0.5 : inputL[i];

      // Apply pre-delay if configured
      let delayedInput = monoInput;
      if (this.predelayLine && this.predelay > 0) {
        this.predelayLine.write(monoInput);
        const predelaySamples = (this.predelay / 1000) * this.sampleRate;
        delayedInput = this.predelayLine.readInterpolated(predelaySamples);
      }

      // Process through comb filters (parallel)
      const combL = this.processCombFilters(delayedInput, 0);
      const combR = this.processCombFilters(delayedInput, 1);

      // Process through allpass filters (series) for diffusion
      const wetL = this.processAllpassFilters(combL, 0);
      const wetR = this.processAllpassFilters(combR, 1);

      // Apply stereo width
      // Mid-side processing: extract mid and side signals
      const mid = (wetL + wetR) * 0.5;
      const side = (wetL - wetR) * 0.5;

      // Adjust width (0 = mono, 1 = full stereo, >1 = wider)
      const wideMid = mid;
      const wideSide = side * this.width;

      // Convert back to left-right
      const finalWetL = wideMid + wideSide;
      const finalWetR = wideMid - wideSide;

      // Mix dry and wet signals
      const dryGain = 1 - this.mix;
      const wetGain = this.mix;

      outputL[i] = inputL[i] * dryGain + finalWetL * wetGain;

      if (outputR) {
        outputR[i] = inputR[i] * dryGain + finalWetR * wetGain;
      }
    }

    // Keep processor alive
    return true;
  }
}

// Register the processor
registerProcessor('reverb-processor', ReverbProcessor);
