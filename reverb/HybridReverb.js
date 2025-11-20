/**
 * HybridReverb.js - Hybrid Reverb Plugin
 *
 * Combines convolution reverb (realistic early reflections) with
 * algorithmic tail (efficient and controllable)
 *
 * Features:
 * - Load impulse response files (WAV)
 * - Convolution for early reflections
 * - Algorithmic tail for efficiency
 * - Crossover blend between IR and algorithmic reverb
 * - IR trimming and normalization
 *
 * @author Agent 5: Reverb & Spatial Effects
 * @version 1.0.0
 */

class HybridReverb {
  /**
   * Create a new HybridReverb instance
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial parameters
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.sampleRate = audioContext.sampleRate;

    // Store original impulse response buffer
    this.originalIRBuffer = null;
    this.currentIRBuffer = null;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Convolution for early reflections
    this.convolver = audioContext.createConvolver();
    this.convolverGain = audioContext.createGain();

    // Pre-delay before convolution
    this.predelay = audioContext.createDelay(0.25);

    // Crossover filters (split signal for IR vs algorithmic)
    this.lowpassForIR = audioContext.createBiquadFilter();
    this.highpassForAlgo = audioContext.createBiquadFilter();
    this.lowpassForIR.type = 'lowpass';
    this.highpassForAlgo.type = 'highpass';

    // Algorithmic tail (using basic reverb structure)
    this.algorithmicTail = this.createAlgorithmicTail();

    // Dry/wet mix
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    // Setup routing
    this.setupRouting();

    // Initialize with default parameters
    this.initialize(options);
  }

  /**
   * Create algorithmic tail network
   * Simplified reverb for the tail portion
   * @returns {Object} Algorithmic tail nodes
   */
  createAlgorithmicTail() {
    const tail = {
      combFilters: [],
      allpassFilters: []
    };

    // 4 comb filters for efficiency
    const combDelayTimes = [0.0297, 0.0371, 0.0411, 0.0437];

    combDelayTimes.forEach(time => {
      const delay = this.context.createDelay(0.1);
      const feedback = this.context.createGain();
      const damping = this.context.createBiquadFilter();
      const output = this.context.createGain();

      delay.delayTime.value = time;
      feedback.gain.value = 0.84;
      damping.type = 'lowpass';
      damping.frequency.value = 5000;
      damping.Q.value = 0.5;
      output.gain.value = 0.25; // 1/4 for mixing 4 combs

      tail.combFilters.push({ delay, feedback, damping, output });
    });

    // 2 all-pass filters for diffusion
    const allpassTimes = [0.0051, 0.0126];

    allpassTimes.forEach(time => {
      const delay = this.context.createDelay(0.1);
      const feedback = this.context.createGain();
      const feedforward = this.context.createGain();

      delay.delayTime.value = time;
      feedback.gain.value = 0.5;
      feedforward.gain.value = -0.5;

      tail.allpassFilters.push({ delay, feedback, feedforward });
    });

    // Create gain nodes for tail
    tail.input = this.context.createGain();
    tail.output = this.context.createGain();

    // Connect comb filters in parallel
    tail.combFilters.forEach(comb => {
      tail.input.connect(comb.delay);
      comb.delay.connect(comb.damping);
      comb.damping.connect(comb.feedback);
      comb.feedback.connect(comb.delay); // Feedback loop
      comb.damping.connect(comb.output);
      comb.output.connect(tail.output);
    });

    // Connect all-pass filters in series
    let current = tail.output;
    tail.allpassFilters.forEach(ap => {
      const apInput = this.context.createGain();
      current.disconnect(tail.output);
      current.connect(apInput);

      apInput.connect(ap.delay);
      apInput.connect(ap.feedforward);
      ap.delay.connect(ap.feedback);
      ap.feedback.connect(ap.delay); // Feedback loop

      const apOutput = this.context.createGain();
      ap.delay.connect(apOutput);
      ap.feedforward.connect(apOutput);

      current = apOutput;
    });

    // Reconnect final output
    const finalOutput = tail.output;
    tail.output = current;

    return tail;
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path with pre-delay
    this.input.connect(this.predelay);

    // === Crossover split ===
    // Low frequencies → convolution (early reflections)
    this.predelay.connect(this.lowpassForIR);
    this.lowpassForIR.connect(this.convolver);
    this.convolver.connect(this.convolverGain);
    this.convolverGain.connect(this.wetGain);

    // High frequencies → algorithmic tail
    this.predelay.connect(this.highpassForAlgo);
    this.highpassForAlgo.connect(this.algorithmicTail.input);
    this.algorithmicTail.output.connect(this.wetGain);

    // Wet to output
    this.wetGain.connect(this.output);
  }

  /**
   * Initialize parameters
   * @param {Object} options - Parameter values
   */
  initialize(options) {
    this.setPreDelay(options.predelay || 0);
    this.setDecayTime(options.decayTime || 2.0);
    this.setCrossover(options.crossover || 2000);
    this.setIRLevel(options.erLevel || -6);
    this.setTailLevel(options.tailLevel || -6);
    this.setDamping(options.damping || 50);
    this.setMix(options.mix || 30);

    // Load impulse response if provided
    if (options.impulseResponse) {
      this.loadImpulseResponse(options.impulseResponse);
    }
  }

  /**
   * Load impulse response from URL
   * @param {string} url - URL of WAV file
   * @returns {Promise<boolean>} Success status
   */
  async loadImpulseResponse(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.context.decodeAudioData(arrayBuffer);

      this.originalIRBuffer = audioBuffer;
      this.currentIRBuffer = audioBuffer;
      this.convolver.buffer = audioBuffer;

      console.log(`Loaded IR: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.numberOfChannels} channels`);
      return true;
    } catch (error) {
      console.error('Failed to load impulse response:', error);
      return false;
    }
  }

  /**
   * Load impulse response from file
   * @param {File} file - WAV file object
   * @returns {Promise<boolean>} Success status
   */
  async loadImpulseResponseFile(file) {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await this.context.decodeAudioData(arrayBuffer);

      this.originalIRBuffer = audioBuffer;
      this.currentIRBuffer = audioBuffer;
      this.convolver.buffer = audioBuffer;

      console.log(`Loaded IR: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.numberOfChannels} channels`);
      return true;
    } catch (error) {
      console.error('Failed to load impulse response:', error);
      return false;
    }
  }

  /**
   * Trim impulse response length
   * @param {number} percent - Length percentage (0-100)
   */
  setIRLength(percent) {
    if (!this.originalIRBuffer) return;

    percent = Math.max(1, Math.min(100, percent));
    const targetLength = Math.floor(
      this.originalIRBuffer.length * (percent / 100)
    );

    // Create trimmed buffer
    const trimmedBuffer = this.context.createBuffer(
      this.originalIRBuffer.numberOfChannels,
      targetLength,
      this.originalIRBuffer.sampleRate
    );

    // Copy trimmed audio data
    for (let ch = 0; ch < this.originalIRBuffer.numberOfChannels; ch++) {
      const originalData = this.originalIRBuffer.getChannelData(ch);
      const trimmedData = trimmedBuffer.getChannelData(ch);
      trimmedData.set(originalData.subarray(0, targetLength));

      // Apply fade-out to prevent clicks (last 10ms)
      const fadeLength = Math.min(
        Math.floor(this.sampleRate * 0.01),
        targetLength
      );
      for (let i = 0; i < fadeLength; i++) {
        const fadeGain = (fadeLength - i) / fadeLength;
        trimmedData[targetLength - fadeLength + i] *= fadeGain;
      }
    }

    this.currentIRBuffer = trimmedBuffer;
    this.convolver.buffer = trimmedBuffer;
  }

  /**
   * Normalize impulse response
   * Adjusts IR to prevent clipping
   */
  normalizeIR() {
    if (!this.originalIRBuffer) return;

    // Find peak value across all channels
    let peak = 0;
    for (let ch = 0; ch < this.originalIRBuffer.numberOfChannels; ch++) {
      const data = this.originalIRBuffer.getChannelData(ch);
      for (let i = 0; i < data.length; i++) {
        peak = Math.max(peak, Math.abs(data[i]));
      }
    }

    if (peak === 0) return;

    // Create normalized buffer
    const normalizedBuffer = this.context.createBuffer(
      this.originalIRBuffer.numberOfChannels,
      this.originalIRBuffer.length,
      this.originalIRBuffer.sampleRate
    );

    const gain = 1.0 / peak;

    for (let ch = 0; ch < this.originalIRBuffer.numberOfChannels; ch++) {
      const originalData = this.originalIRBuffer.getChannelData(ch);
      const normalizedData = normalizedBuffer.getChannelData(ch);

      for (let i = 0; i < originalData.length; i++) {
        normalizedData[i] = originalData[i] * gain;
      }
    }

    this.originalIRBuffer = normalizedBuffer;
    this.currentIRBuffer = normalizedBuffer;
    this.convolver.buffer = normalizedBuffer;
  }

  /**
   * Set pre-delay time
   * @param {number} ms - Pre-delay in milliseconds (0-250)
   */
  setPreDelay(ms) {
    const seconds = Math.max(0, Math.min(250, ms)) / 1000;
    this.predelay.delayTime.setTargetAtTime(
      seconds,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set decay time for algorithmic tail
   * @param {number} seconds - Decay time in seconds (0.1-20)
   */
  setDecayTime(seconds) {
    seconds = Math.max(0.1, Math.min(20, seconds));

    // Calculate feedback gain for desired decay time
    this.algorithmicTail.combFilters.forEach(comb => {
      const delayTime = comb.delay.delayTime.value;
      const feedback = Math.pow(10, (-3 * delayTime) / seconds);
      comb.feedback.gain.setTargetAtTime(
        Math.min(0.98, feedback),
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set crossover frequency
   * @param {number} freq - Frequency in Hz (200-8000)
   */
  setCrossover(freq) {
    freq = Math.max(200, Math.min(8000, freq));

    this.lowpassForIR.frequency.setTargetAtTime(
      freq,
      this.context.currentTime,
      0.01
    );
    this.highpassForAlgo.frequency.setTargetAtTime(
      freq,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set IR (early reflections) level
   * @param {number} db - Level in dB (-inf to 0)
   */
  setIRLevel(db) {
    const gain = db <= -60 ? 0 : Math.pow(10, db / 20);
    this.convolverGain.gain.setTargetAtTime(
      gain,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set algorithmic tail level
   * @param {number} db - Level in dB (-inf to 0)
   */
  setTailLevel(db) {
    const gain = db <= -60 ? 0 : Math.pow(10, db / 20);
    this.algorithmicTail.output.gain.setTargetAtTime(
      gain,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set high-frequency damping for algorithmic tail
   * @param {number} percent - Damping percentage (0-100)
   */
  setDamping(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const freq = 20000 * Math.pow(0.05, percent / 100);

    this.algorithmicTail.combFilters.forEach(comb => {
      comb.damping.frequency.setTargetAtTime(
        freq,
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Mix percentage (0-100)
   */
  setMix(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const wet = percent / 100;
    const dry = 1 - wet;

    this.dryGain.gain.setTargetAtTime(
      dry,
      this.context.currentTime,
      0.01
    );
    this.wetGain.gain.setTargetAtTime(
      wet,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Get current impulse response info
   * @returns {Object|null} IR info or null if no IR loaded
   */
  getIRInfo() {
    if (!this.currentIRBuffer) return null;

    return {
      duration: this.currentIRBuffer.duration,
      sampleRate: this.currentIRBuffer.sampleRate,
      numberOfChannels: this.currentIRBuffer.numberOfChannels,
      length: this.currentIRBuffer.length
    };
  }

  /**
   * Connect this reverb to an audio node
   * @param {AudioNode} destination - Destination node
   * @returns {HybridReverb} This reverb instance (for chaining)
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect this reverb
   * @returns {HybridReverb} This reverb instance (for chaining)
   */
  disconnect() {
    this.output.disconnect();
    return this;
  }

  /**
   * Cleanup and release resources
   */
  dispose() {
    this.disconnect();

    // Disconnect algorithmic tail
    this.algorithmicTail.combFilters.forEach(comb => {
      comb.delay.disconnect();
      comb.feedback.disconnect();
      comb.damping.disconnect();
      comb.output.disconnect();
    });

    this.algorithmicTail.allpassFilters.forEach(ap => {
      ap.delay.disconnect();
      ap.feedback.disconnect();
      ap.feedforward.disconnect();
    });

    // Clear buffers
    this.originalIRBuffer = null;
    this.currentIRBuffer = null;
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = HybridReverb;
}
