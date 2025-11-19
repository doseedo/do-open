/**
 * Reverb.js - Algorithmic Reverb Plugin
 *
 * Creates artificial room ambience using feedback delay networks (FDN)
 * Based on Freeverb/Schroeder reverberator architecture with:
 * - Early reflections (simulating room geometry)
 * - Diffuse reverb tail (parallel comb filters + series all-pass)
 * - Frequency-dependent damping
 * - Subtle modulation to avoid metallic artifacts
 * - Stereo width control
 *
 * @author Agent 5: Reverb & Spatial Effects
 * @version 1.0.0
 */

class Reverb {
  /**
   * Create a new Reverb instance
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial parameters
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.sampleRate = audioContext.sampleRate;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Pre-delay
    this.predelay = audioContext.createDelay(0.25);

    // Early reflections network
    this.earlyReflections = this.createEarlyReflections();

    // Reverb tail (diffuse network)
    this.reverbTail = this.createReverbTail();

    // Gain stages
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();
    this.earlyGain = audioContext.createGain();
    this.tailGain = audioContext.createGain();

    // Stereo width control nodes
    this.stereoSplitter = audioContext.createChannelSplitter(2);
    this.stereoMerger = audioContext.createChannelMerger(2);
    this.midGain = audioContext.createGain();
    this.sideGain = audioContext.createGain();

    // Modulation (LFO for subtle pitch variation)
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();
    this.lfo.frequency.value = 0.5;
    this.lfoGain.gain.value = 0;
    this.lfo.start();

    // Setup routing
    this.setupRouting();

    // Initialize with default parameters
    this.initialize(options);
  }

  /**
   * Create early reflections network
   * Simulates discrete echoes from room surfaces
   * @returns {Array} Array of early reflection delay nodes
   */
  createEarlyReflections() {
    const delays = [];

    // Early reflection delay times (in seconds)
    // Based on typical room geometry
    const delayTimes = [
      0.019, 0.022, 0.027, 0.031, 0.037, 0.043, 0.048, 0.053
    ];

    delayTimes.forEach((time, index) => {
      const delay = this.context.createDelay(0.1);
      const gain = this.context.createGain();
      const pan = this.context.createStereoPanner();

      delay.delayTime.value = time;
      // Exponential decay for early reflections
      gain.gain.value = Math.pow(0.7, index + 1);
      // Spread reflections across stereo field
      pan.pan.value = (Math.random() * 2 - 1) * 0.6;

      delays.push({ delay, gain, pan });
    });

    return delays;
  }

  /**
   * Create reverb tail using Freeverb-style architecture
   * Parallel comb filters + series all-pass filters
   * @returns {Object} Reverb tail nodes
   */
  createReverbTail() {
    const tail = {
      combFilters: [],
      allpassFilters: [],
      modulation: []
    };

    // Comb filter delay times (in seconds)
    // Tuned to avoid resonances (prime-related ratios)
    // 8 comb filters (4 per stereo channel)
    const combDelayTimes = [
      0.0297, 0.0371, 0.0411, 0.0437, // Left channel
      0.0306, 0.0379, 0.0420, 0.0445  // Right channel (slightly detuned)
    ];

    combDelayTimes.forEach((time, index) => {
      const delay = this.context.createDelay(0.1);
      const feedback = this.context.createGain();
      const damping = this.context.createBiquadFilter();
      const output = this.context.createGain();

      delay.delayTime.value = time;
      feedback.gain.value = 0.84; // Initial decay factor
      damping.type = 'lowpass';
      damping.frequency.value = 5000;
      damping.Q.value = 0.5;
      output.gain.value = 0.125; // 1/8 for mixing 8 combs

      tail.combFilters.push({
        delay,
        feedback,
        damping,
        output,
        channel: index < 4 ? 0 : 1 // Left or right channel
      });
    });

    // All-pass filter delay times (for diffusion)
    // Series configuration to increase echo density
    const allpassTimes = [
      0.0051, 0.0126, 0.0100, 0.0077
    ];

    allpassTimes.forEach(time => {
      const delayL = this.context.createDelay(0.1);
      const delayR = this.context.createDelay(0.1);
      const feedbackL = this.context.createGain();
      const feedbackR = this.context.createGain();
      const feedforwardL = this.context.createGain();
      const feedforwardR = this.context.createGain();

      delayL.delayTime.value = time;
      delayR.delayTime.value = time * 1.02; // Slight stereo detuning
      feedbackL.gain.value = 0.5;
      feedbackR.gain.value = 0.5;
      feedforwardL.gain.value = -0.5;
      feedforwardR.gain.value = -0.5;

      tail.allpassFilters.push({
        delayL,
        delayR,
        feedbackL,
        feedbackR,
        feedforwardL,
        feedforwardR
      });
    });

    return tail;
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path: input → predelay → early reflections + reverb tail
    this.input.connect(this.predelay);

    // ===== EARLY REFLECTIONS =====
    this.earlyReflections.forEach(er => {
      this.predelay.connect(er.delay);
      er.delay.connect(er.gain);
      er.gain.connect(er.pan);
      er.pan.connect(this.earlyGain);
    });

    // ===== REVERB TAIL =====
    // Comb filters (parallel configuration)
    const leftCombSum = this.context.createGain();
    const rightCombSum = this.context.createGain();

    this.reverbTail.combFilters.forEach(comb => {
      // Input to comb filter
      this.predelay.connect(comb.delay);

      // Feedback loop: delay → damping → feedback → delay
      comb.delay.connect(comb.damping);
      comb.damping.connect(comb.feedback);
      comb.feedback.connect(comb.delay);

      // Output from feedback to stereo sum
      comb.damping.connect(comb.output);
      if (comb.channel === 0) {
        comb.output.connect(leftCombSum);
      } else {
        comb.output.connect(rightCombSum);
      }
    });

    // All-pass filters (series configuration for diffusion)
    // Create stereo path
    let currentL = leftCombSum;
    let currentR = rightCombSum;

    this.reverbTail.allpassFilters.forEach(ap => {
      // Left channel all-pass
      const inputL = this.context.createGain();
      currentL.connect(inputL);

      inputL.connect(ap.delayL);
      inputL.connect(ap.feedforwardL);
      ap.delayL.connect(ap.feedbackL);
      ap.feedbackL.connect(ap.delayL); // Feedback loop

      const outputL = this.context.createGain();
      ap.delayL.connect(outputL);
      ap.feedforwardL.connect(outputL);
      currentL = outputL;

      // Right channel all-pass
      const inputR = this.context.createGain();
      currentR.connect(inputR);

      inputR.connect(ap.delayR);
      inputR.connect(ap.feedforwardR);
      ap.delayR.connect(ap.feedbackR);
      ap.feedbackR.connect(ap.delayR); // Feedback loop

      const outputR = this.context.createGain();
      ap.delayR.connect(outputR);
      ap.feedforwardR.connect(outputR);
      currentR = outputR;
    });

    // Connect to tail gain
    currentL.connect(this.tailGain);
    currentR.connect(this.tailGain);

    // LFO modulation (connected to some delay times for subtle variation)
    this.lfo.connect(this.lfoGain);

    // Mix early reflections and tail
    this.earlyGain.connect(this.wetGain);
    this.tailGain.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  /**
   * Initialize parameters
   * @param {Object} options - Parameter values
   */
  initialize(options) {
    this.setPreDelay(options.preDelay || 0);
    this.setDecayTime(options.decayTime || 2.0);
    this.setSize(options.size || 50);
    this.setDiffusion(options.diffusion || 70);
    this.setDamping(options.damping || 50);
    this.setModulation(options.modulation || 20);
    this.setStereoWidth(options.stereoWidth || 100);
    this.setEarlyLevel(options.earlyLevel || -12);
    this.setTailLevel(options.tailLevel || -6);
    this.setMix(options.mix || 30);
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
   * Set decay time
   * @param {number} seconds - Decay time in seconds (0.1-20)
   */
  setDecayTime(seconds) {
    seconds = Math.max(0.1, Math.min(20, seconds));

    // Calculate feedback gain for desired decay time
    // Using RT60 formula: feedback = 10^(-3 * delay / RT60)
    this.reverbTail.combFilters.forEach(comb => {
      const delayTime = comb.delay.delayTime.value;
      const feedback = Math.pow(10, (-3 * delayTime) / seconds);
      comb.feedback.gain.setTargetAtTime(
        Math.min(0.98, feedback), // Clamp to prevent runaway
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set room size
   * @param {number} percent - Size percentage (0-100)
   */
  setSize(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const scale = 0.5 + (percent / 100) * 1.5; // 0.5x to 2x

    // Scale comb filter delay times
    const baseCombTimes = [
      0.0297, 0.0371, 0.0411, 0.0437,
      0.0306, 0.0379, 0.0420, 0.0445
    ];

    this.reverbTail.combFilters.forEach((comb, index) => {
      const newTime = baseCombTimes[index] * scale;
      comb.delay.delayTime.setTargetAtTime(
        newTime,
        this.context.currentTime,
        0.01
      );
    });

    // Scale all-pass delay times
    const baseAllpassTimes = [0.0051, 0.0126, 0.0100, 0.0077];
    this.reverbTail.allpassFilters.forEach((ap, index) => {
      const newTime = baseAllpassTimes[index] * scale;
      ap.delayL.delayTime.setTargetAtTime(
        newTime,
        this.context.currentTime,
        0.01
      );
      ap.delayR.delayTime.setTargetAtTime(
        newTime * 1.02,
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set diffusion amount
   * @param {number} percent - Diffusion percentage (0-100)
   */
  setDiffusion(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const amount = percent / 100 * 0.7; // 0 to 0.7

    // Adjust all-pass feedback gains
    this.reverbTail.allpassFilters.forEach(ap => {
      ap.feedbackL.gain.setTargetAtTime(
        amount,
        this.context.currentTime,
        0.01
      );
      ap.feedbackR.gain.setTargetAtTime(
        amount,
        this.context.currentTime,
        0.01
      );
      ap.feedforwardL.gain.setTargetAtTime(
        -amount,
        this.context.currentTime,
        0.01
      );
      ap.feedforwardR.gain.setTargetAtTime(
        -amount,
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set high-frequency damping
   * @param {number} percent - Damping percentage (0-100)
   */
  setDamping(percent) {
    percent = Math.max(0, Math.min(100, percent));
    // Map to lowpass frequency: 100% damping = 1kHz, 0% = 20kHz
    const freq = 20000 * Math.pow(0.05, percent / 100);

    this.reverbTail.combFilters.forEach(comb => {
      comb.damping.frequency.setTargetAtTime(
        freq,
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set modulation amount
   * @param {number} percent - Modulation percentage (0-100)
   */
  setModulation(percent) {
    percent = Math.max(0, Math.min(100, percent));
    // Subtle modulation to prevent metallic sound
    const depth = percent / 100 * 0.0003; // Max 0.3ms variation

    this.lfoGain.gain.setTargetAtTime(
      depth,
      this.context.currentTime,
      0.01
    );

    // Vary LFO frequency slightly
    this.lfo.frequency.setTargetAtTime(
      0.3 + (percent / 100) * 0.7, // 0.3 to 1 Hz
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set stereo width
   * @param {number} percent - Width percentage (0-100)
   */
  setStereoWidth(percent) {
    percent = Math.max(0, Math.min(100, percent));
    // 0% = mono, 100% = full stereo
    // Using mid-side technique
    const width = percent / 100;
    const mid = 1.0;
    const side = width;

    this.midGain.gain.setTargetAtTime(
      mid,
      this.context.currentTime,
      0.01
    );
    this.sideGain.gain.setTargetAtTime(
      side,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set early reflections level
   * @param {number} db - Level in dB (-inf to 0)
   */
  setEarlyLevel(db) {
    const gain = db <= -60 ? 0 : Math.pow(10, db / 20);
    this.earlyGain.gain.setTargetAtTime(
      gain,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set reverb tail level
   * @param {number} db - Level in dB (-inf to 0)
   */
  setTailLevel(db) {
    const gain = db <= -60 ? 0 : Math.pow(10, db / 20);
    this.tailGain.gain.setTargetAtTime(
      gain,
      this.context.currentTime,
      0.01
    );
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
   * Connect this reverb to an audio node
   * @param {AudioNode} destination - Destination node
   * @returns {Reverb} This reverb instance (for chaining)
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect this reverb
   * @returns {Reverb} This reverb instance (for chaining)
   */
  disconnect() {
    this.output.disconnect();
    return this;
  }

  /**
   * Cleanup and release resources
   */
  dispose() {
    this.lfo.stop();
    this.disconnect();

    // Disconnect all internal nodes
    this.earlyReflections.forEach(er => {
      er.delay.disconnect();
      er.gain.disconnect();
      er.pan.disconnect();
    });

    this.reverbTail.combFilters.forEach(comb => {
      comb.delay.disconnect();
      comb.feedback.disconnect();
      comb.damping.disconnect();
      comb.output.disconnect();
    });

    this.reverbTail.allpassFilters.forEach(ap => {
      ap.delayL.disconnect();
      ap.delayR.disconnect();
      ap.feedbackL.disconnect();
      ap.feedbackR.disconnect();
      ap.feedforwardL.disconnect();
      ap.feedforwardR.disconnect();
    });
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Reverb;
}
