/**
 * Phaser Effect
 * Sweeping notches created by all-pass filters with LFO modulation
 *
 * @class Phaser
 * @param {AudioContext} audioContext - Web Audio API context
 * @param {Object} options - Configuration options
 */
class Phaser {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // All-pass filter stages
    this.stages = [];
    this.maxStages = 12;
    this._numStages = 6;

    // Create maximum number of stages
    for (let i = 0; i < this.maxStages; i++) {
      const allpass = this.context.createBiquadFilter();
      allpass.type = 'allpass';
      allpass.Q.value = 1;
      allpass.frequency.value = 1000;
      this.stages.push({
        filter: allpass,
        active: i < this._numStages
      });
    }

    // LFO for frequency modulation
    this.lfo = this.context.createOscillator();
    this.lfoGain = this.context.createGain();

    // Feedback
    this.feedback = this.context.createGain();

    // Dry/Wet mix
    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();

    // Stage output router (for variable stage count)
    this.stageRouter = this.context.createGain();

    // Parameters
    this._rate = 0.5;          // LFO speed in Hz
    this._depth = 50;          // Modulation intensity (0-100%)
    this._feedbackAmount = 0;  // Feedback amount (0-100%)
    this._frequency = 1000;    // Center frequency (200-8000 Hz)
    this._spread = 50;         // Spacing between notches (0-100%)
    this._waveform = 'sine';   // LFO waveform
    this._mix = 50;            // Dry/wet mix (0-100%)

    // Setup audio routing
    this.setupRouting();

    // Initialize with default or provided options
    this.initialize(options);
  }

  /**
   * Setup audio routing for all nodes
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path - cascade all-pass filters
    // Input -> First active stage
    this.input.connect(this.stages[0].filter);

    // Chain all stages together
    for (let i = 0; i < this.maxStages - 1; i++) {
      this.stages[i].filter.connect(this.stages[i + 1].filter);
    }

    // Last stage connects to router
    this.stages[this.maxStages - 1].filter.connect(this.stageRouter);

    // Also connect intermediate stages to router for variable stage count
    for (let i = 0; i < this.maxStages; i++) {
      this.stages[i].filter.connect(this.stageRouter);
    }

    // Router -> Feedback -> Back to first stage
    this.stageRouter.connect(this.feedback);
    this.feedback.connect(this.stages[0].filter);

    // Router -> Wet output
    this.stageRouter.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // LFO modulates all filter frequencies
    this.lfo.connect(this.lfoGain);

    this.stages.forEach(stage => {
      this.lfoGain.connect(stage.filter.frequency);
    });

    // Set initial LFO
    this.lfo.type = this._waveform;
    this.lfo.frequency.value = this._rate;
    this.lfo.start();
  }

  /**
   * Initialize or update parameters
   * @param {Object} options - Parameter values
   */
  initialize(options = {}) {
    if (options.rate !== undefined) this.setRate(options.rate);
    if (options.depth !== undefined) this.setDepth(options.depth);
    if (options.feedback !== undefined) this.setFeedback(options.feedback);
    if (options.stages !== undefined) this.setStages(options.stages);
    if (options.frequency !== undefined) this.setFrequency(options.frequency);
    if (options.spread !== undefined) this.setSpread(options.spread);
    if (options.waveform !== undefined) this.setWaveform(options.waveform);
    if (options.mix !== undefined) this.setMix(options.mix);

    // Update active stages
    this.updateActiveStages();
  }

  /**
   * Update routing based on active stage count
   */
  updateActiveStages() {
    // Disconnect all from router first
    this.stages.forEach(stage => {
      try {
        stage.filter.disconnect(this.stageRouter);
      } catch (e) {
        // Already disconnected
      }
    });

    // Connect only the last active stage to router
    const lastActiveIndex = this._numStages - 1;
    if (lastActiveIndex >= 0 && lastActiveIndex < this.stages.length) {
      this.stages[lastActiveIndex].filter.connect(this.stageRouter);
    }

    // Update active flags
    this.stages.forEach((stage, index) => {
      stage.active = index < this._numStages;
    });
  }

  /**
   * Set LFO rate (speed of modulation)
   * @param {number} hz - Frequency in Hz (0.01 to 10)
   */
  setRate(hz) {
    this._rate = Math.max(0.01, Math.min(10, hz));
    this.lfo.frequency.setValueAtTime(this._rate, this.context.currentTime);
  }

  /**
   * Set modulation depth
   * @param {number} percent - Depth percentage (0 to 100)
   */
  setDepth(percent) {
    this._depth = Math.max(0, Math.min(100, percent));
    const depth = this._depth / 100;

    // Frequency modulation range (±2000 Hz at 100% depth)
    this.lfoGain.gain.setValueAtTime(depth * 2000, this.context.currentTime);
  }

  /**
   * Set feedback amount
   * @param {number} percent - Feedback percentage (0 to 100)
   */
  setFeedback(percent) {
    this._feedbackAmount = Math.max(0, Math.min(100, percent));
    const feedback = this._feedbackAmount / 100;

    // Limit feedback to prevent runaway
    this.feedback.gain.setValueAtTime(feedback * 0.9, this.context.currentTime);
  }

  /**
   * Set number of all-pass filter stages
   * @param {number} num - Number of stages (4, 6, 8, or 12)
   */
  setStages(num) {
    const validStages = [4, 6, 8, 12];
    if (validStages.includes(num)) {
      this._numStages = num;
      this.updateActiveStages();
      this.updateFilterFrequencies();
    }
  }

  /**
   * Set center frequency
   * @param {number} hz - Center frequency in Hz (200 to 8000)
   */
  setFrequency(hz) {
    this._frequency = Math.max(200, Math.min(8000, hz));
    this.updateFilterFrequencies();
  }

  /**
   * Set frequency spread between stages
   * @param {number} percent - Spread percentage (0 to 100)
   */
  setSpread(percent) {
    this._spread = Math.max(0, Math.min(100, percent));
    this.updateFilterFrequencies();
  }

  /**
   * Update all filter frequencies based on center, spread, and stage count
   */
  updateFilterFrequencies() {
    const spread = this._spread / 100;

    this.stages.forEach((stage, index) => {
      if (stage.active) {
        // Exponential spacing for more musical notches
        const spreadFactor = Math.pow(2, (index / this._numStages) * spread);
        const frequency = this._frequency * spreadFactor;

        stage.filter.frequency.setValueAtTime(
          Math.max(200, Math.min(8000, frequency)),
          this.context.currentTime
        );
      }
    });
  }

  /**
   * Set LFO waveform
   * @param {string} type - Waveform type ('sine', 'triangle', 'square', 'sawtooth')
   */
  setWaveform(type) {
    const validTypes = ['sine', 'triangle', 'square', 'sawtooth'];
    if (validTypes.includes(type)) {
      this._waveform = type;

      // Create new LFO with new waveform
      const oldLfo = this.lfo;
      this.lfo = this.context.createOscillator();
      this.lfo.type = this._waveform;
      this.lfo.frequency.value = this._rate;

      // Reconnect to all filters
      oldLfo.disconnect();
      this.lfo.connect(this.lfoGain);
      this.lfo.start();

      // Stop old LFO
      oldLfo.stop();
    }
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet percentage (0 to 100)
   */
  setMix(percent) {
    this._mix = Math.max(0, Math.min(100, percent));
    const wet = this._mix / 100;
    const dry = 1 - wet;

    // Use equal power crossfade
    this.dryGain.gain.setValueAtTime(Math.cos(wet * Math.PI / 2), this.context.currentTime);
    this.wetGain.gain.setValueAtTime(Math.sin(wet * Math.PI / 2), this.context.currentTime);
  }

  /**
   * Get current parameter values
   * @returns {Object} Current parameter values
   */
  getParams() {
    return {
      rate: this._rate,
      depth: this._depth,
      feedback: this._feedbackAmount,
      stages: this._numStages,
      frequency: this._frequency,
      spread: this._spread,
      waveform: this._waveform,
      mix: this._mix
    };
  }

  /**
   * Connect to destination
   * @param {AudioNode} destination - Audio node to connect to
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect output
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.lfo.stop();
    this.lfo.disconnect();
    this.lfoGain.disconnect();

    this.stages.forEach(stage => {
      stage.filter.disconnect();
    });

    this.feedback.disconnect();
    this.stageRouter.disconnect();
    this.input.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
    this.output.disconnect();
  }
}

// Export for use in modules or Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Phaser;
}
