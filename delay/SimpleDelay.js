/**
 * Simple Delay
 * Basic echo effect with feedback and filtering
 *
 * Features:
 * - Tempo synchronization with BPM
 * - Feedback loop with filtering
 * - Stereo or mono operation
 * - Smooth parameter changes (no clicks)
 */

class SimpleDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    this.delay = audioContext.createDelay(5.0); // Max 5 seconds
    this.feedback = audioContext.createGain();
    this.filter = audioContext.createBiquadFilter();

    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // Ping pong nodes (stereo bounce)
    this.splitter = audioContext.createChannelSplitter(2);
    this.merger = audioContext.createChannelMerger(2);
    this.delayL = audioContext.createDelay(5.0);
    this.delayR = audioContext.createDelay(5.0);
    this.feedbackL = audioContext.createGain();
    this.feedbackR = audioContext.createGain();

    // State
    this.bpm = 120;
    this.syncEnabled = false;
    this.currentDivision = '1/4';
    this.pingPongEnabled = false;

    // Setup routing
    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path with feedback (mono mode)
    this.input.connect(this.delay);
    this.delay.connect(this.filter);
    this.filter.connect(this.feedback);

    // Feedback loop
    this.feedback.connect(this.delay);

    // Output
    this.feedback.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // Initially disconnect ping pong routing
    // It will be connected when pingPong is enabled
  }

  setupPingPongRouting() {
    // Disconnect mono routing
    this.input.disconnect();
    this.delay.disconnect();
    this.filter.disconnect();
    this.feedback.disconnect();
    this.wetGain.disconnect();

    // Split input into L/R
    this.input.connect(this.splitter);

    // Left channel
    this.splitter.connect(this.delayL, 0);
    this.delayL.connect(this.feedbackL);

    // Right channel
    this.splitter.connect(this.delayR, 1);
    this.delayR.connect(this.feedbackR);

    // Cross-feedback: L feeds R, R feeds L (ping pong effect)
    this.feedbackL.connect(this.delayR);
    this.feedbackR.connect(this.delayL);

    // Output to merger
    this.feedbackL.connect(this.merger, 0, 0);
    this.feedbackR.connect(this.merger, 0, 1);

    // Merger to wet gain
    this.merger.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // Keep dry path connected
    this.input.connect(this.dryGain);
  }

  restoreMonoRouting() {
    // Disconnect ping pong routing
    this.input.disconnect();
    this.splitter.disconnect();
    this.delayL.disconnect();
    this.delayR.disconnect();
    this.feedbackL.disconnect();
    this.feedbackR.disconnect();
    this.merger.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();

    // Restore mono routing
    this.setupRouting();
  }

  initialize(options) {
    // Set default values
    this.setDelayTime(options.delayTime || 250);
    this.setFeedback(options.feedback || 0);
    this.setMix(options.mix || 50);
    this.setFilter(options.filterType || 'off', options.filterFreq || 5000);

    if (options.sync !== undefined) {
      this.setSync(options.sync, options.division || '1/4');
    }

    if (options.pingPong !== undefined) {
      this.setPingPong(options.pingPong);
    }
  }

  /**
   * Set delay time in milliseconds
   * @param {number} ms - Delay time (0-5000ms)
   * @param {number} rampTime - Smooth transition time (default 0.05s)
   */
  setDelayTime(ms, rampTime = 0.05) {
    // Clamp to valid range
    ms = Math.max(0, Math.min(5000, ms));

    // Smooth delay time changes to prevent artifacts
    const now = this.context.currentTime;
    const delaySeconds = ms / 1000;

    if (this.pingPongEnabled) {
      // Set both L and R delays
      this.delayL.delayTime.cancelScheduledValues(now);
      this.delayL.delayTime.setValueAtTime(this.delayL.delayTime.value, now);
      this.delayL.delayTime.linearRampToValueAtTime(delaySeconds, now + rampTime);

      this.delayR.delayTime.cancelScheduledValues(now);
      this.delayR.delayTime.setValueAtTime(this.delayR.delayTime.value, now);
      this.delayR.delayTime.linearRampToValueAtTime(delaySeconds, now + rampTime);
    } else {
      // Mono delay
      this.delay.delayTime.cancelScheduledValues(now);
      this.delay.delayTime.setValueAtTime(this.delay.delayTime.value, now);
      this.delay.delayTime.linearRampToValueAtTime(delaySeconds, now + rampTime);
    }

    this.currentDelayTime = ms;
  }

  /**
   * Set feedback amount
   * @param {number} amount - Feedback (0-100%)
   */
  setFeedback(amount) {
    // Clamp to valid range
    amount = Math.max(0, Math.min(100, amount));

    // Convert to gain (0-1)
    // Use slight curve for more natural feel
    const gain = Math.pow(amount / 100, 0.8);

    if (this.pingPongEnabled) {
      // Apply to both channels
      this.feedbackL.gain.value = gain * 0.5; // Reduce slightly for ping pong
      this.feedbackR.gain.value = gain * 0.5;
    } else {
      this.feedback.gain.value = gain;
    }
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet mix (0-100%)
   */
  setMix(percent) {
    // Clamp to valid range
    percent = Math.max(0, Math.min(100, percent));

    // Use equal power crossfade for smooth mixing
    const wetAngle = (percent / 100) * (Math.PI / 2);
    const wet = Math.sin(wetAngle);
    const dry = Math.cos(wetAngle);

    this.wetGain.gain.value = wet;
    this.dryGain.gain.value = dry;
  }

  /**
   * Set filter in feedback path
   * @param {string} type - Filter type ('off', 'lowpass', 'highpass')
   * @param {number} frequency - Filter frequency (20-20000 Hz)
   */
  setFilter(type, frequency) {
    frequency = Math.max(20, Math.min(20000, frequency));

    if (type === 'off') {
      // Use allpass filter (passes everything)
      this.filter.type = 'allpass';
    } else {
      this.filter.type = type; // 'lowpass' or 'highpass'
      this.filter.frequency.value = frequency;
      this.filter.Q.value = 0.7071; // Butterworth response
    }
  }

  /**
   * Enable/disable tempo sync
   * @param {boolean} enabled - Sync enabled
   * @param {string} division - Musical division ('1/4', '1/8', etc.)
   */
  setSync(enabled, division = '1/4') {
    this.syncEnabled = enabled;
    this.currentDivision = division;

    if (enabled) {
      const ms = this.syncTimeToMS(division, this.bpm);
      this.setDelayTime(ms);
    }
  }

  /**
   * Set BPM for tempo sync
   * @param {number} bpm - Beats per minute
   */
  setBPM(bpm) {
    bpm = Math.max(20, Math.min(300, bpm));
    this.bpm = bpm;

    if (this.syncEnabled) {
      const ms = this.syncTimeToMS(this.currentDivision, this.bpm);
      this.setDelayTime(ms);
    }
  }

  /**
   * Enable/disable ping pong stereo mode
   * @param {boolean} enabled - Ping pong enabled
   */
  setPingPong(enabled) {
    if (enabled === this.pingPongEnabled) return;

    this.pingPongEnabled = enabled;

    if (enabled) {
      this.setupPingPongRouting();
      // Restore current delay time to both channels
      if (this.currentDelayTime) {
        this.setDelayTime(this.currentDelayTime);
      }
    } else {
      this.restoreMonoRouting();
      // Restore current delay time to mono delay
      if (this.currentDelayTime) {
        this.setDelayTime(this.currentDelayTime);
      }
    }
  }

  /**
   * Convert musical division to milliseconds
   * @param {string} division - Musical division
   * @param {number} bpm - Beats per minute
   * @returns {number} Time in milliseconds
   */
  syncTimeToMS(division, bpm) {
    const beatDuration = 60000 / bpm; // One quarter note in ms

    const divisionMap = {
      '4': 16,      // 4 bars
      '2': 8,       // 2 bars
      '1': 4,       // 1 bar (whole note)
      '1/2': 2,     // Half note
      '1/4': 1,     // Quarter note
      '1/8': 0.5,   // Eighth note
      '1/16': 0.25, // Sixteenth note
      '1/32': 0.125, // Thirty-second note
      '1/2T': 4/3,  // Half note triplet
      '1/4T': 2/3,  // Quarter note triplet
      '1/8T': 1/3,  // Eighth note triplet
      '1/16T': 1/6, // Sixteenth note triplet
      '1/2D': 3,    // Dotted half note
      '1/4D': 1.5,  // Dotted quarter note
      '1/8D': 0.75, // Dotted eighth note
      '1/16D': 0.375 // Dotted sixteenth note
    };

    return beatDuration * (divisionMap[division] || 1);
  }

  /**
   * Get current delay time in milliseconds
   */
  getDelayTime() {
    return this.currentDelayTime;
  }

  /**
   * Clean up and disconnect all nodes
   */
  dispose() {
    // Disconnect all nodes
    this.input.disconnect();
    this.output.disconnect();
    this.delay.disconnect();
    this.feedback.disconnect();
    this.filter.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();
    this.splitter.disconnect();
    this.merger.disconnect();
    this.delayL.disconnect();
    this.delayR.disconnect();
    this.feedbackL.disconnect();
    this.feedbackR.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SimpleDelay;
}
