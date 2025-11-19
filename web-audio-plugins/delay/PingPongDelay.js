/**
 * Ping Pong Delay
 * Stereo delay that bounces between left and right channels
 *
 * Features:
 * - Independent L/R delay times
 * - Cross-feedback (L→R, R→L)
 * - Stereo spread control
 * - Tempo sync with musical divisions
 * - Filter in feedback path
 */

class PingPongDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Main I/O
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Separate L/R channels
    this.splitter = audioContext.createChannelSplitter(2);
    this.merger = audioContext.createChannelMerger(2);

    // L and R delay lines
    this.delayL = audioContext.createDelay(5.0);
    this.delayR = audioContext.createDelay(5.0);

    // Cross-feedback gains
    this.feedbackL = audioContext.createGain();
    this.feedbackR = audioContext.createGain();

    // Filters for each channel
    this.filterL = audioContext.createBiquadFilter();
    this.filterR = audioContext.createBiquadFilter();

    // Dry/wet mix
    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // Spread control (affects delay time offset)
    this.spreadAmount = 0;

    // State
    this.bpm = 120;
    this.syncEnabledL = false;
    this.syncEnabledR = false;
    this.divisionL = '1/4';
    this.divisionR = '1/8';
    this.baseDelayTimeL = 250;
    this.baseDelayTimeR = 125;

    // Setup routing
    this.setupPingPongRouting();
    this.initialize(options);
  }

  setupPingPongRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Split input into L/R
    this.input.connect(this.splitter);

    // Left channel: input → delay → filter → feedback
    this.splitter.connect(this.delayL, 0);
    this.delayL.connect(this.filterL);
    this.filterL.connect(this.feedbackL);

    // Right channel: input → delay → filter → feedback
    this.splitter.connect(this.delayR, 1);
    this.delayR.connect(this.filterR);
    this.filterR.connect(this.feedbackR);

    // Cross-feedback: L feeds R, R feeds L (ping pong effect)
    this.feedbackL.connect(this.delayR);
    this.feedbackR.connect(this.delayL);

    // Output to merger
    this.feedbackL.connect(this.merger, 0, 0);
    this.feedbackR.connect(this.merger, 0, 1);

    // Merger to wet gain to output
    this.merger.connect(this.wetGain);
    this.wetGain.connect(this.output);
  }

  initialize(options) {
    // Set default values
    this.setDelayTimeL(options.delayTimeL || 250);
    this.setDelayTimeR(options.delayTimeR || 125);
    this.setFeedback(options.feedback || 50);
    this.setSpread(options.spread || 50);
    this.setFilter(
      options.filterType || 'lowpass',
      options.filterFreq || 5000
    );
    this.setMix(options.mix || 50);

    if (options.syncL !== undefined) {
      this.setSyncL(options.syncL, options.divisionL || '1/4');
    }
    if (options.syncR !== undefined) {
      this.setSyncR(options.syncR, options.divisionR || '1/8');
    }
  }

  /**
   * Set left channel delay time
   * @param {number} ms - Delay time (0-5000ms)
   * @param {number} rampTime - Smooth transition time
   */
  setDelayTimeL(ms, rampTime = 0.05) {
    ms = Math.max(0, Math.min(5000, ms));
    this.baseDelayTimeL = ms;

    const now = this.context.currentTime;
    const delaySeconds = ms / 1000;

    this.delayL.delayTime.cancelScheduledValues(now);
    this.delayL.delayTime.setValueAtTime(this.delayL.delayTime.value, now);
    this.delayL.delayTime.linearRampToValueAtTime(delaySeconds, now + rampTime);
  }

  /**
   * Set right channel delay time
   * @param {number} ms - Delay time (0-5000ms)
   * @param {number} rampTime - Smooth transition time
   */
  setDelayTimeR(ms, rampTime = 0.05) {
    ms = Math.max(0, Math.min(5000, ms));
    this.baseDelayTimeR = ms;

    const now = this.context.currentTime;
    const delaySeconds = ms / 1000;

    this.delayR.delayTime.cancelScheduledValues(now);
    this.delayR.delayTime.setValueAtTime(this.delayR.delayTime.value, now);
    this.delayR.delayTime.linearRampToValueAtTime(delaySeconds, now + rampTime);
  }

  /**
   * Set feedback amount (applies to both channels)
   * @param {number} amount - Feedback (0-100%)
   */
  setFeedback(amount) {
    amount = Math.max(0, Math.min(100, amount));

    // Convert to gain with curve for natural feel
    // Reduce by 0.5 for cross-feedback to prevent runaway
    const gain = Math.pow(amount / 100, 0.8) * 0.5;

    this.feedbackL.gain.value = gain;
    this.feedbackR.gain.value = gain;
  }

  /**
   * Set stereo spread
   * @param {number} amount - Spread (0-100%)
   * 0 = both channels same, 100 = maximum separation
   */
  setSpread(amount) {
    amount = Math.max(0, Math.min(100, amount));
    this.spreadAmount = amount;

    // Spread affects the delay time offset between L and R
    // At 0%, times are equal; at 100%, R is offset by up to 50ms
    const spreadOffset = (amount / 100) * 50; // Max 50ms offset

    // Apply spread to right channel
    const rightTimeWithSpread = this.baseDelayTimeR + spreadOffset;
    this.setDelayTimeR(rightTimeWithSpread);
  }

  /**
   * Set filter for feedback path (applies to both channels)
   * @param {string} type - Filter type ('off', 'lowpass', 'highpass')
   * @param {number} frequency - Filter frequency (20-20000 Hz)
   */
  setFilter(type, frequency) {
    frequency = Math.max(20, Math.min(20000, frequency));

    if (type === 'off') {
      this.filterL.type = 'allpass';
      this.filterR.type = 'allpass';
    } else {
      this.filterL.type = type;
      this.filterR.type = type;
      this.filterL.frequency.value = frequency;
      this.filterR.frequency.value = frequency;
      this.filterL.Q.value = 0.7071; // Butterworth
      this.filterR.Q.value = 0.7071;
    }
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet mix (0-100%)
   */
  setMix(percent) {
    percent = Math.max(0, Math.min(100, percent));

    // Equal power crossfade
    const wetAngle = (percent / 100) * (Math.PI / 2);
    const wet = Math.sin(wetAngle);
    const dry = Math.cos(wetAngle);

    this.wetGain.gain.value = wet;
    this.dryGain.gain.value = dry;
  }

  /**
   * Enable/disable tempo sync for left channel
   * @param {boolean} enabled - Sync enabled
   * @param {string} division - Musical division
   */
  setSyncL(enabled, division = '1/4') {
    this.syncEnabledL = enabled;
    this.divisionL = division;

    if (enabled) {
      const ms = this.syncTimeToMS(division, this.bpm);
      this.setDelayTimeL(ms);
    }
  }

  /**
   * Enable/disable tempo sync for right channel
   * @param {boolean} enabled - Sync enabled
   * @param {string} division - Musical division
   */
  setSyncR(enabled, division = '1/8') {
    this.syncEnabledR = enabled;
    this.divisionR = division;

    if (enabled) {
      const ms = this.syncTimeToMS(division, this.bpm);
      this.setDelayTimeR(ms);
    }
  }

  /**
   * Set BPM for tempo sync
   * @param {number} bpm - Beats per minute
   */
  setBPM(bpm) {
    bpm = Math.max(20, Math.min(300, bpm));
    this.bpm = bpm;

    if (this.syncEnabledL) {
      const msL = this.syncTimeToMS(this.divisionL, this.bpm);
      this.setDelayTimeL(msL);
    }

    if (this.syncEnabledR) {
      const msR = this.syncTimeToMS(this.divisionR, this.bpm);
      this.setDelayTimeR(msR);
    }
  }

  /**
   * Convert musical division to milliseconds
   * @param {string} division - Musical division
   * @param {number} bpm - Beats per minute
   * @returns {number} Time in milliseconds
   */
  syncTimeToMS(division, bpm) {
    const beatDuration = 60000 / bpm;

    const divisionMap = {
      '4': 16,      // 4 bars
      '2': 8,       // 2 bars
      '1': 4,       // 1 bar
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
   * Get current delay times
   */
  getDelayTimes() {
    return {
      left: this.baseDelayTimeL,
      right: this.baseDelayTimeR
    };
  }

  /**
   * Clean up and disconnect all nodes
   */
  dispose() {
    this.input.disconnect();
    this.output.disconnect();
    this.splitter.disconnect();
    this.merger.disconnect();
    this.delayL.disconnect();
    this.delayR.disconnect();
    this.feedbackL.disconnect();
    this.feedbackR.disconnect();
    this.filterL.disconnect();
    this.filterR.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PingPongDelay;
}
