/**
 * Filter Delay
 * 3 parallel delay lines with individual filtering and panning
 *
 * Features:
 * - 3 independent delay taps
 * - Individual filtering per tap
 * - Pan position per tap
 * - Parallel signal flow
 * - Tempo sync per tap
 */

class FilterDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Main I/O
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Input gain control
    this.inputGain = audioContext.createGain();

    // Dry/wet mix
    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // 3 parallel delay taps
    this.taps = [];
    for (let i = 0; i < 3; i++) {
      this.taps.push(this.createTap(audioContext, i));
    }

    // State
    this.bpm = 120;

    // Setup routing
    this.setupParallelTaps();
    this.initialize(options);
  }

  createTap(context, index) {
    const tap = {
      index: index,
      delay: context.createDelay(5.0),
      feedback: context.createGain(),
      filter: context.createBiquadFilter(),
      pan: context.createStereoPanner ? context.createStereoPanner() : null,
      volume: context.createGain(),

      // Fallback panner for browsers without StereoPanner
      pannerL: context.createGain(),
      pannerR: context.createGain(),
      merger: context.createChannelMerger(2),

      // State
      syncEnabled: false,
      division: '1/4',
      baseDelayTime: 250 * (index + 1), // Stagger taps by default
      currentPan: 0
    };

    // Configure default filter
    tap.filter.type = 'lowpass';
    tap.filter.frequency.value = 5000;
    tap.filter.Q.value = 1.0;

    // Configure default volume
    tap.volume.gain.value = 0.7;

    return tap;
  }

  setupParallelTaps() {
    // Connect input to input gain
    this.input.connect(this.inputGain);

    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Each tap runs in parallel
    this.taps.forEach((tap, index) => {
      // Input → delay
      this.inputGain.connect(tap.delay);

      // Delay → filter → feedback
      tap.delay.connect(tap.filter);
      tap.filter.connect(tap.feedback);

      // Feedback loop
      tap.feedback.connect(tap.delay);

      // Output path: feedback → pan → volume → wet gain
      if (tap.pan) {
        // Use native StereoPanner if available
        tap.feedback.connect(tap.pan);
        tap.pan.connect(tap.volume);
      } else {
        // Fallback: manual panning with gains
        tap.feedback.connect(tap.pannerL);
        tap.feedback.connect(tap.pannerR);
        tap.pannerL.connect(tap.merger, 0, 0);
        tap.pannerR.connect(tap.merger, 0, 1);
        tap.merger.connect(tap.volume);
      }

      tap.volume.connect(this.wetGain);
    });

    // Wet to output
    this.wetGain.connect(this.output);
  }

  initialize(options) {
    // Set global defaults
    this.setInputGain(options.input !== undefined ? options.input : 0);
    this.setMix(options.mix || 50);

    // Initialize each tap with defaults or provided options
    for (let i = 0; i < 3; i++) {
      const tapOptions = options[`tap${i + 1}`] || {};
      this.setTap(i, {
        delayTime: tapOptions.delayTime || (250 * (i + 1)),
        feedback: tapOptions.feedback !== undefined ? tapOptions.feedback : 30,
        pan: tapOptions.pan !== undefined ? tapOptions.pan : (i - 1) * 50, // Spread: -50, 0, 50
        filterType: tapOptions.filterType || 'lowpass',
        filterFreq: tapOptions.filterFreq || 5000,
        filterQ: tapOptions.filterQ || 1.0,
        volume: tapOptions.volume !== undefined ? tapOptions.volume : 70,
        sync: tapOptions.sync || false,
        division: tapOptions.division || '1/4'
      });
    }
  }

  /**
   * Set parameters for a specific tap
   * @param {number} index - Tap index (0-2)
   * @param {object} params - Parameters to set
   */
  setTap(index, params) {
    if (index < 0 || index >= 3) {
      console.error('Invalid tap index:', index);
      return;
    }

    const tap = this.taps[index];

    // Delay time
    if (params.delayTime !== undefined) {
      this.setTapDelayTime(index, params.delayTime);
    }

    // Feedback
    if (params.feedback !== undefined) {
      const feedback = Math.max(0, Math.min(100, params.feedback));
      tap.feedback.gain.value = Math.pow(feedback / 100, 0.8);
    }

    // Pan
    if (params.pan !== undefined) {
      this.setTapPan(index, params.pan);
    }

    // Volume
    if (params.volume !== undefined) {
      const volume = Math.max(0, Math.min(100, params.volume));
      tap.volume.gain.value = volume / 100;
    }

    // Filter frequency
    if (params.filterFreq !== undefined) {
      const freq = Math.max(20, Math.min(20000, params.filterFreq));
      tap.filter.frequency.value = freq;
    }

    // Filter Q
    if (params.filterQ !== undefined) {
      const q = Math.max(0.1, Math.min(10, params.filterQ));
      tap.filter.Q.value = q;
    }

    // Filter type
    if (params.filterType !== undefined) {
      tap.filter.type = params.filterType; // lowpass, highpass, bandpass
    }

    // Sync
    if (params.sync !== undefined) {
      tap.syncEnabled = params.sync;
      if (params.sync && params.division) {
        tap.division = params.division;
        const ms = this.syncTimeToMS(params.division, this.bpm);
        this.setTapDelayTime(index, ms);
      }
    }
  }

  /**
   * Set delay time for a specific tap
   * @param {number} index - Tap index (0-2)
   * @param {number} ms - Delay time (0-5000ms)
   * @param {number} rampTime - Smooth transition time
   */
  setTapDelayTime(index, ms, rampTime = 0.05) {
    if (index < 0 || index >= 3) return;

    const tap = this.taps[index];
    ms = Math.max(0, Math.min(5000, ms));
    tap.baseDelayTime = ms;

    const now = this.context.currentTime;
    const delaySeconds = ms / 1000;

    tap.delay.delayTime.cancelScheduledValues(now);
    tap.delay.delayTime.setValueAtTime(tap.delay.delayTime.value, now);
    tap.delay.delayTime.linearRampToValueAtTime(delaySeconds, now + rampTime);
  }

  /**
   * Set pan position for a specific tap
   * @param {number} index - Tap index (0-2)
   * @param {number} pan - Pan position (-100 to +100)
   */
  setTapPan(index, pan) {
    if (index < 0 || index >= 3) return;

    const tap = this.taps[index];
    pan = Math.max(-100, Math.min(100, pan));
    tap.currentPan = pan;

    const panValue = pan / 100; // -1 to 1

    if (tap.pan) {
      // Use native StereoPanner
      tap.pan.pan.value = panValue;
    } else {
      // Manual panning with constant power law
      const panAngle = (panValue + 1) * (Math.PI / 4); // 0 to PI/2
      const leftGain = Math.cos(panAngle);
      const rightGain = Math.sin(panAngle);

      tap.pannerL.gain.value = leftGain;
      tap.pannerR.gain.value = rightGain;
    }
  }

  /**
   * Set input gain
   * @param {number} db - Input gain in dB (-Infinity to +6)
   */
  setInputGain(db) {
    db = Math.max(-60, Math.min(6, db));

    // Convert dB to linear gain
    const gain = db === -Infinity ? 0 : Math.pow(10, db / 20);
    this.inputGain.gain.value = gain;
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
   * Set BPM for tempo sync
   * @param {number} bpm - Beats per minute
   */
  setBPM(bpm) {
    bpm = Math.max(20, Math.min(300, bpm));
    this.bpm = bpm;

    // Update all synced taps
    this.taps.forEach((tap, index) => {
      if (tap.syncEnabled) {
        const ms = this.syncTimeToMS(tap.division, this.bpm);
        this.setTapDelayTime(index, ms);
      }
    });
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
      '4': 16,
      '2': 8,
      '1': 4,
      '1/2': 2,
      '1/4': 1,
      '1/8': 0.5,
      '1/16': 0.25,
      '1/32': 0.125,
      '1/2T': 4/3,
      '1/4T': 2/3,
      '1/8T': 1/3,
      '1/16T': 1/6,
      '1/2D': 3,
      '1/4D': 1.5,
      '1/8D': 0.75,
      '1/16D': 0.375
    };

    return beatDuration * (divisionMap[division] || 1);
  }

  /**
   * Get all tap parameters
   * @returns {Array} Array of tap parameter objects
   */
  getTaps() {
    return this.taps.map(tap => ({
      delayTime: tap.baseDelayTime,
      feedback: tap.feedback.gain.value * 100,
      pan: tap.currentPan,
      volume: tap.volume.gain.value * 100,
      filterFreq: tap.filter.frequency.value,
      filterQ: tap.filter.Q.value,
      filterType: tap.filter.type,
      syncEnabled: tap.syncEnabled,
      division: tap.division
    }));
  }

  /**
   * Clean up and disconnect all nodes
   */
  dispose() {
    this.input.disconnect();
    this.output.disconnect();
    this.inputGain.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();

    this.taps.forEach(tap => {
      tap.delay.disconnect();
      tap.feedback.disconnect();
      tap.filter.disconnect();
      tap.volume.disconnect();

      if (tap.pan) {
        tap.pan.disconnect();
      } else {
        tap.pannerL.disconnect();
        tap.pannerR.disconnect();
        tap.merger.disconnect();
      }
    });
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = FilterDelay;
}
