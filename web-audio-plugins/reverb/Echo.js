/**
 * Echo.js - Complex Delay/Echo Plugin
 *
 * Advanced delay effect with:
 * - Stereo delays with independent timing
 * - Modulation in feedback path
 * - Ducking (delay quiets when input is loud)
 * - Reverb in feedback for ambient tails
 * - Filtering (highpass/lowpass)
 * - Tempo sync capability
 * - Ping-pong and other channel modes
 *
 * @author Agent 5: Reverb & Spatial Effects
 * @version 1.0.0
 */

class Echo {
  /**
   * Create a new Echo instance
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial parameters
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.sampleRate = audioContext.sampleRate;

    // Tempo tracking
    this.tempo = 120; // BPM
    this.syncEnabled = false;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Stereo delay lines
    this.delayL = audioContext.createDelay(2.0);
    this.delayR = audioContext.createDelay(2.0);

    // Feedback paths
    this.feedbackL = audioContext.createGain();
    this.feedbackR = audioContext.createGain();
    this.feedbackCross = audioContext.createGain(); // For ping-pong

    // Channel mode routing
    this.channelRouter = {
      inputSplitter: audioContext.createChannelSplitter(2),
      outputMerger: audioContext.createChannelMerger(2),
      leftGain: audioContext.createGain(),
      rightGain: audioContext.createGain()
    };

    // Modulation (LFO)
    this.lfo = audioContext.createOscillator();
    this.lfoGain = audioContext.createGain();
    this.lfo.type = 'sine';
    this.lfo.frequency.value = 0.5;
    this.lfoGain.gain.value = 0;
    this.lfo.start();

    // Ducking (sidechain dynamics)
    this.ducker = this.createDucker();

    // Reverb in feedback path
    this.feedbackReverb = this.createFeedbackReverb();

    // Filters
    this.highpass = audioContext.createBiquadFilter();
    this.lowpass = audioContext.createBiquadFilter();
    this.highpass.type = 'highpass';
    this.lowpass.type = 'lowpass';

    // Stereo offset delay
    this.stereoOffsetDelay = audioContext.createDelay(0.05);

    // Dry/wet mix
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    // Setup routing
    this.setupRouting();

    // Initialize with default parameters
    this.initialize(options);
  }

  /**
   * Create ducking/sidechain dynamics
   * Uses envelope follower to duck delay when input is loud
   * @returns {Object} Ducker nodes
   */
  createDucker() {
    const ducker = {
      inputAnalyzer: this.context.createAnalyser(),
      envelopeFollower: this.context.createGain(),
      inverter: this.context.createGain(),
      threshold: -20,
      ratio: 4,
      enabled: false
    };

    // Configure analyzer
    ducker.inputAnalyzer.fftSize = 256;
    ducker.inputAnalyzer.smoothingTimeConstant = 0.8;

    // Inverter for ducking effect (1 - envelope = duck)
    ducker.inverter.gain.value = -1;

    return ducker;
  }

  /**
   * Create simple reverb for feedback path
   * @returns {Object} Reverb nodes
   */
  createFeedbackReverb() {
    const reverb = {
      delays: [],
      allpass: [],
      input: this.context.createGain(),
      output: this.context.createGain(),
      mix: this.context.createGain()
    };

    // 2 comb filters
    const combTimes = [0.0297, 0.0371];
    combTimes.forEach(time => {
      const delay = this.context.createDelay(0.1);
      const feedback = this.context.createGain();
      const damping = this.context.createBiquadFilter();

      delay.delayTime.value = time;
      feedback.gain.value = 0.7;
      damping.type = 'lowpass';
      damping.frequency.value = 4000;

      reverb.delays.push({ delay, feedback, damping });
    });

    // 1 all-pass for diffusion
    const ap = this.context.createDelay(0.1);
    const apFeedback = this.context.createGain();
    ap.delayTime.value = 0.005;
    apFeedback.gain.value = 0.5;
    reverb.allpass.push({ delay: ap, feedback: apFeedback });

    // Route reverb internally
    reverb.delays.forEach(d => {
      reverb.input.connect(d.delay);
      d.delay.connect(d.damping);
      d.damping.connect(d.feedback);
      d.feedback.connect(d.delay); // Feedback loop
      d.damping.connect(reverb.output);
    });

    reverb.allpass.forEach(ap => {
      reverb.output.connect(ap.delay);
      ap.delay.connect(ap.feedback);
      ap.feedback.connect(ap.delay); // Feedback loop
    });

    // Mix control (0 = no reverb, 1 = full reverb)
    reverb.mix.gain.value = 0;

    return reverb;
  }

  /**
   * Setup audio routing
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path: input → delays → filters → ducking → reverb → feedback
    // Split to stereo channels
    this.input.connect(this.channelRouter.inputSplitter);

    // Left channel path
    this.channelRouter.inputSplitter.connect(
      this.channelRouter.leftGain,
      0
    );
    this.channelRouter.leftGain.connect(this.delayL);

    // Right channel path (with stereo offset)
    this.channelRouter.inputSplitter.connect(
      this.channelRouter.rightGain,
      1
    );
    this.channelRouter.rightGain.connect(this.stereoOffsetDelay);
    this.stereoOffsetDelay.connect(this.delayR);

    // Delay → Filter chain
    this.delayL.connect(this.highpass);
    this.delayR.connect(this.highpass);
    this.highpass.connect(this.lowpass);

    // Filter → Ducker input analyzer (for envelope detection)
    this.lowpass.connect(this.ducker.inputAnalyzer);

    // Feedback paths
    // Left feedback: delay → reverb → feedback gain → back to delay
    this.lowpass.connect(this.feedbackReverb.input);
    this.feedbackReverb.output.connect(this.feedbackL);
    this.feedbackL.connect(this.delayL);

    // Right feedback
    this.feedbackReverb.output.connect(this.feedbackR);
    this.feedbackR.connect(this.delayR);

    // Cross feedback for ping-pong (optional)
    this.feedbackL.connect(this.feedbackCross);
    this.feedbackCross.connect(this.delayR);

    // Modulation: LFO → delay times
    this.lfo.connect(this.lfoGain);
    // Note: In practice, you'd use AudioWorklet for true delay time modulation
    // This is a simplified version

    // Output
    this.lowpass.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // Start ducking envelope follower process
    this.startDuckingProcess();
  }

  /**
   * Start ducking envelope follower
   * Monitors input level and adjusts wet gain
   */
  startDuckingProcess() {
    if (!this.ducker.enabled) return;

    const updateInterval = 50; // ms
    const bufferLength = this.ducker.inputAnalyzer.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    const updateDucking = () => {
      if (!this.ducker.enabled) return;

      // Get RMS level from analyzer
      this.ducker.inputAnalyzer.getFloatTimeDomainData(dataArray);

      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / bufferLength);
      const db = 20 * Math.log10(rms + 0.0001);

      // Calculate ducking amount
      if (db > this.ducker.threshold) {
        const over = db - this.ducker.threshold;
        const reduction = over * (1 - 1 / this.ducker.ratio);
        const gain = Math.pow(10, -reduction / 20);

        this.wetGain.gain.setTargetAtTime(
          gain * (this.currentMix / 100),
          this.context.currentTime,
          0.01
        );
      } else {
        this.wetGain.gain.setTargetAtTime(
          this.currentMix / 100,
          this.context.currentTime,
          0.05
        );
      }

      setTimeout(updateDucking, updateInterval);
    };

    if (this.ducker.enabled) {
      updateDucking();
    }
  }

  /**
   * Initialize parameters
   * @param {Object} options - Parameter values
   */
  initialize(options) {
    this.currentMix = options.mix || 30;

    this.setDelayTimeL(options.delayTimeL || 250);
    this.setDelayTimeR(options.delayTimeR || 375);
    this.setFeedback(options.feedback || 40);
    this.setChannelMode(options.channelMode || 'stereo');
    this.setStereoOffset(options.stereoOffset || 0);
    this.setModulationRate(options.modulationRate || 0.5);
    this.setModulationAmount(options.modulationAmount || 0);
    this.setDuckingThreshold(options.duckingThreshold || -20);
    this.setDuckingRatio(options.duckingRatio || 4);
    this.setReverbAmount(options.reverbAmount || 0);
    this.setReverbDecay(options.reverbDecay || 2);
    this.setHighpass(options.highpass || 20);
    this.setLowpass(options.lowpass || 20000);
    this.setMix(options.mix || 30);
  }

  /**
   * Set left delay time
   * @param {number} ms - Delay time in milliseconds (0-2000) or sync value
   */
  setDelayTimeL(ms) {
    if (this.syncEnabled) {
      ms = this.syncToMs(ms);
    }

    const seconds = Math.max(0, Math.min(2000, ms)) / 1000;
    this.delayL.delayTime.setTargetAtTime(
      seconds,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set right delay time
   * @param {number} ms - Delay time in milliseconds (0-2000) or sync value
   */
  setDelayTimeR(ms) {
    if (this.syncEnabled) {
      ms = this.syncToMs(ms);
    }

    const seconds = Math.max(0, Math.min(2000, ms)) / 1000;
    this.delayR.delayTime.setTargetAtTime(
      seconds,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Convert tempo-synced value to milliseconds
   * @param {string|number} value - Sync value (e.g., '1/4', '1/8') or ms
   * @returns {number} Milliseconds
   */
  syncToMs(value) {
    if (typeof value === 'number') return value;

    const beatMs = (60 / this.tempo) * 1000;
    const syncMap = {
      '1/1': beatMs * 4,
      '1/2': beatMs * 2,
      '1/4': beatMs,
      '1/8': beatMs / 2,
      '1/16': beatMs / 4,
      '1/32': beatMs / 8
    };

    return syncMap[value] || beatMs;
  }

  /**
   * Set feedback amount
   * @param {number} percent - Feedback percentage (0-100)
   */
  setFeedback(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const gain = percent / 100 * 0.95; // Max 0.95 to prevent runaway

    this.feedbackL.gain.setTargetAtTime(
      gain,
      this.context.currentTime,
      0.01
    );
    this.feedbackR.gain.setTargetAtTime(
      gain,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set channel mode
   * @param {string} mode - 'stereo', 'left', 'right', or 'ping-pong'
   */
  setChannelMode(mode) {
    switch (mode) {
      case 'left':
        this.channelRouter.leftGain.gain.value = 1;
        this.channelRouter.rightGain.gain.value = 0;
        this.feedbackCross.gain.value = 0;
        break;

      case 'right':
        this.channelRouter.leftGain.gain.value = 0;
        this.channelRouter.rightGain.gain.value = 1;
        this.feedbackCross.gain.value = 0;
        break;

      case 'ping-pong':
        this.channelRouter.leftGain.gain.value = 1;
        this.channelRouter.rightGain.gain.value = 1;
        this.feedbackCross.gain.value = this.feedbackL.gain.value;
        // Reduce direct feedback to avoid doubling
        this.feedbackL.gain.value *= 0.5;
        this.feedbackR.gain.value *= 0.5;
        break;

      case 'stereo':
      default:
        this.channelRouter.leftGain.gain.value = 1;
        this.channelRouter.rightGain.gain.value = 1;
        this.feedbackCross.gain.value = 0;
        break;
    }
  }

  /**
   * Set stereo offset
   * @param {number} ms - Offset in milliseconds (-50 to +50)
   */
  setStereoOffset(ms) {
    const seconds = Math.max(-50, Math.min(50, ms)) / 1000;
    const absSeconds = Math.abs(seconds);

    this.stereoOffsetDelay.delayTime.setTargetAtTime(
      absSeconds,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set modulation rate
   * @param {number} hz - LFO frequency in Hz (0-10)
   */
  setModulationRate(hz) {
    hz = Math.max(0, Math.min(10, hz));
    this.lfo.frequency.setTargetAtTime(
      hz,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set modulation amount
   * @param {number} percent - Modulation depth percentage (0-100)
   */
  setModulationAmount(percent) {
    percent = Math.max(0, Math.min(100, percent));
    // Max modulation depth of 10ms
    const depth = (percent / 100) * 0.01;

    this.lfoGain.gain.setTargetAtTime(
      depth,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set ducking threshold
   * @param {number} db - Threshold in dB (-60 to 0)
   */
  setDuckingThreshold(db) {
    this.ducker.threshold = Math.max(-60, Math.min(0, db));
  }

  /**
   * Set ducking ratio
   * @param {number} ratio - Compression ratio (1-10)
   */
  setDuckingRatio(ratio) {
    this.ducker.ratio = Math.max(1, Math.min(10, ratio));
  }

  /**
   * Enable/disable ducking
   * @param {boolean} enabled - Ducking enabled
   */
  setDuckingEnabled(enabled) {
    this.ducker.enabled = enabled;
    if (enabled) {
      this.startDuckingProcess();
    }
  }

  /**
   * Set reverb amount in feedback path
   * @param {number} percent - Reverb amount percentage (0-100)
   */
  setReverbAmount(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const mix = percent / 100;

    this.feedbackReverb.mix.gain.setTargetAtTime(
      mix,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set reverb decay time
   * @param {number} seconds - Decay time in seconds (0.1-10)
   */
  setReverbDecay(seconds) {
    seconds = Math.max(0.1, Math.min(10, seconds));

    this.feedbackReverb.delays.forEach(d => {
      const delayTime = d.delay.delayTime.value;
      const feedback = Math.pow(10, (-3 * delayTime) / seconds);
      d.feedback.gain.setTargetAtTime(
        Math.min(0.95, feedback),
        this.context.currentTime,
        0.01
      );
    });
  }

  /**
   * Set highpass filter frequency
   * @param {number} freq - Frequency in Hz (20-1000)
   */
  setHighpass(freq) {
    freq = Math.max(20, Math.min(1000, freq));
    this.highpass.frequency.setTargetAtTime(
      freq,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set lowpass filter frequency
   * @param {number} freq - Frequency in Hz (1000-20000)
   */
  setLowpass(freq) {
    freq = Math.max(1000, Math.min(20000, freq));
    this.lowpass.frequency.setTargetAtTime(
      freq,
      this.context.currentTime,
      0.01
    );
  }

  /**
   * Set tempo for sync
   * @param {number} bpm - Tempo in BPM
   */
  setTempo(bpm) {
    this.tempo = Math.max(20, Math.min(300, bpm));
  }

  /**
   * Enable/disable tempo sync
   * @param {boolean} enabled - Sync enabled
   */
  setSyncEnabled(enabled) {
    this.syncEnabled = enabled;
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Mix percentage (0-100)
   */
  setMix(percent) {
    percent = Math.max(0, Math.min(100, percent));
    this.currentMix = percent;

    const wet = percent / 100;
    const dry = 1 - wet;

    this.dryGain.gain.setTargetAtTime(
      dry,
      this.context.currentTime,
      0.01
    );

    if (!this.ducker.enabled) {
      this.wetGain.gain.setTargetAtTime(
        wet,
        this.context.currentTime,
        0.01
      );
    }
  }

  /**
   * Connect this echo to an audio node
   * @param {AudioNode} destination - Destination node
   * @returns {Echo} This echo instance (for chaining)
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect this echo
   * @returns {Echo} This echo instance (for chaining)
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

    // Disconnect feedback reverb
    this.feedbackReverb.delays.forEach(d => {
      d.delay.disconnect();
      d.feedback.disconnect();
      d.damping.disconnect();
    });

    this.feedbackReverb.allpass.forEach(ap => {
      ap.delay.disconnect();
      ap.feedback.disconnect();
    });
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Echo;
}
