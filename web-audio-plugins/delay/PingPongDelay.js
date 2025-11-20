/**
 * Ping Pong Delay - AudioWorklet Version
 * Stereo delay that bounces between left and right channels
 *
 * Features:
 * - Cross-feedback between L and R (ping-pong effect)
 * - Stereo spread control
 * - Tempo sync with musical divisions
 * - High-performance AudioWorklet processing
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 2.0.0 (AudioWorklet)
 */

class PingPongDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.isWorkletLoaded = false;
    this.workletNode = null;

    // Main I/O
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // State
    this.bpm = 120;
    this.syncEnabled = false;
    this.division = '1/4';

    // Parameters
    this.params = {
      delayTime: 0.375,   // seconds
      feedback: 0.4,      // 0-1
      spread: 1.0,        // 0-1
      mix: 0.5            // 0-1
    };

    // Initialize (async)
    this.initialize(options);
  }

  /**
   * Initialize the plugin (async to load AudioWorklet)
   * @param {Object} options - Initial parameters
   * @returns {Promise} Resolves when worklet is loaded
   */
  async initialize(options) {
    try {
      // Load AudioWorklet processor
      await this.loadWorklet();

      // Set initial parameters
      this.setDelayTime(options.delayTime || 375);
      this.setFeedback(options.feedback || 50);
      this.setSpread(options.spread || 100);
      this.setMix(options.mix || 50);

      if (options.bpm !== undefined) {
        this.setBPM(options.bpm);
      }

      if (options.sync !== undefined) {
        this.setSync(options.sync, options.division || '1/4');
      }

    } catch (error) {
      console.error('PingPongDelay: Failed to initialize AudioWorklet:', error);
    }
  }

  /**
   * Load the AudioWorklet processor module
   * @returns {Promise} Resolves when worklet is loaded
   */
  async loadWorklet() {
    if (this.isWorkletLoaded) return;

    try {
      // Load the processor script
      const workletPath = '/web-audio-plugins/delay/worklets/ping-pong-delay-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(this.context, 'ping-pong-delay-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2] // Stereo output
      });

      // Connect the routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.isWorkletLoaded = true;

    } catch (error) {
      console.error('PingPongDelay: Failed to load AudioWorklet module:', error);
      throw error;
    }
  }

  /**
   * Update a parameter in the AudioWorklet processor
   * @param {Object} params - Parameters to update
   */
  updateWorkletParams(params) {
    if (this.workletNode && this.workletNode.port) {
      this.workletNode.port.postMessage({
        type: 'updateParams',
        params: params
      });
    }
  }

  /**
   * Set delay time
   * @param {number} ms - Delay time (0-5000ms)
   */
  setDelayTime(ms) {
    ms = Math.max(0, Math.min(5000, ms));
    const seconds = ms / 1000;

    this.params.delayTime = seconds;
    this.updateWorkletParams({ delayTime: seconds });
  }

  /**
   * Set feedback amount (applies to ping-pong cross-feedback)
   * @param {number} amount - Feedback (0-100%)
   */
  setFeedback(amount) {
    amount = Math.max(0, Math.min(100, amount));

    // Convert to gain with curve for natural feel
    const gain = Math.pow(amount / 100, 0.8) * 0.5; // Reduced for ping-pong

    this.params.feedback = gain;
    this.updateWorkletParams({ feedback: gain });
  }

  /**
   * Set stereo spread
   * @param {number} amount - Spread (0-100%)
   * 0 = mono, 100 = full stereo width
   */
  setSpread(amount) {
    amount = Math.max(0, Math.min(100, amount));
    const spread = amount / 100;

    this.params.spread = spread;
    this.updateWorkletParams({ spread: spread });
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet mix (0-100%)
   */
  setMix(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const mix = percent / 100;

    this.params.mix = mix;
    this.updateWorkletParams({ mix: mix });
  }

  /**
   * Enable/disable tempo sync
   * @param {boolean} enabled - Sync enabled
   * @param {string} division - Musical division
   */
  setSync(enabled, division = '1/4') {
    this.syncEnabled = enabled;
    this.division = division;

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
      const ms = this.syncTimeToMS(this.division, this.bpm);
      this.setDelayTime(ms);
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
   * Get current delay time
   * @returns {Object} Left and right delay times in ms
   */
  getDelayTimes() {
    const ms = this.params.delayTime * 1000;
    return {
      left: ms,
      right: ms
    };
  }

  /**
   * Connect this delay to an audio node
   * @param {AudioNode} destination - Destination node
   * @returns {PingPongDelay} This instance (for chaining)
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect this delay from all outputs
   * @returns {PingPongDelay} This instance (for chaining)
   */
  disconnect() {
    this.output.disconnect();
    return this;
  }

  /**
   * Clean up and disconnect all nodes
   */
  dispose() {
    this.disconnect();
    this.input.disconnect();

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
  }

  /**
   * Check if using AudioWorklet
   * @returns {boolean} True if AudioWorklet is loaded
   */
  usesAudioWorklet() {
    return this.isWorkletLoaded;
  }

  /**
   * Process audio offline (for rendering)
   * @param {AudioBuffer} inputBuffer - Input audio buffer
   * @returns {Promise<AudioBuffer>} Processed audio buffer
   */
  async processOffline(inputBuffer) {
    // Create offline context with same sample rate
    const offlineContext = new OfflineAudioContext(
      Math.max(2, inputBuffer.numberOfChannels), // Ensure stereo for ping-pong
      inputBuffer.length,
      inputBuffer.sampleRate
    );

    // Load worklet in offline context
    const workletPath = '/web-audio-plugins/delay/worklets/ping-pong-delay-processor.js';
    await offlineContext.audioWorklet.addModule(workletPath);

    // Create source and worklet node
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'ping-pong-delay-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2]
    });

    // Apply current parameters
    workletNode.port.postMessage({
      type: 'updateParams',
      params: this.params
    });

    // Connect: source -> worklet -> destination
    source.connect(workletNode);
    workletNode.connect(offlineContext.destination);

    // Render
    source.start();
    const renderedBuffer = await offlineContext.startRendering();

    return renderedBuffer;
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PingPongDelay;
}
