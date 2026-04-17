/**
 * Simple Delay - AudioWorklet Version
 * Basic echo effect with feedback and filtering
 *
 * Features:
 * - Tempo synchronization with BPM
 * - Feedback loop with damping filter
 * - Stereo or mono operation
 * - Smooth parameter changes (no clicks)
 * - High-performance AudioWorklet processing
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 2.0.0 (AudioWorklet)
 */

class SimpleDelay {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.isWorkletLoaded = false;
    this.workletNode = null;

    // Audio nodes (for fallback or routing)
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // State
    this.bpm = 120;
    this.syncEnabled = false;
    this.currentDivision = '1/4';
    this.currentDelayTime = 250; // milliseconds

    // Parameters
    this.params = {
      delayTime: 0.25,  // seconds
      feedback: 0,      // 0-1
      mix: 0.5,         // 0-1
      damping: 0.5      // 0-1
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
      this.setDelayTime(options.delayTime || 250);
      this.setFeedback(options.feedback || 0);
      this.setMix(options.mix || 50);
      this.setDamping(options.damping || 0.5);

      if (options.sync !== undefined) {
        this.setSync(options.sync, options.division || '1/4');
      }

      if (options.bpm !== undefined) {
        this.setBPM(options.bpm);
      }

    } catch (error) {
      console.error('SimpleDelay: Failed to initialize AudioWorklet:', error);
      // Could fall back to native Web Audio nodes here
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
      const workletPath = '/web-audio-plugins/delay/worklets/simple-delay-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(this.context, 'simple-delay-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2]
      });

      // Connect the routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.isWorkletLoaded = true;

    } catch (error) {
      console.error('SimpleDelay: Failed to load AudioWorklet module:', error);
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
   * Set delay time in milliseconds
   * @param {number} ms - Delay time (0-5000ms)
   * @param {number} rampTime - Smooth transition time (ignored in worklet, kept for API compatibility)
   */
  setDelayTime(ms, rampTime = 0.05) {
    // Clamp to valid range
    ms = Math.max(0, Math.min(5000, ms));
    this.currentDelayTime = ms;

    // Convert to seconds
    const seconds = ms / 1000;
    this.params.delayTime = seconds;

    // Update worklet
    this.updateWorkletParams({ delayTime: seconds });
  }

  /**
   * Set feedback amount
   * @param {number} amount - Feedback (0-100%)
   */
  setFeedback(amount) {
    // Clamp to valid range
    amount = Math.max(0, Math.min(100, amount));

    // Convert to gain (0-1) with slight curve
    const gain = Math.pow(amount / 100, 0.8);
    this.params.feedback = gain;

    // Update worklet
    this.updateWorkletParams({ feedback: gain });
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet mix (0-100%)
   */
  setMix(percent) {
    // Clamp to valid range
    percent = Math.max(0, Math.min(100, percent));

    // Convert to 0-1 range
    const mix = percent / 100;
    this.params.mix = mix;

    // Update worklet
    this.updateWorkletParams({ mix: mix });
  }

  /**
   * Set damping (feedback filter)
   * @param {number} amount - Damping amount (0-1)
   */
  setDamping(amount) {
    // Clamp to valid range
    amount = Math.max(0, Math.min(1, amount));
    this.params.damping = amount;

    // Update worklet
    this.updateWorkletParams({ damping: amount });
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
   * @returns {number} Current delay time in ms
   */
  getDelayTime() {
    return this.currentDelayTime;
  }

  /**
   * Connect this delay to an audio node
   * @param {AudioNode} destination - Destination node
   * @returns {SimpleDelay} This instance (for chaining)
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect this delay from all outputs
   * @returns {SimpleDelay} This instance (for chaining)
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
      inputBuffer.numberOfChannels,
      inputBuffer.length,
      inputBuffer.sampleRate
    );

    // Load worklet in offline context
    const workletPath = '/web-audio-plugins/delay/worklets/simple-delay-processor.js';
    await offlineContext.audioWorklet.addModule(workletPath);

    // Create source and worklet node
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'simple-delay-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [inputBuffer.numberOfChannels]
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
  module.exports = SimpleDelay;
}
