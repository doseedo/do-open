/**
 * Echo - AudioWorklet Version
 * Multi-tap delay/echo effect
 *
 * Features:
 * - Multiple delay taps with exponential decay
 * - Tempo synchronization capability
 * - Feedback control
 * - High-performance AudioWorklet processing
 * - Stereo processing
 *
 * @author Agent 3: Delay/Echo Plugins
 * @version 2.0.0 (AudioWorklet)
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
    this.isWorkletLoaded = false;
    this.workletNode = null;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Tempo tracking
    this.tempo = 120; // BPM
    this.syncEnabled = false;

    // Parameters
    this.params = {
      baseDelay: 0.25,  // seconds
      numTaps: 4,       // number of taps
      feedback: 0.3,    // 0-1
      tapDecay: 0.7,    // 0-1
      mix: 0.5          // 0-1
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
      this.setDelayTimeL(options.delayTimeL || 250);
      this.setFeedback(options.feedback || 40);
      this.setNumTaps(options.numTaps || 4);
      this.setTapDecay(options.tapDecay || 70);
      this.setMix(options.mix || 30);

      if (options.tempo !== undefined) {
        this.setTempo(options.tempo);
      }

      if (options.syncEnabled !== undefined) {
        this.setSyncEnabled(options.syncEnabled);
      }

    } catch (error) {
      console.error('Echo: Failed to initialize AudioWorklet:', error);
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
      const workletPath = '/web-audio-plugins/delay/worklets/echo-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(this.context, 'echo-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2]
      });

      // Connect the routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.isWorkletLoaded = true;

    } catch (error) {
      console.error('Echo: Failed to load AudioWorklet module:', error);
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
   * Set base delay time (alias for setDelayTimeL for compatibility)
   * @param {number} ms - Delay time in milliseconds (0-2000) or sync value
   */
  setDelayTimeL(ms) {
    if (this.syncEnabled) {
      ms = this.syncToMs(ms);
    }

    const seconds = Math.max(0, Math.min(5000, ms)) / 1000;
    this.params.baseDelay = seconds;

    this.updateWorkletParams({ baseDelay: seconds });
  }

  /**
   * Set number of delay taps
   * @param {number} taps - Number of taps (1-16)
   */
  setNumTaps(taps) {
    taps = Math.max(1, Math.min(16, Math.floor(taps)));
    this.params.numTaps = taps;

    this.updateWorkletParams({ numTaps: taps });
  }

  /**
   * Set feedback amount
   * @param {number} percent - Feedback percentage (0-100)
   */
  setFeedback(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const gain = percent / 100 * 0.95; // Max 0.95 to prevent runaway

    this.params.feedback = gain;
    this.updateWorkletParams({ feedback: gain });
  }

  /**
   * Set tap decay (how much each tap is quieter than the previous)
   * @param {number} percent - Decay percentage (0-100)
   */
  setTapDecay(percent) {
    percent = Math.max(0, Math.min(100, percent));
    const decay = percent / 100;

    this.params.tapDecay = decay;
    this.updateWorkletParams({ tapDecay: decay });
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
    const mix = percent / 100;

    this.params.mix = mix;
    this.updateWorkletParams({ mix: mix });
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
    const workletPath = '/web-audio-plugins/delay/worklets/echo-processor.js';
    await offlineContext.audioWorklet.addModule(workletPath);

    // Create source and worklet node
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'echo-processor', {
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

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Echo;
}
