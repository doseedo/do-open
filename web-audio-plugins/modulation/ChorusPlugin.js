/**
 * Chorus Effect Plugin (AudioWorklet Version)
 * Creates the illusion of multiple voices/instruments by layering slightly detuned delays
 *
 * This version uses AudioWorklet for high-performance processing in both real-time
 * and offline rendering contexts.
 *
 * @class ChorusPlugin
 * @author Agent 4 (Modulation Plugins)
 * @version 2.0.0
 */
class ChorusPlugin {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // AudioWorklet node (will be created asynchronously)
    this.workletNode = null;
    this.isWorkletReady = false;

    // Parameters
    this._params = {
      rate: 0.5,        // LFO speed in Hz (0.01 to 10)
      depth: 50,        // Modulation intensity (0-100%)
      delay: 20,        // Base delay time in ms (5-50)
      voices: 2,        // Number of chorus voices (1-8)
      mix: 50           // Dry/wet mix (0-100%)
    };

    // Initialize with options
    this.initialize(options);
  }

  /**
   * Initialize or update parameters
   * @param {Object} options - Parameter values
   */
  async initialize(options = {}) {
    // Update parameters from options
    if (options.rate !== undefined) this._params.rate = options.rate;
    if (options.depth !== undefined) this._params.depth = options.depth;
    if (options.delay !== undefined) this._params.delay = options.delay;
    if (options.voices !== undefined) this._params.voices = options.voices;
    if (options.mix !== undefined) this._params.mix = options.mix;

    // Load and create AudioWorklet if not already done
    if (!this.isWorkletReady) {
      await this.setupAudioWorklet();
    }

    // Send initial parameters to worklet
    if (this.workletNode) {
      this.updateWorkletParams();
    }
  }

  /**
   * Setup AudioWorklet processing
   */
  async setupAudioWorklet() {
    try {
      // Load the AudioWorklet module
      const workletPath = new URL('./worklets/chorus-processor.js', import.meta.url).href;
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'chorus-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2]
      });

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Initialize worklet with sample rate
      this.workletNode.port.postMessage({
        type: 'init',
        params: { sampleRate: this.context.sampleRate }
      });

      this.isWorkletReady = true;
    } catch (error) {
      console.error('Failed to setup AudioWorklet for Chorus:', error);
      // Fallback: direct connection (bypass)
      this.input.connect(this.output);
    }
  }

  /**
   * Update worklet parameters
   */
  updateWorkletParams() {
    if (!this.workletNode) return;

    this.workletNode.port.postMessage({
      type: 'setParams',
      params: {
        rate: this._params.rate,
        depth: this._params.depth / 100,  // Convert percentage to 0-1
        delay: this._params.delay / 1000, // Convert ms to seconds
        voices: this._params.voices,
        mix: this._params.mix / 100       // Convert percentage to 0-1
      }
    });
  }

  /**
   * Set LFO rate (speed of modulation)
   * @param {number} hz - Frequency in Hz (0.01 to 10)
   */
  setRate(hz) {
    this._params.rate = Math.max(0.01, Math.min(10, hz));
    this.updateWorkletParams();
  }

  /**
   * Set modulation depth
   * @param {number} percent - Depth percentage (0 to 100)
   */
  setDepth(percent) {
    this._params.depth = Math.max(0, Math.min(100, percent));
    this.updateWorkletParams();
  }

  /**
   * Set number of active voices
   * @param {number} num - Number of voices (1 to 8)
   */
  setVoices(num) {
    this._params.voices = Math.max(1, Math.min(8, Math.floor(num)));
    this.updateWorkletParams();
  }

  /**
   * Set base delay time
   * @param {number} ms - Delay time in milliseconds (5 to 50)
   */
  setDelayTime(ms) {
    this._params.delay = Math.max(5, Math.min(50, ms));
    this.updateWorkletParams();
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet percentage (0 to 100)
   */
  setMix(percent) {
    this._params.mix = Math.max(0, Math.min(100, percent));
    this.updateWorkletParams();
  }

  /**
   * Get current parameter values
   * @returns {Object} Current parameter values
   */
  getParams() {
    return { ...this._params };
  }

  /**
   * Set all parameters at once
   * @param {Object} params - Parameter object
   */
  setParams(params) {
    if (params.rate !== undefined) this.setRate(params.rate);
    if (params.depth !== undefined) this.setDepth(params.depth);
    if (params.voices !== undefined) this.setVoices(params.voices);
    if (params.delay !== undefined) this.setDelayTime(params.delay);
    if (params.mix !== undefined) this.setMix(params.mix);
  }

  /**
   * Connect output to destination
   * @param {AudioNode} destination - Destination node
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
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
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }
    this.input.disconnect();
    this.output.disconnect();
    this.isWorkletReady = false;
  }

  /**
   * Check if using AudioWorklet
   * @returns {boolean} True if using AudioWorklet
   */
  usesAudioWorklet() {
    return this.isWorkletReady;
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ChorusPlugin;
}

export default ChorusPlugin;
