/**
 * Tremolo Effect Plugin (AudioWorklet Version)
 * Creates amplitude modulation effect
 *
 * This version uses AudioWorklet for high-performance processing in both real-time
 * and offline rendering contexts.
 *
 * @class TremoloPlugin
 * @author Agent 4 (Modulation Plugins)
 * @version 2.0.0
 */
class TremoloPlugin {
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
      rate: 5.0,         // LFO speed in Hz (0.1 to 20)
      depth: 50,         // Modulation intensity (0-100%)
      waveform: 'sine'   // LFO waveform: 'sine', 'triangle', 'square', 'saw'
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
    if (options.waveform !== undefined) this._params.waveform = options.waveform;

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
      const workletPath = new URL('./worklets/tremolo-processor.js', import.meta.url).href;
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'tremolo-processor', {
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
      console.error('Failed to setup AudioWorklet for Tremolo:', error);
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
        waveform: this._params.waveform
      }
    });
  }

  /**
   * Set LFO rate (speed of modulation)
   * @param {number} hz - Frequency in Hz (0.1 to 20)
   */
  setRate(hz) {
    this._params.rate = Math.max(0.1, Math.min(20, hz));
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
   * Set LFO waveform
   * @param {string} waveform - Waveform type: 'sine', 'triangle', 'square', 'saw'
   */
  setWaveform(waveform) {
    const validWaveforms = ['sine', 'triangle', 'square', 'saw'];
    if (validWaveforms.includes(waveform)) {
      this._params.waveform = waveform;
      this.updateWorkletParams();
    }
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
    if (params.waveform !== undefined) this.setWaveform(params.waveform);
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
  module.exports = TremoloPlugin;
}

export default TremoloPlugin;
