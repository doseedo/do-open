/**
 * Pan Plugin
 * Stereo panning with constant power curve using AudioWorklet
 *
 * Features:
 * - Constant power panning (maintains perceived loudness)
 * - Smooth parameter automation
 * - Zero-latency processing
 * - Pan range: -1 (full left) to +1 (full right)
 *
 * @class PanPlugin
 */
class PanPlugin {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // AudioWorklet node (will be initialized asynchronously)
    this.workletNode = null;
    this.workletLoaded = false;

    // Parameters
    this.params = {
      pan: 0 // -1 (left) to +1 (right), 0 = center
    };

    // Initialize
    this.initializeWorklet().then(() => {
      this.initialize(options);
    });
  }

  /**
   * Initialize AudioWorklet processor
   */
  async initializeWorklet() {
    try {
      // Load the AudioWorklet module
      const workletPath = '/web-audio-plugins/utility/worklets/pan-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node with 2 channels (stereo)
      this.workletNode = new AudioWorkletNode(this.context, 'pan-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2] // Stereo output
      });

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.workletLoaded = true;
    } catch (error) {
      console.error('Failed to load PanPlugin AudioWorklet:', error);
      // Fallback to simple passthrough
      this.input.connect(this.output);
    }
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.pan !== undefined) {
      this.setPan(options.pan);
    }
  }

  /**
   * Set pan position (-1 to +1)
   * -1 = full left, 0 = center, +1 = full right
   */
  setPan(value) {
    this.params.pan = Math.max(-1, Math.min(1, value));
    this.updateWorklet();
  }

  /**
   * Update worklet with current parameters
   */
  updateWorklet() {
    if (this.workletLoaded && this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'setParams',
        params: {
          pan: this.params.pan
        }
      });
    }
  }

  /**
   * Get current parameters
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Get current pan value
   */
  getPan() {
    return this.params.pan;
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.pan !== undefined) {
      this.setPan(params.pan);
    }
  }

  /**
   * Connect to destination
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Dispose and clean up resources
   */
  dispose() {
    if (this.workletNode) {
      this.workletNode.disconnect();
    }
    this.disconnect();
    this.input.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = PanPlugin;
}
