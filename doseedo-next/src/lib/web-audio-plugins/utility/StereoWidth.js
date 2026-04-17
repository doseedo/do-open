/**
 * Stereo Width Plugin
 * Mid/Side stereo width control using AudioWorklet
 *
 * Features:
 * - Width range: 0% (mono) to 200% (extra wide)
 * - Mid/Side processing for transparent adjustment
 * - Zero-latency processing
 * - Smooth parameter automation
 *
 * @class StereoWidthPlugin
 */
class StereoWidthPlugin {
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
      width: 1.0 // 0 = mono, 1 = normal, 2 = extra wide (0% to 200%)
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
      const workletPath = '/web-audio-plugins/utility/worklets/stereo-width-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node with 2 channels (stereo)
      this.workletNode = new AudioWorkletNode(this.context, 'stereo-width-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2] // Stereo output
      });

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.workletLoaded = true;
    } catch (error) {
      console.error('Failed to load StereoWidthPlugin AudioWorklet:', error);
      // Fallback to simple passthrough
      this.input.connect(this.output);
    }
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.width !== undefined) {
      this.setWidth(options.width);
    }
  }

  /**
   * Set stereo width (0 to 2)
   * 0 = mono, 1 = normal stereo, 2 = extra wide
   * Can also accept percentage (0-200%)
   */
  setWidth(value) {
    // If value is > 2, assume it's a percentage (0-200%)
    if (value > 2) {
      value = value / 100;
    }
    this.params.width = Math.max(0, Math.min(2, value));
    this.updateWorklet();
  }

  /**
   * Set width as percentage (0-200%)
   */
  setWidthPercent(percent) {
    this.setWidth(percent / 100);
  }

  /**
   * Set to mono
   */
  setMono() {
    this.setWidth(0);
  }

  /**
   * Set to normal stereo
   */
  setNormal() {
    this.setWidth(1);
  }

  /**
   * Update worklet with current parameters
   */
  updateWorklet() {
    if (this.workletLoaded && this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'setParams',
        params: {
          width: this.params.width
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
   * Get current width value (0-2)
   */
  getWidth() {
    return this.params.width;
  }

  /**
   * Get current width as percentage (0-200%)
   */
  getWidthPercent() {
    return this.params.width * 100;
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.width !== undefined) {
      this.setWidth(params.width);
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
  module.exports = StereoWidthPlugin;
}
