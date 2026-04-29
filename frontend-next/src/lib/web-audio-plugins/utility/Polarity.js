/**
 * Polarity Plugin
 * Phase inversion per channel using AudioWorklet
 *
 * Features:
 * - Independent L/R phase inversion
 * - Zero-latency processing
 * - Useful for fixing phase issues
 * - Create special effects with phase manipulation
 *
 * @class PolarityPlugin
 */
class PolarityPlugin {
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
      invertLeft: false,
      invertRight: false
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
      const workletPath = '/web-audio-plugins/utility/worklets/polarity-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'polarity-processor');

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.workletLoaded = true;
    } catch (error) {
      console.error('Failed to load PolarityPlugin AudioWorklet:', error);
      // Fallback to simple passthrough
      this.input.connect(this.output);
    }
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.invertLeft !== undefined) {
      this.setInvertLeft(options.invertLeft);
    }
    if (options.invertRight !== undefined) {
      this.setInvertRight(options.invertRight);
    }
  }

  /**
   * Set left channel phase inversion
   * @param {boolean} invert - True to invert phase, false for normal
   */
  setInvertLeft(invert) {
    this.params.invertLeft = Boolean(invert);
    this.updateWorklet();
  }

  /**
   * Set right channel phase inversion
   * @param {boolean} invert - True to invert phase, false for normal
   */
  setInvertRight(invert) {
    this.params.invertRight = Boolean(invert);
    this.updateWorklet();
  }

  /**
   * Toggle left channel phase
   */
  toggleLeft() {
    this.setInvertLeft(!this.params.invertLeft);
  }

  /**
   * Toggle right channel phase
   */
  toggleRight() {
    this.setInvertRight(!this.params.invertRight);
  }

  /**
   * Set phase for specific channel
   * @param {string} channel - 'L' or 'R' or 'left' or 'right'
   * @param {boolean} invert - True to invert phase
   */
  setPhase(channel, invert) {
    const ch = channel.toLowerCase();
    if (ch === 'l' || ch === 'left') {
      this.setInvertLeft(invert);
    } else if (ch === 'r' || ch === 'right') {
      this.setInvertRight(invert);
    }
  }

  /**
   * Update worklet with current parameters
   */
  updateWorklet() {
    if (this.workletLoaded && this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'setParams',
        params: {
          invertLeft: this.params.invertLeft,
          invertRight: this.params.invertRight
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
   * Get left channel inversion state
   */
  getInvertLeft() {
    return this.params.invertLeft;
  }

  /**
   * Get right channel inversion state
   */
  getInvertRight() {
    return this.params.invertRight;
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.invertLeft !== undefined) {
      this.setInvertLeft(params.invertLeft);
    }
    if (params.invertRight !== undefined) {
      this.setInvertRight(params.invertRight);
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
  module.exports = PolarityPlugin;
}
