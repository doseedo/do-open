/**
 * Gain Plugin
 * Precision gain/volume control using AudioWorklet
 *
 * Features:
 * - Linear gain control (0 to infinity)
 * - dB gain control (-infinity to +35 dB)
 * - Zero-latency processing
 * - Smooth parameter automation support
 *
 * @class GainPlugin
 */
class GainPlugin {
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
      gain: 1.0,      // Linear gain (0 to infinity)
      gainDb: 0       // dB gain (-infinity to +35 dB)
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
      const workletPath = '/web-audio-plugins/utility/worklets/gain-processor.js';
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'gain-processor');

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.workletLoaded = true;
    } catch (error) {
      console.error('Failed to load GainPlugin AudioWorklet:', error);
      // Fallback to simple gain node
      this.input.connect(this.output);
    }
  }

  /**
   * Initialize with options
   */
  initialize(options) {
    if (options.gain !== undefined) {
      this.setGain(options.gain);
    } else if (options.gainDb !== undefined) {
      this.setGainDb(options.gainDb);
    }
  }

  /**
   * Convert dB to linear gain
   */
  dbToGain(db) {
    if (db === -Infinity || db < -100) return 0;
    return Math.pow(10, db / 20);
  }

  /**
   * Convert linear gain to dB
   */
  gainToDb(gain) {
    if (gain <= 0) return -Infinity;
    return 20 * Math.log10(gain);
  }

  /**
   * Set linear gain (0 to infinity)
   */
  setGain(gain) {
    this.params.gain = Math.max(0, gain);
    this.params.gainDb = this.gainToDb(this.params.gain);
    this.updateWorklet();
  }

  /**
   * Set dB gain (-infinity to +35 dB)
   */
  setGainDb(db) {
    this.params.gainDb = Math.max(-100, Math.min(35, db));
    this.params.gain = this.dbToGain(this.params.gainDb);
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
          gain: this.params.gain,
          gainDb: this.params.gainDb
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
   * Get linear gain
   */
  getGain() {
    return this.params.gain;
  }

  /**
   * Get dB gain
   */
  getGainDb() {
    return this.params.gainDb;
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.gain !== undefined) {
      this.setGain(params.gain);
    } else if (params.gainDb !== undefined) {
      this.setGainDb(params.gainDb);
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
  module.exports = GainPlugin;
}
