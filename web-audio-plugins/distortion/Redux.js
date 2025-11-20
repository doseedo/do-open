/**
 * Redux Plugin
 * Bit crushing and sample rate reduction for lo-fi digital artifacts
 *
 * Features:
 * - Bit depth reduction (quantization)
 * - Sample rate reduction simulation
 * - Dithering to reduce harsh artifacts
 * - Jitter for analog-style instability
 * - Hardness parameter for quantization curve
 *
 * UPDATED: Now uses AudioWorklet for high-performance processing
 *
 * @class Redux
 */
class Redux {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.workletNode = null;
    this.isWorkletReady = false;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Parameters
    this.params = {
      bitDepth: 8,        // 1 to 16 bits
      sampleRate: 22050,  // 50 to 44100 Hz
      hardness: 50,       // 0 to 100% (quantization curve)
      dither: 0,          // 0 to 100%
      jitter: 0,          // 0 to 100%
      mix: 100            // 0-100%
    };

    // Initialize the plugin
    this.initialize(options);
  }

  /**
   * Initialize with options and setup AudioWorklet
   */
  async initialize(options) {
    // Update parameters from options
    if (options.bitDepth !== undefined) this.params.bitDepth = options.bitDepth;
    if (options.sampleRate !== undefined) this.params.sampleRate = options.sampleRate;
    if (options.hardness !== undefined) this.params.hardness = options.hardness;
    if (options.dither !== undefined) this.params.dither = options.dither;
    if (options.jitter !== undefined) this.params.jitter = options.jitter;
    if (options.mix !== undefined) this.params.mix = options.mix;

    // Setup AudioWorklet
    try {
      // Get the worklet path relative to this file
      const workletPath = new URL('./worklets/redux-processor.js', import.meta.url || window.location.href);

      await this.context.audioWorklet.addModule(workletPath);

      // Create the worklet node
      this.workletNode = new AudioWorkletNode(this.context, 'redux-processor');

      // Connect audio routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Send initial parameters
      this.sendParamsToWorklet();

      this.isWorkletReady = true;

    } catch (error) {
      console.error('Failed to initialize AudioWorklet for Redux:', error);
      console.warn('Redux plugin will not function without AudioWorklet support');
    }

    return this;
  }

  /**
   * Send current parameters to the worklet processor
   */
  sendParamsToWorklet() {
    if (!this.workletNode) return;

    this.workletNode.port.postMessage({
      type: 'setParams',
      params: { ...this.params }
    });
  }

  /**
   * Set bit depth (1 to 16 bits)
   */
  setBitDepth(bits) {
    this.params.bitDepth = Math.max(1, Math.min(16, Math.round(bits)));
    this.sendParamsToWorklet();
  }

  /**
   * Set sample rate (50 to 44100 Hz)
   */
  setSampleRate(rate) {
    this.params.sampleRate = Math.max(50, Math.min(this.context.sampleRate, rate));
    this.sendParamsToWorklet();
  }

  /**
   * Set hardness (0 to 100%)
   * Controls the quantization curve from soft to hard
   */
  setHardness(percent) {
    this.params.hardness = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Set dither amount (0 to 100%)
   * Adds noise to reduce quantization artifacts
   */
  setDither(percent) {
    this.params.dither = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Set jitter amount (0 to 100%)
   * Adds sample timing variation for analog-style instability
   */
  setJitter(percent) {
    this.params.jitter = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Set dry/wet mix (0-100%)
   */
  setMix(percent) {
    this.params.mix = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Get current parameters
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Set all parameters at once
   */
  setParams(params) {
    if (params.bitDepth !== undefined) this.params.bitDepth = Math.max(1, Math.min(16, Math.round(params.bitDepth)));
    if (params.sampleRate !== undefined) this.params.sampleRate = Math.max(50, Math.min(this.context.sampleRate, params.sampleRate));
    if (params.hardness !== undefined) this.params.hardness = Math.max(0, Math.min(100, params.hardness));
    if (params.dither !== undefined) this.params.dither = Math.max(0, Math.min(100, params.dither));
    if (params.jitter !== undefined) this.params.jitter = Math.max(0, Math.min(100, params.jitter));
    if (params.mix !== undefined) this.params.mix = Math.max(0, Math.min(100, params.mix));

    this.sendParamsToWorklet();
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
    this.disconnect();
    this.input.disconnect();

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode.port.close();
      this.workletNode = null;
    }

    this.isWorkletReady = false;
  }
}

// Export for use in modules
export default Redux;

// Also support CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Redux;
}
