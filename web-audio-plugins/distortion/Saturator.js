/**
 * Saturator Plugin
 * Multi-mode saturation from subtle warmth to heavy distortion
 *
 * Features:
 * - Multiple saturation algorithms (warm, digital, analog, clip, foldback, sine-fold)
 * - Pre/post filtering
 * - DC offset removal
 * - Color parameter for harmonic emphasis
 *
 * UPDATED: Now uses AudioWorklet for high-performance processing
 *
 * @class Saturator
 */
class Saturator {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.workletNode = null;
    this.isWorkletReady = false;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Parameters
    this.params = {
      drive: 0,           // 0-100%
      type: 'warm',       // 'warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'
      color: 0,           // 0-100%
      depth: 100,         // 0-100% (wet/dry character)
      dcFilter: true,     // boolean
      output: 0,          // -24 to +24 dB
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
    if (options.drive !== undefined) this.params.drive = options.drive;
    if (options.type !== undefined) this.params.type = options.type;
    if (options.color !== undefined) this.params.color = options.color;
    if (options.depth !== undefined) this.params.depth = options.depth;
    if (options.dcFilter !== undefined) this.params.dcFilter = options.dcFilter;
    if (options.output !== undefined) this.params.output = options.output;
    if (options.mix !== undefined) this.params.mix = options.mix;

    // Setup AudioWorklet
    try {
      // Get the worklet path relative to this file
      const workletPath = new URL('./worklets/saturation-processor.js', import.meta.url || window.location.href);

      await this.context.audioWorklet.addModule(workletPath);

      // Create the worklet node
      this.workletNode = new AudioWorkletNode(this.context, 'saturation-processor');

      // Connect audio routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Send initial parameters
      this.sendParamsToWorklet();

      this.isWorkletReady = true;

    } catch (error) {
      console.error('Failed to initialize AudioWorklet for Saturator:', error);
      console.warn('Saturator plugin will not function without AudioWorklet support');
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
   * Set drive amount (0-100%)
   */
  setDrive(percent) {
    this.params.drive = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Set saturation type
   */
  setType(type) {
    const validTypes = ['warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'];
    if (!validTypes.includes(type)) {
      console.warn(`Invalid saturation type: ${type}. Using 'warm'.`);
      type = 'warm';
    }
    this.params.type = type;
    this.sendParamsToWorklet();
  }

  /**
   * Set color (harmonic emphasis, 0-100%)
   */
  setColor(percent) {
    this.params.color = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Set depth (wet/dry character, 0-100%)
   */
  setDepth(percent) {
    this.params.depth = Math.max(0, Math.min(100, percent));
    this.sendParamsToWorklet();
  }

  /**
   * Enable/disable DC filter
   */
  setDCFilter(enabled) {
    this.params.dcFilter = enabled;
    this.sendParamsToWorklet();
  }

  /**
   * Set output gain (-24 to +24 dB)
   */
  setOutput(db) {
    this.params.output = Math.max(-24, Math.min(24, db));
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
    if (params.drive !== undefined) this.params.drive = Math.max(0, Math.min(100, params.drive));
    if (params.type !== undefined) {
      const validTypes = ['warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'];
      if (validTypes.includes(params.type)) {
        this.params.type = params.type;
      }
    }
    if (params.color !== undefined) this.params.color = Math.max(0, Math.min(100, params.color));
    if (params.depth !== undefined) this.params.depth = Math.max(0, Math.min(100, params.depth));
    if (params.dcFilter !== undefined) this.params.dcFilter = params.dcFilter;
    if (params.output !== undefined) this.params.output = Math.max(-24, Math.min(24, params.output));
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
export default Saturator;

// Also support CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Saturator;
}
