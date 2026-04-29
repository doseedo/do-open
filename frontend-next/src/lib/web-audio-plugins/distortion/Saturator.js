/**
 * Saturator Plugin (AudioWorklet Version)
 * Multi-mode saturation for adding harmonic richness, warmth, and character
 *
 * Features:
 * - Multiple saturation algorithms (warm, digital, analog, clip, foldback, sineFold)
 * - Adjustable drive and color
 * - Harmonic depth control
 * - DC filtering
 * - High-performance AudioWorklet processing
 *
 * @class SaturatorPlugin
 * @version 2.0.0 (AudioWorklet)
 */
class SaturatorPlugin {
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
      curve: 'warm',          // 'warm', 'digital', 'analog', 'clip', 'foldback', 'sineFold'
      drive: 30,              // 0-100% (drive amount)
      color: 50,              // 0-100% (harmonic color frequency emphasis)
      depth: 100,             // 0-100% (wet signal amount)
      dcFilter: true,         // Enable/disable DC blocking
      output: 0,              // -24 to +24 dB (output gain)
      mix: 100                // 0-100% (dry/wet mix)
    };

    // Initialize with options
    this.initialize(options);
  }

  /**
   * Initialize or update parameters
   */
  async initialize(options = {}) {
    // Update parameters from options
    if (options.curve !== undefined) this._params.curve = options.curve;
    if (options.drive !== undefined) this._params.drive = options.drive;
    if (options.color !== undefined) this._params.color = options.color;
    if (options.depth !== undefined) this._params.depth = options.depth;
    if (options.dcFilter !== undefined) this._params.dcFilter = options.dcFilter;
    if (options.output !== undefined) this._params.output = options.output;
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
      const workletPath = new URL('./worklets/saturation-processor.js', import.meta.url).href;
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'saturation-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2]
      });

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.isWorkletReady = true;
    } catch (error) {
      console.error('Failed to setup AudioWorklet for Saturator:', error);
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
        curve: this._params.curve,
        drive: this._params.drive,
        color: this._params.color,
        depth: this._params.depth,
        dcFilter: this._params.dcFilter,
        output: this._params.output,
        mix: this._params.mix
      }
    });
  }

  /**
   * Set saturation curve type
   * @param {string} type - 'warm', 'digital', 'analog', 'clip', 'foldback', or 'sineFold'
   */
  setCurve(type) {
    const validTypes = ['warm', 'digital', 'analog', 'clip', 'foldback', 'sineFold'];
    if (validTypes.includes(type)) {
      this._params.curve = type;
      this.updateWorkletParams();
    }
  }

  /**
   * Set drive amount
   * @param {number} value - Drive (0-100%)
   */
  setDrive(value) {
    this._params.drive = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Set harmonic color emphasis
   * @param {number} value - Color (0-100%)
   */
  setColor(value) {
    this._params.color = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Set depth (wet signal amount)
   * @param {number} value - Depth (0-100%)
   */
  setDepth(value) {
    this._params.depth = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Enable/disable DC blocking filter
   * @param {boolean} enabled - DC filter on/off
   */
  setDCFilter(enabled) {
    this._params.dcFilter = Boolean(enabled);
    this.updateWorkletParams();
  }

  /**
   * Set output gain
   * @param {number} db - Output gain in dB (-24 to +24)
   */
  setOutput(db) {
    this._params.output = Math.max(-24, Math.min(24, db));
    this.updateWorkletParams();
  }

  /**
   * Set dry/wet mix
   * @param {number} value - Mix percentage (0-100%)
   */
  setMix(value) {
    this._params.mix = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Get current parameter value
   */
  getParameter(name) {
    return this._params[name];
  }

  /**
   * Connect the plugin to a destination
   */
  connect(destination) {
    this.output.connect(destination);
    return this;
  }

  /**
   * Disconnect the plugin
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
}

export { SaturatorPlugin };
