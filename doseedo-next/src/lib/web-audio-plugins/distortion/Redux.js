/**
 * Redux Plugin (AudioWorklet Version)
 * Bit crushing and sample rate reduction for lo-fi digital artifacts
 *
 * Features:
 * - Bit depth reduction (1-16 bits)
 * - Sample rate reduction (50Hz-48kHz)
 * - Hardness parameter for quantization curve
 * - Dithering to reduce harsh artifacts
 * - Jitter for analog-style instability
 * - High-performance AudioWorklet processing
 *
 * @class ReduxPlugin
 * @version 2.0.0 (AudioWorklet)
 */
class ReduxPlugin {
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
      bitDepth: 8,            // 1-16 bits
      sampleRate: 22050,      // 50-48000 Hz (target sample rate)
      hardness: 50,           // 0-100% (quantization curve)
      dither: 0,              // 0-100% (dithering amount)
      jitter: 0,              // 0-100% (timing variation)
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
    if (options.bitDepth !== undefined) this._params.bitDepth = options.bitDepth;
    if (options.sampleRate !== undefined) this._params.sampleRate = options.sampleRate;
    if (options.hardness !== undefined) this._params.hardness = options.hardness;
    if (options.dither !== undefined) this._params.dither = options.dither;
    if (options.jitter !== undefined) this._params.jitter = options.jitter;
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
      const workletPath = new URL('./worklets/redux-processor.js', import.meta.url).href;
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'redux-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2]
      });

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.isWorkletReady = true;
    } catch (error) {
      console.error('Failed to setup AudioWorklet for Redux:', error);
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
        bitDepth: this._params.bitDepth,
        sampleRate: this._params.sampleRate,
        hardness: this._params.hardness,
        dither: this._params.dither,
        jitter: this._params.jitter
      }
    });
  }

  /**
   * Set bit depth
   * @param {number} bits - Bit depth (1-16)
   */
  setBitDepth(bits) {
    this._params.bitDepth = Math.max(1, Math.min(16, Math.round(bits)));
    this.updateWorkletParams();
  }

  /**
   * Set sample rate reduction target
   * @param {number} hz - Target sample rate in Hz (50-48000)
   */
  setSampleRate(hz) {
    this._params.sampleRate = Math.max(50, Math.min(this.context.sampleRate, hz));
    this.updateWorkletParams();
  }

  /**
   * Set quantization hardness
   * @param {number} value - Hardness (0-100%)
   */
  setHardness(value) {
    this._params.hardness = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Set dithering amount
   * @param {number} value - Dither (0-100%)
   */
  setDither(value) {
    this._params.dither = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Set timing jitter amount
   * @param {number} value - Jitter (0-100%)
   */
  setJitter(value) {
    this._params.jitter = Math.max(0, Math.min(100, value));
    this.updateWorkletParams();
  }

  /**
   * Set dry/wet mix
   * @param {number} value - Mix percentage (0-100%)
   */
  setMix(value) {
    this._params.mix = Math.max(0, Math.min(100, value));
    // Note: Redux processor doesn't have mix parameter built in
    // You'd need to implement this with a dry/wet mixer in the main thread
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

export { ReduxPlugin };
