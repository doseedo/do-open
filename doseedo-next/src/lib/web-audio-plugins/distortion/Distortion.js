/**
 * Distortion Plugin (AudioWorklet Version)
 * Hard clipping distortion with aggressive harmonic generation
 *
 * Features:
 * - Multiple clipping algorithms (hard, soft, asymmetric, foldback)
 * - Pre/post filtering
 * - Tone control
 * - High-performance AudioWorklet processing
 *
 * @class DistortionPlugin
 * @version 2.0.0 (AudioWorklet)
 */
class DistortionPlugin {
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
      drive: 50,              // 0-100% (drive amount)
      tone: 1000,             // 20-20000 Hz (tone frequency)
      toneWidth: 1.0,         // 0.1-10 (Q factor)
      filterPosition: 'post', // 'pre' or 'post'
      clipType: 'hard',       // 'hard', 'soft', 'asymmetric', 'foldback'
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
    if (options.drive !== undefined) this._params.drive = options.drive;
    if (options.tone !== undefined) this._params.tone = options.tone;
    if (options.toneWidth !== undefined) this._params.toneWidth = options.toneWidth;
    if (options.filterPosition !== undefined) this._params.filterPosition = options.filterPosition;
    if (options.clipType !== undefined) this._params.clipType = options.clipType;
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
      const workletPath = new URL('./worklets/distortion-processor.js', import.meta.url).href;
      await this.context.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.context, 'distortion-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2]
      });

      // Setup routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      this.isWorkletReady = true;
    } catch (error) {
      console.error('Failed to setup AudioWorklet for Distortion:', error);
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
        drive: this._params.drive,
        tone: this._params.tone,
        toneWidth: this._params.toneWidth,
        filterPosition: this._params.filterPosition,
        clipType: this._params.clipType,
        output: this._params.output,
        mix: this._params.mix / 100  // Convert percentage to 0-1
      }
    });
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
   * Set tone frequency
   * @param {number} hz - Frequency in Hz (20-20000)
   */
  setTone(hz) {
    this._params.tone = Math.max(20, Math.min(20000, hz));
    this.updateWorkletParams();
  }

  /**
   * Set tone width (Q factor)
   * @param {number} value - Q factor (0.1-10)
   */
  setToneWidth(value) {
    this._params.toneWidth = Math.max(0.1, Math.min(10, value));
    this.updateWorkletParams();
  }

  /**
   * Set filter position
   * @param {string} position - 'pre' or 'post'
   */
  setFilterPosition(position) {
    if (position === 'pre' || position === 'post') {
      this._params.filterPosition = position;
      this.updateWorkletParams();
    }
  }

  /**
   * Set clipping type
   * @param {string} type - 'hard', 'soft', 'asymmetric', or 'foldback'
   */
  setClipType(type) {
    const validTypes = ['hard', 'soft', 'asymmetric', 'foldback'];
    if (validTypes.includes(type)) {
      this._params.clipType = type;
      this.updateWorkletParams();
    }
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

export { DistortionPlugin };
