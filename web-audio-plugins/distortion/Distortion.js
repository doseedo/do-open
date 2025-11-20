/**
 * Distortion Plugin
 * Hard clipping distortion with aggressive harmonic generation
 *
 * Features:
 * - Hard clipping with pre/post filtering
 * - Tone stack for shaping distortion character
 * - Multiple clipping algorithms
 * - High gain capability
 * - Pre/post filter positioning
 *
 * UPDATED: Now uses AudioWorklet for high-performance processing
 *
 * @class Distortion
 */
class Distortion {
  constructor(audioContext, options = {}) {
    this.context = audioContext;
    this.workletNode = null;
    this.isWorkletReady = false;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Parameters
    this.params = {
      drive: 50,              // 0-100%
      tone: 1000,             // 20 Hz to 20 kHz
      toneWidth: 1,           // 0.1 to 10 (Q factor)
      filterPosition: 'post', // 'pre' or 'post'
      clipType: 'hard',       // 'hard', 'soft', 'asymmetric', 'foldback', 'tanh'
      output: 0,              // -24 to +24 dB
      mix: 100                // 0-100%
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
    if (options.tone !== undefined) this.params.tone = options.tone;
    if (options.toneWidth !== undefined) this.params.toneWidth = options.toneWidth;
    if (options.filterPosition !== undefined) this.params.filterPosition = options.filterPosition;
    if (options.clipType !== undefined) this.params.clipType = options.clipType;
    if (options.output !== undefined) this.params.output = options.output;
    if (options.mix !== undefined) this.params.mix = options.mix;

    // Setup AudioWorklet
    try {
      // Get the worklet path relative to this file
      const workletPath = new URL('./worklets/distortion-processor.js', import.meta.url || window.location.href);

      await this.context.audioWorklet.addModule(workletPath);

      // Create the worklet node
      this.workletNode = new AudioWorkletNode(this.context, 'distortion-processor');

      // Connect audio routing
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Send initial parameters
      this.sendParamsToWorklet();

      this.isWorkletReady = true;

    } catch (error) {
      console.error('Failed to initialize AudioWorklet for Distortion:', error);
      console.warn('Distortion plugin will not function without AudioWorklet support');
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
   * Set tone center frequency (20 Hz to 20 kHz)
   */
  setTone(frequency) {
    this.params.tone = Math.max(20, Math.min(20000, frequency));
    this.sendParamsToWorklet();
  }

  /**
   * Set tone width (Q factor, 0.1 to 10)
   */
  setToneWidth(q) {
    this.params.toneWidth = Math.max(0.1, Math.min(10, q));
    this.sendParamsToWorklet();
  }

  /**
   * Set filter position ('pre' or 'post')
   */
  setFilterPosition(position) {
    if (position !== 'pre' && position !== 'post') {
      console.warn(`Invalid filter position: ${position}. Using 'post'.`);
      position = 'post';
    }
    this.params.filterPosition = position;
    this.sendParamsToWorklet();
  }

  /**
   * Set clipping type
   */
  setClipType(type) {
    const validTypes = ['hard', 'soft', 'asymmetric', 'foldback', 'tanh'];
    if (!validTypes.includes(type)) {
      console.warn(`Invalid clip type: ${type}. Using 'hard'.`);
      type = 'hard';
    }
    this.params.clipType = type;
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
    if (params.tone !== undefined) this.params.tone = Math.max(20, Math.min(20000, params.tone));
    if (params.toneWidth !== undefined) this.params.toneWidth = Math.max(0.1, Math.min(10, params.toneWidth));
    if (params.filterPosition !== undefined) this.params.filterPosition = params.filterPosition;
    if (params.clipType !== undefined) this.params.clipType = params.clipType;
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
export default Distortion;

// Also support CommonJS
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Distortion;
}
