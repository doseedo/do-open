/**
 * Flanger Effect
 * Jet-plane whoosh effect created by short delay with feedback and modulation
 *
 * @class Flanger
 * @param {AudioContext} audioContext - Web Audio API context
 * @param {Object} options - Configuration options
 */
class Flanger {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // Core flanger components
    this.delay = this.context.createDelay(0.02); // Max 20ms for flanger
    this.feedback = this.context.createGain();
    this.lfo = this.context.createOscillator();
    this.lfoGain = this.context.createGain();

    // Dry/Wet mix
    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();

    // Manual offset (static delay control)
    this.manualGain = this.context.createGain();

    // Parameters
    this._rate = 0.5;        // LFO speed in Hz
    this._depth = 50;        // Modulation intensity (0-100%)
    this._feedback = 50;     // Feedback amount (0-100%, can be negative)
    this._delay = 3;         // Base delay time in ms
    this._manual = 50;       // Static delay offset (0-100%)
    this._sync = false;      // Tempo sync (not implemented yet)
    this._waveform = 'sine'; // LFO waveform
    this._mix = 50;          // Dry/wet mix (0-100%)

    // Setup audio routing
    this.setupRouting();

    // Initialize with default or provided options
    this.initialize(options);
  }

  /**
   * Setup audio routing for all nodes
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path with feedback
    // Input -> Delay
    this.input.connect(this.delay);

    // Feedback loop: Delay -> Feedback Gain -> Delay
    this.delay.connect(this.feedback);
    this.feedback.connect(this.delay);

    // Output: Delay -> Wet Gain -> Output
    this.delay.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // LFO modulation of delay time
    this.lfo.connect(this.lfoGain);
    this.lfoGain.connect(this.delay.delayTime);

    // Manual offset modulation
    this.manualGain.gain.value = 0;
    this.manualGain.connect(this.delay.delayTime);

    // Set initial LFO
    this.lfo.type = this._waveform;
    this.lfo.frequency.value = this._rate;
    this.lfo.start();
  }

  /**
   * Initialize or update parameters
   * @param {Object} options - Parameter values
   */
  initialize(options = {}) {
    if (options.rate !== undefined) this.setRate(options.rate);
    if (options.depth !== undefined) this.setDepth(options.depth);
    if (options.feedback !== undefined) this.setFeedback(options.feedback);
    if (options.delay !== undefined) this.setDelayTime(options.delay);
    if (options.manual !== undefined) this.setManual(options.manual);
    if (options.waveform !== undefined) this.setWaveform(options.waveform);
    if (options.mix !== undefined) this.setMix(options.mix);
    if (options.sync !== undefined) this.setSync(options.sync);
  }

  /**
   * Set LFO rate (speed of modulation)
   * @param {number} hz - Frequency in Hz (0.01 to 10)
   */
  setRate(hz) {
    this._rate = Math.max(0.01, Math.min(10, hz));
    this.lfo.frequency.setValueAtTime(this._rate, this.context.currentTime);
  }

  /**
   * Set modulation depth
   * @param {number} percent - Depth percentage (0 to 100)
   */
  setDepth(percent) {
    this._depth = Math.max(0, Math.min(100, percent));
    const depth = this._depth / 100;

    // Flanger typically uses shorter modulation depth (max 3ms)
    this.lfoGain.gain.setValueAtTime(depth * 0.003, this.context.currentTime);
  }

  /**
   * Set feedback amount (can be positive or negative for different character)
   * @param {number} percent - Feedback percentage (-100 to 100)
   */
  setFeedback(percent) {
    this._feedback = Math.max(-100, Math.min(100, percent));
    const feedback = this._feedback / 100;

    // Limit feedback to prevent runaway (max 0.95)
    this.feedback.gain.setValueAtTime(feedback * 0.95, this.context.currentTime);
  }

  /**
   * Set base delay time
   * @param {number} ms - Delay time in milliseconds (0.5 to 10)
   */
  setDelayTime(ms) {
    this._delay = Math.max(0.5, Math.min(10, ms));
    const delaySeconds = this._delay / 1000;

    this.delay.delayTime.setValueAtTime(delaySeconds, this.context.currentTime);
  }

  /**
   * Set manual offset (static delay offset)
   * @param {number} percent - Manual offset percentage (0 to 100)
   */
  setManual(percent) {
    this._manual = Math.max(0, Math.min(100, percent));
    const manual = this._manual / 100;

    // Manual adds up to 5ms of static offset
    this.manualGain.gain.setValueAtTime(manual * 0.005, this.context.currentTime);
  }

  /**
   * Set LFO waveform
   * @param {string} type - Waveform type ('sine', 'triangle', 'square', 'sawtooth')
   */
  setWaveform(type) {
    const validTypes = ['sine', 'triangle', 'square', 'sawtooth'];
    if (validTypes.includes(type)) {
      this._waveform = type;

      // Create new LFO with new waveform
      const oldLfo = this.lfo;
      this.lfo = this.context.createOscillator();
      this.lfo.type = this._waveform;
      this.lfo.frequency.value = this._rate;

      // Reconnect
      oldLfo.disconnect();
      this.lfo.connect(this.lfoGain);
      this.lfo.start();

      // Stop old LFO
      oldLfo.stop();
    }
  }

  /**
   * Set dry/wet mix
   * @param {number} percent - Wet percentage (0 to 100)
   */
  setMix(percent) {
    this._mix = Math.max(0, Math.min(100, percent));
    const wet = this._mix / 100;
    const dry = 1 - wet;

    // Use equal power crossfade
    this.dryGain.gain.setValueAtTime(Math.cos(wet * Math.PI / 2), this.context.currentTime);
    this.wetGain.gain.setValueAtTime(Math.sin(wet * Math.PI / 2), this.context.currentTime);
  }

  /**
   * Set tempo sync (placeholder for future BPM sync implementation)
   * @param {boolean} enabled - Enable tempo sync
   */
  setSync(enabled) {
    this._sync = enabled;
    // TODO: Implement tempo sync with BPM
    console.log('Tempo sync not yet implemented');
  }

  /**
   * Get current parameter values
   * @returns {Object} Current parameter values
   */
  getParams() {
    return {
      rate: this._rate,
      depth: this._depth,
      feedback: this._feedback,
      delay: this._delay,
      manual: this._manual,
      waveform: this._waveform,
      mix: this._mix,
      sync: this._sync
    };
  }

  /**
   * Connect to destination
   * @param {AudioNode} destination - Audio node to connect to
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect output
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.lfo.stop();
    this.lfo.disconnect();
    this.lfoGain.disconnect();
    this.delay.disconnect();
    this.feedback.disconnect();
    this.manualGain.disconnect();
    this.input.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
    this.output.disconnect();
  }
}

// Export for use in modules or Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Flanger;
}
