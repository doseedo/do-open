/**
 * Tremolo/Auto Pan Effect
 * Amplitude or pan modulation for rhythmic movement
 *
 * @class Tremolo
 * @param {AudioContext} audioContext - Web Audio API context
 * @param {Object} options - Configuration options
 */
class Tremolo {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // LFO for modulation
    this.lfo = this.context.createOscillator();
    this.lfoGain = this.context.createGain();

    // Offset for amplitude modulation (to keep signal positive)
    this.lfoOffset = this.context.createGain();

    // For tremolo mode - amplitude modulation
    this.amplitudeModulator = this.context.createGain();

    // For pan mode - stereo panning
    this.panner = this.context.createStereoPanner();

    // Stereo phase offset LFO (for stereo tremolo)
    this.lfoStereo = null;
    this.lfoStereoGain = null;
    this.lfoStereoOffset = null;
    this.amplitudeModulatorStereo = null;

    // Splitter and merger for stereo processing
    this.splitter = this.context.createChannelSplitter(2);
    this.merger = this.context.createChannelMerger(2);

    // Parameters
    this._rate = 5;            // LFO speed in Hz (or BPM if synced)
    this._depth = 50;          // Modulation intensity (0-100%)
    this._waveform = 'sine';   // LFO waveform
    this._phase = 0;           // Phase offset in degrees (0-360)
    this._mode = 'tremolo';    // 'tremolo' or 'pan'
    this._shape = 50;          // Waveform shaping/skew (0-100%)
    this._sync = false;        // Tempo sync
    this._stereo = false;      // Stereo phase offset

    // Setup audio routing
    this.setupRouting();

    // Initialize with default or provided options
    this.initialize(options);
  }

  /**
   * Setup audio routing for all nodes
   */
  setupRouting() {
    // Start with tremolo mode
    this.setupTremoloMode();

    // Set initial LFO
    this.lfo.type = this._waveform;
    this.lfo.frequency.value = this._rate;
    this.lfo.start();
  }

  /**
   * Setup routing for tremolo (amplitude modulation) mode
   */
  setupTremoloMode() {
    // Disconnect everything first
    this.disconnectAll();

    if (this._stereo) {
      // Stereo tremolo with phase offset between L and R
      this.input.connect(this.splitter);

      // Create stereo LFO if not exists
      if (!this.lfoStereo) {
        this.lfoStereo = this.context.createOscillator();
        this.lfoStereoGain = this.context.createGain();
        this.lfoStereoOffset = this.context.createGain();
        this.amplitudeModulatorStereo = this.context.createGain();

        this.lfoStereo.type = this._waveform;
        this.lfoStereo.frequency.value = this._rate;
        this.lfoStereo.start();
      }

      // Left channel (0° phase)
      this.lfo.connect(this.lfoGain);
      this.lfoOffset.gain.value = 1; // Offset to keep signal positive
      this.lfoGain.connect(this.lfoOffset.gain);
      this.splitter.connect(this.amplitudeModulator, 0);
      this.lfoOffset.connect(this.amplitudeModulator.gain);
      this.amplitudeModulator.connect(this.merger, 0, 0);

      // Right channel (180° phase)
      this.lfoStereo.connect(this.lfoStereoGain);
      this.lfoStereoOffset.gain.value = 1;
      this.lfoStereoGain.connect(this.lfoStereoOffset.gain);
      this.splitter.connect(this.amplitudeModulatorStereo, 1);
      this.lfoStereoOffset.connect(this.amplitudeModulatorStereo.gain);
      this.amplitudeModulatorStereo.connect(this.merger, 0, 1);

      this.merger.connect(this.output);
    } else {
      // Mono tremolo
      this.input.connect(this.amplitudeModulator);
      this.amplitudeModulator.connect(this.output);

      // LFO modulates amplitude
      // Use offset to keep signal always positive
      this.lfoOffset.gain.value = 1; // This will be the center point
      this.lfo.connect(this.lfoGain);
      this.lfoGain.connect(this.lfoOffset.gain);
      this.lfoOffset.connect(this.amplitudeModulator.gain);
    }
  }

  /**
   * Setup routing for auto pan mode
   */
  setupPanMode() {
    // Disconnect everything first
    this.disconnectAll();

    // Simple panning modulation
    this.input.connect(this.panner);
    this.panner.connect(this.output);

    // LFO modulates pan position
    this.lfo.connect(this.lfoGain);
    this.lfoGain.connect(this.panner.pan);
  }

  /**
   * Disconnect all nodes
   */
  disconnectAll() {
    try {
      this.input.disconnect();
      this.amplitudeModulator.disconnect();
      this.panner.disconnect();
      this.lfo.disconnect();
      this.lfoGain.disconnect();
      this.lfoOffset.disconnect();
      this.splitter.disconnect();
      this.merger.disconnect();

      if (this.lfoStereo) {
        this.lfoStereo.disconnect();
        this.lfoStereoGain.disconnect();
        this.lfoStereoOffset.disconnect();
        this.amplitudeModulatorStereo.disconnect();
      }
    } catch (e) {
      // Some nodes might not be connected
    }
  }

  /**
   * Initialize or update parameters
   * @param {Object} options - Parameter values
   */
  initialize(options = {}) {
    if (options.mode !== undefined) this.setMode(options.mode);
    if (options.rate !== undefined) this.setRate(options.rate);
    if (options.depth !== undefined) this.setDepth(options.depth);
    if (options.waveform !== undefined) this.setWaveform(options.waveform);
    if (options.phase !== undefined) this.setPhase(options.phase);
    if (options.shape !== undefined) this.setShape(options.shape);
    if (options.sync !== undefined) this.setSync(options.sync);
    if (options.stereo !== undefined) this.setStereo(options.stereo);
  }

  /**
   * Set modulation mode
   * @param {string} mode - 'tremolo' or 'pan'
   */
  setMode(mode) {
    if (mode === 'tremolo' || mode === 'pan') {
      this._mode = mode;

      if (mode === 'tremolo') {
        this.setupTremoloMode();
      } else {
        this.setupPanMode();
      }

      // Reapply depth setting
      this.setDepth(this._depth);
    }
  }

  /**
   * Set LFO rate (speed of modulation)
   * @param {number} hz - Frequency in Hz (0.01 to 40)
   */
  setRate(hz) {
    this._rate = Math.max(0.01, Math.min(40, hz));
    this.lfo.frequency.setValueAtTime(this._rate, this.context.currentTime);

    if (this.lfoStereo) {
      this.lfoStereo.frequency.setValueAtTime(this._rate, this.context.currentTime);
    }
  }

  /**
   * Set modulation depth
   * @param {number} percent - Depth percentage (0 to 100)
   */
  setDepth(percent) {
    this._depth = Math.max(0, Math.min(100, percent));
    const depth = this._depth / 100;

    if (this._mode === 'tremolo') {
      // For tremolo, depth controls amplitude variation
      // Center gain at (1 - depth/2), vary by ±depth/2
      const centerGain = 1 - (depth * 0.5);
      this.lfoOffset.gain.setValueAtTime(centerGain, this.context.currentTime);
      this.lfoGain.gain.setValueAtTime(depth * 0.5, this.context.currentTime);

      if (this.lfoStereoOffset) {
        this.lfoStereoOffset.gain.setValueAtTime(centerGain, this.context.currentTime);
        this.lfoStereoGain.gain.setValueAtTime(depth * 0.5, this.context.currentTime);
      }
    } else {
      // For pan, depth controls pan range (-depth to +depth)
      this.lfoGain.gain.setValueAtTime(depth, this.context.currentTime);
    }
  }

  /**
   * Set LFO waveform
   * @param {string} type - Waveform type ('sine', 'triangle', 'square', 'sawtooth', 'random')
   */
  setWaveform(type) {
    const validTypes = ['sine', 'triangle', 'square', 'sawtooth'];

    if (type === 'random') {
      // Random waveform - use noise
      // This would require a custom implementation with AudioBufferSourceNode
      console.log('Random waveform not yet implemented');
      return;
    }

    if (validTypes.includes(type)) {
      this._waveform = type;

      // Create new LFO with new waveform
      const oldLfo = this.lfo;
      this.lfo = this.context.createOscillator();
      this.lfo.type = this._waveform;
      this.lfo.frequency.value = this._rate;

      // Reconnect based on mode
      oldLfo.disconnect();
      this.lfo.connect(this.lfoGain);
      this.lfo.start();
      oldLfo.stop();

      // Update stereo LFO if exists
      if (this.lfoStereo) {
        const oldStereoLfo = this.lfoStereo;
        this.lfoStereo = this.context.createOscillator();
        this.lfoStereo.type = this._waveform;
        this.lfoStereo.frequency.value = this._rate;

        oldStereoLfo.disconnect();
        this.lfoStereo.connect(this.lfoStereoGain);
        this.lfoStereo.start();
        oldStereoLfo.stop();
      }
    }
  }

  /**
   * Set LFO phase offset
   * @param {number} degrees - Phase offset in degrees (0 to 360)
   */
  setPhase(degrees) {
    this._phase = degrees % 360;
    // Phase offset would require custom waveform or delayed start
    // For simplicity, we'll log this as a placeholder
    console.log('Phase offset requires custom waveform implementation');
  }

  /**
   * Set waveform shaping
   * @param {number} percent - Shape percentage (0 to 100)
   */
  setShape(percent) {
    this._shape = Math.max(0, Math.min(100, percent));
    // Waveform shaping would require custom waveform manipulation
    console.log('Waveform shaping not yet implemented');
  }

  /**
   * Set tempo sync
   * @param {boolean} enabled - Enable tempo sync
   */
  setSync(enabled) {
    this._sync = enabled;
    // TODO: Implement tempo sync with BPM
    console.log('Tempo sync not yet implemented');
  }

  /**
   * Set stereo mode (phase offset between L/R)
   * @param {boolean} enabled - Enable stereo phase offset
   */
  setStereo(enabled) {
    if (this._stereo !== enabled) {
      this._stereo = enabled;

      if (this._mode === 'tremolo') {
        this.setupTremoloMode();
        this.setDepth(this._depth);
      }
    }
  }

  /**
   * Get current parameter values
   * @returns {Object} Current parameter values
   */
  getParams() {
    return {
      rate: this._rate,
      depth: this._depth,
      waveform: this._waveform,
      phase: this._phase,
      mode: this._mode,
      shape: this._shape,
      sync: this._sync,
      stereo: this._stereo
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
    this.disconnectAll();

    if (this.lfoStereo) {
      this.lfoStereo.stop();
    }
  }
}

// Export for use in modules or Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Tremolo;
}
