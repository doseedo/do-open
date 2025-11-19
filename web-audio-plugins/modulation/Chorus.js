/**
 * Chorus Effect
 * Creates the illusion of multiple voices/instruments by layering slightly detuned delays
 *
 * @class Chorus
 * @param {AudioContext} audioContext - Web Audio API context
 * @param {Object} options - Configuration options
 */
class Chorus {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = this.context.createGain();
    this.output = this.context.createGain();

    // Dry/Wet mix
    this.dryGain = this.context.createGain();
    this.wetGain = this.context.createGain();

    // Voice configuration
    this.maxVoices = 8;
    this.voices = [];
    this._numVoices = 4;

    // Parameters
    this._rate = 0.5;        // LFO speed in Hz
    this._depth = 50;        // Modulation intensity (0-100%)
    this._spread = 50;       // Stereo width (0-100%)
    this._feedback = 0;      // Feedback amount (0-100%)
    this._mix = 50;          // Dry/wet mix (0-100%)
    this._delay = 20;        // Base delay time in ms

    // Create maximum number of voices
    for (let i = 0; i < this.maxVoices; i++) {
      this.voices.push(this.createVoice(i));
    }

    // Setup audio routing
    this.setupRouting();

    // Initialize with default or provided options
    this.initialize(options);
  }

  /**
   * Create a single chorus voice with delay, LFO, and panning
   * @param {number} index - Voice index for phase offset
   * @returns {Object} Voice object containing audio nodes
   */
  createVoice(index) {
    const voice = {
      delay: this.context.createDelay(0.1), // Max 100ms delay
      lfo: this.context.createOscillator(),
      lfoGain: this.context.createGain(),
      panner: this.context.createStereoPanner(),
      feedbackGain: this.context.createGain(),
      outputGain: this.context.createGain(),
      active: false
    };

    // Calculate phase offset for this voice (distributed evenly)
    const phaseOffset = (index / this.maxVoices) * Math.PI * 2;

    // Create custom waveform with phase offset for smooth chorus
    const real = new Float32Array(2);
    const imag = new Float32Array(2);
    real[0] = 0;
    real[1] = Math.cos(phaseOffset);
    imag[0] = 0;
    imag[1] = Math.sin(phaseOffset);

    const wave = this.context.createPeriodicWave(real, imag);
    voice.lfo.setPeriodicWave(wave);

    // Set base delay time (20ms default)
    voice.delay.delayTime.value = 0.020;

    // LFO depth (5ms modulation default)
    voice.lfoGain.gain.value = 0.005;

    // Connect LFO to delay time
    voice.lfo.connect(voice.lfoGain);
    voice.lfoGain.connect(voice.delay.delayTime);

    // Set LFO frequency
    voice.lfo.frequency.value = this._rate;

    // Start LFO
    voice.lfo.start();

    // Initialize panning (spread across stereo field)
    voice.panner.pan.value = 0;

    // Initialize feedback
    voice.feedbackGain.gain.value = 0;

    // Initialize output gain
    voice.outputGain.gain.value = 1.0 / this.maxVoices; // Balance multiple voices

    return voice;
  }

  /**
   * Setup audio routing for all nodes
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet paths (all voices in parallel)
    this.voices.forEach(voice => {
      // Input -> Delay
      this.input.connect(voice.delay);

      // Feedback loop: Delay -> Feedback Gain -> Delay
      voice.delay.connect(voice.feedbackGain);
      voice.feedbackGain.connect(voice.delay);

      // Output path: Delay -> Panner -> Output Gain -> Wet
      voice.delay.connect(voice.panner);
      voice.panner.connect(voice.outputGain);
      voice.outputGain.connect(this.wetGain);
    });

    this.wetGain.connect(this.output);
  }

  /**
   * Initialize or update parameters
   * @param {Object} options - Parameter values
   */
  initialize(options = {}) {
    if (options.rate !== undefined) this.setRate(options.rate);
    if (options.depth !== undefined) this.setDepth(options.depth);
    if (options.voices !== undefined) this.setVoices(options.voices);
    if (options.spread !== undefined) this.setSpread(options.spread);
    if (options.feedback !== undefined) this.setFeedback(options.feedback);
    if (options.mix !== undefined) this.setMix(options.mix);
    if (options.delay !== undefined) this.setDelayTime(options.delay);

    // Update active voices
    this.updateActiveVoices();
  }

  /**
   * Update which voices are active based on numVoices setting
   */
  updateActiveVoices() {
    this.voices.forEach((voice, index) => {
      const shouldBeActive = index < this._numVoices;
      if (shouldBeActive !== voice.active) {
        voice.active = shouldBeActive;
        voice.outputGain.gain.value = shouldBeActive ? (1.0 / this._numVoices) : 0;
      }
    });
  }

  /**
   * Set LFO rate (speed of modulation)
   * @param {number} hz - Frequency in Hz (0.01 to 10)
   */
  setRate(hz) {
    this._rate = Math.max(0.01, Math.min(10, hz));
    this.voices.forEach(voice => {
      voice.lfo.frequency.setValueAtTime(this._rate, this.context.currentTime);
    });
  }

  /**
   * Set modulation depth
   * @param {number} percent - Depth percentage (0 to 100)
   */
  setDepth(percent) {
    this._depth = Math.max(0, Math.min(100, percent));
    const depth = this._depth / 100;
    this.voices.forEach(voice => {
      // Max 10ms modulation at 100% depth
      voice.lfoGain.gain.setValueAtTime(depth * 0.010, this.context.currentTime);
    });
  }

  /**
   * Set number of active voices
   * @param {number} num - Number of voices (1 to 8)
   */
  setVoices(num) {
    this._numVoices = Math.max(1, Math.min(this.maxVoices, Math.floor(num)));
    this.updateActiveVoices();
  }

  /**
   * Set stereo spread
   * @param {number} percent - Spread percentage (0 to 100)
   */
  setSpread(percent) {
    this._spread = Math.max(0, Math.min(100, percent));
    const spread = this._spread / 100;

    this.voices.forEach((voice, index) => {
      if (this._numVoices === 1) {
        voice.panner.pan.setValueAtTime(0, this.context.currentTime);
      } else {
        // Distribute voices across stereo field
        const position = (index / (this._numVoices - 1)) * 2 - 1; // -1 to 1
        voice.panner.pan.setValueAtTime(position * spread, this.context.currentTime);
      }
    });
  }

  /**
   * Set feedback amount
   * @param {number} percent - Feedback percentage (0 to 100)
   */
  setFeedback(percent) {
    this._feedback = Math.max(0, Math.min(100, percent));
    const feedback = this._feedback / 100;

    this.voices.forEach(voice => {
      // Use exponential scaling for more natural feedback
      voice.feedbackGain.gain.setValueAtTime(feedback * 0.7, this.context.currentTime);
    });
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
   * Set base delay time
   * @param {number} ms - Delay time in milliseconds (5 to 50)
   */
  setDelayTime(ms) {
    this._delay = Math.max(5, Math.min(50, ms));
    const delaySeconds = this._delay / 1000;

    this.voices.forEach(voice => {
      voice.delay.delayTime.setValueAtTime(delaySeconds, this.context.currentTime);
    });
  }

  /**
   * Get current parameter values
   * @returns {Object} Current parameter values
   */
  getParams() {
    return {
      rate: this._rate,
      depth: this._depth,
      voices: this._numVoices,
      spread: this._spread,
      feedback: this._feedback,
      mix: this._mix,
      delay: this._delay
    };
  }

  /**
   * Connect input source
   * @param {AudioNode} source - Audio source to connect
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
    this.voices.forEach(voice => {
      voice.lfo.stop();
      voice.lfo.disconnect();
      voice.lfoGain.disconnect();
      voice.delay.disconnect();
      voice.panner.disconnect();
      voice.feedbackGain.disconnect();
      voice.outputGain.disconnect();
    });

    this.input.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
    this.output.disconnect();
  }
}

// Export for use in modules or Node.js
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Chorus;
}
