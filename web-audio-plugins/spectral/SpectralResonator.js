/**
 * Spectral Resonator
 * Resonant comb filtering based on spectral analysis and pitch tracking
 *
 * @example
 * const audioContext = new AudioContext();
 * const resonator = new SpectralResonator(audioContext);
 *
 * // Connect audio source
 * source.connect(resonator.input);
 * resonator.connect(audioContext.destination);
 *
 * // Set resonant frequency
 * resonator.setPitch(440); // A4
 *
 * // Set number of harmonics
 * resonator.setHarmonics(16);
 *
 * // Set decay time
 * resonator.setDecay(2.0); // 2 seconds
 */

class SpectralResonator {
  /**
   * Create a Spectral Resonator
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {string} options.mode - Resonance mode ('pitched', 'noise', 'hybrid')
   * @param {number} options.pitch - Resonant frequency in Hz
   * @param {number} options.decay - Decay time in seconds
   * @param {number} options.color - Harmonic emphasis (0 to 1)
   * @param {number} options.harmonics - Number of harmonics (1 to 32)
   * @param {number} options.stretch - Harmonic spacing (0.5 to 2.0)
   * @param {number} options.detune - Detune in cents (-100 to +100)
   * @param {number} options.attack - Attack time in ms
   * @param {number} options.release - Release time in ms
   * @param {number} options.mix - Wet/dry mix (0 to 1)
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    // Envelope follower
    this.envelopeFollower = this.createEnvelopeFollower();

    // Parameters
    this.parameters = {
      mode: options.mode || 'pitched',
      pitch: options.pitch || 440,
      decay: options.decay || 1.0,
      color: options.color !== undefined ? options.color : 0.5,
      harmonics: options.harmonics || 8,
      stretch: options.stretch || 1.0,
      detune: options.detune || 0,
      attack: options.attack || 10,
      release: options.release || 100,
      mix: options.mix !== undefined ? options.mix : 0.5
    };

    // Harmonic resonators
    this.resonators = [];
    this.harmonicGains = [];

    // Setup routing
    this.setupRouting();
    this.initialize(options);
  }

  /**
   * Create envelope follower
   * @private
   */
  createEnvelopeFollower() {
    // Rectifier (abs value approximation)
    const rectifier = this.context.createWaveShaper();
    const curve = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) {
      const x = (i * 2 / 4096) - 1;
      curve[i] = Math.abs(x);
    }
    rectifier.curve = curve;

    // Smoothing filter
    const smoother = this.context.createBiquadFilter();
    smoother.type = 'lowpass';
    smoother.frequency.value = 20;
    smoother.Q.value = 0.707;

    rectifier.connect(smoother);

    return { rectifier, smoother };
  }

  /**
   * Setup audio routing
   * @private
   */
  setupRouting() {
    // Dry path
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path will be setup when harmonics are created
    this.wetGain.connect(this.output);

    // Set initial mix
    this.updateMix();
  }

  /**
   * Initialize with options
   * @private
   */
  initialize(options) {
    this.buildResonators();
    this.updateParameters();
  }

  /**
   * Build harmonic resonators
   * @private
   */
  buildResonators() {
    // Clean up existing resonators
    this.resonators.forEach(resonator => {
      resonator.delay.disconnect();
      resonator.feedback.disconnect();
      resonator.feedforward.disconnect();
      resonator.gain.disconnect();
    });

    this.resonators = [];
    this.harmonicGains = [];

    const numHarmonics = this.parameters.harmonics;

    for (let i = 0; i < numHarmonics; i++) {
      const resonator = this.createHarmonicResonator(i + 1);
      this.resonators.push(resonator);

      // Connect input to resonator
      this.input.connect(resonator.feedforward);

      // Connect resonator to wet output
      resonator.gain.connect(this.wetGain);
    }

    this.updateResonatorFrequencies();
  }

  /**
   * Create a single harmonic resonator (comb filter)
   * @private
   */
  createHarmonicResonator(harmonicNumber) {
    // Delay line
    const delay = this.context.createDelay(1.0);

    // Feedback and feedforward paths
    const feedback = this.context.createGain();
    const feedforward = this.context.createGain();

    // Output gain for this harmonic
    const gain = this.context.createGain();

    // Comb filter structure:
    // input → feedforward → gain → output
    //           ↓
    //         delay → feedback → delay (loop)

    feedforward.connect(delay);
    delay.connect(feedback);
    feedback.connect(delay); // Feedback loop
    delay.connect(gain);

    return {
      delay,
      feedback,
      feedforward,
      gain,
      harmonicNumber
    };
  }

  /**
   * Update resonator frequencies
   * @private
   */
  updateResonatorFrequencies() {
    const fundamentalFreq = this.parameters.pitch;
    const detuneFactor = Math.pow(2, this.parameters.detune / 1200);
    const stretch = this.parameters.stretch;

    this.resonators.forEach(resonator => {
      // Calculate harmonic frequency with stretch
      const harmonicFreq = fundamentalFreq * Math.pow(resonator.harmonicNumber, stretch) * detuneFactor;

      // Clamp to valid range
      const clampedFreq = Math.max(20, Math.min(20000, harmonicFreq));

      // Set delay time (period of the frequency)
      if (clampedFreq > 0) {
        resonator.delay.delayTime.value = 1 / clampedFreq;
      }

      // Set harmonic amplitude (decays with harmonic number)
      const color = this.parameters.color;
      const amplitude = 1 / Math.pow(resonator.harmonicNumber, 1 + color);
      resonator.gain.gain.value = amplitude;
    });
  }

  /**
   * Update feedback gains based on decay time
   * @private
   */
  updateDecay() {
    const decayTime = this.parameters.decay;

    this.resonators.forEach(resonator => {
      // Convert decay time to feedback gain
      // feedback = 0.001^(1 / (decay_time * sample_rate))
      const samplesInDecay = decayTime * this.context.sampleRate;
      const feedback = Math.pow(0.001, 1 / samplesInDecay);

      resonator.feedback.gain.value = Math.min(0.9999, feedback);
    });
  }

  /**
   * Update all parameters
   * @private
   */
  updateParameters() {
    this.updateResonatorFrequencies();
    this.updateDecay();
    this.updateMix();
  }

  /**
   * Update wet/dry mix
   * @private
   */
  updateMix() {
    const mix = this.parameters.mix;
    this.wetGain.gain.value = mix;
    this.dryGain.gain.value = 1 - mix;
  }

  /**
   * Set resonance mode
   * @param {string} mode - Mode ('pitched', 'noise', 'hybrid')
   */
  setMode(mode) {
    this.parameters.mode = mode;
    // Mode affects which frequencies resonate
    // For simplicity, we just use pitched mode
    // Full implementation would filter input differently per mode
  }

  /**
   * Set resonant pitch
   * @param {number} pitch - Frequency in Hz or MIDI note number
   */
  setPitch(pitch) {
    // If pitch > 127, assume it's Hz, otherwise MIDI note
    if (pitch <= 127 && pitch >= 0) {
      // Convert MIDI to Hz
      this.parameters.pitch = 440 * Math.pow(2, (pitch - 69) / 12);
    } else {
      this.parameters.pitch = Math.max(20, Math.min(20000, pitch));
    }

    this.updateResonatorFrequencies();
  }

  /**
   * Set decay time
   * @param {number} seconds - Decay time in seconds (0.1 to 10)
   */
  setDecay(seconds) {
    this.parameters.decay = Math.max(0.1, Math.min(10, seconds));
    this.updateDecay();
  }

  /**
   * Set harmonic color/emphasis
   * @param {number} color - Color (0 to 100%)
   */
  setColor(color) {
    this.parameters.color = Math.max(0, Math.min(100, color)) / 100;
    this.updateResonatorFrequencies();
  }

  /**
   * Set number of harmonics
   * @param {number} num - Number of harmonics (1 to 32)
   */
  setHarmonics(num) {
    const harmonics = Math.max(1, Math.min(32, Math.floor(num)));

    if (harmonics !== this.parameters.harmonics) {
      this.parameters.harmonics = harmonics;
      this.buildResonators();
    }
  }

  /**
   * Set harmonic stretch
   * @param {number} stretch - Stretch factor (0.5 to 2.0)
   */
  setStretch(stretch) {
    this.parameters.stretch = Math.max(0.5, Math.min(2.0, stretch));
    this.updateResonatorFrequencies();
  }

  /**
   * Set detune
   * @param {number} cents - Detune in cents (-100 to +100)
   */
  setDetune(cents) {
    this.parameters.detune = Math.max(-100, Math.min(100, cents));
    this.updateResonatorFrequencies();
  }

  /**
   * Set attack time
   * @param {number} ms - Attack time in milliseconds (0 to 500)
   */
  setAttack(ms) {
    this.parameters.attack = Math.max(0, Math.min(500, ms));
    // Attack/release would be used with envelope follower
    // For now, stored for future use
  }

  /**
   * Set release time
   * @param {number} ms - Release time in milliseconds (10 to 5000)
   */
  setRelease(ms) {
    this.parameters.release = Math.max(10, Math.min(5000, ms));
    // Release would be used with envelope follower
    // For now, stored for future use
  }

  /**
   * Set wet/dry mix
   * @param {number} mix - Mix (0 to 100%)
   */
  setMix(mix) {
    this.parameters.mix = Math.max(0, Math.min(100, mix)) / 100;
    this.updateMix();
  }

  /**
   * Connect to an audio node
   * @param {AudioNode} destination - Destination node
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Dispose of the processor and free resources
   */
  dispose() {
    this.resonators.forEach(resonator => {
      resonator.delay.disconnect();
      resonator.feedback.disconnect();
      resonator.feedforward.disconnect();
      resonator.gain.disconnect();
    });

    this.input.disconnect();
    this.output.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
  }
}

// Export for use in modules or browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SpectralResonator;
}
