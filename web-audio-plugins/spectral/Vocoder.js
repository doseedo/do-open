/**
 * Vocoder
 * Multi-band vocoder for imposing modulator's spectral envelope on carrier
 *
 * @example
 * const audioContext = new AudioContext();
 * const vocoder = new Vocoder(audioContext);
 *
 * // Connect modulator (e.g., voice)
 * micInput.connect(vocoder.input);
 *
 * // Connect carrier (e.g., synthesizer) - optional if using internal carrier
 * synthInput.connect(vocoder.carrierInput);
 *
 * // Connect output
 * vocoder.connect(audioContext.destination);
 *
 * // Configure
 * vocoder.setBands(32);
 * vocoder.setCarrierType('saw');
 */

class Vocoder {
  /**
   * Create a Vocoder
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.bands - Number of bands (8, 16, 32, 40)
   * @param {string} options.range - Frequency range ('low', 'mid', 'high', 'full')
   * @param {number} options.loFreq - Lowest frequency (20 to 500 Hz)
   * @param {number} options.hiFreq - Highest frequency (2k to 20k Hz)
   * @param {number} options.attack - Envelope attack (0.1 to 100 ms)
   * @param {number} options.release - Envelope release (10 to 500 ms)
   * @param {number} options.formant - Formant shift (-4 to +4 semitones)
   * @param {number} options.resonance - Formant emphasis (0 to 1)
   * @param {string} options.carrierSource - Carrier source ('internal', 'external')
   * @param {string} options.carrierType - Internal carrier type ('noise', 'saw', 'pulse')
   * @param {number} options.upperBandLevel - High-frequency emphasis (0 to 1)
   * @param {number} options.mix - Wet/dry mix (0 to 1)
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain(); // Modulator input
    this.carrierInput = audioContext.createGain(); // External carrier input
    this.output = audioContext.createGain();
    this.dryGain = audioContext.createGain();
    this.wetGain = audioContext.createGain();

    // Parameters
    this.parameters = {
      bands: options.bands || 16,
      range: options.range || 'full',
      loFreq: options.loFreq || 80,
      hiFreq: options.hiFreq || 12000,
      attack: options.attack !== undefined ? options.attack : 10,
      release: options.release !== undefined ? options.release : 100,
      formant: options.formant || 0,
      resonance: options.resonance !== undefined ? options.resonance : 0.5,
      carrierSource: options.carrierSource || 'internal',
      carrierType: options.carrierType || 'saw',
      upperBandLevel: options.upperBandLevel !== undefined ? options.upperBandLevel : 1.0,
      mix: options.mix !== undefined ? options.mix : 1.0
    };

    // Internal carrier oscillator
    this.internalCarrier = null;
    this.noiseBuffer = null;

    // Vocoder bands
    this.bands = [];

    // Setup
    this.setupRouting();
    this.createInternalCarrier();
    this.initialize(options);
  }

  /**
   * Setup audio routing
   * @private
   */
  setupRouting() {
    // Dry path (modulator)
    this.input.connect(this.dryGain);
    this.dryGain.connect(this.output);

    // Wet path will be setup when bands are created
    this.wetGain.connect(this.output);

    // Set initial mix
    this.updateMix();
  }

  /**
   * Create internal carrier sources
   * @private
   */
  createInternalCarrier() {
    // Create sawtooth oscillator
    this.internalCarrier = this.context.createOscillator();
    this.internalCarrier.type = 'sawtooth';
    this.internalCarrier.frequency.value = 110; // A2
    this.internalCarrier.start();

    // Create noise buffer
    this.createNoiseBuffer();
  }

  /**
   * Create noise buffer for noise carrier
   * @private
   */
  createNoiseBuffer() {
    const bufferSize = this.context.sampleRate * 2; // 2 seconds of noise
    this.noiseBuffer = this.context.createBuffer(1, bufferSize, this.context.sampleRate);
    const data = this.noiseBuffer.getChannelData(0);

    for (let i = 0; i < bufferSize; i++) {
      data[i] = Math.random() * 2 - 1;
    }
  }

  /**
   * Initialize with options
   * @private
   */
  initialize(options) {
    this.buildBands();
  }

  /**
   * Build vocoder bands
   * @private
   */
  buildBands() {
    // Clean up existing bands
    this.bands.forEach(band => {
      band.modulatorFilter.disconnect();
      band.rectifier.disconnect();
      band.envelopeFilter.disconnect();
      band.carrierFilter.disconnect();
      band.vca.disconnect();
    });

    this.bands = [];

    const numBands = this.parameters.bands;
    const frequencies = this.calculateBandFrequencies();

    for (let i = 0; i < numBands; i++) {
      const band = this.createBand(frequencies[i], i);
      this.bands.push(band);
    }

    // Update carrier routing
    this.updateCarrierRouting();
  }

  /**
   * Calculate band center frequencies
   * @private
   */
  calculateBandFrequencies() {
    const numBands = this.parameters.bands;
    const loFreq = this.parameters.loFreq;
    const hiFreq = this.parameters.hiFreq;

    // Logarithmic spacing for more musical distribution
    const frequencies = [];
    const ratio = Math.pow(hiFreq / loFreq, 1 / numBands);

    for (let i = 0; i < numBands; i++) {
      const freq = loFreq * Math.pow(ratio, i + 0.5);
      frequencies.push(freq);
    }

    return frequencies;
  }

  /**
   * Create a single vocoder band
   * @private
   */
  createBand(centerFreq, index) {
    // Modulator analysis path
    const modulatorFilter = this.context.createBiquadFilter();
    modulatorFilter.type = 'bandpass';
    modulatorFilter.frequency.value = centerFreq;
    modulatorFilter.Q.value = 5 + this.parameters.resonance * 15;

    // Envelope follower (rectifier + lowpass)
    const rectifier = this.context.createWaveShaper();
    rectifier.curve = this.createRectifierCurve();
    rectifier.oversample = 'none';

    const envelopeFilter = this.context.createBiquadFilter();
    envelopeFilter.type = 'lowpass';
    this.updateEnvelopeTime(envelopeFilter);

    // Carrier synthesis path (with formant shift)
    const formantShift = Math.pow(2, this.parameters.formant / 12);
    const carrierFilter = this.context.createBiquadFilter();
    carrierFilter.type = 'bandpass';
    carrierFilter.frequency.value = centerFreq * formantShift;
    carrierFilter.Q.value = 5 + this.parameters.resonance * 15;

    // VCA (voltage-controlled amplifier)
    const vca = this.context.createGain();
    vca.gain.value = 0;

    // Connect modulator path
    this.input.connect(modulatorFilter);
    modulatorFilter.connect(rectifier);
    rectifier.connect(envelopeFilter);

    // Connect envelope to VCA gain
    envelopeFilter.connect(vca.gain);

    // Connect carrier path
    carrierFilter.connect(vca);

    // Apply upper band level
    const bandGain = this.context.createGain();
    const bandLevel = 1.0 - (index / this.parameters.bands) * (1.0 - this.parameters.upperBandLevel);
    bandGain.gain.value = bandLevel;

    vca.connect(bandGain);
    bandGain.connect(this.wetGain);

    return {
      centerFreq,
      index,
      modulatorFilter,
      rectifier,
      envelopeFilter,
      carrierFilter,
      vca,
      bandGain
    };
  }

  /**
   * Create rectifier curve for envelope follower
   * @private
   */
  createRectifierCurve() {
    const curve = new Float32Array(4096);
    for (let i = 0; i < 4096; i++) {
      const x = (i * 2 / 4096) - 1;
      curve[i] = Math.abs(x); // Full-wave rectifier
    }
    return curve;
  }

  /**
   * Update envelope follower time constants
   * @private
   */
  updateEnvelopeTime(filter) {
    // Convert attack/release to cutoff frequency
    // Faster attack/release = higher cutoff
    const avgTime = (this.parameters.attack + this.parameters.release) / 2;
    const cutoff = Math.max(5, Math.min(200, 1000 / avgTime));

    filter.frequency.value = cutoff;
  }

  /**
   * Update carrier routing based on source setting
   * @private
   */
  updateCarrierRouting() {
    // Disconnect all carrier sources first
    this.bands.forEach(band => {
      band.carrierFilter.disconnect();
    });

    if (this.parameters.carrierSource === 'internal') {
      // Use internal carrier
      this.connectInternalCarrier();
    }

    // External carrier is always available via carrierInput
    this.bands.forEach(band => {
      this.carrierInput.connect(band.carrierFilter);
    });
  }

  /**
   * Connect internal carrier to all bands
   * @private
   */
  connectInternalCarrier() {
    if (this.parameters.carrierType === 'noise') {
      // Create noise source
      const noise = this.context.createBufferSource();
      noise.buffer = this.noiseBuffer;
      noise.loop = true;
      noise.start();

      this.bands.forEach(band => {
        noise.connect(band.carrierFilter);
      });

      // Store reference to stop later
      this.currentNoiseSource = noise;
    } else {
      // Use oscillator
      this.internalCarrier.type = this.parameters.carrierType === 'pulse' ? 'square' : 'sawtooth';

      this.bands.forEach(band => {
        this.internalCarrier.connect(band.carrierFilter);
      });
    }
  }

  /**
   * Update all band parameters
   * @private
   */
  updateBandParameters() {
    const formantShift = Math.pow(2, this.parameters.formant / 12);
    const resonance = 5 + this.parameters.resonance * 15;

    this.bands.forEach(band => {
      // Update carrier filter frequency with formant shift
      band.carrierFilter.frequency.value = band.centerFreq * formantShift;

      // Update Q (resonance)
      band.modulatorFilter.Q.value = resonance;
      band.carrierFilter.Q.value = resonance;

      // Update envelope time
      this.updateEnvelopeTime(band.envelopeFilter);

      // Update upper band level
      const bandLevel = 1.0 - (band.index / this.parameters.bands) *
                        (1.0 - this.parameters.upperBandLevel);
      band.bandGain.gain.value = bandLevel;
    });
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
   * Set number of bands
   * @param {number} bands - Number of bands (8, 16, 32, 40)
   */
  setBands(bands) {
    const validBands = [8, 16, 32, 40];
    const closestBand = validBands.reduce((prev, curr) =>
      Math.abs(curr - bands) < Math.abs(prev - bands) ? curr : prev
    );

    if (closestBand !== this.parameters.bands) {
      this.parameters.bands = closestBand;
      this.buildBands();
    }
  }

  /**
   * Set frequency range preset
   * @param {string} range - Range ('low', 'mid', 'high', 'full')
   */
  setRange(range) {
    const ranges = {
      low: { loFreq: 40, hiFreq: 2000 },
      mid: { loFreq: 200, hiFreq: 8000 },
      high: { loFreq: 1000, hiFreq: 20000 },
      full: { loFreq: 80, hiFreq: 12000 }
    };

    if (ranges[range]) {
      this.parameters.range = range;
      this.parameters.loFreq = ranges[range].loFreq;
      this.parameters.hiFreq = ranges[range].hiFreq;
      this.buildBands();
    }
  }

  /**
   * Set lowest frequency
   * @param {number} freq - Frequency in Hz (20 to 500)
   */
  setLoFreq(freq) {
    this.parameters.loFreq = Math.max(20, Math.min(500, freq));
    this.buildBands();
  }

  /**
   * Set highest frequency
   * @param {number} freq - Frequency in Hz (2k to 20k)
   */
  setHiFreq(freq) {
    this.parameters.hiFreq = Math.max(2000, Math.min(20000, freq));
    this.buildBands();
  }

  /**
   * Set envelope attack time
   * @param {number} ms - Attack time (0.1 to 100 ms)
   */
  setAttack(ms) {
    this.parameters.attack = Math.max(0.1, Math.min(100, ms));
    this.updateBandParameters();
  }

  /**
   * Set envelope release time
   * @param {number} ms - Release time (10 to 500 ms)
   */
  setRelease(ms) {
    this.parameters.release = Math.max(10, Math.min(500, ms));
    this.updateBandParameters();
  }

  /**
   * Set formant shift
   * @param {number} semitones - Shift (-4 to +4 semitones)
   */
  setFormant(semitones) {
    this.parameters.formant = Math.max(-4, Math.min(4, semitones));
    this.updateBandParameters();
  }

  /**
   * Set resonance/formant emphasis
   * @param {number} resonance - Resonance (0 to 100%)
   */
  setResonance(resonance) {
    this.parameters.resonance = Math.max(0, Math.min(100, resonance)) / 100;
    this.updateBandParameters();
  }

  /**
   * Set carrier source
   * @param {string} source - Source ('internal', 'external')
   */
  setCarrierSource(source) {
    if (source === 'internal' || source === 'external') {
      this.parameters.carrierSource = source;
      this.updateCarrierRouting();
    }
  }

  /**
   * Set internal carrier type
   * @param {string} type - Type ('noise', 'saw', 'pulse')
   */
  setCarrierType(type) {
    const validTypes = ['noise', 'saw', 'pulse'];
    if (validTypes.includes(type)) {
      // Stop current noise source if switching from noise
      if (this.currentNoiseSource && type !== 'noise') {
        this.currentNoiseSource.stop();
        this.currentNoiseSource = null;
      }

      this.parameters.carrierType = type;

      if (this.parameters.carrierSource === 'internal') {
        this.updateCarrierRouting();
      }
    }
  }

  /**
   * Set upper band level
   * @param {number} level - Level (0 to 100%)
   */
  setUpperBandLevel(level) {
    this.parameters.upperBandLevel = Math.max(0, Math.min(100, level)) / 100;
    this.updateBandParameters();
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
   * Set carrier frequency (for internal carrier)
   * @param {number} freq - Frequency in Hz
   */
  setCarrierFrequency(freq) {
    if (this.internalCarrier) {
      this.internalCarrier.frequency.value = freq;
    }
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
    // Stop internal carrier
    if (this.internalCarrier) {
      this.internalCarrier.stop();
      this.internalCarrier = null;
    }

    if (this.currentNoiseSource) {
      this.currentNoiseSource.stop();
      this.currentNoiseSource = null;
    }

    // Disconnect all bands
    this.bands.forEach(band => {
      band.modulatorFilter.disconnect();
      band.rectifier.disconnect();
      band.envelopeFilter.disconnect();
      band.carrierFilter.disconnect();
      band.vca.disconnect();
      band.bandGain.disconnect();
    });

    this.input.disconnect();
    this.carrierInput.disconnect();
    this.output.disconnect();
    this.dryGain.disconnect();
    this.wetGain.disconnect();
  }
}

// Export for use in modules or browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Vocoder;
}
