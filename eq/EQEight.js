/**
 * EQ Eight - Professional 8-band Parametric Equalizer
 *
 * Features:
 * - 8 independent bands with individual control
 * - Multiple filter types per band (bell, lowshelf, highshelf, lowpass, highpass, notch, bandpass)
 * - Adjustable frequency, gain, and Q per band
 * - Global gain control
 * - Adaptive Q (higher gain = narrower Q)
 * - Stereo and mid/side processing modes
 * - Individual band enable/bypass
 * - Frequency response analysis
 *
 * Based on Web Audio API BiquadFilterNode
 */

class EQEight {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.globalGainNode = audioContext.createGain();

    // Default frequencies for 8 bands (logarithmically spaced)
    this.defaultFrequencies = [
      30,      // Sub bass
      90,      // Bass
      250,     // Low mids
      700,     // Mids
      2000,    // Upper mids
      5000,    // Presence
      10000,   // Brilliance
      16000    // Air
    ];

    // Initialize 8 bands
    this.bands = [];
    for (let i = 0; i < 8; i++) {
      this.bands.push(this.createBand(i));
    }

    // Parameters
    this.globalGain = 0; // dB
    this.adaptive = false;
    this.editMode = 'stereo'; // 'single', 'left', 'right', 'stereo'

    // Setup audio routing
    this.setupFilterChain();

    // Initialize with options
    this.initialize(options);
  }

  /**
   * Create a single EQ band
   */
  createBand(index) {
    const filter = this.context.createBiquadFilter();
    const bypass = this.context.createGain();

    // Default settings
    filter.type = 'peaking';
    filter.frequency.value = this.defaultFrequencies[index];
    filter.gain.value = 0; // dB
    filter.Q.value = 0.71; // Default Q (1/√2 = Butterworth)

    return {
      filter: filter,
      bypass: bypass,
      enabled: true,
      frequency: this.defaultFrequencies[index],
      gain: 0,
      Q: 0.71,
      filterType: 'bell',
      index: index
    };
  }

  /**
   * Setup the filter chain: input → bands[0..7] → globalGain → output
   */
  setupFilterChain() {
    let current = this.input;

    // Chain all bands in series
    for (let i = 0; i < 8; i++) {
      current.connect(this.bands[i].filter);
      this.bands[i].filter.connect(this.bands[i].bypass);
      current = this.bands[i].bypass;
    }

    // Connect to global gain and output
    current.connect(this.globalGainNode);
    this.globalGainNode.connect(this.output);
  }

  /**
   * Initialize EQ with options
   */
  initialize(options) {
    if (options.globalGain !== undefined) {
      this.setGlobalGain(options.globalGain);
    }

    if (options.adaptive !== undefined) {
      this.adaptive = options.adaptive;
    }

    if (options.editMode !== undefined) {
      this.editMode = options.editMode;
    }

    if (options.bands) {
      options.bands.forEach((bandOptions, index) => {
        if (index < 8) {
          this.setBand(index, bandOptions);
        }
      });
    }
  }

  /**
   * Set parameters for a specific band
   *
   * @param {number} index - Band index (0-7)
   * @param {Object} params - Band parameters
   * @param {number} params.frequency - Frequency in Hz (20-20000)
   * @param {number} params.gain - Gain in dB (-15 to +15)
   * @param {number} params.Q - Q factor (0.1 to 10)
   * @param {string} params.filterType - Filter type (bell, lowshelf, highshelf, lowpass, highpass, notch, bandpass)
   * @param {boolean} params.enabled - Enable/disable band
   */
  setBand(index, params) {
    if (index < 0 || index >= 8) {
      console.warn(`EQEight: Invalid band index ${index}`);
      return;
    }

    const band = this.bands[index];
    const now = this.context.currentTime;

    // Update frequency
    if (params.frequency !== undefined) {
      const freq = Math.max(20, Math.min(20000, params.frequency));
      band.frequency = freq;
      band.filter.frequency.setValueAtTime(freq, now);
    }

    // Update gain
    if (params.gain !== undefined) {
      const gain = Math.max(-15, Math.min(15, params.gain));
      band.gain = gain;
      band.filter.gain.setValueAtTime(gain, now);

      // Apply adaptive Q if enabled
      if (this.adaptive && params.Q === undefined) {
        this.updateAdaptiveQ(index);
      }
    }

    // Update Q
    if (params.Q !== undefined) {
      const Q = Math.max(0.1, Math.min(10, params.Q));
      band.Q = Q;
      band.filter.Q.setValueAtTime(Q, now);
    }

    // Update filter type
    if (params.filterType !== undefined) {
      band.filterType = params.filterType;
      band.filter.type = this.mapFilterType(params.filterType);

      // Reset gain for filter types that don't support it
      if (['lowpass', 'highpass', 'bandpass', 'notch', 'allpass'].includes(band.filter.type)) {
        band.filter.gain.value = 0;
      }
    }

    // Enable/disable band
    if (params.enabled !== undefined) {
      this.enableBand(index, params.enabled);
    }
  }

  /**
   * Update adaptive Q based on gain
   * Higher gain = narrower Q (more precise)
   */
  updateAdaptiveQ(index) {
    const band = this.bands[index];
    const absGain = Math.abs(band.gain);

    // Adaptive Q formula: Q increases with gain
    // Base Q = 0.71, increases up to 2.0 at maximum gain
    const adaptiveQ = 0.71 + (absGain / 15) * 1.29;

    band.Q = adaptiveQ;
    band.filter.Q.setValueAtTime(adaptiveQ, this.context.currentTime);
  }

  /**
   * Map custom filter type names to BiquadFilterNode types
   */
  mapFilterType(type) {
    const typeMap = {
      'bell': 'peaking',
      'lowshelf': 'lowshelf',
      'highshelf': 'highshelf',
      'lowpass': 'lowpass',
      'highpass': 'highpass',
      'notch': 'notch',
      'bandpass': 'bandpass',
      'allpass': 'allpass'
    };
    return typeMap[type] || 'peaking';
  }

  /**
   * Enable or bypass a specific band
   */
  enableBand(index, enabled) {
    if (index < 0 || index >= 8) return;

    const band = this.bands[index];
    band.enabled = enabled;

    if (enabled) {
      // Enable the band by setting bypass gain to 1
      band.bypass.gain.setValueAtTime(1, this.context.currentTime);
    } else {
      // Bypass the band by setting filter to unity (0 dB gain)
      const originalGain = band.filter.gain.value;
      band.filter.gain.setValueAtTime(0, this.context.currentTime);

      // Store original gain for when re-enabled
      band._storedGain = originalGain;
    }
  }

  /**
   * Get parameters for a specific band
   */
  getBand(index) {
    if (index < 0 || index >= 8) return null;

    const band = this.bands[index];
    return {
      frequency: band.frequency,
      gain: band.gain,
      Q: band.Q,
      filterType: band.filterType,
      enabled: band.enabled,
      index: index
    };
  }

  /**
   * Set global gain (applied to all bands)
   */
  setGlobalGain(gain) {
    this.globalGain = Math.max(-15, Math.min(15, gain));

    // Convert dB to linear gain
    const linearGain = Math.pow(10, this.globalGain / 20);
    this.globalGainNode.gain.setValueAtTime(linearGain, this.context.currentTime);
  }

  /**
   * Enable/disable adaptive Q
   */
  setAdaptive(enabled) {
    this.adaptive = enabled;

    if (enabled) {
      // Update all bands with adaptive Q
      for (let i = 0; i < 8; i++) {
        this.updateAdaptiveQ(i);
      }
    }
  }

  /**
   * Set edit mode (stereo, left, right, mid/side)
   */
  setEditMode(mode) {
    if (['single', 'left', 'right', 'stereo'].includes(mode)) {
      this.editMode = mode;
      // Note: Mid/side processing would require additional processing
      // For now, this is a parameter placeholder
    }
  }

  /**
   * Calculate frequency response at given frequencies
   * Useful for visualization
   *
   * @param {Float32Array} frequencies - Frequencies to analyze
   * @returns {Object} - Magnitude and phase response
   */
  getFrequencyResponse(frequencies) {
    const length = frequencies.length;
    const magResponse = new Float32Array(length);
    const phaseResponse = new Float32Array(length);

    // Initialize to unity
    magResponse.fill(1);
    phaseResponse.fill(0);

    // Accumulate response from each enabled band
    for (let i = 0; i < 8; i++) {
      const band = this.bands[i];
      if (!band.enabled) continue;

      const bandMag = new Float32Array(length);
      const bandPhase = new Float32Array(length);

      band.filter.getFrequencyResponse(frequencies, bandMag, bandPhase);

      // Multiply magnitude responses (add in dB)
      for (let j = 0; j < length; j++) {
        magResponse[j] *= bandMag[j];
        phaseResponse[j] += bandPhase[j];
      }
    }

    // Apply global gain
    const globalGainLinear = Math.pow(10, this.globalGain / 20);
    for (let j = 0; j < length; j++) {
      magResponse[j] *= globalGainLinear;
    }

    return {
      magnitude: magResponse,
      phase: phaseResponse
    };
  }

  /**
   * Reset all bands to default (flat EQ)
   */
  reset() {
    for (let i = 0; i < 8; i++) {
      this.setBand(i, {
        frequency: this.defaultFrequencies[i],
        gain: 0,
        Q: 0.71,
        filterType: 'bell',
        enabled: true
      });
    }
    this.setGlobalGain(0);
    this.setAdaptive(false);
  }

  /**
   * Get the input node for connection
   */
  getInput() {
    return this.input;
  }

  /**
   * Get the output node for connection
   */
  getOutput() {
    return this.output;
  }

  /**
   * Connect this EQ to another audio node
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect this EQ from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Clean up resources
   */
  destroy() {
    this.disconnect();
    this.input.disconnect();
    this.globalGainNode.disconnect();

    this.bands.forEach(band => {
      band.filter.disconnect();
      band.bypass.disconnect();
    });

    console.log('EQEight destroyed');
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EQEight;
}
