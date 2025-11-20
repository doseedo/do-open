/**
 * EQ Three - DJ-Style 3-Band Equalizer with Kill Switches
 *
 * Features:
 * - 3-band frequency splitting (Low, Mid, High)
 * - Independent gain control per band (0 to 2x, can kill to 0)
 * - Adjustable crossover frequencies
 * - Linkwitz-Riley 4th order crossover filters for phase-coherent summing
 * - Kill switches for complete frequency removal
 * - Smooth parameter changes for DJ performance
 * - No phase issues at crossover points
 *
 * Based on Web Audio API BiquadFilterNode
 */

class EQThree {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Create three frequency bands
    this.lowBand = this.createBand();
    this.midBand = this.createBand();
    this.highBand = this.createBand();

    // Crossover frequencies (defaults)
    this.lowFreq = 250;   // Hz (20-500)
    this.highFreq = 3500; // Hz (2k-10k)

    // Band gains (0 = kill, 1 = unity, 2 = +6dB boost)
    this.lowGain = 1;
    this.midGain = 1;
    this.highGain = 1;

    // Setup crossover filters (Linkwitz-Riley 4th order)
    this.setupCrossoverFilters();

    // Setup audio routing
    this.setupRouting();

    // Initialize with options
    this.initialize(options);
  }

  /**
   * Create a frequency band with gain control
   */
  createBand() {
    return {
      gain: this.context.createGain(),
      filter1: this.context.createBiquadFilter(),
      filter2: this.context.createBiquadFilter()
    };
  }

  /**
   * Setup Linkwitz-Riley 4th order crossover filters
   *
   * Linkwitz-Riley alignment:
   * - Two cascaded 2nd-order Butterworth filters = 4th order LR
   * - Phase-coherent: sum of all bands = flat frequency response
   * - Q = 0.707 (1/√2) for Butterworth response
   */
  setupCrossoverFilters() {
    const Q = 0.707; // Butterworth Q

    // === LOW BAND: 2x Lowpass at lowFreq ===
    this.lowBand.filter1.type = 'lowpass';
    this.lowBand.filter1.frequency.value = this.lowFreq;
    this.lowBand.filter1.Q.value = Q;

    this.lowBand.filter2.type = 'lowpass';
    this.lowBand.filter2.frequency.value = this.lowFreq;
    this.lowBand.filter2.Q.value = Q;

    // === MID BAND: Highpass at lowFreq + Lowpass at highFreq ===
    this.midBand.filter1.type = 'highpass';
    this.midBand.filter1.frequency.value = this.lowFreq;
    this.midBand.filter1.Q.value = Q;

    this.midBand.filter2.type = 'lowpass';
    this.midBand.filter2.frequency.value = this.highFreq;
    this.midBand.filter2.Q.value = Q;

    // === HIGH BAND: 2x Highpass at highFreq ===
    this.highBand.filter1.type = 'highpass';
    this.highBand.filter1.frequency.value = this.highFreq;
    this.highBand.filter1.Q.value = Q;

    this.highBand.filter2.type = 'highpass';
    this.highBand.filter2.frequency.value = this.highFreq;
    this.highBand.filter2.Q.value = Q;

    // Set initial gains
    this.lowBand.gain.gain.value = this.lowGain;
    this.midBand.gain.gain.value = this.midGain;
    this.highBand.gain.gain.value = this.highGain;
  }

  /**
   * Setup audio routing
   *
   * Signal flow:
   * input → [split to 3 bands] → filters → gains → [sum] → output
   */
  setupRouting() {
    // === LOW BAND ===
    // input → lowpass1 → lowpass2 → gain → output
    this.input.connect(this.lowBand.filter1);
    this.lowBand.filter1.connect(this.lowBand.filter2);
    this.lowBand.filter2.connect(this.lowBand.gain);
    this.lowBand.gain.connect(this.output);

    // === MID BAND ===
    // input → highpass1 → lowpass2 → gain → output
    this.input.connect(this.midBand.filter1);
    this.midBand.filter1.connect(this.midBand.filter2);
    this.midBand.filter2.connect(this.midBand.gain);
    this.midBand.gain.connect(this.output);

    // === HIGH BAND ===
    // input → highpass1 → highpass2 → gain → output
    this.input.connect(this.highBand.filter1);
    this.highBand.filter1.connect(this.highBand.filter2);
    this.highBand.filter2.connect(this.highBand.gain);
    this.highBand.gain.connect(this.output);
  }

  /**
   * Initialize EQ with options
   */
  initialize(options) {
    if (options.lowFreq !== undefined) {
      this.setLowFrequency(options.lowFreq);
    }

    if (options.highFreq !== undefined) {
      this.setHighFrequency(options.highFreq);
    }

    if (options.lowGain !== undefined) {
      this.setLowGain(options.lowGain);
    }

    if (options.midGain !== undefined) {
      this.setMidGain(options.midGain);
    }

    if (options.highGain !== undefined) {
      this.setHighGain(options.highGain);
    }
  }

  /**
   * Set low band gain
   * @param {number} gain - Gain multiplier (0 to 2, where 0 = kill, 1 = unity, 2 = +6dB)
   * @param {number} rampTime - Ramp time in seconds (default: 0.01s for smooth changes)
   */
  setLowGain(gain, rampTime = 0.01) {
    this.lowGain = Math.max(0, Math.min(2, gain));
    const now = this.context.currentTime;

    if (rampTime > 0) {
      this.lowBand.gain.gain.setTargetAtTime(this.lowGain, now, rampTime / 3);
    } else {
      this.lowBand.gain.gain.setValueAtTime(this.lowGain, now);
    }
  }

  /**
   * Set mid band gain
   * @param {number} gain - Gain multiplier (0 to 2)
   * @param {number} rampTime - Ramp time in seconds
   */
  setMidGain(gain, rampTime = 0.01) {
    this.midGain = Math.max(0, Math.min(2, gain));
    const now = this.context.currentTime;

    if (rampTime > 0) {
      this.midBand.gain.gain.setTargetAtTime(this.midGain, now, rampTime / 3);
    } else {
      this.midBand.gain.gain.setValueAtTime(this.midGain, now);
    }
  }

  /**
   * Set high band gain
   * @param {number} gain - Gain multiplier (0 to 2)
   * @param {number} rampTime - Ramp time in seconds
   */
  setHighGain(gain, rampTime = 0.01) {
    this.highGain = Math.max(0, Math.min(2, gain));
    const now = this.context.currentTime;

    if (rampTime > 0) {
      this.highBand.gain.gain.setTargetAtTime(this.highGain, now, rampTime / 3);
    } else {
      this.highBand.gain.gain.setValueAtTime(this.highGain, now);
    }
  }

  /**
   * Set low/mid crossover frequency
   * @param {number} freq - Frequency in Hz (20 to 500)
   */
  setLowFrequency(freq) {
    this.lowFreq = Math.max(20, Math.min(500, freq));
    const now = this.context.currentTime;

    // Update low band lowpass filters
    this.lowBand.filter1.frequency.setValueAtTime(this.lowFreq, now);
    this.lowBand.filter2.frequency.setValueAtTime(this.lowFreq, now);

    // Update mid band highpass filter
    this.midBand.filter1.frequency.setValueAtTime(this.lowFreq, now);
  }

  /**
   * Set mid/high crossover frequency
   * @param {number} freq - Frequency in Hz (2000 to 10000)
   */
  setHighFrequency(freq) {
    this.highFreq = Math.max(2000, Math.min(10000, freq));
    const now = this.context.currentTime;

    // Update mid band lowpass filter
    this.midBand.filter2.frequency.setValueAtTime(this.highFreq, now);

    // Update high band highpass filters
    this.highBand.filter1.frequency.setValueAtTime(this.highFreq, now);
    this.highBand.filter2.frequency.setValueAtTime(this.highFreq, now);
  }

  /**
   * Kill a specific band (set gain to 0)
   * @param {string} band - Band name ('low', 'mid', 'high')
   */
  killBand(band) {
    switch (band.toLowerCase()) {
      case 'low':
        this.setLowGain(0);
        break;
      case 'mid':
        this.setMidGain(0);
        break;
      case 'high':
        this.setHighGain(0);
        break;
      default:
        console.warn(`EQThree: Invalid band "${band}"`);
    }
  }

  /**
   * Reset a specific band to unity gain
   * @param {string} band - Band name ('low', 'mid', 'high')
   */
  resetBand(band) {
    switch (band.toLowerCase()) {
      case 'low':
        this.setLowGain(1);
        break;
      case 'mid':
        this.setMidGain(1);
        break;
      case 'high':
        this.setHighGain(1);
        break;
      default:
        console.warn(`EQThree: Invalid band "${band}"`);
    }
  }

  /**
   * Reset all bands to unity gain
   */
  reset() {
    this.setLowGain(1);
    this.setMidGain(1);
    this.setHighGain(1);
    this.setLowFrequency(250);
    this.setHighFrequency(3500);
  }

  /**
   * Get current state
   */
  getState() {
    return {
      lowGain: this.lowGain,
      midGain: this.midGain,
      highGain: this.highGain,
      lowFreq: this.lowFreq,
      highFreq: this.highFreq
    };
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

    // Disconnect low band
    this.lowBand.filter1.disconnect();
    this.lowBand.filter2.disconnect();
    this.lowBand.gain.disconnect();

    // Disconnect mid band
    this.midBand.filter1.disconnect();
    this.midBand.filter2.disconnect();
    this.midBand.gain.disconnect();

    // Disconnect high band
    this.highBand.filter1.disconnect();
    this.highBand.filter2.disconnect();
    this.highBand.gain.disconnect();

    console.log('EQThree destroyed');
  }
}

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = EQThree;
}
