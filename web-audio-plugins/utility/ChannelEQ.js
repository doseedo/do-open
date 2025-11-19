/**
 * Channel EQ Plugin
 * Simple low-cut and high-cut filters for mixing
 *
 * Features:
 * - Low-cut (highpass) filter: 20-500 Hz
 * - High-cut (lowpass) filter: 2k-20k Hz
 * - Multiple slope options: 6, 12, 18, 24, 36, 48 dB/oct
 * - Efficient processing
 * - Visual frequency response
 */

class ChannelEQ {
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Input/Output
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Low cut (highpass) filters - cascade for higher slopes
    // Each biquad filter provides 12 dB/oct for highpass/lowpass
    this.lowCutFilters = [];
    for (let i = 0; i < 4; i++) {
      const filter = audioContext.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.value = 20;
      filter.Q.value = 0.707; // Butterworth response
      this.lowCutFilters.push(filter);
    }

    // High cut (lowpass) filters - cascade for higher slopes
    this.highCutFilters = [];
    for (let i = 0; i < 4; i++) {
      const filter = audioContext.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 20000;
      filter.Q.value = 0.707; // Butterworth response
      this.highCutFilters.push(filter);
    }

    // State
    this.state = {
      lowCutEnabled: false,
      lowCutFreq: 20,
      lowCutSlope: 12,
      highCutEnabled: false,
      highCutFreq: 20000,
      highCutSlope: 12
    };

    this.setupRouting();
    this.initialize(options);
  }

  setupRouting() {
    // Chain all filters in series
    let current = this.input;

    // Low cut filters
    this.lowCutFilters.forEach(filter => {
      current.connect(filter);
      current = filter;
    });

    // High cut filters
    this.highCutFilters.forEach(filter => {
      current.connect(filter);
      current = filter;
    });

    // Connect to output
    current.connect(this.output);

    // Initially disable all filters
    this.disableAllFilters();
  }

  initialize(options) {
    if (options.lowCutFreq !== undefined) {
      this.setLowCut(options.lowCutFreq, options.lowCutSlope || 12);
    }
    if (options.highCutFreq !== undefined) {
      this.setHighCut(options.highCutFreq, options.highCutSlope || 12);
    }
  }

  /**
   * Disable all filters by setting them to allpass
   */
  disableAllFilters() {
    this.lowCutFilters.forEach(filter => {
      filter.type = 'allpass';
    });
    this.highCutFilters.forEach(filter => {
      filter.type = 'allpass';
    });
  }

  /**
   * Set low cut (highpass) filter
   * @param {number} freq - Frequency in Hz (20-500, 0 = off)
   * @param {number} slope - Slope in dB/octave (6, 12, 18, 24, 36, 48)
   */
  setLowCut(freq, slope = 12) {
    // Validate and clamp frequency
    if (freq === 0 || freq === 'off') {
      this.state.lowCutEnabled = false;
      this.state.lowCutFreq = 0;
      this.updateLowCutFilters();
      return;
    }

    freq = Math.max(20, Math.min(500, freq));
    this.state.lowCutFreq = freq;
    this.state.lowCutEnabled = true;

    // Validate slope
    const validSlopes = [6, 12, 18, 24, 36, 48];
    if (!validSlopes.includes(slope)) {
      console.warn(`Invalid slope: ${slope}. Using 12 dB/oct.`);
      slope = 12;
    }

    this.state.lowCutSlope = slope;
    this.updateLowCutFilters();
  }

  /**
   * Update low cut filters based on current settings
   */
  updateLowCutFilters() {
    if (!this.state.lowCutEnabled) {
      // Disable all low cut filters
      this.lowCutFilters.forEach(filter => {
        filter.type = 'allpass';
      });
      return;
    }

    // Calculate number of filters needed
    // Each biquad highpass provides 12 dB/oct
    // For 6 dB/oct, we need special handling (use one filter with different Q)
    const numFilters = this.state.lowCutSlope === 6 ? 1 : Math.floor(this.state.lowCutSlope / 12);

    // Configure filters
    this.lowCutFilters.forEach((filter, index) => {
      if (index < numFilters) {
        filter.type = 'highpass';
        filter.frequency.value = this.state.lowCutFreq;

        // For 6 dB/oct, use a single filter with Q = 0.5
        // For Butterworth response (12 dB/oct and higher), use Q = 0.707
        if (this.state.lowCutSlope === 6) {
          filter.Q.value = 0.5;
        } else {
          // Butterworth Q values for cascaded filters
          filter.Q.value = this.getButterworth Q(numFilters, index);
        }
      } else {
        // Disable unused filters
        filter.type = 'allpass';
      }
    });
  }

  /**
   * Set high cut (lowpass) filter
   * @param {number} freq - Frequency in Hz (2000-20000, 0 = off)
   * @param {number} slope - Slope in dB/octave (6, 12, 18, 24, 36, 48)
   */
  setHighCut(freq, slope = 12) {
    // Validate and clamp frequency
    if (freq === 0 || freq === 'off') {
      this.state.highCutEnabled = false;
      this.state.highCutFreq = 0;
      this.updateHighCutFilters();
      return;
    }

    freq = Math.max(2000, Math.min(20000, freq));
    this.state.highCutFreq = freq;
    this.state.highCutEnabled = true;

    // Validate slope
    const validSlopes = [6, 12, 18, 24, 36, 48];
    if (!validSlopes.includes(slope)) {
      console.warn(`Invalid slope: ${slope}. Using 12 dB/oct.`);
      slope = 12;
    }

    this.state.highCutSlope = slope;
    this.updateHighCutFilters();
  }

  /**
   * Update high cut filters based on current settings
   */
  updateHighCutFilters() {
    if (!this.state.highCutEnabled) {
      // Disable all high cut filters
      this.highCutFilters.forEach(filter => {
        filter.type = 'allpass';
      });
      return;
    }

    // Calculate number of filters needed
    const numFilters = this.state.highCutSlope === 6 ? 1 : Math.floor(this.state.highCutSlope / 12);

    // Configure filters
    this.highCutFilters.forEach((filter, index) => {
      if (index < numFilters) {
        filter.type = 'lowpass';
        filter.frequency.value = this.state.highCutFreq;

        // For 6 dB/oct, use a single filter with Q = 0.5
        if (this.state.highCutSlope === 6) {
          filter.Q.value = 0.5;
        } else {
          filter.Q.value = this.getButterworthQ(numFilters, index);
        }
      } else {
        // Disable unused filters
        filter.type = 'allpass';
      }
    });
  }

  /**
   * Get Butterworth Q value for cascaded filters
   * For a Butterworth filter, Q values depend on filter order and stage
   */
  getButterworthQ(numFilters, filterIndex) {
    // For Butterworth filters, Q = 1 / (2 * cos((2k + n - 1) * π / (2n)))
    // where n = filter order (number of 2nd order sections), k = section index (1 to n)

    if (numFilters === 1) {
      return 0.707; // 2nd order Butterworth
    }

    const n = numFilters;
    const k = filterIndex + 1;
    const q = 1 / (2 * Math.cos(((2 * k + n - 1) * Math.PI) / (2 * n)));

    return q;
  }

  /**
   * Get frequency response at a given frequency
   * Returns gain in dB
   */
  getFrequencyResponse(frequency) {
    let totalGain = 0; // in dB

    // Low cut contribution
    if (this.state.lowCutEnabled) {
      const octaves = Math.log2(frequency / this.state.lowCutFreq);
      if (octaves < 0) {
        // Below cutoff - apply attenuation
        totalGain += octaves * this.state.lowCutSlope;
      }
    }

    // High cut contribution
    if (this.state.highCutEnabled) {
      const octaves = Math.log2(frequency / this.state.highCutFreq);
      if (octaves > 0) {
        // Above cutoff - apply attenuation
        totalGain -= octaves * this.state.highCutSlope;
      }
    }

    return totalGain;
  }

  /**
   * Get frequency response curve
   * Returns array of {frequency, gain} objects
   */
  getFrequencyResponseCurve(numPoints = 100) {
    const minFreq = 20;
    const maxFreq = 20000;
    const curve = [];

    for (let i = 0; i < numPoints; i++) {
      // Logarithmic frequency spacing
      const freq = minFreq * Math.pow(maxFreq / minFreq, i / (numPoints - 1));
      const gain = this.getFrequencyResponse(freq);

      curve.push({ frequency: freq, gain: gain });
    }

    return curve;
  }

  /**
   * Enable/disable low cut
   */
  setLowCutEnabled(enabled) {
    this.state.lowCutEnabled = enabled;
    this.updateLowCutFilters();
  }

  /**
   * Enable/disable high cut
   */
  setHighCutEnabled(enabled) {
    this.state.highCutEnabled = enabled;
    this.updateHighCutFilters();
  }

  /**
   * Get current state
   */
  getState() {
    return { ...this.state };
  }

  /**
   * Connect to destination
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Cleanup
   */
  destroy() {
    this.disconnect();
    this.input.disconnect();

    this.lowCutFilters.forEach(filter => filter.disconnect());
    this.highCutFilters.forEach(filter => filter.disconnect());
  }
}

// Export for use in Node.js or as module
if (typeof module !== 'undefined' && module.exports) {
  module.exports = ChannelEQ;
}
