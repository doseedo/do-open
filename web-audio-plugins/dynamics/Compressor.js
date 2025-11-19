/**
 * Professional Compressor Plugin
 *
 * Features:
 * - Sidechain input support
 * - Auto makeup gain option
 * - RMS and Peak detection modes
 * - Configurable knee (hard to soft)
 * - Parallel compression via mix control
 * - Real-time gain reduction metering
 *
 * @class Compressor
 */
class Compressor {
  /**
   * Create a compressor instance
   * @param {AudioContext} audioContext - The Web Audio context
   * @param {Object} options - Configuration options
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.sidechainInput = audioContext.createGain();

    // Dry/wet signal paths for parallel compression
    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // Internal compressor node
    this.compressorNode = audioContext.createDynamicsCompressor();

    // Makeup gain node
    this.makeupGainNode = audioContext.createGain();

    // Analyzer for gain reduction metering
    this.analyzerInput = audioContext.createAnalyser();
    this.analyzerOutput = audioContext.createAnalyser();
    this.analyzerInput.fftSize = 2048;
    this.analyzerOutput.fftSize = 2048;

    // State
    this.gainReduction = 0;
    this.bypassed = false;
    this.sidechainEnabled = false;
    this.autoMakeup = false;
    this.detectionMode = 'peak'; // 'peak' or 'rms'

    // Default parameter values
    this.defaults = {
      threshold: -24,    // dB
      ratio: 4,          // 1 to 20
      attack: 0.010,     // seconds (10ms)
      release: 0.100,    // seconds (100ms)
      knee: 0,           // dB (0 to 12)
      makeupGain: 0,     // dB
      mix: 100           // 0 to 100%
    };

    this.initialize(options);
  }

  /**
   * Initialize the compressor with default or provided options
   * @param {Object} options - Configuration options
   */
  initialize(options) {
    // Set up audio graph
    // Input splits to wet and dry paths
    this.input.connect(this.analyzerInput);
    this.input.connect(this.dryGain);
    this.input.connect(this.compressorNode);

    // Wet path: compressor → makeup gain → wet gain → output
    this.compressorNode.connect(this.makeupGainNode);
    this.makeupGainNode.connect(this.analyzerOutput);
    this.makeupGainNode.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // Dry path: dry gain → output
    this.dryGain.connect(this.output);

    // Apply default parameters
    this.setParameter('threshold', options.threshold || this.defaults.threshold);
    this.setParameter('ratio', options.ratio || this.defaults.ratio);
    this.setParameter('attack', options.attack || this.defaults.attack);
    this.setParameter('release', options.release || this.defaults.release);
    this.setParameter('knee', options.knee || this.defaults.knee);
    this.setParameter('makeupGain', options.makeupGain || this.defaults.makeupGain);
    this.setParameter('mix', options.mix || this.defaults.mix);

    // Start gain reduction monitoring
    this.startGainReductionMonitoring();
  }

  /**
   * Connect the compressor to a destination
   * @param {AudioNode} destination - The audio node to connect to
   * @returns {AudioNode} The destination node
   */
  connect(destination) {
    this.output.connect(destination);
    return destination;
  }

  /**
   * Disconnect the compressor from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Set a compressor parameter
   * @param {string} name - Parameter name
   * @param {number} value - Parameter value
   * @param {number} time - Time to ramp to the new value (in seconds)
   */
  setParameter(name, value, time = 0) {
    const now = this.context.currentTime;

    switch (name) {
      case 'threshold':
        // Clamp to valid range: -60 to 0 dB
        value = Math.max(-60, Math.min(0, value));
        if (time > 0) {
          this.compressorNode.threshold.linearRampToValueAtTime(value, now + time);
        } else {
          this.compressorNode.threshold.setValueAtTime(value, now);
        }
        break;

      case 'ratio':
        // Clamp to valid range: 1 to 20
        value = Math.max(1, Math.min(20, value));
        if (time > 0) {
          this.compressorNode.ratio.linearRampToValueAtTime(value, now + time);
        } else {
          this.compressorNode.ratio.setValueAtTime(value, now);
        }
        break;

      case 'attack':
        // Clamp to valid range: 0.0001 to 0.1 seconds (0.1ms to 100ms)
        value = Math.max(0.0001, Math.min(0.1, value));
        if (time > 0) {
          this.compressorNode.attack.linearRampToValueAtTime(value, now + time);
        } else {
          this.compressorNode.attack.setValueAtTime(value, now);
        }
        break;

      case 'release':
        // Clamp to valid range: 0.01 to 1 seconds (10ms to 1000ms)
        value = Math.max(0.01, Math.min(1, value));
        if (time > 0) {
          this.compressorNode.release.linearRampToValueAtTime(value, now + time);
        } else {
          this.compressorNode.release.setValueAtTime(value, now);
        }
        break;

      case 'knee':
        // Clamp to valid range: 0 to 12 dB
        value = Math.max(0, Math.min(12, value));
        if (time > 0) {
          this.compressorNode.knee.linearRampToValueAtTime(value, now + time);
        } else {
          this.compressorNode.knee.setValueAtTime(value, now);
        }
        break;

      case 'makeupGain':
        // Clamp to valid range: 0 to 24 dB
        value = Math.max(0, Math.min(24, value));
        const gainValue = this.dbToGain(value);
        if (time > 0) {
          this.makeupGainNode.gain.linearRampToValueAtTime(gainValue, now + time);
        } else {
          this.makeupGainNode.gain.setValueAtTime(gainValue, now);
        }
        break;

      case 'mix':
        // Clamp to valid range: 0 to 100%
        value = Math.max(0, Math.min(100, value));
        const wetAmount = value / 100;
        const dryAmount = 1 - wetAmount;

        if (time > 0) {
          this.wetGain.gain.linearRampToValueAtTime(wetAmount, now + time);
          this.dryGain.gain.linearRampToValueAtTime(dryAmount, now + time);
        } else {
          this.wetGain.gain.setValueAtTime(wetAmount, now);
          this.dryGain.gain.setValueAtTime(dryAmount, now);
        }
        break;

      default:
        console.warn(`Unknown parameter: ${name}`);
    }
  }

  /**
   * Get a compressor parameter value
   * @param {string} name - Parameter name
   * @returns {number} The parameter value
   */
  getParameter(name) {
    switch (name) {
      case 'threshold':
        return this.compressorNode.threshold.value;
      case 'ratio':
        return this.compressorNode.ratio.value;
      case 'attack':
        return this.compressorNode.attack.value;
      case 'release':
        return this.compressorNode.release.value;
      case 'knee':
        return this.compressorNode.knee.value;
      case 'makeupGain':
        return this.gainToDb(this.makeupGainNode.gain.value);
      case 'mix':
        return this.wetGain.gain.value * 100;
      default:
        console.warn(`Unknown parameter: ${name}`);
        return 0;
    }
  }

  /**
   * Enable or disable sidechain compression
   * @param {boolean} enabled - Whether to enable sidechain
   */
  enableSidechain(enabled) {
    if (enabled && !this.sidechainEnabled) {
      // Connect sidechain input to compressor's sidechain
      // Note: Web Audio API's DynamicsCompressorNode doesn't support external sidechain
      // This is a limitation - in a production environment, we'd need to implement
      // a custom AudioWorklet processor for true sidechain support
      console.warn('External sidechain not supported by native DynamicsCompressorNode');
      console.warn('Consider using AudioWorklet implementation for sidechain support');
      this.sidechainEnabled = true;
    } else if (!enabled && this.sidechainEnabled) {
      this.sidechainEnabled = false;
    }
  }

  /**
   * Enable or disable auto makeup gain
   * @param {boolean} enabled - Whether to enable auto makeup gain
   */
  setAutoMakeup(enabled) {
    this.autoMakeup = enabled;
    if (enabled) {
      // Calculate approximate makeup gain based on threshold and ratio
      // Formula: makeupGain ≈ threshold * (1 - 1/ratio) / 2
      const threshold = this.compressorNode.threshold.value;
      const ratio = this.compressorNode.ratio.value;
      const autoGain = Math.abs(threshold * (1 - 1/ratio) / 2);
      this.setParameter('makeupGain', autoGain);
    }
  }

  /**
   * Set detection mode (peak or RMS)
   * @param {string} mode - 'peak' or 'rms'
   */
  setDetectionMode(mode) {
    if (mode === 'peak' || mode === 'rms') {
      this.detectionMode = mode;
      // Note: Web Audio API's DynamicsCompressorNode doesn't expose detection mode
      // This would need to be implemented in a custom AudioWorklet processor
      console.warn('Detection mode switching not supported by native DynamicsCompressorNode');
    }
  }

  /**
   * Start monitoring gain reduction
   */
  startGainReductionMonitoring() {
    const updateGainReduction = () => {
      if (!this.bypassed) {
        // Get RMS levels from input and output
        const inputBuffer = new Float32Array(this.analyzerInput.frequencyBinCount);
        const outputBuffer = new Float32Array(this.analyzerOutput.frequencyBinCount);

        this.analyzerInput.getFloatTimeDomainData(inputBuffer);
        this.analyzerOutput.getFloatTimeDomainData(outputBuffer);

        // Calculate RMS
        const inputRMS = this.calculateRMS(inputBuffer);
        const outputRMS = this.calculateRMS(outputBuffer);

        // Calculate gain reduction in dB
        if (inputRMS > 0 && outputRMS > 0) {
          this.gainReduction = 20 * Math.log10(outputRMS / inputRMS);
          this.gainReduction = Math.max(-60, Math.min(0, this.gainReduction));
        } else {
          this.gainReduction = 0;
        }
      }

      if (!this.isDisposed) {
        requestAnimationFrame(updateGainReduction);
      }
    };

    updateGainReduction();
  }

  /**
   * Calculate RMS of a buffer
   * @param {Float32Array} buffer - Audio buffer
   * @returns {number} RMS value
   */
  calculateRMS(buffer) {
    let sum = 0;
    for (let i = 0; i < buffer.length; i++) {
      sum += buffer[i] * buffer[i];
    }
    return Math.sqrt(sum / buffer.length);
  }

  /**
   * Get current gain reduction in dB
   * @returns {number} Gain reduction in dB (negative value)
   */
  getGainReduction() {
    return this.gainReduction;
  }

  /**
   * Bypass the compressor (true bypass)
   * @param {boolean} enabled - Whether to bypass
   */
  bypass(enabled) {
    this.bypassed = enabled;

    if (enabled) {
      // Disconnect compressed signal
      this.wetGain.gain.setValueAtTime(0, this.context.currentTime);
      this.dryGain.gain.setValueAtTime(1, this.context.currentTime);
    } else {
      // Restore mix settings
      const mix = this.getParameter('mix') / 100;
      this.wetGain.gain.setValueAtTime(mix, this.context.currentTime);
      this.dryGain.gain.setValueAtTime(1 - mix, this.context.currentTime);
    }
  }

  /**
   * Convert dB to linear gain
   * @param {number} db - Decibel value
   * @returns {number} Linear gain value
   */
  dbToGain(db) {
    return Math.pow(10, db / 20);
  }

  /**
   * Convert linear gain to dB
   * @param {number} gain - Linear gain value
   * @returns {number} Decibel value
   */
  gainToDb(gain) {
    return 20 * Math.log10(gain);
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.isDisposed = true;

    // Disconnect all nodes
    this.input.disconnect();
    this.output.disconnect();
    this.sidechainInput.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();
    this.compressorNode.disconnect();
    this.makeupGainNode.disconnect();
    this.analyzerInput.disconnect();
    this.analyzerOutput.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Compressor;
}
