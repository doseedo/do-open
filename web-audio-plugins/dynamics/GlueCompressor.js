/**
 * Professional Glue Compressor Plugin
 *
 * Vintage-style bus compressor inspired by classic VCA designs.
 * Features stepped controls and a unique "gluing" character perfect for mix buses.
 *
 * Features:
 * - Vintage VCA-style compression character
 * - Stepped attack and release times
 * - Auto-release mode
 * - Soft knee compression
 * - Peak mode switch
 * - Dry/wet mix control
 *
 * @class GlueCompressor
 */
class GlueCompressor {
  /**
   * Create a glue compressor instance
   * @param {AudioContext} audioContext - The Web Audio context
   * @param {Object} options - Configuration options
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Dry/wet signal paths
    this.wetGain = audioContext.createGain();
    this.dryGain = audioContext.createGain();

    // Internal compressor node with soft knee
    this.compressorNode = audioContext.createDynamicsCompressor();

    // Makeup gain node
    this.makeupGainNode = audioContext.createGain();

    // Vintage character filter (subtle high-frequency rolloff)
    this.characterFilter = audioContext.createBiquadFilter();
    this.characterFilter.type = 'lowpass';
    this.characterFilter.frequency.value = 18000;
    this.characterFilter.Q.value = 0.707;

    // Analyzers for gain reduction metering
    this.analyzerInput = audioContext.createAnalyser();
    this.analyzerOutput = audioContext.createAnalyser();
    this.analyzerInput.fftSize = 2048;
    this.analyzerOutput.fftSize = 2048;

    // State
    this.gainReduction = 0;
    this.bypassed = false;
    this.peakMode = false;
    this.autoRelease = false;

    // Stepped parameter values (authentic to vintage units)
    this.attackSteps = [0.0001, 0.0003, 0.001, 0.003, 0.010, 0.030]; // 0.1, 0.3, 1, 3, 10, 30 ms
    this.releaseSteps = [0.1, 0.3, 0.6, 1.2]; // 0.1, 0.3, 0.6, 1.2 seconds
    this.ratioSteps = [2, 4, 10, 1000]; // 2:1, 4:1, 10:1, ∞:1

    // Default parameter values
    this.defaults = {
      threshold: -12,    // dB (-40 to 0)
      ratioIndex: 1,     // Index into ratioSteps (default 4:1)
      attackIndex: 3,    // Index into attackSteps (default 3ms)
      releaseIndex: 2,   // Index into releaseSteps (default 0.6s)
      makeupGain: 0,     // dB (0 to 20)
      dryWet: 100        // 0 to 100%
    };

    // Current parameters
    this.params = {
      threshold: options.threshold || this.defaults.threshold,
      ratioIndex: options.ratioIndex !== undefined ? options.ratioIndex : this.defaults.ratioIndex,
      attackIndex: options.attackIndex !== undefined ? options.attackIndex : this.defaults.attackIndex,
      releaseIndex: options.releaseIndex !== undefined ? options.releaseIndex : this.defaults.releaseIndex,
      makeupGain: options.makeupGain || this.defaults.makeupGain,
      dryWet: options.dryWet !== undefined ? options.dryWet : this.defaults.dryWet
    };

    // Auto-release state
    this.lastPeakTime = 0;
    this.autoReleaseCoeff = 0.6;

    this.initialize();
  }

  /**
   * Initialize the glue compressor
   */
  initialize() {
    // Set up audio graph
    // Input splits to wet and dry paths
    this.input.connect(this.analyzerInput);
    this.input.connect(this.dryGain);
    this.input.connect(this.compressorNode);

    // Wet path: compressor → character filter → makeup gain → wet gain → output
    this.compressorNode.connect(this.characterFilter);
    this.characterFilter.connect(this.makeupGainNode);
    this.makeupGainNode.connect(this.analyzerOutput);
    this.makeupGainNode.connect(this.wetGain);
    this.wetGain.connect(this.output);

    // Dry path: dry gain → output
    this.dryGain.connect(this.output);

    // Set soft knee for vintage character
    this.compressorNode.knee.setValueAtTime(6, this.context.currentTime);

    // Apply parameters
    this.applyParameters();

    // Start gain reduction monitoring
    this.startGainReductionMonitoring();
  }

  /**
   * Apply all parameters to the compressor
   */
  applyParameters() {
    const now = this.context.currentTime;

    // Threshold
    this.compressorNode.threshold.setValueAtTime(this.params.threshold, now);

    // Ratio (stepped)
    const ratio = this.ratioSteps[this.params.ratioIndex];
    this.compressorNode.ratio.setValueAtTime(ratio, now);

    // Attack (stepped)
    const attack = this.attackSteps[this.params.attackIndex];
    this.compressorNode.attack.setValueAtTime(attack, now);

    // Release (stepped or auto)
    if (!this.autoRelease) {
      const release = this.releaseSteps[this.params.releaseIndex];
      this.compressorNode.release.setValueAtTime(release, now);
    }

    // Makeup gain
    const makeupGainValue = this.dbToGain(this.params.makeupGain);
    this.makeupGainNode.gain.setValueAtTime(makeupGainValue, now);

    // Dry/wet mix
    const wetAmount = this.params.dryWet / 100;
    const dryAmount = 1 - wetAmount;
    this.wetGain.gain.setValueAtTime(wetAmount, now);
    this.dryGain.gain.setValueAtTime(dryAmount, now);
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
   */
  setParameter(name, value) {
    const now = this.context.currentTime;

    switch (name) {
      case 'threshold':
        // -40 to 0 dB
        this.params.threshold = Math.max(-40, Math.min(0, value));
        this.compressorNode.threshold.setValueAtTime(this.params.threshold, now);
        break;

      case 'ratioIndex':
        // 0 to 3 (maps to 2:1, 4:1, 10:1, ∞:1)
        this.params.ratioIndex = Math.max(0, Math.min(3, Math.floor(value)));
        const ratio = this.ratioSteps[this.params.ratioIndex];
        this.compressorNode.ratio.setValueAtTime(ratio, now);
        break;

      case 'attackIndex':
        // 0 to 5
        this.params.attackIndex = Math.max(0, Math.min(5, Math.floor(value)));
        const attack = this.attackSteps[this.params.attackIndex];
        this.compressorNode.attack.setValueAtTime(attack, now);
        break;

      case 'releaseIndex':
        // 0 to 3
        this.params.releaseIndex = Math.max(0, Math.min(3, Math.floor(value)));
        if (!this.autoRelease) {
          const release = this.releaseSteps[this.params.releaseIndex];
          this.compressorNode.release.setValueAtTime(release, now);
        }
        break;

      case 'makeupGain':
        // 0 to 20 dB
        this.params.makeupGain = Math.max(0, Math.min(20, value));
        const makeupGainValue = this.dbToGain(this.params.makeupGain);
        this.makeupGainNode.gain.setValueAtTime(makeupGainValue, now);
        break;

      case 'dryWet':
        // 0 to 100%
        this.params.dryWet = Math.max(0, Math.min(100, value));
        const wetAmount = this.params.dryWet / 100;
        const dryAmount = 1 - wetAmount;
        this.wetGain.gain.setValueAtTime(wetAmount, now);
        this.dryGain.gain.setValueAtTime(dryAmount, now);
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
        return this.params.threshold;
      case 'ratioIndex':
        return this.params.ratioIndex;
      case 'ratio':
        return this.ratioSteps[this.params.ratioIndex];
      case 'attackIndex':
        return this.params.attackIndex;
      case 'attack':
        return this.attackSteps[this.params.attackIndex];
      case 'releaseIndex':
        return this.params.releaseIndex;
      case 'release':
        return this.autoRelease ? 'auto' : this.releaseSteps[this.params.releaseIndex];
      case 'makeupGain':
        return this.params.makeupGain;
      case 'dryWet':
        return this.params.dryWet;
      default:
        console.warn(`Unknown parameter: ${name}`);
        return 0;
    }
  }

  /**
   * Set peak mode (faster detection)
   * @param {boolean} enabled - Whether to enable peak mode
   */
  setPeakMode(enabled) {
    this.peakMode = enabled;
    // Note: Web Audio API doesn't expose detection mode
    // This would need custom implementation for true peak mode
  }

  /**
   * Set auto-release mode
   * @param {boolean} enabled - Whether to enable auto-release
   */
  setAutoRelease(enabled) {
    this.autoRelease = enabled;

    if (enabled) {
      // Start auto-release monitoring
      this.updateAutoRelease();
    } else {
      // Restore manual release setting
      const release = this.releaseSteps[this.params.releaseIndex];
      this.compressorNode.release.setValueAtTime(release, this.context.currentTime);
    }
  }

  /**
   * Update auto-release based on input signal
   */
  updateAutoRelease() {
    if (!this.autoRelease || this.bypassed || this.isDisposed) {
      return;
    }

    const bufferLength = this.analyzerInput.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);
    this.analyzerInput.getFloatTimeDomainData(dataArray);

    // Find peak
    let peak = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const abs = Math.abs(dataArray[i]);
      if (abs > peak) peak = abs;
    }

    const peakDb = this.gainToDb(peak);

    // If signal is above threshold, we're in attack phase
    if (peakDb > this.params.threshold) {
      this.lastPeakTime = this.context.currentTime;
    }

    // Calculate adaptive release time based on how long since last peak
    const timeSincePeak = this.context.currentTime - this.lastPeakTime;
    const baseRelease = 0.1; // 100ms minimum
    const maxRelease = 1.2; // 1.2s maximum
    const adaptiveRelease = Math.min(maxRelease, baseRelease + timeSincePeak * this.autoReleaseCoeff);

    this.compressorNode.release.setValueAtTime(adaptiveRelease, this.context.currentTime);

    // Continue monitoring
    setTimeout(() => this.updateAutoRelease(), 50); // Update every 50ms
  }

  /**
   * Start monitoring gain reduction
   */
  startGainReductionMonitoring() {
    const updateGainReduction = () => {
      if (!this.bypassed && !this.isDisposed) {
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
      const wetAmount = this.params.dryWet / 100;
      const dryAmount = 1 - wetAmount;
      this.wetGain.gain.setValueAtTime(wetAmount, this.context.currentTime);
      this.dryGain.gain.setValueAtTime(dryAmount, this.context.currentTime);
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
    if (gain <= 0) return -Infinity;
    return 20 * Math.log10(gain);
  }

  /**
   * Get available attack time options (in ms)
   * @returns {Array} Array of attack times in milliseconds
   */
  getAttackOptions() {
    return this.attackSteps.map(t => t * 1000);
  }

  /**
   * Get available release time options (in seconds)
   * @returns {Array} Array of release times in seconds
   */
  getReleaseOptions() {
    return [...this.releaseSteps, 'Auto'];
  }

  /**
   * Get available ratio options
   * @returns {Array} Array of ratio strings
   */
  getRatioOptions() {
    return ['2:1', '4:1', '10:1', '∞:1'];
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.isDisposed = true;

    // Disconnect all nodes
    this.input.disconnect();
    this.output.disconnect();
    this.wetGain.disconnect();
    this.dryGain.disconnect();
    this.compressorNode.disconnect();
    this.makeupGainNode.disconnect();
    this.characterFilter.disconnect();
    this.analyzerInput.disconnect();
    this.analyzerOutput.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = GlueCompressor;
}
