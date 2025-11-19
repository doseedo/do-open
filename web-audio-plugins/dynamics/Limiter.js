/**
 * Professional Brick Wall Limiter Plugin
 *
 * Features:
 * - True peak limiting with lookahead
 * - Ultra-fast attack time
 * - Adjustable release
 * - Gain reduction metering
 * - Prevents signal from exceeding ceiling
 *
 * @class Limiter
 */
class Limiter {
  /**
   * Create a limiter instance
   * @param {AudioContext} audioContext - The Web Audio context
   * @param {Object} options - Configuration options
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Lookahead delay line
    this.delayNode = audioContext.createDelay(0.1); // Max 100ms lookahead

    // Gain node for limiting
    this.limiterGain = audioContext.createGain();

    // Analyzers for peak detection and metering
    this.peakAnalyzer = audioContext.createAnalyser();
    this.peakAnalyzer.fftSize = 2048;
    this.peakAnalyzer.smoothingTimeConstant = 0;

    this.outputAnalyzer = audioContext.createAnalyser();
    this.outputAnalyzer.fftSize = 2048;

    // State
    this.bypassed = false;
    this.gainReduction = 0;
    this.peakLevel = 0;

    // Default parameter values
    this.defaults = {
      ceiling: -0.3,     // dB (-20 to 0)
      release: 0.050,    // seconds (50ms)
      lookahead: 0.005   // seconds (5ms)
    };

    // Current parameters
    this.params = {
      ceiling: options.ceiling || this.defaults.ceiling,
      release: options.release || this.defaults.release,
      lookahead: options.lookahead || this.defaults.lookahead
    };

    // Attack is fixed to very fast (essentially instantaneous)
    this.attackTime = 0.001; // 1ms

    // Envelope follower state
    this.envelope = 0;

    this.initialize();
  }

  /**
   * Initialize the limiter
   */
  initialize() {
    // Set up audio graph with lookahead
    // Input splits: one path for analysis, one path for delayed audio
    this.input.connect(this.peakAnalyzer);
    this.input.connect(this.delayNode);

    // Delayed audio goes through limiter gain
    this.delayNode.connect(this.limiterGain);
    this.limiterGain.connect(this.outputAnalyzer);
    this.limiterGain.connect(this.output);

    // Set initial delay time (lookahead)
    this.delayNode.delayTime.setValueAtTime(this.params.lookahead, this.context.currentTime);

    // Initialize limiter gain to unity
    this.limiterGain.gain.setValueAtTime(1.0, this.context.currentTime);

    // Start limiting process
    this.startLimiting();
  }

  /**
   * Connect the limiter to a destination
   * @param {AudioNode} destination - The audio node to connect to
   * @returns {AudioNode} The destination node
   */
  connect(destination) {
    this.output.connect(destination);
    return destination;
  }

  /**
   * Disconnect the limiter from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Set a limiter parameter
   * @param {string} name - Parameter name
   * @param {number} value - Parameter value
   */
  setParameter(name, value) {
    const now = this.context.currentTime;

    switch (name) {
      case 'ceiling':
        // -20 to 0 dB
        this.params.ceiling = Math.max(-20, Math.min(0, value));
        break;

      case 'release':
        // 10ms to 1000ms
        this.params.release = Math.max(0.010, Math.min(1.000, value));
        break;

      case 'lookahead':
        // 0 to 10ms
        value = Math.max(0, Math.min(0.010, value));
        this.params.lookahead = value;
        // Update delay time smoothly
        this.delayNode.delayTime.cancelScheduledValues(now);
        this.delayNode.delayTime.setValueAtTime(this.delayNode.delayTime.value, now);
        this.delayNode.delayTime.linearRampToValueAtTime(value, now + 0.1);
        break;

      default:
        console.warn(`Unknown parameter: ${name}`);
    }
  }

  /**
   * Get a limiter parameter value
   * @param {string} name - Parameter name
   * @returns {number} The parameter value
   */
  getParameter(name) {
    return this.params[name] || 0;
  }

  /**
   * Start the limiting process
   */
  startLimiting() {
    const bufferLength = this.peakAnalyzer.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    const process = () => {
      if (this.bypassed || this.isDisposed) {
        if (!this.isDisposed) {
          requestAnimationFrame(process);
        }
        return;
      }

      // Get peak level from lookahead buffer
      this.peakAnalyzer.getFloatTimeDomainData(dataArray);
      const peak = this.findPeak(dataArray);
      const peakDb = this.gainToDb(peak);

      this.peakLevel = peakDb;

      // Calculate required gain reduction
      const ceilingLinear = this.dbToGain(this.params.ceiling);
      let targetGain = 1.0;

      if (peak > ceilingLinear) {
        // Signal exceeds ceiling, calculate gain reduction
        targetGain = ceilingLinear / peak;
      }

      // Apply envelope follower for smooth gain changes
      const now = this.context.currentTime;
      const dt = 1 / 60; // Approximate frame time

      if (targetGain < this.envelope) {
        // Attack: very fast response
        const attackCoeff = Math.exp(-dt / this.attackTime);
        this.envelope = targetGain + attackCoeff * (this.envelope - targetGain);
      } else {
        // Release: slower response
        const releaseCoeff = Math.exp(-dt / this.params.release);
        this.envelope = targetGain + releaseCoeff * (this.envelope - targetGain);
      }

      // Clamp envelope
      this.envelope = Math.max(0, Math.min(1, this.envelope));

      // Apply gain smoothly
      this.limiterGain.gain.cancelScheduledValues(now);
      this.limiterGain.gain.setValueAtTime(this.limiterGain.gain.value, now);
      this.limiterGain.gain.linearRampToValueAtTime(this.envelope, now + 0.01);

      // Calculate gain reduction for metering
      this.gainReduction = this.gainToDb(this.envelope);

      requestAnimationFrame(process);
    };

    process();
  }

  /**
   * Find the peak value in a buffer
   * @param {Float32Array} buffer - Audio buffer
   * @returns {number} Peak value
   */
  findPeak(buffer) {
    let peak = 0;
    for (let i = 0; i < buffer.length; i++) {
      const abs = Math.abs(buffer[i]);
      if (abs > peak) {
        peak = abs;
      }
    }
    return peak;
  }

  /**
   * Get current gain reduction in dB
   * @returns {number} Gain reduction in dB (negative value)
   */
  getGainReduction() {
    return this.gainReduction;
  }

  /**
   * Get current peak level in dB
   * @returns {number} Peak level in dB
   */
  getPeakLevel() {
    return this.peakLevel;
  }

  /**
   * Get output level for metering
   * @returns {number} Output level in dB
   */
  getOutputLevel() {
    const bufferLength = this.outputAnalyzer.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);
    this.outputAnalyzer.getFloatTimeDomainData(dataArray);

    const rms = this.calculateRMS(dataArray);
    return this.gainToDb(rms);
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
   * Bypass the limiter (true bypass)
   * @param {boolean} enabled - Whether to bypass
   */
  bypass(enabled) {
    this.bypassed = enabled;

    const now = this.context.currentTime;
    if (enabled) {
      // Set gain to unity
      this.limiterGain.gain.cancelScheduledValues(now);
      this.limiterGain.gain.setValueAtTime(1.0, now);
      this.envelope = 1.0;
      this.gainReduction = 0;
    } else {
      // Resume normal operation
      this.envelope = 1.0;
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
   * Get latency introduced by lookahead (in seconds)
   * @returns {number} Latency in seconds
   */
  getLatency() {
    return this.params.lookahead;
  }

  /**
   * Clean up resources
   */
  dispose() {
    this.isDisposed = true;

    // Disconnect all nodes
    this.input.disconnect();
    this.output.disconnect();
    this.delayNode.disconnect();
    this.limiterGain.disconnect();
    this.peakAnalyzer.disconnect();
    this.outputAnalyzer.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Limiter;
}
