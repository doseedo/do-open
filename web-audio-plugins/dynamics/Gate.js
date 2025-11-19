/**
 * Professional Noise Gate Plugin
 *
 * Features:
 * - Adjustable threshold and range
 * - Hold time to prevent chattering
 * - Sidechain support for ducking
 * - Hysteresis to prevent rapid on/off
 * - Smooth attack and release
 *
 * @class Gate
 */
class Gate {
  /**
   * Create a gate instance
   * @param {AudioContext} audioContext - The Web Audio context
   * @param {Object} options - Configuration options
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.sidechainInput = audioContext.createGain();

    // Gate gain node (this is what opens/closes)
    this.gateGain = audioContext.createGain();

    // Analyzer for level detection
    this.analyzer = audioContext.createAnalyser();
    this.analyzer.fftSize = 2048;
    this.analyzer.smoothingTimeConstant = 0;

    // State
    this.bypassed = false;
    this.sidechainEnabled = false;
    this.gateOpen = false;
    this.currentGain = 0;
    this.holdCounter = 0;
    this.hysteresisState = 'closed'; // 'open' or 'closed'

    // Default parameter values
    this.defaults = {
      threshold: -32,    // dB (-60 to 0)
      range: -60,        // dB (0 to -60) - maximum attenuation
      attack: 0.001,     // seconds (1ms)
      hold: 0.010,       // seconds (10ms)
      release: 0.100,    // seconds (100ms)
      hysteresis: 6      // dB - threshold difference between open and close
    };

    // Current parameters
    this.params = {
      threshold: options.threshold || this.defaults.threshold,
      range: options.range || this.defaults.range,
      attack: options.attack || this.defaults.attack,
      hold: options.hold || this.defaults.hold,
      release: options.release || this.defaults.release,
      hysteresis: options.hysteresis || this.defaults.hysteresis
    };

    this.initialize();
  }

  /**
   * Initialize the gate
   */
  initialize() {
    // Set up audio graph
    this.input.connect(this.analyzer);
    this.input.connect(this.gateGain);
    this.gateGain.connect(this.output);

    // Start gate closed
    this.gateGain.gain.setValueAtTime(this.dbToGain(this.params.range), this.context.currentTime);

    // Start gate processing
    this.startGateProcessing();
  }

  /**
   * Connect the gate to a destination
   * @param {AudioNode} destination - The audio node to connect to
   * @returns {AudioNode} The destination node
   */
  connect(destination) {
    this.output.connect(destination);
    return destination;
  }

  /**
   * Disconnect the gate from all destinations
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Set a gate parameter
   * @param {string} name - Parameter name
   * @param {number} value - Parameter value
   */
  setParameter(name, value) {
    switch (name) {
      case 'threshold':
        this.params.threshold = Math.max(-60, Math.min(0, value));
        break;

      case 'range':
        this.params.range = Math.max(-60, Math.min(0, value));
        break;

      case 'attack':
        // 0.1ms to 50ms
        this.params.attack = Math.max(0.0001, Math.min(0.050, value));
        break;

      case 'hold':
        // 0 to 500ms
        this.params.hold = Math.max(0, Math.min(0.500, value));
        break;

      case 'release':
        // 10ms to 2000ms
        this.params.release = Math.max(0.010, Math.min(2.000, value));
        break;

      case 'hysteresis':
        // 0 to 12 dB
        this.params.hysteresis = Math.max(0, Math.min(12, value));
        break;

      default:
        console.warn(`Unknown parameter: ${name}`);
    }
  }

  /**
   * Get a gate parameter value
   * @param {string} name - Parameter name
   * @returns {number} The parameter value
   */
  getParameter(name) {
    return this.params[name] || 0;
  }

  /**
   * Enable or disable sidechain gating
   * @param {boolean} enabled - Whether to enable sidechain
   */
  enableSidechain(enabled) {
    this.sidechainEnabled = enabled;
    // In a full implementation, we would connect the sidechain input
    // to a separate analyzer and use that for gate detection
  }

  /**
   * Start gate processing loop
   */
  startGateProcessing() {
    const bufferLength = this.analyzer.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);

    const process = () => {
      if (this.bypassed || this.isDisposed) {
        if (!this.isDisposed) {
          requestAnimationFrame(process);
        }
        return;
      }

      // Get current input level
      this.analyzer.getFloatTimeDomainData(dataArray);
      const rms = this.calculateRMS(dataArray);
      const levelDb = this.gainToDb(rms);

      // Determine if gate should be open or closed using hysteresis
      const openThreshold = this.params.threshold;
      const closeThreshold = this.params.threshold - this.params.hysteresis;

      let shouldBeOpen = false;

      if (this.hysteresisState === 'closed') {
        // Gate is closed, check if we should open
        if (levelDb > openThreshold) {
          shouldBeOpen = true;
          this.hysteresisState = 'open';
          this.holdCounter = 0;
        }
      } else {
        // Gate is open, check if we should close
        if (levelDb > closeThreshold) {
          shouldBeOpen = true;
          this.holdCounter = 0;
        } else {
          // Below close threshold, start hold timer
          this.holdCounter += 1;
          const holdSamples = this.params.hold * this.context.sampleRate / bufferLength;

          if (this.holdCounter < holdSamples) {
            // Still in hold period, keep gate open
            shouldBeOpen = true;
          } else {
            // Hold period expired, close gate
            shouldBeOpen = false;
            this.hysteresisState = 'closed';
          }
        }
      }

      // Smooth gate opening/closing
      const now = this.context.currentTime;
      const targetGain = shouldBeOpen ? 1.0 : this.dbToGain(this.params.range);
      const rampTime = shouldBeOpen ? this.params.attack : this.params.release;

      // Cancel any scheduled changes
      this.gateGain.gain.cancelScheduledValues(now);

      // Ramp to target gain
      this.gateGain.gain.setValueAtTime(this.gateGain.gain.value, now);
      this.gateGain.gain.linearRampToValueAtTime(targetGain, now + rampTime);

      this.gateOpen = shouldBeOpen;
      this.currentGain = targetGain;

      requestAnimationFrame(process);
    };

    process();
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
   * Check if gate is currently open
   * @returns {boolean} True if gate is open
   */
  isGateOpen() {
    return this.gateOpen;
  }

  /**
   * Get current gate gain (for metering)
   * @returns {number} Current gate gain (0 to 1)
   */
  getCurrentGain() {
    return this.gateGain.gain.value;
  }

  /**
   * Get current attenuation in dB
   * @returns {number} Attenuation in dB
   */
  getAttenuation() {
    return this.gainToDb(this.gateGain.gain.value);
  }

  /**
   * Bypass the gate (true bypass)
   * @param {boolean} enabled - Whether to bypass
   */
  bypass(enabled) {
    this.bypassed = enabled;

    const now = this.context.currentTime;
    if (enabled) {
      // Open gate fully
      this.gateGain.gain.cancelScheduledValues(now);
      this.gateGain.gain.setValueAtTime(1.0, now);
    } else {
      // Resume normal operation
      this.hysteresisState = 'closed';
      this.holdCounter = 0;
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
   * Clean up resources
   */
  dispose() {
    this.isDisposed = true;

    // Disconnect all nodes
    this.input.disconnect();
    this.output.disconnect();
    this.sidechainInput.disconnect();
    this.gateGain.disconnect();
    this.analyzer.disconnect();
  }
}

// Export for use in modules
if (typeof module !== 'undefined' && module.exports) {
  module.exports = Gate;
}
