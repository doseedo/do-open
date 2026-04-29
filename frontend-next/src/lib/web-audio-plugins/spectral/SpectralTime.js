/**
 * Spectral Time
 * Time stretching, pitch shifting, and spectral freezing using phase vocoder
 *
 * @example
 * const audioContext = new AudioContext();
 * const spectralTime = new SpectralTime(audioContext);
 *
 * // Connect audio source
 * source.connect(spectralTime.input);
 * spectralTime.connect(audioContext.destination);
 *
 * // Time stretch to 2x (slower)
 * spectralTime.setStretch(2.0);
 *
 * // Freeze spectrum
 * spectralTime.setFreeze(true);
 *
 * // Pitch shift up 7 semitones
 * spectralTime.setPitchShift(7);
 */

class SpectralTime {
  /**
   * Create a Spectral Time processor
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.stretch - Time stretch factor (0.1 to 4.0)
   * @param {boolean} options.freeze - Spectral freeze enabled
   * @param {number} options.blur - Spectral blur amount (0 to 1)
   * @param {number} options.shift - Pitch shift in semitones (-24 to +24)
   * @param {number} options.formant - Formant shift in semitones (-4 to +4)
   * @param {number} options.residual - Transient preservation (0 to 1)
   * @param {number} options.mix - Wet/dry mix (0 to 1)
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.workletNode = null;

    // Parameters
    this.parameters = {
      stretch: options.stretch || 1.0,
      freeze: options.freeze || false,
      blur: options.blur || 0.0,
      shift: options.shift || 0,
      formant: options.formant || 0,
      residual: options.residual || 0.0,
      mix: options.mix || 1.0
    };

    // Setup state
    this.isReady = false;
    this.setupPromise = this.setupWorklet();
  }

  /**
   * Setup AudioWorklet processor
   * @private
   */
  async setupWorklet() {
    try {
      // Get the base path for worklet files
      const basePath = this.getBasePath();

      // Add worklet modules
      await this.context.audioWorklet.addModule(`${basePath}/worklets/fft-lib.js`);
      await this.context.audioWorklet.addModule(`${basePath}/worklets/spectral-time-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'spectral-time-processor'
      );

      // Connect nodes
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Apply initial parameters
      this.applyParameters();

      this.isReady = true;
    } catch (error) {
      console.error('Error setting up Spectral Time worklet:', error);
      throw error;
    }
  }

  /**
   * Get base path for worklet files
   * @private
   */
  getBasePath() {
    // Try to determine the base path from the script location
    if (typeof document !== 'undefined') {
      const scripts = document.getElementsByTagName('script');
      for (let script of scripts) {
        if (script.src && script.src.includes('SpectralTime.js')) {
          return script.src.substring(0, script.src.lastIndexOf('/'));
        }
      }
    }
    return './spectral';
  }

  /**
   * Apply all parameters to worklet
   * @private
   */
  applyParameters() {
    if (!this.workletNode) return;

    Object.entries(this.parameters).forEach(([key, value]) => {
      this.workletNode.port.postMessage({
        type: key,
        value: value
      });
    });
  }

  /**
   * Wait for the processor to be ready
   * @returns {Promise<void>}
   */
  async ready() {
    await this.setupPromise;
  }

  /**
   * Set time stretch factor
   * @param {number} factor - Stretch factor (0.1 to 4.0)
   *   1.0 = normal speed
   *   < 1.0 = faster (e.g., 0.5 = 2x speed)
   *   > 1.0 = slower (e.g., 2.0 = half speed)
   */
  setStretch(factor) {
    this.parameters.stretch = Math.max(0.1, Math.min(4.0, factor));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'stretch',
        value: this.parameters.stretch
      });
    }
  }

  /**
   * Set spectral freeze
   * @param {boolean} enabled - Enable freeze
   */
  setFreeze(enabled) {
    this.parameters.freeze = enabled;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'freeze',
        value: this.parameters.freeze
      });
    }
  }

  /**
   * Set spectral blur
   * @param {number} amount - Blur amount (0 to 100%)
   */
  setBlur(amount) {
    this.parameters.blur = Math.max(0, Math.min(100, amount)) / 100;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'blur',
        value: this.parameters.blur
      });
    }
  }

  /**
   * Set pitch shift
   * @param {number} semitones - Shift amount (-24 to +24 semitones)
   */
  setPitchShift(semitones) {
    this.parameters.shift = Math.max(-24, Math.min(24, semitones));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'shift',
        value: this.parameters.shift
      });
    }
  }

  /**
   * Set formant shift
   * @param {number} semitones - Formant shift (-4 to +4 semitones)
   */
  setFormantShift(semitones) {
    this.parameters.formant = Math.max(-4, Math.min(4, semitones));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'formant',
        value: this.parameters.formant
      });
    }
  }

  /**
   * Set transient preservation
   * @param {number} amount - Residual amount (0 to 100%)
   */
  setResidual(amount) {
    this.parameters.residual = Math.max(0, Math.min(100, amount)) / 100;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'residual',
        value: this.parameters.residual
      });
    }
  }

  /**
   * Set wet/dry mix
   * @param {number} mix - Mix amount (0 to 100%)
   */
  setMix(mix) {
    this.parameters.mix = Math.max(0, Math.min(100, mix)) / 100;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'mix',
        value: this.parameters.mix
      });
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
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    this.input.disconnect();
    this.output.disconnect();
  }
}

// Export for use in modules or browser
if (typeof module !== 'undefined' && module.exports) {
  module.exports = SpectralTime;
}
