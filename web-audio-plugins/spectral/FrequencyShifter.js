/**
 * Frequency Shifter
 * Linear frequency shifting using single-sideband modulation
 *
 * @example
 * const audioContext = new AudioContext();
 * const shifter = new FrequencyShifter(audioContext);
 *
 * // Connect audio source
 * source.connect(shifter.input);
 * shifter.connect(audioContext.destination);
 *
 * // Shift up by 100 Hz
 * shifter.setFrequency(100);
 *
 * // Use wide mode for stereo spread
 * shifter.setMode('wide');
 * shifter.setWideAmount(50);
 */

class FrequencyShifter {
  /**
   * Create a Frequency Shifter
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.frequency - Shift frequency in Hz (-5000 to +5000)
   * @param {number} options.fine - Fine tuning in Hz (-100 to +100)
   * @param {string} options.mode - Shift mode ('up', 'down', 'wide')
   * @param {number} options.wideAmount - Stereo spread for wide mode (0 to 1)
   * @param {number} options.drive - Harmonic saturation (0 to 1)
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
      frequency: options.frequency || 0,
      fine: options.fine || 0,
      mode: options.mode || 'up',
      wideAmount: options.wideAmount !== undefined ? options.wideAmount : 0.5,
      drive: options.drive || 0,
      mix: options.mix !== undefined ? options.mix : 1.0
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
      await this.context.audioWorklet.addModule(`${basePath}/worklets/frequency-shifter-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'frequency-shifter-processor',
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2]
        }
      );

      // Connect nodes
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Apply initial parameters
      this.applyParameters();

      this.isReady = true;
    } catch (error) {
      console.error('Error setting up Frequency Shifter worklet:', error);
      throw error;
    }
  }

  /**
   * Get base path for worklet files
   * @private
   */
  getBasePath() {
    if (typeof document !== 'undefined') {
      const scripts = document.getElementsByTagName('script');
      for (let script of scripts) {
        if (script.src && script.src.includes('FrequencyShifter.js')) {
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
   * Set shift frequency
   * @param {number} hz - Frequency shift in Hz (-5000 to +5000)
   *   Positive = shift up, Negative = shift down
   */
  setFrequency(hz) {
    this.parameters.frequency = Math.max(-5000, Math.min(5000, hz));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'frequency',
        value: this.parameters.frequency
      });
    }
  }

  /**
   * Set fine tuning
   * @param {number} hz - Fine tuning in Hz (-100 to +100)
   */
  setFine(hz) {
    this.parameters.fine = Math.max(-100, Math.min(100, hz));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'fine',
        value: this.parameters.fine
      });
    }
  }

  /**
   * Set shift mode
   * @param {string} mode - Mode ('up', 'down', 'wide')
   *   'up' = upper sideband (shift up)
   *   'down' = lower sideband (shift down)
   *   'wide' = stereo spread
   */
  setMode(mode) {
    const validModes = ['up', 'down', 'wide'];
    if (validModes.includes(mode)) {
      this.parameters.mode = mode;
      if (this.workletNode) {
        this.workletNode.port.postMessage({
          type: 'mode',
          value: this.parameters.mode
        });
      }
    }
  }

  /**
   * Set wide amount (for wide mode)
   * @param {number} amount - Stereo spread (0 to 100%)
   */
  setWideAmount(amount) {
    this.parameters.wideAmount = Math.max(0, Math.min(100, amount)) / 100;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'wideAmount',
        value: this.parameters.wideAmount
      });
    }
  }

  /**
   * Set harmonic drive/saturation
   * @param {number} drive - Drive amount (0 to 100%)
   */
  setDrive(drive) {
    this.parameters.drive = Math.max(0, Math.min(100, drive)) / 100;
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'drive',
        value: this.parameters.drive
      });
    }
  }

  /**
   * Set wet/dry mix
   * @param {number} mix - Mix (0 to 100%)
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
  module.exports = FrequencyShifter;
}
