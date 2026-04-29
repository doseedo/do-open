/**
 * Frequency Shifter Plugin
 * Linear frequency shifting using single-sideband modulation
 *
 * Unlike pitch shifting (which multiplies frequencies) or ring modulation
 * (which creates sum and difference), frequency shifting ADDS a constant
 * to all frequencies, creating unique metallic and dissonant effects.
 *
 * @example
 * const audioContext = new AudioContext();
 * const shifter = new FrequencyShifter(audioContext);
 *
 * // Connect audio source
 * source.connect(shifter.input);
 * shifter.connect(audioContext.destination);
 *
 * // Shift all frequencies up by 100 Hz
 * shifter.setShift(100);
 * shifter.setMix(0.5); // 50% wet
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

class FrequencyShifter {
  /**
   * Create a Frequency Shifter
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.shift - Frequency shift in Hz (-5000 to +5000)
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
      shift: options.shift || 0,
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

      // Add worklet module
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
    return './creative';
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
   * Set frequency shift amount
   * @param {number} hz - Frequency shift in Hz (-5000 to +5000)
   *   Positive = shift up
   *   Negative = shift down
   *
   * Examples:
   *   100 Hz: Subtle metallic character
   *   500 Hz: Strong dissonance
   *   -100 Hz: Shift down (darkening effect)
   */
  setShift(hz) {
    this.parameters.shift = Math.max(-5000, Math.min(5000, hz));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'shift',
        value: this.parameters.shift
      });
    }
  }

  /**
   * Set wet/dry mix
   * @param {number} mix - Mix (0 to 1)
   */
  setMix(mix) {
    this.parameters.mix = Math.max(0, Math.min(1, mix));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'mix',
        value: this.parameters.mix
      });
    }
  }

  /**
   * Get current parameters
   * @returns {Object} Current parameter values
   */
  getParams() {
    return { ...this.parameters };
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
