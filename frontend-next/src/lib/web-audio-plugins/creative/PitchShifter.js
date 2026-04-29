/**
 * Pitch Shifter Plugin
 * Time-domain pitch shifting with formant preservation
 *
 * Uses overlap-add time-domain method for real-time pitch shifting.
 * Shifts pitch up or down while maintaining time duration.
 *
 * @example
 * const audioContext = new AudioContext();
 * const pitchShifter = new PitchShifter(audioContext);
 *
 * // Connect audio source
 * source.connect(pitchShifter.input);
 * pitchShifter.connect(audioContext.destination);
 *
 * // Shift up by 7 semitones (perfect fifth)
 * pitchShifter.setPitchShift(7);
 * pitchShifter.setMix(1.0);
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

class PitchShifter {
  /**
   * Create a Pitch Shifter
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.pitchShift - Pitch shift in semitones (-12 to +12)
   * @param {number} options.windowSize - Window size in seconds (0.05 to 0.2)
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
      pitchShift: options.pitchShift || 0,
      windowSize: options.windowSize || 0.1,
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
      await this.context.audioWorklet.addModule(`${basePath}/worklets/dsp-utils.js`);
      await this.context.audioWorklet.addModule(`${basePath}/worklets/pitch-shifter-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'pitch-shifter-processor',
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
      console.error('Error setting up Pitch Shifter worklet:', error);
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
        if (script.src && script.src.includes('PitchShifter.js')) {
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
   * Set pitch shift amount
   * @param {number} semitones - Pitch shift in semitones (-12 to +12)
   *   Positive = shift up, Negative = shift down
   *   12 semitones = 1 octave
   */
  setPitchShift(semitones) {
    this.parameters.pitchShift = Math.max(-12, Math.min(12, semitones));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'pitchShift',
        value: this.parameters.pitchShift
      });
    }
  }

  /**
   * Set window size (affects quality vs latency trade-off)
   * @param {number} seconds - Window size in seconds (0.05 to 0.2)
   *   Larger = better quality, more latency
   *   Smaller = lower latency, more artifacts
   */
  setWindowSize(seconds) {
    this.parameters.windowSize = Math.max(0.05, Math.min(0.2, seconds));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'windowSize',
        value: this.parameters.windowSize
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
  module.exports = PitchShifter;
}
