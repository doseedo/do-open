/**
 * Ring Modulator Plugin
 * Creates inharmonic sidebands through amplitude modulation
 *
 * Ring modulation multiplies the input signal with a carrier oscillator,
 * producing sum and difference frequencies (f1±f2)
 *
 * @example
 * const audioContext = new AudioContext();
 * const ringMod = new RingModulator(audioContext);
 *
 * // Connect audio source
 * source.connect(ringMod.input);
 * ringMod.connect(audioContext.destination);
 *
 * // Set carrier frequency
 * ringMod.setFrequency(440);
 * ringMod.setMix(0.5);
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

class RingModulator {
  /**
   * Create a Ring Modulator
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.frequency - Carrier frequency in Hz (0.1 to 20000)
   * @param {string} options.waveform - Carrier waveform ('sine', 'triangle', 'square', 'saw')
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
      frequency: options.frequency || 440,
      waveform: options.waveform || 'sine',
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
      await this.context.audioWorklet.addModule(`${basePath}/worklets/ring-modulator-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'ring-modulator-processor',
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
      console.error('Error setting up Ring Modulator worklet:', error);
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
        if (script.src && script.src.includes('RingModulator.js')) {
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
   * Set carrier frequency
   * @param {number} hz - Frequency in Hz (0.1 to 20000)
   */
  setFrequency(hz) {
    this.parameters.frequency = Math.max(0.1, Math.min(20000, hz));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'frequency',
        value: this.parameters.frequency
      });
    }
  }

  /**
   * Set carrier waveform
   * @param {string} waveform - Waveform type ('sine', 'triangle', 'square', 'saw')
   */
  setWaveform(waveform) {
    const validWaveforms = ['sine', 'triangle', 'square', 'saw'];
    if (validWaveforms.includes(waveform)) {
      this.parameters.waveform = waveform;
      if (this.workletNode) {
        this.workletNode.port.postMessage({
          type: 'waveform',
          value: this.parameters.waveform
        });
      }
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
  module.exports = RingModulator;
}
