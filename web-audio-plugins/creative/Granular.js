/**
 * Granular Synthesis Plugin
 * Creates evolving textures by manipulating micro-segments of audio
 *
 * Granular synthesis breaks audio into tiny grains (10-100ms) and
 * recombines them with various transformations. Excellent for pads,
 * atmospheres, and experimental sound design.
 *
 * @example
 * const audioContext = new AudioContext();
 * const granular = new Granular(audioContext);
 *
 * // Connect audio source
 * source.connect(granular.input);
 * granular.connect(audioContext.destination);
 *
 * // Configure granular parameters
 * granular.setGrainSize(0.05); // 50ms grains
 * granular.setDensity(20); // 20 grains per second
 * granular.setRandomness(0.7); // High randomization
 * granular.setPitch(1.5); // Pitch up grains
 *
 * @author Agent 9 (Creative Effects)
 * @version 1.0.0
 */

class Granular {
  /**
   * Create a Granular Synthesizer
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.grainSize - Grain size in seconds (0.01 to 0.5)
   * @param {number} options.density - Grains per second (1 to 100)
   * @param {number} options.randomness - Randomization amount (0 to 1)
   * @param {number} options.pitch - Pitch/playback rate (0.25 to 4.0)
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
      grainSize: options.grainSize || 0.05,
      density: options.density || 10,
      randomness: options.randomness !== undefined ? options.randomness : 0.5,
      pitch: options.pitch || 1.0,
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
      await this.context.audioWorklet.addModule(`${basePath}/worklets/granular-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'granular-processor',
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
      console.error('Error setting up Granular worklet:', error);
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
        if (script.src && script.src.includes('Granular.js')) {
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
   * Set grain size
   * @param {number} seconds - Grain size in seconds (0.01 to 0.5)
   *   Smaller = sharper, more textural
   *   Larger = smoother, more melodic
   */
  setGrainSize(seconds) {
    this.parameters.grainSize = Math.max(0.01, Math.min(0.5, seconds));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'grainSize',
        value: this.parameters.grainSize
      });
    }
  }

  /**
   * Set grain density
   * @param {number} grainsPerSecond - Grains per second (1 to 100)
   *   Lower = sparse, rhythmic
   *   Higher = dense, continuous
   */
  setDensity(grainsPerSecond) {
    this.parameters.density = Math.max(1, Math.min(100, grainsPerSecond));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'density',
        value: this.parameters.density
      });
    }
  }

  /**
   * Set randomness
   * @param {number} amount - Randomization amount (0 to 1)
   *   0 = deterministic, mechanical
   *   1 = chaotic, evolving
   */
  setRandomness(amount) {
    this.parameters.randomness = Math.max(0, Math.min(1, amount));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'randomness',
        value: this.parameters.randomness
      });
    }
  }

  /**
   * Set pitch/playback rate
   * @param {number} pitch - Playback rate (0.25 to 4.0)
   *   < 1.0 = lower pitch
   *   1.0 = original pitch
   *   > 1.0 = higher pitch
   */
  setPitch(pitch) {
    this.parameters.pitch = Math.max(0.25, Math.min(4.0, pitch));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'pitch',
        value: this.parameters.pitch
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
  module.exports = Granular;
}
