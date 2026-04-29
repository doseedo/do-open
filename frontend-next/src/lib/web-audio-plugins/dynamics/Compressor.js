/**
 * CompressorPlugin - Professional Dynamics Compressor
 *
 * AudioWorklet-based compressor with:
 * - Soft knee compression
 * - Configurable attack/release
 * - Makeup gain
 * - Parallel compression (wet/dry mix)
 * - Real-time gain reduction metering
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class CompressorPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'dynamics',
      description: 'Professional dynamics compressor with soft knee and parallel compression',
      name: 'Compressor',
      ...options
    });

    this.workletNode = null;
    this.gainReduction = 0;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      threshold: -24,
      ratio: 4,
      attack: 0.010,
      release: 0.100,
      knee: 0,
      makeupGain: 0,
      mix: 1.0
    };
  }

  /**
   * Initialize the plugin (loads AudioWorklet)
   */
  async initialize() {
    if (this.initialized) {
      return;
    }

    try {
      // Load the AudioWorklet module
      await this.audioContext.audioWorklet.addModule(
        new URL('../worklets/compressor-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'compressor-processor',
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2],
        }
      );

      // Connect input -> worklet -> output
      this.input.disconnect();
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Listen for gain reduction messages
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'gainReduction') {
          this.gainReduction = event.data.value;
        }
      };

      // Register parameters
      this.registerParameter('threshold', this.workletNode.parameters.get('threshold'), {
        min: -60,
        max: 0,
        default: this.defaults.threshold,
        unit: 'dB',
        label: 'Threshold',
        type: 'continuous'
      });

      this.registerParameter('ratio', this.workletNode.parameters.get('ratio'), {
        min: 1,
        max: 20,
        default: this.defaults.ratio,
        unit: ':1',
        label: 'Ratio',
        type: 'continuous'
      });

      this.registerParameter('attack', this.workletNode.parameters.get('attack'), {
        min: 0.0001,
        max: 0.5,
        default: this.defaults.attack,
        unit: 's',
        label: 'Attack',
        type: 'continuous'
      });

      this.registerParameter('release', this.workletNode.parameters.get('release'), {
        min: 0.001,
        max: 2.0,
        default: this.defaults.release,
        unit: 's',
        label: 'Release',
        type: 'continuous'
      });

      this.registerParameter('knee', this.workletNode.parameters.get('knee'), {
        min: 0,
        max: 12,
        default: this.defaults.knee,
        unit: 'dB',
        label: 'Knee',
        type: 'continuous'
      });

      this.registerParameter('makeupGain', this.workletNode.parameters.get('makeupGain'), {
        min: 0,
        max: 24,
        default: this.defaults.makeupGain,
        unit: 'dB',
        label: 'Makeup Gain',
        type: 'continuous'
      });

      this.registerParameter('mix', this.workletNode.parameters.get('mix'), {
        min: 0,
        max: 1.0,
        default: this.defaults.mix,
        unit: '%',
        label: 'Mix',
        type: 'continuous'
      });

      // Set default values
      this.setParameter('threshold', this.defaults.threshold);
      this.setParameter('ratio', this.defaults.ratio);
      this.setParameter('attack', this.defaults.attack);
      this.setParameter('release', this.defaults.release);
      this.setParameter('knee', this.defaults.knee);
      this.setParameter('makeupGain', this.defaults.makeupGain);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize CompressorPlugin:', error);
      throw error;
    }
  }

  /**
   * Get current gain reduction for metering
   * @returns {number} Gain reduction in dB (negative value)
   */
  getGainReduction() {
    return this.gainReduction;
  }

  /**
   * Enable auto makeup gain calculation
   * @param {boolean} enabled - Enable auto makeup gain
   */
  setAutoMakeup(enabled) {
    if (enabled) {
      const threshold = this.getParameter('threshold').param.value;
      const ratio = this.getParameter('ratio').param.value;

      // Approximate makeup gain: makeupGain ≈ |threshold * (1 - 1/ratio) / 2|
      const autoGain = Math.abs(threshold * (1 - 1/ratio) / 2);
      this.setParameter('makeupGain', autoGain);
    }
  }

  /**
   * Process audio offline (for bouncing/rendering)
   * @param {AudioBuffer} inputBuffer - Input audio buffer
   * @returns {Promise<AudioBuffer>} Processed audio buffer
   */
  async processOffline(inputBuffer) {
    if (!this.initialized) {
      await this.initialize();
    }

    const offlineContext = new OfflineAudioContext(
      inputBuffer.numberOfChannels,
      inputBuffer.length,
      inputBuffer.sampleRate
    );

    // Load worklet in offline context
    await offlineContext.audioWorklet.addModule(
      new URL('../worklets/compressor-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'compressor-processor');

    // Copy current parameter values
    workletNode.parameters.get('threshold').value = this.getParameter('threshold').param.value;
    workletNode.parameters.get('ratio').value = this.getParameter('ratio').param.value;
    workletNode.parameters.get('attack').value = this.getParameter('attack').param.value;
    workletNode.parameters.get('release').value = this.getParameter('release').param.value;
    workletNode.parameters.get('knee').value = this.getParameter('knee').param.value;
    workletNode.parameters.get('makeupGain').value = this.getParameter('makeupGain').param.value;
    workletNode.parameters.get('mix').value = this.getParameter('mix').param.value;

    // Connect graph
    source.connect(workletNode);
    workletNode.connect(offlineContext.destination);

    // Render
    source.start(0);
    const renderedBuffer = await offlineContext.startRendering();

    return renderedBuffer;
  }

  /**
   * Override dispose to clean up worklet
   */
  dispose() {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode.port.close();
      this.workletNode = null;
    }

    super.dispose();
  }

  /**
   * Check if plugin uses AudioWorklet
   * @returns {boolean}
   */
  usesAudioWorklet() {
    return true;
  }
}

export default CompressorPlugin;
