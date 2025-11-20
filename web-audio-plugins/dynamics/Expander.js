/**
 * ExpanderPlugin - Downward Expander
 *
 * AudioWorklet-based expander with:
 * - Downward expansion (increases dynamic range)
 * - Configurable ratio
 * - Attack/release times
 * - More subtle than a gate
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class ExpanderPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'dynamics',
      description: 'Downward expander for increasing dynamic range subtly',
      name: 'Expander',
      ...options
    });

    this.workletNode = null;
    this.expansion = 0;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      threshold: -40, // -40 dB
      ratio: 2, // 1:2 expansion
      attack: 0.010, // 10ms
      release: 0.100 // 100ms
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
        new URL('../worklets/expander-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'expander-processor',
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

      // Listen for expansion messages
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'expansion') {
          this.expansion = event.data.value;
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
        max: 10,
        default: this.defaults.ratio,
        unit: ':1',
        label: 'Ratio',
        type: 'continuous'
      });

      this.registerParameter('attack', this.workletNode.parameters.get('attack'), {
        min: 0.0001,
        max: 0.100,
        default: this.defaults.attack,
        unit: 's',
        label: 'Attack',
        type: 'continuous'
      });

      this.registerParameter('release', this.workletNode.parameters.get('release'), {
        min: 0.010,
        max: 2.0,
        default: this.defaults.release,
        unit: 's',
        label: 'Release',
        type: 'continuous'
      });

      // Set default values
      this.setParameter('threshold', this.defaults.threshold);
      this.setParameter('ratio', this.defaults.ratio);
      this.setParameter('attack', this.defaults.attack);
      this.setParameter('release', this.defaults.release);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize ExpanderPlugin:', error);
      throw error;
    }
  }

  /**
   * Get current expansion amount
   * @returns {number} Expansion in dB (negative value)
   */
  getExpansion() {
    return this.expansion;
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
      new URL('../worklets/expander-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'expander-processor');

    // Copy current parameter values
    workletNode.parameters.get('threshold').value = this.getParameter('threshold').param.value;
    workletNode.parameters.get('ratio').value = this.getParameter('ratio').param.value;
    workletNode.parameters.get('attack').value = this.getParameter('attack').param.value;
    workletNode.parameters.get('release').value = this.getParameter('release').param.value;

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

export default ExpanderPlugin;
