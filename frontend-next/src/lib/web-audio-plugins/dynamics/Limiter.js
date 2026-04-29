/**
 * LimiterPlugin - Peak Limiter
 *
 * AudioWorklet-based limiter with:
 * - Hard limiting (infinite ratio)
 * - Fast attack time
 * - Transparent limiting
 * - Automatic makeup gain option
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class LimiterPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'dynamics',
      description: 'Peak limiter for mastering and preventing clipping',
      name: 'Limiter',
      ...options
    });

    this.workletNode = null;
    this.gainReduction = 0;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      threshold: -1, // -1 dB for mastering
      attack: 0.001, // 1ms - very fast
      release: 0.100, // 100ms
      makeupGain: 0
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
        new URL('../worklets/limiter-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'limiter-processor',
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
        min: -24,
        max: 0,
        default: this.defaults.threshold,
        unit: 'dB',
        label: 'Threshold',
        type: 'continuous'
      });

      this.registerParameter('attack', this.workletNode.parameters.get('attack'), {
        min: 0.0001,
        max: 0.050,
        default: this.defaults.attack,
        unit: 's',
        label: 'Attack',
        type: 'continuous'
      });

      this.registerParameter('release', this.workletNode.parameters.get('release'), {
        min: 0.010,
        max: 1.0,
        default: this.defaults.release,
        unit: 's',
        label: 'Release',
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

      // Set default values
      this.setParameter('threshold', this.defaults.threshold);
      this.setParameter('attack', this.defaults.attack);
      this.setParameter('release', this.defaults.release);
      this.setParameter('makeupGain', this.defaults.makeupGain);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize LimiterPlugin:', error);
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
   * Enable auto makeup gain (brings level up to threshold)
   * @param {boolean} enabled - Enable auto makeup gain
   */
  setAutoMakeup(enabled) {
    if (enabled) {
      const threshold = this.getParameter('threshold').param.value;
      // Auto makeup gain brings the limited signal close to 0 dBFS
      const autoGain = Math.abs(threshold);
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
      new URL('../worklets/limiter-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'limiter-processor');

    // Copy current parameter values
    workletNode.parameters.get('threshold').value = this.getParameter('threshold').param.value;
    workletNode.parameters.get('attack').value = this.getParameter('attack').param.value;
    workletNode.parameters.get('release').value = this.getParameter('release').param.value;
    workletNode.parameters.get('makeupGain').value = this.getParameter('makeupGain').param.value;

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

export default LimiterPlugin;
