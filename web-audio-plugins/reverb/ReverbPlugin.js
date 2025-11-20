/**
 * ReverbPlugin - Algorithmic Reverb (AudioWorklet)
 *
 * AudioWorklet-based Schroeder reverb with:
 * - Parallel comb filters for reverb tail
 * - Series allpass filters for diffusion
 * - Frequency-dependent damping
 * - Configurable room size and decay
 * - Pre-delay control
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class ReverbPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'reverb',
      description: 'Algorithmic reverb using Schroeder/Freeverb architecture with comb and allpass filters',
      name: 'Reverb',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      preDelay: 0,
      decayTime: 2.0,
      size: 50,
      diffusion: 70,
      damping: 50,
      mix: 30
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
        new URL('../worklets/reverb-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'reverb-processor',
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

      // Listen for messages from worklet (if any)
      this.workletNode.port.onmessage = (event) => {
        // Handle any messages from the worklet processor
        if (event.data.type === 'meter') {
          // Could handle metering data here
        }
      };

      // Register parameters
      this.registerParameter('preDelay', this.workletNode.parameters.get('preDelay'), {
        min: 0,
        max: 250,
        default: this.defaults.preDelay,
        unit: 'ms',
        label: 'Pre-Delay',
        type: 'continuous'
      });

      this.registerParameter('decayTime', this.workletNode.parameters.get('decayTime'), {
        min: 0.1,
        max: 20,
        default: this.defaults.decayTime,
        unit: 's',
        label: 'Decay Time',
        type: 'continuous'
      });

      this.registerParameter('size', this.workletNode.parameters.get('size'), {
        min: 0,
        max: 100,
        default: this.defaults.size,
        unit: '%',
        label: 'Room Size',
        type: 'continuous'
      });

      this.registerParameter('diffusion', this.workletNode.parameters.get('diffusion'), {
        min: 0,
        max: 100,
        default: this.defaults.diffusion,
        unit: '%',
        label: 'Diffusion',
        type: 'continuous'
      });

      this.registerParameter('damping', this.workletNode.parameters.get('damping'), {
        min: 0,
        max: 100,
        default: this.defaults.damping,
        unit: '%',
        label: 'Damping',
        type: 'continuous'
      });

      this.registerParameter('mix', this.workletNode.parameters.get('mix'), {
        min: 0,
        max: 100,
        default: this.defaults.mix,
        unit: '%',
        label: 'Mix',
        type: 'continuous'
      });

      // Set default values
      this.setParameter('preDelay', this.defaults.preDelay);
      this.setParameter('decayTime', this.defaults.decayTime);
      this.setParameter('size', this.defaults.size);
      this.setParameter('diffusion', this.defaults.diffusion);
      this.setParameter('damping', this.defaults.damping);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize ReverbPlugin:', error);
      throw error;
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
      new URL('../worklets/reverb-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'reverb-processor');

    // Copy current parameter values
    workletNode.parameters.get('preDelay').value = this.getParameter('preDelay').param.value;
    workletNode.parameters.get('decayTime').value = this.getParameter('decayTime').param.value;
    workletNode.parameters.get('size').value = this.getParameter('size').param.value;
    workletNode.parameters.get('diffusion').value = this.getParameter('diffusion').param.value;
    workletNode.parameters.get('damping').value = this.getParameter('damping').param.value;
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

export default ReverbPlugin;
