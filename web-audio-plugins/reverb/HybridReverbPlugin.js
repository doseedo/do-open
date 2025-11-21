/**
 * HybridReverbPlugin - Hybrid Reverb (AudioWorklet)
 *
 * AudioWorklet-based hybrid reverb with:
 * - Early reflections (multi-tap delays simulating room geometry)
 * - Algorithmic tail (comb + allpass filters)
 * - Independent level control for early/tail sections
 * - Pre-delay control
 * - Frequency-dependent damping
 *
 * Note: This is a simplified version. A full hybrid reverb would use
 * convolution for early reflections, but this uses delay-based simulation.
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class HybridReverbPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'reverb',
      description: 'Hybrid reverb combining early reflections with algorithmic tail',
      name: 'HybridReverb',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      preDelay: 0,
      decayTime: 2.0,
      earlyLevel: -6,
      tailLevel: -6,
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
        new URL('../worklets/hybrid-reverb-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'hybrid-reverb-processor',
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

      this.registerParameter('earlyLevel', this.workletNode.parameters.get('earlyLevel'), {
        min: -60,
        max: 0,
        default: this.defaults.earlyLevel,
        unit: 'dB',
        label: 'Early Level',
        type: 'continuous'
      });

      this.registerParameter('tailLevel', this.workletNode.parameters.get('tailLevel'), {
        min: -60,
        max: 0,
        default: this.defaults.tailLevel,
        unit: 'dB',
        label: 'Tail Level',
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
      this.setParameter('earlyLevel', this.defaults.earlyLevel);
      this.setParameter('tailLevel', this.defaults.tailLevel);
      this.setParameter('damping', this.defaults.damping);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize HybridReverbPlugin:', error);
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
      new URL('../worklets/hybrid-reverb-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'hybrid-reverb-processor');

    // Copy current parameter values
    workletNode.parameters.get('preDelay').value = this.getParameter('preDelay').param.value;
    workletNode.parameters.get('decayTime').value = this.getParameter('decayTime').param.value;
    workletNode.parameters.get('earlyLevel').value = this.getParameter('earlyLevel').param.value;
    workletNode.parameters.get('tailLevel').value = this.getParameter('tailLevel').param.value;
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

export default HybridReverbPlugin;
