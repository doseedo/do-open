/**
 * EchoPlugin - Multi-Tap Delay/Echo (AudioWorklet)
 *
 * AudioWorklet-based echo effect with:
 * - Multiple delay taps with exponential decay
 * - Stereo delays with independent timing
 * - Feedback control with filtering
 * - Highpass/lowpass filters in feedback path
 * - Stereo offset for width control
 *
 * @author Agent 6 - Reverb Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class EchoPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'reverb',
      description: 'Multi-tap delay with stereo processing and filtered feedback',
      name: 'Echo',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      delayTimeL: 250,
      delayTimeR: 375,
      feedback: 40,
      numTaps: 4,
      tapDecay: 0.7,
      highpass: 20,
      lowpass: 20000,
      stereoOffset: 0,
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
        new URL('../worklets/echo-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'echo-processor',
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
      this.registerParameter('delayTimeL', this.workletNode.parameters.get('delayTimeL'), {
        min: 0,
        max: 2000,
        default: this.defaults.delayTimeL,
        unit: 'ms',
        label: 'Delay Time L',
        type: 'continuous'
      });

      this.registerParameter('delayTimeR', this.workletNode.parameters.get('delayTimeR'), {
        min: 0,
        max: 2000,
        default: this.defaults.delayTimeR,
        unit: 'ms',
        label: 'Delay Time R',
        type: 'continuous'
      });

      this.registerParameter('feedback', this.workletNode.parameters.get('feedback'), {
        min: 0,
        max: 100,
        default: this.defaults.feedback,
        unit: '%',
        label: 'Feedback',
        type: 'continuous'
      });

      this.registerParameter('numTaps', this.workletNode.parameters.get('numTaps'), {
        min: 1,
        max: 8,
        default: this.defaults.numTaps,
        unit: '',
        label: 'Number of Taps',
        type: 'discrete'
      });

      this.registerParameter('tapDecay', this.workletNode.parameters.get('tapDecay'), {
        min: 0.1,
        max: 1.0,
        default: this.defaults.tapDecay,
        unit: '',
        label: 'Tap Decay',
        type: 'continuous'
      });

      this.registerParameter('highpass', this.workletNode.parameters.get('highpass'), {
        min: 20,
        max: 1000,
        default: this.defaults.highpass,
        unit: 'Hz',
        label: 'Highpass',
        type: 'continuous'
      });

      this.registerParameter('lowpass', this.workletNode.parameters.get('lowpass'), {
        min: 1000,
        max: 20000,
        default: this.defaults.lowpass,
        unit: 'Hz',
        label: 'Lowpass',
        type: 'continuous'
      });

      this.registerParameter('stereoOffset', this.workletNode.parameters.get('stereoOffset'), {
        min: -50,
        max: 50,
        default: this.defaults.stereoOffset,
        unit: 'ms',
        label: 'Stereo Offset',
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
      this.setParameter('delayTimeL', this.defaults.delayTimeL);
      this.setParameter('delayTimeR', this.defaults.delayTimeR);
      this.setParameter('feedback', this.defaults.feedback);
      this.setParameter('numTaps', this.defaults.numTaps);
      this.setParameter('tapDecay', this.defaults.tapDecay);
      this.setParameter('highpass', this.defaults.highpass);
      this.setParameter('lowpass', this.defaults.lowpass);
      this.setParameter('stereoOffset', this.defaults.stereoOffset);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize EchoPlugin:', error);
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
      new URL('../worklets/echo-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'echo-processor');

    // Copy current parameter values
    workletNode.parameters.get('delayTimeL').value = this.getParameter('delayTimeL').param.value;
    workletNode.parameters.get('delayTimeR').value = this.getParameter('delayTimeR').param.value;
    workletNode.parameters.get('feedback').value = this.getParameter('feedback').param.value;
    workletNode.parameters.get('numTaps').value = this.getParameter('numTaps').param.value;
    workletNode.parameters.get('tapDecay').value = this.getParameter('tapDecay').param.value;
    workletNode.parameters.get('highpass').value = this.getParameter('highpass').param.value;
    workletNode.parameters.get('lowpass').value = this.getParameter('lowpass').param.value;
    workletNode.parameters.get('stereoOffset').value = this.getParameter('stereoOffset').param.value;
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

export default EchoPlugin;
