/**
 * ChorusPlugin - Professional Chorus Effect
 *
 * AudioWorklet-based chorus with:
 * - Multiple voices (1-8) with phase-offset LFOs
 * - DelayLine-based modulation
 * - Stereo spread control
 * - Feedback for richer texture
 * - Configurable LFO rate and depth
 * - Multiple waveform types
 *
 * @author Agent 4 - Modulation Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class ChorusPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'modulation',
      description: 'Multi-voice chorus effect with stereo spread and feedback',
      name: 'Chorus',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      rate: 0.5,
      depth: 50,
      voices: 4,
      spread: 50,
      feedback: 0,
      mix: 50,
      delay: 20,
      waveform: 0 // 0=sine, 1=triangle, 2=square, 3=sawtooth
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
        new URL('../worklets/chorus-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'chorus-processor',
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

      // Register parameters
      this.registerParameter('rate', this.workletNode.parameters.get('rate'), {
        min: 0.01,
        max: 10,
        default: this.defaults.rate,
        unit: 'Hz',
        label: 'Rate',
        type: 'continuous'
      });

      this.registerParameter('depth', this.workletNode.parameters.get('depth'), {
        min: 0,
        max: 100,
        default: this.defaults.depth,
        unit: '%',
        label: 'Depth',
        type: 'continuous'
      });

      this.registerParameter('voices', this.workletNode.parameters.get('voices'), {
        min: 1,
        max: 8,
        default: this.defaults.voices,
        unit: '',
        label: 'Voices',
        type: 'discrete'
      });

      this.registerParameter('spread', this.workletNode.parameters.get('spread'), {
        min: 0,
        max: 100,
        default: this.defaults.spread,
        unit: '%',
        label: 'Spread',
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

      this.registerParameter('mix', this.workletNode.parameters.get('mix'), {
        min: 0,
        max: 100,
        default: this.defaults.mix,
        unit: '%',
        label: 'Mix',
        type: 'continuous'
      });

      this.registerParameter('delay', this.workletNode.parameters.get('delay'), {
        min: 5,
        max: 50,
        default: this.defaults.delay,
        unit: 'ms',
        label: 'Delay',
        type: 'continuous'
      });

      this.registerParameter('waveform', this.workletNode.parameters.get('waveform'), {
        min: 0,
        max: 3,
        default: this.defaults.waveform,
        unit: '',
        label: 'Waveform',
        type: 'discrete'
      });

      // Set default values
      this.setParameter('rate', this.defaults.rate);
      this.setParameter('depth', this.defaults.depth);
      this.setParameter('voices', this.defaults.voices);
      this.setParameter('spread', this.defaults.spread);
      this.setParameter('feedback', this.defaults.feedback);
      this.setParameter('mix', this.defaults.mix);
      this.setParameter('delay', this.defaults.delay);
      this.setParameter('waveform', this.defaults.waveform);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize ChorusPlugin:', error);
      throw error;
    }
  }

  /**
   * Set waveform type by name
   * @param {string} type - 'sine', 'triangle', 'square', 'sawtooth'
   */
  setWaveformType(type) {
    const waveformMap = {
      'sine': 0,
      'triangle': 1,
      'square': 2,
      'sawtooth': 3
    };

    const value = waveformMap[type];
    if (value !== undefined) {
      this.setParameter('waveform', value);
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
      new URL('../worklets/chorus-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'chorus-processor');

    // Copy current parameter values
    workletNode.parameters.get('rate').value = this.getParameter('rate').param.value;
    workletNode.parameters.get('depth').value = this.getParameter('depth').param.value;
    workletNode.parameters.get('voices').value = this.getParameter('voices').param.value;
    workletNode.parameters.get('spread').value = this.getParameter('spread').param.value;
    workletNode.parameters.get('feedback').value = this.getParameter('feedback').param.value;
    workletNode.parameters.get('mix').value = this.getParameter('mix').param.value;
    workletNode.parameters.get('delay').value = this.getParameter('delay').param.value;
    workletNode.parameters.get('waveform').value = this.getParameter('waveform').param.value;

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

export default ChorusPlugin;
