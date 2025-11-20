/**
 * PhaserPlugin - Professional Phaser Effect
 *
 * AudioWorklet-based phaser with:
 * - Cascade of allpass filters (2-12 stages)
 * - LFO-modulated frequency sweep
 * - Feedback for resonance
 * - Frequency spread control
 * - Multiple LFO waveforms
 *
 * @author Agent 4 - Modulation Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class PhaserPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'modulation',
      description: 'Sweeping notch filter phaser with adjustable stages and feedback',
      name: 'Phaser',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      rate: 0.5,
      depth: 50,
      feedback: 0,
      stages: 6,
      frequency: 1000,
      spread: 50,
      waveform: 0, // 0=sine, 1=triangle, 2=square, 3=sawtooth
      mix: 50
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
        new URL('../worklets/phaser-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'phaser-processor',
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

      this.registerParameter('feedback', this.workletNode.parameters.get('feedback'), {
        min: 0,
        max: 100,
        default: this.defaults.feedback,
        unit: '%',
        label: 'Feedback',
        type: 'continuous'
      });

      this.registerParameter('stages', this.workletNode.parameters.get('stages'), {
        min: 2,
        max: 12,
        default: this.defaults.stages,
        unit: '',
        label: 'Stages',
        type: 'discrete'
      });

      this.registerParameter('frequency', this.workletNode.parameters.get('frequency'), {
        min: 200,
        max: 8000,
        default: this.defaults.frequency,
        unit: 'Hz',
        label: 'Frequency',
        type: 'continuous'
      });

      this.registerParameter('spread', this.workletNode.parameters.get('spread'), {
        min: 0,
        max: 100,
        default: this.defaults.spread,
        unit: '%',
        label: 'Spread',
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

      this.registerParameter('mix', this.workletNode.parameters.get('mix'), {
        min: 0,
        max: 100,
        default: this.defaults.mix,
        unit: '%',
        label: 'Mix',
        type: 'continuous'
      });

      // Set default values
      this.setParameter('rate', this.defaults.rate);
      this.setParameter('depth', this.defaults.depth);
      this.setParameter('feedback', this.defaults.feedback);
      this.setParameter('stages', this.defaults.stages);
      this.setParameter('frequency', this.defaults.frequency);
      this.setParameter('spread', this.defaults.spread);
      this.setParameter('waveform', this.defaults.waveform);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize PhaserPlugin:', error);
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
      new URL('../worklets/phaser-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'phaser-processor');

    // Copy current parameter values
    workletNode.parameters.get('rate').value = this.getParameter('rate').param.value;
    workletNode.parameters.get('depth').value = this.getParameter('depth').param.value;
    workletNode.parameters.get('feedback').value = this.getParameter('feedback').param.value;
    workletNode.parameters.get('stages').value = this.getParameter('stages').param.value;
    workletNode.parameters.get('frequency').value = this.getParameter('frequency').param.value;
    workletNode.parameters.get('spread').value = this.getParameter('spread').param.value;
    workletNode.parameters.get('waveform').value = this.getParameter('waveform').param.value;
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

export default PhaserPlugin;
