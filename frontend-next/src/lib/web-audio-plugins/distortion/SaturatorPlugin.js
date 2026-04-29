/**
 * SaturatorPlugin - Multi-Mode Saturation
 *
 * AudioWorklet-based saturator with:
 * - Multiple saturation algorithms (warm, digital, analog, clip, foldback, sine-fold)
 * - Harmonic emphasis with color filter
 * - DC offset removal
 * - Depth control for saturation intensity
 * - Dry/wet mix control
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class SaturatorPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'distortion',
      description: 'Multi-mode saturation from subtle warmth to heavy distortion',
      name: 'Saturator',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      drive: 0,
      type: 0, // 0=warm, 1=digital, 2=analog, 3=clip, 4=foldback, 5=sine-fold
      color: 0,
      depth: 100,
      dcFilter: 1, // 1=enabled
      output: 0,
      mix: 100
    };

    // Saturation type mapping
    this.saturationTypes = ['warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'];
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
        new URL('../worklets/saturator-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'saturator-processor',
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
      this.registerParameter('drive', this.workletNode.parameters.get('drive'), {
        min: 0,
        max: 100,
        default: this.defaults.drive,
        unit: '%',
        label: 'Drive',
        type: 'continuous'
      });

      this.registerParameter('type', this.workletNode.parameters.get('type'), {
        min: 0,
        max: 5,
        default: this.defaults.type,
        unit: '',
        label: 'Type',
        type: 'discrete'
      });

      this.registerParameter('color', this.workletNode.parameters.get('color'), {
        min: 0,
        max: 100,
        default: this.defaults.color,
        unit: '%',
        label: 'Color',
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

      this.registerParameter('dcFilter', this.workletNode.parameters.get('dcFilter'), {
        min: 0,
        max: 1,
        default: this.defaults.dcFilter,
        unit: '',
        label: 'DC Filter',
        type: 'discrete'
      });

      this.registerParameter('output', this.workletNode.parameters.get('output'), {
        min: -24,
        max: 24,
        default: this.defaults.output,
        unit: 'dB',
        label: 'Output',
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
      this.setParameter('drive', this.defaults.drive);
      this.setParameter('type', this.defaults.type);
      this.setParameter('color', this.defaults.color);
      this.setParameter('depth', this.defaults.depth);
      this.setParameter('dcFilter', this.defaults.dcFilter);
      this.setParameter('output', this.defaults.output);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize SaturatorPlugin:', error);
      throw error;
    }
  }

  /**
   * Set saturation type by name
   * @param {string} type - Saturation type: 'warm', 'digital', 'analog', 'clip', 'foldback', 'sine-fold'
   */
  setSaturationType(type) {
    const index = this.saturationTypes.indexOf(type);
    if (index >= 0) {
      this.setParameter('type', index);
    } else {
      console.warn(`Invalid saturation type: ${type}. Using 'warm'.`);
      this.setParameter('type', 0);
    }
  }

  /**
   * Get current saturation type name
   * @returns {string} Saturation type name
   */
  getSaturationTypeName() {
    const index = Math.floor(this.getParameter('type').param.value);
    return this.saturationTypes[index] || 'warm';
  }

  /**
   * Enable/disable DC filter
   * @param {boolean} enabled - Enable DC filter
   */
  setDCFilter(enabled) {
    this.setParameter('dcFilter', enabled ? 1 : 0);
  }

  /**
   * Get DC filter state
   * @returns {boolean} True if DC filter is enabled
   */
  getDCFilterEnabled() {
    return this.getParameter('dcFilter').param.value > 0.5;
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
      new URL('../worklets/saturator-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'saturator-processor');

    // Copy current parameter values
    workletNode.parameters.get('drive').value = this.getParameter('drive').param.value;
    workletNode.parameters.get('type').value = this.getParameter('type').param.value;
    workletNode.parameters.get('color').value = this.getParameter('color').param.value;
    workletNode.parameters.get('depth').value = this.getParameter('depth').param.value;
    workletNode.parameters.get('dcFilter').value = this.getParameter('dcFilter').param.value;
    workletNode.parameters.get('output').value = this.getParameter('output').param.value;
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

export default SaturatorPlugin;
