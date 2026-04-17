/**
 * DistortionPlugin - Hard Clipping Distortion
 *
 * AudioWorklet-based distortion with:
 * - Multiple waveshaping algorithms (hard, soft, asymmetric, foldback)
 * - Pre/post filtering with tone control
 * - High gain capability
 * - DC blocking
 * - Dry/wet mix control
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class DistortionPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'distortion',
      description: 'Hard clipping distortion with multiple waveshaping algorithms',
      name: 'Distortion',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      drive: 50,
      tone: 1000,
      toneWidth: 1,
      clipType: 0, // 0=hard, 1=soft, 2=asymmetric, 3=foldback
      filterPosition: 0, // 0=post, 1=pre
      output: 0,
      mix: 100
    };

    // Clip type mapping
    this.clipTypes = ['hard', 'soft', 'asymmetric', 'foldback'];
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
        new URL('../worklets/distortion-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'distortion-processor',
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

      this.registerParameter('tone', this.workletNode.parameters.get('tone'), {
        min: 20,
        max: 20000,
        default: this.defaults.tone,
        unit: 'Hz',
        label: 'Tone',
        type: 'continuous'
      });

      this.registerParameter('toneWidth', this.workletNode.parameters.get('toneWidth'), {
        min: 0.1,
        max: 10,
        default: this.defaults.toneWidth,
        unit: 'Q',
        label: 'Tone Width',
        type: 'continuous'
      });

      this.registerParameter('clipType', this.workletNode.parameters.get('clipType'), {
        min: 0,
        max: 3,
        default: this.defaults.clipType,
        unit: '',
        label: 'Clip Type',
        type: 'discrete'
      });

      this.registerParameter('filterPosition', this.workletNode.parameters.get('filterPosition'), {
        min: 0,
        max: 1,
        default: this.defaults.filterPosition,
        unit: '',
        label: 'Filter Position',
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
      this.setParameter('tone', this.defaults.tone);
      this.setParameter('toneWidth', this.defaults.toneWidth);
      this.setParameter('clipType', this.defaults.clipType);
      this.setParameter('filterPosition', this.defaults.filterPosition);
      this.setParameter('output', this.defaults.output);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize DistortionPlugin:', error);
      throw error;
    }
  }

  /**
   * Set clip type by name
   * @param {string} type - Clip type: 'hard', 'soft', 'asymmetric', 'foldback'
   */
  setClipType(type) {
    const index = this.clipTypes.indexOf(type);
    if (index >= 0) {
      this.setParameter('clipType', index);
    } else {
      console.warn(`Invalid clip type: ${type}. Using 'hard'.`);
      this.setParameter('clipType', 0);
    }
  }

  /**
   * Get current clip type name
   * @returns {string} Clip type name
   */
  getClipTypeName() {
    const index = Math.floor(this.getParameter('clipType').param.value);
    return this.clipTypes[index] || 'hard';
  }

  /**
   * Set filter position
   * @param {string} position - 'pre' or 'post'
   */
  setFilterPosition(position) {
    if (position === 'pre') {
      this.setParameter('filterPosition', 1);
    } else if (position === 'post') {
      this.setParameter('filterPosition', 0);
    } else {
      console.warn(`Invalid filter position: ${position}. Using 'post'.`);
      this.setParameter('filterPosition', 0);
    }
  }

  /**
   * Get current filter position
   * @returns {string} 'pre' or 'post'
   */
  getFilterPosition() {
    return this.getParameter('filterPosition').param.value === 1 ? 'pre' : 'post';
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
      new URL('../worklets/distortion-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'distortion-processor');

    // Copy current parameter values
    workletNode.parameters.get('drive').value = this.getParameter('drive').param.value;
    workletNode.parameters.get('tone').value = this.getParameter('tone').param.value;
    workletNode.parameters.get('toneWidth').value = this.getParameter('toneWidth').param.value;
    workletNode.parameters.get('clipType').value = this.getParameter('clipType').param.value;
    workletNode.parameters.get('filterPosition').value = this.getParameter('filterPosition').param.value;
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

export default DistortionPlugin;
