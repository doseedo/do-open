/**
 * OverdrivePlugin - Tube-Style Soft Clipping
 *
 * AudioWorklet-based overdrive with:
 * - Soft clipping with multiple curve types (tanh, atan, softClip)
 * - Asymmetric distortion for even harmonics
 * - Tone stack (post-distortion lowshelf EQ)
 * - Auto gain compensation
 * - Dry/wet mix control
 *
 * @author Agent 5 - Distortion Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class OverdrivePlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'distortion',
      description: 'Tube-style soft clipping for warm, musical distortion',
      name: 'Overdrive',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      drive: 30,
      tone: 50,
      bias: 0,
      curveType: 0, // 0=tanh, 1=atan, 2=softClip
      output: 0,
      mix: 100
    };

    // Curve type mapping
    this.curveTypes = ['tanh', 'atan', 'softClip'];
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
        new URL('../worklets/overdrive-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'overdrive-processor',
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
        min: 0,
        max: 100,
        default: this.defaults.tone,
        unit: '%',
        label: 'Tone',
        type: 'continuous'
      });

      this.registerParameter('bias', this.workletNode.parameters.get('bias'), {
        min: -100,
        max: 100,
        default: this.defaults.bias,
        unit: '%',
        label: 'Bias',
        type: 'continuous'
      });

      this.registerParameter('curveType', this.workletNode.parameters.get('curveType'), {
        min: 0,
        max: 2,
        default: this.defaults.curveType,
        unit: '',
        label: 'Curve Type',
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
      this.setParameter('bias', this.defaults.bias);
      this.setParameter('curveType', this.defaults.curveType);
      this.setParameter('output', this.defaults.output);
      this.setParameter('mix', this.defaults.mix);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize OverdrivePlugin:', error);
      throw error;
    }
  }

  /**
   * Set curve type by name
   * @param {string} type - Curve type: 'tanh', 'atan', 'softClip'
   */
  setCurveType(type) {
    const index = this.curveTypes.indexOf(type);
    if (index >= 0) {
      this.setParameter('curveType', index);
    } else {
      console.warn(`Invalid curve type: ${type}. Using 'tanh'.`);
      this.setParameter('curveType', 0);
    }
  }

  /**
   * Get current curve type name
   * @returns {string} Curve type name
   */
  getCurveTypeName() {
    const index = Math.floor(this.getParameter('curveType').param.value);
    return this.curveTypes[index] || 'tanh';
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
      new URL('../worklets/overdrive-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'overdrive-processor');

    // Copy current parameter values
    workletNode.parameters.get('drive').value = this.getParameter('drive').param.value;
    workletNode.parameters.get('tone').value = this.getParameter('tone').param.value;
    workletNode.parameters.get('bias').value = this.getParameter('bias').param.value;
    workletNode.parameters.get('curveType').value = this.getParameter('curveType').param.value;
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

export default OverdrivePlugin;
