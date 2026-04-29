/**
 * TremoloPlugin - Professional Tremolo/Auto-Pan Effect
 *
 * AudioWorklet-based tremolo with:
 * - Amplitude modulation (tremolo mode)
 * - Stereo panning modulation (auto-pan mode)
 * - Stereo mode with 180° phase offset
 * - Multiple LFO waveforms
 * - Configurable rate and depth
 *
 * @author Agent 4 - Modulation Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class TremoloPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'modulation',
      description: 'Amplitude or pan modulation with LFO control',
      name: 'Tremolo',
      ...options
    });

    this.workletNode = null;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      rate: 5,
      depth: 50,
      waveform: 0, // 0=sine, 1=triangle, 2=square, 3=sawtooth
      mode: 0, // 0=tremolo, 1=pan
      stereo: 0 // 0=mono, 1=stereo
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
        new URL('../worklets/tremolo-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'tremolo-processor',
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
        max: 40,
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

      this.registerParameter('waveform', this.workletNode.parameters.get('waveform'), {
        min: 0,
        max: 3,
        default: this.defaults.waveform,
        unit: '',
        label: 'Waveform',
        type: 'discrete'
      });

      this.registerParameter('mode', this.workletNode.parameters.get('mode'), {
        min: 0,
        max: 1,
        default: this.defaults.mode,
        unit: '',
        label: 'Mode',
        type: 'discrete'
      });

      this.registerParameter('stereo', this.workletNode.parameters.get('stereo'), {
        min: 0,
        max: 1,
        default: this.defaults.stereo,
        unit: '',
        label: 'Stereo',
        type: 'discrete'
      });

      // Set default values
      this.setParameter('rate', this.defaults.rate);
      this.setParameter('depth', this.defaults.depth);
      this.setParameter('waveform', this.defaults.waveform);
      this.setParameter('mode', this.defaults.mode);
      this.setParameter('stereo', this.defaults.stereo);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize TremoloPlugin:', error);
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
   * Set mode by name
   * @param {string} mode - 'tremolo' or 'pan'
   */
  setMode(mode) {
    const modeMap = {
      'tremolo': 0,
      'pan': 1
    };

    const value = modeMap[mode];
    if (value !== undefined) {
      this.setParameter('mode', value);
    }
  }

  /**
   * Enable/disable stereo mode
   * @param {boolean} enabled - Enable stereo phase offset
   */
  setStereo(enabled) {
    this.setParameter('stereo', enabled ? 1 : 0);
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
      new URL('../worklets/tremolo-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'tremolo-processor');

    // Copy current parameter values
    workletNode.parameters.get('rate').value = this.getParameter('rate').param.value;
    workletNode.parameters.get('depth').value = this.getParameter('depth').param.value;
    workletNode.parameters.get('waveform').value = this.getParameter('waveform').param.value;
    workletNode.parameters.get('mode').value = this.getParameter('mode').param.value;
    workletNode.parameters.get('stereo').value = this.getParameter('stereo').param.value;

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

export default TremoloPlugin;
