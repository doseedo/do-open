/**
 * GatePlugin - Noise Gate
 *
 * AudioWorklet-based noise gate with:
 * - Threshold-based gating
 * - Configurable attack/release
 * - Range parameter for attenuation depth
 * - Real-time gate state monitoring
 *
 * @author Agent 1 - Dynamics Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class GatePlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      category: 'dynamics',
      description: 'Noise gate for removing low-level signals and background noise',
      name: 'Gate',
      ...options
    });

    this.workletNode = null;
    this.isGateOpen = false;
    this.initialized = false;

    // Default parameter values
    this.defaults = {
      threshold: -40, // -40 dB
      attack: 0.010, // 10ms
      release: 0.100, // 100ms
      range: -60 // Attenuate by 60 dB when closed
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
        new URL('../worklets/gate-processor.js', import.meta.url).href
      );

      // Create the AudioWorkletNode
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'gate-processor',
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

      // Listen for gate state messages
      this.workletNode.port.onmessage = (event) => {
        if (event.data.type === 'gateState') {
          this.isGateOpen = event.data.value;
        }
      };

      // Register parameters
      this.registerParameter('threshold', this.workletNode.parameters.get('threshold'), {
        min: -60,
        max: 0,
        default: this.defaults.threshold,
        unit: 'dB',
        label: 'Threshold',
        type: 'continuous'
      });

      this.registerParameter('attack', this.workletNode.parameters.get('attack'), {
        min: 0.0001,
        max: 0.100,
        default: this.defaults.attack,
        unit: 's',
        label: 'Attack',
        type: 'continuous'
      });

      this.registerParameter('release', this.workletNode.parameters.get('release'), {
        min: 0.010,
        max: 2.0,
        default: this.defaults.release,
        unit: 's',
        label: 'Release',
        type: 'continuous'
      });

      this.registerParameter('range', this.workletNode.parameters.get('range'), {
        min: -80,
        max: 0,
        default: this.defaults.range,
        unit: 'dB',
        label: 'Range',
        type: 'continuous'
      });

      // Set default values
      this.setParameter('threshold', this.defaults.threshold);
      this.setParameter('attack', this.defaults.attack);
      this.setParameter('release', this.defaults.release);
      this.setParameter('range', this.defaults.range);

      this.initialized = true;

    } catch (error) {
      console.error('Failed to initialize GatePlugin:', error);
      throw error;
    }
  }

  /**
   * Get current gate state
   * @returns {boolean} True if gate is open, false if closed
   */
  getGateState() {
    return this.isGateOpen;
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
      new URL('../worklets/gate-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const workletNode = new AudioWorkletNode(offlineContext, 'gate-processor');

    // Copy current parameter values
    workletNode.parameters.get('threshold').value = this.getParameter('threshold').param.value;
    workletNode.parameters.get('attack').value = this.getParameter('attack').param.value;
    workletNode.parameters.get('release').value = this.getParameter('release').param.value;
    workletNode.parameters.get('range').value = this.getParameter('range').param.value;

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

export default GatePlugin;
