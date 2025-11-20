/**
 * EQPlugin - 3-Band Parametric Equalizer
 * AudioWorklet-based plugin for high-performance EQ processing
 *
 * Features:
 * - 3 independent peaking filters (Low, Mid, High)
 * - Adjustable frequency, gain, and Q per band
 * - Smooth parameter changes
 * - Optimized for offline rendering (20x+ real-time)
 * - Compatible with BasePlugin architecture
 *
 * @author Agent 2 (EQ Plugins)
 * @version 2.0.0 (AudioWorklet)
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class EQPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      name: 'EQ-3',
      category: 'eq',
      description: '3-Band Parametric Equalizer'
    });

    // AudioWorklet node
    this.workletNode = null;

    // Track initialization state
    this.initialized = false;

    // Default parameters
    this.params = {
      freq1: 100,
      gain1: 0,
      q1: 1.0,
      freq2: 1000,
      gain2: 0,
      q2: 1.0,
      freq3: 10000,
      gain3: 0,
      q3: 1.0,
      outputGain: 1.0
    };

    // Apply options
    Object.assign(this.params, options);
  }

  /**
   * Initialize the AudioWorklet
   * Must be called before using the plugin
   */
  async initialize() {
    if (this.initialized) {
      return;
    }

    try {
      // Load the AudioWorklet module
      const workletPath = new URL('./worklets/eq-processor.js', import.meta.url).href;
      await this.audioContext.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.audioContext, 'eq-processor', {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        channelCount: 2,
        channelCountMode: 'explicit',
        channelInterpretation: 'speakers'
      });

      // Connect audio routing: input -> worklet -> output
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Initialize with parameters
      this.workletNode.port.postMessage({
        type: 'init',
        params: this.params
      });

      this.initialized = true;

      console.log('EQPlugin initialized');
    } catch (error) {
      console.error('Failed to initialize EQPlugin:', error);
      throw error;
    }
  }

  /**
   * Check if this plugin uses AudioWorklet
   * @returns {boolean} True (this plugin uses AudioWorklet)
   */
  usesAudioWorklet() {
    return true;
  }

  /**
   * Set band 1 (Low) parameters
   * @param {Object} params - Band parameters
   * @param {number} params.frequency - Frequency in Hz (20-20000)
   * @param {number} params.gain - Gain in dB (-15 to +15)
   * @param {number} params.q - Q factor (0.1 to 10)
   */
  setBand1(params) {
    if (params.frequency !== undefined) {
      this.updateParam('freq1', Math.max(20, Math.min(20000, params.frequency)));
    }
    if (params.gain !== undefined) {
      this.updateParam('gain1', Math.max(-15, Math.min(15, params.gain)));
    }
    if (params.q !== undefined) {
      this.updateParam('q1', Math.max(0.1, Math.min(10, params.q)));
    }
  }

  /**
   * Set band 2 (Mid) parameters
   * @param {Object} params - Band parameters
   */
  setBand2(params) {
    if (params.frequency !== undefined) {
      this.updateParam('freq2', Math.max(20, Math.min(20000, params.frequency)));
    }
    if (params.gain !== undefined) {
      this.updateParam('gain2', Math.max(-15, Math.min(15, params.gain)));
    }
    if (params.q !== undefined) {
      this.updateParam('q2', Math.max(0.1, Math.min(10, params.q)));
    }
  }

  /**
   * Set band 3 (High) parameters
   * @param {Object} params - Band parameters
   */
  setBand3(params) {
    if (params.frequency !== undefined) {
      this.updateParam('freq3', Math.max(20, Math.min(20000, params.frequency)));
    }
    if (params.gain !== undefined) {
      this.updateParam('gain3', Math.max(-15, Math.min(15, params.gain)));
    }
    if (params.q !== undefined) {
      this.updateParam('q3', Math.max(0.1, Math.min(10, params.q)));
    }
  }

  /**
   * Set output gain
   * @param {number} gain - Linear gain (0 to 2)
   */
  setOutputGain(gain) {
    this.updateParam('outputGain', Math.max(0, Math.min(2, gain)));
  }

  /**
   * Update a parameter and send to worklet
   */
  updateParam(param, value) {
    this.params[param] = value;

    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'update',
        param: param,
        value: value
      });
    }
  }

  /**
   * Get current state
   * @returns {Object} Current parameters
   */
  getState() {
    return { ...this.params };
  }

  /**
   * Load state from preset
   * @param {Object} state - State object with parameters
   */
  setState(state) {
    Object.assign(this.params, state);

    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'init',
        params: this.params
      });
    }
  }

  /**
   * Reset all bands to unity (flat response)
   */
  reset() {
    this.setBand1({ frequency: 100, gain: 0, q: 1.0 });
    this.setBand2({ frequency: 1000, gain: 0, q: 1.0 });
    this.setBand3({ frequency: 10000, gain: 0, q: 1.0 });
    this.setOutputGain(1.0);
  }

  /**
   * Process audio offline (for rendering)
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
    const workletPath = new URL('./worklets/eq-processor.js', import.meta.url).href;
    await offlineContext.audioWorklet.addModule(workletPath);

    // Create worklet node in offline context
    const offlineWorklet = new AudioWorkletNode(offlineContext, 'eq-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      channelCount: inputBuffer.numberOfChannels,
      channelCountMode: 'explicit'
    });

    // Initialize with current parameters
    offlineWorklet.port.postMessage({
      type: 'init',
      params: this.params
    });

    // Create buffer source
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    // Connect: source -> worklet -> destination
    source.connect(offlineWorklet);
    offlineWorklet.connect(offlineContext.destination);

    // Start and render
    source.start(0);
    const renderedBuffer = await offlineContext.startRendering();

    return renderedBuffer;
  }

  /**
   * Clean up resources
   */
  dispose() {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    super.dispose();
    this.initialized = false;
  }

  /**
   * Get plugin info
   * @returns {Object} Plugin metadata
   */
  getInfo() {
    return {
      ...super.getInfo(),
      type: 'audioworklet',
      version: '2.0.0',
      bands: 3
    };
  }
}

// Export for use in other modules
export default EQPlugin;
