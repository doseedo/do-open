/**
 * FilterPlugin - Single Versatile Filter
 * AudioWorklet-based plugin for multi-mode filtering
 *
 * Features:
 * - Multiple filter types (lowpass, highpass, bandpass, notch, peaking, lowshelf, highshelf, allpass)
 * - Adjustable frequency, Q, and gain
 * - Dry/wet mix control
 * - Optimized for offline rendering (20x+ real-time)
 * - Compatible with BasePlugin architecture
 *
 * @author Agent 2 (EQ Plugins)
 * @version 2.0.0 (AudioWorklet)
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class FilterPlugin extends BasePlugin {
  constructor(audioContext, options = {}) {
    super(audioContext, {
      name: 'Filter',
      category: 'eq',
      description: 'Single Versatile Filter'
    });

    // AudioWorklet node
    this.workletNode = null;

    // Track initialization state
    this.initialized = false;

    // Available filter types
    this.filterTypes = [
      'lowpass',
      'highpass',
      'bandpass',
      'notch',
      'peaking',
      'lowshelf',
      'highshelf',
      'allpass'
    ];

    // Default parameters
    this.params = {
      type: 'lowpass',
      frequency: 1000,
      q: 1.0,
      gain: 0,
      mix: 1.0,
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
      const workletPath = new URL('./worklets/filter-processor.js', import.meta.url).href;
      await this.audioContext.audioWorklet.addModule(workletPath);

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(this.audioContext, 'filter-processor', {
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

      console.log('FilterPlugin initialized');
    } catch (error) {
      console.error('Failed to initialize FilterPlugin:', error);
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
   * Set filter type
   * @param {string} type - Filter type (lowpass, highpass, bandpass, notch, peaking, lowshelf, highshelf, allpass)
   */
  setType(type) {
    if (!this.filterTypes.includes(type)) {
      console.warn(`Invalid filter type: ${type}`);
      return;
    }

    this.updateParam('type', type);
  }

  /**
   * Get current filter type
   * @returns {string} Current filter type
   */
  getType() {
    return this.params.type;
  }

  /**
   * Get available filter types
   * @returns {Array<string>} Array of filter type names
   */
  getAvailableTypes() {
    return [...this.filterTypes];
  }

  /**
   * Set filter frequency
   * @param {number} frequency - Frequency in Hz (20-20000)
   */
  setFrequency(frequency) {
    this.updateParam('frequency', Math.max(20, Math.min(20000, frequency)));
  }

  /**
   * Get current frequency
   * @returns {number} Frequency in Hz
   */
  getFrequency() {
    return this.params.frequency;
  }

  /**
   * Set Q factor (resonance/bandwidth)
   * @param {number} q - Q factor (0.1-20)
   */
  setQ(q) {
    this.updateParam('q', Math.max(0.1, Math.min(20, q)));
  }

  /**
   * Get current Q factor
   * @returns {number} Q factor
   */
  getQ() {
    return this.params.q;
  }

  /**
   * Set gain (for peaking and shelving filters)
   * @param {number} gainDb - Gain in dB (-15 to +15)
   */
  setGain(gainDb) {
    this.updateParam('gain', Math.max(-15, Math.min(15, gainDb)));
  }

  /**
   * Get current gain
   * @returns {number} Gain in dB
   */
  getGain() {
    return this.params.gain;
  }

  /**
   * Set dry/wet mix
   * @param {number} mix - Mix amount (0 = dry, 1 = wet)
   */
  setMix(mix) {
    this.updateParam('mix', Math.max(0, Math.min(1, mix)));
  }

  /**
   * Get current mix
   * @returns {number} Mix amount (0-1)
   */
  getMix() {
    return this.params.mix;
  }

  /**
   * Set output gain
   * @param {number} gain - Linear gain (0 to 2)
   */
  setOutputGain(gain) {
    this.updateParam('outputGain', Math.max(0, Math.min(2, gain)));
  }

  /**
   * Get current output gain
   * @returns {number} Output gain
   */
  getOutputGain() {
    return this.params.outputGain;
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
   * Set multiple parameters at once
   * @param {Object} params - Object with parameter values
   */
  setParameters(params) {
    Object.assign(this.params, params);

    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'init',
        params: this.params
      });
    }
  }

  /**
   * Reset to default settings
   */
  reset() {
    this.setType('lowpass');
    this.setFrequency(1000);
    this.setQ(1.0);
    this.setGain(0);
    this.setMix(1.0);
    this.setOutputGain(1.0);
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
    const workletPath = new URL('./worklets/filter-processor.js', import.meta.url).href;
    await offlineContext.audioWorklet.addModule(workletPath);

    // Create worklet node in offline context
    const offlineWorklet = new AudioWorkletNode(offlineContext, 'filter-processor', {
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
      filterTypes: this.filterTypes
    };
  }
}

// Export for use in other modules
export default FilterPlugin;
