/**
 * ReverbPlugin - AudioWorklet-based algorithmic reverb
 *
 * Freeverb-style reverb using Schroeder reverberator architecture.
 * High-performance implementation for production use.
 *
 * Features:
 * - Room size control
 * - Decay time (RT60)
 * - High-frequency damping
 * - Stereo width control
 * - Pre-delay
 * - Dry/wet mix
 * - Offline rendering support
 *
 * @author Agent 6: Reverb Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class ReverbPlugin extends BasePlugin {
  /**
   * Create a new ReverbPlugin
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Plugin configuration
   */
  constructor(audioContext, options = {}) {
    super(audioContext, {
      name: 'Algorithmic Reverb',
      category: 'reverb',
      description: 'Freeverb-style algorithmic reverb with full parameter control',
      ...options
    });

    // AudioWorklet node
    this.workletNode = null;

    // Plugin state
    this.isInitialized = false;

    // Default parameters
    this.params = {
      roomSize: 50,      // 0-100%
      decayTime: 2.0,    // 0.1-20 seconds
      damping: 50,       // 0-100%
      width: 100,        // 0-100%
      predelay: 0,       // 0-250ms
      mix: 30            // 0-100%
    };
  }

  /**
   * Initialize the plugin (load AudioWorklet)
   * Must be called before use
   * @returns {Promise<boolean>} Success status
   */
  async initialize() {
    if (this.isInitialized) {
      return true;
    }

    try {
      // Load the AudioWorklet module
      await this.audioContext.audioWorklet.addModule(
        new URL('./worklets/reverb-processor.js', import.meta.url).href
      );

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'reverb-processor',
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2]
        }
      );

      // Connect input -> worklet -> output
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Send initial parameters
      this.updateWorkletParams();

      this.isInitialized = true;
      return true;

    } catch (error) {
      console.error('Failed to initialize ReverbPlugin:', error);
      return false;
    }
  }

  /**
   * Set room size
   * @param {number} percent - Room size percentage (0-100)
   */
  setRoomSize(percent) {
    this.params.roomSize = Math.max(0, Math.min(100, percent));
    this.updateWorkletParams();
  }

  /**
   * Set decay time (RT60)
   * @param {number} seconds - Decay time in seconds (0.1-20)
   */
  setDecayTime(seconds) {
    this.params.decayTime = Math.max(0.1, Math.min(20, seconds));
    this.updateWorkletParams();
  }

  /**
   * Set high-frequency damping
   * @param {number} percent - Damping percentage (0-100)
   */
  setDamping(percent) {
    this.params.damping = Math.max(0, Math.min(100, percent));
    this.updateWorkletParams();
  }

  /**
   * Set stereo width
   * @param {number} percent - Width percentage (0-100)
   */
  setWidth(percent) {
    this.params.width = Math.max(0, Math.min(100, percent));
    this.updateWorkletParams();
  }

  /**
   * Set pre-delay
   * @param {number} ms - Pre-delay in milliseconds (0-250)
   */
  setPreDelay(ms) {
    this.params.predelay = Math.max(0, Math.min(250, ms));
    this.updateWorkletParams();
  }

  /**
   * Set mix parameter (dry/wet)
   * @param {number} percent - Mix percentage (0-100)
   */
  setMix(percent) {
    this.params.mix = Math.max(0, Math.min(100, percent));
    this.updateWorkletParams();
  }

  /**
   * Set size (alias for setRoomSize for compatibility)
   * @param {number} percent - Size percentage (0-100)
   */
  setSize(percent) {
    this.setRoomSize(percent);
  }

  /**
   * Update worklet parameters
   */
  updateWorkletParams() {
    if (!this.workletNode) return;

    this.workletNode.port.postMessage({
      type: 'setParams',
      params: {
        roomSize: this.params.roomSize,
        decayTime: this.params.decayTime,
        damping: this.params.damping,
        width: this.params.width,
        predelay: this.params.predelay,
        mix: this.params.mix
      }
    });
  }

  /**
   * Get current parameters
   * @returns {Object} Parameter values
   */
  getParams() {
    return { ...this.params };
  }

  /**
   * Load a preset
   * @param {Object} preset - Preset configuration
   */
  loadPreset(preset) {
    if (!preset || !preset.parameters) {
      console.warn('Invalid preset');
      return;
    }

    const { parameters } = preset;

    if (parameters.roomSize !== undefined) this.setRoomSize(parameters.roomSize);
    if (parameters.decayTime !== undefined) this.setDecayTime(parameters.decayTime);
    if (parameters.damping !== undefined) this.setDamping(parameters.damping);
    if (parameters.width !== undefined) this.setWidth(parameters.width);
    if (parameters.predelay !== undefined) this.setPreDelay(parameters.predelay);
    if (parameters.mix !== undefined) this.setMix(parameters.mix);
  }

  /**
   * Save current state as preset
   * @param {string} name - Preset name
   * @returns {Object} Preset object
   */
  savePreset(name) {
    return {
      name,
      plugin: 'ReverbPlugin',
      category: 'reverb',
      parameters: { ...this.params }
    };
  }

  /**
   * Process audio offline (for rendering)
   * @param {AudioBuffer} inputBuffer - Input audio buffer
   * @returns {Promise<AudioBuffer>} Processed audio buffer
   */
  async processOffline(inputBuffer) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    // Create offline context
    const offlineContext = new OfflineAudioContext(
      inputBuffer.numberOfChannels,
      inputBuffer.length,
      inputBuffer.sampleRate
    );

    // Load worklet in offline context
    await offlineContext.audioWorklet.addModule(
      new URL('./worklets/reverb-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const worklet = new AudioWorkletNode(offlineContext, 'reverb-processor', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2]
    });

    // Send parameters
    worklet.port.postMessage({
      type: 'setParams',
      params: this.params
    });

    // Connect and render
    source.connect(worklet);
    worklet.connect(offlineContext.destination);
    source.start();

    return await offlineContext.startRendering();
  }

  /**
   * Check if plugin uses AudioWorklet
   * @returns {boolean} Always true for this plugin
   */
  usesAudioWorklet() {
    return true;
  }

  /**
   * Cleanup and release resources
   */
  dispose() {
    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    this.isInitialized = false;

    super.dispose();
  }
}

// Factory presets
export const ReverbPresets = {
  smallRoom: {
    name: 'Small Room',
    parameters: {
      roomSize: 25,
      decayTime: 0.8,
      damping: 60,
      width: 80,
      predelay: 5,
      mix: 25
    }
  },

  mediumHall: {
    name: 'Medium Hall',
    parameters: {
      roomSize: 50,
      decayTime: 2.0,
      damping: 50,
      width: 100,
      predelay: 20,
      mix: 35
    }
  },

  largeHall: {
    name: 'Large Hall',
    parameters: {
      roomSize: 75,
      decayTime: 4.0,
      damping: 40,
      width: 100,
      predelay: 30,
      mix: 40
    }
  },

  cathedral: {
    name: 'Cathedral',
    parameters: {
      roomSize: 90,
      decayTime: 8.0,
      damping: 30,
      width: 100,
      predelay: 40,
      mix: 45
    }
  },

  plate: {
    name: 'Plate',
    parameters: {
      roomSize: 40,
      decayTime: 2.5,
      damping: 70,
      width: 100,
      predelay: 0,
      mix: 30
    }
  },

  spring: {
    name: 'Spring',
    parameters: {
      roomSize: 30,
      decayTime: 1.5,
      damping: 80,
      width: 60,
      predelay: 0,
      mix: 35
    }
  }
};

export default ReverbPlugin;
