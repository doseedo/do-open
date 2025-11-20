/**
 * ConvolutionReverbPlugin - AudioWorklet-based convolution reverb
 *
 * Uses impulse responses for realistic room acoustics.
 * This is the modern AudioWorklet implementation for production use.
 *
 * Features:
 * - Load impulse response files
 * - Pre-delay control
 * - Dry/wet mix
 * - High-performance offline rendering
 *
 * @author Agent 6: Reverb Plugins
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class ConvolutionReverbPlugin extends BasePlugin {
  /**
   * Create a new ConvolutionReverbPlugin
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Plugin configuration
   */
  constructor(audioContext, options = {}) {
    super(audioContext, {
      name: 'Convolution Reverb',
      category: 'reverb',
      description: 'Convolution reverb with impulse response support',
      ...options
    });

    // AudioWorklet node
    this.workletNode = null;

    // Impulse response data
    this.impulseResponseBuffer = null;

    // Plugin state
    this.isInitialized = false;

    // Default parameters
    this.params = {
      mix: 30,        // 0-100%
      predelay: 0     // 0-250ms
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
        new URL('./worklets/convolution-reverb-processor.js', import.meta.url).href
      );

      // Create the AudioWorklet node
      this.workletNode = new AudioWorkletNode(
        this.audioContext,
        'convolution-reverb-processor'
      );

      // Connect input -> worklet -> output
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Listen for messages from the worklet
      this.workletNode.port.onmessage = (event) => {
        this.handleWorkletMessage(event.data);
      };

      // Send initial parameters
      this.updateWorkletParams();

      this.isInitialized = true;
      return true;

    } catch (error) {
      console.error('Failed to initialize ConvolutionReverbPlugin:', error);
      return false;
    }
  }

  /**
   * Handle messages from the worklet
   * @param {Object} data - Message data
   */
  handleWorkletMessage(data) {
    const { type } = data;

    if (type === 'irLoaded') {
      console.log(`Impulse response loaded: ${data.duration.toFixed(2)}s`);
    }
  }

  /**
   * Load impulse response from URL
   * @param {string} url - URL of impulse response file (WAV)
   * @returns {Promise<boolean>} Success status
   */
  async loadImpulseResponse(url) {
    try {
      const response = await fetch(url);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const arrayBuffer = await response.arrayBuffer();
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

      return this.setImpulseResponse(audioBuffer);

    } catch (error) {
      console.error('Failed to load impulse response:', error);
      return false;
    }
  }

  /**
   * Load impulse response from file
   * @param {File} file - Audio file
   * @returns {Promise<boolean>} Success status
   */
  async loadImpulseResponseFile(file) {
    try {
      const arrayBuffer = await file.arrayBuffer();
      const audioBuffer = await this.audioContext.decodeAudioData(arrayBuffer);

      return this.setImpulseResponse(audioBuffer);

    } catch (error) {
      console.error('Failed to load impulse response file:', error);
      return false;
    }
  }

  /**
   * Set impulse response from AudioBuffer
   * @param {AudioBuffer} audioBuffer - Impulse response buffer
   * @returns {boolean} Success status
   */
  setImpulseResponse(audioBuffer) {
    if (!audioBuffer) {
      console.warn('Invalid audio buffer');
      return false;
    }

    if (!this.isInitialized) {
      console.warn('Plugin not initialized. Call initialize() first.');
      return false;
    }

    // Store the buffer
    this.impulseResponseBuffer = audioBuffer;

    // Convert to mono if stereo (sum channels)
    const channelData = audioBuffer.getChannelData(0);
    let impulseResponse;

    if (audioBuffer.numberOfChannels > 1) {
      // Sum all channels to mono
      impulseResponse = new Float32Array(audioBuffer.length);
      for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
        const data = audioBuffer.getChannelData(ch);
        for (let i = 0; i < data.length; i++) {
          impulseResponse[i] += data[i];
        }
      }
      // Normalize
      const scale = 1 / audioBuffer.numberOfChannels;
      for (let i = 0; i < impulseResponse.length; i++) {
        impulseResponse[i] *= scale;
      }
    } else {
      impulseResponse = new Float32Array(channelData);
    }

    // Send to worklet
    this.workletNode.port.postMessage({
      type: 'setImpulseResponse',
      impulseResponse: impulseResponse
    });

    return true;
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
   * Set pre-delay
   * @param {number} ms - Pre-delay in milliseconds (0-250)
   */
  setPreDelay(ms) {
    this.params.predelay = Math.max(0, Math.min(250, ms));
    this.updateWorkletParams();
  }

  /**
   * Update worklet parameters
   */
  updateWorkletParams() {
    if (!this.workletNode) return;

    this.workletNode.port.postMessage({
      type: 'setParams',
      params: {
        mix: this.params.mix,
        predelay: this.params.predelay
      }
    });
  }

  /**
   * Get impulse response info
   * @returns {Object|null} IR info
   */
  getIRInfo() {
    if (!this.impulseResponseBuffer) {
      return null;
    }

    return {
      duration: this.impulseResponseBuffer.duration,
      sampleRate: this.impulseResponseBuffer.sampleRate,
      numberOfChannels: this.impulseResponseBuffer.numberOfChannels,
      length: this.impulseResponseBuffer.length
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
      new URL('./worklets/convolution-reverb-processor.js', import.meta.url).href
    );

    // Create nodes
    const source = offlineContext.createBufferSource();
    source.buffer = inputBuffer;

    const worklet = new AudioWorkletNode(offlineContext, 'convolution-reverb-processor');

    // Send parameters and IR
    worklet.port.postMessage({
      type: 'setParams',
      params: this.params
    });

    if (this.impulseResponseBuffer) {
      const channelData = this.impulseResponseBuffer.getChannelData(0);
      worklet.port.postMessage({
        type: 'setImpulseResponse',
        impulseResponse: new Float32Array(channelData)
      });
    }

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

    this.impulseResponseBuffer = null;
    this.isInitialized = false;

    super.dispose();
  }
}

export default ConvolutionReverbPlugin;
