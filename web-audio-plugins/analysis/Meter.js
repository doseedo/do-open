/**
 * Meter Plugin
 * Real-time audio level metering with RMS and peak detection
 *
 * Features:
 * - RMS (Root Mean Square) level measurement
 * - Peak level detection
 * - Peak hold with configurable hold time
 * - Per-channel metering (stereo support)
 * - Configurable update rate
 * - dB and linear level outputs
 * - Event-based level updates
 *
 * @example
 * const audioContext = new AudioContext();
 * const meter = new MeterPlugin(audioContext);
 *
 * // Connect audio source
 * source.connect(meter.input);
 * meter.connect(audioContext.destination);
 *
 * // Listen for level updates
 * meter.on('update', (levels) => {
 *   console.log('RMS:', levels.rmsDb);
 *   console.log('Peak:', levels.peakDb);
 *   console.log('Peak Hold:', levels.peakHoldDb);
 * });
 *
 * // Configure hold time
 * meter.setHoldTime(2.0); // 2 seconds
 *
 * // Reset peak holds
 * meter.reset();
 *
 * @author Agent 8 (Analyzer Plugins)
 * @version 1.0.0
 */

class MeterPlugin {
  /**
   * Create a Meter plugin
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Initial configuration
   * @param {number} options.holdTime - Peak hold time in seconds (default: 1.5)
   * @param {number} options.updateRate - Updates per second (default: 60)
   */
  constructor(audioContext, options = {}) {
    this.context = audioContext;

    // Create audio nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();
    this.workletNode = null;

    // Parameters
    this.parameters = {
      holdTime: options.holdTime !== undefined ? options.holdTime : 1.5,
      updateRate: options.updateRate !== undefined ? options.updateRate : 60
    };

    // Event listeners
    this.listeners = new Map();

    // Current levels (cached for sync access)
    this.currentLevels = {
      rms: [0, 0],
      rmsDb: [-Infinity, -Infinity],
      peak: [0, 0],
      peakDb: [-Infinity, -Infinity],
      peakHold: [0, 0],
      peakHoldDb: [-Infinity, -Infinity],
      timestamp: 0
    };

    // Setup state
    this.isReady = false;
    this.setupPromise = this.setupWorklet();
  }

  /**
   * Setup AudioWorklet processor
   * @private
   */
  async setupWorklet() {
    try {
      // Get the base path for worklet files
      const basePath = this.getBasePath();

      // Add worklet module
      await this.context.audioWorklet.addModule(`${basePath}/worklets/meter-processor.js`);

      // Create worklet node
      this.workletNode = new AudioWorkletNode(
        this.context,
        'meter-processor',
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          outputChannelCount: [2]
        }
      );

      // Setup message handler
      this.workletNode.port.onmessage = (e) => {
        this.handleWorkletMessage(e.data);
      };

      // Connect nodes
      this.input.connect(this.workletNode);
      this.workletNode.connect(this.output);

      // Apply initial parameters
      this.applyParameters();

      this.isReady = true;
    } catch (error) {
      console.error('Error setting up Meter worklet:', error);
      throw error;
    }
  }

  /**
   * Get base path for worklet files
   * @private
   */
  getBasePath() {
    // Try to determine the base path from the script location
    if (typeof document !== 'undefined') {
      const scripts = document.getElementsByTagName('script');
      for (let script of scripts) {
        if (script.src && script.src.includes('MeterPlugin.js')) {
          return script.src.substring(0, script.src.lastIndexOf('/'));
        }
      }
    }
    return './analysis';
  }

  /**
   * Handle messages from worklet
   * @private
   */
  handleWorkletMessage(data) {
    if (data.type === 'meter-update') {
      // Update current levels cache
      this.currentLevels = {
        rms: data.rms,
        rmsDb: data.rmsDb,
        peak: data.peak,
        peakDb: data.peakDb,
        peakHold: data.peakHold,
        peakHoldDb: data.peakHoldDb,
        timestamp: data.timestamp
      };

      // Emit update event
      this.emit('update', this.currentLevels);
    }
  }

  /**
   * Apply all parameters to worklet
   * @private
   */
  applyParameters() {
    if (!this.workletNode) return;

    Object.entries(this.parameters).forEach(([key, value]) => {
      this.workletNode.port.postMessage({
        type: key,
        value: value
      });
    });
  }

  /**
   * Wait for the processor to be ready
   * @returns {Promise<void>}
   */
  async ready() {
    await this.setupPromise;
  }

  /**
   * Set peak hold time
   * @param {number} seconds - Hold time in seconds (0.1 to 10)
   */
  setHoldTime(seconds) {
    this.parameters.holdTime = Math.max(0.1, Math.min(10, seconds));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'holdTime',
        value: this.parameters.holdTime
      });
    }
  }

  /**
   * Set update rate
   * @param {number} rate - Updates per second (1 to 120)
   */
  setUpdateRate(rate) {
    this.parameters.updateRate = Math.max(1, Math.min(120, rate));
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'updateRate',
        value: this.parameters.updateRate
      });
    }
  }

  /**
   * Reset all peak holds
   */
  reset() {
    if (this.workletNode) {
      this.workletNode.port.postMessage({
        type: 'reset'
      });
    }
  }

  /**
   * Get current levels (synchronous, may be slightly outdated)
   * @returns {Object} Current levels
   */
  getLevels() {
    return { ...this.currentLevels };
  }

  /**
   * Connect to destination
   * @param {AudioNode|AudioParam} destination - Destination to connect to
   */
  connect(destination) {
    this.output.connect(destination);
  }

  /**
   * Disconnect from all outputs
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Add event listener
   * @param {string} event - Event name ('update')
   * @param {Function} callback - Event callback
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);
  }

  /**
   * Remove event listener
   * @param {string} event - Event name
   * @param {Function} callback - Event callback to remove
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index !== -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Emit event to all listeners
   * @private
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        callback(data);
      });
    }
  }

  /**
   * Cleanup and release resources
   */
  dispose() {
    this.disconnect();
    this.input.disconnect();

    if (this.workletNode) {
      this.workletNode.disconnect();
      this.workletNode = null;
    }

    this.listeners.clear();
  }

  /**
   * Get plugin info
   * @returns {Object} Plugin metadata
   */
  getInfo() {
    return {
      name: 'Meter',
      category: 'analysis',
      description: 'Real-time audio level metering with RMS and peak detection',
      author: 'Agent 8',
      version: '1.0.0'
    };
  }
}

// Export for use in Node.js or as module
if (typeof module !== 'undefined' && module.exports) {
  module.exports = MeterPlugin;
}
