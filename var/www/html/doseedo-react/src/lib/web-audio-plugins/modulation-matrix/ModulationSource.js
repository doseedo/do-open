/**
 * ModulationSource - Base class for all modulation sources
 *
 * Provides common functionality for modulation sources (LFOs, envelopes, etc.):
 * - Modulation output signal generation
 * - Routing to multiple destinations
 * - Depth/amount control per destination
 * - Bipolar/unipolar output
 * - Start/stop control
 *
 * @author Agent 17 (Modulation Matrix & Advanced LFOs)
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';

export class ModulationSource extends BasePlugin {
  /**
   * Create a modulation source
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Configuration options
   * @param {boolean} options.bipolar - Output range: bipolar (-1 to 1) or unipolar (0 to 1)
   * @param {number} options.frequency - Modulation frequency in Hz
   */
  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: options.category || 'modulation',
      description: options.description || 'Modulation source'
    });

    // Modulation output characteristics
    this.bipolar = options.bipolar !== undefined ? options.bipolar : true;
    this._running = false;

    // Create constant source for DC offset (used for unipolar conversion)
    this._dcOffset = audioContext.createConstantSource();
    this._dcOffset.offset.value = this.bipolar ? 0 : 0.5;
    this._dcOffset.start();

    // Create gain node for modulation depth/scaling
    this._depthGain = audioContext.createGain();
    this._depthGain.gain.value = 1.0;

    // Modulation destinations: { targetParam, depthGain, connection }
    this._destinations = [];

    // Register common modulation parameters
    this.registerParameter('depth', this._depthGain.gain, {
      min: 0,
      max: 1,
      default: 1.0,
      unit: '%',
      label: 'Modulation Depth'
    });
  }

  /**
   * Start the modulation source
   * @param {number} time - Start time (AudioContext time)
   */
  start(time = null) {
    if (this._running) {
      console.warn(`${this.name} is already running`);
      return;
    }

    const startTime = time !== null ? time : this.audioContext.currentTime;
    this._onStart(startTime);
    this._running = true;
  }

  /**
   * Stop the modulation source
   * @param {number} time - Stop time (AudioContext time)
   */
  stop(time = null) {
    if (!this._running) {
      console.warn(`${this.name} is not running`);
      return;
    }

    const stopTime = time !== null ? time : this.audioContext.currentTime;
    this._onStop(stopTime);
    this._running = false;
  }

  /**
   * Check if modulation source is running
   * @returns {boolean} Running state
   */
  isRunning() {
    return this._running;
  }

  /**
   * Override in subclasses to implement start behavior
   * @protected
   * @param {number} time - Start time
   */
  _onStart(time) {
    // To be implemented by subclasses
  }

  /**
   * Override in subclasses to implement stop behavior
   * @protected
   * @param {number} time - Stop time
   */
  _onStop(time) {
    // To be implemented by subclasses
  }

  /**
   * Route modulation to a target parameter
   * @param {AudioParam} targetParam - Destination audio parameter
   * @param {number} depth - Modulation depth/amount (0-1)
   * @returns {Object} Routing object with disconnect method
   */
  routeTo(targetParam, depth = 1.0) {
    if (!(targetParam instanceof AudioParam)) {
      throw new Error('Target must be an AudioParam');
    }

    // Create a gain node for this specific routing
    const routingGain = this.audioContext.createGain();
    routingGain.gain.value = depth;

    // Connect: modulation output -> depth control -> routing gain -> target
    this.output.connect(this._depthGain);
    this._depthGain.connect(routingGain);
    routingGain.connect(targetParam);

    // Store routing information
    const routing = {
      targetParam,
      depthGain: routingGain,
      disconnect: () => {
        routingGain.disconnect();
        const index = this._destinations.indexOf(routing);
        if (index !== -1) {
          this._destinations.splice(index, 1);
        }
      }
    };

    this._destinations.push(routing);
    return routing;
  }

  /**
   * Set modulation depth for a specific destination
   * @param {AudioParam} targetParam - Target parameter
   * @param {number} depth - New depth value (0-1)
   */
  setRoutingDepth(targetParam, depth) {
    const routing = this._destinations.find(d => d.targetParam === targetParam);
    if (routing) {
      routing.depthGain.gain.value = Math.max(0, Math.min(1, depth));
    }
  }

  /**
   * Disconnect from a specific target
   * @param {AudioParam} targetParam - Target to disconnect
   */
  disconnectFrom(targetParam) {
    const routing = this._destinations.find(d => d.targetParam === targetParam);
    if (routing) {
      routing.disconnect();
    }
  }

  /**
   * Disconnect from all targets
   */
  disconnectAll() {
    this._destinations.forEach(routing => {
      routing.depthGain.disconnect();
    });
    this._destinations = [];
  }

  /**
   * Get all active destinations
   * @returns {Array<Object>} Destination routings
   */
  getDestinations() {
    return [...this._destinations];
  }

  /**
   * Set bipolar/unipolar mode
   * @param {boolean} bipolar - True for -1 to 1, false for 0 to 1
   */
  setBipolar(bipolar) {
    this.bipolar = bipolar;
    this._dcOffset.offset.value = bipolar ? 0 : 0.5;
  }

  /**
   * Get current output value (for visualization)
   * Subclasses should override this to provide real-time value
   * @returns {number} Current modulation value
   */
  getCurrentValue() {
    return 0;
  }

  /**
   * Cleanup resources
   */
  dispose() {
    this.stop();
    this.disconnectAll();
    this._dcOffset.stop();
    this._dcOffset.disconnect();
    this._depthGain.disconnect();
    super.dispose();
  }

  /**
   * Get modulation source info
   * @returns {Object} Source metadata
   */
  getInfo() {
    return {
      ...super.getInfo(),
      running: this._running,
      bipolar: this.bipolar,
      destinationCount: this._destinations.length
    };
  }
}

export default ModulationSource;
