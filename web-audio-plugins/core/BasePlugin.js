/**
 * BasePlugin - Foundation class for all Web Audio plugins
 *
 * Provides core functionality:
 * - Parameter registration and management
 * - Audio routing (input/output nodes)
 * - Bypass functionality
 * - Preset management
 * - Resource cleanup
 *
 * @author Agent 10 (Core Infrastructure)
 * @version 1.0.0
 */

export class BasePlugin {
  /**
   * Create a new plugin instance
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Plugin configuration options
   * @param {string} options.category - Plugin category
   * @param {string} options.description - Plugin description
   * @param {string} options.name - Plugin name (optional, defaults to class name)
   */
  constructor(audioContext, options = {}) {
    if (!audioContext) {
      throw new Error('AudioContext is required');
    }

    this.audioContext = audioContext;
    this.category = options.category || 'uncategorized';
    this.description = options.description || '';
    this.name = options.name || this.constructor.name;

    // Create input and output nodes for routing
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Bypass functionality
    this._bypassed = false;
    this._bypassGain = audioContext.createGain();
    this._bypassGain.connect(this.output);

    // Parameter tracking
    this._parameters = new Map();

    // Preset storage
    this._currentPreset = null;

    // Performance metrics
    this._processingTime = 0;
  }

  /**
   * Register a parameter for automation and preset management
   * @param {string} name - Parameter name
   * @param {AudioParam} audioParam - Web Audio AudioParam to control
   * @param {Object} config - Parameter configuration
   * @param {number} config.min - Minimum value
   * @param {number} config.max - Maximum value
   * @param {number} config.default - Default value
   * @param {string} config.unit - Unit label (Hz, dB, %, etc.)
   * @param {string} config.label - Display label
   * @param {string} config.type - Parameter type (continuous, discrete, switch)
   */
  registerParameter(name, audioParam, config = {}) {
    this._parameters.set(name, {
      param: audioParam,
      config: {
        min: config.min ?? 0,
        max: config.max ?? 1,
        default: config.default ?? 0.5,
        unit: config.unit || '',
        label: config.label || name,
        type: config.type || 'continuous'
      }
    });
  }

  /**
   * Get a registered parameter
   * @param {string} name - Parameter name
   * @returns {Object} Parameter object with param and config
   */
  getParameter(name) {
    return this._parameters.get(name);
  }

  /**
   * Get all registered parameters
   * @returns {Map} All parameters
   */
  getAllParameters() {
    return this._parameters;
  }

  /**
   * Set a parameter value
   * @param {string} name - Parameter name
   * @param {number} value - New value
   * @param {number} time - Time to apply change (AudioContext time)
   */
  setParameter(name, value, time = null) {
    const param = this._parameters.get(name);
    if (!param) {
      console.warn(`Parameter ${name} not found in ${this.name}`);
      return;
    }

    const { min, max } = param.config;
    const clampedValue = Math.max(min, Math.min(max, value));

    if (time === null) {
      param.param.value = clampedValue;
    } else {
      param.param.setValueAtTime(clampedValue, time);
    }
  }

  /**
   * Connect this plugin to another audio node or plugin
   * @param {AudioNode|BasePlugin} destination - Destination node or plugin
   */
  connect(destination) {
    if (destination instanceof BasePlugin) {
      this.output.connect(destination.input);
    } else if (destination instanceof AudioNode) {
      this.output.connect(destination);
    } else {
      throw new Error('Invalid connection destination');
    }
  }

  /**
   * Disconnect this plugin from all outputs
   */
  disconnect() {
    this.output.disconnect();
  }

  /**
   * Bypass/enable the plugin
   * @param {boolean} bypassed - True to bypass, false to enable
   */
  setBypass(bypassed) {
    this._bypassed = bypassed;

    if (bypassed) {
      // Direct connection: input -> output
      this.input.disconnect();
      this.input.connect(this._bypassGain);
    } else {
      // Normal processing chain
      this.input.disconnect();
      // Subclasses will implement their own processing chain
    }
  }

  /**
   * Check if plugin is bypassed
   * @returns {boolean} Bypass state
   */
  isBypassed() {
    return this._bypassed;
  }

  /**
   * Load a preset
   * @param {Object} preset - Preset object with parameter values
   * @param {string} preset.name - Preset name
   * @param {Object} preset.parameters - Parameter values
   */
  loadPreset(preset) {
    if (!preset || !preset.parameters) {
      console.warn('Invalid preset format');
      return;
    }

    Object.entries(preset.parameters).forEach(([name, value]) => {
      this.setParameter(name, value);
    });

    this._currentPreset = preset;
  }

  /**
   * Save current state as a preset
   * @param {string} name - Preset name
   * @returns {Object} Preset object
   */
  savePreset(name) {
    const parameters = {};

    this._parameters.forEach((param, paramName) => {
      parameters[paramName] = param.param.value;
    });

    const preset = {
      name,
      plugin: this.name,
      category: this.category,
      parameters
    };

    this._currentPreset = preset;
    return preset;
  }

  /**
   * Get current preset
   * @returns {Object|null} Current preset or null
   */
  getCurrentPreset() {
    return this._currentPreset;
  }

  /**
   * Cleanup and release resources
   * Must be called when plugin is no longer needed
   */
  dispose() {
    this.disconnect();
    this.input.disconnect();
    this._bypassGain.disconnect();
    this._parameters.clear();
  }

  /**
   * Get plugin info
   * @returns {Object} Plugin metadata
   */
  getInfo() {
    return {
      name: this.name,
      category: this.category,
      description: this.description,
      parameterCount: this._parameters.size,
      bypassed: this._bypassed
    };
  }
}

export default BasePlugin;
