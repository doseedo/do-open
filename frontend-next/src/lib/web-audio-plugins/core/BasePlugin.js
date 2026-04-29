/**
 * BasePlugin - Foundation class for all Web Audio plugins
 *
 * @description
 * Base class that provides common functionality for all audio plugins:
 * - Audio node routing (input/output)
 * - Parameter management and registration
 * - Preset save/load
 * - Bypass functionality
 * - Resource cleanup
 *
 * All plugins MUST extend this class and register with PluginFactory
 *
 * @example
 * class MyPlugin extends BasePlugin {
 *   constructor(audioContext, options = {}) {
 *     super(audioContext, options);
 *     // Your implementation
 *   }
 * }
 */

export class BasePlugin {
  /**
   * Create a new plugin instance
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Object} options - Plugin configuration
   * @param {string} options.category - Plugin category (e.g., 'vintage', 'dynamics')
   * @param {string} options.description - Plugin description
   */
  constructor(audioContext, options = {}) {
    if (!audioContext) {
      throw new Error('AudioContext is required');
    }

    this.audioContext = audioContext;
    this.category = options.category || 'uncategorized';
    this.description = options.description || '';
    this.name = this.constructor.name;

    // Create input and output nodes
    this.input = audioContext.createGain();
    this.output = audioContext.createGain();

    // Bypass functionality
    this._bypass = false;
    this._bypassGain = audioContext.createGain();

    // Parameter tracking
    this.parameters = new Map();
    this.parameterValues = new Map();

    // Performance tracking
    this.cpuUsage = 0;
    this._processingStartTime = 0;

    // Unique instance ID
    this.instanceId = this._generateId();

    // Internal nodes for subclasses
    this._nodes = [];
  }

  /**
   * Register a parameter for automation and preset management
   * @param {string} name - Parameter name
   * @param {AudioParam} audioParam - Web Audio AudioParam
   * @param {Object} config - Parameter configuration
   */
  registerParameter(name, audioParam, config = {}) {
    const paramConfig = {
      param: audioParam,
      min: config.min !== undefined ? config.min : audioParam.minValue || 0,
      max: config.max !== undefined ? config.max : audioParam.maxValue || 1,
      default: config.default !== undefined ? config.default : audioParam.value,
      unit: config.unit || '',
      label: config.label || name,
      type: config.type || 'continuous', // continuous, discrete, boolean
      ...config
    };

    this.parameters.set(name, paramConfig);
    this.parameterValues.set(name, audioParam.value);

    return this;
  }

  /**
   * Set a parameter value
   * @param {string} name - Parameter name
   * @param {number} value - New value
   * @param {number} rampTime - Ramp time in seconds (optional)
   */
  setParameter(name, value, rampTime = 0) {
    const paramConfig = this.parameters.get(name);
    if (!paramConfig) {
      console.warn(`Parameter "${name}" not found in ${this.name}`);
      return this;
    }

    // Clamp value to min/max
    const clampedValue = Math.max(paramConfig.min, Math.min(paramConfig.max, value));

    if (rampTime > 0) {
      const now = this.audioContext.currentTime;
      paramConfig.param.setTargetAtTime(clampedValue, now, rampTime);
    } else {
      paramConfig.param.value = clampedValue;
    }

    this.parameterValues.set(name, clampedValue);

    return this;
  }

  /**
   * Get a parameter value
   * @param {string} name - Parameter name
   * @returns {number} Current value
   */
  getParameter(name) {
    return this.parameterValues.get(name);
  }

  /**
   * Get all parameters
   * @returns {Object} All parameter values
   */
  getAllParameters() {
    const params = {};
    for (const [name, value] of this.parameterValues.entries()) {
      params[name] = value;
    }
    return params;
  }

  /**
   * Save current state as a preset
   * @param {string} name - Preset name
   * @param {string} category - Preset category
   * @param {string} description - Preset description
   * @returns {Object} Preset data
   */
  savePreset(name, category = 'User', description = '') {
    return {
      name,
      category,
      description,
      plugin: this.name,
      version: '1.0.0',
      timestamp: Date.now(),
      parameters: this.getAllParameters()
    };
  }

  /**
   * Load a preset
   * @param {Object} preset - Preset data
   * @param {number} morphTime - Morph time in seconds
   */
  loadPreset(preset, morphTime = 0) {
    if (preset.plugin !== this.name) {
      console.warn(`Preset is for ${preset.plugin}, but this is ${this.name}`);
      return this;
    }

    for (const [name, value] of Object.entries(preset.parameters)) {
      this.setParameter(name, value, morphTime);
    }

    return this;
  }

  /**
   * Enable/disable bypass
   * @param {boolean} bypass - Bypass state
   */
  setBypass(bypass) {
    this._bypass = bypass;

    if (bypass) {
      // Route input directly to output
      this.input.disconnect();
      this.input.connect(this.output);
    } else {
      // TODO: Subclasses should override _reconnect() method
      this._reconnect();
    }

    return this;
  }

  /**
   * Get bypass state
   * @returns {boolean} Bypass state
   */
  getBypass() {
    return this._bypass;
  }

  /**
   * Connect this plugin to another node
   * @param {AudioNode|BasePlugin} destination - Destination node
   */
  connect(destination) {
    if (destination instanceof BasePlugin) {
      this.output.connect(destination.input);
    } else {
      this.output.connect(destination);
    }
    return this;
  }

  /**
   * Disconnect this plugin
   */
  disconnect() {
    this.output.disconnect();
    return this;
  }

  /**
   * Reconnect internal routing (called after bypass)
   * Subclasses should override this
   * @protected
   */
  _reconnect() {
    // Default: direct connection
    this.input.connect(this.output);
  }

  /**
   * Clean up resources
   * IMPORTANT: Always call super.dispose() in subclasses
   */
  dispose() {
    // Disconnect all nodes
    this.input.disconnect();
    this.output.disconnect();
    this._bypassGain.disconnect();

    // Disconnect tracked nodes
    for (const node of this._nodes) {
      if (node && typeof node.disconnect === 'function') {
        node.disconnect();
      }
    }

    // Clear maps
    this.parameters.clear();
    this.parameterValues.clear();
    this._nodes = [];
  }

  /**
   * Generate a unique ID
   * @private
   * @returns {string} Unique ID
   */
  _generateId() {
    return `${this.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Track an audio node for cleanup
   * @protected
   * @param {AudioNode} node - Node to track
   */
  _trackNode(node) {
    this._nodes.push(node);
    return node;
  }

  /**
   * Get plugin info
   * @returns {Object} Plugin information
   */
  getInfo() {
    return {
      name: this.name,
      instanceId: this.instanceId,
      category: this.category,
      description: this.description,
      parameterCount: this.parameters.size,
      bypass: this._bypass,
      cpuUsage: this.cpuUsage
    };
  }
}

export default BasePlugin;
