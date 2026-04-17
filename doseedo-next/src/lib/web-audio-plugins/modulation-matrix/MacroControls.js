/**
 * MacroControls - Macro Control System
 *
 * Features:
 * - 8 macro knobs (expandable to 16)
 * - Each macro controls multiple parameters
 * - Customizable parameter ranges (min/max)
 * - Curve types: linear, exponential, logarithmic
 * - Preset morphing via macros
 * - MIDI CC mapping
 * - Automation recording/playback
 * - Learn mode for quick assignment
 *
 * Inspired by: Ableton Macro Controls, Bitwig Macros, Reaktor Macros
 *
 * @author Agent 17 (Modulation Matrix & Advanced LFOs)
 * @version 1.0.0
 */

import { BasePlugin } from '../core/BasePlugin.js';
import PluginFactory from '../core/PluginFactory.js';

export class MacroControls extends BasePlugin {
  /**
   * Curve types for parameter mapping
   */
  static CURVE_TYPES = {
    LINEAR: 'linear',
    EXPONENTIAL: 'exponential',
    LOGARITHMIC: 'logarithmic',
    S_CURVE: 's-curve'
  };

  /**
   * Default number of macro knobs
   */
  static DEFAULT_MACRO_COUNT = 8;

  constructor(audioContext, options = {}) {
    super(audioContext, {
      ...options,
      category: 'modulation',
      description: 'Macro control system with 8 knobs'
    });

    this.macroCount = options.macroCount || MacroControls.DEFAULT_MACRO_COUNT;

    // Macro state: value (0-1) and label
    this._macros = [];

    // Macro mappings: macroIndex -> [{ targetParam, min, max, curve, plugin }]
    this._mappings = new Map();

    // Learn mode
    this._learnMode = false;
    this._learningMacro = null;

    // Automation
    this._automationData = new Map();
    this._isRecording = false;
    this._recordingStartTime = 0;

    // Initialize macros
    for (let i = 0; i < this.macroCount; i++) {
      this._createMacro(i);
    }
  }

  /**
   * Create a macro control
   * @private
   * @param {number} index - Macro index
   */
  _createMacro(index) {
    const constantSource = this.audioContext.createConstantSource();
    constantSource.offset.value = 0; // Default position (0-1)
    constantSource.start();

    const macro = {
      index,
      label: `Macro ${index + 1}`,
      source: constantSource,
      value: 0,
      midiCC: -1, // Not mapped by default
      color: this._generateMacroColor(index)
    };

    this._macros[index] = macro;
    this._mappings.set(index, []);

    // Register as parameter
    this.registerParameter(`macro${index}`, constantSource.offset, {
      min: 0,
      max: 1,
      default: 0,
      unit: '',
      label: macro.label
    });
  }

  /**
   * Generate a color for macro (for UI visualization)
   * @private
   * @param {number} index - Macro index
   * @returns {string} CSS color
   */
  _generateMacroColor(index) {
    const hue = (index * 45) % 360;
    return `hsl(${hue}, 70%, 60%)`;
  }

  /**
   * Set macro value
   * @param {number} macroIndex - Macro index (0-7)
   * @param {number} value - Value (0-1)
   * @param {number} time - Time to apply change
   */
  setMacroValue(macroIndex, value, time = null) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      console.warn(`Invalid macro index: ${macroIndex}`);
      return;
    }

    const macro = this._macros[macroIndex];
    const clampedValue = Math.max(0, Math.min(1, value));
    macro.value = clampedValue;

    const setTime = time !== null ? time : this.audioContext.currentTime;
    macro.source.offset.setValueAtTime(clampedValue, setTime);

    // Update all mapped parameters
    this._updateMappedParameters(macroIndex, clampedValue, setTime);

    // Record automation if recording
    if (this._isRecording) {
      this._recordAutomation(macroIndex, clampedValue, setTime);
    }
  }

  /**
   * Get macro value
   * @param {number} macroIndex - Macro index
   * @returns {number} Current value (0-1)
   */
  getMacroValue(macroIndex) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      return 0;
    }

    return this._macros[macroIndex].value;
  }

  /**
   * Set macro label
   * @param {number} macroIndex - Macro index
   * @param {string} label - New label
   */
  setMacroLabel(macroIndex, label) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      return;
    }

    this._macros[macroIndex].label = label;
  }

  /**
   * Map a macro to a parameter
   * @param {number} macroIndex - Macro index
   * @param {AudioParam} targetParam - Target parameter
   * @param {Object} options - Mapping options
   * @param {number} options.min - Minimum value for parameter
   * @param {number} options.max - Maximum value for parameter
   * @param {string} options.curve - Curve type
   * @param {BasePlugin} options.plugin - Source plugin (for reference)
   * @param {string} options.paramName - Parameter name (for reference)
   * @returns {Object} Mapping object
   */
  mapParameter(macroIndex, targetParam, options = {}) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      console.warn(`Invalid macro index: ${macroIndex}`);
      return null;
    }

    if (!(targetParam instanceof AudioParam)) {
      console.warn('Target must be an AudioParam');
      return null;
    }

    const mapping = {
      targetParam,
      min: options.min !== undefined ? options.min : 0,
      max: options.max !== undefined ? options.max : 1,
      curve: options.curve || MacroControls.CURVE_TYPES.LINEAR,
      plugin: options.plugin || null,
      paramName: options.paramName || 'unknown',
      enabled: true
    };

    this._mappings.get(macroIndex).push(mapping);

    // Apply current macro value to the new mapping
    const macroValue = this._macros[macroIndex].value;
    this._applyMappingValue(mapping, macroValue);

    return mapping;
  }

  /**
   * Remove a parameter mapping
   * @param {number} macroIndex - Macro index
   * @param {AudioParam} targetParam - Target parameter to unmap
   * @returns {boolean} True if unmapped successfully
   */
  unmapParameter(macroIndex, targetParam) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      return false;
    }

    const mappings = this._mappings.get(macroIndex);
    const index = mappings.findIndex(m => m.targetParam === targetParam);

    if (index !== -1) {
      mappings.splice(index, 1);
      return true;
    }

    return false;
  }

  /**
   * Clear all mappings for a macro
   * @param {number} macroIndex - Macro index
   */
  clearMappings(macroIndex) {
    if (macroIndex >= 0 && macroIndex < this.macroCount) {
      this._mappings.set(macroIndex, []);
    }
  }

  /**
   * Get all mappings for a macro
   * @param {number} macroIndex - Macro index
   * @returns {Array<Object>} Mappings
   */
  getMappings(macroIndex) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      return [];
    }

    return [...this._mappings.get(macroIndex)];
  }

  /**
   * Update mapped parameters when macro value changes
   * @private
   * @param {number} macroIndex - Macro index
   * @param {number} macroValue - Macro value (0-1)
   * @param {number} time - Time to apply
   */
  _updateMappedParameters(macroIndex, macroValue, time) {
    const mappings = this._mappings.get(macroIndex);

    mappings.forEach(mapping => {
      if (mapping.enabled) {
        this._applyMappingValue(mapping, macroValue, time);
      }
    });
  }

  /**
   * Apply macro value to a mapped parameter
   * @private
   * @param {Object} mapping - Mapping object
   * @param {number} macroValue - Macro value (0-1)
   * @param {number} time - Time to apply
   */
  _applyMappingValue(mapping, macroValue, time = null) {
    const { targetParam, min, max, curve } = mapping;

    // Apply curve
    let scaledValue;

    switch (curve) {
      case MacroControls.CURVE_TYPES.LINEAR:
        scaledValue = min + (max - min) * macroValue;
        break;

      case MacroControls.CURVE_TYPES.EXPONENTIAL:
        // Exponential curve (more change at high values)
        scaledValue = min + (max - min) * Math.pow(macroValue, 2);
        break;

      case MacroControls.CURVE_TYPES.LOGARITHMIC:
        // Logarithmic curve (more change at low values)
        scaledValue = min + (max - min) * Math.sqrt(macroValue);
        break;

      case MacroControls.CURVE_TYPES.S_CURVE:
        // S-curve (smooth acceleration and deceleration)
        const t = macroValue;
        const smoothValue = t * t * (3 - 2 * t); // Smoothstep function
        scaledValue = min + (max - min) * smoothValue;
        break;

      default:
        scaledValue = min + (max - min) * macroValue;
    }

    // Apply to parameter
    const setTime = time !== null ? time : this.audioContext.currentTime;
    targetParam.setValueAtTime(scaledValue, setTime);
  }

  /**
   * Enable learn mode for a macro
   * @param {number} macroIndex - Macro index to learn
   */
  enableLearnMode(macroIndex) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      return;
    }

    this._learnMode = true;
    this._learningMacro = macroIndex;
  }

  /**
   * Disable learn mode
   */
  disableLearnMode() {
    this._learnMode = false;
    this._learningMacro = null;
  }

  /**
   * Learn a parameter (called when user touches a parameter in learn mode)
   * @param {AudioParam} targetParam - Parameter to learn
   * @param {Object} options - Mapping options
   */
  learnParameter(targetParam, options = {}) {
    if (!this._learnMode || this._learningMacro === null) {
      console.warn('Learn mode is not enabled');
      return;
    }

    this.mapParameter(this._learningMacro, targetParam, options);
    console.log(`Learned parameter for Macro ${this._learningMacro + 1}`);

    // Auto-disable learn mode after learning
    this.disableLearnMode();
  }

  /**
   * Map macro to MIDI CC
   * @param {number} macroIndex - Macro index
   * @param {number} ccNumber - MIDI CC number (0-127)
   */
  mapMIDICC(macroIndex, ccNumber) {
    if (macroIndex < 0 || macroIndex >= this.macroCount) {
      return;
    }

    if (ccNumber < 0 || ccNumber > 127) {
      console.warn('MIDI CC must be between 0 and 127');
      return;
    }

    this._macros[macroIndex].midiCC = ccNumber;
  }

  /**
   * Handle MIDI CC input
   * @param {number} ccNumber - MIDI CC number
   * @param {number} value - MIDI CC value (0-127)
   */
  handleMIDICC(ccNumber, value) {
    // Find macro mapped to this CC
    const macro = this._macros.find(m => m.midiCC === ccNumber);

    if (macro) {
      // Convert MIDI value (0-127) to normalized value (0-1)
      const normalizedValue = value / 127;
      this.setMacroValue(macro.index, normalizedValue);
    }
  }

  /**
   * Start automation recording
   */
  startRecording() {
    this._isRecording = true;
    this._recordingStartTime = this.audioContext.currentTime;
    this._automationData.clear();

    // Initialize automation tracks for each macro
    for (let i = 0; i < this.macroCount; i++) {
      this._automationData.set(i, []);
    }
  }

  /**
   * Stop automation recording
   */
  stopRecording() {
    this._isRecording = false;
  }

  /**
   * Record automation point
   * @private
   * @param {number} macroIndex - Macro index
   * @param {number} value - Value
   * @param {number} time - Time
   */
  _recordAutomation(macroIndex, value, time) {
    const relativeTime = time - this._recordingStartTime;
    const track = this._automationData.get(macroIndex);

    if (track) {
      track.push({ time: relativeTime, value });
    }
  }

  /**
   * Playback recorded automation
   * @param {number} startTime - Start time (AudioContext time)
   */
  playbackAutomation(startTime = null) {
    const playbackStart = startTime !== null ? startTime : this.audioContext.currentTime;

    this._automationData.forEach((track, macroIndex) => {
      track.forEach(point => {
        this.setMacroValue(macroIndex, point.value, playbackStart + point.time);
      });
    });
  }

  /**
   * Export automation data
   * @returns {Object} Automation data
   */
  exportAutomation() {
    const data = {};

    this._automationData.forEach((track, macroIndex) => {
      data[`macro${macroIndex}`] = track;
    });

    return data;
  }

  /**
   * Import automation data
   * @param {Object} data - Automation data
   */
  importAutomation(data) {
    this._automationData.clear();

    Object.entries(data).forEach(([key, track]) => {
      const macroIndex = parseInt(key.replace('macro', ''));
      if (macroIndex >= 0 && macroIndex < this.macroCount) {
        this._automationData.set(macroIndex, track);
      }
    });
  }

  /**
   * Reset all macros to default values
   */
  resetAll() {
    for (let i = 0; i < this.macroCount; i++) {
      this.setMacroValue(i, 0);
    }
  }

  /**
   * Get all macro states
   * @returns {Array<Object>} Macro states
   */
  getAllMacros() {
    return this._macros.map(macro => ({
      index: macro.index,
      label: macro.label,
      value: macro.value,
      midiCC: macro.midiCC,
      color: macro.color,
      mappingCount: this._mappings.get(macro.index).length
    }));
  }

  /**
   * Get macro control info
   * @returns {Object} Macro control metadata
   */
  getInfo() {
    return {
      ...super.getInfo(),
      macroCount: this.macroCount,
      learnMode: this._learnMode,
      learningMacro: this._learningMacro,
      recording: this._isRecording,
      totalMappings: Array.from(this._mappings.values()).reduce((sum, m) => sum + m.length, 0)
    };
  }

  /**
   * Cleanup
   */
  dispose() {
    this._macros.forEach(macro => {
      macro.source.stop();
      macro.source.disconnect();
    });

    this._mappings.clear();
    this._automationData.clear();

    super.dispose();
  }
}

// Register with PluginFactory
PluginFactory.register('MacroControls', MacroControls, {
  category: 'modulation',
  description: 'Macro control system with 8 knobs, MIDI mapping, and automation',
  tags: ['macro', 'control', 'modulation', 'midi', 'automation'],
  version: '1.0.0',
  author: 'Agent 17'
});

export default MacroControls;
