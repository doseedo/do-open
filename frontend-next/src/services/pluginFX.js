/**
 * PluginFX Service - Professional Web Audio Effects Chain
 *
 * Replaces tunaFX.js with the comprehensive web-audio-plugins library
 * Manages an 8-slot FX chain with dynamic plugin loading
 *
 * Signal chain: Input → Slot1 → Slot2 → ... → Slot8 → Output
 */

import PluginFactory from '../lib/web-audio-plugins/core/PluginFactory.js';
import { registerAllPlugins } from '../lib/web-audio-plugins/register-all.js';

// Default FX chain configuration (matches previous tunaFX setup)
// Only includes plugins that are confirmed to exist in web-audio-plugins
const DEFAULT_FX_CHAIN = [
  { pluginName: 'Reverb', enabled: true },
  { pluginName: 'SimpleDelay', enabled: true },
  { pluginName: 'Chorus', enabled: true },
  { pluginName: 'Compressor', enabled: true },
  { pluginName: 'EQThree', enabled: true },  // Using EQThree instead of AutoFilter
  { pluginName: 'Phaser', enabled: true },
  { pluginName: null, enabled: false },  // Empty slot
  { pluginName: null, enabled: false },  // Empty slot
];

class PluginFXService {
  constructor() {
    this.audioContext = null;
    this.fxBusInput = null;
    this.fxBusOutput = null;

    // 8 FX slots
    this.slots = new Array(8).fill(null).map((_, index) => ({
      id: index,
      plugin: null,
      pluginName: null,
      enabled: false,
      inputGain: null,
      outputGain: null,
    }));

    this.initialized = false;
    this.pluginsRegistered = false;
  }

  /**
   * Initialize the FX service with an audio context
   * @param {AudioContext} audioContext - Web Audio API context
   * @param {Array} initialChain - Optional initial FX chain configuration
   */
  async initialize(audioContext, initialChain = DEFAULT_FX_CHAIN) {
    if (this.initialized) {
      console.log('🎛️ PluginFX already initialized');
      return;
    }

    this.audioContext = audioContext;

    // Register all plugins with the factory (only once)
    if (!this.pluginsRegistered) {
      try {
        registerAllPlugins();
        this.pluginsRegistered = true;
        console.log('✅ All plugins registered with PluginFactory');
        console.log('📦 Available plugins:', PluginFactory.getPluginNames());
      } catch (error) {
        console.error('❌ Failed to register plugins:', error);
      }
    }

    // Create FX bus input/output nodes
    this.fxBusInput = audioContext.createGain();
    this.fxBusInput.gain.value = 1.0;

    this.fxBusOutput = audioContext.createGain();
    this.fxBusOutput.gain.value = 1.0;

    // Initialize each slot with gain nodes for routing
    this.slots.forEach((slot, index) => {
      slot.inputGain = audioContext.createGain();
      slot.outputGain = audioContext.createGain();
      slot.inputGain.gain.value = 1.0;
      slot.outputGain.gain.value = 1.0;
    });

    // Load initial FX chain
    for (let i = 0; i < Math.min(initialChain.length, 8); i++) {
      const config = initialChain[i];
      if (config.pluginName) {
        await this.setSlot(i, config.pluginName, config.enabled);
      }
    }

    // Connect the chain
    this._connectChain();

    this.initialized = true;
    console.log('🎛️ PluginFX initialized with 8-slot FX chain');
  }

  /**
   * Connect all slots in series
   * @private
   */
  _connectChain() {
    // Disconnect everything first
    this.fxBusInput.disconnect();
    this.slots.forEach(slot => {
      slot.inputGain.disconnect();
      slot.outputGain.disconnect();
      if (slot.plugin) {
        slot.plugin.disconnect();
      }
    });

    // Build the chain: input → slot0 → slot1 → ... → slot7 → output
    let previousNode = this.fxBusInput;

    for (let i = 0; i < this.slots.length; i++) {
      const slot = this.slots[i];

      // Connect previous node to this slot's input
      previousNode.connect(slot.inputGain);

      if (slot.plugin && slot.enabled) {
        // Plugin is active: input → plugin → output
        slot.inputGain.connect(slot.plugin.input);
        slot.plugin.connect(slot.outputGain);
      } else {
        // No plugin or bypassed: direct connection
        slot.inputGain.connect(slot.outputGain);
      }

      previousNode = slot.outputGain;
    }

    // Connect last slot to FX bus output
    previousNode.connect(this.fxBusOutput);

    console.log('🔗 FX chain connected');
  }

  /**
   * Set a plugin in a specific slot
   * @param {number} slotIndex - Slot index (0-7)
   * @param {string} pluginName - Name of the plugin to load
   * @param {boolean} enabled - Whether the plugin is enabled
   * @returns {Object} The created plugin instance
   */
  async setSlot(slotIndex, pluginName, enabled = true) {
    if (slotIndex < 0 || slotIndex >= 8) {
      console.error(`Invalid slot index: ${slotIndex}`);
      return null;
    }

    const slot = this.slots[slotIndex];

    // Dispose of existing plugin
    if (slot.plugin) {
      try {
        slot.plugin.dispose();
      } catch (e) {
        console.warn('Error disposing plugin:', e);
      }
      slot.plugin = null;
    }

    // If no plugin name, clear the slot
    if (!pluginName) {
      slot.pluginName = null;
      slot.enabled = false;
      this._connectChain();
      console.log(`🗑️ Cleared slot ${slotIndex}`);
      return null;
    }

    // Check if plugin exists
    if (!PluginFactory.has(pluginName)) {
      console.error(`Plugin "${pluginName}" not found. Available: ${PluginFactory.getPluginNames().join(', ')}`);
      return null;
    }

    try {
      // Create new plugin instance
      const plugin = PluginFactory.create(pluginName, this.audioContext);

      slot.plugin = plugin;
      slot.pluginName = pluginName;
      slot.enabled = enabled;

      // Reconnect the chain
      this._connectChain();

      console.log(`✅ Slot ${slotIndex}: Loaded ${pluginName} (${enabled ? 'enabled' : 'bypassed'})`);
      return plugin;
    } catch (error) {
      console.error(`❌ Failed to create plugin "${pluginName}":`, error);
      return null;
    }
  }

  /**
   * Clear a slot (remove plugin)
   * @param {number} slotIndex - Slot index (0-7)
   */
  clearSlot(slotIndex) {
    return this.setSlot(slotIndex, null);
  }

  /**
   * Enable/disable (bypass) a slot
   * @param {number} slotIndex - Slot index (0-7)
   * @param {boolean} enabled - Enable state
   */
  setSlotEnabled(slotIndex, enabled) {
    if (slotIndex < 0 || slotIndex >= 8) return;

    const slot = this.slots[slotIndex];
    slot.enabled = enabled;

    if (slot.plugin) {
      slot.plugin.setBypass(!enabled);
    }

    this._connectChain();
    console.log(`🎛️ Slot ${slotIndex}: ${enabled ? 'enabled' : 'bypassed'}`);
  }

  /**
   * Toggle a slot's enabled state
   * @param {number} slotIndex - Slot index (0-7)
   */
  toggleSlot(slotIndex) {
    if (slotIndex < 0 || slotIndex >= 8) return;
    this.setSlotEnabled(slotIndex, !this.slots[slotIndex].enabled);
  }

  /**
   * Get a plugin instance from a slot
   * @param {number} slotIndex - Slot index (0-7)
   * @returns {Object} Plugin instance or null
   */
  getPlugin(slotIndex) {
    if (slotIndex < 0 || slotIndex >= 8) return null;
    return this.slots[slotIndex].plugin;
  }

  /**
   * Get slot info
   * @param {number} slotIndex - Slot index (0-7)
   * @returns {Object} Slot information
   */
  getSlotInfo(slotIndex) {
    if (slotIndex < 0 || slotIndex >= 8) return null;

    const slot = this.slots[slotIndex];
    return {
      id: slot.id,
      pluginName: slot.pluginName,
      enabled: slot.enabled,
      plugin: slot.plugin,
      parameters: slot.plugin ? slot.plugin.getAllParameters() : {},
      parameterConfigs: slot.plugin ? Array.from(slot.plugin.parameters.entries()).map(([name, config]) => ({
        name,
        ...config,
        value: slot.plugin.getParameter(name)
      })) : []
    };
  }

  /**
   * Get all slots info
   * @returns {Array} All slot information
   */
  getAllSlotsInfo() {
    return this.slots.map((_, index) => this.getSlotInfo(index));
  }

  /**
   * Set a parameter on a plugin
   * @param {number} slotIndex - Slot index (0-7)
   * @param {string} paramName - Parameter name
   * @param {number} value - Parameter value
   * @param {number} rampTime - Ramp time in seconds
   */
  setParameter(slotIndex, paramName, value, rampTime = 0) {
    const plugin = this.getPlugin(slotIndex);
    if (plugin) {
      plugin.setParameter(paramName, value, rampTime);
    }
  }

  /**
   * Get available plugins by category
   * @returns {Object} Plugins organized by category
   */
  getAvailablePlugins() {
    const categories = PluginFactory.getCategories();
    const result = {};

    categories.forEach(category => {
      result[category] = PluginFactory.getPluginsByCategory(category);
    });

    return result;
  }

  /**
   * Get all available plugin names
   * @returns {Array} Plugin names
   */
  getPluginNames() {
    return PluginFactory.getPluginNames();
  }

  /**
   * Get plugin info
   * @param {string} pluginName - Plugin name
   * @returns {Object} Plugin metadata
   */
  getPluginInfo(pluginName) {
    return PluginFactory.getPluginInfo(pluginName);
  }

  /**
   * Save current FX chain as preset
   * @param {string} name - Preset name
   * @returns {Object} Chain preset data
   */
  saveChainPreset(name) {
    return {
      name,
      timestamp: Date.now(),
      slots: this.slots.map(slot => ({
        pluginName: slot.pluginName,
        enabled: slot.enabled,
        parameters: slot.plugin ? slot.plugin.getAllParameters() : {}
      }))
    };
  }

  /**
   * Load a chain preset
   * @param {Object} preset - Chain preset data
   */
  async loadChainPreset(preset) {
    if (!preset.slots) return;

    for (let i = 0; i < preset.slots.length && i < 8; i++) {
      const slotData = preset.slots[i];

      if (slotData.pluginName) {
        await this.setSlot(i, slotData.pluginName, slotData.enabled);

        // Restore parameters
        const plugin = this.getPlugin(i);
        if (plugin && slotData.parameters) {
          for (const [name, value] of Object.entries(slotData.parameters)) {
            plugin.setParameter(name, value);
          }
        }
      } else {
        this.clearSlot(i);
      }
    }

    console.log(`✅ Loaded chain preset: ${preset.name}`);
  }

  /**
   * Get the FX bus input node (connect tracks here)
   * @returns {GainNode} Input gain node
   */
  getFXBusInput() {
    return this.fxBusInput;
  }

  /**
   * Get the FX bus output node (connect to master here)
   * @returns {GainNode} Output gain node
   */
  getFXBusOutput() {
    return this.fxBusOutput;
  }

  /**
   * Swap two slots
   * @param {number} slotA - First slot index
   * @param {number} slotB - Second slot index
   */
  swapSlots(slotA, slotB) {
    if (slotA < 0 || slotA >= 8 || slotB < 0 || slotB >= 8) return;

    const tempPlugin = this.slots[slotA].plugin;
    const tempName = this.slots[slotA].pluginName;
    const tempEnabled = this.slots[slotA].enabled;

    this.slots[slotA].plugin = this.slots[slotB].plugin;
    this.slots[slotA].pluginName = this.slots[slotB].pluginName;
    this.slots[slotA].enabled = this.slots[slotB].enabled;

    this.slots[slotB].plugin = tempPlugin;
    this.slots[slotB].pluginName = tempName;
    this.slots[slotB].enabled = tempEnabled;

    this._connectChain();
    console.log(`🔄 Swapped slots ${slotA} and ${slotB}`);
  }

  /**
   * Clean up all resources
   */
  destroy() {
    // Dispose all plugins
    this.slots.forEach(slot => {
      if (slot.plugin) {
        try {
          slot.plugin.dispose();
        } catch (e) {
          console.warn('Error disposing plugin:', e);
        }
      }
      if (slot.inputGain) slot.inputGain.disconnect();
      if (slot.outputGain) slot.outputGain.disconnect();
    });

    if (this.fxBusInput) this.fxBusInput.disconnect();
    if (this.fxBusOutput) this.fxBusOutput.disconnect();

    this.initialized = false;
    console.log('🗑️ PluginFX destroyed');
  }

  // ===== LEGACY COMPATIBILITY METHODS =====
  // These maintain compatibility with existing tunaFX calls

  /**
   * Update reverb parameters (legacy compatibility)
   */
  updateReverb(params) {
    // Find reverb plugin in slots
    const reverbSlot = this.slots.findIndex(s =>
      s.pluginName === 'Reverb' || s.pluginName === 'HybridReverb'
    );

    if (reverbSlot >= 0 && this.slots[reverbSlot].plugin) {
      const plugin = this.slots[reverbSlot].plugin;
      if (params.decay !== undefined) plugin.setParameter('decay', params.decay);
      if (params.roomSize !== undefined) plugin.setParameter('roomSize', params.roomSize);
      if (params.damping !== undefined) plugin.setParameter('damping', params.damping);
      if (params.mix !== undefined) plugin.setParameter('mix', params.mix);
    }
  }

  /**
   * Update delay parameters (legacy compatibility)
   */
  updateDelay(params) {
    const delaySlot = this.slots.findIndex(s =>
      s.pluginName === 'SimpleDelay' || s.pluginName === 'PingPongDelay' || s.pluginName === 'FilterDelay'
    );

    if (delaySlot >= 0 && this.slots[delaySlot].plugin) {
      const plugin = this.slots[delaySlot].plugin;
      if (params.time !== undefined) plugin.setParameter('delayTime', params.time);
      if (params.feedback !== undefined) plugin.setParameter('feedback', params.feedback);
      if (params.mix !== undefined) plugin.setParameter('mix', params.mix);
    }
  }

  /**
   * Update chorus parameters (legacy compatibility)
   */
  updateChorus(params) {
    const chorusSlot = this.slots.findIndex(s => s.pluginName === 'Chorus');

    if (chorusSlot >= 0 && this.slots[chorusSlot].plugin) {
      const plugin = this.slots[chorusSlot].plugin;
      if (params.rate !== undefined) plugin.setParameter('rate', params.rate);
      if (params.depth !== undefined) plugin.setParameter('depth', params.depth);
      if (params.feedback !== undefined) plugin.setParameter('feedback', params.feedback);
    }
  }

  /**
   * Update compressor parameters (legacy compatibility)
   */
  updateCompressor(params) {
    const compSlot = this.slots.findIndex(s =>
      s.pluginName === 'Compressor' || s.pluginName === 'GlueCompressor'
    );

    if (compSlot >= 0 && this.slots[compSlot].plugin) {
      const plugin = this.slots[compSlot].plugin;
      if (params.threshold !== undefined) plugin.setParameter('threshold', params.threshold);
      if (params.ratio !== undefined) plugin.setParameter('ratio', params.ratio);
      if (params.attack !== undefined) plugin.setParameter('attack', params.attack);
      if (params.release !== undefined) plugin.setParameter('release', params.release);
    }
  }

  /**
   * Update filter parameters (legacy compatibility)
   */
  updateFilter(params) {
    const filterSlot = this.slots.findIndex(s =>
      s.pluginName === 'AutoFilter' || s.pluginName === 'EQThree'
    );

    if (filterSlot >= 0 && this.slots[filterSlot].plugin) {
      const plugin = this.slots[filterSlot].plugin;
      if (params.frequency !== undefined) plugin.setParameter('frequency', params.frequency);
      if (params.resonance !== undefined) plugin.setParameter('resonance', params.resonance);
      if (params.gain !== undefined) plugin.setParameter('gain', params.gain);
    }
  }

  /**
   * Update phaser parameters (legacy compatibility)
   */
  updatePhaser(params) {
    const phaserSlot = this.slots.findIndex(s => s.pluginName === 'Phaser');

    if (phaserSlot >= 0 && this.slots[phaserSlot].plugin) {
      const plugin = this.slots[phaserSlot].plugin;
      if (params.rate !== undefined) plugin.setParameter('rate', params.rate);
      if (params.depth !== undefined) plugin.setParameter('depth', params.depth);
      if (params.feedback !== undefined) plugin.setParameter('feedback', params.feedback);
    }
  }
}

// Export singleton instance
export const pluginFX = new PluginFXService();
export default pluginFX;
