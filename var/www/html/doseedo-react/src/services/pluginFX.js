/**
 * PluginFX Service - Professional Web Audio Effects Chain
 *
 * 8-slot FX chain. Each slot now hosts a DSPPresetSlot instance, which
 * is a thin adapter over WebAudioDSPEngine — the same engine that
 * powers /plugins/create. This unifies the plugin format: built-in
 * studio presets and user-authored plugins use the same runtime, the
 * same parameter system, and the same loader.
 *
 * Legacy BasePlugin / PluginFactory is still imported as a fallback
 * for any historical plugin name not yet in the DSP-preset registry,
 * so existing FX chain saves don't break.
 *
 * Signal chain: Input → Slot1 → Slot2 → ... → Slot8 → Output
 */

import PluginFactory from '../lib/web-audio-plugins/core/PluginFactory.js';
import { registerAllPlugins } from '../lib/web-audio-plugins/register-all.js';
import { DSP_PRESETS } from '../lib/dsp-presets/index.js';
import { DSPPresetSlot } from './dspPresetSlot.js';
import { createPluginSlot, resolvePluginRef } from './userPluginLoader.js';

// Default FX chain — uses the new DSP-preset registry. Each entry is
// either a string (= preset key in DSP_PRESETS) or a {source, id} ref
// (for user/community plugins). Strings are resolved against
// DSP_PRESETS first, then fall back to PluginFactory by name.
const DEFAULT_FX_CHAIN = [
  { plugin: 'reverb',       enabled: true },
  { plugin: 'simpleDelay',  enabled: true },
  { plugin: 'chorus',       enabled: true },
  { plugin: 'compressor',   enabled: true },
  { plugin: 'parametricEQ', enabled: true },
  { plugin: 'phaser',       enabled: true },
  { plugin: null,           enabled: false },
  { plugin: null,           enabled: false },
];

/**
 * Map a legacy PascalCase plugin name (e.g. 'Reverb', 'SimpleDelay')
 * to the corresponding DSP preset key. Returns null if no match.
 */
function legacyNameToPresetKey(name) {
  if (!name || typeof name !== 'string') return null;
  // exact match first
  if (DSP_PRESETS[name]) return name;
  // case-insensitive match
  const lower = name.toLowerCase();
  for (const k of Object.keys(DSP_PRESETS)) {
    if (k.toLowerCase() === lower) return k;
  }
  // PascalCase → camelCase: 'SimpleDelay' → 'simpleDelay'
  const camel = name.charAt(0).toLowerCase() + name.slice(1);
  if (DSP_PRESETS[camel]) return camel;
  return null;
}

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

    // Load initial FX chain. Each entry uses the new `plugin` key
    // (either a preset key string or a {source,id} ref); accept the
    // legacy `pluginName` key as a fallback for old saved chains.
    for (let i = 0; i < Math.min(initialChain.length, 8); i++) {
      const config = initialChain[i];
      const ref = config.plugin ?? config.pluginName ?? null;
      if (ref) {
        await this.setSlot(i, ref, config.enabled);
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
   * Set a plugin in a specific slot.
   *
   * Three input shapes for `pluginRef`:
   *   1. A preset key from src/lib/dsp-presets         e.g.  'reverb'
   *   2. A user/community plugin ref                    e.g.  {source:'user', id:'abc-123'}
   *   3. (Legacy) a PascalCase web-audio-plugins name   e.g.  'Reverb', 'EQThree'
   *
   * Resolution order:
   *   - Object → resolvePluginRef() (DSP-lang path, async fetch if user/community)
   *   - String → DSP-preset registry (case-insensitive, with PascalCase
   *              → camelCase fallback). Hits the new path.
   *   - String not in DSP_PRESETS → fallback to legacy PluginFactory
   *              so historical FX chain saves keep working.
   *
   * Both paths produce something with .input/.output/.setParameter/.dispose,
   * so the chain wiring code below doesn't care which one is in the slot.
   *
   * @param {number} slotIndex
   * @param {string|Object} pluginRef
   * @param {boolean} enabled
   * @returns {Object|null} the slot's plugin instance (DSPPresetSlot or BasePlugin)
   */
  async setSlot(slotIndex, pluginRef, enabled = true) {
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

    // Clear-slot path
    if (!pluginRef) {
      slot.pluginName = null;
      slot.pluginRef = null;
      slot.kind = null;
      slot.enabled = false;
      this._connectChain();
      console.log(`🗑️ Cleared slot ${slotIndex}`);
      return null;
    }

    let plugin = null;
    let displayName = null;
    let kind = null;

    // ── Path 1: object ref (user / community / explicit builtin) ──
    if (typeof pluginRef === 'object' && pluginRef !== null) {
      try {
        plugin = await createPluginSlot(this.audioContext, pluginRef);
        displayName = plugin?.meta?.name || `${pluginRef.source}:${pluginRef.id}`;
        kind = 'dsp';
      } catch (e) {
        console.error(`❌ Failed to load plugin ref`, pluginRef, e);
        return null;
      }
    } else {
      // ── Path 2: string — try DSP-preset registry first ─────────
      const presetKey = legacyNameToPresetKey(pluginRef);
      if (presetKey) {
        try {
          plugin = await createPluginSlot(this.audioContext, {
            source: 'builtin',
            id: presetKey,
          });
          displayName = plugin?.meta?.name || presetKey;
          kind = 'dsp';
        } catch (e) {
          console.error(`❌ Failed to instantiate DSP preset "${presetKey}":`, e);
        }
      }

      // ── Path 3: legacy fallback to PluginFactory ───────────────
      if (!plugin && PluginFactory.has(pluginRef)) {
        try {
          plugin = PluginFactory.create(pluginRef, this.audioContext);
          displayName = pluginRef;
          kind = 'legacy';
        } catch (e) {
          console.error(`❌ Failed to create legacy plugin "${pluginRef}":`, e);
          return null;
        }
      }

      if (!plugin) {
        console.error(
          `Plugin "${pluginRef}" not found in DSP presets or PluginFactory. ` +
          `DSP presets: ${Object.keys(DSP_PRESETS).join(', ')}. ` +
          `Legacy: ${PluginFactory.getPluginNames().join(', ')}.`
        );
        return null;
      }
    }

    slot.plugin = plugin;
    slot.pluginName = displayName;
    slot.pluginRef = pluginRef;
    slot.kind = kind;
    slot.enabled = enabled;

    this._connectChain();
    console.log(`✅ Slot ${slotIndex}: Loaded ${displayName} (${kind}, ${enabled ? 'enabled' : 'bypassed'})`);
    return plugin;
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
   * Get slot info. Works uniformly for both DSPPresetSlot and legacy
   * BasePlugin instances (DSPPresetSlot exposes a compatible
   * .getAllParameters() and a Map-like .parameters getter).
   *
   * @param {number} slotIndex
   * @returns {Object|null}
   */
  getSlotInfo(slotIndex) {
    if (slotIndex < 0 || slotIndex >= 8) return null;

    const slot = this.slots[slotIndex];
    if (!slot.plugin) {
      return {
        id: slot.id,
        pluginName: slot.pluginName,
        pluginRef: slot.pluginRef,
        kind: slot.kind,
        enabled: slot.enabled,
        plugin: null,
        parameters: {},
        parameterConfigs: [],
      };
    }
    let parameterConfigs = [];
    try {
      parameterConfigs = Array.from(slot.plugin.parameters.entries()).map(([name, config]) => ({
        name,
        ...config,
        value: slot.plugin.getParameter(name),
      }));
    } catch (e) {
      console.warn('[pluginFX] failed to read slot params:', e);
    }
    return {
      id: slot.id,
      pluginName: slot.pluginName,
      pluginRef: slot.pluginRef,
      kind: slot.kind,
      enabled: slot.enabled,
      plugin: slot.plugin,
      parameters: slot.plugin.getAllParameters(),
      parameterConfigs,
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
   * Get available plugins by category. Now sources from BOTH the new
   * DSP-preset registry (preferred) and the legacy PluginFactory.
   *
   * @returns {Object} { category: [{ key, name, description, source }] }
   */
  getAvailablePlugins() {
    const result = {};
    // 1. DSP presets first (new system)
    for (const [key, preset] of Object.entries(DSP_PRESETS)) {
      const cat = preset.category || 'misc';
      if (!result[cat]) result[cat] = [];
      result[cat].push({
        key,
        name: preset.name,
        description: preset.description,
        source: 'builtin',
      });
    }
    // 2. Legacy PluginFactory entries that DO NOT shadow a DSP preset
    try {
      const legacyCategories = PluginFactory.getCategories();
      for (const cat of legacyCategories) {
        const legacy = PluginFactory.getPluginsByCategory(cat) || [];
        for (const item of legacy) {
          const name = typeof item === 'string' ? item : item.name;
          if (legacyNameToPresetKey(name)) continue; // already covered by DSP preset
          if (!result[cat]) result[cat] = [];
          result[cat].push({
            key: name,
            name,
            description: PluginFactory.getPluginInfo(name)?.description,
            source: 'legacy',
          });
        }
      }
    } catch (e) { /* legacy registry empty / not loaded */ }
    return result;
  }

  /** All available plugin names (DSP presets + legacy fallback). */
  getPluginNames() {
    const names = new Set(Object.keys(DSP_PRESETS));
    try {
      for (const n of PluginFactory.getPluginNames()) names.add(n);
    } catch (e) {}
    return Array.from(names);
  }

  /**
   * Get plugin metadata by name (preset key OR legacy name).
   */
  getPluginInfo(pluginName) {
    const presetKey = legacyNameToPresetKey(pluginName);
    if (presetKey) {
      const p = DSP_PRESETS[presetKey];
      return {
        name: p.name,
        category: p.category,
        description: p.description,
        source: 'builtin',
      };
    }
    return PluginFactory.getPluginInfo(pluginName);
  }

  /**
   * Save current FX chain as preset. Persists `pluginRef` (the original
   * ref or preset key handed to setSlot) so user-authored plugins reload
   * from the chatbot-backend correctly.
   */
  saveChainPreset(name) {
    return {
      name,
      timestamp: Date.now(),
      version: 2,
      slots: this.slots.map(slot => ({
        pluginRef: slot.pluginRef ?? slot.pluginName ?? null,
        pluginName: slot.pluginName,  // for human readability + legacy compat
        kind: slot.kind,
        enabled: slot.enabled,
        parameters: slot.plugin ? slot.plugin.getAllParameters() : {},
      })),
    };
  }

  /**
   * Load a chain preset. Accepts both v1 (legacy) and v2 (new) shapes.
   */
  async loadChainPreset(preset) {
    if (!preset.slots) return;

    for (let i = 0; i < preset.slots.length && i < 8; i++) {
      const slotData = preset.slots[i];
      const ref = slotData.pluginRef ?? slotData.pluginName ?? null;

      if (ref) {
        await this.setSlot(i, ref, slotData.enabled);
        const plugin = this.getPlugin(i);
        if (plugin && slotData.parameters) {
          for (const [pname, value] of Object.entries(slotData.parameters)) {
            try { plugin.setParameter(pname, value); }
            catch (e) { /* harmless: param not present in this build */ }
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
  // These maintain compatibility with existing tunaFX calls. They were
  // written against the old web-audio-plugins parameter names. With
  // DSP-preset slots the names are slightly different, so we try a
  // remap table first and fall through to the original name if needed.

  /**
   * Try to set a parameter using a list of candidate names so legacy
   * call sites work whether the slot is hosting a DSP-preset slot or a
   * legacy BasePlugin. Stops at the first param that the plugin
   * actually exposes.
   */
  _setParamCompat(plugin, candidateNames, value) {
    if (!plugin || value === undefined) return;
    for (const name of candidateNames) {
      try {
        // Skip silently if the param doesn't exist on this plugin
        const exists = (plugin.parameters && plugin.parameters.has?.(name))
                    || (plugin.getParameter && plugin.getParameter(name) !== undefined);
        if (exists) {
          plugin.setParameter(name, value);
          return;
        }
      } catch (e) { /* try next candidate */ }
    }
    // Last resort — just try the first candidate name and let it warn
    try { plugin.setParameter(candidateNames[0], value); }
    catch (e) {}
  }

  /** Update reverb parameters (legacy compatibility) */
  updateReverb(params) {
    const reverbSlot = this.slots.findIndex(s =>
      /reverb/i.test(s.pluginName || '')
    );
    if (reverbSlot < 0 || !this.slots[reverbSlot].plugin) return;
    const plugin = this.slots[reverbSlot].plugin;
    this._setParamCompat(plugin, ['decayTime', 'decay'],     params.decay);
    this._setParamCompat(plugin, ['size', 'roomSize'],       params.roomSize);
    this._setParamCompat(plugin, ['damping'],                params.damping);
    this._setParamCompat(plugin, ['mix'],                    params.mix);
  }

  /** Update delay parameters (legacy compatibility) */
  updateDelay(params) {
    const slot = this.slots.findIndex(s => /delay|echo/i.test(s.pluginName || ''));
    if (slot < 0 || !this.slots[slot].plugin) return;
    const plugin = this.slots[slot].plugin;
    this._setParamCompat(plugin, ['time', 'delayTime'], params.time);
    this._setParamCompat(plugin, ['feedback'],          params.feedback);
    this._setParamCompat(plugin, ['mix'],               params.mix);
  }

  /** Update chorus parameters (legacy compatibility) */
  updateChorus(params) {
    const slot = this.slots.findIndex(s => /chorus/i.test(s.pluginName || ''));
    if (slot < 0 || !this.slots[slot].plugin) return;
    const plugin = this.slots[slot].plugin;
    this._setParamCompat(plugin, ['rate'],     params.rate);
    this._setParamCompat(plugin, ['depth'],    params.depth);
    this._setParamCompat(plugin, ['feedback'], params.feedback);
  }

  /** Update compressor parameters (legacy compatibility) */
  updateCompressor(params) {
    const slot = this.slots.findIndex(s => /compress/i.test(s.pluginName || ''));
    if (slot < 0 || !this.slots[slot].plugin) return;
    const plugin = this.slots[slot].plugin;
    this._setParamCompat(plugin, ['threshold'], params.threshold);
    this._setParamCompat(plugin, ['ratio'],     params.ratio);
    this._setParamCompat(plugin, ['attack'],    params.attack);
    this._setParamCompat(plugin, ['release'],   params.release);
  }

  /** Update filter / EQ parameters (legacy compatibility) */
  updateFilter(params) {
    const slot = this.slots.findIndex(s =>
      /filter|eq/i.test(s.pluginName || '')
    );
    if (slot < 0 || !this.slots[slot].plugin) return;
    const plugin = this.slots[slot].plugin;
    this._setParamCompat(plugin, ['frequency', 'cutoff'], params.frequency);
    this._setParamCompat(plugin, ['resonance', 'q'],      params.resonance);
    this._setParamCompat(plugin, ['gain'],                params.gain);
  }

  /** Update phaser parameters (legacy compatibility) */
  updatePhaser(params) {
    const slot = this.slots.findIndex(s => /phaser/i.test(s.pluginName || ''));
    if (slot < 0 || !this.slots[slot].plugin) return;
    const plugin = this.slots[slot].plugin;
    this._setParamCompat(plugin, ['rate'],     params.rate);
    this._setParamCompat(plugin, ['depth'],    params.depth);
    this._setParamCompat(plugin, ['feedback'], params.feedback);
  }
}

// Export singleton instance
export const pluginFX = new PluginFXService();
export default pluginFX;
