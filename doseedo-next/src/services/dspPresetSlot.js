/**
 * DSPPresetSlot — adapter that lets a `dspConfig` (from /plugins/create or
 * from src/lib/dsp-presets/) act as an FX slot inside the studio's
 * pluginFX chain.
 *
 * The same `WebAudioDSPEngine` that powers /plugins/create is reused
 * here, with two adjustments:
 *  - The engine's AudioContext is replaced with the studio's existing
 *    context so all slots and tracks share one clock.
 *  - The engine's auto-connection to `ctx.destination` is severed, since
 *    the studio routes its own master output. The slot exposes
 *    `.input` (the engine's `_chainInput`) and `.output` (the engine's
 *    `masterGain`) for the surrounding pluginFX chain to wire up.
 *
 * Parameter contract:
 *  - `setParameter(paramId, normalizedValue)` → engine.setParameter()
 *  - `getParameters()` returns the dspConfig's parameter array (id,
 *    label, min, max, default, unit) for the FX UI to render controls.
 *  - `getParameterValue(paramId)` returns the *normalized* current value.
 *
 * Lifecycle:
 *  - `dispose()` tears down the engine graph (disconnect/stop all nodes).
 */

import WebAudioDSPEngine from '../audio/WebAudioDSPEngine.js';

export class DSPPresetSlot {
  /**
   * @param {AudioContext} audioContext  — studio's existing context (shared clock)
   * @param {Object} dspConfig           — { dspChain, parameters, routing } JSON
   * @param {Object} [meta]              — optional { name, category, description } for FX picker
   */
  constructor(audioContext, dspConfig, meta = {}) {
    if (!audioContext) throw new Error('DSPPresetSlot: audioContext is required');
    if (!dspConfig)   throw new Error('DSPPresetSlot: dspConfig is required');

    this.meta = meta;
    this.dspConfig = dspConfig;

    // Instantiate the engine and inject the studio's AudioContext BEFORE
    // building the graph, so _ensureContext() doesn't allocate its own.
    this.engine = new WebAudioDSPEngine(dspConfig);
    this.engine.ctx = audioContext;
    this.engine._buildGraph();

    // Sever the engine's auto-routing to ctx.destination — the studio's
    // master bus handles the final output. The chain remains alive in
    // the graph, just routed through our `output` gain node instead.
    try {
      if (this.engine.analyser) this.engine.analyser.disconnect();
    } catch (e) {}

    // The chain endpoints exposed to the studio FX slot:
    //   .input  = first node of the dsp chain (where audio is fed in)
    //   .output = engine.masterGain (last node of the dsp chain)
    this.input  = this.engine._chainInput || this.engine.masterGain;
    this.output = this.engine.masterGain;

    // If something went wrong and there's no chainInput, fall back to a
    // passthrough so the slot doesn't break the studio bus.
    if (!this.input || !this.output) {
      console.warn('[DSPPresetSlot] engine produced no chain endpoints; falling back to passthrough');
      const pass = audioContext.createGain();
      this.input = pass;
      this.output = pass;
    }

    // Apply default param values once so the engine's param bindings
    // are populated even before the FX UI starts moving sliders.
    if (dspConfig.parameters) {
      for (const p of dspConfig.parameters) {
        const norm = (p.default != null && p.max != null && p.min != null && p.max > p.min)
          ? (p.default - p.min) / (p.max - p.min)
          : 0.5;
        this.engine.setParameter(p.id, norm);
      }
    }
  }

  /**
   * Set a parameter by id.
   *
   * Three call styles for compatibility with the old BasePlugin API:
   *   slot.setParameter('size', 75);            // native units
   *   slot.setParameter('size', 75, 0.05);      // native + rampTime (rampTime ignored — engine has its own ramping)
   *   slot.setParameter('size', 0.5, 0, true);  // normalized
   */
  setParameter(paramId, value, _rampTime = 0, normalized = false) {
    if (normalized) {
      this.engine.setParameter(paramId, Math.max(0, Math.min(1, value)));
      return;
    }
    const def = this.dspConfig.parameters?.find(p => p.id === paramId);
    if (!def) {
      console.warn(`[DSPPresetSlot] unknown param ${paramId}`);
      return;
    }
    const norm = def.max > def.min
      ? (value - def.min) / (def.max - def.min)
      : 0;
    this.engine.setParameter(paramId, Math.max(0, Math.min(1, norm)));
  }

  /** Set a parameter by id with a normalized 0..1 value (matches engine API). */
  setParameterNormalized(paramId, normalizedValue) {
    this.engine.setParameter(paramId, Math.max(0, Math.min(1, normalizedValue)));
  }

  /** Returns the parameter schema array for the FX UI. */
  getParameters() {
    return this.dspConfig.parameters || [];
  }

  /** Current normalized value (0..1) for a param id. */
  getParameterValue(paramId) {
    return this.engine.paramValues?.[paramId] ?? 0;
  }

  // ── BasePlugin-compatible surface ─────────────────────────────────
  // The studio's pluginFX.js was written against the old BasePlugin
  // interface. Implementing these methods on DSPPresetSlot lets it
  // drop into the existing FX chain code without per-type branching.

  /** Returns the current native value for a single param. */
  getParameter(paramId) {
    const def = this.dspConfig.parameters?.find(p => p.id === paramId);
    if (!def) return 0;
    const norm = this.engine.paramValues?.[paramId] ?? 0;
    return def.min + norm * (def.max - def.min);
  }

  /** Returns {paramId: nativeValue} for the whole plugin. */
  getAllParameters() {
    const out = {};
    for (const p of this.dspConfig.parameters || []) {
      out[p.id] = this.getParameter(p.id);
    }
    return out;
  }

  /** Connect this slot's output node to a downstream Web Audio node. */
  connect(target) {
    if (!this.output) return;
    return this.output.connect(target);
  }

  /** Disconnect this slot's output node from everything (or one target). */
  disconnect(target) {
    if (!this.output) return;
    try {
      if (target) this.output.disconnect(target);
      else        this.output.disconnect();
    } catch (e) { /* harmless when not connected */ }
  }

  /**
   * Bypass / un-bypass.
   *
   * The studio's pluginFX rebuilds its chain via _connectChain() on
   * every bypass toggle, so all this needs to do is record the flag —
   * the rewiring step takes care of routing input → output direct.
   * (We also internally short the engine's chain via masterGain.gain
   * to mute its tail when bypassed.)
   */
  setBypass(bypassed) {
    this._bypassed = !!bypassed;
    if (this.engine?.masterGain) {
      const t = this.engine.ctx?.currentTime ?? 0;
      this.engine.masterGain.gain.cancelScheduledValues(t);
      this.engine.masterGain.gain.setTargetAtTime(bypassed ? 0 : 0.8, t, 0.005);
    }
  }

  /**
   * Map-like .parameters property used by pluginFX.getSlotInfo() to
   * iterate `[name, config]` pairs. Mirrors BasePlugin.parameters.
   */
  get parameters() {
    const map = new Map();
    for (const p of this.dspConfig.parameters || []) {
      map.set(p.id, {
        min: p.min,
        max: p.max,
        default: p.default,
        unit: p.unit,
        label: p.label,
      });
    }
    return map;
  }

  /** Tear down the engine and free its nodes. */
  dispose() {
    try {
      // Disconnect what we own
      try { this.engine.masterGain?.disconnect(); } catch (e) {}
      // Engine's own teardown handles internal nodes/oscillators
      if (typeof this.engine._teardownGraph === 'function') {
        this.engine._teardownGraph();
      }
    } catch (e) {
      console.warn('[DSPPresetSlot] dispose error', e);
    }
    this.input = null;
    this.output = null;
    this.engine = null;
  }
}
