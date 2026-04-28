/**
 * PluginAdapter — bridge between Logic-plugin records emitted by the
 * desktop sync (logic_engine/logic_sync.py → track.logicPlugins[]) and a
 * live Web Audio graph powered by `WebAudioDSPEngine`.
 *
 * Architecture
 * ────────────
 *   sync JSON track.logicPlugins[i] = {
 *     plugin_id, plugin_name, au_type, au_subtype, au_manufacturer,
 *     parameters: [{id, name, value}, ...], classinfo_xml
 *   }
 *
 * R10's calibration harness emits one mapping JSON per supported plugin
 * to `/plugin-mappings/{plugin_id}.json`. That mapping carries:
 *   - `web_topology`: a dspConfig (dspChain + parameters + routing) that
 *     WebAudioDSPEngine knows how to instantiate
 *   - `param_map[]`: per-Logic-parameter curve fit + domain/range mapping
 *     onto a `web_param` (i.e. an entry in web_topology.parameters)
 *   - `param_map_by_name`: name-based fallback lookup when Logic ids drift
 *
 * Per Logic plugin slot, `instantiate(logicPlugin)`:
 *   1. fetches the mapping (lazy + cached) keyed by plugin_id
 *   2. instantiates a `WebAudioDSPEngine` over the project's shared
 *      AudioContext, severing its destination wiring (same trick as
 *      DSPPresetSlot)
 *   3. applies each Logic parameter through the curve fit to the matching
 *      web param
 *   4. returns `{engine, input, output, setLogicParam, dispose}` — the
 *      caller wires `input` to the upstream node and `output` to the
 *      downstream node, indistinguishable from any other AudioNode
 *
 * Fallback: if no mapping exists, `instantiate` returns `null` and the
 * caller MUST fall back to the bounce-cache audio path (current behaviour).
 *
 * Real-time editing: when a user drags a knob, the studio calls
 * `slot.setLogicParam(logic_id, value)` and the adapter:
 *   - looks up the matching `param_map` row
 *   - runs the value through the curve fit
 *   - converts to the engine's normalized [0, 1] range
 *   - calls `engine.setParameter(web_param_id, normalized)`
 *
 * The engine internally schedules an audio-rate ramp via
 * `setTargetAtTime` so knob drags glide rather than zip.
 */

import WebAudioDSPEngine from '../audio/WebAudioDSPEngine.js';

// ─────────────────────────────────────────────────────────────────────────
// Curve fit evaluators
// ─────────────────────────────────────────────────────────────────────────

/** Convert a Logic-native value to a web-native value via the mapping curve. */
function applyCurve(curve, value, domain, range, breakpoints) {
  const [d0, d1] = domain || [0, 1];
  const [r0, r1] = range  || [0, 1];

  if (curve === 'linear' || !curve) {
    if (d1 === d0) return r0;
    const t = (value - d0) / (d1 - d0);
    return r0 + t * (r1 - r0);
  }

  if (curve === 'log') {
    // domain is the [0,1] knob position, range is the audible target.
    // Logic params often arrive normalized; treat the value as
    // a position in [d0, d1] and map exponentially onto [r0, r1].
    if (d1 === d0 || r0 <= 0 || r1 <= 0) return r0;
    const t = Math.max(0, Math.min(1, (value - d0) / (d1 - d0)));
    return r0 * Math.pow(r1 / r0, t);
  }

  if (curve === 'exp') {
    if (d1 === d0) return r0;
    const t = Math.max(0, Math.min(1, (value - d0) / (d1 - d0)));
    return r0 + (r1 - r0) * (Math.exp(t) - 1) / (Math.E - 1);
  }

  if (curve === 'piecewise' && Array.isArray(breakpoints) && breakpoints.length >= 2) {
    // Linear interpolation between adjacent (logic, web) breakpoints.
    // Breakpoints are assumed sorted by logic value.
    const bp = breakpoints;
    if (value <= bp[0][0]) return bp[0][1];
    if (value >= bp[bp.length - 1][0]) return bp[bp.length - 1][1];
    for (let i = 1; i < bp.length; i++) {
      const [x1, y1] = bp[i];
      const [x0, y0] = bp[i - 1];
      if (value <= x1) {
        if (x1 === x0) return y0;
        const t = (value - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
      }
    }
    return bp[bp.length - 1][1];
  }

  // Unknown curve → identity
  return value;
}

/** Engine params live in normalized [0,1] space. Convert a target value. */
function toNormalized(value, paramDef) {
  if (!paramDef) return Math.max(0, Math.min(1, value));
  const min = paramDef.min ?? 0;
  const max = paramDef.max ?? 1;
  const skew = paramDef.skew || 1;
  if (max === min) return 0.5;
  const linear = (value - min) / (max - min);
  const clamped = Math.max(0, Math.min(1, linear));
  // Inverse of WebAudioDSPEngine.scaleParam: shaped = clamped^skew
  return Math.pow(clamped, skew);
}

// ─────────────────────────────────────────────────────────────────────────
// L/R full split-engine wrapper
// ─────────────────────────────────────────────────────────────────────────

/**
 * Full split-engine routing for `lr_paired`: input ChannelSplitter
 * routes L → engineL.input and R → engineR.input; engine outputs merge
 * back to a stereo pair. Each engine processes one channel
 * independently — knobs targeted via setLeftLogicParam /
 * setRightLogicParam diverge per-channel. Doubles DSP cost; only built
 * when the second engine instance constructed successfully (caller
 * checks). Falls back to `_buildLrPairedWrap` (output-gain split) when
 * the second engine refuses to build.
 *
 * `engines` shape: ``{left: {input, output}, right: {input, output}}``.
 * Returns ``{input, output, dispose}``. Dispose disconnects every node
 * we created here without touching the engines themselves (caller
 * disposes engines separately).
 */
function _buildLrSplitEngineWrap(ctx, engines) {
  if (typeof ctx.createChannelSplitter !== 'function'
      || typeof ctx.createChannelMerger !== 'function') {
    return null;
  }
  let inputSplit, lLeg, rLeg, merger, output;
  try {
    inputSplit = ctx.createChannelSplitter(2);
    // Per-leg gain stages give us a stable handoff between the splitter
    // (mono outputs) and each engine (stereo input expected). Each leg
    // gain is unity; we never touch them at runtime — they exist purely
    // so the splitter's mono channel can connect into a node that the
    // engine's stereo input accepts.
    lLeg = ctx.createGain();
    rLeg = ctx.createGain();
    merger = ctx.createChannelMerger(2);
    output = ctx.createGain();

    inputSplit.connect(lLeg, 0);
    inputSplit.connect(rLeg, 1);
    lLeg.connect(engines.left.input);
    rLeg.connect(engines.right.input);
    engines.left.output.connect(merger, 0, 0);
    engines.right.output.connect(merger, 0, 1);
    merger.connect(output);
  } catch (e) {
    return null;
  }
  // ChannelSplitter is the slot's input; downstream connects flow
  // through its main port (it accepts a stereo input via channel 0).
  // We expose `input` as a passthrough Gain wrapping the splitter so
  // callers' connect/disconnect lifecycle stays predictable.
  const inputGain = ctx.createGain();
  try { inputGain.connect(inputSplit); } catch (e) { /* noop */ }

  return {
    input: inputGain,
    output,
    dispose: () => {
      const ns = [inputGain, inputSplit, lLeg, rLeg, merger, output];
      for (const n of ns) {
        try { n.disconnect(); } catch (_) { /* noop */ }
      }
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────
// L/R-paired routing wrapper
// ─────────────────────────────────────────────────────────────────────────

/**
 * For plugins that expose paired Left/Right knobs (Stereo Delay,
 * dual-mono level controls), `routing.mode === "lr_paired"` is the
 * hint that the engine output should split into independent L/R post-
 * gains. We don't duplicate the *whole* engine — that would double DSP
 * cost — but we DO insert a ChannelSplitter at the output, run each
 * channel through its own gain stage, and remerge. PluginAdapter
 * exposes ``setLeftGain`` / ``setRightGain`` on the slot for paired-
 * level params; rack UI can layout L and R columns and feed them
 * independently. The DSP body is shared, which matches Logic for the
 * common case of "left output level / right output level" being the
 * only differing knobs (Stereo Delay's L/R taps are scheduled inside
 * a shared multitap; their level controls are post-stage gain trims).
 *
 * Returns `{input, output, setLeftGain, setRightGain, dispose}`. The
 * caller wires its `input` ahead of the engine and reads `output` as
 * the slot's user-facing output. setLeftGain/setRightGain accept
 * normalized [0..1] (0=silence, 1=unity, >1=boost up to 4x).
 *
 * Full split-engine processing (each channel through its own
 * WebAudioDSPEngine) is still a Tier-2 follow-up — needed for plugins
 * whose DSP itself differs per-channel (rare). The output-gain split
 * here covers the common cases.
 */
function _buildLrPairedWrap(ctx, engineIn, engineOut) {
  // Engine output → splitter → per-channel gain → merger → wrap output.
  // Engine input is exposed unchanged so the caller's existing
  // ``slot.input.connect(...)`` patterns work.
  let splitter, leftGain, rightGain, merger, output;
  try {
    splitter = ctx.createChannelSplitter(2);
    leftGain = ctx.createGain(); leftGain.gain.value = 1;
    rightGain = ctx.createGain(); rightGain.gain.value = 1;
    merger = ctx.createChannelMerger(2);
    output = ctx.createGain();
    engineOut.connect(splitter);
    splitter.connect(leftGain, 0);
    splitter.connect(rightGain, 1);
    leftGain.connect(merger, 0, 0);
    rightGain.connect(merger, 0, 1);
    merger.connect(output);
  } catch (e) {
    // If the wrap fails for any reason (atypical channel counts,
    // mock AudioContext without splitter/merger, broken graph), fall
    // through to a passthrough so we don't mute audio. The caller
    // still receives the engine output via the regular path.
    return null;
  }

  return {
    input: engineIn,
    output,
    setLeftGain: (v) => {
      const t = (ctx.currentTime || 0) + 0.001;
      try { leftGain.gain.setTargetAtTime(Math.max(0, Math.min(4, v)), t, 0.005); }
      catch { leftGain.gain.value = Math.max(0, Math.min(4, v)); }
    },
    setRightGain: (v) => {
      const t = (ctx.currentTime || 0) + 0.001;
      try { rightGain.gain.setTargetAtTime(Math.max(0, Math.min(4, v)), t, 0.005); }
      catch { rightGain.gain.value = Math.max(0, Math.min(4, v)); }
    },
    dispose: () => {
      const ns = [splitter, leftGain, rightGain, merger, output];
      for (const n of ns) {
        try { n.disconnect(); } catch (_) { /* noop */ }
      }
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────
// M-S routing wrapper
// ─────────────────────────────────────────────────────────────────────────

/**
 * Build a Mid-Side-around-engine wrapper using ChannelSplitter / Merger.
 *
 * The wrap produces an `input` (stereo) and `output` (stereo) pair that
 * encode L/R → M/S, route through the engine's chain, then decode back
 * to L/R. The encode/decode matrix is the standard sum-and-difference
 * with √2 normalization so a round-trip is unity gain.
 *
 * Returns `{input, output, dispose}` — dispose disconnects every node
 * we created here without touching the engine itself.
 */
function _buildMidSideWrap(ctx, engineIn, engineOut) {
  // Bail when the AudioContext doesn't expose channel split/merge —
  // typically a test mock. Caller treats null as "no wrap" and uses
  // the engine output directly.
  if (typeof ctx.createChannelSplitter !== 'function'
      || typeof ctx.createChannelMerger !== 'function') {
    return null;
  }

  // Encoder: L/R → M/S. M = (L+R)/√2; S = (L-R)/√2.
  const inputSplit = ctx.createChannelSplitter(2);
  const negR = ctx.createGain(); negR.gain.value = -1;
  const sumScale = 1 / Math.sqrt(2);
  const midA = ctx.createGain(); midA.gain.value = sumScale;
  const midB = ctx.createGain(); midB.gain.value = sumScale;
  const sideA = ctx.createGain(); sideA.gain.value = sumScale;
  const sideB = ctx.createGain(); sideB.gain.value = sumScale;
  const msMerger = ctx.createChannelMerger(2);
  const input = ctx.createGain();

  input.connect(inputSplit);
  inputSplit.connect(midA, 0); inputSplit.connect(midB, 1);
  inputSplit.connect(sideA, 0); inputSplit.connect(negR, 1);
  negR.connect(sideB);
  midA.connect(msMerger, 0, 0); midB.connect(msMerger, 0, 0);
  sideA.connect(msMerger, 0, 1); sideB.connect(msMerger, 0, 1);

  msMerger.connect(engineIn);

  // Decoder: M/S → L/R. L = (M+S)/√2; R = (M-S)/√2. Same matrix
  // structure as the encoder; we simply swap which side gets negated.
  const outSplit = ctx.createChannelSplitter(2);
  const negS = ctx.createGain(); negS.gain.value = -1;
  const lA = ctx.createGain(); lA.gain.value = sumScale;
  const lB = ctx.createGain(); lB.gain.value = sumScale;
  const rA = ctx.createGain(); rA.gain.value = sumScale;
  const rB = ctx.createGain(); rB.gain.value = sumScale;
  const lrMerger = ctx.createChannelMerger(2);
  const output = ctx.createGain();

  engineOut.connect(outSplit);
  outSplit.connect(lA, 0); outSplit.connect(lB, 1);
  outSplit.connect(rA, 0); outSplit.connect(negS, 1);
  negS.connect(rB);
  lA.connect(lrMerger, 0, 0); lB.connect(lrMerger, 0, 0);
  rA.connect(lrMerger, 0, 1); rB.connect(lrMerger, 0, 1);
  lrMerger.connect(output);

  return {
    input,
    output,
    dispose: () => {
      const nodes = [inputSplit, negR, midA, midB, sideA, sideB, msMerger,
                     outSplit, negS, lA, lB, rA, rB, lrMerger, input, output];
      for (const n of nodes) {
        try { n.disconnect(); } catch (e) { /* noop */ }
      }
    },
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Mapping registry
// ─────────────────────────────────────────────────────────────────────────

const MAPPINGS_BASE = '/plugin-mappings';

/**
 * Resolve a Logic plugin row to its web_param id. Tries:
 *   1. Match by `logic_id`
 *   2. Match by `logic_name` (via param_map[].logic_name OR param_map_by_name)
 */
function resolveParamRow(mapping, logicParam) {
  if (!mapping?.param_map) return null;

  const byId = mapping.param_map.find(
    row => row.logic_id != null && row.logic_id === logicParam.id
  );
  if (byId) return byId;

  const byName = mapping.param_map.find(
    row => row.logic_name && logicParam.name && row.logic_name === logicParam.name
  );
  if (byName) return byName;

  // Last-ditch: param_map_by_name → web_param string lookup
  if (mapping.param_map_by_name && logicParam.name) {
    const webParam = mapping.param_map_by_name[logicParam.name];
    if (webParam) {
      // Fabricate a default linear identity row.
      return { web_param: webParam, curve: 'linear' };
    }
  }
  return null;
}

// ─────────────────────────────────────────────────────────────────────────
// PluginAdapter
// ─────────────────────────────────────────────────────────────────────────

export default class PluginAdapter {
  /**
   * @param {AudioContext|OfflineAudioContext} audioContext  shared studio context
   * @param {Object} [options]
   * @param {Object} [options.mappingsRegistry]  pre-loaded {plugin_id: mapping} map
   *   (skips fetching index.json if provided — useful for tests/SSR)
   * @param {string} [options.basePath='/plugin-mappings']
   * @param {boolean} [options.strictMode=false]  if true, instantiate throws
   *   when any plugin lacks a mapping. Default false → silent fallback.
   * @param {(url: string) => Promise<Response>} [options.fetchImpl=globalThis.fetch]
   */
  constructor(audioContext, options = {}) {
    if (!audioContext) throw new Error('PluginAdapter: audioContext required');
    this.ctx = audioContext;
    this.basePath = options.basePath || MAPPINGS_BASE;
    this.strictMode = !!options.strictMode;
    // strictSampleRate: when true, mappings whose expected_sample_rate
    // disagrees with the AudioContext's sampleRate refuse to load (the
    // adapter returns null, caller falls back to bounce-cache audio).
    // Default false → log a console.warn and proceed; matches existing
    // behaviour. Useful for plugins where SR matters audibly
    // (oversampled limiters, IR-based reverbs) and silent miscalibration
    // is worse than no audio.
    this.strictSampleRate = !!options.strictSampleRate;
    // `fetchImpl: null` is honored as "no fetch" — useful for tests and
    // pre-seeded registries where a network roundtrip would be wrong.
    this.fetchImpl = (options.fetchImpl !== undefined)
      ? options.fetchImpl
      : (typeof fetch !== 'undefined' ? fetch.bind(globalThis) : null);

    // Edit callbacks: invoked AFTER a local plugin edit lands so the
    // studio can broadcast the same edit to peers via the session
    // edit-log. Optional — when absent, edits stay local. Carries the
    // identifying triple (trackUuid, slot, paramId) which the studio
    // resolves into an enqueueSetPluginParam call. We deliberately
    // keep this side at "fire callback with metadata"; transport
    // (sessionEditsAPI) lives in the app, not in the adapter.
    this.editCallbacks = options.editCallbacks || null;

    // plugin_id → mapping JSON (or null = miss tombstone, don't refetch)
    this.mappings = new Map();
    if (options.mappingsRegistry) {
      for (const [k, v] of Object.entries(options.mappingsRegistry)) {
        this.mappings.set(String(k), v);
      }
    }

    // Available plugin_ids (loaded from index.json or seeded by registry).
    this.available = new Set(this.mappings.keys());

    this._indexLoaded = false;
    this._inflight = new Map(); // plugin_id → Promise<mapping|null>

    // Cache of instantiated engines keyed by `${plugin_id}|${param_hash}`.
    // Reuses the engine + graph if the same plugin/parameter snapshot is
    // requested twice (e.g. duplicate tracks).
    this._engineCache = new Map();
  }

  /**
   * Load the mapping registry index. Safe to call multiple times — only
   * the first call hits the network.
   */
  async load() {
    if (this._indexLoaded) return;
    this._indexLoaded = true;
    if (!this.fetchImpl) return; // no fetch available (tests can pre-seed)
    try {
      const res = await this.fetchImpl(`${this.basePath}/index.json`);
      if (!res.ok) {
        // No registry → adapter degenerates to "always fall back"; perfectly fine.
        return;
      }
      const data = await res.json();
      const ids = data?.mappings || [];
      for (const id of ids) this.available.add(String(id));
    } catch (err) {
      // Network error → behave as if registry is empty.
      console.warn('[PluginAdapter] index load failed:', err?.message || err);
    }
  }

  /** Returns true if a mapping is known to exist for this plugin_id. */
  hasMapping(pluginId) {
    return this.available.has(String(pluginId));
  }

  /**
   * Lazy-fetch a single mapping JSON by plugin_id. De-dupes concurrent
   * loads and caches both hits and misses.
   */
  async _loadMapping(pluginId) {
    const id = String(pluginId);
    if (this.mappings.has(id)) return this.mappings.get(id);
    if (this._inflight.has(id)) return this._inflight.get(id);
    if (!this.fetchImpl) {
      this.mappings.set(id, null);
      return null;
    }
    // Only consult the network if `index.json` told us the mapping exists.
    // Avoids warn-spam every time a Logic plugin shows up that isn't in the
    // registry — overwhelmingly the common case until R10 lands more fits.
    if (this._indexLoaded && !this.available.has(id)) {
      this.mappings.set(id, null);
      return null;
    }
    const p = (async () => {
      try {
        const res = await this.fetchImpl(`${this.basePath}/${id}.json`);
        if (!res.ok) {
          this.mappings.set(id, null);
          return null;
        }
        const data = await res.json();
        this.mappings.set(id, data);
        this.available.add(id);
        return data;
      } catch (err) {
        console.warn(`[PluginAdapter] mapping ${id} fetch failed:`, err?.message || err);
        this.mappings.set(id, null);
        return null;
      } finally {
        this._inflight.delete(id);
      }
    })();
    this._inflight.set(id, p);
    return p;
  }

  /**
   * Instantiate a live web DSP slot for a Logic plugin record.
   * Returns null if no mapping is available (caller must fall back).
   * Rejects if `strictMode` is set and the mapping is missing.
   *
   * @param {Object} logicPlugin
   * @returns {Promise<null | {
   *   engine: WebAudioDSPEngine,
   *   input: AudioNode,
   *   output: AudioNode,
   *   setLogicParam: (idOrName: number|string, value: number) => void,
   *   getMapping: () => Object,
   *   dispose: () => void,
   * }>}
   */
  async instantiate(logicPlugin, editContext = null) {
    if (!logicPlugin?.plugin_id) return null;
    const id = String(logicPlugin.plugin_id);

    if (!this.mappings.has(id) && this.fetchImpl) {
      await this._loadMapping(id);
    }
    const mapping = this.mappings.get(id);
    if (!mapping || !mapping.web_topology) {
      if (this.strictMode) {
        throw new Error(`PluginAdapter: no mapping for plugin_id=${id} (strict mode)`);
      }
      return null;
    }

    return this._buildSlotFromMapping(mapping, logicPlugin, editContext);
  }

  /**
   * Synchronous variant — used when the mapping is already known to be
   * loaded (e.g. by `load()` + a pre-fetch). Returns null if not cached.
   */
  instantiateSync(logicPlugin, editContext = null) {
    if (!logicPlugin?.plugin_id) return null;
    const id = String(logicPlugin.plugin_id);
    const mapping = this.mappings.get(id);
    if (!mapping || !mapping.web_topology) {
      if (this.strictMode) {
        throw new Error(`PluginAdapter: no mapping for plugin_id=${id} (strict mode)`);
      }
      return null;
    }
    return this._buildSlotFromMapping(mapping, logicPlugin, editContext);
  }

  _buildSlotFromMapping(mapping, logicPlugin, editContext = null) {
    const id = String(logicPlugin.plugin_id);
    const dspConfig = mapping.web_topology;

    // Sample-rate sanity check. Most Logic plugins behave fine across
    // 44.1k/48k but oversampling-aware compressors and IR-based reverbs
    // change behaviour with the host SR. Default policy: warn and
    // proceed — slightly-off audio beats no audio. With
    // ``strictSampleRate``: refuse to load, caller falls back to
    // bounce-cache.
    const expectedSR = mapping.expected_sample_rate;
    if (expectedSR && this.ctx?.sampleRate && expectedSR !== this.ctx.sampleRate) {
      const msg =
        `[PluginAdapter] sample-rate mismatch for ${mapping.plugin_name || id}: ` +
        `mapping calibrated at ${expectedSR}Hz but AudioContext is ` +
        `${this.ctx.sampleRate}Hz`;
      if (this.strictSampleRate) {
        if (this.strictMode) {
          throw new Error(msg + ' (strictSampleRate)');
        }
        console.warn(msg + ' — refusing to load (strictSampleRate)');
        return null;
      }
      console.warn(msg);
    }

    // Build the primary engine, inject our shared context, then build graph.
    const engine = new WebAudioDSPEngine(dspConfig);
    engine.ctx = this.ctx;
    engine._buildGraph();

    // Sever destination so the caller controls routing. The engine still
    // wires masterGain → analyser → ctx.destination internally, but we
    // expose masterGain as our output and additionally disconnect the
    // analyser → destination link to prevent double-summing into the
    // studio's master bus.
    try {
      if (engine.analyser) engine.analyser.disconnect();
    } catch (e) { /* noop */ }

    const engineIn  = engine._chainInput || engine.masterGain;
    const engineOut = engine.masterGain;

    // For lr_paired plugins, build a SECOND engine instance that
    // processes only the right channel. The two engines start with
    // identical params; setLeftLogicParam / setRightLogicParam diverge
    // them on operator demand. setLogicParam (no L/R suffix) dispatches
    // to both so the common case (a knob that should affect both
    // channels equally) Just Works.
    const wantLrPair = (dspConfig.routing?.mode || 'stereo') === 'lr_paired';
    let engineRight = null;
    let engineRightIn = null;
    let engineRightOut = null;
    if (wantLrPair && engineIn && engineOut) {
      try {
        engineRight = new WebAudioDSPEngine(dspConfig);
        engineRight.ctx = this.ctx;
        engineRight._buildGraph();
        try { if (engineRight.analyser) engineRight.analyser.disconnect(); } catch (_) { /* noop */ }
        engineRightIn  = engineRight._chainInput || engineRight.masterGain;
        engineRightOut = engineRight.masterGain;
      } catch (e) {
        console.warn('[PluginAdapter] lr_paired second-engine build failed; falling back to shared-engine wrap:', e?.message || e);
        engineRight = null;
      }
    }

    let input;
    let output;
    let msNodes = null;
    let lrNodes = null;
    if (!engineIn || !engineOut) {
      // Engine produced nothing usable — return a passthrough so the
      // caller's track chain stays intact.
      const pass = this.ctx.createGain();
      input = pass;
      output = pass;
    } else if ((dspConfig.routing?.mode || 'stereo') === 'ms') {
      // Mid-Side wrap: encode L/R → M/S, run engine on the pair, decode
      // M/S → L/R. Mid = (L+R)/√2, Side = (L-R)/√2; the inverse uses the
      // same scaling so M-S then S-M is unity. We chain ChannelSplitter
      // and ChannelMerger nodes around the engine: this is the cheapest
      // M-S codec the WebAudio API offers and is sample-accurate.
      msNodes = _buildMidSideWrap(this.ctx, engineIn, engineOut);
      input = msNodes?.input || engineIn;
      output = msNodes?.output || engineOut;
    } else if (wantLrPair && engineRight) {
      // Full split-engine: input ChannelSplitter routes L → engineL,
      // R → engineR; engine outputs merge into the slot output. Doubles
      // DSP cost but lets paired params (Left Delay vs. Right Delay)
      // diverge per channel.
      lrNodes = _buildLrSplitEngineWrap(
        this.ctx,
        { left: { input: engineIn, output: engineOut },
          right: { input: engineRightIn, output: engineRightOut } },
      );
      input = lrNodes?.input || engineIn;
      output = lrNodes?.output || engineOut;
    } else if (wantLrPair && !engineRight) {
      // Second engine refused to build — fall back to the simpler
      // output-gain split. setLeftGain / setRightGain still work; the
      // setLeftLogicParam / setRightLogicParam paths become no-ops
      // (only one engine to dispatch to), which we surface via
      // hasIndependentLrEngines=false on the slot below.
      lrNodes = _buildLrPairedWrap(this.ctx, engineIn, engineOut);
      input = lrNodes?.input || engineIn;
      output = lrNodes?.output || engineOut;
    } else {
      input = engineIn;
      output = engineOut;
    }

    // Bypass plumbing: lazy. Until the user toggles bypass, the chain
    // is a single linear path (input → engine → output) — same topology
    // as before, so existing graph walkers (offline-render harnesses,
    // analyzers) see no surprise. On first setBypassed(true) we splice a
    // parallel passthrough alongside the engine and crossfade between
    // them. The cost of laziness is a 30ms one-shot click on the very
    // first toggle of every plugin's lifetime — acceptable.
    let bypassWired = false;
    let passthrough = null;
    let enginePathGain = null;
    let bypassSumOut = null;

    // Build an index for parameter dispatch
    const webParamDefs = {};
    for (const p of (dspConfig.parameters || [])) webParamDefs[p.id] = p;

    // Apply each Logic parameter through its curve fit
    const initialParams = logicPlugin.parameters || [];
    for (const lp of initialParams) {
      this._applyLogicParam(engine, mapping, webParamDefs, lp);
    }

    // Bind a fast lookup by Logic id and Logic name for the live setter.
    const rowByLogicId = new Map();
    const rowByLogicName = new Map();
    for (const row of (mapping.param_map || [])) {
      if (row.logic_id != null) rowByLogicId.set(row.logic_id, row);
      if (row.logic_name) rowByLogicName.set(row.logic_name, row);
    }

    // Per-engine dispatcher — the body of setLogicParam minus the
    // broadcast hook. Used both directly (single-engine plugins) and
    // mirrored to engineLeft + engineRight for lr_paired splits.
    const dispatchToEngine = (eng, row, value) => {
      if (row.indexed) {
        const intValue = Math.round(Number(value) || 0);
        if (typeof eng.setIndexedParameter === 'function') {
          eng.setIndexedParameter(row.web_param, intValue);
        } else {
          eng.setParameter(row.web_param, intValue);
        }
        return;
      }
      const webParamDef = webParamDefs[row.web_param];
      const webValue = applyCurve(row.curve, value, row.domain, row.range, row.breakpoints);
      const norm = toNormalized(webValue, webParamDef);
      eng.setParameter(row.web_param, norm);
    };

    const setLogicParam = (idOrName, value, opts = {}) => {
      let row = (typeof idOrName === 'number')
        ? rowByLogicId.get(idOrName)
        : rowByLogicName.get(idOrName);

      // Fallback to param_map_by_name string-lookup
      if (!row && typeof idOrName === 'string' && mapping.param_map_by_name) {
        const webParam = mapping.param_map_by_name[idOrName];
        if (webParam) row = { web_param: webParam, curve: 'linear' };
      }
      if (!row) return false;

      // Broadcast the edit to the session log unless the caller passed
      // ``broadcast: false`` — typically because they're applying an
      // *inbound* edit from a peer and don't want to echo it back. The
      // logical id we emit is the AU's logic_id (stable across sessions),
      // not the resolved row's web_param.
      const shouldBroadcast = opts.broadcast !== false
        && this.editCallbacks?.onParamEdit
        && editContext?.trackUuid != null
        && editContext?.slotIndex != null
        && row.logic_id != null
        && !row.indexed; // indexed params get their own broadcast path
      if (shouldBroadcast) {
        try {
          this.editCallbacks.onParamEdit({
            trackUuid: editContext.trackUuid,
            slot: editContext.slotIndex,
            paramId: row.logic_id,
            value: Number(value),
          });
        } catch (e) {
          console.warn('[PluginAdapter] onParamEdit threw:', e?.message || e);
        }
      }

      // Dispatch to primary engine, plus the right-channel engine when
      // the lr_paired split built one — both stay in sync until the
      // operator calls setLeftLogicParam / setRightLogicParam to
      // diverge them.
      dispatchToEngine(engine, row, value);
      if (engineRight) dispatchToEngine(engineRight, row, value);
      return true;
    };

    // Per-channel param targeting for lr_paired plugins. setLeft/Right
    // operate on a single engine without echoing to the other; ideal
    // for "Left Delay Time" / "Right Delay Time" knob pairs. When the
    // slot doesn't have an independent right engine (single-engine
    // fallback), setRight* maps onto the shared engine so the value
    // still has SOME effect — the L/R divergence just collapses.
    const setLeftLogicParam = (idOrName, value, opts = {}) => {
      const row = rowByLogicId.get(idOrName) || rowByLogicName.get(idOrName);
      if (!row) return false;
      if (opts.broadcast !== false
          && this.editCallbacks?.onParamEdit
          && editContext?.trackUuid != null
          && editContext?.slotIndex != null
          && row.logic_id != null && !row.indexed) {
        try {
          this.editCallbacks.onParamEdit({
            trackUuid: editContext.trackUuid,
            slot: editContext.slotIndex,
            paramId: row.logic_id,
            value: Number(value),
            channel: 'left',
          });
        } catch (e) { /* noop */ }
      }
      dispatchToEngine(engine, row, value);
      return true;
    };
    const setRightLogicParam = (idOrName, value, opts = {}) => {
      const row = rowByLogicId.get(idOrName) || rowByLogicName.get(idOrName);
      if (!row) return false;
      if (opts.broadcast !== false
          && this.editCallbacks?.onParamEdit
          && editContext?.trackUuid != null
          && editContext?.slotIndex != null
          && row.logic_id != null && !row.indexed) {
        try {
          this.editCallbacks.onParamEdit({
            trackUuid: editContext.trackUuid,
            slot: editContext.slotIndex,
            paramId: row.logic_id,
            value: Number(value),
            channel: 'right',
          });
        } catch (e) { /* noop */ }
      }
      dispatchToEngine(engineRight || engine, row, value);
      return true;
    };

    // Bypass control: lazy-splice the parallel passthrough on first use.
    // ``slot.output`` reference may need updating once we splice — we
    // capture the current `output` in a closure-mutable local so callers
    // who already grabbed `slot.output` see the spliced node next time.
    let bypassed = false;
    let userOutput = output;
    const _wireBypass = () => {
      if (bypassWired) return;
      const ctx = this.ctx;
      passthrough = ctx.createGain(); passthrough.gain.value = 0;
      enginePathGain = ctx.createGain(); enginePathGain.gain.value = 1;
      bypassSumOut = ctx.createGain(); bypassSumOut.gain.value = 1;
      try {
        // Insert enginePathGain between the current `output` and a new
        // summing gain. Keep the existing connections out of `output`
        // intact — we don't disconnect what callers may already have
        // wired downstream; we just route an additional sum node.
        userOutput.connect(enginePathGain);
        enginePathGain.connect(bypassSumOut);
        input.connect(passthrough);
        passthrough.connect(bypassSumOut);
        // Patch the slot.output to point at the summing node. Callers
        // that did slot.output.connect(...) before the first toggle keep
        // their connection from the old node; new calls go through the
        // sum.
        slot.output = bypassSumOut;
        bypassWired = true;
      } catch (e) {
        // Splice failed — leave bypass a no-op rather than mute audio.
        bypassWired = false;
      }
    };
    const setBypassed = (b, opts = {}) => {
      const target = !!b;
      if (target === bypassed) return;
      bypassed = target;
      if (!bypassWired) _wireBypass();
      if (!bypassWired) return;
      const t = (this.ctx?.currentTime || 0) + 0.001;
      const fade = 0.030;
      try {
        enginePathGain.gain.setTargetAtTime(target ? 0 : 1, t, fade / 4);
        passthrough.gain.setTargetAtTime(target ? 1 : 0, t, fade / 4);
      } catch (e) {
        enginePathGain.gain.value = target ? 0 : 1;
        passthrough.gain.value = target ? 1 : 0;
      }
      // Broadcast unless suppressed (peer echo). Bypass at the AU layer
      // is a host-set property; the desktop dispatcher routes it to
      // set_plugin_bypass via doo_hook so Logic mirrors the toggle.
      if (opts.broadcast !== false
          && this.editCallbacks?.onBypassChange
          && editContext?.trackUuid != null
          && editContext?.slotIndex != null) {
        try {
          this.editCallbacks.onBypassChange({
            trackUuid: editContext.trackUuid,
            slot: editContext.slotIndex,
            bypassed: target,
          });
        } catch (e) {
          console.warn('[PluginAdapter] onBypassChange threw:', e?.message || e);
        }
      }
    };
    const isBypassed = () => bypassed;

    const dispose = () => {
      try {
        if (slot.output && typeof slot.output.disconnect === 'function') slot.output.disconnect();
      } catch (e) { /* noop */ }
      try { if (passthrough) passthrough.disconnect(); } catch (e) { /* noop */ }
      try { if (enginePathGain) enginePathGain.disconnect(); } catch (e) { /* noop */ }
      try { if (bypassSumOut) bypassSumOut.disconnect(); } catch (e) { /* noop */ }
      try { if (msNodes) msNodes.dispose(); } catch (e) { /* noop */ }
      try { if (lrNodes) lrNodes.dispose(); } catch (e) { /* noop */ }
      try { engine._teardownGraph(); } catch (e) { /* noop */ }
      try { if (engineRight) engineRight._teardownGraph(); } catch (e) { /* noop */ }
    };

    const slot = {
      pluginId: id,
      pluginName: logicPlugin.plugin_name || mapping.plugin_name,
      engine,
      input,
      output,
      setLogicParam,
      setBypassed,
      isBypassed,
      // Convenience flags so the React layer can decide whether to render
      // a bypass button / Mix knob / dropdown UI without a re-fetch.
      bypassSupported: mapping.bypass_supported !== false,
      hasWetParam: typeof mapping.wet_param_id === 'number',
      routingMode: dspConfig.routing?.mode || 'stereo',
      // Paired-level setters surface on the slot only when the wrap
      // actually built. Calling them when the wrap was a no-op is a
      // silent no-op rather than a crash — UI can call them
      // unconditionally without checking routingMode first.
      setLeftGain: (v) => { lrNodes?.setLeftGain?.(v); },
      setRightGain: (v) => { lrNodes?.setRightGain?.(v); },
      // Per-channel param dispatchers for lr_paired splits. setLogicParam
      // continues to update both engines symmetrically; these are the
      // diverge knobs (Stereo Delay's Left Delay vs. Right Delay).
      setLeftLogicParam,
      setRightLogicParam,
      // True when an independent right-channel engine actually built.
      // Lets paired-knob UI know whether L/R divergence is real or
      // collapses onto the shared single engine.
      hasIndependentLrEngines: !!engineRight,
      getMapping: () => mapping,
      dispose,
    };
    return slot;
  }

  _applyLogicParam(engine, mapping, webParamDefs, logicParam) {
    const row = resolveParamRow(mapping, logicParam);
    if (!row) return;
    if (row.indexed) {
      const intValue = Math.round(Number(logicParam.value) || 0);
      if (typeof engine.setIndexedParameter === 'function') {
        engine.setIndexedParameter(row.web_param, intValue);
      } else {
        engine.setParameter(row.web_param, intValue);
      }
      return;
    }
    const webParamDef = webParamDefs[row.web_param];
    const webValue = applyCurve(row.curve, logicParam.value, row.domain, row.range, row.breakpoints);
    const norm = toNormalized(webValue, webParamDef);
    engine.setParameter(row.web_param, norm);
  }

  /**
   * Build a complete track-playback chain:
   *   audioInputNode → [plugin1] → [plugin2] → ... → outputNode
   *
   * Returns { input, output, slots, fallback, dispose } where:
   *   - `input` is the AudioNode the caller pipes the track's source into
   *   - `output` is the AudioNode the caller pipes onward (bus / master)
   *   - `slots` is the array of instantiated PluginAdapter slot objects
   *   - `fallback` is `true` if any plugin lacked a mapping AND
   *     strict_mode is false → caller should play bounce audio for this
   *     track instead of feeding source through this chain
   *   - `dispose` releases all engines + disconnects nodes
   *
   * If `track.logicPlugins` is empty, returns a pure passthrough chain
   * with `fallback=false`. If strict mode is set, throws on any miss.
   */
  async buildTrackChain(track) {
    const ctx = this.ctx;
    const input = ctx.createGain();
    const output = ctx.createGain();
    input.gain.value = 1;
    output.gain.value = 1;

    const logicPlugins = track?.logicPlugins || [];
    if (logicPlugins.length === 0) {
      input.connect(output);
      return {
        input, output, slots: [],
        fallback: false,
        dispose: () => {
          try { input.disconnect(); } catch (e) {}
          try { output.disconnect(); } catch (e) {}
        },
      };
    }

    const slots = [];
    let cursor = input;
    let fallback = false;
    // Edit-broadcast plumbing: passing trackUuid + slotIndex into each
    // instantiate call lets setLogicParam / setBypassed fire the
    // adapter's editCallbacks with the address every time the user
    // touches a knob or bypass button. trackUuid comes from the synced
    // shape; slotIndex is just the position in logicPlugins[].
    const trackUuid = track?.uuid || track?.metadata?.uuid || null;

    for (let slotIndex = 0; slotIndex < logicPlugins.length; slotIndex++) {
      const lp = logicPlugins[slotIndex];
      const slot = await this.instantiate(lp, { trackUuid, slotIndex });
      if (!slot) {
        fallback = true;
        // Stop building further — caller will use bounce audio for the
        // whole track. Tear down anything we already built.
        for (const s of slots) {
          try { s.dispose(); } catch (e) {}
        }
        try { input.disconnect(); } catch (e) {}
        try { output.disconnect(); } catch (e) {}
        return {
          input: null, output: null, slots: [],
          fallback: true,
          missingPluginId: String(lp.plugin_id || ''),
          dispose: () => {},
        };
      }
      try { cursor.connect(slot.input); } catch (e) { /* noop */ }
      cursor = slot.output;
      slots.push(slot);
    }

    try { cursor.connect(output); } catch (e) { /* noop */ }

    const dispose = () => {
      for (const s of slots) {
        try { s.dispose(); } catch (e) {}
      }
      try { input.disconnect(); } catch (e) {}
      try { output.disconnect(); } catch (e) {}
    };

    return { input, output, slots, fallback: false, dispose };
  }

  /** Drop all cached mappings + engines (e.g. on logout / project switch). */
  clearCache() {
    for (const slot of this._engineCache.values()) {
      try { slot.dispose(); } catch (e) {}
    }
    this._engineCache.clear();
  }
}

// Export curve helpers separately so unit tests + R10's calibration
// harness can reuse them without instantiating the full adapter.
export { applyCurve, toNormalized, resolveParamRow };
