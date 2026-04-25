/**
 * ModRouter — runtime modulation matrix for Doseedo's Web Audio engine.
 *
 * Resolves "nodeId.paramKey" target strings on `lfo`, `macro`, and `mod_envelope`
 * "ghost" nodes (io: {in:0, out:0}) into live parameter updates against the
 * builtNodes graph produced by WebAudioDSPEngine.
 *
 * Three routing strategies are picked per-target:
 *   - audio-rate (a-rate): when the target exposes an AudioParam, we feed the
 *     LFO/envelope OutputGain straight into it via AudioNode.connect(audioParam).
 *     Web Audio sums modulator output with the param's value. We insert an
 *     intermediate `scaling GainNode` whose .gain holds (amount × paramRange)
 *     so the modulator's [-1..+1] (or [0..1]) sweeps the desired range.
 *   - control-rate (k-rate): when the target only has a `customSetter`
 *     (e.g. waveshaper drive curve, reverb decay IR regen), we sample the
 *     modulator's value at ~60 Hz via setInterval and call the setter.
 *   - immediate (one-shot): macros push values once on setMacroValue() —
 *     no continuous modulation needed.
 *
 * Bipolar / unipolar conventions:
 *   - LFO output is **bipolar** ([-1..+1]) by construction (OscillatorNode +
 *     unipolar shaper if needed). depth ∈ [0,1] scales the swing.
 *   - mod_envelope output is **unipolar** ([0..1] following an ADSR), then
 *     amount ∈ [-1,+1] flips/scales it.
 *   - macro value is **unipolar** ([0..1]); per-target amount ∈ [-1,+1].
 *
 * BPM sync:
 *   - LFO with sync_to_bpm=true derives rate_hz from current BPM and sync_rate.
 *     Calling setBPM() updates all synced LFOs.
 *
 * The "ghost" node problem:
 *   - lfo / macro / mod_envelope have io {in:0, out:0} so they don't appear in
 *     audio edges. ModRouter scans `dspGraph.nodes` separately, looks them up
 *     in `builtNodes` (where they may exist as inert placeholders) and replaces
 *     or augments them with the modulation infrastructure constructed here.
 *
 * Author: Agent R7 (Modulation Matrix)
 */

// We re-implement target-string parsing locally (see parseTarget below) to
// avoid a hard dependency on dspNodeDefinitions / modulationUtils. This keeps
// ModRouter usable from offline-render unit tests (no React stack required).

// ── Constants ──────────────────────────────────────────────────────────────

const MOD_NODE_TYPES = new Set(['lfo', 'macro', 'mod_envelope', 'midi_cc']);
const K_RATE_INTERVAL_MS = 16; // ~60 Hz update rate for k-rate setters

// Map sync_rate strings → multiplier (cycles per beat)
// 1/4 = quarter note = 1 cycle per beat at 60 BPM → rate_hz = bpm/60
const SYNC_RATE_MAP = {
  '1/16': 4, '1/8': 2, '1/4': 1, '1/2': 0.5,
  '1': 0.25, '2': 0.125, '4': 0.0625,
};

// ── Local target string resolver ───────────────────────────────────────────
//   Avoids a circular dep on dspNodeDefinitions; equivalent to
//   `resolveModTarget` in that file.

function parseTarget(targetStr) {
  if (!targetStr || typeof targetStr !== 'string') return null;
  if (!targetStr.includes('.')) return null;
  const dot = targetStr.indexOf('.');
  const nodeId = targetStr.slice(0, dot);
  const paramKey = targetStr.slice(dot + 1);
  if (!nodeId || !paramKey) return null;
  return { nodeId, paramKey };
}

// Map a node's paramKey (from dspNodeDefinitions) to the AudioParam attached
// to a built node via WebAudioDSPEngine's builders. Builders only expose
// AudioParams that were bound to a `@global_param` string — so for direct
// node-to-node routing we have to introspect the actual AudioNode types.
//
// We accept the convention that builtNodes[id].input / .output are AudioNodes
// with well-known fields (BiquadFilter has .frequency/.Q/.gain, GainNode has
// .gain, DelayNode has .delayTime, DynamicsCompressor has threshold/ratio/etc).

function resolveAudioParamOnBuiltNode(builtNode, paramKey) {
  if (!builtNode) return null;
  // 1. paramTargets is the canonical map for params bound via @paramId; check
  //    if this paramKey was published there. (Unlikely for direct-targeting
  //    but supported.)
  if (builtNode.paramTargets) {
    for (const t of Object.values(builtNode.paramTargets)) {
      if (t.audioParam && t._paramKey === paramKey) {
        return { audioParam: t.audioParam, paramDef: t.paramDef, scale: t.scale };
      }
    }
  }
  // 2. Walk the input/output AudioNode for matching fields. Try input first
  //    (most filters/gain nodes share input==output anyway).
  const candidates = [builtNode.input, builtNode.output];
  for (const n of candidates) {
    if (!n) continue;
    const ap = pickAudioParam(n, paramKey);
    if (ap) return { audioParam: ap, paramDef: null, scale: null };
  }
  // 3. Multi-stage built nodes (ladder, multitap delay) may stash audioParams
  //    in builtNode.audioParams — convention.
  if (builtNode.audioParams && builtNode.audioParams[paramKey]) {
    return { audioParam: builtNode.audioParams[paramKey], paramDef: null, scale: null };
  }
  // 4. customSetter fallback for paramKey — keyed by paramKey on the builtNode.
  if (builtNode.customSetters && typeof builtNode.customSetters[paramKey] === 'function') {
    return { customSetter: builtNode.customSetters[paramKey], paramDef: null };
  }
  return null;
}

function pickAudioParam(audioNode, paramKey) {
  // BiquadFilterNode
  if (audioNode.frequency && audioNode.Q) {
    if (paramKey === 'cutoff' || paramKey === 'frequency' || paramKey === 'freq')
      return audioNode.frequency;
    if (paramKey === 'resonance' || paramKey === 'q' || paramKey === 'Q')
      return audioNode.Q;
    if (paramKey === 'gain' && audioNode.gain) return audioNode.gain;
  }
  // GainNode
  if (audioNode.gain && !audioNode.frequency) {
    if (paramKey === 'gain' || paramKey === 'level' || paramKey === 'volume')
      return audioNode.gain;
  }
  // DelayNode
  if (audioNode.delayTime) {
    if (paramKey === 'time' || paramKey === 'delay_time' || paramKey === 'time_ms' || paramKey === 'delay_ms')
      return audioNode.delayTime;
  }
  // StereoPannerNode
  if (audioNode.pan && !audioNode.frequency) {
    if (paramKey === 'pan' || paramKey === 'position') return audioNode.pan;
  }
  // DynamicsCompressorNode
  if (audioNode.threshold && audioNode.ratio) {
    if (paramKey === 'threshold') return audioNode.threshold;
    if (paramKey === 'ratio') return audioNode.ratio;
    if (paramKey === 'attack') return audioNode.attack;
    if (paramKey === 'release') return audioNode.release;
    if (paramKey === 'knee') return audioNode.knee;
  }
  // OscillatorNode
  if (audioNode.frequency && audioNode.detune && !audioNode.Q) {
    if (paramKey === 'rate' || paramKey === 'rate_hz' || paramKey === 'frequency')
      return audioNode.frequency;
    if (paramKey === 'detune') return audioNode.detune;
  }
  return null;
}

// Compute the linear-domain "range" used to scale a [-1..+1] modulator into
// the param's range. paramDef is the dspNodeDefinitions schema entry (may be
// null — we fall back to AudioParam.minValue/maxValue or sensible defaults).
function getRange(paramDef, audioParam) {
  if (paramDef && paramDef.min != null && paramDef.max != null) {
    return paramDef.max - paramDef.min;
  }
  if (audioParam && isFinite(audioParam.maxValue) && isFinite(audioParam.minValue)) {
    return Math.min(audioParam.maxValue - audioParam.minValue, 1e6); // cap for sanity
  }
  return 1; // default unit-range
}

// ── ModRouter ──────────────────────────────────────────────────────────────

export default class ModRouter {
  /**
   * @param {BaseAudioContext} ctx
   * @param {object} dspGraph - {nodes:[{id,type,params}], edges:[]}
   * @param {object} builtNodes - {nodeId: {input, output, paramTargets, ...}}
   * @param {object} paramDefs - global parameter definitions map
   * @param {object} [opts]
   * @param {object} [opts.nodeSchema] - map of node-type → schema (params)
   *                                     so we can look up min/max for paramKeys.
   * @param {number} [opts.bpm=120]
   */
  constructor(ctx, dspGraph, builtNodes, paramDefs, opts = {}) {
    this.ctx = ctx;
    this.dspGraph = dspGraph || { nodes: [], edges: [] };
    this.builtNodes = builtNodes || {};
    this.paramDefs = paramDefs || {};
    this.nodeSchema = opts.nodeSchema || {};
    this.bpm = opts.bpm || 120;

    // Tracking
    this._lfos = new Map();           // nodeId → { osc, depthGain, scaleGains:[], cfg }
    this._macros = new Map();         // nodeId → { value, targets:[{audioParam, scaleGain, customSetter, amount, range, baseValue}] }
    this._modEnvs = new Map();        // nodeId → { cfg, target:{...}, currentValue, scheduledEnd, gateOn }
    this._kRateUpdaters = [];         // [{ stop:Function }] — interval handles

    this._resolved = false;
  }

  /**
   * Walk the graph and wire up modulation routing.
   * Idempotent: calling resolveTargets() twice tears down + rebuilds.
   */
  resolveTargets() {
    if (this._resolved) this.dispose();
    this._resolved = true;

    for (const nodeDef of (this.dspGraph.nodes || [])) {
      if (!MOD_NODE_TYPES.has(nodeDef.type)) continue;
      try {
        switch (nodeDef.type) {
          case 'lfo':          this._buildLFO(nodeDef);         break;
          case 'macro':        this._buildMacro(nodeDef);       break;
          case 'mod_envelope': this._buildModEnvelope(nodeDef); break;
          case 'midi_cc':      this._buildMidiCC(nodeDef);      break;
          default: break;
        }
      } catch (err) {
        // Don't let one bad mod node break the rest of the graph.
        // eslint-disable-next-line no-console
        console.warn(`[ModRouter] failed to resolve node ${nodeDef.id} (${nodeDef.type}):`, err);
      }
    }
  }

  // ── LFO ──────────────────────────────────────────────────────────────────

  _buildLFO(nodeDef) {
    const params = nodeDef.params || {};
    const targetStr = params.target || '';
    const target = parseTarget(targetStr);
    if (!target) {
      console.warn(`[ModRouter] LFO ${nodeDef.id} has no/invalid target: "${targetStr}"`);
      return;
    }
    const tgtBuilt = this.builtNodes[target.nodeId];
    if (!tgtBuilt) {
      console.warn(`[ModRouter] LFO ${nodeDef.id} → unknown target node "${target.nodeId}"`);
      return;
    }
    const resolved = resolveAudioParamOnBuiltNode(tgtBuilt, target.paramKey);
    if (!resolved) {
      console.warn(
        `[ModRouter] LFO ${nodeDef.id} → cannot resolve param "${target.paramKey}" on ${target.nodeId}`,
      );
      return;
    }

    // Build LFO source: OscillatorNode for sine/triangle/saw/square, AudioWorklet
    // (or fallback to ScriptProcessor) for 'random' / S&H. To stay zero-dep here
    // we approximate 'random' as a sample-and-held white noise source via a
    // BufferSource looping a coarse random buffer.
    const shape = (params.shape || 'sine').toLowerCase();
    const rateHz = this._lfoRateHz(params);
    const depth = clamp(params.depth ?? 0.5, 0, 1);

    const { source, isRandom } = this._buildLFOSource(shape, rateHz, params.phase || 0);

    // depthGain scales the bipolar [-1..+1] LFO by depth (0..1).
    const depthGain = this.ctx.createGain();
    depthGain.gain.value = depth;
    source.connect(depthGain);

    const cfg = {
      source,
      depthGain,
      rateParam: source.frequency || null,
      shape,
      isRandom,
      depth,
      sync: !!params.sync_to_bpm,
      syncRate: params.sync_rate || '1/4',
      manualRateHz: rateHz,
      target,
      tgtBuilt,
      resolved,
      scaleGain: null,
    };

    if (resolved.audioParam) {
      // a-rate routing
      const range = getRange(this._lookupParamDef(target, resolved), resolved.audioParam);
      const scaleGain = this.ctx.createGain();
      // Bipolar LFO sweeps ±(range/2) × depth around the param's current value.
      // (range/2 because LFO output is in [-1..+1], not [0..1].)
      scaleGain.gain.value = range / 2;
      depthGain.connect(scaleGain);
      try {
        scaleGain.connect(resolved.audioParam);
      } catch (e) {
        console.warn(`[ModRouter] LFO ${nodeDef.id} could not connect to audioParam:`, e);
        return;
      }
      cfg.scaleGain = scaleGain;
    } else if (resolved.customSetter) {
      // k-rate routing — sample LFO via AnalyserNode
      const analyser = this.ctx.createAnalyser();
      analyser.fftSize = 256;
      depthGain.connect(analyser);
      const buf = new Float32Array(analyser.fftSize);
      const paramDef = this._lookupParamDef(target, resolved);
      const range = getRange(paramDef, null);
      const baseMin = paramDef?.min ?? 0;
      const updater = setInterval(() => {
        analyser.getFloatTimeDomainData(buf);
        const sample = buf[buf.length - 1] || 0; // last sample
        // sample is ~bipolar [-1..1]*depth; map to absolute param value:
        //   value = base + sample * range/2  (range/2 because bipolar swing).
        const value = baseMin + range / 2 + sample * range / 2;
        try { resolved.customSetter(value); } catch (e) { /* swallow */ }
      }, K_RATE_INTERVAL_MS);
      this._kRateUpdaters.push({ stop: () => clearInterval(updater) });
    }

    // Start oscillators
    if (typeof source.start === 'function') {
      try { source.start(); } catch (e) { /* already started */ }
    }

    this._lfos.set(nodeDef.id, cfg);
  }

  _buildLFOSource(shape, rateHz, phaseDeg) {
    // OscillatorNode handles sine/sawtooth/square/triangle natively.
    // 'random' = sample-and-hold: build a buffer of stepped random values
    // looping at rateHz.
    if (shape === 'random' || shape === 'sample_hold' || shape === 's&h') {
      const sr = this.ctx.sampleRate;
      const periodSamples = Math.max(2, Math.floor(sr / Math.max(rateHz, 0.01)));
      const stepsPerPeriod = 32; // # of S&H steps per "cycle"
      const stepLen = Math.max(1, Math.floor(periodSamples / stepsPerPeriod));
      const totalLen = stepLen * stepsPerPeriod;
      const buf = this.ctx.createBuffer(1, totalLen, sr);
      const data = buf.getChannelData(0);
      for (let s = 0; s < stepsPerPeriod; s++) {
        const v = Math.random() * 2 - 1; // bipolar
        const start = s * stepLen;
        for (let i = 0; i < stepLen; i++) data[start + i] = v;
      }
      const src = this.ctx.createBufferSource();
      src.buffer = buf;
      src.loop = true;
      // Fake .frequency for sync compat (no-op)
      src.frequency = { value: rateHz, setTargetAtTime: () => {} };
      return { source: src, isRandom: true };
    }
    const osc = this.ctx.createOscillator();
    osc.type = shape === 'saw' ? 'sawtooth' : shape;
    osc.frequency.value = rateHz;
    // Phase: WebAudio doesn't expose phase directly; approximate by delaying start.
    if (phaseDeg) {
      const offsetSec = (phaseDeg / 360) / Math.max(rateHz, 0.001);
      // schedule start in the past (won't actually go negative — harmless on AudioContext)
      try { osc.start(this.ctx.currentTime, offsetSec); } catch (e) { /* not all impls support */ }
    }
    return { source: osc, isRandom: false };
  }

  _lfoRateHz(params) {
    if (params.sync_to_bpm) {
      const mult = SYNC_RATE_MAP[params.sync_rate] ?? 1;
      return (this.bpm / 60) * mult;
    }
    return params.rate_hz ?? params.rate ?? 1;
  }

  _lookupParamDef(target, resolved) {
    if (resolved && resolved.paramDef) return resolved.paramDef;
    // Look up the target node's schema to find the param min/max.
    const tgtNodeDef = (this.dspGraph.nodes || []).find(n => n.id === target.nodeId);
    if (!tgtNodeDef) return null;
    const schema = this.nodeSchema[tgtNodeDef.type];
    if (schema && schema.params && schema.params[target.paramKey]) {
      return schema.params[target.paramKey];
    }
    return null;
  }

  // ── Macro ────────────────────────────────────────────────────────────────

  _buildMacro(nodeDef) {
    const params = nodeDef.params || {};
    const value = clamp(params.value ?? 0.5, 0, 1);
    const cfg = { value, targets: [] };

    for (let i = 1; i <= 8; i++) {
      const tStr = params[`target_${i}`];
      const amt = params[`amount_${i}`];
      if (tStr == null && amt == null && i > 2) break; // only iterate beyond the schema's 1/2 if present
      if (!tStr) continue;
      const t = parseTarget(tStr);
      if (!t) continue;
      const tgtBuilt = this.builtNodes[t.nodeId];
      if (!tgtBuilt) {
        console.warn(`[ModRouter] Macro ${nodeDef.id} target_${i} unknown node "${t.nodeId}"`);
        continue;
      }
      const resolved = resolveAudioParamOnBuiltNode(tgtBuilt, t.paramKey);
      if (!resolved) {
        console.warn(`[ModRouter] Macro ${nodeDef.id} target_${i} unresolvable param "${t.paramKey}" on ${t.nodeId}`);
        continue;
      }
      const paramDef = this._lookupParamDef(t, resolved);
      const range = getRange(paramDef, resolved.audioParam);
      const baseMin = paramDef?.min ?? 0;
      cfg.targets.push({
        target: t,
        amount: clamp(amt ?? 1, -1, 1),
        range,
        baseMin,
        audioParam: resolved.audioParam,
        customSetter: resolved.customSetter,
        baseValue: resolved.audioParam ? resolved.audioParam.value : baseMin,
      });
    }

    this._macros.set(nodeDef.id, cfg);
    // Apply initial value
    this._applyMacro(nodeDef.id);
  }

  _applyMacro(nodeId) {
    const cfg = this._macros.get(nodeId);
    if (!cfg) return;
    const t = this.ctx.currentTime;
    for (const m of cfg.targets) {
      // unipolar value × bipolar amount × range → offset added to base
      const offset = (cfg.value) * m.amount * m.range;
      const newValue = m.baseValue + offset;
      if (m.audioParam) {
        try {
          m.audioParam.cancelScheduledValues(t);
          m.audioParam.setTargetAtTime(newValue, t, 0.01);
        } catch (e) { /* offline ctx in test: setTargetAtTime is fine */ }
      } else if (m.customSetter) {
        try { m.customSetter(newValue); } catch (e) { /* swallow */ }
      }
    }
  }

  setMacroValue(nodeId, value) {
    const cfg = this._macros.get(nodeId);
    if (!cfg) return;
    cfg.value = clamp(value, 0, 1);
    this._applyMacro(nodeId);
  }

  // ── Mod Envelope ─────────────────────────────────────────────────────────

  _buildModEnvelope(nodeDef) {
    const params = nodeDef.params || {};
    const targetStr = params.target || '';
    const target = parseTarget(targetStr);
    if (!target) return;
    const tgtBuilt = this.builtNodes[target.nodeId];
    if (!tgtBuilt) {
      console.warn(`[ModRouter] ModEnvelope ${nodeDef.id} unknown target node "${target.nodeId}"`);
      return;
    }
    const resolved = resolveAudioParamOnBuiltNode(tgtBuilt, target.paramKey);
    if (!resolved) {
      console.warn(`[ModRouter] ModEnvelope ${nodeDef.id} unresolvable param "${target.paramKey}"`);
      return;
    }
    const paramDef = this._lookupParamDef(target, resolved);
    const range = getRange(paramDef, resolved.audioParam);

    const cfg = {
      target, resolved, paramDef, range,
      attackMs: params.attack_ms ?? 10,
      decayMs: params.decay_ms ?? 300,
      sustain: clamp(params.sustain ?? 0.7, 0, 1),
      releaseMs: params.release_ms ?? 500,
      amount: clamp(params.amount ?? 1, -1, 1),
      baseValue: resolved.audioParam ? resolved.audioParam.value : (paramDef?.min ?? 0),
      gateOn: false,
      // For k-rate setters: we step the envelope value in JS.
      _kRateInterval: null,
      _kRateStartTime: 0,
      _kRateState: 'idle',
    };
    this._modEnvs.set(nodeDef.id, cfg);
  }

  /**
   * Trigger or release a mod envelope.
   * @param {string} nodeId
   * @param {boolean} gateOn  true=note-on, false=note-off
   */
  triggerModEnvelope(nodeId, gateOn) {
    const cfg = this._modEnvs.get(nodeId);
    if (!cfg) return;
    const t = this.ctx.currentTime;
    const { resolved, range, amount, baseValue } = cfg;
    const peak = baseValue + range * amount;          // attack target
    const sustainVal = baseValue + range * amount * cfg.sustain;
    const A = cfg.attackMs / 1000;
    const D = cfg.decayMs / 1000;
    const R = cfg.releaseMs / 1000;

    if (resolved.audioParam) {
      const ap = resolved.audioParam;
      ap.cancelScheduledValues(t);
      if (gateOn) {
        ap.setValueAtTime(baseValue, t);
        ap.linearRampToValueAtTime(peak, t + Math.max(0.001, A));
        ap.linearRampToValueAtTime(sustainVal, t + Math.max(0.001, A) + Math.max(0.001, D));
        cfg.gateOn = true;
      } else {
        // Release from current value (best-effort: read .value)
        const cur = (typeof ap.value === 'number') ? ap.value : sustainVal;
        ap.setValueAtTime(cur, t);
        ap.linearRampToValueAtTime(baseValue, t + Math.max(0.001, R));
        cfg.gateOn = false;
      }
    } else if (resolved.customSetter) {
      // k-rate: drive the setter via setInterval over A/D/R.
      if (cfg._kRateInterval) { clearInterval(cfg._kRateInterval); cfg._kRateInterval = null; }
      const startT = nowMs();
      cfg._kRateStartTime = startT;
      cfg._kRateState = gateOn ? 'attack' : 'release';
      const releaseStartValue = cfg._kRateLastValue ?? baseValue;
      cfg._kRateInterval = setInterval(() => {
        const elapsed = (nowMs() - startT) / 1000;
        let v = baseValue;
        if (cfg._kRateState === 'attack') {
          if (elapsed < A) {
            v = baseValue + (peak - baseValue) * (elapsed / A);
          } else if (elapsed < A + D) {
            const f = (elapsed - A) / D;
            v = peak + (sustainVal - peak) * f;
          } else {
            v = sustainVal;
            cfg._kRateState = 'sustain';
          }
        } else if (cfg._kRateState === 'release') {
          if (elapsed < R) {
            const f = elapsed / R;
            v = releaseStartValue + (baseValue - releaseStartValue) * f;
          } else {
            v = baseValue;
            clearInterval(cfg._kRateInterval);
            cfg._kRateInterval = null;
            cfg._kRateState = 'idle';
          }
        }
        cfg._kRateLastValue = v;
        try { resolved.customSetter(v); } catch (e) { /* swallow */ }
      }, K_RATE_INTERVAL_MS);
      this._kRateUpdaters.push({ stop: () => {
        if (cfg._kRateInterval) clearInterval(cfg._kRateInterval);
      }});
      cfg.gateOn = gateOn;
    }
  }

  // ── MIDI CC stub (for completeness) ──────────────────────────────────────

  _buildMidiCC(nodeDef) {
    const params = nodeDef.params || {};
    const targetStr = params.target || '';
    const target = parseTarget(targetStr);
    if (!target) return;
    const tgtBuilt = this.builtNodes[target.nodeId];
    if (!tgtBuilt) return;
    const resolved = resolveAudioParamOnBuiltNode(tgtBuilt, target.paramKey);
    if (!resolved) return;
    // Store for later setMidiCC() calls
    if (!this._midiCC) this._midiCC = new Map();
    this._midiCC.set(nodeDef.id, {
      ccNumber: params.cc_number ?? 1,
      target, resolved,
      minVal: params.min_val ?? 0,
      maxVal: params.max_val ?? 1,
    });
  }

  setMidiCC(ccNumber, value0to1) {
    if (!this._midiCC) return;
    for (const cfg of this._midiCC.values()) {
      if (cfg.ccNumber !== ccNumber) continue;
      const v = cfg.minVal + (cfg.maxVal - cfg.minVal) * clamp(value0to1, 0, 1);
      if (cfg.resolved.audioParam) {
        const t = this.ctx.currentTime;
        try {
          cfg.resolved.audioParam.cancelScheduledValues(t);
          cfg.resolved.audioParam.setTargetAtTime(v, t, 0.01);
        } catch (e) { /* ignore */ }
      } else if (cfg.resolved.customSetter) {
        try { cfg.resolved.customSetter(v); } catch (e) { /* ignore */ }
      }
    }
  }

  // ── BPM sync ─────────────────────────────────────────────────────────────

  setBPM(bpm) {
    this.bpm = bpm;
    for (const cfg of this._lfos.values()) {
      if (!cfg.sync) continue;
      const mult = SYNC_RATE_MAP[cfg.syncRate] ?? 1;
      const newRate = (this.bpm / 60) * mult;
      if (cfg.rateParam && typeof cfg.rateParam.setTargetAtTime === 'function') {
        try {
          cfg.rateParam.setTargetAtTime(newRate, this.ctx.currentTime, 0.01);
        } catch (e) { cfg.rateParam.value = newRate; }
      } else if (cfg.rateParam) {
        cfg.rateParam.value = newRate;
      }
    }
  }

  // ── Cleanup ──────────────────────────────────────────────────────────────

  dispose() {
    for (const cfg of this._lfos.values()) {
      try { cfg.source.stop(); } catch (e) {}
      try { cfg.source.disconnect(); } catch (e) {}
      try { cfg.depthGain.disconnect(); } catch (e) {}
      if (cfg.scaleGain) {
        try { cfg.scaleGain.disconnect(); } catch (e) {}
      }
    }
    this._lfos.clear();
    this._macros.clear();
    for (const cfg of this._modEnvs.values()) {
      if (cfg._kRateInterval) clearInterval(cfg._kRateInterval);
    }
    this._modEnvs.clear();
    for (const u of this._kRateUpdaters) { try { u.stop(); } catch (e) {} }
    this._kRateUpdaters = [];
    this._resolved = false;
  }
}

// ── helpers ──────────────────────────────────────────────────────────────

function clamp(v, lo, hi) {
  if (v < lo) return lo;
  if (v > hi) return hi;
  return v;
}

function nowMs() {
  if (typeof performance !== 'undefined' && performance.now) return performance.now();
  return Date.now();
}

// Re-export the parser so callers can validate target strings without
// reaching into dspNodeDefinitions.
export { parseTarget };
