/**
 * /src/audio/builders/r8.js
 *
 * Sidechain key-input builders. Created by Agent R8.
 *
 * Provides three node builders that the WebAudioDSPEngine NODE_BUILDERS
 * registry can adopt:
 *
 *   sidechain      — pure routing tap. A GainNode whose `output` is meant
 *                    to be wired to the input[1] of a sibling compressor_sc
 *                    or gate_sc node.
 *
 *   compressor_sc  — sidechain-capable compressor backed by the
 *                    'r8-compressor-sc-processor' AudioWorkletProcessor.
 *                    Built with numberOfInputs:2 so the second input slot
 *                    can carry the key signal. Exposes `signal_in` and
 *                    `key_in` so the graph compiler can target either.
 *
 *   gate_sc        — same idea, backed by 'r8-gate-sc-processor'.
 *
 * --- Worklet module loading -------------------------------------------------
 * The two new processors live alongside the existing ones at:
 *
 *   /src/lib/web-audio-plugins/worklets/r8-compressor-sc-processor.js
 *   /src/lib/web-audio-plugins/worklets/r8-gate-sc-processor.js
 *
 * The engine is responsible for calling
 *   await ctx.audioWorklet.addModule(<worklet-url>)
 * before any of these builders run. `ensureR8WorkletsLoaded(ctx)` is exported
 * as a convenience helper. It memoizes per-context.
 *
 * --- Edge routing convention -----------------------------------------------
 * The graph compiler is expected to wire edges like:
 *
 *   { source: 'kickTrack',   target: 'comp1' }                  // input[0]
 *   { source: 'sidechainTap', target: 'comp1', input: 1 }       // input[1] (key)
 *
 * Builders publish their input slots as:
 *   built.input        — input[0] (audio); kept for backwards compat
 *   built.inputs[0]    — same as built.input
 *   built.inputs[1]    — sidechain / key input
 *   built.signal_in    — alias for built.inputs[0]
 *   built.key_in       — alias for built.inputs[1]
 *
 * The engine's _buildGraphFromNodes() should select the correct slot from
 * an edge's `input` field; see INTEGRATION_R8.md for the patch description.
 */

const R8_COMPRESSOR_PROCESSOR = 'r8-compressor-sc-processor';
const R8_GATE_PROCESSOR = 'r8-gate-sc-processor';

const R8_COMPRESSOR_URL =
  '/src/lib/web-audio-plugins/worklets/r8-compressor-sc-processor.js';
const R8_GATE_URL =
  '/src/lib/web-audio-plugins/worklets/r8-gate-sc-processor.js';

const _loadedContexts = new WeakSet();

/**
 * Load the R8 worklet modules into an AudioContext (idempotent per-context).
 * Engine bootstrap should `await` this before invoking the builders below.
 */
export async function ensureR8WorkletsLoaded(ctx, urls = {}) {
  if (_loadedContexts.has(ctx)) return;
  const compressorUrl = urls.compressor || R8_COMPRESSOR_URL;
  const gateUrl = urls.gate || R8_GATE_URL;
  await Promise.all([
    ctx.audioWorklet.addModule(compressorUrl),
    ctx.audioWorklet.addModule(gateUrl)
  ]);
  _loadedContexts.add(ctx);
}

/**
 * Resolve a parameter binding ("@paramId") into a paramTargets entry pointing
 * at an AudioParam. Used to keep the binding logic identical across builders.
 */
function bindAudioParam(paramTargets, paramDefs, paramId, audioParam, scale) {
  const entry = { audioParam, paramDef: paramDefs?.[paramId] };
  if (scale) entry.scale = scale;
  paramTargets[paramId] = entry;
}

// ── sidechain ────────────────────────────────────────────────────────────────
// A passthrough gain. Whatever feeds it should be wired (by the graph
// compiler) to a downstream dynamics node's input[1] via an edge with
// `input: 1`. The node itself contains no detector logic — its sole job is
// to be a stable named output that other nodes can reference as a sidechain
// source.
//
// dspNodeDefinitions has io {in:0, out:1}, but in practice the compiler may
// also choose to feed it from another node — keeping `input` writable is
// harmless. The OUTPUT is what feeds compressor_sc.key_in.
export function sidechain(ctx, node, paramDefs) {
  const tap = ctx.createGain();
  tap.gain.value = 1;

  const targets = {};
  const params = node.params || {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'gain' || key === 'gain_db' || key === 'level') {
        // gain_db wants linear scaling — caller's modulator will apply it.
        bindAudioParam(targets, paramDefs, paramId, tap.gain);
      }
    } else {
      if (key === 'gain' || key === 'level') tap.gain.value = val;
      // gain_db handled at param-init time outside this builder
    }
  }

  return {
    input: tap,
    output: tap,
    inputs: [tap],
    paramTargets: targets,
    isSidechainSource: true
  };
}

// ── compressor_sc ────────────────────────────────────────────────────────────
// Sidechain-capable compressor. Uses an AudioWorkletNode with two inputs:
//   inputs[0] — audio
//   inputs[1] — key
// Caller must have already ensured the worklet module is loaded; see
// ensureR8WorkletsLoaded(). The k-rate `sidechain_active` parameter is set
// to 1 so the processor reads the key channel — but the processor itself
// also no-ops to inputs[0] if the key channel is silent / missing, so this
// is a safe default.
export function compressor_sc(ctx, node, paramDefs) {
  const params = node.params || {};

  const workletNode = new AudioWorkletNode(ctx, R8_COMPRESSOR_PROCESSOR, {
    numberOfInputs: 2,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      threshold: params.threshold_db ?? params.threshold ?? -24,
      ratio: params.ratio ?? 4,
      attack: ((params.attack_ms ?? 10) / 1000),
      release: ((params.release_ms ?? 150) / 1000),
      knee: params.knee_db ?? 6,
      makeupGain: params.makeup_db ?? 0,
      mix: params.mix ?? 1.0,
      sidechain_active: params.sidechain_input === false ? 0 : 1
    }
  });

  // The two input slots are exposed via the AudioWorkletNode's input-index
  // protocol on connect(), but the engine wants explicit refs to wire to.
  // Web Audio doesn't give us "child" GainNodes for input slots automatically,
  // so we expose an `inputResolver(index)` helper used by the graph compiler:
  // for input 0 you connect(workletNode, 0, 0); for input 1, connect(workletNode, 0, 1).
  // To keep the existing engine code (which calls `src.output.connect(tgt.input)`)
  // working for input 0, `built.input` IS the worklet itself. The engine will
  // route input:1 edges through `built.inputs[1]` by passing the index to
  // the underlying AudioNode.connect().
  const targets = {};
  const wp = workletNode.parameters;
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'threshold_db' || key === 'threshold') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('threshold'));
      } else if (key === 'ratio') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('ratio'));
      } else if (key === 'attack_ms' || key === 'attack') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('attack'), (v) => v / 1000);
      } else if (key === 'release_ms' || key === 'release') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('release'), (v) => v / 1000);
      } else if (key === 'knee_db' || key === 'knee') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('knee'));
      } else if (key === 'makeup_db' || key === 'makeup' || key === 'makeup_gain') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('makeupGain'));
      } else if (key === 'mix') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('mix'));
      } else if (key === 'sidechain_input') {
        // Toggle the active flag (booleans → 0/1)
        const flagParam = wp.get('sidechain_active');
        targets[paramId] = {
          paramDef: paramDefs?.[paramId],
          customSetter: (v) => { flagParam.value = v ? 1 : 0; }
        };
      }
    }
  }

  return {
    input: workletNode,         // input[0] — backwards-compat: existing
                                // edges connect here with no `input` field.
    output: workletNode,
    inputs: [workletNode, workletNode], // index 0 + 1 share the same node;
                                        // graph compiler must pass slot index
                                        // to AudioNode.connect(dst, 0, slot)
    signal_in: workletNode,
    key_in: workletNode,
    keyInputIndex: 1,                   // slot to use on connect()
    signalInputIndex: 0,
    paramTargets: targets,
    workletNode,
    sidechainCapable: true
  };
}

// ── gate_sc ──────────────────────────────────────────────────────────────────
export function gate_sc(ctx, node, paramDefs) {
  const params = node.params || {};

  const workletNode = new AudioWorkletNode(ctx, R8_GATE_PROCESSOR, {
    numberOfInputs: 2,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: {
      threshold: params.threshold_db ?? params.threshold ?? -40,
      attack: ((params.attack_ms ?? 1) / 1000),
      release: ((params.release_ms ?? 100) / 1000),
      range: params.range_db ?? params.range ?? -60,
      sidechain_active: params.sidechain_input === false ? 0 : 1
    }
  });

  const targets = {};
  const wp = workletNode.parameters;
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      if (key === 'threshold_db' || key === 'threshold') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('threshold'));
      } else if (key === 'attack_ms' || key === 'attack') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('attack'), (v) => v / 1000);
      } else if (key === 'release_ms' || key === 'release') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('release'), (v) => v / 1000);
      } else if (key === 'range_db' || key === 'range') {
        bindAudioParam(targets, paramDefs, paramId, wp.get('range'));
      } else if (key === 'sidechain_input') {
        const flagParam = wp.get('sidechain_active');
        targets[paramId] = {
          paramDef: paramDefs?.[paramId],
          customSetter: (v) => { flagParam.value = v ? 1 : 0; }
        };
      }
    }
  }

  return {
    input: workletNode,
    output: workletNode,
    inputs: [workletNode, workletNode],
    signal_in: workletNode,
    key_in: workletNode,
    keyInputIndex: 1,
    signalInputIndex: 0,
    paramTargets: targets,
    workletNode,
    sidechainCapable: true
  };
}

// Aggregate export for easy registry merging:
//   import * as R8 from './builders/r8';
//   Object.assign(NODE_BUILDERS, { sidechain: R8.sidechain, ... })
export const R8_BUILDERS = {
  sidechain,
  compressor_sc,
  gate_sc
};

export default R8_BUILDERS;
