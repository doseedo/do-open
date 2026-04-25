/**
 * Agent R2 — WDF (Wave Digital Filter) builders.
 *
 * Each builder returns the standard shape consumed by WebAudioDSPEngine:
 *   { input, output, paramTargets }
 *
 * Since these are AudioWorkletNode-backed, the builder synchronously
 * returns input/output GainNodes (initially wired through, behaving as
 * unity-gain passthrough) and asynchronously loads the worklet module +
 * splices the worklet into the chain when ready.  This keeps the engine's
 * synchronous build path (`builder(ctx, node, paramDefs)`) clean while
 * still letting the worklet do its thing.
 *
 * paramTargets uses `customSetter` for each modulatable param — the setter
 * forwards to the worklet's k-rate AudioParam once the worklet is ready,
 * and buffers the latest value until then.
 *
 * Author: Agent R2 (one of 12 in the runtime expansion).
 */

// Map of node-type → worklet path (relative to import.meta.url base).
// Builders use new URL(...) to make the path bundler-resolvable.
const WORKLET_PATHS = {
  wdf_diode_clipper:       '../../lib/web-audio-plugins/worklets/r2-wdf-diode-clipper-processor.js',
  wdf_tube_triode:         '../../lib/web-audio-plugins/worklets/r2-wdf-tube-triode-processor.js',
  wdf_tube_amp:            '../../lib/web-audio-plugins/worklets/r2-wdf-tube-amp-processor.js',
  wdf_transistor_clipper:  '../../lib/web-audio-plugins/worklets/r2-wdf-transistor-clipper-processor.js',
  wdf_tone_stack:          '../../lib/web-audio-plugins/worklets/r2-wdf-tone-stack-processor.js',
};

const PROCESSOR_NAMES = {
  wdf_diode_clipper:       'r2-wdf-diode-clipper-processor',
  wdf_tube_triode:         'r2-wdf-tube-triode-processor',
  wdf_tube_amp:            'r2-wdf-tube-amp-processor',
  wdf_transistor_clipper:  'r2-wdf-transistor-clipper-processor',
  wdf_tone_stack:          'r2-wdf-tone-stack-processor',
};

// Cache of loaded worklet modules per AudioContext+nodeType.  WeakMap on ctx
// → Map of nodeType → Promise so we don't double-load.
const _moduleCache = new WeakMap();

function _loadWorklet(ctx, nodeType) {
  let perCtx = _moduleCache.get(ctx);
  if (!perCtx) {
    perCtx = new Map();
    _moduleCache.set(ctx, perCtx);
  }
  if (perCtx.has(nodeType)) return perCtx.get(nodeType);
  const path = WORKLET_PATHS[nodeType];
  if (!path) {
    const err = Promise.reject(new Error(`r2: unknown worklet for nodeType ${nodeType}`));
    perCtx.set(nodeType, err);
    return err;
  }
  const url = new URL(path, import.meta.url).href;
  const p = ctx.audioWorklet.addModule(url).catch((err) => {
    // eslint-disable-next-line no-console
    console.error(`r2: failed to load worklet ${nodeType}`, err);
    throw err;
  });
  perCtx.set(nodeType, p);
  return p;
}

/**
 * Generic factory that all 5 builders share.  Differences are only the
 * processor name + the list of params to expose as paramTargets.
 *
 * paramKeys: array of {key, defaultValue, dbScale?} — the keys that match
 * both the dspGraph node.params and the worklet's parameterDescriptors.
 */
function _build(ctx, node, paramDefs, nodeType, paramKeys) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  input.gain.value = 1;
  output.gain.value = 1;

  // Initially connect input → output (passthrough) until worklet ready.
  // Once worklet loads we'll disconnect and reroute through it.
  input.connect(output);

  // Buffer for parameter values that arrive before the worklet is ready.
  const pending = {};
  // Live worklet reference; null until ready.
  let workletNode = null;

  // paramTargets: for each modulatable param, return a customSetter that
  // either writes to worklet AudioParam (when ready) or buffers the value.
  const targets = {};
  const params = node.params || {};
  const processorName = PROCESSOR_NAMES[nodeType];

  for (const { key } of paramKeys) {
    const val = params[key];
    if (typeof val === 'string' && val.startsWith('@')) {
      const paramId = val.slice(1);
      targets[paramId] = {
        paramDef: paramDefs[paramId] || {},
        customSetter: (v) => {
          if (workletNode) {
            const p = workletNode.parameters.get(key);
            if (p) p.value = v;
            else workletNode.port.postMessage({ type: 'setParam', key, value: v });
          } else {
            pending[key] = v;
          }
        },
      };
    } else if (val != null) {
      pending[key] = Number(val);
    }
  }

  // Async load + splice
  _loadWorklet(ctx, nodeType).then(() => {
    try {
      workletNode = new AudioWorkletNode(ctx, processorName, {
        numberOfInputs: 1,
        numberOfOutputs: 1,
        outputChannelCount: [2],
      });
      // Apply any pending param values
      for (const [key, v] of Object.entries(pending)) {
        const p = workletNode.parameters.get(key);
        if (p) p.value = v;
      }
      // Splice into the chain: input → workletNode → output
      try { input.disconnect(output); } catch (_) { /* already disconnected */ }
      input.connect(workletNode);
      workletNode.connect(output);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(`r2: failed to instantiate ${processorName}`, err);
      // leave passthrough in place
    }
  }).catch(() => {
    // _loadWorklet already logged; passthrough remains
  });

  return { input, output, paramTargets: targets };
}

// ── Per-type builders ─────────────────────────────────────────────────────

export function buildWdfDiodeClipper(ctx, node, paramDefs) {
  return _build(ctx, node, paramDefs, 'wdf_diode_clipper', [
    { key: 'drive' }, { key: 'ideality' }, { key: 'symmetry' }, { key: 'mix' },
  ]);
}

export function buildWdfTubeTriode(ctx, node, paramDefs) {
  return _build(ctx, node, paramDefs, 'wdf_tube_triode', [
    { key: 'drive' }, { key: 'bias' }, { key: 'mix' },
  ]);
}

export function buildWdfTubeAmp(ctx, node, paramDefs) {
  return _build(ctx, node, paramDefs, 'wdf_tube_amp', [
    { key: 'gain' }, { key: 'bias' }, { key: 'stages' },
    { key: 'output_level' }, { key: 'mix' },
  ]);
}

export function buildWdfTransistorClipper(ctx, node, paramDefs) {
  return _build(ctx, node, paramDefs, 'wdf_transistor_clipper', [
    { key: 'drive' }, { key: 'beta' }, { key: 'fuzz' }, { key: 'mix' },
  ]);
}

export function buildWdfToneStack(ctx, node, paramDefs) {
  return _build(ctx, node, paramDefs, 'wdf_tone_stack', [
    { key: 'bass' }, { key: 'mid' }, { key: 'treble' }, { key: 'mix' },
  ]);
}

// ── Default export: { nodeType: builderFn } map for direct merge into NODE_BUILDERS

export default {
  wdf_diode_clipper:       buildWdfDiodeClipper,
  wdf_tube_triode:         buildWdfTubeTriode,
  wdf_tube_amp:            buildWdfTubeAmp,
  wdf_transistor_clipper:  buildWdfTransistorClipper,
  wdf_tone_stack:          buildWdfToneStack,
};
