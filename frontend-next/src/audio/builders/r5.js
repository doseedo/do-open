/**
 * r5.js — runtime builders for the R5 batch of spectral-domain DSP nodes.
 *
 * Implements:
 *   pitch_shift     — phase-vocoder pitch shift (±24 semitones)
 *   spectral_filter — FFT-bin gating between [low_bin, high_bin]
 *   spectral_freeze — magnitude-latch freeze with phase advance / random
 *
 * Each builder honors the standard interface:
 *   buildFoo(ctx, nodeDef, paramDefs) → { input, output, paramTargets }
 *
 * Worklets used (registered via ensureR5Worklets):
 *   r5-pitch-shift-processor
 *   r5-spectral-filter-processor
 *   r5-spectral-freeze-processor
 *
 * '@'-prefixed param values bind to parameter IDs for live updates. Because
 * the builder API is synchronous but `audioWorklet.addModule` is async,
 * builders kick off registration eagerly and fall back to a passthrough
 * GainNode until the worklet is ready. On the *next* graph rebuild the
 * worklet has loaded and the spectral processor takes over.
 *
 * NOTE: r1.js also defines a `pitch_shift` builder (SOLA / time-domain).
 * Whichever one is merged later into NODE_BUILDERS wins; integrators
 * should pick the spectral version for higher-shift quality and the SOLA
 * version for lower latency. See INTEGRATION_R5.md.
 */

// ── Worklet loading ───────────────────────────────────────────────────────

const WORKLET_BASE = '../../lib/web-audio-plugins/worklets';

const R5_WORKLETS = {
  'r5-pitch-shift-processor':     `${WORKLET_BASE}/r5-pitch-shift-processor.js`,
  'r5-spectral-filter-processor': `${WORKLET_BASE}/r5-spectral-filter-processor.js`,
  'r5-spectral-freeze-processor': `${WORKLET_BASE}/r5-spectral-freeze-processor.js`,
};

const _registeredCtxs = new WeakMap();

export async function ensureR5Worklets(ctx) {
  if (!ctx || !ctx.audioWorklet) return false;
  if (_registeredCtxs.get(ctx) === true) return true;

  const pending = _registeredCtxs.get(ctx);
  if (pending && typeof pending.then === 'function') return pending;

  const promise = (async () => {
    for (const [, relPath] of Object.entries(R5_WORKLETS)) {
      try {
        const url = new URL(relPath, import.meta.url).href;
        await ctx.audioWorklet.addModule(url);
      } catch (err) {
        // Some environments (jsdom, SSR) may fail; swallow so non-worklet
        // builders still come up.
        // eslint-disable-next-line no-console
        console.warn('[R5] failed to load worklet', relPath, err);
      }
    }
    _registeredCtxs.set(ctx, true);
    return true;
  })();

  _registeredCtxs.set(ctx, promise);
  return promise;
}

function workletsReady(ctx) {
  return _registeredCtxs.get(ctx) === true;
}

// ── Helpers ───────────────────────────────────────────────────────────────

function bindParam(targets, paramId, paramDef, audioParam, opts = {}) {
  targets[paramId] = { audioParam, paramDef, ...opts };
}

function bindCustom(targets, paramId, paramDef, customSetter) {
  targets[paramId] = { paramDef, customSetter };
}

function isAtBinding(v) {
  return typeof v === 'string' && v.startsWith('@');
}

// Synchronously try to instantiate a worklet node; returns null if the
// registration hasn't completed yet so the caller can fall back.
function tryCreateWorklet(ctx, processorName, options) {
  ensureR5Worklets(ctx);
  if (!workletsReady(ctx)) return null;
  try {
    return new AudioWorkletNode(ctx, processorName, options);
  } catch (e) {
    return null;
  }
}

// Apply a possibly-bound param value: for literals, set immediately via
// `applyLiteral`; for `@` bindings, install a target entry.
function bindOrApply(targets, paramDefs, params, key, paramTargetSetup, applyLiteral) {
  const val = params[key];
  if (val === undefined) return;
  if (isAtBinding(val)) {
    const paramId = val.slice(1);
    paramTargetSetup(paramId, paramDefs[paramId]);
  } else {
    applyLiteral(val);
  }
}

// ── pitch_shift ────────────────────────────────────────────────────────────
//
// Phase-vocoder pitch shift driven by AudioParams + port messages.
// Builder produces a wet/dry chain so we can return stable input/output
// gains even when the worklet isn't registered yet.

export function buildPitchShift(ctx, nodeDef, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const targets = {};
  const params = (nodeDef && nodeDef.params) || {};

  const node = tryCreateWorklet(ctx, 'r5-pitch-shift-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: { fftSize: 2048 },
  });

  if (!node) {
    // Worklet not yet loaded — passthrough. Next graph rebuild will pick it up.
    input.connect(output);

    // Still expose param bindings as no-ops so the engine doesn't error out.
    if (isAtBinding(params.semitones)) {
      const id = params.semitones.slice(1);
      bindCustom(targets, id, paramDefs[id], () => {});
    }
    if (isAtBinding(params.mix)) {
      const id = params.mix.slice(1);
      bindCustom(targets, id, paramDefs[id], () => {});
    }
    return { input, output, paramTargets: targets };
  }

  input.connect(node);
  node.connect(output);

  const semParam = node.parameters.get('semitones');
  const mixParam = node.parameters.get('mix');

  bindOrApply(targets, paramDefs, params, 'semitones',
    (id, def) => bindParam(targets, id, def, semParam),
    (v) => { if (semParam) semParam.value = +v; }
  );
  bindOrApply(targets, paramDefs, params, 'mix',
    (id, def) => bindParam(targets, id, def, mixParam),
    (v) => { if (mixParam) mixParam.value = +v; }
  );

  return { input, output, paramTargets: targets, workletNode: node };
}

// ── spectral_filter ────────────────────────────────────────────────────────

export function buildSpectralFilter(ctx, nodeDef, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const targets = {};
  const params = (nodeDef && nodeDef.params) || {};

  const node = tryCreateWorklet(ctx, 'r5-spectral-filter-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: { fftSize: 2048 },
  });

  if (!node) {
    input.connect(output);
    for (const k of ['low_bin', 'high_bin', 'mix']) {
      if (isAtBinding(params[k])) {
        const id = params[k].slice(1);
        bindCustom(targets, id, paramDefs[id], () => {});
      }
    }
    return { input, output, paramTargets: targets };
  }

  input.connect(node);
  node.connect(output);

  const loP  = node.parameters.get('lowBin');
  const hiP  = node.parameters.get('highBin');
  const mixP = node.parameters.get('mix');

  bindOrApply(targets, paramDefs, params, 'low_bin',
    (id, def) => bindParam(targets, id, def, loP),
    (v) => { if (loP) loP.value = +v; }
  );
  bindOrApply(targets, paramDefs, params, 'high_bin',
    (id, def) => bindParam(targets, id, def, hiP),
    (v) => { if (hiP) hiP.value = +v; }
  );
  bindOrApply(targets, paramDefs, params, 'mix',
    (id, def) => bindParam(targets, id, def, mixP),
    (v) => { if (mixP) mixP.value = +v; }
  );

  return { input, output, paramTargets: targets, workletNode: node };
}

// ── spectral_freeze ────────────────────────────────────────────────────────

export function buildSpectralFreeze(ctx, nodeDef, paramDefs) {
  const input  = ctx.createGain();
  const output = ctx.createGain();
  const targets = {};
  const params = (nodeDef && nodeDef.params) || {};

  const phaseMode = (params && typeof params.phase_mode === 'string')
    ? params.phase_mode : 'advance';

  const node = tryCreateWorklet(ctx, 'r5-spectral-freeze-processor', {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: { fftSize: 2048, phaseMode },
  });

  if (!node) {
    input.connect(output);
    for (const k of ['freeze', 'mix']) {
      if (isAtBinding(params[k])) {
        const id = params[k].slice(1);
        bindCustom(targets, id, paramDefs[id], () => {});
      }
    }
    return { input, output, paramTargets: targets };
  }

  input.connect(node);
  node.connect(output);

  const fzP  = node.parameters.get('freeze');
  const mixP = node.parameters.get('mix');

  bindOrApply(targets, paramDefs, params, 'freeze',
    (id, def) => bindParam(targets, id, def, fzP),
    (v) => { if (fzP) fzP.value = +v; }
  );
  bindOrApply(targets, paramDefs, params, 'mix',
    (id, def) => bindParam(targets, id, def, mixP),
    (v) => { if (mixP) mixP.value = +v; }
  );

  return { input, output, paramTargets: targets, workletNode: node };
}

// ── Exports ───────────────────────────────────────────────────────────────

// `pitch_shift_pv` — phase-vocoder version. R1 ships the SOLA `pitch_shift`
// (lower latency, gentler at small shifts). Use `_pv` for ≥±12 semitones,
// `pitch_shift` (SOLA) for small/transient-sensitive shifts.
const r5Builders = {
  pitch_shift_pv:  buildPitchShift,
  spectral_filter: buildSpectralFilter,
  spectral_freeze: buildSpectralFreeze,
};

export default r5Builders;
