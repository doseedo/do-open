/**
 * R13 — Adaptive Limiter (Logic Pro stock parity)
 *
 * Registers `adaptive_limiter` as a NEW node type backed by a single
 * AudioWorklet (`r13-adaptive-limiter-processor`) implementing a multi-stage
 * lookahead limiter with adaptive release. See INTEGRATION_R13_ADAPTIVE_LIMITER.md
 * for the algorithm rationale.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildAdaptiveLimiter(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Schema:
 *   {
 *     type: 'adaptive_limiter',
 *     params: {
 *       gain:               number (0..24, dB)         | '@<id>',
 *       out_ceiling:        number (-30..0, dB)        | '@<id>',
 *       lookahead_ms:       number (1..12)             | '@<id>',
 *       release_min_ms:     number (1..50)             | '@<id>',
 *       release_max_ms:     number (100..2000)         | '@<id>',
 *       release_adaptation: number (0..1)              | '@<id>',
 *       true_peak:          0|1                        | '@<id>',
 *       soft_clip_amount:   number (0..1)              | '@<id>',
 *       link_lr:            0|1                        | '@<id>',
 *     }
 *   }
 *
 * Param values may be literals or '@<paramId>' bindings (live-modulated).
 *
 * Fallback: if the worklet processor isn't registered yet (or worklets are
 * unavailable, e.g. SSR / Node test harness), the builder constructs a
 * primitive DynamicsCompressorNode configured for hard-knee 20:1 limiting at
 * `out_ceiling` threshold. The fallback is intentionally simple — it preserves
 * the ceiling-enforcement contract (peaks ≤ ceiling) but loses the adaptive
 * release character.
 *
 * @author Doseedo R13 — Adaptive Limiter
 */

const R13_PROCESSOR = 'r13-adaptive-limiter-processor';

function _safeWorklet(ctx, name, options) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13/AdaptiveLimiter] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

// Helper — return AudioParam by name from the worklet (null when no worklet).
function _wpar(worklet, name) {
  return (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;
}

// Per-param spec: paramId → AudioParam name + numeric coercer (so we can
// drive the worklet AudioParam directly via @-modulation).
const PARAM_SPECS = {
  gain:               { workletName: 'gain' },
  out_ceiling:        { workletName: 'out_ceiling' },
  lookahead_ms:       { workletName: 'lookahead_ms' },
  release_min_ms:     { workletName: 'release_min_ms' },
  release_max_ms:     { workletName: 'release_max_ms' },
  release_adaptation: { workletName: 'release_adaptation' },
  true_peak:          { workletName: 'true_peak',     coerce: (v) => (v ? 1 : 0) },
  soft_clip_amount:   { workletName: 'soft_clip_amount' },
  link_lr:            { workletName: 'link_lr',       coerce: (v) => (v ? 1 : 0) },
};

export function buildAdaptiveLimiter(ctx, node, paramDefs) {
  const params = node.params || {};
  const targets = {};

  const input = ctx.createGain();
  const output = ctx.createGain();

  // Pull initial parameter literals so the worklet starts at the right
  // operating point even before @-modulated targets fire.
  const parameterData = {};
  for (const [key, val] of Object.entries(params)) {
    if (typeof val === 'string' && val.startsWith('@')) continue;
    if (PARAM_SPECS[key] != null) {
      const spec = PARAM_SPECS[key];
      const coerced = spec.coerce ? spec.coerce(val) : val;
      if (typeof coerced === 'number' && Number.isFinite(coerced)) {
        parameterData[spec.workletName] = coerced;
      }
    }
  }

  const worklet = _safeWorklet(ctx, R13_PROCESSOR, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData,
  });

  // ── Fallback path ───────────────────────────────────────────────────────
  // No worklet: use a DynamicsCompressorNode in brickwall config + a pre-gain
  // for the `gain` param. Loses adaptive-release character but preserves
  // the ceiling-enforcement contract.
  let preGain = null;
  let comp = null;
  let postGain = null;
  let fallbackCeilDb = (typeof params.out_ceiling === 'number') ? params.out_ceiling : -0.3;
  if (!worklet) {
    preGain = ctx.createGain();
    comp = ctx.createDynamicsCompressor();
    postGain = ctx.createGain();

    const initialGainDb = (typeof params.gain === 'number') ? params.gain : 0;
    preGain.gain.value = Math.pow(10, initialGainDb / 20);

    // Brickwall configuration.
    comp.threshold.value = fallbackCeilDb;
    comp.knee.value = 0;
    comp.ratio.value = 20;
    comp.attack.value = 0.001;
    comp.release.value = 0.100;

    postGain.gain.value = 1.0;

    input.connect(preGain);
    preGain.connect(comp);
    comp.connect(postGain);
    postGain.connect(output);
  } else {
    input.connect(worklet);
    worklet.connect(output);
  }

  // ── Param wiring ────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const spec = PARAM_SPECS[key];
    if (!spec) continue;

    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    if (isModulated) {
      const ap = _wpar(worklet, spec.workletName);
      if (ap) {
        // Preferred path: bind the engine's AudioParam directly.
        targets[paramId] = {
          paramDef: paramDefs[paramId],
          audioParam: ap,
          customSetter: spec.coerce ? (v) => { ap.value = spec.coerce(v); } : null,
        };
      } else {
        // Fallback path: route specific params to the DynamicsCompressorNode.
        targets[paramId] = {
          paramDef: paramDefs[paramId],
          customSetter: (v) => {
            if (key === 'gain' && preGain) {
              preGain.gain.value = Math.pow(10, v / 20);
            } else if (key === 'out_ceiling' && comp) {
              comp.threshold.value = v;
              fallbackCeilDb = v;
            } else if (key === 'release_max_ms' && comp) {
              // Use the slow-release bound as the fallback follower release.
              comp.release.value = Math.max(0.01, v / 1000);
            } else if (key === 'lookahead_ms' && comp) {
              comp.attack.value = Math.max(0.0001, (v / 1000) / 4);
            }
            // soft_clip_amount, true_peak, link_lr, release_min_ms,
            // release_adaptation: no fallback DSP target — silently no-op.
          },
        };
      }
    } else if (val !== undefined && val !== null) {
      // Literal value: already pushed via parameterData on the worklet path.
      // For the fallback path, params not handled above are no-op.
      // (preGain / comp.threshold / etc. were primed from `params` above.)
    }
  }

  return { input, output, paramTargets: targets };
}

// ── Default export: NODE_BUILDERS map ───────────────────────────────────────
const R13_ADAPTIVE_LIMITER_BUILDERS = {
  adaptive_limiter: buildAdaptiveLimiter,
};

export default R13_ADAPTIVE_LIMITER_BUILDERS;
