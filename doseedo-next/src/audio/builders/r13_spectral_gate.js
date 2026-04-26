/**
 * R13 — Spectral Gate (frequency-domain noise gate).
 *
 * Registers `spectral_gate` as a NEW node type. Pipeline (per-frame):
 *
 *   STFT → per-bin |X[k]| → compare against (threshold + tilt(k)) →
 *     bins below threshold get target gain = 10^(reduction_db/20), bins
 *     above pass through (target = 1) → per-bin one-pole low-pass
 *     envelope on the gate decision (attack / release ms) →
 *     X'[k] = X[k] · env[k] → ISTFT → wet/dry mix.
 *
 * Built on the same R5 spectral foundation (Hann window, 75% overlap,
 * inlined Cooley–Tukey FFT). The worklet
 * `r13-spectral-gate-processor` carries the heavy lifting; this builder
 * only wires AudioParams + port-message overrides + a non-worklet fallback.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildSpectralGate(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Fallback path (when the worklet processor isn't registered yet, e.g.
 * `addModule` hasn't resolved): a serial WebAudio DynamicsCompressorNode
 * configured as a wide-band gate (high ratio, low knee). Frequency-
 * dependent gating isn't possible without the worklet — we still produce
 * audible audio + bind every param surface so the engine doesn't error.
 *
 * @author Doseedo R13
 */

const R13_SPECTRAL_GATE = 'r13-spectral-gate-processor';

// ── Worklet auto-registration (mirrors r5.js pattern) ─────────────────────

const WORKLET_PATH = '../../lib/web-audio-plugins/worklets/r13-spectral-gate-processor.js';
const _registeredCtxs = new WeakMap();

export async function ensureR13SpectralGateWorklet(ctx) {
  if (!ctx || !ctx.audioWorklet) return false;
  if (_registeredCtxs.get(ctx) === true) return true;
  const pending = _registeredCtxs.get(ctx);
  if (pending && typeof pending.then === 'function') return pending;

  const promise = (async () => {
    try {
      const url = new URL(WORKLET_PATH, import.meta.url).href;
      await ctx.audioWorklet.addModule(url);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('[R13/spectral_gate] worklet failed to load:', err && err.message);
    }
    _registeredCtxs.set(ctx, true);
    return true;
  })();
  _registeredCtxs.set(ctx, promise);
  return promise;
}

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13/spectral_gate] worklet ${name} unavailable, using fallback:`, e && e.message);
    }
    return null;
  }
}

// dB → linear gain.
function _db2gain(db) { return Math.pow(10, db / 20); }

// Param key → AudioParam name on the worklet.
const PARAM_NAME_MAP = {
  threshold_db: 'thresholdDb',
  reduction_db: 'reductionDb',
  attack_ms:    'attackMs',
  release_ms:   'releaseMs',
  low_cut:      'lowCut',
  high_cut:     'highCut',
  tilt_db:      'tiltDb',
  mix:          'mix',
};

// ── spectral_gate builder ─────────────────────────────────────────────────
//
// Schema:
//   {
//     type: 'spectral_gate',
//     params: {
//       threshold_db: number (-60..0, default -40)         | '@<id>',
//       reduction_db: number (-60..0, default -40)         | '@<id>',
//       attack_ms:    number (1..100,  default 10)         | '@<id>',
//       release_ms:   number (10..1000, default 100)       | '@<id>',
//       low_cut:      number (0..1, normalized 0..N/2 bin) | '@<id>',
//       high_cut:     number (0..1)                        | '@<id>',
//       tilt_db:      number (-12..+12, default 0)         | '@<id>',
//       mix:          number (0..1, default 1)             | '@<id>',
//     }
//   }
export function buildSpectralGate(ctx, node, paramDefs) {
  const params = (node && node.params) || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Kick off async worklet registration; it'll be ready on the *next* graph
  // build. The first build after page load is a fallback.
  ensureR13SpectralGateWorklet(ctx);

  // Pre-seed parameterData from any literal values so the first analysis
  // frame uses the correct threshold without waiting for setParameter().
  const initParamData = {};
  for (const [k, v] of Object.entries(params)) {
    if (typeof v !== 'number') continue;
    const apName = PARAM_NAME_MAP[k];
    if (apName) initParamData[apName] = v;
  }

  const worklet = _safeWorklet(ctx, R13_SPECTRAL_GATE, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: initParamData,
    processorOptions: { fftSize: 2048 },
  });

  // ── Fallback path ─────────────────────────────────────────────────────
  // Wide-band gate via DynamicsCompressorNode (high ratio, low threshold).
  // Spectrum-aware behavior is impossible without the worklet, but graph
  // binding + audible audio still flow.
  let fallbackGate = null;
  let fallbackDry = null, fallbackWet = null;
  if (!worklet) {
    fallbackGate = ctx.createDynamicsCompressor();
    // Configure as a gate-ish device: high ratio, low threshold so quiet
    // signals are squashed. Negative threshold roughly matches threshold_db.
    fallbackGate.threshold.value = (typeof params.threshold_db === 'number') ? params.threshold_db : -40;
    fallbackGate.ratio.value     = 20;
    fallbackGate.attack.value    = 0.005;
    fallbackGate.release.value   = 0.100;
    fallbackGate.knee.value      = 1;

    fallbackDry = ctx.createGain();
    fallbackWet = ctx.createGain();
    fallbackDry.gain.value = 0;
    fallbackWet.gain.value = (typeof params.mix === 'number') ? params.mix : 1;

    input.connect(fallbackDry);
    input.connect(fallbackGate);
    fallbackGate.connect(fallbackWet);
    fallbackDry.connect(output);
    fallbackWet.connect(output);
  } else {
    input.connect(worklet);
    worklet.connect(output);
  }

  // Helper: get a worklet AudioParam by *AudioParam name* (camelCase),
  // null if no worklet.
  const wpar = (name) => (worklet && worklet.parameters)
    ? (worklet.parameters.get(name) || null)
    : null;

  // ── Param wiring ─────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    const isModulated = typeof val === 'string' && val.startsWith('@');
    const paramId = isModulated ? val.slice(1) : null;

    const apName = PARAM_NAME_MAP[key];
    if (!apName) continue;
    const ap = wpar(apName);

    if (isModulated) {
      if (ap) {
        targets[paramId] = { audioParam: ap, paramDef: paramDefs[paramId] };
      } else {
        // Worklet missing → synthesize a custom setter that drives the
        // fallback compressor where it makes sense.
        const setter = _fallbackSetter(key, fallbackGate, fallbackDry, fallbackWet);
        targets[paramId] = { paramDef: paramDefs[paramId], customSetter: setter };
      }
    } else if (typeof val === 'number') {
      if (ap) {
        ap.value = val;
      } else {
        const setter = _fallbackSetter(key, fallbackGate, fallbackDry, fallbackWet);
        if (setter) setter(val);
      }
    }
  }

  return { input, output, paramTargets: targets, workletNode: worklet || null };
}

// Map a schema param (snake_case) to a fallback DynamicsCompressor setter.
// Returns a function that takes a numeric value, or null for params that
// don't map to the fallback (e.g. tilt_db, low_cut).
function _fallbackSetter(key, gate, dryG, wetG) {
  if (!gate) return () => {};
  switch (key) {
    case 'threshold_db':
      return (v) => { gate.threshold.value = Math.max(-100, Math.min(0, +v)); };
    case 'reduction_db':
      // Fallback can't directly set reduction-floor; emulate by tweaking ratio.
      // -60 dB reduction ≈ ratio 50, -10 dB reduction ≈ ratio 2.
      return (v) => {
        const r = Math.max(1, Math.min(50, Math.pow(10, -(+v) / 40)));
        gate.ratio.value = r;
      };
    case 'attack_ms':
      return (v) => { gate.attack.value = Math.max(0.001, +v / 1000); };
    case 'release_ms':
      return (v) => { gate.release.value = Math.max(0.01, +v / 1000); };
    case 'mix':
      return (v) => {
        if (!wetG || !dryG) return;
        const w = Math.max(0, Math.min(1, +v));
        wetG.gain.value = w;
        dryG.gain.value = 1 - w;
      };
    case 'low_cut':
    case 'high_cut':
    case 'tilt_db':
      // No-op in fallback; binding still installed so engine doesn't error.
      return () => {};
    default:
      return () => {};
  }
}

// ── Default export: NODE_BUILDERS map ────────────────────────────────────
const R13_SPECTRAL_GATE_BUILDERS = {
  spectral_gate: buildSpectralGate,
};

export default R13_SPECTRAL_GATE_BUILDERS;
