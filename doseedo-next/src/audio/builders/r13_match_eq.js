/**
 * R13 — Match EQ builder.
 *
 * Registers `match_eq` as a new node type. Internally it constructs an
 * AudioWorkletNode running `r13-match-eq-processor` (long-FFT magnitude
 * matcher with three modes: analyze_target / analyze_source / apply).
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildMatchEQ(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Worklet not yet registered? We still bind a passthrough GainNode and
 * surface the param IDs as no-ops so the engine doesn't error out. On the
 * next graph rebuild after `audioWorklet.addModule(...)` resolves, the
 * worklet path lights up.
 *
 * Schema (all values may be a literal OR an `@<paramId>` binding):
 *   {
 *     type: 'match_eq',
 *     params: {
 *       mode:                  'analyze_target' | 'analyze_source' | 'apply' | '@<id>',
 *       curve_smoothing_octave: 0.33,        // (UI hint; smoothing is host-side)
 *       curve_amount:           0..1,
 *       low_cut:                Hz,          // mapped to lowBin via fftSize/sr
 *       high_cut:               Hz,
 *       gain_makeup:            -12..+12 dB,
 *       target_curve:           Float32Array | number[]   // halfSize+1 magnitudes
 *       source_curve:           Float32Array | number[]
 *       fft_size:               1024 | 2048 | 4096 | 8192  // default 4096
 *     }
 *   }
 *
 * Authored to match the Match-EQ row of `PLUGIN_PARITY_ROADMAP.md` (Tier 2).
 *
 * @author Doseedo R13
 */

const PROC_NAME = 'r13-match-eq-processor';
const WORKLET_REL = '../../lib/web-audio-plugins/worklets/r13-match-eq-processor.js';

const MODE_NAME_TO_INDEX = {
  analyze_target: 0,
  analyze_source: 1,
  apply:          2,
};

function _modeIndex(value) {
  if (typeof value === 'number') return Math.max(0, Math.min(2, Math.round(value)));
  if (typeof value === 'string') {
    const idx = MODE_NAME_TO_INDEX[value.toLowerCase()];
    return idx == null ? 2 : idx;
  }
  return 2;
}

function _isAt(v) { return typeof v === 'string' && v.startsWith('@'); }

// Async-load the worklet module idempotently per-context. Synchronous
// builders kick this off and fall through to passthrough on the first call.
const _r13Registered = new WeakMap();
export async function ensureR13MatchEQWorklet(ctx) {
  if (!ctx || !ctx.audioWorklet) return false;
  if (_r13Registered.get(ctx) === true) return true;
  const pending = _r13Registered.get(ctx);
  if (pending && typeof pending.then === 'function') return pending;

  const promise = (async () => {
    try {
      const url = new URL(WORKLET_REL, import.meta.url).href;
      await ctx.audioWorklet.addModule(url);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn('[R13 match_eq] worklet load failed', err);
    }
    _r13Registered.set(ctx, true);
    return true;
  })();
  _r13Registered.set(ctx, promise);
  return promise;
}

function _safeWorklet(ctx, name, options = {}) {
  try {
    return new AudioWorkletNode(ctx, name, options);
  } catch (e) {
    if (typeof console !== 'undefined' && console.debug) {
      console.debug(`[R13 match_eq] worklet ${name} unavailable, fallback engaged:`, e && e.message);
    }
    return null;
  }
}

// Convert Hz → 0..1 normalized bin position relative to Nyquist (sr/2).
function _hzToNorm(hz, sampleRate) {
  if (!isFinite(hz) || hz <= 0) return 0;
  const nyq = sampleRate * 0.5;
  return Math.max(0, Math.min(1, hz / nyq));
}

function _dbToLin(db) { return Math.pow(10, db / 20); }

// Lightweight Float32Array coercion (accept arrays from JSON or typed input)
function _asF32(v) {
  if (!v) return null;
  if (v instanceof Float32Array) return v;
  if (Array.isArray(v)) return Float32Array.from(v);
  if (ArrayBuffer.isView(v)) return Float32Array.from(v);
  return null;
}

export function buildMatchEQ(ctx, nodeDef, paramDefs) {
  const params = (nodeDef && nodeDef.params) || {};
  const targets = {};

  const input  = ctx.createGain();
  const output = ctx.createGain();

  // Kick off worklet registration; returns immediately, ready next rebuild.
  ensureR13MatchEQWorklet(ctx);

  const fftSize = (typeof params.fft_size === 'number' && [1024, 2048, 4096, 8192].includes(params.fft_size))
    ? params.fft_size
    : 4096;
  const halfSize = fftSize >> 1;

  const initialMode = _modeIndex(_isAt(params.mode) ? 'apply' : (params.mode ?? 'apply'));

  const targetCurveF32 = _asF32(params.target_curve);
  const sourceCurveF32 = _asF32(params.source_curve);

  const node = _safeWorklet(ctx, PROC_NAME, {
    numberOfInputs: 1,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    parameterData: { mode: initialMode },
    processorOptions: {
      fftSize,
      targetCurve: (targetCurveF32 && targetCurveF32.length === halfSize + 1) ? targetCurveF32 : undefined,
      sourceCurve: (sourceCurveF32 && sourceCurveF32.length === halfSize + 1) ? sourceCurveF32 : undefined,
    },
  });

  // ── Fallback path: passthrough. Param surface still binds so the engine
  // doesn't see a different shape when the worklet isn't loaded yet.
  if (!node) {
    input.connect(output);
    for (const k of ['mode', 'curve_amount', 'low_cut', 'high_cut', 'gain_makeup',
                      'curve_smoothing_octave', 'target_curve', 'source_curve']) {
      if (_isAt(params[k])) {
        const id = params[k].slice(1);
        targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
      }
    }
    return { input, output, paramTargets: targets };
  }

  // Worklet wired. If both target + source curves were preloaded, ask the
  // worklet to compute its match curve right now (otherwise it stays flat).
  if (targetCurveF32 && sourceCurveF32) {
    try { node.port.postMessage({ type: 'recompute' }); } catch (e) { /* ignore */ }
  }

  input.connect(node);
  node.connect(output);

  const modeP    = node.parameters.get('mode');
  const amtP     = node.parameters.get('amount');
  const loP      = node.parameters.get('lowBin');
  const hiP      = node.parameters.get('highBin');
  const makeupP  = node.parameters.get('gainMakeup');

  // ── mode (string or number, optionally bound) ────────────────────────────
  if (_isAt(params.mode)) {
    const id = params.mode.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => {
        const idx = (typeof v === 'string') ? _modeIndex(v) :
                    (v >= 0 && v <= 1) ? Math.min(2, Math.floor(v * 3)) :
                    _modeIndex(v);
        try { node.port.postMessage({ type: 'mode', value: idx }); } catch (e) { /* ignore */ }
        if (modeP) modeP.value = idx;
      },
    };
  } else if (params.mode !== undefined) {
    const idx = _modeIndex(params.mode);
    if (modeP) modeP.value = idx;
  }

  // ── curve_amount (0..1) ──────────────────────────────────────────────────
  if (_isAt(params.curve_amount)) {
    const id = params.curve_amount.slice(1);
    if (amtP) targets[id] = { audioParam: amtP, paramDef: paramDefs[id] };
    else targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
  } else if (typeof params.curve_amount === 'number') {
    if (amtP) amtP.value = +params.curve_amount;
  }

  // ── low_cut / high_cut (Hz) → normalized bin position ────────────────────
  const sr = ctx.sampleRate || 44100;
  if (_isAt(params.low_cut)) {
    const id = params.low_cut.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => {
        if (loP) loP.value = _hzToNorm(+v, sr);
      },
    };
  } else if (typeof params.low_cut === 'number') {
    if (loP) loP.value = _hzToNorm(params.low_cut, sr);
  }

  if (_isAt(params.high_cut)) {
    const id = params.high_cut.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => {
        if (hiP) hiP.value = _hzToNorm(+v, sr);
      },
    };
  } else if (typeof params.high_cut === 'number') {
    if (hiP) hiP.value = _hzToNorm(params.high_cut, sr);
  }

  // ── gain_makeup (dB → linear) ────────────────────────────────────────────
  if (_isAt(params.gain_makeup)) {
    const id = params.gain_makeup.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => { if (makeupP) makeupP.value = _dbToLin(+v); },
    };
  } else if (typeof params.gain_makeup === 'number') {
    if (makeupP) makeupP.value = _dbToLin(params.gain_makeup);
  }

  // ── target_curve / source_curve port messages ────────────────────────────
  if (_isAt(params.target_curve)) {
    const id = params.target_curve.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => {
        const f = _asF32(v);
        if (f) {
          try { node.port.postMessage({ type: 'set_target', curve: f }); } catch (e) { /* ignore */ }
          try { node.port.postMessage({ type: 'recompute' }); } catch (e) { /* ignore */ }
        }
      },
    };
  }
  if (_isAt(params.source_curve)) {
    const id = params.source_curve.slice(1);
    targets[id] = {
      paramDef: paramDefs[id],
      customSetter: (v) => {
        const f = _asF32(v);
        if (f) {
          try { node.port.postMessage({ type: 'set_source', curve: f }); } catch (e) { /* ignore */ }
          try { node.port.postMessage({ type: 'recompute' }); } catch (e) { /* ignore */ }
        }
      },
    };
  }

  // curve_smoothing_octave is informational — the actual smoothing is done
  // host-side before posting `set_curve`. Surface as a no-op binding when
  // wired so the param shape stays consistent across builds.
  if (_isAt(params.curve_smoothing_octave)) {
    const id = params.curve_smoothing_octave.slice(1);
    targets[id] = { paramDef: paramDefs[id], customSetter: () => {} };
  }

  return { input, output, paramTargets: targets, workletNode: node };
}

// ── Smoothing helper (1/N-octave geometric mean) — exposed for hosts ──────
//
// `mags` is a length=halfSize+1 magnitude array. `octaveFraction` ∈ (0..1]
// (e.g. 1/3 for third-octave smoothing). Smoothing happens in log-magnitude
// space so a +6 dB peak averages to +3 dB across a centered window of
// equal weight, not √2× linear.
export function smoothCurveOctave(mags, sampleRate, octaveFraction = 1 / 3) {
  const N = mags.length;
  if (N < 2 || octaveFraction <= 0) return Float32Array.from(mags);
  const out = new Float32Array(N);
  const nyq = sampleRate * 0.5;
  const dbIn = new Float32Array(N);
  for (let i = 0; i < N; i++) dbIn[i] = 20 * Math.log10(Math.max(1e-12, mags[i]));
  for (let i = 0; i < N; i++) {
    const fc = (i / (N - 1)) * nyq;
    if (fc <= 0) { out[i] = mags[i]; continue; }
    const lo = fc * Math.pow(2, -octaveFraction * 0.5);
    const hi = fc * Math.pow(2,  octaveFraction * 0.5);
    const loIdx = Math.max(0, Math.floor((lo / nyq) * (N - 1)));
    const hiIdx = Math.min(N - 1, Math.ceil((hi / nyq) * (N - 1)));
    let sum = 0, count = 0;
    for (let k = loIdx; k <= hiIdx; k++) { sum += dbIn[k]; count++; }
    const avgDb = count > 0 ? sum / count : dbIn[i];
    out[i] = Math.pow(10, avgDb / 20);
  }
  return out;
}

// Compute a match curve (target / source) with eps guard. Both inputs are
// linear-magnitude arrays of equal length. Output is linear-gain.
export function computeMatchCurve(targetMags, sourceMags, eps = 1e-9) {
  const N = Math.min(targetMags.length, sourceMags.length);
  const out = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    out[i] = targetMags[i] / Math.max(eps, sourceMags[i]);
  }
  return out;
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_MATCH_EQ_BUILDERS = {
  match_eq: buildMatchEQ,
};

export default R13_MATCH_EQ_BUILDERS;
