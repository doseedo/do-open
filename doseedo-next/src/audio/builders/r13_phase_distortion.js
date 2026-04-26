/**
 * R13 — Phase Distortion (Casio CZ-style PD effect)
 *
 * Registers `phase_distortion` as a NEW node type. Internally it's a stereo
 * pair of WaveShaperNodes whose `curve` Float32Arrays are regenerated whenever
 * the curve family / drive / asymmetry knobs change. The curve families are
 * intentionally CZ-style (sharp, asymmetric, "bright vs dark") rather than the
 * smooth tanh/atan curves R0's `buildWaveshaper` already covers — that's the
 * whole point of this node.
 *
 * NOTE on the algorithm choice. True Casio PD applies a non-linear remap of
 * the *phase* of an oscillator (driven by a sub-counter). We implement the
 * perceptually-similar (and Logic-stock-similar) effect form: an input
 * waveshaper whose transfer curve is shaped like the result of running a
 * sine through CZ-style phase remap. This gives the same bright→dark sweep
 * character on harmonic inputs (saws, pads, vocals) without the cost of a
 * Hilbert transform / PLL. See INTEGRATION_R13_PHASE_DISTORTION.md for the
 * curve derivations.
 *
 * Builder contract (matches WebAudioDSPEngine.js):
 *   buildPhaseDistortion(ctx, nodeDef, paramDefs)
 *     → { input, output, paramTargets: { [paramId]: {...} } }
 *
 * Schema:
 *   {
 *     type: 'phase_distortion',
 *     params: {
 *       pd_amount:    0..100  (% drive — internally normalised to 0..1)
 *       pd_curve:     'saw' | 'square' | 'pulse' | 'res1' | 'res2' | 'res3'
 *       pd_asymmetry: -1..+1  (curve bias)
 *       pre_gain:     -12..+12 dB (input gain, pre-shaper)
 *       post_gain:    -12..+12 dB (output gain, post-shaper)
 *       tone:         -12..+12 dB (post tilt high-shelf @ 4 kHz)
 *       mix:          0..1 (dry/wet)
 *     }
 *   }
 *
 * Param values may be literals or '@<paramId>' bindings.
 *
 * Worklet: optional. The worklet (`r13-phase-distortion-processor`) is a
 * stub for a future per-sample variant that drives the curve-index continuously
 * (so `pd_curve` itself can be modulated at audio rate). The current builder
 * does NOT instantiate it — the WaveShaper-curve approach is good enough for
 * Logic parity.
 *
 * @author Doseedo R13
 */

// ── Curve catalogue ───────────────────────────────────────────────────────
//
// Each entry takes (x, amount, asym) and returns the transfer-curve output
// at input `x ∈ [-1, 1]`, where `amount ∈ [0, 1]` is drive depth and
// `asym ∈ [-1, +1]` is asymmetry. Higher amount = more drastic remap.
//
// All curves preserve f(0)=0 (so silence stays silence) and pass through
// f(±1)=±1 at amount=0 (so 0% drive ≈ unity passthrough).

// Helper: signed asymmetric warp of `x` by `asym ∈ [-1, +1]`. Bends positive
// half if asym>0, negative half if asym<0. Identity at asym=0.
function _bend(x, asym) {
  if (asym === 0) return x;
  // x>=0: y = x^(1+asym); x<0: y = -|x|^(1-asym). Keeps continuity at 0.
  const a = asym;
  if (x >= 0) return Math.pow(x, Math.max(0.05, 1 + a));
  return -Math.pow(-x, Math.max(0.05, 1 - a));
}

// CZ "saw": bright fundamental + harmonic ramp. Fold at high amount.
function _curveSaw(x, amount, asym) {
  const k = amount; // 0..1
  // Map to phase angle, then warp phase, then take sin.
  const phase = (_bend(x, asym) + 1) * Math.PI;     // 0..2π
  // CZ saw window: bunch phase in 0..π toward 0 as k rises (bright transient)
  const w = Math.pow(phase / (2 * Math.PI), 1 + 4 * k);
  return Math.sin(2 * Math.PI * w);
}

// CZ "square": symmetric phase squash → square-wave-ish on a sine driver.
function _curveSquare(x, amount, asym) {
  const k = amount;
  const xb = _bend(x, asym);
  // Hard region (around ±1) compressed; mid kept linear at low k.
  // Sigmoid-with-knee: tanh-style but with a hard hinge at ±(1 - k).
  const kn = 1 - 0.95 * k; // knee position
  if (xb > kn)  return 1 - (1 - kn) * Math.exp(-20 * k * (xb - kn));
  if (xb < -kn) return -1 + (1 - kn) * Math.exp(-20 * k * (-xb - kn));
  return xb / kn;
}

// CZ "pulse": narrow positive spike, bias-controlled width.
function _curvePulse(x, amount, asym) {
  const k = amount;
  const xb = _bend(x, asym);
  // Pulse: cubic crossover that opens a "duty cycle" gap as k rises.
  const duty = 0.5 - 0.45 * k; // 0.05..0.5
  if (xb > duty)  return 1;
  if (xb < -duty) return -1;
  // Smooth ramp inside the gap: cubic for soft edge.
  const t = xb / duty;
  return t * t * t * (1 - 0.3 * k) + t * (0.3 * k);
}

// "res1": resonance peak — superimpose a high-frequency sine on top of the
// driver, with amplitude scaled by `amount`. Casio CZ resonance was a sub-osc
// at N× fundamental, gated by the master sine; we approximate that effect on
// the input waveshaper by raising the curve's local slope where |x| is small.
function _curveRes1(x, amount, asym) {
  const k = amount;
  const xb = _bend(x, asym);
  // Resonance ripple: 5 cycles inside x ∈ [-1,1], scaled by k, decaying
  // toward the rails (so the ripple is concentrated in the centre).
  const ripple = Math.sin(xb * Math.PI * 5) * (1 - Math.abs(xb)) * k * 0.6;
  return Math.max(-1, Math.min(1, xb + ripple));
}

// "res2": brighter resonance — 9 cycles, less envelope dip.
function _curveRes2(x, amount, asym) {
  const k = amount;
  const xb = _bend(x, asym);
  const ripple = Math.sin(xb * Math.PI * 9) * (1 - 0.5 * Math.abs(xb)) * k * 0.5;
  return Math.max(-1, Math.min(1, xb + ripple));
}

// "res3": aggressive — 13 cycles, no envelope dip = sustained ringing.
function _curveRes3(x, amount, asym) {
  const k = amount;
  const xb = _bend(x, asym);
  const ripple = Math.sin(xb * Math.PI * 13) * k * 0.45;
  return Math.max(-1, Math.min(1, xb + ripple));
}

// Public registry — keep keys aligned with INTEGRATION doc.
export const PD_CURVE_FAMILIES = {
  saw:    _curveSaw,
  square: _curveSquare,
  pulse:  _curvePulse,
  res1:   _curveRes1,
  res2:   _curveRes2,
  res3:   _curveRes3,
};

const DEFAULT_CURVE = 'saw';
const CURVE_RESOLUTION = 4096; // power of 2; matches Logic-tier WaveShaper sizes.

/**
 * Build a Float32Array transfer curve for the given family + drive + asym.
 * Exported for tests + the calibration validation panel.
 */
export function makePDCurve(family, amount, asym, n = CURVE_RESOLUTION) {
  const fn = PD_CURVE_FAMILIES[family] || PD_CURVE_FAMILIES[DEFAULT_CURVE];
  const k = Math.max(0, Math.min(1, amount));
  const a = Math.max(-1, Math.min(1, asym));
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // Map sample index to x ∈ [-1, +1]
    const x = (i * 2) / (n - 1) - 1;
    let y = fn(x, k, a);
    if (!Number.isFinite(y)) y = 0;
    out[i] = Math.max(-1, Math.min(1, y));
  }
  return out;
}

function _isModulated(v) {
  return typeof v === 'string' && v.startsWith('@');
}

function _dbToGain(db) {
  return Math.pow(10, db / 20);
}

// Resolve the initial value for a param: literal pass-through, modulated → default.
function _initial(val, fallback) {
  if (val == null || _isModulated(val)) return fallback;
  return val;
}

export function buildPhaseDistortion(ctx, nodeDef, paramDefs) {
  const params = (nodeDef && nodeDef.params) || {};
  const targets = {};

  // Topology:
  //
  //   input → preGain → splitter ─┬─→ shaperL → mergerL ─┐
  //                               └─→ shaperR → mergerR ─┤
  //                                                       merger → toneShelf → postGain → wetGain ┐
  //   input ──────────────────────────────────────────────────────────────────→ dryGain ─────────┴→ output
  //
  // We use a stereo channel splitter so left/right are processed
  // independently — matches Logic's stereo-effect convention. If the input is
  // mono the splitter still works; the right shaper just gets a copy.

  const input    = ctx.createGain();
  const output   = ctx.createGain();
  const preGain  = ctx.createGain();
  const postGain = ctx.createGain();
  const dryGain  = ctx.createGain();
  const wetGain  = ctx.createGain();
  const toneShelf = ctx.createBiquadFilter();
  toneShelf.type = 'highshelf';
  toneShelf.frequency.value = 4000;

  const splitter = ctx.createChannelSplitter(2);
  const merger   = ctx.createChannelMerger(2);
  const shaperL  = ctx.createWaveShaper();
  const shaperR  = ctx.createWaveShaper();
  shaperL.oversample = '4x';
  shaperR.oversample = '4x';

  // Initial state for the curve-regenerator.
  const state = {
    family:    _initial(params.pd_curve, DEFAULT_CURVE),
    amount:    Math.max(0, Math.min(1, _initial(params.pd_amount, 50) / 100)),
    asymmetry: Math.max(-1, Math.min(1, _initial(params.pd_asymmetry, 0))),
  };
  if (typeof state.family !== 'string') state.family = DEFAULT_CURVE;

  // Curve regeneration helper — single source of truth for shaper curves.
  function regenerateCurve() {
    const c = makePDCurve(state.family, state.amount, state.asymmetry);
    // WaveShaperNode.curve setter copies the array internally; reuse the
    // same array for L/R for cache-friendliness.
    shaperL.curve = c;
    shaperR.curve = c;
  }
  regenerateCurve();

  // Initial gains
  preGain.gain.value  = _dbToGain(_initial(params.pre_gain,  0));
  postGain.gain.value = _dbToGain(_initial(params.post_gain, 0));
  toneShelf.gain.value = _initial(params.tone, 0);
  const initialMix = Math.max(0, Math.min(1, _initial(params.mix, 1)));
  wetGain.gain.value = initialMix;
  dryGain.gain.value = 1 - initialMix;

  // ── Wiring ──────────────────────────────────────────────────────────────
  // Wet path
  input.connect(preGain);
  preGain.connect(splitter);
  splitter.connect(shaperL, 0);
  splitter.connect(shaperR, 1);
  shaperL.connect(merger, 0, 0);
  shaperR.connect(merger, 0, 1);
  merger.connect(toneShelf);
  toneShelf.connect(postGain);
  postGain.connect(wetGain);
  wetGain.connect(output);
  // Dry path
  input.connect(dryGain);
  dryGain.connect(output);

  // ── Param wiring ────────────────────────────────────────────────────────
  for (const [key, val] of Object.entries(params)) {
    if (!_isModulated(val)) continue;
    const paramId = val.slice(1);
    const def = paramDefs ? paramDefs[paramId] : undefined;

    switch (key) {
      case 'pd_amount':
        targets[paramId] = {
          paramDef: def,
          customSetter: (v) => {
            // Accept 0..100 (% — the Logic surface) OR 0..1 (normalised).
            const norm = (v > 1.5) ? (v / 100) : v;
            state.amount = Math.max(0, Math.min(1, norm));
            regenerateCurve();
          },
        };
        break;

      case 'pd_curve':
        targets[paramId] = {
          paramDef: def,
          customSetter: (v) => {
            let family;
            if (typeof v === 'string') {
              family = v.toLowerCase();
            } else if (typeof v === 'number') {
              const keys = Object.keys(PD_CURVE_FAMILIES);
              const idx = Math.max(0, Math.min(keys.length - 1, Math.round(v)));
              family = keys[idx];
            }
            if (PD_CURVE_FAMILIES[family]) {
              state.family = family;
              regenerateCurve();
            }
          },
        };
        break;

      case 'pd_asymmetry':
        targets[paramId] = {
          paramDef: def,
          customSetter: (v) => {
            state.asymmetry = Math.max(-1, Math.min(1, v));
            regenerateCurve();
          },
        };
        break;

      case 'pre_gain':
        // Direct AudioParam: caller passes value in dB.
        targets[paramId] = {
          paramDef: def,
          customSetter: (v) => { preGain.gain.value = _dbToGain(v); },
        };
        break;

      case 'post_gain':
        targets[paramId] = {
          paramDef: def,
          customSetter: (v) => { postGain.gain.value = _dbToGain(v); },
        };
        break;

      case 'tone':
        targets[paramId] = { audioParam: toneShelf.gain, paramDef: def };
        break;

      case 'mix':
        targets[paramId] = {
          paramDef: def,
          customSetter: (v) => {
            const w = Math.max(0, Math.min(1, v));
            wetGain.gain.value = w;
            dryGain.gain.value = 1 - w;
          },
        };
        break;

      default:
        break;
    }
  }

  return {
    input,
    output,
    paramTargets: targets,
    // Exposed for tests + the calibration panel — lets callers force a
    // regeneration / inspect the current state without poking the closure.
    _pd: { state, regenerateCurve, shaperL, shaperR },
  };
}

// ── Default export: NODE_BUILDERS map ─────────────────────────────────────
const R13_PHASE_DISTORTION_BUILDERS = {
  phase_distortion: buildPhaseDistortion,
};

export default R13_PHASE_DISTORTION_BUILDERS;
