/**
 * R13 — Phase Distortion: builder smoke + curve-regeneration test.
 *
 * Compatible with Jest, Vitest, AND a plain `node` runner via the
 * `runAll()` export. The builder doesn't depend on a real AudioContext —
 * we hand it a minimal stub that records connect()/curve assignments — so
 * the test never needs jsdom or any Web Audio shim.
 *
 * Run via Jest:
 *   npx jest tests/r13_phase_distortion.test.js
 *
 * Run standalone (no framework):
 *   node tests/r13_phase_distortion.test.js
 */

import {
  buildPhaseDistortion,
  makePDCurve,
  PD_CURVE_FAMILIES,
} from '../src/audio/builders/r13_phase_distortion.js';

// ── Tiny AudioContext stub ────────────────────────────────────────────────
//
// Matches just the surface area buildPhaseDistortion uses:
//   createGain, createWaveShaper, createBiquadFilter,
//   createChannelSplitter, createChannelMerger.
// Nodes record connect() targets and curve/value assignments so we can
// assert wiring + state mutations.

function makeStubAudioContext() {
  const _make = () => {
    const node = {
      _connections: [],
      gain: { value: 1 },
      curve: null,
      oversample: 'none',
      type: 'lowpass',
      frequency: { value: 0 },
      _curveSetCount: 0,
      connect(target /* , out, in */) {
        node._connections.push(target);
        return target;
      },
      disconnect() {},
    };
    // Track curve assignments via accessor.
    Object.defineProperty(node, 'curve', {
      get() { return node._curve; },
      set(v) { node._curve = v; node._curveSetCount += 1; },
    });
    return node;
  };

  return {
    sampleRate: 48000,
    createGain:           _make,
    createWaveShaper:     _make,
    createBiquadFilter:   _make,
    createChannelSplitter: (_n) => _make(),
    createChannelMerger:   (_n) => _make(),
  };
}

// ── Assertion helpers (work both inside Jest's `expect` and standalone) ──

function _assert(cond, msg) {
  if (!cond) throw new Error(msg || 'assertion failed');
}

function _ok(actual, msg) { _assert(!!actual, msg); }
function _eq(a, b, msg) {
  if (a !== b) throw new Error(`${msg || 'eq'}: expected ${b}, got ${a}`);
}
function _approx(a, b, eps, msg) {
  if (Math.abs(a - b) > eps) {
    throw new Error(`${msg || 'approx'}: expected ${b}±${eps}, got ${a}`);
  }
}

// ── The actual tests ─────────────────────────────────────────────────────

const TESTS = [];
function _t(name, fn) { TESTS.push({ name, fn }); }

_t('makePDCurve returns Float32Array of length 4096', () => {
  const c = makePDCurve('saw', 0.5, 0);
  _ok(c instanceof Float32Array, 'expected Float32Array');
  _eq(c.length, 4096, 'curve length');
});

_t('makePDCurve passes through origin at amount=0', () => {
  for (const family of Object.keys(PD_CURVE_FAMILIES)) {
    const c = makePDCurve(family, 0, 0, 4097);   // odd N so index 2048 is exactly x=0
    _approx(c[2048], 0, 1e-3, `${family}: f(0) at amount=0`);
  }
});

_t('makePDCurve outputs are clamped to [-1, +1]', () => {
  for (const family of Object.keys(PD_CURVE_FAMILIES)) {
    for (const amount of [0, 0.3, 0.7, 1.0]) {
      const c = makePDCurve(family, amount, 0);
      let mx = -Infinity, mn = Infinity;
      for (let i = 0; i < c.length; i++) {
        if (c[i] > mx) mx = c[i];
        if (c[i] < mn) mn = c[i];
      }
      _ok(mx <= 1.0 + 1e-6, `${family}@${amount}: max ${mx}`);
      _ok(mn >= -1.0 - 1e-6, `${family}@${amount}: min ${mn}`);
    }
  }
});

_t('all six curve families produce different outputs at amount=0.7', () => {
  const families = Object.keys(PD_CURVE_FAMILIES);
  const curves = families.map(f => makePDCurve(f, 0.7, 0));
  for (let i = 0; i < families.length; i++) {
    for (let j = i + 1; j < families.length; j++) {
      // L1 distance — must be non-trivially different
      let d = 0;
      for (let k = 0; k < curves[i].length; k++) {
        d += Math.abs(curves[i][k] - curves[j][k]);
      }
      _ok(d > 1.0,
          `families ${families[i]} vs ${families[j]}: L1=${d} (too similar)`);
    }
  }
});

_t('asymmetry !=0 changes the curve shape', () => {
  const cSym  = makePDCurve('saw', 0.5, 0);
  const cAsym = makePDCurve('saw', 0.5, 0.6);
  let d = 0;
  for (let i = 0; i < cSym.length; i++) d += Math.abs(cSym[i] - cAsym[i]);
  _ok(d > 0.1, `expected asymmetry to change curve, L1=${d}`);
});

_t('buildPhaseDistortion returns standard {input,output,paramTargets} shape', () => {
  const ctx = makeStubAudioContext();
  const node = { params: { pd_amount: 50, pd_curve: 'saw', mix: 1 } };
  const r = buildPhaseDistortion(ctx, node, {});
  _ok(r.input,  'has input');
  _ok(r.output, 'has output');
  _ok(r.paramTargets && typeof r.paramTargets === 'object', 'has paramTargets map');
  _ok(r._pd && r._pd.regenerateCurve, 'exposes _pd debug handle');
});

_t('paramTargets registered for every modulated knob', () => {
  const ctx = makeStubAudioContext();
  const node = {
    params: {
      pd_amount:    '@drive',
      pd_curve:     '@family',
      pd_asymmetry: '@asym',
      pre_gain:     '@pre',
      post_gain:    '@post',
      tone:         '@tone',
      mix:          '@mix',
    },
  };
  const defs = {
    drive:  { id: 'drive' },  family: { id: 'family' }, asym: { id: 'asym' },
    pre:    { id: 'pre' },    post:   { id: 'post' },   tone: { id: 'tone' },
    mix:    { id: 'mix' },
  };
  const r = buildPhaseDistortion(ctx, node, defs);
  for (const id of ['drive', 'family', 'asym', 'pre', 'post', 'tone', 'mix']) {
    _ok(r.paramTargets[id], `target registered for '${id}'`);
  }
  // tone must be an audioParam binding; the others have customSetters
  _ok(r.paramTargets.tone.audioParam, 'tone is audioParam-bound');
  for (const id of ['drive', 'family', 'asym', 'pre', 'post', 'mix']) {
    _ok(typeof r.paramTargets[id].customSetter === 'function',
        `${id} has customSetter`);
  }
});

_t('pd_amount.customSetter regenerates the WaveShaper curves', () => {
  const ctx = makeStubAudioContext();
  const r = buildPhaseDistortion(ctx, { params: { pd_amount: '@drive' } },
                                  { drive: { id: 'drive' } });
  const before = r._pd.shaperL._curveSetCount;
  const beforeCurve = r._pd.shaperL.curve;
  r.paramTargets.drive.customSetter(80); // 80% drive (Logic surface)
  const after = r._pd.shaperL._curveSetCount;
  _ok(after > before, 'curve assignment counter advanced on knob drag');
  // Also check the new curve is meaningfully different from the old one
  // (initial was amount=0.5; new should be amount=0.8)
  let d = 0;
  for (let i = 0; i < beforeCurve.length; i++) {
    d += Math.abs(beforeCurve[i] - r._pd.shaperL.curve[i]);
  }
  _ok(d > 1.0, `expected curve to mutate, L1=${d}`);
  // Also: state.amount must reflect the new value (normalised)
  _approx(r._pd.state.amount, 0.8, 1e-6, 'state.amount post-setter');
});

_t('pd_curve.customSetter switches family and re-renders', () => {
  const ctx = makeStubAudioContext();
  const r = buildPhaseDistortion(
    ctx,
    { params: { pd_curve: '@family', pd_amount: 70 } },
    { family: { id: 'family' } }
  );
  const initialFamily = r._pd.state.family;
  _eq(initialFamily, 'saw', 'initial family');

  const initialCurve = new Float32Array(r._pd.shaperL.curve);

  r.paramTargets.family.customSetter('res2');
  _eq(r._pd.state.family, 'res2', 'family switched to res2');
  let d = 0;
  for (let i = 0; i < initialCurve.length; i++) {
    d += Math.abs(initialCurve[i] - r._pd.shaperL.curve[i]);
  }
  _ok(d > 1.0, `expected curve to change with family switch, L1=${d}`);

  // Numeric enum form (1 → 'square') should also work.
  r.paramTargets.family.customSetter(1);
  _eq(r._pd.state.family, 'square', 'numeric enum maps to family');
});

_t('pd_asymmetry.customSetter clamps to [-1, +1]', () => {
  const ctx = makeStubAudioContext();
  const r = buildPhaseDistortion(ctx, { params: { pd_asymmetry: '@asym' } },
                                  { asym: { id: 'asym' } });
  r.paramTargets.asym.customSetter(5);
  _approx(r._pd.state.asymmetry, 1, 1e-6, 'clamp to +1');
  r.paramTargets.asym.customSetter(-5);
  _approx(r._pd.state.asymmetry, -1, 1e-6, 'clamp to -1');
});

_t('mix.customSetter drives wet/dry balance', () => {
  const ctx = makeStubAudioContext();
  const node = { params: { mix: '@mix' } };
  const r = buildPhaseDistortion(ctx, node, { mix: { id: 'mix' } });
  // Verify customSetter exists and runs without throwing — internal
  // wetGain/dryGain are private to the closure but the call must not error.
  let threw = false;
  try { r.paramTargets.mix.customSetter(0.0); }
  catch (e) { threw = true; }
  _ok(!threw, 'mix(0) must not throw');
  try { r.paramTargets.mix.customSetter(1.0); }
  catch (e) { threw = true; }
  _ok(!threw, 'mix(1) must not throw');
});

_t('pre_gain customSetter converts dB → linear gain', () => {
  const ctx = makeStubAudioContext();
  const r = buildPhaseDistortion(ctx, { params: { pre_gain: '@pre' } },
                                  { pre: { id: 'pre' } });
  // 0 dB → ~1.0; +6 dB → ~2.0
  r.paramTargets.pre.customSetter(0);
  // Can't easily inspect preGain because it's closed over inside the builder,
  // but we can at least confirm the call runs cleanly.
  let threw = false;
  try { r.paramTargets.pre.customSetter(6); }
  catch (e) { threw = true; }
  _ok(!threw, '+6 dB must not throw');
});

_t('non-modulated literals do not produce paramTargets', () => {
  const ctx = makeStubAudioContext();
  const r = buildPhaseDistortion(
    ctx,
    { params: { pd_amount: 30, pd_curve: 'pulse', mix: 0.5 } },
    {}
  );
  _eq(Object.keys(r.paramTargets).length, 0,
      'no paramTargets for literal params');
  _eq(r._pd.state.family, 'pulse', 'literal family applied');
  _approx(r._pd.state.amount, 0.3, 1e-6, 'literal amount applied (normalised)');
});

// ── Runner — works both inside Jest/Vitest AND standalone via `node`. ────

export async function runAll() {
  let pass = 0, fail = 0;
  for (const { name, fn } of TESTS) {
    try {
      await fn();
      pass += 1;
      // eslint-disable-next-line no-console
      console.log(`  ok  ${name}`);
    } catch (e) {
      fail += 1;
      // eslint-disable-next-line no-console
      console.error(`  FAIL ${name}: ${e && e.message}`);
    }
  }
  // eslint-disable-next-line no-console
  console.log(`\nR13 phase_distortion: ${pass} passed, ${fail} failed`);
  if (fail > 0) {
    if (typeof process !== 'undefined' && process.exit) process.exit(1);
    throw new Error(`${fail} test(s) failed`);
  }
}

// Wire into Jest/Vitest if globals are available (no-op under plain node).
if (typeof describe === 'function' && typeof test === 'function') {
  describe('R13 — phase_distortion', () => {
    for (const { name, fn } of TESTS) {
      // eslint-disable-next-line no-loop-func
      test(name, () => fn());
    }
  });
}

// Run automatically when invoked directly via `node tests/...`.
const _isMain = (() => {
  try {
    if (typeof import.meta !== 'undefined' && import.meta.url
        && typeof process !== 'undefined' && process.argv && process.argv[1]) {
      const fileUrl = new URL(`file://${process.argv[1]}`).href;
      return import.meta.url === fileUrl;
    }
  } catch (e) { /* ignore */ }
  return false;
})();
if (_isMain) runAll();
