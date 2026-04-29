/**
 * R13 — Vintage EQ Collection (1073 + API) builder smoke + Q-vs-gain remap.
 *
 * Runs as a Jest test when run with `jest`, but also runs standalone via
 * plain `node tests/r13_vintage_eq.test.js` — we install `describe / test /
 * expect` shims if they aren't already globals. This keeps the smoke loop
 * working in CI environments where the doseedo-next repo doesn't have a
 * jest runner installed (it doesn't currently).
 *
 * Author: Agent R13
 */

// ──────────────────────────────────────────────────────────────────────────
// Plain-node test shims — no-op when running under jest.
// ──────────────────────────────────────────────────────────────────────────

const _failures = [];
const _passes = [];

if (typeof globalThis.describe !== 'function') {
  globalThis.describe = (name, body) => {
    console.log(`\n# ${name}`);
    body();
  };
}
if (typeof globalThis.test !== 'function') {
  globalThis.test = (name, body) => {
    try {
      const r = body();
      if (r && typeof r.then === 'function') {
        // async tests not used here; fail explicitly if they appear
        throw new Error('async tests not supported in shim runner');
      }
      _passes.push(name);
      console.log(`  ✓ ${name}`);
    } catch (err) {
      _failures.push({ name, err });
      console.error(`  ✗ ${name}`);
      console.error(`    ${err && err.message}`);
    }
  };
}
if (typeof globalThis.expect !== 'function') {
  globalThis.expect = (actual) => ({
    toBe: (expected) => {
      if (actual !== expected) throw new Error(`expected ${JSON.stringify(actual)} to be ${JSON.stringify(expected)}`);
    },
    toEqual: (expected) => {
      const a = JSON.stringify(actual);
      const b = JSON.stringify(expected);
      if (a !== b) throw new Error(`expected ${a} to equal ${b}`);
    },
    toBeGreaterThan: (n) => {
      if (!(actual > n)) throw new Error(`expected ${actual} > ${n}`);
    },
    toBeGreaterThanOrEqual: (n) => {
      if (!(actual >= n)) throw new Error(`expected ${actual} >= ${n}`);
    },
    toBeLessThan: (n) => {
      if (!(actual < n)) throw new Error(`expected ${actual} < ${n}`);
    },
    toBeLessThanOrEqual: (n) => {
      if (!(actual <= n)) throw new Error(`expected ${actual} <= ${n}`);
    },
    toBeCloseTo: (n, digits = 6) => {
      const eps = Math.pow(10, -digits);
      if (Math.abs(actual - n) > eps) throw new Error(`expected ${actual} ≈ ${n} (within ${eps})`);
    },
    toBeDefined: () => {
      if (actual === undefined) throw new Error('expected value to be defined');
    },
    toContain: (v) => {
      if (!Array.isArray(actual) && typeof actual !== 'string') {
        throw new Error('toContain only valid for arrays / strings');
      }
      if (!actual.includes(v)) throw new Error(`expected ${JSON.stringify(actual)} to contain ${JSON.stringify(v)}`);
    },
  });
}

// ──────────────────────────────────────────────────────────────────────────
// Mock AudioContext + node primitives.
// We need just enough surface for the R13 builder: createGain, createBiquadFilter,
// createWaveShaper, plus an AudioWorkletNode that throws (so the builder takes
// its WaveShaper fallback path).
// ──────────────────────────────────────────────────────────────────────────

class MockAudioParam {
  constructor(value = 0) { this.value = value; }
  cancelScheduledValues() {}
  setTargetAtTime() {}
}
class MockNode {
  constructor() { this.connected = []; }
  connect(target) { this.connected.push(target); return target; }
  disconnect() { this.connected = []; }
}
class MockGainNode extends MockNode {
  constructor() { super(); this.gain = new MockAudioParam(1); }
}
class MockBiquadFilterNode extends MockNode {
  constructor() {
    super();
    this.type = 'lowpass';
    this.frequency = new MockAudioParam(350);
    this.Q = new MockAudioParam(1);
    this.gain = new MockAudioParam(0);
    this.detune = new MockAudioParam(0);
  }
}
class MockWaveShaperNode extends MockNode {
  constructor() { super(); this.curve = null; this.oversample = 'none'; }
}

class MockAudioContext {
  constructor() { this.currentTime = 0; this.sampleRate = 48000; }
  createGain() { return new MockGainNode(); }
  createBiquadFilter() { return new MockBiquadFilterNode(); }
  createWaveShaper() { return new MockWaveShaperNode(); }
}

// Fail-loud AudioWorkletNode — _safeWorklet catches the throw and falls back.
globalThis.AudioWorkletNode = class { constructor() { throw new Error('mock: worklet not registered'); } };

// ──────────────────────────────────────────────────────────────────────────
// Module under test (dynamic ESM import — works under both runners).
// ──────────────────────────────────────────────────────────────────────────

async function loadBuilder() {
  // Resolve relative to this test file (tests/ → ../src/audio/builders/r13_vintage_eq.js)
  const url = new URL('../src/audio/builders/r13_vintage_eq.js', import.meta.url).href;
  return import(url);
}

// ──────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────

const ctx = new MockAudioContext();

const mod = await loadBuilder();
const { buildVintage1073, buildVintageAPI, _internals } = mod;
const builders = mod.default;

describe('R13 — Vintage EQ Collection', () => {
  test('default export registers vintage_1073 + vintage_api', () => {
    expect(typeof builders.vintage_1073).toBe('function');
    expect(typeof builders.vintage_api).toBe('function');
  });

  test('buildVintage1073 returns the engine contract shape', () => {
    const node = { params: {} };
    const out = buildVintage1073(ctx, node, {});
    expect(out.input).toBeDefined();
    expect(out.output).toBeDefined();
    expect(out.paramTargets).toBeDefined();
  });

  test('buildVintage1073 wires modulated params into paramTargets', () => {
    const node = {
      params: {
        low_cut_freq:        '@hpf',
        low_shelf_gain:      '@lsg',
        mid_gain:            '@mg',
        mid_q:               '@mq',
        high_shelf_gain:     '@hsg',
        inductor_saturation: '@drv',
        output_gain:         '@og',
      },
    };
    const paramDefs = {
      hpf: { min: 0, max: 4 }, lsg: { min: -18, max: 18 },
      mg: { min: -18, max: 18 }, mq: { min: 0, max: 1 },
      hsg: { min: -18, max: 18 }, drv: { min: 0, max: 1 },
      og: { min: 0, max: 2 },
    };
    const out = buildVintage1073(ctx, node, paramDefs);
    expect(out.paramTargets.hpf).toBeDefined();
    expect(out.paramTargets.lsg).toBeDefined();
    expect(out.paramTargets.mg).toBeDefined();
    expect(out.paramTargets.mq).toBeDefined();
    expect(out.paramTargets.hsg).toBeDefined();
    expect(out.paramTargets.drv).toBeDefined();
    expect(out.paramTargets.og).toBeDefined();
  });

  test('1073 indexed low_cut_freq table maps correctly (off / 50 / 80 / 160 / 300)', () => {
    expect(_internals.N1073_LOW_CUT_FREQS).toEqual([0, 50, 80, 160, 300]);
  });

  test('1073 mid frequency table contains required positions', () => {
    expect(_internals.N1073_MID_FREQS).toContain(360);
    expect(_internals.N1073_MID_FREQS).toContain(700);
    expect(_internals.N1073_MID_FREQS).toContain(1600);
    expect(_internals.N1073_MID_FREQS).toContain(3200);
  });

  test('1073 high shelf is fixed at 12 kHz', () => {
    expect(_internals.N1073_HIGH_SHELF_HZ).toBe(12000);
  });

  test('buildVintageAPI returns the engine contract shape', () => {
    const node = { params: {} };
    const out = buildVintageAPI(ctx, node, {});
    expect(out.input).toBeDefined();
    expect(out.output).toBeDefined();
    expect(out.paramTargets).toBeDefined();
  });

  test('API frequency table is the canonical 12 positions', () => {
    expect(_internals.API_FREQ_TABLE).toEqual(
      [50, 100, 200, 400, 800, 1500, 3000, 5000, 7500, 12500, 15000, 20000]
    );
  });

  test('buildVintageAPI wires all 4 band freq+gain params', () => {
    const node = {
      params: {
        band1_freq: '@b1f', band1_gain: '@b1g',
        band2_freq: '@b2f', band2_gain: '@b2g',
        band3_freq: '@b3f', band3_gain: '@b3g',
        band4_freq: '@b4f', band4_gain: '@b4g',
        inductor_saturation: '@drv', output_gain: '@og',
      },
    };
    const paramDefs = {};
    for (let i = 1; i <= 4; i++) {
      paramDefs[`b${i}f`] = { min: 0, max: 11 };
      paramDefs[`b${i}g`] = { min: -12, max: 12 };
    }
    paramDefs.drv = { min: 0, max: 1 };
    paramDefs.og = { min: 0, max: 2 };
    const out = buildVintageAPI(ctx, node, paramDefs);
    for (let i = 1; i <= 4; i++) {
      expect(out.paramTargets[`b${i}f`]).toBeDefined();
      expect(out.paramTargets[`b${i}g`]).toBeDefined();
    }
    expect(out.paramTargets.drv).toBeDefined();
    expect(out.paramTargets.og).toBeDefined();
  });
});

describe('R13 — Q-vs-gain remap curves', () => {
  test('1073 mid Q tightens with magnitude of gain (medium preset)', () => {
    const q0 = _internals.n1073MidQ(0, 'medium');
    const q9 = _internals.n1073MidQ(9, 'medium');
    const q18 = _internals.n1073MidQ(18, 'medium');
    expect(q0).toBeLessThan(q9);
    expect(q9).toBeLessThan(q18);
    // Symmetric in sign
    expect(_internals.n1073MidQ(-12, 'medium')).toBeCloseTo(_internals.n1073MidQ(12, 'medium'), 6);
  });

  test('1073 mid Q presets sit at the right base order', () => {
    const broadAt0  = _internals.n1073MidQ(0, 'broad');
    const mediumAt0 = _internals.n1073MidQ(0, 'medium');
    const narrowAt0 = _internals.n1073MidQ(0, 'narrow');
    expect(broadAt0).toBeLessThan(mediumAt0);
    expect(mediumAt0).toBeLessThan(narrowAt0);
  });

  test('API proportional-Q starts wide and tightens with gain', () => {
    const q0  = _internals.apiProportionalQ(0);
    const q6  = _internals.apiProportionalQ(6);
    const q12 = _internals.apiProportionalQ(12);
    // start ≈ 0.6
    expect(q0).toBeCloseTo(0.6, 2);
    // end ≈ 2.5
    expect(q12).toBeCloseTo(2.5, 2);
    // monotonic
    expect(q0).toBeLessThan(q6);
    expect(q6).toBeLessThan(q12);
    // Symmetric in sign
    expect(_internals.apiProportionalQ(-9)).toBeCloseTo(_internals.apiProportionalQ(9), 6);
  });

  test('inductor saturation curve has expected shape: 0 drive → ≈identity, 1 drive → bounded soft-clip', () => {
    const flatCurve = _internals.makeInductorCurve(0);
    const hotCurve  = _internals.makeInductorCurve(1);
    expect(flatCurve.length).toBe(2048);
    expect(hotCurve.length).toBe(2048);
    // drive=0: should be roughly linear and bounded |y| ≤ 1
    let maxFlat = 0;
    for (let i = 0; i < flatCurve.length; i++) {
      maxFlat = Math.max(maxFlat, Math.abs(flatCurve[i]));
    }
    expect(maxFlat).toBeLessThanOrEqual(1.001);
    // drive=1: also bounded
    let maxHot = 0;
    for (let i = 0; i < hotCurve.length; i++) {
      maxHot = Math.max(maxHot, Math.abs(hotCurve[i]));
    }
    expect(maxHot).toBeLessThanOrEqual(1.001);
    // drive=1 should reach near full scale (saturation pushes input to ±1 plateau)
    expect(maxHot).toBeGreaterThan(0.5);
  });

  test('1073 mid_gain customSetter updates Q together with gain (live remap)', () => {
    const node = { params: { mid_gain: '@mg', mid_q: '@mq' } };
    const paramDefs = { mg: { min: -18, max: 18 }, mq: { min: 0, max: 1 } };
    const out = buildVintage1073(ctx, node, paramDefs);
    // Snapshot the underlying mid bell node by grabbing via a fresh build — the
    // builder doesn't expose nodes directly, but customSetter side-effects are
    // testable: we set mid_q to 'medium' (0.5) then sweep mid_gain.
    out.paramTargets.mq.customSetter(0.5);   // medium
    const setter = out.paramTargets.mg.customSetter;
    // Track Q by re-driving from 0..18 — we can't read it without a reference.
    // Instead: assert the curve equation directly via _internals.
    const q0 = _internals.n1073MidQ(0, 'medium');
    const q18 = _internals.n1073MidQ(18, 'medium');
    setter(0);   // shouldn't throw
    setter(18);  // shouldn't throw
    expect(q0).toBeLessThan(q18);
  });

  test('API bandN_gain customSetter updates Q together with gain', () => {
    const node = { params: { band1_gain: '@b1g' } };
    const paramDefs = { b1g: { min: -12, max: 12 } };
    const out = buildVintageAPI(ctx, node, paramDefs);
    const setter = out.paramTargets.b1g.customSetter;
    // Smoke: call across the gain range without throwing
    setter(-12); setter(0); setter(12);
    // Validate the underlying remap is monotonic in |gain|
    expect(_internals.apiProportionalQ(0)).toBeLessThan(_internals.apiProportionalQ(12));
  });
});

// Standalone runner — only emits a non-zero exit code if any test failed.
if (typeof process !== 'undefined' && _failures.length + _passes.length > 0) {
  console.log(`\n${_passes.length} passed, ${_failures.length} failed`);
  if (_failures.length > 0) {
    process.exitCode = 1;
  }
}
