/**
 * R13 Bass Amp Designer composite — builder smoke + behaviour tests.
 *
 * Run with:
 *   node --test tests/r13_bass_amp_designer.test.js
 *
 * The Web Audio API is not available in Node, so we stub a minimal
 * AudioContext + force the worklet path to fail. The composite builder
 * must cleanly fall back to native-node substitutes (waveshaper / biquad /
 * dynamics-compressor / convolver / gain) and still return a
 * `{ input, output, paramTargets }` shape.
 *
 * Coverage:
 *   1. Builder smoke — fallback path returns the canonical contract with
 *      every modulated param mapped into paramTargets.
 *   2. amp_model + cab_model setters — every enum value invokes without
 *      throwing.
 *   3. tube_blend extremes — 0.0 zeroes the tube stage gain.
 *   4. compression knob — higher value lowers the DynamicsCompressor's
 *      threshold (more aggressive comp).
 *   5. DI vs Mic blend — direct_out_mix=1.0 routes signal entirely
 *      through DI (mic_gain=0, di_gain=1).
 *   6. cab_model='di_only' — clears the convolver IR buffer.
 *   7. Default export — registers `bass_amp_designer` node type.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  buildBassAmpDesigner,
  AMP_MODELS,
  CAB_MODELS,
  GRAPHIC_EQ_FREQS,
  DEFAULT_AMP_MODEL,
  DEFAULT_CAB_MODEL,
} from '../src/audio/builders/r13_bass_amp_designer.js';
import R13_BASS_AMP_BUILDERS from '../src/audio/builders/r13_bass_amp_designer.js';

// ── Minimal AudioContext stub ────────────────────────────────────────────
function makeStubCtx() {
  const sampleRate = 48000;
  const currentTime = 0;

  function makeAudioParam(initial = 0) {
    return {
      value: initial,
      setTargetAtTime(v) { this.value = v; },
      setValueAtTime(v)  { this.value = v; },
      cancelScheduledValues() {},
    };
  }
  function createGain() {
    return {
      _kind: 'gain',
      gain: makeAudioParam(1),
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createBiquadFilter() {
    return {
      _kind: 'biquad',
      type: 'lowpass',
      frequency: makeAudioParam(440),
      Q: makeAudioParam(1),
      gain: makeAudioParam(0),
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createWaveShaper() {
    return {
      _kind: 'shaper',
      curve: null,
      oversample: 'none',
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createConvolver() {
    return {
      _kind: 'convolver',
      buffer: null,
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createDynamicsCompressor() {
    return {
      _kind: 'compressor',
      threshold: makeAudioParam(-24),
      ratio:     makeAudioParam(12),
      attack:    makeAudioParam(0.003),
      release:   makeAudioParam(0.25),
      knee:      makeAudioParam(30),
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createDelay() {
    return {
      _kind: 'delay',
      delayTime: makeAudioParam(0),
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createBuffer(channels, length, sr) {
    const data = [];
    for (let c = 0; c < channels; c++) data.push(new Float32Array(length));
    return {
      numberOfChannels: channels,
      length,
      sampleRate: sr,
      getChannelData: (i) => data[i],
    };
  }
  return {
    sampleRate, currentTime,
    createGain, createBiquadFilter, createWaveShaper,
    createConvolver, createDynamicsCompressor, createDelay, createBuffer,
  };
}

function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class { constructor() { throw new Error('worklet unavailable in Node'); } };
}
function uninstallWorklet() {
  delete globalThis.AudioWorkletNode;
}

const PARAM_DEFS = {
  amp:        { id: 'amp',        type: 'enum',   default: DEFAULT_AMP_MODEL },
  gain:       { id: 'gain',       min: 0, max: 2, default: 1 },
  bass:       { id: 'bass',       min: 0, max: 1, default: 0.5 },
  mid_low:    { id: 'mid_low',    min: 0, max: 1, default: 0.5 },
  mid_hi:     { id: 'mid_hi',     min: 0, max: 1, default: 0.5 },
  treble:     { id: 'treble',     min: 0, max: 1, default: 0.5 },
  master:     { id: 'master',     min: 0, max: 2, default: 1 },
  comp:       { id: 'comp',       min: 0, max: 1, default: 0.3 },
  geq:        { id: 'geq',        type: 'array',  default: [0, 0, 0, 0, 0] },
  blend:      { id: 'blend',      min: 0, max: 1, default: 0.75 },
  cab:        { id: 'cab',        type: 'enum',   default: DEFAULT_CAB_MODEL },
  mic:        { id: 'mic',        min: 0, max: 1, default: 0.3 },
  direct:     { id: 'direct',     min: 0, max: 1, default: 0 },
  outlevel:   { id: 'outlevel',   min: 0, max: 2, default: 1 },
};

function expectBuilderContract(result, label) {
  assert.ok(result, `${label}: builder returned a value`);
  assert.ok(result.input, `${label}: result.input present`);
  assert.ok(result.output, `${label}: result.output present`);
  assert.ok(result.paramTargets && typeof result.paramTargets === 'object',
    `${label}: paramTargets is an object`);
  const sink = { _connections: [], connect() {}, disconnect() {} };
  assert.doesNotThrow(() => result.output.connect(sink),
    `${label}: output.connect() works`);
}

// ── Tests ───────────────────────────────────────────────────────────────

test('module shape — exports + defaults', () => {
  assert.equal(typeof buildBassAmpDesigner, 'function', 'builder is exported');
  assert.ok(R13_BASS_AMP_BUILDERS.bass_amp_designer === buildBassAmpDesigner,
    'default export registers bass_amp_designer node type');
  assert.equal(GRAPHIC_EQ_FREQS.length, 5, '5-band graphic EQ');
  assert.ok(Object.keys(AMP_MODELS).length >= 6, 'at least 6 amp models');
  assert.ok('di_only' in CAB_MODELS, 'di_only cab option exists');
  assert.equal(DEFAULT_AMP_MODEL, 'classic_bass');
  assert.equal(DEFAULT_CAB_MODEL, '4x10');
});

test('buildBassAmpDesigner — fallback path returns valid contract with all params', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'bass_amp_designer',
      params: {
        amp_model:      '@amp',
        gain:           '@gain',
        bass:           '@bass',
        mid_low:        '@mid_low',
        mid_hi:         '@mid_hi',
        treble:         '@treble',
        master:         '@master',
        compression:    '@comp',
        graphic_eq:     '@geq',
        tube_blend:     '@blend',
        cab_model:      '@cab',
        mic_position:   '@mic',
        direct_out_mix: '@direct',
        output_level:   '@outlevel',
      },
    };
    const result = buildBassAmpDesigner(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'bass_amp_designer');

    // All ~14 params should map into paramTargets
    // (master and output_level both map to postGain — the second wins, so
    // expect exactly one entry for each unique paramId).
    const expectedIds = ['amp', 'gain', 'bass', 'mid_low', 'mid_hi', 'treble',
      'master', 'comp', 'geq', 'blend', 'cab', 'mic', 'direct', 'outlevel'];
    for (const id of expectedIds) {
      assert.ok(id in result.paramTargets,
        `paramTargets contains '${id}'`);
    }
    assert.ok(Object.keys(result.paramTargets).length >= 12,
      `at least 12 paramTargets (got ${Object.keys(result.paramTargets).length})`);
  } finally {
    uninstallWorklet();
  }
});

test('amp_model setter — every model name + numeric index applies without throwing', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const result = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { amp_model: '@amp' },
    }, PARAM_DEFS);
    const setAmp = result.paramTargets.amp.customSetter;
    assert.equal(typeof setAmp, 'function');
    for (const name of Object.keys(AMP_MODELS)) {
      assert.doesNotThrow(() => setAmp(name), `setAmp('${name}')`);
    }
    assert.doesNotThrow(() => setAmp(0));
    assert.doesNotThrow(() => setAmp(0.5));
    assert.doesNotThrow(() => setAmp(1));
    assert.doesNotThrow(() => setAmp('not_a_real_amp')); // graceful no-op
  } finally {
    uninstallWorklet();
  }
});

test('cab_model setter — every cab + di_only applies without throwing', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const result = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { cab_model: '@cab' },
    }, PARAM_DEFS);
    const setCab = result.paramTargets.cab.customSetter;
    assert.equal(typeof setCab, 'function');
    for (const name of Object.keys(CAB_MODELS)) {
      assert.doesNotThrow(() => setCab(name), `setCab('${name}')`);
    }
    assert.doesNotThrow(() => setCab(0.0));
    assert.doesNotThrow(() => setCab(0.5));
    assert.doesNotThrow(() => setCab(1.0));
  } finally {
    uninstallWorklet();
  }
});

test('tube_blend=0 zeroes the tube stage; tube_blend=1 zeroes the SS stage', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    // We need to walk the graph to find the tube/ss stage gains. Easier
    // path: re-instantiate at extremes by setting a static value, then
    // observe which gain value got pushed via the customSetter.
    const r = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { tube_blend: '@blend' },
    }, PARAM_DEFS);
    const setBlend = r.paramTargets.blend.customSetter;

    // Find both stage-out gains by walking input.connect path
    const allGains = [];
    const seen = new Set();
    function walk(n, depth = 0) {
      if (!n || seen.has(n) || depth > 12) return;
      seen.add(n);
      if (n._kind === 'gain') allGains.push(n);
      for (const out of (n._connections || [])) walk(out, depth + 1);
    }
    walk(r.input);

    // Pre-test: at default tube_blend (model.tubeMix=0.75 for classic_bass)
    // tube gain should sum w/ ss gain to 1.
    setBlend(0.0);
    // Find any gain whose value is now 0, and any whose value is now 1
    const zeros = allGains.filter((g) => Math.abs(g.gain.value - 0.0) < 1e-9);
    const ones  = allGains.filter((g) => Math.abs(g.gain.value - 1.0) < 1e-9);
    assert.ok(zeros.length >= 1, 'at least one stage gain at 0 when tube_blend=0');
    assert.ok(ones.length  >= 1, 'at least one stage gain at 1 when tube_blend=0');

    // Same logic flipped
    setBlend(1.0);
    const zeros2 = allGains.filter((g) => Math.abs(g.gain.value - 0.0) < 1e-9);
    const ones2  = allGains.filter((g) => Math.abs(g.gain.value - 1.0) < 1e-9);
    assert.ok(zeros2.length >= 1, 'at least one stage gain at 0 when tube_blend=1');
    assert.ok(ones2.length  >= 1, 'at least one stage gain at 1 when tube_blend=1');
  } finally {
    uninstallWorklet();
  }
});

test('compression knob — higher value lowers compressor threshold', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const r = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { compression: '@comp' },
    }, PARAM_DEFS);
    const setComp = r.paramTargets.comp.customSetter;

    // Reach the compressor by walking the graph
    function findCompressor(start) {
      const seen = new Set();
      const stack = [start];
      while (stack.length) {
        const n = stack.pop();
        if (!n || seen.has(n)) continue;
        seen.add(n);
        if (n._kind === 'compressor') return n;
        for (const out of (n._connections || [])) stack.push(out);
      }
      return null;
    }
    const comp = findCompressor(r.input);
    assert.ok(comp, 'DynamicsCompressorNode is in the graph');

    setComp(0.0);
    const thresh0 = comp.threshold.value;
    const ratio0  = comp.ratio.value;
    setComp(1.0);
    const thresh1 = comp.threshold.value;
    const ratio1  = comp.ratio.value;
    assert.ok(thresh1 < thresh0,
      `comp=1.0 threshold (${thresh1}) lower than comp=0.0 threshold (${thresh0})`);
    assert.ok(ratio1 > ratio0,
      `comp=1.0 ratio (${ratio1}) > comp=0.0 ratio (${ratio0})`);
  } finally {
    uninstallWorklet();
  }
});

test('direct_out_mix=1 routes through DI; =0 routes through mic', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const r = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { direct_out_mix: '@direct' },
    }, PARAM_DEFS);
    const setDirect = r.paramTargets.direct.customSetter;

    // Find all gains with values close to {0, 1} after the setter runs
    function collectGains(start) {
      const seen = new Set();
      const out = [];
      const stack = [start];
      while (stack.length) {
        const n = stack.pop();
        if (!n || seen.has(n)) continue;
        seen.add(n);
        if (n._kind === 'gain') out.push(n);
        for (const o of (n._connections || [])) stack.push(o);
      }
      return out;
    }
    const allGains = collectGains(r.input);

    setDirect(1.0);
    const ones = allGains.filter((g) => Math.abs(g.gain.value - 1.0) < 1e-9);
    const zeros = allGains.filter((g) => Math.abs(g.gain.value - 0.0) < 1e-9);
    assert.ok(ones.length  >= 1, 'mix=1 → some gain pinned to 1 (DI)');
    assert.ok(zeros.length >= 1, 'mix=1 → some gain pinned to 0 (mic)');

    setDirect(0.0);
    const ones2 = allGains.filter((g) => Math.abs(g.gain.value - 1.0) < 1e-9);
    const zeros2 = allGains.filter((g) => Math.abs(g.gain.value - 0.0) < 1e-9);
    assert.ok(ones2.length  >= 1, 'mix=0 → some gain pinned to 1 (mic)');
    assert.ok(zeros2.length >= 1, 'mix=0 → some gain pinned to 0 (DI)');

    // 50/50: both gains at 0.5
    setDirect(0.5);
    const halves = allGains.filter((g) => Math.abs(g.gain.value - 0.5) < 1e-9);
    assert.ok(halves.length >= 2, 'mix=0.5 → mic and DI both at 0.5');
  } finally {
    uninstallWorklet();
  }
});

test('cab_model=di_only clears the convolver buffer', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const r = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { cab_model: '@cab' },
    }, PARAM_DEFS);

    // Walk graph for the convolver
    function findConvolver(start) {
      const seen = new Set();
      const stack = [start];
      while (stack.length) {
        const n = stack.pop();
        if (!n || seen.has(n)) continue;
        seen.add(n);
        if (n._kind === 'convolver') return n;
        for (const o of (n._connections || [])) stack.push(o);
      }
      return null;
    }
    const cab = findConvolver(r.input);
    assert.ok(cab, 'convolver in graph for default cab_model');
    // Default cab_model should have built an IR buffer (cabinet-ir.js
    // returns a non-null AudioBuffer-like)
    assert.ok(cab.buffer !== null, 'cab buffer initially loaded');

    const setCab = r.paramTargets.cab.customSetter;
    setCab('di_only');
    assert.equal(cab.buffer, null, 'cab buffer cleared after di_only');

    setCab('8x10');
    assert.ok(cab.buffer !== null, 'cab buffer rebuilt for 8x10');
  } finally {
    uninstallWorklet();
  }
});

test('graphic_eq array setter — applies all 5 band gains', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const r = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: { graphic_eq: '@geq' },
    }, PARAM_DEFS);
    const setGeq = r.paramTargets.geq.customSetter;
    assert.equal(typeof setGeq, 'function');
    assert.doesNotThrow(() => setGeq([3, -2, 4, 0, -5]));
    // Pass non-arrays — must not throw, just no-op
    assert.doesNotThrow(() => setGeq(null));
    assert.doesNotThrow(() => setGeq(0.5));
  } finally {
    uninstallWorklet();
  }
});

test('static (non-modulated) params apply without paramTargets entries', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const r = buildBassAmpDesigner(ctx, {
      type: 'bass_amp_designer',
      params: {
        amp_model:    'svt_classic',
        cab_model:    '8x10',
        gain:         1.5,
        compression:  0.7,
        graphic_eq:   [2, 4, 0, -3, 1],
        tube_blend:   0.6,
        direct_out_mix: 0.4,
      },
    }, PARAM_DEFS);
    expectBuilderContract(r, 'static-params');
    // No '@' refs → no paramTargets
    assert.equal(Object.keys(r.paramTargets).length, 0,
      'no paramTargets when no params are @-modulated');
  } finally {
    uninstallWorklet();
  }
});
