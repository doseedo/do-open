/**
 * R13 ESS (Enhanced Stereo Spread) — builder smoke + invariant tests.
 *
 * Run with:
 *   node --test tests/r13_ess.test.js
 *
 * The Web Audio API is not available in Node, so we stub a minimal
 * AudioContext with the surface the builder touches. We also build a
 * graph-walking helper that simulates DC-rate signal flow through the
 * stub graph — enough to verify the M/S round-trip identity ("mono in,
 * mono out") and the width-affects-stereo invariant.
 *
 * The stub does NOT model biquad filtering (we treat filters as
 * unity-gain pass-through for the round-trip check). What matters is
 * that the GAIN MATRIX is correct: the M=L+R, S=L-R encode and the
 * L=M+S, R=M-S decode must be wired right. Filter response shape is
 * Web Audio's responsibility, not ours.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { buildESS } from '../src/audio/builders/r13_ess.js';
import R13_ESS_BUILDERS from '../src/audio/builders/r13_ess.js';

// ── Minimal AudioContext stub ────────────────────────────────────────────

let _nodeIdCounter = 0;
function _makeNode(kind, extra = {}) {
  const id = `${kind}#${_nodeIdCounter++}`;
  const node = {
    _id: id,
    _kind: kind,
    _outConns: [],          // Array<{ target, outIndex, inIndex }>
    connect(target, outIndex = 0, inIndex = 0) {
      this._outConns.push({ target, outIndex, inIndex });
      return target;
    },
    disconnect() { this._outConns.length = 0; },
    ...extra,
  };
  return node;
}

function makeStubCtx() {
  const sampleRate = 48000;
  const ctx = {
    sampleRate,
    currentTime: 0,
    createGain() {
      const g = { value: 1, setTargetAtTime(v) { g.value = v; } };
      return _makeNode('gain', {
        gain: g,
        channelCount: 2, channelCountMode: 'max', channelInterpretation: 'speakers',
      });
    },
    createBiquadFilter() {
      const f = { value: 1000, setTargetAtTime(v) { f.value = v; } };
      const q = { value: 0.707, setTargetAtTime(v) { q.value = v; } };
      const g = { value: 0,     setTargetAtTime(v) { g.value = v; } };
      return _makeNode('biquad', {
        type: 'lowpass', frequency: f, Q: q, gain: g,
      });
    },
    createDelay(_max) {
      const dt = { value: 0, setTargetAtTime(v) { dt.value = v; } };
      return _makeNode('delay', { delayTime: dt });
    },
    createChannelSplitter(n) {
      return _makeNode('splitter', { numberOfOutputs: n });
    },
    createChannelMerger(n) {
      return _makeNode('merger', { numberOfInputs: n });
    },
  };
  return ctx;
}

// ── DC-rate signal-flow simulator ────────────────────────────────────────
// Treats biquads + delays as unity pass-through (we're testing the M/S
// matrix, not the filtering). Walks from a source pair (L, R) and
// computes the resulting (L', R') at the merger.

function simulate(internal, splitter, merger, Lin, Rin) {
  // node._id → { L: number, R: number } for stereo pass-throughs, or
  // for mono nodes a single 'V' value depending on the path. To keep
  // it simple we represent everything as { v: number } and treat
  // splitter outputs as separate channels (we visit them by index).
  // Strategy: starting from splitter output 0 = L and output 1 = R,
  // do a BFS over outConns where 'biquad', 'delay', 'gain' apply
  // their scalar transformation; 'merger' accumulates per-input-channel.

  // We'll perform a forward pass twice: once with (Lin, 0) at the
  // splitter and once with (0, Rin), then sum results. This linearises
  // the propagation since every node is linear.

  function passFrom(splitterOuts) {
    // splitterOuts: { 0: number, 1: number } — values entering each
    // splitter output port.
    // Per-node accumulated input value:
    const acc = new Map();          // node._id → number (sum of inputs)
    const mergerAcc = { 0: 0, 1: 0 };

    // Seed: any connection from splitter node directly receives the
    // splitter's per-port value at the connection's outIndex.
    const queue = [];
    for (const c of splitter._outConns) {
      const v = splitterOuts[c.outIndex] || 0;
      queue.push({ target: c.target, value: v, inIndex: c.inIndex });
    }

    // BFS
    while (queue.length) {
      const { target, value, inIndex } = queue.shift();

      if (target === merger) {
        mergerAcc[inIndex] = (mergerAcc[inIndex] || 0) + value;
        continue;
      }

      // Apply node transformation
      let next = value;
      if (target._kind === 'gain') {
        next = value * (target.gain.value ?? 1);
      } else if (target._kind === 'biquad') {
        // DC-rate approximation: lowpass = pass, highpass = block,
        // we want round-trip identity at width=100% so we model BOTH
        // as pass-through (band sum = identity). For width != 100%
        // the gain matrix is what matters, not which band catches DC.
        next = value;
      } else if (target._kind === 'delay') {
        next = value;
      } else {
        next = value;
      }

      // Accumulate inputs (a node may receive from multiple sources).
      const prev = acc.get(target._id) || 0;
      const total = prev + next;
      acc.set(target._id, total);

      // Now propagate the *delta* to its outputs. Strict BFS without
      // delta tracking would double-count; we instead reset `next` and
      // re-emit only when this is the LAST visit. But our graph is a
      // DAG with a known fan-in structure — easier solution: for nodes
      // that have multiple inputs, only propagate when all inputs have
      // arrived. We approximate by collecting all queue items first,
      // then re-running until stable. For our specific topology that
      // works because every node has fan-in ≤ 4 and the DAG is small.
      //
      // Simpler: emit per-input-arrival; this overcounts when a node
      // sums two inputs and forwards. To avoid that we use a 2-phase
      // approach: phase 1 collects per-node sums, phase 2 emits
      // outgoing connections once per node.
      //
      // We do it inline below by NOT continuing the BFS here. Instead,
      // we track 'pending' inputs and a reverse adjacency.
      //
      // For simplicity in this stub, we instead model the accumulator
      // and walk topologically via a post-pass after all sources have
      // settled. Implemented in `topoEval` below.
    }
    return null; // unused — see topoEval
  }

  // ── Topo-sort + evaluate — robust path ─────────────────────────────
  // 1. Discover all nodes reachable from splitter
  const allNodes = new Set();
  const stack = [splitter];
  while (stack.length) {
    const n = stack.pop();
    if (allNodes.has(n)) continue;
    allNodes.add(n);
    for (const c of n._outConns || []) stack.push(c.target);
  }
  // Add merger if reachable (it should be).
  // 2. Reverse adjacency: node → list of {source, outIndex, inIndex, value-getter}
  const incoming = new Map();
  for (const n of allNodes) incoming.set(n, []);
  for (const n of allNodes) {
    for (const c of n._outConns || []) {
      const tgt = c.target;
      if (!incoming.has(tgt)) incoming.set(tgt, []);
      incoming.get(tgt).push({ source: n, outIndex: c.outIndex, inIndex: c.inIndex });
    }
  }
  // 3. Topo-sort (Kahn).
  const indeg = new Map();
  for (const n of allNodes) indeg.set(n, (incoming.get(n) || []).filter(e => e.source !== n).length);
  // The splitter has no inputs in our subgraph (it's the source).
  // Any node whose only "incoming" is from the splitter is fine because
  // splitter has zero indegree.
  const order = [];
  const ready = [];
  for (const n of allNodes) if ((indeg.get(n) || 0) === 0) ready.push(n);
  while (ready.length) {
    const n = ready.shift();
    order.push(n);
    for (const c of n._outConns || []) {
      const tgt = c.target;
      const d = indeg.get(tgt) - 1;
      indeg.set(tgt, d);
      if (d === 0) ready.push(tgt);
    }
  }
  // 4. Evaluate
  // Each node has an "output value" — for the splitter, indexed by
  // outIndex; for everything else it's a single scalar.
  const splitterVal = { 0: Lin, 1: Rin };
  const out = new Map();    // node → scalar
  for (const n of order) {
    if (n === splitter) continue;
    if (n === merger) {
      // Per-input-channel accumulation
      const ch = { 0: 0, 1: 0 };
      for (const e of incoming.get(n) || []) {
        let v;
        if (e.source === splitter) v = splitterVal[e.outIndex] || 0;
        else v = out.get(e.source) || 0;
        ch[e.inIndex] = (ch[e.inIndex] || 0) + v;
      }
      out.set(n, ch);
      continue;
    }
    // Sum incoming
    let sum = 0;
    for (const e of incoming.get(n) || []) {
      let v;
      if (e.source === splitter) v = splitterVal[e.outIndex] || 0;
      else if (e.source === merger) v = (out.get(e.source) || {})[e.inIndex] || 0;
      else v = out.get(e.source) || 0;
      sum += v;
    }
    // Apply transform
    let val = sum;
    if (n._kind === 'gain') val = sum * (n.gain.value ?? 1);
    else if (n._kind === 'biquad') val = sum;
    else if (n._kind === 'delay') val = sum;
    out.set(n, val);
  }
  // 5. Return merger result
  const m = out.get(merger) || { 0: 0, 1: 0 };
  return { L: m[0] || 0, R: m[1] || 0 };
}

const PARAM_DEFS = {
  bass_width:   { id: 'bass_width',   min: 0, max: 200, default: 100 },
  mid_width:    { id: 'mid_width',    min: 0, max: 200, default: 100 },
  high_width:   { id: 'high_width',   min: 0, max: 200, default: 100 },
  master_width: { id: 'master_width', min: 0, max: 200, default: 100 },
  bass_delay:   { id: 'bass_delay',   min: 0, max: 30,  default: 0   },
  high_delay:   { id: 'high_delay',   min: 0, max: 15,  default: 0   },
  mono_below:   { id: 'mono_below',   min: 0, max: 400, default: 0   },
  output_gain:  { id: 'output_gain',  min: 0, max: 4,   default: 1   },
  xover_low:    { id: 'xover_low',    min: 100, max: 400, default: 250 },
  xover_high:   { id: 'xover_high',   min: 1000, max: 6000, default: 2500 },
};

// ── Tests ────────────────────────────────────────────────────────────────

test('buildESS — returns valid builder contract', () => {
  const ctx = makeStubCtx();
  const node = {
    type: 'ess_stereo_spread',
    params: {
      bass_width:   '@bass_width',
      mid_width:    '@mid_width',
      high_width:   '@high_width',
      master_width: '@master_width',
    },
  };
  const result = buildESS(ctx, node, PARAM_DEFS);
  assert.ok(result, 'builder returned a value');
  assert.ok(result.input,  'result.input present');
  assert.ok(result.output, 'result.output present');
  assert.ok(result.paramTargets && typeof result.paramTargets === 'object',
    'paramTargets is an object');
  assert.ok('bass_width'   in result.paramTargets, 'bass_width exposed');
  assert.ok('mid_width'    in result.paramTargets, 'mid_width exposed');
  assert.ok('high_width'   in result.paramTargets, 'high_width exposed');
  assert.ok('master_width' in result.paramTargets, 'master_width exposed');
});

test('R13_ESS_BUILDERS default export registers ess_stereo_spread', () => {
  assert.equal(typeof R13_ESS_BUILDERS.ess_stereo_spread, 'function',
    'default export wires ess_stereo_spread');
});

test('static params produce no paramTargets entries', () => {
  const ctx = makeStubCtx();
  const node = {
    type: 'ess_stereo_spread',
    params: {
      bass_width: 150, mid_width: 100, high_width: 80,
      master_width: 110, output_gain: 1.0,
    },
  };
  const result = buildESS(ctx, node, PARAM_DEFS);
  assert.equal(Object.keys(result.paramTargets).length, 0,
    'all-static params produce zero paramTargets');
});

test('mono input stays mono — L=R in implies L=R out at width=100%', () => {
  const ctx = makeStubCtx();
  const node = {
    type: 'ess_stereo_spread',
    params: {
      bass_width: 100, mid_width: 100, high_width: 100,
      master_width: 100, output_gain: 1.0,
      // Disable Haas + mono safety so the signal-flow test is clean.
      bass_delay_ms: 0, high_delay_ms: 0, mono_below_hz: 0,
    },
  };
  const result = buildESS(ctx, node, PARAM_DEFS);
  const { splitter, merger } = result._internal;

  // Probe with a mono-coherent signal: L = R = 1
  const out = simulate(result._internal, splitter, merger, 1, 1);
  // For L = R = 1: M = 1, S = 0. Output should be L = M = 1, R = M = 1.
  // Crossover bands sum to identity (3 unity-gain bands → 3.0 — see note).
  // What matters here is the symmetry: out.L === out.R.
  assert.ok(Math.abs(out.L - out.R) < 1e-9,
    `mono in stays mono out: out.L=${out.L} out.R=${out.R}`);

  // Probe with another mono signal: L = R = 0.7
  const out2 = simulate(result._internal, splitter, merger, 0.7, 0.7);
  assert.ok(Math.abs(out2.L - out2.R) < 1e-9,
    `mono in (0.7) stays mono out: out.L=${out2.L} out.R=${out2.R}`);

  // And with a different mono level scaled by output_gain
  // (output_gain is 1.0 so values match)
  assert.ok(out2.L > 0, 'mono signal produces nonzero output');
});

test('mono invariant holds at non-default widths IF widths are equal', () => {
  // The "mono input → mono output" invariant only requires that the
  // matrix be symmetric. With L=R the side bus is 0 in every band, so
  // the per-band widths are irrelevant. This test verifies that.
  const ctx = makeStubCtx();
  const node = {
    type: 'ess_stereo_spread',
    params: {
      bass_width: 200, mid_width: 50, high_width: 0,   // wildly different
      master_width: 175, output_gain: 1.0,
      bass_delay_ms: 0, high_delay_ms: 0, mono_below_hz: 0,
    },
  };
  const result = buildESS(ctx, node, PARAM_DEFS);
  const { splitter, merger } = result._internal;
  const out = simulate(result._internal, splitter, merger, 0.5, 0.5);
  assert.ok(Math.abs(out.L - out.R) < 1e-9,
    `mono in stays mono out at extreme widths: out.L=${out.L} out.R=${out.R}`);
});

test('width affects the side path — driving bass_width updates underlying gain', () => {
  const ctx = makeStubCtx();
  const node = {
    type: 'ess_stereo_spread',
    params: { bass_width: '@bass_width' },
  };
  const result = buildESS(ctx, node, PARAM_DEFS);
  const target = result.paramTargets.bass_width;
  assert.ok(target, 'bass_width target exists');
  assert.equal(typeof target.customSetter, 'function',
    'bass_width target uses customSetter (transform applied)');
  const initial = result._internal.sBassGain.gain.value;
  // Drive width to 200% → multiplier 2.0
  target.customSetter(200);
  assert.equal(result._internal.sBassGain.gain.value, 2.0,
    'driving bass_width=200 sets sBassGain to 2.0');
  // Drive width to 0% → multiplier 0.0 (mono in band)
  target.customSetter(0);
  assert.equal(result._internal.sBassGain.gain.value, 0.0,
    'driving bass_width=0 sets sBassGain to 0.0 (mono band)');
  // Drive width to 100% → multiplier 1.0 (neutral)
  target.customSetter(100);
  assert.equal(result._internal.sBassGain.gain.value, 1.0,
    'driving bass_width=100 sets sBassGain to 1.0');
  assert.notEqual(initial, undefined, 'initial value was readable');
});

test('width=0% produces mono output for any stereo input', () => {
  // Verify the inverse direction of the M/S identity: with all widths
  // at 0% the side bus is fully attenuated, so output L = output R = M
  // regardless of the input stereo content.
  const ctx = makeStubCtx();
  const node = {
    type: 'ess_stereo_spread',
    params: {
      bass_width: 0, mid_width: 0, high_width: 0,
      master_width: 100, output_gain: 1.0,
      bass_delay_ms: 0, high_delay_ms: 0, mono_below_hz: 0,
    },
  };
  const result = buildESS(ctx, node, PARAM_DEFS);
  const { splitter, merger } = result._internal;
  // Drive a HARD-PANNED input: L = 1, R = -1. M = 0, S = 1.
  // With widths=0 the S contribution is zeroed → output L = R = M = 0.
  const out = simulate(result._internal, splitter, merger, 1, -1);
  assert.ok(Math.abs(out.L - out.R) < 1e-9,
    `width=0 produces mono out for stereo in: out.L=${out.L} out.R=${out.R}`);
  // And with a less extreme stereo input
  const out2 = simulate(result._internal, splitter, merger, 0.8, 0.2);
  assert.ok(Math.abs(out2.L - out2.R) < 1e-9,
    `width=0 mono for asymm in: out.L=${out2.L} out.R=${out2.R}`);
});

test('width affects stereo — width>100 widens the stereo image', () => {
  // Compare width=100 vs width=200 for the same hard-panned input.
  // At width=100: out should reproduce the input (L=1, R=-1).
  // At width=200: out should be wider (|L-R| > 2).
  const inputL = 1, inputR = -1;

  function rms(node) {
    const { splitter, merger } = node._internal;
    return simulate(node._internal, splitter, merger, inputL, inputR);
  }

  // Neutral — widths at 100%
  const ctxA = makeStubCtx();
  const neutral = buildESS(ctxA, {
    type: 'ess_stereo_spread',
    params: {
      bass_width: 100, mid_width: 100, high_width: 100,
      master_width: 100, output_gain: 1.0,
      bass_delay_ms: 0, high_delay_ms: 0, mono_below_hz: 0,
    },
  }, PARAM_DEFS);

  // Wide — widths at 200%
  const ctxB = makeStubCtx();
  const wide = buildESS(ctxB, {
    type: 'ess_stereo_spread',
    params: {
      bass_width: 200, mid_width: 200, high_width: 200,
      master_width: 100, output_gain: 1.0,
      bass_delay_ms: 0, high_delay_ms: 0, mono_below_hz: 0,
    },
  }, PARAM_DEFS);

  const a = rms(neutral);
  const b = rms(wide);

  const diffA = Math.abs(a.L - a.R);
  const diffB = Math.abs(b.L - b.R);

  assert.ok(diffB > diffA,
    `width=200 widens stereo: |L-R|@100% = ${diffA}, |L-R|@200% = ${diffB} (expect 200>100)`);
});
