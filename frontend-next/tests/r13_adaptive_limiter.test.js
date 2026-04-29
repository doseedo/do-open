/**
 * R13 Adaptive Limiter — builder smoke + ceiling-enforcement tests.
 *
 * Run with:
 *   node --test tests/r13_adaptive_limiter.test.js
 *
 * The Web Audio API isn't available in Node, so we stub a minimal
 * AudioContext + DynamicsCompressorNode and force the worklet path to fail.
 * The builder must fall back to the DynamicsCompressorNode brickwall config
 * and still return a `{ input, output, paramTargets }` shape compatible with
 * the WebAudioDSPEngine builder contract.
 *
 * Ceiling-enforcement test re-implements the worklet's gain-follower math
 * in plain JS (the worklet itself can't run in Node) and asserts that
 * feeding a -1 dBFS sine through it with ceiling at -6 dBFS yields a
 * peak ≤ -6 dBFS at the output.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  buildAdaptiveLimiter,
} from '../src/audio/builders/r13_adaptive_limiter.js';
import R13_ADAPTIVE_LIMITER_BUILDERS from '../src/audio/builders/r13_adaptive_limiter.js';

// ── Minimal AudioContext stub ────────────────────────────────────────────
function makeStubCtx() {
  const sampleRate = 48000;
  function createGain() {
    return {
      gain: { value: 1 },
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createDynamicsCompressor() {
    return {
      threshold: { value: -24 },
      knee:      { value: 30 },
      ratio:     { value: 12 },
      attack:    { value: 0.003 },
      release:   { value: 0.250 },
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  return { sampleRate, createGain, createDynamicsCompressor };
}

// Force the worklet path to fail so we exercise the fallback branch.
function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class { constructor() { throw new Error('not available in Node'); } };
}
function uninstallWorklet() {
  delete globalThis.AudioWorkletNode;
}

const PARAM_DEFS = {
  gain:         { id: 'gain',         min: 0,    max: 24,   default: 0    },
  ceiling:      { id: 'ceiling',      min: -30,  max: 0,    default: -0.3 },
  lookahead:    { id: 'lookahead',    min: 1,    max: 12,   default: 5    },
  release_min:  { id: 'release_min',  min: 1,    max: 50,   default: 5    },
  release_max:  { id: 'release_max',  min: 100,  max: 2000, default: 500  },
  adaptation:   { id: 'adaptation',   min: 0,    max: 1,    default: 0.7  },
  soft_clip:    { id: 'soft_clip',    min: 0,    max: 1,    default: 0.3  },
};

function expectBuilderContract(result, label) {
  assert.ok(result, `${label}: builder returned a value`);
  assert.ok(result.input, `${label}: result.input present`);
  assert.ok(result.output, `${label}: result.output present`);
  assert.ok(result.paramTargets && typeof result.paramTargets === 'object',
    `${label}: paramTargets is an object`);
  const sink = { connect() {}, disconnect() {} };
  assert.doesNotThrow(() => result.output.connect(sink),
    `${label}: output.connect() works`);
}

// ── Smoke tests ─────────────────────────────────────────────────────────

test('buildAdaptiveLimiter — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'adaptive_limiter',
      params: {
        gain:               '@gain',
        out_ceiling:        '@ceiling',
        lookahead_ms:       '@lookahead',
        release_min_ms:     '@release_min',
        release_max_ms:     '@release_max',
        release_adaptation: '@adaptation',
        true_peak:          1,
        soft_clip_amount:   '@soft_clip',
        link_lr:            1,
      },
    };
    const result = buildAdaptiveLimiter(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'adaptive_limiter');

    // Modulated params should all be present in paramTargets
    for (const id of ['gain', 'ceiling', 'lookahead', 'release_min', 'release_max', 'adaptation', 'soft_clip']) {
      assert.ok(id in result.paramTargets, `paramTargets includes ${id}`);
    }

    // The fallback path drives gain via customSetter (preGain), and
    // ceiling via customSetter (compressor threshold).
    assert.equal(typeof result.paramTargets.gain.customSetter, 'function',
      'gain has customSetter in fallback');
    assert.equal(typeof result.paramTargets.ceiling.customSetter, 'function',
      'ceiling has customSetter in fallback');

    // Drive the setters — must not throw.
    assert.doesNotThrow(() => result.paramTargets.gain.customSetter(6),
      'gain setter executes');
    assert.doesNotThrow(() => result.paramTargets.ceiling.customSetter(-6),
      'ceiling setter executes');
    assert.doesNotThrow(() => result.paramTargets.lookahead.customSetter(8),
      'lookahead setter executes');
    assert.doesNotThrow(() => result.paramTargets.release_max.customSetter(800),
      'release_max setter executes');
    // Params with no fallback target must still call the setter without error
    assert.doesNotThrow(() => result.paramTargets.adaptation.customSetter(0.5),
      'adaptation setter is a no-op on fallback');
    assert.doesNotThrow(() => result.paramTargets.soft_clip.customSetter(0.5),
      'soft_clip setter is a no-op on fallback');
  } finally {
    uninstallWorklet();
  }
});

test('buildAdaptiveLimiter — static params produce zero paramTargets', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'adaptive_limiter',
      params: {
        gain: 3, out_ceiling: -1, lookahead_ms: 5,
        release_min_ms: 5, release_max_ms: 400,
        release_adaptation: 0.7, true_peak: 1,
        soft_clip_amount: 0.3, link_lr: 1,
      },
    };
    const result = buildAdaptiveLimiter(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'adaptive_limiter (static)');
    assert.equal(Object.keys(result.paramTargets).length, 0,
      'static params produce no paramTargets');
  } finally {
    uninstallWorklet();
  }
});

test('R13_ADAPTIVE_LIMITER_BUILDERS default export registers `adaptive_limiter`', () => {
  assert.equal(typeof R13_ADAPTIVE_LIMITER_BUILDERS.adaptive_limiter, 'function');
});

// ── Worklet algorithm regression — ceiling-enforcement ────────────────────
//
// Re-implements the gain-follower core from r13-adaptive-limiter-processor.js
// in plain JS so we can verify the brickwall contract end-to-end without
// AudioWorklet support. We feed a -1 dBFS 440 Hz sine, ceiling -6 dBFS,
// and assert the output peak ≤ -6 dBFS.

function dbToLin(db) { return Math.pow(10, db / 20); }
function linToDb(lin) { return 20 * Math.log10(Math.max(1e-12, lin)); }

function tau(timeSec, sr) {
  if (!(timeSec > 0)) return 0;
  return Math.exp(-1 / (Math.max(1e-6, timeSec) * sr));
}

function runWorkletAlgo(input, sr, opts) {
  const {
    gainDb        = 0,
    ceilDb        = -6,
    lookMs        = 5,
    relMinMs      = 5,
    relMaxMs      = 500,
    relAdapt      = 0.7,
    softClipAmt   = 0.3,
    truePeakOn    = false,   // disable for reference-test simplicity
    windowSec     = 0.150,
    maxEventsPerSec = 200,
  } = opts || {};

  const N = input.length;
  const out = new Float32Array(N);
  const preGain = dbToLin(gainDb);
  const ceilLin = dbToLin(ceilDb);
  const lookSamples = Math.max(1, Math.floor((lookMs / 1000) * sr));
  const ringSize = Math.max(lookSamples + 4, 64);
  const ring = new Float32Array(ringSize);
  let rW = 0;

  const attackSec = (lookMs / 1000) / 3;
  const attackA = tau(attackSec, sr);

  // Adaptive-release event window
  const windowSamples = Math.max(1, Math.floor(windowSec * sr));
  const eventRing = new Uint8Array(windowSamples);
  let eventW = 0;
  let eventCount = 0;
  const maxEventsInWindow = Math.max(1, Math.floor(maxEventsPerSec * windowSec));

  let env = 1.0;

  for (let i = 0; i < N; i++) {
    const x = input[i] * preGain;
    const peak = Math.abs(x);

    // Adaptive release
    const event = peak > ceilLin ? 1 : 0;
    const oldEvent = eventRing[eventW];
    if (event !== oldEvent) eventCount += (event - oldEvent);
    eventRing[eventW] = event;
    eventW = (eventW + 1) % windowSamples;

    const density = Math.min(1, eventCount / maxEventsInWindow);
    const relAdaptSec = (relMinMs + (relMaxMs - relMinMs) * (1 - density)) / 1000;
    const fixedSec = relMaxMs / 1000;
    const releaseSec = (1 - relAdapt) * fixedSec + relAdapt * relAdaptSec;
    const releaseA = tau(releaseSec, sr);

    // Push raw to lookahead
    ring[rW] = x;
    rW = (rW + 1) % ringSize;

    // Gain target
    const target = peak > ceilLin ? (ceilLin / peak) : 1.0;
    const a = (target < env) ? attackA : releaseA;
    env = a * env + (1 - a) * target;

    // Read delayed sample
    const ri = ((rW - lookSamples) % ringSize + ringSize) % ringSize;
    let y = ring[ri] * env;

    // Soft-clip + brickwall
    if (softClipAmt > 0) {
      const sc = y / ceilLin;
      const clamped = Math.max(-1.5, Math.min(1.5, sc));
      const s = clamped - (clamped * clamped * clamped) / 3;
      y = ((1 - softClipAmt) * sc + softClipAmt * s) * ceilLin;
    }
    if (y >  ceilLin) y =  ceilLin;
    if (y < -ceilLin) y = -ceilLin;
    out[i] = y;
  }
  return out;
}

test('worklet algorithm — -1 dBFS sine, ceiling -6 dBFS → output peak ≤ -6 dBFS', () => {
  const sr = 48000;
  // 200 ms of -1 dBFS 440 Hz sine. The first 7-8 ms is the limiter still
  // converging to its attack envelope, so we ignore that prefix when taking
  // the peak. (The brickwall still hard-clamps every sample, so even the
  // prefix should be ≤ ceiling — included as a strict check below.)
  const N = Math.floor(sr * 0.200);
  const input = new Float32Array(N);
  const amp = dbToLin(-1);  // -1 dBFS sine peak
  for (let i = 0; i < N; i++) {
    input[i] = amp * Math.sin(2 * Math.PI * 440 * (i / sr));
  }

  const ceilDb = -6;
  const out = runWorkletAlgo(input, sr, {
    gainDb: 0,
    ceilDb,
    lookMs: 5,
    relMinMs: 5,
    relMaxMs: 200,
    relAdapt: 0.7,
    softClipAmt: 0.3,
    truePeakOn: false,
  });

  // Strict brickwall: every sample must be within ±ceiling, by virtue of
  // the final hard-clamp. Tolerance = 1e-6 for FP rounding.
  const ceilLin = dbToLin(ceilDb);
  let strictMax = 0;
  for (let i = 0; i < N; i++) {
    const a = Math.abs(out[i]);
    if (a > strictMax) strictMax = a;
  }
  const strictPeakDb = linToDb(strictMax);
  assert.ok(strictPeakDb <= ceilDb + 1e-3,
    `strict peak ${strictPeakDb.toFixed(3)} dBFS exceeds ceiling ${ceilDb} dBFS`);

  // Also check the steady-state peak (after the limiter has settled).
  const settleStart = Math.floor(sr * 0.030);  // skip first 30 ms
  let steadyMax = 0;
  for (let i = settleStart; i < N; i++) {
    const a = Math.abs(out[i]);
    if (a > steadyMax) steadyMax = a;
  }
  const steadyPeakDb = linToDb(steadyMax);
  assert.ok(steadyPeakDb <= ceilDb + 1e-3,
    `steady-state peak ${steadyPeakDb.toFixed(3)} dBFS exceeds ceiling ${ceilDb} dBFS`);

  // Sanity: the limiter actually engaged — output peak is meaningfully
  // above silence (i.e. we're not just outputting zeros).
  assert.ok(steadyMax > dbToLin(-12),
    `output is too quiet (peak ${steadyPeakDb.toFixed(2)} dBFS) — limiter likely killed signal`);
});

test('worklet algorithm — sparse vs. dense input drives release adaptation', () => {
  // Build two test inputs: a sparse one (1-sample spike every 50 ms — at most
  // 3 events in any 150 ms window) and a dense one (constant -1 dBFS, so
  // every sample over the half-cycle hits the ceiling). The density tracker
  // should saturate at 1.0 for dense and stay near zero for sparse.
  const sr = 48000;
  const ceilDb = -6;
  const ceilLin = dbToLin(ceilDb);
  const amp = dbToLin(-1);

  // 500 ms test length (gives several full window cycles)
  const N  = Math.floor(sr * 0.500);

  // Dense: constant sine well over the ceiling
  const dense = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    dense[i] = amp * Math.sin(2 * Math.PI * 440 * (i / sr));
  }

  // Sparse: a single 1-sample over-ceiling spike every 50 ms
  const sparse = new Float32Array(N);
  const spikePeriod = Math.floor(sr * 0.050);
  for (let i = 0; i < N; i++) {
    if (i % spikePeriod === 0) sparse[i] = amp;
  }

  // Re-implement just the density tracker (matching the worklet exactly).
  function measureDensity(input) {
    const windowSec = 0.150;
    const maxEventsPerSec = 200;
    const windowSamples = Math.floor(windowSec * sr);
    const maxEventsInWindow = Math.max(1, Math.floor(maxEventsPerSec * windowSec));
    const ring = new Uint8Array(windowSamples);
    let w = 0, count = 0;
    let lastDensity = 0;
    for (let i = 0; i < input.length; i++) {
      const event = Math.abs(input[i]) > ceilLin ? 1 : 0;
      const old = ring[w];
      if (event !== old) count += (event - old);
      ring[w] = event;
      w = (w + 1) % windowSamples;
      lastDensity = Math.min(1, count / maxEventsInWindow);
    }
    return lastDensity;
  }

  const denseDensity  = measureDensity(dense);
  const sparseDensity = measureDensity(sparse);

  // Dense saturates the density.
  assert.ok(denseDensity >= 0.99,
    `dense density should be ≥ 0.99, got ${denseDensity.toFixed(3)}`);
  // Sparse: at most 3 events / 30-event cap → density ≤ 0.1.
  assert.ok(sparseDensity <= 0.15,
    `sparse density should be ≤ 0.15, got ${sparseDensity.toFixed(3)}`);

  // And the corresponding release time mapping: dense → near release_min,
  // sparse → near release_max.
  const relMinMs = 5;
  const relMaxMs = 500;
  const denseRel  = relMinMs + (relMaxMs - relMinMs) * (1 - denseDensity);
  const sparseRel = relMinMs + (relMaxMs - relMinMs) * (1 - sparseDensity);
  assert.ok(denseRel < 20,
    `dense release should be < 20 ms, got ${denseRel.toFixed(2)} ms`);
  assert.ok(sparseRel > 400,
    `sparse release should be > 400 ms, got ${sparseRel.toFixed(2)} ms`);
});
