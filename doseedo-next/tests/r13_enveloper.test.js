/**
 * R13 Enveloper (transient shaper) — builder smoke + transient-detection.
 *
 * Run with:
 *   node --test tests/r13_enveloper.test.js
 *
 * The Web Audio API is not available in Node, so we stub a minimal
 * AudioContext and force the worklet path to fail. The builder must
 * cleanly fall back to a DynamicsCompressor + makeup gain pair and still
 * return a `{ input, output, paramTargets }` shape compatible with the
 * WebAudioDSPEngine builder contract.
 *
 * The transient-detection test re-implements the worklet's follower math
 * in plain JS (no AudioWorkletNode dependency) and verifies that a
 * synthetic drum hit followed by silence produces a visible attack
 * envelope spike — i.e. the algorithm actually detects transients.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { buildEnveloper } from '../src/audio/builders/r13_enveloper.js';
import R13_ENVELOPER_BUILDERS from '../src/audio/builders/r13_enveloper.js';

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
      knee:      { value: 6   },
      ratio:     { value: 2   },
      attack:    { value: 0.003 },
      release:   { value: 0.25  },
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  return { sampleRate, createGain, createDynamicsCompressor };
}

function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class { constructor() { throw new Error('not available in Node'); } };
}
function uninstallWorklet() { delete globalThis.AudioWorkletNode; }

const PARAM_DEFS = {
  p_attack:  { id: 'p_attack',  min: -100, max: 100, default: 0 },
  p_sustain: { id: 'p_sustain', min: -100, max: 100, default: 0 },
  p_makeup:  { id: 'p_makeup',  min: -12,  max: 12,  default: 0 },
  p_mix:     { id: 'p_mix',     min: 0,    max: 1,   default: 1 },
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

// ── Builder smoke ─────────────────────────────────────────────────────────
test('buildEnveloper — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'enveloper',
      params: {
        attack:      '@p_attack',
        sustain:     '@p_sustain',
        output_gain: '@p_makeup',
        mix:         '@p_mix',
        attack_time_ms: 1.0,  // static
      },
    };
    const result = buildEnveloper(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'enveloper');

    assert.ok('p_attack'  in result.paramTargets, 'attack target exposed');
    assert.ok('p_sustain' in result.paramTargets, 'sustain target exposed');
    assert.ok('p_makeup'  in result.paramTargets, 'output_gain target exposed');
    assert.ok('p_mix'     in result.paramTargets, 'mix target exposed');

    // Each fallback target should expose a customSetter (no AudioParam path)
    assert.equal(typeof result.paramTargets.p_attack.customSetter,  'function');
    assert.equal(typeof result.paramTargets.p_sustain.customSetter, 'function');
    assert.equal(typeof result.paramTargets.p_makeup.customSetter,  'function');
    assert.equal(typeof result.paramTargets.p_mix.customSetter,     'function');

    // Driving the setters must not throw — proves the fallback wiring is live
    assert.doesNotThrow(() => result.paramTargets.p_attack.customSetter(50),  'attack setter');
    assert.doesNotThrow(() => result.paramTargets.p_sustain.customSetter(-30),'sustain setter');
    assert.doesNotThrow(() => result.paramTargets.p_makeup.customSetter(3),   'output_gain setter');
    assert.doesNotThrow(() => result.paramTargets.p_mix.customSetter(0.5),    'mix setter');
  } finally {
    uninstallWorklet();
  }
});

test('buildEnveloper — static params produce zero paramTargets', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'enveloper',
      params: {
        attack: 30,
        sustain: -20,
        output_gain: 2,
        mix: 0.8,
      },
    };
    const result = buildEnveloper(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'enveloper-static');
    assert.equal(Object.keys(result.paramTargets).length, 0,
      'static params produce no paramTargets');
  } finally {
    uninstallWorklet();
  }
});

test('R13_ENVELOPER_BUILDERS default export registers `enveloper`', () => {
  assert.equal(typeof R13_ENVELOPER_BUILDERS.enveloper, 'function');
});

// ── Transient detection ──────────────────────────────────────────────────
//
// Reimplements the worklet follower math in plain JS so we can verify the
// algorithm without a real AudioWorkletNode. A 5 ms drum hit followed by
// 200 ms of silence is fed to the followers; we assert:
//   - the attack envelope spikes during the hit (RMS_hit ≫ RMS_tail)
//   - the sustain envelope is bounded in [0, 1]
//   - the spike time aligns with the input transient (within a few ms)

function coefFromMs(ms, sr) {
  const samples = Math.max(1, (ms / 1000) * sr);
  return 1 - Math.exp(-1 / samples);
}

function runFollowers(input, sr, opts = {}) {
  const fa = coefFromMs(opts.fastAttackMs  ?? 1,    sr);
  const fr = coefFromMs(opts.fastReleaseMs ?? 10,   sr);
  const sa = coefFromMs(opts.slowAttackMs  ?? 30,   sr);
  const srC = coefFromMs(opts.slowReleaseMs ?? 300, sr);
  const pa = coefFromMs(5,    sr);
  const pr = coefFromMs(1000, sr);

  const N = input.length;
  const fastE = new Float32Array(N);
  const slowE = new Float32Array(N);
  const peakE = new Float32Array(N);
  const attackEnv  = new Float32Array(N);
  const sustainEnv = new Float32Array(N);

  let fast = 0, slow = 0, peak = 1e-6;
  for (let i = 0; i < N; i++) {
    const ax = Math.abs(input[i]);
    const cF = (ax > fast) ? fa  : fr;
    const cS = (ax > slow) ? sa  : srC;
    const cP = (ax > peak) ? pa  : pr;
    fast = (1 - cF) * fast + cF * ax;
    slow = (1 - cS) * slow + cS * ax;
    peak = (1 - cP) * peak + cP * ax;
    fastE[i] = fast; slowE[i] = slow; peakE[i] = peak;
    const denom = peak > 1e-5 ? peak : 1e-5;
    let a = (fast - slow) / denom;       if (a < 0) a = 0; else if (a > 1) a = 1;
    let s = (slow - 0.6 * fast) / denom; if (s < 0) s = 0; else if (s > 1) s = 1;
    attackEnv[i]  = a;
    sustainEnv[i] = s;
  }
  return { fastE, slowE, peakE, attackEnv, sustainEnv };
}

function rmsRange(arr, lo, hi) {
  let sum = 0; let n = 0;
  for (let i = lo; i < Math.min(hi, arr.length); i++) { sum += arr[i] * arr[i]; n++; }
  return Math.sqrt(sum / Math.max(1, n));
}

function maxRange(arr, lo, hi) {
  let m = 0;
  for (let i = lo; i < Math.min(hi, arr.length); i++) { if (arr[i] > m) m = arr[i]; }
  return m;
}

test('worklet algorithm — drum hit produces visible attack spike', () => {
  const sr = 48000;
  const totalMs = 250;
  const N = Math.floor(sr * totalMs / 1000);
  const input = new Float32Array(N);

  // 5 ms exponentially-decaying drum-hit-like burst at t = 20 ms
  const hitStart = Math.floor(sr * 20 / 1000);
  const hitLen   = Math.floor(sr *  5 / 1000);
  for (let i = 0; i < hitLen; i++) {
    const t = i / hitLen;
    // High-amplitude, fast-decaying burst (amplitude 0.95 at start)
    input[hitStart + i] = 0.95 * Math.exp(-6 * t) * Math.sin(2 * Math.PI * 200 * (i / sr));
  }
  // Trailing 50ms of low-level body to give the slow follower something
  // to track (so attack vs sustain envelopes are distinguishable).
  const bodyStart = hitStart + hitLen;
  const bodyLen   = Math.floor(sr * 50 / 1000);
  for (let i = 0; i < bodyLen; i++) {
    input[bodyStart + i] = 0.05 * Math.sin(2 * Math.PI * 200 * (i / sr));
  }
  // Rest is zero (silence)

  const { attackEnv, sustainEnv } = runFollowers(input, sr);

  // Window definitions (samples)
  const transientLo = hitStart;
  const transientHi = hitStart + Math.floor(sr * 8 / 1000);     // 8 ms after hit
  const tailLo      = hitStart + Math.floor(sr * 100 / 1000);   // 80 ms after body
  const tailHi      = N;

  const attackPeakHit = maxRange(attackEnv, transientLo, transientHi);
  const attackRmsTail = rmsRange(attackEnv, tailLo, tailHi);

  // 1) The attack envelope must peak meaningfully during the hit.
  assert.ok(attackPeakHit > 0.05,
    `attack envelope peak during hit should be > 0.05, got ${attackPeakHit.toFixed(4)}`);

  // 2) The attack envelope must be ≥ ~5x larger during the hit than during
  //    the silent tail — that's the whole point of "transient detection".
  const ratio = attackPeakHit / Math.max(1e-9, attackRmsTail);
  assert.ok(ratio > 5,
    `attack peak/tail ratio should be > 5 (transient-vs-silence), got ${ratio.toFixed(2)}`);

  // 3) Both envelopes must remain bounded in [0, 1] (clamp invariant).
  for (let i = 0; i < N; i++) {
    assert.ok(attackEnv[i]  >= 0 && attackEnv[i]  <= 1, `attack[${i}]=${attackEnv[i]} out of [0,1]`);
    assert.ok(sustainEnv[i] >= 0 && sustainEnv[i] <= 1, `sustain[${i}]=${sustainEnv[i]} out of [0,1]`);
  }

  // 4) Attack peak should occur within ~10 ms of the hit start, not later.
  let argmax = transientLo;
  let m = 0;
  for (let i = transientLo; i < transientHi; i++) {
    if (attackEnv[i] > m) { m = attackEnv[i]; argmax = i; }
  }
  const peakDelayMs = (argmax - hitStart) * 1000 / sr;
  assert.ok(peakDelayMs < 10,
    `attack envelope peak should land within 10 ms of transient, got ${peakDelayMs.toFixed(2)} ms`);
});

test('worklet algorithm — gain math: positive attack increases peak gain at transient', () => {
  // Verify that, given an attack envelope value > 0, applying a positive
  // attack_pct produces total gain > 1 — the actual transient boost.
  const sr = 48000;
  const N  = Math.floor(sr * 0.05);
  const input = new Float32Array(N);
  const hitStart = Math.floor(sr * 0.005);
  const hitLen   = Math.floor(sr * 0.003);
  for (let i = 0; i < hitLen; i++) {
    input[hitStart + i] = 0.9 * Math.exp(-6 * (i / hitLen));
  }

  const { attackEnv, sustainEnv } = runFollowers(input, sr);
  const aPeak = maxRange(attackEnv, hitStart, hitStart + hitLen + 200);

  // Gain math from the worklet
  const attackPct  = 80;
  const sustainPct = 0;
  const aGain = 1 + (attackPct  / 100) * aPeak;
  const sGain = 1 + (sustainPct / 100) * 0.0;
  const total = aGain * sGain;

  assert.ok(total > 1.05,
    `positive attack should produce total gain > 1.05, got ${total.toFixed(3)}`);
  // And negative attack should reduce it
  const aGainNeg = 1 + (-80 / 100) * aPeak;
  assert.ok(aGainNeg < 1.0,
    `negative attack should produce gain < 1.0, got ${aGainNeg.toFixed(3)}`);
});
