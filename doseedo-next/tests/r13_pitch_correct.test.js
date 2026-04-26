/**
 * r13_pitch_correct.test.js — unit tests for R13 Pitch Correction.
 *
 * Two tests:
 *   1. Builder smoke — buildPitchCorrect returns { input, output,
 *      paramTargets } using a minimal Web Audio mock, both with literal
 *      and '@<id>'-modulated params. AudioWorklet isn't available under
 *      plain node, so the builder hits its passthrough fallback —
 *      that's intentional and exercises the fallback path.
 *
 *   2. YIN pitch detection — 1 second of a synthetic 440 Hz sine is
 *      fed through a node-side reimplementation of the YIN difference /
 *      CMND / parabolic-interp pipeline used in the worklet. Asserts
 *      detected f0 is within ±2 Hz of 440.
 *
 *      The YIN implementation is duplicated here on purpose: the
 *      worklet is self-contained (no importScripts) so the algorithm
 *      itself isn't importable from node. We instead validate the
 *      algorithm's spec — the same recipe runs in the worklet.
 *
 * Run:
 *   node tests/r13_pitch_correct.test.js
 *
 * The test exits non-zero on failure.
 */

import { strict as assert } from 'node:assert';
import { buildPitchCorrect, R13_SCALE_MASKS } from '../src/audio/builders/r13_pitch_correct.js';

// ─────────────────────────────────────────────────────────────────────
// Minimal Web Audio mock — just enough for the builder.
// AudioWorkletNode is intentionally undefined on globalThis so the
// builder's _safeWorklet() catch path engages.
// ─────────────────────────────────────────────────────────────────────

class MockAudioParam {
  constructor(v = 0) { this.value = v; }
  setValueAtTime(v) { this.value = v; }
  setTargetAtTime(v) { this.value = v; }
}

class MockGainNode {
  constructor() {
    this.gain = new MockAudioParam(1);
    this._connected = new Set();
  }
  connect(t) { if (t) this._connected.add(t); return t; }
  disconnect() { this._connected.clear(); }
}

class MockAudioContext {
  constructor() {
    this.sampleRate = 48000;
    this.audioWorklet = { addModule: async () => {} };
  }
  createGain() { return new MockGainNode(); }
}

// Make sure no AudioWorkletNode is defined globally so _safeWorklet falls back.
if (typeof globalThis.AudioWorkletNode !== 'undefined') {
  delete globalThis.AudioWorkletNode;
}

// ─────────────────────────────────────────────────────────────────────
// Tiny test harness
// ─────────────────────────────────────────────────────────────────────
let passed = 0;
let failed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    process.stdout.write(`  ✓ ${name}\n`);
    passed++;
  } catch (err) {
    process.stdout.write(`  ✗ ${name}\n`);
    failed++;
    failures.push({ name, err });
  }
}

process.stdout.write('\nR13 Pitch Correction — unit tests\n');

// ─────────────────────────────────────────────────────────────────────
// Test 1: builder smoke
// ─────────────────────────────────────────────────────────────────────

await test('buildPitchCorrect returns input/output/paramTargets shape (literal params)', () => {
  const ctx = new MockAudioContext();
  const nodeDef = {
    type: 'pitch_correct',
    params: {
      key: 0,
      scale: 'major',
      response_ms: 50,
      correction_amount: 1,
      formant_preserve: 0,
      mix: 1,
    },
  };
  const result = buildPitchCorrect(ctx, nodeDef, {});
  assert.ok(result, 'expected result object');
  assert.ok(result.input, 'expected input node');
  assert.ok(result.output, 'expected output node');
  assert.equal(typeof result.paramTargets, 'object');
  // Literal params with no '@' bindings → no targets
  assert.equal(Object.keys(result.paramTargets).length, 0);
});

await test('buildPitchCorrect installs targets for @-bound params (fallback path)', () => {
  const ctx = new MockAudioContext();
  const paramDefs = {
    p_key:    { id: 'p_key', min: 0, max: 11 },
    p_resp:   { id: 'p_resp', min: 0, max: 500 },
    p_mask:   { id: 'p_mask', min: 0, max: 4095 },
    p_corr:   { id: 'p_corr', min: 0, max: 1 },
  };
  const nodeDef = {
    type: 'pitch_correct',
    params: {
      key: '@p_key',
      response_ms: '@p_resp',
      scale: 'custom',
      scale_mask: '@p_mask',
      correction_amount: '@p_corr',
    },
  };
  const result = buildPitchCorrect(ctx, nodeDef, paramDefs);
  assert.ok(result.paramTargets.p_key,  'expected p_key target');
  assert.ok(result.paramTargets.p_resp, 'expected p_resp target');
  assert.ok(result.paramTargets.p_mask, 'expected p_mask target');
  assert.ok(result.paramTargets.p_corr, 'expected p_corr target');
  // In the fallback (no AudioWorkletNode) path, targets are no-op customSetters.
  assert.equal(typeof result.paramTargets.p_key.customSetter, 'function');
  // No-op shouldn't throw.
  result.paramTargets.p_key.customSetter(5);
  result.paramTargets.p_mask.customSetter(0xAB);
});

await test('R13_SCALE_MASKS exposes major/minor/chromatic with correct cardinality', () => {
  // major scale → 7 notes
  let cnt = 0;
  for (let i = 0; i < 12; i++) if ((R13_SCALE_MASKS.major >> i) & 1) cnt++;
  assert.equal(cnt, 7, 'major must contain 7 pitch classes');
  cnt = 0;
  for (let i = 0; i < 12; i++) if ((R13_SCALE_MASKS.minor >> i) & 1) cnt++;
  assert.equal(cnt, 7, 'natural minor must contain 7 pitch classes');
  assert.equal(R13_SCALE_MASKS.chromatic, 0xFFF);
});

// ─────────────────────────────────────────────────────────────────────
// Test 2: YIN pitch detection on synthetic 440 Hz sine.
// We re-implement the same YIN pipeline that the worklet uses (the
// worklet is self-contained / not importable from node).
// ─────────────────────────────────────────────────────────────────────

function yinDetect(buf, sampleRate, opts = {}) {
  const W = opts.window || 2048;
  const halfW = W >> 1;
  const threshold = opts.threshold || 0.15;
  const minF0 = opts.minF0 || 70;
  const maxF0 = opts.maxF0 || 1100;

  // Take the last W samples
  const start = buf.length - W;
  const x = buf.slice(start, start + W);

  // d(τ)
  const d = new Float32Array(halfW);
  for (let tau = 0; tau < halfW; tau++) {
    let sum = 0;
    for (let i = 0; i < halfW; i++) {
      const diff = x[i] - x[i + tau];
      sum += diff * diff;
    }
    d[tau] = sum;
  }
  // Cumulative-mean normalised difference
  const cmnd = new Float32Array(halfW);
  cmnd[0] = 1;
  let running = 0;
  for (let tau = 1; tau < halfW; tau++) {
    running += d[tau];
    cmnd[tau] = d[tau] * tau / Math.max(running, 1e-12);
  }
  const minTau = Math.floor(sampleRate / maxF0);
  const maxTau = Math.min(halfW - 1, Math.floor(sampleRate / minF0));
  let tauEst = -1;
  for (let tau = Math.max(2, minTau); tau <= maxTau; tau++) {
    if (cmnd[tau] < threshold) {
      while (tau + 1 <= maxTau && cmnd[tau + 1] < cmnd[tau]) tau++;
      tauEst = tau;
      break;
    }
  }
  if (tauEst < 0) return 0;
  // Parabolic interpolation
  let betterTau = tauEst;
  if (tauEst > 0 && tauEst < halfW - 1) {
    const s0 = cmnd[tauEst - 1];
    const s1 = cmnd[tauEst];
    const s2 = cmnd[tauEst + 1];
    const denom = (s0 + s2 - 2 * s1);
    if (Math.abs(denom) > 1e-12) {
      betterTau = tauEst + 0.5 * (s0 - s2) / denom;
    }
  }
  if (betterTau <= 0) return 0;
  return sampleRate / betterTau;
}

await test('YIN detects 440 Hz sine within ±2 Hz', () => {
  const sr = 48000;
  const dur = 1.0;
  const N = Math.floor(sr * dur);
  const buf = new Float32Array(N);
  const omega = 2 * Math.PI * 440;
  for (let i = 0; i < N; i++) {
    buf[i] = Math.sin((omega * i) / sr);
  }
  const f0 = yinDetect(buf, sr, { window: 2048 });
  assert.ok(f0 > 0, `expected voiced detection, got ${f0}`);
  assert.ok(Math.abs(f0 - 440) < 2,
    `expected ~440 Hz, got ${f0.toFixed(3)} Hz (delta=${(f0 - 440).toFixed(3)})`);
});

await test('YIN detects 220 Hz sine within ±2 Hz (low end)', () => {
  const sr = 48000;
  const N = Math.floor(sr * 1.0);
  const buf = new Float32Array(N);
  const omega = 2 * Math.PI * 220;
  for (let i = 0; i < N; i++) {
    buf[i] = Math.sin((omega * i) / sr);
  }
  const f0 = yinDetect(buf, sr, { window: 2048 });
  assert.ok(f0 > 0, `expected voiced detection, got ${f0}`);
  assert.ok(Math.abs(f0 - 220) < 2,
    `expected ~220 Hz, got ${f0.toFixed(3)} Hz (delta=${(f0 - 220).toFixed(3)})`);
});

await test('YIN returns ~0 (or wide window) on white noise (unvoiced-ish)', () => {
  const sr = 48000;
  const N = sr;
  const buf = new Float32Array(N);
  for (let i = 0; i < N; i++) buf[i] = Math.random() * 2 - 1;
  const f0 = yinDetect(buf, sr, { window: 2048 });
  // We don't strictly demand 0, but f0 should not lock to a clean low pitch.
  // Sanity bound: shouldn't return a value within ±2 Hz of 440 by coincidence.
  if (f0 > 0) {
    assert.ok(Math.abs(f0 - 440) > 5,
      `noise unexpectedly locked near 440 Hz: ${f0}`);
  }
});

// ─────────────────────────────────────────────────────────────────────
// Done
// ─────────────────────────────────────────────────────────────────────

process.stdout.write(`\n${passed} passed, ${failed} failed\n`);
if (failed > 0) {
  for (const { name, err } of failures) {
    process.stderr.write(`\n--- ${name} ---\n${err.stack || err.message}\n`);
  }
  process.exit(1);
}
