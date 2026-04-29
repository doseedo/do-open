/**
 * R13 ChromaVerb extended FDN reverb — builder smoke tests.
 *
 * Run with:
 *   node --test tests/r13_chromaverb.test.js
 *
 * The Web Audio API is not available in Node, so we stub a minimal
 * AudioContext and force the worklet path to fail. Each builder must
 * cleanly fall back to a ConvolverNode + dry/wet gain pair and still
 * return a `{ input, output, paramTargets }` shape compatible with the
 * WebAudioDSPEngine builder contract.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  buildFdnSmooth,
  buildFdnStrange,
  buildFdnDense,
} from '../src/audio/builders/r13_chromaverb.js';
import R13_BUILDERS from '../src/audio/builders/r13_chromaverb.js';

// ── Minimal AudioContext stub ────────────────────────────────────────────
// Returns the bare surface the R13 builders touch in the fallback path.
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
  function createConvolver() {
    return {
      buffer: null,
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
  return { sampleRate, createGain, createConvolver, createBuffer };
}

// Force the worklet path to fail so we exercise the fallback branch.
// _safeWorklet() in the builder calls `new AudioWorkletNode(...)` and
// catches; defining a throwing constructor pushes builds onto the
// ConvolverNode fallback.
function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class { constructor() { throw new Error('not available in Node'); } };
}

function uninstallWorklet() {
  delete globalThis.AudioWorkletNode;
}

const PARAM_DEFS = {
  decay: { id: 'decay', min: 0.1, max: 20.0, default: 2.5 },
  mix:   { id: 'mix',   min: 0,   max: 1,    default: 0.3 },
};

// Each test instantiates a fresh ctx + node spec, runs the builder,
// asserts the contract, and confirms the audio path connects cleanly.
function expectBuilderContract(result, label) {
  assert.ok(result, `${label}: builder returned a value`);
  assert.ok(result.input, `${label}: result.input present`);
  assert.ok(result.output, `${label}: result.output present`);
  assert.ok(result.paramTargets && typeof result.paramTargets === 'object',
    `${label}: paramTargets is an object`);
  // Connect a downstream node to ensure output is connectable
  const sink = { _connections: [], connect() {}, disconnect() {} };
  // builder's input.connect(downstream) should not throw — fallback wires
  // input → dryF + convolver internally; output.connect should also work.
  assert.doesNotThrow(() => result.output.connect(sink),
    `${label}: output.connect() works`);
}

test('buildFdnSmooth — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'fdn_smooth',
      params: { decay_time: '@decay', mix: '@mix', damping: 0.5, width: 0.8 },
    };
    const result = buildFdnSmooth(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'fdn_smooth');
    assert.ok('decay' in result.paramTargets, 'fdn_smooth: decay is exposed');
    assert.ok('mix' in result.paramTargets,   'fdn_smooth: mix is exposed');
    // mix on fallback should drive the dry/wet pair via customSetter
    assert.equal(typeof result.paramTargets.mix.customSetter, 'function',
      'fdn_smooth: mix has customSetter in fallback');
  } finally {
    uninstallWorklet();
  }
});

test('buildFdnStrange — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'fdn_strange',
      params: { decay_time: 4.0, mix: 0.4, diffusion: 0.7 },
    };
    const result = buildFdnStrange(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'fdn_strange');
    // Static values: no paramTargets entries should be created
    assert.equal(Object.keys(result.paramTargets).length, 0,
      'fdn_strange: static params produce no paramTargets');
  } finally {
    uninstallWorklet();
  }
});

test('buildFdnDense — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'fdn_dense',
      params: { decay_time: '@decay', pre_delay: 20, damping: 0.2, mix: 0.5 },
    };
    const result = buildFdnDense(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'fdn_dense');
    assert.ok('decay' in result.paramTargets, 'fdn_dense: modulated decay exposed');
    // Driving the customSetter must not throw — proves the fallback IR
    // regen path is wired
    assert.doesNotThrow(
      () => result.paramTargets.decay.customSetter(2.0),
      'fdn_dense: decay setter executes',
    );
  } finally {
    uninstallWorklet();
  }
});

test('R13_BUILDERS default export registers the three node types', () => {
  assert.equal(typeof R13_BUILDERS.fdn_smooth,  'function');
  assert.equal(typeof R13_BUILDERS.fdn_strange, 'function');
  assert.equal(typeof R13_BUILDERS.fdn_dense,   'function');
});
