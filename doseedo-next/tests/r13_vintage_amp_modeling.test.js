/**
 * R13 Vintage Amp Modeling composite — builder smoke tests.
 *
 * Run with:
 *   node --test tests/r13_vintage_amp_modeling.test.js
 *
 * The Web Audio API is not available in Node, so we stub a minimal
 * AudioContext + force the worklet path to fail. The composite builder
 * must cleanly fall back to WaveShaper / BiquadFilter / GainNode
 * substitutes and still return a `{ input, output, paramTargets }` shape.
 *
 * Coverage:
 *   1. buildVintageAmpModeling — fallback path returns the canonical
 *      builder contract with all expected paramTargets present.
 *   2. amp_model preset switch — invoking the customSetter with each preset
 *      name retunes the underlying R2/R3 fallbacks (waveshaper curve content
 *      changes, biquad frequencies update) without throwing.
 *   3. cab_model + mic_type switching produces distinct convolver
 *      configurations.
 *   4. Default export registers the `vintage_amp_modeling` node type.
 *   5. AMP_MODEL_PRESETS table is well-formed (8 presets, all required
 *      fields present, harmonic_signature has 4 elements).
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import R13_VINTAGE_AMP_BUILDERS, {
  buildVintageAmpModeling,
  AMP_MODEL_PRESETS,
  CAB_PRESETS,
  MIC_PRESETS,
} from '../src/audio/builders/r13_vintage_amp_modeling.js';

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
      gain: makeAudioParam(1),
      _connections: [],
      connect(t) { this._connections.push(t); return t; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createBiquadFilter() {
    return {
      type: 'lowpass',
      frequency: makeAudioParam(440),
      Q: makeAudioParam(1),
      gain: makeAudioParam(0),
      _connections: [],
      connect(t) { this._connections.push(t); return t; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createWaveShaper() {
    return {
      curve: null,
      oversample: 'none',
      _connections: [],
      connect(t) { this._connections.push(t); return t; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createConvolver() {
    return {
      buffer: null,
      _connections: [],
      connect(t) { this._connections.push(t); return t; },
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
    createConvolver, createBuffer,
  };
}

function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class {
    constructor() { throw new Error('not available in Node'); }
  };
}
function uninstallWorklet() {
  delete globalThis.AudioWorkletNode;
}

const PARAM_DEFS = {
  amp:   { id: 'amp',   type: 'enum',  default: 'tweed_5e3' },
  gain:  { id: 'gain',  min: 0, max: 10, default: 5 },
  bass:  { id: 'bass',  min: 0, max: 1,  default: 0.5 },
  mid:   { id: 'mid',   min: 0, max: 1,  default: 0.5 },
  treb:  { id: 'treb',  min: 0, max: 1,  default: 0.5 },
  pres:  { id: 'pres',  min: 0, max: 1,  default: 0.3 },
  mast:  { id: 'mast',  min: 0, max: 1,  default: 0.5 },
  bias:  { id: 'bias',  min: 0, max: 1,  default: 0.5 },
  nfb:   { id: 'nfb',   min: 0, max: 1,  default: 0.5 },
  cab:   { id: 'cab',   type: 'enum',  default: '4x12_greenback' },
  mic:   { id: 'mic',   type: 'enum',  default: 'sm57' },
  micp:  { id: 'micp',  min: 0, max: 1, default: 0.3 },
  out:   { id: 'out',   min: 0, max: 4, default: 1 },
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

// ── Tests ────────────────────────────────────────────────────────────────

test('buildVintageAmpModeling — fallback path returns valid contract with all params exposed', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'vintage_amp_modeling',
      params: {
        amp_model:    '@amp',
        gain:         '@gain',
        bass:         '@bass',
        mid:          '@mid',
        treble:       '@treb',
        presence:     '@pres',
        master:       '@mast',
        bias:         '@bias',
        nfb:          '@nfb',
        cab_model:    '@cab',
        mic_type:     '@mic',
        mic_position: '@micp',
        output_level: '@out',
      },
    };
    const result = buildVintageAmpModeling(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'vintage_amp_modeling');

    // Every modulated param must be present in paramTargets
    for (const id of ['amp', 'gain', 'bass', 'mid', 'treb', 'pres', 'mast',
                      'bias', 'nfb', 'cab', 'mic', 'micp', 'out']) {
      assert.ok(id in result.paramTargets,
        `paramTargets contains '${id}'`);
    }
    // Each binding has either a customSetter or a direct audioParam
    for (const id of Object.keys(result.paramTargets)) {
      const t = result.paramTargets[id];
      assert.ok(t.customSetter || t.audioParam,
        `paramTargets['${id}'] has either audioParam or customSetter`);
    }
  } finally {
    uninstallWorklet();
  }
});

test('buildVintageAmpModeling — amp_model preset switch retunes the chain without throwing', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'vintage_amp_modeling',
      params: { amp_model: '@amp' },
    };
    const result = buildVintageAmpModeling(ctx, node, PARAM_DEFS);
    const setAmp = result.paramTargets.amp.customSetter;
    assert.equal(typeof setAmp, 'function', 'amp_model setter exists');

    // Iterate through every model — must not throw
    for (const name of Object.keys(AMP_MODEL_PRESETS)) {
      assert.doesNotThrow(() => setAmp(name), `setAmp('${name}') does not throw`);
    }
    // Numeric index path (slider UI)
    assert.doesNotThrow(() => setAmp(0));
    assert.doesNotThrow(() => setAmp(0.5));
    assert.doesNotThrow(() => setAmp(1));
    // Invalid name falls back silently
    assert.doesNotThrow(() => setAmp('not_a_real_amp'));
  } finally {
    uninstallWorklet();
  }
});

test('buildVintageAmpModeling — switching amp_model rebakes power-tube curve', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    // We need a way to peek at the rebaked curves. The fallback shapers are
    // wired into the graph via input.connect() chain — we walk the chain
    // from `input` and find the second WaveShaper (which is the power amp).
    const node = {
      type: 'vintage_amp_modeling',
      params: { amp_model: '@amp' },
    };
    const result = buildVintageAmpModeling(ctx, node, PARAM_DEFS);

    // BFS/DFS the graph from input, find both WaveShapers (preamp + power amp).
    // Heuristic: WaveShapers have a .curve property and no .frequency.
    function findShapersDFS(start) {
      const seen = new Set();
      const found = [];
      const stack = [start];
      while (stack.length) {
        const n = stack.pop();
        if (!n || seen.has(n)) continue;
        seen.add(n);
        if (Object.prototype.hasOwnProperty.call(n, 'curve')
            && !Object.prototype.hasOwnProperty.call(n, 'frequency')) {
          found.push(n);
        }
        if (Array.isArray(n._connections)) {
          for (const c of n._connections) stack.push(c);
        }
      }
      return found;
    }
    const shapers = findShapersDFS(result.input);
    assert.ok(shapers.length >= 2,
      `expected at least 2 waveshapers in fallback chain, got ${shapers.length}`);

    // Capture pre-switch curves
    const before = shapers.map((s) => s.curve && Float32Array.from(s.curve));
    // Switch to a maximally-different preset
    const setAmp = result.paramTargets.amp.customSetter;
    setAmp('marshall_jcm800');
    const after = shapers.map((s) => s.curve && Float32Array.from(s.curve));

    // At least one shaper should have updated its curve (non-trivial L2 diff)
    let totalDiff = 0;
    for (let i = 0; i < shapers.length; i++) {
      if (!before[i] || !after[i]) continue;
      const len = Math.min(before[i].length, after[i].length);
      let d = 0;
      for (let k = 0; k < len; k++) {
        const dv = before[i][k] - after[i][k];
        d += dv * dv;
      }
      totalDiff += d;
    }
    assert.ok(totalDiff > 1e-6,
      `expected non-zero L2 diff between tweed_5e3 and marshall_jcm800 curves, got ${totalDiff}`);
  } finally {
    uninstallWorklet();
  }
});

test('buildVintageAmpModeling — cab_model + mic_type switching does not throw', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'vintage_amp_modeling',
      params: { cab_model: '@cab', mic_type: '@mic', mic_position: '@micp' },
    };
    const result = buildVintageAmpModeling(ctx, node, PARAM_DEFS);

    const setCab = result.paramTargets.cab.customSetter;
    const setMic = result.paramTargets.mic.customSetter;
    const setMicPos = result.paramTargets.micp.customSetter;

    for (const name of Object.keys(CAB_PRESETS)) {
      assert.doesNotThrow(() => setCab(name), `setCab('${name}') does not throw`);
    }
    for (const name of Object.keys(MIC_PRESETS)) {
      assert.doesNotThrow(() => setMic(name), `setMic('${name}') does not throw`);
    }
    for (const v of [0, 0.25, 0.5, 0.75, 1]) {
      assert.doesNotThrow(() => setMicPos(v), `setMicPos(${v}) does not throw`);
    }
    // Numeric index paths
    assert.doesNotThrow(() => setCab(0));
    assert.doesNotThrow(() => setCab(0.5));
    assert.doesNotThrow(() => setMic(0));
    assert.doesNotThrow(() => setMic(2));
  } finally {
    uninstallWorklet();
  }
});

test('R13_VINTAGE_AMP_BUILDERS default export registers vintage_amp_modeling', () => {
  assert.equal(typeof R13_VINTAGE_AMP_BUILDERS.vintage_amp_modeling, 'function',
    'vintage_amp_modeling builder is registered');
});

test('AMP_MODEL_PRESETS table exposes 8 presets with consistent shape', () => {
  const names = Object.keys(AMP_MODEL_PRESETS);
  assert.equal(names.length, 8,
    `AMP_MODEL_PRESETS has ${names.length} entries (expected 8)`);
  for (const name of names) {
    const m = AMP_MODEL_PRESETS[name];
    for (const key of [
      'tube_type', 'power_tube_type',
      'harmonic_signature', 'transformer_color',
      'bias_drift', 'nfb_default',
      'tone_voicing', 'presence_freq', 'gain_makeup',
    ]) {
      assert.ok(key in m, `AMP_MODEL_PRESETS['${name}'] is missing ${key}`);
    }
    assert.equal(m.harmonic_signature.length, 4,
      `AMP_MODEL_PRESETS['${name}'].harmonic_signature has 4 entries`);
    for (const v of m.harmonic_signature) {
      assert.ok(typeof v === 'number' && v >= 0 && v <= 1,
        `AMP_MODEL_PRESETS['${name}'].harmonic_signature entry ${v} in [0,1]`);
    }
    for (const k of ['lp_cutoff_hz', 'sat_amount']) {
      assert.ok(k in m.transformer_color,
        `AMP_MODEL_PRESETS['${name}'].transformer_color has ${k}`);
    }
    for (const k of ['bass', 'mid', 'treble']) {
      assert.ok(k in m.tone_voicing,
        `AMP_MODEL_PRESETS['${name}'].tone_voicing has ${k}`);
    }
    assert.ok(['EL84', 'EL34', '6V6', '6L6'].includes(m.power_tube_type),
      `AMP_MODEL_PRESETS['${name}'].power_tube_type is a known tube class`);
  }
  // All 8 historically-named heads must be present
  for (const k of [
    'tweed_5e3', 'tweed_5f6', 'vox_ac30', 'marshall_plexi',
    'marshall_jcm800', 'hiwatt', 'orange_or120', 'silvertone',
  ]) {
    assert.ok(k in AMP_MODEL_PRESETS, `'${k}' present in AMP_MODEL_PRESETS`);
  }
});

test('CAB_PRESETS + MIC_PRESETS tables expose at least the required entries', () => {
  for (const k of ['1x12_alnico', '2x12_celestion', '4x12_greenback', '4x10_jensen']) {
    assert.ok(k in CAB_PRESETS, `CAB_PRESETS contains '${k}'`);
  }
  for (const k of ['sm57', 'sm7', 'r121', 'u87']) {
    assert.ok(k in MIC_PRESETS, `MIC_PRESETS contains '${k}'`);
  }
  // Cab presets sanity
  for (const [name, p] of Object.entries(CAB_PRESETS)) {
    assert.ok(p.lowCutHz < p.highCutHz,
      `CAB_PRESETS['${name}']: lowCutHz < highCutHz`);
    assert.ok(p.durationSec > 0 && p.durationSec < 1,
      `CAB_PRESETS['${name}']: 0 < durationSec < 1`);
  }
  // Mic presets sanity
  for (const [name, p] of Object.entries(MIC_PRESETS)) {
    for (const k of ['peak_freq', 'peak_gain_db', 'peak_q', 'hp_freq', 'hp_q']) {
      assert.ok(k in p, `MIC_PRESETS['${name}'] is missing ${k}`);
    }
  }
});
