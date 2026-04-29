/**
 * r13_vocoder.test.js — smoke tests for the Vocoder builder
 *
 * Run with plain node (zero-install):
 *   node tests/r13_vocoder.test.js
 *
 * The repo doesn't ship a Web Audio test harness, so we provide a tiny
 * in-process Web Audio mock — just enough surface for buildVocoder to
 * construct its graph in either the worklet path or the fallback path.
 *
 * Tests:
 *   1. Builder returns the right shape (input / output / paramTargets / inputs).
 *   2. inputs.length === 2 (modulator + carrier slots).
 *   3. Worklet path uses the worklet-name we expect, and parameterData
 *      matches our schema (band_idx + carrier_type enum mapping).
 *   4. Fallback path constructs N analysis bandpasses + N synthesis
 *      bandpasses + N gain controls.
 *   5. Logarithmic band-center generator produces N strictly-increasing
 *      centers spanning 100 Hz .. 8 kHz.
 */

import { strict as assert } from 'node:assert';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ─────────────────────────────────────────────────────────────────────────
// Web Audio mock
// ─────────────────────────────────────────────────────────────────────────

class MockAudioParam {
  constructor(v = 0) { this.value = v; }
  setValueAtTime(v) { this.value = v; }
  setTargetAtTime(v) { this.value = v; }
  cancelScheduledValues() {}
  linearRampToValueAtTime(v) { this.value = v; }
}

let __nodeId = 0;
class MockNode {
  constructor(ctx, kind) {
    this.id = ++__nodeId;
    this.ctx = ctx;
    this.kind = kind;
    this._outs = new Set();
  }
  connect(target) { this._outs.add(target); return target; }
  disconnect() { this._outs.clear(); }
}
class MockGain extends MockNode {
  constructor(ctx) { super(ctx, 'gain'); this.gain = new MockAudioParam(1); }
}
class MockBiquad extends MockNode {
  constructor(ctx) {
    super(ctx, 'biquad');
    this.frequency = new MockAudioParam(1000);
    this.Q = new MockAudioParam(1);
    this.gain = new MockAudioParam(0);
    this.type = 'lowpass';
  }
}
class MockAnalyser extends MockNode {
  constructor(ctx) { super(ctx, 'analyser'); this.fftSize = 256; this.smoothingTimeConstant = 0.5; }
  getByteTimeDomainData(arr) { for (let i = 0; i < arr.length; i++) arr[i] = 128; }
}
class MockOsc extends MockNode {
  constructor(ctx) { super(ctx, 'osc'); this.frequency = new MockAudioParam(440); this.detune = new MockAudioParam(0); this.type = 'sine'; }
  start() {} stop() {}
}
class MockBufferSource extends MockNode {
  constructor(ctx) { super(ctx, 'bufsrc'); this.buffer = null; this.loop = false; }
  start() {} stop() {}
}
class MockBuffer {
  constructor(channels, length, sr) {
    this.numberOfChannels = channels; this.length = length; this.sampleRate = sr;
    this._data = Array.from({ length: channels }, () => new Float32Array(length));
  }
  getChannelData(ch) { return this._data[ch]; }
}

class MockAudioWorkletNode extends MockNode {
  constructor(ctx, name, options = {}) {
    super(ctx, 'worklet');
    this.processorName = name;
    this.options = options;
    this.numberOfInputs = options.numberOfInputs || 1;
    this.numberOfOutputs = options.numberOfOutputs || 1;
    // Build parameters Map from parameterDescriptors-equivalent set
    this.parameters = new Map();
    const knownParams = [
      'bands_idx', 'attack_ms', 'release_ms', 'formant_shift',
      'carrier_type', 'carrier_freq', 'mix', 'unvoiced_mix', 'q',
    ];
    for (const p of knownParams) {
      const initial = (options.parameterData && options.parameterData[p] != null)
        ? options.parameterData[p] : 0;
      this.parameters.set(p, new MockAudioParam(initial));
    }
  }
}

class MockCtx {
  constructor({ workletAvailable = true } = {}) {
    this.sampleRate = 48000;
    this.currentTime = 0;
    this.state = 'running';
    this._workletAvailable = workletAvailable;
    this._created = []; // every node we ever made (for assertions)
  }
  _track(n) { this._created.push(n); return n; }
  createGain()           { return this._track(new MockGain(this)); }
  createBiquadFilter()   { return this._track(new MockBiquad(this)); }
  createAnalyser()       { return this._track(new MockAnalyser(this)); }
  createOscillator()     { return this._track(new MockOsc(this)); }
  createBufferSource()   { return this._track(new MockBufferSource(this)); }
  createBuffer(c, l, sr) { return new MockBuffer(c, l, sr); }
}

// ─────────────────────────────────────────────────────────────────────────
// Wire up globals so r13_vocoder.js can `new AudioWorkletNode(...)`
// ─────────────────────────────────────────────────────────────────────────

let __installedAvailable = true;
function installWorkletGlobal(available) {
  __installedAvailable = available;
  globalThis.AudioWorkletNode = function (ctx, name, options) {
    if (!__installedAvailable) throw new Error('worklet not registered');
    return new MockAudioWorkletNode(ctx, name, options);
  };
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

let passed = 0, failed = 0;
function test(name, fn) {
  try { fn(); console.log(`ok   ${name}`); passed++; }
  catch (e) { console.error(`FAIL ${name}\n  ${e.stack || e}`); failed++; }
}

const builderPath = path.resolve(__dirname, '../src/audio/builders/r13_vocoder.js');
const { buildVocoder, default: R13_BUILDERS } =
  await import(`file://${builderPath}`);

// ── 1. Default export shape ──────────────────────────────────────────────
test('default export registers `vocoder`', () => {
  assert.ok(R13_BUILDERS, 'no default export');
  assert.equal(typeof R13_BUILDERS.vocoder, 'function', 'vocoder builder missing');
  assert.equal(R13_BUILDERS.vocoder, buildVocoder);
});

// ── 2. Worklet path: builder returns the right shape ─────────────────────
test('worklet path: builder returns input/output/paramTargets/inputs[2]', () => {
  installWorkletGlobal(true);
  const ctx = new MockCtx();
  const result = buildVocoder(ctx, {
    type: 'vocoder',
    params: { bands: 16, carrier_type: 'saw', mix: 0.8 },
  }, {});
  assert.ok(result.input, 'no input');
  assert.ok(result.output, 'no output');
  assert.ok(Array.isArray(result.inputs), 'inputs missing');
  assert.equal(result.inputs.length, 2, 'expected 2 input slots');
  assert.equal(typeof result.paramTargets, 'object');
});

// ── 3. Worklet path: parameterData passes through correctly ──────────────
test('worklet path: bands enum + carrier_type enum map to integers', () => {
  installWorkletGlobal(true);
  const ctx = new MockCtx();
  const result = buildVocoder(ctx, {
    type: 'vocoder',
    params: {
      bands: 32,                  // → bands_idx = 3
      carrier_type: 'external',   // → carrier_type = 3
      attack_ms: 12.5,
      release_ms: 200,
      formant_shift: -3,
      carrier_freq: 220,
      mix: 0.6,
      unvoiced_mix: 0.4,
      q: 18,
    },
  }, {});
  // The output of the worklet path is the AudioWorkletNode itself.
  const w = result.output;
  assert.equal(w.processorName, 'r13-vocoder-processor');
  const pd = w.options.parameterData;
  assert.equal(pd.bands_idx, 3);
  assert.equal(pd.carrier_type, 3);
  assert.equal(pd.attack_ms, 12.5);
  assert.equal(pd.release_ms, 200);
  assert.equal(pd.formant_shift, -3);
  assert.equal(pd.carrier_freq, 220);
  assert.equal(pd.mix, 0.6);
  assert.equal(pd.unvoiced_mix, 0.4);
  assert.equal(pd.q, 18);
  // numberOfInputs must be 2 so the engine can wire mod + carrier
  assert.equal(w.numberOfInputs, 2);
});

// ── 4. Worklet path: '@'-bound params register paramTargets ──────────────
test('worklet path: @-bound params show up in paramTargets', () => {
  installWorkletGlobal(true);
  const ctx = new MockCtx();
  const paramDefs = {
    pid_attack:  { id: 'pid_attack',  min: 0.1, max: 100, default: 5 },
    pid_release: { id: 'pid_release', min: 1,   max: 500, default: 50 },
    pid_mix:     { id: 'pid_mix',     min: 0,   max: 1,   default: 1 },
  };
  const result = buildVocoder(ctx, {
    type: 'vocoder',
    params: {
      attack_ms:  '@pid_attack',
      release_ms: '@pid_release',
      mix:        '@pid_mix',
    },
  }, paramDefs);
  assert.ok(result.paramTargets.pid_attack);
  assert.ok(result.paramTargets.pid_release);
  assert.ok(result.paramTargets.pid_mix);
  // Each target should have either an audioParam or a customSetter
  for (const k of ['pid_attack', 'pid_release', 'pid_mix']) {
    const t = result.paramTargets[k];
    const hasAP = !!t.audioParam;
    const hasCS = typeof t.customSetter === 'function';
    assert.ok(hasAP || hasCS, `target ${k} has neither audioParam nor customSetter`);
  }
});

// ── 5. Fallback path: builds correct # of bandpass filters ───────────────
test('fallback path: N analysis BPs + N synthesis BPs + N gains', () => {
  installWorkletGlobal(false);
  const ctx = new MockCtx({ workletAvailable: false });
  const result = buildVocoder(ctx, {
    type: 'vocoder',
    params: { bands: 16, carrier_type: 'saw', carrier_freq: 110, mix: 1 },
  }, {});
  assert.ok(result.input);
  assert.ok(result.output);
  assert.equal(result.inputs.length, 2);
  // Count bandpass biquads built. Each band creates: 1 ana BP + 1 syn BP.
  const biquads = ctx._created.filter((n) => n.kind === 'biquad');
  // Synthesis bandpasses get type='bandpass' set; analysis ones too.
  // We know for N=16: 32 biquads.
  assert.equal(biquads.length, 32, `expected 32 biquads, got ${biquads.length}`);
  // analysers: one per analysis band
  const analysers = ctx._created.filter((n) => n.kind === 'analyser');
  assert.equal(analysers.length, 16, `expected 16 analyser nodes, got ${analysers.length}`);
});

// ── 6. Fallback path: external carrier wires inputs[1] not internal osc ──
test('fallback path: carrier_type=external uses input[1] (no internal osc)', () => {
  installWorkletGlobal(false);
  const ctx = new MockCtx({ workletAvailable: false });
  const result = buildVocoder(ctx, {
    type: 'vocoder',
    params: { bands: 8, carrier_type: 'external' },
  }, {});
  const oscs = ctx._created.filter((n) => n.kind === 'osc');
  const bufsrcs = ctx._created.filter((n) => n.kind === 'bufsrc');
  assert.equal(oscs.length, 0, 'no internal oscillator should be created for external carrier');
  assert.equal(bufsrcs.length, 0, 'no internal noise source should be created for external carrier');
  // Carrier slot is inputs[1]
  assert.ok(result.inputs[1], 'carrier input slot missing');
});

// ── 7. Fallback path: carrier_type=noise builds a buffer source ──────────
test('fallback path: carrier_type=noise builds a BufferSource', () => {
  installWorkletGlobal(false);
  const ctx = new MockCtx({ workletAvailable: false });
  buildVocoder(ctx, {
    type: 'vocoder',
    params: { bands: 8, carrier_type: 'noise' },
  }, {});
  const bufsrcs = ctx._created.filter((n) => n.kind === 'bufsrc');
  assert.equal(bufsrcs.length, 1, 'expected exactly 1 BufferSource for noise carrier');
  const oscs = ctx._created.filter((n) => n.kind === 'osc');
  assert.equal(oscs.length, 0, 'no oscillator should be created for noise carrier');
});

// ── 8. Logarithmic band-center spacing ───────────────────────────────────
test('log-band centers: N bands span 100 Hz .. 8 kHz, strictly increasing', () => {
  // Reach into the worklet processor's helper by re-implementing the same
  // formula here — the worklet itself can't be imported in node, so we
  // verify the builder's fallback uses an equivalent layout. Center
  // frequencies are exposed indirectly via the BiquadFilter.frequency.value
  // we set in fallback.
  installWorkletGlobal(false);
  const ctx = new MockCtx({ workletAvailable: false });
  buildVocoder(ctx, {
    type: 'vocoder',
    params: { bands: 16, carrier_type: 'saw' },
  }, {});
  const biquads = ctx._created.filter((n) => n.kind === 'biquad');
  // The fallback builds them interleaved: [ana0, syn0, ana1, syn1, ...].
  // For formant_shift=0, ana[i] === syn[i], so even-indexed = analysis,
  // odd-indexed = synthesis at matching center.
  const anaFreqs = [];
  for (let i = 0; i < biquads.length; i += 2) anaFreqs.push(biquads[i].frequency.value);
  assert.equal(anaFreqs.length, 16, `expected 16 analysis biquads, got ${anaFreqs.length}`);
  assert.ok(anaFreqs[0] >= 99 && anaFreqs[0] <= 101,
    `first analysis freq ~100 Hz, got ${anaFreqs[0]}`);
  assert.ok(anaFreqs[15] >= 7900 && anaFreqs[15] <= 8100,
    `last analysis freq ~8 kHz, got ${anaFreqs[15]}`);
  for (let i = 1; i < anaFreqs.length; i++) {
    assert.ok(anaFreqs[i] > anaFreqs[i - 1],
      `freqs not strictly increasing at ${i}: ${anaFreqs[i - 1]} → ${anaFreqs[i]}`);
  }
});

// ── 9. Synthesis bands shift with formant_shift ──────────────────────────
test('formant_shift: synthesis BP centers = analysis × 2^(semi/12)', () => {
  installWorkletGlobal(false);
  const ctx = new MockCtx({ workletAvailable: false });
  buildVocoder(ctx, {
    type: 'vocoder',
    params: { bands: 8, carrier_type: 'saw', formant_shift: 12 },
  }, {});
  const biquads = ctx._created.filter((n) => n.kind === 'biquad');
  // Build order is interleaved [ana0, syn0, ana1, syn1, ...].
  const ana = []; const syn = [];
  for (let i = 0; i < biquads.length; i += 2) {
    ana.push(biquads[i].frequency.value);
    syn.push(biquads[i + 1].frequency.value);
  }
  assert.equal(ana.length, 8); assert.equal(syn.length, 8);
  for (let i = 0; i < 8; i++) {
    const expected = ana[i] * 2;
    const ratio = syn[i] / expected;
    assert.ok(ratio > 0.99 && ratio < 1.01,
      `band ${i}: synthesis center ${syn[i]} should be 2× analysis ${ana[i]} (got ratio ${ratio})`);
  }
});

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
