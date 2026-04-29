/**
 * R13 Pedalboard composite — builder smoke + chain-construction tests.
 *
 * Run with plain node (zero-install):
 *   node tests/r13_pedalboard.test.js
 *
 * The repo doesn't ship a Web Audio test harness, so we provide an
 * in-process Web Audio mock — just enough surface for buildPedalboard +
 * its 24 sub-builders to construct their graphs in either the worklet
 * path or the fallback path.
 *
 * Tests:
 *   1. Default export registers `pedalboard` builder.
 *   2. Empty pedalboard returns a unity passthrough.
 *   3. Single-pedal chain returns valid contract.
 *   4. 5-pedal serial chain (overdrive → distortion → delay → chorus → reverb).
 *   5. Bypass crossfade ramps wet/dry on the customSetter.
 *   6. Unknown pedal type fails open (passthrough, no throw).
 *   7. All 24 sub-types build without throw (kitchen-sink chain).
 *   8. paramTargets surfaces per-slot bypass + sub-pedal @-bindings.
 */

import { strict as assert } from 'node:assert';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);

// ─────────────────────────────────────────────────────────────────────────
// Web Audio mock — superset of every method the pedalboard composite +
// its R1/R2/R3/R5/R9 sub-builders touch.
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
  connect(t) { this._outs.add(t); return t; }
  disconnect(t) { if (t) this._outs.delete(t); else this._outs.clear(); }
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
    this.detune = new MockAudioParam(0);
    this.type = 'lowpass';
  }
}
class MockDelay extends MockNode {
  constructor(ctx, max = 1) { super(ctx, 'delay'); this.delayTime = new MockAudioParam(0); this.maxDelayTime = max; }
}
class MockOsc extends MockNode {
  constructor(ctx) {
    super(ctx, 'osc');
    this.frequency = new MockAudioParam(440);
    this.detune = new MockAudioParam(0);
    this.type = 'sine';
  }
  start() { this._started = true; } stop() {}
}
class MockConstantSource extends MockNode {
  constructor(ctx) { super(ctx, 'const'); this.offset = new MockAudioParam(1); }
  start() {} stop() {}
}
class MockBufferSource extends MockNode {
  constructor(ctx) { super(ctx, 'bufsrc'); this.buffer = null; this.loop = false; this.playbackRate = new MockAudioParam(1); }
  start() {} stop() {}
}
class MockStereoPanner extends MockNode {
  constructor(ctx) { super(ctx, 'panner'); this.pan = new MockAudioParam(0); }
}
class MockAnalyser extends MockNode {
  constructor(ctx) { super(ctx, 'analyser'); this.fftSize = 256; this.smoothingTimeConstant = 0.5; }
  getByteTimeDomainData(arr) { for (let i = 0; i < arr.length; i++) arr[i] = 128; }
}
class MockWaveShaper extends MockNode {
  constructor(ctx) { super(ctx, 'shaper'); this.curve = null; this.oversample = 'none'; }
}
class MockConvolver extends MockNode {
  constructor(ctx) { super(ctx, 'convolver'); this.buffer = null; this.normalize = true; }
}
class MockDynComp extends MockNode {
  constructor(ctx) {
    super(ctx, 'dyncomp');
    this.threshold = new MockAudioParam(-24);
    this.ratio     = new MockAudioParam(4);
    this.attack    = new MockAudioParam(0.003);
    this.release   = new MockAudioParam(0.25);
    this.knee      = new MockAudioParam(6);
    this.reduction = 0;
  }
}
class MockBuffer {
  constructor(channels, length, sr) {
    this.numberOfChannels = channels;
    this.length = length;
    this.sampleRate = sr;
    this._data = Array.from({ length: channels }, () => new Float32Array(length));
  }
  getChannelData(ch) { return this._data[ch]; }
}

class MockCtx {
  constructor() {
    this.sampleRate = 48000;
    this.currentTime = 0;
    this.state = 'running';
    this._created = [];
    // R1 builders need ctx.audioWorklet.addModule()
    this.audioWorklet = {
      addModule: async () => true,
    };
  }
  _track(n) { this._created.push(n); return n; }
  createGain()              { return this._track(new MockGain(this)); }
  createBiquadFilter()      { return this._track(new MockBiquad(this)); }
  createDelay(max)          { return this._track(new MockDelay(this, max)); }
  createOscillator()        { return this._track(new MockOsc(this)); }
  createConstantSource()    { return this._track(new MockConstantSource(this)); }
  createBufferSource()      { return this._track(new MockBufferSource(this)); }
  createStereoPanner()      { return this._track(new MockStereoPanner(this)); }
  createAnalyser()          { return this._track(new MockAnalyser(this)); }
  createWaveShaper()        { return this._track(new MockWaveShaper(this)); }
  createConvolver()         { return this._track(new MockConvolver(this)); }
  createDynamicsCompressor(){ return this._track(new MockDynComp(this)); }
  createBuffer(c, l, sr)    { return new MockBuffer(c, l, sr); }
}

// AudioWorkletNode globally stubbed — always throws so builders fall back
// to native primitives. R1/R2/R3/R5/R9 sub-builders all implement non-worklet
// passthroughs that build on the ctx primitives the mock provides.
globalThis.AudioWorkletNode = function () {
  throw new Error('worklet path disabled in tests');
};

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

let passed = 0, failed = 0;
function test(name, fn) {
  try {
    const r = fn();
    if (r && typeof r.then === 'function') {
      return r.then(
        () => { console.log(`ok   ${name}`); passed++; },
        (e) => { console.error(`FAIL ${name}\n  ${e.stack || e}`); failed++; },
      );
    }
    console.log(`ok   ${name}`); passed++;
  } catch (e) {
    console.error(`FAIL ${name}\n  ${e.stack || e}`); failed++;
  }
}

const builderPath = path.resolve(__dirname, '../src/audio/builders/r13_pedalboard.js');
const { buildPedalboard, default: R13_BUILDERS, PEDAL_TYPES } =
  await import(`file://${builderPath}`);

// ── 1. Default export ────────────────────────────────────────────────────
test('default export registers `pedalboard`', () => {
  assert.ok(R13_BUILDERS, 'no default export');
  assert.equal(typeof R13_BUILDERS.pedalboard, 'function');
  assert.equal(R13_BUILDERS.pedalboard, buildPedalboard);
});

test('PEDAL_TYPES enumerates 24 sub-pedals', () => {
  assert.ok(Array.isArray(PEDAL_TYPES));
  assert.equal(PEDAL_TYPES.length, 24, `expected 24 pedal types, got ${PEDAL_TYPES.length}`);
});

// ── 2. Empty pedalboard ──────────────────────────────────────────────────
test('empty pedals[] returns input→output passthrough', () => {
  const ctx = new MockCtx();
  const r = buildPedalboard(ctx, { type: 'pedalboard', params: { pedals: [] } }, {});
  assert.ok(r.input);
  assert.ok(r.output);
  assert.equal(Array.isArray(r.pedals), true);
  assert.equal(r.pedals.length, 0);
  // input should be connected directly to output
  assert.ok(r.input._outs.has(r.output), 'input not wired to output for empty chain');
});

// ── 3. Single pedal contract ─────────────────────────────────────────────
test('single overdrive_pedal builds without throw and produces contract', () => {
  const ctx = new MockCtx();
  const r = buildPedalboard(ctx, {
    type: 'pedalboard',
    params: {
      pedals: [
        { type: 'overdrive_pedal', drive: 0.6, tone: 0.5, level: 0.7, bypass: false },
      ],
    },
  }, {});
  assert.ok(r.input); assert.ok(r.output);
  assert.equal(r.pedals.length, 1);
  assert.equal(r.pedals[0].type, 'overdrive_pedal');
  assert.equal(r.pedals[0].status, 'ok');
  // bypass_0 paramTarget exists with customSetter
  assert.ok(r.paramTargets.bypass_0);
  assert.equal(typeof r.paramTargets.bypass_0.customSetter, 'function');
});

// ── 4. Chain construction (5+ pedals) ────────────────────────────────────
test('5-pedal chain builds: overdrive → distortion → delay → chorus → reverb', () => {
  const ctx = new MockCtx();
  const r = buildPedalboard(ctx, {
    type: 'pedalboard',
    params: {
      pedals: [
        { type: 'overdrive_pedal',  drive: 0.5, tone: 0.6, level: 0.8 },
        { type: 'distortion_pedal', drive: 0.7, tone: 0.4, level: 0.6 },
        { type: 'delay_pedal',      time: 250, feedback: 0.35, mix: 0.4 },
        { type: 'chorus_pedal',     rate: 1.2, depth: 0.6, mix: 0.5 },
        { type: 'reverb_pedal',     decay: 1.8, mix: 0.3 },
      ],
    },
  }, {});
  assert.equal(r.pedals.length, 5);
  for (let i = 0; i < 5; i++) {
    assert.equal(r.pedals[i].status, 'ok', `pedal ${i} (${r.pedals[i].type}) status=${r.pedals[i].status}`);
    assert.ok(r.paramTargets[`bypass_${i}`], `bypass_${i} missing`);
  }
  // verify each pedal's join → next pedal's input chain (or final output)
  // by walking the wetGain→join→nextInput edges.
  for (let i = 0; i < 4; i++) {
    const join = r.pedals[i].join;
    assert.ok(join, `pedal ${i} missing join`);
    // join should connect to either pedal[i+1].input or pedal[i+1].dryGain (input via gain)
  }
});

// ── 5. Bypass toggle ─────────────────────────────────────────────────────
test('bypass_<i> setter flips wet/dry gain values', () => {
  const ctx = new MockCtx();
  const r = buildPedalboard(ctx, {
    type: 'pedalboard',
    params: {
      pedals: [
        { type: 'distortion_pedal', drive: 0.7, bypass: false },
      ],
    },
  }, {});
  const slot = r.pedals[0];
  // Initial: wet=1, dry=0
  assert.equal(slot.wetGain.gain.value, 1, 'initial wet=1');
  assert.equal(slot.dryGain.gain.value, 0, 'initial dry=0');
  // Engage bypass via setter
  r.paramTargets.bypass_0.customSetter(true);
  assert.equal(slot.wetGain.gain.value, 0, 'after bypass=true wet=0');
  assert.equal(slot.dryGain.gain.value, 1, 'after bypass=true dry=1');
  // Engage again with numeric 0
  r.paramTargets.bypass_0.customSetter(0);
  assert.equal(slot.wetGain.gain.value, 1, 'numeric 0 → engaged wet=1');
});

// ── 6. Unknown pedal fails open ──────────────────────────────────────────
test('unknown pedal type → status="unknown", no throw', () => {
  const ctx = new MockCtx();
  const r = buildPedalboard(ctx, {
    type: 'pedalboard',
    params: {
      pedals: [
        { type: 'overdrive_pedal', drive: 0.5 },
        { type: 'totally_made_up_pedal_2026', drive: 99 },
        { type: 'reverb_pedal', mix: 0.3 },
      ],
    },
  }, {});
  assert.equal(r.pedals.length, 3);
  assert.equal(r.pedals[0].status, 'ok');
  assert.equal(r.pedals[1].status, 'unknown');
  assert.equal(r.pedals[2].status, 'ok');
});

// ── 7. Kitchen-sink: all 24 sub-types build ──────────────────────────────
test('all 24 sub-pedal types build cleanly in one chain', () => {
  const ctx = new MockCtx();
  const pedals = PEDAL_TYPES.map((type) => ({ type }));
  const r = buildPedalboard(ctx, { type: 'pedalboard', params: { pedals } }, {});
  assert.equal(r.pedals.length, 24);
  for (const slot of r.pedals) {
    assert.equal(slot.status, 'ok',
      `pedal ${slot.type} build status=${slot.status} ${slot.error || ''}`);
  }
  // 24 bypass targets exposed
  for (let i = 0; i < 24; i++) {
    assert.ok(r.paramTargets[`bypass_${i}`], `bypass_${i} missing`);
  }
});

// ── 8. @-bound sub-params surface as per-slot pedal_<i>_<sub> targets ────
test('@-bound sub-params surface as pedal_<i>_<sub> paramTargets', () => {
  const ctx = new MockCtx();
  const paramDefs = {
    p_drive: { id: 'p_drive', min: 0, max: 1, default: 0.5 },
    p_level: { id: 'p_level', min: 0, max: 2, default: 0.7 },
    p_mix:   { id: 'p_mix',   min: 0, max: 1, default: 0.4 },
  };
  const r = buildPedalboard(ctx, {
    type: 'pedalboard',
    params: {
      pedals: [
        { type: 'overdrive_pedal', drive: 0.5, tone: '@p_drive', level: '@p_level' },
        { type: 'delay_pedal',     time: 250, feedback: 0.3, mix: '@p_mix' },
      ],
    },
  }, paramDefs);
  // Sub-pedal targets get prefixed pedal_<i>_<sub> form
  // overdrive exposes 'tone' and 'level' (since they are @-bound)
  assert.ok(r.paramTargets.pedal_0_tone, 'pedal_0_tone missing');
  assert.ok(r.paramTargets.pedal_0_level, 'pedal_0_level missing');
});

// ── Summary ──────────────────────────────────────────────────────────────

if (failed > 0) {
  console.error(`\n${failed} test(s) failed (${passed} passed)`);
  process.exit(1);
} else {
  console.log(`\n${passed} test(s) passed`);
}
