/**
 * r13_space_designer.test.js — unit tests for the Space Designer builder.
 *
 * Strategy: ship our own minimal AudioContext mock that implements the
 * subset of Web Audio surface our builder uses (createGain, createBuffer,
 * createConvolver, createBiquadFilter, decodeAudioData). Tests verify:
 *
 *   1. shapeIR truncates the IR to `length` of the source
 *   2. shapeIR prepends `predelay` ms of silence
 *   3. shapeIR reverses the IR when `reverse=true`
 *   4. shapeIR fades in linearly over `attack_time`
 *   5. shapeIR fades out exponentially over `decay_time`
 *   6. buildConvolutionSD returns {input, output, paramTargets} shape
 *   7. customSetters fire on every shape param and update conv.buffer
 *   8. low_cut / high_cut bind to BiquadFilter AudioParams
 *
 * Runs under Jest/Vitest OR plain `node` via `runAll()` / direct import.
 *
 * Author: Agent R13
 */

import {
  buildConvolutionSD,
  makeDefaultSourceIR,
  shapeIR,
} from '../src/audio/builders/r13_space_designer.js';

// ── Minimal AudioContext mock ─────────────────────────────────────────────
// Implements just enough surface for the R13 builder.

class MockAudioParam {
  constructor(value = 0) {
    this.value = value;
    this._scheduled = [];
  }
  setValueAtTime(v, _t)        { this.value = v; this._scheduled.push(['s', v, _t]); return this; }
  setTargetAtTime(v, _t, _tau) { this.value = v; this._scheduled.push(['t', v, _t]); return this; }
  linearRampToValueAtTime(v, _t) { this.value = v; this._scheduled.push(['l', v, _t]); return this; }
}

class MockNode {
  constructor() { this._connections = []; }
  connect(target) { this._connections.push(target); return target; }
  disconnect()   { this._connections = []; }
}

class MockGainNode extends MockNode {
  constructor() { super(); this.gain = new MockAudioParam(1); }
}

class MockBiquadFilter extends MockNode {
  constructor() {
    super();
    this.type = 'lowpass';
    this.frequency = new MockAudioParam(350);
    this.Q = new MockAudioParam(1);
    this.detune = new MockAudioParam(0);
    this.gain = new MockAudioParam(0);
  }
}

class MockAudioBuffer {
  constructor(numberOfChannels, length, sampleRate) {
    this.numberOfChannels = numberOfChannels;
    this.length = length;
    this.sampleRate = sampleRate;
    this.duration = length / sampleRate;
    this._channels = [];
    for (let i = 0; i < numberOfChannels; i++) {
      this._channels.push(new Float32Array(length));
    }
  }
  getChannelData(ch) { return this._channels[ch]; }
  copyFromChannel(dest, ch, off = 0) {
    const src = this._channels[ch];
    const n = Math.min(dest.length, src.length - off);
    for (let i = 0; i < n; i++) dest[i] = src[off + i];
  }
}

class MockConvolverNode extends MockNode {
  constructor() {
    super();
    this.normalize = true;
    this._buffer = null;
  }
  set buffer(b) { this._buffer = b; }
  get buffer() { return this._buffer; }
}

class MockAudioContext {
  constructor(sampleRate = 48000) { this.sampleRate = sampleRate; this.currentTime = 0; }
  createGain()           { return new MockGainNode(); }
  createBiquadFilter()   { return new MockBiquadFilter(); }
  createConvolver()      { return new MockConvolverNode(); }
  createBuffer(ch, len, sr) { return new MockAudioBuffer(ch, len, sr || this.sampleRate); }
  createStereoPanner()   { const n = new MockNode(); n.pan = new MockAudioParam(0); return n; }
  createConstantSource() {
    const n = new MockNode();
    n.offset = new MockAudioParam(1);
    n.start = () => {};
    n.stop  = () => {};
    return n;
  }
  decodeAudioData(_arrayBuffer) {
    // Synthetic decode — returns a half-second 48k stereo buffer of noise.
    return Promise.resolve(new MockAudioBuffer(2, this.sampleRate / 2, this.sampleRate));
  }
}

// ── Assert + run helpers ──────────────────────────────────────────────────

function assert(cond, msg) {
  if (!cond) throw new Error('ASSERT: ' + msg);
}
function assertClose(actual, expected, tol, msg) {
  if (Math.abs(actual - expected) > tol) {
    throw new Error(`ASSERT close: ${msg} — got ${actual}, expected ${expected} ± ${tol}`);
  }
}

// ── The actual tests ─────────────────────────────────────────────────────

function test_shapeIR_length_truncates() {
  const ctx = new MockAudioContext(48000);
  const src = makeDefaultSourceIR(ctx, 1.0, 3.0); // 1 s, 48k samples
  const out = shapeIR(ctx, src, { length: 0.5, attack_time: 0, decay_time: 0, predelay: 0, reverse: false, density: 1.0 });
  // length=0.5 → output ≈ 0.5 * srcLen + 0 predelay = ~24000 samples
  assertClose(out.length, src.length * 0.5, 4, 'length=0.5 truncates to ~half');
}

function test_shapeIR_predelay_is_silent() {
  const ctx = new MockAudioContext(48000);
  const src = makeDefaultSourceIR(ctx, 0.5, 3.0);
  const predelayMs = 100;
  const expectedSilent = Math.floor((predelayMs / 1000) * 48000);
  const out = shapeIR(ctx, src, { length: 1.0, attack_time: 0, decay_time: 0, predelay: predelayMs, reverse: false, density: 1.0 });
  const data = out.getChannelData(0);
  // First `expectedSilent` samples must be exactly 0 (silence prepended).
  for (let i = 0; i < expectedSilent; i++) {
    assert(data[i] === 0, `predelay region must be silence at sample ${i}, got ${data[i]}`);
  }
  // Sample right after the predelay window should generally NOT be zero
  // (raw IR is noise so the first non-silent sample is non-zero w.p. → 1).
  // We don't require it strictly because Math.random() *could* return 0;
  // instead check that AT LEAST ONE sample in the next 100 is non-zero.
  let anyNonZero = false;
  for (let i = expectedSilent; i < expectedSilent + 100 && i < data.length; i++) {
    if (data[i] !== 0) { anyNonZero = true; break; }
  }
  assert(anyNonZero, 'IR content should follow the predelay silence');
}

function test_shapeIR_reverse_flips_energy() {
  const ctx = new MockAudioContext(48000);
  // Build a deterministic source: a single impulse at the FIRST sample only.
  const src = ctx.createBuffer(1, 1000, 48000);
  src.getChannelData(0)[0] = 1.0;
  // Forward: energy at the head
  const fwd = shapeIR(ctx, src, { length: 1.0, attack_time: 0, decay_time: 0, predelay: 0, reverse: false, density: 1.0 });
  const fwdHead = Math.abs(fwd.getChannelData(0)[0]);
  const fwdTail = Math.abs(fwd.getChannelData(0)[fwd.length - 1]);
  assert(fwdHead > fwdTail, `forward IR: head energy (${fwdHead}) should exceed tail (${fwdTail})`);
  // Reverse: energy now at the tail
  const rev = shapeIR(ctx, src, { length: 1.0, attack_time: 0, decay_time: 0, predelay: 0, reverse: true, density: 1.0 });
  const revHead = Math.abs(rev.getChannelData(0)[0]);
  const revTail = Math.abs(rev.getChannelData(0)[rev.length - 1]);
  assert(revTail > revHead, `reversed IR: tail energy (${revTail}) should exceed head (${revHead})`);
}

function test_shapeIR_attack_fades_in() {
  const ctx = new MockAudioContext(48000);
  // DC source so we can read the envelope directly off the output.
  const src = ctx.createBuffer(1, 4800, 48000);
  src.getChannelData(0).fill(1.0);
  const attackMs = 50; // 2400 samples at 48k
  const out = shapeIR(ctx, src, { length: 1.0, attack_time: attackMs, decay_time: 0, predelay: 0, reverse: false, density: 1.0 });
  const data = out.getChannelData(0);
  // First sample is 0 (linear ramp starts at i/attackSamps with i=0).
  assertClose(data[0], 0.0, 1e-9, 'attack envelope starts at 0');
  // Halfway through the attack window, envelope ≈ 0.5
  const attackSamps = (attackMs / 1000) * 48000;
  const mid = Math.floor(attackSamps / 2);
  assertClose(data[mid], 0.5, 0.05, 'attack envelope at midpoint should be ~0.5');
  // Just past the attack window, envelope should be ~1.0 (no decay set)
  const past = Math.floor(attackSamps) + 10;
  assertClose(data[past], 1.0, 0.05, 'attack envelope reaches ~1.0 after attack window');
}

function test_shapeIR_decay_decays() {
  const ctx = new MockAudioContext(48000);
  const src = ctx.createBuffer(1, 4800, 48000);
  src.getChannelData(0).fill(1.0);
  const decayMs = 50;
  const out = shapeIR(ctx, src, { length: 1.0, attack_time: 0, decay_time: decayMs, predelay: 0, reverse: false, density: 1.0 });
  const data = out.getChannelData(0);
  // First sample = e^0 = 1
  assertClose(data[0], 1.0, 1e-9, 'decay envelope starts at 1.0');
  // At the end of the decay window, envelope ≈ 1/1000 = -60 dB
  const decaySamps = Math.floor((decayMs / 1000) * 48000);
  const tailVal = data[decaySamps - 1];
  assert(tailVal < 0.01 && tailVal > 0, `decay tail should be ~-60 dB, got ${tailVal}`);
}

function test_buildConvolutionSD_smoke() {
  const ctx = new MockAudioContext(48000);
  const node = { type: 'convolution_sd', params: {} };
  const result = buildConvolutionSD(ctx, node, {});
  assert(result.input instanceof MockGainNode, 'returns a Gain input');
  assert(result.output instanceof MockGainNode, 'returns a Gain output');
  assert(typeof result.paramTargets === 'object', 'returns a paramTargets object');
  assert(typeof result.loadIR === 'function', 'returns loadIR');
  assert(result._convolver._buffer instanceof MockAudioBuffer, 'conv.buffer is set on construction (default IR)');
}

function test_param_targets_wired_for_shape_params() {
  const ctx = new MockAudioContext(48000);
  const paramDefs = {
    p_len:    { default: 1.0,  min: 0.0,  max: 1.0 },
    p_atk:    { default: 0,    min: 0,    max: 500 },
    p_dec:    { default: 0,    min: 0,    max: 5000 },
    p_pre:    { default: 0,    min: 0,    max: 500 },
    p_lc:     { default: 20,   min: 20,   max: 2000 },
    p_hc:     { default: 20000,min: 1000, max: 20000 },
    p_mix:    { default: 0.3,  min: 0,    max: 1 },
    p_rev:    { default: 0,    min: 0,    max: 1 },
    p_den:    { default: 1.0,  min: 0.05, max: 4.0 },
  };
  const node = {
    type: 'convolution_sd',
    params: {
      length:      '@p_len',
      attack_time: '@p_atk',
      decay_time:  '@p_dec',
      predelay:    '@p_pre',
      low_cut:     '@p_lc',
      high_cut:    '@p_hc',
      mix:         '@p_mix',
      reverse:     '@p_rev',
      density:     '@p_den',
    },
  };
  const result = buildConvolutionSD(ctx, node, paramDefs);
  // Each modulated param should produce a target keyed by the bare paramId.
  for (const id of Object.keys(paramDefs)) {
    assert(result.paramTargets[id], `paramTargets.${id} should be wired`);
  }
  // low_cut / high_cut should bind to AudioParams (not customSetter)
  assert(result.paramTargets.p_lc.audioParam instanceof MockAudioParam,
         'low_cut binds to a BiquadFilter.frequency AudioParam');
  assert(result.paramTargets.p_hc.audioParam instanceof MockAudioParam,
         'high_cut binds to a BiquadFilter.frequency AudioParam');
  // Shape params should be customSetter
  for (const id of ['p_len', 'p_atk', 'p_dec', 'p_pre', 'p_rev', 'p_den']) {
    assert(typeof result.paramTargets[id].customSetter === 'function',
           `${id} should expose a customSetter (shape param)`);
  }
  // mix should be customSetter (drives the dry/wet pair)
  assert(typeof result.paramTargets.p_mix.customSetter === 'function',
         'mix should be a customSetter (dry/wet pair)');
}

function test_knob_drag_mutates_ir_buffer() {
  // Fire the `length` customSetter and verify the convolver's buffer length
  // actually changes — the IR is mutated in real-time as Logic claims.
  const ctx = new MockAudioContext(48000);
  const node = {
    type: 'convolution_sd',
    params: { length: '@p_len' },
  };
  const result = buildConvolutionSD(ctx, node, { p_len: { default: 1.0, min: 0, max: 1 } });
  const initialBuffer = result._convolver._buffer;
  const initialLen = initialBuffer.length;

  // Drag length down to 0.25 — should produce a ~4× shorter shaped IR.
  result.paramTargets.p_len.customSetter(0.25);
  // The rebuild is scheduled on a microtask. Force-fire by also calling
  // the synchronous rebuild that the builder exposes for tests.
  result._rebuildNow();
  const afterBuffer = result._convolver._buffer;
  assert(afterBuffer !== initialBuffer || afterBuffer.length !== initialLen,
         'conv.buffer must change when `length` knob is dragged');
  // After length=0.25 the shaped buffer is ~25% of source. Source default is
  // 2.5 s @ 48k = 120k samples; shaped should be ~30k.
  assert(afterBuffer.length < initialLen,
         `after length=0.25, buffer should be shorter than initial (got ${afterBuffer.length} vs ${initialLen})`);
}

function test_predelay_lengthens_buffer() {
  const ctx = new MockAudioContext(48000);
  const node = { type: 'convolution_sd', params: { predelay: '@p_pre' } };
  const result = buildConvolutionSD(ctx, node, { p_pre: { default: 0, min: 0, max: 500 } });
  const initialLen = result._convolver._buffer.length;
  result.paramTargets.p_pre.customSetter(100); // 100 ms predelay
  result._rebuildNow();
  const after = result._convolver._buffer.length;
  // 100 ms @ 48k = 4800 extra samples
  assertClose(after - initialLen, 4800, 4, 'predelay=100ms adds ~4800 samples to the IR length');
}

// ── Test runner ───────────────────────────────────────────────────────────

const TESTS = [
  ['shapeIR length truncates',          test_shapeIR_length_truncates],
  ['shapeIR predelay is silent',        test_shapeIR_predelay_is_silent],
  ['shapeIR reverse flips energy',      test_shapeIR_reverse_flips_energy],
  ['shapeIR attack fades in',           test_shapeIR_attack_fades_in],
  ['shapeIR decay decays',              test_shapeIR_decay_decays],
  ['buildConvolutionSD smoke',          test_buildConvolutionSD_smoke],
  ['param targets wired for shape',     test_param_targets_wired_for_shape_params],
  ['knob drag mutates IR buffer',       test_knob_drag_mutates_ir_buffer],
  ['predelay lengthens buffer',         test_predelay_lengthens_buffer],
];

export async function runAll() {
  let passed = 0, failed = 0;
  const failures = [];
  for (const [name, fn] of TESTS) {
    try {
      await fn();
      // eslint-disable-next-line no-console
      console.log(`  ok  ${name}`);
      passed++;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.log(`  FAIL ${name} — ${err.message}`);
      failures.push({ name, message: err.message });
      failed++;
    }
  }
  // eslint-disable-next-line no-console
  console.log(`\n${passed}/${TESTS.length} passed${failed ? `, ${failed} failed` : ''}`);
  return { passed, failed, total: TESTS.length, failures };
}

// ── Jest / Vitest harness ─────────────────────────────────────────────────

if (typeof describe === 'function' && typeof it === 'function') {
  describe('R13 Space Designer (convolution_sd)', () => {
    for (const [name, fn] of TESTS) {
      // eslint-disable-next-line no-undef
      it(name, async () => { await fn(); });
    }
  });
}

// ── Auto-run when invoked directly via `node` ────────────────────────────
// import.meta.url === argv-resolved path → CLI invocation
const isDirect = (() => {
  try {
    if (typeof process === 'undefined' || !process.argv || !import.meta || !import.meta.url) return false;
    const argv1 = process.argv[1];
    if (!argv1) return false;
    return import.meta.url.endsWith(argv1.replace(/^file:\/\//, ''))
        || import.meta.url === `file://${argv1}`;
  } catch (_e) { return false; }
})();
if (isDirect) {
  runAll().then(({ failed }) => {
    if (failed > 0) process.exit(1);
  });
}
