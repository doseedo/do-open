/**
 * tests/r13_multipressor.test.js — unit tests for the R13 Multipressor builder.
 *
 * The builder offers a worklet-preferred path with a BiquadFilter +
 * DynamicsCompressor fallback. Since AudioWorkletNode is unavailable in node
 * (and won't be registered without a real AudioContext anyway), these tests
 * exercise the FALLBACK path only — which is exactly what ships in the
 * absence of `_ensurePhase1Worklets` having loaded the worklet module.
 *
 * Mock model:
 *   - MockOfflineAudioContext implements just enough Web Audio surface area
 *     to build the multipressor graph and run startRendering() per-sample.
 *   - BiquadFilter mock implements RBJ Butterworth (q=0.7071) for lowpass
 *     and highpass — that's exactly what the LR4 split needs.
 *   - DynamicsCompressor mock implements the standard one-pole envelope
 *     follower + soft-knee gain function (lifted from PluginAdapter.test.js).
 *   - DelayNode mock is a fixed-length ring buffer (lookahead).
 *
 * Tests:
 *   1. Builder smoke      — exports + node-builder registration shape.
 *   2. Crossover split    — sine at 60/500/6k passes the right band(s) and
 *                           attenuates the wrong ones (per-band probe via
 *                           splitTap RMS).
 *   3. Per-band compress  — loud band-1 sine, threshold low + ratio 20:1 →
 *                           output RMS drops vs. bypass-all baseline.
 *
 * Run:  node tests/r13_multipressor.test.js
 *       (exits non-zero on any failure)
 */

import { strict as assert } from 'node:assert';

// ─────────────────────────────────────────────────────────────────────────
// Web Audio mock
// ─────────────────────────────────────────────────────────────────────────

class MockAudioParam {
  constructor(value = 0) { this.value = value; }
  setValueAtTime(v) { this.value = v; }
  setTargetAtTime(v) { this.value = v; }
  cancelScheduledValues() {}
  linearRampToValueAtTime(v) { this.value = v; }
}

let __nodeId = 0;
class MockAudioNode {
  constructor(ctx, kind) {
    this.id = ++__nodeId;
    this.ctx = ctx;
    this.kind = kind;
    this._outs = new Set();
    this._ins  = new Set();
  }
  connect(target) {
    if (!target) return target;
    this._outs.add(target);
    if (target instanceof MockAudioNode) target._ins.add(this);
    return target;
  }
  disconnect(target) {
    if (target) {
      this._outs.delete(target);
      if (target instanceof MockAudioNode) target._ins.delete(this);
    } else {
      for (const t of this._outs) {
        if (t instanceof MockAudioNode) t._ins.delete(this);
      }
      this._outs.clear();
    }
  }
}

class MockGainNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'gain'); this.gain = new MockAudioParam(1); }
}

// Biquad — RBJ formulas, transposed direct-form II for one stage. We
// provide just lowpass / highpass / allpass / notch / bandpass and accept
// the default q = 1/sqrt(2) for the LR-4 case.
class MockBiquadFilter extends MockAudioNode {
  constructor(ctx) {
    super(ctx, 'biquad');
    this.frequency = new MockAudioParam(1000);
    this.Q         = new MockAudioParam(Math.SQRT1_2);
    this.gain      = new MockAudioParam(0);
    this.type      = 'lowpass';
    this._z1 = [0, 0]; // per-channel state (we render at most 2 ch)
    this._z2 = [0, 0];
    this._coefsCutoff = -1;
    this._coefsType   = '';
    this._coefsQ      = -1;
  }
  _computeCoefs(sr) {
    if (this.frequency.value === this._coefsCutoff
      && this.type === this._coefsType
      && this.Q.value === this._coefsQ) return;
    const f = Math.max(20, Math.min(sr * 0.45, this.frequency.value));
    const q = Math.max(0.01, this.Q.value);
    const w0 = 2 * Math.PI * f / sr;
    const cw = Math.cos(w0);
    const sw = Math.sin(w0);
    const alpha = sw / (2 * q);
    let b0, b1, b2, a0, a1, a2;
    switch (this.type) {
      case 'lowpass':
        b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = (1 - cw) / 2; break;
      case 'highpass':
        b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = (1 + cw) / 2; break;
      case 'bandpass':
        b0 = alpha; b1 = 0; b2 = -alpha; break;
      case 'allpass':
        b0 = 1 - alpha; b1 = -2 * cw; b2 = 1 + alpha; break;
      default:
        b0 = 1; b1 = 0; b2 = 0; break;
    }
    a0 = 1 + alpha; a1 = -2 * cw; a2 = 1 - alpha;
    this._b0 = b0 / a0; this._b1 = b1 / a0; this._b2 = b2 / a0;
    this._a1 = a1 / a0; this._a2 = a2 / a0;
    this._coefsCutoff = this.frequency.value;
    this._coefsType   = this.type;
    this._coefsQ      = this.Q.value;
  }
  step(ch, x, sr) {
    this._computeCoefs(sr);
    const y = this._b0 * x + this._z1[ch];
    this._z1[ch] = this._b1 * x - this._a1 * y + this._z2[ch];
    this._z2[ch] = this._b2 * x - this._a2 * y;
    return y;
  }
}

class MockDynamicsCompressorNode extends MockAudioNode {
  constructor(ctx) {
    super(ctx, 'compressor');
    this.threshold = new MockAudioParam(-24);
    this.ratio     = new MockAudioParam(4);
    this.attack    = new MockAudioParam(0.003);
    this.release   = new MockAudioParam(0.25);
    this.knee      = new MockAudioParam(6);
    this.reduction = 0;
    this._envDb = [-120, -120];
  }
  step(ch, x, sr) {
    const inDb = 20 * Math.log10(Math.max(1e-6, Math.abs(x)));
    const a = Math.max(0.0001, this.attack.value);
    const r = Math.max(0.001,  this.release.value);
    const aCoef = Math.exp(-1 / (a * sr));
    const rCoef = Math.exp(-1 / (r * sr));
    let env = this._envDb[ch];
    if (inDb > env) env = aCoef * env + (1 - aCoef) * inDb;
    else            env = rCoef * env + (1 - rCoef) * inDb;
    this._envDb[ch] = env;
    const t = this.threshold.value;
    const k = Math.max(0, this.knee.value);
    const r_ratio = Math.max(1, this.ratio.value);
    const overshoot = env - t;
    let gainDb = 0;
    if (overshoot >= k / 2) {
      gainDb = -overshoot * (1 - 1 / r_ratio);
    } else if (overshoot > -k / 2 && k > 0) {
      const xx = overshoot + k / 2;
      gainDb = -xx * xx * (1 - 1 / r_ratio) / (2 * k);
    }
    this.reduction = gainDb;
    return x * Math.pow(10, gainDb / 20);
  }
}

class MockDelayNode extends MockAudioNode {
  constructor(ctx, maxSec = 1.0) {
    super(ctx, 'delay');
    this.delayTime = new MockAudioParam(0);
    this._maxSec = maxSec;
    this._ring = [new Float32Array(Math.ceil(maxSec * (ctx.sampleRate || 48000)) + 4),
                  new Float32Array(Math.ceil(maxSec * (ctx.sampleRate || 48000)) + 4)];
    this._w = 0;
  }
  step(ch, x, sr) {
    const buf = this._ring[ch];
    const n = buf.length;
    const dSamp = Math.max(0, Math.min(n - 2, Math.floor(this.delayTime.value * sr)));
    let r = this._w - dSamp;
    if (r < 0) r += n;
    const out = buf[r];
    if (ch === 1 || this.ctx.numberOfChannels === 1) {
      // advance write pointer once per sample-frame, after both channels handled.
    }
    buf[this._w] = x;
    return out;
  }
  // Helper: bump write index after a stereo pair has been processed
  advance() {
    this._w = (this._w + 1) % this._ring[0].length;
  }
}

class MockBufferSourceNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'buffersrc'); this.buffer = null; this.loop = false; }
  start() {} stop() {}
}

class MockAudioBuffer {
  constructor(channels, length, sampleRate) {
    this.numberOfChannels = channels;
    this.length = length;
    this.sampleRate = sampleRate;
    this.duration = length / sampleRate;
    this._data = Array.from({ length: channels }, () => new Float32Array(length));
  }
  getChannelData(ch) { return this._data[ch]; }
}

class MockOfflineAudioContext {
  constructor(channels, length, sampleRate) {
    this.numberOfChannels = channels;
    this.length = length;
    this.sampleRate = sampleRate;
    this.currentTime = 0;
    this.state = 'running';
    this.destination = new MockGainNode(this);
    this.destination.kind = 'destination';
    this._nodes = new Set();
    this._sources = [];
    const wrap = (n) => { this._nodes.add(n); return n; };
    this.createGain               = () => wrap(new MockGainNode(this));
    this.createBiquadFilter       = () => wrap(new MockBiquadFilter(this));
    this.createDynamicsCompressor = () => wrap(new MockDynamicsCompressorNode(this));
    this.createDelay              = (max = 1) => wrap(new MockDelayNode(this, max));
    const _self = this;
    this.createBufferSource = () => {
      const n = wrap(new MockBufferSourceNode(_self));
      _self._sources.push(n);
      return n;
    };
    this.createBuffer = (ch, len, sr) => new MockAudioBuffer(ch, len, sr);
  }

  /**
   * Walk every source per-sample through the connected graph. The graph is
   * a DAG of mock nodes; per-sample we DFS to evaluate. To avoid combinatorial
   * blowup we evaluate each node lazily once per sample (memoize by id).
   */
  async startRendering() {
    const out = new MockAudioBuffer(this.numberOfChannels, this.length, this.sampleRate);
    const sr = this.sampleRate;
    const nodeOrder = this._topoOrder();
    const valByNode = new Map(); // node.id → [L, R]

    for (let n = 0; n < this.length; n++) {
      // 1. Source contributions for sample n.
      valByNode.clear();
      for (const node of nodeOrder) {
        let inL = 0, inR = 0;
        for (const inNode of node._ins) {
          const v = valByNode.get(inNode.id);
          if (v) { inL += v[0]; inR += v[1]; }
        }
        let outL = inL, outR = inR;
        if (node instanceof MockBufferSourceNode) {
          if (node.buffer && n < node.buffer.length) {
            outL = node.buffer.getChannelData(0)[n] || 0;
            outR = node.buffer.numberOfChannels > 1
              ? (node.buffer.getChannelData(1)[n] || 0)
              : outL;
          } else { outL = 0; outR = 0; }
        } else if (node instanceof MockGainNode) {
          outL = inL * node.gain.value;
          outR = inR * node.gain.value;
        } else if (node instanceof MockBiquadFilter) {
          outL = node.step(0, inL, sr);
          outR = node.step(1, inR, sr);
        } else if (node instanceof MockDynamicsCompressorNode) {
          outL = node.step(0, inL, sr);
          outR = node.step(1, inR, sr);
        } else if (node instanceof MockDelayNode) {
          outL = node.step(0, inL, sr);
          outR = node.step(1, inR, sr);
          node.advance();
        }
        valByNode.set(node.id, [outL, outR]);
      }
      const destV = valByNode.get(this.destination.id);
      if (destV) {
        out.getChannelData(0)[n] += destV[0];
        if (out.numberOfChannels > 1) out.getChannelData(1)[n] += destV[1];
      }
    }
    return out;
  }

  _topoOrder() {
    const order = [];
    const visited = new Set();
    const visit = (node) => {
      if (visited.has(node.id)) return;
      visited.add(node.id);
      for (const inNode of node._ins) visit(inNode);
      order.push(node);
    };
    for (const n of this._nodes) visit(n);
    visit(this.destination);
    return order;
  }
}

// Patch globals so the builder (which reaches for AudioWorkletNode) doesn't
// crash; the safe-worklet helper inside r13_multipressor catches the error.
globalThis.AudioWorkletNode = class { constructor() { throw new Error('not registered'); } };
globalThis.AudioContext = MockOfflineAudioContext;
globalThis.OfflineAudioContext = MockOfflineAudioContext;

// ─────────────────────────────────────────────────────────────────────────
// Test harness
// ─────────────────────────────────────────────────────────────────────────

let passed = 0, failed = 0;
const failures = [];
async function test(name, fn) {
  try {
    await fn();
    passed++;
    process.stdout.write(`  ok  ${name}\n`);
  } catch (err) {
    failed++;
    failures.push({ name, err });
    process.stdout.write(`  FAIL ${name}\n    ${err.message}\n`);
  }
}

function rms(arr) {
  let s = 0;
  for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
  return Math.sqrt(s / arr.length);
}

function makeSineBuffer(ctx, freqHz, durationSec, amplitude = 0.5) {
  const len = Math.floor(ctx.sampleRate * durationSec);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = amplitude * Math.sin(2 * Math.PI * freqHz * i / ctx.sampleRate);
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

const SR = 48000;

const builderMod = await import('../src/audio/builders/r13_multipressor.js');
const { buildMultipressor } = builderMod;
const R13_DEFAULT = builderMod.default;

await test('builder smoke — exports buildMultipressor + default registry', () => {
  assert.equal(typeof buildMultipressor, 'function');
  assert.ok(R13_DEFAULT && typeof R13_DEFAULT === 'object');
  assert.equal(typeof R13_DEFAULT.multipressor, 'function');
  assert.equal(R13_DEFAULT.multipressor, buildMultipressor);
});

await test('builder smoke — returns {input, output, paramTargets} of expected shape', () => {
  const ctx = new MockOfflineAudioContext(2, SR * 0.05, SR);
  const built = buildMultipressor(ctx, { params: {} }, {});
  assert.ok(built.input,  'should expose input node');
  assert.ok(built.output, 'should expose output node');
  assert.ok(built.paramTargets && typeof built.paramTargets === 'object',
    'should expose paramTargets');
  // 4 bands (compressor + makeup + bypassWet + bypassDry + splitTap)
  assert.ok(Array.isArray(built.multipressorBands) && built.multipressorBands.length === 4,
    `expected 4 bands, got ${built.multipressorBands && built.multipressorBands.length}`);
  assert.ok(built.multipressorCrossovers && built.multipressorCrossovers.lp.length === 3,
    'should expose 3 LR-LP crossovers');
  assert.ok(built.multipressorCrossovers.hp.length === 3,
    'should expose 3 LR-HP crossovers');
});

await test('builder — @-bound crossover_1 installs a customSetter that drives both LR4 biquads', () => {
  const ctx = new MockOfflineAudioContext(2, SR * 0.05, SR);
  const built = buildMultipressor(
    ctx,
    { params: { crossover_1: '@xo1' } },
    { xo1: { id: 'xo1', min: 50, max: 500, default: 120 } }
  );
  const tgt = built.paramTargets['xo1'];
  assert.ok(tgt && typeof tgt.customSetter === 'function',
    'crossover_1 should bind a customSetter');
  // Drive it to a new value and verify both biquads in the LP[0] cascade moved
  tgt.customSetter(250);
  assert.equal(built.multipressorCrossovers.lp[0].frequencyParams[0].value, 250);
  assert.equal(built.multipressorCrossovers.lp[0].frequencyParams[1].value, 250);
  assert.equal(built.multipressorCrossovers.hp[0].frequencyParams[0].value, 250);
});

// ── Crossover split ──────────────────────────────────────────────────────

/**
 * Helper: feed a sine at `freqHz` through the multipressor configured to
 * pass-through (very high thresholds — no compression), and probe each
 * band's splitTap RMS. With LR4 crossovers, energy should localise to the
 * one or two adjacent bands containing the source frequency.
 */
async function renderSinePerBand(freqHz, params = {}) {
  const ctx = new MockOfflineAudioContext(1, SR * 0.5, SR);
  const built = buildMultipressor(ctx, {
    params: {
      // High thresholds → no band compresses (we're testing the filter split,
      // not the compression).
      band1_threshold_db: 0,  band2_threshold_db: 0,
      band3_threshold_db: 0,  band4_threshold_db: 0,
      band1_ratio: 1, band2_ratio: 1, band3_ratio: 1, band4_ratio: 1,
      ...params,
    },
  }, {});

  // Source: long sine
  const src = ctx.createBufferSource();
  src.buffer = makeSineBuffer(ctx, freqHz, 0.5, 0.5);
  src.connect(built.input);

  // Tap each band's splitTap → independent destination-sums via per-band gain.
  // We simulate "destination" as the multipressor output, but to read per-band
  // values we capture the post-filter (pre-compressor) splitTap state via a
  // dedicated probe gain that we wire to the destination.
  const probeGains = [];
  for (let i = 0; i < 4; i++) {
    const probe = ctx.createGain();
    built.multipressorBands[i].splitTap.connect(probe);
    probeGains.push(probe);
  }
  // Sum probes into destination, but only ONE at a time so each render
  // returns a single band. Call this 4 times.
  const bandRMS = [];
  for (let i = 0; i < 4; i++) {
    // Reset connections so this render only sees probe[i] → destination
    for (let j = 0; j < 4; j++) {
      probeGains[j].disconnect();
    }
    probeGains[i].connect(ctx.destination);

    // Re-render a fresh ctx — the existing ctx has accumulated state.
    const ctx2 = new MockOfflineAudioContext(1, SR * 0.3, SR);
    const built2 = buildMultipressor(ctx2, {
      params: {
        band1_threshold_db: 0, band2_threshold_db: 0,
        band3_threshold_db: 0, band4_threshold_db: 0,
        band1_ratio: 1, band2_ratio: 1, band3_ratio: 1, band4_ratio: 1,
        ...params,
      },
    }, {});
    const src2 = ctx2.createBufferSource();
    src2.buffer = makeSineBuffer(ctx2, freqHz, 0.3, 0.5);
    src2.connect(built2.input);
    built2.multipressorBands[i].splitTap.connect(ctx2.destination);
    const rendered = await ctx2.startRendering();
    // Discard first 0.05s for filter settling
    const settle = Math.floor(SR * 0.05);
    bandRMS.push(rms(rendered.getChannelData(0).slice(settle)));
  }
  return bandRMS;
}

await test('crossover split — 60 Hz lives in band 0 (LP@120 Hz)', async () => {
  const r = await renderSinePerBand(60);
  // Band 0 = lowpass @ 120 Hz, 60 Hz is well in-band → high RMS.
  // Band 3 = highpass @ 4 kHz, 60 Hz is way below → near zero.
  // Source amplitude 0.5 → raw sine RMS ≈ 0.354. Expect band 0 ≥ 0.7×, others much less.
  const reference = 0.5 / Math.SQRT2;
  assert.ok(r[0] > reference * 0.7,
    `band 0 (LP) should pass 60 Hz: got ${r[0].toFixed(4)}, expect > ${(reference*0.7).toFixed(4)}`);
  assert.ok(r[3] < reference * 0.05,
    `band 3 (HP@4k) should reject 60 Hz: got ${r[3].toFixed(4)}, expect < ${(reference*0.05).toFixed(4)}`);
});

await test('crossover split — 6 kHz lives in band 3 (HP@4 kHz)', async () => {
  const r = await renderSinePerBand(6000);
  const reference = 0.5 / Math.SQRT2;
  assert.ok(r[3] > reference * 0.7,
    `band 3 (HP@4k) should pass 6 kHz: got ${r[3].toFixed(4)}, expect > ${(reference*0.7).toFixed(4)}`);
  assert.ok(r[0] < reference * 0.05,
    `band 0 (LP@120) should reject 6 kHz: got ${r[0].toFixed(4)}, expect < ${(reference*0.05).toFixed(4)}`);
});

await test('crossover split — 500 Hz lives in band 1 (BP 120..800)', async () => {
  const r = await renderSinePerBand(500);
  const reference = 0.5 / Math.SQRT2;
  // Band 1 = HP@120 → LP@800; 500 Hz is well in-band.
  assert.ok(r[1] > reference * 0.6,
    `band 1 should pass 500 Hz: got ${r[1].toFixed(4)}, expect > ${(reference*0.6).toFixed(4)}`);
  // Bands 0 and 3 should both be far away.
  assert.ok(r[0] < reference * 0.2,
    `band 0 (LP@120) should attenuate 500 Hz: got ${r[0].toFixed(4)}`);
  assert.ok(r[3] < reference * 0.05,
    `band 3 (HP@4k) should reject 500 Hz: got ${r[3].toFixed(4)}`);
});

// ── Per-band compression ────────────────────────────────────────────────

/**
 * Render a 60 Hz sine through the full multipressor. Compare two configs:
 *   A. All bands inactive (very low threshold but ratio=1 = unity gain).
 *   B. Band 1 (the low band) heavily compressed (threshold=-40, ratio=20).
 * The output RMS in (B) should be measurably lower than (A) because the
 * 60 Hz signal lives in band 0 and gets clobbered.
 */
async function renderFullChain(bandParams) {
  const ctx = new MockOfflineAudioContext(1, SR * 0.6, SR);
  const built = buildMultipressor(ctx, { params: bandParams }, {});
  const src = ctx.createBufferSource();
  src.buffer = makeSineBuffer(ctx, 60, 0.6, 0.5);
  src.connect(built.input);
  built.output.connect(ctx.destination);
  const rendered = await ctx.startRendering();
  // Skip first 0.1 s for compressor envelope to settle
  const settle = Math.floor(SR * 0.1);
  return rms(rendered.getChannelData(0).slice(settle));
}

await test('per-band compression — band 0 with low threshold + 20:1 ratio reduces output ≥ 6 dB', async () => {
  const baseline = await renderFullChain({
    band1_threshold_db: 0,  band1_ratio: 1,
    band2_threshold_db: 0,  band2_ratio: 1,
    band3_threshold_db: 0,  band3_ratio: 1,
    band4_threshold_db: 0,  band4_ratio: 1,
    band1_attack_ms: 1, band1_release_ms: 50,
  });
  const compressed = await renderFullChain({
    band1_threshold_db: -40, band1_ratio: 20,
    band2_threshold_db: 0,   band2_ratio: 1,
    band3_threshold_db: 0,   band3_ratio: 1,
    band4_threshold_db: 0,   band4_ratio: 1,
    band1_attack_ms: 1, band1_release_ms: 50,
  });
  const dropDb = 20 * Math.log10(Math.max(1e-9, compressed) / Math.max(1e-9, baseline));
  process.stdout.write(`     baseline RMS=${baseline.toFixed(4)}, compressed RMS=${compressed.toFixed(4)}, drop=${dropDb.toFixed(2)} dB\n`);
  assert.ok(dropDb <= -6,
    `expected ≥ 6 dB drop with band 1 heavily compressed; got ${dropDb.toFixed(2)} dB`);
});

await test('per-band compression — band 4 (HF) compression does NOT affect 60 Hz signal', async () => {
  const baseline = await renderFullChain({
    band1_threshold_db: 0, band2_threshold_db: 0,
    band3_threshold_db: 0, band4_threshold_db: 0,
    band1_ratio: 1, band2_ratio: 1, band3_ratio: 1, band4_ratio: 1,
  });
  // Heavy HF compression — 60 Hz signal lives in band 0 so band 4 pulldown
  // shouldn't change the output significantly.
  const compressed = await renderFullChain({
    band1_threshold_db: 0,   band1_ratio: 1,
    band2_threshold_db: 0,   band2_ratio: 1,
    band3_threshold_db: 0,   band3_ratio: 1,
    band4_threshold_db: -50, band4_ratio: 20,
  });
  const dropDb = Math.abs(20 * Math.log10(Math.max(1e-9, compressed) / Math.max(1e-9, baseline)));
  process.stdout.write(`     band-4 only compressed: |Δ|=${dropDb.toFixed(2)} dB\n`);
  assert.ok(dropDb <= 1.5,
    `60 Hz output should be ~unchanged when only band 4 compresses; got |Δ|=${dropDb.toFixed(2)} dB`);
});

// ─────────────────────────────────────────────────────────────────────────
process.stdout.write(`\n${passed} passed, ${failed} failed\n`);
if (failed > 0) {
  for (const { name, err } of failures) {
    process.stderr.write(`\n--- ${name} ---\n${err.stack || err.message}\n`);
  }
  process.exit(1);
}
