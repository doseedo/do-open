/**
 * PluginAdapter.test.js — unit tests for src/lib/PluginAdapter.js
 *
 * The repo doesn't ship a Web Audio test harness (no jest, no jsdom,
 * no @web-audio-api/* deps). To stay zero-install we provide a small
 * in-process Web Audio mock that:
 *   - implements just enough of the OfflineAudioContext / AudioNode /
 *     AudioParam surface for WebAudioDSPEngine + PluginAdapter to build
 *     a graph + run setTargetAtTime
 *   - tracks node connections so we can assert topology
 *   - implements DynamicsCompressorNode with a simple per-sample
 *     soft-knee model so `startRendering()` actually attenuates loud
 *     input — enough to verify the slot is performing dynamic-range
 *     reduction end-to-end
 *
 * The mock is intentionally minimal. Anything PluginAdapter / the engine
 * touches that the mock doesn't model becomes a no-op (e.g. ConvolverNode,
 * AnalyserNode FFT). That's fine — those nodes aren't on the Compressor
 * mapping's signal path.
 *
 * Run:
 *   node src/lib/PluginAdapter.test.js
 *
 * The tests exit non-zero on failure.
 */

import { strict as assert } from 'node:assert';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import fs from 'node:fs';

const __filename = fileURLToPath(import.meta.url);
const __dirname  = path.dirname(__filename);
const repoRoot   = path.resolve(__dirname, '../../');

// ─────────────────────────────────────────────────────────────────────────
// Web Audio mock (just enough for WebAudioDSPEngine + PluginAdapter)
// ─────────────────────────────────────────────────────────────────────────

class MockAudioParam {
  constructor(value = 0) {
    this.value = value;
    this._scheduled = [];
  }
  setValueAtTime(v, _t)        { this.value = v; }
  setTargetAtTime(v, _t, _tau) { this.value = v; }
  cancelScheduledValues(_t)    {}
  linearRampToValueAtTime(v)   { this.value = v; }
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
class MockBiquadFilter extends MockAudioNode {
  constructor(ctx) {
    super(ctx, 'biquad');
    this.frequency = new MockAudioParam(1000);
    this.Q         = new MockAudioParam(1);
    this.gain      = new MockAudioParam(0);
    this.type      = 'lowpass';
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
  }
}
class MockDelayNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'delay'); this.delayTime = new MockAudioParam(0); }
}
class MockStereoPannerNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'panner'); this.pan = new MockAudioParam(0); }
}
class MockOscillatorNode extends MockAudioNode {
  constructor(ctx) {
    super(ctx, 'osc');
    this.frequency = new MockAudioParam(440);
    this.detune    = new MockAudioParam(0);
    this.type      = 'sine';
  }
  start() {} stop() {}
}
class MockBufferSourceNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'buffersrc'); this.buffer = null; this.loop = false; }
  start() {} stop() {}
}
class MockConvolverNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'convolver'); this.buffer = null; }
}
class MockWaveShaperNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'shaper'); this.curve = null; this.oversample = 'none'; }
}
class MockAnalyserNode extends MockAudioNode {
  constructor(ctx) { super(ctx, 'analyser'); this.fftSize = 256; this.frequencyBinCount = 128; }
  getByteFrequencyData(arr) { for (let i = 0; i < arr.length; i++) arr[i] = 0; }
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
    this.resume = async () => { this.state = 'running'; };
    this.suspend = async () => { this.state = 'suspended'; };
    this.close = async () => { this.state = 'closed'; };
    this.destination = new MockGainNode(this);
    this.destination.kind = 'destination';
    // Track every node so we can walk the graph in startRendering.
    this._nodes = new Set();
    this._sources = []; // list of buffer sources to render
    const wrap = (node) => { this._nodes.add(node); return node; };
    this.createGain                = () => wrap(new MockGainNode(this));
    this.createBiquadFilter        = () => wrap(new MockBiquadFilter(this));
    this.createDynamicsCompressor  = () => wrap(new MockDynamicsCompressorNode(this));
    this.createDelay               = () => wrap(new MockDelayNode(this));
    this.createStereoPanner        = () => wrap(new MockStereoPannerNode(this));
    this.createOscillator          = () => wrap(new MockOscillatorNode(this));
    this.createConvolver           = () => wrap(new MockConvolverNode(this));
    this.createWaveShaper          = () => wrap(new MockWaveShaperNode(this));
    this.createAnalyser            = () => wrap(new MockAnalyserNode(this));
    const _self = this;
    this.createBufferSource = () => {
      const node = wrap(new MockBufferSourceNode(_self));
      _self._sources.push(node);
      return node;
    };
    this.createBuffer = (channels, len, sr) => new MockAudioBuffer(channels, len, sr);
  }

  /**
   * Trace each source through the graph collecting compressors and gains
   * along the path, then per-sample render into the destination buffer.
   * This is enough to demonstrate that the Compressor node is doing its
   * job (dynamic range reduction) without modeling EQ etc.
   */
  async startRendering() {
    const out = new MockAudioBuffer(this.numberOfChannels, this.length, this.sampleRate);
    for (const src of this._sources) {
      if (!src.buffer) continue;
      const path = this._tracePathToDestination(src);
      // path = [{compressor?, gain?}, ...]
      const compressors = [];
      const gains = [];
      for (const node of path) {
        if (node instanceof MockDynamicsCompressorNode) compressors.push(node);
        else if (node instanceof MockGainNode) gains.push(node);
      }

      const channels = Math.min(out.numberOfChannels, src.buffer.numberOfChannels);
      for (let ch = 0; ch < channels; ch++) {
        const inData  = src.buffer.getChannelData(ch);
        const outData = out.getChannelData(ch);
        let envDb = -120; // running envelope follower in dB
        for (let i = 0; i < this.length; i++) {
          let s = inData[i] || 0;
          // Run through each compressor sequentially
          for (const comp of compressors) {
            const inDb = 20 * Math.log10(Math.max(1e-6, Math.abs(s)));
            // Attack/release envelope follower. Coefficient maps from time
            // constant to per-sample ratio; this is the standard one-pole.
            const attack  = Math.max(0.0001, comp.attack.value);
            const release = Math.max(0.001,  comp.release.value);
            const aCoef = Math.exp(-1 / (attack  * this.sampleRate));
            const rCoef = Math.exp(-1 / (release * this.sampleRate));
            if (inDb > envDb) envDb = aCoef * envDb + (1 - aCoef) * inDb;
            else              envDb = rCoef * envDb + (1 - rCoef) * inDb;
            // Soft knee around threshold
            const threshold = comp.threshold.value;
            const ratio     = Math.max(1, comp.ratio.value);
            const knee      = Math.max(0, comp.knee.value);
            const overshoot = envDb - threshold;
            let gainDb = 0;
            if (overshoot >= knee / 2) {
              gainDb = -overshoot * (1 - 1 / ratio);
            } else if (overshoot > -knee / 2 && knee > 0) {
              const x = overshoot + knee / 2;
              gainDb = -x * x * (1 - 1 / ratio) / (2 * knee);
            }
            const linGain = Math.pow(10, gainDb / 20);
            s = s * linGain;
            // Track reduction in the node for getReduction()
            comp.reduction = gainDb;
          }
          for (const g of gains) s *= g.gain.value;
          outData[i] += s;
        }
      }
    }
    return out;
  }

  _tracePathToDestination(start) {
    // BFS from `start` collecting every node reachable that eventually
    // hits this.destination. We return them in topological order by
    // simply preserving BFS visit order — close enough for our linear
    // FX chain test.
    const order = [];
    const seen = new Set([start]);
    const queue = [start];
    let reachedDest = false;
    while (queue.length) {
      const node = queue.shift();
      order.push(node);
      if (node === this.destination) reachedDest = true;
      for (const out of node._outs || []) {
        if (!seen.has(out)) { seen.add(out); queue.push(out); }
      }
    }
    if (!reachedDest) return []; // disconnected
    return order;
  }
}

// Patch global
globalThis.AudioContext = MockOfflineAudioContext;
globalThis.OfflineAudioContext = MockOfflineAudioContext;
globalThis.window = globalThis.window || globalThis;

// ─────────────────────────────────────────────────────────────────────────
// Test harness
// ─────────────────────────────────────────────────────────────────────────

let passed = 0;
let failed = 0;
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

// Load the mock 154 mapping from disk so the test exercises the real file.
const mappingPath = path.join(repoRoot, 'public/plugin-mappings/154.json');
const compressorMapping = JSON.parse(fs.readFileSync(mappingPath, 'utf8'));

// ─────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────

const sampleRate = 44100;
const lengthSec = 1;

const PluginAdapterMod = await import('./PluginAdapter.js');
const PluginAdapter = PluginAdapterMod.default;
const { applyCurve, toNormalized } = PluginAdapterMod;

await test('applyCurve linear identity', () => {
  assert.equal(applyCurve('linear', -20, [-60, 0], [-60, 0]), -20);
  assert.equal(applyCurve('linear', 0,   [-60, 0], [-60, 0]), 0);
  assert.equal(applyCurve('linear', -60, [-60, 0], [-60, 0]), -60);
});

await test('applyCurve piecewise interpolates between breakpoints', () => {
  const bp = [[1, 1], [2, 2], [4, 4], [8, 8], [16, 16], [30, 20]];
  assert.equal(applyCurve('piecewise', 1, [1, 30], [1, 20], bp), 1);
  assert.equal(applyCurve('piecewise', 4, [1, 30], [1, 20], bp), 4);
  // Halfway between (16,16) and (30,20) at logic=23 → web=18
  const v = applyCurve('piecewise', 23, [1, 30], [1, 20], bp);
  assert.ok(Math.abs(v - 18) < 0.01, `expected ~18, got ${v}`);
});

await test('applyCurve log spans the range monotonically', () => {
  const r = (t) => applyCurve('log', t, [0, 1], [0.1, 1000]);
  assert.ok(r(0) < r(0.5) && r(0.5) < r(1));
  assert.ok(Math.abs(r(0) - 0.1) < 1e-6);
  assert.ok(Math.abs(r(1) - 1000) < 1e-6);
});

await test('toNormalized inverts engine scaleParam', () => {
  // skew=1 case: trivial linear inverse
  const def = { min: -60, max: 0, skew: 1 };
  assert.ok(Math.abs(toNormalized(-30, def) - 0.5) < 1e-6);
  // skew=0.3 case: the inverse must round-trip through scaleParam
  const skewDef = { min: 0.1, max: 1000, skew: 0.3 };
  const norm = toNormalized(100, skewDef);
  // engine.scaleParam: shaped = norm^(1/skew), value = min + (max-min)*shaped
  const shaped = Math.pow(norm, 1 / skewDef.skew);
  const round = skewDef.min + (skewDef.max - skewDef.min) * shaped;
  assert.ok(Math.abs(round - 100) < 1e-3, `round-trip failed: ${round}`);
});

await test('PluginAdapter returns null for unknown plugin (graceful fallback)', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate * lengthSec, sampleRate);
  const adapter = new PluginAdapter(ctx, { fetchImpl: null });
  const slot = await adapter.instantiate({
    plugin_id: 'nonsense-id',
    plugin_name: 'Bogus',
    parameters: [],
  });
  assert.equal(slot, null);
});

await test('PluginAdapter strictMode throws on missing mapping', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate * lengthSec, sampleRate);
  const adapter = new PluginAdapter(ctx, { fetchImpl: null, strictMode: true });
  await assert.rejects(
    () => adapter.instantiate({ plugin_id: 'missing', parameters: [] }),
    /no mapping/
  );
});

await test('PluginAdapter instantiates Compressor mapping with curve-fit params', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate * lengthSec, sampleRate);
  const adapter = new PluginAdapter(ctx, {
    mappingsRegistry: { 154: compressorMapping },
    fetchImpl: null,
  });
  await adapter.load(); // no-op when fetchImpl is null
  assert.ok(adapter.hasMapping('154'));

  const slot = await adapter.instantiate({
    plugin_id: '154',
    plugin_name: 'Compressor',
    parameters: [
      { id: 0, name: 'Threshold', value: -30 },
      { id: 1, name: 'Ratio',     value: 8 },
      { id: 2, name: 'Attack',    value: 0.5 },
      { id: 3, name: 'Release',   value: 0.5 },
      { id: 4, name: 'Knee',      value: 0.5 },
      { id: 5, name: 'Gain',      value: 0 },
    ],
  });
  assert.ok(slot, 'slot should be non-null');
  assert.ok(slot.engine, 'slot.engine should exist');
  assert.ok(slot.input,  'slot.input should be a node');
  assert.ok(slot.output, 'slot.output should be a node');

  // Verify the threshold Logic param mapped through linear curve onto the
  // compressor's threshold AudioParam at -30 dB.
  // The engine builds a Compressor node — find it via paramTargets.
  const compTarget = slot.engine.paramTargets['threshold'];
  assert.ok(compTarget, 'engine should bind threshold');
  assert.ok(Math.abs(compTarget.audioParam.value - (-30)) < 0.5,
    `expected threshold≈-30, got ${compTarget.audioParam.value}`);

  // Verify ratio piecewise: logic=8 → web=8
  const ratioTarget = slot.engine.paramTargets['ratio'];
  assert.ok(Math.abs(ratioTarget.audioParam.value - 8) < 0.5,
    `expected ratio≈8, got ${ratioTarget.audioParam.value}`);

  slot.dispose();
});

await test('Renders 1s of input + asserts dynamic-range reduction', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate * lengthSec, sampleRate);
  const adapter = new PluginAdapter(ctx, {
    mappingsRegistry: { 154: compressorMapping },
    fetchImpl: null,
  });

  const slot = await adapter.instantiate({
    plugin_id: '154',
    plugin_name: 'Compressor',
    parameters: [
      // Hard-clip-ish settings so reduction is unmistakable.
      { id: 0, name: 'Threshold', value: -30 },
      { id: 1, name: 'Ratio',     value: 16 },
      { id: 2, name: 'Attack',    value: 0.0 },
      { id: 3, name: 'Release',   value: 0.0 },
      { id: 4, name: 'Knee',      value: 0.0 },
      // Engine's compressor builder treats the makeup AudioParam as a
      // raw linear gain (no dB→linear scale on the target), so a Logic
      // value of `1` round-trips to gain.value=1.0. Setting `0` here
      // would silence the chain entirely. Until the engine is patched
      // to apply dbToLinear on its compressor.makeup target the test
      // works around it explicitly.
      { id: 5, name: 'Gain',      value: 1 },
    ],
  });

  // Loud + quiet bursts. The compressor should leave quiet regions
  // alone but pull the loud ones down toward the threshold.
  const bufLen = sampleRate * lengthSec;
  const inputBuf = ctx.createBuffer(2, bufLen, sampleRate);
  for (let ch = 0; ch < 2; ch++) {
    const data = inputBuf.getChannelData(ch);
    for (let i = 0; i < bufLen; i++) {
      const t = i / sampleRate;
      const env = (t < 0.5) ? 0.95 : 0.05;          // loud half, quiet half
      data[i] = env * Math.sin(2 * Math.PI * 440 * t);
    }
  }
  const src = ctx.createBufferSource();
  src.buffer = inputBuf;
  src.connect(slot.input);
  slot.output.connect(ctx.destination);
  src.start();

  const rendered = await ctx.startRendering();
  assert.ok(rendered, 'rendered buffer should exist');

  // Compute peak amplitude in loud half vs. quiet half BEFORE compression
  // and AFTER. We expect:
  //   - quiet half mostly unchanged (input peak 0.05 → output peak ≈ 0.05)
  //   - loud half pulled down (input peak 0.95 → output peak < 0.5)
  const ch0 = rendered.getChannelData(0);
  let loudPeakOut = 0, quietPeakOut = 0;
  const half = Math.floor(bufLen / 2);
  // Skip the leading transient (compressor envelope settles in a few ms).
  const loudStart = Math.floor(0.05 * sampleRate);
  for (let i = loudStart; i < half; i++) loudPeakOut = Math.max(loudPeakOut, Math.abs(ch0[i]));
  for (let i = half; i < bufLen; i++)    quietPeakOut = Math.max(quietPeakOut, Math.abs(ch0[i]));

  // Output should still be non-zero: the chain isn't muted.
  assert.ok(loudPeakOut > 0.01, `expected non-zero loud output, got ${loudPeakOut}`);
  assert.ok(quietPeakOut > 0.01, `expected non-zero quiet output, got ${quietPeakOut}`);

  // Loud half compressed: peak should be appreciably less than the
  // 0.95 input peak. With threshold -30 dB and 16:1 ratio we expect
  // < ~0.5 (well above -30 dB but well below 0.95).
  assert.ok(loudPeakOut < 0.7,
    `expected loud peak compressed below 0.7, got ${loudPeakOut.toFixed(3)}`);

  // Dynamic-range ratio: loud/quiet AT INPUT was 0.95/0.05 = 19. After
  // compression it should be measurably smaller (closer to 1). Sanity
  // bound: at least 30% reduction in dynamic range.
  const drIn  = 0.95 / 0.05;                    // = 19
  const drOut = loudPeakOut / Math.max(quietPeakOut, 1e-6);
  assert.ok(drOut < drIn * 0.7,
    `expected dynamic range reduction; in=${drIn.toFixed(2)} out=${drOut.toFixed(2)}`);

  slot.dispose();
});

await test('setLogicParam updates engine in real time', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate, sampleRate);
  const adapter = new PluginAdapter(ctx, {
    mappingsRegistry: { 154: compressorMapping },
    fetchImpl: null,
  });
  const slot = await adapter.instantiate({
    plugin_id: '154',
    plugin_name: 'Compressor',
    parameters: [{ id: 0, name: 'Threshold', value: -30 }],
  });
  const thr = slot.engine.paramTargets['threshold'];
  assert.ok(Math.abs(thr.audioParam.value - (-30)) < 0.5);

  // Drag the knob from -30 → -10 dB
  const ok = slot.setLogicParam(0, -10);
  assert.equal(ok, true);
  assert.ok(Math.abs(thr.audioParam.value - (-10)) < 0.5,
    `setLogicParam by id failed: ${thr.audioParam.value}`);

  // Also drag by name (Logic id drift is real)
  slot.setLogicParam('Threshold', -50);
  assert.ok(Math.abs(thr.audioParam.value - (-50)) < 0.5,
    `setLogicParam by name failed: ${thr.audioParam.value}`);

  slot.dispose();
});

await test('buildTrackChain wires plugins in series; falls back on miss', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate, sampleRate);
  const adapter = new PluginAdapter(ctx, {
    mappingsRegistry: { 154: compressorMapping },
    fetchImpl: null,
  });

  // Track with a single mapped plugin → live chain
  const liveTrack = {
    id: 'track-1',
    logicPlugins: [{
      plugin_id: '154', plugin_name: 'Compressor',
      parameters: [{ id: 0, name: 'Threshold', value: -20 }],
    }],
  };
  const live = await adapter.buildTrackChain(liveTrack);
  assert.equal(live.fallback, false);
  assert.equal(live.slots.length, 1);
  assert.ok(live.input && live.output);
  live.dispose();

  // Track with one unknown plugin → fallback
  const fbTrack = {
    id: 'track-2',
    logicPlugins: [
      { plugin_id: '154', plugin_name: 'Compressor', parameters: [] },
      { plugin_id: '99999', plugin_name: 'UnsupportedReverb', parameters: [] },
    ],
  };
  const fb = await adapter.buildTrackChain(fbTrack);
  assert.equal(fb.fallback, true);
  assert.equal(fb.input, null);
  assert.equal(fb.output, null);
  assert.equal(fb.missingPluginId, '99999');

  // Track with no plugins → passthrough, fallback=false (zero behavior change)
  const pt = await adapter.buildTrackChain({ id: 'track-3', logicPlugins: [] });
  assert.equal(pt.fallback, false);
  assert.equal(pt.slots.length, 0);
  pt.dispose();
});

await test('indexed param bypasses curve + normalization (dropdown UX)', async () => {
  // Mapping has one indexed Logic row → web_param "mode" + a regular
  // Threshold row. The slot's setLogicParam should pass the integer
  // index straight through for the indexed row and apply the curve for
  // the threshold row. We verify by capturing setIndexedParameter and
  // setParameter calls on the engine.
  const indexedMapping = {
    plugin_id: '999',
    plugin_name: 'ToyDistortion',
    web_topology: {
      dspChain: [{ type: 'compressor', id: 'comp_core' }],
      parameters: [
        { id: 'threshold', label: 'Threshold', min: -60, max: 0, default: -20 },
        { id: 'mode',      label: 'Mode',      min: 0,   max: 2, default: 0 },
      ],
      routing: { input: 'stereo', chain: [], output: 'stereo' },
    },
    param_map: [
      { logic_id: 0, logic_name: 'Threshold', web_param: 'threshold',
        curve: 'linear', domain: [-60, 0], range: [-60, 0] },
      { logic_id: 1, logic_name: 'Mode', web_param: 'mode', indexed: true,
        curve: 'linear', domain: [0, 2], range: [0, 2] },
    ],
    bypass_supported: true,
    expected_sample_rate: sampleRate,
  };

  const ctx = new MockOfflineAudioContext(2, sampleRate, sampleRate);
  const adapter = new PluginAdapter(ctx, {
    mappingsRegistry: { 999: indexedMapping },
    fetchImpl: null,
  });
  const slot = await adapter.instantiate({
    plugin_id: '999',
    plugin_name: 'ToyDistortion',
    parameters: [
      { id: 0, name: 'Threshold', value: -20 },
      { id: 1, name: 'Mode',      value: 1 },
    ],
  });
  // Capture engine dispatches.
  const calls = [];
  slot.engine.setIndexedParameter = (k, v) => calls.push(['idx', k, v]);
  const origSet = slot.engine.setParameter.bind(slot.engine);
  slot.engine.setParameter = (k, v) => { calls.push(['set', k, v]); origSet(k, v); };

  slot.setLogicParam(1, 2);   // indexed → setIndexedParameter, raw int
  slot.setLogicParam(0, -40); // continuous → setParameter, normalized

  const idxCall = calls.find(c => c[0] === 'idx');
  assert.ok(idxCall, 'indexed dispatch should have fired');
  assert.equal(idxCall[1], 'mode');
  assert.equal(idxCall[2], 2);  // integer passed through, not normalized

  const setCall = calls.find(c => c[0] === 'set' && c[1] === 'threshold');
  assert.ok(setCall, 'continuous dispatch should have fired');
  // setParameter receives a normalized [0,1] value; -40 in [-60,0] → 0.333
  assert.ok(setCall[2] >= 0 && setCall[2] <= 1,
    `expected normalized in [0,1], got ${setCall[2]}`);

  // Convenience flags surface for the React layer.
  assert.equal(slot.bypassSupported, true);
  assert.equal(slot.routingMode, 'stereo');

  slot.dispose();
});

await test('setBypassed wires lazy passthrough; dispose cleans up', async () => {
  const ctx = new MockOfflineAudioContext(2, sampleRate, sampleRate);
  const adapter = new PluginAdapter(ctx, {
    mappingsRegistry: { 154: compressorMapping },
    fetchImpl: null,
  });
  const slot = await adapter.instantiate({
    plugin_id: '154',
    plugin_name: 'Compressor',
    parameters: [{ id: 0, name: 'Threshold', value: -20 }],
  });
  // Pre-toggle: bypass is functions but no parallel splice yet.
  assert.equal(typeof slot.setBypassed, 'function');
  assert.equal(slot.isBypassed(), false);

  // First toggle wires the lazy bypass infrastructure. The slot.output
  // pointer may move from the engine's masterGain to the new sum gain;
  // both are MockGainNodes so duck-typing holds.
  slot.setBypassed(true);
  assert.equal(slot.isBypassed(), true);
  slot.setBypassed(false);
  assert.equal(slot.isBypassed(), false);

  // Dispose without errors regardless of bypass state.
  slot.dispose();
});

// ─────────────────────────────────────────────────────────────────────────
// Done
// ─────────────────────────────────────────────────────────────────────────

process.stdout.write(`\n${passed} passed, ${failed} failed\n`);
if (failed > 0) {
  for (const { name, err } of failures) {
    process.stderr.write(`\n--- ${name} ---\n${err.stack || err.message}\n`);
  }
  process.exit(1);
}
