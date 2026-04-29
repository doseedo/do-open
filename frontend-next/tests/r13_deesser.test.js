/**
 * R13 DeEsser 2 — builder smoke + sibilance-attenuation tests.
 *
 * Run with:
 *   node --test tests/r13_deesser.test.js
 *
 * Two test groups:
 *
 *   1. Builder smoke (no Web Audio runtime). We stub a minimal AudioContext
 *      and force AudioWorkletNode construction to throw so the static-cut
 *      BiquadFilterNode fallback is exercised. Every builder must return
 *      `{ input, output, paramTargets }` and bind '@'-modulated params
 *      either as audioParam refs or customSetters.
 *
 *   2. Sibilance attenuation. We re-implement the worklet's DSP inline
 *      (see _runDeesser below — same RBJ peaking-EQ + envelope follower)
 *      and feed it a pink-noise + 7 kHz tone mixture. The output's energy
 *      in the sibilant band (5–10 kHz) must drop relative to the dry
 *      signal by at least 4 dB, while the wide-band level outside the band
 *      stays within 1 dB. This validates that the dynamic peaking EQ
 *      attenuates the targeted frequency range without affecting the rest
 *      of the spectrum.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import { buildDeEsser } from '../src/audio/builders/r13_deesser.js';
import R13_DEESSER_BUILDERS from '../src/audio/builders/r13_deesser.js';

// ── Minimal AudioContext stub (matches r13_chromaverb.test.js style) ─────
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
  function createBiquadFilter() {
    return {
      type: 'lowpass',
      frequency: { value: 350 },
      Q:         { value: 1 },
      gain:      { value: 0 },
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  return { sampleRate, createGain, createBiquadFilter };
}

function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class { constructor() { throw new Error('not available in Node'); } };
}

function uninstallWorklet() {
  delete globalThis.AudioWorkletNode;
}

const PARAM_DEFS = {
  freq_low:     { id: 'freq_low',     min: 1500, max: 10000, default: 4000 },
  freq_high:    { id: 'freq_high',    min: 5000, max: 15000, default: 9000 },
  threshold_db: { id: 'threshold_db', min: -60,  max: 0,     default: -28 },
  range_db:     { id: 'range_db',     min: 0,    max: 24,    default: 12 },
  q:            { id: 'q',            min: 0.5,  max: 10,    default: 2.0 },
  monitor:      { id: 'monitor',      min: 0,    max: 1,     default: 0 },
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

// ── Group 1: builder smoke ───────────────────────────────────────────────

test('buildDeEsser — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'deesser',
      params: {
        freq_low: '@freq_low',
        freq_high: '@freq_high',
        threshold_db: -30,
        range_db: '@range_db',
        q: 2.5,
        monitor: 0,
      },
    };
    const result = buildDeEsser(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'deesser');
    assert.ok('freq_low'  in result.paramTargets, 'modulated freq_low exposed');
    assert.ok('freq_high' in result.paramTargets, 'modulated freq_high exposed');
    assert.ok('range_db'  in result.paramTargets, 'modulated range_db exposed');
    // freq_low setter must update the underlying biquad frequency
    assert.equal(typeof result.paramTargets.freq_low.customSetter, 'function');
    assert.doesNotThrow(() => result.paramTargets.freq_low.customSetter(5000));
    assert.doesNotThrow(() => result.paramTargets.range_db.customSetter(18));
    // The fallback peaking biquad should be present
    assert.ok(result.fallbackPeak, 'fallback biquad is exposed for inspection');
    assert.equal(result.fallbackPeak.type, 'peaking');
  } finally {
    uninstallWorklet();
  }
});

test('buildDeEsser — static params bind no targets', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'deesser',
      params: {
        freq_low: 5000, freq_high: 10000,
        threshold_db: -20, range_db: 10, q: 3, monitor: 0,
      },
    };
    const result = buildDeEsser(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'deesser-static');
    assert.equal(Object.keys(result.paramTargets).length, 0,
      'static params produce no paramTargets');
    // Initial gain should be -range_db / 2 = -5 dB (mid-amount cut)
    assert.equal(result.fallbackPeak.gain.value, -5);
  } finally {
    uninstallWorklet();
  }
});

test('R13_DEESSER_BUILDERS default export registers the deesser node type', () => {
  assert.equal(typeof R13_DEESSER_BUILDERS.deesser, 'function');
});

// ── Group 2: sibilance attenuation — DSP equivalence test ─────────────────
//
// We re-implement the worklet's DSP inline (same RBJ peaking-EQ + cascade
// HP/LP detection bandpass + envelope follower). This exercises the exact
// math the worklet runs without needing an AudioWorklet runtime, which Node
// doesn't provide.

const SR = 48000;

function rbjHighpass(state, fc) {
  const w0 = 2 * Math.PI * fc / SR;
  const cos_w0 = Math.cos(w0);
  const sin_w0 = Math.sin(w0);
  const Q = Math.SQRT1_2;
  const alpha = sin_w0 / (2 * Q);
  const a0 = 1 + alpha;
  state.b0 =  ((1 + cos_w0) / 2) / a0;
  state.b1 = (-(1 + cos_w0))     / a0;
  state.b2 =  ((1 + cos_w0) / 2) / a0;
  state.a1 = (-2 * cos_w0)       / a0;
  state.a2 = (1 - alpha)         / a0;
}

function rbjLowpass(state, fc) {
  const w0 = 2 * Math.PI * fc / SR;
  const cos_w0 = Math.cos(w0);
  const sin_w0 = Math.sin(w0);
  const Q = Math.SQRT1_2;
  const alpha = sin_w0 / (2 * Q);
  const a0 = 1 + alpha;
  state.b0 = ((1 - cos_w0) / 2) / a0;
  state.b1 =  (1 - cos_w0)      / a0;
  state.b2 = ((1 - cos_w0) / 2) / a0;
  state.a1 = (-2 * cos_w0)      / a0;
  state.a2 = (1 - alpha)        / a0;
}

function rbjPeaking(coef, fc, Q, gainDb) {
  const A = Math.pow(10, gainDb / 40);
  const w0 = 2 * Math.PI * fc / SR;
  const cos_w0 = Math.cos(w0);
  const sin_w0 = Math.sin(w0);
  const alpha = sin_w0 / (2 * Math.max(0.05, Q));
  const a0 = 1 + alpha / A;
  coef.b0 = (1 + alpha * A) / a0;
  coef.b1 = (-2 * cos_w0) / a0;
  coef.b2 = (1 - alpha * A) / a0;
  coef.a1 = (-2 * cos_w0) / a0;
  coef.a2 = (1 - alpha / A) / a0;
}

function bq(state, coef, x) {
  const y = coef.b0 * x + coef.b1 * state.x1 + coef.b2 * state.x2
            - coef.a1 * state.y1 - coef.a2 * state.y2;
  state.x2 = state.x1; state.x1 = x;
  state.y2 = state.y1; state.y1 = y;
  return y;
}

function newState() { return { x1: 0, x2: 0, y1: 0, y2: 0, b0: 1, b1: 0, b2: 0, a1: 0, a2: 0 }; }

function _runDeesser(input, p) {
  const N = input.length;
  const out = new Float32Array(N);

  const hp = newState(); rbjHighpass(hp, p.freq_low);
  const lp = newState(); rbjLowpass(lp,  p.freq_high);
  const peak = newState();
  const peakCoef = {};

  const coefA = 1 - Math.exp(-1 / Math.max(1, (p.attack_ms  / 1000) * SR));
  const coefR = 1 - Math.exp(-1 / Math.max(1, (p.release_ms / 1000) * SR));

  let env = 0;
  let lastFc = -1, lastQ = -1, lastGdb = 0;
  const fc = Math.sqrt(p.freq_low * p.freq_high);

  for (let i = 0; i < N; i++) {
    const x = input[i];
    // Detection cascade
    const hpOut = bq(hp, hp, x);
    const lpOut = bq(lp, lp, hpOut);
    const rect = Math.abs(lpOut);
    const c = (rect > env) ? coefA : coefR;
    env = (1 - c) * env + c * rect;
    const envDb = (env > 1e-6) ? 20 * Math.log10(env) : -120;
    const overshoot = envDb - p.threshold_db;
    const knee = 6;
    let amount;
    if      (overshoot <= -knee / 2) amount = 0;
    else if (overshoot >=  knee / 2) amount = 1;
    else amount = (overshoot + knee / 2) / knee;
    const cutDb = -p.range_db * amount;

    if (Math.abs(fc - lastFc) > 0.5
        || Math.abs(p.q - lastQ) > 0.001
        || Math.abs(cutDb - lastGdb) > 0.05) {
      rbjPeaking(peakCoef, fc, p.q, cutDb);
      lastFc = fc; lastQ = p.q; lastGdb = cutDb;
    }
    out[i] = bq(peak, peakCoef, x);
  }
  return out;
}

// Goertzel-based single-bin power estimate at frequency f (Hz)
function bandPowerDb(buf, fLo, fHi, samples = buf.length) {
  // Simple wide-band Goertzel-aggregate at ~50-Hz bins between fLo and fHi
  const binStep = 50;
  let totalPower = 0;
  let bins = 0;
  for (let f = fLo; f <= fHi; f += binStep) {
    const w = 2 * Math.PI * f / SR;
    const coef = 2 * Math.cos(w);
    let s0 = 0, s1 = 0, s2 = 0;
    for (let i = 0; i < samples; i++) {
      s0 = buf[i] + coef * s1 - s2;
      s2 = s1;
      s1 = s0;
    }
    const power = s1 * s1 + s2 * s2 - coef * s1 * s2;
    totalPower += power;
    bins++;
  }
  const avg = totalPower / Math.max(1, bins);
  return 10 * Math.log10(avg + 1e-30);
}

function makePinkNoise(N) {
  // Voss-McCartney-ish 6-stage pink noise (cheap and adequate)
  const out = new Float32Array(N);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
  for (let i = 0; i < N; i++) {
    const white = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + white * 0.0555179;
    b1 = 0.99332 * b1 + white * 0.0750759;
    b2 = 0.96900 * b2 + white * 0.1538520;
    b3 = 0.86650 * b3 + white * 0.3104856;
    b4 = 0.55000 * b4 + white * 0.5329522;
    b5 = -0.7616 * b5 - white * 0.0168980;
    out[i] = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362) * 0.11;
    b6 = white * 0.115926;
  }
  return out;
}

function addTone(buf, freqHz, amplitude) {
  const w = 2 * Math.PI * freqHz / SR;
  for (let i = 0; i < buf.length; i++) {
    buf[i] += amplitude * Math.sin(w * i);
  }
}

test('DeEsser DSP — attenuates 7 kHz tone embedded in pink noise', () => {
  const N = SR; // 1 second
  const dry = makePinkNoise(N);
  // Embed a strong 7 kHz tone — well above the threshold once the bandpass
  // selects it. Amplitude 0.5 ≈ -6 dBFS; pink noise floor ≈ -25 dBFS RMS.
  addTone(dry, 7000, 0.5);

  const params = {
    freq_low: 5000,
    freq_high: 10000,
    threshold_db: -36,
    range_db: 18,
    attack_ms: 1.5,
    release_ms: 40,
    q: 2.0,
  };
  const wet = _runDeesser(dry, params);

  // Targeted-band attenuation in 5–10 kHz
  const dryBand = bandPowerDb(dry, 5000, 10000);
  const wetBand = bandPowerDb(wet, 5000, 10000);
  const reductionDb = dryBand - wetBand;

  // Outside-band check: 200–1500 Hz should be largely unaffected
  const dryOut = bandPowerDb(dry, 200, 1500);
  const wetOut = bandPowerDb(wet, 200, 1500);
  const outDeltaDb = Math.abs(dryOut - wetOut);

  assert.ok(reductionDb >= 4,
    `expected ≥ 4 dB cut in sibilant band, got ${reductionDb.toFixed(2)} dB`);
  assert.ok(outDeltaDb <= 1.0,
    `expected outside-band delta ≤ 1 dB, got ${outDeltaDb.toFixed(2)} dB`);
});

test('DeEsser DSP — leaves clean signal alone (no envelope crossing)', () => {
  // Pink noise WITHOUT the sibilant tone, far below threshold ⇒ no cut.
  const N = SR / 2;
  const dry = makePinkNoise(N);
  // Scale way down so the band never crosses threshold_db
  for (let i = 0; i < N; i++) dry[i] *= 0.05;

  const params = {
    freq_low: 5000,
    freq_high: 10000,
    threshold_db: -20,   // high threshold; clean material won't trip it
    range_db: 18,
    attack_ms: 1.5,
    release_ms: 40,
    q: 2.0,
  };
  const wet = _runDeesser(dry, params);

  const dryBand = bandPowerDb(dry, 5000, 10000);
  const wetBand = bandPowerDb(wet, 5000, 10000);
  const delta = Math.abs(dryBand - wetBand);
  assert.ok(delta <= 1.5,
    `expected ≤ 1.5 dB delta on clean signal, got ${delta.toFixed(2)} dB`);
});
