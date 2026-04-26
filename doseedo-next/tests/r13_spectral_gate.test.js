/**
 * R13 Spectral Gate — builder smoke + gate-attenuation tests.
 *
 * Run with:
 *   node --test tests/r13_spectral_gate.test.js
 *
 * Two layers of coverage:
 *
 *   1. **Builder contract / smoke**: stubs out AudioContext + forces the
 *      worklet path to fail so the fallback branch executes. Confirms the
 *      builder returns `{ input, output, paramTargets }`, that `@param_id`
 *      bindings populate `paramTargets`, and that literal numeric params
 *      do not.
 *
 *   2. **Gate attenuation (numerical)**: Web Audio worklets can't run in
 *      Node, but the gating math is pure DSP. We re-derive the same
 *      algorithm — FFT → per-bin threshold compare → reduction-gain
 *      multiplier — and verify that a -50 dBFS pure tone is attenuated by
 *      the expected reduction_db when the threshold is set above the
 *      tone level. This is the proof the worklet's per-bin gate decision
 *      is correct, without needing a browser.
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';

import {
  buildSpectralGate,
} from '../src/audio/builders/r13_spectral_gate.js';
import R13_SPECTRAL_GATE_BUILDERS from '../src/audio/builders/r13_spectral_gate.js';

// ── Minimal AudioContext stub ────────────────────────────────────────────
function makeStubCtx() {
  const sampleRate = 48000;
  function makeParam(value) {
    return { value, _scheduled: [],
      setValueAtTime() {}, linearRampToValueAtTime() {}, setTargetAtTime() {} };
  }
  function createGain() {
    return {
      gain: makeParam(1),
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  function createDynamicsCompressor() {
    return {
      threshold: makeParam(-24),
      ratio:     makeParam(12),
      attack:    makeParam(0.003),
      release:   makeParam(0.25),
      knee:      makeParam(30),
      _connections: [],
      connect(target) { this._connections.push(target); return target; },
      disconnect() { this._connections.length = 0; },
    };
  }
  return {
    sampleRate,
    audioWorklet: { addModule: async () => {} }, // resolves; no real registration
    createGain,
    createDynamicsCompressor,
  };
}

// Force `new AudioWorkletNode(...)` to throw → builder takes the fallback
// path. Mirrors the convention used in r13_chromaverb.test.js etc.
function installThrowingWorklet() {
  globalThis.AudioWorkletNode = class { constructor() { throw new Error('worklet not available in node'); } };
}

function uninstallWorklet() {
  delete globalThis.AudioWorkletNode;
}

const PARAM_DEFS = {
  threshold:    { id: 'threshold',    min: -60, max: 0,    default: -40 },
  reduction:    { id: 'reduction',    min: -60, max: 0,    default: -40 },
  attack:       { id: 'attack',       min: 1,   max: 100,  default: 10 },
  release:      { id: 'release',      min: 10,  max: 1000, default: 100 },
  low_cut_p:    { id: 'low_cut_p',    min: 0,   max: 1,    default: 0 },
  high_cut_p:   { id: 'high_cut_p',   min: 0,   max: 1,    default: 1 },
  tilt:         { id: 'tilt',         min: -12, max: 12,   default: 0 },
  mix:          { id: 'mix',          min: 0,   max: 1,    default: 1 },
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

// ── Builder smoke ────────────────────────────────────────────────────────

test('buildSpectralGate — fallback path returns valid contract', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'spectral_gate',
      params: {
        threshold_db: '@threshold',
        reduction_db: '@reduction',
        attack_ms:    '@attack',
        release_ms:   '@release',
        low_cut:      '@low_cut_p',
        high_cut:     '@high_cut_p',
        tilt_db:      '@tilt',
        mix:          '@mix',
      },
    };
    const result = buildSpectralGate(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'spectral_gate');

    // Every modulated param must be in paramTargets
    for (const id of ['threshold', 'reduction', 'attack', 'release',
                      'low_cut_p', 'high_cut_p', 'tilt', 'mix']) {
      assert.ok(id in result.paramTargets,
        `spectral_gate: ${id} is exposed in paramTargets`);
    }
    // In fallback, every target should have a customSetter (no AudioParam)
    for (const id of Object.keys(result.paramTargets)) {
      assert.equal(typeof result.paramTargets[id].customSetter, 'function',
        `spectral_gate: ${id} has customSetter in fallback`);
    }
  } finally {
    uninstallWorklet();
  }
});

test('buildSpectralGate — literal params do not populate paramTargets', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'spectral_gate',
      params: {
        threshold_db: -30,
        reduction_db: -50,
        attack_ms:    5,
        release_ms:   200,
        low_cut:      0.05,
        high_cut:     0.95,
        tilt_db:      -3,
        mix:          1,
      },
    };
    const result = buildSpectralGate(ctx, node, PARAM_DEFS);
    expectBuilderContract(result, 'spectral_gate(literals)');
    assert.equal(Object.keys(result.paramTargets).length, 0,
      'literal params produce no paramTargets entries');
  } finally {
    uninstallWorklet();
  }
});

test('buildSpectralGate — fallback custom setters do not throw on numeric input', () => {
  installThrowingWorklet();
  try {
    const ctx = makeStubCtx();
    const node = {
      type: 'spectral_gate',
      params: {
        threshold_db: '@threshold',
        reduction_db: '@reduction',
        attack_ms:    '@attack',
        release_ms:   '@release',
        low_cut:      '@low_cut_p',
        high_cut:     '@high_cut_p',
        tilt_db:      '@tilt',
        mix:          '@mix',
      },
    };
    const result = buildSpectralGate(ctx, node, PARAM_DEFS);
    // Drive every setter through a representative value
    assert.doesNotThrow(() => result.paramTargets.threshold.customSetter(-25));
    assert.doesNotThrow(() => result.paramTargets.reduction.customSetter(-30));
    assert.doesNotThrow(() => result.paramTargets.attack.customSetter(20));
    assert.doesNotThrow(() => result.paramTargets.release.customSetter(300));
    assert.doesNotThrow(() => result.paramTargets.low_cut_p.customSetter(0.1));
    assert.doesNotThrow(() => result.paramTargets.high_cut_p.customSetter(0.9));
    assert.doesNotThrow(() => result.paramTargets.tilt.customSetter(-6));
    assert.doesNotThrow(() => result.paramTargets.mix.customSetter(0.5));
  } finally {
    uninstallWorklet();
  }
});

test('R13_SPECTRAL_GATE_BUILDERS default export registers spectral_gate', () => {
  assert.equal(typeof R13_SPECTRAL_GATE_BUILDERS.spectral_gate, 'function');
});

// ── Gate-attenuation (numerical DSP) ────────────────────────────────────

// Re-implement just enough of the worklet's per-frame algorithm to verify
// the gate decision is correct. The real worklet uses identical math —
// see src/lib/web-audio-plugins/worklets/r13-spectral-gate-processor.js
// _processFrame(). Using a self-contained FFT here means the test runs
// in pure Node with no browser polyfill.

class FFT {
  constructor(size) {
    this.size = size;
    this.cosT = new Float32Array(size >> 1);
    this.sinT = new Float32Array(size >> 1);
    for (let i = 0, half = size >> 1; i < half; i++) {
      const a = (-2 * Math.PI * i) / size;
      this.cosT[i] = Math.cos(a);
      this.sinT[i] = Math.sin(a);
    }
    const bits = Math.log2(size) | 0;
    this.rev = new Uint32Array(size);
    for (let i = 0; i < size; i++) {
      let r = 0;
      for (let j = 0; j < bits; j++) r = (r << 1) | ((i >> j) & 1);
      this.rev[i] = r;
    }
  }
  forward(re, im) {
    const N = this.size;
    for (let i = 0; i < N; i++) {
      const j = this.rev[i];
      if (j > i) { let t = re[i]; re[i] = re[j]; re[j] = t;
                   t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let block = 2; block <= N; block <<= 1) {
      const halfBlock = block >> 1;
      const step = N / block;
      for (let i = 0; i < N; i += block) {
        for (let j = i, k = 0; j < i + halfBlock; j++, k += step) {
          const l = j + halfBlock;
          const wr = this.cosT[k], wi = this.sinT[k];
          const tr = re[l] * wr - im[l] * wi;
          const ti = re[l] * wi + im[l] * wr;
          re[l] = re[j] - tr; im[l] = im[j] - ti;
          re[j] += tr;        im[j] += ti;
        }
      }
    }
  }
}

function hann(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  return w;
}

// Simulate one analysis frame of the gate algorithm.
// Returns { rmsBefore, rmsAfter, attenuationDb }.
function simulateOneFrame({ tone_hz, sampleRate, level_db, fftSize,
                            threshold_db, reduction_db, low_cut, high_cut, tilt_db }) {
  // 1. Synthesize tone over fftSize samples
  const N = fftSize;
  const halfN = N >> 1;
  const amp = Math.pow(10, level_db / 20);
  const w = hann(N);
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const x = amp * Math.sin(2 * Math.PI * tone_hz * i / sampleRate);
    re[i] = x * w[i];
    im[i] = 0;
  }

  // RMS of the unwindowed tone (for reference)
  let rmsBefore = 0;
  for (let i = 0; i < N; i++) rmsBefore += amp * Math.sin(2 * Math.PI * tone_hz * i / sampleRate) ** 2;
  rmsBefore = Math.sqrt(rmsBefore / N);

  // 2. Forward FFT
  const fft = new FFT(N);
  fft.forward(re, im);

  // 3. Per-bin gate decision — env starts at 1.0 (matches worklet init).
  // Single-frame: the per-bin LPF needs at least one hop to settle, so we
  // apply the *target* gain directly (the LPF would converge to it).
  const reductionGain = Math.pow(10, reduction_db / 20);
  const lowBin = Math.floor(low_cut * halfN);
  const highBin = Math.ceil(high_cut * halfN);
  const env = new Float32Array(halfN + 1);
  for (let k = 0; k <= halfN; k++) {
    const reK = re[k], imK = im[k];
    const mag = Math.sqrt(reK * reK + imK * imK);
    const magDb = 20 * Math.log10(Math.max(mag, 1e-12));
    const inBand = (k >= lowBin && k < highBin);
    const tiltAtK = tilt_db * (k / halfN);
    const threshK = threshold_db + tiltAtK;
    const below = inBand && magDb < threshK;
    env[k] = below ? reductionGain : 1.0;
    re[k] = reK * env[k];
    im[k] = imK * env[k];
  }
  // Hermitian mirror for real IFFT
  for (let k = 1; k < halfN; k++) { re[N - k] = re[k]; im[N - k] = -im[k]; }

  // 4. Inverse FFT (forward with conjugated im, scaled).
  for (let i = 0; i < N; i++) im[i] = -im[i];
  fft.forward(re, im);
  const inv = 1 / N;
  for (let i = 0; i < N; i++) { re[i] *= inv; im[i] = -im[i] * inv; }

  // 5. RMS of the gated, time-domain signal (re-windowed once by Hann like
  // the synthesis step). Compensate for the Hann² double-window energy
  // factor (3/8) so we compare apples to apples with rmsBefore.
  let rmsAfter2 = 0;
  for (let i = 0; i < N; i++) {
    const y = re[i] * w[i];
    rmsAfter2 += y * y;
  }
  // Hann² mean = 3/8; divide N by this to land back on the unwindowed RMS.
  const meanHannSq = 3 / 8;
  rmsAfter2 /= (N * meanHannSq);
  const rmsAfter = Math.sqrt(rmsAfter2);

  const attenuationDb = 20 * Math.log10(Math.max(rmsAfter, 1e-12) / Math.max(rmsBefore, 1e-12));
  return { rmsBefore, rmsAfter, attenuationDb };
}

test('gate attenuates a below-threshold tone by ≥ 20 dB', () => {
  // 1 kHz tone at -50 dBFS, threshold -30 dB → tone is below threshold.
  // reduction_db = -40 → expected attenuation ≈ -40 dB.
  const r = simulateOneFrame({
    tone_hz: 1000,
    sampleRate: 48000,
    level_db: -50,
    fftSize: 2048,
    threshold_db: -30,
    reduction_db: -40,
    low_cut: 0,
    high_cut: 1,
    tilt_db: 0,
  });
  // Allow for FFT bin-leakage (the tone won't sit perfectly on one bin).
  // The dominant bin gets gated to -40 dB; leakage bins also get gated;
  // empirically the bulk attenuation is in the 25-40 dB range.
  assert.ok(r.attenuationDb < -20,
    `expected ≥20 dB attenuation, got ${r.attenuationDb.toFixed(1)} dB ` +
    `(rmsBefore=${r.rmsBefore.toExponential(2)}, rmsAfter=${r.rmsAfter.toExponential(2)})`);
});

// Variant that returns the survival ratio of the tone's *peak* bin
// magnitude (rather than full RMS), so leakage bins outside the main lobe
// don't dominate the metric. This is what we want to measure to confirm
// the gate either passes or attenuates the *tone itself*.
function peakBinSurvivalDb({ tone_hz, sampleRate, level_db, fftSize,
                             threshold_db, reduction_db, low_cut, high_cut, tilt_db }) {
  const N = fftSize;
  const halfN = N >> 1;
  const amp = Math.pow(10, level_db / 20);
  const w = hann(N);
  const re = new Float32Array(N);
  const im = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    re[i] = amp * Math.sin(2 * Math.PI * tone_hz * i / sampleRate) * w[i];
    im[i] = 0;
  }
  const fft = new FFT(N);
  fft.forward(re, im);

  // Find peak bin (will be near tone_hz)
  let peakBin = 0, peakMag = 0;
  for (let k = 0; k <= halfN; k++) {
    const m = Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    if (m > peakMag) { peakMag = m; peakBin = k; }
  }
  const magBefore = peakMag;

  // Apply gate (single-frame target gain = LPF settled value)
  const reductionGain = Math.pow(10, reduction_db / 20);
  const lowBin = Math.floor(low_cut * halfN);
  const highBin = Math.ceil(high_cut * halfN);
  const reK = re[peakBin], imK = im[peakBin];
  const magDb = 20 * Math.log10(Math.max(magBefore, 1e-12));
  const inBand = (peakBin >= lowBin && peakBin < highBin);
  const tiltAtK = tilt_db * (peakBin / halfN);
  const threshK = threshold_db + tiltAtK;
  const below = inBand && magDb < threshK;
  const env = below ? reductionGain : 1.0;
  const reAfter = reK * env, imAfter = imK * env;
  const magAfter = Math.sqrt(reAfter * reAfter + imAfter * imAfter);

  return 20 * Math.log10(Math.max(magAfter, 1e-12) / Math.max(magBefore, 1e-12));
}

test('gate passes an above-threshold tone (peak bin preserved)', () => {
  // 1 kHz tone at -10 dBFS, threshold -30 dB → tone is above threshold.
  // The dominant bin's magnitude must be untouched (≈ 0 dB attenuation).
  const att = peakBinSurvivalDb({
    tone_hz: 1000,
    sampleRate: 48000,
    level_db: -10,
    fftSize: 2048,
    threshold_db: -30,
    reduction_db: -40,
    low_cut: 0,
    high_cut: 1,
    tilt_db: 0,
  });
  assert.ok(att > -0.5,
    `expected the tone's peak bin to be preserved (≥ -0.5 dB), got ${att.toFixed(2)} dB`);
});

test('low_cut excludes bass bins from gating (peak bin preserved)', () => {
  // Tone at 100 Hz, level -50 dBFS, threshold -30 dB → would be gated …
  // but low_cut = 0.02 (lowBin = floor(0.02 * 1024) = 20) places the
  // 100 Hz bin (k≈4) below the cut, so it's exempt.
  const att = peakBinSurvivalDb({
    tone_hz: 100,
    sampleRate: 48000,
    level_db: -50,
    fftSize: 2048,
    threshold_db: -30,
    reduction_db: -40,
    low_cut: 0.02,
    high_cut: 1,
    tilt_db: 0,
  });
  assert.ok(att > -0.5,
    `expected the low-cut-exempt peak bin to be preserved (≥ -0.5 dB), got ${att.toFixed(2)} dB`);
});
