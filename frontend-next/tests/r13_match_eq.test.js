/**
 * r13_match_eq.test.js — smoke + spectrum-match-on-pink-noise tests.
 *
 * Coverage:
 *   1. Builder smoke — buildMatchEQ(...) returns the expected
 *      { input, output, paramTargets } shape under a stub AudioContext
 *      (no AudioWorklet available → falls back to passthrough).
 *   2. Host-side helpers — smoothCurveOctave + computeMatchCurve produce
 *      sensible numerical results.
 *   3. Spectrum-match-on-pink-noise — generate pink noise (source) and
 *      a low-passed copy of the same pink noise (target). The match
 *      curve should approximate a low-pass response, and the apply
 *      pipeline (reference JS implementation mirroring the worklet)
 *      should make the source's spectrum approach the target's within
 *      the [low_cut, high_cut] band.
 *
 * Runner: this file uses describe/it/expect when present (Jest/Vitest)
 * and exposes a `runAll()` async function for direct `node` execution.
 *
 * @author Doseedo R13
 */

import {
  buildMatchEQ,
  smoothCurveOctave,
  computeMatchCurve,
} from '../src/audio/builders/r13_match_eq.js';

// ── Stub AudioContext / AudioWorkletNode for builder smoke ──────────────────

class StubAudioParam { constructor(v = 0) { this.value = v; } }

class StubGainNode {
  constructor() { this.gain = new StubAudioParam(1); this._connections = []; }
  connect(dst) { this._connections.push(dst); return dst; }
  disconnect() { this._connections = []; }
}

class StubAudioContext {
  constructor() {
    this.sampleRate = 44100;
    // Intentionally NO `audioWorklet` property → builder returns null worklet,
    // exercises the passthrough fallback path.
  }
  createGain() { return new StubGainNode(); }
}

function builderSmokeTest() {
  const ctx = new StubAudioContext();
  const nodeDef = {
    type: 'match_eq',
    params: {
      mode: 'apply',
      curve_amount: '@amt',
      low_cut: 100,
      high_cut: 16000,
      gain_makeup: 0,
    },
  };
  const paramDefs = { amt: { min: 0, max: 1 } };
  const built = buildMatchEQ(ctx, nodeDef, paramDefs);
  const okShape = built && built.input && built.output && built.paramTargets;
  if (!okShape) return { pass: false, message: 'builder returned wrong shape' };
  // Passthrough: input should be connected to output.
  const passthroughOK = built.input._connections.includes(built.output);
  // Param target for amt should be a custom-setter no-op in fallback mode.
  const amtTargetOK = !!built.paramTargets.amt;
  const pass = passthroughOK && amtTargetOK;
  return {
    pass,
    message: pass
      ? 'builder smoke OK (fallback passthrough + param surface bound)'
      : `builder smoke FAIL — passthrough=${passthroughOK} amtTarget=${amtTargetOK}`,
  };
}

// ── Host-side helpers numerical sanity ──────────────────────────────────────

function helperSanityTest() {
  // Construct a target with a +6 dB peak at bin 200, source flat at 0.5.
  const N = 513; // halfSize+1 for a 1024-pt FFT
  const sr = 44100;
  const target = new Float32Array(N).fill(0.5);
  const source = new Float32Array(N).fill(0.5);
  target[200] = 0.5 * Math.pow(10, 6 / 20); // +6 dB at bin 200

  const matched = computeMatchCurve(target, source);
  // matched[200] should be ≈ 10^(6/20) = 1.995…
  const expected = Math.pow(10, 6 / 20);
  const errMag = Math.abs(matched[200] - expected);
  if (errMag > 1e-3) {
    return { pass: false, message: `computeMatchCurve peak err=${errMag}` };
  }

  // Smoothing across 1/3-octave should drop the +6 dB peak (single-bin spike)
  // a lot — neighbours pull the geo-mean way down.
  const smoothed = smoothCurveOctave(matched, sr, 1 / 3);
  const peakDb       = 20 * Math.log10(matched[200]);   // ≈ +6 dB
  const smoothedPeakDb = 20 * Math.log10(smoothed[200]); // → small
  const dropOK = smoothedPeakDb < peakDb - 1.5; // smoothing must reduce peak
  if (!dropOK) {
    return {
      pass: false,
      message: `smoothing did not reduce single-bin peak: raw=${peakDb.toFixed(2)} dB, smoothed=${smoothedPeakDb.toFixed(2)} dB`,
    };
  }

  return {
    pass: true,
    message: `helper sanity OK (computeMatchCurve peak=${matched[200].toFixed(3)} ≈ ${expected.toFixed(3)}, smoothing reduced ${peakDb.toFixed(1)} → ${smoothedPeakDb.toFixed(1)} dB)`,
  };
}

// ── Reference JS port of the worklet's analyze + apply path ────────────────

class FFT {
  constructor(size) {
    this.size = size;
    this.half = size >> 1;
    this.cosT = new Float32Array(this.half);
    this.sinT = new Float32Array(this.half);
    for (let i = 0; i < this.half; i++) {
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
  _bitReverse(re, im) {
    const rev = this.rev;
    for (let i = 0; i < this.size; i++) {
      const j = rev[i];
      if (j > i) {
        const tr = re[i]; re[i] = re[j]; re[j] = tr;
        const ti = im[i]; im[i] = im[j]; im[j] = ti;
      }
    }
  }
  forward(re, im) {
    this._bitReverse(re, im);
    const N = this.size;
    for (let block = 2; block <= N; block <<= 1) {
      const halfBlock = block >> 1;
      const step = N / block;
      for (let i = 0; i < N; i += block) {
        for (let j = i, k = 0; j < i + halfBlock; j++, k += step) {
          const l = j + halfBlock;
          const wr = this.cosT[k];
          const wi = this.sinT[k];
          const tr = re[l] * wr - im[l] * wi;
          const ti = re[l] * wi + im[l] * wr;
          re[l] = re[j] - tr;
          im[l] = im[j] - ti;
          re[j] += tr;
          im[j] += ti;
        }
      }
    }
  }
  inverse(re, im) {
    for (let i = 0; i < this.size; i++) im[i] = -im[i];
    this.forward(re, im);
    const inv = 1 / this.size;
    for (let i = 0; i < this.size; i++) {
      re[i] *= inv;
      im[i] = -im[i] * inv;
    }
  }
}

function hannWindow(N) {
  const w = new Float32Array(N);
  for (let i = 0; i < N; i++) w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (N - 1)));
  return w;
}

// Accumulate average magnitude spectrum over a long signal.
function analyzeSpectrum(signal, fftSize = 4096) {
  const fft = new FFT(fftSize);
  const win = hannWindow(fftSize);
  const half = fftSize >> 1;
  const hop = fftSize >> 2;
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);
  const acc = new Float32Array(half + 1);
  let frames = 0;
  for (let pos = 0; pos + fftSize <= signal.length; pos += hop) {
    for (let i = 0; i < fftSize; i++) {
      re[i] = signal[pos + i] * win[i];
      im[i] = 0;
    }
    fft.forward(re, im);
    for (let k = 0; k <= half; k++) {
      acc[k] += Math.sqrt(re[k] * re[k] + im[k] * im[k]);
    }
    frames++;
  }
  if (frames > 0) {
    for (let k = 0; k <= half; k++) acc[k] /= frames;
  }
  return acc;
}

// Apply the match curve to a signal (mirrors the worklet exactly).
function applyMatchCurve(signal, curve, fftSize, opts = {}) {
  const fft = new FFT(fftSize);
  const win = hannWindow(fftSize);
  const halfN = fftSize >> 1;
  const hop = fftSize >> 2;
  const windowNorm = 2 / 3;
  const amount = opts.amount != null ? opts.amount : 1;
  const gainMakeup = opts.gainMakeup != null ? opts.gainMakeup : 1;
  const loBin = opts.loBin != null ? opts.loBin : 0;
  const hiBin = opts.hiBin != null ? opts.hiBin : halfN;

  const out = new Float32Array(signal.length);
  const ola = new Float32Array(fftSize);
  const re = new Float32Array(fftSize);
  const im = new Float32Array(fftSize);

  let writePos = 0;
  for (let pos = 0; pos + fftSize <= signal.length; pos += hop) {
    for (let i = 0; i < fftSize; i++) {
      re[i] = signal[pos + i] * win[i];
      im[i] = 0;
    }
    fft.forward(re, im);
    for (let k = 0; k <= halfN; k++) {
      let g;
      if (k < loBin || k >= hiBin) g = 1;
      else {
        const c = curve[k] != null ? curve[k] : 1;
        const clamped = Math.min(64, Math.max(1 / 64, c));
        g = 1 + amount * (clamped - 1);
      }
      g *= gainMakeup;
      re[k] *= g;
      im[k] *= g;
      if (k > 0 && k < halfN) {
        re[fftSize - k] =  re[k];
        im[fftSize - k] = -im[k];
      }
    }
    fft.inverse(re, im);
    for (let i = 0; i < fftSize; i++) {
      const idx = pos + i;
      if (idx < out.length) {
        out[idx] += re[i] * win[i] * windowNorm;
      }
    }
    writePos = pos + hop;
  }
  void writePos;
  return out;
}

// Pink noise generator (Voss algorithm, simple & cheap)
function makePinkNoise(N, seed = 1) {
  const out = new Float32Array(N);
  const numRows = 16;
  const rows = new Float32Array(numRows);
  let runningSum = 0;
  // mulberry32 seeded RNG so the test is deterministic
  let s = seed >>> 0;
  const rand = () => {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  let counter = 0;
  for (let i = 0; i < N; i++) {
    counter++;
    let row = 0;
    while ((counter & (1 << row)) === 0 && row < numRows - 1) row++;
    runningSum -= rows[row];
    rows[row] = (rand() * 2 - 1) * 0.1;
    runningSum += rows[row];
    out[i] = runningSum + (rand() * 2 - 1) * 0.05;
  }
  // Normalize to ~unity peak
  let peak = 1e-9;
  for (let i = 0; i < N; i++) peak = Math.max(peak, Math.abs(out[i]));
  for (let i = 0; i < N; i++) out[i] /= peak;
  return out;
}

// One-pole low-pass filter (used as a known target spectral shape)
function onePoleLowPass(signal, fc, sampleRate) {
  const out = new Float32Array(signal.length);
  const dt = 1 / sampleRate;
  const rc = 1 / (2 * Math.PI * fc);
  const alpha = dt / (rc + dt);
  let y = 0;
  for (let i = 0; i < signal.length; i++) {
    y = y + alpha * (signal[i] - y);
    out[i] = y;
  }
  return out;
}

// Spectrum-match test: target = LP-filtered pink, source = pink. Match
// curve should approximate the LP shape, and the applied output's
// spectrum should be closer to target than the unprocessed source.
function spectrumMatchTest() {
  const sampleRate = 44100;
  const fftSize = 4096;
  const halfN = fftSize >> 1;
  const dur = 2.0;
  const N = Math.floor(sampleRate * dur);

  const pink = makePinkNoise(N, 0xC0FFEE);
  const fcLow = 1000; // 1 kHz one-pole LP
  const targetSig = onePoleLowPass(pink, fcLow, sampleRate);

  const targetMag = analyzeSpectrum(targetSig, fftSize);
  const sourceMag = analyzeSpectrum(pink,      fftSize);

  // Smooth + divide
  const targetSm = smoothCurveOctave(targetMag, sampleRate, 1 / 3);
  const sourceSm = smoothCurveOctave(sourceMag, sampleRate, 1 / 3);
  const matchCurve = computeMatchCurve(targetSm, sourceSm);

  // Sanity: at bin near 5 kHz, curve should be << 1 (LP rolloff)
  const bin5k = Math.round((5000 / (sampleRate / 2)) * halfN);
  const bin500 = Math.round((500 / (sampleRate / 2)) * halfN);
  const gainAt5k_dB  = 20 * Math.log10(Math.max(1e-9, matchCurve[bin5k]));
  const gainAt500_dB = 20 * Math.log10(Math.max(1e-9, matchCurve[bin500]));
  // 1 kHz LP: at 5 kHz expect ~ -14 dB attenuation; at 500 Hz roughly flat (≈ 0 dB).
  const curveShapeOK = gainAt5k_dB < -8 && gainAt500_dB > -3;
  if (!curveShapeOK) {
    return {
      pass: false,
      message: `match curve shape wrong: 500Hz=${gainAt500_dB.toFixed(1)}dB (want ~0), 5kHz=${gainAt5k_dB.toFixed(1)}dB (want < -8)`,
    };
  }

  // Apply curve to pink, compare resulting spectrum to target spectrum.
  const applied = applyMatchCurve(pink, matchCurve, fftSize, {
    amount: 1,
    gainMakeup: 1,
    loBin: 0,
    hiBin: halfN,
  });
  const appliedMag   = analyzeSpectrum(applied, fftSize);
  const appliedMagSm = smoothCurveOctave(appliedMag, sampleRate, 1 / 3);

  // Score: average dB error over [200 Hz .. 8 kHz]
  const lo = Math.round((200  / (sampleRate / 2)) * halfN);
  const hi = Math.round((8000 / (sampleRate / 2)) * halfN);
  let sumErrApplied = 0;
  let sumErrUnproc  = 0;
  let count = 0;
  for (let k = lo; k <= hi; k++) {
    const tDb = 20 * Math.log10(Math.max(1e-9, targetSm[k]));
    const aDb = 20 * Math.log10(Math.max(1e-9, appliedMagSm[k]));
    const sDb = 20 * Math.log10(Math.max(1e-9, sourceSm[k]));
    sumErrApplied += Math.abs(tDb - aDb);
    sumErrUnproc  += Math.abs(tDb - sDb);
    count++;
  }
  const meanApplied = sumErrApplied / count;
  const meanUnproc  = sumErrUnproc  / count;

  // Match should beat unprocessed by a wide margin and itself land within ~3 dB.
  const beatsUnproc   = meanApplied < meanUnproc - 5;
  const closeAbsolute = meanApplied < 5; // dB
  const pass = beatsUnproc && closeAbsolute;
  return {
    pass,
    meanAppliedDb: meanApplied,
    meanUnprocDb: meanUnproc,
    gainAt500_dB,
    gainAt5k_dB,
    message: pass
      ? `spectrum match OK — applied avg err ${meanApplied.toFixed(2)} dB vs unprocessed ${meanUnproc.toFixed(2)} dB (curve@500Hz=${gainAt500_dB.toFixed(1)}, curve@5kHz=${gainAt5k_dB.toFixed(1)})`
      : `spectrum match FAIL — applied avg err ${meanApplied.toFixed(2)} dB, unprocessed ${meanUnproc.toFixed(2)} dB (need < unproc-5 AND <5)`,
  };
}

// ── Top-level runner ────────────────────────────────────────────────────────

export async function runAll() {
  const r1 = builderSmokeTest();
  const r2 = helperSanityTest();
  const r3 = spectrumMatchTest();
  return {
    builderSmoke: r1,
    helperSanity: r2,
    spectrumMatch: r3,
    pass: r1.pass && r2.pass && r3.pass,
  };
}

// ── Test-framework hooks ────────────────────────────────────────────────────

if (typeof describe === 'function' && typeof it === 'function') {
  describe('R13 match_eq', () => {
    it('builder returns shape + binds params under fallback', () => {
      const r = builderSmokeTest();
      // eslint-disable-next-line no-console
      console.log('[R13 match_eq]', r.message);
      // eslint-disable-next-line no-undef
      expect(r.pass).toBe(true);
    });
    it('host helpers (smoothing + match) are numerically sane', () => {
      const r = helperSanityTest();
      // eslint-disable-next-line no-console
      console.log('[R13 match_eq]', r.message);
      // eslint-disable-next-line no-undef
      expect(r.pass).toBe(true);
    });
    it('match curve on pink-noise + LP target approximates target spectrum', () => {
      const r = spectrumMatchTest();
      // eslint-disable-next-line no-console
      console.log('[R13 match_eq]', r.message);
      // eslint-disable-next-line no-undef
      expect(r.pass).toBe(true);
    });
  });
}

// Direct-execution path (so `node tests/r13_match_eq.test.js` works in CI
// without a runner).
const isMain = (() => {
  try {
    return typeof process !== 'undefined'
      && process.argv && process.argv[1]
      && import.meta && import.meta.url
      && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/').split('/').pop());
  } catch (e) { return false; }
})();

if (isMain) {
  runAll().then((res) => {
    // eslint-disable-next-line no-console
    console.log('builderSmoke :', res.builderSmoke.message);
    // eslint-disable-next-line no-console
    console.log('helperSanity :', res.helperSanity.message);
    // eslint-disable-next-line no-console
    console.log('spectrumMatch:', res.spectrumMatch.message);
    if (!res.pass) {
      // eslint-disable-next-line no-console
      console.error('FAIL');
      if (typeof process !== 'undefined') process.exit(1);
    } else {
      // eslint-disable-next-line no-console
      console.log('PASS');
    }
  }).catch((err) => {
    // eslint-disable-next-line no-console
    console.error('test threw:', err);
    if (typeof process !== 'undefined') process.exit(1);
  });
}
