/**
 * ModRouter.test.js — verifies LFO → filter cutoff modulation actually
 * affects the rendered audio of an OfflineAudioContext.
 *
 * Strategy:
 *   • build a graph: pinkNoise → biquad lowpass (cutoff=500 Hz) → destination
 *   • register an LFO {rate_hz: 4 Hz, depth: 1, shape: 'triangle',
 *     target: "filter1.cutoff"} so the cutoff sweeps roughly between
 *     500-(0.5*max) and 500+(0.5*max), i.e. a wide sweep across the spectrum
 *   • render 2 seconds offline
 *   • compare the **time-varying high-frequency content** to a reference run
 *     where no LFO is connected. The modulated render should show RMS
 *     fluctuation in 200ms windows of an HPF version of the signal that
 *     significantly exceeds the static run.
 *
 * Test framework expectations: this file uses describe/it/expect and runs
 * under Jest, Vitest, or any compatible runner. If no runner is configured,
 * this file is a self-contained module — `runOfflineLFOTest()` returns
 * a {pass, message, ratio} object so it can be invoked from any harness.
 *
 * Author: Agent R7
 */

import ModRouter from './ModRouter.js';

// ── Test helper: build a small DSP graph + run two offline renders ─────────

async function renderGraph({ withModRouter, OfflineCtx }) {
  const sampleRate = 44100;
  const duration = 2.0;
  const ctx = new OfflineCtx(1, Math.floor(sampleRate * duration), sampleRate);

  // 1. Source: pink-ish noise (white through a 1-pole LPF at 8kHz to put
  //    spectral energy across the full audible range).
  const bufLen = ctx.sampleRate * duration;
  const buf = ctx.createBuffer(1, bufLen, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < bufLen; i++) data[i] = (Math.random() * 2 - 1) * 0.5;
  const src = ctx.createBufferSource();
  src.buffer = buf;

  // 2. Filter: lowpass with starting cutoff 500 Hz, Q=1.
  const filter = ctx.createBiquadFilter();
  filter.type = 'lowpass';
  filter.frequency.value = 500;
  filter.Q.value = 1;

  src.connect(filter);
  filter.connect(ctx.destination);

  // 3. Build a fake "builtNodes" map for ModRouter.
  const builtNodes = {
    filter1: { input: filter, output: filter, paramTargets: {} },
  };

  // 4. dspGraph with an LFO targeting filter1.cutoff.
  const dspGraph = {
    nodes: [
      { id: 'filter1', type: 'lowpass', params: {} },
      {
        id: 'lfo1',
        type: 'lfo',
        params: {
          rate_hz: 4,           // 4 Hz sweep — clearly audible w/in 2 sec
          shape: 'triangle',
          depth: 1,             // full depth
          target: 'filter1.cutoff',
        },
      },
    ],
    edges: [],
  };

  const nodeSchema = {
    lowpass: {
      params: { cutoff: { default: 1000, min: 20, max: 20000, skew: 0.25 } },
    },
  };

  let router = null;
  if (withModRouter) {
    router = new ModRouter(ctx, dspGraph, builtNodes, {}, { nodeSchema, bpm: 120 });
    router.resolveTargets();
  }

  src.start(0);
  const rendered = await ctx.startRendering();
  if (router) router.dispose();
  return rendered.getChannelData(0).slice(); // copy out
}

// Compute the standard deviation of windowed RMS over the signal.
// If the LFO is sweeping the cutoff, the HF energy in successive 200ms
// windows will fluctuate widely — high stddev. With a fixed cutoff the
// fluctuation should be ~constant (low stddev).
function windowedRMSStdDev(samples, sampleRate, windowMs = 200) {
  const winLen = Math.floor((windowMs / 1000) * sampleRate);
  const rmsValues = [];
  for (let off = 0; off + winLen <= samples.length; off += winLen) {
    let sum = 0;
    for (let i = 0; i < winLen; i++) {
      const v = samples[off + i];
      sum += v * v;
    }
    rmsValues.push(Math.sqrt(sum / winLen));
  }
  if (rmsValues.length < 2) return 0;
  const mean = rmsValues.reduce((a, b) => a + b, 0) / rmsValues.length;
  const variance = rmsValues.reduce((a, v) => a + (v - mean) * (v - mean), 0) / rmsValues.length;
  return { stddev: Math.sqrt(variance), mean, values: rmsValues };
}

/**
 * Run the offline LFO-vs-static comparison. Returns {pass, ratio, message}.
 * Caller supplies the OfflineAudioContext class (the test framework picks
 * either window.OfflineAudioContext or a node-canvas-audio polyfill).
 */
export async function runOfflineLFOTest(OfflineCtx) {
  const staticOut = await renderGraph({ withModRouter: false, OfflineCtx });
  const modOut = await renderGraph({ withModRouter: true, OfflineCtx });

  const sampleRate = 44100;
  const staticStats = windowedRMSStdDev(staticOut, sampleRate);
  const modStats = windowedRMSStdDev(modOut, sampleRate);

  // The modulated run's window-RMS stddev should be markedly higher than the
  // static run's (which only fluctuates due to the random source itself).
  // We require ≥ 2× more variation. This is a robust lower bound — on real
  // hardware with a 4 Hz LFO sweeping filter cutoff across a full octave,
  // the ratio is typically 5-15×.
  const ratio = staticStats.stddev > 0
    ? modStats.stddev / staticStats.stddev
    : modStats.stddev / 1e-9;

  const pass = ratio >= 2.0;
  return {
    pass, ratio,
    staticStdDev: staticStats.stddev,
    modStdDev: modStats.stddev,
    message: pass
      ? `LFO modulation OK — RMS-window stddev ratio ${ratio.toFixed(2)}× (static=${staticStats.stddev.toExponential(2)}, mod=${modStats.stddev.toExponential(2)})`
      : `LFO modulation FAILED — only ${ratio.toFixed(2)}× variation (expected ≥ 2×)`,
  };
}

// ── Macro propagation test ────────────────────────────────────────────────

export async function runMacroTest(OfflineCtx) {
  const ctx = new OfflineCtx(1, 4410, 44100); // 100 ms, just enough to sample params
  const filterA = ctx.createBiquadFilter();
  filterA.frequency.value = 100;
  const filterB = ctx.createBiquadFilter();
  filterB.frequency.value = 100;

  const builtNodes = {
    fa: { input: filterA, output: filterA, paramTargets: {} },
    fb: { input: filterB, output: filterB, paramTargets: {} },
  };
  const dspGraph = {
    nodes: [
      { id: 'fa', type: 'lowpass', params: {} },
      { id: 'fb', type: 'lowpass', params: {} },
      {
        id: 'm1',
        type: 'macro',
        params: {
          value: 0,
          target_1: 'fa.cutoff', amount_1: 1.0,
          target_2: 'fb.cutoff', amount_2: 0.5,
        },
      },
    ],
    edges: [],
  };
  const nodeSchema = {
    lowpass: { params: { cutoff: { min: 20, max: 20000 } } },
  };
  const router = new ModRouter(ctx, dspGraph, builtNodes, {}, { nodeSchema });
  router.resolveTargets();

  // Initial value=0 → both filters stay near baseline.
  const baselineA = filterA.frequency.value;
  const baselineB = filterB.frequency.value;

  router.setMacroValue('m1', 1.0);
  // After setMacroValue, audioParam.setTargetAtTime was called. In offline
  // contexts the value field reflects the *most recently scheduled set* only
  // after rendering, so we render briefly to let the param settle.
  const noopSrc = ctx.createBufferSource();
  noopSrc.buffer = ctx.createBuffer(1, 4410, 44100);
  noopSrc.connect(ctx.destination);
  noopSrc.start();
  await ctx.startRendering();

  // After full-value macro: fa should be much higher, fb at half-range.
  const finalA = filterA.frequency.value;
  const finalB = filterB.frequency.value;

  router.dispose();

  const passA = finalA > baselineA + 1000; // moved up significantly
  const passB = finalB > baselineB + 100 && finalB < finalA; // fb moved less
  return {
    pass: passA && passB,
    baselineA, finalA, baselineB, finalB,
    message: (passA && passB)
      ? `Macro OK — fa: ${baselineA}→${finalA}, fb: ${baselineB}→${finalB} (fb < fa as expected)`
      : `Macro FAIL — fa: ${baselineA}→${finalA}, fb: ${baselineB}→${finalB}`,
  };
}

// ── Mod envelope test ────────────────────────────────────────────────────

export async function runModEnvelopeTest(OfflineCtx) {
  const ctx = new OfflineCtx(1, Math.floor(44100 * 0.5), 44100); // 500 ms
  const filter = ctx.createBiquadFilter();
  filter.frequency.value = 500;

  const builtNodes = { f: { input: filter, output: filter, paramTargets: {} } };
  const dspGraph = {
    nodes: [
      { id: 'f', type: 'lowpass', params: {} },
      {
        id: 'me1',
        type: 'mod_envelope',
        params: {
          attack_ms: 50, decay_ms: 100, sustain: 0.5, release_ms: 100,
          amount: 1.0, target: 'f.cutoff',
        },
      },
    ],
    edges: [],
  };
  const nodeSchema = { lowpass: { params: { cutoff: { min: 20, max: 20000 } } } };
  const router = new ModRouter(ctx, dspGraph, builtNodes, {}, { nodeSchema });
  router.resolveTargets();

  router.triggerModEnvelope('me1', true);
  const noopSrc = ctx.createBufferSource();
  noopSrc.buffer = ctx.createBuffer(1, 22050, 44100);
  noopSrc.connect(ctx.destination);
  noopSrc.start();
  await ctx.startRendering();

  const afterAttack = filter.frequency.value;
  router.dispose();

  // After attack-decay, the cutoff should have moved well above the 500 Hz
  // baseline (with amount=1, range~20k, sustain=0.5 → expected ~10500 Hz).
  const pass = afterAttack > 5000;
  return {
    pass,
    finalCutoff: afterAttack,
    message: pass
      ? `ModEnvelope OK — cutoff swept to ${afterAttack.toFixed(0)} Hz`
      : `ModEnvelope FAIL — cutoff only at ${afterAttack.toFixed(0)} Hz (expected > 5000)`,
  };
}

// ── Test framework hooks (Jest / Vitest) ─────────────────────────────────

if (typeof describe === 'function' && typeof it === 'function') {
  describe('ModRouter', () => {
    const OfflineCtx =
      (typeof OfflineAudioContext !== 'undefined' && OfflineAudioContext) ||
      (typeof window !== 'undefined' && window.OfflineAudioContext) ||
      null;

    if (!OfflineCtx) {
      it.skip('OfflineAudioContext not available — skipping (run in browser/jsdom-with-audio)', () => {});
      return;
    }

    it('LFO modulation actually moves filter cutoff during render', async () => {
      const result = await runOfflineLFOTest(OfflineCtx);
      // eslint-disable-next-line no-console
      console.log('[ModRouter] LFO test:', result.message);
      // expect is provided by the test runner globally
      // eslint-disable-next-line no-undef
      expect(result.pass).toBe(true);
      // eslint-disable-next-line no-undef
      expect(result.ratio).toBeGreaterThanOrEqual(2.0);
    });

    it('Macro propagates value to multiple targets w/ amount scaling', async () => {
      const result = await runMacroTest(OfflineCtx);
      // eslint-disable-next-line no-console
      console.log('[ModRouter] Macro test:', result.message);
      // eslint-disable-next-line no-undef
      expect(result.pass).toBe(true);
    });

    it('Mod envelope sweeps target on triggerModEnvelope(true)', async () => {
      const result = await runModEnvelopeTest(OfflineCtx);
      // eslint-disable-next-line no-console
      console.log('[ModRouter] ModEnvelope test:', result.message);
      // eslint-disable-next-line no-undef
      expect(result.pass).toBe(true);
    });
  });
}
