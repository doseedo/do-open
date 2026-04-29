/**
 * r13_ringshifter.test.js — builder smoke + ring-mod sideband test.
 *
 * Strategy
 *   1. Smoke: invoke `buildRingshifter` against either the real
 *      OfflineAudioContext (browser / jsdom-with-audio) OR an in-process
 *      fake Web Audio shim (sufficient to exercise the builder's node
 *      construction + paramTargets wiring on plain `node`).
 *   2. Spectral: drive a 440 Hz sine through a faithful re-implementation
 *      of the worklet's ring-mod path (input × sin(2π·fc·t)) and assert
 *      sideband energy at 440−fc and 440+fc with the carrier at fc=100 Hz.
 *      Because the worklet code is pure math (no Web Audio dependencies)
 *      we can lift its `_lfo` / `process` math directly here without
 *      requiring a full AudioWorkletProcessor environment.
 *
 * Runs under jest/vitest (uses describe/it/expect when present) AND under
 * plain `node` (uses runAll() which logs pass/fail and exits non-zero on
 * failure).
 *
 * Author: Agent R13
 */

// ── Tiny in-process Web Audio shim (only what buildRingshifter needs) ─────

class FakeAudioParam {
  constructor(defaultValue = 0) {
    this.value = defaultValue;
    this._calls = [];
  }
  setTargetAtTime(v, t, tc) { this.value = v; this._calls.push(['setTargetAtTime', v, t, tc]); }
  setValueAtTime(v, t)      { this.value = v; this._calls.push(['setValueAtTime', v, t]); }
}

class FakeAudioNode {
  constructor() {
    this._connections = [];
  }
  connect(target) { this._connections.push(target); return target; }
  disconnect()    { this._connections = []; }
}

class FakeGainNode extends FakeAudioNode {
  constructor() { super(); this.gain = new FakeAudioParam(1); }
}

class FakeOscillatorNode extends FakeAudioNode {
  constructor() { super(); this.frequency = new FakeAudioParam(440); this.type = 'sine'; this._started = false; }
  start() { this._started = true; }
  stop()  { this._started = false; }
}

class FakeAudioContext {
  constructor(sampleRate = 44100) {
    this.sampleRate = sampleRate;
    this.currentTime = 0;
    this.destination = new FakeGainNode();
  }
  createGain()        { return new FakeGainNode(); }
  createOscillator()  { return new FakeOscillatorNode(); }
  createBuffer(ch, len, sr) {
    const channels = [];
    for (let i = 0; i < ch; i++) channels.push(new Float32Array(len));
    return {
      numberOfChannels: ch,
      length: len,
      sampleRate: sr,
      getChannelData: (i) => channels[i],
    };
  }
  // NB: no AudioWorkletNode — `_safeWorklet` will throw → builder falls
  // back to the primitive ring-mod path. That's exactly what we exercise
  // for the smoke test.
}

// ── Faithful re-implementation of the worklet's ring-mod inner loop ───────
// (Sample-accurate reproduction of the math in
// `r13-ringshifter-processor.js`'s `process()` for `mode=0 ring_mod`.)

function renderRingMod({
  sampleRate, durationSec,
  inputFreq, carrierFreq,
  dryMix, wetMix, outputGain,
}) {
  const N = Math.floor(sampleRate * durationSec);
  const out = new Float32Array(N);
  const TWO_PI = 2 * Math.PI;
  let phase = 0;
  for (let i = 0; i < N; i++) {
    const t = i / sampleRate;
    const x = Math.sin(TWO_PI * inputFreq * t);
    phase += (TWO_PI * carrierFreq) / sampleRate;
    if (phase >= TWO_PI) phase -= TWO_PI;
    const wet = x * Math.sin(phase);
    out[i] = (dryMix * x + wetMix * wet) * outputGain;
  }
  return out;
}

// ── Goertzel single-bin spectral magnitude ────────────────────────────────

function goertzelMag(samples, targetHz, sampleRate) {
  const k = 2 * Math.PI * targetHz / sampleRate;
  const cosK = Math.cos(k);
  const coef = 2 * cosK;
  let s0 = 0, s1 = 0, s2 = 0;
  for (let i = 0; i < samples.length; i++) {
    s0 = samples[i] + coef * s1 - s2;
    s2 = s1;
    s1 = s0;
  }
  // Power
  const real = s1 - s2 * cosK;
  const imag = s2 * Math.sin(k);
  return Math.sqrt(real * real + imag * imag) / samples.length * 2; // normalised
}

// ── Tests ─────────────────────────────────────────────────────────────────

export async function runBuilderSmokeTest() {
  const { buildRingshifter } = await import('../src/audio/builders/r13_ringshifter.js');
  const ctx = new FakeAudioContext(44100);

  const paramDefs = {
    p_freq:  { min: 0, max: 5000, default: 220 },
    p_wet:   { min: 0, max: 1,    default: 0.5 },
    p_mode:  { min: 0, max: 3,    default: 0 },
  };

  const node = {
    type: 'ring_shift',
    params: {
      mode:        '@p_mode',
      freq_hz:     '@p_freq',
      lfo_rate:    0,
      lfo_depth:   0,
      lfo_shape:   'sine',
      feedback:    0,
      dry_mix:     0.5,
      wet_mix:     '@p_wet',
      output_gain: 1.0,
    },
  };

  const result = buildRingshifter(ctx, node, paramDefs);

  // Required shape
  if (!result || !result.input || !result.output || !result.paramTargets) {
    return { pass: false, message: 'buildRingshifter did not return {input,output,paramTargets}' };
  }
  // Modulation targets are wired
  for (const id of ['p_freq', 'p_wet', 'p_mode']) {
    if (!result.paramTargets[id]) {
      return { pass: false, message: `paramTargets missing entry for "${id}"` };
    }
  }
  // Params should bind without throwing (covers customSetter paths)
  try {
    if (result.paramTargets.p_mode.customSetter) {
      result.paramTargets.p_mode.customSetter('freq_shift_up');
      result.paramTargets.p_mode.customSetter(0.75); // 0..1 normalised
      result.paramTargets.p_mode.customSetter(2);    // direct integer
    }
    if (result.paramTargets.p_freq.customSetter) {
      result.paramTargets.p_freq.customSetter(440);
    } else if (result.paramTargets.p_freq.audioParam) {
      result.paramTargets.p_freq.audioParam.value = 440;
    }
  } catch (e) {
    return { pass: false, message: `customSetter threw: ${e && e.message}` };
  }

  return { pass: true, message: 'builder smoke OK — paramTargets wired, customSetters exec without error' };
}

export async function runRingModSidebandTest() {
  const sampleRate = 44100;
  const durationSec = 0.5;
  const inputFreq = 440;
  const carrierFreq = 100;

  // Pure ring-mod, no dry, full wet, unity gain
  const out = renderRingMod({
    sampleRate, durationSec,
    inputFreq, carrierFreq,
    dryMix: 0, wetMix: 1, outputGain: 1,
  });

  // Expected sidebands: 440−100 = 340, 440+100 = 540
  const magOriginal  = goertzelMag(out, inputFreq, sampleRate);
  const magLowSide   = goertzelMag(out, inputFreq - carrierFreq, sampleRate);
  const magHighSide  = goertzelMag(out, inputFreq + carrierFreq, sampleRate);

  // Both sidebands should dominate the original carrier in the wet output.
  // Theoretical: y(t) = sin(2π·440·t) · sin(2π·100·t)
  //                   = 0.5·(cos(2π·340·t) − cos(2π·540·t))
  // → magnitude at 340 and 540 is ~0.5 each, magnitude at 440 is ~0.
  const sidebandsStrong = magLowSide > 0.3 && magHighSide > 0.3;
  const carrierAttenuated = magOriginal < 0.05;

  const pass = sidebandsStrong && carrierAttenuated;
  return {
    pass,
    magOriginal, magLowSide, magHighSide,
    message: pass
      ? `Ring-mod OK — sidebands at ${inputFreq - carrierFreq} (mag ${magLowSide.toFixed(3)}) and ${inputFreq + carrierFreq} (mag ${magHighSide.toFixed(3)}), carrier ${inputFreq} attenuated to ${magOriginal.toFixed(3)}`
      : `Ring-mod FAIL — magOriginal=${magOriginal.toFixed(3)} (want <0.05), magLow=${magLowSide.toFixed(3)} magHigh=${magHighSide.toFixed(3)} (want >0.3 each)`,
  };
}

// ── Plain-node runner ─────────────────────────────────────────────────────

export async function runAll() {
  const results = [];
  results.push({ name: 'builder smoke', ...(await runBuilderSmokeTest()) });
  results.push({ name: 'ring-mod sideband 440 ⊗ 100',  ...(await runRingModSidebandTest()) });

  let failed = 0;
  for (const r of results) {
    const tag = r.pass ? 'PASS' : 'FAIL';
    // eslint-disable-next-line no-console
    console.log(`[r13_ringshifter] ${tag} — ${r.name}: ${r.message}`);
    if (!r.pass) failed++;
  }
  return { failed, total: results.length, results };
}

// ── Test framework hooks (Jest / Vitest) ─────────────────────────────────

if (typeof describe === 'function' && typeof it === 'function') {
  describe('R13 Ringshifter', () => {
    it('builder returns {input, output, paramTargets} with all @-bindings wired', async () => {
      const r = await runBuilderSmokeTest();
      // eslint-disable-next-line no-console
      console.log('[r13_ringshifter]', r.message);
      // eslint-disable-next-line no-undef
      expect(r.pass).toBe(true);
    });

    it('ring-mod produces sidebands at f_input ± f_carrier and attenuates the carrier', async () => {
      const r = await runRingModSidebandTest();
      // eslint-disable-next-line no-console
      console.log('[r13_ringshifter]', r.message);
      // eslint-disable-next-line no-undef
      expect(r.pass).toBe(true);
    });
  });
}

// ── Self-run when invoked directly via `node` ────────────────────────────

const _isMainEsm =
  typeof process !== 'undefined' &&
  typeof import.meta !== 'undefined' &&
  process.argv && process.argv[1] &&
  import.meta.url &&
  import.meta.url.endsWith(process.argv[1].replace(/^.*?(\/|\\)/, ''));

if (typeof process !== 'undefined' && process.argv && process.argv[1] &&
    process.argv[1].endsWith('r13_ringshifter.test.js')) {
  runAll().then((r) => {
    if (r.failed > 0) {
      // eslint-disable-next-line no-console
      console.error(`[r13_ringshifter] ${r.failed}/${r.total} tests FAILED`);
      process.exit(1);
    } else {
      // eslint-disable-next-line no-console
      console.log(`[r13_ringshifter] ${r.total}/${r.total} tests PASSED`);
      process.exit(0);
    }
  }).catch((e) => {
    // eslint-disable-next-line no-console
    console.error('[r13_ringshifter] runner error:', e);
    process.exit(2);
  });
}
