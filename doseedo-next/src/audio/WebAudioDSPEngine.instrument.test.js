/**
 * WebAudioDSPEngine — instrument-mode (R6 VoiceManager) regression test.
 *
 * Renders 1 second of audio with a 4-voice C-major chord through an
 * oscillator → lowpass → adsr voice template, asserts non-zero RMS / peak.
 *
 * The engine's `_ensureContext()` constructs a real `window.AudioContext` —
 * not available in node/jest. We bypass it by directly injecting an
 * `OfflineAudioContext` onto `engine.ctx` and skipping the worklet boot
 * (worklet helpers no-op when audioWorklet is absent anyway).
 *
 * Mirrors the test harness pattern from VoiceManager.test.js.
 */

import WebAudioDSPEngine from './WebAudioDSPEngine.js';

const SAMPLE_RATE = 48000;

function rms(buffer) {
  let sum = 0;
  let n = 0;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      sum += data[i] * data[i];
      n++;
    }
  }
  return Math.sqrt(sum / Math.max(1, n));
}

function peak(buffer) {
  let m = 0;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = 0; i < data.length; i++) {
      const a = Math.abs(data[i]);
      if (a > m) m = a;
    }
  }
  return m;
}

// dspGraph for a 4-voice subtractive synth: oscillator → lowpass → adsr → out
const SUBTRACTIVE_INSTRUMENT_CONFIG = {
  pluginType: 'instrument',
  parameters: [
    { id: 'cutoff',     min: 20,    max: 20000, default: 2000, skew: 0.25 },
    { id: 'resonance',  min: 0.5,   max: 12,    default: 1 },
    { id: 'amp_attack', min: 0.001, max: 2,     default: 0.01 },
    { id: 'amp_decay',  min: 0.001, max: 4,     default: 0.1 },
    { id: 'amp_sustain', min: 0,    max: 1,     default: 0.7 },
    { id: 'amp_release', min: 0.001, max: 4,    default: 0.2 },
  ],
  voice: { polyphony: 4, mode: 'poly', stealing: 'oldest' },
  dspGraph: {
    nodes: [
      { id: 'osc1',   type: 'oscillator', params: { waveform: 'sawtooth', gain: 0.4 } },
      { id: 'filt',   type: 'lowpass',    params: { cutoff: '@cutoff', resonance: '@resonance' } },
      { id: 'env',    type: 'adsr',       params: {
        attack: '@amp_attack', decay: '@amp_decay',
        sustain: '@amp_sustain', release: '@amp_release',
      } },
      { id: 'output', type: 'output' },
    ],
    edges: [
      { source: 'osc1', target: 'filt' },
      { source: 'filt', target: 'env' },
      { source: 'env',  target: 'output' },
    ],
  },
};

/**
 * Construct an engine, skip its real-AudioContext bootstrap, and wire its
 * `masterGain` into an injected OfflineAudioContext's `destination`. We
 * monkey-patch `_ensureContext` (no-op, ctx already set) and skip
 * `_ensurePhase1Worklets` so test runs deterministically without worklets.
 */
function makeOfflineEngine(config, offlineCtx) {
  const engine = new WebAudioDSPEngine(config);
  engine.ctx = offlineCtx;
  // No-op the bootstrappers — ctx is already wired and we don't need worklets
  // for the simple oscillator + filter + adsr graph in this test.
  engine._ensureContext = () => {};
  engine._ensurePhase1Worklets = () => {};
  return engine;
}

function attachToDestination(engine) {
  // The engine builds masterGain inside _buildGraph(), then connects it to
  // ctx.destination via masterGain → analyser → destination. No work needed
  // here — we just trigger the build.
  engine._buildGraph();
}

describe('WebAudioDSPEngine — R6 instrument mode', () => {
  test('isInstrument detects oscillator-typed graph nodes', () => {
    const engine = new WebAudioDSPEngine(SUBTRACTIVE_INSTRUMENT_CONFIG);
    expect(engine.isInstrument).toBe(true);
  });

  test('extracts a per-voice template separating sources/envelopes from FX', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const cfg = {
      ...SUBTRACTIVE_INSTRUMENT_CONFIG,
      dspGraph: {
        nodes: [
          { id: 'osc1', type: 'oscillator', params: { waveform: 'sawtooth' } },
          { id: 'env',  type: 'adsr',       params: {} },
          // A reverb is global FX (per-voice category set excludes it)
          { id: 'verb', type: 'reverb',     params: { mix: 0.2 } },
          { id: 'output', type: 'output' },
        ],
        edges: [
          { source: 'osc1', target: 'env' },
          { source: 'env',  target: 'verb' },
          { source: 'verb', target: 'output' },
        ],
      },
    };
    const engine = makeOfflineEngine(cfg, ctx);
    const { voiceTemplate, globalFx } = engine._extractVoiceTemplate();
    const voiceIds = voiceTemplate.nodes.map(n => n.id);
    expect(voiceIds).toContain('osc1');
    expect(voiceIds).toContain('env');
    expect(voiceIds).toContain('output');
    expect(voiceIds).not.toContain('verb');
    expect(globalFx.map(n => n.id)).toEqual(['verb']);
  });

  test('renders non-zero audio for a 4-voice C-major chord', async () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1.0, SAMPLE_RATE);
    const engine = makeOfflineEngine(SUBTRACTIVE_INSTRUMENT_CONFIG, ctx);
    attachToDestination(engine);

    // Trigger 4-note chord at t=0.05
    const t0 = 0.05;
    [60, 64, 67, 72].forEach(n => engine.voiceManager.noteOn(n, 0.7, t0));
    [60, 64, 67, 72].forEach(n => engine.voiceManager.noteOff(n, 0.6));

    const buf = await ctx.startRendering();
    const rmsLevel = rms(buf);
    const peakLevel = peak(buf);
    expect(rmsLevel).toBeGreaterThan(1e-4);
    expect(peakLevel).toBeGreaterThan(1e-3);
  });

  test('engine.noteOn forwards to VoiceManager (no voiceId returned for tracking)', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const engine = makeOfflineEngine(SUBTRACTIVE_INSTRUMENT_CONFIG, ctx);
    attachToDestination(engine);

    expect(engine.voiceManager).toBeDefined();
    expect(engine.voiceManager.voiceCount()).toBe(0);

    engine.noteOn(60, 0.8);
    engine.noteOn(64, 0.8);
    expect(engine.voiceManager.voiceCount()).toBe(2);

    // R6: noteOff is keyed by MIDI note number (not voiceId)
    engine.noteOff(60);
    // Voice released → activeNotes no longer contains 60
    expect(engine.voiceManager.activeNotes()).not.toContain(60);
    expect(engine.voiceManager.activeNotes()).toContain(64);
  });

  test('engine.setParameter forwards scaled value to VoiceManager.setParam', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const engine = makeOfflineEngine(SUBTRACTIVE_INSTRUMENT_CONFIG, ctx);
    attachToDestination(engine);

    const calls = [];
    const origSetParam = engine.voiceManager.setParam.bind(engine.voiceManager);
    engine.voiceManager.setParam = (id, v) => {
      calls.push([id, v]);
      origSetParam(id, v);
    };

    // normalized 0.5 on cutoff (skew=0.25, min=20, max=20000) → ~ 0.5^4 * 19980 + 20
    engine.setParameter('cutoff', 0.5);
    const cutoffCall = calls.find(c => c[0] === 'cutoff');
    expect(cutoffCall).toBeDefined();
    // Sanity check: scaled value should land somewhere between min and max.
    expect(cutoffCall[1]).toBeGreaterThan(20);
    expect(cutoffCall[1]).toBeLessThanOrEqual(20000);
  });

  test('setSustain delegates to VoiceManager', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const engine = makeOfflineEngine(SUBTRACTIVE_INSTRUMENT_CONFIG, ctx);
    attachToDestination(engine);

    engine.noteOn(60, 0.8);
    engine.setSustain(true);
    engine.noteOff(60);
    // Sustained → voice still alive (deferred release)
    expect(engine.voiceManager._allVoices[0].released).toBe(false);
    engine.setSustain(false);
    expect(engine.voiceManager._allVoices[0].released).toBe(true);
  });

  test('bindCC routes through MidiInput to the voice manager', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const engine = makeOfflineEngine(SUBTRACTIVE_INSTRUMENT_CONFIG, ctx);
    attachToDestination(engine);

    expect(engine.midiInput).toBeDefined();
    const calls = [];
    engine.voiceManager.setParam = (id, v) => calls.push([id, v]);
    engine.bindCC(74, 'cutoff');
    engine.midiInput.feed({ type: 'cc', cc: 74, value: 0.42 });
    expect(calls).toEqual([['cutoff', 0.42]]);
  });

  test('auto-binds midi_cc graph nodes during build', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const cfg = {
      ...SUBTRACTIVE_INSTRUMENT_CONFIG,
      dspGraph: {
        ...SUBTRACTIVE_INSTRUMENT_CONFIG.dspGraph,
        nodes: [
          ...SUBTRACTIVE_INSTRUMENT_CONFIG.dspGraph.nodes,
          { id: 'cc1', type: 'midi_cc', params: { cc_number: 71, target: 'resonance', min_val: 0.5, max_val: 12 } },
        ],
      },
    };
    const engine = makeOfflineEngine(cfg, ctx);
    attachToDestination(engine);

    // Drive a CC#71 event and verify it lands on resonance via the
    // midiInput's auto-binding (built by _buildInstrumentGraphV2).
    const calls = [];
    engine.voiceManager.setParam = (id, v) => calls.push([id, v]);
    engine.midiInput.feed({ type: 'cc', cc: 71, value: 0.5 });
    expect(calls).toContainEqual(['resonance', 0.5 * (12 - 0.5) + 0.5]);
  });

  test('legacy fallback path activates when config.voice.enabled === false', () => {
    const cfg = {
      ...SUBTRACTIVE_INSTRUMENT_CONFIG,
      voice: { enabled: false },
    };
    const engine = new WebAudioDSPEngine(cfg);
    expect(engine.isInstrument).toBe(true);
    expect(engine._useVoiceManagerPath).toBe(false);
  });
});

describe('WebAudioDSPEngine — non-instrument graph (R6 regression)', () => {
  test('effect-only chain still uses _buildGraphFromNodes path', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const fxConfig = {
      pluginType: 'effect',
      parameters: [
        { id: 'cutoff', min: 20, max: 20000, default: 1000, skew: 0.25 },
      ],
      dspGraph: {
        nodes: [
          { id: 'input',  type: 'input' },
          { id: 'filt',   type: 'lowpass', params: { cutoff: '@cutoff' } },
          { id: 'output', type: 'output' },
        ],
        edges: [
          { source: 'input', target: 'filt' },
          { source: 'filt',  target: 'output' },
        ],
      },
    };
    const engine = makeOfflineEngine(fxConfig, ctx);
    expect(engine.isInstrument).toBe(false);
    expect(engine._useVoiceManagerPath).toBe(false);
    attachToDestination(engine);
    // Graph built without VoiceManager
    expect(engine.voiceManager).toBeFalsy();
    expect(engine.masterGain).toBeDefined();
  });
});
