/**
 * VoiceManager — offline-render unit tests.
 *
 * Renders 1s of audio with a 4-voice synth playing a chord and asserts the
 * output is non-zero. Also exercises mono/legato modes, sustain pedal, and
 * voice stealing under polyphony pressure.
 *
 * Uses OfflineAudioContext so we can run in CI without a real audio device.
 */

import VoiceManager from './VoiceManager.js';
import MidiInput from './MidiInput.js';

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

// Simple 4-voice subtractive template:
//   osc → filter → adsr → output
const SIMPLE_TEMPLATE = {
  nodes: [
    { id: 'osc1', type: 'oscillator', params: { waveform: 'sawtooth', gain: 0.4 } },
    { id: 'filt', type: 'lowpass', params: { cutoff: 2000, resonance: 1 } },
    { id: 'env',  type: 'adsr', params: { attack: 0.01, decay: 0.1, sustain: 0.7, release: 0.2 } },
    { id: 'output', type: 'output' },
  ],
  edges: [
    { source: 'osc1', target: 'filt' },
    { source: 'filt', target: 'env' },
    { source: 'env',  target: 'output' },
  ],
};

// More complex ES2-like template: dual osc + sub + filter + adsr
const ES2_LIKE_TEMPLATE = {
  nodes: [
    { id: 'osc1', type: 'oscillator', params: { waveform: 'sawtooth', gain: 0.35, detune: -7 } },
    { id: 'osc2', type: 'oscillator', params: { waveform: 'square',   gain: 0.30, detune: +7 } },
    { id: 'sub',  type: 'sub_oscillator', params: { waveform: 'sine', octave_offset: -1, gain: 0.25 } },
    { id: 'mix',  type: 'gain', params: { gain: 1 } },
    { id: 'filt', type: 'lowpass', params: { cutoff: 1800, resonance: 2 } },
    { id: 'env',  type: 'adsr', params: { attack: 0.005, decay: 0.15, sustain: 0.6, release: 0.3 } },
    { id: 'output', type: 'output' },
  ],
  edges: [
    { source: 'osc1', target: 'mix' },
    { source: 'osc2', target: 'mix' },
    { source: 'sub',  target: 'mix' },
    { source: 'mix',  target: 'filt' },
    { source: 'filt', target: 'env' },
    { source: 'env',  target: 'output' },
  ],
};

describe('VoiceManager', () => {
  test('renders non-zero audio for a 4-voice C-major chord', async () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1.0, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8 });
    vm.output.connect(ctx.destination);

    // Trigger 4-note chord at t=0.05 (after some headroom)
    const t0 = 0.05;
    [60, 64, 67, 72].forEach(n => vm.noteOn(n, 0.7, t0));
    // Release at t=0.6
    [60, 64, 67, 72].forEach(n => vm.noteOff(n, 0.6));

    const buf = await ctx.startRendering();
    const rmsLevel = rms(buf);
    const peakLevel = peak(buf);
    expect(rmsLevel).toBeGreaterThan(1e-4);
    expect(peakLevel).toBeGreaterThan(1e-3);
  });

  test('voice count tracks active notes', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8 });
    expect(vm.voiceCount()).toBe(0);
    vm.noteOn(60, 0.8, 0.0);
    vm.noteOn(64, 0.8, 0.0);
    expect(vm.voiceCount()).toBe(2);
  });

  test('voice stealing kicks in beyond polyphony cap', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 4, stealing: 'oldest' });
    for (let n = 60; n < 66; n++) vm.noteOn(n, 0.8, 0.0);
    // 6 noteOns into a 4-voice manager → 2 should be stolen
    const liveCount = vm._allVoices.filter(v => !v.released && !v.stolen).length;
    expect(liveCount).toBeLessThanOrEqual(4);
    const stolen = vm._allVoices.filter(v => v.stolen).length;
    expect(stolen).toBeGreaterThanOrEqual(1);
  });

  test('sustain pedal defers releases', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 4 });
    vm.noteOn(60, 0.8, 0.0);
    vm.setSustain(true);
    vm.noteOff(60, 0.1);
    // Voice should still be active (not released) because pedal is down
    const v = vm._allVoices[0];
    expect(v.released).toBe(false);
    vm.setSustain(false);
    expect(v.released).toBe(true);
  });

  test('mono mode reuses a single voice', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8, mode: 'mono' });
    vm.noteOn(60, 0.8, 0.0);
    vm.noteOn(64, 0.8, 0.05);
    // Only one voice should be alive
    const live = vm._allVoices.filter(v => !v.released && !v.stolen);
    expect(live.length).toBe(1);
    expect(live[0].note).toBe(64);
  });

  test('legato re-pitches without retriggering envelope', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8, mode: 'legato' });
    vm.noteOn(60, 0.8, 0.0);
    const v = vm._allVoices[0];
    const startTime = v.startTime;
    vm.noteOn(64, 0.8, 0.05);
    // Same voice, re-pitched, but startTime should not have been reset
    expect(v.note).toBe(64);
    expect(v.startTime).toBe(startTime);
  });

  test('ES2-like multi-osc template renders audio', async () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1.0, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, ES2_LIKE_TEMPLATE, {}, { polyphony: 4 });
    vm.output.connect(ctx.destination);
    [48, 52, 55, 60].forEach(n => vm.noteOn(n, 0.7, 0.05));
    [48, 52, 55, 60].forEach(n => vm.noteOff(n, 0.6));
    const buf = await ctx.startRendering();
    expect(peak(buf)).toBeGreaterThan(1e-3);
  });
});

describe('MidiInput', () => {
  test('feed() routes noteon/noteoff to voice manager', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8 });
    const midi = new MidiInput(vm);
    midi.feed({ type: 'noteon', note: 60, velocity: 0.7, time: 0.0 });
    expect(vm.activeNotes()).toContain(60);
    midi.feed({ type: 'noteoff', note: 60, time: 0.1 });
    // After noteOff the note should leave the active map (voice still in
    // the all-list during release, but no longer keyed by note).
    expect(vm.activeNotes()).not.toContain(60);
  });

  test('CC routing forwards to setParam', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8 });
    const calls = [];
    vm.setParam = (id, v) => calls.push([id, v]);
    const midi = new MidiInput(vm);
    midi.bindCC(74, 'cutoff');
    midi.feed({ type: 'cc', cc: 74, value: 0.5 });
    expect(calls).toEqual([['cutoff', 0.5]]);
  });

  test('sustain CC defers releases', () => {
    const ctx = new OfflineAudioContext(2, SAMPLE_RATE * 1, SAMPLE_RATE);
    const vm = new VoiceManager(ctx, SIMPLE_TEMPLATE, {}, { polyphony: 8 });
    const midi = new MidiInput(vm);
    midi.feed({ type: 'noteon', note: 60, velocity: 0.7, time: 0.0 });
    midi.feed({ type: 'sustain', value: true });
    midi.feed({ type: 'noteoff', note: 60, time: 0.1 });
    expect(vm._allVoices[0].released).toBe(false);
    midi.feed({ type: 'sustain', value: false });
    expect(vm._allVoices[0].released).toBe(true);
  });
});
