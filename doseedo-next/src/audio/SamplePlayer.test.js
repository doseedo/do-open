/**
 * SamplePlayer / Keymap / EXS24Reader — offline-render unit tests.
 *
 * Tests use OfflineAudioContext for deterministic CI runs (no audio device).
 * Pattern follows VoiceManager.test.js:
 *   - Build a synthetic AudioBuffer (sine wave) so we can verify pitch math.
 *   - Trigger SamplePlayer at note=rootNote (rate=1×) and note=rootNote+12
 *     (rate=2×) — measure how much of the source plays in a fixed window
 *     and assert the duration ratio is ~2:1.
 *   - Round-trip a synthetic .exs blob through EXS24Reader.parse and assert
 *     the parsed zones / sampleRefs match what we wrote.
 *   - Exercise Keymap zone selection + round-robin advancement.
 */

import SamplePlayer from './SamplePlayer.js';
import Keymap from './Keymap.js';
import EXS24Reader, { buildSyntheticExs } from './EXS24Reader.js';

const SAMPLE_RATE = 48000;

function makeSineBuffer(ctx, freq = 440, durationSec = 0.5) {
  const len = Math.floor(ctx.sampleRate * durationSec);
  const buf = ctx.createBuffer(1, len, ctx.sampleRate);
  const data = buf.getChannelData(0);
  for (let i = 0; i < len; i++) {
    data[i] = Math.sin(2 * Math.PI * freq * i / ctx.sampleRate) * 0.5;
  }
  return buf;
}

/**
 * Find the last sample index (in samples) where any channel exceeds `threshold`.
 * Used as a proxy for "how long did the source play before reaching its end".
 * Faster playbackRate → source ends sooner → last-active sample is earlier.
 */
function lastActiveSample(buffer, threshold = 1e-3) {
  let lastIdx = -1;
  for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
    const data = buffer.getChannelData(ch);
    for (let i = data.length - 1; i >= 0; i--) {
      if (Math.abs(data[i]) > threshold) {
        if (i > lastIdx) lastIdx = i;
        break;
      }
    }
  }
  return lastIdx;
}

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

describe('SamplePlayer', () => {
  test('renders pitched audio at rootNote (rate = 1x)', async () => {
    const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 1.0, SAMPLE_RATE);
    const sample = makeSineBuffer(ctx, 440, 0.5);
    const sp = new SamplePlayer(ctx, { sampleBuffer: sample, rootNote: 60 });
    sp.output.connect(ctx.destination);
    sp.play(60, 0.8, 0);
    const buf = await ctx.startRendering();
    expect(rms(buf)).toBeGreaterThan(1e-3);
  });

  test('octave-up note plays at 2x speed (source ends ~half as soon)', async () => {
    // Render 1.0s of audio at two pitches.
    const SOURCE_DUR = 0.5;
    const RENDER_DUR = 1.0;

    const renderAt = async (note) => {
      const ctx = new OfflineAudioContext(1, SAMPLE_RATE * RENDER_DUR, SAMPLE_RATE);
      const sample = makeSineBuffer(ctx, 440, SOURCE_DUR);
      const sp = new SamplePlayer(ctx, { sampleBuffer: sample, rootNote: 60 });
      sp.output.connect(ctx.destination);
      sp.play(note, 0.8, 0);
      const out = await ctx.startRendering();
      return out;
    };

    const bufBase  = await renderAt(60);       // rate = 1x → source plays for ~0.5s
    const bufHigh  = await renderAt(72);       // rate = 2x → source plays for ~0.25s

    const lastBase = lastActiveSample(bufBase);
    const lastHigh = lastActiveSample(bufHigh);

    expect(lastBase).toBeGreaterThan(0);
    expect(lastHigh).toBeGreaterThan(0);

    // High pitch should finish *earlier* than base — and the ratio should
    // be roughly 2:1 (allow ±25% slack for buffer source ramp-down).
    const ratio = lastBase / lastHigh;
    expect(ratio).toBeGreaterThan(1.5);
    expect(ratio).toBeLessThan(3.0);
  });

  test('reverse playback produces non-zero audio', async () => {
    const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 0.6, SAMPLE_RATE);
    const sample = makeSineBuffer(ctx, 440, 0.5);
    const sp = new SamplePlayer(ctx, { sampleBuffer: sample, rootNote: 60, reverse: true });
    sp.output.connect(ctx.destination);
    sp.play(60, 0.8, 0);
    const buf = await ctx.startRendering();
    expect(rms(buf)).toBeGreaterThan(1e-3);
  });

  test('release ramps voice gain down', async () => {
    const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 1.0, SAMPLE_RATE);
    const sample = makeSineBuffer(ctx, 440, 1.0);
    const sp = new SamplePlayer(ctx, { sampleBuffer: sample, rootNote: 60 });
    sp.output.connect(ctx.destination);
    const v = sp.play(60, 0.8, 0);
    v.release(0.1, 0.05);    // release at 0.1s with 50ms decay
    const buf = await ctx.startRendering();
    // Tail (after t=0.2s) should be effectively silent
    const data = buf.getChannelData(0);
    let tailSum = 0;
    let tailN = 0;
    for (let i = Math.floor(0.25 * SAMPLE_RATE); i < data.length; i++) {
      tailSum += data[i] * data[i];
      tailN++;
    }
    const tailRMS = Math.sqrt(tailSum / Math.max(1, tailN));
    expect(tailRMS).toBeLessThan(0.01);
  });

  test('velocity scales output gain', async () => {
    const renderAtVel = async (vel) => {
      const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 0.5, SAMPLE_RATE);
      const sample = makeSineBuffer(ctx, 440, 0.5);
      const sp = new SamplePlayer(ctx, { sampleBuffer: sample, rootNote: 60 });
      sp.output.connect(ctx.destination);
      sp.play(60, vel, 0);
      const out = await ctx.startRendering();
      return rms(out);
    };
    const r1 = await renderAtVel(1.0);
    const r2 = await renderAtVel(0.25);
    // Lower velocity → quieter output
    expect(r1).toBeGreaterThan(r2 * 2);
  });
});

describe('Keymap', () => {
  test('selectZones picks zones whose key+vel range matches', async () => {
    const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const buf = makeSineBuffer(ctx, 440, 0.1);
    const km = new Keymap(ctx, [
      { keyRange: [60, 72], velRange: [1, 64],   sampleBuffer: buf, rootNote: 60 },
      { keyRange: [60, 72], velRange: [65, 127], sampleBuffer: buf, rootNote: 60 },
      { keyRange: [40, 50], velRange: [1, 127],  sampleBuffer: buf, rootNote: 40 },
    ]);
    const lo = km.selectZones(64, 0.3);     // velocity-low zone
    expect(lo).toHaveLength(1);
    expect(lo[0].velRange[1]).toBe(64);

    const hi = km.selectZones(64, 0.9);     // velocity-high zone
    expect(hi).toHaveLength(1);
    expect(hi[0].velRange[0]).toBe(65);

    const out = km.selectZones(45, 0.5);    // separate key zone
    expect(out).toHaveLength(1);
    expect(out[0].rootNote).toBe(40);

    const none = km.selectZones(80, 0.5);   // out of range entirely
    expect(none).toHaveLength(0);
  });

  test('round-robin cycles through zones in a group', async () => {
    const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 0.1, SAMPLE_RATE);
    const buf = makeSineBuffer(ctx, 440, 0.1);
    const km = new Keymap(ctx, [
      { keyRange: [60, 60], velRange: [1, 127], sampleBuffer: buf, rootNote: 60, roundRobin: 1, gain: 0.1 },
      { keyRange: [60, 60], velRange: [1, 127], sampleBuffer: buf, rootNote: 60, roundRobin: 1, gain: 0.5 },
      { keyRange: [60, 60], velRange: [1, 127], sampleBuffer: buf, rootNote: 60, roundRobin: 1, gain: 0.9 },
    ]);
    const a = km.selectZones(60, 0.8)[0];
    const b = km.selectZones(60, 0.8)[0];
    const c = km.selectZones(60, 0.8)[0];
    const d = km.selectZones(60, 0.8)[0];
    // Cycle a→b→c→a (gain field is unique per zone)
    expect(a.gain).toBe(0.1);
    expect(b.gain).toBe(0.5);
    expect(c.gain).toBe(0.9);
    expect(d.gain).toBe(0.1);
  });

  test('trigger() returns voice handles and renders audio', async () => {
    const ctx = new OfflineAudioContext(1, SAMPLE_RATE * 0.5, SAMPLE_RATE);
    const buf = makeSineBuffer(ctx, 440, 0.3);
    const km = new Keymap(ctx, [
      { keyRange: [60, 72], velRange: [1, 127], sampleBuffer: buf, rootNote: 60 },
    ]);
    km.output.connect(ctx.destination);
    const handles = km.trigger(64, 0.7, 0);
    expect(handles).toHaveLength(1);
    const out = await ctx.startRendering();
    expect(rms(out)).toBeGreaterThan(1e-3);
  });
});

describe('EXS24Reader', () => {
  test('round-trips a synthetic 2-zone instrument', () => {
    const blob = buildSyntheticExs({
      name: 'TestInstr',
      version: 0x10000,
      zones: [
        {
          name: 'low',
          keyRange: [36, 59], velRange: [1, 64],
          rootNote: 48, fineTune: 5, coarseTune: 0,
          volume: 0.8, pan: -0.5,
          sampleStart: 0, sampleEnd: 44100,
          loopStart: 1000, loopEnd: 2000,
          loopOn: true, reverse: false,
          sampleIndex: 0, groupIndex: 0,
        },
        {
          name: 'high',
          keyRange: [60, 84], velRange: [65, 127],
          rootNote: 72, fineTune: -10, coarseTune: 1,
          volume: 1.0, pan: 0.5,
          sampleStart: 0, sampleEnd: 22050,
          loopStart: 0, loopEnd: 0,
          loopOn: false, reverse: true,
          sampleIndex: 1, groupIndex: 0,
        },
      ],
      groups: [
        { name: 'main', polyphony: 16, exclusiveGroup: 0, volume: 0, pan: 0 },
      ],
      sampleRefs: [
        { name: 'kick.wav',  filename: 'kick.wav',  length: 44100, sampleRate: 44100, bitDepth: 16 },
        { name: 'snare.wav', filename: 'snare.wav', length: 22050, sampleRate: 44100, bitDepth: 24 },
      ],
    });

    const parsed = EXS24Reader.parse(blob);

    expect(parsed.name).toBe('TestInstr');
    expect(parsed.zones).toHaveLength(2);
    expect(parsed.groups).toHaveLength(1);
    expect(parsed.sampleRefs).toHaveLength(2);

    const z0 = parsed.zones[0];
    expect(z0.keyRange).toEqual([36, 59]);
    expect(z0.velRange).toEqual([1, 64]);
    expect(z0.rootNote).toBe(48);
    expect(z0.tuning).toBe(5);                   // 0 semis * 100 + 5 cents
    expect(z0.sampleIndex).toBe(0);
    expect(z0.groupIndex).toBe(0);
    expect(z0.loopOn).toBe(true);
    expect(z0.reverse).toBe(false);
    expect(z0.sampleEnd).toBe(44100);

    const z1 = parsed.zones[1];
    expect(z1.keyRange).toEqual([60, 84]);
    expect(z1.rootNote).toBe(72);
    expect(z1.tuning).toBe(100 - 10);            // 1 semi * 100 + (-10) cents = 90
    expect(z1.sampleIndex).toBe(1);
    expect(z1.reverse).toBe(true);
    expect(z1.loopOn).toBe(false);

    const g = parsed.groups[0];
    expect(g.name).toBe('main');
    expect(g.polyphony).toBe(16);

    const r0 = parsed.sampleRefs[0];
    expect(r0.filename).toBe('kick.wav');
    expect(r0.length).toBe(44100);
    expect(r0.sampleRate).toBe(44100);
    expect(r0.bitDepth).toBe(16);

    const r1 = parsed.sampleRefs[1];
    expect(r1.filename).toBe('snare.wav');
    expect(r1.length).toBe(22050);
    expect(r1.bitDepth).toBe(24);
  });

  test('handles empty input gracefully', () => {
    const empty = buildSyntheticExs({ name: 'Empty' });
    const parsed = EXS24Reader.parse(empty);
    expect(parsed.zones).toHaveLength(0);
    expect(parsed.groups).toHaveLength(0);
    expect(parsed.sampleRefs).toHaveLength(0);
  });

  test('accepts ArrayBuffer, Uint8Array, and DataView', () => {
    const blob = buildSyntheticExs({
      name: 'Multi',
      zones: [{ keyRange: [60, 60], velRange: [1, 127], rootNote: 60, sampleIndex: 0, groupIndex: 0 }],
      sampleRefs: [{ filename: 'x.wav', length: 1, sampleRate: 44100, bitDepth: 16 }],
    });
    expect(EXS24Reader.parse(blob).zones).toHaveLength(1);
    expect(EXS24Reader.parse(new Uint8Array(blob)).zones).toHaveLength(1);
    expect(EXS24Reader.parse(new DataView(blob)).zones).toHaveLength(1);
  });
});
