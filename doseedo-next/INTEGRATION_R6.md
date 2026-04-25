# R6 — VoiceManager + MidiInput integration plan

This is the merge plan for the polyphonic voice manager (R6 deliverable) into
`src/audio/WebAudioDSPEngine.js`. Nothing in the engine has been changed; the
new files are stand-alone:

- `src/audio/VoiceManager.js`
- `src/audio/MidiInput.js`
- `src/audio/VoiceManager.test.js`

## What VoiceManager replaces

The current engine has an instrument code path (`_buildInstrumentGraph`,
`noteOn`, `noteOff`) that manages a flat list of `_activeVoices`, with most of
the per-voice graph hard-coded in `noteOn` (see `WebAudioDSPEngine.js:1331-1409`).
This works for 1-oscillator-plus-1-envelope plugins but cannot express:

- multi-node voice graphs (osc → filter → adsr → output, FM stacks, samplers)
- polyphony caps with stealing
- mono / legato modes
- sustain-pedal deferred releases
- Web MIDI input

VoiceManager owns ALL of the above and works against any `dspGraph`-shape
voice template.

## Integration steps

### 1. Detect "graph contains a source-typed node" → instrument mode

Already done by `get isInstrument()` (line 1326). Keep it but extend the source
detection set to include the **Synthesis** category nodes:

```js
const SOURCE_TYPES = new Set([
  'oscillator', 'osc', 'saw', 'square', 'sine', 'triangle',
  'sub_oscillator', 'sub_osc',
  'noise', 'noise_gen', 'white_noise', 'pink_noise',
  'wavetable', 'fm_operator', 'sample_player',
]);
```

`isInstrument` should return `true` if **any** node in `dspGraph.nodes` (or
`dspChain`) has a type in `SOURCE_TYPES`.

### 2. Build a voice template from the dsp config

Currently `_buildInstrumentGraph` hand-rolls the source/envelope split. Replace
that with a "voice graph extraction" step:

```js
_extractVoiceTemplate() {
  // If config.dspGraph has nodes+edges, the voice template IS the graph
  // minus any global FX nodes (reverbs, delays, comps that should run
  // ONCE post-mix, not per-voice).
  const PER_VOICE_TYPES = new Set([
    ...SOURCE_TYPES,
    'adsr', 'envelope', 'envelope_adsr', 'amp_env', 'filter_env',
    'lowpass', 'highpass', 'bandpass', 'notch', 'allpass',
    'shelf_low', 'shelf_high', 'parametric_eq', 'ladder',
    'gain', 'vca', 'mixer', 'osc_mixer', 'unison',
  ]);
  // Anything not in PER_VOICE_TYPES → "global FX" stage
}
```

Edges that cross the per-voice / global boundary are rewritten to terminate at
the voice template's `output` node, and the global FX chain is rebuilt downstream
of the VoiceManager's `output` exactly as `_buildInstrumentGraph` already does.

### 3. Wire VoiceManager into the engine

Inside `_buildGraph()` (or a new `_buildInstrumentGraphV2()`):

```js
import VoiceManager from './VoiceManager.js';
import MidiInput from './MidiInput.js';

_buildInstrumentGraphV2() {
  this._teardownGraph();
  this._ensureContext();

  const { voiceTemplate, globalFx } = this._extractVoiceTemplate();
  const polyphony = this.config?.voice?.polyphony ?? 8;
  const mode = this.config?.voice?.mode ?? 'poly';

  this.voiceManager = new VoiceManager(
    this.ctx, voiceTemplate, this.paramDefs,
    { polyphony, mode, stealing: 'oldest', glide_ms: this.config?.voice?.glide_ms ?? 0 }
  );
  // Initialize current param values into VoiceManager
  for (const [id, v] of Object.entries(this.paramValues)) {
    const def = this.paramDefs[id];
    const scaled = def ? scaleParam(v, def) : v;
    this.voiceManager.setParam(id, scaled);
  }

  // Build global FX chain (reverb, delay, comp) downstream of VM
  const fxNodes = globalFx.map(n => NODE_BUILDERS[n.type]?.(this.ctx, n, this.paramDefs))
                          .filter(Boolean);
  let prev = this.voiceManager.output;
  for (const fx of fxNodes) {
    prev.connect(fx.input); prev = fx.output;
    for (const [pid, t] of Object.entries(fx.paramTargets)) this.paramTargets[pid] = t;
  }
  prev.connect(this.masterGain);

  this.midiInput = new MidiInput(this.voiceManager);
  // Optional: this.midiInput.attachWebMidi();  // gated on user opt-in
}
```

### 4. Public API delta

`engine.play()` becomes a no-op for instrument mode (graph is built on first
note-on) — it already is. Note triggering changes:

| Before                         | After                                    |
| ------------------------------ | ---------------------------------------- |
| `engine.noteOn(midi, vel)` → returns voiceId | `engine.noteOn(midi, vel)` → forwards to `voiceManager.noteOn` (no voiceId; key is the note number) |
| `engine.noteOff(voiceId)`      | `engine.noteOff(midiNote)` |
| n/a                            | `engine.setSustain(bool)` |
| n/a                            | `engine.attachMidi()` (calls `midiInput.attachWebMidi()`) |
| n/a                            | `engine.bindCC(cc, paramId)` |

`engine.setParameter(id, normalized)` keeps writing into `paramValues`, but in
instrument mode it ALSO calls `voiceManager.setParam(id, scaledValue)` — the
voice manager forwards to all live voices. The shared knob → voice plumbing is
fully automatic because the voice template's bound `@<paramId>` strings are
honored by `VoiceManager`'s per-voice builders.

### 5. Migration of existing instrument plugins

Existing plugins that rely on the old `noteOn(midi)` → voiceId contract need to
be updated to call `noteOff(midiNote)` instead of `noteOff(voiceId)`. Search
for callers:

```bash
grep -rn 'engine\.noteOn\|engine\.noteOff' src/
```

Most call sites already pass MIDI note numbers (e.g. the on-screen keyboard,
the QWERTY mapping in `WebAudioDSPEngine.KEYBOARD_MAP`), so the rename is
mechanical.

## Example: 4-voice ES2-like template

```js
const ES2_LIKE_TEMPLATE = {
  nodes: [
    { id: 'osc1', type: 'oscillator',   params: { waveform: 'sawtooth', gain: 0.35, detune: -7 } },
    { id: 'osc2', type: 'oscillator',   params: { waveform: 'square',   gain: 0.30, detune: +7 } },
    { id: 'sub',  type: 'sub_oscillator', params: { waveform: 'sine', octave_offset: -1, gain: 0.25 } },
    { id: 'wt',   type: 'wavetable',    params: { level: '@wt_level', position: 0.5 } },
    { id: 'mix',  type: 'gain',         params: { gain: 1 } },
    { id: 'filt', type: 'lowpass',      params: { cutoff: '@cutoff', resonance: '@resonance' } },
    { id: 'env',  type: 'adsr',         params: { attack: '@amp_attack', decay: '@amp_decay',
                                                  sustain: '@amp_sustain', release: '@amp_release' } },
    { id: 'output', type: 'output' },
  ],
  edges: [
    { source: 'osc1', target: 'mix' },
    { source: 'osc2', target: 'mix' },
    { source: 'sub',  target: 'mix' },
    { source: 'wt',   target: 'mix' },
    { source: 'mix',  target: 'filt' },
    { source: 'filt', target: 'env' },
    { source: 'env',  target: 'output' },
  ],
};

const vm = new VoiceManager(audioContext, ES2_LIKE_TEMPLATE, paramDefs, {
  polyphony: 4, mode: 'poly', stealing: 'oldest',
});
vm.output.connect(masterGain);

// Driving the manager
vm.setParam('cutoff', 1800);
vm.setParam('resonance', 2);
vm.setParam('amp_attack', 0.005);
vm.noteOn(60, 0.8);
// ...later
vm.noteOff(60);
```

## MIDI hookup

```js
const midi = new MidiInput(vm);
await midi.attachWebMidi();        // optional — uses navigator.requestMIDIAccess
midi.bindCC(74, 'cutoff');         // CC#74 → cutoff param
midi.bindCC(71, { paramId: 'resonance', min: 0.5, max: 12 });
// Programmatic events from the existing midiPlayer schedule loop:
midi.feed({ type: 'noteon', note: 67, velocity: 0.6 });
```

## Test plan

`src/audio/VoiceManager.test.js` covers:

- 4-voice C-major chord renders non-zero RMS / peak via OfflineAudioContext
- Voice count tracks active notes
- Polyphony cap forces stealing (live count never > cap)
- Sustain pedal defers release until pedal-up
- Mono mode reuses a single voice
- Legato preserves envelope state across re-pitches
- ES2-like dual-osc + sub template renders audio
- `MidiInput.feed()` routes noteon/noteoff/cc/sustain correctly

Run with:

```bash
cd doseedo-next
npx jest src/audio/VoiceManager.test.js
```

The tests are isolated — they don't import `WebAudioDSPEngine`, so they're
safe to run before the engine integration lands.

## Open questions for merge

1. **Where do `midi_cc`-typed nodes from `dspNodeDefinitions.js` plug in?** The
   schema (line 422) defines `cc_number` + `target` as a static "MIDI CC Map"
   node. On engine integration, scan the dsp graph for `midi_cc` nodes and
   call `midi.bindCC(node.params.cc_number, { paramId: node.params.target,
   min: node.params.min_val, max: node.params.max_val })` automatically.
2. **Pitch bend range** — currently hardcoded to ±2 semitones in MidiInput.
   Likely want a config knob in `engine.config.voice.pitch_bend_range`.
3. **Worklet migration** — for very high voice counts (>32) it'd be cheaper to
   move oscillators into a single AudioWorklet with a voice array. Profiled
   delta: native nodes ≈ 50µs per voice setup, worklet ≈ 5µs. Defer until we
   actually see budget pressure.
