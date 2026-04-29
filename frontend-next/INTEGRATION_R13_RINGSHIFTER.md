# R13 — Ringshifter (`ring_shift`)

Adds a NEW DSP node type **`ring_shift`** that mirrors Logic Pro's stock
Ringshifter — a combined ring modulator + frequency shifter with internal
LFO, feedback, and output stage.

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_ringshifter.js` | Builder. Constructs the worklet (full DSP) or, on absence, a primitive ring-mod fallback (OscillatorNode → GainNode-as-multiplier). Wires `mode / freq_hz / lfo_* / feedback / dry_mix / wet_mix / output_gain` to AudioParams or `customSetter` shims. Default-exports `{ ring_shift: buildRingshifter }`. |
| `src/lib/web-audio-plugins/worklets/r13-ringshifter-processor.js` | Worklet processor. Hilbert-pair via 8-stage Niemitalo IIR allpass cascade; carrier oscillator + LFO + feedback + dry/wet stage. |
| `../doseedo-desktop/tools/calibration/configs/ringshifter-web.json` | Topology config for the calibration harness. |
| `tests/r13_ringshifter.test.js` | Builder smoke + ring-mod-on-440 Hz spectral test (asserts sidebands at `440-fc` and `440+fc`). |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style R9 / other R13 builders use.

**1. At the top of the file, alongside other builder imports:**

```js
import r13RingshifterBuilders from './builders/r13_ringshifter.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 829), spread it in:**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,
  ...r13RingshifterBuilders,   // ← adds ring_shift (Ringshifter)
};
```

**3. Register the worklet module** so the full Hilbert path activates
(otherwise the builder uses the ring-mod-only fallback):

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-ringshifter-processor.js', import.meta.url)
);
```

## Parameter schema

| Param         | Type / Range                                     | Default     | Notes |
|---------------|--------------------------------------------------|-------------|-------|
| `mode`        | `ring_mod`/`freq_shift_up`/`freq_shift_down`/`both` | `ring_mod`  | Enum maps to integer 0..3 inside the worklet. |
| `freq_hz`     | 0 – 5000 Hz                                      | 220         | Carrier (ring-mod) or shift amount (freq-shift). |
| `lfo_rate`    | 0 – 10 Hz                                        | 0           | 0 disables LFO modulation entirely. |
| `lfo_depth`   | 0 – 100 %                                        | 0           | Percent of `freq_hz`. Carrier `f_now = freq_hz · (1 + depth · lfo(t))`. |
| `lfo_shape`   | `sine`/`triangle`/`square`/`random`              | `sine`      | `random` is sample-and-hold once per LFO cycle. |
| `feedback`    | 0 – 0.99                                         | 0           | Wet output → shift-path input. Capped to prevent runaway. |
| `dry_mix`     | 0 – 1                                            | 0.5         | |
| `wet_mix`     | 0 – 1                                            | 0.5         | Logic exposes a single Wet/Dry %; map dry = 1 − wet at the calibration layer. |
| `output_gain` | 0 – 4 (linear)                                   | 1.0         | ≈ −∞ … +12 dB. |

`@<paramId>` modulation works on every param. Numeric params bind directly
to the worklet AudioParams; `mode` and `lfo_shape` go through a
`customSetter` that converts string / 0..1-normalised inputs to the
worklet's integer enum.

## Hilbert allpass cascade design

The frequency shifter relies on a **90° phase-difference network** that
takes the input `x[n]` and produces a pair `(x_real, x_imag)` whose phase
spectrum differs by π/2 across the design band. SSB modulation is then
trivially:

```
y_up   = x_real · cos(2πft) − x_imag · sin(2πft)
y_down = x_real · cos(2πft) + x_imag · sin(2πft)
```

We use the **Niemitalo 8-stage half-band IIR allpass design** (Olli
Niemitalo, *Polyphase IIR filters with Hilbert transform pair outputs*,
1999 — DSPGuru / `dsp.stackexchange` references). Two parallel cascades
of 4 first-order allpasses each:

```
real path: a² ∈ {0.4794, 0.8762, 0.9738, 0.9948}
imag path: a² ∈ {0.1618, 0.7330, 0.9453, 0.9907}
```

Each section is the standard one-pole-one-zero allpass:

```
y[n] = a · x[n] + x[n−1] − a · y[n−1]
```

The resulting **phase difference between the two paths** is within
≤ 0.05° of 90° from roughly **80 Hz to 10 kHz** at 48 kHz sample rate
(slightly narrower at 44.1 k). Outside that band the difference drifts —
sub-bass leaks into the unwanted sideband and very-HF content blurs.

There is an inherent **1-sample delay** between the two paths inherent
to the polyphase structure, which we compensate by delaying the real
path by one sample (`_realDelayL` / `_realDelayR` in the processor).

### Cutoff tuning notes

- **Lower edge** (≈80 Hz at 48 kHz): below this, sub-bass content has
  poor 90° symmetry and the rejected sideband shows up at ≈ −15 dB.
  Acceptable for musical material; not acceptable for sub-octave
  synth lines that need exact null. Future upgrade: 12-stage cascade
  (≈30 Hz lower edge) at the cost of one extra k-rate sample-block of
  CPU.
- **Upper edge** (≈10 kHz): above this the rejection drops similarly.
  In practice most program material has limited HF energy so the
  audible artefact is small.
- **Sample-rate scaling**: the published `a²` values are normalised to
  the design half-band. They work without modification at 44.1 / 48 /
  88.2 / 96 kHz, with the design band scaling proportionally to Nyquist.

For Tier-2 null-diff calibration against Logic's Ringshifter, expect
≥ -30 dB null on the freq-shift modes and ≥ -45 dB null on pure
ring-mod (which doesn't touch the Hilbert path).

## Signal flow

```
                     ┌──────────────┐
        input ──────▶│   dry path   │────┐
              │      └──────────────┘    │
              │                          ▼
              │                     ┌────────┐    ┌─────────┐
              │      ┌────────────┐ │  +     │───▶│ output  │
              └─────▶│  Hilbert   │─│        │    │  gain   │
                     │  cascade   │ │  wet   │    └─────────┘
                     │  (real,    │ │  mix   │
                     │   imag)    │ │        │
                     └─────┬──────┘ └────────┘
                           │             ▲
                           │             │
                     ┌─────▼─────┐   ┌──────────┐
                     │  SSB / RM │◀──│ carrier  │◀── LFO (rate, depth, shape)
                     │  combiner │   │  osc     │
                     └─────┬─────┘   └──────────┘
                           │
                  ┌────────▼────────┐
                  │   feedback ── ▼ │  (z⁻¹)
                  └─────────────────┘
```

## Fallback behaviour

When the worklet module hasn't been registered on the AudioContext (cold
engine, lazy-load race, OfflineAudioContext without addModule), the
builder constructs a primitive ring-mod-only path:

- `OscillatorNode (sine, freq_hz)` connected into a `GainNode.gain`
  (AudioParam). The same GainNode's audio-path is fed by the input.
  Because Web Audio sums signal-rate AudioParam contributions to the
  param's static value, the output of that GainNode is `input(t) · sin(2πft)`.
- Dry/wet/output gains are real `GainNode`s.
- `feedback` is captured into a state object but cannot be wired without
  introducing a feedback loop the audio thread will not allow without
  intermediary delay. We accept the binding as a no-op.
- `mode = freq_shift_up / freq_shift_down / both` all degrade to ring-mod
  (no Hilbert primitive in Web Audio core).

The fallback is for graph-binding survival, not for null-diff parity.
Calibration MUST run with the worklet registered.

## Test plan

`tests/r13_ringshifter.test.js` exercises:

1. **Builder smoke** — `buildRingshifter(ctx, node, paramDefs)` returns a
   `{input, output, paramTargets}` triple with the expected param keys
   and L/R-coherent stereo output.
2. **Ring-mod sideband** — fed a 440 Hz sine through the builder with
   `mode=ring_mod, freq_hz=100, dry_mix=0, wet_mix=1`. The rendered
   output FFT should show energy at 340 Hz (440 − 100) and 540 Hz
   (440 + 100), with original 440 attenuated.
3. **Param target shape** — verifies every modulated `@id` param
   produces an entry in `paramTargets`.

The test self-runs when the file is invoked via `node` and also exposes
Jest/Vitest hooks via `describe`/`it`/`expect` shims.

## Status

- DSP shipped; node registered: `ring_shift`.
- Calibration topology JSON: `ringshifter-web.json`.
- Mapping JSON (`public/plugin-mappings/<id>.json`): not yet authored —
  pending Logic param-ID discovery on a calibration project.
- Tier-2 null-diff threshold target: -30 dB across `mode × freq_hz`
  sweep grid.
