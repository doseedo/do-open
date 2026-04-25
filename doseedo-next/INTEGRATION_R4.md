# R4 — Circuit Models integration

Agent: **R4** (composite circuit builders)
Status: builders shipped, awaiting R2 + R3 worklets to come online
Files:
- `src/audio/builders/r4.js` — composite builders for the four named hardware emulations
- `src/audio/cabinet-ir.js` — synthetic 4×10 cabinet IR for `circuit_fender_bassman`

## NODE_BUILDERS additions

Append to the `NODE_BUILDERS` map in `src/audio/WebAudioDSPEngine.js` once
WebAudioDSPEngine.js is open for the runtime-expansion merge:

```js
import R4_BUILDERS from './builders/r4.js';
// …
const NODE_BUILDERS = {
  // …existing builders…
  ...R4_BUILDERS,
};
```

`R4_BUILDERS` resolves to:

| `nodeDef.type`            | builder fn                  |
| ------------------------- | --------------------------- |
| `circuit_fender_bassman`  | `buildCircuitFenderBassman` |
| `circuit_pultec_eq`       | `buildCircuitPultecEq`      |
| `circuit_tape_machine`    | `buildCircuitTapeMachine`   |
| `circuit_tube_preamp`     | `buildCircuitTubePreamp`    |

Builder contract matches the existing engine convention — every R4 builder
returns `{ input, output, paramTargets }` (some also return `oscillators` so
the engine can `.stop()` LFOs on teardown). No changes to the engine class
itself are required — drop-in.

## Dependency graph (R2 / R3 worklets)

R4 composes black-box AudioWorkletNodes registered by other agents. The
worklet `name` strings R4 imports:

| Worklet name (R4 references)        | Owner | Purpose in R4 composites                           |
| ----------------------------------- | ----- | -------------------------------------------------- |
| `r2-wdf-tube-amp-processor`         | R2    | Tube preamp stage in `bassman`, `tube_preamp`      |
| `r2-wdf-tone-stack-processor`       | R2    | Bass/Mid/Treble stack in `bassman`                 |
| `r3-wdf-transformer-processor`      | R3    | Output transformer in `bassman`                    |
| `r3-wdf-tape-sat-processor`         | R3    | Tape saturation in `tape_machine`                  |
| `r3-wdf-rc-filter-processor`        | R3    | Reserved (not needed once `wdf_tone_stack` exists) |

### Soft-failure wrapper

Every `new AudioWorkletNode(ctx, name, …)` in `r4.js` is wrapped in
`_safeWorklet()`. If R2/R3 haven't yet called
`audioContext.audioWorklet.addModule(…)` for their processor, construction
throws and `_safeWorklet()` returns `null`. The composite then falls back to
a primitive substitute so the graph still plays:

| Missing worklet         | R4 fallback                                                    |
| ----------------------- | -------------------------------------------------------------- |
| `r2-wdf-tube-amp`       | `tanh(x·2.0)` `WaveShaperNode` (single stage)                  |
| `r2-wdf-tone-stack`     | 3 cascaded biquads (lowshelf 100 Hz, peaking 500 Hz, highshelf 3.5 kHz) |
| `r3-wdf-transformer`    | Gentle `tanh(x·1.5) · 0.95` `WaveShaperNode`                   |
| `r3-wdf-tape-sat`       | Asymmetric `tanh` `WaveShaperNode` (positive half compressed)  |

This means the schema-listed circuit models are *audible right now*, even
before R2/R3 finish. Once they ship and their `audioWorklet.addModule()`
calls run on the live `AudioContext`, the same builders will pick up the
real WDF processors with no code change in `r4.js`.

### Required boot order

To get the "real" sound the engine must, before any composite is built:

```js
await ctx.audioWorklet.addModule('/path/to/r2-wdf-worklets.js');
await ctx.audioWorklet.addModule('/path/to/r3-wdf-worklets.js');
```

R2 and R3 are responsible for shipping the actual worklet bundles and for
naming their processors exactly as listed in the table above. If either
agent picks a different name they MUST coordinate the rename with R4.

## Per-composite topology (quick reference)

### `circuit_fender_bassman`
```
input → wet ─┬─→ [R2 wdf_tube_amp stages=2]
             │       ↓
             │   [R2 wdf_tone_stack OR 3-biquad FMV fallback]
             │       ↓
             │   [R3 wdf_transformer OR tanh shaper]
             │       ↓
             │   [ConvolverNode w/ buildCabinetIR()]   ← src/audio/cabinet-ir.js
             │       ↓
             │   [presence highshelf @ 3 kHz]
             │       ↓
             │   [master gain] ───→ output
       dry ──────────────────────→ output
```

Params: `gain`, `bass`, `mid`, `treble`, `presence`, `master` (alias
`output_level`), `mix`.

### `circuit_pultec_eq`
```
input → wet → [lowshelf BOOST @ low_freq]
            → [lowshelf CUT   @ low_freq · 1.5]
            → [peaking BOOST  @ high_freq, Q=high_bandwidth]
            → [highshelf CUT  @ 20 kHz with high_atten]
            → [output gain] ─────→ output
       dry ────────────────────→ output
```

The "boost+attenuate same freq creates a notch on the boost shoulder" trick
is reproduced by giving boost and cut **different** corner frequencies
(cut tracks at `1.5 ×` boost on the low band, fixed at 20 kHz on the high
band). This is the same shape Pultec EQP-1A produces in hardware.

Params: `low_freq`, `low_boost`, `low_atten`, `high_freq`, `high_boost`,
`high_atten`, `high_bandwidth`, `output`, `mix`.

### `circuit_tape_machine`
```
input → wet → [input_level gain]
            → [highshelf @ 10 kHz, +6 dB]                  ← record EQ
            → [R3 wdf_tape_sat OR asymmetric tanh shaper]
            → [highshelf @ 10 kHz, −6 dB]                  ← playback EQ
            → [DelayNode @ 25 ms]
                ↑ wow LFO 0.5 Hz × wow_depth
                ↑ flutter LFO 6 Hz × flutter_depth
            → [peaking @ 80 Hz, +0..8 dB]                  ← head bump
            → output
       dry ────────────────────────────────────────────→ output
```

Wow + flutter are summed onto a single `delayTime` AudioParam (cheap,
phase-coherent, no ScriptProcessor needed). Modulation depths are scaled by
the schema's `wow_flutter` parameter so a single knob feels right.

Params: `input_level`, `bias`, `speed`, `wow_flutter`, `head_bump`, `mix`.

### `circuit_tube_preamp`
```
input → wet → [highshelf @ 4 kHz, engaged when bright > 0.5]
            → [R2 wdf_tube_amp w/ stages param OR tanh shaper]
            → [highshelf tilt EQ @ 1.5 kHz]
            → [output_level gain] ─→ output
       dry ────────────────────────→ output
```

Params: `gain`, `stages`, `bright`, `output_level` (+ optional `tone`,
`mix`).

## Known approximations

- Cabinet IR is a procedurally-generated bandpass impulse — *not* a real
  speaker measurement. ~40 ms long. Real users can replace it by writing
  their own `AudioBuffer` into the `ConvolverNode`. This is intentional:
  shipping a 200 KB binary IR isn't worth the bundle weight for a fallback
  path.
- Pultec "broad/sharp" bandwidth is a single Q sweep (0.3 → 3.0). Real
  EQP-1A bandwidth is two switched stages — close enough for a UI knob.
- Tape `bias` has no audible effect when the R3 worklet is missing
  (no fallback). Engine still binds the param target via `customSetter:
  () => {}` so automation isn't dropped.
- `circuit_fender_bassman`'s presence control is a post-cab high-shelf, not
  in the negative-feedback loop as on real hardware. Sounds close enough at
  modest gain settings.
- `wow_flutter`, `bright` switching, and Pultec band-cut "boost+atten" all
  live entirely on the AudioParam graph — no ScriptProcessor / no main-thread
  audio code.

## Testing checklist

- [ ] After R2/R3 land their worklets, confirm `_safeWorklet` no longer logs
  `[R4] worklet … unavailable, using fallback` at runtime.
- [ ] Verify that mapping `mix` → engine param values produces a smooth dry/wet
  crossfade for each composite (R4 sets `wet = v, dry = 1 − v`).
- [ ] Open one of each composite in PluginCreator, automate a parameter, confirm
  smooth modulation (no zipper noise — most params are AudioParams or run
  through `customSetter` debounced by the engine's existing param-update path).
- [ ] Confirm `cabinet-ir.js` is tree-shake-friendly (only imported when the
  Bassman model is used) — current import in `r4.js` is static, so the IR
  generator ships whenever any circuit model ships. If bundle size matters
  later, lazy-import it inside `buildCircuitFenderBassman`.
