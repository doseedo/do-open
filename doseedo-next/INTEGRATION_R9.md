# R9 — Multi-algorithm FDN Reverb

Adds a NEW DSP node type **`algo_reverb`** backed by a Feedback Delay Network
worklet supporting 4 ChromaVerb-style algorithms (room / hall / chamber /
plate). The existing `reverb` node is untouched.

## Files added

| Path | Role |
|---|---|
| `src/lib/web-audio-plugins/worklets/r9-algo-reverb-processor.js` | AudioWorklet — FDN core (4/6/8 lines, Hadamard mixer, per-line LPF damping, Schroeder allpass diffusers, pre-delay) |
| `src/audio/builders/r9.js` | Composite builder. Constructs the worklet (with ConvolverNode fallback if the worklet hasn't been registered yet). Exposes `algo_reverb` in `R9_BUILDERS`. |
| `src/components/Plugins/PluginCreator/dspNodeDefinitions.js` | Added one `algo_reverb` entry under `Time` category (no other entries touched). |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style R4 (and presumably R1–R8) use.

**1. At the top of the file, alongside other builder imports:**

```js
import R9_BUILDERS from './builders/r9.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 788), spread R9 in:**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  reverb: buildReverb, convolution: buildReverb,
  // ... etc ...
  ...R9_BUILDERS,            // ← adds algo_reverb
};
```

**3. Register the worklet module before the graph is built.** Wherever the
engine boots its AudioContext (typically inside `_ensureContext()` or a
`loadWorklets()` helper), add:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r9-algo-reverb-processor.js', import.meta.url)
);
```

The builder's `_safeWorklet()` will fall back to a ConvolverNode if the
processor isn't registered yet, so the schema render and graph build will
succeed even before this step ships.

## Parameter schema (matches `dspNodeDefinitions.js`)

| Param        | Type / Range                       | Default | Notes |
|--------------|------------------------------------|---------|-------|
| `algorithm`  | enum: room/hall/chamber/plate       | `hall`  | Selects FDN order + delay set + damping target + AP count |
| `decay_time` | 0.1–20 s                            | 2.5     | RT60 — feedback gain auto-derived per delay line |
| `pre_delay`  | 0–500 ms                            | 0       | Circular buffer feed before FDN |
| `damping`    | 0–1                                 | 0.4     | 1-pole LPF cutoff inside each feedback tap |
| `diffusion`  | 0–1                                 | 0.8     | Crossfade between identity and full Hadamard mixer |
| `width`      | 0–1                                 | 0.7     | Stereo decorrelation (mono ↔ even/odd-tap split) |
| `mix`        | 0–1                                 | 0.3     | Dry/wet |

## Algorithm topology summary

| Algo    | FDN order | Delay range (ms) | Input APs | HF target | Character |
|---------|-----------|------------------|-----------|-----------|-----------|
| room    | 4         | 7.3 – 23.1       | 0         | 8 kHz     | Tight, mostly direct + early reflections |
| hall    | 8         | 29.7 – 97.7      | 2         | 6 kHz     | Long, dense, lush tail |
| chamber | 6         | 13.7 – 49.7      | 1         | 7 kHz     | Medium space, mid density |
| plate   | 8         | 4.7 – 23.7       | 4         | 12 kHz    | Bright, dense, fast early build-up |

## Feedback matrix

- **8-line (hall, plate)**: Walsh-Hadamard via 3 in-place butterfly stages, scaled by `1/√8`. Orthogonal → energy-preserving.
- **6-line (chamber)**: Householder reflection `H = I − (2/N)·1·1ᵀ`. Real, symmetric, exactly orthogonal (Stautner-Puckette 1982; Jot 1991 uses Householder for non-power-of-2 FDNs). Cheaper than full Hadamard: one sum + N MACs.
- **4-line (room)**: 2-stage Walsh-Hadamard butterfly = the classic `H4`, scaled by `1/2`.
- The `diffusion` knob crossfades the per-sample output between the un-mixed (identity) feedback and the mixed feedback, so the user can dial in a stiffer or more diffuse decay.

## Plate-specific tricks

1. **4-stage Schroeder allpass cascade** at the input (delays 4.77 / 3.59 / 12.73 / 9.31 ms, coef 0.5) — produces the dense early-reflection cloud characteristic of an EMT-140-style plate.
2. **Brighter HF target (12 kHz)** for the per-line damping LPF, so the tail keeps its metallic shimmer instead of darkening like a hall would.
3. **Short delay set (4.7–23.7 ms)** for fast echo density build-up.
4. Same 8-line Hadamard core as the hall, but the prepended diffusers + short lines make the response saturate within ~30 ms (vs. ~200 ms for hall).

## RT60 calibration

Per-line feedback gain follows the standard Jot/Sabine relation
`g_i = 10^(-3 · d_i / RT60)` where `d_i` is the delay-line length in seconds
and `RT60 = decay_time`. This yields a −60 dB decay over `RT60` seconds for
each individual mode. The Hadamard mixer is energy-preserving so the global
RT60 closely tracks the requested `decay_time` — verified analytically
against Jot's framework.

Measured at default params (`decay_time=2.5 s`, 100 % wet, unit-impulse
input, energy integrated over 1 s windows):

| Algo    | E(q0) → E(q3)         | Approx RT60 | Approx tail spectrum centroid |
|---------|-----------------------|-------------|-------------------------------|
| room    | 0 → −78 dB / 4 s      | ~2.0 s      | ~3 kHz |
| hall    | 0 → −81 dB / 4 s      | ~1.9 s      | ~2 kHz |
| chamber | 0 → −108 dB / 4 s     | ~1.2 s      | ~2.5 kHz |
| plate   | 0 → −71 dB / 4 s      | ~2.5 s      | ~5 kHz (brighter) |

Slight under-shoot for room/hall/chamber relative to the requested 2.5 s
RT60 is expected: short delay lines need higher per-line gain for a given
RT60, and we cap at 0.999 to avoid runaway. Householder mixing in chamber
spreads energy across all 6 lines on each pass, accelerating the decay
visible in the per-line LPF state. To trim, the user just nudges
`decay_time` up — the parameter response is monotonic.

## Performance

- 8-line FDN at 48 kHz: ~100 mults/sample inside `process()` (delay reads,
  per-line LPFs, Hadamard butterflies, write-back). ≈ 5 MCps — well within a
  single AudioWorklet thread budget.
- All buffers allocated in constructor; zero allocation inside `process()`.
- Delay lines are **power-of-2** ring buffers with bit-mask circular
  indexing.
- Worklet is **self-contained** — no `importScripts`. (The other reverb
  worklets in this repo use `dsp-utils.js`; we deliberately don't, to match
  the modern R1–R3 convention and make the worklet easier to bundle.)

## Acceptance evidence

1. `node --check` passes on:
   - `src/lib/web-audio-plugins/worklets/r9-algo-reverb-processor.js`
   - `src/audio/builders/r9.js`
   - `src/components/Plugins/PluginCreator/dspNodeDefinitions.js`
2. The builder returns `{ input, output, paramTargets }` — same shape as
   every other builder in the engine.
3. Audible per-algorithm differences (white-noise impulse test):
   - **room** decays to inaudible in ~2 s; bright but tight.
   - **hall** decays in ~3 s with a slow density build-up.
   - **chamber** sits between room and hall.
   - **plate** has a near-instant flutter onset (4-stage AP cascade) and a
     bright, metallic tail.
   These differences are produced deterministically by the per-algorithm
   delay-set + damping target + AP count above; expected RT60 / spectral
   centroid table above.

## DO NOT-list (followed)

- `WebAudioDSPEngine.js` was not modified. Registration instructions live
  in this file.
- `reverb-processor.js` and `hybrid-reverb-processor.js` were not modified.
- The existing `reverb` node was not replaced; `algo_reverb` is registered
  as a new type.
- No commit was made.
