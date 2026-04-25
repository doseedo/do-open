# R5 — Spectral DSP Worklets (integration notes)

This batch adds three FFT-domain DSP nodes to the WebAudioDSPEngine builder
registry: `pitch_shift`, `spectral_filter`, `spectral_freeze`.

## Files added

```
src/lib/web-audio-plugins/worklets/r5-pitch-shift-processor.js
src/lib/web-audio-plugins/worklets/r5-spectral-filter-processor.js
src/lib/web-audio-plugins/worklets/r5-spectral-freeze-processor.js
src/audio/builders/r5.js
```

Each worklet inlines its own Cooley–Tukey radix-2 FFT (one class, ~80 lines)
plus a Hann window helper. They never call `importScripts` — Doseedo serves
worklets from arbitrary roots (Vercel + the legacy CRA stack), so an inline
copy is simpler and more portable than a shared `r5-fft.js`.

## NODE_BUILDERS additions

In `src/audio/WebAudioDSPEngine.js`, merge the R5 builders into the existing
`NODE_BUILDERS` map. Recommended insertion (right after the R1/R2/R3/R4
imports and merges):

```js
import r5Builders from './builders/r5.js';

// existing NODE_BUILDERS object …
Object.assign(NODE_BUILDERS, r5Builders);
```

R5 exports three keys: `pitch_shift`, `spectral_filter`, `spectral_freeze`.

### Conflict: `pitch_shift`

`r1.js` already defines a `pitch_shift` builder (SOLA / time-domain).
`r5.js` defines a phase-vocoder version. Whichever Object.assign runs last
wins. The integrator should pick one of:

1. **Spectral (R5) wins** — better quality at large shifts (±12 → ±24
   semitones), 46 ms latency at 2048-pt FFT. Recommended for music
   creative effects.

   ```js
   Object.assign(NODE_BUILDERS, r1Builders);
   Object.assign(NODE_BUILDERS, r5Builders); // R5 overrides R1's pitch_shift
   ```

2. **SOLA (R1) wins** — lower latency, gentler artifacts on small shifts
   (±3 to ±6 semitones), but transient smearing at extremes.

   ```js
   Object.assign(NODE_BUILDERS, r5Builders);
   Object.assign(NODE_BUILDERS, r1Builders); // R1 overrides R5's pitch_shift
   ```

3. **Both, distinct keys** — if the schema is updated to expose two node
   types (e.g. `pitch_shift_pv` and `pitch_shift_sola`), import explicitly:

   ```js
   import { buildPitchShift as buildPitchShiftPV } from './builders/r5.js';
   import { buildPitchShift as buildPitchShiftSOLA } from './builders/r1.js';
   NODE_BUILDERS.pitch_shift_pv   = buildPitchShiftPV;
   NODE_BUILDERS.pitch_shift_sola = buildPitchShiftSOLA;
   ```

   This is the cleanest long-term option but requires a schema edit in
   `dspNodeDefinitions.js`.

## FFT design choices

- **Algorithm**: Cooley–Tukey radix-2, in-place, with pre-computed cosine /
  sine tables and a precomputed bit-reversal lookup. ~80 lines, public-domain
  derivation.
- **Precision**: `Float32Array` throughout. Matches the rest of the Doseedo
  worklet codebase (`fft-lib.js` pattern).
- **Inlined per worklet**: each R5 worklet owns its own copy of `R5FFT` and
  `r5HannWindow`. Worklets cannot `import` ES modules at runtime in a
  consistent way across browsers — `addModule(url)` only loads one file
  and there is no equivalent of Node-style `require`. `importScripts` is
  unavailable in modern AudioWorkletGlobalScope. The cost of duplication
  is ~80 LOC × 3 = ~240 LOC; far cheaper than the bundle-time complexity
  of a shared module.
- **Window**: Hann at the configured FFT size. Hann + 75% overlap (`hop = N/4`)
  satisfies the COLA condition with a constant overlap-sum of 1.5, so we
  multiply the synthesis frame by `2/3` (`windowNorm`) to bring unit-gain
  output. No additional normalization needed.
- **Default size**: 2048 samples. Configurable via constructor
  `processorOptions.fftSize`. Latency = `fftSize` samples (≈ 46 ms at
  44.1 kHz). Halving to 1024 cuts latency in half but loses
  low-frequency resolution.

## Builder pattern

The engine's `NODE_BUILDERS` map calls builders **synchronously**, but
`audioWorklet.addModule()` is **asynchronous**. R5 (matching R1's
convention) handles this with:

1. `ensureR5Worklets(ctx)` — idempotent, per-context. Kicks off
   `addModule` for all three R5 worklets the first time it's called.
2. `tryCreateWorklet(ctx, name, opts)` — eagerly calls
   `ensureR5Worklets`, then if registration is **already complete**
   creates the `AudioWorkletNode`; otherwise returns `null`.
3. If `tryCreateWorklet` returns `null`, the builder falls back to a
   passthrough: `input.connect(output)`. The next graph rebuild (after
   the user changes a node, presses re-render, etc.) will find the
   worklets registered and instantiate the real processor.

This means the **first build after page load** of any R5 node will be a
silent passthrough; the second build will work. To pre-warm, callers can
explicitly `await ensureR5Worklets(ctx)` after creating the AudioContext.

## Per-node parameter binding

All three nodes follow the same pattern. Schema params (from
`dspNodeDefinitions.js`) bind to the worklet's `AudioParam`s:

| node            | schema param | AudioParam       | range        |
|-----------------|--------------|------------------|--------------|
| pitch_shift     | semitones    | `semitones`      | -24 .. +24   |
| pitch_shift     | mix          | `mix`            | 0 .. 1       |
| spectral_filter | low_bin      | `lowBin`         | 0 .. 1       |
| spectral_filter | high_bin     | `highBin`        | 0 .. 1       |
| spectral_filter | mix          | `mix`            | 0 .. 1       |
| spectral_freeze | freeze       | `freeze`         | 0 .. 1       |
| spectral_freeze | mix          | `mix`            | 0 .. 1       |

`@param_id` bindings install entries in `paramTargets` for live updates
through `engine.setParameter(...)`.

Each worklet also accepts `port.postMessage` overrides for cases where the
host wants to drive the parameter outside the AudioParam graph
(e.g. `{type: 'semitones', value: 7}` or `{type: 'reset'}`). Setting an
override pins the param until reset; AudioParam values are still wired
but are ignored while overridden.

`spectral_freeze` additionally accepts `{type: 'phase_mode', value: 'advance' | 'random'}`.
The default is `'advance'` (phase-vocoder true-bin propagation, gives a
tonal pad). `'random'` randomizes phase per hop for a noisier ambient pad.

## Latency

| node            | fftSize | hop | latency (samples) | latency (ms @ 44.1 kHz) |
|-----------------|---------|-----|-------------------|--------------------------|
| pitch_shift     | 2048    | 512 | 2048              | ≈ 46 ms                  |
| spectral_filter | 2048    | 512 | 2048              | ≈ 46 ms                  |
| spectral_freeze | 2048    | 512 | 2048              | ≈ 46 ms                  |

All three pay one full window of latency on the input side (we wait for
the first complete frame before emitting). They do **not** add output
latency beyond OLA itself, because each synthesis hop produces `hop`
fresh samples that are emitted immediately by the per-sample shift loop.

## Known limits

- **Transient smearing on extreme pitch shifts**. At ±18 semitones and
  beyond, the phase-vocoder smears percussive transients across multiple
  hops because the synthesis reconstruction averages magnitudes over a
  full window. Mitigations available but not implemented:
  - Phase-locked vocoder (Laroche-Dolson "rigid phase locking") — keep
    phase coherence within peak regions of the spectrum.
  - Transient detection + bypass during transient hops.
  - Smaller FFT size (1024 → ~23 ms latency, less smearing) at the cost
    of frequency resolution.

- **Spectral filter gating is rectangular**. A hard zero-bin gate produces
  ringing in the time domain (sinc roll-off). The Hann analysis/synthesis
  windows mask most of it, but rapid sweeps of `low_bin` / `high_bin` will
  click. A future revision could apply a smoothed taper at the bin edges
  (e.g. raised-cosine over the outermost 3-5 bins on each side).

- **Spectral freeze on noisy material**. `'advance'` phase mode produces a
  tonal pad; on broadband noise input this can sound metallic because the
  phase advance for each bin is a discrete frequency. Switch to
  `'random'` mode for ambient material: `worklet.port.postMessage({type:'phase_mode', value:'random'})`.

- **Mono output**. All three worklets are mono-in / mono-out internally
  (left channel). They duplicate the wet signal to L+R on output for the
  engine's stereo bus, but stereo input phase / image is not preserved.
  Future work: dual independent FFT per channel (~2× CPU).

- **First-build silence**. As noted above: the very first build of an R5
  node on a new AudioContext is a passthrough until `addModule` finishes.
  Pre-warm with `await ensureR5Worklets(ctx)` if you need it ready
  before the first build.

- **Memory cost per node**. ~3 × 2048 × 4 bytes = 24 KB FFT scratch +
  ~3 × 1024 × 4 bytes = 12 KB phase / mag state per node. Plus an
  18-element ring buffer + OLA buffer. Total ≈ 60 KB per instance —
  negligible, but worth noting if hundreds are stamped at once.
