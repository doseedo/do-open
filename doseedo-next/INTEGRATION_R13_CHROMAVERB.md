# R13 — ChromaVerb full-algorithm parity

R13 extends R9's algorithmic FDN reverb (`algo_reverb`, 4 algos) with three
new node types covering the rest of Logic's ChromaVerb 14-algorithm set:

| Node type     | Logic algos covered                              | FDN order | Mixer       |
|---------------|--------------------------------------------------|-----------|-------------|
| `fdn_smooth`  | Vocal Hall, Dark Hall, Gentle Chamber            | 8         | Hadamard-8  |
| `fdn_strange` | Strange Room, Synth Hall, Modulated Chamber      | 6         | Householder |
| `fdn_dense`   | Bright Room, Drum Plate, Early Reflections       | 16        | Hadamard-16 |

Together with R9's `algo_reverb`, the 14 ChromaVerb algorithms are mapped
in `tools/calibration/configs/chromaverb-full-web.json` (desktop tree) via
an `algorithm_map` that selects an engine node type per Logic algorithm
index. Calibration changes the slot's node type when `algorithm` changes
(not a worklet enum like in R9).

## Param surface

All three variants accept the same parameters as `algo_reverb`:

```
decay_time  s     0.1 .. 20
pre_delay   ms    0   .. 500
damping     0..1
diffusion   0..1
width       0..1
mix         0..1
```

The `algorithm` knob itself is handled at the topology level (a slot
rebuild) rather than as a worklet param.

## Per-variant tuning

### `fdn_smooth` — Vocal Hall / Dark Hall

- 8-line FDN, base delays 61–157 ms (long, mutually-prime).
- 4-stage Schroeder input diffuser, allpass coef 0.6 — high pre-density.
- HF target 5.5 kHz at damping=0 (already noticeably dark — the
  signature character of "vocal hall" / "dark hall" presets).
- Default decay 3.5 s, default damping 0.5, default diffusion 0.9.

The longer base-delay set and stronger diffuser cascade vs. R9 hall give
a slower modal density build-up — perceived as a smoother, denser tail.

### `fdn_strange` — Strange Room / Synth Hall

- 6-line FDN with **Householder reflection mixer** (real symmetric
  orthogonal): `H = I - (2/N)·1·1ᵀ`. Energy-preserving, cheaper than
  Hadamard, and breaks the power-of-2 modal symmetry of standard FDNs.
- Per-line LFO modulating the read tap (fractional-delay interpolation):
  rates 0.31–1.27 Hz (mutually irrational), depths 1.7–4.7 ms.
- 1-stage Schroeder diffuser only — preserves transient identity so
  the modulation is the audible feature, not raw early-reflection
  density.
- Default decay 2.5 s, default damping 0.3, default width 0.9.

The irrational LFO rates ensure the modulation never phase-locks; that's
what gives the chorused, drifting tail without metallic combing.

### `fdn_dense` — Bright Room / Drum Plate

- **16-line FDN** (double R9's largest), short delays 3.1–22.1 ms.
- Walsh-Hadamard mixer on 16 elements (4 butterfly stages, in-place).
- 6-stage Schroeder pre-diffuser, allpass coef 0.65 — aggressive early
  density.
- HF target 13 kHz at damping=0 — very bright.
- Stereo decorrelation by **halves split** (taps 0–7 vs. 8–15) rather
  than even/odd interleave — gives a wider perceived field for dense
  FDNs where adjacent taps are highly correlated.
- Default decay 1.5 s, default damping 0.2, default diffusion 1.0.

Per-sample cost is roughly 2× R9 hall (16 LPFs + 16 delay reads + 64
mults for Hadamard-16). Comfortably real-time at 48 kHz on Apple Silicon.

## Builder + worklet files

Builders (`Do/doseedo-next/src/audio/builders/r13_chromaverb.js`):

- `buildFdnSmooth(ctx, node, paramDefs)`
- `buildFdnStrange(ctx, node, paramDefs)`
- `buildFdnDense(ctx, node, paramDefs)`
- `default` export: `{ fdn_smooth, fdn_strange, fdn_dense }`

Worklets (`Do/doseedo-next/src/lib/web-audio-plugins/worklets/`):

- `r13-fdn-smooth-processor.js`  → registered as `r13-fdn-smooth-processor`
- `r13-fdn-strange-processor.js` → registered as `r13-fdn-strange-processor`
- `r13-fdn-dense-processor.js`   → registered as `r13-fdn-dense-processor`

## Engine wiring

Add to `WebAudioDSPEngine.js`:

```js
import r13Builders from './builders/r13_chromaverb.js';
// ...
const NODE_BUILDERS = {
  // ...
  ...r9Builders,
  ...r13Builders,   // fdn_smooth, fdn_strange, fdn_dense
};
```

The worklets are loaded via the existing `_ensurePhase1Worklets` plumbing —
just add the three filenames to that loader.

## Fallback behavior

Every builder uses `_safeWorklet()`: if the `AudioWorkletNode` constructor
throws (worklet module not yet loaded, or running in a non-browser test
context), the builder returns a primitive ConvolverNode + dry/wet gain
pair instead. The fallback IR profile is tuned per variant:

| Variant       | Duration | Decay exp | Spectral tilt |
|---------------|----------|-----------|---------------|
| `fdn_smooth`  | 4.5 s    | 2.5       | -0.3 (LF)     |
| `fdn_strange` | 2.8 s    | 3.0       |  0.0          |
| `fdn_dense`   | 1.6 s    | 4.5       | +0.4 (HF)     |

The fallback exposes the same `paramTargets` keys as the worklet path so
the engine sees a consistent shape regardless of whether the worklet
loaded. `decay_time` and `mix` are functional in fallback (rebuild IR /
adjust dry-wet gains); `pre_delay`, `damping`, `diffusion`, `width` are
no-op customSetters in fallback (graph still binds — null-diff just
won't be tight until the real worklet loads).

## Testing

`Do/doseedo-next/tests/r13_chromaverb.test.js` is a node:test smoke
suite covering one build per variant. It uses an `OfflineAudioContext`
and a stub `AudioWorkletNode` global — the builders fall through to the
primitive fallback path and the test asserts that each builder returns
`{ input, output, paramTargets }` with both audio nodes connectable.

Run with:

```
node --test tests/r13_chromaverb.test.js
```

## Calibration plan

`tools/calibration/configs/chromaverb-full-web.json` defines the topology
and a 14-entry `algorithm_map`. Calibration order (low to high effort):

1. `fdn_dense` drum-plate variant — short tail, easy curve-fit on decay.
2. `fdn_smooth` vocal-hall variant — single dominant param (decay).
3. `fdn_strange` synth-hall variant — modulation depth/rate are
   currently fixed; if R12 null-diff misses, expose them as worklet
   params in a follow-up.

Target -40 dB RMS null-diff per algorithm against a Logic bounce of the
preset on `gen://drums` source.
