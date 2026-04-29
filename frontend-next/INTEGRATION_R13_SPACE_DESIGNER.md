# R13 — Space Designer (`convolution_sd`)

Adds a NEW DSP node type **`convolution_sd`** that mirrors Logic Pro's Space
Designer convolution reverb. Built on top of the R1 `convolution` primitive
(ConvolverNode) plus IR-shape mutation: length truncation, attack/decay
envelopes, predelay, reverse, density resampling, and pre-IR EQ.

The existing R1 `convolution` node is untouched.

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_space_designer.js` | Composite builder. ConvolverNode + pre-IR HPF/LPF + dry/wet bus. Mutates `conv.buffer` on every shape-param change. Exposes `convolution_sd` in `R13_BUILDERS`. |
| `src/lib/web-audio-plugins/worklets/r13-convolution-sd-processor.js` | Stub worklet. Registers the processor name (so future partitioned-FFT upgrades drop in) but is never instantiated by the current builder. |
| `tests/r13_space_designer.test.js` | Unit tests — IR shape math + builder smoke + IR-buffer mutation on knob drag. Runs under jest/vitest OR plain `node` via `runAll(OfflineAudioContext)`. |
| `../doseedo-desktop/tools/calibration/configs/space-designer-web.json` | Topology config for the calibration harness (`tools/calibration/auto_driver/`). |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style R9 uses.

**1. At the top of the file, alongside other builder imports:**

```js
import r13Builders from './builders/r13_space_designer.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 829), spread R13 in:**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,    // algo_reverb (FDN, 4 algos)
  ...r13Builders,   // ← adds convolution_sd (Space Designer)
};
```

**3. (Optional)** Register the worklet module so future partitioned-FFT
upgrades activate without engine surgery. The current builder works whether
or not this runs:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-convolution-sd-processor.js', import.meta.url)
);
```

## Parameter schema

| Param         | Type / Range          | Default | Notes |
|---------------|-----------------------|---------|-------|
| `length`      | 0.0 – 1.0 ratio       | 1.0     | Truncates the IR to a fraction of the source length. Applied after density. |
| `attack_time` | 0 – 500 ms            | 0       | Linear fade-in at the IR start. |
| `decay_time`  | 0 – 10000 ms          | 0       | Exponential fade-out (Sabine RT60 mapping → ≈ −60 dB over `decay_time`). |
| `predelay`    | 0 – 500 ms            | 0       | Silence prepended in front of the shaped IR. |
| `low_cut`     | 20 – 2000 Hz          | 20      | Highpass biquad BEFORE the convolver. |
| `high_cut`    | 1000 – 20000 Hz       | 20000   | Lowpass biquad BEFORE the convolver. |
| `mix`         | 0 – 1                 | 0.3     | Wet/dry crossfade. |
| `reverse`     | bool                  | false   | Read IR tail-to-head. |
| `density`     | 0.05 – 4.0            | 1.0     | IR resampling stride. <1 packs grains; >1 stretches them. |

`@<paramId>` modulation works on every param. Numeric, EQ-frequency, and mix
params bind directly to AudioParams; IR-shape params bind through a
`customSetter` that mutates the cached `shape{}` object and schedules a
`queueMicrotask` IR rebuild.

## Default IR

A 2.5 s exponentially-decaying noise IR is generated synthetically at build
time so the node makes audible reverb without an external file. Swap it via
`buildResult.loadIR(srcOrUrlOrBlob)` — accepts URL strings, ArrayBuffers,
Blobs, File objects, and `{url}` wrappers. After load, the node reapplies
the current shape{} to the new source.

## Signal flow

```
input ──┬─→ dryGain ─────────────────────────────────────────┐
        └─→ low_cut HPF → high_cut LPF → conv (shaped IR) → wetGain ─┴─→ output
```

`conv.buffer` is rebuilt on shape-param changes via `shapeIR(...)`. Order of
operations inside `shapeIR` (matches Logic's flow):

1. Reverse the source IR if `reverse=true`
2. Resample by `density` (linear interp; <1 = denser, >1 = sparser)
3. Truncate to `length` of the resampled buffer
4. Linear fade-in over `attack_time`
5. Exponential fade-out over `decay_time` (Sabine RT60)
6. Prepend `predelay` ms of silence

## Performance notes

- IR rebuild cost: O(channels × shapedSamples). For a 2.5 s stereo IR @ 48
  kHz that's ~240k mults per shape change. Coalesced via `queueMicrotask` so
  a 50 Hz knob drag = 1 rebuild per frame max.
- ConvolverNode FFT cost is dominated by the IR length, not the rebuild rate.
  For IRs > ~5 s expect dropouts on slower machines — the partitioned-FFT
  worklet path (TODO) addresses this.
- No allocation in the hot graph path — only on rebuild, which happens off
  the audio thread (microtask).

## Testing

```bash
# Smoke (Jest/Vitest):
cd doseedo-next && npx jest tests/r13_space_designer.test.js
# Or run standalone (no framework needed):
cd doseedo-next && node --experimental-vm-modules \
  -e "import('./tests/r13_space_designer.test.js').then(m => m.runAll())"
```

Tests cover:

1. `shapeIR` — length truncation gives shorter buffer
2. `shapeIR` — predelay produces leading silence of correct length
3. `shapeIR` — reverse produces tail energy at the head
4. `shapeIR` — attack envelope ramps from silence
5. `buildConvolutionSD` — returns the standard {input, output, paramTargets} shape
6. Param-target wiring registers customSetters for every shape param
7. Knob drag (calling `paramTargets.length.customSetter`) actually mutates `conv.buffer`
8. Pre-IR EQ filters are addressable (low_cut / high_cut are AudioParam-bound)

## Calibration handoff

Topology config: `doseedo-desktop/tools/calibration/configs/space-designer-web.json`.
Add a `PluginEntry` row for "Space Designer" in
`tools/calibration/auto_driver/registry.py` referencing this config.

Per-param sweep ranges in the config double as the curve-fit search domain.
The `notes` field on each param tells the calibration engineer where to
expect Logic vs. web behavior to diverge (esp. `density`).

## Limitations / future work

- The ConvolverNode-based path is acceptable for IRs up to ~5 s; the
  partitioned-FFT worklet (`r13-convolution-sd-processor.js`) needs to be
  implemented for very long IRs / ultra-low latency. The stub is in place.
- `density` resampling is linear-interp; Logic's Space Designer uses a more
  sophisticated band-limited resampler. Calibration may need a piecewise
  curve to compensate.
- Pre-IR EQ only — Space Designer also has a post-IR EQ section. Add as
  a downstream node in the dspChain if the null-diff requires it; the
  topology config flags this.
