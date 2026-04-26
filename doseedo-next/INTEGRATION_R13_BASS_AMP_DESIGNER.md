# R13 — Bass Amp Designer integration

A single composite node `bass_amp_designer` that mirrors Logic's stock
Bass Amp Designer plugin in the web DSP runtime.

## Files

| Path | Role |
|---|---|
| `src/audio/builders/r13_bass_amp_designer.js` | Builder. Default-exports `{ bass_amp_designer: buildBassAmpDesigner }`. |
| `src/lib/web-audio-plugins/worklets/r13-bass-amp-designer-processor.js` | Optional worklet — single-loop fallback that collapses the composite into a single per-sample DSP path. Not on the default code path. |
| `tests/r13_bass_amp_designer.test.js` | Builder smoke + DI/mic blend behavior test (zero-install Web Audio mock). |
| `../doseedo-desktop/tools/calibration/configs/bass-amp-designer-web.json` | Calibration topology + param schema for the desktop calibration harness. |

## Node type registered

`bass_amp_designer` (single composite — not split into preamp/cab/etc).

## Engine wiring

Two lines in `src/audio/WebAudioDSPEngine.js`:

```js
// near the other R-round imports at top:
import r13BassAmpBuilders from './builders/r13_bass_amp_designer.js';

// inside NODE_BUILDERS, append after r9Builders:
  ...r13BassAmpBuilders,    // bass_amp_designer (composite)
```

These additions don't conflict with the parallel agents' Amp Designer or
Vintage Amp Modeling builders — each owns a separate file and a distinct
node type (`bass_amp_designer` vs. `amp_designer` vs. `vintage_amp`).

The R13 ChromaVerb agent's file (`r13_chromaverb.js`) is also separate and
is currently not wired into `NODE_BUILDERS`. Adding both `r13` builders in
the same engine commit is a one-liner — the file names and exported
default-key sets are disjoint.

## Topology

```
input → pre_gain ┬─→ tube_path  (waveshaper, asymmetric soft-clip) ─┐
                 ├─→ ss_path    (waveshaper, sharper soft-clip)    ─┤
                 │                                  tube_blend ──→ stage_sum
                 │
                 └─→ DI tap (HPF 35 Hz + lowshelf 90 Hz + LPF 8 kHz)─→ DI gain
                                                              ↓
stage_sum → graphic_eq_5band (50/120/300/800/2000 Hz peaks)
            → wdf_tone_stack (R2 worklet OR 4-biquad fallback at 80/250/900/3000 Hz)
            → DynamicsCompressor (mild bass-amp leveling)
            → cab_ir (ConvolverNode, per cab_model: 1x15 / 2x15 / 4x10 / 6x10 / 8x10 / di_only)
            → mic_position high-shelf tilt
            → mic_gain ──┐
                         ├─→ post_gain → output
            DI gain ────┘
```

## Params

| Param | Type | Range / Enum | Notes |
|---|---|---|---|
| `amp_model` | string | `flip_top \| classic_bass \| fender_bass \| modern_bass \| svt_classic \| svt_modern \| hiwatt_bass \| acoustic_360` | Sets initial preamp curves + tone-stack biases. 8 models covers Logic's range. |
| `gain` | number | 0..2 | Pre-gain into both preamp paths. |
| `bass` | number | 0..1 | ±12 dB low-shelf at 80 Hz, additive with model bias. |
| `mid_low` | number | 0..1 | ±9 dB peak at 250 Hz. |
| `mid_hi` | number | 0..1 | ±9 dB peak at 900 Hz. |
| `treble` | number | 0..1 | ±12 dB high-shelf at 3 kHz, additive with model bias. |
| `master` (or `output_level`) | number | 0..2 | Post-gain. |
| `compression` | number | 0..1 | Scales DynamicsCompressor threshold (-6 → -30 dB) + ratio (1.5 → 6). |
| `graphic_eq` | number[5] | -12..+12 dB each | Gains at 50/120/300/800/2000 Hz. |
| `tube_blend` | number | 0..1 | 0 = solid-state only, 1 = tube only. |
| `cab_model` | string | `1x15 \| 2x15 \| 4x10 \| 6x10 \| 8x10 \| di_only` | Selects synthetic cabinet IR shape. `di_only` zeroes the IR. |
| `mic_position` | number | 0..1 | 0 = on-axis (+2 dB HF), 1 = off-axis (-6 dB HF). |
| `direct_out_mix` | number | 0..1 | Mic vs. DI blend. 0 = full mic, 1 = full DI. |

All numeric params support both literal numbers and `'@<paramId>'` modulation
references via the standard `paramTargets` shape.

## Worklet fallback policy

The composite uses zero net-new worklets — it composes existing AudioNodes
plus the R2 `wdf_tone_stack` worklet (when registered). On contexts where R2
hasn't loaded yet, the 4-biquad cascade still produces audible bass-amp
coloration. This is the same fallback strategy used by R4's circuit builders.

The optional `r13-bass-amp-designer-processor.js` worklet is NOT loaded by
default. It's available for future use if a single-worklet bass amp is
desirable (e.g. for tighter latency in 30-track sessions). To opt in, load
the module via `ctx.audioWorklet.addModule(...)` before constructing the
node and route signal through it manually.

## Calibration plan

1. Bounce 5 representative Logic Bass Amp Designer presets with a clean DI
   bass loop as the source: `BAss-Amp_Default`, `Modern Slap`, `SVT
   Rock`, `Reggae Dub`, `DI Only`.
2. For each preset, sweep these knobs in Logic:
   - `Amp Model` (8 steps — one per enum value)
   - `Gain` (11 steps × `Master` (5 steps) — 2D grid for combined drive)
   - `Bass` / `Mid Low` / `Mid Hi` / `Treble` (11 steps each)
   - `Compression` (11 steps)
   - `Direct Out Mix` (11 steps)
   - `Cab Model` (6 steps — one per enum)
3. Author per-preset `web_topology` overrides if any model-specific drift
   appears (e.g. SVT preamp's particular asymmetric clip).
4. Run R12 null-diff against each captured Logic bounce. Target ≤ −36 dB
   RMS (lower than Tier 1's −40 dB target — composites have multiplicative
   error stack, this is acceptable for a first pass).

## Tests

Run: `node Do/doseedo-next/tests/r13_bass_amp_designer.test.js`

The test:
1. Builds the node with each amp_model + cab_model and asserts the
   builder returns a valid `{ input, output, paramTargets }` shape.
2. Asserts at least 12 paramTargets are exposed when all params are
   `@`-modulated.
3. Confirms `direct_out_mix=1.0` routes signal entirely through the DI
   path (mic_gain==0, di_gain==1) — not 0.5 / 0.5 — so the DI/mic
   blend logic isn't accidentally swapped.
4. Confirms `tube_blend=0.0` zeroes the tube path output gain.
5. Confirms `compression` knob increases threshold attenuation (a higher
   value yields a lower `threshold` setting on the DynamicsCompressorNode).
6. Confirms switching `cab_model` to `'di_only'` clears the convolver
   buffer.

The mock context implements `createConvolver`, `createWaveShaper`,
`createBiquadFilter`, `createDynamicsCompressor`, `createGain`,
`createDelay` — enough to exercise the composite's full graph.

## Known gaps

- The synthetic cabinet IRs are procedural — fine for null-diff at low
  frequencies but won't beat a real bass-cab IR file. Calibration may
  swap to user-supplied IRs once a corpus is available.
- The R2 `wdf_tone_stack` worklet only takes `bass / mid / treble` (3
  bands), not the 4 the bass-amp uses. Today the worklet runs in
  parallel to the 4-biquad fallback (worklet handles the legacy 3-band
  shape on top of which the fallback adds the mid_low / mid_hi
  distinction). A future R-round could ship a `wdf_tone_stack_4band`
  variant.
- `amp_model` and `cab_model` are discrete strings. The standard A5
  param-delta channel (kind=7 patcher) emits floats; the calibration
  config's `enum` field hints the harness should round to the nearest
  enum value.

## How this fits the Tier 2 roadmap

This row in the Tier 2 coverage matrix is now 🔵 (DSP exists, calibration
not started). Calibration unblocks → 🟢. Parallel agents are doing
`amp_designer` and `vintage_amp` in separate files; merging all three
adds three rows of value with zero file conflicts.
