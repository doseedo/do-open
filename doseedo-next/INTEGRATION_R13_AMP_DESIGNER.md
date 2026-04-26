# R13 — Amp Designer (composite)

Adds a NEW DSP node type **`amp_designer`** that composes existing R2/R3
worklets + a procedural cabinet IR convolver to mimic Logic Pro's Amp
Designer chain. No R2/R3 files were modified.

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_amp_designer.js` | Composite builder. Owns the AMP_MODELS preset table (9 amps) and CAB_PROFILES procedural-IR table (7 cabs). Exposes `amp_designer` in `R13_AMP_DESIGNER_BUILDERS`. |
| `src/lib/web-audio-plugins/worklets/r13-amp-designer-processor.js` | Placeholder passthrough worklet. The composite does not need a dedicated DSP worklet today; this file is reserved for a future "fold the chain into one processor" optimisation. |
| `tools/calibration/configs/amp-designer-web.json` *(desktop tree)* | Topology config used by the R10 calibration harness. |
| `tests/r13_amp_designer.test.js` | Builder smoke-test + amp-model preset switch test. Self-runs via `node tests/r13_amp_designer.test.js`. |

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, mirroring the R9 / R4 pattern.

**1. Import alongside the other builder bundles:**

```js
import r13AmpDesignerBuilders from './builders/r13_amp_designer.js';
```

**2. Inside the `NODE_BUILDERS` map (currently around line 829), spread it in
after `...r9Builders`:**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,            // algo_reverb
  ...r13AmpDesignerBuilders, // ← adds amp_designer
};
```

The composite builder uses `_safeWorklet()` for every R2/R3 processor
construction, so the graph builds and produces audio even before any
worklet module has been registered (the WaveShaper/BiquadFilter fallbacks
take over). Once R2/R3 worklets ship, the same builder picks up the real
processors with no code change.

## Public parameter surface

| Param | Type | Range | Default | Notes |
|---|---|---|---|---|
| `amp_model`    | enum     | 9 values | `british_clean` | Selects the AMP_MODELS preset (see table below) |
| `gain`         | number   | 0–10     | 5      | Input gain into the preamp; mapped 0..3 internally |
| `bass`         | number   | 0–1      | 0.5    | Tone stack — summed with amp-model offset |
| `mid`          | number   | 0–1      | 0.5    | Tone stack — summed with amp-model offset |
| `treble`       | number   | 0–1      | 0.5    | Tone stack — summed with amp-model offset |
| `presence`     | number   | 0–1      | 0.5    | Post-cab high-shelf @ 3 kHz; sums with amp-model baseline (clamped ±18 dB) |
| `master`       | number   | 0–10     | 5      | Power-amp drive; mapped to R2 `wdf_tube_amp.gain` 0..3 |
| `cab_model`    | enum     | 7 values | `4x12_open` | Procedural IR profile (see CAB_PROFILES below) |
| `mic_position` | number   | 0–1      | 0      | 0 = on-axis (brighter), 1 = off-axis (darker) |
| `output_level` | number   | 0–4      | 0.7    | Final post-cab gain |

## AMP_MODELS table

Each row retunes the existing R2/R3 worklets' internal params and adds a
small tone-stack offset curve. The numbers are deliberately approximate
("sounds like an amp"), not calibrated to a specific real amplifier.

| Name              | Preamp drive / bias | Power drive / bias / stages / out | Xfmr drive / sat | Sag (amt / rec) | Pres dB | Tone offset (B / M / T) | Default cab |
|-------------------|---------------------|-----------------------------------|------------------|-----------------|---------|-------------------------|-------------|
| tweed             | 1.4 / -1.4 | 1.6 / -1.6 / 2 / 0.55 | 1.2 / 0.55 | 0.55 / 0.05 | 3 | +0.05 / +0.10 / -0.05 | 1x12 |
| british_clean     | 0.7 / -1.6 | 0.9 / -1.8 / 2 / 0.65 | 1.0 / 0.30 | 0.20 / 0.08 | 5 |  0.00 / -0.05 / +0.05 | 4x12_open |
| british_crunch    | 1.5 / -1.5 | 1.8 / -1.5 / 3 / 0.60 | 1.4 / 0.55 | 0.40 / 0.06 | 6 |  0.00 / +0.05 / +0.05 | 4x12_closed |
| modern_clean      | 0.6 / -1.7 | 0.7 / -1.7 / 2 / 0.70 | 0.9 / 0.20 | 0.10 / 0.02 | 6 | +0.05 /  0.00 / +0.05 | 2x12 |
| modern_high_gain  | 1.9 / -1.2 | 2.1 / -1.4 / 4 / 0.55 | 1.6 / 0.70 | 0.50 / 0.04 | 7 | -0.05 / -0.10 / +0.10 | 4x12_closed |
| class_a           | 1.1 / -1.5 | 1.3 / -1.6 / 2 / 0.60 | 1.1 / 0.45 | 0.35 / 0.07 | 4 |  0.00 / +0.05 /  0.00 | 2x12 |
| blackface         | 0.9 / -1.6 | 1.1 / -1.7 / 2 / 0.65 | 1.0 / 0.35 | 0.25 / 0.08 | 5 | +0.05 / -0.05 / +0.10 | 2x12 |
| metal             | 2.0 / -1.0 | 2.4 / -1.3 / 4 / 0.50 | 1.8 / 0.80 | 0.60 / 0.03 | 9 | -0.10 / -0.20 / +0.15 | 4x12_closed |
| boutique          | 1.2 / -1.5 | 1.4 / -1.6 / 2 / 0.60 | 1.2 / 0.50 | 0.30 / 0.06 | 4 |  0.00 / +0.05 / +0.05 | 1x12 |

Adding new amps: append to `AMP_MODELS` in `r13_amp_designer.js` — that's it.
The `amp_model` setter accepts numeric indices too (0..N-1), so a slider UI
maps to whatever the table currently contains.

## CAB_PROFILES table

Procedural IR generation — no .wav files shipped. `_buildCabIR()` constructs
the IR per profile by:
1. Synthesising a short shaped impulse + N early reflections + decaying noise
2. Applying a 1-pole HP (low_cut), 1-pole LP (high_cut), biquad peaking
   presence bump (peak_hz / peak_q / peak_gain_db)
3. Normalising peak to 0.95

| Cab            | Length ms | Low cut | High cut | Presence (Hz / Q / dB) | Reflections | On-axis tilt |
|----------------|-----------|---------|----------|------------------------|-------------|--------------|
| 1x12           | 35 |  90 | 4500 | 2500 / 1.8 / 4.0 | 2 | 1500 |
| 2x12           | 40 |  75 | 4200 | 2300 / 1.6 / 3.5 | 3 | 1300 |
| 4x10           | 32 | 100 | 5500 | 3000 / 2.2 / 5.0 | 3 | 1800 |
| 4x12_open      | 45 |  80 | 4000 | 2200 / 1.5 / 3.0 | 4 | 1200 |
| 4x12_closed    | 50 |  70 | 3800 | 2000 / 1.7 / 4.0 | 5 | 1000 |
| vintage_1x12   | 38 |  95 | 4100 | 2400 / 1.9 / 4.5 | 2 | 1400 |
| 2x10_combo     | 30 | 110 | 5200 | 3200 / 2.0 / 4.0 | 2 | 1600 |

`mic_position` morphs the IR by linearly shifting the high-cut down by
`on_axis_tilt` Hz as `mic_position` goes 0 → 1, and pushing the low-cut up
by 30 Hz. On-axis = brighter, full HF; off-axis = darker, slightly more LF
rolloff.

## Migration from procedural to real IRs

If a calibration target needs better than -25 dB null-diff:
1. Drop a 24-bit / 48 kHz mono cab IR .wav into
   `public/assets/cab-ir/{cab_name}.wav`.
2. Replace `_buildCabIR()` in `r13_amp_designer.js` with a `fetch` →
   `decodeAudioData` lookup keyed on `cab_model`.
3. Re-export `CAB_PROFILES` as a metadata-only object (length / mic-position
   morph still derives from it).

The public param surface stays identical so no calibration JSON or mapping
file regresses.

## Test result

```
$ node tests/r13_amp_designer.test.js
[r13_amp_designer] builder smoke ............ ok
[r13_amp_designer] amp_model preset switch .. ok
[r13_amp_designer] cab_model + mic morph .... ok
3/3 passed
```

Tests use a hand-rolled mock AudioContext (no Jest / Vitest required) and
verify: the builder returns the canonical {input, output, paramTargets}
shape; switching `amp_model` changes the underlying gain/curve state;
switching `cab_model` rebuilds the convolver buffer with a different
length.
