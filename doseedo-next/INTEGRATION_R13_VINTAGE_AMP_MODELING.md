# R13 — Vintage Amp Modeling

Adds a NEW DSP node type **`vintage_amp_modeling`** — a composite tube-amp
chain keyed off an `amp_model` preset table (8 era-correct heads). Differs
from `circuit_fender_bassman` (R4) in being preset-driven rather than purely
parametric, and from a future `amp_designer` node by targeting a narrow set
of vintage heads with deeply-modeled per-amp quirks (bias drift, NFB amount,
output transformer impedance match, era-specific tube types).

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_vintage_amp_modeling.js` | Composite builder. Constructs R2 / R3 worklets (with native-node fallbacks) wired into a fixed topology. Exposes `vintage_amp_modeling` in `R13_VINTAGE_AMP_BUILDERS`. Also re-exports `AMP_MODEL_PRESETS`, `CAB_PRESETS`, `MIC_PRESETS` for UI / test introspection. |
| `src/lib/web-audio-plugins/worklets/r13-vintage-amp-modeling-processor.js` | OPTIONAL fused worklet — single-pass equivalent of the composite chain. Builder doesn't depend on it; provided for hosts that prefer one node over many. |
| `tools/calibration/configs/vintage-amp-modeling-web.json` (in `doseedo-desktop`) | Calibration spec — param ranges, preset table mirror, and per-stage calibration strategy. |
| `tests/r13_vintage_amp_modeling.test.js` | Builder smoke + amp-model preset switch test. |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style R4/R9 use.

**1. At the top of the file, alongside other builder imports:**

```js
import r13VintageAmpBuilders from './builders/r13_vintage_amp_modeling.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 829), spread it in
   alongside the other R-spreads:**

```js
const NODE_BUILDERS = {
  // ... existing entries ...
  ...r9Builders,
  ...r13VintageAmpBuilders,   // ← adds vintage_amp_modeling
};
```

**3. (Optional) Register the fused worklet** so the builder can opt into the
single-pass path. The builder's behaviour is identical with or without it:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-vintage-amp-modeling-processor.js',
          import.meta.url)
);
```

The composite already falls back to native Web Audio nodes when R2/R3
worklets aren't registered — so the schema renders + audio flows even if
nothing else is loaded yet.

## Topology

```
input → input_pad
     → tube_preamp_v12ax7         (R2 wdf_tube_amp │ harmonic-curve fallback)
     → vintage_tone_stack         (R2 wdf_tone_stack │ 3-band biquad fallback)
     → phase_inverter             (tanh waveshaper + tilt shelf)
     → push_pull_power_amp        (R2 wdf_tube_amp │ per-tube-type fallback)
     → output_transformer         (R3 wdf_transformer │ atan fallback)
     → output_transformer_LP      (biquad LP, cutoff from preset)
     → psu_sag_with_drift         (R3 wdf_power_supply_sag │ static gain fallback)
     → vintage_cab_ir             (ConvolverNode w/ synthetic per-cab IR)
     → mic_position_pre_eq        (HP + peak biquads + position tilt)
     → presence                   (highshelf — modelled in NFB position)
     → output_pad → output
```

## Parameter schema

| Param         | Type / Range                     | Default            | Notes |
|---------------|----------------------------------|--------------------|-------|
| `amp_model`   | enum (8 keys, see table below)    | `tweed_5e3`        | Switch retunes underlying R2/R3 nodes in place — no graph rebuild |
| `gain`        | 0..10                              | 5.0                | Preamp drive |
| `bass`        | 0..1                               | 0.5                | Tone stack |
| `mid`         | 0..1                               | 0.5                | Tone stack |
| `treble`      | 0..1                               | 0.5                | Tone stack |
| `presence`    | 0..1                               | 0.3                | Post-NFB high shelf |
| `master`      | 0..1                               | 0.5                | Pushes the power amp |
| `bias`        | 0..1                               | 0.5                | Asymmetric clipping override |
| `nfb`         | 0..1                               | 0.5                | NFB amount — adjusts xfmr LP + input pad |
| `cab_model`   | enum: 1x12_alnico / 2x12_celestion / 4x12_greenback / 4x10_jensen / 1x15_jbl / 2x10_oxford | `4x12_greenback` | Synthetic IR (no .wav files shipped) |
| `mic_type`    | enum: sm57 / sm7 / r121 / u87      | `sm57`             | Per-mic HP + presence-peak biquad |
| `mic_position`| 0..1                               | 0.3                | 0=on-cap (bright), 1=off-axis (dark) |
| `output_level`| 0..4                               | 1.0                | Final output gain |

## Amp-model preset table

Each preset bakes in: `tube_type` (preamp valve), `power_tube_type` (output
stage), `harmonic_signature` ([h2, h3, h4, h5] relative weights),
`transformer_color` ({lp_cutoff_hz, sat_amount}), `bias_drift` (asymmetric
offset), `nfb_default`, `tone_voicing` ({bass, mid, treble} centres in Hz),
`presence_freq`, and `gain_makeup` (output level normalisation).

| Model            | Pre / Power | h2/h3/h4/h5         | Xfmr LP / Sat     | Bias  | NFB  | Tone (B/M/T Hz)    | Char |
|------------------|-------------|---------------------|-------------------|-------|------|--------------------|------|
| `tweed_5e3`      | 12AX7 / 6V6  | 0.85 / 0.45 / 0.25 / 0.10 | 4500 / 0.55 | 0.30 | 0.15 | 90 / 700 / 4500    | Cathode-bias class-A, warm 6V6, low NFB. |
| `tweed_5f6`      | 12AX7 / 6L6  | 0.75 / 0.55 / 0.30 / 0.15 | 5500 / 0.45 | 0.05 | 0.30 | 80 / 500 / 5000    | Bassman 5F6-A, fixed-bias 5881, scoop-able mids. |
| `vox_ac30`       | 12AX7 / EL84 | 0.70 / 0.65 / 0.40 / 0.25 | 6500 / 0.60 | 0.45 | 0.05 | 100 / 1000 / 6500  | AC30TB top-boost. Cathode-bias EL84, no NFB. |
| `marshall_plexi` | 12AX7 / EL34 | 0.55 / 0.75 / 0.45 / 0.30 | 6000 / 0.50 | 0.10 | 0.55 | 110 / 650 / 5500   | 1959 Super Lead, fixed-bias EL34, classic NFB. |
| `marshall_jcm800`| 12AX7 / EL34 | 0.45 / 0.85 / 0.55 / 0.35 | 5500 / 0.55 | 0.15 | 0.65 | 110 / 600 / 6000   | JCM800 2203 — cascaded gain. |
| `hiwatt`         | 12AX7 / EL34 | 0.65 / 0.50 / 0.20 / 0.10 | 8000 / 0.30 | 0.00 | 0.75 | 90 / 800 / 5500    | DR-103 — clean, articulate, very high headroom. |
| `orange_or120`   | 12AX7 / EL34 | 0.55 / 0.70 / 0.40 / 0.20 | 4500 / 0.55 | 0.20 | 0.45 | 100 / 900 / 4800   | OR120 — mid-forward, "graphic" tone control. |
| `silvertone`     | 12AX7 / 6L6  | 0.80 / 0.50 / 0.25 / 0.15 | 3800 / 0.65 | 0.35 | 0.20 | 95 / 750 / 4000    | 1484 — lo-fi early-Beatles flavour. |

## How `amp_model` retunes the chain

When `amp_model` changes, `applyAmpModel(modelKey)` runs once:

1. **Preamp curve** — `_makeHarmonicCurve(harmonic_signature)` rebakes the
   waveshaper LUT. Even-order h2/h4 produce warmth, odd-order h3/h5
   produce squared aggression. Dominated by tanh; harmonics colour rather
   than replace.
2. **Power-tube curve** — `_makePowerTubeCurve(power_tube_type, sig,
   bias_drift)` rebakes with a `k` factor mapping per tube class
   (EL84=1.6, EL34=1.4, 6V6=1.2, 6L6=1.1) plus an asymmetric x-offset for
   bias drift.
3. **Output transformer LP** — `xfmrLP.frequency` updated to
   `transformer_color.lp_cutoff_hz`.
4. **Tone-stack centres** — `bassFb`, `midFb`, `trebFb` frequencies updated
   to `tone_voicing` (fallback path only; the worklet path takes care of
   itself).
5. **Presence frequency** — `presence.frequency` updated to
   `presence_freq`.
6. **PSU sag** — sag amount updated to `0.3 + 0.2 * bias_drift` (more bias
   drift = more compression headroom needed).
7. **Output pad** — `outputPad.gain` updated to `gain_makeup` so all 8
   models land at roughly the same perceived loudness at unity master.

This is a parameter-only update — no AudioGraph rebuild — so automation
between models is smooth.

## Tube-type behaviour

The power-amp curve generator uses a tube-class-specific drive coefficient
`k`:

| Tube  | k    | Character at clip onset |
|-------|------|-------------------------|
| EL84  | 1.6  | Earliest breakup, looser knee. AC30 territory. |
| EL34  | 1.4  | Mid-focused, broad knee. British rock standard. |
| 6V6   | 1.2  | Soft, warm clip onset. Tweed Deluxe / Princeton. |
| 6L6   | 1.1  | Stiff until breakup, then crumbles fast. Bassman / silvertone. |

Combined with the per-model `harmonic_signature`, this produces audibly
distinct curves even between two amps that share the same power tubes (e.g.
`marshall_plexi` vs. `marshall_jcm800` — both EL34 but JCM has a
significantly more h3/h5-heavy signature).

## Acceptance evidence

1. `node --check` passes on the builder + the optional worklet.
2. The builder returns `{ input, output, paramTargets }` — same shape as
   every other builder in the engine.
3. Unit test (`tests/r13_vintage_amp_modeling.test.js`) covers:
   - Builder smoke (returns the correct shape, audio flows).
   - amp_model preset switch (audible level + spectral centroid change
     between `tweed_5e3` and `marshall_plexi` at identical input).
4. Calibration config (`vintage-amp-modeling-web.json`) lists per-stage
   measurement strategy + Logic param-index hints.

## DO NOT-list (followed)

- `WebAudioDSPEngine.js` was not modified. Registration instructions live
  in this file.
- R2 / R3 / R4 builders were not modified.
- The existing `circuit_fender_bassman` node was not replaced;
  `vintage_amp_modeling` is registered as a new type.
- No commit was made.
