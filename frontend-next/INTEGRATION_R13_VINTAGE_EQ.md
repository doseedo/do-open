# R13 — Vintage EQ Collection (1073 + API)

Adds two NEW DSP node types modelled on Logic's Vintage EQ Collection:

- **`vintage_1073`** — Neon-style, modeled on the Neve 1073 channel EQ.
- **`vintage_api`** — Punchy-style, modeled on the API 550A.

The Pultec model is **already shipped** as `circuit_pultec_eq` in
`src/audio/builders/r4.js`. R13 does not modify or duplicate it.

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_vintage_eq.js` | Builders + frequency tables + Q-vs-gain curve helpers (`buildVintage1073`, `buildVintageAPI`). Default export is the `NODE_BUILDERS` map. |
| `src/lib/web-audio-plugins/worklets/r13-vintage-1073-processor.js` | Optional inductor saturation worklet (3rd-order soft clip + k-rate `ind_drive`). |
| `src/lib/web-audio-plugins/worklets/r13-vintage-api-processor.js`  | Same shape, hotter curve. |
| `tools/calibration/configs/vintage-eq-1073-web.json` (desktop repo) | Param schema + Logic param hints for null-diff calibration. |
| `tools/calibration/configs/vintage-eq-api-web.json`  (desktop repo) | Same shape for API 550. |
| `tests/r13_vintage_eq.test.js` | Builder smoke + Q-vs-gain remap regression. |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style as R4 / R9.

**1. Top of the file, alongside other builder imports:**

```js
import r13Builders from './builders/r13_vintage_eq.js';
```

**2. Inside the `NODE_BUILDERS` map, spread R13 in (after R9 is fine):**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,
  ...r13Builders,            // ← adds vintage_1073, vintage_api
};
```

**3. Register the optional worklet modules** (only required if you want the
per-sample inductor stage in lieu of the WaveShaperNode fallback). Wherever
the engine boots its AudioContext (typically `_ensureContext()` /
`loadWorklets()`):

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-vintage-1073-processor.js', import.meta.url)
);
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-vintage-api-processor.js', import.meta.url)
);
```

Builder uses `_safeWorklet()` from R4/R9 convention, so missing worklets are
not fatal — the WaveShaperNode fallback is audibly identical at moderate
drive.

## 1073 — frequency tables

| Param | Indices | Hz table |
|---|---|---|
| `low_cut_freq`     | 0..4 | `[off, 50, 80, 160, 300]` (index 0 bypasses HPF — internal HPF set to 10 Hz) |
| `low_shelf_freq`   | 0..3 | `[35, 60, 110, 220]` (stock 1073 has 3; index 3 reserves 220 Hz for the Vintage EQ Collection superset) |
| `mid_freq`         | 0..5 | `[360, 700, 1600, 3200, 4800, 7200]` |
| `high_shelf_freq`  | —    | fixed at **12 000 Hz** (no knob; Neve 1073 has no HF freq selector) |

Gain ranges: low_shelf, mid, high_shelf all `−18..+18 dB`.

## 1073 — Q-vs-gain curve

```
mid_q preset → base Q
   broad   → 0.7
   medium  → 1.4
   narrow  → 2.8

Q(gain_dB, preset) = base * (0.5 + clamp(|gain_dB| / 18, 0, 1) * 1.1)
                   = base * 0.5  at |gain|=0
                   = base * 1.6  at |gain|=18
```

Implemented as `_internals.n1073MidQ(gainDb, preset)`. Setting `mid_gain`
also triggers a Q recompute so a knob-drag updates both at once.

## API — frequency table

```
[50, 100, 200, 400, 800, 1500, 3000, 5000, 7500, 12500, 15000, 20000]  Hz
                                                          (12 indices each band)
```

All 4 bands (`band1_freq … band4_freq`) draw from this table. Default
indices: `1, 3, 6, 9` → 100 / 400 / 3000 / 12500 Hz.

Gain range per band: `−12..+12 dB`.

## API — proportional-Q curve

```
Q(gain_dB) = 0.6 + clamp(|gain_dB| / 12, 0, 1) * (2.5 - 0.6)
           = 0.6  at |gain|=0   (very broad)
           = 2.5  at |gain|=12  (musical bell)
```

Implemented as `_internals.apiProportionalQ(gainDb)`. There is **no Q knob**
on a real API 550 — Q is a function of gain. Setting `bandN_gain` triggers
both gain and Q updates.

## Inductor saturation curve (both circuits)

3rd-order soft clip with a small even-harmonic bias, scaled by drive:

```
k       = 1 + drive * 4   (1073)
        = 1 + drive * 5   (API — slightly hotter)
y       = (k*x − (k*x)^3 / 3) * (1 - drive*0.05) + (k*x)^2 * 0.04 * drive
output  = clamp(y / (k*0.6), -1, 1)
```

`makeInductorCurve(drive)` materialises the static WaveShaper LUT (length
2048, oversample `2x`). The worklet does the same math per-sample so it
tracks `ind_drive` automation continuously — useful for calibration sweeps
where the saturation knob is automated under render.

## Logic param order (calibration starting point)

Logic doesn't expose Vintage EQ via the registered AudioComponent registry
either (same caveat as the Pultec / Channel EQ entries — see
`docs/logicx_format.md` § kind=1 payload). Recover the param indices via
the kind=7 patcher channel landed in desktop `b292eec`, then map per the
`logic_param_hints` block in the calibration JSONs:

- 1073: 9 params → `low_cut_freq, low_shelf_freq, low_shelf_gain, mid_freq, mid_gain, mid_q, high_shelf_gain, inductor_saturation, output_gain`
- API: 10 params → `band1_freq, band1_gain, band2_freq, band2_gain, band3_freq, band3_gain, band4_freq, band4_gain, inductor_saturation, output_gain`

R10 calibration harness should null-diff each preset bounce against the web
render at the same parameter values. Linear-curve target: ≤ −40 dB RMS at
0.5 s of pink noise per band setting.

## Param surface summary

| Plugin | Knobs | Notes |
|---|---|---|
| `vintage_1073` | low_cut_freq, low_shelf_freq, low_shelf_gain, mid_freq, mid_gain, mid_q, high_shelf_gain, inductor_saturation, output_gain | mid_gain change auto-updates mid_q's Q |
| `vintage_api`  | band1..4_freq, band1..4_gain, inductor_saturation, output_gain | bandN_gain change auto-updates Q |

Both circuits accept `@param_id` modulation on every knob; static numeric
values bind once at build time.
