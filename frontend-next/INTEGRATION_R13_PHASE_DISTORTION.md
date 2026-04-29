# R13 — Phase Distortion (`phase_distortion`)

Adds a NEW DSP node type **`phase_distortion`** that mirrors Logic Pro's
stock Phase Distortion effect. Casio-CZ-style bright/dark phase-domain
character, applied as a colour / distortion processor (not a synth osc).

## Why this isn't just a tanh waveshaper

Logic's `Distortion II`, `Overdrive`, and `Saturator` already cover smooth
saturation. Phase Distortion is in the catalogue *because* its transfer
curves are sharp and asymmetric — closer to the Casio CZ-101 phase-domain
remap windows than to anything in `buildWaveshaper`. Six curve families
implement the CZ character: three primary (`saw`/`square`/`pulse`) and three
resonance variants (`res1`/`res2`/`res3` with 5/9/13-cycle ripples).

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_phase_distortion.js` | Builder. Stereo `WaveShaperNode` pair; curves regenerate on shape-param mutation. Exposes `phase_distortion` in `R13_BUILDERS`. |
| `src/lib/web-audio-plugins/worklets/r13-phase-distortion-processor.js` | Stub worklet. Reserved for a future sample-accurate variant where `pd_curve` is interpolated at audio rate. Not instantiated by the current builder. |
| `tests/r13_phase_distortion.test.js` | Unit tests — curve smoke, monotonicity, regeneration on knob drag, builder graph shape. |
| `../doseedo-desktop/tools/calibration/configs/phase-distortion-web.json` | Topology config for the calibration harness. |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style R9 / R13 Space Designer use.

**1. Top of the file, alongside other builder imports:**

```js
import r13PhaseDistortionBuilders from './builders/r13_phase_distortion.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 830):**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,
  ...r13PhaseDistortionBuilders,   // ← adds phase_distortion
};
```

**3. (Optional)** Register the worklet module so future sample-accurate
upgrades activate without engine surgery. The current builder works
whether or not this runs:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-phase-distortion-processor.js', import.meta.url)
);
```

## Parameter schema

| Param          | Type / Range          | Default | Notes |
|----------------|-----------------------|---------|-------|
| `pd_amount`    | 0 – 100 %             | 50      | Drive depth. Accepts 0..100 (Logic surface) or 0..1 (normalised). |
| `pd_curve`     | enum (6 families)     | `'saw'` | Curve family. See "Curve families" below. |
| `pd_asymmetry` | -1 – +1               | 0       | Pre-warp bias on `x` before the curve fn. |
| `pre_gain`     | -12 – +12 dB          | 0       | Input gain (dB) BEFORE the shaper. |
| `post_gain`    | -12 – +12 dB          | 0       | Output gain (dB) AFTER the shaper / tone shelf. |
| `tone`         | -12 – +12 dB          | 0       | Post-shaper high-shelf tilt @ 4 kHz. |
| `mix`          | 0 – 1                 | 1       | Wet/dry crossfade. |

`@<paramId>` modulation works on every param. `tone` binds direct to the
high-shelf `gain` AudioParam; the rest go through `customSetter`s — most
trigger a `regenerateCurve()` call.

## Curve families

Each curve is `f(x, amount, asym) ∈ [-1, +1]`, evaluated over 4096 samples.
All curves preserve `f(0)=0` and `f(±1)=±1` at `amount=0` (identity at zero
drive).

| Family   | Character | Math sketch |
|----------|-----------|-------------|
| `saw`    | Bright fundamental + harmonic ramp. | Map `x` → phase angle, warp via `phase^(1 + 4·k)`, take `sin(2π · w)`. CZ's brightest window. |
| `square` | Symmetric phase squash (square-ish on sine). | Knee at `±(1 - 0.95·k)`, `tanh`-style soft hinge above. |
| `pulse`  | Narrow pulse, asym-controlled width. | Cubic crossover with `duty = 0.5 - 0.45·k` rail-to-rail. |
| `res1`   | Resonance — 5 cycles, env-decayed toward rails. | `xb + sin(xb·5π) · (1 - |xb|) · k · 0.6`. |
| `res2`   | Brighter resonance — 9 cycles, less env dip. | `xb + sin(xb·9π) · (1 - 0.5|xb|) · k · 0.5`. |
| `res3`   | Aggressive resonance — 13 cycles, no env dip. | `xb + sin(xb·13π) · k · 0.45`. |

`xb` = `_bend(x, asym)` — pre-warps `x` so positive/negative halves can be
biased independently.

## Signal flow

```
                     ┌→ shaperL ─┐
input → preGain → splitter        └→ merger → toneShelf → postGain → wetGain ─┐
                     └→ shaperR ─┘                                              ├→ output
input ──────────────────────────────────────────────────────────→ dryGain ────┘
```

Stereo splitter / merger keeps L/R independent, matching Logic's stereo-effect
convention. Mono input feeds both shapers identically (the splitter still
binds; right shaper just gets a copy on channel 0).

## Performance notes

- Curve regen is a single `for` loop over 4096 samples (~0.02 ms on M1). Fine
  to run synchronously inside a knob `customSetter`.
- `WaveShaperNode.curve =` setter copies the array internally, so we share
  the same `Float32Array` between L/R for cache friendliness.
- `oversample = '4x'` — mandatory. Removing it produces aliasing above
  ~10 kHz on `square` / `res3` at high drive that the R12 null-diff will
  flag immediately.
- No allocations in the audio thread — everything happens in the WebAudio
  graph or in the `customSetter` (UI thread).

## Testing

```bash
# Smoke (Jest/Vitest):
cd doseedo-next && npx jest tests/r13_phase_distortion.test.js
```

Tests cover:

1. `makePDCurve` returns Float32Array of length 4096
2. Curve passes through origin at `amount=0` (all families)
3. `makePDCurve` outputs are clamped to `[-1, +1]`
4. All six curve families produce *different* outputs for the same `amount`
5. `buildPhaseDistortion` returns the standard `{input, output, paramTargets}` shape
6. Param targets registered for every modulated knob
7. Calling `paramTargets.pd_amount.customSetter(80)` mutates the shaper curve
8. Calling `paramTargets.pd_curve.customSetter('res2')` switches family and re-renders curve
9. `tone` param binds directly to the high-shelf AudioParam

## Calibration handoff

Topology config: `doseedo-desktop/tools/calibration/configs/phase-distortion-web.json`.
Add a `PluginEntry` row for "Phase Distortion" in
`tools/calibration/auto_driver/registry.py` referencing this config.

Per-param sweep ranges in the config double as the curve-fit search domain.
The `notes` field on each param tells the calibration engineer where to
expect Logic vs. web behaviour to diverge (esp. `pd_curve` family selection
and resonance variants).

## Limitations / future work

- `pd_curve` family changes are step-quantised — switching families during
  playback can produce a click. The worklet stub
  (`r13-phase-distortion-processor.js`) exists so a future sample-accurate
  variant can crossfade between families at audio rate. Until then, expose
  `pd_curve` as a discrete UI control only.
- `pd_asymmetry` is implemented as a pre-curve `x → x^(1±a)` warp. Logic's
  Phase Distortion may use a different asymmetry model — calibration should
  confirm and add piecewise breakpoints if R² < 0.95.
- True Casio CZ-style phase distortion would require a Hilbert transform
  or PLL on the input. We chose the WaveShaper-curve approximation
  intentionally (cheap, sample-rate-agnostic, audio-faithful for harmonic
  inputs). If a customer reports a perceptual mismatch on transient-heavy
  material (drums), revisit with a Hilbert-transform variant in a worklet.
