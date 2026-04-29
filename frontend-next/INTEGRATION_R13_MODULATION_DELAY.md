# R13 — Modulation Delay (`modulation_delay`)

Adds a NEW DSP node type **`modulation_delay`** that mirrors Logic Pro's
stock Modulation Delay (also covers Chorus, Flanger, and Tape Delay with
appropriate param defaults — same DSP core, different presets).

Tape-style chorus/flanger combo. Pipeline:

1. Variable delay line per channel (0.1..80 ms range).
2. LFO modulates delay time (sine / triangle / random / square shapes).
3. Feedback loop — HPF → LPF → asymmetric tape saturation → gain.
4. Wet/dry equal-power crossfade.
5. Stereo width via offset LFO phases (`stereo_phase`).

## Files added

| Path | Role |
|---|---|
| `src/audio/builders/r13_modulation_delay.js` | Builder. Tries the worklet; on failure returns a static-DelayNode fallback so audio still flows. Exposes `modulation_delay` in `R13_BUILDERS`. |
| `src/lib/web-audio-plugins/worklets/r13-modulation-delay-processor.js` | Real-DSP worklet. Per-channel `DelayLine` + `LFO` + per-sample feedback path with one-pole HPF + LPF + asymmetric soft-clip saturator. |
| `tests/r13_modulation_delay.test.js` | Builder smoke test + LFO-modulated-delay rendering test (OfflineAudioContext, no real audio device). |
| `../doseedo-desktop/tools/calibration/configs/modulation-delay-web.json` | Topology config for `tools/calibration/auto_driver/`. Includes 5 starter presets (Subtle Chorus / Jet Flanger / Tape Echo / Inverted Slap / Strange Drift). |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring in `WebAudioDSPEngine.js`

Two single-line additions (R9-style):

**1.** Top of the file, alongside other builder imports:

```js
import r13ModulationDelayBuilders from './builders/r13_modulation_delay.js';
```

**2.** Inside the `NODE_BUILDERS` map (around line 829), spread it in:

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,                  // algo_reverb (FDN, 4 algos)
  ...r13ModulationDelayBuilders,  // ← adds modulation_delay
};
```

**3. (Worklet pre-load — recommended).** Add the worklet filename to the
existing `_ensurePhase1Worklets` loader so first-build doesn't fall through
to the static-delay fallback path:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-modulation-delay-processor.js', import.meta.url)
);
```

## Parameter schema

| Param             | Type / Range          | Default | Notes |
|-------------------|-----------------------|---------|-------|
| `delay_ms`        | 0.1..80 ms            | 8.0     | Base delay. <5 ms = flanger, 5–30 = chorus, >30 = slap. |
| `rate_hz`         | 0.05..10 Hz           | 0.5     | LFO frequency (both channels). |
| `depth`           | 0..100 %              | 30      | LFO sweep depth. 100% = ±40 ms. |
| `feedback`        | -100..+100 %          | 0       | Signed. Negative inverts polarity. Hard-capped to ±0.95 internally. |
| `tape_saturation` | 0..1                  | 0       | Dry/wet of asymmetric soft-clip in feedback path. |
| `lfo_shape`       | enum/string           | 'sine'  | 0/'sine', 1/'triangle', 2/'random' (smoothed S&H), 3/'square'. |
| `stereo_phase`    | 0..360 deg            | 90      | Phase offset between L and R LFOs. |
| `low_cut`         | 20..2000 Hz           | 50      | 1-pole HPF in feedback path. |
| `high_cut`        | 1000..20000 Hz        | 12000   | 1-pole LPF in feedback path. |
| `mix`             | 0..1                  | 0.5     | Dry/wet equal-power crossfade. |

`@<paramId>` modulation works on every param. Numeric params bind directly
to AudioParams when the worklet is active, or to fallback-path setters
(delay-time / feedback-gain / dry-wet gains) otherwise.

## Signal flow

```
              ┌──────────── dry ────────────┐
              │                             │
input ──┬─→ delay_L (LFO_L) ─┬──────────── wet_L ──┴──→ outL
        │           ▲        │
        │           │        └─→ HPF → LPF → tape_sat → fb_gain ┐
        │           └─────────────────────────────────────────────┘
        │
        └─→ delay_R (LFO_R, phase = LFO_L + stereo_phase) ─┬─ wet_R ──→ outR
                    ▲                                       │
                    │                                       └─→ (same fb chain)
                    └────────────────── (feedback) ─────────────────────────
```

## Fallback behavior

If `new AudioWorkletNode(ctx, 'r13-modulation-delay-processor')` throws
(worklet not loaded, or running in a non-browser context), the builder
returns a primitive path:

- `ctx.createDelay(0.2)` per bus
- A `GainNode` for feedback
- Equal-power dry/wet `GainNode` pair

Audio still flows; LFO modulation is inactive in fallback. `delay_ms`,
`feedback`, and `mix` remain functional via `customSetter`s. Other params
become no-ops in fallback. The engine sees a consistent
`{ input, output, paramTargets }` shape regardless.

Calibration MUST run with the worklet active — `auto_driver`'s
`web_renderer.py` already awaits worklet readiness before each render.

## Testing

`tests/r13_modulation_delay.test.js` covers:

1. Builder smoke — returns `{ input, output, paramTargets }` with audio
   nodes connectable end-to-end through OfflineAudioContext.
2. `@`-binding — modulated `mix` param produces a `paramTargets` entry.
3. LFO modulation — render 0.5 s of impulse train through a high-feedback
   high-depth config; assert non-trivial RMS on the wet output AND that
   peak energy is distributed across time (not concentrated at one delay
   tap, which would indicate no modulation).

Run with:

```
cd Do/doseedo-next
node --test tests/r13_modulation_delay.test.js
```

The tests stub `AudioWorkletNode` so they exercise the fallback path
without requiring a worklet-loaded context.

## Calibration plan

`tools/calibration/configs/modulation-delay-web.json` is the topology
config. Calibration order (low to high effort):

1. **Subtle Chorus** preset — small depth, no feedback. Dominant params:
   `delay_ms`, `rate_hz`, `depth`, `mix`. Easy curve fit.
2. **Tape Echo** preset — saturator + filters become audible. Validates
   the feedback-path tone shaping.
3. **Jet Flanger** preset — high feedback (75%) stresses the HPF+LPF
   stability margin and inverts polarity halfway. Most likely to need a
   piecewise feedback curve.
4. **Strange Drift** preset — random LFO shape. Validates the smoothed
   S&H matches Logic's "random" mode (Logic may not smooth — if null-diff
   misses, expose a `random_smoothing` param in a follow-up).

Target -40 dB RMS null-diff per preset against a Logic bounce on
`gen://drums` source.

## Notes

- Logic exposes the same DSP core under three plugin names (Chorus,
  Flanger, Tape Delay) with different param defaults and UI surfaces.
  Once Modulation Delay is calibrated, those three Tier-1 entries can
  reuse this engine node with mapping-level param defaults — no further
  DSP work needed.
- `stereo_phase` is the OFFSET between L and R LFOs, anchored on every
  change. Logic's UI tends to call this "L/R phase" or "stereo offset".
- Feedback ordering (HPF → LPF → sat) was chosen so the saturator sees
  DC-blocked, band-limited material. The LPF before the saturator is
  load-bearing for the tape-repeat darkening character.
