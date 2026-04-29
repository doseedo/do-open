# INTEGRATION_R13_MULTIPRESSOR — 4-band parallel multiband compressor

**Round:** R13 — Tier 2 plugin parity (`Multipressor` row).
**Author:** Agent R13.
**Status:** DSP shipped (worklet + non-worklet builders); calibration pending.

## What ships

- `src/audio/builders/r13_multipressor.js` — registers a new node type
  `multipressor` via `buildMultipressor`. Dual-path: an
  `AudioWorkletNode` running `r13-multipressor-processor` is preferred,
  with a transparent `BiquadFilterNode + DynamicsCompressorNode`
  fallback when the worklet module hasn't been registered yet.
- `src/lib/web-audio-plugins/worklets/r13-multipressor-processor.js` —
  single-pass worklet implementation (LR4 split + per-band feed-forward
  compression + lookahead + master gain).
- `tools/calibration/configs/multipressor-web.json` (desktop tree) —
  topology + parameter ranges + Logic-param-id hints for the auto-driver
  harness.

## Engine wiring (do once when promoting from review)

```js
// src/audio/WebAudioDSPEngine.js
import r13MultipressorBuilders from './builders/r13_multipressor.js';

const NODE_BUILDERS = {
  // ...existing...
  ...r13MultipressorBuilders,    // multipressor
};
```

Optionally pre-warm the worklet inside `_ensurePhase1Worklets`:

```js
await ctx.audioWorklet.addModule(
  '/src/lib/web-audio-plugins/worklets/r13-multipressor-processor.js'
);
```

If the module isn't loaded, the builder silently uses the BiquadFilter +
DynamicsCompressor fallback — same param surface, same paramTargets
shape, no graph-build error.

## Pipeline / topology

```
input ── lookahead (DelayNode ≤ 10 ms) ──┬─ LR-LP@xo1 ──── band0 comp ──┐
                                          ├─ LR-HP@xo1 → LR-LP@xo2 ── band1 comp ──┤
                                          ├─ LR-HP@xo2 → LR-LP@xo3 ── band2 comp ──┤── Σ ── output_gain ── output
                                          └─ LR-HP@xo3 ────────────── band3 comp ──┘
```

Per-band: `splitTap → DynamicsCompressor → makeup_gain → bypassWet`,
plus a parallel `splitTap → bypassDry` so a band can be bypassed without
re-routing.

## Linkwitz-Riley crossover topology — why this works

A Linkwitz-Riley 4th-order (LR4) crossover is constructed by cascading
two Butterworth 2nd-order biquads at the same cutoff frequency, each
with `q = 1/√2 ≈ 0.7071`.

Key property: **`LP_LR4(f) + HP_LR4(f) = ALLPASS(f)`**. The two
LR4-filtered branches are 360° out of phase across the crossover, but
their sum has unity magnitude everywhere — a pure phase rotation. So
when no band compresses, summing the four bands reconstructs the input
(magnitude-flat, with predictable phase).

Cascaded LR4s for the inner bands (`HP@xo[i] → LP@xo[i+1]`) hold this
property exactly when the crossovers are well-separated. With
overlapping crossovers (e.g. `xo2 = 1.05 × xo1`) some band-edge
amplitude ripple appears; the calibration harness should pick crossover
defaults that respect ≥ 1 octave separation.

Both the worklet path (handcoded biquads, RBJ formulas) and the
fallback path (`BiquadFilterNode` with `type='lowpass'/'highpass'`,
q=0.7071) implement the LR4 cascade identically. Web Audio's
`BiquadFilterNode` already uses Direct-Form-I with double-precision
state, so cascading two of them at the same cutoff is exactly an LR4.

## Parameters

| Param | Range | Unit | Notes |
|---|---|---|---|
| `crossover_1` | 50–500 | Hz | Bass / low-mid split |
| `crossover_2` | 200–2000 | Hz | Low-mid / high-mid split |
| `crossover_3` | 1000–10000 | Hz | High-mid / highs split |
| `band[N]_threshold_db` | -60 – 0 | dB | Per-band, N ∈ {1,2,3,4} |
| `band[N]_ratio` | 1 – 30 | — | |
| `band[N]_attack_ms` | 0.1 – 300 | ms | |
| `band[N]_release_ms` | 1 – 2000 | ms | |
| `band[N]_gain_db` | -24 – +24 | dB | Per-band makeup |
| `band[N]_bypass` | 0/1 | — | Routes splitTap → summer dry |
| `lookahead_ms` | 0 – 10 | ms | Pre-split delay (both paths) |
| `output_gain` | -24 – +24 | dB | Master post-Σ trim |

Per-band keys are accepted in three spellings — `band1_threshold_db`,
`band[1]_threshold_db`, and `band0_threshold_db` (0-indexed). Pick one
in your `web_topology` JSON; all three resolve to band index `i ∈ [0,3]`.

## Calibration handshake

1. Build `Cal_Multipressor.logicx` (operator step) — empty audio track,
   one Multipressor instance, no other inserts.
2. Add a `PluginEntry` to `tools/calibration/auto_driver/registry.py`
   pointing at `multipressor-web.json`.
3. Run `python -m tools.calibration.auto_driver.batch --plugin Multipressor`.
4. The `param_sweeps` block will sweep crossovers + per-band threshold/
   ratio/attack/release across N steps; `curve_fit.fit_param` selects
   linear/log/exp/piecewise per param.
5. `python -m tools.calibration.auto_driver.publish` lifts the resulting
   `{logic_id}.json` into `public/plugin-mappings/` and refreshes the
   `index.json`.

The Logic-param-id hints in the config are **stubs** — confirm against
the live `kind=7` patcher dump before promoting.

## Tests

`tests/r13_multipressor.test.js` covers:

- Builder smoke: `multipressor` is reachable through `NODE_BUILDERS`,
  returns `{input, output, paramTargets}` with the expected shape and
  no missing nodes.
- Crossover frequency split: feed a sine at f → verify the band whose
  range contains f passes through with > 0.7× of input RMS, while bands
  outside that range attenuate to < 0.2× of input RMS. Three test
  frequencies (60 Hz, 500 Hz, 6 kHz) hit bands 0/1/3.
- Per-band compression: feed a loud sine at 60 Hz with band1 threshold
  set very low and ratio 20:1; verify the output RMS drops by ≥ 6 dB
  vs. the bypass-all baseline.

Run: `node src/audio/builders/r13_multipressor.test.js` (no Jest needed
— the test file ships its own `MockOfflineAudioContext`, mirroring the
PluginAdapter test pattern).

## Known gaps

- `_ensurePhase1Worklets` is not yet wired to load
  `r13-multipressor-processor.js`. Until it is, every web instance runs
  the BiquadFilter fallback. Audible difference is small (the fallback
  uses native BiquadFilter + native DynamicsCompressor, both
  battle-tested DSP), but per-sample latency differs by one render
  quantum.
- Per-band knee parameter is currently fixed to 6 dB. Add
  `band[N]_knee_db` to the worklet param descriptor + fallback wiring
  if calibration indicates Logic exposes per-band knee separately.
- Logic param ids in the config are **provisional**. Validate via
  kind=7 dump before shipping the mapping JSON.
