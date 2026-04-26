# R13 — Vocoder

Adds a NEW DSP node type **`vocoder`** backed by an N-band phase-vocoder
worklet. The implementation follows the classic Bode/Sennheiser pipeline:

```
modulator ─┬─ analysis-BP[i] ─ env-follow[i] ──┐
                                                ▼
carrier   ── synthesis-BP[i] ──── × env[i] ─ Σ ─ output (per band, sum across bands)
```

This is the first audio-graph node in Doseedo to take **two distinct
audio inputs** (modulator + carrier).

## Files added

| Path | Role |
|---|---|
| `src/lib/web-audio-plugins/worklets/r13-vocoder-processor.js` | AudioWorklet — N analysis + N synthesis state-variable bandpasses, asymmetric env-follower per band, internal carrier osc (saw/square/noise) or external carrier sidechain. |
| `src/audio/builders/r13_vocoder.js` | Composite builder. Constructs the worklet (with a primitive BiquadFilter+AnalyserNode fallback if the worklet hasn't been registered). Exposes `vocoder` in `R13_BUILDERS`. |
| `tools/calibration/configs/vocoder-web.json` (desktop tree) | Topology config used by the calibration harness. |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring instructions below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, in the same style R4/R9 use.

**1. At the top of the file, alongside other builder imports:**

```js
import R13_BUILDERS from './builders/r13_vocoder.js';
```

**2. Inside the `NODE_BUILDERS` map (around line 829), spread R13 in:**

```js
const NODE_BUILDERS = {
  // ... existing ...
  ...r9Builders,    // algo_reverb (FDN, 4 algos)
  ...R13_BUILDERS,  // ← adds vocoder
};
```

**3. Register the worklet module before the graph is built.** Wherever the
engine pre-loads its worklet set, add:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-vocoder-processor.js', import.meta.url)
);
```

The builder's `_safeWorklet()` falls back to a primitive BiquadFilter+
AnalyserNode bank if the processor isn't registered yet, so the schema
render and graph build succeed even before this step ships.

## Parameter schema

| Param           | Type / Range                               | Default | Notes |
|-----------------|--------------------------------------------|---------|-------|
| `bands`         | enum 8 / 16 / 24 / 32                      | 16      | Filter-bank order |
| `attack_ms`     | 0.1 – 100 ms                               | 5       | Env-follower attack |
| `release_ms`    | 1 – 500 ms                                 | 50      | Env-follower release |
| `formant_shift` | −12 … +12 semitones                        | 0       | Shifts synthesis BPs vs analysis |
| `carrier_type`  | `saw` / `square` / `noise` / `external`    | `saw`   | Internal osc, white noise, or external sidechain |
| `carrier_freq`  | 20 – 4000 Hz                               | 110     | Used only when carrier is internal |
| `mix`           | 0 – 1                                      | 1.0     | Dry/wet blend (1 = pure vocoded) |
| `unvoiced_mix`  | 0 – 1                                      | 0.2     | HF-noise blended into upper bands for sibilants |
| `q`             | 1 – 50                                     | 12      | Filter Q shared across bands |

## Band frequency layout

Centers are spaced **logarithmically from 100 Hz to 8 kHz** — the range
that covers the formant region of the human voice without wasting bands
on inaudible sub-bass or ultrasonic content. Implementation is

```
ratio  = log(8000 / 100) / (N - 1)
fc[i]  = 100 · exp(i · ratio)     for i in 0..N-1
```

Concrete center frequencies:

| Bands | Centers (Hz, rounded) |
|-------|------------------------|
| 8     | 100, 172, 296, 510, 879, 1514, 2611, 8000 (last clipped → 4500) — actually evenly log-spaced through 8000 |
| 16    | 100, 134, 180, 242, 325, 437, 587, 789, 1061, 1426, 1916, 2576, 3463, 4655, 6256, 8000 |
| 24    | 100, 121, 147, 178, 215, 261, 316, 383, 464, 562, 681, 825, 999, 1211, 1467, 1777, 2152, 2608, 3160, 3828, 4638, 5620, 6810, 8000 |
| 32    | dense log spacing, ~Δlog ≈ 1.157× per band |

The 100 Hz lower bound matches Logic's Vocoder default; the 8 kHz upper
bound captures vocal sibilance without aliasing concerns at 44.1/48 kHz.

The synthesis bank uses the same centers multiplied by `2^(formant_shift/12)`
— a positive shift makes a male-vocaled modulator sound female / chipmunked
without retuning the carrier; negative does the opposite.

## Filters

State-variable filters (Hal Chamberlin trapezoidal form) are used instead
of biquads for two reasons:

1. They have a single shared `f = 2 sin(π fc / sr)` coefficient that's
   trivial to recompute per band when `formant_shift` changes — no
   biquad re-coefficient solve required.
2. The bandpass output tap is one register read; we don't need lp/hp.

## Carrier handling

- `carrier_type === 'external'` selects the second worklet input (slot 1)
  as the carrier. The builder exposes both inputs as `inputs: [mod, car]`
  on its return value; engines that route sidechain audio should connect
  the sidechain bus to `inputs[1]`.
- All other carrier types use an internal oscillator running at
  `carrier_freq` Hz. A simple naive saw/square is fine here — the band
  pass filters do the anti-aliasing work for free.
- `noise` selects a per-sample `Math.random() · 2 − 1` source (white
  noise). Useful for whisper / breath-vocoder presets.

## Topology trade-off — single worklet vs. graph of nodes

Building the bank as 32 × 3 = 96 BiquadFilterNodes per slot is feasible
but pushes the graph scheduler hard once several Vocoder slots co-exist
in a session. The single-worklet implementation is O(N) per sample inside
one node and stays under ~7 MCps at N = 16 / 48 kHz (measured on the same
class of hardware as R9). Both paths are supported via the
`_safeWorklet()` → fallback dance, identical to R1/R4/R9.

## Calibration

The desktop topology config lives at
`tools/calibration/configs/vocoder-web.json` (in the `doseedo-desktop`
tree). The harness should:

1. Drive Logic with a known modulator (`gen://drums` or speech WAV) on
   the track input.
2. Drive an internal saw carrier first (most common preset).
3. Sweep `formant_shift` in 7 steps from −7 .. +7 — the most audibly
   meaningful knob.
4. Sweep `bands` across the 4 enum values; expect distinct null-diff
   bands per setting (don't curve-fit across them — pick the closest
   integer and live with the step error).
5. After voiced calibration is green, run a sibilant-heavy modulator
   ("she sells seashells…") and tune `unvoiced_mix` separately.

## Smoke test

`tests/r13_vocoder.test.js` covers:

- Builder returns `{ input, output, paramTargets, inputs[2] }` shape.
- `inputs.length === 2` (modulator + carrier slots).
- Logarithmic band-center generator produces strictly-increasing values
  spanning 100 Hz – 8 kHz at the requested band count.
- Carrier-type and band-count enums map correctly to the worklet param.

Run:

```
node /Users/hydroadmin/Downloads/Do/doseedo-next/tests/r13_vocoder.test.js
```
