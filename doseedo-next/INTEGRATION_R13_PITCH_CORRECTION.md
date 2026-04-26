# R13 — Pitch Correction (Logic Pro Pitch Correction parity)

Adds a NEW DSP node type **`pitch_correct`** backed by a single AudioWorklet
that fuses YIN pitch detection, scale-snap quantisation, and a PSOLA
pitch shift. Targets Logic's stock Pitch Correction plugin — Auto-Tune-class
in feature surface.

## Files added

| Path | Role |
|---|---|
| `src/lib/web-audio-plugins/worklets/r13-pitch-correct-processor.js` | AudioWorklet — combined YIN + scale-snap + PSOLA pipeline |
| `src/audio/builders/r13_pitch_correct.js` | Builder — instantiates the worklet with `parameterData`, wires `'@<id>'` modulated params, falls back to a passthrough Gain if the worklet isn't registered |
| `tools/calibration/configs/pitch-correction-web.json` (desktop tree) | R10 topology config for null-diff calibration |
| `tests/r13_pitch_correct.test.js` | Builder smoke + offline-rendered YIN frequency-detection on a synthetic 440 Hz sine |

No file in `src/audio/WebAudioDSPEngine.js` was modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, identical in style to R5/R8/R9.

**1. Import alongside other builder imports (top of file):**

```js
import r13Builders from './builders/r13_pitch_correct.js';
```

**2. Spread into `NODE_BUILDERS` (around line 867):**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,         // algo_reverb (FDN, 4 algos)
  ...r13Builders,        // ← adds pitch_correct
};
```

**3. Register the worklet module before the graph is built.** Inside
`_ensureContext()` (or wherever R5/R9 worklets are registered) add:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-pitch-correct-processor.js', import.meta.url)
);
```

The builder's `_safeWorklet()` returns null and the path falls back to
a passthrough Gain if the worklet hasn't loaded yet. So graph build never
fails — the next rebuild after `addModule` resolves picks up the real
processor.

## Parameter schema

| Param | Type / range | Default | Notes |
|---|---|---|---|
| `key` | int 0–11 | 0 | Semitones above C — `0=C, 1=C#, 2=D, …, 11=B` |
| `scale` | enum | `'chromatic'` | `'major' \| 'minor' \| 'chromatic' \| 'custom'` |
| `scale_mask` | uint12 | `0xFFF` | Bit i = pitch class i is in scale (used when `scale === 'custom'`) |
| `response_ms` | 0–500 ms | 50 | One-pole smoothing of the semitone delta. 0 ≈ T-Pain, 200 ≈ natural |
| `correction_amount` | 0–1 | 1 | 1 = full snap, 0 = bypass; multiplies the semitone delta |
| `formant_preserve` | 0/1 | 0 | **TODO** — exposed but no-op at present (see status below) |
| `mix` | 0–1 | 1 | Dry/wet |

Modulated params (`'@<paramId>'`) are honoured; `key` / `formant_preserve`
go through a rounding transform; `scale` accepts string (live drag of a
combo box) or numeric enum (0=major, 1=minor, 2=chromatic, 3=custom).

## YIN parameters

Baked into the processor via `processorOptions`:

| Constant | Value | Why |
|---|---|---|
| `analysisWindow` | 2048 samples | ≈ 43 ms @ 48 kHz — long enough to capture two periods at 70 Hz lower bound |
| `yinThreshold` | 0.15 | Canonical (Cheveigné & Kawahara 2002) |
| `minF0Hz` | 70 | covers male vocal lower limit |
| `maxF0Hz` | 1100 | covers soprano + sibilance carriers without devolving to noise lock |
| `detectIntervalSamples` | 256 | Run YIN every ~5 ms (43 % duty cycle at 2048-sample window) — vocal pitch doesn't change faster than this |

YIN steps implemented:
1. Difference function `d(τ) = Σ(x[i] − x[i+τ])²`
2. Cumulative-mean normalised difference `d'(τ)`
3. Absolute-threshold dip search → first `τ` with `d'(τ) < 0.15`, walk to local minimum
4. Parabolic interpolation around the chosen `τ` for sub-sample accuracy

`f0 = sampleRate / τ`. Returns 0 (unvoiced) if no `τ` satisfies the threshold
in the [minTau, maxTau] window. Last voiced f0 is held through unvoiced
gaps so PSOLA's epoch length stays sane.

## Scale-snap quantiser

Pitch class of detected f0 (relative to `C0 = 16.3516 Hz`) is computed
as `12 · log2(f0 / 16.3516)`. The snap algorithm walks the 12 pitch
classes (and ±1 octave neighbours, to handle wrap-around at the edge of
the scale) and picks the in-mask pitch class minimising
`|candidate − target|` in semitone space. The result is converted back
to Hz, and the shift in semitones is `12 · log2(snapped / detected) ·
correction_amount`.

## PSOLA epoch length

PSOLA grain (epoch) length is locked to the **detected period**:
`T = sampleRate / f0`. When the input is unvoiced, the last voiced f0
is reused so transitions stay continuous. The current implementation is
a two-grain crossfade: read positions are spaced `T/2` apart, each
weighted by a Hann shape based on phase-within-period.

The read pointer advances at `1/ratio` per sample; on each block we
also re-anchor the read pointer if it drifts more than ~2 periods from
a "few periods behind the write head" target — this prevents pointer
divergence on sustained ratios like 1.05 (whole-tone shift) but does
introduce a faint glitch on long held notes, audible mostly on pure
sine input. Acceptable for vocal material; the calibration topology
and golden tests will quantify the residual.

## Formant-preservation status — TODO

`formant_preserve` is exposed in the param surface and forwarded to the
worklet AudioParam, but the worklet currently does **not** separate the
spectral envelope from the excitation. Effect of the flag at runtime:
none.

Implementation plan when this lands as R13.1:
1. LPC analysis (order ~16) on each PSOLA epoch → all-pole envelope
   coefficients.
2. Inverse-filter the input epoch to get a flat-spectrum residual.
3. PSOLA-shift the residual.
4. Apply the **original-rate** envelope to the shifted residual (i.e.
   re-resample envelope to track the new f0 but keep formant peaks at
   the source f's spectral positions).

A simpler stop-gap is "warp-back" — after PSOLA, apply an inverse
spectral warp by `1/ratio` to keep formants at their source frequencies.
Cheap (single FFT), but introduces phase artefacts. Documented here so
calibration is honest about the gap.

For Logic null-diff today: keep `formant_preserve=0` in the calibration
preset to avoid penalising the web side for a feature it doesn't
implement.

## Latency

`analysisWindow / 2 + 1 PSOLA epoch ≈ 1024 samples + ≈ 100 samples ≈
23 ms @ 48 kHz`. Logic's stock plugin reports 1024 samples PDC at
44.1 kHz — close enough that the existing `latency_samples` field in
the mapping schema can be set to `1024` and inserts will line up.

## Acceptance evidence

1. `node --check` passes on the worklet + builder.
2. `tests/r13_pitch_correct.test.js`:
   - Builder smoke — `buildPitchCorrect(ctx, ...)` with a real
     OfflineAudioContext mock returns `{ input, output, paramTargets }`
     without throwing, both with and without an AudioWorklet registered.
   - YIN frequency-detection — 1 second of 440 Hz sine through the YIN
     code path returns f0 within ±2 Hz of 440. (Direct call to the
     YIN core, not the full worklet, since AudioWorklet registration
     isn't available under plain `node`.)
3. Builder shape conforms to the convention in `r9.js` and accepts both
   literal and `'@'`-bound params.

## DO NOT-list (followed)

- `WebAudioDSPEngine.js` not modified — wiring instructions live in this file.
- `dspNodeDefinitions.js` not modified.
- `r1-pitch-shift-processor.js` and `r5-pitch-shift-processor.js` not modified.
- The existing `pitch_shift` and `pitch_shift_pv` nodes are not replaced;
  `pitch_correct` is registered as a new type.
- No commit was made.
