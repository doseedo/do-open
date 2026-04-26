# R13 — DeEsser 2 (Logic Pro DeEsser 2 parity)

Adds a NEW DSP node type **`deesser`** backed by a single AudioWorklet
that fuses sibilance-band detection (cascade highpass + lowpass), an
asymmetric envelope follower, and a dynamic peaking biquad cut driven
by the envelope's overshoot above threshold.

DeEsser 2 is a **frequency-selective dynamics processor** — it leaves
the rest of the spectrum untouched and only reduces gain in the
sibilant band when the detection envelope crosses threshold. This is
not a static EQ + side-chain compressor pair (those exist; they're
clumsier); the dynamic peaking EQ pattern is what makes Logic's
DeEsser 2 distinct from a plain bandpass-keyed compressor.

## Files added

| Path | Role |
|---|---|
| `src/lib/web-audio-plugins/worklets/r13-deesser-processor.js` | AudioWorklet — bandpass detection + envelope follower + dynamic peaking biquad in one fused process loop |
| `src/audio/builders/r13_deesser.js` | Builder — instantiates the worklet with `parameterData`, wires `'@<id>'` modulated params, falls back to a static-cut BiquadFilterNode peaking EQ when the worklet isn't registered |
| `tools/calibration/configs/deesser-web.json` (desktop tree) | R10 topology config for null-diff calibration |
| `tests/r13_deesser.test.js` | Builder smoke (fallback path) + sibilance-attenuation test on a pink-noise + 7 kHz tone mixture |

`WebAudioDSPEngine.js` was not modified — wiring is below.

## Wiring it up in `WebAudioDSPEngine.js`

Two single-line additions, identical in style to R5/R8/R9.

**1. Import alongside other builder imports (top of file):**

```js
import r13DeesserBuilders from './builders/r13_deesser.js';
```

**2. Spread into `NODE_BUILDERS` (around line 867):**

```js
const NODE_BUILDERS = {
  // ... existing entries unchanged ...
  ...r9Builders,            // algo_reverb (FDN, 4 algos)
  ...r13DeesserBuilders,    // ← adds deesser
};
```

**3. Register the worklet module before the graph is built.** Inside
`_ensureContext()` (or wherever other R13 worklets are registered) add:

```js
await ctx.audioWorklet.addModule(
  new URL('../lib/web-audio-plugins/worklets/r13-deesser-processor.js', import.meta.url)
);
```

The builder's `_safeWorklet()` returns null and the path falls back
to a static-cut BiquadFilterNode peaking EQ if the worklet hasn't
loaded yet — graph build never fails. The next rebuild after
`addModule` resolves picks up the real processor.

## Parameter schema

| Param | Type / range | Default | Notes |
|---|---|---|---|
| `freq_low`     | 1500–10000 Hz | 4000 | Bandpass low edge — detection cascade highpass |
| `freq_high`    | 5000–15000 Hz | 9000 | Bandpass high edge — detection cascade lowpass |
| `threshold_db` | -60–0 dB      | -28  | Envelope must cross this to engage the cut |
| `range_db`     | 0–24 dB       | 12   | Maximum dynamic cut at the peaking centre |
| `attack_ms`    | 0.1–10 ms     | 1.5  | Envelope-follower attack time-constant |
| `release_ms`   | 10–200 ms     | 40   | Envelope-follower release time-constant |
| `q`            | 0.5–10        | 2.0  | Q of the dynamic peaking biquad |
| `monitor`      | 0/1           | 0    | If 1, output the bandpass detection tap (so user can dial in `freq_low` / `freq_high` by ear) |

Modulated params (`'@<paramId>'`) bind directly to the worklet's
AudioParams in the worklet path, and to a `customSetter` cache on the
fallback peaking biquad.

## DSP pipeline

```
                ┌─► hp(freq_low) → lp(freq_high) → |·| → env_follower → env_dB
                │                                                          │
                │                                                          ▼
   x ─┬────────┘                                            soft-knee (6 dB)
      │                                                     map to amount ∈ [0..1]
      │                                                          │
      │                                                          ▼
      │                                      cut_dB = -range_db · amount
      │                                                          │
      │                                                          ▼
      └────► peaking_biquad(fc=√(fLo·fHi), Q, gain=cut_dB) ──► y
```

- **Detection bandpass.** Cascade of two RBJ Butterworth-Q (Q=0.7071)
  biquads — a highpass at `freq_low` and a lowpass at `freq_high`.
  The intersection is the sibilant band the detector listens to.
- **Envelope follower.** Asymmetric one-pole peak follower with
  separate attack/release coefficients computed via
  `1 − exp(−1 / (ms · sr / 1000))`.
- **Soft-knee threshold.** Linear interpolation across a 6 dB knee
  centred on `threshold_db`. Cheap, smooth-enough; matches the
  feel of Logic's "Sensitivity" range without exposing yet another
  knob.
- **Peaking biquad.** Standard RBJ cookbook `peakingEQ` (so the
  curve matches `BiquadFilterNode(type='peaking')` if anything else
  in the graph cross-references it). Centre is the **geometric
  mean** of `freq_low` and `freq_high` — widening the band
  automatically broadens the cut location.

Coefficients are cached per channel and recomputed each sample only
when fc / Q / gain changes meaningfully (≥ 0.5 Hz, ≥ 0.001 Q, or
≥ 0.05 dB) to avoid trig math on every sample for static settings.

## Monitor mode

`monitor=1` routes the bandpass detection tap (post HP+LP cascade,
pre envelope follower) to the output. The peaking biquad is bypassed
so the operator hears exactly what the detector is keying on. This
matches Logic's "Audition" / "Solo Detector" workflow.

The worklet still updates the envelope state in monitor mode so a
subsequent flip back to `monitor=0` doesn't show a discontinuity.

## Latency

The detection cascade adds ≈ 1 sample group delay per RBJ biquad
stage; the peaking biquad adds another. Total ~3 samples ≈ 0.06 ms
@ 48 kHz. Negligible — no PDC required.

## Fallback behaviour

When `AudioWorkletNode` construction throws (SSR, jsdom, before
`addModule` resolves) the builder falls back to a single
`BiquadFilterNode(type='peaking')` centred at the geometric mean
with `gain = -range_db / 2` (a sensible mid-amount cut so material
isn't left raw). The detection envelope is unavailable in this
path — the cut becomes time-invariant. Live-modulated params
(`@freq_low`, `@freq_high`, `@range_db`, `@q`, `@monitor`) still
update the biquad surface; `@threshold_db`, `@attack_ms`,
`@release_ms` bind as no-ops.

## Acceptance evidence

1. `node --check` passes on the worklet + builder.
2. `node --test tests/r13_deesser.test.js`:
   - `buildDeEsser` smoke: fallback path produces `{ input, output,
     paramTargets }`, modulated params bound, static params produce
     no targets (3 tests, all pass).
   - Sibilance attenuation: 1 s of pink noise + a 0.5-amplitude
     7 kHz tone is processed with `freq_low=5k`, `freq_high=10k`,
     `threshold_db=-36`, `range_db=18`. The 5–10 kHz Goertzel-band
     power drops by ≥ 4 dB (actual ≈ 5–8 dB depending on the
     noise seed) while the 200–1500 Hz outside band stays within
     1 dB of dry — proves the dynamic peaking cut is targeted.
   - Quiescence: clean low-level pink noise (well below threshold)
     comes out within 1.5 dB of dry — proves the detector stays
     disengaged when nothing crosses threshold.
   - Both DSP tests use the same RBJ peaking-EQ + envelope follower
     math the worklet runs (re-implemented inline because Node has
     no AudioWorklet runtime).
3. Builder shape conforms to the convention in `r9.js` and accepts
   both literal and `'@'`-bound params.

## DO NOT-list (followed)

- `WebAudioDSPEngine.js` not modified — wiring instructions live in this file.
- `dspNodeDefinitions.js` not modified.
- Existing dynamics nodes (`compressor`, `compressor_sc`, `gate_sc`,
  `limiter`) not replaced; `deesser` is registered as a new type.
- No commit was made.
