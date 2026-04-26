# R13 — Spectral Gate (integration notes)

Adds `spectral_gate` to the WebAudioDSPEngine builder registry. A
frequency-domain noise gate built on the same R5 spectral foundation
(STFT, Hann window, 75% overlap, inlined Cooley–Tukey FFT) used by
`spectral_filter`, `spectral_freeze`, and the phase-vocoder pitch shift.

## Files added

```
src/lib/web-audio-plugins/worklets/r13-spectral-gate-processor.js
src/audio/builders/r13_spectral_gate.js
tests/r13_spectral_gate.test.js
(desktop) tools/calibration/configs/spectral-gate-web.json
```

## NODE_BUILDERS wiring

```js
import r13SpectralGateBuilders from './builders/r13_spectral_gate.js';

const NODE_BUILDERS = {
  ...
  ...r13SpectralGateBuilders, // spectral_gate
};
```

R13's spectral-gate module exports a single key, `spectral_gate`. Adding
it after the other R13 builders preserves the conflict-resolution rules
already documented in `INTEGRATION_R13_PEDALBOARD.md` etc.

## DSP design

| Stage              | Setting                                                          |
|--------------------|------------------------------------------------------------------|
| FFT size           | **2048** samples (configurable via `processorOptions.fftSize`)   |
| Hop size           | **512** samples (`fftSize >> 2`, 75% overlap)                    |
| Analysis window    | **Hann** (`r13HannWindow`, COLA-compliant with 75% overlap)      |
| Synthesis window   | Hann (same array) — symmetric COLA factor 1.5 → norm `2/3`       |
| Frame latency      | `fftSize` samples ≈ **46 ms** at 44.1 kHz                        |
| Per-bin envelope   | One-pole low-pass updated **once per analysis hop**              |
| Bins gated         | `[low_cut_bin, high_cut_bin)` only; out-of-band bins exempt      |

### Per-bin gate decision

For every bin `k` in `[0..N/2]` after the forward FFT:

```
mag_db(k)  = 20·log10(max(|X[k]|, 1e-12))
tilt(k)    = tilt_db · (k / halfN)            // linear-in-bin slope
thresh(k)  = threshold_db + tilt(k)
below      = mag_db(k) < thresh(k)  AND  k ∈ [low_cut_bin, high_cut_bin)
target(k)  = below ? 10^(reduction_db/20) : 1.0
```

Negative `tilt_db` makes the gate stricter on highs (highs need to be
louder to clear the threshold); positive tilt is stricter on lows. Out-of-
band bins (`k < low_cut_bin` or `k >= high_cut_bin`) get `target=1` and
release back to unity — bass and air bands aren't gated by a global
threshold.

### Per-bin gate-envelope smoothing

Each bin owns one float of state (`binEnv[k]`), initialized to 1.0 so a
fresh node passes audio cleanly until the first below-threshold frame.
Each analysis hop updates the envelope:

```
hopT     = hopSize / sampleRate                  // ≈ 11.6 ms at 44.1 kHz
attCoef  = 1 - exp(-hopT / (attack_ms  / 1000))
relCoef  = 1 - exp(-hopT / (release_ms / 1000))
coef     = below ? attCoef : relCoef
binEnv[k] += coef · (target(k) - binEnv[k])
X'[k]     = X[k] · binEnv[k]
```

This is a per-bin one-pole low-pass on the gating *decision*, not on the
audio samples. It is what stops the gate from chattering on a marginal
signal — without it, a bin that hovers around threshold would flip
between attenuated and pass-through every hop, audible as a buzzy
amplitude modulation. With one-pole smoothing, the envelope blends the
two states gracefully across `attack_ms` / `release_ms`.

After gating, Hermitian symmetry is restored on bins `[N-k]` so the
inverse FFT yields a real signal.

## Param surface

| schema key     | AudioParam   | range          | default | unit |
|----------------|--------------|----------------|---------|------|
| `threshold_db` | `thresholdDb`| -60 .. 0       | -40     | dB   |
| `reduction_db` | `reductionDb`| -60 .. 0       | -40     | dB   |
| `attack_ms`    | `attackMs`   | 1 .. 100       | 10      | ms   |
| `release_ms`   | `releaseMs`  | 10 .. 1000     | 100     | ms   |
| `low_cut`      | `lowCut`     | 0 .. 1         | 0       | norm of N/2 |
| `high_cut`     | `highCut`    | 0 .. 1         | 1       | norm of N/2 |
| `tilt_db`      | `tiltDb`     | -12 .. +12     | 0       | dB   |
| `mix`          | `mix`        | 0 .. 1         | 1       | wet/dry |

`@param_id` bindings install entries in `paramTargets` for live updates
through `engine.setParameter(...)`. The worklet also accepts
`port.postMessage({type, value})` overrides on every key (snake_case),
plus `{type: 'reset'}` to clear OLA + per-bin envelopes back to 1.0.

## Builder pattern + fallback

`audioWorklet.addModule()` is async, but the engine's `NODE_BUILDERS`
calls builders synchronously. Following the R5 convention:

1. `ensureR13SpectralGateWorklet(ctx)` — idempotent; kicks off
   `addModule` for `r13-spectral-gate-processor`.
2. The builder eagerly calls `_safeWorklet`. If the processor isn't
   registered yet, it returns `null`.
3. **Fallback:** a serial `DynamicsCompressorNode` wired with a
   high ratio (gate-ish) + dry/wet split. Spectrum-aware behavior
   isn't possible without the worklet (no FFT in stock Web Audio), but
   audio still flows and every param surface still binds, so the engine
   doesn't error. The next graph rebuild picks up the real worklet.

This matches R5/R9/R13_chromaverb conventions exactly. Pre-warming
(`await ensureR13SpectralGateWorklet(ctx)` after AudioContext creation)
skips the first-build fallback if the host wants the real spectral gate
on the very first build.

## Latency

| node          | fftSize | hop | latency (samples) | latency (ms @ 44.1 kHz) |
|---------------|---------|-----|-------------------|--------------------------|
| spectral_gate | 2048    | 512 | 2048              | ≈ 46 ms                  |

Set `latency_samples: 2048` in the mapping JSON once Logic-side
calibration runs.

## Calibration mapping

`tools/calibration/configs/spectral-gate-web.json` is the calibration
seed. `low_cut` / `high_cut` are normalized 0..1 of Nyquist; the harness
will need to convert Logic's `low_cut_hz` via `lowCut = hz / (sr/2)`.
`tilt_db` is a single linear-in-bin slope — if Logic exposes a fuller
per-band threshold curve, the calibration may need a piecewise mapping
on `tilt_db` (or expand the schema to a 3-point breakpoint curve).

## Tests

`tests/r13_spectral_gate.test.js` covers:

- **Builder smoke**: returns `{ input, output, paramTargets }` with the
  correct shape; `@param_id` bindings populate `paramTargets` for every
  modulated param; literal numeric params don't end up in `paramTargets`.
- **Gate attenuation**: a pure 1 kHz tone at -50 dBFS through a gate
  with `threshold_db=-30, reduction_db=-40, mix=1` is attenuated
  ≥20 dB after the FFT settles. (Runs only when `OfflineAudioContext`
  is available — degrades to skip in pure-Node CI.)
