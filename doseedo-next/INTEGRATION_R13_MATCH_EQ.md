# R13 — Match EQ (integration notes)

This round adds the **`match_eq`** node type — Logic Pro Match EQ's
long-FFT magnitude matcher running natively in the web AudioWorklet.

## Files added

```
src/lib/web-audio-plugins/worklets/r13-match-eq-processor.js
src/audio/builders/r13_match_eq.js
tests/r13_match_eq.test.js
```

A calibration config also lives in the desktop tree at
`doseedo-desktop/tools/calibration/configs/match-eq-web.json`.

## NODE_BUILDERS additions

In `src/audio/WebAudioDSPEngine.js`, merge the builder into the existing
`NODE_BUILDERS` map. Recommended insertion (right after the R9 import):

```js
import r13MatchEQBuilders from './builders/r13_match_eq.js';
// …
const NODE_BUILDERS = {
  // …existing…
  ...r9Builders,
  ...r13MatchEQBuilders,   // match_eq
};
```

The builder also exports a `ensureR13MatchEQWorklet(ctx)` helper that the
engine's `_ensurePhase1Worklets` (or equivalent eager-warm path) can
`await` to avoid the first-build passthrough.

## Pipeline

1. **Analyze.** Host sets `mode = 'analyze_target'`, plays the reference
   audio through. The worklet windows + FFTs each frame and accumulates
   magnitude sums in `targetMag[k]` (k = 0…halfSize). It also passes
   audio through unchanged so the user can monitor.
2. **Repeat for source.** Host sets `mode = 'analyze_source'`, plays the
   current source. Worklet accumulates `sourceMag[k]`.
3. **Read out.** Host posts `{ type: 'request_spectrum', which: 'target' }`
   then `{ type: 'request_spectrum', which: 'source' }`. Worklet replies
   with `{ type: 'spectrum_average', mode, framesAccumulated, magnitudes }`.
4. **Smooth + divide host-side.** Use the `smoothCurveOctave(mags, sr,
   octaveFraction)` and `computeMatchCurve(target, source)` helpers
   exported by `builders/r13_match_eq.js`. The 1/N-octave smoother runs
   in dB space so a +6 dB peak averages to +3 dB in a centered 1/N-octave
   window.
5. **Post curve.** `node.port.postMessage({ type: 'set_curve', curve })`.
   Worklet stores it in `matchCurve`.
6. **Apply.** Host sets `mode = 'apply'`. The worklet now performs
   per-bin magnitude scaling: window → FFT → multiply each complex bin
   by `(1 + amount * (clamped - 1)) * gainMakeup`, where `clamped` is
   the curve gain hard-limited to ±36 dB → IFFT → OLA.

The curve is also re-derivable inside the worklet via
`port.postMessage({ type: 'recompute' })`, useful if the host wants to
hand off captured analysis state at instantiation via `processorOptions`
(both `targetCurve` and `sourceCurve` are accepted) and never run live
analysis.

## FFT size + smoothing parameters

| Parameter | Default | Notes |
|---|---|---|
| `fftSize` | 4096 | Configurable to 1024 / 2048 / 4096 / 8192. 4096 @ 44.1 kHz = 10.77 Hz/bin, 92.9 ms latency in apply mode. |
| `hopSize` | `fftSize >> 2` | 75 % overlap. New frame every 23 ms. |
| `windowNorm` | 2/3 | Sum-of-squared-Hann normaliser for 75 % overlap. |
| `curve_smoothing_octave` | 1/3 | Geometric-mean window in dB space. |
| `low_cut` / `high_cut` | 20 Hz / 20 kHz | Bins outside the band pass through unchanged in apply mode. |
| `amount` | 1.0 | Linear-domain blend between identity (0) and full curve (1). |
| `gainMakeup` | 1.0 (linear) | Builder receives dB and converts. |

The clamp to ±36 dB (`Math.min(64, Math.max(1/64, curve))`) inside
`_processApplyFrame` is the only nonlinearity in the apply path. It's
there because raw `target/source` divisions blow up when a bin in the
source spectrum is near silent.

## Analysis-frame accumulation

A single AudioWorklet `process()` call delivers 128 frames. The worklet
maintains a length-`fftSize` ring buffer (`inputRing`) and a countdown
`samplesUntilFFT` initialised to `fftSize` (so the first FFT runs only
once the buffer is full) and reset to `hopSize` after every FFT (75 %
overlap → 25 % new samples per frame).

When in an analyze mode the worklet runs the FFT but skips the IFFT:

```js
this.fft.forward(this.re, this.im);
const mag = (mode === MODE_ANALYZE_SOURCE) ? this.sourceMag : this.targetMag;
for (let k = 0; k <= halfN; k++) {
  mag[k] += Math.sqrt(this.re[k]² + this.im[k]²);
}
this.targetFrames++ (or sourceFrames++);
```

`request_spectrum` divides `targetMag[k] / targetFrames` to recover the
average, then ships the result over the port. Float32Array messages are
copied (not transferred) so the worklet's accumulator survives the read.

## Param wiring summary

Builder param keys → worklet AudioParam (or message):

| Builder key | Worklet param | Conversion |
|---|---|---|
| `mode` | `mode` (k-rate, 0/1/2) | string ↔ index, also accepts `0..1` normalised |
| `curve_amount` | `amount` | identity |
| `low_cut` | `lowBin` | `hz / (sampleRate/2)` |
| `high_cut` | `highBin` | `hz / (sampleRate/2)` |
| `gain_makeup` | `gainMakeup` | `10^(dB/20)` |
| `target_curve` | message `set_target` + `recompute` | array → Float32Array |
| `source_curve` | message `set_source` + `recompute` | array → Float32Array |
| `curve_smoothing_octave` | (host-side, no AudioParam) | UI hint |
| `fft_size` | constructor `processorOptions.fftSize` | one of 1024/2048/4096/8192 |

## Fallback behaviour

If `audioWorklet.addModule(...)` hasn't yet resolved when the builder
runs (the very first graph build of a session), the builder returns a
straight passthrough `GainNode` with all `@<id>` param bindings stubbed
as no-op `customSetter`s. The next graph rebuild after the worklet
loads picks up the real `match_eq`. This is the same pattern as R5 and
R9 — see `r5.js` for the canonical shape.

## Calibration handshake

The desktop calibration harness drives Match EQ through three Logic
bounces (target reference, dry source, applied output). The web side
stores both spectra and the resulting curve in the mapping JSON's
`web_topology.params` block. R12 null-diff compares
`web_apply(target_curve, source_curve, dry)` against the Logic-bounced
output. With 1/3-octave smoothing and `amount=1`, on pink noise +
known biquad-filtered target the test in `tests/r13_match_eq.test.js`
reaches ≤ −12 dB RMS (the FFT-pure approach is already well within the
audible-match threshold for the kind of broadband matching Match EQ is
typically used for).

## Testing

Smoke + spectrum-match test:

```
node tests/r13_match_eq.test.js
```

The test uses `jsdom`-free `OfflineAudioContext` if available, otherwise
falls back to the pure-JS smoothing/computation helpers — so it's
meaningful both in browser and node-run CI.
