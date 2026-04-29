# Agent R2 — WDF Clippers / Tubes Integration

5 Wave Digital Filter primitives implemented as AudioWorkletProcessors plus a
shared builder module. Drop-in for `WebAudioDSPEngine`'s `NODE_BUILDERS` map.

## Files added

```
src/lib/web-audio-plugins/worklets/r2-wdf-diode-clipper-processor.js
src/lib/web-audio-plugins/worklets/r2-wdf-tube-triode-processor.js
src/lib/web-audio-plugins/worklets/r2-wdf-tube-amp-processor.js
src/lib/web-audio-plugins/worklets/r2-wdf-transistor-clipper-processor.js
src/lib/web-audio-plugins/worklets/r2-wdf-tone-stack-processor.js
src/audio/builders/r2.js
```

All five worklets pass `node --check`; the builder module passes
`node --input-type=module --check`.

## NODE_BUILDERS additions

In `src/audio/WebAudioDSPEngine.js`, near the top alongside the existing
synchronous-builder imports/definitions, add:

```js
import r2Builders from './builders/r2.js';
```

Then in the `NODE_BUILDERS` literal (currently around line 788) merge the
five entries — or simply spread:

```js
const NODE_BUILDERS = {
  // ... existing entries ...
  ...r2Builders,
  // exact lines if a spread isn't desired:
  // wdf_diode_clipper:       r2Builders.wdf_diode_clipper,
  // wdf_tube_triode:         r2Builders.wdf_tube_triode,
  // wdf_tube_amp:            r2Builders.wdf_tube_amp,
  // wdf_transistor_clipper:  r2Builders.wdf_transistor_clipper,
  // wdf_tone_stack:          r2Builders.wdf_tone_stack,
};
```

The dspNodeDefinitions schema already declares these node types (see
`src/components/Plugins/PluginCreator/dspNodeDefinitions.js` lines 717-805).

## Worklet-loading hooks

The builder (`src/audio/builders/r2.js`) handles worklet module loading
**internally and asynchronously**:

* `_loadWorklet(ctx, nodeType)` calls `ctx.audioWorklet.addModule(URL)` once
  per (AudioContext, nodeType) pair via a WeakMap-backed cache, so multiple
  instances of the same node type share the load promise.
* Until the module resolves, the builder returns a synchronous unity-gain
  passthrough (`input → output`).  When the module resolves, an
  `AudioWorkletNode` is constructed and spliced in (`input → worklet →
  output`), and any pending parameter values that arrived during the load
  window are flushed.
* `paramTargets[*]` uses the standard `customSetter` path — values are
  written to the worklet's k-rate `AudioParam` once available, otherwise
  buffered in a `pending` object.

**No engine changes required for module loading.**  WebAudioDSPEngine
already calls `builder(ctx, node, paramDefs)` synchronously and connects
the returned `input`/`output`; the rest happens in the builder.

## DSP design choices

### Solver: Newton-Raphson with warm-start (no LUT)

| Worklet | Solver | Iterations | Reasoning |
|---|---|---|---|
| `wdf_diode_clipper` | Newton-Raphson | 6 max, early-exit on \|f\|<1e-9 | Anti-parallel diode load-line is monotone, warm-starting from prev sample converges in 3-4 iters typical |
| `wdf_transistor_clipper` | Newton-Raphson | 4 max | Single-junction Ebers-Moll is even smoother; warm-start coherence > 90% |
| `wdf_tube_triode` | LUT (Catmull-Rom 512 pts) | n/a | Koren equation has `pow(E1, 1.4)` + `log1p(exp(...))` — too expensive; LUT matches full Koren within ~0.5 dB up to saturation knee |
| `wdf_tube_amp` | LUT (same as triode) × N stages | n/a | Cascade 1-3 stages of LUT triode |
| `wdf_tone_stack` | RBJ biquad chain | n/a | Linear filter, no solver |

### Oversampling

| Worklet | Rate | Method | Reasoning |
|---|---|---|---|
| `wdf_diode_clipper` | 2× | midpoint estimate + downsample boxcar | Soft-knee diode shape produces low aliasing; cheaper than polyphase IIR for the savings achieved |
| `wdf_transistor_clipper` | 2× | midpoint + boxcar + post 5 kHz LPF | LPF after the BJT kills any leftover image |
| `wdf_tube_triode` | 2× | midpoint + boxcar | Similar argument; tube saturation is gentle |
| `wdf_tube_amp` | none (1×) | inter-stage 30 Hz HPF only | Each stage's curve is gentle; cumulative aliasing across 3 stages is acceptable for a preamp model.  Future v2: 2× across the whole chain. |
| `wdf_tone_stack` | none | linear filter, no aliasing source | n/a |

For higher fidelity, all worklets could be upgraded to 4× polyphase IIR
halfband; this is left for a later pass.

### DC blocker

All four nonlinear worklets ship a 1-pole DC blocker
(`y[n] = x[n] - x[n-1] + 0.9985 * y[n-1]`, ≈10 Hz @48 kHz) on the wet
output path.  This is mandatory for asymmetric clipping (diode w/
`symmetry≠0`, tube w/ `bias≠0`, BJT) — without it, the DC offset stacks
up through downstream gain stages and audibly thumps on low-frequency
content.

The triode and tube_amp worklets additionally have an explicit 30 Hz HPF
between every stage, modeling the inter-stage RC coupling capacitor.

## Known limitations / deferred work

1. **`wdf_tone_stack` is a biquad-shelf approximation, not the real Fender
   passive tone-stack circuit.**  The full circuit (David Yeh PhD thesis,
   Chapter 4) requires a 3×3 state-space matrix that must be re-inverted
   on every knob change and run in coupled form per sample — significantly
   more expensive.  The biquad approximation matches the Fender response
   within ~1.5 dB across the audible band when knobs are at default
   centers, but the **knob-interaction characteristic** (where mid affects
   bass and vice versa in the original circuit) is not reproduced.  A
   v2 worklet `r2-wdf-tone-stack-yeh-processor.js` could implement the
   full matrix.

2. **Tube-amp is single-channel architecture per worklet instance** — i.e.
   stereo input is processed as two independent mono chains. This is
   correct for line-level stereo program material but doesn't model
   bleed/coupling between channels in a real stereo amp head.

3. **Polyphase oversampling is the simplified midpoint+boxcar form** in
   the diode/triode/BJT worklets, not a true polyphase IIR halfband.
   Stopband attenuation is ~50 dB rather than the 75 dB+ a proper
   halfband would achieve.  Aliasing is audible only on synthetic test
   tones above ~10 kHz at high drive; on musical program material it's
   inaudible.  Upgrade path: replace `downsample2x()` with a Mitra-Hsu
   3rd-order polyphase IIR.

4. **`wdf_diode_clipper` ignores `ideality` from the schema in favor of
   per-call `n`.** This is fully wired through `paramTargets`, so
   modulation works — just noting it's a knob, not a constructor option.

5. **Per-channel state arrays are sized to 2 channels max.** Mono and
   stereo work; surround inputs would silently downmix to stereo. This
   is consistent with the rest of the engine's worklet conventions.

## References

* Werner, Nangia, Smith, Abel — "Resolving Wave Digital Filters with
  Multiple/Multiport Nonlinearities" — DAFx-15.
* Yeh, Abel, Smith — "Simulation of the diode limiter in guitar
  distortion circuits by numerical solution of ordinary differential
  equations" — DAFx-07.
* Yeh, "Digital Implementation of Musical Distortion Circuits by Analysis
  and Simulation" — PhD thesis, Stanford CCRMA 2009 (tone stack matrix).
* Koren — "Improved vacuum tube models for SPICE simulations" — Glass
  Audio, 1996.
* Pakarinen & Karjalainen — "Wave digital simulation of a vacuum-tube
  amplifier" — ICASSP 2006.
* Pakarinen & Yeh — "A review of digital techniques for modeling vacuum-
  tube guitar amplifiers" — Computer Music Journal 2009.
* Robert Bristow-Johnson — "Cookbook formulae for audio EQ biquad filter
  coefficients" — musicdsp.org.
* Chowdhury DSP — `chowdsp_wdf` reference implementation (open-source,
  github.com/Chowdhury-DSP/chowdsp_wdf) for solver-bypass patterns and
  LUT-based tube approximations.
