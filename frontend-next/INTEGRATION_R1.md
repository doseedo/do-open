# R1 — Integration into WebAudioDSPEngine

This document lists the exact edits required to wire the R1 batch of node-type
runtime builders into `src/audio/WebAudioDSPEngine.js`.

R1 adds runtime support for these 7 schema-defined node *categories* (14 total
node types):

- `bitcrusher`
- `multitap_delay`
- `convolution`
- `envelope_follower`
- `pitch_shift`
- `comb`
- `math_add` / `math_multiply` / `math_abs` / `math_rectifier` / `math_slew` / `math_scale` / `math_crossfade` / `math_constant`

All implementations live outside `WebAudioDSPEngine.js`; the engine only needs
to import them and register them in the `NODE_BUILDERS` table.

---

## 1. Add the import

Add this near the top of `src/audio/WebAudioDSPEngine.js`, after the existing
helper functions and before `const NODE_BUILDERS = …`:

```js
import r1Builders, { ensureR1Worklets } from './builders/r1.js';
```

(Or if you prefer named imports for clarity:)

```js
import {
  buildBitcrusher, buildMultitapDelay, buildConvolution,
  buildEnvelopeFollower, buildPitchShift, buildComb,
  buildMathAdd, buildMathMultiply, buildMathConstant,
  buildMathScale, buildMathCrossfade, buildMathAbs,
  buildMathRectifier, buildMathSlew, ensureR1Worklets,
} from './builders/r1.js';
```

---

## 2. Replace the wrong / stubbed entries in `NODE_BUILDERS`

Around **line 788** of `WebAudioDSPEngine.js`, the current registry has these
incorrect mappings:

```js
delay: buildDelay, multitap_delay: buildDelay, ping_pong_delay: buildDelay,
                  // ^^^ wrong — single-tap stub
reverb: buildReverb, convolution: buildReverb,
                     // ^^^ wrong — uses algorithmic reverb
overdrive: buildWaveshaper, waveshaper: buildWaveshaper,
saturation: buildWaveshaper, foldback: buildWaveshaper,
// `bitcrusher` is currently absent from the table → falls through
// to the no-op passthrough branch in `_buildGraph`.
```

Remove `multitap_delay` and `convolution` from those existing lines, then add
the following block to the registry **using the spread of `r1Builders`**:

```js
const NODE_BUILDERS = {
  // … existing entries …

  // R1 — corrected / new node-type runtimes
  ...r1Builders,
};
```

Or if you'd rather list them explicitly (recommended for grep-ability), append
the following lines at the end of `NODE_BUILDERS`:

```js
bitcrusher:        r1Builders.bitcrusher,
multitap_delay:    r1Builders.multitap_delay,
convolution:       r1Builders.convolution,
envelope_follower: r1Builders.envelope_follower,
pitch_shift:       r1Builders.pitch_shift,
comb:              r1Builders.comb,
math_add:          r1Builders.math_add,
math_multiply:     r1Builders.math_multiply,
math_constant:     r1Builders.math_constant,
math_scale:        r1Builders.math_scale,
math_crossfade:    r1Builders.math_crossfade,
math_abs:          r1Builders.math_abs,
math_rectifier:    r1Builders.math_rectifier,
math_slew:         r1Builders.math_slew,
```

These spread/override entries **must come after** the existing
`delay/multitap_delay/ping_pong_delay` and `reverb/convolution` lines so they
take precedence — the spread above does this naturally.

---

## 3. Hook worklet pre-loading into context creation

The R1 builders for `bitcrusher`, `multitap_delay`, `envelope_follower`,
`pitch_shift`, and `comb` use AudioWorklets. They each *kick off* worklet
loading on first call, but because `addModule` is async and the builder API
is sync, the **first** build of a graph using those node types will fall back
to a non-worklet implementation.

To make the worklets available on the first build, call `ensureR1Worklets()`
in `_ensureContext()` — patch the existing method around **line 842**:

```js
_ensureContext() {
  if (!this.ctx || this.ctx.state === 'closed') {
    try {
      this.ctx = new (window.AudioContext || window.webkitAudioContext)();
      this.contextError = null;
    } catch (err) {
      this.contextError = err.message || 'Failed to create AudioContext';
      return;
    }
  }
  if (this.ctx.state === 'suspended') {
    this.ctx.resume().catch(() => {});
  }
  // R1: kick off worklet registration (idempotent, returns a promise).
  // Most engines call play() after a user gesture, so by the time the graph
  // is built the worklets are usually ready.
  ensureR1Worklets(this.ctx);
}
```

If you need *guaranteed* worklet availability before `_buildGraph()` runs,
make `play()` async and `await ensureR1Worklets(this.ctx)` before building.
For now the fallback paths are safe — they just won't sound as good.

---

## 4. Verify

After the patch, every node type listed above should pick up its R1 builder
when the engine is constructed against a `dspChain` or `dspGraph` containing
those types. The fallback non-worklet implementations also satisfy the builder
interface, so removing the worklet files (or running in an environment without
AudioWorklet support) won't break the engine — it'll just sound less faithful.

No other files in the engine need to change. `_applyParam`, `setParameter`,
`_teardownGraph`, instrument mode, etc. all already understand the
`paramTargets` shape (audioParam / audioParams / customSetter / scale).

---

## 5. Optional follow-ups (not required for R1 acceptance)

- The math_slew fallback is currently a passthrough; to get true per-sample
  slew rates on full-band audio, we'd add an `r1-slew-processor.js` worklet.
- The pitch_shift fallback is silent (dry-only) when the worklet hasn't loaded;
  acceptable for first-paint but a `setTimeout`-based lazy rebuild after the
  worklets register would smooth the UX.
- math_crossfade exposes `inputs: [a, b]` on its return; the current
  `_buildGraphFromNodes()` only consumes `input`. If the graph schema starts
  emitting two-input nodes, the engine's edge resolver should learn to use
  `inputs[port]` when the target node exposes one.
