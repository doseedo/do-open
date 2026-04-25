# R8 — Sidechain Key-Input Routing

Owner: Agent R8
Scope: lets a Compressor or Gate have its detector driven by an external
signal (the "key input"), as in Logic Pro's Compressor / Noise Gate /
Enveloper. Required for stock plugins that expose a Side Chain selector.

This integration document describes how the new artifacts plug into the
existing `WebAudioDSPEngine`. **No engine source files were modified by R8.**

---

## New artifacts

| File | Purpose |
| --- | --- |
| `src/lib/web-audio-plugins/worklets/r8-compressor-sc-processor.js` | Sidechain-capable compressor `AudioWorkletProcessor`. Reads `inputs[1]` as the detector signal when `sidechain_active >= 0.5` and the channel is non-empty; otherwise mirrors stock `compressor-processor` behaviour. Registered as `'r8-compressor-sc-processor'`. |
| `src/lib/web-audio-plugins/worklets/r8-gate-sc-processor.js` | Sidechain-capable gate. Same protocol; registered as `'r8-gate-sc-processor'`. |
| `src/audio/builders/r8.js` | Three builder functions plus `ensureR8WorkletsLoaded(ctx)`. Exports `R8_BUILDERS = { sidechain, compressor_sc, gate_sc }`. |

The existing `compressor-processor.js` and `gate-processor.js` are
**unchanged**. Existing edges in the graph that target `compressor` or
`gate` continue to use `buildCompressor`. Sidechain support is opt-in via
the new `compressor_sc` / `gate_sc` node types.

---

## NODE_BUILDERS additions

In `src/audio/WebAudioDSPEngine.js`, near the existing `NODE_BUILDERS`
table (~line 788), import and merge:

```js
import { R8_BUILDERS, ensureR8WorkletsLoaded } from './builders/r8';

const NODE_BUILDERS = {
  // ...existing entries...
  sidechain:     R8_BUILDERS.sidechain,
  compressor_sc: R8_BUILDERS.compressor_sc,
  gate_sc:       R8_BUILDERS.gate_sc,
};
```

`ensureR8WorkletsLoaded(ctx)` MUST be `await`-ed during engine bootstrap
(somewhere inside the `init()` flow, before any node builder runs) — the
worklet processors must be registered with the AudioContext before the
`new AudioWorkletNode(...)` calls inside `compressor_sc` / `gate_sc` will
succeed.

Suggested placement: right after `this.ctx = new AudioContext(...)` and any
existing `addModule()` calls. The function is idempotent per-context.

The dspNodeDefinitions schema entries for `compressor` / `gate` already
expose `sidechain_input: { default: false }` (Dynamics category at
`src/components/Plugins/PluginCreator/dspNodeDefinitions.js`). When the
plugin author flips that toggle, the UI layer should write the node's
`type` as `compressor_sc` / `gate_sc` (instead of `compressor` / `gate`)
when serialising the dspGraph for runtime use, OR add a registry alias
that points the legacy types to the SC builders when the parameter is
truthy. The SC builders gracefully degrade to internal-detector behaviour
when `inputs[1]` is silent, so always promoting to `compressor_sc` is also
safe — this trades a tiny AudioWorkletNode allocation cost for uniformity.

---

## Edge format extension

Today, `_buildGraphFromNodes()` (~line 940 in WebAudioDSPEngine.js) walks
edges with this shape:

```js
{ source: 'kickTrack', target: 'compressor1' }
```

and connects them with the equivalent of:

```js
src.output.connect(tgt.input);
```

R8 introduces an OPTIONAL third field `input` on edges:

```js
{ source: 'kickTrack',    target: 'comp1' }                  // input slot 0 (audio)
{ source: 'sidechainTap', target: 'comp1', input: 1 }        // input slot 1 (key)
```

### Convention

- `input` is a non-negative integer naming an input slot on the target
  node. Default is `0` (audio path), preserving backwards compatibility
  with every existing edge in the codebase.
- `input: 1` wires to the **sidechain key input** of a dynamics node.
  Currently only `compressor_sc` and `gate_sc` honour it; for other node
  types `input: 1` is undefined behaviour and the engine SHOULD ignore it
  (or warn in dev) rather than throwing.
- Higher values (`input: 2+`) are reserved for future fan-in nodes
  (e.g. multi-band sidechain).

### Required engine patch (description, not code)

Update the edge-iteration block inside `_buildGraphFromNodes()` so that:

1. It reads `edge.input ?? 0` into a local `slotIndex`.
2. It picks the right target endpoint for the slot.
   - If the built node exposes `built.inputs` (an array) and
     `built.inputs[slotIndex]` exists, connect to that.
   - Else, fall back to today's behaviour: `built.input`.
3. It uses the three-argument form of `AudioNode.connect()` so the slot
   index is passed through:
   - `src.output.connect(tgtAudioNode, 0, slotIndex)`
   - This is required because `compressor_sc` and `gate_sc` use a single
     `AudioWorkletNode` instance for both slots; the slot disambiguation
     happens at the Web Audio connect() level via the `inputIndex`
     argument, not via two distinct JS objects.
4. It still wraps the call in a try/catch so a slot mismatch on a
   non-sidechain node doesn't tear down the whole graph build.

Pseudocode shape (engine maintainer to write the actual patch):

```
for (const edge of graph.edges) {
  const src = builtNodes[edge.source];
  const tgt = builtNodes[edge.target];
  if (!src || !tgt) continue;
  const slot = edge.input ?? 0;
  const tgtNode = tgt.inputs?.[slot] ?? tgt.input;
  try {
    src.output.connect(tgtNode, 0, slot);
  } catch (e) { /* incompatible slot — log and continue */ }
}
```

The two-argument call (`src.output.connect(tgt.input)`) used today is
equivalent to the three-argument form with `outputIndex=0, inputIndex=0`,
so swapping is fully backwards-compatible for every edge that omits
`input` (i.e., every existing edge).

---

## Graph-compiler convention summary

- A `sidechain` node provides the externally-tapped signal (e.g. tapped
  from a kick track). It outputs a single audio stream.
- An edge from `sidechain` to `compressor_sc` (or `gate_sc`) MUST set
  `input: 1`. Otherwise it lands on the audio path and audibly mixes with
  the signal being compressed — which is wrong.
- Edges that omit `input` continue to wire to slot 0 (audio) — preserving
  every legacy graph.
- `compressor_sc` / `gate_sc` builders set `parameterData.sidechain_active = 1`
  by default so the processor uses `inputs[1]` as detector when something
  is wired there. If nothing is wired, the worklet detects empty input and
  silently falls back to `inputs[0]` — no audible difference vs the stock
  compressor. This means a `compressor_sc` with no sidechain edge is a
  drop-in replacement for `compressor`.

---

## Migration / activation checklist

1. Bootstrap: `await ensureR8WorkletsLoaded(ctx)` in engine init.
2. Register: merge `R8_BUILDERS` into `NODE_BUILDERS`.
3. Patch: extend `_buildGraphFromNodes()` per the description above to
   honour `edge.input`.
4. UI: when the `sidechain_input` toggle on a Compressor/Gate UI is set,
   either rewrite the serialised node `type` to `compressor_sc` / `gate_sc`
   OR alias the legacy types when the toggle is truthy. The SC builders
   are safe drop-ins, so a uniform registry alias is acceptable.
5. Verify: a `sidechain` node feeding a `compressor_sc` via
   `{ input: 1 }` produces ducking driven by the key signal; removing the
   sidechain edge produces self-keyed compression identical to the stock
   compressor.
