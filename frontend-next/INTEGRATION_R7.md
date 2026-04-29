# R7 — Modulation Matrix Integration

Agent R7 ships `src/audio/ModRouter.js` (plus tests) without modifying
`WebAudioDSPEngine.js` or R6's `VoiceManager.js`. This document is the
merge plan for those two files.

## What R7 delivers

- `src/audio/ModRouter.js` — runtime resolver for `lfo`, `macro`,
  `mod_envelope`, and `midi_cc` "ghost" nodes (io: `{in:0, out:0}`).
  Reads each node's `target` / `target_N` strings (e.g. `"filter1.cutoff"`),
  looks the target up in `builtNodes`, and wires either:
    1. **a-rate** modulation via `OscillatorNode → GainNode → AudioParam` for
       AudioParam-bearing targets (filter cutoff, gain, delay time, pan, etc.), or
    2. **k-rate** modulation via a `setInterval(updater, 16ms)` reading the
       LFO value through an `AnalyserNode` for `customSetter`-only targets
       (waveshaper drive curve, reverb decay IR regen).
- `src/audio/ModRouter.test.js` — three offline-render tests:
    1. LFO targeting filter cutoff produces ≥ 2× more windowed-RMS variation
       than a static run.
    2. Macro `setMacroValue(1.0)` propagates to two targets with different amounts.
    3. `triggerModEnvelope(id, true)` sweeps the target AudioParam.

## What R7 does NOT touch

- `WebAudioDSPEngine.js` is unchanged.
- `VoiceManager.js` (R6) is unchanged.

## Required edits in WebAudioDSPEngine

These three diffs are the entire integration surface.

### 1. Build `builtNodes` map keyed by node id (not just an array)

Inside `_buildGraphFromNodes()` we already build `const builtNodes = {}`
keyed by `nodeDef.id`. Good — that's the shape ModRouter consumes. **Save it
on `this`** so we can hand it to ModRouter:

```diff
   _buildGraphFromNodes() {
     const ctx = this.ctx;
     const graph = this.config.dspGraph;
     const builtNodes = {};
     // ...build pass...
+    this._builtNodesById = builtNodes;
   }
```

For `_buildGraph()` (the chain-mode path) and `_buildInstrumentGraph()`,
populate `this._builtNodesById` from the chain by using each `nodeDef.id`
(or auto-assigning `${type}_${index}` if missing):

```diff
   _buildGraph() {
     // ...
     const builtNodes = [];
+    const builtNodesById = {};
     for (const nodeDef of chain) {
       // ...
       builtNodes.push(built);
+      builtNodesById[nodeDef.id || `${nodeDef.type}_${builtNodes.length - 1}`] = built;
     }
+    this._builtNodesById = builtNodesById;
   }
```

### 2. Construct ModRouter after `_buildGraph()`

At the end of each of the three graph-building methods (`_buildGraph`,
`_buildGraphFromNodes`, `_buildInstrumentGraph`), after the call to
`_applyParam` for initial values:

```diff
+    // Wire up modulation routing (LFO, macro, mod_envelope, midi_cc).
+    if (this._modRouter) { this._modRouter.dispose(); }
+    if (this.config?.dspGraph?.nodes) {
+      // pull node-schema map for min/max lookups
+      const nodeSchema = this._buildNodeSchemaMap();
+      this._modRouter = new ModRouter(
+        this.ctx,
+        this.config.dspGraph,
+        this._builtNodesById,
+        this.paramDefs,
+        { nodeSchema, bpm: this.bpm || 120 },
+      );
+      this._modRouter.resolveTargets();
+    }
```

Add an import at the top:
```js
import ModRouter from './ModRouter.js';
import { NODE_CATEGORIES } from '../components/Plugins/PluginCreator/dspNodeDefinitions.js';
```

…and a helper:
```js
_buildNodeSchemaMap() {
  if (this._nodeSchemaCache) return this._nodeSchemaCache;
  const map = {};
  for (const list of Object.values(NODE_CATEGORIES)) {
    for (const def of list) map[def.type] = def;
  }
  this._nodeSchemaCache = map;
  return map;
}
```

### 3. Tear-down + lifecycle

In `_teardownGraph()`:
```diff
+    if (this._modRouter) { this._modRouter.dispose(); this._modRouter = null; }
     this.oscillators.forEach(o => { try { o.stop(); } catch (e) {} });
```

Add a public passthrough:
```js
setBPM(bpm) {
  this.bpm = bpm;
  if (this._modRouter) this._modRouter.setBPM(bpm);
}
```

## Hook into VoiceManager (R6)

R6 owns note-on / note-off lifecycle. The simplest contract:

- VoiceManager already has access to the engine instance. On note-on,
  after building the per-voice envelope, call:
  ```js
  // gateOn for every mod_envelope node in the graph
  for (const node of (this.engine.config?.dspGraph?.nodes || [])) {
    if (node.type === 'mod_envelope') {
      this.engine._modRouter?.triggerModEnvelope(node.id, true);
    }
  }
  ```
- On note-off, mirror with `false`.

If R6's voice manager is per-voice and we want **per-voice** mod envelopes,
that requires a second pass: ModRouter would have to instantiate one envelope
per voice. The current implementation is **monophonic / shared envelope** —
fine for filter sweeps, drum mods, and most monosynth scenarios. For full
polyphonic mod envelopes, R7-followup would need a `voiceId` arg on
`triggerModEnvelope` and per-voice envelope state. Doc'd as a known gap.

## Order of operations on engine.play()

```
engine.play() →
  _buildGraph() →
    builders run (filter, gain, etc.) → builtNodes populated
    _applyParam(id, normVal) for each global param        // existing
    new ModRouter(...).resolveTargets()                   // NEW (R7)
  source.connect(_chainInput); source.start();
```

## What the convention "nodeId.paramKey" maps to

ModRouter's `resolveAudioParamOnBuiltNode(builtNode, paramKey)` introspects
the AudioNode at `builtNode.input` / `builtNode.output` and matches paramKey
to a known field. Coverage:

| Built node type    | paramKeys recognized                           |
| ------------------ | ---------------------------------------------- |
| BiquadFilterNode   | `cutoff` / `frequency` / `freq` / `resonance` / `q` / `Q` / `gain` |
| GainNode           | `gain` / `level` / `volume`                    |
| DelayNode          | `time` / `delay_time` / `time_ms` / `delay_ms` |
| StereoPannerNode   | `pan` / `position`                             |
| DynamicsCompressor | `threshold` / `ratio` / `attack` / `release` / `knee` |
| OscillatorNode     | `rate` / `rate_hz` / `frequency` / `detune`    |

Compound built nodes (ladder, multitap delay) should expose
`builtNode.audioParams = { cutoff: filters[0].frequency, ... }` for direct
lookup. Today only the BiquadFilter ladder builder exists, and it uses the
multi-param `paramTargets` pattern; a small builder update could publish
`audioParams: { cutoff: filters.map(f=>f.frequency)[0] }` for ModRouter
visibility. Out-of-scope for R7 — file an issue if a ladder needs LFO mod.

## Testing in CI

`ModRouter.test.js` exports `runOfflineLFOTest(OfflineCtx)`,
`runMacroTest(OfflineCtx)`, `runModEnvelopeTest(OfflineCtx)` so it can be
invoked from any harness. The `describe`/`it` blocks at the bottom of the
file are picked up automatically by Jest or Vitest if either is added to
the project. Until then, the test functions are usable from
`PluginCreator.js` debug panel or a Cypress browser test that gets
`window.OfflineAudioContext` for free.

## Known limitations

1. **'random' LFO shape** uses a looping random buffer (32 steps per cycle),
   not a per-cycle re-randomized S&H. For true per-cycle S&H you'd promote
   to an AudioWorklet — out of R7 scope.
2. **Phase offset** is approximated by negative-time `osc.start(t, offset)`
   — works in modern browsers, no-op in older ones.
3. **K-rate updaters** use `setInterval` with `performance.now()` for
   timing. Drift under heavy main-thread load. For sample-accurate k-rate,
   migrate to AudioWorklet — out of R7 scope.
4. **Per-voice mod envelopes** — current implementation is one envelope
   instance per `mod_envelope` node, not per voice. (See VoiceManager
   section above.)
