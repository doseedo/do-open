# R11 Integration Plan — PluginAdapter

PluginAdapter splices Logic plugin records (from desktop sync JSON) into a
live Web Audio graph. It is **fully passive** until somebody explicitly
calls it from the existing track-playback orchestrator. As long as no one
imports it, behavior is identical to today.

---

## Files added

- `src/lib/PluginAdapter.js`                     — adapter class
- `src/lib/PluginAdapter.test.js`                — 10 unit tests, run with `node src/lib/PluginAdapter.test.js`
- `public/plugin-mappings/index.json`            — registry of available mappings (`["154"]` for now)
- `public/plugin-mappings/154.json`              — Compressor mock mapping (replaced by R10 calibration output later)

No existing files are modified.

---

## Integration point in the existing orchestrator

The track-playback orchestrator that needs to learn about PluginAdapter is:

> `src/hooks/useAudioPlayback.js`

The function is `scheduleTrack(...)` (defined ~line 1358) and the `gainNode.connect(outputNode)` call at **line 1386**. That single line is the seam:

```js
//  src/hooks/useAudioPlayback.js, line 1386 (existing)
gainNode.connect(outputNode);
//  ^ today: trackGain → busGain (per-bus output)
```

The proposed splice (do **not** apply yet — this doc is the plan):

```js
// 1) At hook init time, lazily build a singleton adapter for this AudioContext.
const pluginAdapter = useRef(null);
useEffect(() => {
  if (!audioContext) return;
  pluginAdapter.current = new PluginAdapter(audioContext);
  pluginAdapter.current.load(); // fire-and-forget index.json fetch
}, [audioContext]);

// 2) When scheduling a track that has logicPlugins, build a chain and
//    insert it between the track gain and the bus gain.
async function maybeBuildPluginChain(track) {
  if (!pluginAdapter.current) return null;
  if (!track.logicPlugins?.length) return null;
  const chain = await pluginAdapter.current.buildTrackChain(track);
  if (chain.fallback) return null;       // some plugin lacked a mapping → bounce
  return chain;
}

// 3) In scheduleTrack(), replace the single connect() with:
const chain = await maybeBuildPluginChain(track);
if (chain) {
  gainNode.connect(chain.input);
  chain.output.connect(outputNode);
  trackResources.push(chain);             // dispose on stop()
} else {
  gainNode.connect(outputNode);           // unchanged: bounce-cache path
}
```

The same seam applies to the other three `scheduleTrack`-family helpers
in the same file (search for `outputNode` parameter at `scheduleTrack`,
`scheduleTrackWithSchedule`, `scheduleTrackClips`).

A second touch-up (later, optional): when the user **edits a Logic plugin
parameter** in the studio UI, dispatch through the adapter:

```js
const chain = trackChainsRef.current.get(trackId);
const slot = chain.slots[pluginIndex];
slot.setLogicParam(logicParamId, newValue);
// engine.setParameter() schedules an audio-rate ramp internally
```

---

## Fallback path (kill switch)

PluginAdapter is designed so **the registry being empty is a no-op**:

- `public/plugin-mappings/index.json` lists which `plugin_id` values have
  a mapping. Today it lists `"154"` only.
- `adapter.instantiate(logicPlugin)` returns `null` for any plugin not in
  the registry (or whose mapping JSON 404s).
- `adapter.buildTrackChain(track)` returns `{ fallback: true, input: null,
  output: null, slots: [] }` if **any** plugin on the track lacks a
  mapping. The orchestrator MUST then take the existing bounce-cache
  path for that track.
- If `index.json` itself is missing or empty, `available` stays an empty
  Set, every `instantiate` short-circuits to `null`, and every
  `buildTrackChain` returns `fallback: true`. **Behavior is identical to
  the current state.**

To kill the new path entirely without code changes:
- delete `public/plugin-mappings/index.json` (or set `mappings: []`), **or**
- set `strictMode: false` (default) and ship a track-playback splice that
  always falls back when `chain.fallback === true`.

To force-disable the splice in the orchestrator without touching
`PluginAdapter`, gate the `maybeBuildPluginChain` call behind a feature
flag:

```js
if (!process.env.NEXT_PUBLIC_LIVE_PLUGINS) return null;
```

---

## Real-time editing flow

The killer feature: dragging a knob in the Logic-plugin UI updates the
Web Audio graph in real time, with audio-rate ramping (no zipper noise),
without re-bouncing.

```
┌────────────────────┐    knob drag       ┌──────────────────────┐
│ Plugin UI (knob)   │ ─────────────────► │ track store / hook   │
│ "Threshold = -25"  │                    │ logicPlugins[i].params│
└────────────────────┘                    └──────────┬───────────┘
                                                     │
                                                     ▼
                                        ┌────────────────────────┐
                                        │ slot.setLogicParam(    │
                                        │   logic_id, newValue)  │
                                        └──────────┬─────────────┘
                                                   │ runs curve fit
                                                   │ runs toNormalized
                                                   ▼
                                        ┌────────────────────────┐
                                        │ engine.setParameter(   │
                                        │   web_param, norm)     │
                                        └──────────┬─────────────┘
                                                   │
                                                   ▼
                                  AudioParam.setTargetAtTime(scaled, t, 0.02)
                                                   │
                                                   ▼
                                            audible change in ~20ms
```

The PluginAdapter caches per-slot fast lookups (`rowByLogicId` Map and
`rowByLogicName` Map) so the path from knob to AudioParam runs in O(1)
and is allocation-free after construction.

When a Logic id drifts between sessions (Logic occasionally renumbers
parameter IDs after plugin upgrades), the setter falls back to
`logic_name`, then to `mapping.param_map_by_name`. So as long as the
parameter's display name is stable, real-time editing keeps working.

---

## Testing

```
cd /Users/hydroadmin/Downloads/Do/doseedo-next
node src/lib/PluginAdapter.test.js
```

10 tests, expected output:

```
  ok  applyCurve linear identity
  ok  applyCurve piecewise interpolates between breakpoints
  ok  applyCurve log spans the range monotonically
  ok  toNormalized inverts engine scaleParam
  ok  PluginAdapter returns null for unknown plugin (graceful fallback)
  ok  PluginAdapter strictMode throws on missing mapping
  ok  PluginAdapter instantiates Compressor mapping with curve-fit params
  ok  Renders 1s of input + asserts dynamic-range reduction
  ok  setLogicParam updates engine in real time
  ok  buildTrackChain wires plugins in series; falls back on miss

10 passed, 0 failed
```

The render test asserts:
- output is non-zero for both loud and quiet sections (chain isn't muted)
- loud-section peak is compressed to < 0.7 (input was 0.95)
- output-side dynamic-range ratio is < 70% of input-side ratio

The mock `OfflineAudioContext` ships inline with the test file (the repo
has no jest/jsdom). Anything PluginAdapter or WebAudioDSPEngine touches
that the mock doesn't model (ConvolverNode IR generation, AnalyserNode
FFT) becomes a no-op and is not on the Compressor signal path.

---

## Known gaps for follow-up

- **Engine compressor `makeup` target**: `WebAudioDSPEngine.buildCompressor`
  binds the makeup gain target without `scale: dbToLinear`, so a Logic
  Gain of 0 dB writes 0 to `gain.value` and silences the chain. The test
  works around this by setting Logic Gain = 1. Patching the engine to
  add `scale: dbToLinear` on the makeup target (one line) fixes it
  permanently. R11 is forbidden from modifying the engine; flag it for a
  later pass.
- **Cache by (plugin_id, param_state) hash**: the brief mentions
  caching instantiated engines for fast re-instantiation. The fields
  (`_engineCache` Map, `clearCache()`) are wired but not yet populated —
  re-using engines across track rebuilds requires that the engines be
  guaranteed stateless w.r.t. the track's audio source, which is true
  today but worth a follow-up audit.
- **WebSockets / live editing transport**: real-time Logic→web parameter
  sync requires the desktop pipeline to push param changes; today the
  pipeline does a full sync JSON dump on save. R12 should land a
  delta channel.
