# R13 — Enveloper (Transient Shaper)

Adds a single new DSP node type — **`enveloper`** — that mirrors Logic Pro's
stock *Enveloper* plugin: a transient designer in the SPL Transient
Designer / Waves TransX family. Two parallel envelope followers (fast +
slow) decompose the input into an attack envelope and a sustain envelope;
two independent gain knobs let you boost or cut each.

## Files

- `src/audio/builders/r13_enveloper.js` — builder + default `{ enveloper: buildEnveloper }`.
- `src/lib/web-audio-plugins/worklets/r13-enveloper-processor.js` — the real DSP path.
- `tests/r13_enveloper.test.js` — builder smoke + transient-detection unit tests.
- `tools/calibration/configs/enveloper-web.json` (desktop tree) — calibration config.

## Algorithm

```
            ┌─ fast follower (1 ms / 10 ms) ──┐
input ─┬────┤                                  ├──► attack_env  = max(0, fast − slow) / peak
       │    └─ slow follower (30 ms / 300 ms)─┘    sustain_env = max(0, slow − 0.6·fast) / peak
       │
       └────► slow peak follower (5 ms / 1 s) ──── peak (auto-normaliser)

attack_gain  = 1 + (attack_pct  / 100) · attack_env
sustain_gain = 1 + (sustain_pct / 100) · sustain_env
y = mix · (x · attack_gain · sustain_gain · 10^(output_gain/20))
  + (1 − mix) · x
```

Auto-normalisation by a slow-tracking peak keeps the shape factors in
[0, 1] regardless of input level, so the same `attack_pct` produces the
same perceptual emphasis at -20 dBFS as at 0 dBFS.

Total wet gain is clamped at 16× (~24 dB) — matches Logic's internal
soft-knee ceiling under extreme attack+sustain settings.

## Param surface

| Param | Range | Unit | Notes |
|---|---|---|---|
| `attack`             | −100…+100 | %   | Boost (+) / soften (−) the transient peak |
| `sustain`            | −100…+100 | %   | Boost (+) / cut (−) the body decay |
| `attack_time_ms`     | 0.1…10    | ms  | Fast follower attack |
| `attack_release_ms`  | 1…50      | ms  | Fast follower release |
| `sustain_time_ms`    | 10…200    | ms  | Slow follower attack |
| `sustain_release_ms` | 100…2000  | ms  | Slow follower release |
| `output_gain`        | −12…+12   | dB  | Static makeup, post transient gain |
| `mix`                | 0…1       |     | Dry/wet crossfade |

Logic's stock UI only exposes Attack + Sustain (+ output gain). Treat the
follower timings as "advanced" params: probe them during calibration; if
Logic's response is invariant under timing tweaks, lock them to their
defaults in the mapping JSON.

## Engine wiring

The builder is **not** auto-wired into `WebAudioDSPEngine.NODE_BUILDERS`
yet — that's a shared file (don't touch). When wiring lands, add:

```js
// src/audio/WebAudioDSPEngine.js
import r13EnveloperBuilders from './builders/r13_enveloper.js';

const NODE_BUILDERS = {
  // ...existing builders
  ...r13EnveloperBuilders,   // enveloper
};
```

And add the worklet to the pre-warm list (whichever helper loads R13's
sister processors):

```js
await ctx.audioWorklet.addModule(
  '/lib/web-audio-plugins/worklets/r13-enveloper-processor.js'
);
```

Until that one-line wiring lands, the `enveloper` node type is reachable
by importing the builder directly in mapping code or test harnesses.

## Mapping JSON shape

```json
{
  "plugin_id": "<logic id, e.g. 1118>",
  "schema_version": "1.0",
  "web_topology": {
    "nodes": [
      { "id": "env",
        "type": "enveloper",
        "params": {
          "attack":      "@p_attack",
          "sustain":     "@p_sustain",
          "output_gain": "@p_makeup",
          "mix":         1.0
        }
      },
      { "id": "out", "type": "output" }
    ],
    "edges": [{ "source": "env", "target": "out" }]
  },
  "param_map": [
    { "logic_id": 0, "web_param": "p_attack",  "curve": "linear" },
    { "logic_id": 1, "web_param": "p_sustain", "curve": "linear" },
    { "logic_id": 2, "web_param": "p_makeup",  "curve": "linear" }
  ]
}
```

Set `param_map[i].curve` from R10 calibration. The defaults assume Logic
ships a linear knob → percent mapping; verify with a sweep before
locking.

## Fallback path

When the worklet processor isn't yet registered on the
`AudioContext` (SSR, first-frame, or a test harness without
`audioWorklet.addModule`), the builder substitutes a
`DynamicsCompressor` + makeup gain configured as a transient-emphasis
stage. The fallback honours the same param surface — the engine sees
the same shape regardless of which path is alive. Once the worklet
loads, the next graph build picks up the real DSP with no code change.

Expect ≥ −25 dB null-diff against Logic in fallback mode (the
DynamicsCompressor only approximates positive-attack behaviour); the
worklet path is the calibration target.

## Test plan

- `tests/r13_enveloper.test.js` covers two cases:
  1. **Builder smoke**: `buildEnveloper(ctx, node, paramDefs)` returns
     `{input, output, paramTargets}` with matching keys and the graph
     binds without throwing.
  2. **Transient detection**: a synthetic drum-hit-then-silence input
     fed through the worklet path should show the attack envelope
     spiking at the hit (i.e. RMS in the first 5 ms is significantly
     higher than RMS during the silent tail). The test uses a small
     reference implementation of the follower math in plain JS so it
     runs without a real `AudioWorkletNode`.

## Calibration backlog

1. Bounce 5 presets from Logic Enveloper at fixed input (drum loop):
   `+50% attack`, `−50% attack`, `+50% sustain`, `−50% sustain`, neutral.
2. Drive each Logic param across 11 steps; capture the resulting
   bounces.
3. Fit `param_map` curves with `tools/calibration/curve_fit.py`.
4. Author `public/plugin-mappings/<logic_id>.json` with the topology
   above + fitted curves.
5. R12 null-diff goal: ≤ −40 dB RMS across all five presets.
