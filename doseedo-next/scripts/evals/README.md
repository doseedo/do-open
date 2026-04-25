# Ref-renderer evaluation harness

Measures how well the vision + DSL pipeline reproduces real reference plugin
UIs. Run before and after any prompt or vocabulary change to see what
improved and what regressed.

## What's here

```
references/       source plugin screenshots (PNG)
truth.js          per-image expectations (archetype, chassis, canvas, row kinds,
                  knob/slider counts, accent hue) + known vocabulary gaps
outputs/<name>/   artifacts produced by one eval run:
    dsl.json         generated PluginDSL
    vision.json      raw Moondream answers + latencies
    render.html      Babel-standalone harness with components inlined
    render.png       headless-Chromium screenshot of the rendered DSL
    score.json       per-criterion pass/fail + full scorecard record
summary.md        rewritten on every run — diffable table of all cases
VOCABULARY_GAP.md design-side doc listing the components/row-kinds our system
                  needs to cover the long tail
```

## Run

```sh
# full pipeline — hits Modal for vision + DSL, re-renders, screenshots
node scripts/eval.mjs

# single case
node scripts/eval.mjs wavetable-synth

# use cached DSLs and only re-render + re-score (fast iteration after schema
# or renderer changes — no LLM calls)
node scripts/eval.mjs --skip-vision
```

Artifacts are keyed by image basename. Deleting `outputs/<name>/` triggers a
full re-run for that case on the next invocation.

## Scorecard dimensions

Each case scores ~7 criteria (scaled based on presence of module-kinds etc.):

| criterion | rule |
|-----------|------|
| archetype | `meta.productType` contains truth keyword (case-insensitive) |
| chassis | exact match |
| canvas | within ±20% of truth on both axes |
| row-kinds | all expected kinds appear; extras OK |
| knobs | count within `[minKnobs, maxKnobs]` |
| sliders | count within `[minSliders, maxSliders]` |
| module-kinds | at least one expected kind appears |
| accent-hue | oklch H within `accentHueTolerance`° of truth H |

`summary.md` is written to this directory after each run — diff it between
commits to see whether a prompt/schema change helped or hurt.

## Adding a new case

1. Drop a reference screenshot in `references/<name>.png`.
2. Add a truth entry in `truth.js` with expected structural metrics and
   a `gaps` list of any design patterns the current vocabulary can't cover.
3. `node scripts/eval.mjs <name>` to generate the artifacts.
4. Review `outputs/<name>/render.png` vs the reference and refine the
   truth entry.

## Why `truth.js` is soft

Qwen is expected to diverge from the reference (different brand name,
hue-shifted palette, original preset names). What it should *preserve* is
structural fidelity: correct archetype, correct chassis, correct row
vocabulary, control density in the right range. That's what the scorecard
measures. Tight pixel matching would double-punish IP-safe divergence.
