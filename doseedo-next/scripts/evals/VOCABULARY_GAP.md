# Vocabulary gap — what's needed to match the new reference set

**Reference set** (scripts/evals/references/):
- `wavetable-synth.png` — generic Serum-archetype wavetable synth
- `digital-reverb.png` — Lexicon 224 archetype (rack digital reverb)
- `lofi-color.png` — XLN Audio RC-20 Retro Color (tape/vinyl character processor)
- `granular-engine.png` — Output Portal / granular-engine archetype

**Hand-authored targets** (from `/Users/hydroadmin/Downloads/plugin editor (2)/`):
- `helix/` — wavetable synth (already ported as `helixDSL` golden)
- `strata/` — digital reverb (already ported as `strataDSL` golden)
- `vhs/` — VHS-88 Analog Grain — character processor modeled on RC-20
- `nebula/` — NEBULA Granular Engine — modeled on granular reference

Two new archetypes (`lofi-color`, `granular-engine`) that the current system **cannot reproduce structurally**, even with a correct vision pass. Gap summary below, ranked by implementation cost.

---

## 1. `character-row` + `CharacterKnob` primitive  — RC-20 / VHS-88 archetype

**What's missing**

Hero "amount" knob row (a single row of ~6 large knobs, one per character module). Currently we have no row kind that renders a flat grid of large labeled knobs with bypass LEDs.

**New components**
- `<CharacterKnob>` primitive — larger diameter (~86px), ivory/cream fill, subtle rim ticks, single black indicator line, bypass LED square in top-left corner.
- `<CharacterRow>` row kind — `{ kind: "character-row", knobs: [{label, value, active, tint?}] }`.

**Estimated cost:** ~2 hours (primitive + row renderer + CSS + one new module-kind `character-module` that maps to a single column in the module-strip above).

**Unlocks:** RC-20-family (tape/vinyl color), multi-effect character processors (iZotope VocalSynth archetype), anything with a "6 hero knobs across the bottom" pattern.

---

## 2. Tinted module variant + `MorphSelector` + `FluxLane` — per-column character-module

**What's missing**

Current `module-strip` modules use one shared palette. The RC-20 archetype requires per-column accent tints (sienna / bronze / moss / steel-blue / seafoam / graphite in VHS-88) and two new in-module primitives.

**New components**
- `ModuleSpec.tint?: oklch` — per-module accent override that cascades into the module's CSS vars.
- `<MorphSelector labelA labelB position>` — "A ↔ B" dual-label track with a draggable dot (e.g. WOW ↔ FLUTTER).
- `<FluxLane>` — mini 22px strip with a sine polyline + a position dot, representing per-module modulation.
- `<StereoToggle>` — two overlapping circles glyph used in VHS-88's stereo buttons.

**Estimated cost:** ~2 hours (three primitives + CSS-var cascade for tints).

**Unlocks:** character processors where each column has distinct visual identity.

---

## 3. `EQStrip` row kind + magnitude slider + preset bar — RC-20 footer pattern

**What's missing**

The RC-20 footer is a single wide row: IN GAIN · [EQ enable LED | ← mode curve | CUT slider | mode curve → | TONE | mode toggle] · OUT WIDTH · OUT GAIN. We don't have a composite row like this. Similar shape appears in SSL channels, channel-strip plugins.

Also missing: header patterns like the RC-20/VHS-88 top bar (brand mark + preset picker w/ up/down arrows + LOAD/SAVE + MAGNITUDE horizontal slider).

**New components**
- `<EQStripRow>` — `{ kind: "eq-strip", io: {in, out}, eq: {curve, cutSlider, tone}, toggles: [...] }`.
- Header extension: `header.magnitude?: {value}` (horizontal slider in top-right).
- `<PresetPicker>` — name text + ▲▼ buttons + LOAD + SAVE chip group.

**Estimated cost:** ~2 hours.

**Unlocks:** RC-20 footer, SSL/Waves channel strip bottom bar, anything with inline EQ + I/O trim.

---

## 4. `xy-pad` row kind + constellation visualization — granular archetype

**What's missing**

The granular reference has a square XY pad with a starfield/constellation backdrop (~60 random dots + faint radial lines) and a trailing cursor dot. Qwen has no vocabulary to emit this.

**New components**
- `<XYPad>` module — `{ kind: "xy-pad", width, background: "constellation"|"grid"|"plain", cursor?: [x,y], axisLabels?: [x,y] }`.
- Constellation background generator (seeded random, deterministic for a given size).

**Estimated cost:** ~1.5 hours.

**Unlocks:** granular engines, Output Portal-style samplers, anything with interactive 2D control.

---

## 5. `mod-lane` module — step + curve editable envelopes

**What's missing**

Granular reference has two editable mod lanes (MOD 2 = step sequencer in yellow, MOD 3 = smooth curve in red). Each lane has 8–12 breakpoints, connecting lines, dot handles, and a side panel with RATE + HUMANIZE knobs + SYNC/RNDM/CLEAR chips.

We currently only have static `lfo-curve` and `env-curve` displays; no editable breakpoint lane.

**New components**
- `<ModLane>` module — `{ kind: "mod-lane", mode: "step"|"curve", color, points: [0..1]×N, sideKnobs: [{label, value}], chips: string[] }`.
- Chip primitive already exists (`Chip`) — just needs the render pattern.

**Estimated cost:** ~2 hours.

**Unlocks:** modulators in virtually every modern synth/sampler plugin.

---

## 6. `MacroKnob` variant with tick ring + live-pane layout

**What's missing**

Granular has two **~100px-diameter macro knobs** (GRANULATION + DIFFUSION) rendered with a dotted-tick value ring around a hollow inner knob. Different from the standard 270°-arc Knob. Also, the granular layout has **two panes** side-by-side (main + live) — our current `<RenderDSL>` always stacks rows vertically across the full canvas.

**New components**
- `<MacroKnob>` primitive — 48 rim ticks (lit proportional to value) + hollow inner disc + small indicator dot.
- `meta.layout?: "single-pane" | "dual-pane-60-40" | "dual-pane-50-50"` — adds a layout selector.
- `rows` can be split into `mainRows` + `livePaneRows` (or tag rows with `pane: "main" | "live"`).

**Estimated cost:** ~3 hours (renderer layout change is the larger piece).

**Unlocks:** Output Portal, Massive X macro-heavy workflows, performance-oriented layouts.

---

## 7. Variable-sweep knobs — `sweep?: 0..1` prop on `<Knob>`

**What's missing**

The granular reference uses **78% sweep** knobs (open at bottom, like an infinity encoder rather than a volume-style 270°). Ours are fixed 270°. Qwen can't currently request a different aesthetic.

**New param**
- `KnobSpec.sweep?: number` (0.5..1, default 0.75 for our existing look, 0.78 for granular-style).

**Estimated cost:** ~20 minutes.

**Unlocks:** style-matching to modern minimal UIs.

---

## Priority order + rough effort

| Phase | Item | Hours | Unlocks |
|-------|------|-------|---------|
| P1 | #7 variable-sweep knob | 0.3 | any plugin with non-270° knobs |
| P1 | #1 character-row + knob | 2 | RC-20 family |
| P1 | #2 tinted modules + morph + flux | 2 | VHS-88 quality character |
| P2 | #3 eq-strip + preset-bar | 2 | RC-20 footer, channel strips |
| P2 | #5 mod-lane module | 2 | modulation in any synth |
| P2 | #4 xy-pad module | 1.5 | granular, performance UIs |
| P3 | #6 macro-knob + dual-pane | 3 | granular live-pane, Portal |

**Total P1** (lo-fi/character processor support): ~4.3 hours, two new goldens afterward (VHS-88 as golden #3, maybe a compressor as golden #4 to round out the coverage).

**Total P1+P2** (adds granular + channel strip): ~7.8 hours.

**Total P1+P2+P3** (complete modern-plugin vocabulary): ~11 hours of focused implementation + 3 hours of vocabulary-addition to the system prompt + 2 hours porting VHS-88 and NEBULA as goldens = **~16 hours** to reach production parity with the hand-authored examples.

## Side note on goldens

Qwen3-14B overfits few-shots — we've seen it copy STRATA's program names verbatim. Adding VHS-88 and NEBULA as goldens will help archetype-match the lo-fi and granular cases, but it will also increase the chance of Qwen verbatim-cribbing their module names and preset names. Monitor for this in the eval harness and consider a post-generation pass that detects matches against golden tokens and asks Qwen to rename.
