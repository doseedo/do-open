# Ref-renderer evaluation — 2026-04-24T17:47:06.198Z

| case | score | archetype | chassis | canvas | rows | knobs | sliders | accent |
|------|-------|-----------|---------|--------|------|-------|---------|--------|
| wavetable-synth | **100%** (8/8) | Wavetable Synthesizer | plugin-window | 1280×800 | module-strip, mod-matrix, keyboard-strip | 52 | 0 | `oklch(0.80 0.15 230)` |
| digital-reverb | **100%** (7/7) | Digital Reverb | rack-hardware | 480×780 | led-display, button-row, slider-bank | 0 | 6 | `oklch(0.62 0.22 25)` |
| lofi-color | **100%** (8/8) | Analog Grain / Lo-Fi Color Plugin | plugin-window | 1000×820 | module-strip, character-row, eq-strip | 20 | 0 | `oklch(0.80 0.13 230)` |
| granular-engine | **100%** (8/8) | Granular Synthesizer | plugin-window | 1200×800 | module-strip, eq-strip | 33 | 0 | `oklch(0.80 0.13 230)` |

## Per-case detail

### wavetable-synth

- ✅ **archetype** — got "Wavetable Synthesizer", expected contains one of [wavetable]
- ✅ **chassis** — got "plugin-window", expected "plugin-window"
- ✅ **canvas** — got 1280x800, expected ~1440x900
- ✅ **row-kinds** — got [module-strip, mod-matrix, keyboard-strip]
- ✅ **knobs** — got 52, expected 30..60
- ✅ **sliders** — got 0, expected 0..10
- ✅ **module-kinds** — got [wavetable-osc, noise, filter, macros, envelope, lfo, velocity]
- ✅ **accent-hue** — got H=230, distance=160° (tolerance 180°)

_latency: vision 17.9s · dsl —s · prompt — / completion — tokens_

### digital-reverb

- ✅ **archetype** — got "Digital Reverb", expected contains one of [reverb]
- ✅ **chassis** — got "rack-hardware", expected "rack-hardware"
- ✅ **canvas** — got 480x780, expected ~480x780
- ✅ **row-kinds** — got [led-display, button-row, slider-bank]
- ✅ **knobs** — got 0, expected 0..0
- ✅ **sliders** — got 6, expected 4..8
- ✅ **accent-hue** — got H=25, distance=0° (tolerance 180°)

_latency: vision 15.9s · dsl —s · prompt — / completion — tokens_

### lofi-color

- ✅ **archetype** — got "Analog Grain / Lo-Fi Color Plugin", expected contains one of [color, lofi, lo-fi, tape, character, color]
- ✅ **chassis** — got "plugin-window", expected "plugin-window"
- ✅ **canvas** — got 1000x820, expected ~1000x820
- ✅ **row-kinds** — got [module-strip, character-row, eq-strip]
- ✅ **knobs** — got 20, expected 16..48
- ✅ **sliders** — got 0, expected 0..6
- ✅ **module-kinds** — got [character-module]
- ✅ **accent-hue** — got H=230, distance=150° (tolerance 180°)

**Known vocabulary gaps** (from truth.js, these are not scored):
- no character-row row kind
- no character-knob primitive (big cream-faced with bypass LED)
- no morph-selector primitive
- no flux-lane display
- no tinted module variant (each column needs own accent hue)
- no EQ-strip row kind (IN/EQ/OUT inline)
- no magnitude-slider primitive
- no preset-picker primitive

_latency: vision 10.6s · dsl —s · prompt — / completion — tokens_

### granular-engine

- ✅ **archetype** — got "Granular Synthesizer", expected contains one of [granular]
- ✅ **chassis** — got "plugin-window", expected "plugin-window"
- ✅ **canvas** — got 1200x800, expected ~1000x720
- ✅ **row-kinds** — got [module-strip, eq-strip]
- ✅ **knobs** — got 33, expected 10..40
- ✅ **sliders** — got 0, expected 0..4
- ✅ **module-kinds** — got [wavetable-osc, xy-pad, mod-lane]
- ✅ **accent-hue** — got H=230, distance=90° (tolerance 180°)

**Known vocabulary gaps** (from truth.js, these are not scored):
- no xy-pad component
- no mod-lane (step/curve editor) component
- no macro-knob with tick-ring
- no asymmetric pane layout (all current rows are full-width)
- no variable-sweep knob option (only 270° knobs)
- no brand-plate outside header
- no DRY/WET histogram
- no side-rail primitive

_latency: vision 16.9s · dsl —s · prompt — / completion — tokens_
