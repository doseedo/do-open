/**
 * Ground-truth expectations for each reference image in ../references/.
 *
 * Populated by hand from the source images + the corresponding
 * hand-authored plugin outputs in /Users/hydroadmin/Downloads/plugin editor*.
 * These are NOT strict assertions — the eval harness uses them to compute
 * soft scores (±20% on counts, ±30° on hue distance, required row-kinds
 * must all be present but extras are allowed).
 *
 * Each entry:
 *   archetype           — canonical productType keyword
 *   chassis             — plugin-window | rack-hardware | pedal
 *   canvas              — approx [w,h], ±20% tolerance
 *   expectedRowKinds    — at least these kinds must appear
 *   minKnobs / maxKnobs — knob count range
 *   minSliders / maxSliders — slider count range
 *   accentHue           — approx oklch H, ±30° tolerance
 *   moduleKinds         — at least one of these must appear
 *   requiredPatterns    — free-form list of design patterns the target
 *                         implementation needs to support (used for
 *                         vocabulary-gap reporting, not scoring).
 */

export const truth = {
  'wavetable-synth': {
    archetype: 'wavetable',           // productType should contain this
    chassis: 'plugin-window',
    canvas: [1440, 900],
    expectedRowKinds: ['module-strip', 'mod-matrix', 'keyboard-strip'],
    minKnobs: 30, maxKnobs: 60,
    minSliders: 0, maxSliders: 10,
    accentHue: 70,                    // amber — but any is fine, this is just the source
    accentHueTolerance: 180,          // accept any accent — Qwen is supposed to diverge
    moduleKinds: ['wavetable-osc', 'filter', 'envelope', 'lfo', 'macros'],
    requiredPatterns: [
      'module-strip row with wavetable-osc + filter + envelope modules',
      'mod-matrix row with macros + envelope + lfo + velocity panels',
      'keyboard-strip at bottom',
      'procedural wavetable display per OSC',
      'filter curve display in filter module',
      'ADSR envelope curve display',
    ],
  },

  'digital-reverb': {
    archetype: 'reverb',
    chassis: 'rack-hardware',
    canvas: [480, 780],
    expectedRowKinds: ['led-display', 'button-row', 'slider-bank'],
    minKnobs: 0, maxKnobs: 0,
    minSliders: 4, maxSliders: 8,
    accentHue: 25,                    // red
    accentHueTolerance: 180,
    moduleKinds: [],                  // rack-style — no module-strip
    requiredPatterns: [
      '7-segment LED display row',
      '8-button program picker',
      '8-button parameter row with LEDs',
      'vertical slider bank with REVERB TIME bracket',
      'stereo meter with LED segments',
    ],
  },

  'lofi-color': {
    archetype: 'color',               // accept any of: lofi, lo-fi, color, tape, character
    archetypeAlt: ['lofi', 'lo-fi', 'tape', 'character', 'color'],
    chassis: 'plugin-window',
    canvas: [1000, 820],              // VHS-88 golden dims; RC-20 is ~600x480 but Qwen will aim at VHS scale
    expectedRowKinds: ['module-strip', 'character-row'],
    minKnobs: 16, maxKnobs: 48,       // character plugins: ~6 character knobs + 6×(2-4) small knobs ≈ 18-30
    minSliders: 0, maxSliders: 6,
    accentHue: 80,                    // olive/amber on RC-20; VHS-88 uses 230. We allow any per tolerance.
    accentHueTolerance: 180,
    moduleKinds: ['character-module'],
    requiredPatterns: [
      'header with brand + preset picker + magnitude slider',
      '6 tinted module columns with per-module knob clusters',
      'morph-selector ("A ↔ B" dual-label bar) inside modules',
      'flux-lane mini waveform strip per module',
      'BIG character-knob row (hero knobs, one per module)',
      'per-character bypass LED indicator',
      'footer EQ + I/O strip (IN GAIN | EQ CUT + tone | OUT WIDTH/GAIN)',
      'nameplate at bottom',
    ],
    gaps: [
      'no character-row row kind',
      'no character-knob primitive (big cream-faced with bypass LED)',
      'no morph-selector primitive',
      'no flux-lane display',
      'no tinted module variant (each column needs own accent hue)',
      'no EQ-strip row kind (IN/EQ/OUT inline)',
      'no magnitude-slider primitive',
      'no preset-picker primitive',
    ],
  },

  'granular-engine': {
    archetype: 'granular',
    chassis: 'plugin-window',
    canvas: [1000, 720],
    expectedRowKinds: ['module-strip'],       // module-strip is the carrier for xy-pad + mod-lane modules
    minKnobs: 10, maxKnobs: 40,               // granular engines can be dense (Portal has 30+)
    minSliders: 0, maxSliders: 4,
    accentHue: 140,                           // green (lime)
    accentHueTolerance: 180,
    moduleKinds: ['xy-pad', 'mod-lane'],      // at least one of these must appear
    requiredPatterns: [
      'asymmetric 2-column layout (main pane + live pane)',
      'dense knob grid with mixed-size knobs (7-8 variable-size knobs)',
      'tick-ring macro knobs (dotted ring showing value position)',
      'XY pad with constellation / particle visualization',
      'mod lanes with step + curve editing (editable breakpoints)',
      '78%-sweep knobs (not 270°) — shorter arc for minimal aesthetic',
      'DRY↔WET track slider (horizontal with both endpoint labels)',
      'DRY/WET particle histogram',
      'brand plate in corner (separate from top header)',
      'side rail with icon glyphs',
    ],
    gaps: [
      'no xy-pad component',
      'no mod-lane (step/curve editor) component',
      'no macro-knob with tick-ring',
      'no asymmetric pane layout (all current rows are full-width)',
      'no variable-sweep knob option (only 270° knobs)',
      'no brand-plate outside header',
      'no DRY/WET histogram',
      'no side-rail primitive',
    ],
  },
};

export default truth;
