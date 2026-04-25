/**
 * Golden DSL: VHS-88 ANALOG GRAIN — lo-fi / character processor.
 *
 * Modeled after the hand-authored plugin editor (2)/vhs/app.jsx.
 * Teaches Qwen the character-row + tinted-character-module pattern
 * used by RC-20 / VHS-family plugins. The 6-column module-strip maps
 * one-to-one onto the 6-knob character-row below it.
 *
 * Brand-divergent from the reference (XLN RC-20 Retro Color):
 *   - original name VHS-88 (vs RC-20)
 *   - ice-blue accent hue 230 (vs RC-20's olive/teal ~80)
 *   - module labels renamed (NOISE→GRAIN, WOBBLE→WARP, etc.)
 */

export const vhsDSL = {
  meta: {
    name: 'VHS-88',
    productType: 'Analog Grain / Lo-Fi Color Plugin',
    version: 'v1',
    canvas: [1000, 820],
    chassis: 'plugin-window',
    theme: 'charcoal-ice-blue',
  },
  palette: {
    bg: [
      'oklch(0.16 0.008 260)',
      'oklch(0.20 0.010 260)',
      'oklch(0.25 0.012 260)',
      'oklch(0.30 0.014 260)',
      'oklch(0.35 0.016 260)',
    ],
    ink: [
      'oklch(0.92 0.015 260)',
      'oklch(0.75 0.012 260)',
      'oklch(0.55 0.010 260)',
      'oklch(0.42 0.008 260)',
    ],
    line:     'oklch(0.32 0.012 260)',
    lineSoft: 'oklch(0.26 0.010 260)',
    accent:   'oklch(0.80 0.13 230)',  // ice-blue
    accent2:  'oklch(0.72 0.12 30)',   // warm sienna secondary
    led:      'oklch(0.80 0.13 230)',
    ok:       'oklch(0.75 0.14 155)',
    warn:     'oklch(0.72 0.17 30)',
  },
  type: {
    brand: 'Space Grotesk',
    ui:    'Inter',
    mono:  'JetBrains Mono',
  },
  header: {
    tabs: ['MAIN', 'MIX', 'FX', 'MATRIX', 'GLOBAL'],
    preset: { name: 'Ferric Saturation 04', author: 'Ash Keller' },
  },
  rows: [
    {
      kind: 'module-strip',
      height: 260,
      modules: [
        {
          kind: 'character-module', label: 'TAPE',
          tint: 'oklch(0.55 0.15 30)',   // burnt sienna
          typeSelect: 'VINYL 1',
          knobs: [
            { label: 'TONE',    value: 0.40 },
            { label: 'ROUTING', value: 0.75 },
            { label: 'FOLLOW',  value: 0.35 },
            { label: 'DUCK',    value: 0.55 },
          ],
          fluxLane: true,
        },
        {
          kind: 'character-module', label: 'WARP',
          tint: 'oklch(0.55 0.12 60)',   // bronze
          morph: { a: 'WOW', b: 'FLUTTER', position: 0.4 },
          knobs: [
            { label: 'RATE',   value: 0.40 },
            { label: 'DEPTH',  value: 0.50 },
            { label: 'STEREO', value: 0.60 },
            { label: 'MIX',    value: 0.60 },
          ],
          fluxLane: true,
        },
        {
          kind: 'character-module', label: 'TUBE',
          tint: 'oklch(0.50 0.10 120)',  // moss
          typeSelect: 'TUBEPAIR',
          knobs: [
            { label: 'FOCUS', value: 0.55 },
            { label: 'MIX',   value: 0.60 },
          ],
          fluxLane: true,
        },
        {
          kind: 'character-module', label: 'BITS',
          tint: 'oklch(0.50 0.08 240)',  // steel-blue
          morph: { a: 'RATE', b: 'BITS', position: 0.50 },
          knobs: [
            { label: 'FOCUS',  value: 0.45 },
            { label: 'SMOOTH', value: 0.45 },
            { label: 'MIX',    value: 0.60 },
          ],
          fluxLane: true,
        },
        {
          kind: 'character-module', label: 'ROOM',
          tint: 'oklch(0.55 0.10 175)',  // seafoam
          knobs: [
            { label: 'DECAY',    value: 0.65 },
            { label: 'PREDELAY', value: 0.40 },
            { label: 'STEREO',   value: 0.60 },
            { label: 'FOCUS',    value: 0.35 },
          ],
          fluxLane: true,
        },
        {
          kind: 'character-module', label: 'TAPE',
          tint: 'oklch(0.35 0.06 270)',  // graphite
          morph: { a: 'WEAR', b: 'FLUTTER', position: 0.55 },
          knobs: [
            { label: 'RATE',     value: 0.50 },
            { label: 'DROPOUTS', value: 0.40 },
            { label: 'STEREO',   value: 0.50 },
          ],
          fluxLane: true,
        },
      ],
    },
    {
      kind: 'character-row',
      height: 150,
      knobs: [
        { label: 'GRAIN',    value: 0.30, active: true,  tint: 'oklch(0.55 0.15 30)' },
        { label: 'WARP',     value: 0.55, active: true,  tint: 'oklch(0.55 0.12 60)' },
        { label: 'CRUSH',    value: 0.45, active: true,  tint: 'oklch(0.50 0.10 120)' },
        { label: 'BITS',     value: 0.70, active: false, tint: 'oklch(0.50 0.08 240)' },
        { label: 'ROOM',     value: 0.40, active: true,  tint: 'oklch(0.55 0.10 175)' },
        { label: 'TAPE',     value: 0.60, active: true,  tint: 'oklch(0.60 0.13 230)' },
      ],
    },
    {
      kind: 'eq-strip',
      height: 72,
      io: {
        in: { label: 'GAIN', value: 0.5 },
        out: { labels: ['WIDTH', 'GAIN'], values: [0.6, 0.55] },
      },
      eq: { cutLeft: 0.15, cutRight: 0.85, tone: 0.5 },
    },
  ],
};

export default vhsDSL;
