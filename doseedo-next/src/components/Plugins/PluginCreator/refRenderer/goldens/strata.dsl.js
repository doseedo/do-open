/**
 * Golden DSL: STRATA 240 Digital Reverb.
 *
 * Extracted from /Users/hydroadmin/Downloads/plugin editor/strata/*.jsx.
 * STRATA is an original (IP-diverged) brand modeled on the classic
 * rack-digital-reverb archetype. The small-screen program-bank + fader
 * layout is the signature of this archetype.
 */

export const strataDSL = {
  meta: {
    name: 'STRATA',
    model: '240',
    productType: 'Digital Reverb',
    version: 'v1',
    canvas: [480, 780],
    chassis: 'rack-hardware',
    theme: 'cream-faceplate-red-led',
  },
  palette: {
    bg: [
      'oklch(0.92 0.025 82)',     // cream faceplate
      'oklch(0.88 0.022 82)',
      'oklch(0.78 0.020 82)',
      'oklch(0.55 0.015 82)',
    ],
    ink: [
      'oklch(0.22 0.01 60)',
      'oklch(0.40 0.01 60)',
      'oklch(0.55 0.01 60)',
    ],
    line:     'oklch(0.58 0.015 60)',
    lineSoft: 'oklch(0.72 0.012 60)',
    accent:   'oklch(0.62 0.22 25)',  // red — the LED
    led:      'oklch(0.62 0.22 25)',
    ok:       'oklch(0.65 0.16 140)',
    warn:     'oklch(0.62 0.22 25)',
  },
  type: {
    brand:   'Bitter',          // slab-serif for the faceplate wordmark
    ui:      'IBM Plex Sans',
    mono:    'IBM Plex Mono',
    display: 'Bitter',
  },
  rows: [
    {
      kind: 'led-display',
      height: 110,
      spec: {
        value: '2.4',
        digitSize: 54,
        units: ['sec', 'ms', 'Hz', 'kHz'],
        activeUnit: 0,
        meter: { lChannel: 5, rChannel: 4 },
        indicator: 'OPEN',
        captions: ['HEADROOM · dB', '⇧ + CLK = CHORUS'],
      },
    },
    {
      kind: 'button-row',
      style: 'program',
      height: 82,
      buttons: [
        { n: 1, name: 'Long Plate' },
        { n: 2, name: 'Tape Room' },
        { n: 3, name: 'Hall 01' },
        { n: 4, name: 'Booth' },
        { n: 5, name: 'Vocal Glass', active: true },
        { n: 6, name: 'Infinity' },
        { n: 7, name: 'Short Room' },
        { n: 8, name: 'Drum Plate' },
      ],
    },
    {
      kind: 'button-row',
      style: 'param-led',
      height: 82,
      buttons: [
        { label: 'GATE' },
        { label: 'FLOOR<br/>NOISE', ledOn: true, active: true },
        { label: 'REAR<br/>SENDS' },
        { label: 'SHAPE<br/>ALT', ledOn: true, active: true },
        { label: 'TAIL<br/>OPT' },
        { label: '&lt;DRY/WET&gt;' },
        { label: 'MIX' },
        { label: 'SOLO' },
      ],
    },
    {
      kind: 'slider-bank',
      height: 220,
      sliders: [
        { label: 'BASS',              value: 0.45 },
        { label: 'MID',               value: 0.55 },
        { label: 'CROSS<br/>OVER',    value: 0.35 },
        { label: 'TREBLE<br/>DECAY',  value: 0.70 },
        { label: 'DEPTH',             value: 0.50 },
        { label: 'PRE-<br/>DELAY',    value: 0.30 },
      ],
      brackets: [
        { label: 'REVERB TIME', span: [0, 1] },
      ],
    },
  ],
};

export default strataDSL;
