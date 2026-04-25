/**
 * Golden DSL: HELIX Wavetable Synthesizer.
 *
 * Extracted by reading the reference implementation at
 * /Users/hydroadmin/Downloads/plugin editor/helix/*.jsx + styles.css.
 * Used both as a renderer snapshot target and as a few-shot for the
 * Qwen DSL generator. Keep terse — every byte counts toward the 32K
 * context window.
 *
 * Name "HELIX" is an original (IP-diverged) brand, picked at design
 * time to be distinct from the wavetable-synth archetypes it was
 * modeled on. Safe to ship and use as a few-shot.
 */

export const helixDSL = {
  meta: {
    name: 'HELIX',
    productType: 'Wavetable Synthesizer',
    version: 'v2',
    canvas: [1440, 900],
    chassis: 'plugin-window',
    theme: 'amber-dark',
  },
  palette: {
    bg:   [
      'oklch(0.15 0.005 60)',
      'oklch(0.19 0.006 60)',
      'oklch(0.23 0.007 60)',
      'oklch(0.28 0.008 60)',
      'oklch(0.33 0.009 60)',
    ],
    ink:  [
      'oklch(0.88 0.01 60)',
      'oklch(0.72 0.01 60)',
      'oklch(0.52 0.01 60)',
      'oklch(0.40 0.008 60)',
    ],
    line:     'oklch(0.30 0.008 60)',
    lineSoft: 'oklch(0.26 0.006 60)',
    accent:   'oklch(0.80 0.15 70)',  // amber
    led:      'oklch(0.80 0.15 70)',
    ok:       'oklch(0.75 0.14 155)',
    warn:     'oklch(0.72 0.17 30)',
  },
  type: {
    brand: 'IBM Plex Serif',
    ui:    'Inter',
    mono:  'JetBrains Mono',
  },
  header: {
    tabs: ['MAIN', 'MIX', 'FX', 'MATRIX', 'GLOBAL'],
    preset: { name: 'Ember Drift', author: 'Ash Keller' },
  },
  rows: [
    {
      kind: 'module-strip',
      height: 340,
      modules: [
        {
          kind: 'sub',
          label: 'SUB',
          waveform: 'sine',
          oct: '-2',
          crs: '—',
          knobs: [
            { label: 'PAN',   value: 0.5, bipolar: true, format: 'pan' },
            { label: 'LEVEL', value: 0.65, format: 'raw' },
          ],
        },
        {
          kind: 'wavetable-osc',
          id: 'A',
          label: 'A',
          active: false,
          flex: 1.3,
          header: { engine: 'WAVETABLE', tableName: 'Ember Flow', dest: 'F1' },
          meta: [
            { k: 'OCT', v: '+2' },
            { k: 'SEM', v: '-4' },
            { k: 'FIN', v: '-41' },
            { k: 'CRS', v: '-64.0' },
          ],
          display: { variant: 'wavetable-stacked' },
          knobs: [
            { label: 'PAN',    value: 0.5,  bipolar: true, format: 'pan', primary: true },
            { label: 'WT POS', value: 0.3,  format: 'raw' },
            { label: 'UNISON', value: 0.25, format: 'count' },
            { label: 'DETUNE', value: 0.3,  format: 'raw' },
            { label: 'BLEND',  value: 0.5,  format: 'raw' },
            { label: 'LEVEL',  value: 0.72, format: 'raw' },
          ],
        },
        {
          kind: 'wavetable-osc',
          id: 'B',
          label: 'B',
          flex: 1.3,
          header: { engine: 'WAVETABLE', tableName: 'Ribbon Stack' },
          meta: [
            { k: 'OCT', v: '-2' }, { k: 'SEM', v: '-7' },
            { k: 'FIN', v: '0' },  { k: 'CRS', v: '64.0' },
          ],
          display: { variant: 'wavetable-stepped' },
          knobs: [
            { label: 'PAN',    value: 0.5,  bipolar: true, format: 'pan' },
            { label: 'WT POS', value: 0.6,  format: 'raw' },
            { label: 'UNISON', value: 0.4,  format: 'count' },
            { label: 'DETUNE', value: 0.35, format: 'raw' },
            { label: 'BLEND',  value: 0.5,  format: 'raw' },
            { label: 'LEVEL',  value: 0.68, format: 'raw' },
          ],
        },
        {
          kind: 'wavetable-osc',
          id: 'C',
          label: 'C',
          active: true,
          flex: 1.4,
          header: { engine: 'GRANULAR', tableName: 'Dust Impact', dest: 'F2' },
          meta: [
            { k: 'OCT', v: '-1' }, { k: 'SEM', v: '0' },
            { k: 'FIN', v: '0' },  { k: 'CRS', v: '0.0' },
          ],
          display: { variant: 'granular-sample' },
          knobs: [
            { label: 'WARP',   value: 0.3, format: 'raw', accent: 'ok' },
            { label: 'SCAN',   value: 0.6, format: 'raw', accent: 'ok' },
            { label: 'DENS',   value: 0.4, format: 'raw', accent: 'ok' },
            { label: 'LENGTH', value: 0.5, format: 'raw', accent: 'ok' },
            { label: 'PAN',    value: 0.5, bipolar: true, format: 'pan', accent: 'ok' },
            { label: 'LEVEL',  value: 0.7, format: 'raw', accent: 'ok' },
            { label: 'OFFSET', value: 0.2, format: 'raw', accent: 'ok' },
            { label: 'DIR',    value: 0.5, bipolar: true, format: 'raw', accent: 'ok' },
            { label: 'PITCH',  value: 0.4, bipolar: true, format: 'semi', accent: 'ok' },
          ],
        },
        {
          kind: 'noise',
          label: 'NOISE',
          header: { tableName: 'Cream' },
          knobs: [
            { label: 'PAN',   value: 0.5, bipolar: true, format: 'pan' },
            { label: 'LEVEL', value: 0.55, format: 'raw' },
          ],
        },
        {
          kind: 'filter',
          label: 'FILTER 1',
          hue: 'amber',
          header: { routing: 'Low-Pass 24' },
          modes: ['PRE', 'A', 'B', 'C', 'POST', 'VIZ'],
          activeMode: 1,
          knobs: [
            { label: 'CUTOFF', value: 0.6, format: 'hz' },
            { label: 'RES',    value: 0.4, format: 'raw' },
            { label: 'PAN',    value: 0.5, bipolar: true, format: 'pan' },
            { label: 'DRIVE',  value: 0.3, format: 'raw' },
            { label: 'DAMP',   value: 0.45, format: 'raw' },
            { label: 'MIX',    value: 0.7, format: 'raw' },
            { label: 'LEVEL',  value: 0.65, format: 'raw' },
          ],
        },
        {
          kind: 'filter',
          label: 'FILTER 2',
          hue: 'amber',
          active: true,
          header: { routing: 'Plate Verb' },
          modes: ['PRE', 'A', 'B', 'C', 'POST', 'VIZ'],
          activeMode: 0,
          knobs: [
            { label: 'CUTOFF', value: 0.55, format: 'hz' },
            { label: 'RES',    value: 0.5, format: 'raw' },
            { label: 'PAN',    value: 0.5, bipolar: true, format: 'pan' },
            { label: 'DRIVE',  value: 0.4, format: 'raw' },
            { label: 'DAMP',   value: 0.5, format: 'raw' },
            { label: 'MIX',    value: 0.6, format: 'raw' },
            { label: 'LEVEL',  value: 0.6, format: 'raw' },
          ],
        },
      ],
    },
    {
      kind: 'mod-matrix',
      height: 220,
      panels: [
        {
          kind: 'macros',
          label: 'MACROS',
          knobs: [
            { label: 'M1', value: 0.4, primary: true },
            { label: 'M2', value: 0.6 },
            { label: 'M3', value: 0.3 },
            { label: 'M4', value: 0.7 },
            { label: 'M5', value: 0.5 },
            { label: 'M6', value: 0.2 },
            { label: 'M7', value: 0.55 },
            { label: 'M8', value: 0.45 },
          ],
        },
        {
          kind: 'envelope',
          label: 'ENV 1 · AMP',
          dest: 'VOL',
          display: { variant: 'env-curve' },
          knobs: [
            { label: 'ATK',  value: 0.15, format: 'ms' },
            { label: 'DEC',  value: 0.25, format: 'ms' },
            { label: 'SUS',  value: 0.6,  format: 'percent' },
            { label: 'REL',  value: 0.35, format: 'ms' },
            { label: 'HOLD', value: 0.1,  format: 'ms' },
          ],
        },
        {
          kind: 'lfo',
          label: 'LFO 1',
          dest: 'PITCH',
          display: { variant: 'lfo-curve' },
          knobs: [
            { label: 'RATE',  value: 0.4, format: 'hz' },
            { label: 'DEPTH', value: 0.6, format: 'percent' },
            { label: 'PHASE', value: 0.5, bipolar: true, format: 'raw' },
          ],
        },
        {
          kind: 'velocity',
          label: 'VELOCITY',
          display: { variant: 'velocity-curve', curve: 1.4 },
          knobs: [
            { label: 'CURVE', value: 0.6, format: 'raw' },
            { label: 'DEPTH', value: 0.7, format: 'percent' },
          ],
        },
      ],
    },
    {
      kind: 'keyboard-strip',
      height: 70,
      spec: { octaves: 5, pressed: [48, 52, 55] },
    },
  ],
};

export default helixDSL;
