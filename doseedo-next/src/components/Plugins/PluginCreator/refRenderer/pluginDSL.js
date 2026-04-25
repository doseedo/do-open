/**
 * Plugin DSL — the structured JSON the LLM emits, and the renderer
 * consumes. Based on /Users/hydroadmin/Downloads/plugin editor/ARCHITECTURE.md §2.
 *
 * Shape overview (see validatePluginDSL for enforced rules):
 *
 *   {
 *     meta: { name, productType, version, canvas: [w,h], chassis, theme },
 *     palette: { bg: [oklch×5], ink: [oklch×4], line, lineSoft,
 *                accent, accent2?, led?, ok?, warn? },
 *     type:    { brand, ui, mono, display? },        // Google font families
 *     header?: HeaderSpec,
 *     rows:    RowSpec[],
 *   }
 *
 *   RowSpec kinds:
 *     - "module-strip"   — columns of modules (synth voice row)
 *     - "mod-matrix"     — macros / env / lfo / velocity panels
 *     - "keyboard-strip" — piano keyboard
 *     - "button-row"     — program/param buttons (STRATA-style)
 *     - "slider-bank"    — column of vertical sliders
 *     - "led-display"    — 7-segment readout + meter (STRATA-style)
 *
 *   ModuleSpec kinds:
 *     - "sub"             — tiny sub osc
 *     - "wavetable-osc"   — wavetable / stepped / granular / noise
 *     - "noise"           — noise source
 *     - "filter"          — filter w/ FilterCurve display
 *     - "envelope"        — ADSR with EnvCurve
 *     - "lfo"             — LFO with LFOCurve
 *     - "reverb-programs" — program button grid + meters (STRATA)
 */

// ────────────────────────────────────────────────────────────────
// Constants + enums (exported so the prompt can cite them)

export const CHASSIS = ['plugin-window', 'rack-hardware', 'pedal'];
export const ROW_KINDS = [
  'module-strip',
  'mod-matrix',
  'keyboard-strip',
  'button-row',
  'slider-bank',
  'led-display',
  // Character processors (RC-20 / VHS-88 archetype) — a horizontal
  // row of large "amount" knobs with LED bypass indicators, one per
  // module in the module-strip above. Use ALONGSIDE a module-strip.
  'character-row',
  // Compact footer strip used by RC-20-family: IN GAIN · EQ band ·
  // OUT WIDTH · OUT GAIN, all inline.
  'eq-strip',
];
export const MODULE_KINDS = [
  'sub',
  'wavetable-osc',
  'noise',
  'filter',
  'envelope',
  'lfo',
  'macros',
  'velocity',
  'reverb-programs',
  // Character-processor column (RC-20 / VHS-88). Has an optional tint,
  // optional top-selector (VINYL / TAPE / TUBE etc.), optional morph
  // bar (WOW ↔ FLUTTER style), a cluster of 2-4 small knobs, and an
  // optional flux-lane strip at the bottom.
  'character-module',
  // Square 2D control pad with a cursor and optional constellation
  // background. Used in granular engines / performance UIs.
  'xy-pad',
  // Editable modulation lane — step sequencer OR smooth curve, with
  // breakpoints and optional side-knobs (RATE/HUMANIZE/DEPTH).
  'mod-lane',
  // Generic knob-bank panel (for when no existing kind fits — compressor,
  // saturator, etc.). Renderer falls back to a grid of knobs.
  'knob-bank',
];
export const KNOB_FORMATS = ['percent', 'db', 'semi', 'hz', 'ms', 'count', 'raw', 'pan'];
export const DISPLAY_VARIANTS = [
  'wavetable-stacked',
  'wavetable-stepped',
  'granular-sample',
  'noise-wave',
  'filter-curve',
  'env-curve',
  'lfo-curve',
  'velocity-curve',
  // New
  'step-lane',              // mod-lane step mode
  'curve-lane',             // mod-lane smooth curve
  'xy-constellation',       // xy-pad with starfield
  'xy-grid',                // xy-pad with plain grid
  'flux-lane',              // mini waveform strip for character modules
];

// IP safety — products we refuse to reproduce by name. Extend as needed.
export const BANNED_BRANDS = [
  'serum', 'serum 2', 'vital', 'massive', 'massive x',
  'lexicon', 'lexicon 480', 'lexicon 224',
  'valhalla', 'valhalla vintage verb', 'valhalla shimmer',
  'fabfilter', 'pro-q', 'pro-l',
  'waves', 'ssl', 'neve',
  'soundtoys', 'echoboy',
  'sylenth1', 'diva', 'zebra',
  // Lo-fi / color processors (added for new eval cases)
  'rc-20', 'rc-20 retro color', 'rc20', 'xln audio',
  'decapitator', 'sausage fattener', 'saturn',
  // Granular engines
  'output', 'portal', 'arcade', 'kontakt', 'maschine',
];

// ────────────────────────────────────────────────────────────────
// Validator — hand-rolled, no zod dep. Returns {ok, errors}.

const isOklch = (s) =>
  typeof s === 'string' &&
  /^oklch\(\s*\d*\.?\d+\s+\d*\.?\d+\s+\d+(\.\d+)?\s*(\/\s*\d*\.?\d+\s*)?\)$/.test(s);

const isHex = (s) => typeof s === 'string' && /^#[0-9a-fA-F]{3,8}$/.test(s);
const isColor = (s) => isOklch(s) || isHex(s);

export function validatePluginDSL(dsl) {
  const errors = [];
  const push = (path, msg) => errors.push(`${path}: ${msg}`);

  if (!dsl || typeof dsl !== 'object') {
    return { ok: false, errors: ['dsl: must be an object'] };
  }

  // meta
  const m = dsl.meta;
  if (!m || typeof m !== 'object') push('meta', 'required object');
  else {
    if (!m.name || typeof m.name !== 'string') push('meta.name', 'required string');
    else if (BANNED_BRANDS.includes(m.name.toLowerCase()))
      push('meta.name', `"${m.name}" is a banned brand — pick an original name`);
    if (!m.productType) push('meta.productType', 'required string');
    if (!Array.isArray(m.canvas) || m.canvas.length !== 2)
      push('meta.canvas', 'required [width, height]');
    else {
      const [w, h] = m.canvas;
      if (!Number.isFinite(w) || w < 320 || w > 2400)
        push('meta.canvas[0]', `width ${w} must be 320..2400`);
      if (!Number.isFinite(h) || h < 200 || h > 1600)
        push('meta.canvas[1]', `height ${h} must be 200..1600`);
    }
    if (m.chassis && !CHASSIS.includes(m.chassis))
      push('meta.chassis', `must be one of ${CHASSIS.join('|')}`);
  }

  // palette
  const p = dsl.palette;
  if (!p || typeof p !== 'object') push('palette', 'required object');
  else {
    if (!Array.isArray(p.bg) || p.bg.length < 3)
      push('palette.bg', 'required array of ≥3 colors');
    else p.bg.forEach((c, i) => {
      if (!isColor(c)) push(`palette.bg[${i}]`, `not a valid color: ${c}`);
    });
    if (!Array.isArray(p.ink) || p.ink.length < 2)
      push('palette.ink', 'required array of ≥2 colors');
    else p.ink.forEach((c, i) => {
      if (!isColor(c)) push(`palette.ink[${i}]`, `not a valid color: ${c}`);
    });
    if (!isColor(p.accent)) push('palette.accent', `not a valid color: ${p.accent}`);
    if (p.accent2 != null && !isColor(p.accent2))
      push('palette.accent2', `not a valid color: ${p.accent2}`);
  }

  // type
  const t = dsl.type;
  if (!t || typeof t !== 'object') push('type', 'required object');
  else {
    if (!t.brand) push('type.brand', 'required Google font family');
    if (!t.ui) push('type.ui', 'required Google font family');
    if (!t.mono) push('type.mono', 'required Google font family');
  }

  // rows
  if (!Array.isArray(dsl.rows) || dsl.rows.length === 0)
    push('rows', 'required non-empty array');
  else {
    dsl.rows.forEach((row, i) => {
      const path = `rows[${i}]`;
      if (!row || !ROW_KINDS.includes(row.kind))
        return push(`${path}.kind`, `must be one of ${ROW_KINDS.join('|')}`);
      validateRow(row, path, push);
    });
  }

  return { ok: errors.length === 0, errors };
}

function validateRow(row, path, push) {
  switch (row.kind) {
    case 'module-strip':
      if (!Array.isArray(row.modules) || row.modules.length === 0)
        push(`${path}.modules`, 'required non-empty array');
      else row.modules.forEach((mod, j) => {
        const mp = `${path}.modules[${j}]`;
        if (!mod || !MODULE_KINDS.includes(mod.kind))
          return push(`${mp}.kind`, `must be one of ${MODULE_KINDS.join('|')}`);
        validateModule(mod, mp, push);
      });
      break;
    case 'mod-matrix':
      if (!Array.isArray(row.panels) || row.panels.length === 0)
        push(`${path}.panels`, 'required non-empty array');
      else row.panels.forEach((panel, j) => {
        const pp = `${path}.panels[${j}]`;
        if (!panel.kind)
          push(`${pp}.kind`, 'required: macros|envelope|lfo|velocity');
      });
      break;
    case 'keyboard-strip':
      if (!row.spec) push(`${path}.spec`, 'required KeyboardSpec');
      break;
    case 'button-row':
      if (!Array.isArray(row.buttons) || row.buttons.length === 0)
        push(`${path}.buttons`, 'required non-empty array');
      break;
    case 'character-row':
      if (!Array.isArray(row.knobs) || row.knobs.length === 0)
        push(`${path}.knobs`, 'character-row requires knobs[]');
      else row.knobs.forEach((k, j) => {
        const kp = `${path}.knobs[${j}]`;
        if (typeof k.label !== 'string') push(`${kp}.label`, 'required');
        if (typeof k.value !== 'number' || k.value < 0 || k.value > 1)
          push(`${kp}.value`, 'number in [0,1]');
      });
      break;
    case 'eq-strip':
      // Shape: { kind: "eq-strip", io: {in, out?}, eq?: {cut, tone?} }
      if (!row.io && !row.eq) push(`${path}`, 'eq-strip must have io or eq');
      break;
    case 'slider-bank':
      if (!Array.isArray(row.sliders) || row.sliders.length === 0)
        push(`${path}.sliders`, 'required non-empty array');
      break;
    case 'led-display':
      if (!row.spec) push(`${path}.spec`, 'required LedDisplaySpec');
      break;
    default:
      push(`${path}.kind`, `unknown kind ${row.kind}`);
  }
}

function validateModule(mod, path, push) {
  // Modules without knobs: xy-pad (cursor only), mod-lane (points only),
  // reverb-programs (button grid), lfo (curve-only variant allowed).
  const KNOBLESS = new Set(['xy-pad', 'mod-lane', 'reverb-programs', 'lfo']);
  if (!Array.isArray(mod.knobs)) {
    if (!KNOBLESS.has(mod.kind))
      push(`${path}.knobs`, 'required array (may be empty)');
  } else {
    mod.knobs.forEach((k, i) => {
      const kp = `${path}.knobs[${i}]`;
      if (typeof k.label !== 'string') push(`${kp}.label`, 'required string');
      if (typeof k.value !== 'number' || k.value < 0 || k.value > 1)
        push(`${kp}.value`, 'required number in [0,1]');
      if (k.format && !KNOB_FORMATS.includes(k.format))
        push(`${kp}.format`, `must be one of ${KNOB_FORMATS.join('|')}`);
    });
  }

  // Module-kind-specific refinements
  if (mod.kind === 'filter') {
    const labels = (mod.knobs || []).map((k) => k.label.toLowerCase());
    if (!labels.some((l) => l.includes('cut')))
      push(`${path}.knobs`, 'filter module must have a CUTOFF knob');
  }
  if (mod.kind === 'envelope') {
    const labels = (mod.knobs || []).map((k) => k.label.toLowerCase());
    const needed = ['a', 'd', 's', 'r']; // ATK/DEC/SUS/REL initials
    const hasAll = needed.every((n) => labels.some((l) => l.startsWith(n) || l.includes(n)));
    if (!hasAll) push(`${path}.knobs`, 'envelope should have ATK/DEC/SUS/REL knobs');
  }
}

// ────────────────────────────────────────────────────────────────
// A terse, model-facing description of the schema. Include this in the
// system prompt instead of the full TypeScript types.

export const SCHEMA_FOR_PROMPT = `
PluginDSL shape:
{
  "meta": {
    "name": string,                   // ORIGINAL brand (never a real product name)
    "productType": string,            // e.g. "Wavetable Synthesizer", "Digital Reverb"
    "version": string,
    "canvas": [width, height],        // 320..2400 × 200..1600
    "chassis": ${JSON.stringify(CHASSIS)},
    "theme": string
  },
  "palette": {
    "bg":   [oklch×≥3],               // darkest → lightest background shades
    "ink":  [oklch×≥2],               // primary text → faint text
    "line": oklch, "lineSoft": oklch,
    "accent": oklch,                  // hero color
    "accent2": oklch?, "led": oklch?, "ok": oklch?, "warn": oklch?
  },
  "type": {
    "brand": GoogleFont, "ui": GoogleFont, "mono": GoogleFont, "display"?: GoogleFont
  },
  "header"?: {                        // synth-style top bar
    "tabs"?: string[], "preset"?: {name, author, desc}
  },
  "rows": RowSpec[]                   // ordered top-to-bottom
}

RowSpec kinds: ${JSON.stringify(ROW_KINDS)}
  module-strip:   { kind, height, modules: ModuleSpec[] }
  mod-matrix:     { kind, height, panels: [{kind: "macros"|"envelope"|"lfo"|"velocity", ...}] }
  slider-bank:    { kind, height, sliders: [{label, value: 0..1}], brackets?: [{label, span:[from,to]}] }
  button-row:     { kind, height, style: "program"|"param-led", buttons: [{n?, label, active?, ledOn?}] }
  led-display:    { kind, height, spec: {value: string, units: ["sec","ms","Hz","kHz"]?, meter?: {lChannel, rChannel}} }
  keyboard-strip: { kind, height, spec: {octaves: number, pressed?: number[]} }
  character-row:  { kind, height, knobs: [{label, value: 0..1, active?, tint?: oklch}] }
                  // Hero "amount" knobs for lo-fi/color processors (RC-20 / VHS-88).
                  // USE THIS for NOISE/WOBBLE/DISTORT/DIGITAL/SPACE/MAGNETIC-style rows.
                  // One knob per column in the module-strip above.
  eq-strip:       { kind, height, io?: {in: {label, value}, out?: {labels: [], values: []}},
                    eq?: {cutLeft, cutRight, tone?} }
                  // Compact footer: IN GAIN · EQ CUT band · OUT WIDTH/GAIN inline.

ModuleSpec kinds: ${JSON.stringify(MODULE_KINDS)}
  wavetable-osc:    { kind, id, label, header?: {engine, tableName, dest}, meta?: [{k,v}],
                      display: {variant: "wavetable-stacked"|"wavetable-stepped"|"granular-sample"},
                      knobs: KnobSpec[], warpRow?: {...} }
  filter:           { kind, label, display: {variant: "filter-curve", cutoff: 0..1, res: 0..1},
                      modes?: string[], activeMode?: number, knobs: KnobSpec[] }
  envelope:         { kind, label, display: {variant:"env-curve", atk,dec,sus,rel ∈ 0..1},
                      knobs: KnobSpec[] }        // must include ATK/DEC/SUS/REL
  lfo:              { kind, label, display: {variant:"lfo-curve", points: [[x,y], ...]}, knobs: KnobSpec[] }
  macros:           { kind, label, knobs: KnobSpec[] }
  velocity:         { kind, label, display: {variant:"velocity-curve"}, knobs?: KnobSpec[] }
  noise:            { kind, label, display: {variant:"noise-wave"}, knobs: KnobSpec[] }
  sub:              { kind, label, waveform: "sine"|"tri"|"sq"|"saw", knobs: KnobSpec[] }
  reverb-programs:  { kind, programs: [{n,name}], activeProgram: number, paramButtons?: [...] }
  character-module: { kind, label, tint?: oklch, typeSelect?: string,
                      morph?: {a: string, b: string, position: 0..1},
                      knobs: KnobSpec[],                  // small knobs, 2-4 per module
                      fluxLane?: boolean }                // tiny waveform strip at bottom
                      // Use for RC-20 / VHS-88-style character columns:
                      //   TAPE / TUBE / NOISE / BITS / SPACE / MAGNETIC
  xy-pad:           { kind, label, background?: "constellation"|"grid"|"plain",
                      cursor?: [x: 0..1, y: 0..1],
                      axisLabels?: [string, string] }
                      // Square 2D control pad. Use in granular / performance UIs.
  mod-lane:         { kind, label, mode: "step"|"curve",
                      points: number[],                    // 6-16 values in 0..1
                      color?: "accent"|"accent2"|"ok"|"warn",
                      knobs?: [{label, value}] }           // RATE / HUMANIZE / DEPTH
                      // Editable modulation lane for granular / synth mod sections.
  knob-bank:        { kind, label, columns?: number, knobs: KnobSpec[] }
                      // Generic fallback — grid of labeled knobs.

KnobSpec: { label, value: 0..1, bipolar?: boolean,
            format?: ${JSON.stringify(KNOB_FORMATS)}, accent?: "primary"|"secondary"|"ok", primary?: boolean }

Hard rules:
  1. meta.name MUST NOT be a real product (Serum, Vital, Lexicon 480, Valhalla, Pro-Q, RC-20, etc).
  2. All colors MUST be oklch(L C H) strings, e.g. "oklch(0.78 0.15 70)".
  3. Every knob value MUST be in [0,1]. Use knob.format for the display transform.
  4. Canvas width × height MUST be in [320,2400] × [200,1600].
  5. filter modules MUST have a CUTOFF knob. envelope modules MUST have ATK/DEC/SUS/REL knobs.

Archetype → vocabulary guide (use this to pick the right row/module kinds):
  - Wavetable / subtractive / FM SYNTH → module-strip (OSC/FILTER/NOISE modules)
    + mod-matrix (macros/env/lfo/velocity) + keyboard-strip.
  - Digital REVERB (rack-style) → led-display + button-row×2 + slider-bank.
  - Lo-fi / tape / vinyl / character / color PROCESSOR (e.g. multi-stage tape color)
    → module-strip with 6 character-module columns + character-row (6 hero knobs
    matching the column count, one-to-one) + optional eq-strip footer.
    meta.productType should contain "lofi" or "color" or "tape" or "character".
  - Granular ENGINE / sampler performance UI → module-strip mixing wavetable-osc
    (set display.variant="granular-sample") with xy-pad and mod-lane modules.
    meta.productType should contain "granular".
  - EQ / compressor / saturator (single-function FX without obvious archetype
    above) → module-strip with knob-bank modules + optional eq-strip footer.
`.trim();
