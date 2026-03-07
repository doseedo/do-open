/**
 * Complex Synth Template Builder — Randomized Variation System
 *
 * Generates varied, high-quality 90-120+ component designbrief JSONs for Serum-class synths.
 * Each generation is unique — different names, aesthetics, tab arrangements, section compositions.
 * Bypasses GPT-4o for structure (it consistently generates sparse designs ~59 components)
 * while producing professional-level variety through randomized section pools.
 */

import { generateTheme } from './themeGenerator';

// ── Helpers ──────────────────────────────────────────────────────────────────

const pick = arr => arr[Math.floor(Math.random() * arr.length)];
const shuffle = arr => { const a = [...arr]; for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
const coinFlip = (p = 0.5) => Math.random() < p;

// ── Detection ────────────────────────────────────────────────────────────────

const COMPLEX_SYNTH_RE = /\b(wavetable|serum|vital|massive|phase\s*plant|pigments|complex\s*synth|full\s*synth|wavetable\s*synth)\b/i;
const SERUM_LIKE_RE = /\b(serum|vital|massive|phase\s*plant|pigments)[\s-]*(like|style|class|type|level|inspired|quality|tier)?\b/i;

export function isComplexSynthRequest(text) {
  const lower = text.toLowerCase();
  if (!COMPLEX_SYNTH_RE.test(lower) && !SERUM_LIKE_RE.test(lower)) return false;
  if (/\b(compressor|comp|limiter|eq|equalizer|reverb|delay|chorus|phaser|flanger|distortion|overdrive|gate|de-?esser|saturator)\b/i.test(lower) &&
      !/\bsynth/i.test(lower)) return false;
  return true;
}

// ── Style pairings — flux knobs (AI-generated) with themed waveform styles ────
// Each pairing uses flux for consistent knob generation across the plugin.
// Waveform style is ONE style per pairing — all waveforms in the plugin match.

const FLUX_PROMPTS = [
  { aesthetic: 'modern-minimal', prompt: 'sleek matte black minimal synth knob with thin white indicator line on dark background, clean modern design, subtle shadow', accent: '#4ecdc4' },
  { aesthetic: 'modern-minimal', prompt: 'smooth dark grey rubber knob with glowing cyan dot indicator, minimalist design, soft matte finish', accent: '#00d4aa' },
  { aesthetic: 'cyberpunk-neon', prompt: 'translucent dark glass knob with glowing neon purple ring, cyberpunk style, dark background, subtle internal glow', accent: '#bb86fc' },
  { aesthetic: 'cyberpunk-neon', prompt: 'black knob with bright magenta LED ring indicator, futuristic electronic design, dark metallic finish', accent: '#ff00aa' },
  { aesthetic: 'vintage-analog', prompt: 'vintage cream colored bakelite knob with brown skirt and pointer tab, warm analog style, 1970s synthesizer', accent: '#d4a76a' },
  { aesthetic: 'pro-studio', prompt: 'brushed aluminum studio knob with silver cap and blue indicator line, professional recording equipment style', accent: '#4488cc' },
  { aesthetic: 'industrial-eurorack', prompt: 'small black hexagonal knob with red indicator notch, industrial eurorack module style, matte metal finish', accent: '#ff4444' },
  { aesthetic: 'classic-hifi', prompt: 'dark wood and brass knob with gold pointer, premium hi-fi amplifier style, warm vintage luxury', accent: '#c8a050' },
];

// Waveform styles per aesthetic — consistent throughout the plugin
const WAVEFORM_THEMES = {
  'modern-minimal': ['3d-wavetable', 'glass-panel'],
  'cyberpunk-neon': ['neon-glow', 'holographic'],
  'vintage-analog': ['retro-crt'],
  'pro-studio': ['glass-panel', 'gradient-fill'],
  'industrial-eurorack': ['led-matrix'],
  'classic-hifi': ['retro-crt', 'glass-panel'],
};

function buildStylePairing(fluxEntry) {
  // Pick ONE waveform style for the whole plugin
  const waveOptions = WAVEFORM_THEMES[fluxEntry.aesthetic] || ['glass-panel'];
  const waveform = pick(waveOptions);
  return {
    aesthetic: fluxEntry.aesthetic,
    knob: 'flux',
    small: 'flux',
    fluxPrompt: fluxEntry.prompt,
    slider: pick(['minimal-track', 'channel-fader', 'led-bar']),
    button: pick(['pill-glow', 'toggle-led', 'rocker']),
    waveform,   // same style for ALL waveforms in plugin
    accent: fluxEntry.accent,
  };
}

// ── Plugin name generation ───────────────────────────────────────────────────

const NAME_PREFIXES = ['Nova', 'Zenith', 'Flux', 'Prism', 'Helix', 'Vortex', 'Nebula', 'Axiom', 'Pulse', 'Drift', 'Aether', 'Stratos', 'Quasar', 'Synapse', 'Orbit', 'Phantom', 'Vertex', 'Echo', 'Nexus', 'Chromatic'];
const NAME_SUFFIXES = ['', '', ' XT', ' Pro', ' II', ' MK2', '', '', ' One', ''];

function generatePluginName(userText) {
  const quoted = userText.match(/["']([^"']+)["']/);
  if (quoted) return quoted[1];
  const namedMatch = userText.match(/(?:called|named)\s+["']?([A-Z][\w\s]{1,30}?)["']?(?:\s|$|,|\.)/i);
  if (namedMatch) return namedMatch[1].trim();
  return pick(NAME_PREFIXES) + pick(NAME_SUFFIXES);
}

// ── Accent color from user keywords or random ────────────────────────────────

function pickAccentColor(text, pairing) {
  const lower = text.toLowerCase();
  if (/\bred\b|crimson|scarlet/i.test(lower)) return '#ff3b3b';
  if (/\bblue\b|cobalt|azure/i.test(lower)) return '#3b82f6';
  if (/\bgreen\b|emerald/i.test(lower)) return '#10b981';
  if (/purple|violet/i.test(lower)) return '#a855f7';
  if (/\borange\b|amber/i.test(lower)) return '#f97316';
  if (/\bpink\b|rose|fuchsia/i.test(lower)) return '#ec4899';
  if (/gold|yellow|brass/i.test(lower)) return '#eab308';
  if (/cyan|teal|aqua/i.test(lower)) return '#06b6d4';
  if (/neon\s*green/i.test(lower)) return '#00ff88';
  return pairing.accent;
}

// ── Pick style pairing from user keywords ────────────────────────────────────

function pickStylePairing(text) {
  const lower = text.toLowerCase();
  const matching = [];
  if (/neon|cyber|glow|futur|synthwave|outrun|vital/i.test(lower))
    matching.push(...FLUX_PROMPTS.filter(p => p.aesthetic === 'cyberpunk-neon'));
  if (/vintage|analog|retro|warm|classic|tube/i.test(lower))
    matching.push(...FLUX_PROMPTS.filter(p => p.aesthetic === 'vintage-analog'));
  if (/studio|pro|clean|professional/i.test(lower))
    matching.push(...FLUX_PROMPTS.filter(p => p.aesthetic === 'pro-studio'));
  if (/industrial|euro|metal|harsh|raw/i.test(lower))
    matching.push(...FLUX_PROMPTS.filter(p => p.aesthetic === 'industrial-eurorack'));
  if (/hifi|hi-fi|luxur|premium/i.test(lower))
    matching.push(...FLUX_PROMPTS.filter(p => p.aesthetic === 'classic-hifi'));
  const fluxEntry = matching.length > 0 ? pick(matching) : pick(FLUX_PROMPTS);
  return buildStylePairing(fluxEntry);
}

// ══════════════════════════════════════════════════════════════════════════════
// SECTION BUILDERS — each returns a section object with randomized components
// ══════════════════════════════════════════════════════════════════════════════

function buildOscSection(label, prefix, s) {
  const fp = s.fluxPrompt;
  const comps = [
    { type: 'waveform', label: `${label} Wave`, waveformStyle: s.waveform },
    { type: 'dropdown', label: `${prefix} Wavetable` },
    { type: 'dropdown', label: `${prefix} Warp Mode` },
    { type: 'knob', label: `${prefix} WT Pos`, svgStyle: s.knob, size: 'large', ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Warp`, svgStyle: s.knob, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Level`, svgStyle: s.knob, ...(fp && { fluxPrompt: fp }) },
  ];
  // Random tuning knobs (5-8 small knobs)
  const tuningPool = shuffle([`${prefix} Unison`, `${prefix} Detune`, `${prefix} Blend`, `${prefix} Phase`, `${prefix} Oct`, `${prefix} Semi`, `${prefix} Fine`, `${prefix} Pan`]);
  const tuningCount = 5 + Math.floor(Math.random() * 4); // 5-8
  for (let i = 0; i < Math.min(tuningCount, tuningPool.length); i++) {
    comps.push({ type: 'knob', label: tuningPool[i], svgStyle: s.small, size: 'small', ...(fp && { fluxPrompt: fp }) });
  }
  comps.push({ type: 'button', label: `${prefix} On`, svgStyle: s.button });
  return { label, weight: 1, layout: 'display-controls', components: comps };
}

function buildFilterSection(label, s) {
  const fp = s.fluxPrompt;
  const comps = [
    { type: 'waveform', label: `${label} Response`, waveformStyle: s.waveform },
    { type: 'dropdown', label: `${label} Type` },
    { type: 'knob', label: 'Cutoff', svgStyle: s.knob, size: 'large', ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: 'Resonance', svgStyle: s.knob, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: 'Drive', svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: 'Key Track', svgStyle: s.small, size: 'small', ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: 'Env Amt', svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: 'Filt Mix', svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
  ];
  const extraPool = shuffle(['Filt Vel', 'Filt Fat', 'Slope', 'Morph', 'Filt Pan', 'Spread', 'Filt Mod']);
  const extraCount = 3 + Math.floor(Math.random() * 3);
  for (let i = 0; i < Math.min(extraCount, extraPool.length); i++) {
    comps.push({ type: 'knob', label: extraPool[i], svgStyle: s.small, size: 'small', ...(fp && { fluxPrompt: fp }) });
  }
  if (coinFlip(0.6)) comps.push({ type: 'dropdown', label: 'Route' });
  comps.push({ type: 'button', label: `${label} On`, svgStyle: s.button });
  return { label, weight: 1, layout: 'display-controls', components: comps };
}

function buildEnvelopeSection(label, prefix, s) {
  const fp = s.fluxPrompt;
  const comps = [
    { type: 'adsr', label: `${label}` },
    { type: 'knob', label: `${prefix} Atk`, svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Dcy`, svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Sus`, svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Rel`, svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Amt`, svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
  ];
  const extraPool = shuffle([`${prefix} Hold`, `${prefix} Crv`, `${prefix} Vel`, `${prefix} Slope`, `${prefix} Delay`, `${prefix} Depth`, `${prefix} Loop`]);
  const extraCount = 3 + Math.floor(Math.random() * 3);
  for (let i = 0; i < Math.min(extraCount, extraPool.length); i++) {
    comps.push({ type: 'knob', label: extraPool[i], svgStyle: s.small, size: 'small', ...(fp && { fluxPrompt: fp }) });
  }
  if (coinFlip(0.5)) comps.push({ type: 'dropdown', label: `${prefix} Target` });
  comps.push({ type: 'button', label: `${prefix} On`, svgStyle: s.button });
  return { label, weight: 1, layout: 'display-controls', components: comps };
}

function buildLFOSection(label, prefix, s) {
  const fp = s.fluxPrompt;
  const comps = [
    { type: 'waveform', label: `${label} Shape`, waveformStyle: s.waveform },
    { type: 'dropdown', label: `${prefix} Wave` },
    { type: 'knob', label: `${prefix} Rate`, svgStyle: s.knob, size: 'large', ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: `${prefix} Depth`, svgStyle: s.knob, ...(fp && { fluxPrompt: fp }) },
  ];
  const extraPool = shuffle([`${prefix} Phase`, `${prefix} Smooth`, `${prefix} Attack`, `${prefix} Delay`, `${prefix} Fade`, `${prefix} Offset`, `${prefix} Slew`, `${prefix} Steps`]);
  const extraCount = 4 + Math.floor(Math.random() * 3);
  for (let i = 0; i < Math.min(extraCount, extraPool.length); i++) {
    comps.push({ type: 'knob', label: extraPool[i], svgStyle: s.small, size: 'small', ...(fp && { fluxPrompt: fp }) });
  }
  if (coinFlip(0.6)) comps.push({ type: 'dropdown', label: `${prefix} Target` });
  comps.push({ type: 'button', label: `${prefix} Sync`, svgStyle: s.button });
  if (coinFlip(0.6)) comps.push({ type: 'button', label: `${prefix} Retrig`, svgStyle: s.button });
  return { label, weight: 1, layout: 'display-controls', components: comps };
}

function buildFxSection(label, prefix, s) {
  const fp = s.fluxPrompt;
  const comps = [
    { type: 'waveform', label: `${label} View`, waveformStyle: s.waveform },
  ];
  if (coinFlip(0.6)) comps.push({ type: 'dropdown', label: `${prefix} Mode` });
  comps.push({ type: 'knob', label: `${prefix} ${pick(['Amount', 'Drive', 'Size', 'Time', 'Rate'])}`, svgStyle: s.knob, size: 'large', ...(fp && { fluxPrompt: fp }) });
  const fxKnobPool = shuffle([`${prefix} Mix`, `${prefix} Tone`, `${prefix} Damp`, `${prefix} Width`, `${prefix} Feed`, `${prefix} Depth`, `${prefix} Mod`, `${prefix} Pre`, `${prefix} Hi-Cut`, `${prefix} Lo-Cut`, `${prefix} Decay`]);
  const fxKnobCount = 5 + Math.floor(Math.random() * 4);
  for (let i = 0; i < Math.min(fxKnobCount, fxKnobPool.length); i++) {
    comps.push({ type: 'knob', label: fxKnobPool[i], svgStyle: s.small, size: coinFlip(0.3) ? 'small' : undefined, ...(fp && { fluxPrompt: fp }) });
  }
  comps.push({ type: 'button', label: `${prefix} On`, svgStyle: s.button });
  return { label, weight: 1, layout: 'display-controls', components: comps };
}

function buildMixerSection(s) {
  const fp = s.fluxPrompt;
  const comps = [
    { type: 'slider', label: 'Osc A', svgStyle: s.slider },
    { type: 'slider', label: 'Osc B', svgStyle: s.slider },
  ];
  if (coinFlip(0.6)) comps.push({ type: 'slider', label: 'Sub', svgStyle: s.slider });
  if (coinFlip(0.5)) comps.push({ type: 'slider', label: 'Noise', svgStyle: s.slider });
  comps.push(
    { type: 'knob', label: 'Width', svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'knob', label: 'Glide', svgStyle: s.small, ...(fp && { fluxPrompt: fp }) },
    { type: 'meter', label: 'Level', svgStyle: 'led-bar' },
  );
  if (coinFlip(0.4)) comps.push({ type: 'knob', label: 'Voices', svgStyle: s.small, size: 'small', ...(fp && { fluxPrompt: fp }) });
  return { label: 'MIXER', weight: 1, components: comps };
}

function buildMasterStrip(s) {
  const fp = s.fluxPrompt;
  const macroCount = pick([3, 4, 4, 4, 5]);
  const comps = [];
  for (let i = 1; i <= macroCount; i++) {
    comps.push({ type: 'knob', label: `Macro ${i}`, svgStyle: s.knob, ...(fp && { fluxPrompt: fp }) });
  }
  comps.push(
    { type: 'slider', label: 'Volume', svgStyle: s.slider },
    { type: 'meter', label: 'Output', svgStyle: 'led-bar' },
  );
  if (coinFlip(0.4)) comps.push({ type: 'button', label: 'Bypass', svgStyle: s.button });
  return { label: 'MASTER', position: 'bottom', components: comps };
}

// ══════════════════════════════════════════════════════════════════════════════
// TAB ARRANGEMENT VARIATIONS
// ══════════════════════════════════════════════════════════════════════════════

function arrangement3Tab(s) {
  // Serum-style: OSC | FX | MATRIX
  const oscTab = {
    label: pick(['OSC', 'OSCILLATORS', 'SYNTH']),
    sections: [
      buildOscSection('OSC A', 'A', s),
      buildOscSection('OSC B', 'B', s),
    ],
  };
  const fxNames = shuffle(['REVERB', 'DELAY', 'CHORUS', 'PHASER', 'DISTORTION', 'FLANGER', 'COMP']);
  const fxPrefixes = { REVERB: 'Rev', DELAY: 'Dly', CHORUS: 'Ch', PHASER: 'Ph', DISTORTION: 'Dist', FLANGER: 'Fl', COMP: 'Cmp' };
  const fxTab = {
    label: pick(['FX', 'EFFECTS']),
    sections: fxNames.slice(0, pick([3, 4])).map((name) =>
      buildFxSection(name, fxPrefixes[name], s)
    ),
  };
  const matrixTab = {
    label: pick(['MATRIX', 'MOD', 'MODULATION']),
    sections: [
      buildLFOSection('LFO 1', 'LFO1', s),
      buildLFOSection('LFO 2', 'LFO2', s),
      ...(coinFlip(0.4) ? [buildMixerSection(s)] : []),
    ],
  };
  return [oscTab, fxTab, matrixTab];
}

function arrangement4Tab(s) {
  // Vital-style: OSC | FILTER & ENV | FX | MOD
  const oscCount = coinFlip(0.3) ? 3 : 2;
  const oscSections = [];
  const oscLabels = oscCount === 3 ? ['OSC 1', 'OSC 2', 'OSC 3'] : ['OSC A', 'OSC B'];
  const oscPrefixes = oscCount === 3 ? ['O1', 'O2', 'O3'] : ['A', 'B'];
  for (let i = 0; i < oscCount; i++) {
    oscSections.push(buildOscSection(oscLabels[i], oscPrefixes[i], s));
  }

  const oscTab = { label: pick(['OSC', 'OSCILLATORS']), sections: oscSections };

  const filterEnvTab = {
    label: pick(['FILTER', 'FILTER & ENV', 'SHAPE']),
    sections: [
      buildFilterSection('FILTER', s),
      buildEnvelopeSection('AMP ENV', 'Amp', s),
      ...(coinFlip(0.5) ? [buildEnvelopeSection('MOD ENV', 'Mod', s)] : []),
    ],
  };

  const fxNames = shuffle(['REVERB', 'DELAY', 'CHORUS', 'DISTORTION', 'PHASER', 'COMP', 'EQ']);
  const fxPrefixes = { REVERB: 'Rev', DELAY: 'Dly', CHORUS: 'Ch', DISTORTION: 'Dist', PHASER: 'Ph', COMP: 'Cmp', EQ: 'EQ' };
  const fxTab = {
    label: pick(['FX', 'EFFECTS']),
    sections: fxNames.slice(0, pick([3, 4])).map((name) =>
      buildFxSection(name, fxPrefixes[name], s)
    ),
  };

  const modTab = {
    label: pick(['MOD', 'MODULATION', 'MATRIX']),
    sections: [
      buildLFOSection('LFO 1', 'LFO1', s),
      buildLFOSection('LFO 2', 'LFO2', s),
      ...(coinFlip(0.5) ? [buildLFOSection('LFO 3', 'LFO3', s)] : []),
    ],
  };

  return [oscTab, filterEnvTab, fxTab, modTab];
}

function arrangement5Tab(s) {
  // Workstation-style: OSC | FILTER | MOD | FX | GLOBAL
  const oscTab = {
    label: 'OSC',
    sections: [
      buildOscSection('OSC A', 'A', s),
      buildOscSection('OSC B', 'B', s),
    ],
  };

  const filterTab = {
    label: 'FILTER',
    sections: [
      buildFilterSection('FILTER 1', s),
      buildFilterSection('FILTER 2', s),
    ],
  };

  const modTab = {
    label: 'MOD',
    sections: [
      buildEnvelopeSection('AMP ENV', 'Amp', s),
      buildEnvelopeSection('MOD ENV', 'Mod', s),
      buildLFOSection('LFO 1', 'LFO1', s),
      buildLFOSection('LFO 2', 'LFO2', s),
    ],
  };

  const fxNames = shuffle(['REVERB', 'DELAY', 'CHORUS', 'DISTORTION', 'PHASER', 'COMP']);
  const fxPrefixes = { REVERB: 'Rev', DELAY: 'Dly', CHORUS: 'Ch', DISTORTION: 'Dist', PHASER: 'Ph', COMP: 'Cmp' };
  const fxTab = {
    label: 'FX',
    sections: fxNames.slice(0, pick([3, 4])).map((name) =>
      buildFxSection(name, fxPrefixes[name], s)
    ),
  };

  const globalTab = {
    label: pick(['GLOBAL', 'VOICE', 'SETUP']),
    sections: [
      buildMixerSection(s),
      buildLFOSection('LFO 3', 'LFO3', s),
    ],
  };

  return [oscTab, filterTab, modTab, fxTab, globalTab];
}

// ══════════════════════════════════════════════════════════════════════════════
// MAIN BUILDER — produces a unique brief each time
// ══════════════════════════════════════════════════════════════════════════════

export function buildComplexSynthBrief(userMessage) {
  const style = pickStylePairing(userMessage);
  const accentColor = pickAccentColor(userMessage, style);
  const pluginName = generatePluginName(userMessage);
  const themeObj = generateTheme(accentColor, style.aesthetic);

  // Pick random tab arrangement
  const arrangementFn = pick([arrangement3Tab, arrangement4Tab, arrangement4Tab, arrangement5Tab]);
  const tabs = arrangementFn(style);

  return {
    pluginName,
    width: 1000,
    height: 560,
    aesthetic: style.aesthetic,
    accentColor,
    bgColor: themeObj.bgColor,
    titleBarColor: themeObj.titleBarColor,
    backgroundPrompt: themeObj.dallePrompt,
    layout: 'tabbed',
    tabs,
    persistentSections: [buildMasterStrip(style)],
  };
}

// ── Count total components in a brief ────────────────────────────────────────

export function countBriefComponents(brief) {
  let count = 0;
  if (brief.tabs) {
    for (const tab of brief.tabs) {
      for (const sec of (tab.sections || [])) {
        count += (sec.components || []).length;
      }
    }
  }
  if (brief.sections) {
    for (const sec of brief.sections) {
      count += (sec.components || []).length;
    }
  }
  if (brief.persistentSections) {
    for (const sec of brief.persistentSections) {
      count += (sec.components || []).length;
    }
  }
  return count;
}
