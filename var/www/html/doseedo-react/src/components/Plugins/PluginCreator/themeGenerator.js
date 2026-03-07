/**
 * Algorithmic Theme Generator
 * Derives complete plugin themes from a seed color + aesthetic category
 * using color theory (HSL manipulation, harmony rules).
 */

// ── HSL ↔ Hex conversions ────────────────────────────────────────────────────

function hexToRgb(hex) {
  hex = hex.replace('#', '');
  if (hex.length === 3) hex = hex[0]+hex[0]+hex[1]+hex[1]+hex[2]+hex[2];
  const n = parseInt(hex, 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function rgbToHex(r, g, b) {
  const clamp = v => Math.max(0, Math.min(255, Math.round(v)));
  return '#' + [clamp(r), clamp(g), clamp(b)].map(v => v.toString(16).padStart(2, '0')).join('');
}

function rgbToHsl(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const l = (max + min) / 2;
  if (max === min) return { h: 0, s: 0, l };
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  let h;
  if (max === r) h = ((g - b) / d + (g < b ? 6 : 0)) / 6;
  else if (max === g) h = ((b - r) / d + 2) / 6;
  else h = ((r - g) / d + 4) / 6;
  return { h, s, l };
}

function hslToRgb(h, s, l) {
  if (s === 0) { const v = Math.round(l * 255); return { r: v, g: v, b: v }; }
  const hue2rgb = (p, q, t) => {
    if (t < 0) t += 1; if (t > 1) t -= 1;
    if (t < 1/6) return p + (q - p) * 6 * t;
    if (t < 1/2) return q;
    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
    return p;
  };
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  return {
    r: Math.round(hue2rgb(p, q, h + 1/3) * 255),
    g: Math.round(hue2rgb(p, q, h) * 255),
    b: Math.round(hue2rgb(p, q, h - 1/3) * 255),
  };
}

export function hexToHsl(hex) {
  const { r, g, b } = hexToRgb(hex);
  return rgbToHsl(r, g, b);
}

export function hslToHex(h, s, l) {
  const { r, g, b } = hslToRgb(h, s, l);
  return rgbToHex(r, g, b);
}

// ── Color manipulation ───────────────────────────────────────────────────────

export function adjustLightness(hex, amount) {
  const hsl = hexToHsl(hex);
  return hslToHex(hsl.h, hsl.s, Math.max(0, Math.min(1, hsl.l + amount)));
}

export function adjustSaturation(hex, amount) {
  const hsl = hexToHsl(hex);
  return hslToHex(hsl.h, Math.max(0, Math.min(1, hsl.s + amount)), hsl.l);
}

export function complementary(hex) {
  const hsl = hexToHsl(hex);
  return hslToHex((hsl.h + 0.5) % 1, hsl.s, hsl.l);
}

export function analogous(hex, offset = 1/12) {
  const hsl = hexToHsl(hex);
  return [
    hslToHex((hsl.h - offset + 1) % 1, hsl.s, hsl.l),
    hex,
    hslToHex((hsl.h + offset) % 1, hsl.s, hsl.l),
  ];
}

export function triadic(hex) {
  const hsl = hexToHsl(hex);
  return [
    hex,
    hslToHex((hsl.h + 1/3) % 1, hsl.s, hsl.l),
    hslToHex((hsl.h + 2/3) % 1, hsl.s, hsl.l),
  ];
}

export function splitComplementary(hex) {
  const hsl = hexToHsl(hex);
  return [
    hex,
    hslToHex((hsl.h + 5/12) % 1, hsl.s, hsl.l),
    hslToHex((hsl.h + 7/12) % 1, hsl.s, hsl.l),
  ];
}

function hexToRgba(hex, alpha) {
  const { r, g, b } = hexToRgb(hex);
  return `rgba(${r},${g},${b},${alpha})`;
}

// ── Aesthetic categories ─────────────────────────────────────────────────────

export const AESTHETIC_CATEGORIES = {
  'vintage-analog': {
    label: 'Vintage Analog',
    knobSvgStyle: 'moog-pointer', sliderSvgStyle: 'vintage-slot', buttonSvgStyle: 'moog-rocker',
    bgLightness: 0.06, bgSaturation: 0.15, bgHueShift: 0,
    textLightness: 0.90, textSaturation: 0.12,
    palette: 'warm',
  },
  'pro-studio': {
    label: 'Pro Studio',
    knobSvgStyle: 'dome-line', sliderSvgStyle: 'channel-fader', buttonSvgStyle: 'toggle-led',
    bgLightness: 0.12, bgSaturation: 0.10, bgHueShift: 0.6,
    textLightness: 0.82, textSaturation: 0.05,
    palette: 'cool',
  },
  'modern-minimal': {
    label: 'Modern Minimal',
    knobSvgStyle: 'soft-rubber', sliderSvgStyle: 'minimal-track', buttonSvgStyle: 'pill-glow',
    bgLightness: 0.10, bgSaturation: 0.08, bgHueShift: 0,
    textLightness: 0.85, textSaturation: 0.05,
    palette: 'neutral',
  },
  'cyberpunk-neon': {
    label: 'Cyberpunk / Neon',
    knobSvgStyle: 'glass-ring', sliderSvgStyle: 'led-bar', buttonSvgStyle: 'pill-glow',
    bgLightness: 0.04, bgSaturation: 0.15, bgHueShift: 0.7,
    textLightness: 0.80, textSaturation: 0.30,
    palette: 'vivid',
  },
  'industrial-eurorack': {
    label: 'Industrial / Eurorack',
    knobSvgStyle: 'hex-bolt', sliderSvgStyle: 'slot-thumb', buttonSvgStyle: 'rocker',
    bgLightness: 0.08, bgSaturation: 0.05, bgHueShift: 0,
    textLightness: 0.65, textSaturation: 0.05,
    palette: 'desaturated',
  },
  'classic-hifi': {
    label: 'Classic Hi-Fi',
    knobSvgStyle: 'bakelite', sliderSvgStyle: 'channel-fader', buttonSvgStyle: 'rocker',
    bgLightness: 0.08, bgSaturation: 0.20, bgHueShift: 0.08,
    textLightness: 0.85, textSaturation: 0.10,
    palette: 'warm',
  },
};

// ── Theme generation ─────────────────────────────────────────────────────────

/**
 * Generate a full plugin theme from accent color + aesthetic.
 * Returns an object compatible with the PLUGIN_THEMES format in themes.js.
 */
export function generateTheme(accentColor, aesthetic = 'pro-studio') {
  const cat = AESTHETIC_CATEGORIES[aesthetic] || AESTHETIC_CATEGORIES['pro-studio'];
  const accentHsl = hexToHsl(accentColor);

  // Background: very dark, tinted toward the accent hue
  const bgHue = (accentHsl.h + cat.bgHueShift) % 1;
  const bgColor = hslToHex(bgHue, cat.bgSaturation, cat.bgLightness);
  const titleBarColor = hslToHex(bgHue, cat.bgSaturation, cat.bgLightness + 0.05);

  // Text: light, slightly tinted
  const textColor = hslToHex(accentHsl.h, cat.textSaturation, cat.textLightness);

  // Panel colors — visible backgrounds with clear borders for section grouping
  const panelColor = hexToRgba(accentColor, 0.16);
  const panelBorder = hexToRgba(accentColor, 0.40);

  // Tab bar colors for tabbed layouts
  const tabActiveColor = accentColor;
  const tabInactiveColor = hexToRgba(accentColor, 0.12);
  const tabBarBg = hslToHex(bgHue, cat.bgSaturation, cat.bgLightness + 0.03);

  return {
    id: `gen-${aesthetic}-${accentColor.replace('#', '')}`,
    name: `${cat.label}`,
    generated: true,
    bgColor,
    titleBarColor,
    accentColor,
    textColor,
    panelColor,
    panelBorder,
    tabActiveColor,
    tabInactiveColor,
    tabBarBg,
    knobSvgStyle: cat.knobSvgStyle,
    sliderSvgStyle: cat.sliderSvgStyle,
    buttonSvgStyle: cat.buttonSvgStyle,
    dallePrompt: generateDallePrompt(aesthetic, accentColor),
  };
}

/**
 * Generate a full color palette from a seed color and harmony type.
 */
export function generatePalette(seedColor, harmony = 'complementary') {
  const hsl = hexToHsl(seedColor);
  let colors;
  switch (harmony) {
    case 'analogous': colors = analogous(seedColor); break;
    case 'triadic': colors = triadic(seedColor); break;
    case 'split-complementary': colors = splitComplementary(seedColor); break;
    default: colors = [seedColor, complementary(seedColor)];
  }
  return {
    primary: colors[0],
    secondary: colors[1] || colors[0],
    tertiary: colors[2] || colors[1] || colors[0],
    bg: hslToHex(hsl.h, 0.15, 0.08),
    titleBar: hslToHex(hsl.h, 0.15, 0.13),
    text: hslToHex(hsl.h, 0.08, 0.85),
    panel: hexToRgba(seedColor, 0.04),
    panelBorder: hexToRgba(seedColor, 0.12),
  };
}

/**
 * Generate 6 theme variants (one per aesthetic) from a single accent color.
 */
export function generateThemeVariants(accentColor) {
  return Object.keys(AESTHETIC_CATEGORIES).map(key =>
    generateTheme(accentColor, key)
  );
}

// ── DALL-E prompt generation per aesthetic ────────────────────────────────────

function generateDallePrompt(aesthetic, accentColor) {
  const hsl = hexToHsl(accentColor);
  const warmCool = hsl.h < 0.17 || hsl.h > 0.92 ? 'warm' : hsl.h < 0.5 ? 'neutral' : 'cool';

  // IMPORTANT: These prompts describe ONLY material textures — no mention of
  // instruments, panels, faceplates, controls, or anything electronic.
  // DALL-E 3 will render synths/knobs if it sees those words.
  const prompts = {
    'vintage-analog': `Top-down macro photograph of dark ${warmCool}-toned walnut wood grain surface, deeply aged natural hardwood with rich amber patina and grain variation, four small brass Phillips-head screws near corners embedded in wood, thick satin lacquer finish with subtle depth, dramatic ${warmCool} brown tones with deep shadows in grain, photorealistic 4k detail, flat lay product photography lighting, shallow depth of field on texture`,
    'pro-studio': `Top-down macro photograph of premium brushed steel metal surface, blue-gray tinted precision-machined industrial steel with fine linear brush marks, subtle anodized coating with very faint blue tint, thin recessed channel lines creating subtle rectangular zone divisions, professional aerospace-grade metal finish, photorealistic 4k detail, diffused studio lighting with soft reflections`,
    'modern-minimal': `Top-down photograph of matte dark charcoal aluminum surface with extremely subtle metallic grain, deep space gray anodized metal with hairline finish, barely visible recessed division lines creating clean zones, premium Apple-style industrial design material, photorealistic 4k detail, even soft diffused lighting, near-black sophisticated surface`,
    'cyberpunk-neon': `Top-down photograph of matte black carbon fiber composite surface, visible tight hexagonal weave pattern with subtle depth, very faint electric cyan and magenta circuit trace lines etched near edges, frosted dark glass inlay strip across center, dark technology material with subtle iridescence, photorealistic 4k detail, moody underlighting`,
    'industrial-eurorack': `Top-down photograph of dark gunmetal anodized aluminum surface, heavy industrial brushed metal with visible cross-hatched machining marks, raw hex bolt heads in corners, matte dark gray to black metal with subtle oil-slick patina, exposed rivet details at edges, photorealistic 4k detail, harsh overhead lighting`,
    'classic-hifi': `Top-down photograph of vintage champagne gold painted steel surface with ${warmCool} aged patina, thick glossy enamel coating with subtle orange-peel texture, thin chrome trim strip at edges, hairline crack patterns in aged enamel finish, retro 1970s appliance aesthetic, photorealistic 4k detail, warm tungsten lighting`,
  };

  return prompts[aesthetic] || prompts['pro-studio'];
}
