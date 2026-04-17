import React, { useState, useRef, useEffect, useCallback, useMemo } from 'react';
import * as chatAPI from '../../../services/chatAPI';
import { buildPluginStream, buildPlugin, buildPluginAutoFix } from '../../../services/pluginProjectsAPI';
import styles from './PluginCreator.module.css';
import { PLUGIN_TEMPLATES } from './templates';
import { PLUGIN_THEMES, applyTheme } from './themes';
import { getDesignDirectorPrompt, DSP_ARCHITECT_PROMPT, getRefinementPrompt, isDesignRequest } from './expertPrompts';
import { computeLayout } from './layoutEngine';
import { generateTheme } from './themeGenerator';
import { isComplexSynthRequest, buildComplexSynthBrief, countBriefComponents } from './complexSynthTemplate';

// ── Style randomizer (creative direction seed) ───────────────────────────────
const STYLE_SEEDS = [
  "Asymmetric layout — off-center hero knob, diagonal visual flow.",
  "Moog Minimoog Model D inspired — wood sides, black face, cream labels.",
  "Brutalist — monochrome, thick borders, raw industrial feel, no rounded corners.",
  "Eurorack faceplate — dense vertical (300x550), tiny packed knobs, patch points, metal texture.",
  "Maximalist — pack every pixel with controls, dense grid of small knobs, info-rich.",
  "Circular/radial — knobs in a ring around a central display, radar screen aesthetic.",
  "Fairchild 670 — cream faceplate, large VU meters, bakelite knobs, vintage tube warmth.",
  "Anime/vaporwave — pastel gradient, pixel art character, Japanese-styled labels, pill buttons.",
  "Minimal white studio — light #f8f8f8 bg, single accent color, hairline borders, generous whitespace.",
  "Roland TR-808 — wide horizontal, colored step buttons, orange/red accents on dark gray.",
  "Art deco — gold on navy, geometric patterns, sunburst gradients, ornate borders.",
  "Sci-fi holographic — dark bg, cyan wireframe panels, floating knobs, HUD meters.",
  "1970s analog console — olive/brown/cream, wood grain bg, chicken-head knobs, warm LEDs.",
  "Lo-fi bitcrushed — pixelated bg, 8-bit color palette, chunky controls, retro game UI.",
  "Live performance — extra-large knobs (100x110), high-contrast, minimal secondary controls.",
  "Neve console — steel blue-gray, red/blue knob accents, horizontal strip layout.",
  "Outrun synthwave — dark purple, neon pink + electric blue, grid bg, chrome knobs.",
];

function getRandomStyleSeed() {
  return STYLE_SEEDS[Math.floor(Math.random() * STYLE_SEEDS.length)];
}

// ── Variation elements — randomized per-request for design diversity ─────────
const TAB_VARIATIONS = [
  { tabs: 3, labels: ['OSC', 'FX', 'MATRIX'], desc: '3-tab synth (Serum-style)' },
  { tabs: 4, labels: ['OSC A', 'OSC B', 'FILTER/ENV', 'FX'], desc: '4-tab with separate oscillators' },
  { tabs: 3, labels: ['OSCILLATORS', 'MODULATION', 'EFFECTS'], desc: '3-tab with combined oscillators' },
  { tabs: 4, labels: ['SYNTH', 'FILTER', 'MOD', 'FX'], desc: '4-tab signal-flow layout' },
  { tabs: 3, labels: ['TONE', 'SHAPE', 'SPACE'], desc: '3-tab creative naming' },
  { tabs: 5, labels: ['OSC', 'FILTER', 'ENV/LFO', 'FX', 'MATRIX'], desc: '5-tab deep synth' },
];

const SECTION_VARIATIONS = [
  'Use display-controls layout for oscillator sections (waveform left, controls right)',
  'Use slider-group layout for mixer sections (vertical faders with detail knobs below)',
  'Include an XY-pad section for performance control',
  'Use a spectrum-analyzer display in the FX/output section',
  'Include ADSR envelope displays alongside envelope controls',
  'Use a large wavetable_3d display as the hero visual element',
  'Include a modulation matrix section with dropdown routing',
];

const AESTHETIC_VARIATIONS = [
  'Dark theme with neon accent — like Vital or Phase Plant',
  'Matte black with warm orange accents — hardware feel',
  'Deep navy with gold/brass accents — premium studio look',
  'Dark gray with cyan/teal accents — modern digital',
  'Black with red accents — aggressive/distortion aesthetic',
  'Charcoal with purple/violet accents — creative/artistic',
  'Dark with green phosphor accents — retro digital/CRT feel',
];

function getVariationDirective(messageText) {
  // Parse user's message for explicit oscillator count
  const oscMatch = messageText && messageText.match(/(\d+)\s*(?:osc(?:illator)?s?)/i);
  let tab;
  if (oscMatch) {
    const oscCount = parseInt(oscMatch[1], 10);
    if (oscCount >= 3) {
      // Group oscillators into tabs — max 5 tabs total to keep UI manageable
      // 2 oscs per tab when there are many, 1 per tab when few
      const letters = 'ABCDEFGH';
      const maxOsc = Math.min(oscCount, 8);
      const oscLabels = [];
      if (maxOsc <= 4) {
        // 1 osc per tab: OSC A, OSC B, OSC C, OSC D + FX
        for (let i = 0; i < maxOsc; i++) oscLabels.push(`OSC ${letters[i]}`);
      } else {
        // Pair oscs into tabs: OSC A/B, OSC C/D, etc.
        for (let i = 0; i < maxOsc; i += 2) {
          if (i + 1 < maxOsc) {
            oscLabels.push(`OSC ${letters[i]}/${letters[i + 1]}`);
          } else {
            oscLabels.push(`OSC ${letters[i]}`);
          }
        }
      }
      oscLabels.push('FX');
      tab = { tabs: oscLabels.length, labels: oscLabels, desc: `${maxOsc}-oscillator layout` };
    } else {
      tab = TAB_VARIATIONS[Math.floor(Math.random() * TAB_VARIATIONS.length)];
    }
  } else {
    tab = TAB_VARIATIONS[Math.floor(Math.random() * TAB_VARIATIONS.length)];
  }
  const sectionHints = [];
  const shuffled = [...SECTION_VARIATIONS].sort(() => Math.random() - 0.5);
  sectionHints.push(shuffled[0], shuffled[1]); // pick 2 random section hints
  const aesthetic = AESTHETIC_VARIATIONS[Math.floor(Math.random() * AESTHETIC_VARIATIONS.length)];
  return {
    tabSuggestion: tab,
    sectionHints,
    aesthetic,
    variationId: Math.random().toString(36).slice(2, 8), // unique ID to prevent caching
  };
}

// ── Synth/plugin design guidance ─────────────────────────────────────────────
// Instead of forcing rigid blueprints, give the LLM flexible structural rules
// and let it use its knowledge of specific synths/hardware.

// Regex for specific hardware unit names (effects processors, compressors, EQs, etc.)
const HARDWARE_KEYWORDS_RE = /\b(la-?2a|1176|distressor|fairchild|pultec|ssl|neve|dbx|api\s*(500|550|560|2500)?|manley|tube-?tech|lexicon|emt|roland\s*(re-?201|space\s*echo|dimension)|boss\s*(ce-?1|dm-?2|re-?20)|mxr|klon|proco|pro\s*co|ehx|electro-?harmonix|teletronix|urei|eventide|tc\s*electronic|universal\s*audio|warm\s*audio|shadow\s*hills|chandler|heritage|drawmer|bss|focusrite\s*isa|avalon|summit|empirical\s*labs|retro\s*instruments|elysia|tegeler|bettermaker|dangerous|great\s*river|a-?designs|inward\s*connections|smart\s*research|alan\s*smart|overstayer|culture\s*vulture|thermionic)\b/i;

// Regex for effect-type keywords (not synths — audio processors)
const EFFECT_TYPE_RE = /\b(compressor|comp|limiter|limiting\s*amplifier|eq|equalizer|equaliser|reverb|delay|echo|chorus|phaser|flanger|distortion|overdrive|fuzz|saturator|saturation|de-?esser|gate|noise\s*gate|transient\s*(shaper|designer)|tremolo|vibrato|exciter|enhancer|pre-?amp|preamp|channel\s*strip|bus\s*comp|master\s*comp|stereo\s*comp|opto\s*comp|vari-?mu|fet\s*comp|vca\s*comp)\b/i;

function isHardwareRequest(text) {
  return HARDWARE_KEYWORDS_RE.test(text);
}

function isEffectRequest(text) {
  return EFFECT_TYPE_RE.test(text);
}

function injectTemplateGuidance(message) {
  const lower = message.toLowerCase();

  // Detect if this is a synth/instrument/effect design request
  const isSynthRequest = /\b(synth|synthesizer|make|create|build|design|generate)\b/i.test(lower) &&
    /\b(synth|plugin|effect|compressor|reverb|delay|eq|distortion|limiter|gate|chorus|flanger|phaser|saturator|amp|preamp|channel\s*strip)\b/i.test(lower);

  // Detect if the user references a SPECIFIC real-world synth/hardware
  const referencesSpecific = /\b(arp\s*2600|moog|minimoog|prophet|jupiter|juno|ob-?x[a-d]?|sh-101|ms-?20|dx-?7|fm-?8|operator|sub\s*37|grandmother|matriarch|model\s*d|poly\s*d|subsequent|1176|la-?2a|pultec|neve|ssl|api|fairchild|distressor|teletronix|dbx|urei|lexicon|eventide|tc\s*electronic|boss|roland|korg|yamaha|nord|access\s*virus|waldorf|dsi|sequential|oberheim|ems|buchla|serge|make\s*noise|mutable|behringer|arturia|native\s*instruments|u-?he|xfer|serum|vital|massive|phase\s*plant|pigments|diva|repro|zebra|surge|helm|tyrell|synth1|sylenth|spire|avenger|omnisphere|kontakt)\b/i;
  const hasSpecificRef = referencesSpecific.test(lower);

  // Detect if this is a specific hardware unit recreation
  const isHardware = isHardwareRequest(lower);

  // Detect if this is an effect (not a synth)
  const isEffect = isEffectRequest(lower);

  if (!isSynthRequest) return message;

  // Build guidance sections based on request type
  let guidance = '\n\n[PLUGIN DESIGN GUIDANCE]\n\n';

  if (isHardware) {
    // ── SPECIFIC HARDWARE RECREATION (most restrictive) ──
    guidance += `CRITICAL HARDWARE RECREATION RULES: This is a SPECIFIC HARDWARE UNIT. You MUST:
1. Use 'horizontal' layout (NOT tabbed, NOT grid) — real hardware has ONE faceplate
2. Use ONLY the controls that exist on the real hardware — check your knowledge of the actual unit
3. Canvas size: 650-800px wide, 280-380px tall (rackmount proportions)
4. Maximum 1-2 sections — most hardware units have a single panel
5. DO NOT add synthesizer controls (oscillators, wavetables, envelopes, mod matrix) — this is NOT a synth
6. Match the real faceplate color and aesthetic from the actual hardware
7. DO NOT use tabbed layout — hardware units do not have tabs
8. DO NOT add controls that don't exist on the real unit — less is more
9. Use the correct control types: if the real unit has buttons, use buttons; if it has knobs, use knobs
10. Include the correct meter type (VU meter, gain reduction meter, LED ladder) if the real unit has one

For example:
- 1176: 4 ratio buttons (20, 12, 8, 4), large VU meter, input/output knobs, attack/release knobs, ~700x320, blue/black faceplate
- LA-2A: Only 2 knobs (Gain, Peak Reduction) + meter + Comp/Limit switch, ~650x300, silver faceplate
- Pultec EQP-1A: Boost/Atten/Bandwidth knobs, frequency selectors, ~750x320, cream/blue faceplate
- Fairchild 670: Large VU meters, threshold/time-constant knobs, bakelite knobs, ~800x380, cream faceplate
- Distressor: Ratio buttons, attack/release knobs, input/output, meter, ~700x300, purple/black faceplate
- SSL Bus Comp: Threshold/ratio/attack/release/makeup knobs, ~700x300, gray/black faceplate
- Neve 1073: EQ bands with frequency selectors, gain trim, HPF, ~400x350, blue/gray faceplate

Recreate the AUTHENTIC layout with ONLY the controls that exist on the real unit.`;
  } else if (hasSpecificRef) {
    guidance += `IMPORTANT: The user references a SPECIFIC real-world synth/hardware. You KNOW what this looks like — use your knowledge of its actual control layout, panel organization, and visual style. Match the real hardware's sections, control types (sliders vs knobs vs buttons), and overall aesthetic. DO NOT substitute a generic wavetable synth layout.

For example:
- ARP 2600: Semi-modular, left-to-right signal flow (VCO→VCF→VCA), sliders NOT knobs, patch points, spring reverb section, distinctive gray/orange aesthetic
- Moog Minimoog: Left-to-right panels (Controllers, Oscillators, Mixer, Filters, Envelopes), iconic pitch/mod wheels, wood endcheeks
- Prophet-5: Two oscillator rows, poly-mod section, filter section, horizontal layout with membrane buttons
- Roland Juno: Slider-heavy, DCO section, HPF/VCF/VCA/ENV/LFO sections in a row, chorus button
- DX7: Algorithm display, operator controls, minimal physical controls, membrane buttons, LCD-style

Recreate the AUTHENTIC layout with the correct control types and organization.`;
  } else if (isEffect) {
    // ── GENERIC EFFECT (not specific hardware, but clearly an effect not a synth) ──
    guidance += `EFFECT PLUGIN RULES: This is an AUDIO EFFECT, NOT a synthesizer. You MUST:
1. Use simple 'horizontal' layout with 1-2 sections — effects are simpler than synths
2. DO NOT add synthesizer controls (oscillators, wavetables, ADSR envelopes, mod matrix, note/pitch controls)
3. Canvas size: 600-800px wide, 280-400px tall
4. Focus on the controls appropriate for this effect type (gain, threshold, ratio, frequency, mix, etc.)
5. Include appropriate metering (VU, gain reduction, spectrum) if relevant
6. Keep it clean and focused — real effect plugins are NOT cluttered with dozens of controls`;
  } else {
    guidance += `Design a layout that matches what this type of plugin should look like. Use your knowledge of real-world hardware and software in this category.`;
  }

  // ── COMPLEX SYNTH (maximize density) ──
  const isSynth = /\b(synth|synthesizer|wavetable|serum|vital|massive|moog|prophet|juno|fm\s*synth|subtractive|additive|modular)\b/i.test(lower);
  if (isSynth && !isHardware && !isEffect) {
    guidance += `\nCOMPLEX SYNTHESIZER — MANDATORY STRUCTURE:

You MUST use this EXACT tab/section structure. Do NOT simplify or reduce. Output ALL of these sections with ALL listed components. This is the MINIMUM — you may add more but NEVER less.

Tab "OSC" — 2 sections side by side:
  Section "OSC A" (15 components): wavetable_3d "Wavetable A", dropdown "A Wavetable", dropdown "A Warp Mode", knob "A WT Pos" size=large, knob "A Warp", knob "A Unison" size=small, knob "A Detune" size=small, knob "A Blend" size=small, knob "A Phase" size=small, knob "A Oct" size=small, knob "A Semi" size=small, knob "A Fine" size=small, knob "A Pan" size=small, knob "A Level", button "A On"
  Section "OSC B" (15 components): IDENTICAL structure to OSC A but with "B" prefix

Tab "FILTER" — 2 sections side by side:
  Section "FILTER" (11 components): waveform "Filter Response", dropdown "Filter Type", dropdown "Filter Routing", knob "Cutoff" size=large, knob "Resonance", knob "Drive", knob "Key Track" size=small, knob "Env Amount", knob "Filter Mix", knob "Filter Vel" size=small, button "Filter On"
  Section "ENVELOPES" (10 components): adsr "Amp Envelope", knob "Attack", knob "Decay", knob "Sustain", knob "Release", knob "Env Vel" size=small, knob "Env Amount" size=small, dropdown "Env Curve", button "Env On", waveform "Mod Envelope"

Tab "MOD" — 2 sections side by side:
  Section "LFO 1" (10 components): waveform "LFO 1 Shape", dropdown "LFO 1 Shape Select", dropdown "LFO 1 Target", knob "LFO 1 Rate" size=large, knob "LFO 1 Depth", knob "LFO 1 Phase" size=small, knob "LFO 1 Delay" size=small, button "LFO 1 Sync", button "LFO 1 Retrigger", button "LFO 1 On"
  Section "LFO 2" (10 components): IDENTICAL structure to LFO 1 but with "LFO 2" prefix

Tab "FX" — 4 sections in 2x2 grid:
  Section "REVERB" (7 components): waveform "Reverb Space", knob "Rev Mix", knob "Rev Size", knob "Rev Decay", knob "Rev Damping", knob "Rev Pre-Delay" size=small, button "Rev On"
  Section "DELAY" (7 components): waveform "Delay Time", knob "Dly Time", knob "Dly Feedback", knob "Dly Mix", dropdown "Dly Type", knob "Dly Tone" size=small, button "Dly On"
  Section "CHORUS" (6 components): knob "Chorus Rate", knob "Chorus Depth", knob "Chorus Mix", knob "Chorus Width" size=small, dropdown "Chorus Mode", button "Chorus On"
  Section "DISTORTION" (6 components): knob "Dist Drive", knob "Dist Tone", knob "Dist Mix", dropdown "Dist Type", knob "Dist Output" size=small, button "Dist On"

persistentSections — "MASTER" (7 components): knob "Macro 1", knob "Macro 2", knob "Macro 3", knob "Macro 4", slider "Volume" svgStyle=channel-fader, meter "Output" svgStyle=led-bar, button "Bypass"

TOTAL: ~104 components across 4 tabs. Do NOT reduce this count. Adapt the aesthetic/colors/svgStyles to match the user's request but keep ALL sections and ALL components.\n`;
  }

  guidance += `

STRUCTURAL RULES (apply to ALL designs):
- Use "size": "large" on hero/primary controls — rendered 1.4x bigger
- Use "size": "small" on secondary controls — rendered 0.7x
- Sections with waveform/oscilloscope/meter displays alongside knobs SHOULD use "layout": "display-controls"
- Use "groups" array in sections with 6+ controls for organization: { "name": "Group Name", "items": ["Control1", "Control2"] }
- For complex/tabbed synths (Serum-class), use 1000x560 canvas. For simple effects/pedals, 350-450px tall. Width 800-1100px.
- Use section weights of 1 (equal sizing) — the auto-fit system handles the rest
- Use at least 4 different component types (knobs, sliders, waveforms, buttons, meters, dropdowns)
- Choose the RIGHT control type: sliders for faders/levels, knobs for frequency/time/amount, buttons for on/off/mode, dropdowns for type selectors
- Customize colors, aesthetic, accentColor, and backgroundPrompt to match the referenced hardware's era and style
- For hardware-inspired designs, use appropriate svgStyles (e.g., "chicken-head" knobs for vintage, "minimal-dot" for modern)
- EVERY section in a tabbed layout MUST have a display component (waveform, wavetable_3d, adsr, or spectrum-analyzer) plus 8-14 knobs/controls. Pack panels DENSE — no empty space.`;

  return message + guidance;
}


// ── Expert prompts (cached on module load) ───────────────────────────────────
const DESIGN_DIRECTOR_PROMPT = getDesignDirectorPrompt();
const REFINEMENT_PROMPT = getRefinementPrompt();

// Legacy prompt kept as fallback for edge cases
const MASTER_PROMPT = `You are a world-class audio plugin UI/UX designer AND DSP architect. You think like a product designer at Native Instruments or Arturia — every plugin you create looks like a real, shippable product. You obsess over visual cohesion, materials, lighting, and the relationship between the background and controls.

When a user describes a plugin, design BOTH the visual interface AND the audio processing:
- Complete plugin → output BOTH pluginlang AND dsplang
- Visual/layout changes only → output only pluginlang
- DSP/audio changes only → output only dsplang
- Modify something specific → modify only what they asked

## DESIGN PHILOSOPHY — Think like a hardware designer

Every plugin should look like a REAL physical product photographed from above. Ask yourself:
1. **What material is the faceplate?** (black anodized aluminum, cream-painted steel, brushed titanium, walnut wood)
2. **What era/style?** (1970s analog warmth, modern minimalism, sci-fi, brutalist industrial)
3. **What knobs would this hardware use?** (chicken-head pointers, skirted knobs with numbers, rubberized soft-touch, chrome potentiometers)
4. **How are sections organized?** (silk-screened panel labels, recessed zones, screw holes, ventilation slots)

The background image and controls must feel like ONE cohesive product. The background is the FACEPLATE — the controls sit ON it.

## BACKGROUND IMAGES — The faceplate/panel

The bgImage creates the hardware faceplate. It should look like a TOP-DOWN photograph of an EMPTY control panel:

For hardware recreations (Moog, Neve, 1176, etc.):
- Generate a bird's-eye view of the EMPTY panel/faceplate with appropriate materials, colors, panel sections, and label areas — but NO knobs, NO buttons, NO sliders, NO controls
- Include panel details: screw holes, section dividers, ventilation, silk-screen text areas, brand badge zones
- Example: "Top-down view of empty vintage analog synthesizer faceplate, dark charcoal panel with cream-colored section labels, walnut wood side cheeks, panel mounting screws visible, silk-screened text areas, matte finish, photorealistic, no knobs no buttons no controls no sliders, empty panel ready for controls"

For abstract/modern designs:
- Use material textures with depth: "Dark brushed aluminum control surface with subtle machined grooves, recessed control zones, matte black anodized finish, photorealistic top-down view"

For creative/artistic designs:
- Use atmospheric surfaces: "Deep space nebula surface with subtle grid overlay, dark edges fading to deep purple center, abstract technology aesthetic"

NEVER generate a photo of an actual complete instrument with controls — that creates visual conflict with the SVG components.

## PLUGINLANG (UI) — output inside \`\`\`pluginlang blocks

{
  "pluginConfig": { "name": "Plugin Name", "width": 600, "height": 400, "bgColor": "#1a1a2e", "titleBarColor": "#2d2d4e", "bgImage": {"generate": "DALL-E prompt"} OR {"search": "query"} OR "url" },
  "components": [ ... ],
  "mode": "replace" OR "merge"
}

Component types and default sizes: knob (60x70), slider (30x120), button (70x28), label (100x24), led (12x12), dropdown (120x28), image (80x80), panel (200x150), meter (24x100), waveform (180x60), xy-pad (120x120)

Common props: x, y, width, height, color, label, opacity, rotation, borderRadius, fontSize, zIndex
Panel props: borderColor, bgColor, bgGradient, backdropBlur, boxShadow
Image: {"generate": "prompt"} or {"search": "query"} for auto-generation

## LAYOUT — Think in spatial zones

Organize into logical sections like real hardware:
- **Header**: Plugin name, mode selectors
- **Main section**: Primary controls grouped by function (oscillators, filter, envelope, etc.)
- **Display area**: Meters, waveforms, XY pads
- **Output section**: Mix, volume, master controls

Use panel components to create visual sections (recessed areas, bordered groups, tinted zones). Use label components for section titles. Use image components for decorative elements (brand logos, wood side panels, decorative screws).

VARY layouts: Try asymmetric, radial, vertical eurorack (300x550), ultra-wide (900x350), stacked horizontal strips. Vary plugin sizes. Vary colors. NEVER default to the same blue/purple dark theme.

## CONTROL STYLES — svgStyle field (REQUIRED on ALL knobs, sliders, buttons)

EVERY knob, slider, and button MUST have an "svgStyle" field. The frontend renders beautiful parameterized SVGs automatically from this style name. Do NOT write raw SVG — just pick a style and set colors.

### Knob styles ("svgStyle" values for type "knob"):
- "skirted-pointer" — Classic Moog skirted knob with triangular pointer. Best for: vintage synths, classic analog.
- "chicken-head" — Vintage chicken-head wedge pointer with concentric rings. Best for: 1970s amps, retro, warm analog.
- "dome-line" — Modern studio dome with line indicator, chrome highlights. Best for: Neve/SSL, pro studio, mixing.
- "soft-rubber" — Rubberized matte with dot indicator, crosshatch grip. Best for: modern controllers, clean.
- "chrome-cap" — Shiny chrome center cap with colored line. Best for: polished studio gear, high-end.
- "glass-ring" — Dark glass body with glowing neon ring. Best for: cyberpunk, neon, futuristic, Vital.
- "hex-bolt" — Hexagonal industrial shape. Best for: eurorack, industrial, dense layouts.
- "minimal-dot" — Thin outlined circle with dot indicator. Best for: minimal, Ableton-like, modern.
- "led-ring" — 12 segmented LED dots in arc around dark center. Best for: DJ gear, electronic, encoders.
- "bakelite" — Classic bakelite with cream wedge pointer. Best for: tube amps, 1950s-60s vintage.

### Slider styles ("svgStyle" values for type "slider"):
- "channel-fader" — Console fader with sculpted cap and recessed track. Best for: mixing consoles.
- "slot-thumb" — Recessed slot with rectangular thumb, grip lines. Best for: eurorack, industrial.
- "led-bar" — Vertical LED bar graph segments. Best for: meters, neon, electronic.
- "minimal-track" — Thin line with circle thumb. Best for: minimal, modern.
- "vintage-slot" — Vintage rounded thumb with textured grip. Best for: analog, retro.

### Button styles ("svgStyle" values for type "button"):
- "toggle-led" — Toggle with LED indicator dot that glows when active. Best for: on/off, bypass.
- "rocker" — 3D rocker switch with beveled gradient. Best for: hardware, vintage.
- "momentary" — Round momentary push button. Best for: triggers, taps.
- "pill-glow" — Pill shape with glow border effect. Best for: modern, neon, cyberpunk.
- "vintage-toggle" — Classic lever toggle switch. Best for: vintage, retro.

### Color overrides (ALWAYS include these to match the design):
- "bodyColor" — main body fill (hex). Dark for most styles.
- "indicatorColor" — pointer/LED/indicator color (hex). Should contrast with body.
- "accentColor" — rings, outlines, secondary highlights (hex).

### Examples:

Vintage Moog knob:
{ "type": "knob", "x": 50, "y": 80, "width": 60, "height": 60, "label": "Cutoff", "svgStyle": "chicken-head", "bodyColor": "#2a1f14", "indicatorColor": "#f5deb3", "accentColor": "#5a4a38", "color": "#d4a76a" }

Studio channel strip knob:
{ "type": "knob", "x": 50, "y": 80, "width": 60, "height": 60, "label": "Gain", "svgStyle": "dome-line", "bodyColor": "#666", "indicatorColor": "#ff3333", "accentColor": "#999", "color": "#ff3333" }

Cyberpunk knob:
{ "type": "knob", "x": 50, "y": 80, "width": 60, "height": 60, "label": "Rate", "svgStyle": "glass-ring", "bodyColor": "#0a0a1a", "indicatorColor": "#00ffc8", "accentColor": "#1a3a3a", "color": "#00ffc8" }

Console fader:
{ "type": "slider", "x": 200, "y": 60, "width": 30, "height": 120, "label": "Volume", "svgStyle": "channel-fader", "bodyColor": "#222", "indicatorColor": "#667eea", "accentColor": "#555", "color": "#667eea" }

Bypass button:
{ "type": "button", "x": 300, "y": 10, "width": 70, "height": 28, "label": "Bypass", "svgStyle": "toggle-led", "bodyColor": "#1a1a2e", "indicatorColor": "#4caf50", "accentColor": "#444", "color": "#4caf50" }

### MATCHING STYLE TO AESTHETIC (always follow this):
- Vintage/Analog (Moog, Buchla, ARP): knobs=chicken-head or skirted-pointer or bakelite, sliders=vintage-slot, buttons=vintage-toggle or rocker
- Pro Studio (Neve, SSL, API): knobs=dome-line or chrome-cap, sliders=channel-fader, buttons=toggle-led
- Modern Minimal (Ableton, Serum): knobs=minimal-dot or soft-rubber, sliders=minimal-track, buttons=pill-glow
- Cyberpunk/Neon (Vital, futuristic): knobs=glass-ring or led-ring, sliders=led-bar, buttons=pill-glow
- Eurorack/Industrial: knobs=hex-bolt, sliders=slot-thumb, buttons=rocker
- Classic Hi-Fi (tube amps): knobs=bakelite or skirted-pointer, sliders=channel-fader, buttons=rocker

### Sprites (photorealistic DALL-E components — optional, slow)
For 1-3 hero components that need photorealism, use "sprite" instead of "svgStyle":
{ "type": "knob", "width": 80, "height": 80, "sprite": {"generate": "photorealistic chrome audio knob, top-down, centered, black bg, studio lighting, no text"} }

### Flux-generated components (AI photorealistic, fast, cached)
For distinctive or unusual components, use svgStyle "flux" with a "fluxPrompt" field:
{ "type": "knob", "x": 50, "y": 80, "width": 60, "height": 60, "label": "Cutoff", "svgStyle": "flux", "fluxPrompt": "chrome dome knob with blue LED ring indicator" }
{ "type": "slider", "x": 200, "y": 60, "width": 30, "height": 120, "label": "Mix", "svgStyle": "flux", "fluxPrompt": "brushed steel channel fader with rubber grip" }
{ "type": "button", "x": 300, "y": 10, "width": 70, "height": 28, "label": "Bypass", "svgStyle": "flux", "fluxPrompt": "illuminated red arcade button" }
The fluxPrompt should concisely describe the physical appearance (10-20 words). Use flux for unique or custom hardware styles.

## DSPLANG (Audio DSP) — output inside \`\`\`dsplang blocks

{
  "pluginType": "effect"|"instrument",
  "name": "Plugin Name",
  "parameters": [{ "id": "param_id", "name": "Display Name", "min": 0, "max": 1, "default": 0.5, "skew": 1.0, "unit": "" }],
  "dspChain": [{ "type": "node_type", "id": "unique_id", "params": { "cutoff": "@cutoff_param_id", "resonance": 0.707 } }],
  "routing": { "input": "stereo", "chain": ["id1", "id2"], "output": "stereo" },
  "mode": "replace"
}

DSP nodes and their param keys (use EXACTLY these keys in dspChain params):
- lowpass/highpass/bandpass/notch/allpass: cutoff, resonance
- ladder: cutoff, resonance, mode (LPF24/LPF12/HPF24/HPF12/BPF24/BPF12)
- delay: time, feedback, mix
- ping_pong_delay: time, feedback
- multitap_delay: time, feedback, mix
- comb: time, feedback, mix
- reverb: room_size, damping, wet, dry, width
- chorus: rate, depth, feedback, mix
- phaser: rate, depth, feedback, mix
- flanger: rate, depth, feedback, mix
- tremolo: rate, depth
- lfo: rate, depth
- compressor: threshold, ratio, attack, release
- limiter: threshold, release
- gate: threshold, ratio, attack, release
- expander: threshold, ratio, attack, release
- envelope_follower: attack, release
- gain: gain_db
- overdrive: drive
- saturation: amount
- bitcrusher: bit_depth, rate_reduction
- foldback: threshold
- ring_mod: frequency
- oscillator/wavetable/fm_operator: frequency
- noise: level
- pan: pan
- mix (dry/wet): ratio (alias: mix)
- envelope_adsr: attack, decay, sustain, release
- shelf_low/shelf_high: frequency, gain
- parametric_eq: frequency, q, gain
- waveshaper/convolution/dc_blocker/peak_meter/rms_meter: (no params)

## PARAMETER BINDING — CRITICAL (MUST follow)

Every knob/slider the user can turn MUST be connected to a DSP node via "@param_id" bindings in the dspChain params.

WRONG (knob does nothing — hardcoded value):
  "parameters": [{"id": "cutoff", ...}],
  "dspChain": [{"type": "lowpass", "id": "lp", "params": {"cutoff": 1000}}]

CORRECT (knob controls the filter):
  "parameters": [{"id": "cutoff", ...}],
  "dspChain": [{"type": "lowpass", "id": "lp", "params": {"cutoff": "@cutoff"}}]

Reverb example (ALL params bound):
  "parameters": [{"id": "room_size", "name": "Room Size", "min": 0, "max": 1, "default": 0.5}, {"id": "damping", "name": "Damping", "min": 0, "max": 1, "default": 0.5}, {"id": "wet", "name": "Wet", "min": 0, "max": 1, "default": 0.33}, {"id": "dry", "name": "Dry", "min": 0, "max": 1, "default": 0.67}, {"id": "width", "name": "Width", "min": 0, "max": 1, "default": 1.0}],
  "dspChain": [{"type": "reverb", "id": "reverb1", "params": {"room_size": "@room_size", "damping": "@damping", "wet": "@wet", "dry": "@dry", "width": "@width"}}]

Rules:
- "@param_id" in node params binds to the exposed parameter. The user's knob controls it.
- Use literal numbers ONLY for values that should be fixed (not user-controllable).
- EVERY parameter in the "parameters" array MUST appear as "@param_id" in at least one node's params.
- USE THE EXACT PARAM KEYS listed above. E.g. for reverb use "room_size" NOT "roomSize" or "size".
- Parameter IDs SHOULD match node param keys (e.g. param id "cutoff" → node key "cutoff": "@cutoff").
- For mix/dry-wet: use param "mix" → node params {"ratio": "@mix"} on a "mix" type node.
- EVERY node param that should be user-controllable MUST use "@param_id" binding, NOT a literal value.

Skew: <1 = log (use 0.25 for frequency), 1.0 = linear, >1 = high end resolution.
Instruments: add "midi": { "voices": 8, "pitchBendRange": 2 }.

Advanced patterns: Parallel processing (NY compression), feedback networks (Karplus-Strong), sidechain (envelope_follower → filter), multi-band (crossover filters), modulation matrix (multiple LFOs), creative combos (bitcrusher→reverb, waveshaper→bandpass→delay).

## AUTO-BINDING UI ↔ DSP

When you output BOTH pluginlang and dsplang, make the component labels match DSP parameter names so they auto-bind:
- DSP param: { "id": "cutoff", "name": "Cutoff" } → UI knob: { "label": "Cutoff" }
- DSP param: { "id": "mix", "name": "Mix" } → UI slider: { "label": "Mix" }
ALWAYS match labels to parameter names.

## PLUGIN TYPE — CRITICAL
- ALWAYS include "pluginType" in every dsplang block: "effect", "instrument", or "midi_effect".
- "effect" = audio effect (reverb, delay, compressor, EQ, etc.). Does NOT accept MIDI.
- "instrument" = generates audio from MIDI input (synth, sampler). Accepts MIDI. Add "midi" field.
- "midi_effect" = processes MIDI (arpeggiator, chord generator). Very rare.
- When the user asks to "make a reverb/delay/compressor/EQ/etc." → pluginType MUST be "effect".
- When the user asks to "make a synth/instrument/sampler" → pluginType MUST be "instrument".
- If switching from instrument to effect (or vice versa), explicitly set the new pluginType.

## EDITING RULES
- Partial edits: KEEP all existing components, only change what's asked.
- Full redesign: Replace everything.
- When in doubt, preserve existing components.

## OUTPUT RULES
- Always output pluginlang/dsplang blocks when changes are requested.
- Brief explanation (2-3 sentences) before/after.
- NEVER produce the same layout twice. Every design must be visually distinct.
- When outputting both, put pluginlang FIRST, then dsplang.
- pluginlang FIRST, then dsplang.

## QUALITY CHECKLIST (verify EVERY design before output)
1. Does the background image describe an EMPTY faceplate/panel appropriate to the aesthetic? (Never photos of complete instruments)
2. Does EVERY knob/slider/button have an "svgStyle" field? And matching bodyColor/indicatorColor/accentColor?
3. Is the svgStyle appropriate to the aesthetic? (chicken-head for vintage, dome-line for studio, glass-ring for cyberpunk)
4. Do the color values match the overall design theme? (Warm browns for vintage, cool grays for studio, dark+neon for cyberpunk)
5. Are there section labels, panel groupings, and decorative elements?
6. Is the layout organized into logical zones like real hardware?`;

// ── Parse both block types from AI response ──────────────────────────────────

function parseBlocks(text) {
  const pluginLangBlocks = [];
  const dspLangBlocks = [];
  const designBriefBlocks = [];
  const parseErrors = []; // Track parse errors for better reporting

  const plRegex = /\`\`\`pluginlang\s*([\s\S]*?)\`\`\`/g;
  let match;
  while ((match = plRegex.exec(text)) !== null) {
    try { pluginLangBlocks.push(JSON.parse(match[1].trim())); } catch (e) {
      parseErrors.push({ blockType: 'pluginlang', raw: match[1].trim(), error: e.message });
    }
  }

  const dspRegex = /\`\`\`dsplang\s*([\s\S]*?)\`\`\`/g;
  while ((match = dspRegex.exec(text)) !== null) {
    try { dspLangBlocks.push(JSON.parse(match[1].trim())); } catch (e) {
      parseErrors.push({ blockType: 'dsplang', raw: match[1].trim(), error: e.message });
    }
  }

  const dbRegex = /\`\`\`designbrief\s*([\s\S]*?)\`\`\`/g;
  while ((match = dbRegex.exec(text)) !== null) {
    try { designBriefBlocks.push(JSON.parse(match[1].trim())); } catch (e) {
      parseErrors.push({ blockType: 'designbrief', raw: match[1].trim(), error: e.message });
    }
  }

  return { pluginLangBlocks, dspLangBlocks, designBriefBlocks, parseErrors };
}

/** Render message text with Apply buttons for all block types */
function renderMessageContent(text, onApplyLayout, onApplyDsp, onApplyBoth, onGenerateCode, onApplyDesignBrief) {
  const parts = [];
  // Match all block types
  const regex = /\`\`\`(pluginlang|dsplang|designbrief)\s*([\s\S]*?)\`\`\`/g;
  let lastIdx = 0;
  let blockIdx = 0;
  let match;

  // Collect all blocks first for "Apply Both" detection
  const allBlocks = [];
  const regexScan = /\`\`\`(pluginlang|dsplang|designbrief)\s*([\s\S]*?)\`\`\`/g;
  while ((match = regexScan.exec(text)) !== null) {
    let parsed = null;
    let parseError = null;
    let rawContent = match[2].trim();
    try { parsed = JSON.parse(rawContent); } catch (e) { parseError = e.message; }
    allBlocks.push({ type: match[1], parsed, parseError, rawContent, index: match.index, fullMatch: match[0] });
  }

  const hasUI = allBlocks.some(b => b.type === 'pluginlang' && b.parsed);
  const hasDSP = allBlocks.some(b => b.type === 'dsplang' && b.parsed);
  const hasDesignBrief = allBlocks.some(b => b.type === 'designbrief' && b.parsed);
  const hasBoth = (hasUI || hasDesignBrief) && hasDSP;

  // Now render
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIdx) {
      parts.push(<span key={`t-${blockIdx}`}>{text.slice(lastIdx, match.index)}</span>);
    }
    const blockType = match[1];
    let parsed = null;
    try { parsed = JSON.parse(match[2].trim()); } catch {}

    // Find matching block info (with parseError & rawContent) from allBlocks
    const matchingBlock = allBlocks.find(b => b.index === match.index);

    if (parsed && blockType === 'pluginlang') {
      const compCount = parsed.components?.length || 0;
      const name = parsed.pluginConfig?.name || 'Layout';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div className={styles.pluginlangHeader}>
            <i className="fa-solid fa-puzzle-piece" />
            <span>{name} — {compCount} components</span>
          </div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <button className={styles.applyLayoutBtn} onClick={() => onApplyLayout(parsed)}>
              <i className="fa-solid fa-wand-magic-sparkles" /> Apply UI
            </button>
            {hasBoth && blockIdx === 0 && (
              <button className={styles.applyLayoutBtn} style={{ background: 'linear-gradient(135deg, #667eea, #764ba2)' }}
                onClick={() => {
                  const uiBlock = allBlocks.find(b => b.type === 'pluginlang' && b.parsed);
                  const dspBlock = allBlocks.find(b => b.type === 'dsplang' && b.parsed);
                  if (uiBlock && dspBlock) onApplyBoth(uiBlock.parsed, dspBlock.parsed);
                }}>
                <i className="fa-solid fa-layer-group" /> Apply Both
              </button>
            )}
          </div>
        </div>
      );
    } else if (parsed && blockType === 'dsplang') {
      const nodeCount = parsed.dspChain?.length || 0;
      const paramCount = parsed.parameters?.length || 0;
      const name = parsed.name || 'DSP Config';
      const pType = parsed.pluginType || 'effect';
      const typeIcon = pType === 'instrument' ? 'fa-piano-keyboard' : pType === 'midi_effect' ? 'fa-music' : 'fa-wave-square';
      const typeLabel = pType === 'instrument' ? 'Instrument' : pType === 'midi_effect' ? 'MIDI FX' : 'Effect';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.dsplangBlock || styles.pluginlangBlock}>
          <div className={styles.dsplangHeader || styles.pluginlangHeader}>
            <i className="fa-solid fa-microchip" />
            <span>{name} — {nodeCount} nodes, {paramCount} params</span>
            <span style={{ marginLeft: 'auto', fontSize: 10, padding: '2px 8px', borderRadius: 10, background: pType === 'instrument' ? 'rgba(255,152,0,0.2)' : pType === 'midi_effect' ? 'rgba(156,39,176,0.2)' : 'rgba(76,175,80,0.2)', color: pType === 'instrument' ? '#ffb74d' : pType === 'midi_effect' ? '#ce93d8' : '#81c784' }}>
              <i className={`fa-solid ${typeIcon}`} style={{ marginRight: 4 }} />{typeLabel}
            </span>
          </div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <button className={styles.applyDspBtn || styles.applyLayoutBtn} onClick={() => onApplyDsp(parsed)}>
              <i className="fa-solid fa-check" /> Apply DSP
            </button>
            <button className={styles.generateCodeBtn || styles.applyLayoutBtn} onClick={() => onGenerateCode(parsed)}>
              <i className="fa-solid fa-code" /> Generate Code
            </button>
          </div>
        </div>
      );
    } else if (parsed && blockType === 'designbrief') {
      const sectionCount = parsed.sections?.length || 0;
      const compCount = (parsed.sections || []).reduce((sum, s) => sum + (s.components?.length || 0), 0);
      const name = parsed.pluginName || 'Design';
      const aesthetic = parsed.aesthetic || '';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div className={styles.pluginlangHeader}>
            <i className="fa-solid fa-compass-drafting" />
            <span>{name} — {sectionCount} sections, {compCount} controls</span>
            {aesthetic && (
              <span style={{ marginLeft: 'auto', fontSize: 10, padding: '2px 8px', borderRadius: 10, background: 'rgba(186,156,255,0.15)', color: '#cbb3ff' }}>
                {aesthetic}
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            <button className={styles.applyLayoutBtn} style={{ background: 'linear-gradient(135deg, #667eea, #764ba2)' }}
              onClick={() => onApplyDesignBrief && onApplyDesignBrief(parsed)}>
              <i className="fa-solid fa-wand-magic-sparkles" /> Apply Design
            </button>
            {hasBoth && (
              <button className={styles.applyLayoutBtn} style={{ background: 'linear-gradient(135deg, #4caf50, #2e7d32)' }}
                onClick={() => {
                  if (onApplyDesignBrief) onApplyDesignBrief(parsed);
                  const dspBlock = allBlocks.find(b => b.type === 'dsplang' && b.parsed);
                  if (dspBlock) onApplyDsp(dspBlock.parsed);
                }}>
                <i className="fa-solid fa-layer-group" /> Apply All
              </button>
            )}
          </div>
        </div>
      );
    } else {
      const errMsg = matchingBlock?.parseError || 'Unknown parse error';
      const rawSnippet = matchingBlock?.rawContent
        ? matchingBlock.rawContent.slice(0, 100) + (matchingBlock.rawContent.length > 100 ? '...' : '')
        : '';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div style={{
            padding: '8px 10px', borderRadius: 6,
            background: 'rgba(255,80,60,0.08)', border: '1px solid rgba(255,80,60,0.2)',
          }}>
            <div style={{ color: 'rgba(255,150,100,0.9)', fontSize: 11, marginBottom: 4 }}>
              <i className="fa-solid fa-triangle-exclamation" /> Parse error in {blockType} block
            </div>
            <div style={{ color: 'rgba(255,120,100,0.8)', fontSize: 10, fontFamily: 'monospace', marginBottom: 2 }}>
              {errMsg}
            </div>
            {rawSnippet && (
              <div style={{
                color: 'rgba(255,255,255,0.4)', fontSize: 9, fontFamily: 'monospace',
                padding: '4px 6px', background: 'rgba(0,0,0,0.2)', borderRadius: 3,
                whiteSpace: 'pre-wrap', wordBreak: 'break-all', marginTop: 4,
              }}>
                {rawSnippet}
              </div>
            )}
          </div>
        </div>
      );
    }
    lastIdx = match.index + match[0].length;
    blockIdx++;
  }

  if (lastIdx < text.length) {
    parts.push(<span key="end">{text.slice(lastIdx)}</span>);
  }
  return parts.length > 0 ? parts : text;
}

// ── Main Component ───────────────────────────────────────────────────────────

const MasterChat = ({
  pluginConfig, components, dspConfig,
  onApplyLayout, onApplyDsp, onOpenImageBrowser,
  chatHistory, onChatHistoryChange,
  activeTheme, onApplyTheme,
}) => {
  const defaultMsg = {
    role: 'assistant',
    content: "I'm your plugin designer — tell me what you want to build and I'll create both the UI and DSP. Or pick a template below to start fast.\n\nTry: \"Design a warm analog tape delay with ping-pong mode, a vinyl saturator, and a lo-fi vibe\"",
    timestamp: new Date().toISOString(),
  };

  const [messages, setMessages] = useState(chatHistory?.length > 0 ? chatHistory : [defaultMsg]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [referenceImage, setReferenceImage] = useState(null);
  const [styleSeed, setStyleSeed] = useState(() => getRandomStyleSeed());
  const [showTemplates, setShowTemplates] = useState(() => !components || components.length === 0);
  const [showThemes, setShowThemes] = useState(false);
  const [codePreview, setCodePreview] = useState(null);
  const [activeCodeFile, setActiveCodeFile] = useState(null);
  const [generatingCode, setGeneratingCode] = useState(false);
  const [buildingPlugin, setBuildingPlugin] = useState(false);
  const [buildLogs, setBuildLogs] = useState([]);
  const [buildStage, setBuildStage] = useState('');
  const [revertToast, setRevertToast] = useState(null); // { type: 'ui'|'dsp'|'both', timer }
  const [autoApply, setAutoApply] = useState(true);
  const autoApplyRef = useRef(true);
  autoApplyRef.current = autoApply;
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  // Restore chat history from localStorage on mount if chatHistory prop is empty
  const CHAT_STORAGE_KEY = 'plugin_creator_chat_history';
  const hasRestoredRef = useRef(false);
  useEffect(() => {
    if (hasRestoredRef.current) return;
    hasRestoredRef.current = true;
    if (chatHistory && chatHistory.length > 0) return; // parent already has history
    try {
      const saved = localStorage.getItem(CHAT_STORAGE_KEY);
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setMessages(parsed);
        }
      }
    } catch (e) {
      console.warn('[MasterChat] Failed to restore chat from localStorage:', e);
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Reset messages when parent signals a new project (chatHistory goes empty)
  const prevChatHistoryLenRef = useRef(chatHistory?.length || 0);
  useEffect(() => {
    const prevLen = prevChatHistoryLenRef.current;
    const curLen = chatHistory?.length || 0;
    prevChatHistoryLenRef.current = curLen;
    // Parent reset chatHistory to [] — reset messages to default
    if (prevLen > 0 && curLen === 0) {
      console.log('[MasterChat] Parent reset chatHistory — clearing messages');
      setMessages([defaultMsg]);
      setReferenceImage(null);
      setError(null);
      setIsLoading(false);
      setShowTemplates(true);
    }
  }, [chatHistory]); // eslint-disable-line react-hooks/exhaustive-deps

  // Persist chat history to localStorage and parent (limit to last 50 messages)
  useEffect(() => {
    if (messages.length > 1) {
      const toSave = messages.slice(-50);
      try {
        localStorage.setItem(CHAT_STORAGE_KEY, JSON.stringify(toSave));
      } catch (e) {
        console.warn('[MasterChat] Failed to save chat to localStorage:', e);
      }
      if (onChatHistoryChange) {
        onChatHistoryChange(messages);
      }
    }
  }, [messages]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => { inputRef.current?.focus(); }, []);

  // Hide templates once user starts chatting
  useEffect(() => {
    if (messages.length > 2 || components.length > 0) setShowTemplates(false);
  }, [messages.length, components.length]);

  const buildContext = useCallback(() => {
    const ctx = {
      pluginName: pluginConfig.name,
      canvasSize: `${pluginConfig.width}x${pluginConfig.height}`,
      bgColor: pluginConfig.bgColor,
      componentCount: components.length,
      creativeDirection: styleSeed,
    };
    if (activeTheme) {
      ctx.activeTheme = {
        name: activeTheme.name,
        accentColor: activeTheme.accentColor,
        knobStyle: activeTheme.knobStyle,
        dallePrompt: activeTheme.dallePrompt,
      };
    }
    if (components.length > 0) {
      ctx.components = components.map(c => ({
        type: c.type, label: c.label, x: c.x, y: c.y, width: c.width, height: c.height,
        ...(c.svgStyle ? { svgStyle: c.svgStyle } : {}),
        ...(c.color ? { color: c.color } : {}),
      }));
      // Group components by containing panel (section structure)
      const panels = components.filter(c => c.type === 'panel');
      const assigned = new Set();
      const sectionList = panels.map(p => {
        const contained = components.filter(c => {
          if (c.type === 'panel' || assigned.has(c.id)) return false;
          const ccx = c.x + (c.width || 0) / 2;
          const ccy = c.y + (c.height || 0) / 2;
          return ccx >= p.x && ccx <= p.x + p.width && ccy >= p.y && ccy <= p.y + p.height;
        });
        contained.forEach(c => assigned.add(c.id));
        return {
          name: p.label || 'Section',
          panel: { x: p.x, y: p.y, width: p.width, height: p.height },
          components: contained,
        };
      });
      ctx.sections = sectionList.map(s => ({
        name: s.name,
        panel: s.panel,
        components: s.components.map(c => ({
          type: c.type, label: c.label, x: c.x, y: c.y, width: c.width, height: c.height,
        })),
      }));

      // Reconstruct a designbrief representation so the AI can modify it for structural edits
      const hasWaveformOrXY = (comps) => comps.some(c => c.type === 'waveform' || c.type === 'xy-pad');
      ctx.currentDesignBrief = {
        pluginName: pluginConfig.name,
        width: pluginConfig.width,
        height: pluginConfig.height,
        bgColor: pluginConfig.bgColor,
        titleBarColor: pluginConfig.titleBarColor,
        aesthetic: activeTheme?.name || 'modern-minimal',
        accentColor: activeTheme?.accentColor || pluginConfig.bgColor,
        layout: panels.length >= 4 ? 'grid' : panels.length >= 2 ? 'horizontal' : 'vertical',
        sections: sectionList.map(s => {
          // Estimate weight from panel area relative to canvas
          const canvasArea = pluginConfig.width * pluginConfig.height;
          const panelArea = s.panel.width * s.panel.height;
          const areaRatio = panelArea / canvasArea;
          let weight = areaRatio > 0.25 ? 3 : areaRatio > 0.12 ? 2 : 1;

          const sectionDef = {
            label: s.name,
            weight,
            components: s.components.map(c => {
              const comp = { type: c.type, label: c.label };
              if (c.svgStyle) comp.svgStyle = c.svgStyle;
              if (c.fluxPrompt) comp.fluxPrompt = c.fluxPrompt;
              // Estimate size modifier
              const defaultW = c.type === 'knob' ? 60 : c.type === 'slider' ? 30 : 0;
              if (defaultW && c.width > defaultW * 1.2) comp.size = 'large';
              else if (defaultW && c.width < defaultW * 0.8) comp.size = 'small';
              return comp;
            }),
          };
          if (hasWaveformOrXY(s.components)) {
            sectionDef.layout = 'display-controls';
          }
          // Detect sidebar: panel spans near-full height and is on the right edge
          const panelRight = s.panel.x + s.panel.width;
          const nearRightEdge = panelRight >= pluginConfig.width * 0.85;
          const nearFullHeight = s.panel.height >= pluginConfig.height * 0.7;
          if (nearRightEdge && nearFullHeight) {
            sectionDef.position = 'right';
          }
          return sectionDef;
        }),
      };
    }
    if (dspConfig) {
      ctx.currentDsp = {
        pluginType: dspConfig.pluginType || 'effect',
        parameters: (dspConfig.parameters || []).map(p => `${p.name} (${p.id}: ${p.min}-${p.max} ${p.unit || ''})`),
        chainSummary: (dspConfig.dspChain || []).map(n => `${n.type}:${n.id}`).join(' → '),
      };
      ctx.IMPORTANT_pluginType = `Current plugin type is "${dspConfig.pluginType || 'effect'}". If the user asks for a different type of plugin, you MUST change pluginType accordingly.`;
    }
    if (referenceImage?.description) {
      ctx.referenceImageAnalysis = referenceImage.description;
    }
    return ctx;
  }, [pluginConfig, components, dspConfig, styleSeed, referenceImage, activeTheme]);

  // ── Reference image ──────────────────────────────────────────────────────
  const handleReferenceImage = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async (ev) => {
      const dataUrl = ev.target.result;
      setReferenceImage({ dataUrl, analyzing: true, description: null });
      try {
        const resp = await chatAPI.analyzeReferenceImage({ image_data: dataUrl });
        setReferenceImage({ dataUrl, analyzing: false, description: resp.description });
      } catch {
        setReferenceImage({ dataUrl, analyzing: false, description: 'Reference uploaded (analysis unavailable)' });
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }, []);

  // ── Send message (MoE pipeline) ──────────────────────────────────────────
  const handleSend = useCallback(async () => {
    if (!inputMessage.trim() || isLoading) return;
    let messageText = inputMessage;
    if (referenceImage?.description && !messages.some(m => m.content?.includes('[Reference'))) {
      messageText = `[Reference image: ${referenceImage.description}]\n\n${messageText}`;
    }
    const userMsg = { role: 'user', content: inputMessage, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const newDesign = isDesignRequest(messageText, components.length);
      console.log('[MasterChat] isDesignRequest:', newDesign, '| componentCount:', components.length, '| msg:', messageText.slice(0, 80));

      if (newDesign) {
        // ═══ Check for complex synth — use pre-built template for guaranteed density ═══
        const complexSynth = isComplexSynthRequest(messageText);

        if (complexSynth) {
          // ═══ COMPLEX SYNTH TEMPLATE PATH (bypasses GPT-4o for structure) ═══
          console.log('[MasterChat] Complex synth detected — using pre-built template');

          const brief = buildComplexSynthBrief(messageText);
          const totalComponents = countBriefComponents(brief);
          const canvasW = brief.width;
          const canvasH = brief.height;
          const layout = computeLayout(brief, canvasW, canvasH);

          console.log('[MasterChat] Template applied:', brief.pluginName, totalComponents, 'components');

          // Apply immediately — no waiting for GPT-4o (skip confirm dialog)
          layout.skipConfirm = true;
          onApplyLayout(layout);

          // Show assistant message
          const tabLabels = brief.tabs.map(t => t.label).join(', ');
          const templateMsg = {
            role: 'assistant',
            content: `Designed **${brief.pluginName}** — ${brief.tabs.length}-tab wavetable synth with **${totalComponents} components** (${tabLabels}) + master strip.\n\nStyle: *${brief.aesthetic}* | Accent: \`${brief.accentColor}\`\n\nGenerating DSP processing chain...`,
            timestamp: new Date().toISOString(),
          };
          setMessages(prev => [...prev, templateMsg]);

          // Save for debugging
          try {
            window.__masterChatDebug = {
              templateBrief: brief,
              totalComponents,
              isComplexSynth: true,
              timestamp: new Date().toISOString(),
            };
          } catch (e) { /* noop */ }

          // Generate DSP for the template (same flow as normal)
          try {
            // Build component summary from ALL tabs + persistent sections
            const allComponents = [];
            for (const tab of brief.tabs) {
              for (const sec of (tab.sections || [])) {
                for (const c of (sec.components || [])) {
                  allComponents.push(`${c.type}: "${c.label}"`);
                }
              }
            }
            for (const sec of (brief.persistentSections || [])) {
              for (const c of (sec.components || [])) {
                allComponents.push(`${c.type}: "${c.label}"`);
              }
            }
            const componentSummary = allComponents.join(', ');

            const dspResponse = await chatAPI.sendChatMessage({
              system_prompt: DSP_ARCHITECT_PROMPT,
              daw_context: {
                pluginName: brief.pluginName,
                pluginType: 'instrument',
                uiComponents: componentSummary,
                uiComponentCount: totalComponents,
                userRequest: messageText,
              },
              message: `Design the DSP processing chain for: ${messageText}\n\nThis is a Serum-class wavetable synthesizer with ${totalComponents} UI controls.\n\nUI components available for parameter binding (${totalComponents} total): ${componentSummary}\n\nIMPORTANT: This is a COMPLEX instrument with ${totalComponents} UI controls. Generate DSP parameters for ALL listed UI components. Do NOT simplify — match the full complexity of the design.`,
              conversation_history: [],
            });

            setMessages(prev => [...prev, {
              role: 'assistant',
              content: dspResponse.message,
              timestamp: dspResponse.timestamp || new Date().toISOString(),
            }]);

            if (autoApplyRef.current) {
              const { dspLangBlocks: autoDspBlocks } = parseBlocks(dspResponse.message);
              if (autoDspBlocks.length > 0) {
                console.log('[MasterChat] Auto-applying DSP config for complex synth');
                onApplyDsp(autoDspBlocks[0]);
              }
            }
          } catch (dspErr) {
            console.warn('DSP Architect call failed for complex synth:', dspErr);
          }
        } else {
        // ═══ NORMAL MoE PIPELINE: Design Director → Layout Engine → DSP Architect ═══
        console.log('[MasterChat] Using Design Director prompt (MoE pipeline)');

        // Rotate style seed for each new design request (ensures variety)
        const freshSeed = getRandomStyleSeed();
        setStyleSeed(freshSeed);

        // Generate per-request variation directives
        const variation = getVariationDirective(messageText);
        console.log('[MasterChat] Variation:', variation.variationId, variation.tabSuggestion.desc, variation.aesthetic);

        // Inject template guidance for known synth types (wavetable, etc.)
        let guidedMessage = injectTemplateGuidance(messageText);
        if (guidedMessage !== messageText) {
          console.log('[MasterChat] Template guidance injected for known synth type');
        }

        // Inject variation directive for non-specific requests
        // (don't override when user gave very specific instructions)
        const isVague = messageText.split(/\s+/).length < 20;
        if (isVague) {
          guidedMessage += `\n\n[CREATIVE VARIATION — variation seed: ${variation.variationId}]
IMPORTANT: Create a UNIQUE design that is DIFFERENT from any previous generation.
- Suggested tab structure: ${variation.tabSuggestion.tabs} tabs (${variation.tabSuggestion.labels.join(', ')}) — ${variation.tabSuggestion.desc}. Feel free to adapt or use a different structure if it better fits the concept.
- Layout ideas: ${variation.sectionHints.join('. ')}.
- Aesthetic direction: ${variation.aesthetic}
- Creative direction: ${freshSeed}
Do NOT copy a previous layout. Vary the number of sections, controls per section, control types, and visual hierarchy.`;
        }

        // Step 1: Design Director (focused on visual design)
        const ctx = buildContext();
        ctx.creativeDirection = freshSeed; // override with fresh seed

        // Lower temperature for specific hardware recreations (precision matters)
        // Higher temperature for generic creative requests (variety matters)
        const useHardwareTemp = isHardwareRequest(messageText.toLowerCase());
        const designTemperature = useHardwareTemp ? 0.5 : 0.9;
        if (useHardwareTemp) {
          console.log('[MasterChat] Hardware recreation detected — using lower temperature (0.5) for precision');
        }

        const designResponse = await chatAPI.sendChatMessage({
          system_prompt: DESIGN_DIRECTOR_PROMPT,
          daw_context: ctx,
          message: guidedMessage,
          conversation_history: messages,
          temperature: designTemperature,
          max_tokens: 12000,
        });

        // Show Design Director response immediately
        const designMsg = {
          role: 'assistant',
          content: designResponse.message,
          timestamp: designResponse.timestamp || new Date().toISOString(),
        };
        setMessages(prev => [...prev, designMsg]);

        // Step 2: Check if we need DSP (detect if user asked for a complete plugin)
        const { designBriefBlocks } = parseBlocks(designResponse.message);
        console.log('[MasterChat] Design Director response parsed:', designBriefBlocks.length, 'designbrief blocks');
        if (designBriefBlocks.length > 0) {
          console.log('[MasterChat] DesignBrief:', JSON.stringify(designBriefBlocks[0], null, 2).slice(0, 500));
        }
        // Save full AI response for debugging
        try {
          window.__masterChatDebug = {
            rawResponse: designResponse.message,
            designBriefBlocks,
            isDesignRequest: newDesign,
            timestamp: new Date().toISOString(),
          };
        } catch (e) { /* noop */ }

        // Auto-apply design brief
        if (autoApplyRef.current && designBriefBlocks.length > 0) {
          const brief = designBriefBlocks[0];
          const canvasW = brief.width || pluginConfig.width || 600;
          const canvasH = brief.height || pluginConfig.height || 400;
          const layout = computeLayout(brief, canvasW, canvasH);
          console.log('[MasterChat] Auto-applying design brief:', brief.pluginName);
          layout.skipConfirm = true;
          onApplyLayout(layout);
        }

        const needsDsp = /\b(synth|effect|reverb|delay|compressor|eq|distortion|plugin|instrument|sampler|chorus|phaser|flanger|saturation|overdrive|filter)\b/i.test(messageText);

        if (needsDsp && designBriefBlocks.length > 0) {
          // Step 3: DSP Architect (parallel-ish — fires after design director)
          const brief = designBriefBlocks[0];
          const componentSummary = (brief.sections || []).flatMap(s =>
            (s.components || []).map(c => `${c.type}: "${c.label}"`)
          ).join(', ');

          try {
            const componentCount = (brief.sections || []).reduce((sum, s) => sum + (s.components?.length || 0), 0);
            const complexityHint = componentCount > 20
              ? `\n\nIMPORTANT: This is a COMPLEX instrument with ${componentCount} UI controls. Generate DSP parameters for ALL listed UI components. Do NOT simplify — match the full complexity of the design.`
              : componentCount > 10
              ? `\n\nThis is a medium-complexity instrument with ${componentCount} UI controls. Generate parameters for all listed UI components.`
              : '';

            const dspResponse = await chatAPI.sendChatMessage({
              system_prompt: DSP_ARCHITECT_PROMPT,
              daw_context: {
                pluginName: brief.pluginName || 'Plugin',
                pluginType: /\b(synth|instrument|sampler)\b/i.test(messageText) ? 'instrument' : 'effect',
                uiComponents: componentSummary,
                uiComponentCount: componentCount,
                userRequest: messageText,
              },
              message: `Design the DSP processing chain for: ${messageText}\n\nUI components available for parameter binding (${componentCount} total): ${componentSummary}${complexityHint}`,
              conversation_history: [],
            });

            // Append DSP response as follow-up
            setMessages(prev => [...prev, {
              role: 'assistant',
              content: dspResponse.message,
              timestamp: dspResponse.timestamp || new Date().toISOString(),
            }]);

            // Auto-apply DSP config
            if (autoApplyRef.current) {
              const { dspLangBlocks: autoDspBlocks } = parseBlocks(dspResponse.message);
              if (autoDspBlocks.length > 0) {
                console.log('[MasterChat] Auto-applying DSP config');
                onApplyDsp(autoDspBlocks[0]);
              }
            }
          } catch (dspErr) {
            // DSP generation is optional — don't fail the whole request
            console.warn('DSP Architect call failed:', dspErr);
          }
        }
        } // end else (normal pipeline)
      } else {
        // ═══ REFINEMENT: Single focused call ═══
        const response = await chatAPI.sendChatMessage({
          system_prompt: REFINEMENT_PROMPT,
          daw_context: buildContext(),
          message: messageText,
          conversation_history: messages,
        });
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: response.message,
          timestamp: response.timestamp || new Date().toISOString(),
        }]);

        // Auto-apply any blocks in refinement response
        if (autoApplyRef.current) {
          const { designBriefBlocks: refBriefs, dspLangBlocks: refDsp, pluginLangBlocks: refUI } = parseBlocks(response.message);
          if (refBriefs.length > 0) {
            const brief = refBriefs[0];
            const canvasW = brief.width || pluginConfig.width || 600;
            const canvasH = brief.height || pluginConfig.height || 400;
            const layout = computeLayout(brief, canvasW, canvasH);
            console.log('[MasterChat] Auto-applying refinement design brief');
            layout.skipConfirm = true;
            onApplyLayout(layout);
          } else if (refUI.length > 0) {
            console.log('[MasterChat] Auto-applying refinement UI layout');
            refUI[0].skipConfirm = true;
            onApplyLayout(refUI[0]);
          }
          if (refDsp.length > 0) {
            console.log('[MasterChat] Auto-applying refinement DSP config');
            onApplyDsp(refDsp[0]);
          }
        }
      }
    } catch (err) {
      setError(err.message || 'Failed to get response');
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [inputMessage, isLoading, messages, buildContext, referenceImage, components.length, onApplyLayout, onApplyDsp, pluginConfig]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); }
  };

  // ── Apply actions with revert toast ──────────────────────────────────────
  const handleApplyLayout = useCallback((layout) => {
    onApplyLayout(layout);
    setRevertToast({ type: 'ui' });
    setTimeout(() => setRevertToast(null), 6000);
  }, [onApplyLayout]);

  const handleApplyDsp = useCallback((dsp) => {
    onApplyDsp(dsp);
    setRevertToast({ type: 'dsp' });
    setTimeout(() => setRevertToast(null), 6000);
  }, [onApplyDsp]);

  const handleApplyBoth = useCallback((layout, dsp) => {
    onApplyLayout(layout);
    onApplyDsp(dsp);
    setRevertToast({ type: 'both' });
    setTimeout(() => setRevertToast(null), 6000);
  }, [onApplyLayout, onApplyDsp]);

  // ── Apply design brief (MoE pipeline: layout engine + theme generator) ──
  const handleApplyDesignBrief = useCallback((brief) => {
    const canvasW = brief.width || pluginConfig.width || 600;
    const canvasH = brief.height || pluginConfig.height || 400;
    console.log('[MasterChat] Applying design brief:', brief.pluginName, canvasW + 'x' + canvasH, brief.aesthetic);

    // Compute layout through the layout engine (deterministic positioning)
    const layout = computeLayout(brief, canvasW, canvasH);
    console.log('[MasterChat] Layout computed:', layout.components?.length, 'components,', layout.components?.filter(c => c.svg).length, 'with SVGs');
    console.log('[MasterChat] Plugin config:', layout.pluginConfig);

    // Save for debugging
    try {
      window.__applyDesignBriefDebug = {
        brief, layout,
        svgSample: layout.components?.find(c => c.svg)?.svg?.slice(0, 200),
        timestamp: new Date().toISOString(),
      };
    } catch (e) { /* noop */ }

    // Apply through existing applyLayout pipeline
    onApplyLayout(layout);
    setRevertToast({ type: 'ui' });
    setTimeout(() => setRevertToast(null), 6000);
  }, [onApplyLayout, pluginConfig]);

  // ── Regenerate last assistant message ────────────────────────────────────
  const handleRegenerate = useCallback(() => {
    if (isLoading) return;
    // Find the last user message
    const lastUserIdx = messages.reduce((acc, m, i) => m.role === 'user' ? i : acc, -1);
    if (lastUserIdx === -1) return;
    const lastUserMsg = messages[lastUserIdx].content;
    // Remove all assistant messages after the last user message
    const trimmed = messages.slice(0, lastUserIdx + 1);
    setMessages(trimmed);
    // Re-send by setting inputMessage and triggering send
    setInputMessage(lastUserMsg);
    // Use a small delay to let state update before sending
    setTimeout(() => {
      // We set inputMessage above; the user can press send, or we auto-trigger
      inputRef.current?.focus();
    }, 50);
  }, [messages, isLoading]);

  // ── Retry after error ──────────────────────────────────────────────────
  const handleRetry = useCallback(() => {
    if (isLoading) return;
    setError(null);
    const lastUserIdx = messages.reduce((acc, m, i) => m.role === 'user' ? i : acc, -1);
    if (lastUserIdx === -1) return;
    const lastUserMsg = messages[lastUserIdx].content;
    setInputMessage(lastUserMsg);
    setTimeout(() => {
      inputRef.current?.focus();
    }, 50);
  }, [messages, isLoading]);

  // ── Code generation (validated) ──────────────────────────────────────────
  const handleGenerateCode = useCallback(async (dsp) => {
    setGeneratingCode(true);
    setError(null);
    try {
      let result;
      try {
        const resp = await fetch('/_chat/api/codegen/generate-validated', {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(dsp),
        });
        if (resp.ok) { result = await resp.json(); }
        else { result = await chatAPI.generatePluginCode(dsp); }
      } catch { result = await chatAPI.generatePluginCode(dsp); }
      setCodePreview(result);
      setActiveCodeFile(Object.keys(result.files)[0] || null);
    } catch (err) {
      setError(err.message || 'Code generation failed');
    } finally {
      setGeneratingCode(false);
    }
  }, []);

  // ── Build ────────────────────────────────────────────────────────────────
  const handleBuild = useCallback(async () => {
    if (!codePreview?.files) return;
    setBuildingPlugin(true);
    setBuildLogs([]);
    setBuildStage('Queuing build...');
    setError(null);
    try {
      const name = pluginConfig?.name || 'MyPlugin';
      const result = await buildPluginAutoFix(name, codePreview.files, null, {
        onLog: (line) => setBuildLogs(prev => [...prev.slice(-50), line]),
        onStage: (stage) => setBuildStage(stage),
        onAutoFix: (fixedFiles) => {
          // Update code preview with AI-fixed files
          if (fixedFiles) {
            setBuildLogs(prev => [...prev, '✓ AI auto-fixed compile errors']);
          }
        },
      });
      const url = URL.createObjectURL(result);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${name.replace(/[^a-zA-Z0-9_-]/g, '_')}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      setBuildStage('Build complete!');
    } catch (err) {
      setError(err.message || 'Build failed');
      setBuildStage('');
    } finally {
      setBuildingPlugin(false);
    }
  }, [codePreview, pluginConfig]);

  // ── Template selection ───────────────────────────────────────────────────
  const handleTemplateSelect = useCallback((template) => {
    if (template.pluginConfig && template.components) {
      onApplyLayout({ pluginConfig: template.pluginConfig, components: template.components, mode: 'replace' });
    }
    if (template.dspConfig) {
      onApplyDsp(template.dspConfig);
    }
    setShowTemplates(false);
    setMessages(prev => [...prev, {
      role: 'assistant',
      content: `Loaded the **${template.name}** template! The UI and DSP are pre-wired and ready to go. You can now:\n\n- Ask me to modify the design ("make it more vintage", "add a reverb section")\n- Click **Test** mode to hear the DSP\n- Click **Generate Code** to get compilable JUCE C++\n\nWhat would you like to change?`,
      timestamp: new Date().toISOString(),
    }]);
  }, [onApplyLayout, onApplyDsp]);

  const clearChat = () => {
    setMessages([{ ...defaultMsg, timestamp: new Date().toISOString() }]);
    setError(null);
    setCodePreview(null);
    setShowTemplates(true);
    try { localStorage.removeItem(CHAT_STORAGE_KEY); } catch (e) { /* noop */ }
  };

  const formatTime = (ts) => new Date(ts).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });

  // ── Auto-bind suggestion ────────────────────────────────────────────────
  const unboundParams = useMemo(() => {
    if (!dspConfig?.parameters || components.length === 0) return [];
    const interactiveLabels = components
      .filter(c => ['knob', 'slider', 'xy-pad'].includes(c.type))
      .map(c => (c.label || '').toLowerCase().replace(/[_\s-]/g, ''));
    return dspConfig.parameters.filter(p => {
      const name = (p.name || p.id || '').toLowerCase().replace(/[_\s-]/g, '');
      return !interactiveLabels.some(l => l === name || l.includes(name) || name.includes(l));
    });
  }, [dspConfig, components]);

  return (
    <>
      {/* Header */}
      <div className={styles.chatHeader}>
        <div className={styles.chatHeaderTitle}>
          <i className="fa-solid fa-bolt" />
          <span>Plugin Designer</span>
        </div>
        <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 3, fontSize: 10, color: autoApply ? 'rgba(76,175,80,0.9)' : 'rgba(255,255,255,0.4)', cursor: 'pointer', userSelect: 'none', marginRight: 4 }}
            title="Automatically apply generated designs and DSP configs">
            <input type="checkbox" checked={autoApply} onChange={e => setAutoApply(e.target.checked)}
              style={{ width: 12, height: 12, accentColor: '#4caf50' }} />
            Auto-apply
          </label>
          <input ref={fileInputRef} type="file" accept="image/*" onChange={handleReferenceImage} style={{ display: 'none' }} />
          <button className={styles.chatClearBtn} onClick={() => fileInputRef.current?.click()}
            title="Upload reference image" style={referenceImage ? { color: '#4caf50' } : undefined}>
            <i className="fa-solid fa-camera" />
          </button>
          {onOpenImageBrowser && (
            <button className={styles.chatClearBtn} onClick={() => onOpenImageBrowser('new-component')} title="Image browser">
              <i className="fa-solid fa-image" />
            </button>
          )}
          <button className={styles.chatClearBtn} onClick={() => setShowTemplates(prev => !prev)} title="Templates">
            <i className="fa-solid fa-grid-2" />
          </button>
          {onApplyTheme && (
            <button className={styles.chatClearBtn} onClick={() => setShowThemes(prev => !prev)} title="Themes"
              style={showThemes ? { color: '#ba9cff' } : undefined}>
              <i className="fa-solid fa-swatchbook" />
            </button>
          )}
          <button className={styles.chatClearBtn} onClick={clearChat} title="Clear chat">
            <i className="fa-solid fa-trash" />
          </button>
        </div>
      </div>

      {/* Reference image banner */}
      {referenceImage && (
        <div style={{ padding: '6px 10px', background: 'rgba(76,175,80,0.08)', borderBottom: '1px solid rgba(76,175,80,0.15)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
          <img src={referenceImage.dataUrl} alt="ref" style={{ width: 36, height: 36, objectFit: 'cover', borderRadius: 4 }} />
          <span style={{ flex: 1, color: 'rgba(255,255,255,0.7)' }}>
            {referenceImage.analyzing ? 'Analyzing reference...' : 'Reference loaded'}
          </span>
          <button onClick={() => setReferenceImage(null)} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)', cursor: 'pointer', fontSize: 12 }}>
            <i className="fa-solid fa-xmark" />
          </button>
        </div>
      )}

      {/* Auto-bind warning */}
      {unboundParams.length > 0 && (
        <div style={{
          padding: '6px 10px', background: 'rgba(255,170,0,0.06)', borderBottom: '1px solid rgba(255,170,0,0.12)',
          fontSize: 11, color: 'rgba(255,200,100,0.8)', display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <i className="fa-solid fa-link-slash" />
          <span>{unboundParams.length} DSP param{unboundParams.length > 1 ? 's' : ''} not bound to UI: {unboundParams.map(p => p.name).join(', ')}</span>
          <button
            onClick={() => {
              setInputMessage(`Add UI controls for these unbound DSP parameters: ${unboundParams.map(p => `${p.name} (${p.id}, ${p.min}-${p.max} ${p.unit})`).join(', ')}. Match the labels exactly to the parameter names so they auto-bind.`);
              inputRef.current?.focus();
            }}
            style={{ marginLeft: 'auto', background: 'rgba(255,170,0,0.15)', border: '1px solid rgba(255,170,0,0.25)', color: '#ffcc66', borderRadius: 4, padding: '2px 8px', fontSize: 10, cursor: 'pointer', whiteSpace: 'nowrap' }}
          >
            Fix it
          </button>
        </div>
      )}

      {/* Code Preview Panel */}
      {codePreview && (
        <div className={styles.codePreviewPanel} style={{ maxHeight: 200 }}>
          <div className={styles.codeFileTabs}>
            {Object.keys(codePreview.files).map(fname => (
              <button key={fname}
                className={activeCodeFile === fname ? styles.codeFileTabActive : styles.codeFileTab}
                onClick={() => setActiveCodeFile(fname)}>
                {fname}
              </button>
            ))}
            <button onClick={handleBuild} disabled={buildingPlugin}
              style={{
                marginLeft: 'auto', padding: '3px 10px', borderRadius: 6, border: 'none',
                fontSize: 11, fontWeight: 600, cursor: buildingPlugin ? 'wait' : 'pointer',
                background: buildingPlugin ? 'rgba(102,126,234,0.2)' : 'linear-gradient(135deg, #667eea, #764ba2)',
                color: '#fff', display: 'flex', alignItems: 'center', gap: 4,
              }}>
              <i className={`fa-solid ${buildingPlugin ? 'fa-spinner fa-spin' : 'fa-hammer'}`} />
              {buildingPlugin ? 'Building...' : 'Build'}
            </button>
            <button onClick={() => setCodePreview(null)}
              style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)', cursor: 'pointer', fontSize: 12 }}>
              <i className="fa-solid fa-xmark" />
            </button>
          </div>
          {buildingPlugin && buildLogs.length > 0 ? (
            <div style={{
              padding: '6px 10px', background: '#0a0a0a', fontSize: 10, fontFamily: 'monospace',
              maxHeight: 120, overflowY: 'auto', color: '#88ff88',
            }}>
              {buildStage && <div style={{ color: '#ffaa00', marginBottom: 2 }}>{buildStage}</div>}
              {buildLogs.map((line, i) => <div key={i} style={{ opacity: 0.8 }}>{line}</div>)}
            </div>
          ) : (
            <pre className={styles.codePreview} style={{ maxHeight: 140 }}>
              <code>{codePreview.files[activeCodeFile] || ''}</code>
            </pre>
          )}
        </div>
      )}

      {/* Template Selector */}
      {showTemplates && (
        <div style={{
          padding: '8px', borderBottom: '1px solid rgba(255,255,255,0.06)',
          display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 150, overflowY: 'auto',
        }}>
          {PLUGIN_TEMPLATES.map((t, i) => (
            <button key={i} onClick={() => handleTemplateSelect(t)} style={{
              padding: '6px 10px', borderRadius: 8, border: '1px solid rgba(186,156,255,0.15)',
              background: 'rgba(186,156,255,0.06)', color: 'rgba(255,255,255,0.8)',
              fontSize: 11, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
              transition: 'all 0.15s',
            }}
            onMouseEnter={e => { e.target.style.background = 'rgba(186,156,255,0.15)'; }}
            onMouseLeave={e => { e.target.style.background = 'rgba(186,156,255,0.06)'; }}
            >
              <i className={`fa-solid ${t.icon}`} style={{ fontSize: 12, color: t.color || '#ba9cff' }} />
              <div style={{ textAlign: 'left' }}>
                <div style={{ fontWeight: 600, fontSize: 11 }}>{t.name}</div>
                <div style={{ fontSize: 9, opacity: 0.5 }}>{t.description}</div>
              </div>
            </button>
          ))}
        </div>
      )}

      {/* Theme Selector */}
      {showThemes && onApplyTheme && (
        <div style={{
          padding: '8px', borderBottom: '1px solid rgba(255,255,255,0.06)',
          display: 'flex', flexWrap: 'wrap', gap: 6, maxHeight: 120, overflowY: 'auto',
        }}>
          {PLUGIN_THEMES.map((theme) => (
            <button key={theme.id} onClick={() => { onApplyTheme(theme); setShowThemes(false); }} style={{
              padding: '5px 10px', borderRadius: 8,
              border: activeTheme?.id === theme.id ? `1px solid ${theme.accentColor}` : '1px solid rgba(255,255,255,0.08)',
              background: activeTheme?.id === theme.id ? `rgba(${parseInt(theme.accentColor.slice(1,3),16)},${parseInt(theme.accentColor.slice(3,5),16)},${parseInt(theme.accentColor.slice(5,7),16)},0.12)` : 'rgba(255,255,255,0.03)',
              color: 'rgba(255,255,255,0.8)',
              fontSize: 11, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 6,
              transition: 'all 0.15s',
            }}>
              <div style={{
                width: 14, height: 14, borderRadius: 3,
                background: `linear-gradient(135deg, ${theme.bgColor}, ${theme.accentColor})`,
                border: '1px solid rgba(255,255,255,0.15)',
              }} />
              <span style={{ fontWeight: 500 }}>{theme.name}</span>
            </button>
          ))}
        </div>
      )}

      {/* Chat Messages */}
      <div className={styles.chatMessages}>
        {messages.map((msg, i) => {
          // Determine if this is the last assistant message (for showing regenerate)
          const isLastAssistant = msg.role === 'assistant' && i === messages.reduce((acc, m, j) => m.role === 'assistant' ? j : acc, -1);
          return (
          <div key={i} className={`${styles.message} ${msg.role === 'user' ? styles.userMessage : styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className={msg.role === 'user' ? 'fa-solid fa-user' : 'fa-solid fa-bolt'} />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.messageText}>
                {msg.role === 'assistant'
                  ? renderMessageContent(msg.content, handleApplyLayout, handleApplyDsp, handleApplyBoth, handleGenerateCode, handleApplyDesignBrief)
                  : msg.content}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <div className={styles.messageTime}>{formatTime(msg.timestamp)}</div>
                {isLastAssistant && !isLoading && messages.some(m => m.role === 'user') && (
                  <button
                    onClick={handleRegenerate}
                    title="Regenerate response"
                    style={{
                      background: 'none', border: 'none', cursor: 'pointer', padding: '1px 4px',
                      color: 'rgba(255,255,255,0.3)', fontSize: 11, borderRadius: 4,
                      transition: 'color 0.15s',
                    }}
                    onMouseEnter={e => { e.currentTarget.style.color = 'rgba(186,156,255,0.8)'; }}
                    onMouseLeave={e => { e.currentTarget.style.color = 'rgba(255,255,255,0.3)'; }}
                  >
                    <i className="fa-solid fa-rotate" />
                  </button>
                )}
              </div>
            </div>
          </div>
          );
        })}

        {isLoading && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.messageIcon}><i className="fa-solid fa-bolt" /></div>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}><span /><span /><span /></div>
            </div>
          </div>
        )}

        {generatingCode && (
          <div className={styles.chatGenerating}>
            <i className="fa-solid fa-spinner fa-spin" /> Generating & validating JUCE C++ code...
          </div>
        )}

        {error && (
          <div className={styles.chatError} style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <i className="fa-solid fa-exclamation-triangle" /><span style={{ flex: 1 }}>{error}</span>
            {messages.some(m => m.role === 'user') && (
              <button
                onClick={handleRetry}
                disabled={isLoading}
                style={{
                  background: 'rgba(255,100,80,0.15)', border: '1px solid rgba(255,100,80,0.3)',
                  color: 'rgba(255,180,160,0.9)', borderRadius: 4, padding: '3px 10px',
                  fontSize: 11, cursor: isLoading ? 'wait' : 'pointer', whiteSpace: 'nowrap',
                  fontWeight: 600, display: 'flex', alignItems: 'center', gap: 4,
                }}
              >
                <i className="fa-solid fa-rotate" /> Retry
              </button>
            )}
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Revert toast */}
      {revertToast && (
        <div style={{
          padding: '6px 12px', background: 'rgba(76,175,80,0.1)', borderTop: '1px solid rgba(76,175,80,0.15)',
          fontSize: 11, color: 'rgba(76,175,80,0.9)', display: 'flex', alignItems: 'center', gap: 6,
        }}>
          <i className="fa-solid fa-check" />
          <span>
            {revertToast.type === 'both' ? 'UI + DSP applied' : revertToast.type === 'ui' ? 'Layout applied' : 'DSP applied'}
            {' '} — press <strong>Ctrl+Z</strong> to undo
          </span>
        </div>
      )}

      {/* Input Area */}
      <div className={styles.chatInputArea}>
        <textarea
          ref={inputRef}
          className={styles.chatInput}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe your plugin — I'll design both the UI and DSP..."
          rows={1}
          disabled={isLoading}
        />
        <button className={styles.chatSendBtn} onClick={handleSend} disabled={!inputMessage.trim() || isLoading}>
          <i className="fa-solid fa-paper-plane" />
        </button>
      </div>
    </>
  );
};

export default MasterChat;
