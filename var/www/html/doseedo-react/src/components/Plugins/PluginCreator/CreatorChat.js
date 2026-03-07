import React, { useState, useRef, useEffect, useCallback } from 'react';
import * as chatAPI from '../../../services/chatAPI';
import styles from './PluginCreator.module.css';

// ── Style randomizer: injected as a creative direction hint per conversation ──
const STYLE_SEEDS = [
  "Use an asymmetric layout — uneven panel sizes, off-center hero knob, diagonal visual flow from top-left to bottom-right.",
  "Take inspiration from the Moog Minimoog Model D — wood side panels (brown side bars), black metal face, cream/white labels, chunky physical switches, vintage toggle buttons.",
  "Go brutalist — monochrome palette (#111/#eee), thick 3px borders, raw industrial feel, blocky sans-serif labels at large sizes (14-16px), no rounded corners, high-contrast VU meters.",
  "Think modular Eurorack faceplate — dense vertical layout (tall narrow: 300x500), patch point circles, small tightly-packed knobs (40x50), colored section dividers, metal panel texture background.",
  "Use a maximalist approach — pack every pixel with controls. Dense grid of small knobs (45x55), thin separators, information-rich: multiple meters, waveform displays, and readout labels showing values.",
  "Circular/radial layout — arrange knobs in a ring around a central waveform or XY-pad display. Use a round overall shape feel (large borderRadius panels). Think radar/sonar screen aesthetic.",
  "Recreate the Fairchild 670 — cream/beige faceplate (#f5e8d0), large VU meters at top, big bakelite knobs, red/black labels, vintage tube warmth, skeumorphic metal screws in corners.",
  "Anime/vaporwave mashup — pastel pink/cyan/purple gradient background, pixel art character image, Japanese-styled labels mixed with English, rounded pill-shaped buttons, LED dot-matrix style meters.",
  "Minimal white studio — light background (#f8f8f8), single accent color (#2563eb), hairline borders, generous whitespace, small precise knobs, elegant thin typography at 9-10px.",
  "Think Roland TR-808 — wide horizontal layout (900x350), colored step-sequencer buttons in a row, orange/yellow/red accents on dark gray, segmented LED displays, distinctive rounded rectangle buttons.",
  "Art deco luxury — gold (#d4a843) on deep navy (#0a1628), geometric patterns, sunburst panel gradients, ornate border decorations via boxShadow, elegant serif-feeling labels.",
  "Go sci-fi holographic — dark transparent background (#050510), cyan (#00f0ff) wireframe panels with no fill, floating knobs, scan-line texture overlay image, futuristic HUD-style meters.",
  "1970s analog console — olive/brown/cream palette, wood grain background image, large chicken-head knobs (75x85), rocker switches, tape-style VU meters, warm incandescent LED colors (#ff8c00).",
  "Bitcrushed lo-fi — pixelated background image, limited 8-bit color palette (#ff004d, #29adff, #00e436, #ffec27), chunky controls, retro game UI feel, pixel-font-sized labels.",
  "Use overlapping panels with depth — panels stacked with slight offsets and increasing boxShadow depth. Controls on the top panel feel elevated. Create a layered card-stack composition.",
  "Design for a live performance context — extra-large knobs (100x110) for the 2-3 most important params, minimal secondary controls, high-contrast colors visible in dim lighting, huge meters.",
  "Inspired by vintage Neve console — steel blue-gray faceplate (#4a5568), red and blue knob accents, precise detented-looking knobs, engraved-style white labels, horizontal strip layout.",
  "Underwater/organic — deep teal background (#0a2f2f), bioluminescent accent glows (#00ffa3, #00c3ff), flowing curved panel shapes (high borderRadius 20-30), coral/sea texture background image.",
  "Split personality — left half is pristine white minimal, right half is dark neon cyberpunk. A dramatic center divide. Controls mirror across the split with contrasting styles.",
  "Outrun/synthwave — dark purple-black (#1a0a2e) background, neon pink (#ff2d95) and electric blue (#00d4ff) accents, grid-line background image, chrome/metallic knob colors, sunset gradient panels.",
];

function getRandomStyleSeed() {
  return STYLE_SEEDS[Math.floor(Math.random() * STYLE_SEEDS.length)];
}

const SYSTEM_PROMPT = `You are an elite VST plugin UI designer who creates stunning, diverse, and ORIGINAL audio plugin interfaces. You push creative boundaries — every design should feel distinct and surprising. You draw inspiration from real hardware (Moog, Neve, Fairchild, Roland, Buchla, Eventide), modern design movements, art styles, and unexpected sources (architecture, nature, fashion, retro computing, sci-fi).

CRITICAL: Do NOT default to the same dark-blue glassmorphic layout every time. Vary your style, layout structure, color palette, and component arrangement dramatically between designs. Each plugin should have a unique visual identity.

You output layouts using a structured "Plugin Language" format inside \`\`\`pluginlang code blocks.

## PLUGINLANG SCHEMA

{
  "pluginConfig": {
    "name": "Plugin Name",
    "width": 600, "height": 400,
    "bgColor": "#1a1a2e",
    "titleBarColor": "#2d2d4e",
    "bgImage": {"generate": "DALL-E prompt for background"} OR {"search": "Unsplash search query"} OR "url"
  },
  "components": [ ... ],
  "mode": "replace" OR "merge"
}

## COMPONENT TYPES

| Type | Default Size | Use For |
|------|-------------|---------|
| knob | 60x70 | Rotary control: gain, freq, mix, drive |
| slider | 30x120 | Vertical fader: volume, sends |
| button | 70x28 | Bypass, mode toggle |
| label | 100x24 | Section titles, value readouts |
| led | 12x12 | Status indicator |
| dropdown | 120x28 | Algorithm/mode select |
| image | 80x80 | Custom graphics, logos, artwork |
| panel | 200x150 | Section container/grouping |
| meter | 24x100 | VU/level meter |
| waveform | 180x60 | Oscilloscope display |
| xy-pad | 120x120 | 2D modulation control |

## ALL COMPONENT PROPERTIES

Common: x, y, width, height, color, label, opacity (0-1), rotation (deg), borderRadius (px), fontSize (px), zIndex (int)

Panel-specific: borderColor, bgColor, bgGradient (CSS gradient string like "linear-gradient(135deg, rgba(255,100,200,0.15), rgba(100,50,150,0.05))"), backdropBlur (px, for glass effects), boxShadow (CSS shadow string)

Image-specific: image — can be a URL string, OR {"generate": "DALL-E prompt"} to auto-generate, OR {"search": "query"} to auto-search stock photos. ALWAYS use image generation/search when the user wants custom visuals — never leave image blank.

## AUTO IMAGE GENERATION

When the user wants themed visuals (anime, retro, abstract art, textures, etc.), use the image generation system:
- For backgrounds: set pluginConfig.bgImage to {"generate": "detailed DALL-E prompt"} or {"search": "query"}
- For image components: set image to {"generate": "prompt"} or {"search": "query"}
- Write detailed DALL-E prompts: include style, colors, mood, composition. Example: {"generate": "kawaii anime girl with pink headphones and cat ears, pastel pink and purple gradient background, sparkles and stars, cute chibi style, digital illustration"}
- The system will auto-generate/search and apply the image. DO NOT tell the user to upload their own image.

## DESIGN STYLES — Use these creatively, MIX them, SUBVERT them:

**Glassmorphism**: Semi-transparent panels with backdrop blur. Use bgColor "rgba(255,255,255,0.08)", backdropBlur 12, borderColor "rgba(255,255,255,0.15)". Layer over a colorful background image.

**Skeumorphism**: Rich textures, realistic shadows, metallic knobs. Dark grays (#2a2a2a, #1a1a1a), thick borders, boxShadow "0 4px 15px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1)".

**Neumorphism**: Soft raised/pressed elements. Light bg (#e0e0e0 area), boxShadow with dual light/dark. Panel: boxShadow "8px 8px 16px rgba(0,0,0,0.15), -8px -8px 16px rgba(255,255,255,0.05)".

**Cyberpunk/Neon**: Dark backgrounds (#0a0a1a), neon accent colors (#00ff88, #ff00ff, #00ddff), glowing boxShadow borders, high contrast.

**Anime/Kawaii**: Pastel colors (#ffb7d5, #b5deff, #c5b3ff), rounded corners (borderRadius 12-20), generated anime background image.

**Retro/Vintage**: Warm browns (#3a2a1a), amber (#d4a043), cream (#f5e6c8), textured backgrounds, classic VU meter aesthetics.

**Modern Minimal**: Lots of whitespace, subtle accent color, thin lines, clean typography.

**Brutalist**: Raw, monochrome, thick borders, no decoration. High contrast black/white with one bold accent.

**Art Deco**: Gold on navy, geometric patterns, sunburst gradients, ornate framing.

**Hardware Recreation**: Model real gear — wood sides, metal faceplates, chicken-head knobs, specific hardware color palettes.

**Outrun/Synthwave**: Neon on dark, chrome accents, grid backgrounds, sunset gradients.

**Organic/Nature**: Flowing shapes (high borderRadius), earth tones or bioluminescent accents, nature-textured backgrounds.

## COORDINATE SYSTEM
Origin (0,0) = top-left. X right, Y down. Title bar is 32px above canvas.

## EDITING RULES — CRITICAL

**Partial edits (user asks to change something specific)**:
- When the user says "change the background", "make it pink", "add a knob", "move the reverb knob" etc., KEEP everything else and only change what they asked.
- Use "mode": "replace" but include ALL existing components from context, modified as requested.
- Read the current layout from the context carefully. Preserve all components the user didn't mention.
- Only use "mode": "merge" when the user explicitly says "add" something new.

**Full redesign (user asks for a new plugin or complete restyle)**:
- When the user says "design a compressor", "make me a new plugin", "completely redesign" — do a full replace.

**When in doubt, preserve existing components.** Users get frustrated when their work disappears.

## LAYOUT CREATIVITY GUIDELINES

1. **Vary your layouts** — Don't always use two symmetric side-by-side panels. Try: single wide panel, L-shaped, stacked horizontal strips, radial/circular, overlapping depth cards, asymmetric split, dense grid, or open floating.
2. **Vary your sizes** — Not everything is 600x400. Try: tall narrow (350x550), ultra-wide (900x350), compact (380x280), or expansive (950x600).
3. **Vary your color palettes** — Don't default to blue/purple. Use: warm amber/cream, red/black, green/gold, pink/white, monochrome, earth tones, neon triads.
4. Create visual hierarchy: large hero knobs (80x90+) for main controls, smaller knobs (45x55) for secondary.
5. Use panels with subtle borders + glass/gradient effects to group controls.
6. Labels ABOVE or BELOW controls, centered, font-size 10-11px, low opacity (0.4-0.6).
7. Use background images for themed plugins — always generate them.
8. Layer: panels at zIndex 0-1, displays at 2, controls at 3-5, labels at 4-6.

## EXAMPLE 1 — Glassmorphic Delay (dark, cosmic):
\`\`\`pluginlang
{
  "pluginConfig": {
    "name": "Crystal Delay",
    "width": 650, "height": 420,
    "bgColor": "#0a0a2e",
    "titleBarColor": "rgba(20,20,60,0.8)",
    "bgImage": {"generate": "abstract cosmic nebula with deep blue and purple swirls, stars and light particles, ethereal space atmosphere, digital art, dark background"}
  },
  "components": [
    {"type": "panel", "x": 20, "y": 15, "width": 300, "height": 390, "borderColor": "rgba(130,170,255,0.15)", "bgColor": "rgba(130,170,255,0.06)", "bgGradient": "linear-gradient(180deg, rgba(130,170,255,0.1) 0%, rgba(130,170,255,0.02) 100%)", "backdropBlur": 12, "borderRadius": 16, "zIndex": 0, "label": ""},
    {"type": "panel", "x": 330, "y": 15, "width": 300, "height": 390, "borderColor": "rgba(200,130,255,0.15)", "bgColor": "rgba(200,130,255,0.06)", "backdropBlur": 12, "borderRadius": 16, "zIndex": 0, "label": ""},
    {"type": "knob", "label": "Time", "x": 55, "y": 60, "width": 80, "height": 90, "color": "#82aaff", "zIndex": 3},
    {"type": "knob", "label": "Feedback", "x": 175, "y": 60, "width": 80, "height": 90, "color": "#82aaff", "zIndex": 3},
    {"type": "knob", "label": "Tone", "x": 370, "y": 60, "width": 80, "height": 90, "color": "#c882ff", "zIndex": 3},
    {"type": "waveform", "label": "Delay Shape", "x": 40, "y": 190, "width": 260, "height": 65, "color": "#82aaff", "zIndex": 3},
    {"type": "xy-pad", "label": "Space", "x": 360, "y": 180, "width": 240, "height": 100, "color": "#c882ff", "zIndex": 3, "borderRadius": 12},
    {"type": "slider", "label": "Mix", "x": 70, "y": 290, "width": 25, "height": 90, "color": "#82aaff", "zIndex": 3},
    {"type": "meter", "label": "L", "x": 375, "y": 310, "width": 18, "height": 70, "color": "#4caf50", "zIndex": 3},
    {"type": "meter", "label": "R", "x": 400, "y": 310, "width": 18, "height": 70, "color": "#4caf50", "zIndex": 3},
    {"type": "button", "label": "Ping-Pong", "x": 450, "y": 320, "width": 80, "height": 26, "color": "#c882ff", "borderRadius": 6, "fontSize": 10, "zIndex": 3}
  ]
}
\`\`\`

## EXAMPLE 2 — Vintage Hardware Compressor (warm skeumorphic):
\`\`\`pluginlang
{
  "pluginConfig": {
    "name": "Fairfield Comp",
    "width": 800, "height": 340,
    "bgColor": "#d4c5a0",
    "titleBarColor": "#5a4a32",
    "bgImage": {"generate": "beige cream metal faceplate texture with subtle brushed aluminum grain, vintage audio equipment surface, warm lighting, photorealistic"}
  },
  "components": [
    {"type": "panel", "x": 15, "y": 10, "width": 770, "height": 320, "borderColor": "#8a7a5a", "bgColor": "rgba(90,74,50,0.08)", "boxShadow": "inset 0 2px 8px rgba(0,0,0,0.15), 0 1px 0 rgba(255,255,255,0.3)", "borderRadius": 4, "zIndex": 0, "label": ""},
    {"type": "meter", "label": "GR", "x": 340, "y": 25, "width": 120, "height": 80, "color": "#cc3333", "zIndex": 3},
    {"type": "label", "label": "GAIN REDUCTION", "x": 340, "y": 108, "width": 120, "height": 16, "color": "#5a4a32", "fontSize": 9, "zIndex": 5},
    {"type": "knob", "label": "Input", "x": 40, "y": 40, "width": 85, "height": 95, "color": "#2a2a2a", "zIndex": 3},
    {"type": "knob", "label": "Output", "x": 160, "y": 40, "width": 85, "height": 95, "color": "#2a2a2a", "zIndex": 3},
    {"type": "knob", "label": "Attack", "x": 510, "y": 40, "width": 75, "height": 85, "color": "#1a3a6a", "zIndex": 3},
    {"type": "knob", "label": "Release", "x": 620, "y": 40, "width": 75, "height": 85, "color": "#1a3a6a", "zIndex": 3},
    {"type": "knob", "label": "Ratio", "x": 80, "y": 175, "width": 70, "height": 80, "color": "#3a1a1a", "zIndex": 3},
    {"type": "knob", "label": "Threshold", "x": 190, "y": 175, "width": 70, "height": 80, "color": "#3a1a1a", "zIndex": 3},
    {"type": "knob", "label": "Mix", "x": 565, "y": 175, "width": 70, "height": 80, "color": "#2a4a2a", "zIndex": 3},
    {"type": "dropdown", "label": "Mode", "x": 350, "y": 180, "width": 100, "height": 26, "color": "#5a4a32", "fontSize": 10, "zIndex": 3},
    {"type": "button", "label": "BYPASS", "x": 350, "y": 220, "width": 100, "height": 30, "color": "#993333", "borderRadius": 3, "fontSize": 11, "zIndex": 3},
    {"type": "led", "label": "", "x": 340, "y": 228, "width": 8, "height": 8, "color": "#ff4444", "zIndex": 4},
    {"type": "label", "label": "FAIRFIELD COMP", "x": 300, "y": 290, "width": 200, "height": 20, "color": "#5a4a32", "fontSize": 14, "opacity": 0.7, "zIndex": 5}
  ]
}
\`\`\`

## EXAMPLE 3 — Minimal White EQ (clean, modern):
\`\`\`pluginlang
{
  "pluginConfig": {
    "name": "Pure EQ",
    "width": 520, "height": 380,
    "bgColor": "#f5f5f5",
    "titleBarColor": "#e8e8e8"
  },
  "components": [
    {"type": "waveform", "label": "", "x": 30, "y": 20, "width": 460, "height": 120, "color": "#2563eb", "zIndex": 2, "borderRadius": 8},
    {"type": "panel", "x": 30, "y": 155, "width": 460, "height": 200, "borderColor": "#e0e0e0", "bgColor": "#ffffff", "boxShadow": "0 1px 3px rgba(0,0,0,0.06)", "borderRadius": 10, "zIndex": 0, "label": ""},
    {"type": "label", "label": "LOW", "x": 55, "y": 165, "width": 50, "height": 14, "color": "#9ca3af", "fontSize": 9, "zIndex": 5},
    {"type": "knob", "label": "Freq", "x": 45, "y": 185, "width": 50, "height": 60, "color": "#2563eb", "zIndex": 3},
    {"type": "knob", "label": "Gain", "x": 105, "y": 185, "width": 50, "height": 60, "color": "#2563eb", "zIndex": 3},
    {"type": "label", "label": "MID", "x": 205, "y": 165, "width": 50, "height": 14, "color": "#9ca3af", "fontSize": 9, "zIndex": 5},
    {"type": "knob", "label": "Freq", "x": 185, "y": 185, "width": 50, "height": 60, "color": "#7c3aed", "zIndex": 3},
    {"type": "knob", "label": "Gain", "x": 245, "y": 185, "width": 50, "height": 60, "color": "#7c3aed", "zIndex": 3},
    {"type": "knob", "label": "Q", "x": 215, "y": 260, "width": 45, "height": 55, "color": "#7c3aed", "zIndex": 3},
    {"type": "label", "label": "HIGH", "x": 365, "y": 165, "width": 50, "height": 14, "color": "#9ca3af", "fontSize": 9, "zIndex": 5},
    {"type": "knob", "label": "Freq", "x": 345, "y": 185, "width": 50, "height": 60, "color": "#dc2626", "zIndex": 3},
    {"type": "knob", "label": "Gain", "x": 405, "y": 185, "width": 50, "height": 60, "color": "#dc2626", "zIndex": 3},
    {"type": "slider", "label": "Out", "x": 468, "y": 175, "width": 18, "height": 80, "color": "#374151", "zIndex": 3},
    {"type": "button", "label": "Bypass", "x": 420, "y": 320, "width": 60, "height": 24, "color": "#e5e7eb", "fontSize": 10, "borderRadius": 6, "zIndex": 3}
  ]
}
\`\`\`

## EXAMPLE 4 — Dense Eurorack Module (modular synth):
\`\`\`pluginlang
{
  "pluginConfig": {
    "name": "Resonant Void",
    "width": 320, "height": 550,
    "bgColor": "#1a1a1a",
    "titleBarColor": "#0d0d0d",
    "bgImage": {"generate": "dark brushed aluminum metal panel texture with subtle vertical grain lines, industrial eurorack module faceplate, matte black finish, photorealistic"}
  },
  "components": [
    {"type": "label", "label": "RESONANT VOID", "x": 60, "y": 8, "width": 200, "height": 20, "color": "#e0e0e0", "fontSize": 13, "zIndex": 5},
    {"type": "panel", "x": 15, "y": 35, "width": 290, "height": 2, "bgColor": "#ff4444", "borderRadius": 0, "zIndex": 1, "label": ""},
    {"type": "knob", "label": "FREQ", "x": 25, "y": 50, "width": 55, "height": 65, "color": "#ff4444", "zIndex": 3},
    {"type": "knob", "label": "RES", "x": 100, "y": 50, "width": 55, "height": 65, "color": "#ff4444", "zIndex": 3},
    {"type": "knob", "label": "DRIVE", "x": 175, "y": 50, "width": 55, "height": 65, "color": "#ffaa00", "zIndex": 3},
    {"type": "knob", "label": "FM", "x": 250, "y": 50, "width": 45, "height": 55, "color": "#44aaff", "zIndex": 3},
    {"type": "panel", "x": 15, "y": 130, "width": 290, "height": 1, "bgColor": "#333333", "borderRadius": 0, "zIndex": 1, "label": ""},
    {"type": "knob", "label": "ATK", "x": 25, "y": 145, "width": 45, "height": 55, "color": "#44ff88", "zIndex": 3},
    {"type": "knob", "label": "DEC", "x": 85, "y": 145, "width": 45, "height": 55, "color": "#44ff88", "zIndex": 3},
    {"type": "knob", "label": "SUS", "x": 145, "y": 145, "width": 45, "height": 55, "color": "#44ff88", "zIndex": 3},
    {"type": "knob", "label": "REL", "x": 205, "y": 145, "width": 45, "height": 55, "color": "#44ff88", "zIndex": 3},
    {"type": "knob", "label": "AMT", "x": 265, "y": 145, "width": 40, "height": 50, "color": "#88ff44", "zIndex": 3},
    {"type": "waveform", "label": "ENV", "x": 20, "y": 215, "width": 280, "height": 50, "color": "#44ff88", "zIndex": 2, "borderRadius": 4},
    {"type": "panel", "x": 15, "y": 280, "width": 290, "height": 1, "bgColor": "#333333", "borderRadius": 0, "zIndex": 1, "label": ""},
    {"type": "knob", "label": "LFO", "x": 25, "y": 295, "width": 50, "height": 60, "color": "#ff44ff", "zIndex": 3},
    {"type": "knob", "label": "DEPTH", "x": 95, "y": 295, "width": 50, "height": 60, "color": "#ff44ff", "zIndex": 3},
    {"type": "dropdown", "label": "SHAPE", "x": 175, "y": 305, "width": 80, "height": 24, "color": "#ff44ff", "fontSize": 9, "zIndex": 3},
    {"type": "button", "label": "SYNC", "x": 265, "y": 305, "width": 40, "height": 22, "color": "#ff44ff", "fontSize": 9, "borderRadius": 2, "zIndex": 3},
    {"type": "panel", "x": 15, "y": 375, "width": 290, "height": 1, "bgColor": "#333333", "borderRadius": 0, "zIndex": 1, "label": ""},
    {"type": "xy-pad", "label": "MORPH", "x": 20, "y": 390, "width": 130, "height": 100, "color": "#ffaa44", "zIndex": 3, "borderRadius": 4},
    {"type": "meter", "label": "L", "x": 175, "y": 390, "width": 16, "height": 100, "color": "#44ff88", "zIndex": 3},
    {"type": "meter", "label": "R", "x": 200, "y": 390, "width": 16, "height": 100, "color": "#44ff88", "zIndex": 3},
    {"type": "knob", "label": "MIX", "x": 235, "y": 400, "width": 55, "height": 65, "color": "#ffffff", "zIndex": 3},
    {"type": "button", "label": "BYPASS", "x": 230, "y": 500, "width": 70, "height": 24, "color": "#333333", "fontSize": 9, "borderRadius": 2, "zIndex": 3},
    {"type": "led", "label": "", "x": 220, "y": 507, "width": 8, "height": 8, "color": "#ff4444", "zIndex": 4}
  ]
}
\`\`\`

## EXAMPLE 5 — Synthwave Reverb (outrun neon):
\`\`\`pluginlang
{
  "pluginConfig": {
    "name": "Neon Halls",
    "width": 700, "height": 380,
    "bgColor": "#0d0018",
    "titleBarColor": "#1a0030",
    "bgImage": {"generate": "synthwave retro grid landscape, neon pink and cyan horizon line, dark purple sky with stars, 1980s outrun aesthetic, digital art, wide format"}
  },
  "components": [
    {"type": "panel", "x": 20, "y": 15, "width": 420, "height": 350, "borderColor": "rgba(255,45,149,0.25)", "bgColor": "rgba(255,45,149,0.04)", "bgGradient": "linear-gradient(180deg, rgba(255,45,149,0.08) 0%, rgba(0,212,255,0.04) 100%)", "backdropBlur": 8, "borderRadius": 12, "zIndex": 0, "label": ""},
    {"type": "panel", "x": 460, "y": 15, "width": 220, "height": 350, "borderColor": "rgba(0,212,255,0.25)", "bgColor": "rgba(0,212,255,0.04)", "backdropBlur": 8, "borderRadius": 12, "zIndex": 0, "label": ""},
    {"type": "label", "label": "NEON HALLS", "x": 140, "y": 22, "width": 180, "height": 22, "color": "#ff2d95", "fontSize": 16, "zIndex": 5},
    {"type": "waveform", "label": "Impulse", "x": 40, "y": 55, "width": 380, "height": 70, "color": "#00d4ff", "zIndex": 2, "borderRadius": 6},
    {"type": "knob", "label": "Size", "x": 50, "y": 150, "width": 75, "height": 85, "color": "#ff2d95", "zIndex": 3},
    {"type": "knob", "label": "Decay", "x": 160, "y": 150, "width": 75, "height": 85, "color": "#ff2d95", "zIndex": 3},
    {"type": "knob", "label": "Damping", "x": 270, "y": 150, "width": 65, "height": 75, "color": "#ff6ab3", "zIndex": 3},
    {"type": "knob", "label": "Pre-Delay", "x": 50, "y": 265, "width": 60, "height": 70, "color": "#00d4ff", "zIndex": 3},
    {"type": "knob", "label": "Width", "x": 145, "y": 265, "width": 60, "height": 70, "color": "#00d4ff", "zIndex": 3},
    {"type": "knob", "label": "Mod", "x": 240, "y": 265, "width": 55, "height": 65, "color": "#00ffa3", "zIndex": 3},
    {"type": "dropdown", "label": "Hall Type", "x": 325, "y": 275, "width": 95, "height": 26, "color": "#ff2d95", "fontSize": 10, "zIndex": 3},
    {"type": "xy-pad", "label": "Tone Shape", "x": 480, "y": 45, "width": 180, "height": 130, "color": "#00d4ff", "zIndex": 3, "borderRadius": 8},
    {"type": "slider", "label": "Mix", "x": 490, "y": 210, "width": 22, "height": 100, "color": "#ff2d95", "zIndex": 3},
    {"type": "meter", "label": "L", "x": 570, "y": 210, "width": 18, "height": 100, "color": "#00ffa3", "zIndex": 3},
    {"type": "meter", "label": "R", "x": 598, "y": 210, "width": 18, "height": 100, "color": "#00ffa3", "zIndex": 3},
    {"type": "button", "label": "FREEZE", "x": 490, "y": 325, "width": 70, "height": 26, "color": "#00d4ff", "borderRadius": 4, "fontSize": 10, "zIndex": 3},
    {"type": "button", "label": "BYPASS", "x": 570, "y": 325, "width": 60, "height": 26, "color": "rgba(255,255,255,0.1)", "borderRadius": 4, "fontSize": 10, "zIndex": 3}
  ]
}
\`\`\`

IMPORTANT:
- Always output pluginlang when the user asks for a layout, redesign, or any visual change.
- When modifying, preserve components the user didn't mention — read the context.
- When the user wants themed visuals, ALWAYS use {"generate": "..."} for images — never tell them to upload manually.
- Brief explanation of design choices (2-3 sentences) before/after the block.
- NEVER produce the same layout twice — vary structure, palette, and style every time.`;

/** Extract pluginlang JSON blocks from AI message text */
function parsePluginLang(text) {
  const blocks = [];
  const regex = /```pluginlang\s*([\s\S]*?)```/g;
  let match;
  while ((match = regex.exec(text)) !== null) {
    try {
      const parsed = JSON.parse(match[1].trim());
      blocks.push(parsed);
    } catch {
      // malformed JSON — skip
    }
  }
  return blocks;
}

/** Render message text, replacing pluginlang blocks with Apply buttons */
function renderMessageContent(text, onApply) {
  const parts = [];
  const regex = /```pluginlang\s*([\s\S]*?)```/g;
  let lastIdx = 0;
  let match;
  let blockIdx = 0;

  while ((match = regex.exec(text)) !== null) {
    // Text before the block
    if (match.index > lastIdx) {
      parts.push(
        <span key={`t-${blockIdx}`}>{text.slice(lastIdx, match.index)}</span>
      );
    }
    // Try to parse the block
    let parsed = null;
    try { parsed = JSON.parse(match[1].trim()); } catch {}

    if (parsed) {
      const compCount = parsed.components ? parsed.components.length : 0;
      const name = parsed.pluginConfig?.name || 'Layout';
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div className={styles.pluginlangHeader}>
            <i className="fa-solid fa-puzzle-piece" />
            <span>{name} — {compCount} components</span>
          </div>
          <button
            className={styles.applyLayoutBtn}
            onClick={() => onApply(parsed)}
          >
            <i className="fa-solid fa-wand-magic-sparkles" /> Apply Layout
          </button>
        </div>
      );
    } else {
      parts.push(
        <div key={`b-${blockIdx}`} className={styles.pluginlangBlock}>
          <div className={styles.pluginlangError}>
            <i className="fa-solid fa-triangle-exclamation" /> Layout parse error
          </div>
        </div>
      );
    }
    lastIdx = match.index + match[0].length;
    blockIdx++;
  }

  // Remaining text after last block
  if (lastIdx < text.length) {
    parts.push(<span key="end">{text.slice(lastIdx)}</span>);
  }

  return parts.length > 0 ? parts : text;
}

const CreatorChat = ({ pluginConfig, components, dspContext, onApplyLayout, onOpenImageBrowser }) => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hey! I'm your plugin UI designer. Describe the plugin you want to build — I'll generate a complete layout you can apply with one click.\n\nTry: \"Design a warm analog compressor with input/output knobs, a gain reduction meter, and attack/release controls\"",
      timestamp: new Date().toISOString(),
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [referenceImage, setReferenceImage] = useState(null); // { dataUrl, analyzing }
  const [styleSeed] = useState(() => getRandomStyleSeed());
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);
  const fileInputRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const buildContext = useCallback(() => {
    const ctx = {
      pluginName: pluginConfig.name,
      canvasSize: `${pluginConfig.width}x${pluginConfig.height}`,
      bgColor: pluginConfig.bgColor,
      componentCount: components.length,
      components: components.map(c => ({
        type: c.type, label: c.label, x: c.x, y: c.y, width: c.width, height: c.height,
      })),
      creativeDirection: styleSeed,
    };
    if (dspContext) {
      ctx.dspPartnerContext = {
        pluginType: dspContext.pluginType,
        parameters: (dspContext.parameters || []).map(p => `${p.name} (${p.id}: ${p.min}-${p.max} ${p.unit})`),
        chainSummary: (dspContext.dspChain || []).map(n => `${n.type}:${n.id}`).join(' → '),
      };
    }
    if (referenceImage?.description) {
      ctx.referenceImageAnalysis = referenceImage.description;
    }
    return ctx;
  }, [pluginConfig, components, dspContext, styleSeed, referenceImage]);

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
      } catch (err) {
        setReferenceImage({ dataUrl, analyzing: false, description: 'Reference image uploaded (analysis unavailable)' });
      }
    };
    reader.readAsDataURL(file);
    e.target.value = '';
  }, []);

  const clearReferenceImage = useCallback(() => {
    setReferenceImage(null);
  }, []);

  const handleSend = useCallback(async () => {
    if (!inputMessage.trim() || isLoading) return;
    let messageText = inputMessage;
    if (referenceImage?.description && !messages.some(m => m.content?.includes('[Reference image'))) {
      messageText = `[Reference image analysis: ${referenceImage.description}]\n\n${messageText}`;
    }
    const userMsg = { role: 'user', content: inputMessage, timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setIsLoading(true);
    setError(null);

    try {
      const response = await chatAPI.sendChatMessage({
        system_prompt: SYSTEM_PROMPT,
        daw_context: buildContext(),
        message: messageText,
        conversation_history: messages,
      });
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: response.message,
        timestamp: response.timestamp,
      }]);
    } catch (err) {
      setError(err.message || 'Failed to get response');
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  }, [inputMessage, isLoading, messages, buildContext, referenceImage]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    setMessages([{
      role: 'assistant',
      content: 'Chat cleared. Describe a plugin and I\'ll design a layout for you.',
      timestamp: new Date().toISOString(),
    }]);
    setError(null);
  };

  const formatTime = (ts) => {
    const d = new Date(ts);
    return d.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
  };

  const handleApply = useCallback((layout) => {
    if (onApplyLayout) onApplyLayout(layout);
  }, [onApplyLayout]);

  return (
    <>
      <div className={styles.chatHeader}>
        <div className={styles.chatHeaderTitle}>
          <i className="fa-solid fa-wand-magic-sparkles" />
          <span>UI Designer</span>
        </div>
        <div style={{ display: 'flex', gap: 4 }}>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleReferenceImage}
            style={{ display: 'none' }}
          />
          <button
            className={styles.chatClearBtn}
            onClick={() => fileInputRef.current?.click()}
            title="Upload reference image"
            style={referenceImage ? { color: '#4caf50' } : undefined}
          >
            <i className="fa-solid fa-camera" />
          </button>
          {onOpenImageBrowser && (
            <button className={styles.chatClearBtn} onClick={() => onOpenImageBrowser('new-component')} title="Image browser">
              <i className="fa-solid fa-image" />
            </button>
          )}
          <button className={styles.chatClearBtn} onClick={clearChat} title="Clear chat">
            <i className="fa-solid fa-trash" />
          </button>
        </div>
      </div>

      {referenceImage && (
        <div style={{ padding: '6px 10px', background: 'rgba(76,175,80,0.08)', borderBottom: '1px solid rgba(76,175,80,0.15)', display: 'flex', alignItems: 'center', gap: 8, fontSize: 11 }}>
          <img src={referenceImage.dataUrl} alt="ref" style={{ width: 36, height: 36, objectFit: 'cover', borderRadius: 4 }} />
          <span style={{ flex: 1, color: 'rgba(255,255,255,0.7)' }}>
            {referenceImage.analyzing ? 'Analyzing reference...' : 'Reference loaded'}
          </span>
          <button onClick={clearReferenceImage} style={{ background: 'none', border: 'none', color: 'rgba(255,255,255,0.4)', cursor: 'pointer', fontSize: 12 }}>
            <i className="fa-solid fa-xmark" />
          </button>
        </div>
      )}

      <div className={styles.chatMessages}>
        {messages.map((msg, i) => (
          <div key={i} className={`${styles.message} ${msg.role === 'user' ? styles.userMessage : styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className={msg.role === 'user' ? 'fa-solid fa-user' : 'fa-solid fa-wand-magic-sparkles'} />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.messageText}>
                {msg.role === 'assistant'
                  ? renderMessageContent(msg.content, handleApply)
                  : msg.content
                }
              </div>
              <div className={styles.messageTime}>{formatTime(msg.timestamp)}</div>
            </div>
          </div>
        ))}

        {isLoading && (
          <div className={`${styles.message} ${styles.assistantMessage}`}>
            <div className={styles.messageIcon}>
              <i className="fa-solid fa-wand-magic-sparkles" />
            </div>
            <div className={styles.messageContent}>
              <div className={styles.typingIndicator}>
                <span /><span /><span />
              </div>
            </div>
          </div>
        )}

        {error && (
          <div className={styles.chatError}>
            <i className="fa-solid fa-exclamation-triangle" />
            <span>{error}</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      <div className={styles.chatInputArea}>
        <textarea
          ref={inputRef}
          className={styles.chatInput}
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Describe your plugin layout..."
          rows={1}
          disabled={isLoading}
        />
        <button
          className={styles.chatSendBtn}
          onClick={handleSend}
          disabled={!inputMessage.trim() || isLoading}
        >
          <i className="fa-solid fa-paper-plane" />
        </button>
      </div>
    </>
  );
};

export default CreatorChat;
