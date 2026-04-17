/**
 * Plugin Creator Theme Presets
 * Each theme provides semantic color tokens + visual style preferences
 */
import { generateKnobSVG, generateSliderSVG, generateButtonSVG } from './svgComponentLibrary';
import { generateThemeVariants } from './themeGenerator';

export const PLUGIN_THEMES = [
  {
    id: 'midnight',
    name: 'Midnight',
    icon: 'fa-moon',
    preview: '#0a0a1a',
    bgColor: '#0a0a1a',
    titleBarColor: '#14142e',
    accentColor: '#00d4ff',
    textColor: '#e0e8ff',
    panelColor: 'rgba(0,212,255,0.04)',
    panelBorder: 'rgba(0,212,255,0.12)',
    knobStyle: 'metallic',
    knobSvgStyle: 'dome-line', sliderSvgStyle: 'channel-fader', buttonSvgStyle: 'toggle-led',
    dallePrompt: 'deep space dark blue-black background with subtle nebula glow at edges, faint star field, completely empty dark center area for control placement, no text no buttons no UI',
  },
  {
    id: 'analog-warmth',
    name: 'Analog Warmth',
    icon: 'fa-fire',
    preview: '#2a1f1a',
    bgColor: '#2a1f1a',
    titleBarColor: '#3d2e1e',
    accentColor: '#ff9800',
    textColor: '#f5e6c8',
    panelColor: 'rgba(255,152,0,0.04)',
    panelBorder: 'rgba(255,152,0,0.12)',
    knobStyle: 'vintage',
    knobSvgStyle: 'chicken-head', sliderSvgStyle: 'vintage-slot', buttonSvgStyle: 'vintage-toggle',
    dallePrompt: 'warm wood grain texture panel background with rich mahogany tones, subtle leather texture accent, vintage analog studio feel, empty center for controls, no knobs no buttons no text',
  },
  {
    id: 'neon-pulse',
    name: 'Neon Pulse',
    icon: 'fa-bolt',
    preview: '#0d0d0d',
    bgColor: '#0d0d0d',
    titleBarColor: '#1a0a1f',
    accentColor: '#ff00ff',
    textColor: '#f0c0ff',
    panelColor: 'rgba(255,0,255,0.04)',
    panelBorder: 'rgba(255,0,255,0.15)',
    knobStyle: 'led-ring',
    knobSvgStyle: 'glass-ring', sliderSvgStyle: 'led-bar', buttonSvgStyle: 'pill-glow',
    dallePrompt: 'pitch black background with neon magenta and cyan light streaks at edges, cyberpunk circuit board traces barely visible, dark empty center area, no UI elements no text',
  },
  {
    id: 'clean-studio',
    name: 'Clean Studio',
    icon: 'fa-sun',
    preview: '#f0f0f0',
    bgColor: '#f0f0f0',
    titleBarColor: '#e0e0e0',
    accentColor: '#333333',
    textColor: '#1a1a1a',
    panelColor: 'rgba(0,0,0,0.03)',
    panelBorder: 'rgba(0,0,0,0.08)',
    knobStyle: 'minimal',
    knobSvgStyle: 'minimal-dot', sliderSvgStyle: 'minimal-track', buttonSvgStyle: 'pill-glow',
    dallePrompt: 'clean white brushed aluminum panel background, subtle light gray gradient, professional studio equipment aesthetic, empty center area, no controls no text no buttons',
  },
  {
    id: 'outrun',
    name: 'Outrun',
    icon: 'fa-car',
    preview: '#1a0030',
    bgColor: '#1a0030',
    titleBarColor: '#2d0050',
    accentColor: '#ff1493',
    textColor: '#ff88cc',
    panelColor: 'rgba(255,20,147,0.04)',
    panelBorder: 'rgba(255,20,147,0.15)',
    knobStyle: 'led-ring',
    knobSvgStyle: 'led-ring', sliderSvgStyle: 'led-bar', buttonSvgStyle: 'pill-glow',
    dallePrompt: 'retro synthwave sunset gradient background, purple to pink to orange horizon line in lower third, perspective grid floor fading into darkness, upper area empty and dark, no text no UI',
  },
  {
    id: 'arctic',
    name: 'Arctic',
    icon: 'fa-snowflake',
    preview: '#0e1a2e',
    bgColor: '#0e1a2e',
    titleBarColor: '#162840',
    accentColor: '#4fc3f7',
    textColor: '#b8e0f8',
    panelColor: 'rgba(79,195,247,0.04)',
    panelBorder: 'rgba(79,195,247,0.1)',
    knobStyle: 'metallic',
    knobSvgStyle: 'chrome-cap', sliderSvgStyle: 'channel-fader', buttonSvgStyle: 'toggle-led',
    dallePrompt: 'dark icy blue-black background with subtle frost crystal texture at edges, arctic aurora borealis glow in upper portion, dark empty center area for controls, no text no buttons',
  },
  {
    id: 'emerald',
    name: 'Emerald',
    icon: 'fa-gem',
    preview: '#0a1a12',
    bgColor: '#0a1a12',
    titleBarColor: '#142e1e',
    accentColor: '#00e676',
    textColor: '#b8f0d0',
    panelColor: 'rgba(0,230,118,0.04)',
    panelBorder: 'rgba(0,230,118,0.1)',
    knobStyle: 'metallic',
    knobSvgStyle: 'dome-line', sliderSvgStyle: 'channel-fader', buttonSvgStyle: 'toggle-led',
    dallePrompt: 'dark green-black background with subtle emerald crystal refraction pattern at edges, deep forest atmospheric glow, empty dark center area, no UI elements no text',
  },
  {
    id: 'rust',
    name: 'Rust & Steel',
    icon: 'fa-gear',
    preview: '#1c1210',
    bgColor: '#1c1210',
    titleBarColor: '#2e1e18',
    accentColor: '#e65100',
    textColor: '#d4a88c',
    panelColor: 'rgba(230,81,0,0.04)',
    panelBorder: 'rgba(230,81,0,0.12)',
    knobStyle: 'vintage',
    knobSvgStyle: 'hex-bolt', sliderSvgStyle: 'slot-thumb', buttonSvgStyle: 'rocker',
    dallePrompt: 'dark rusted steel panel background with rivets and weathered metal texture, industrial grunge aesthetic, dark empty center area for control placement, no knobs no text',
  },
  {
    id: 'vapor',
    name: 'Vaporwave',
    icon: 'fa-palette',
    preview: '#1a0828',
    bgColor: '#1a0828',
    titleBarColor: '#2e1040',
    accentColor: '#e040fb',
    textColor: '#e8b0ff',
    panelColor: 'rgba(224,64,251,0.04)',
    panelBorder: 'rgba(224,64,251,0.12)',
    knobStyle: 'led-ring',
    knobSvgStyle: 'glass-ring', sliderSvgStyle: 'led-bar', buttonSvgStyle: 'pill-glow',
    dallePrompt: 'vaporwave aesthetic background with pastel pink and purple gradient, subtle Greek statue silhouette at edge, palm tree shadows, dreamy haze, empty center area, no text no UI controls',
  },
  {
    id: 'military',
    name: 'Military',
    icon: 'fa-shield',
    preview: '#1a1c14',
    bgColor: '#1a1c14',
    titleBarColor: '#2a2e20',
    accentColor: '#8bc34a',
    textColor: '#c8d8a0',
    panelColor: 'rgba(139,195,74,0.04)',
    panelBorder: 'rgba(139,195,74,0.1)',
    knobStyle: 'default',
    knobSvgStyle: 'skirted-pointer', sliderSvgStyle: 'slot-thumb', buttonSvgStyle: 'rocker',
    dallePrompt: 'dark olive green military equipment panel background with subtle stencil markings at edges, matte metal texture, tactical hardware aesthetic, empty center area, no text no controls',
  },
];

/**
 * Apply a theme to the plugin config and optionally recolor all components
 * Returns { pluginConfig updates, component color remapping }
 */
export function applyTheme(theme, pluginConfig, components) {
  const configUpdate = {
    bgColor: theme.bgColor,
    titleBarColor: theme.titleBarColor,
  };

  const recoloredComponents = components.map(comp => {
    const updated = { ...comp };
    // Recolor interactive components with accent
    if (['knob', 'slider', 'xy-pad', 'meter', 'led', 'waveform'].includes(comp.type)) {
      updated.color = theme.accentColor;
    }
    // Recolor text with text color
    if (['label'].includes(comp.type)) {
      updated.color = theme.textColor;
    }
    // Recolor buttons with accent
    if (['button', 'dropdown'].includes(comp.type)) {
      updated.color = theme.accentColor;
    }
    // Recolor panels with theme panel colors
    if (comp.type === 'panel') {
      updated.bgColor = theme.panelColor;
      updated.borderColor = theme.panelBorder;
    }
    // Apply visual style to interactive controls
    if (['knob', 'slider', 'button'].includes(comp.type)) {
      updated.knobStyle = theme.knobStyle;
      const uid = (comp.id || 'c').replace(/[^a-zA-Z0-9]/g, '').slice(0, 12);
      const params = {
        width: comp.width || 60, height: comp.height || 60,
        bodyColor: comp.bodyColor || updated.color || '#333',
        indicatorColor: comp.indicatorColor || updated.color || '#fff',
        accentColor: comp.accentColor || updated.color || '#888',
        uid, label: comp.label || '',
      };
      if (comp.type === 'knob' && theme.knobSvgStyle) {
        updated.svgStyle = theme.knobSvgStyle;
        updated.svg = generateKnobSVG(theme.knobSvgStyle, params);
      } else if (comp.type === 'slider' && theme.sliderSvgStyle) {
        updated.svgStyle = theme.sliderSvgStyle;
        updated.svg = generateSliderSVG(theme.sliderSvgStyle, params);
      } else if (comp.type === 'button' && theme.buttonSvgStyle) {
        updated.svgStyle = theme.buttonSvgStyle;
        updated.svg = generateButtonSVG(theme.buttonSvgStyle, params);
      }
    }
    return updated;
  });

  return { configUpdate, recoloredComponents };
}

/**
 * Generate theme variants from a single accent color.
 * Returns array of themes (one per aesthetic category) compatible with PLUGIN_THEMES format.
 */
export function generateThemesFromColor(accentColor) {
  return generateThemeVariants(accentColor);
}
