/**
 * Theme Manager
 * Centralized color management system for real-time theme switching
 *
 * Usage:
 * import { setColor, getColor, applyTheme, exportTheme } from './utils/themeManager';
 *
 * // Change a single color
 * setColor('--color-primary-blue', '#ff0000');
 *
 * // Apply a complete theme
 * applyTheme(customTheme);
 *
 * // Export current theme
 * const currentTheme = exportTheme();
 */

/**
 * Set a CSS variable color in real-time
 * @param {string} varName - CSS variable name (e.g., '--color-primary-blue')
 * @param {string} value - Color value (e.g., '#667eea' or 'rgba(102, 126, 234, 0.5)')
 */
export function setColor(varName, value) {
  document.documentElement.style.setProperty(varName, value);
  console.log(`🎨 Theme: ${varName} = ${value}`);
  // Dispatch custom event to notify components
  window.dispatchEvent(new CustomEvent('themeChanged', { detail: { varName, value } }));
}

/**
 * Get the current value of a CSS variable
 * @param {string} varName - CSS variable name
 * @returns {string} The current value
 */
export function getColor(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
}

/**
 * Apply an entire theme object
 * @param {Object} theme - Object with CSS variable names as keys and color values as values
 */
export function applyTheme(theme) {
  Object.entries(theme).forEach(([varName, value]) => {
    setColor(varName, value);
  });
  console.log('🎨 Theme applied:', theme);
}

/**
 * Export the current theme as a JSON object
 * @returns {Object} Current theme colors
 */
export function exportTheme() {
  const rootStyles = getComputedStyle(document.documentElement);
  const theme = {};

  // Get all CSS variables from :root
  const cssVars = Array.from(document.styleSheets)
    .flatMap(sheet => {
      try {
        return Array.from(sheet.cssRules);
      } catch (e) {
        return [];
      }
    })
    .filter(rule => rule.selectorText === ':root')
    .flatMap(rule => {
      return Array.from(rule.style)
        .filter(prop => prop.startsWith('--'));
    });

  // Get unique variables
  const uniqueVars = [...new Set(cssVars)];

  uniqueVars.forEach(varName => {
    theme[varName] = rootStyles.getPropertyValue(varName).trim();
  });

  return theme;
}

/**
 * Reset theme to default values (reload colors.css)
 */
export function resetTheme() {
  // Remove all inline styles from :root
  const root = document.documentElement;
  const inlineStyles = root.style;

  // Remove all CSS variable overrides
  Array.from(inlineStyles).forEach(prop => {
    if (prop.startsWith('--')) {
      root.style.removeProperty(prop);
    }
  });

  console.log('🎨 Theme reset to defaults');
}

/**
 * Predefined theme presets
 */
export const themePresets = {
  default: {
    '--color-primary-blue': '#667eea',
    '--color-primary-blue-light': '#88a3f7',
    '--color-primary-purple': '#8b5cf6',
    '--color-primary-purple-alt': '#ba9cff',
    '--color-primary-purple-dark': '#764ba2',
    '--gradient-primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, #8b5cf6 0%, #667eea 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, #88a3f7 0%, #8b5cf6 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, #667eea 0%, #ba9cff 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, #764ba2 0%, #667eea 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(102, 126, 234, 0.08) 0%, rgba(118, 75, 162, 0.08) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(102, 126, 234, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(102, 126, 234, 0.15) 0%, rgba(118, 75, 162, 0.15) 100%)',
    '--color-primary-blue-10': 'rgba(102, 126, 234, 0.1)',
    '--color-primary-blue-20': 'rgba(102, 126, 234, 0.2)',
    '--color-primary-blue-30': 'rgba(102, 126, 234, 0.3)',
    '--color-primary-blue-50': 'rgba(102, 126, 234, 0.5)',
    '--color-button-solo-bg': '#667eea',
  },
  ocean: {
    '--color-primary-blue': '#00b4d8',
    '--color-primary-blue-light': '#48cae4',
    '--color-primary-purple': '#0077b6',
    '--color-primary-purple-alt': '#90e0ef',
    '--color-primary-purple-dark': '#023e8a',
    '--gradient-primary': 'linear-gradient(135deg, #00b4d8 0%, #023e8a 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, #0077b6 0%, #00b4d8 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, #48cae4 0%, #0077b6 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, #00b4d8 0%, #90e0ef 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, #023e8a 0%, #00b4d8 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(0, 180, 216, 0.08) 0%, rgba(2, 62, 138, 0.08) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(0, 180, 216, 0.1) 0%, rgba(0, 119, 182, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(0, 180, 216, 0.2) 0%, rgba(0, 119, 182, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(0, 180, 216, 0.15) 0%, rgba(2, 62, 138, 0.15) 100%)',
    '--color-primary-blue-10': 'rgba(0, 180, 216, 0.1)',
    '--color-primary-blue-20': 'rgba(0, 180, 216, 0.2)',
    '--color-primary-blue-30': 'rgba(0, 180, 216, 0.3)',
    '--color-primary-blue-50': 'rgba(0, 180, 216, 0.5)',
    '--color-button-solo-bg': '#00b4d8',
  },
  sunset: {
    '--color-primary-blue': '#ff6b6b',
    '--color-primary-blue-light': '#ff8787',
    '--color-primary-purple': '#ee5a6f',
    '--color-primary-purple-alt': '#ffa5a5',
    '--color-primary-purple-dark': '#c92a2a',
    '--gradient-primary': 'linear-gradient(135deg, #ff6b6b 0%, #c92a2a 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, #ee5a6f 0%, #ff6b6b 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, #ff8787 0%, #ee5a6f 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, #ff6b6b 0%, #ffa5a5 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, #c92a2a 0%, #ff6b6b 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(255, 107, 107, 0.08) 0%, rgba(238, 90, 111, 0.08) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(238, 90, 111, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(255, 107, 107, 0.2) 0%, rgba(238, 90, 111, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(255, 107, 107, 0.15) 0%, rgba(238, 90, 111, 0.15) 100%)',
    '--color-primary-blue-10': 'rgba(255, 107, 107, 0.1)',
    '--color-primary-blue-20': 'rgba(255, 107, 107, 0.2)',
    '--color-primary-blue-30': 'rgba(255, 107, 107, 0.3)',
    '--color-primary-blue-50': 'rgba(255, 107, 107, 0.5)',
    '--color-button-solo-bg': '#ff6b6b',
  },
  forest: {
    '--color-primary-blue': '#2d6a4f',
    '--color-primary-blue-light': '#52b788',
    '--color-primary-purple': '#1b4332',
    '--color-primary-purple-alt': '#74c69d',
    '--color-primary-purple-dark': '#081c15',
    '--gradient-primary': 'linear-gradient(135deg, #2d6a4f 0%, #081c15 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, #52b788 0%, #1b4332 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, #2d6a4f 0%, #74c69d 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, #081c15 0%, #2d6a4f 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(45, 106, 79, 0.08) 0%, rgba(27, 67, 50, 0.08) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(45, 106, 79, 0.1) 0%, rgba(27, 67, 50, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(45, 106, 79, 0.2) 0%, rgba(27, 67, 50, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(45, 106, 79, 0.15) 0%, rgba(27, 67, 50, 0.15) 100%)',
    '--color-primary-blue-10': 'rgba(45, 106, 79, 0.1)',
    '--color-primary-blue-20': 'rgba(45, 106, 79, 0.2)',
    '--color-primary-blue-30': 'rgba(45, 106, 79, 0.3)',
    '--color-primary-blue-50': 'rgba(45, 106, 79, 0.5)',
    '--color-button-solo-bg': '#2d6a4f',
  },
  neon: {
    '--color-primary-blue': '#00ff00',
    '--color-primary-blue-light': '#7fff00',
    '--color-primary-purple': '#00dd00',
    '--color-primary-purple-alt': '#66ff66',
    '--color-primary-purple-dark': '#00aa00',
    '--gradient-primary': 'linear-gradient(135deg, #00ff00 0%, #00aa00 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, #00dd00 0%, #00ff00 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, #7fff00 0%, #00dd00 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, #00ff00 0%, #66ff66 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, #00aa00 0%, #00ff00 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(0, 255, 0, 0.08) 0%, rgba(0, 170, 0, 0.08) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(0, 255, 0, 0.1) 0%, rgba(0, 170, 0, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(0, 255, 0, 0.2) 0%, rgba(0, 170, 0, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(0, 255, 0, 0.15) 0%, rgba(0, 170, 0, 0.15) 100%)',
    '--color-primary-blue-10': 'rgba(0, 255, 0, 0.1)',
    '--color-primary-blue-20': 'rgba(0, 255, 0, 0.2)',
    '--color-primary-blue-30': 'rgba(0, 255, 0, 0.3)',
    '--color-primary-blue-50': 'rgba(0, 255, 0, 0.5)',
    '--color-button-solo-bg': '#00ff00',
  },
  monochrome: {
    '--color-primary-blue': '#888888',
    '--color-primary-blue-light': '#aaaaaa',
    '--color-primary-purple': '#666666',
    '--color-primary-purple-alt': '#bbbbbb',
    '--color-primary-purple-dark': '#444444',
    '--gradient-primary': 'linear-gradient(135deg, #888888 0%, #444444 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, #666666 0%, #888888 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, #aaaaaa 0%, #666666 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, #888888 0%, #bbbbbb 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, #444444 0%, #888888 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(136, 136, 136, 0.08) 0%, rgba(102, 102, 102, 0.08) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(136, 136, 136, 0.1) 0%, rgba(102, 102, 102, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(136, 136, 136, 0.2) 0%, rgba(102, 102, 102, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(136, 136, 136, 0.15) 0%, rgba(102, 102, 102, 0.15) 100%)',
    '--color-primary-blue-10': 'rgba(136, 136, 136, 0.1)',
    '--color-primary-blue-20': 'rgba(136, 136, 136, 0.2)',
    '--color-primary-blue-30': 'rgba(136, 136, 136, 0.3)',
    '--color-primary-blue-50': 'rgba(136, 136, 136, 0.5)',
    '--color-button-solo-bg': '#888888',
  },
  glass: {
    '--color-primary-blue': '#b8c0ff',
    '--color-primary-blue-light': '#d8ddff',
    '--color-primary-purple': '#c8b8ff',
    '--color-primary-purple-alt': '#e0d8ff',
    '--color-primary-purple-dark': '#a090d8',
    '--gradient-primary': 'linear-gradient(135deg, rgba(184, 192, 255, 0.2) 0%, rgba(160, 144, 216, 0.15) 100%)',
    '--gradient-primary-reverse': 'linear-gradient(135deg, rgba(200, 184, 255, 0.2) 0%, rgba(184, 192, 255, 0.2) 100%)',
    '--gradient-variant-1': 'linear-gradient(135deg, rgba(216, 221, 255, 0.2) 0%, rgba(200, 184, 255, 0.15) 100%)',
    '--gradient-variant-2': 'linear-gradient(135deg, rgba(184, 192, 255, 0.2) 0%, rgba(224, 216, 255, 0.15) 100%)',
    '--gradient-variant-3': 'linear-gradient(135deg, rgba(160, 144, 216, 0.2) 0%, rgba(184, 192, 255, 0.2) 100%)',
    '--gradient-glow-subtle': 'linear-gradient(135deg, rgba(184, 192, 255, 0.08) 0%, rgba(160, 144, 216, 0.05) 100%)',
    '--gradient-glow-10': 'linear-gradient(135deg, rgba(184, 192, 255, 0.1) 0%, rgba(200, 184, 255, 0.1) 100%)',
    '--gradient-glow-20': 'linear-gradient(135deg, rgba(184, 192, 255, 0.2) 0%, rgba(200, 184, 255, 0.2) 100%)',
    '--gradient-glow-hover': 'linear-gradient(135deg, rgba(184, 192, 255, 0.15) 0%, rgba(160, 144, 216, 0.12) 100%)',
    '--color-primary-blue-10': 'rgba(184, 192, 255, 0.1)',
    '--color-primary-blue-20': 'rgba(184, 192, 255, 0.2)',
    '--color-primary-blue-30': 'rgba(184, 192, 255, 0.3)',
    '--color-primary-blue-50': 'rgba(184, 192, 255, 0.5)',
    '--color-button-solo-bg': '#b8c0ff',
  }
};

/**
 * Apply a preset theme
 * @param {string} presetName - Name of the preset ('ocean', 'sunset', etc.)
 */
export function applyPreset(presetName) {
  const preset = themePresets[presetName];
  if (preset) {
    applyTheme(preset);

    // Add/remove glass theme class on body
    if (presetName === 'glass') {
      document.body.classList.add('theme-glass');
    } else {
      document.body.classList.remove('theme-glass');
    }

    // Dispatch event to notify components of theme change
    window.dispatchEvent(new CustomEvent('themeChanged', { detail: { preset: presetName } }));
  } else {
    console.warn(`🎨 Theme preset "${presetName}" not found`);
  }
}
