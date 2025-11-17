import { useState, useEffect } from 'react';

/**
 * Hook to get a CSS variable color that updates when the theme changes
 * @param {string} varName - CSS variable name (e.g., '--color-primary-blue')
 * @param {string} fallback - Fallback color if variable not found
 * @returns {string} Current color value
 */
export function useThemeColor(varName, fallback = '#667eea') {
  const [color, setColor] = useState(() => {
    const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
    return value || fallback;
  });

  useEffect(() => {
    // Function to update color
    const updateColor = () => {
      const value = getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
      setColor(value || fallback);
    };

    // Update immediately
    updateColor();

    // Listen for theme changes via MutationObserver
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.type === 'attributes' && mutation.attributeName === 'style') {
          updateColor();
        }
      });
    });

    // Observe style attribute changes on document root
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['style']
    });

    // Also listen for custom theme change events
    const handleThemeChange = () => updateColor();
    window.addEventListener('themeChanged', handleThemeChange);

    return () => {
      observer.disconnect();
      window.removeEventListener('themeChanged', handleThemeChange);
    };
  }, [varName, fallback]);

  return color;
}
