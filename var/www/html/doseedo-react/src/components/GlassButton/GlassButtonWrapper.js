import React, { useState, useEffect } from 'react';
import { GlassButton } from 'react-glass-ui';
import './GlassButtonWrapper.css';

/**
 * GlassButtonWrapper - Conditionally wraps button content with GlassButton
 * Only applies glass effect when theme is 'glass'
 */
function GlassButtonWrapper({ children, className = '', style = {}, ...props }) {
  const [isGlassTheme, setIsGlassTheme] = useState(false);

  useEffect(() => {
    // Check initial theme
    const checkTheme = () => {
      setIsGlassTheme(document.body.classList.contains('theme-glass'));
    };

    checkTheme();

    // Listen for theme changes
    const handleThemeChange = () => {
      checkTheme();
    };

    window.addEventListener('themeChanged', handleThemeChange);

    return () => {
      window.removeEventListener('themeChanged', handleThemeChange);
    };
  }, []);

  // Always render normal button - glass effect applied via CSS
  return (
    <button className={className} style={style} {...props}>
      {children}
    </button>
  );
}

export default GlassButtonWrapper;
