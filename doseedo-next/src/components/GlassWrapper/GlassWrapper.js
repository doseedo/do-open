import React, { useState, useEffect } from 'react';
import GlassCard from '../GlassCard/GlassCard';

/**
 * GlassWrapper - Conditionally wraps children with GlassCard from liquid-glass library
 * Only applies glass effect when theme is 'glass'
 */
function GlassWrapper({
  children,
  className = '',
  style = {},
  displacementScale = 80,
  blurAmount = 0.015,
  cornerRadius = 8,
  padding = '0px',
  shadowMode = false,
  onClick,
  ...props
}) {
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

  // Only apply glass effect when theme is 'glass'
  if (isGlassTheme) {
    return (
      <GlassCard
        className={className}
        style={style}
        displacementScale={displacementScale}
        blurAmount={blurAmount}
        cornerRadius={cornerRadius}
        padding={padding}
        shadowMode={shadowMode}
        onClick={onClick}
        {...props}
      >
        {children}
      </GlassCard>
    );
  }

  // Otherwise, render children without glass effect
  return <>{children}</>;
}

export default GlassWrapper;
