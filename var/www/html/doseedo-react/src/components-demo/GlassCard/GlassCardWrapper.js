import React, { useState, useEffect } from 'react';
import { GlassCard } from 'react-glass-ui';
import './GlassCardWrapper.css';

/**
 * GlassCardWrapper - Conditionally wraps content with GlassCard
 * Only applies glass effect when theme is 'glass'
 */
function GlassCardWrapper({ children, className = '', style = {}, ...props }) {
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
      <div className="glass-card-container">
        <GlassCard
          blur={25}
          distortion={20}
          chromaticAberration={0}
          saturation={170}
          brightness={110}
          borderRadius={12}
          borderSize={1.5}
          borderColor="#ffffff"
          borderOpacity={0.25}
          backgroundColor="rgba(255, 255, 255, 0.08)"
          backgroundOpacity={0.08}
          lightEffect={true}
          lightIntensity={0.2}
          lightColor="#ffffff"
          innerLightIntensity={0.15}
          innerLightColor="#ffffff"
          glowEffect={true}
          glowIntensity={0.15}
          glowColor="#764ba2"
          style={{ width: '100%', ...style }}
          {...props}
        >
          {children}
        </GlassCard>
      </div>
    );
  }

  // Otherwise, render children with normal styling
  return (
    <div className={className} style={style} {...props}>
      {children}
    </div>
  );
}

export default GlassCardWrapper;
