import React, { useState, useEffect } from 'react';
import { setColor, getColor, applyPreset, exportTheme, resetTheme } from '../../utils/themeManager';
import './ThemeEditor.css';

/**
 * ThemeEditor Component
 * Real-time color theme editor UI
 *
 * Add this component anywhere in your app to edit colors live:
 * <ThemeEditor />
 */
function ThemeEditor() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeTab, setActiveTab] = useState('primary');
  const [refreshKey, setRefreshKey] = useState(0);
  const [isGlassMode, setIsGlassMode] = useState(document.body.classList.contains('theme-glass'));

  // Load saved glass settings on mount
  useEffect(() => {
    const loadGlassSettings = () => {
      const savedSettings = localStorage.getItem('glassThemeSettings');
      if (savedSettings && document.body.classList.contains('theme-glass')) {
        try {
          const settings = JSON.parse(savedSettings);
          console.log('[ThemeEditor] Loading saved glass settings:', settings);

          // Apply saved settings to body
          Object.entries(settings).forEach(([key, value]) => {
            document.body.style.setProperty(key, value);
          });
        } catch (e) {
          console.error('[ThemeEditor] Failed to load glass settings:', e);
        }
      }
    };

    loadGlassSettings();
  }, []);

  // Check for glass mode changes
  useEffect(() => {
    const checkGlassMode = () => {
      setIsGlassMode(document.body.classList.contains('theme-glass'));
    };
    window.addEventListener('themeChanged', checkGlassMode);
    return () => window.removeEventListener('themeChanged', checkGlassMode);
  }, []);

  // Color groups for organized editing
  const colorGroups = {
    primary: [
      { name: 'Primary Blue', var: '--color-primary-blue', description: 'Main brand color, waveforms' },
      { name: 'Light Blue', var: '--color-primary-blue-light', description: 'Selections, highlights' },
      { name: 'Primary Purple', var: '--color-primary-purple', description: 'Accent color' },
      { name: 'Purple Alt', var: '--color-primary-purple-alt', description: 'Light purple variant' },
      { name: 'Dark Purple', var: '--color-primary-purple-dark', description: 'Dark purple for gradients' },
    ],
    backgrounds: [
      { name: 'Darkest BG', var: '--color-bg-darkest' },
      { name: 'Dark BG', var: '--color-bg-dark' },
      { name: 'Medium BG', var: '--color-bg-medium' },
      { name: 'Light BG', var: '--color-bg-light' },
    ],
    borders: [
      { name: 'Dark Border', var: '--color-border-dark' },
      { name: 'Standard Border', var: '--color-border' },
      { name: 'Light Border', var: '--color-border-light' },
    ],
    text: [
      { name: 'Primary Text', var: '--color-text-primary' },
      { name: 'Secondary Text', var: '--color-text-secondary' },
      { name: 'Muted Text', var: '--color-text-muted' },
    ],
    buttons: [
      { name: 'Mute Button BG', var: '--color-button-mute-bg' },
      { name: 'Mute Button Text', var: '--color-button-mute-text' },
      { name: 'Solo Button BG', var: '--color-button-solo-bg' },
    ]
  };

  // Default theme gradient controls
  const defaultGradients = [
    { name: 'Primary Gradient (Hover)', var: '--gradient-primary', isGradient: true },
    { name: 'Primary Reverse (Selected)', var: '--gradient-primary-reverse', isGradient: true },
    { name: 'Dark Gradient', var: '--gradient-dark', isGradient: true },
  ];

  // Glass mode gradient controls
  const glassControls = {
    gradients: [
      { name: 'Normal Gradient', var: '--glass-gradient-normal', isGradient: true },
      { name: 'Hover Gradient', var: '--glass-gradient-hover', isGradient: true },
      { name: 'Active Gradient', var: '--glass-gradient-full', isGradient: true },
    ],
    opacity: [
      { name: 'Normal Opacity', var: '--glass-opacity-normal', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Hover Opacity', var: '--glass-opacity-hover', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Active Opacity', var: '--glass-opacity-active', type: 'range', min: 0, max: 1, step: 0.05 },
    ],
    blur: [
      { name: 'Normal Blur', var: '--glass-blur-normal', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
      { name: 'Hover Blur', var: '--glass-blur-hover', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
      { name: 'Active Blur', var: '--glass-blur-active', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
    ],
    glow: [
      { name: 'Normal Glow Intensity', var: '--glass-glow-normal', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Hover Glow Intensity', var: '--glass-glow-hover', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Active Glow Intensity', var: '--glass-glow-active', type: 'range', min: 0, max: 1, step: 0.05 },
    ],
    glowDetails: [
      { name: 'Glow Color', var: '--glass-glow-color', type: 'color' },
      { name: 'Glow Blur Radius', var: '--glass-glow-blur', type: 'range', min: 0, max: 50, step: 1, unit: 'px' },
      { name: 'Glow Spread', var: '--glass-glow-spread', type: 'range', min: -20, max: 20, step: 1, unit: 'px' },
      { name: 'Glow Offset X', var: '--glass-glow-offset-x', type: 'range', min: -20, max: 20, step: 1, unit: 'px' },
      { name: 'Glow Offset Y', var: '--glass-glow-offset-y', type: 'range', min: -20, max: 20, step: 1, unit: 'px' },
    ],
    trackNormal: [
      { name: 'Normal Track Gradient', var: '--glass-track-normal-gradient', isGradient: true },
      { name: 'Normal Track Opacity', var: '--glass-track-normal-opacity', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Normal Track Blur', var: '--glass-track-normal-blur', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
      { name: 'Normal Track Speed', var: '--glass-track-normal-speed', type: 'range', min: 1, max: 30, step: 1, unit: 's' },
    ],
    trackSelected: [
      { name: 'Selected Track Gradient', var: '--glass-track-selected-gradient', isGradient: true },
      { name: 'Selected Track Opacity', var: '--glass-track-selected-opacity', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Selected Track Blur', var: '--glass-track-selected-blur', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
      { name: 'Selected Track Speed', var: '--glass-track-selected-speed', type: 'range', min: 1, max: 30, step: 1, unit: 's' },
    ],
    uploadContainer: [
      { name: 'Upload Normal Gradient', var: '--glass-upload-gradient-normal', isGradient: true },
      { name: 'Upload Hover Gradient', var: '--glass-upload-gradient-hover', isGradient: true },
      { name: 'Upload Normal Opacity', var: '--glass-upload-opacity-normal', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Upload Hover Opacity', var: '--glass-upload-opacity-hover', type: 'range', min: 0, max: 1, step: 0.05 },
      { name: 'Upload Normal Blur', var: '--glass-upload-blur-normal', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
      { name: 'Upload Hover Blur', var: '--glass-upload-blur-hover', type: 'range', min: 0, max: 10, step: 0.5, unit: 'px' },
    ]
  };

  const handleColorChange = (varName, value) => {
    // Special handling for glass glow color - convert hex to RGB
    if (varName === '--glass-glow-color') {
      const hex = value.replace('#', '');
      const r = parseInt(hex.substring(0, 2), 16);
      const g = parseInt(hex.substring(2, 4), 16);
      const b = parseInt(hex.substring(4, 6), 16);
      const rgbValue = `${r}, ${g}, ${b}`;
      document.body.style.setProperty(varName, rgbValue);
    } else {
      setColor(varName, value);
    }
  };

  const handleExport = () => {
    const theme = exportTheme();
    const json = JSON.stringify(theme, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'theme.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleReset = () => {
    if (window.confirm('Reset theme to defaults?')) {
      resetTheme();
      window.location.reload(); // Reload to apply default CSS
    }
  };

  const handleSaveGlassSettings = () => {
    if (!document.body.classList.contains('theme-glass')) {
      alert('Please switch to Glass theme first to save glass settings');
      return;
    }

    // Collect all glass-related CSS variables from body
    const glassVars = [
      '--glass-gradient-normal',
      '--glass-gradient-hover',
      '--glass-gradient-full',
      '--glass-opacity-normal',
      '--glass-opacity-hover',
      '--glass-opacity-active',
      '--glass-blur-normal',
      '--glass-blur-hover',
      '--glass-blur-active',
      '--glass-glow-normal',
      '--glass-glow-hover',
      '--glass-glow-active',
      '--glass-glow-color',
      '--glass-glow-blur',
      '--glass-glow-spread',
      '--glass-glow-offset-x',
      '--glass-glow-offset-y',
      '--glass-upload-gradient-normal',
      '--glass-upload-gradient-hover',
      '--glass-upload-opacity-normal',
      '--glass-upload-opacity-hover',
      '--glass-upload-blur-normal',
      '--glass-upload-blur-hover',
      '--glass-track-normal-gradient',
      '--glass-track-normal-opacity',
      '--glass-track-normal-blur',
      '--glass-track-normal-speed',
      '--glass-track-selected-gradient',
      '--glass-track-selected-opacity',
      '--glass-track-selected-blur',
      '--glass-track-selected-speed'
    ];

    const settings = {};
    const bodyStyles = getComputedStyle(document.body);

    glassVars.forEach(varName => {
      const value = bodyStyles.getPropertyValue(varName).trim();
      if (value) {
        settings[varName] = value;
      }
    });

    localStorage.setItem('glassThemeSettings', JSON.stringify(settings));
    console.log('[ThemeEditor] Saved glass settings:', settings);
    alert('Glass theme settings saved! They will load automatically when you switch to glass theme.');
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        className="theme-editor-toggle"
        title="Open Theme Editor"
      >
        🎨
      </button>
    );
  }

  return (
    <div className="theme-editor">
      <div className="theme-editor-header">
        <h3>Theme Editor</h3>
        <button onClick={() => setIsOpen(false)} className="theme-editor-close">×</button>
      </div>

      {/* Preset Themes */}
      <div className="theme-editor-section">
        <h4>Quick Presets</h4>
        <div className="theme-presets">
          <button onClick={() => { applyPreset('default'); setRefreshKey(k => k + 1); }}>Default</button>
          <button onClick={() => { applyPreset('ocean'); setRefreshKey(k => k + 1); }}>Ocean</button>
          <button onClick={() => { applyPreset('sunset'); setRefreshKey(k => k + 1); }}>Sunset</button>
          <button onClick={() => { applyPreset('forest'); setRefreshKey(k => k + 1); }}>Forest</button>
          <button onClick={() => { applyPreset('neon'); setRefreshKey(k => k + 1); }}>Neon</button>
          <button onClick={() => {
            applyPreset('glass');
            // Load saved glass settings after a short delay to let preset apply
            setTimeout(() => {
              const savedSettings = localStorage.getItem('glassThemeSettings');
              if (savedSettings) {
                try {
                  const settings = JSON.parse(savedSettings);
                  Object.entries(settings).forEach(([key, value]) => {
                    document.body.style.setProperty(key, value);
                  });
                } catch (e) {
                  console.error('[ThemeEditor] Failed to load glass settings:', e);
                }
              }
            }, 100);
            setRefreshKey(k => k + 1);
          }}>Glass</button>
          <button onClick={() => { applyPreset('monochrome'); setRefreshKey(k => k + 1); }}>Mono</button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="theme-editor-tabs">
        {Object.keys(colorGroups).map(tab => (
          <button
            key={tab}
            className={activeTab === tab ? 'active' : ''}
            onClick={() => setActiveTab(tab)}
          >
            {tab.charAt(0).toUpperCase() + tab.slice(1)}
          </button>
        ))}
      </div>

      {/* Color Inputs */}
      <div className="theme-editor-colors">
        {colorGroups[activeTab].map(color => (
          <ColorInput
            key={`${color.var}-${refreshKey}`}
            label={color.name}
            varName={color.var}
            description={color.description}
            onChange={handleColorChange}
          />
        ))}
      </div>

      {/* Default Theme Gradients */}
      {!isGlassMode && (
        <div className="theme-editor-section">
          <h4>🎨 Theme Gradients</h4>
          <div className="glass-control-group">
            {defaultGradients.map(control => (
              <GradientInput
                key={`${control.var}-${refreshKey}`}
                label={control.name}
                varName={control.var}
                onChange={handleColorChange}
              />
            ))}
          </div>
        </div>
      )}

      {/* Glass Mode Controls */}
      {isGlassMode && (
        <div className="theme-editor-section glass-controls">
          <h4>🔮 Liquid Glass Controls</h4>

          <div className="glass-control-group">
            <h5>Gradient Colors</h5>
            {glassControls.gradients.map(control => (
              <GradientInput
                key={`${control.var}-${refreshKey}`}
                label={control.name}
                varName={control.var}
                onChange={handleColorChange}
              />
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Opacity Controls</h5>
            {glassControls.opacity.map(control => (
              <SliderInput
                key={`${control.var}-${refreshKey}`}
                label={control.name}
                varName={control.var}
                min={control.min}
                max={control.max}
                step={control.step}
                unit={control.unit}
                onChange={handleColorChange}
              />
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Blur Controls</h5>
            {glassControls.blur.map(control => (
              <SliderInput
                key={`${control.var}-${refreshKey}`}
                label={control.name}
                varName={control.var}
                min={control.min}
                max={control.max}
                step={control.step}
                unit={control.unit || ''}
                onChange={handleColorChange}
              />
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Glow Intensity</h5>
            {glassControls.glow.map(control => (
              <SliderInput
                key={`${control.var}-${refreshKey}`}
                label={control.name}
                varName={control.var}
                min={control.min}
                max={control.max}
                step={control.step}
                onChange={handleColorChange}
              />
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Glow Details</h5>
            {glassControls.glowDetails.map(control => (
              control.type === 'color' ? (
                <ColorInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  onChange={handleColorChange}
                />
              ) : (
                <SliderInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  unit={control.unit || ''}
                  onChange={handleColorChange}
                />
              )
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Normal Track Settings</h5>
            {glassControls.trackNormal.map(control => (
              control.isGradient ? (
                <GradientInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  onChange={handleColorChange}
                />
              ) : (
                <SliderInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  unit={control.unit || ''}
                  onChange={handleColorChange}
                />
              )
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Selected Track Settings</h5>
            {glassControls.trackSelected.map(control => (
              control.isGradient ? (
                <GradientInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  onChange={handleColorChange}
                />
              ) : (
                <SliderInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  unit={control.unit || ''}
                  onChange={handleColorChange}
                />
              )
            ))}
          </div>

          <div className="glass-control-group">
            <h5>Upload Container Settings</h5>
            {glassControls.uploadContainer.map(control => (
              control.isGradient ? (
                <GradientInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  onChange={handleColorChange}
                />
              ) : (
                <SliderInput
                  key={`${control.var}-${refreshKey}`}
                  label={control.name}
                  varName={control.var}
                  min={control.min}
                  max={control.max}
                  step={control.step}
                  unit={control.unit || ''}
                  onChange={handleColorChange}
                />
              )
            ))}
          </div>
        </div>
      )}

      {/* Actions */}
      <div className="theme-editor-actions">
        <button onClick={handleExport} className="btn-export">Export Theme</button>
        <button onClick={handleReset} className="btn-reset">Reset</button>
        {isGlassMode && (
          <button onClick={handleSaveGlassSettings} className="btn-save-glass">Save Glass Settings</button>
        )}
      </div>
    </div>
  );
}

function ColorInput({ label, varName, description, onChange }) {
  const [value, setValue] = useState('');

  useEffect(() => {
    // Special handling for RGB values (like glow color)
    if (varName === '--glass-glow-color') {
      const rgbValue = getComputedStyle(document.body).getPropertyValue(varName).trim();
      if (rgbValue && rgbValue.includes(',')) {
        // Convert RGB to hex
        const [r, g, b] = rgbValue.split(',').map(v => parseInt(v.trim()));
        const hex = `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
        setValue(hex);
      } else {
        setValue('#667eea');
      }
    } else {
      setValue(getColor(varName));
    }
  }, [varName]);

  const handleChange = (e) => {
    const newValue = e.target.value;
    setValue(newValue);
    onChange(varName, newValue);
  };

  return (
    <div className="color-input">
      <label>
        <span className="color-input-label">{label}</span>
        {description && <span className="color-input-desc">{description}</span>}
      </label>
      <div className="color-input-controls">
        <input
          type="color"
          value={value.startsWith('#') ? value : '#667eea'}
          onChange={handleChange}
          className="color-picker"
        />
        <input
          type="text"
          value={value}
          onChange={handleChange}
          className="color-text"
          placeholder="#667eea"
        />
      </div>
    </div>
  );
}

function GradientInput({ label, varName, onChange }) {
  const [colors, setColors] = useState([]);

  useEffect(() => {
    // Read from :root for default theme, body for glass theme
    const element = document.querySelector(':root');
    const gradientValue = getComputedStyle(element).getPropertyValue(varName).trim();
    console.log(`[ThemeEditor] Reading ${varName}:`, gradientValue);
    if (gradientValue) {
      // Extract hex colors from gradient string (handles both hex and rgb)
      const hexMatches = gradientValue.match(/#[0-9a-fA-F]{6}/g);
      const rgbMatches = gradientValue.match(/rgb\((\d+),\s*(\d+),\s*(\d+)\)/g);

      let extractedColors = [];
      if (hexMatches && hexMatches.length > 0) {
        extractedColors = hexMatches;
      } else if (rgbMatches && rgbMatches.length > 0) {
        // Convert rgb to hex
        extractedColors = rgbMatches.map(rgb => {
          const [r, g, b] = rgb.match(/\d+/g).map(Number);
          return `#${((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1)}`;
        });
      }

      if (extractedColors.length > 0) {
        console.log(`[ThemeEditor] Extracted colors for ${varName}:`, extractedColors);
        setColors(extractedColors);
      } else {
        // Fallback defaults
        console.log(`[ThemeEditor] Using fallback colors for ${varName}`);
        setColors(['#667eea', '#764ba2']);
      }
    } else {
      // Fallback defaults if not set
      console.log(`[ThemeEditor] Using fallback colors for ${varName}`);
      setColors(['#667eea', '#764ba2']);
    }
  }, [varName]);

  const handleColorChange = (index, newColor) => {
    const newColors = [...colors];
    newColors[index] = newColor;
    setColors(newColors);

    // Build gradient string
    const stops = newColors.map((c, i) => `${c} ${Math.round((i / (newColors.length - 1)) * 100)}%`).join(', ');
    const gradient = `linear-gradient(135deg, ${stops})`;

    // Apply to CSS variable - use :root for default theme, body for glass theme
    console.log(`[ThemeEditor] Setting ${varName} to:`, gradient);

    // Set on both :root and body to ensure it works for all themes
    document.documentElement.style.setProperty(varName, gradient);
    document.body.style.setProperty(varName, gradient);

    // Force repaint by toggling glass theme class
    const body = document.body;
    const hadGlass = body.classList.contains('theme-glass');
    if (hadGlass) {
      body.classList.remove('theme-glass');
      void body.offsetHeight; // Force reflow
      body.classList.add('theme-glass');
    }
  };

  const addColor = () => {
    if (colors.length < 8) {
      setColors([...colors, '#8b7cf6']);
    }
  };

  const removeColor = (index) => {
    if (colors.length > 2) {
      const newColors = colors.filter((_, i) => i !== index);
      setColors(newColors);
      const stops = newColors.map((c, i) => `${c} ${Math.round((i / (newColors.length - 1)) * 100)}%`).join(', ');
      const gradient = `linear-gradient(135deg, ${stops})`;
      document.documentElement.style.setProperty(varName, gradient);
    }
  };

  if (colors.length === 0) {
    return <div className="gradient-input">Loading...</div>;
  }

  return (
    <div className="gradient-input">
      <label className="gradient-input-label">{label}</label>
      <div className="gradient-preview" style={{ background: `linear-gradient(135deg, ${colors.join(', ')})` }} />
      <div className="gradient-colors">
        {colors.map((color, index) => (
          <div key={index} className="gradient-color-stop">
            <input
              type="color"
              value={color}
              onChange={(e) => handleColorChange(index, e.target.value)}
              className="gradient-color-picker"
            />
            <input
              type="text"
              value={color}
              onChange={(e) => handleColorChange(index, e.target.value)}
              className="gradient-color-text"
            />
            {colors.length > 2 && (
              <button onClick={() => removeColor(index)} className="gradient-remove">×</button>
            )}
          </div>
        ))}
      </div>
      {colors.length < 8 && (
        <button onClick={addColor} className="gradient-add">+ Add Color</button>
      )}
    </div>
  );
}

function SliderInput({ label, varName, min, max, step, unit = '', onChange }) {
  // Set default values based on variable name
  const getDefaultValue = () => {
    if (varName.includes('opacity-normal')) return 0.45;
    if (varName.includes('opacity-hover')) return 0.55;
    if (varName.includes('opacity-active')) return 0.65;
    if (varName.includes('blur-normal')) return 3;
    if (varName.includes('blur-hover')) return 4;
    if (varName.includes('blur-active')) return 5;
    if (varName.includes('glow-normal')) return 0.15;
    if (varName.includes('glow-hover')) return 0.25;
    if (varName.includes('glow-active')) return 0.4;
    if (varName.includes('glow-blur')) return 12;
    if (varName.includes('glow-spread')) return 0;
    if (varName.includes('glow-offset-x')) return 0;
    if (varName.includes('glow-offset-y')) return 4;
    return min;
  };

  const [value, setValue] = useState(getDefaultValue());

  useEffect(() => {
    // Read from body element since glass variables are scoped to body.theme-glass
    const cssValue = getComputedStyle(document.body).getPropertyValue(varName);
    if (cssValue && cssValue.trim() !== '') {
      const numValue = parseFloat(cssValue);
      if (!isNaN(numValue)) {
        setValue(numValue);
      } else {
        // Initialize with default if not set
        const defaultVal = getDefaultValue();
        setValue(defaultVal);
        document.body.style.setProperty(varName, `${defaultVal}${unit}`);
      }
    } else {
      // Initialize with default if not set
      const defaultVal = getDefaultValue();
      setValue(defaultVal);
      document.body.style.setProperty(varName, `${defaultVal}${unit}`);
    }
  }, [varName, unit]);

  const handleChange = (e) => {
    const newValue = parseFloat(e.target.value);
    setValue(newValue);
    // Set on body element since glass variables are scoped to body.theme-glass
    document.body.style.setProperty(varName, `${newValue}${unit}`);

    // Force repaint by toggling glass theme class
    const body = document.body;
    const hadGlass = body.classList.contains('theme-glass');
    if (hadGlass) {
      body.classList.remove('theme-glass');
      void body.offsetHeight; // Force reflow
      body.classList.add('theme-glass');
    }
  };

  return (
    <div className="slider-input">
      <label className="slider-input-label">
        <span>{label}</span>
        <span className="slider-value">{value.toFixed(unit === 'px' ? 1 : 2)}{unit}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        className="slider"
      />
    </div>
  );
}

export default ThemeEditor;
