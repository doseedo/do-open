import React, { useCallback } from 'react';

/**
 * Reusable Slider Component
 * Used for gain, volume, zoom, etc.
 */
const Slider = React.memo(({
  id,
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
  orientation = 'horizontal',
  label,
  className = ''
}) => {
  const handleChange = useCallback((e) => {
    const newValue = parseFloat(e.target.value);
    onChange(newValue);
  }, [onChange]);

  const sliderStyle = orientation === 'vertical'
    ? {
        WebkitAppearance: 'slider-vertical',
        writingMode: 'bt-lr',
        height: '100%'
      }
    : {};

  return (
    <div className={`slider-wrapper ${className}`}>
      {label && <label htmlFor={id}>{label}</label>}
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
        style={sliderStyle}
      />
    </div>
  );
});

Slider.displayName = 'Slider';

export default Slider;
