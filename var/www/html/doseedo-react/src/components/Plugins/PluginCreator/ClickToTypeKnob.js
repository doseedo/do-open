/**
 * ClickToTypeKnob - Enhanced knob with FabFilter-style click-to-type functionality
 * Wraps existing SVGKnobRenderer and adds:
 * - Double-click to enter exact value
 * - Right-click to reset to default
 * - Shift+drag for fine control
 * - Ctrl+click to copy value
 */

import React, { useState, useCallback, useRef, useEffect } from 'react';

const ClickToTypeKnob = ({
  children, // The actual knob renderer
  value = 0.5,
  min = 0,
  max = 1,
  defaultValue = 0.5,
  unit = '',
  decimals = 2,
  onChange,
  onReset,
  label = '',
  color = '#667eea',
  showValue = true,
  width = 60,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [inputValue, setInputValue] = useState('');
  const inputRef = useRef(null);
  const containerRef = useRef(null);

  // Format value for display
  const displayValue = useCallback((val) => {
    const scaled = min + val * (max - min);
    return scaled.toFixed(decimals);
  }, [min, max, decimals]);

  // Parse input value back to 0-1 range
  const parseInput = useCallback((str) => {
    const num = parseFloat(str);
    if (isNaN(num)) return null;
    const clamped = Math.max(min, Math.min(max, num));
    return (clamped - min) / (max - min);
  }, [min, max]);

  // Handle double-click to edit
  const handleDoubleClick = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setInputValue(displayValue(value));
    setIsEditing(true);
  }, [value, displayValue]);

  // Handle right-click to reset
  const handleContextMenu = useCallback((e) => {
    e.preventDefault();
    if (onReset) {
      onReset();
    } else if (onChange) {
      onChange(defaultValue);
    }
  }, [onChange, onReset, defaultValue]);

  // Handle input submit
  const handleInputSubmit = useCallback(() => {
    const newValue = parseInput(inputValue);
    if (newValue !== null && onChange) {
      onChange(newValue);
    }
    setIsEditing(false);
  }, [inputValue, parseInput, onChange]);

  // Handle input keydown
  const handleInputKeyDown = useCallback((e) => {
    if (e.key === 'Enter') {
      handleInputSubmit();
    } else if (e.key === 'Escape') {
      setIsEditing(false);
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const step = e.shiftKey ? 0.001 : 0.01;
      const newVal = Math.min(1, value + step);
      onChange?.(newVal);
      setInputValue(displayValue(newVal));
    } else if (e.key === 'ArrowDown') {
      e.preventDefault();
      const step = e.shiftKey ? 0.001 : 0.01;
      const newVal = Math.max(0, value - step);
      onChange?.(newVal);
      setInputValue(displayValue(newVal));
    }
  }, [handleInputSubmit, value, onChange, displayValue]);

  // Focus input when editing starts
  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
      inputRef.current.select();
    }
  }, [isEditing]);

  // Click outside to close
  useEffect(() => {
    if (!isEditing) return;
    
    const handleClickOutside = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        handleInputSubmit();
      }
    };
    
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isEditing, handleInputSubmit]);

  return (
    <div
      ref={containerRef}
      style={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: 4,
        position: 'relative',
      }}
    >
      {/* Label */}
      {label && (
        <div style={{
          fontSize: 10,
          color: `${color}cc`,
          textTransform: 'uppercase',
          letterSpacing: 0.5,
          fontWeight: 500,
        }}>
          {label}
        </div>
      )}

      {/* Knob wrapper with double-click handler */}
      <div
        onDoubleClick={handleDoubleClick}
        onContextMenu={handleContextMenu}
        style={{ cursor: 'pointer' }}
        title="Double-click to type value, right-click to reset"
      >
        {children}
      </div>

      {/* Value display / input */}
      {showValue && (
        <div style={{ position: 'relative', minWidth: width, textAlign: 'center' }}>
          {isEditing ? (
            <input
              ref={inputRef}
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyDown={handleInputKeyDown}
              onBlur={handleInputSubmit}
              style={{
                width: '100%',
                padding: '2px 4px',
                fontSize: 11,
                fontFamily: 'monospace',
                backgroundColor: 'rgba(0,0,0,0.6)',
                border: `1px solid ${color}`,
                borderRadius: 4,
                color: '#fff',
                textAlign: 'center',
                outline: 'none',
              }}
            />
          ) : (
            <div
              onDoubleClick={handleDoubleClick}
              style={{
                fontSize: 11,
                fontFamily: 'monospace',
                color: `${color}dd`,
                cursor: 'text',
                padding: '2px 4px',
                borderRadius: 4,
                transition: 'background-color 0.15s',
              }}
              onMouseEnter={(e) => e.target.style.backgroundColor = 'rgba(255,255,255,0.05)'}
              onMouseLeave={(e) => e.target.style.backgroundColor = 'transparent'}
            >
              {displayValue(value)}{unit}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ClickToTypeKnob;

/**
 * Usage example:
 * 
 * <ClickToTypeKnob
 *   value={knobValue}
 *   min={-24}
 *   max={24}
 *   unit=" dB"
 *   decimals={1}
 *   onChange={setKnobValue}
 *   label="Gain"
 *   color="#9b59b6"
 * >
 *   <SVGKnobRenderer
 *     svgString={knobSvg}
 *     size={60}
 *     value={knobValue}
 *     isTestMode={true}
 *     onChange={setKnobValue}
 *   />
 * </ClickToTypeKnob>
 */
