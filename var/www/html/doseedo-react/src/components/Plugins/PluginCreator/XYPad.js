/**
 * XYPad Component - 2D control surface for Plugin Creator
 * Portal/Serum-style XY controller with visual feedback
 */

import React, { useRef, useState, useCallback, useEffect } from 'react';

const XYPad = ({
  width = 200,
  height = 200,
  x = 0.5,
  y = 0.5,
  onChange,
  color = '#9b59b6',
  backgroundColor = 'rgba(0,0,0,0.3)',
  label = '',
  xLabel = 'X',
  yLabel = 'Y',
  showGrid = true,
  showCrosshair = true,
  borderRadius = 12,
}) => {
  const padRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [position, setPosition] = useState({ x, y });

  // Update position when props change
  useEffect(() => {
    setPosition({ x, y });
  }, [x, y]);

  const getPositionFromEvent = useCallback((e) => {
    if (!padRef.current) return { x: 0.5, y: 0.5 };
    
    const rect = padRef.current.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    
    const newX = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    const newY = Math.max(0, Math.min(1, 1 - (clientY - rect.top) / rect.height)); // Invert Y
    
    return { x: newX, y: newY };
  }, []);

  const handleStart = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
    const pos = getPositionFromEvent(e);
    setPosition(pos);
    onChange?.(pos);
  }, [getPositionFromEvent, onChange]);

  const handleMove = useCallback((e) => {
    if (!isDragging) return;
    e.preventDefault();
    const pos = getPositionFromEvent(e);
    setPosition(pos);
    onChange?.(pos);
  }, [isDragging, getPositionFromEvent, onChange]);

  const handleEnd = useCallback(() => {
    setIsDragging(false);
  }, []);

  // Global mouse/touch listeners for drag outside pad
  useEffect(() => {
    if (isDragging) {
      const handleGlobalMove = (e) => handleMove(e);
      const handleGlobalEnd = () => handleEnd();
      
      window.addEventListener('mousemove', handleGlobalMove);
      window.addEventListener('mouseup', handleGlobalEnd);
      window.addEventListener('touchmove', handleGlobalMove, { passive: false });
      window.addEventListener('touchend', handleGlobalEnd);
      
      return () => {
        window.removeEventListener('mousemove', handleGlobalMove);
        window.removeEventListener('mouseup', handleGlobalEnd);
        window.removeEventListener('touchmove', handleGlobalMove);
        window.removeEventListener('touchend', handleGlobalEnd);
      };
    }
  }, [isDragging, handleMove, handleEnd]);

  const dotX = position.x * width;
  const dotY = (1 - position.y) * height;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
      {label && (
        <div style={{ color: color, fontSize: 12, fontWeight: 600, textTransform: 'uppercase', letterSpacing: 1 }}>
          {label}
        </div>
      )}
      <div
        ref={padRef}
        onMouseDown={handleStart}
        onTouchStart={handleStart}
        style={{
          width,
          height,
          backgroundColor,
          borderRadius,
          border: `1px solid ${color}40`,
          position: 'relative',
          cursor: isDragging ? 'grabbing' : 'crosshair',
          overflow: 'hidden',
          boxShadow: `0 0 20px ${color}20, inset 0 0 30px rgba(0,0,0,0.3)`,
        }}
      >
        {/* Grid lines */}
        {showGrid && (
          <svg width={width} height={height} style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
            {/* Vertical grid lines */}
            {[0.25, 0.5, 0.75].map((pos) => (
              <line
                key={`v-${pos}`}
                x1={pos * width}
                y1={0}
                x2={pos * width}
                y2={height}
                stroke={color}
                strokeOpacity={0.15}
                strokeWidth={1}
              />
            ))}
            {/* Horizontal grid lines */}
            {[0.25, 0.5, 0.75].map((pos) => (
              <line
                key={`h-${pos}`}
                x1={0}
                y1={pos * height}
                x2={width}
                y2={pos * height}
                stroke={color}
                strokeOpacity={0.15}
                strokeWidth={1}
              />
            ))}
          </svg>
        )}

        {/* Crosshair lines to current position */}
        {showCrosshair && (
          <svg width={width} height={height} style={{ position: 'absolute', top: 0, left: 0, pointerEvents: 'none' }}>
            <line
              x1={dotX}
              y1={0}
              x2={dotX}
              y2={height}
              stroke={color}
              strokeOpacity={0.4}
              strokeWidth={1}
              strokeDasharray="4,4"
            />
            <line
              x1={0}
              y1={dotY}
              x2={width}
              y2={dotY}
              stroke={color}
              strokeOpacity={0.4}
              strokeWidth={1}
              strokeDasharray="4,4"
            />
          </svg>
        )}

        {/* Control dot with glow */}
        <div
          style={{
            position: 'absolute',
            left: dotX - 8,
            top: dotY - 8,
            width: 16,
            height: 16,
            borderRadius: '50%',
            backgroundColor: color,
            boxShadow: `0 0 12px ${color}, 0 0 24px ${color}80`,
            transition: isDragging ? 'none' : 'left 0.05s, top 0.05s',
            pointerEvents: 'none',
          }}
        />

        {/* Inner dot */}
        <div
          style={{
            position: 'absolute',
            left: dotX - 3,
            top: dotY - 3,
            width: 6,
            height: 6,
            borderRadius: '50%',
            backgroundColor: '#fff',
            pointerEvents: 'none',
          }}
        />
      </div>

      {/* Axis labels and values */}
      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', fontSize: 10, color: `${color}aa` }}>
        <span>{xLabel}: {(position.x * 100).toFixed(0)}%</span>
        <span>{yLabel}: {(position.y * 100).toFixed(0)}%</span>
      </div>
    </div>
  );
};

export default XYPad;
