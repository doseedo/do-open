import React, { useMemo } from 'react';

/**
 * PlayheadCursor Component - Complete Rewrite with GPU Acceleration
 *
 * Features:
 * - Transform-based positioning (GPU accelerated)
 * - Will-change hints for browser optimization
 * - Memoized to prevent unnecessary re-renders
 * - Clean visual design with triangle indicator
 * - Smooth 60fps animation support
 */
const PlayheadCursor = React.memo(({
  position,
  totalDuration,
  width,
  zoomLevel = 1,
  showTriangle = true,
  showLine = true
}) => {
  // Calculate pixel position using simple formula
  // Apply zoom level to the width calculation
  const pixelPosition = useMemo(() => {
    if (totalDuration === 0 || width === 0) return 0;
    return (position / totalDuration) * width * zoomLevel;
  }, [position, totalDuration, width, zoomLevel]);

  // Use transform for GPU acceleration instead of left property
  const cursorTransform = `translateX(${pixelPosition}px)`;
  const triangleTransform = `translateX(${pixelPosition}px) translateX(-50%)`;

  return (
    <>
      {/* Triangle indicator at top */}
      {showTriangle && <div
        id="timeline-cursor-triangle"
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          width: 0,
          height: 0,
          borderLeft: '10px solid transparent',
          borderRight: '10px solid transparent',
          borderTop: '15px solid #fff',
          zIndex: 3001,
          pointerEvents: 'none',
          transform: triangleTransform,
          willChange: 'transform',
          transition: 'none'
        }}
      />}

      {/* Vertical line spanning the entire height - extends upward to meet timeline */}
      {showLine && <div
        id="timeline-cursor-line"
        style={{
          position: 'absolute',
          top: '-100px', // Negative value to extend upward into timeline area
          left: 0,
          width: '1px',
          height: '10100px', // Increased to compensate for negative top
          background: 'linear-gradient(to bottom, rgba(255, 255, 255, 0) 0px, rgba(255, 255, 255, 0) 100px, rgba(255, 255, 255, 1) 250px, rgba(255, 255, 255, 1) 100%)',
          boxShadow: '0 0 4px rgba(255, 255, 255, 0.1)',
          zIndex: 5, // Below sticky labels (z-index: 10-100)
          pointerEvents: 'none',
          transform: cursorTransform,
          willChange: 'transform',
          transition: 'none'
        }}
      />}

    </>
  );
});

PlayheadCursor.displayName = 'PlayheadCursor';

export default PlayheadCursor;
