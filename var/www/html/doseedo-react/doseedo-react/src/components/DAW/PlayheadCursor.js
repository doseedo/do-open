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
  showLine = true,
  showTimeDisplay = true
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
          width: '2px',
          height: '10100px', // Increased to compensate for negative top
          backgroundColor: '#fff',
          boxShadow: '0 0 4px rgba(255, 255, 255, 0.5)',
          zIndex: 5, // Below sticky labels (z-index: 10-100)
          pointerEvents: 'none',
          transform: cursorTransform,
          willChange: 'transform',
          transition: 'none'
        }}
      />}

      {/* Time display tooltip (optional, shows during playback) */}
      {showTimeDisplay && position > 0 && (
        <div
          id="timeline-cursor-time"
          style={{
            position: 'absolute',
            top: '-25px',
            left: 0,
            transform: `${triangleTransform} translateX(-50%)`,
            backgroundColor: 'rgba(0, 0, 0, 0.8)',
            color: '#fff',
            padding: '4px 8px',
            borderRadius: '4px',
            fontSize: '11px',
            fontWeight: 'bold',
            whiteSpace: 'nowrap',
            pointerEvents: 'none',
            zIndex: 3002,
            opacity: 0.9,
            willChange: 'transform'
          }}
        >
          {formatTime(position)}
        </div>
      )}
    </>
  );
});

// Format time as MM:SS or SS.s
function formatTime(seconds) {
  if (seconds >= 60) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
  return `${seconds.toFixed(1)}s`;
}

PlayheadCursor.displayName = 'PlayheadCursor';

export default PlayheadCursor;
