import React, { useCallback, useState, useEffect } from 'react';
import styles from './ResizeBar.module.css';

/**
 * ResizeBar - Draggable resize handle for adjusting panel/tracklist width
 *
 * @param {number} leftPosition - Current left position in pixels
 * @param {function} onResize - Callback when resize occurs, receives new left position
 * @param {number} minWidth - Minimum width allowed (default 200px)
 * @param {number} maxWidth - Maximum width allowed (default 600px)
 */
const ResizeBar = React.memo(({ leftPosition, onResize, minWidth = 200, maxWidth = 600 }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);

    const startX = e.clientX;
    const startPosition = leftPosition;
    let rafId = null;
    let pendingTarget = startPosition;

    const flush = () => {
      rafId = null;
      onResize(Math.max(minWidth, Math.min(pendingTarget, maxWidth)));
    };

    const handleMouseMove = (moveEvent) => {
      // Coalesce mousemove → single update per animation frame. Prevents
      // React from queueing dozens of state updates per second and keeps
      // the clamp at min/max feeling smooth instead of snappy.
      pendingTarget = startPosition + (moveEvent.clientX - startX);
      if (rafId === null) rafId = requestAnimationFrame(flush);
    };

    const handleMouseUp = () => {
      if (rafId !== null) {
        cancelAnimationFrame(rafId);
        flush();
      }
      setIsDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [onResize, minWidth, maxWidth, leftPosition]);

  // Prevent text selection while dragging
  useEffect(() => {
    if (isDragging) {
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'ew-resize';
    } else {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    }

    return () => {
      document.body.style.userSelect = '';
      document.body.style.cursor = '';
    };
  }, [isDragging]);

  return (
    <div
      className={`${styles.resizeBar} ${isDragging ? styles.dragging : ''}`}
      style={{ left: `${leftPosition}px` }}
      onMouseDown={handleMouseDown}
    />
  );
});

ResizeBar.displayName = 'ResizeBar';

export default ResizeBar;
