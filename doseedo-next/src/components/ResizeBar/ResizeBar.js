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

    // Store the offset between mouse and bar position
    const startX = e.clientX;
    const startPosition = leftPosition;

    const handleMouseMove = (moveEvent) => {
      const deltaX = moveEvent.clientX - startX;
      const newLeft = startPosition + deltaX;

      // Clamp between min and max
      const clampedLeft = Math.max(minWidth, Math.min(newLeft, maxWidth));

      onResize(clampedLeft);
    };

    const handleMouseUp = () => {
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
