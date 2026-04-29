import React, { useCallback, useState, useEffect } from 'react';
import styles from './VerticalResizeBar.module.css';

/**
 * VerticalResizeBar - Draggable vertical resize handle for adjusting panel/DAW height
 *
 * @param {number} topPosition - Current top position in pixels
 * @param {function} onResize - Callback when resize occurs, receives new top position
 * @param {number} minHeight - Minimum height allowed (default 300px)
 * @param {number} maxHeight - Maximum height allowed (default 800px)
 */
const VerticalResizeBar = React.memo(({ topPosition, onResize, minHeight = 300, maxHeight = 800 }) => {
  const [isDragging, setIsDragging] = useState(false);

  const handleMouseDown = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);

    // Store the offset between mouse and bar position
    const startY = e.clientY;
    const startPosition = topPosition;

    const handleMouseMove = (moveEvent) => {
      const deltaY = moveEvent.clientY - startY;
      const newTop = startPosition + deltaY;

      // Clamp between min and max
      const clampedTop = Math.max(minHeight, Math.min(newTop, maxHeight));

      onResize(clampedTop);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };

    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
  }, [onResize, minHeight, maxHeight, topPosition]);

  // Prevent text selection while dragging
  useEffect(() => {
    if (isDragging) {
      document.body.style.userSelect = 'none';
      document.body.style.cursor = 'ns-resize';
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
      style={{ top: `${topPosition}px` }}
      onMouseDown={handleMouseDown}
    />
  );
});

VerticalResizeBar.displayName = 'VerticalResizeBar';

export default VerticalResizeBar;
