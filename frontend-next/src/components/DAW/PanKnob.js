import React, { useEffect, useRef, useState } from 'react';
import styles from './PanKnob.module.css';

/**
 * PanKnob Component
 * Logic Pro-style rotary pan knob
 *
 * @param {number} pan - Current pan value (-1 to 1, 0 is center)
 * @param {Function} onPanChange - Callback when pan changes
 */
const PanKnob = ({ pan = 0, onPanChange }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const knobRef = useRef(null);
  const dragStartRef = useRef({ startY: 0, startPan: 0 });

  const handleMouseDown = (e) => {
    e.stopPropagation();
    e.preventDefault();
    setIsDragging(true);
    // Position is fixed at center of knob
    setMousePosition({ x: 14.4, y: 0 }); // 14.4px = half of 28.8px knob width
    dragStartRef.current = {
      startY: e.clientY,
      startPan: pan
    };
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      // Keep position fixed above the knob, don't follow cursor

      // Calculate pan based on vertical mouse movement
      const deltaY = dragStartRef.current.startY - e.clientY;
      const sensitivity = 0.005; // Adjust for feel
      let newPan = dragStartRef.current.startPan + (deltaY * sensitivity);
      newPan = Math.max(-1, Math.min(1, newPan));

      onPanChange({ target: { value: newPan } });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging]);

  // Convert pan to display string
  const panToDisplay = (p) => {
    if (Math.abs(p) < 0.01) return 'C'; // Center
    const percentage = Math.abs(Math.round(p * 100));
    const direction = p < 0 ? 'L' : 'R';
    return `${direction}${percentage}`;
  };

  // Calculate rotation angle (-135deg to +135deg for 270 degree travel)
  const getRotation = () => {
    return pan * 135; // -135 to +135 degrees
  };

  return (
    <div className={styles.panKnobContainer}>
      {/* Fixed center tick mark - doesn't rotate */}
      <div className={styles.centerTickMark} />

      <div
        ref={knobRef}
        className={styles.knob}
        onMouseDown={handleMouseDown}
        style={{
          transform: `rotate(${getRotation()}deg)`
        }}
      >
        {/* Knob indicator line */}
        <div className={styles.indicator} />

        {/* Center dot */}
        <div className={styles.centerDot} />
      </div>

      {/* Floating pan value - only visible when dragging */}
      {isDragging && (
        <div
          className={styles.floatingValue}
          style={{
            left: `${mousePosition.x + 6.4}px`, // Add 6.4px for container padding (scaled from 8px)
            top: '-17.6px' // Scaled from -22px
          }}
        >
          {panToDisplay(pan)}
        </div>
      )}
    </div>
  );
};

export default PanKnob;
