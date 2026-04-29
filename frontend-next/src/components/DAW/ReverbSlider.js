import React, { useEffect, useRef, useState } from 'react';
import styles from './ReverbSlider.module.css';

/**
 * ReverbSlider Component
 * Professional DAW-style horizontal reverb send slider
 *
 * @param {number} reverbSend - Current reverb send value (0 to 1)
 * @param {Function} onReverbChange - Callback when reverb send changes
 * @param {string} label - Label for the slider
 */
const ReverbSlider = ({ reverbSend = 0.15, onReverbChange, label }) => {
  const [isDragging, setIsDragging] = useState(false);
  const sliderRef = useRef(null);

  const handleMouseDown = (e) => {
    e.stopPropagation(); // Prevent track selection
    setIsDragging(true);
    updateReverb(e);
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      updateReverb(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const updateReverb = (e) => {
    if (!sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;

    // Map: Left = 0, Right = 1
    let newReverb = x / width;
    newReverb = Math.max(0, Math.min(1, newReverb));

    onReverbChange({ target: { value: newReverb } });
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

  // Convert reverb send to display string (percentage)
  const reverbToDisplay = (r) => {
    return `${Math.round(r * 100)}%`;
  };

  // Calculate position for visual indicator
  const getReverbPosition = () => {
    return reverbSend * 100; // Convert 0..1 to 0..100%
  };

  return (
    <div className={styles.reverbSliderContainer}>
      <div
        ref={sliderRef}
        className={styles.reverbTrack}
        onMouseDown={handleMouseDown}
        title={`Reverb: ${reverbToDisplay(reverbSend)}`}
      >
        {/* Reverb fill indicator */}
        <div
          className={styles.reverbFill}
          style={{
            width: `${getReverbPosition()}%`
          }}
        />

        {/* Fader thumb */}
        <div
          className={styles.fader}
          style={{
            left: `${getReverbPosition()}%`
          }}
        />
      </div>
    </div>
  );
};

export default ReverbSlider;
