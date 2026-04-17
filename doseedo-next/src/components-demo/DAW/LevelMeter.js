import React, { useEffect, useRef, useState } from 'react';
import styles from './LevelMeter.module.css';

/**
 * LevelMeter Component
 * Professional DAW-style horizontal level meter with metering
 *
 * @param {number} gain - Current gain value (0-1)
 * @param {Function} onGainChange - Callback when gain changes
 * @param {number} level - Current audio level (0-1) for metering
 * @param {AudioNode} audioNode - Audio node to analyze for metering (optional)
 */
const LevelMeter = ({ gain, onGainChange, level = 0, audioNode }) => {
  const [isDragging, setIsDragging] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
  const sliderRef = useRef(null);
  const [displayLevel, setDisplayLevel] = useState(0);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const dataArrayRef = useRef(null);

  // Set up audio analysis if audioNode is provided
  useEffect(() => {
    if (!audioNode || !audioNode.context) {
      console.log('⚠️ LevelMeter: No audioNode provided');
      return;
    }

    console.log('🎚️ LevelMeter: Setting up analyser for gain node');

    // Create analyser node
    const analyser = audioNode.context.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.8;

    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);

    analyserRef.current = analyser;
    dataArrayRef.current = dataArray;

    // Connect audioNode -> analyser (for analysis, doesn't affect audio output)
    // This creates a "tap" into the audio stream without interrupting it
    try {
      audioNode.connect(analyser);
      console.log('✅ LevelMeter: Analyser connected');
    } catch (error) {
      console.error('❌ LevelMeter: Failed to connect analyser:', error);
      return;
    }

    // Start metering loop
    const updateMeter = () => {
      if (!analyserRef.current || !dataArrayRef.current) return;

      analyser.getByteFrequencyData(dataArray);

      // Calculate RMS level
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / bufferLength) / 255;

      setDisplayLevel(rms);

      animationFrameRef.current = requestAnimationFrame(updateMeter);
    };

    updateMeter();

    return () => {
      console.log('🧹 LevelMeter: Cleaning up analyser');
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      if (analyserRef.current) {
        try {
          // Disconnect the analyser from the audioNode
          analyserRef.current.disconnect();
        } catch (error) {
          console.error('⚠️ LevelMeter: Error disconnecting analyser:', error);
        }
        analyserRef.current = null;
      }
      dataArrayRef.current = null;
    };
  }, [audioNode]);

  // Fallback smooth level decay for metering when no audioNode
  useEffect(() => {
    if (audioNode) return; // Skip if we have real metering

    if (level > displayLevel) {
      setDisplayLevel(level);
    } else {
      // Decay slowly
      const timeout = setTimeout(() => {
        setDisplayLevel(Math.max(0, displayLevel * 0.95));
      }, 50);
      return () => clearTimeout(timeout);
    }
  }, [level, displayLevel, audioNode]);

  const handleMouseDown = (e) => {
    e.stopPropagation(); // Prevent track selection
    setIsDragging(true);
    // Get cursor position relative to the meterTrack element
    const rect = sliderRef.current.getBoundingClientRect();
    const relativeX = e.clientX - rect.left;
    setMousePosition({ x: relativeX, y: 0 });
    updateGain(e);
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      // Update X position relative to meterTrack
      const rect = sliderRef.current.getBoundingClientRect();
      const relativeX = e.clientX - rect.left;
      setMousePosition({ x: relativeX, y: 0 });
      updateGain(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const updateGain = (e) => {
    if (!sliderRef.current) return;

    const rect = sliderRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const width = rect.width;

    // Left = 0.0, Right = 1.0
    let newGain = x / width;
    newGain = Math.max(0, Math.min(1, newGain));

    onGainChange({ target: { value: newGain } });
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

  // Calculate colors based on level
  const getLevelColor = (position) => {
    if (position > 0.9) return '#ff3b3b'; // Red (> -2dB)
    if (position > 0.75) return '#ffcc00'; // Yellow (> -6dB)
    return '#4ade80'; // Green
  };

  // Convert gain to dB for display
  const gainToDb = (g) => {
    if (g === 0) return '-∞';
    const db = 20 * Math.log10(g);
    return db > -0.5 ? '0.0' : db.toFixed(1);
  };

  return (
    <div className={styles.levelMeterContainer}>
      <div
        ref={sliderRef}
        className={styles.meterTrack}
        onMouseDown={handleMouseDown}
      >
        {/* Background level segments */}
        <div className={styles.levelSegments}>
          {Array.from({ length: 30 }).map((_, i) => {
            const position = (i + 1) / 30;
            const isLit = displayLevel >= position;
            return (
              <div
                key={i}
                className={`${styles.segment} ${isLit ? styles.lit : ''}`}
                style={{
                  backgroundColor: isLit ? getLevelColor(position) : '#151515'
                }}
              />
            );
          })}
        </div>

        {/* Fader thumb */}
        <div
          className={styles.fader}
          style={{
            left: `${gain * 100}%`
          }}
        />
      </div>

      {/* Floating dB value - only visible when dragging */}
      {isDragging && (
        <div
          className={styles.floatingValue}
          style={{
            left: `${mousePosition.x + 4}px`, // Add 4px for container padding
            top: '-26px'
          }}
        >
          {gainToDb(gain)}dB
        </div>
      )}
    </div>
  );
};

export default LevelMeter;
