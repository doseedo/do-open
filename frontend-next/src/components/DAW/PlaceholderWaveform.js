import React, { useEffect, useRef, useState } from 'react';
import styles from './DAW.module.css';

/**
 * PlaceholderWaveform - Animated placeholder showing denoising process
 * Simulates waveform generation with moving vertical lines that settle into place
 */
const PlaceholderWaveform = ({ width, height, duration = 30, settling = false, actualWaveform = null }) => {
  const canvasRef = useRef(null);
  const animationFrameRef = useRef(null);
  const startTimeRef = useRef(Date.now());
  const [finalWaveform, setFinalWaveform] = useState(null);
  const previousAmplitudesRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size accounting for device pixel ratio
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    ctx.scale(dpr, dpr);

    // Generate simplified waveform (vertical lines)
    const lineCount = Math.floor(width / 4); // One line every 4 pixels
    const centerY = height / 2;

    // Generate target waveform ONLY when settling
    // During generation, stay pure noise with no target
    if (settling && !finalWaveform) {
      let waveform = [];

      if (actualWaveform && actualWaveform.length > 0) {
        // Use the actual waveform data when settling
        // Resample to match lineCount
        for (let i = 0; i < lineCount; i++) {
          const sourceIndex = Math.floor((i / lineCount) * actualWaveform.length);
          const normalizedValue = actualWaveform[sourceIndex] || 0; // 0 to 1
          const amplitude = normalizedValue * (height * 0.45);
          waveform.push(amplitude);
        }
      } else {
        // No actual waveform - just stay as noise (no target shape)
        waveform = new Array(lineCount).fill(0);
      }
      setFinalWaveform(waveform);
    }

    // Initialize previous amplitudes if not set
    if (!previousAmplitudesRef.current) {
      previousAmplitudesRef.current = new Array(lineCount).fill(0);
    }

    const animate = () => {
      const elapsed = (Date.now() - startTimeRef.current) / 1000;

      // Clear canvas
      ctx.clearRect(0, 0, width, height);

      // Draw animated waveform lines
      ctx.strokeStyle = 'rgba(139, 92, 246, 0.6)';
      ctx.lineWidth = 2;

      for (let i = 0; i < lineCount; i++) {
        const x = (i / lineCount) * width;
        const targetAmplitude = (finalWaveform && finalWaveform[i]) || 0;
        const previousAmplitude = previousAmplitudesRef.current[i] || 0;

        let amplitude;

        if (settling && finalWaveform) {
          // Settling phase - morph into actual waveform in 0.5 seconds
          const settleProgress = Math.min(elapsed / 0.5, 1); // 0.5 seconds to settle
          const easeOut = 1 - Math.pow(1 - settleProgress, 3); // Cubic ease-out

          // Start from current noise position and ease to actual waveform
          const noise = (1 - easeOut) * ((Math.random() - 0.5) * height * 0.5);
          amplitude = targetAmplitude * easeOut + noise;
        } else {
          // Denoising phase - pure random noise, NO morphing
          // Generate new random target but interpolate from previous value
          const randomTarget = (Math.random() - 0.5) * height * 0.6; // Louder: increased to 60% of height

          // Smoothly interpolate between previous and new random value (slower/smoother: 0.1 instead of 0.15)
          const lerpFactor = 0.08; // Slower, smoother motion
          const smoothedNoise = previousAmplitude + (randomTarget - previousAmplitude) * lerpFactor;

          // Pure noise only - no target shape
          amplitude = smoothedNoise;
        }

        // Store current amplitude for next frame
        previousAmplitudesRef.current[i] = amplitude;

        // Draw line from center
        ctx.beginPath();
        ctx.moveTo(x, centerY - amplitude);
        ctx.lineTo(x, centerY + amplitude);
        ctx.stroke();

        // Add subtle glow effect
        if (!settling || (settling && Math.random() > 0.95)) {
          ctx.strokeStyle = 'rgba(139, 92, 246, 0.3)';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(x, centerY - amplitude);
          ctx.lineTo(x, centerY + amplitude);
          ctx.stroke();
          ctx.strokeStyle = 'rgba(139, 92, 246, 0.6)';
          ctx.lineWidth = 2;
        }
      }

      // Continue animating unless fully settled (0.5s for settling)
      if (!settling || elapsed < 0.5) {
        animationFrameRef.current = requestAnimationFrame(animate);
      }
    };

    animate();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [width, height, duration, settling, finalWaveform, actualWaveform]);

  // Reset start time when settling begins
  useEffect(() => {
    if (settling) {
      startTimeRef.current = Date.now();
    }
  }, [settling]);

  return (
    <canvas
      ref={canvasRef}
      className={styles.placeholderWaveform}
      style={{ width: `${width}px`, height: `${height}px` }}
    />
  );
};

export default PlaceholderWaveform;
