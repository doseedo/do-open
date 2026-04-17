/**
 * Oscilloscope - Real-time waveform display
 * Shows audio waveform like hardware scope
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';

const Oscilloscope = ({
  width = 300,
  height = 150,
  audioContext = null,
  sourceNode = null,
  color = '#00ff88',
  backgroundColor = 'rgba(0,0,0,0.5)',
  gridColor = 'rgba(0,255,136,0.15)',
  lineWidth = 2,
  showGrid = true,
  triggerLevel = 0,
  timeScale = 1, // 1 = normal, 2 = zoomed in
  borderRadius = 8,
}) => {
  const canvasRef = useRef(null);
  const analyserRef = useRef(null);
  const dataArrayRef = useRef(null);
  const animationRef = useRef(null);
  const [isActive, setIsActive] = useState(false);

  // Setup analyser
  useEffect(() => {
    if (!audioContext || !sourceNode) {
      setIsActive(false);
      return;
    }

    try {
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 2048;
      sourceNode.connect(analyser);
      analyserRef.current = analyser;
      dataArrayRef.current = new Float32Array(analyser.fftSize);
      setIsActive(true);

      return () => {
        try { sourceNode.disconnect(analyser); } catch (e) {}
      };
    } catch (e) {
      setIsActive(false);
    }
  }, [audioContext, sourceNode]);

  // Find trigger point (zero crossing)
  const findTrigger = useCallback((data, threshold = 0) => {
    for (let i = 1; i < data.length - 1; i++) {
      if (data[i - 1] <= threshold && data[i] > threshold) {
        return i;
      }
    }
    return 0;
  }, []);

  // Animation loop
  useEffect(() => {
    if (!isActive || !canvasRef.current || !analyserRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    const analyser = analyserRef.current;
    const dataArray = dataArrayRef.current;

    const draw = () => {
      analyser.getFloatTimeDomainData(dataArray);

      // Clear
      ctx.fillStyle = backgroundColor;
      ctx.fillRect(0, 0, width, height);

      // Grid
      if (showGrid) {
        ctx.strokeStyle = gridColor;
        ctx.lineWidth = 1;

        // Vertical lines
        for (let i = 0; i <= 10; i++) {
          const x = (i / 10) * width;
          ctx.beginPath();
          ctx.moveTo(x, 0);
          ctx.lineTo(x, height);
          ctx.stroke();
        }

        // Horizontal lines
        for (let i = 0; i <= 8; i++) {
          const y = (i / 8) * height;
          ctx.beginPath();
          ctx.moveTo(0, y);
          ctx.lineTo(width, y);
          ctx.stroke();
        }

        // Center line (brighter)
        ctx.strokeStyle = `${gridColor.slice(0, -4)}0.4)`;
        ctx.beginPath();
        ctx.moveTo(0, height / 2);
        ctx.lineTo(width, height / 2);
        ctx.stroke();
      }

      // Find trigger point for stable display
      const triggerIndex = findTrigger(dataArray, triggerLevel);
      const samplesToShow = Math.floor(dataArray.length / timeScale);

      // Draw waveform
      ctx.beginPath();
      ctx.strokeStyle = color;
      ctx.lineWidth = lineWidth;
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;

      for (let i = 0; i < width; i++) {
        const sampleIndex = triggerIndex + Math.floor((i / width) * samplesToShow);
        const sample = dataArray[sampleIndex] || 0;
        const y = ((1 - sample) / 2) * height;

        if (i === 0) {
          ctx.moveTo(i, y);
        } else {
          ctx.lineTo(i, y);
        }
      }

      ctx.stroke();
      ctx.shadowBlur = 0;

      animationRef.current = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [isActive, width, height, color, backgroundColor, gridColor, showGrid, lineWidth, triggerLevel, timeScale, findTrigger]);

  // Draw static placeholder
  useEffect(() => {
    if (isActive || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    if (showGrid) {
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 10; i++) {
        ctx.beginPath();
        ctx.moveTo((i / 10) * width, 0);
        ctx.lineTo((i / 10) * width, height);
        ctx.stroke();
      }
      for (let i = 0; i <= 8; i++) {
        ctx.beginPath();
        ctx.moveTo(0, (i / 8) * height);
        ctx.lineTo(width, (i / 8) * height);
        ctx.stroke();
      }
    }

    // Flat line
    ctx.strokeStyle = `${color}66`;
    ctx.lineWidth = lineWidth;
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    ctx.fillStyle = `${color}44`;
    ctx.font = '11px monospace';
    ctx.textAlign = 'center';
    ctx.fillText('No signal', width / 2, height / 2 + 4);
  }, [isActive, width, height, color, backgroundColor, gridColor, showGrid, lineWidth]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ borderRadius, display: 'block' }}
    />
  );
};

export default Oscilloscope;
