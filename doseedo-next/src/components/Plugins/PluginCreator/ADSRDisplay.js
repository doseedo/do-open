/**
 * ADSRDisplay - Visual envelope display with draggable handles
 */

import React, { useRef, useEffect, useCallback, useState } from 'react';

const ADSRDisplay = ({
  width = 200,
  height = 100,
  attack = 0.1,
  decay = 0.2,
  sustain = 0.7,
  release = 0.3,
  onChange,
  color = '#e74c3c',
  backgroundColor = 'rgba(0,0,0,0.3)',
  gridColor = 'rgba(255,255,255,0.1)',
  showGrid = true,
  borderRadius = 8,
}) => {
  const canvasRef = useRef(null);
  const [dragging, setDragging] = useState(null);

  // Normalize values
  const a = Math.max(0, Math.min(1, attack));
  const d = Math.max(0, Math.min(1, decay));
  const s = Math.max(0, Math.min(1, sustain));
  const r = Math.max(0, Math.min(1, release));

  // Calculate points
  const getPoints = useCallback(() => {
    const totalTime = a + d + 0.3 + r; // 0.3 for sustain hold
    const scale = width / totalTime;
    
    return {
      start: { x: 0, y: height },
      attack: { x: a * scale, y: 0 },
      decay: { x: (a + d) * scale, y: height * (1 - s) },
      sustainEnd: { x: (a + d + 0.3) * scale, y: height * (1 - s) },
      end: { x: width, y: height },
    };
  }, [width, height, a, d, s, r]);

  // Draw envelope
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const points = getPoints();

    // Clear
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Grid
    if (showGrid) {
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      for (let i = 0; i <= 4; i++) {
        const y = (i / 4) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    // Fill under envelope
    ctx.beginPath();
    ctx.moveTo(points.start.x, points.start.y);
    ctx.lineTo(points.attack.x, points.attack.y);
    ctx.lineTo(points.decay.x, points.decay.y);
    ctx.lineTo(points.sustainEnd.x, points.sustainEnd.y);
    ctx.lineTo(points.end.x, points.end.y);
    ctx.lineTo(points.start.x, points.start.y);
    ctx.closePath();
    ctx.fillStyle = `${color}33`;
    ctx.fill();

    // Draw envelope line
    ctx.beginPath();
    ctx.moveTo(points.start.x, points.start.y);
    ctx.lineTo(points.attack.x, points.attack.y);
    ctx.lineTo(points.decay.x, points.decay.y);
    ctx.lineTo(points.sustainEnd.x, points.sustainEnd.y);
    ctx.lineTo(points.end.x, points.end.y);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw handles
    const drawHandle = (x, y, label) => {
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    };

    drawHandle(points.attack.x, points.attack.y, 'A');
    drawHandle(points.decay.x, points.decay.y, 'D');
    drawHandle(points.sustainEnd.x, points.sustainEnd.y, 'S');

    // Labels
    ctx.fillStyle = `${color}cc`;
    ctx.font = '10px sans-serif';
    ctx.fillText('A', points.attack.x - 3, height - 5);
    ctx.fillText('D', points.decay.x - 3, height - 5);
    ctx.fillText('S', (points.decay.x + points.sustainEnd.x) / 2 - 3, height - 5);
    ctx.fillText('R', (points.sustainEnd.x + points.end.x) / 2 - 3, height - 5);

  }, [width, height, getPoints, color, backgroundColor, gridColor, showGrid]);

  // Mouse handlers
  const handleMouseDown = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const points = getPoints();

    const dist = (px, py) => Math.sqrt((x - px) ** 2 + (y - py) ** 2);

    if (dist(points.attack.x, points.attack.y) < 15) {
      setDragging('attack');
    } else if (dist(points.decay.x, points.decay.y) < 15) {
      setDragging('decay');
    } else if (dist(points.sustainEnd.x, points.sustainEnd.y) < 15) {
      setDragging('sustain');
    }
  }, [getPoints]);

  const handleMouseMove = useCallback((e) => {
    if (!dragging) return;

    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const xNorm = Math.max(0, Math.min(1, x / width));
    const yNorm = Math.max(0, Math.min(1, 1 - y / height));

    if (dragging === 'attack') {
      onChange?.({ attack: xNorm * 0.5, decay: d, sustain: s, release: r });
    } else if (dragging === 'decay') {
      onChange?.({ attack: a, decay: Math.max(0, xNorm - a) * 0.5, sustain: yNorm, release: r });
    } else if (dragging === 'sustain') {
      onChange?.({ attack: a, decay: d, sustain: yNorm, release: r });
    }
  }, [dragging, width, height, a, d, s, r, onChange]);

  const handleMouseUp = useCallback(() => {
    setDragging(null);
  }, []);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      style={{ 
        borderRadius, 
        cursor: dragging ? 'grabbing' : 'pointer',
        display: 'block',
      }}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    />
  );
};

export default ADSRDisplay;
