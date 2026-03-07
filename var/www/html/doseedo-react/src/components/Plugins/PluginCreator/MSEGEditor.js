/**
 * MSEGEditor - Multi-Stage Envelope Generator
 * Serum/Vital-style draggable envelope curve editor
 */

import React, { useRef, useState, useCallback, useEffect } from 'react';

const MSEGEditor = ({
  width = 400,
  height = 200,
  points = null, // [{x: 0-1, y: 0-1, curve: -1 to 1}]
  onChange,
  color = '#667eea',
  backgroundColor = 'rgba(0,0,0,0.4)',
  gridColor = 'rgba(255,255,255,0.1)',
  showGrid = true,
  snapToGrid = false,
  gridSize = 0.125,
  minPoints = 2,
  maxPoints = 32,
  borderRadius = 8,
}) => {
  const canvasRef = useRef(null);
  const [localPoints, setLocalPoints] = useState(
    points || [
      { x: 0, y: 0, curve: 0 },
      { x: 0.2, y: 1, curve: 0.3 },
      { x: 0.4, y: 0.7, curve: -0.2 },
      { x: 0.6, y: 0.5, curve: 0 },
      { x: 1, y: 0, curve: 0.5 },
    ]
  );
  const [draggingIndex, setDraggingIndex] = useState(null);
  const [hoveredIndex, setHoveredIndex] = useState(null);

  // Sync with external points
  useEffect(() => {
    if (points) setLocalPoints(points);
  }, [points]);

  // Convert canvas coords to normalized
  const canvasToNorm = useCallback((canvasX, canvasY) => {
    const x = Math.max(0, Math.min(1, canvasX / width));
    const y = Math.max(0, Math.min(1, 1 - canvasY / height));
    
    if (snapToGrid) {
      return {
        x: Math.round(x / gridSize) * gridSize,
        y: Math.round(y / gridSize) * gridSize,
      };
    }
    return { x, y };
  }, [width, height, snapToGrid, gridSize]);

  // Convert normalized to canvas coords
  const normToCanvas = useCallback((normX, normY) => ({
    x: normX * width,
    y: (1 - normY) * height,
  }), [width, height]);

  // Find point near position
  const findPointNear = useCallback((canvasX, canvasY, threshold = 12) => {
    for (let i = 0; i < localPoints.length; i++) {
      const { x, y } = normToCanvas(localPoints[i].x, localPoints[i].y);
      const dist = Math.sqrt((x - canvasX) ** 2 + (y - canvasY) ** 2);
      if (dist < threshold) return i;
    }
    return null;
  }, [localPoints, normToCanvas]);

  // Bezier curve calculation
  const getCurvePoint = useCallback((p1, p2, t) => {
    const curve = p1.curve || 0;
    const midY = (p1.y + p2.y) / 2 + curve * 0.5;
    const oneMinusT = 1 - t;
    
    return {
      x: oneMinusT * p1.x + t * p2.x,
      y: oneMinusT * oneMinusT * p1.y + 2 * oneMinusT * t * midY + t * t * p2.y,
    };
  }, []);

  // Draw the envelope
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    // Clear
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // Grid
    if (showGrid) {
      ctx.strokeStyle = gridColor;
      ctx.lineWidth = 1;
      
      for (let i = 0; i <= 1; i += gridSize) {
        const x = i * width;
        const y = (1 - i) * height;
        
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
      }
    }

    // Draw envelope curve
    ctx.beginPath();
    const firstPoint = normToCanvas(localPoints[0].x, localPoints[0].y);
    ctx.moveTo(firstPoint.x, firstPoint.y);

    for (let i = 0; i < localPoints.length - 1; i++) {
      const p1 = localPoints[i];
      const p2 = localPoints[i + 1];
      
      // Draw curved segment
      for (let t = 0; t <= 1; t += 0.05) {
        const point = getCurvePoint(p1, p2, t);
        const canvasPoint = normToCanvas(point.x, point.y);
        ctx.lineTo(canvasPoint.x, canvasPoint.y);
      }
    }

    // Fill under curve
    ctx.lineTo(width, height);
    ctx.lineTo(0, height);
    ctx.closePath();
    ctx.fillStyle = `${color}33`;
    ctx.fill();

    // Stroke curve
    ctx.beginPath();
    ctx.moveTo(firstPoint.x, firstPoint.y);
    for (let i = 0; i < localPoints.length - 1; i++) {
      const p1 = localPoints[i];
      const p2 = localPoints[i + 1];
      for (let t = 0; t <= 1; t += 0.05) {
        const point = getCurvePoint(p1, p2, t);
        const canvasPoint = normToCanvas(point.x, point.y);
        ctx.lineTo(canvasPoint.x, canvasPoint.y);
      }
    }
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw points
    localPoints.forEach((point, i) => {
      const { x, y } = normToCanvas(point.x, point.y);
      const isHovered = hoveredIndex === i;
      const isDragging = draggingIndex === i;
      
      ctx.beginPath();
      ctx.arc(x, y, isHovered || isDragging ? 8 : 6, 0, Math.PI * 2);
      ctx.fillStyle = isDragging ? '#fff' : color;
      ctx.fill();
      
      if (isHovered || isDragging) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

  }, [localPoints, width, height, color, backgroundColor, gridColor, showGrid, gridSize, normToCanvas, getCurvePoint, hoveredIndex, draggingIndex]);

  // Mouse handlers
  const handleMouseDown = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const pointIndex = findPointNear(x, y);
    
    if (pointIndex !== null) {
      setDraggingIndex(pointIndex);
    } else if (localPoints.length < maxPoints) {
      // Add new point
      const norm = canvasToNorm(x, y);
      const newPoints = [...localPoints, { ...norm, curve: 0 }]
        .sort((a, b) => a.x - b.x);
      setLocalPoints(newPoints);
      onChange?.(newPoints);
    }
  }, [findPointNear, localPoints, maxPoints, canvasToNorm, onChange]);

  const handleMouseMove = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    if (draggingIndex !== null) {
      const norm = canvasToNorm(x, y);
      const newPoints = [...localPoints];
      
      // First and last points locked to x=0 and x=1
      if (draggingIndex === 0) {
        newPoints[draggingIndex] = { ...newPoints[draggingIndex], y: norm.y };
      } else if (draggingIndex === localPoints.length - 1) {
        newPoints[draggingIndex] = { ...newPoints[draggingIndex], y: norm.y };
      } else {
        newPoints[draggingIndex] = { ...newPoints[draggingIndex], ...norm };
      }
      
      setLocalPoints(newPoints);
      onChange?.(newPoints);
    } else {
      setHoveredIndex(findPointNear(x, y));
    }
  }, [draggingIndex, localPoints, canvasToNorm, findPointNear, onChange]);

  const handleMouseUp = useCallback(() => {
    setDraggingIndex(null);
  }, []);

  const handleDoubleClick = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    
    const pointIndex = findPointNear(x, y);
    
    if (pointIndex !== null && pointIndex !== 0 && pointIndex !== localPoints.length - 1) {
      // Delete point (except first and last)
      if (localPoints.length > minPoints) {
        const newPoints = localPoints.filter((_, i) => i !== pointIndex);
        setLocalPoints(newPoints);
        onChange?.(newPoints);
      }
    }
  }, [findPointNear, localPoints, minPoints, onChange]);

  // Scroll to adjust curve
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    
    if (hoveredIndex !== null && hoveredIndex < localPoints.length - 1) {
      const newPoints = [...localPoints];
      const delta = e.deltaY > 0 ? -0.1 : 0.1;
      newPoints[hoveredIndex] = {
        ...newPoints[hoveredIndex],
        curve: Math.max(-1, Math.min(1, (newPoints[hoveredIndex].curve || 0) + delta)),
      };
      setLocalPoints(newPoints);
      onChange?.(newPoints);
    }
  }, [hoveredIndex, localPoints, onChange]);

  return (
    <div style={{ position: 'relative' }}>
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        style={{ 
          borderRadius, 
          cursor: draggingIndex !== null ? 'grabbing' : hoveredIndex !== null ? 'grab' : 'crosshair' 
        }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onDoubleClick={handleDoubleClick}
        onWheel={handleWheel}
      />
      <div style={{
        position: 'absolute',
        bottom: 4,
        right: 8,
        fontSize: 9,
        color: `${color}88`,
        pointerEvents: 'none',
      }}>
        Click to add • Double-click to delete • Scroll to curve
      </div>
    </div>
  );
};

export default MSEGEditor;
