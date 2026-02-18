import React, { useRef, useEffect, useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useThemeColor } from '../../hooks/useThemeColor';
import './AutomationWindow.css';

/**
 * AutomationWindow Component
 * Handles volume automation envelope drawing and editing
 * Based on the automation canvas from doseedo2.html with all features
 * - Volume automation with adjustable points
 * - Drag bars for bulk X/Y adjustment
 * - Edge points that span the timeline
 * - Scene change restoration
 */
function AutomationWindow() {
  const { state, dispatch } = useApp();
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [selectedPointIndex, setSelectedPointIndex] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isDraggingX, setIsDraggingX] = useState(false);

  // Get theme colors
  const primaryBlue = useThemeColor('--color-primary-blue', '#667eea');
  const [isDraggingY, setIsDraggingY] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartY, setDragStartY] = useState(0);
  const [timelineWidth, setTimelineWidth] = useState(950);

  const totalDuration = state.totalDuration || 10;
  const minVolume = 0.2;
  const maxVolume = 1.0;

  // Convert canvas Y to volume (0.2 - 1.0)
  const yToVolume = useCallback((y, canvasHeight) => {
    const normalized = 1 - (y / canvasHeight);
    return minVolume + normalized * (maxVolume - minVolume);
  }, [minVolume, maxVolume]);

  // Convert volume to canvas Y
  const volumeToY = useCallback((volume, canvasHeight) => {
    const normalized = (volume - minVolume) / (maxVolume - minVolume);
    return canvasHeight - (normalized * canvasHeight);
  }, [minVolume, maxVolume]);

  // Initialize edge points if empty
  const initializePoints = useCallback(() => {
    if (state.automationWindow.points.length === 0) {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const midVolume = 0.5; // 50% default volume
      const edgePoints = [
        { time: 0, volume: midVolume, isEdge: true },
        { time: totalDuration, volume: midVolume, isEdge: true }
      ];

      dispatch({ type: 'UPDATE_AUTOMATION_POINTS', payload: edgePoints });
    }
  }, [state.automationWindow.points.length, totalDuration, dispatch]);

  // Listen for timeline width changes and match the timeline's width calculation
  useEffect(() => {
    const handleWidthChange = () => {
      // Match Timeline.js width calculation: containerWidth * zoomLevel
      const containerWidth = state.timelineWidth || 950;
      const zoomLevel = state.zoomLevel || 1.0;
      const calculatedWidth = containerWidth * zoomLevel;
      console.log('🎨 Automation window width:', { containerWidth, zoomLevel, calculatedWidth });
      setTimelineWidth(calculatedWidth);
    };

    handleWidthChange();
    window.addEventListener('busLabelWidthChanged', handleWidthChange);
    window.addEventListener('resize', handleWidthChange);
    return () => {
      window.removeEventListener('busLabelWidthChanged', handleWidthChange);
      window.removeEventListener('resize', handleWidthChange);
    };
  }, [state.timelineWidth, state.zoomLevel]);

  // Initialize when automation window becomes visible
  useEffect(() => {
    if (state.automationWindow.isVisible && canvasRef.current) {
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          initializePoints();
          resizeCanvas();
          draw();
        });
      });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.automationWindow.isVisible, timelineWidth]);

  // Redraw when points change or selection changes
  useEffect(() => {
    if (state.automationWindow.isVisible && canvasRef.current) {
      draw();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [state.automationWindow.points, totalDuration, selectedPointIndex, primaryBlue]);

  // Update edge points when duration changes
  useEffect(() => {
    if (state.automationWindow.points.length > 0) {
      const updatedPoints = state.automationWindow.points.map(point => {
        if (point.isEdge && point.time !== 0) {
          return { ...point, time: totalDuration };
        }
        return point;
      });
      if (JSON.stringify(updatedPoints) !== JSON.stringify(state.automationWindow.points)) {
        dispatch({ type: 'UPDATE_AUTOMATION_POINTS', payload: updatedPoints });
      }
    }
  }, [totalDuration]);

  const resizeCanvas = () => {
    if (!canvasRef.current) return;
    const canvas = canvasRef.current;
    const dpr = window.devicePixelRatio || 1;

    // Set display size
    canvas.style.width = '100%';
    canvas.style.height = '80px';

    // Get actual size
    const rect = canvas.getBoundingClientRect();

    // Set canvas resolution (accounting for device pixel ratio for sharp rendering)
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;

    // Scale context to match device pixel ratio
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };

  const draw = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const points = state.automationWindow.points;
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw grid (spacing of 10px like legacy)
    ctx.strokeStyle = '#222';
    ctx.lineWidth = 1;

    for (let y = 0; y < height; y += 10) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw automation line
    if (points.length > 0) {
      const sorted = [...points].sort((a, b) => a.time - b.time);

      ctx.strokeStyle = primaryBlue;
      ctx.lineWidth = 2;
      ctx.beginPath();

      sorted.forEach((point, index) => {
        const x = (point.time / totalDuration) * width;
        const y = volumeToY(point.volume, height);

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw points (but not edge points)
      sorted.forEach((point, index) => {
        if (!point.isEdge) {
          const x = (point.time / totalDuration) * width;
          const y = volumeToY(point.volume, height);

          // Find original index for comparison with selectedPointIndex
          const originalIndex = state.automationWindow.points.findIndex(p =>
            p.time === point.time && p.volume === point.volume && p.isEdge === point.isEdge
          );

          ctx.fillStyle = originalIndex === selectedPointIndex ? '#ffffff' : primaryBlue;
          ctx.strokeStyle = '#333';
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(x, y, 6, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
      });
    }
  };

  const getCanvasCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    return { x, y };
  };

  const findPointNearCursor = (x, y) => {
    const canvas = canvasRef.current;
    if (!canvas) return -1;

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const threshold = 15; // pixels - increased for easier grabbing

    for (let i = 0; i < state.automationWindow.points.length; i++) {
      const point = state.automationWindow.points[i];
      if (point.isEdge) continue; // Can't select edge points

      const px = (point.time / totalDuration) * width;
      const py = volumeToY(point.volume, height);

      const distance = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
      if (distance <= threshold) {
        return i;
      }
    }
    return -1;
  };

  const handleCanvasMouseDown = (e) => {
    const { x, y } = getCanvasCoords(e);
    const pointIndex = findPointNearCursor(x, y);

    if (pointIndex !== -1) {
      // Start dragging existing point
      setSelectedPointIndex(pointIndex);
      setIsDragging(true);
      if (canvasRef.current) {
        canvasRef.current.style.cursor = 'grabbing';
      }
    } else {
      // Add new point
      const canvas = canvasRef.current;
      const width = canvas.offsetWidth;
      const height = canvas.offsetHeight;
      const time = (x / width) * totalDuration;
      const volume = Math.max(minVolume, Math.min(maxVolume, yToVolume(y, height)));

      const newPoint = { time, volume, isEdge: false };
      const updatedPoints = [...state.automationWindow.points, newPoint].sort((a, b) => a.time - b.time);

      dispatch({ type: 'UPDATE_AUTOMATION_POINTS', payload: updatedPoints });
    }
  };

  const handleCanvasMouseMove = (e) => {
    if (!isDragging || selectedPointIndex === null) return;

    const { x, y } = getCanvasCoords(e);
    const canvas = canvasRef.current;
    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;

    const time = Math.max(0, Math.min(totalDuration, (x / width) * totalDuration));
    const volume = Math.max(minVolume, Math.min(maxVolume, yToVolume(y, height)));

    const updatedPoints = state.automationWindow.points.map((point, idx) => {
      if (idx === selectedPointIndex) {
        return { ...point, time, volume };
      }
      return point;
    });

    // Re-sort points by time and update selected index
    const sortedPoints = [...updatedPoints].sort((a, b) => a.time - b.time);

    // Find the new index after sorting
    const draggedPoint = updatedPoints[selectedPointIndex];
    const newIndex = sortedPoints.findIndex(p =>
      p.time === draggedPoint.time && p.volume === draggedPoint.volume
    );

    setSelectedPointIndex(newIndex);
    dispatch({ type: 'UPDATE_AUTOMATION_POINTS', payload: sortedPoints });
  };

  const handleCanvasMouseUp = () => {
    setIsDragging(false);
    setSelectedPointIndex(null);
    if (canvasRef.current) {
      canvasRef.current.style.cursor = 'crosshair';
    }
  };

  const handleCanvasRightClick = (e) => {
    e.preventDefault();
    const { x, y } = getCanvasCoords(e);
    const pointIndex = findPointNearCursor(x, y);

    if (pointIndex !== -1 && !state.automationWindow.points[pointIndex].isEdge) {
      // Remove point (but not edge points)
      const updatedPoints = state.automationWindow.points.filter((_, i) => i !== pointIndex);
      dispatch({ type: 'UPDATE_AUTOMATION_POINTS', payload: updatedPoints });
    }
  };

  // Update cursor style when hovering over points
  const handleCanvasHover = (e) => {
    if (isDragging) return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const width = canvas.offsetWidth;
    const height = canvas.offsetHeight;
    const threshold = 15;

    let foundPoint = false;
    for (let i = 0; i < state.automationWindow.points.length; i++) {
      const point = state.automationWindow.points[i];
      if (point.isEdge) continue;

      const px = (point.time / totalDuration) * width;
      const py = volumeToY(point.volume, height);

      const distance = Math.sqrt((x - px) ** 2 + (y - py) ** 2);
      if (distance <= threshold) {
        foundPoint = true;
        break;
      }
    }

    canvas.style.cursor = foundPoint ? 'grab' : 'crosshair';
  };

  // Drag bar handlers for bulk X/Y movement
  const handleDragBarXDown = (e) => {
    e.stopPropagation();
    setIsDraggingX(true);
    setDragStartX(e.clientX);
    e.currentTarget.classList.add('dragging');
  };

  const handleDragBarYDown = (e) => {
    e.stopPropagation();
    setIsDraggingY(true);
    setDragStartY(e.clientY);
    e.currentTarget.classList.add('dragging');
  };

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (isDraggingX || isDraggingY) {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const width = canvas.offsetWidth;
        const height = canvas.offsetHeight;
        const dx = e.clientX - dragStartX;
        const dy = e.clientY - dragStartY;

        setDragStartX(e.clientX);
        setDragStartY(e.clientY);

        const updatedPoints = state.automationWindow.points.map(point => {
          let newPoint = { ...point };

          if (isDraggingX && !point.isEdge) {
            const deltaTime = (dx / width) * totalDuration;
            newPoint.time = Math.min(totalDuration, Math.max(0, point.time + deltaTime));
          }

          if (isDraggingY) {
            const currentY = volumeToY(point.volume, height);
            const newY = Math.max(0, Math.min(height, currentY + dy));
            newPoint.volume = Math.max(minVolume, Math.min(maxVolume, yToVolume(newY, height)));
          }

          return newPoint;
        });

        dispatch({ type: 'UPDATE_AUTOMATION_POINTS', payload: updatedPoints });
      }
    };

    const handleMouseUp = () => {
      if (isDraggingX || isDraggingY) {
        setIsDraggingX(false);
        setIsDraggingY(false);

        const dragBarX = document.getElementById('automation-drag-bar-x');
        const dragBarY = document.getElementById('automation-drag-bar-y');
        if (dragBarX) dragBarX.classList.remove('dragging');
        if (dragBarY) dragBarY.classList.remove('dragging');
      }
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDraggingX, isDraggingY, dragStartX, dragStartY, state.automationWindow.points, totalDuration, yToVolume, volumeToY, dispatch]);

  if (!state.automationWindow.isVisible) {
    return null;
  }

  return (
    <div className="automation-window scrollable" ref={containerRef} style={{ width: `${timelineWidth}px` }}>
      <div
        id="automation-drag-bar-x"
        className="automation-drag-bar-x"
        onMouseDown={handleDragBarXDown}
        title="Drag to shift all points horizontally"
      ></div>
      <div
        id="automation-drag-bar-y"
        className="automation-drag-bar-y"
        onMouseDown={handleDragBarYDown}
        title="Drag to shift all points vertically"
      ></div>
      <canvas
        ref={canvasRef}
        id="automation-canvas"
        className="automation-canvas"
        onMouseDown={handleCanvasMouseDown}
        onMouseMove={(e) => {
          handleCanvasMouseMove(e);
          if (!isDragging) handleCanvasHover(e);
        }}
        onMouseUp={handleCanvasMouseUp}
        onMouseLeave={handleCanvasMouseUp}
        onContextMenu={handleCanvasRightClick}
      ></canvas>
    </div>
  );
}

export default AutomationWindow;
