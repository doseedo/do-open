import React, { useState, useCallback, useMemo, useRef } from 'react';
import Draggable from 'react-draggable';
import { useApp } from '../../context/AppContext';
import { useWaveform } from '../../hooks/useWaveform';

/**
 * DraggableTrack - React-efficient draggable, resizable track with crop masks
 */
const DraggableTrack = React.memo(({ track, busId, mode, index, isSelected, pixelsPerSecond }) => {
  const { dispatch } = useApp();
  const TRACK_HEIGHT = 60;

  // Track position and dimensions
  const [position, setPosition] = useState({ x: (track.startPosition || 0) * pixelsPerSecond, y: 0 });
  const [cropStart, setCropStart] = useState(track.cropStart || 0); // In seconds
  const [cropEnd, setCropEnd] = useState(track.cropEnd || 0); // In seconds
  const [isResizing, setIsResizing] = useState(false);
  const nodeRef = useRef(null);

  // Waveform rendering
  const { canvasRef } = useWaveform(
    track.audioUrl,
    track.width || 800,
    TRACK_HEIGHT,
    isSelected ? '#8b5cf6' : (track.isPlaceholder ? '#666' : '#667eea')
  );

  // Handle drag
  const handleDrag = useCallback((e, data) => {
    const newPositionSeconds = data.x / pixelsPerSecond;
    setPosition({ x: data.x, y: 0 });

    // Update track metadata in state
    dispatch({
      type: 'UPDATE_TRACK',
      payload: {
        busId,
        trackId: track.id,
        updates: { startPosition: newPositionSeconds }
      }
    });
  }, [dispatch, track.id, busId, pixelsPerSecond]);

  // Handle track selection
  const handleTrackClick = useCallback(() => {
    dispatch({ type: 'SELECT_TRACK', payload: { trackId: track.id, busId } });
  }, [dispatch, track.id, busId]);

  // Crop mask styles
  const leftMaskStyle = useMemo(() => ({
    position: 'absolute',
    left: 0,
    top: 0,
    width: `${cropStart * pixelsPerSecond}px`,
    height: '100%',
    background: 'rgba(0,0,0,0.6)',
    pointerEvents: 'none',
    zIndex: 10
  }), [cropStart, pixelsPerSecond]);

  const rightMaskStyle = useMemo(() => {
    const trackDuration = track.duration || 5;
    const cropEndPx = cropEnd * pixelsPerSecond;
    return {
      position: 'absolute',
      right: 0,
      top: 0,
      width: `${cropEndPx}px`,
      height: '100%',
      background: 'rgba(0,0,0,0.6)',
      pointerEvents: 'none',
      zIndex: 10
    };
  }, [cropEnd, pixelsPerSecond, track.duration]);

  // Track container style
  const trackStyle = useMemo(() => ({
    position: 'relative',
    height: `${TRACK_HEIGHT}px`,
    cursor: isResizing ? 'ew-resize' : 'grab',
    border: isSelected ? '2px solid #8b5cf6' : '1px solid #444',
    borderRadius: '4px',
    overflow: 'hidden',
    background: '#1a1a1a'
  }), [isSelected, isResizing]);

  // Resize handles
  const ResizeHandle = ({ side }) => {
    const handleMouseDown = (e) => {
      e.stopPropagation();
      setIsResizing(true);

      const startX = e.clientX;
      const startCrop = side === 'left' ? cropStart : cropEnd;

      const handleMouseMove = (moveEvent) => {
        const delta = (moveEvent.clientX - startX) / pixelsPerSecond;

        if (side === 'left') {
          const newCropStart = Math.max(0, Math.min(startCrop + delta, (track.duration || 5) - cropEnd));
          setCropStart(newCropStart);
        } else {
          const newCropEnd = Math.max(0, Math.min(startCrop - delta, (track.duration || 5) - cropStart));
          setCropEnd(newCropEnd);
        }
      };

      const handleMouseUp = () => {
        setIsResizing(false);
        document.removeEventListener('mousemove', handleMouseMove);
        document.removeEventListener('mouseup', handleMouseUp);

        // Update track crop in state
        dispatch({
          type: 'UPDATE_TRACK',
          payload: {
            busId,
            trackId: track.id,
            updates: { cropStart, cropEnd }
          }
        });
      };

      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    };

    return (
      <div
        onMouseDown={handleMouseDown}
        style={{
          position: 'absolute',
          [side]: 0,
          top: 0,
          width: '8px',
          height: '100%',
          background: 'rgba(139, 92, 246, 0.5)',
          cursor: 'ew-resize',
          zIndex: 20,
          display: isSelected ? 'block' : 'none'
        }}
      />
    );
  };

  return (
    <Draggable
      nodeRef={nodeRef}
      axis="x"
      position={position}
      onDrag={handleDrag}
      bounds={{ left: 0 }}
      disabled={isResizing}
    >
      <div
        ref={nodeRef}
        style={{
          position: 'absolute',
          top: `${index * TRACK_HEIGHT}px`,
          width: `${(track.width || 800)}px`,
          height: `${TRACK_HEIGHT}px`,
          zIndex: 10 + index
        }}
      >
        <div style={trackStyle} onClick={handleTrackClick}>
          {/* Waveform canvas */}
          <canvas
            ref={canvasRef}
            width={track.width || 800}
            height={TRACK_HEIGHT}
            style={{ display: 'block' }}
          />

          {/* Crop masks */}
          {cropStart > 0 && <div style={leftMaskStyle} />}
          {cropEnd > 0 && <div style={rightMaskStyle} />}

          {/* Resize handles */}
          <ResizeHandle side="left" />
          <ResizeHandle side="right" />

          {/* Track label */}
          <div style={{
            position: 'absolute',
            bottom: 0,
            left: 0,
            right: 0,
            padding: '2px 6px',
            background: 'rgba(0,0,0,0.7)',
            color: '#fff',
            fontSize: '11px',
            whiteSpace: 'nowrap',
            overflow: 'hidden',
            textOverflow: 'ellipsis'
          }}>
            {track.name || track.audioUrl?.split('/').pop() || 'Untitled'}
          </div>
        </div>
      </div>
    </Draggable>
  );
});

DraggableTrack.displayName = 'DraggableTrack';

export default DraggableTrack;
