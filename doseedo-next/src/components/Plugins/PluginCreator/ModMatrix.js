/**
 * ModMatrix - Serum-style modulation routing matrix
 * Drag sources to destinations with depth control
 */

import React, { useState, useCallback } from 'react';

const ModMatrix = ({
  sources = ['LFO 1', 'LFO 2', 'Env 1', 'Env 2', 'Velocity', 'Mod Wheel'],
  destinations = ['Cutoff', 'Resonance', 'Pitch', 'Pan', 'Volume', 'LFO Rate'],
  connections = [], // [{source: 0, dest: 1, depth: 0.5}]
  onChange,
  color = '#667eea',
  backgroundColor = 'rgba(0,0,0,0.3)',
  width = 400,
  cellSize = 40,
}) => {
  const [dragging, setDragging] = useState(null);
  const [hoveredCell, setHoveredCell] = useState(null);

  const getConnection = useCallback((srcIdx, destIdx) => {
    return connections.find(c => c.source === srcIdx && c.dest === destIdx);
  }, [connections]);

  const handleCellClick = useCallback((srcIdx, destIdx) => {
    const existing = getConnection(srcIdx, destIdx);
    
    if (existing) {
      // Remove connection
      const newConnections = connections.filter(
        c => !(c.source === srcIdx && c.dest === destIdx)
      );
      onChange?.(newConnections);
    } else {
      // Add connection with default depth
      onChange?.([...connections, { source: srcIdx, dest: destIdx, depth: 0.5 }]);
    }
  }, [connections, getConnection, onChange]);

  const handleDepthChange = useCallback((srcIdx, destIdx, depth) => {
    const newConnections = connections.map(c => {
      if (c.source === srcIdx && c.dest === destIdx) {
        return { ...c, depth: Math.max(-1, Math.min(1, depth)) };
      }
      return c;
    });
    onChange?.(newConnections);
  }, [connections, onChange]);

  const handleWheel = useCallback((e, srcIdx, destIdx) => {
    e.preventDefault();
    const conn = getConnection(srcIdx, destIdx);
    if (conn) {
      const delta = e.deltaY > 0 ? -0.05 : 0.05;
      handleDepthChange(srcIdx, destIdx, conn.depth + delta);
    }
  }, [getConnection, handleDepthChange]);

  const height = (sources.length + 1) * cellSize;
  const actualWidth = (destinations.length + 1) * cellSize;

  return (
    <div style={{ 
      display: 'inline-block',
      backgroundColor,
      borderRadius: 8,
      padding: 8,
      overflow: 'hidden',
    }}>
      {/* Header row - destinations */}
      <div style={{ display: 'flex' }}>
        <div style={{ width: cellSize * 1.5, height: cellSize }} />
        {destinations.map((dest, i) => (
          <div
            key={i}
            style={{
              width: cellSize,
              height: cellSize,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 9,
              color: `${color}cc`,
              writingMode: 'vertical-rl',
              textOrientation: 'mixed',
              transform: 'rotate(180deg)',
            }}
          >
            {dest}
          </div>
        ))}
      </div>

      {/* Source rows */}
      {sources.map((src, srcIdx) => (
        <div key={srcIdx} style={{ display: 'flex' }}>
          {/* Source label */}
          <div style={{
            width: cellSize * 1.5,
            height: cellSize,
            display: 'flex',
            alignItems: 'center',
            paddingLeft: 8,
            fontSize: 10,
            color: `${color}cc`,
          }}>
            {src}
          </div>

          {/* Matrix cells */}
          {destinations.map((_, destIdx) => {
            const conn = getConnection(srcIdx, destIdx);
            const isHovered = hoveredCell?.src === srcIdx && hoveredCell?.dest === destIdx;
            
            return (
              <div
                key={destIdx}
                onClick={() => handleCellClick(srcIdx, destIdx)}
                onWheel={(e) => handleWheel(e, srcIdx, destIdx)}
                onMouseEnter={() => setHoveredCell({ src: srcIdx, dest: destIdx })}
                onMouseLeave={() => setHoveredCell(null)}
                style={{
                  width: cellSize,
                  height: cellSize,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  cursor: 'pointer',
                  backgroundColor: isHovered ? `${color}22` : 'transparent',
                  border: `1px solid ${color}33`,
                  transition: 'background-color 0.15s',
                }}
              >
                {conn && (
                  <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                    {/* Depth indicator */}
                    <div style={{
                      position: 'absolute',
                      bottom: 0,
                      left: '10%',
                      width: '80%',
                      height: `${Math.abs(conn.depth) * 100}%`,
                      backgroundColor: conn.depth >= 0 ? color : '#e74c3c',
                      opacity: 0.6,
                      borderRadius: 2,
                      transformOrigin: conn.depth >= 0 ? 'bottom' : 'top',
                    }} />
                    {/* Depth value */}
                    <div style={{
                      position: 'absolute',
                      top: '50%',
                      left: '50%',
                      transform: 'translate(-50%, -50%)',
                      fontSize: 9,
                      fontWeight: 600,
                      color: '#fff',
                      textShadow: '0 1px 2px rgba(0,0,0,0.5)',
                    }}>
                      {conn.depth > 0 ? '+' : ''}{(conn.depth * 100).toFixed(0)}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      ))}

      {/* Legend */}
      <div style={{
        marginTop: 8,
        fontSize: 9,
        color: `${color}88`,
        textAlign: 'center',
      }}>
        Click to toggle • Scroll to adjust depth
      </div>
    </div>
  );
};

export default ModMatrix;
