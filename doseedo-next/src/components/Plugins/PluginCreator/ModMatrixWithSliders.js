/**
 * ModMatrix with depth sliders - enhanced version
 */
import React, { useState, useCallback } from 'react';

const ModMatrixWithSliders = ({
  sources = ['LFO 1', 'LFO 2', 'Env 1', 'Env 2', 'Velocity'],
  destinations = ['Cutoff', 'Resonance', 'Pitch', 'Volume'],
  connections = [],
  onChange,
  color = '#667eea',
}) => {
  const getConnection = (srcIdx, destIdx) => 
    connections.find(c => c.source === srcIdx && c.dest === destIdx);

  const updateDepth = (srcIdx, destIdx, depth) => {
    const newConns = connections.map(c => 
      (c.source === srcIdx && c.dest === destIdx) 
        ? { ...c, depth: parseFloat(depth) } 
        : c
    );
    onChange?.(newConns);
  };

  const toggleConnection = (srcIdx, destIdx) => {
    const existing = getConnection(srcIdx, destIdx);
    if (existing) {
      onChange?.(connections.filter(c => !(c.source === srcIdx && c.dest === destIdx)));
    } else {
      onChange?.([...connections, { source: srcIdx, dest: destIdx, depth: 0.5 }]);
    }
  };

  return (
    <div style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, padding: 12 }}>
      <div style={{ display: 'grid', gridTemplateColumns: `100px repeat(${destinations.length}, 80px)`, gap: 4 }}>
        <div />
        {destinations.map((d, i) => (
          <div key={i} style={{ fontSize: 10, color: `${color}aa`, textAlign: 'center', transform: 'rotate(-45deg)', height: 40 }}>{d}</div>
        ))}
        {sources.map((src, srcIdx) => (
          <React.Fragment key={srcIdx}>
            <div style={{ fontSize: 11, color: `${color}cc`, display: 'flex', alignItems: 'center' }}>{src}</div>
            {destinations.map((_, destIdx) => {
              const conn = getConnection(srcIdx, destIdx);
              return (
                <div key={destIdx} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', padding: 4, border: `1px solid ${color}33`, borderRadius: 4 }}>
                  <input
                    type="checkbox"
                    checked={!!conn}
                    onChange={() => toggleConnection(srcIdx, destIdx)}
                    style={{ cursor: 'pointer' }}
                  />
                  {conn && (
                    <input
                      type="range"
                      min="-1"
                      max="1"
                      step="0.05"
                      value={conn.depth}
                      onChange={(e) => updateDepth(srcIdx, destIdx, e.target.value)}
                      style={{ width: 60, marginTop: 4, accentColor: color }}
                    />
                  )}
                  {conn && <span style={{ fontSize: 9, color: '#fff' }}>{(conn.depth * 100).toFixed(0)}%</span>}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

export default ModMatrixWithSliders;
