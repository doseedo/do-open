/**
 * MSEGBindingPicker - Dropdown to select modulation target
 */

import React, { useState, useMemo } from 'react';
import { getModulatableTargets, createModConnection } from './modulationUtils';

const MSEGBindingPicker = ({
  msegId = 'mseg1',
  dspConfig,
  connections = [],
  onConnect,
  onDisconnect,
  color = '#667eea',
}) => {
  const [isOpen, setIsOpen] = useState(false);
  
  const targets = useMemo(() => 
    getModulatableTargets(dspConfig), 
    [dspConfig]
  );
  
  const connectedTargets = useMemo(() => 
    connections.filter(c => c.source === msegId).map(c => c.target),
    [connections, msegId]
  );

  const handleToggle = (targetId) => {
    if (connectedTargets.includes(targetId)) {
      const conn = connections.find(c => c.source === msegId && c.target === targetId);
      if (conn) onDisconnect?.(conn.id);
    } else {
      const newConn = createModConnection(msegId, targetId, 0.5);
      onConnect?.(newConn);
    }
  };

  return (
    <div style={{ position: 'relative', display: 'inline-block' }}>
      <button
        onClick={() => setIsOpen(!isOpen)}
        style={{
          padding: '6px 12px',
          backgroundColor: `${color}22`,
          border: `1px solid ${color}66`,
          borderRadius: 6,
          color: color,
          fontSize: 11,
          cursor: 'pointer',
          display: 'flex',
          alignItems: 'center',
          gap: 6,
        }}
      >
        <i className="fa-solid fa-link" />
        Bind to... ({connectedTargets.length})
        <i className={`fa-solid fa-chevron-${isOpen ? 'up' : 'down'}`} style={{ fontSize: 9 }} />
      </button>

      {isOpen && (
        <div style={{
          position: 'absolute',
          top: '100%',
          left: 0,
          marginTop: 4,
          backgroundColor: '#1a1a2e',
          border: `1px solid ${color}44`,
          borderRadius: 8,
          padding: 8,
          minWidth: 180,
          zIndex: 1000,
          maxHeight: 200,
          overflowY: 'auto',
        }}>
          {targets.length === 0 ? (
            <div style={{ color: '#888', fontSize: 11, padding: 8 }}>
              No modulatable parameters
            </div>
          ) : (
            targets.map(target => {
              const isConnected = connectedTargets.includes(target.id);
              return (
                <div
                  key={target.id}
                  onClick={() => handleToggle(target.id)}
                  style={{
                    padding: '6px 10px',
                    borderRadius: 4,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    backgroundColor: isConnected ? `${color}33` : 'transparent',
                    marginBottom: 2,
                  }}
                  onMouseEnter={(e) => e.target.style.backgroundColor = `${color}22`}
                  onMouseLeave={(e) => e.target.style.backgroundColor = isConnected ? `${color}33` : 'transparent'}
                >
                  <span style={{ color: '#fff', fontSize: 11 }}>
                    {target.name}
                  </span>
                  {isConnected && (
                    <i className="fa-solid fa-check" style={{ color, fontSize: 10 }} />
                  )}
                </div>
              );
            })
          )}
        </div>
      )}
    </div>
  );
};

export default MSEGBindingPicker;
