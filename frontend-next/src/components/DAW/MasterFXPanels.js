import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';

/**
 * MasterFXPanels Component - Master Reverb and EQ panels
 * Matches original doseedo2.html master FX panels
 */
function MasterFXPanels() {
  const { state } = useApp();
  const [reverbMix, setReverbMix] = useState(0.3);
  const [eqBands, setEqBands] = useState({
    '60Hz': 0,
    '250Hz': 0,
    '1kHz': 0,
    '4kHz': 0,
    '12kHz': 0
  });

  const showReverb = state.masterFX?.showReverb || false;
  const showEQ = state.masterFX?.showEQ || false;

  const handleReverbMixChange = (e) => {
    const value = parseFloat(e.target.value);
    setReverbMix(value);
  };

  const handleEQChange = (band, value) => {
    setEqBands(prev => ({
      ...prev,
      [band]: parseFloat(value)
    }));
  };

  return (
    <>
      {/* Master Reverb Panel */}
      <div
        id="master-reverb-panel"
        style={{
          display: showReverb ? 'block' : 'none',
          position: 'absolute',
          bottom: '70px',
          left: '10px',
          right: '10px',
          background: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '4px',
          padding: '10px',
          zIndex: 1000
        }}
      >
        <div style={{ marginBottom: '10px', fontSize: '12px', color: '#aaa' }}>
          Master Reverb
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <label style={{ fontSize: '11px', color: '#888', minWidth: '60px' }}>
            Mix:
          </label>
          <input
            type="range"
            id="master-reverb-mix"
            min="0"
            max="1"
            step="0.01"
            value={reverbMix}
            onChange={handleReverbMixChange}
            style={{ flex: 1 }}
          />
          <span
            id="master-reverb-mix-value"
            style={{ fontSize: '11px', color: '#aaa', minWidth: '40px' }}
          >
            {Math.round(reverbMix * 100)}%
          </span>
        </div>
      </div>

      {/* Master EQ Panel */}
      <div
        id="master-eq-panel"
        style={{
          display: showEQ ? 'block' : 'none',
          position: 'absolute',
          bottom: '70px',
          left: '10px',
          right: '10px',
          background: '#1a1a1a',
          border: '1px solid #333',
          borderRadius: '4px',
          padding: '10px',
          zIndex: 1000
        }}
      >
        <div style={{ marginBottom: '10px', fontSize: '12px', color: '#aaa' }}>
          Master EQ
        </div>
        <div
          style={{
            display: 'flex',
            gap: '15px',
            justifyContent: 'space-around',
            alignItems: 'flex-end',
            height: '120px'
          }}
        >
          {Object.entries(eqBands).map(([band, value], idx) => (
            <div
              key={band}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '5px'
              }}
            >
              <span style={{ fontSize: '10px', color: '#aaa' }}>
                {value > 0 ? '+' : ''}{value}dB
              </span>
              <input
                type="range"
                id={`master-eq-${idx + 1}`}
                min="-12"
                max="12"
                step="0.5"
                value={value}
                onChange={(e) => handleEQChange(band, e.target.value)}
                orient="vertical"
                style={{
                  writingMode: 'bt-lr',
                  WebkitAppearance: 'slider-vertical',
                  width: '20px',
                  height: '80px'
                }}
              />
              <span style={{ fontSize: '10px', color: '#666' }}>{band}</span>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

export default MasterFXPanels;
