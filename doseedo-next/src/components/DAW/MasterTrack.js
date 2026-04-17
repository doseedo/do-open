import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';

/**
 * MasterTrack Component - Master volume and FX controls
 * Matches original doseedo2.html master track section
 */
function MasterTrack() {
  const { state, dispatch } = useApp();
  const [masterGain, setMasterGain] = useState(1);
  const [showReverb, setShowReverb] = useState(false);
  const [showEQ, setShowEQ] = useState(false);

  const handleMasterGainChange = (e) => {
    const value = parseFloat(e.target.value);
    setMasterGain(value);
    dispatch({
      type: 'UPDATE_MASTER_GAIN',
      payload: value
    });
  };

  const toggleReverbPanel = () => {
    setShowReverb(!showReverb);
    setShowEQ(false); // Close EQ if open
    dispatch({
      type: 'TOGGLE_MASTER_REVERB_PANEL'
    });
  };

  const toggleEQPanel = () => {
    setShowEQ(!showEQ);
    setShowReverb(false); // Close reverb if open
    dispatch({
      type: 'TOGGLE_MASTER_EQ_PANEL'
    });
  };

  return (
    <div
      className="trackselect"
      id="master-track"
      style={{
        position: 'absolute',
        bottom: '10px',
        left: '10px',
        right: '10px',
        height: '50px',
        background: '#252525',
        border: '1px solid #444',
        borderRadius: '4px',
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        padding: '0 10px'
      }}
    >
      <span style={{ fontSize: '12px', color: '#aaa', minWidth: '50px' }}>
        MASTER
      </span>

      <input
        type="range"
        id="master-gain"
        min="0"
        max="1"
        step="0.01"
        value={masterGain}
        onChange={handleMasterGainChange}
        style={{ flex: 1, cursor: 'pointer' }}
      />

      <span
        id="master-gain-value"
        style={{ fontSize: '11px', color: '#aaa', minWidth: '35px' }}
      >
        {Math.round(masterGain * 100)}%
      </span>

      <button
        id="master-reverb-btn"
        onClick={toggleReverbPanel}
        className={showReverb ? 'active' : ''}
        style={{
          width: '35px',
          height: '25px',
          fontSize: '10px',
          background: showReverb ? '#667eea' : '#2a2a2a',
          color: showReverb ? '#fff' : '#aaa',
          border: '1px solid #444',
          borderRadius: '3px',
          cursor: 'pointer'
        }}
      >
        REV
      </button>

      <button
        id="master-eq-btn"
        onClick={toggleEQPanel}
        className={showEQ ? 'active' : ''}
        style={{
          width: '30px',
          height: '25px',
          fontSize: '10px',
          background: showEQ ? '#667eea' : '#2a2a2a',
          color: showEQ ? '#fff' : '#aaa',
          border: '1px solid #444',
          borderRadius: '3px',
          cursor: 'pointer'
        }}
      >
        EQ
      </button>
    </div>
  );
}

export default MasterTrack;
