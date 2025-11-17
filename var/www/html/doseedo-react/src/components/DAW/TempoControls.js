import React from 'react';
import { useApp } from '../../context/AppContext';
import styles from './TempoControls.module.css';

/**
 * TempoControls Component - BPM and metronome controls
 * Matches original doseedo2.html .tempocontrols structure
 */
function TempoControls() {
  const { state, dispatch } = useApp();

  // Use global state directly instead of local state
  const bpm = state.bpm;
  const isBPMMode = state.isBPMMode;
  const isMetronomeOn = state.isMetronomeOn;

  const handleBPMChange = (e) => {
    const value = parseInt(e.target.value, 10);
    dispatch({
      type: 'UPDATE_BPM',
      payload: value
    });
  };

  const toggleBPMMode = () => {
    dispatch({
      type: 'TOGGLE_BPM_MODE'
    });
  };

  const toggleMetronome = () => {
    dispatch({
      type: 'TOGGLE_METRONOME'
    });
  };

  return (
    <div className={styles.container} style={{ float: 'right' }}>
      <button
        id="bpm-mode-btn"
        onClick={toggleBPMMode}
        title="Toggle BPM Mode"
        className={`${styles.button} ${isBPMMode ? styles.active : ''}`}
      >
        <i className="fa-solid fa-music"></i>
      </button>

      <button
        id="metronome-btn"
        onClick={toggleMetronome}
        title="Toggle Metronome"
        className={`${styles.button} ${isMetronomeOn ? styles.active : ''}`}
      >
        <i className="fa-solid fa-drum"></i>
      </button>

      <div className={styles.bpmInputContainer}>
        <label htmlFor="bpm-input">BPM</label>
        <input
          type="number"
          id="bpm-input"
          className={styles.bpmInput}
          min="40"
          max="240"
          value={bpm}
          onChange={handleBPMChange}
        />
      </div>
    </div>
  );
}

export default TempoControls;
