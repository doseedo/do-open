import React from 'react';
import { useApp } from '../../context/AppContext';
import useAutoRepaintMeter from '../../hooks/useAutoRepaintMeter';
import { entryAtTime } from '../../services/tempoMap';
import styles from './TempoControls.module.css';

/**
 * TempoControls Component - master tempo + meter for the studio.
 *
 * Layout (left → right):
 *   [metronome] [BPM input] [Meter select] [repaint spinner]
 *
 * When state.tempoMap is populated (by /api/analyze-rhythm on upload),
 * the BPM and meter fields DISPLAY the LOCAL value at the current
 * playhead position — i.e., tempo/meter follow the song as it plays
 * through an in-song change. User edits still dispatch the usual
 * UPDATE_BPM / SET_METER actions (which are interpreted as overrides
 * for the current bar and forward until the next map entry).
 */
function TempoControls() {
  const { state, dispatch } = useApp();

  // Local-at-playhead lookup. When no tempoMap, fall back to the project
  // globals (this is the legacy path — constant tempo/meter everywhere).
  const activeEntry = state.tempoMap
    ? entryAtTime(state.tempoMap, state.playheadPosition || 0)
    : null;
  const bpm = activeEntry ? Math.round(activeEntry.bpm) : state.bpm;
  const beatsPerBar = (activeEntry && activeEntry.meter?.[0]) || state.beatsPerBar || 4;
  const meterDen = (activeEntry && activeEntry.meter?.[1]) || state.meterDenominator || 4;
  const meterStr = `${beatsPerBar}/${meterDen}`;
  const isMetronomeOn = state.isMetronomeOn;

  // Shared auto-repaint hook — same one /studio-dev uses.
  const { applying } = useAutoRepaintMeter();

  const handleBPMChange = (e) => {
    dispatch({ type: 'UPDATE_BPM', payload: parseInt(e.target.value, 10) });
  };

  const handleMeterChange = (e) => {
    dispatch({ type: 'SET_METER', payload: e.target.value });
  };

  const toggleMetronome = () => dispatch({ type: 'TOGGLE_METRONOME' });

  return (
    <div className={styles.container} style={{ float: 'right', display: 'flex', alignItems: 'center', gap: 6 }}>
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

      <div className={styles.bpmInputContainer} title="Time signature">
        <label htmlFor="meter-input">Meter</label>
        <select
          id="meter-input"
          className={styles.bpmInput}
          value={meterStr}
          onChange={handleMeterChange}
          style={{ minWidth: 60 }}
        >
          <option value="3/4">3/4</option>
          <option value="4/4">4/4</option>
          <option value="5/4">5/4</option>
          <option value="6/8">6/8</option>
          <option value="7/8">7/8</option>
        </select>
      </div>

      {applying && (
        <span
          className={styles.button}
          title="Repainting tracks via stemphonic stage2d-130k"
          style={{ pointerEvents: 'none' }}
        >
          <i className="fa-solid fa-wand-magic-sparkles fa-spin" style={{ color: '#8B7FF0' }}></i>
        </span>
      )}
    </div>
  );
}

export default TempoControls;
