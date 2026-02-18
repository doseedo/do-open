import React, { useEffect } from 'react';
import { useApp } from '../../context/AppContext';
import tunaFX from '../../services/tunaFX';
import styles from './FXPanel.module.css';

/**
 * FXPanel - Global FX controls for reverb bus and master effects
 * Displayed in FX Mode
 */
const FXPanel = () => {
  const { state, dispatch } = useApp();

  // Get the reverb bus (typically the last bus)
  const reverbBus = state.buses?.find(bus => bus.name?.toLowerCase().includes('reverb')) || state.buses?.[state.buses.length - 1];

  // Get all tracks from all buses
  const allTracks = state.buses?.flatMap(bus =>
    bus.tracks?.map(track => ({
      ...track,
      busId: bus.id,
      busName: bus.name
    })) || []
  ) || [];

  // Handler for master send - applies to all tracks
  const handleMasterSendChange = (value) => {
    const sendValue = parseFloat(value);
    // Update reverb send for all tracks
    allTracks.forEach(track => {
      dispatch({
        type: 'UPDATE_TRACK_FX',
        payload: {
          trackId: track.id,
          fx: { reverb: sendValue }
        }
      });
    });
  };

  const handleReverbBusVolumeChange = (value) => {
    if (reverbBus) {
      dispatch({
        type: 'UPDATE_BUS_VOLUME',
        payload: {
          busId: reverbBus.id,
          volume: parseFloat(value)
        }
      });
    }
  };

  // Calculate average send value across all tracks for display
  const averageSend = allTracks.length > 0
    ? allTracks.reduce((sum, track) => sum + (track.fx?.reverb || 0), 0) / allTracks.length
    : 0;

  // Sync state changes to Tuna FX
  useEffect(() => {
    if (!tunaFX.initialized) return;

    tunaFX.updateReverb({
      decay: state.reverbDecay,
      preDelay: state.reverbPreDelay,
      roomSize: state.reverbRoomSize,
      damping: state.reverbDamping,
      mix: state.masterFX.reverbMix
    });
  }, [state.reverbDecay, state.reverbPreDelay, state.reverbRoomSize, state.reverbDamping, state.masterFX.reverbMix]);

  useEffect(() => {
    if (!tunaFX.initialized) return;
    tunaFX.updateDelay({
      time: state.delayTime,
      feedback: state.delayFeedback,
      cutoff: state.delayCutoff
    });
  }, [state.delayTime, state.delayFeedback, state.delayCutoff]);

  useEffect(() => {
    if (!tunaFX.initialized) return;
    tunaFX.updateChorus({
      rate: state.chorusRate,
      depth: state.chorusDepth,
      feedback: state.chorusFeedback
    });
  }, [state.chorusRate, state.chorusDepth, state.chorusFeedback]);

  useEffect(() => {
    if (!tunaFX.initialized) return;
    tunaFX.updateCompressor({
      threshold: state.compressorThreshold,
      ratio: state.compressorRatio,
      attack: state.compressorAttack
    });
  }, [state.compressorThreshold, state.compressorRatio, state.compressorAttack]);

  useEffect(() => {
    if (!tunaFX.initialized) return;
    tunaFX.updateFilter({
      frequency: state.filterFrequency,
      resonance: state.filterResonance,
      gain: state.filterGain
    });
  }, [state.filterFrequency, state.filterResonance, state.filterGain]);

  useEffect(() => {
    if (!tunaFX.initialized) return;
    tunaFX.updatePhaser({
      rate: state.phaserRate,
      depth: state.phaserDepth,
      feedback: state.phaserFeedback
    });
  }, [state.phaserRate, state.phaserDepth, state.phaserFeedback]);

  return (
    <div className={styles.fxPanelContainer}>
      <div className={styles.fxHeader}>
        <i className="fa-solid fa-sliders"></i>
        <h3>FX Controls</h3>
      </div>

      {/* Global Controls - Master Send, Bus Volume, Dry/Wet Mix */}
      <div className={styles.globalControls}>
        {/* Master Send */}
        <div className={styles.globalControl}>
          <div className={styles.globalControlHeader}>
            <i className="fa-solid fa-arrow-right-to-bracket"></i>
            <h5>Master Send</h5>
          </div>
          <div className={styles.globalControlInput}>
            <label>
              <span>Amount:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={averageSend}
                onChange={(e) => handleMasterSendChange(e.target.value)}
              />
              <span className={styles.fxValue}>
                {Math.round(averageSend * 100)}%
              </span>
            </label>
          </div>
        </div>

        {/* Reverb Bus Volume */}
        {reverbBus && (
          <div className={styles.globalControl}>
            <div className={styles.globalControlHeader}>
              <i className="fa-solid fa-volume-high"></i>
              <h5>Bus Volume</h5>
            </div>
            <div className={styles.globalControlInput}>
              <label>
                <span>Level:</span>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={reverbBus.volume || 0}
                  onChange={(e) => handleReverbBusVolumeChange(e.target.value)}
                />
                <span className={styles.fxValue}>
                  {Math.round((reverbBus.volume || 0) * 100)}%
                </span>
              </label>
            </div>
          </div>
        )}

        {/* Dry/Wet Mix */}
        <div className={styles.globalControl}>
          <div className={styles.globalControlHeader}>
            <i className="fa-solid fa-droplet"></i>
            <h5>Dry/Wet Mix</h5>
          </div>
          <div className={styles.globalControlInput}>
            <label>
              <span>Mix:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state.reverbMix || 0.5}
                onChange={(e) => dispatch({ type: 'SET_REVERB_MIX', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>
                {Math.round((state.reverbMix || 0.5) * 100)}%
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* Plugin Grid - 4x2 */}
      <div className={styles.fxGrid}>
        {/* Plugin Slot 1: Reverb */}
        <div className={`${styles.fxSection} ${styles.compact}`}>
          <div className={styles.sectionHeader}>
            <i className="fa-solid fa-water"></i>
            <h4>Reverb</h4>
          </div>

          <div className={styles.fxControl}>
            <label>
              <span>Decay:</span>
              <input
                type="range"
                min="0.1"
                max="10"
                step="0.1"
                value={state.reverbDecay || 2.5}
                onChange={(e) => dispatch({ type: 'SET_REVERB_DECAY', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>
                {(state.reverbDecay || 2.5).toFixed(1)}s
              </span>
            </label>
          </div>

          <div className={styles.fxControl}>
            <label>
              <span>Pre-Delay:</span>
              <input
                type="range"
                min="0"
                max="100"
                step="1"
                value={state.reverbPreDelay || 0}
                onChange={(e) => dispatch({ type: 'SET_REVERB_PREDELAY', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>
                {Math.round(state.reverbPreDelay || 0)}ms
              </span>
            </label>
          </div>

          <div className={styles.fxControl}>
            <label>
              <span>Room Size:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state.reverbRoomSize || 0.5}
                onChange={(e) => dispatch({ type: 'SET_REVERB_ROOM_SIZE', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>
                {Math.round((state.reverbRoomSize || 0.5) * 100)}%
              </span>
            </label>
          </div>

          <div className={styles.fxControl}>
            <label>
              <span>Damping:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state.reverbDamping || 0.5}
                onChange={(e) => dispatch({ type: 'SET_REVERB_DAMPING', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>
                {Math.round((state.reverbDamping || 0.5) * 100)}%
              </span>
            </label>
          </div>
        </div>

        {/* Plugin Slot 2: Delay */}
        <div className={`${styles.fxSection} ${styles.compact}`}>
          <div className={styles.sectionHeader}>
            <i className="fa-solid fa-clock"></i>
            <h4>Delay</h4>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Time:</span>
              <input
                type="range"
                min="20"
                max="1000"
                step="10"
                value={state.delayTime || 100}
                onChange={(e) => dispatch({ type: 'SET_DELAY_TIME', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round(state.delayTime || 100)}ms</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Feedback:</span>
              <input
                type="range"
                min="0"
                max="0.9"
                step="0.01"
                value={state.delayFeedback || 0.45}
                onChange={(e) => dispatch({ type: 'SET_DELAY_FEEDBACK', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round((state.delayFeedback || 0.45) * 100)}%</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Cutoff:</span>
              <input
                type="range"
                min="20"
                max="22050"
                step="10"
                value={state.delayCutoff || 20000}
                onChange={(e) => dispatch({ type: 'SET_DELAY_CUTOFF', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{((state.delayCutoff || 20000) / 1000).toFixed(1)}kHz</span>
            </label>
          </div>
        </div>

        {/* Plugin Slot 3: Chorus */}
        <div className={`${styles.fxSection} ${styles.compact}`}>
          <div className={styles.sectionHeader}>
            <i className="fa-solid fa-wave-square"></i>
            <h4>Chorus</h4>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Rate:</span>
              <input
                type="range"
                min="0"
                max="8"
                step="0.1"
                value={state.chorusRate || 1.5}
                onChange={(e) => dispatch({ type: 'SET_CHORUS_RATE', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{(state.chorusRate || 1.5).toFixed(1)}Hz</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Depth:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state.chorusDepth || 0.7}
                onChange={(e) => dispatch({ type: 'SET_CHORUS_DEPTH', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round((state.chorusDepth || 0.7) * 100)}%</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Feedback:</span>
              <input
                type="range"
                min="0"
                max="0.95"
                step="0.01"
                value={state.chorusFeedback || 0.4}
                onChange={(e) => dispatch({ type: 'SET_CHORUS_FEEDBACK', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round((state.chorusFeedback || 0.4) * 100)}%</span>
            </label>
          </div>
        </div>

        {/* Plugin Slot 4: Compressor */}
        <div className={`${styles.fxSection} ${styles.compact}`}>
          <div className={styles.sectionHeader}>
            <i className="fa-solid fa-compress"></i>
            <h4>Compressor</h4>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Threshold:</span>
              <input
                type="range"
                min="-100"
                max="0"
                step="1"
                value={state.compressorThreshold || -20}
                onChange={(e) => dispatch({ type: 'SET_COMPRESSOR_THRESHOLD', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round(state.compressorThreshold || -20)}dB</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Ratio:</span>
              <input
                type="range"
                min="1"
                max="20"
                step="1"
                value={state.compressorRatio || 4}
                onChange={(e) => dispatch({ type: 'SET_COMPRESSOR_RATIO', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round(state.compressorRatio || 4)}:1</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Attack:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.001"
                value={state.compressorAttack || 0.003}
                onChange={(e) => dispatch({ type: 'SET_COMPRESSOR_ATTACK', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round((state.compressorAttack || 0.003) * 1000)}ms</span>
            </label>
          </div>
        </div>

        {/* Plugin Slot 5: Filter */}
        <div className={`${styles.fxSection} ${styles.compact}`}>
          <div className={styles.sectionHeader}>
            <i className="fa-solid fa-filter"></i>
            <h4>Filter</h4>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Frequency:</span>
              <input
                type="range"
                min="20"
                max="22050"
                step="1"
                value={state.filterFrequency || 800}
                onChange={(e) => dispatch({ type: 'SET_FILTER_FREQUENCY', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round(state.filterFrequency || 800)}Hz</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Resonance:</span>
              <input
                type="range"
                min="0.001"
                max="1000"
                step="1"
                value={state.filterResonance || 1}
                onChange={(e) => dispatch({ type: 'SET_FILTER_RESONANCE', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round(state.filterResonance || 1)}Q</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Gain:</span>
              <input
                type="range"
                min="-40"
                max="40"
                step="1"
                value={state.filterGain || 0}
                onChange={(e) => dispatch({ type: 'SET_FILTER_GAIN', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round(state.filterGain || 0)}dB</span>
            </label>
          </div>
        </div>

        {/* Plugin Slot 6: Phaser */}
        <div className={`${styles.fxSection} ${styles.compact}`}>
          <div className={styles.sectionHeader}>
            <i className="fa-solid fa-circle-notch"></i>
            <h4>Phaser</h4>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Rate:</span>
              <input
                type="range"
                min="0.01"
                max="8"
                step="0.01"
                value={state.phaserRate || 0.1}
                onChange={(e) => dispatch({ type: 'SET_PHASER_RATE', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{(state.phaserRate || 0.1).toFixed(2)}Hz</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Depth:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state.phaserDepth || 0.6}
                onChange={(e) => dispatch({ type: 'SET_PHASER_DEPTH', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round((state.phaserDepth || 0.6) * 100)}%</span>
            </label>
          </div>
          <div className={styles.fxControl}>
            <label>
              <span>Feedback:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={state.phaserFeedback || 0.7}
                onChange={(e) => dispatch({ type: 'SET_PHASER_FEEDBACK', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>{Math.round((state.phaserFeedback || 0.7) * 100)}%</span>
            </label>
          </div>
        </div>

        {/* Plugin Slots 7-8: Empty */}
        {[...Array(2)].map((_, i) => (
          <div key={i + 5} className={styles.fxSection}>
            <div className={styles.sectionHeader}>
              <i className="fa-solid fa-plus"></i>
              <h4>Empty Slot</h4>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default FXPanel;
