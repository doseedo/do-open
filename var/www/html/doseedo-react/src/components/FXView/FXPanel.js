import React, { useEffect, useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import pluginFX from '../../services/pluginFX';
import PluginSlot from './PluginSlot';
import styles from './FXPanel.module.css';

/**
 * FXPanel - Dynamic FX Chain with 8 Plugin Slots
 * Replaces the old static Tuna-based FX panel
 */
const FXPanel = () => {
  const { state, dispatch } = useApp();
  const [isInitialized, setIsInitialized] = useState(false);
  const [showPresetModal, setShowPresetModal] = useState(false);
  const [presetName, setPresetName] = useState('');

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

  // Calculate average send value across all tracks for display
  const averageSend = allTracks.length > 0
    ? allTracks.reduce((sum, track) => sum + (track.fx?.reverb || 0), 0) / allTracks.length
    : 0;

  // Handler for master send - applies to all tracks
  const handleMasterSendChange = (value) => {
    const sendValue = parseFloat(value);
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

  // Sync FX slots from state to pluginFX service
  const syncFXSlots = useCallback(async () => {
    if (!pluginFX.initialized) return;

    for (const slot of state.fxSlots) {
      const currentSlotInfo = pluginFX.getSlotInfo(slot.id);

      // Check if slot plugin needs to be changed
      if (currentSlotInfo?.pluginName !== slot.pluginName) {
        await pluginFX.setSlot(slot.id, slot.pluginName, slot.enabled);
      }

      // Sync enabled state
      if (currentSlotInfo?.enabled !== slot.enabled) {
        pluginFX.setSlotEnabled(slot.id, slot.enabled);
      }

      // Sync parameters
      if (slot.params && Object.keys(slot.params).length > 0) {
        for (const [paramName, value] of Object.entries(slot.params)) {
          pluginFX.setParameter(slot.id, paramName, value);
        }
      }
    }
  }, [state.fxSlots]);

  // Sync when fxSlots change
  useEffect(() => {
    if (isInitialized) {
      syncFXSlots();
    }
  }, [state.fxSlots, isInitialized, syncFXSlots]);

  // Handle saving preset
  const handleSavePreset = () => {
    if (!presetName.trim()) return;

    dispatch({
      type: 'SAVE_FX_CHAIN_PRESET',
      payload: { name: presetName.trim() }
    });

    setPresetName('');
    setShowPresetModal(false);
  };

  // Handle loading preset
  const handleLoadPreset = async (presetId) => {
    dispatch({
      type: 'LOAD_FX_CHAIN_PRESET',
      payload: { presetId }
    });

    // Re-sync after loading
    setTimeout(() => syncFXSlots(), 100);
  };

  // Handle reset to default
  const handleResetChain = async () => {
    dispatch({ type: 'RESET_FX_CHAIN' });
    setTimeout(() => syncFXSlots(), 100);
  };

  // Mark as initialized when pluginFX is ready
  useEffect(() => {
    if (pluginFX.initialized && !isInitialized) {
      setIsInitialized(true);
      syncFXSlots();
    }
  }, [pluginFX.initialized, isInitialized, syncFXSlots]);

  return (
    <div className={styles.fxPanelContainer}>
      <div className={styles.fxHeader}>
        <i className="fa-solid fa-sliders"></i>
        <h3>FX Chain</h3>
        <div className={styles.headerActions}>
          <button
            className={styles.headerButton}
            onClick={() => setShowPresetModal(true)}
            title="Save Preset"
          >
            <i className="fa-solid fa-save"></i>
          </button>
          <button
            className={styles.headerButton}
            onClick={handleResetChain}
            title="Reset to Default"
          >
            <i className="fa-solid fa-undo"></i>
          </button>
        </div>
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
              <h5>FX Return</h5>
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
                value={state.masterFX?.reverbMix || 0.5}
                onChange={(e) => dispatch({ type: 'SET_REVERB_MIX', payload: parseFloat(e.target.value) })}
              />
              <span className={styles.fxValue}>
                {Math.round((state.masterFX?.reverbMix || 0.5) * 100)}%
              </span>
            </label>
          </div>
        </div>
      </div>

      {/* 8-Slot FX Grid */}
      <div className={styles.fxGrid}>
        {state.fxSlots?.map((slot, index) => (
          <PluginSlot key={slot.id} slotId={slot.id} />
        ))}
      </div>

      {/* Presets Section */}
      {state.fxChainPresets?.length > 0 && (
        <div className={styles.presetsSection}>
          <h4>Saved Presets</h4>
          <div className={styles.presetList}>
            {state.fxChainPresets.map(preset => (
              <div key={preset.id} className={styles.presetItem}>
                <span className={styles.presetName}>{preset.name}</span>
                <div className={styles.presetActions}>
                  <button
                    onClick={() => handleLoadPreset(preset.id)}
                    title="Load Preset"
                  >
                    <i className="fa-solid fa-upload"></i>
                  </button>
                  <button
                    onClick={() => dispatch({
                      type: 'DELETE_FX_CHAIN_PRESET',
                      payload: { presetId: preset.id }
                    })}
                    title="Delete Preset"
                  >
                    <i className="fa-solid fa-trash"></i>
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Save Preset Modal */}
      {showPresetModal && (
        <div className={styles.modalOverlay} onClick={() => setShowPresetModal(false)}>
          <div className={styles.modal} onClick={e => e.stopPropagation()}>
            <h4>Save FX Chain Preset</h4>
            <input
              type="text"
              placeholder="Preset name..."
              value={presetName}
              onChange={(e) => setPresetName(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSavePreset()}
              autoFocus
            />
            <div className={styles.modalActions}>
              <button onClick={() => setShowPresetModal(false)}>Cancel</button>
              <button onClick={handleSavePreset} disabled={!presetName.trim()}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Status indicator */}
      <div className={styles.statusBar}>
        <span className={`${styles.statusIndicator} ${isInitialized ? styles.active : ''}`}></span>
        <span className={styles.statusText}>
          {isInitialized ? 'FX Chain Active' : 'Initializing...'}
        </span>
        <span className={styles.statusInfo}>
          {state.fxSlots?.filter(s => s.pluginName && s.enabled).length || 0} effects active
        </span>
      </div>
    </div>
  );
};

export default FXPanel;
