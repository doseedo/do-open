import React from 'react';
import styles from './ModeSelector.module.css';

/**
 * ModeSelector - Vertical column for switching between Video, MIDI, Audio, Image, and FX modes
 */
const ModeSelector = ({ selectedMode, onModeChange }) => {
  const modes = [
    { id: 'video', label: 'Video', icon: 'fa-video', isFX: false },
    { id: 'midi', label: 'MIDI', icon: 'fa-music', isFX: false },
    { id: 'audio', label: 'Audio', icon: 'fa-wave-square', isFX: false },
    { id: 'fx', label: 'FX', icon: 'fa-sliders', isPlugin: true }
  ];

  return (
    <div className={styles.modeSelectorContainer}>
      {modes.map(mode => (
        <button
          key={mode.id}
          className={`${mode.isPlugin ? styles.pluginToggle : styles.modeButton} ${selectedMode === mode.id ? styles.active : ''}`}
          onClick={() => onModeChange(mode.id)}
          title={`${mode.label} Mode`}
        >
          <i className={`fa-solid ${mode.icon} ${mode.isPlugin ? styles.pluginIcon : styles.modeIcon}`}></i>
          <span className={mode.isPlugin ? styles.pluginLabel : styles.modeLabel}>{mode.label}</span>
        </button>
      ))}
    </div>
  );
};

export default ModeSelector;
