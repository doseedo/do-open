import React, { useCallback } from 'react';
import styles from './TransportControls.module.css';

/**
 * TransportControls Component
 * Reusable play/pause/stop controls with time display
 *
 * @param {boolean} isPlaying - Current playing state
 * @param {number} playheadPosition - Current playhead position in seconds
 * @param {Function} onPlayPause - Callback for play/pause button
 * @param {Function} onStop - Callback for stop button
 */
const TransportControls = React.memo(({
  isPlaying,
  playheadPosition,
  onPlayPause,
  onStop
}) => {
  const formatTime = useCallback((seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  return (
    <div className={styles.transportControls}>
      <button
        className={`${styles.button} ${isPlaying ? styles.active : ''}`}
        onClick={onPlayPause}
        title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
      >
        <i className={`fa-solid ${isPlaying ? 'fa-pause' : 'fa-play'}`}></i>
      </button>
      <button
        className={styles.button}
        onClick={onStop}
        title="Stop"
      >
        <i className="fa-solid fa-stop"></i>
      </button>
      <span className={styles.timeDisplay}>
        {formatTime(playheadPosition || 0)}
      </span>
    </div>
  );
});

TransportControls.displayName = 'TransportControls';

export default TransportControls;
