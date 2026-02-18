import React, { useState } from 'react';
import { useApp } from '../../../context/AppContext';
import styles from './TrackInfoSidebar.module.css';

/**
 * BusInfoSidebar - Shows information about a selected bus
 * Displays number of tracks, bus controls, and download all button
 */
const BusInfoSidebar = () => {
  const { state, dispatch } = useApp();
  const [isDownloading, setIsDownloading] = useState(false);

  const selectedBus = state.selectedBus;

  // If no bus is selected, don't render
  if (!selectedBus) return null;

  const handleClose = () => {
    dispatch({ type: 'SELECT_BUS', payload: { busId: null } });
  };

  const handleDownloadAll = async () => {
    setIsDownloading(true);
    try {
      // TODO: Implement zip download of all tracks
      console.log('📦 Downloading all tracks from bus:', selectedBus.name);

      // For now, just download each track individually
      for (const track of selectedBus.tracks) {
        if (track.audioUrl) {
          const link = document.createElement('a');
          link.href = track.audioUrl;
          link.download = track.name || `track_${track.id}.wav`;
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);

          // Small delay between downloads
          await new Promise(resolve => setTimeout(resolve, 500));
        }
      }

      console.log('✅ All tracks downloaded');
    } catch (error) {
      console.error('❌ Error downloading tracks:', error);
      alert('Failed to download tracks');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <div className={styles.sidebar}>
      <div className={styles.header}>
        <h3 className={styles.title}>Bus Info</h3>
        <button className={styles.closeButton} onClick={handleClose}>
          ✕
        </button>
      </div>

      <div className={styles.content}>
        {/* Bus Name */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Bus Name</h4>
          <div className={styles.busName}>{selectedBus.name}</div>
        </div>

        {/* Track Count */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Tracks</h4>
          <div className={styles.info}>
            <span className={styles.label}>Total Tracks:</span>
            <span className={styles.value}>{selectedBus.tracks.length}</span>
          </div>
        </div>

        {/* Track List */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Track List</h4>
          <div className={styles.trackList}>
            {selectedBus.tracks.map((track, index) => (
              <div key={track.id} className={styles.trackItem}>
                <span className={styles.trackNumber}>#{index + 1}</span>
                <span className={styles.trackName}>
                  {track.name || track.audioUrl?.split('/').pop() || 'Untitled'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Bus Controls */}
        <div className={styles.section}>
          <h4 className={styles.sectionTitle}>Bus Controls</h4>

          <div className={styles.controlRow}>
            <span className={styles.label}>Gain:</span>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={selectedBus.gain}
              onChange={(e) => dispatch({
                type: 'UPDATE_BUS_GAIN',
                payload: { busId: selectedBus.id, gain: parseFloat(e.target.value) }
              })}
              className={styles.slider}
            />
            <span className={styles.value}>{Math.round(selectedBus.gain * 100)}%</span>
          </div>

          <div className={styles.controlRow}>
            <button
              className={`${styles.button} ${selectedBus.mute ? styles.active : ''}`}
              onClick={() => dispatch({
                type: 'TOGGLE_BUS_MUTE',
                payload: { busId: selectedBus.id }
              })}
            >
              {selectedBus.mute ? 'Unmute' : 'Mute'}
            </button>
            <button
              className={`${styles.button} ${selectedBus.solo ? styles.active : ''}`}
              onClick={() => dispatch({
                type: 'TOGGLE_BUS_SOLO',
                payload: { busId: selectedBus.id }
              })}
            >
              {selectedBus.solo ? 'Unsolo' : 'Solo'}
            </button>
          </div>
        </div>

        {/* Download All Button */}
        <div className={styles.section}>
          <button
            className={`${styles.downloadAllButton} ${isDownloading ? styles.downloading : ''}`}
            onClick={handleDownloadAll}
            disabled={isDownloading || selectedBus.tracks.length === 0}
          >
            {isDownloading ? '📦 Downloading...' : '📦 Download All Tracks'}
          </button>
        </div>
      </div>
    </div>
  );
};

export default BusInfoSidebar;
