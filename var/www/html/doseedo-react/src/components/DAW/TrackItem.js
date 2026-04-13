import React, { useCallback, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useWaveform } from '../../hooks/useWaveform';

/**
 * TrackItem Component - Individual track with waveform (MEMOIZED)
 * Uses React patterns: declarative rendering, custom hooks, memoization
 *
 * @param {Object} track - Track data object
 * @param {string} mode - Track mode (VO, Music, SFX)
 * @param {number} index - Track index in the list
 * @param {boolean} isSelected - Whether this track is selected
 */
const TrackItem = React.memo(({ track, mode, index, isSelected }) => {
  const { state, dispatch } = useApp();

  // Track height constant (matches original)
  const TRACK_HEIGHT = 60;

  // Use custom waveform hook (only renders when audioUrl changes)
  // Pass envelopeData from latent_visual so the envelope is preserved
  // until the full decoded audio is ready.
  const { canvasRef } = useWaveform(
    track.audioUrl,
    track.width || 800,
    TRACK_HEIGHT,
    track.isPlaceholder ? '#666' : '#667eea',
    0,  // cropStart
    0,  // cropEnd
    track.metadata?.envelopeData || null,
    25  // envelopeFps
  );

  // Memoize track style to prevent recalculation on every render
  const trackStyle = useMemo(() => ({
    position: 'absolute',
    top: `${index * TRACK_HEIGHT}px`,
    width: '100%',
    height: `${TRACK_HEIGHT}px`,
    zIndex: 10 + index,
    cursor: track.isPlaceholder ? 'wait' : 'pointer',
    opacity: track.isPlaceholder ? 0.6 : 1
  }), [index, track.isPlaceholder]);

  // Memoize event handlers to prevent recreating on every render
  const handleTrackClick = useCallback(() => {
    dispatch({
      type: 'SELECT_TRACK',
      payload: { trackId: track.id, mode }
    });
  }, [dispatch, track.id, mode]);

  const handleTrackDragStart = useCallback((e) => {
    // Plugin mode: cancel HTML5 drag and use dodrag:// URL scheme for JUCE
    if (state.pluginMode && track.audioUrl) {
      e.preventDefault();
      const fullUrl = track.audioUrl.startsWith('http') || track.audioUrl.startsWith('blob:')
        ? track.audioUrl
        : `${window.location.origin}${track.audioUrl}`;
      const encodedUrl = encodeURIComponent(fullUrl);
      window.location.href = 'dodrag://' + encodedUrl;
      return;
    }
    e.dataTransfer.setData('trackId', track.id);
    e.dataTransfer.setData('mode', mode);
  }, [track.id, track.audioUrl, mode, state.pluginMode]);

  return (
    <li
      className={`track-item ${track.isPlaceholder ? 'loading-placeholder' : ''} ${isSelected ? 'selected' : ''}`}
      style={trackStyle}
      data-voice-index={track.voiceIndex}
      data-voice-number={track.voiceNumber}
      data-is-placeholder={track.isPlaceholder || 'false'}
      data-track-id={track.id}
      onClick={handleTrackClick}
      draggable={!track.isPlaceholder}
      onDragStart={handleTrackDragStart}
    >
      {track.isPlaceholder ? (
        <div className="waveform-container loading">
          <div
            style={{
              position: 'absolute',
              top: '50%',
              left: '50%',
              transform: 'translate(-50%, -50%)',
              color: 'rgba(255, 255, 255, 0.8)',
              fontSize: '14px',
              fontWeight: 'bold'
            }}
          >
            <i className="fa-solid fa-spinner fa-spin"></i>
          </div>
        </div>
      ) : (
        <div className="waveform-container">
          <canvas
            ref={canvasRef}
            width={track.width || 800}
            height={TRACK_HEIGHT}
            style={{
              width: '100%',
              height: '100%',
              borderRadius: '4px'
            }}
          />
        </div>
      )}
    </li>
  );
});

TrackItem.displayName = 'TrackItem';

export default TrackItem;
