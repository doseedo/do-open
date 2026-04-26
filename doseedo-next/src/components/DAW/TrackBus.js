import React, { useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useThemeColor } from '../../hooks/useThemeColor';
import { enqueueVolumeEdit } from '../../services/sessionEditsAPI';

/**
 * TrackBus Component - Represents a track bus control panel (VO, Music, or SFX)
 * Shows track labels that align with the actual tracks in the Downloads component
 */
function TrackBus({ bus, icon }) {
  const { state, dispatch } = useApp();

  // Get theme color for selected track background
  const selectedBgColor = useThemeColor('--color-primary-blue-20', 'rgba(102, 126, 234, 0.2)');

  // Get tracks and state from bus object
  const tracks = bus.tracks || [];
  const gain = bus.gain || 1;
  const isMuted = bus.mute || false;
  const isSolo = bus.solo || false;
  const isExpanded = bus.expanded ?? true;

  const handleGainChange = (e) => {
    const value = parseFloat(e.target.value);
    dispatch({
      type: 'UPDATE_BUS_GAIN',
      payload: { busId: bus.id, gain: value }
    });
  };

  const handleMuteToggle = () => {
    dispatch({
      type: 'TOGGLE_BUS_MUTE',
      payload: { busId: bus.id }
    });
  };

  const handleSoloToggle = () => {
    dispatch({
      type: 'TOGGLE_BUS_SOLO',
      payload: { busId: bus.id }
    });
  };

  const handleExpandToggle = () => {
    dispatch({
      type: 'TOGGLE_BUS_EXPANDED',
      payload: { busId: bus.id }
    });
  };

  // Calculate tracklist height based on number of tracks
  const tracklistHeight = useMemo(() => {
    return isExpanded ? tracks.length * 60 : 0; // 60px per track
  }, [isExpanded, tracks.length]);

  return (
    <div className="trackselect" id={bus.id} data-mode={bus.type}>
      <div>
        <p data-target={bus.id}>
          <span className="bus-name">{bus.name}</span>
          <i className={`fa-solid ${icon}`}></i>
          <i
            className={`fa-solid fa-caret-down caret-icon ${!isExpanded ? 'collapsed' : ''}`}
            onClick={handleExpandToggle}
            style={{ cursor: 'pointer' }}
          ></i>

          <input
            type="range"
            id={`${bus.id}-gain`}
            min="0"
            max="1"
            step="0.01"
            value={gain}
            onChange={handleGainChange}
          />

          <button
            className={`bus-mute ${isMuted ? 'active' : ''}`}
            data-bus={bus.id}
            onClick={handleMuteToggle}
          >
            M
          </button>
          <button
            className={`bus-solo ${isSolo ? 'active' : ''}`}
            data-bus={bus.id}
            onClick={handleSoloToggle}
          >
            S
          </button>
        </p>

        {/* Track labels section - aligns with tracks in Downloads */}
        <div
          className="tracklist"
          style={{
            height: `${tracklistHeight}px`,
            overflow: 'hidden',
            transition: 'height 0.3s ease'
          }}
        >
          <div className="track-label-container">
            {tracks.map((track, index) => (
              <div
                key={track.id}
                className="track-label"
                style={{
                  height: '60px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  padding: '0 10px',
                  borderBottom: '1px solid #333',
                  background: state.selectedTrack?.id === track.id ? selectedBgColor : 'transparent',
                  color: '#aaa',
                  fontSize: '12px'
                }}
              >
                <span
                  style={{ marginRight: '4px', cursor: 'pointer' }}
                  onClick={() => dispatch({ type: 'SELECT_TRACK', payload: { trackId: track.id, busId: bus.id } })}
                >
                  #{index + 1}
                </span>
                <span
                  style={{
                    flex: 1,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                    cursor: 'pointer',
                    minWidth: '60px'
                  }}
                  onClick={() => dispatch({ type: 'SELECT_TRACK', payload: { trackId: track.id, busId: bus.id } })}
                >
                  {track.name || track.audioUrl?.split('/').pop() || 'Untitled'}
                </span>

                {/* Track Volume Slider */}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={track.gain || 1}
                  onChange={(e) => {
                    e.stopPropagation();
                    const newGain = parseFloat(e.target.value);
                    dispatch({
                      type: 'UPDATE_TRACK',
                      payload: {
                        busId: bus.id,
                        trackId: track.id,
                        updates: { gain: newGain }
                      }
                    });
                    // Mirror the edit to the desktop's edit-log so doo_hook
                    // moves the matching Logic mixer fader. Logic strips are
                    // 1-based (track 0 = Stereo Out), so channel = idx + 1.
                    if (state.activeSessionId && typeof track.logicTrackIndex === 'number') {
                      enqueueVolumeEdit(state.activeSessionId, track.logicTrackIndex + 1, newGain);
                    }
                  }}
                  style={{
                    width: '60px',
                    cursor: 'pointer'
                  }}
                  title={`Volume: ${Math.round((track.gain || 1) * 100)}%`}
                />

                {/* Track Mute Button */}
                <button
                  className={`bus-mute ${track.mute ? 'active' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    dispatch({
                      type: 'UPDATE_TRACK',
                      payload: {
                        busId: bus.id,
                        trackId: track.id,
                        updates: { mute: !track.mute }
                      }
                    });
                  }}
                  style={{
                    minWidth: '24px',
                    height: '24px',
                    fontSize: '10px',
                    padding: '0'
                  }}
                  title="Mute track"
                >
                  M
                </button>

                {/* Track Solo Button */}
                <button
                  className={`bus-solo ${track.solo ? 'active' : ''}`}
                  onClick={(e) => {
                    e.stopPropagation();
                    dispatch({
                      type: 'UPDATE_TRACK',
                      payload: {
                        busId: bus.id,
                        trackId: track.id,
                        updates: { solo: !track.solo }
                      }
                    });
                  }}
                  style={{
                    minWidth: '24px',
                    height: '24px',
                    fontSize: '10px',
                    padding: '0'
                  }}
                  title="Solo track"
                >
                  S
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

export default TrackBus;
