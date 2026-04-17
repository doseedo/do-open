import React, { useRef, useEffect, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { useWaveSurfer } from '../../hooks/useWaveSurfer';
import { formatDuration } from '../../utils/audioUtils';
import './AudioWorkspace.css';

/**
 * AudioWorkspace Component
 * Handles the main audio waveform display and track management
 * Uses WaveSurfer.js for audio visualization
 */
function AudioWorkspace() {
  const { state, dispatch } = useApp();
  const waveformRef = useRef(null);
  const [volume, setVolume] = useState(0.5);
  const [zoom, setZoom] = useState(1);

  // Initialize WaveSurfer with custom hook
  const {
    wavesurfer,
    isReady,
    isPlaying,
    duration,
    currentTime,
    playPause,
    load,
    seekTo
  } = useWaveSurfer(waveformRef, {
    waveColor: '#667eea',
    progressColor: '#764ba2',
    cursorColor: '#ffffff',
    barWidth: 2,
    barRadius: 3,
    height: 150,
    barGap: 2
  });

  // Update global playing state when WaveSurfer state changes
  useEffect(() => {
    if (isPlaying !== state.isPlaying) {
      dispatch({ type: 'SET_PLAYING', payload: isPlaying });
    }
  }, [isPlaying]);

  // Handle volume changes
  useEffect(() => {
    if (wavesurfer) {
      wavesurfer.setVolume(volume);
    }
  }, [volume, wavesurfer]);

  // Handle zoom changes
  useEffect(() => {
    if (wavesurfer) {
      wavesurfer.zoom(zoom);
    }
  }, [zoom, wavesurfer]);

  // Load current track when it changes
  useEffect(() => {
    if (state.currentTrack && wavesurfer) {
      load(state.currentTrack.url);
    }
  }, [state.currentTrack, wavesurfer]);

  const handlePlayPause = () => {
    playPause();
  };

  const handleStop = () => {
    if (wavesurfer) {
      wavesurfer.stop();
      dispatch({ type: 'SET_PLAYING', payload: false });
    }
  };

  const handleRewind = () => {
    if (wavesurfer) {
      const newTime = Math.max(0, currentTime - 5);
      seekTo(newTime / duration);
    }
  };

  const handleForward = () => {
    if (wavesurfer) {
      const newTime = Math.min(duration, currentTime + 5);
      seekTo(newTime / duration);
    }
  };

  const handleVolumeChange = (e) => {
    setVolume(parseFloat(e.target.value));
  };

  const handleZoomChange = (e) => {
    setZoom(parseInt(e.target.value));
  };

  const handleTrackClick = (track) => {
    dispatch({ type: 'SET_CURRENT_TRACK', payload: track });
  };

  const handleDeleteTrack = (trackId) => {
    if (window.confirm('Delete this track?')) {
      dispatch({ type: 'REMOVE_AUDIO_TRACK', payload: trackId });

      // If deleting current track, stop playback
      if (state.currentTrack && state.currentTrack.id === trackId) {
        handleStop();
        dispatch({ type: 'SET_CURRENT_TRACK', payload: null });
      }
    }
  };

  return (
    <div className="audio-workspace">
      <div className="workspace-header">
        <h3>Audio Workspace</h3>
        <div className="workspace-info">
          {isReady && (
            <>
              <span className="time-display">
                {formatDuration(currentTime)} / {formatDuration(duration)}
              </span>
              <span className="track-name">
                {state.currentTrack ? state.currentTrack.name : 'No track loaded'}
              </span>
            </>
          )}
        </div>
      </div>

      {/* Waveform Display */}
      <div className="waveform-container" ref={waveformRef}>
        {!isReady && state.currentTrack && (
          <div className="waveform-loading">
            <i className="fa-solid fa-spinner fa-spin"></i>
            <p>Loading waveform...</p>
          </div>
        )}
      </div>

      {/* Transport Controls */}
      <div className="workspace-controls">
        <div className="transport-controls">
          <button onClick={handleRewind} disabled={!isReady} title="Rewind 5s">
            <i className="fa-solid fa-backward"></i>
          </button>
          <button onClick={handlePlayPause} disabled={!isReady} className="play-button">
            <i className={`fa-solid ${isPlaying ? 'fa-pause' : 'fa-play'}`}></i>
          </button>
          <button onClick={handleStop} disabled={!isReady} title="Stop">
            <i className="fa-solid fa-stop"></i>
          </button>
          <button onClick={handleForward} disabled={!isReady} title="Forward 5s">
            <i className="fa-solid fa-forward"></i>
          </button>
        </div>

        <div className="volume-control">
          <i className="fa-solid fa-volume-up"></i>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={volume}
            onChange={handleVolumeChange}
            className="volume-slider"
          />
          <span className="volume-value">{Math.round(volume * 100)}%</span>
        </div>

        <div className="zoom-control">
          <i className="fa-solid fa-magnifying-glass"></i>
          <input
            type="range"
            min="1"
            max="100"
            step="1"
            value={zoom}
            onChange={handleZoomChange}
            className="zoom-slider"
          />
          <span className="zoom-value">{zoom}x</span>
        </div>
      </div>

      {/* Track List */}
      <div className="tracks-container">
        <h4>Tracks ({state.audioTracks.length})</h4>
        {state.audioTracks.length === 0 ? (
          <div className="empty-state">
            <i className="fa-solid fa-music" style={{ fontSize: '48px', opacity: 0.3, marginBottom: '10px' }}></i>
            <p>No audio tracks loaded</p>
            <p style={{ fontSize: '14px', opacity: 0.7 }}>Upload a file or generate audio to get started</p>
          </div>
        ) : (
          <div className="track-list">
            {state.audioTracks.map(track => (
              <div
                key={track.id}
                className={`audio-track ${state.currentTrack && state.currentTrack.id === track.id ? 'active' : ''}`}
                onClick={() => handleTrackClick(track)}
              >
                <div className="track-info">
                  <i className="fa-solid fa-music"></i>
                  <span className="track-name">{track.name}</span>
                  {track.duration && (
                    <span className="track-duration">{formatDuration(track.duration)}</span>
                  )}
                </div>
                <button
                  className="delete-track-btn"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDeleteTrack(track.id);
                  }}
                  title="Delete track"
                >
                  <i className="fa-solid fa-trash"></i>
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

export default AudioWorkspace;
