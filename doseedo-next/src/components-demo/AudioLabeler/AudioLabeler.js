import React, { useState, useRef, useEffect, useCallback } from 'react';
import WaveSurfer from 'wavesurfer.js';
import './AudioLabeler.css';

const GROUPS = [
  'guitar', 'piano', 'bass', 'strings', 'brass',
  'winds', 'voice', 'drums', 'synth', 'keys', 'percussion'
];

const AudioLabeler = () => {
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [selectedGroup, setSelectedGroup] = useState(null);
  const [labels, setLabels] = useState([]);
  const [status, setStatus] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const waveformRef = useRef(null);
  const wavesurferRef = useRef(null);
  const fileInputRef = useRef(null);

  // Initialize WaveSurfer
  useEffect(() => {
    if (waveformRef.current && !wavesurferRef.current) {
      wavesurferRef.current = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'rgba(100, 149, 237, 0.6)',
        progressColor: 'rgba(65, 105, 225, 0.9)',
        cursorColor: '#ff6b6b',
        cursorWidth: 2,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
        height: 128,
        responsive: true,
        normalize: true,
        backend: 'WebAudio',
      });

      wavesurferRef.current.on('ready', () => {
        setDuration(wavesurferRef.current.getDuration());
        setIsLoading(false);
      });

      wavesurferRef.current.on('audioprocess', () => {
        setCurrentTime(wavesurferRef.current.getCurrentTime());
      });

      wavesurferRef.current.on('seek', () => {
        setCurrentTime(wavesurferRef.current.getCurrentTime());
      });

      wavesurferRef.current.on('play', () => setIsPlaying(true));
      wavesurferRef.current.on('pause', () => setIsPlaying(false));
      wavesurferRef.current.on('finish', () => setIsPlaying(false));

      // Click to seek and play
      wavesurferRef.current.on('interaction', () => {
        if (!isPlaying) {
          wavesurferRef.current.play();
        }
      });
    }

    return () => {
      if (wavesurferRef.current) {
        wavesurferRef.current.destroy();
        wavesurferRef.current = null;
      }
    };
  }, []);

  // Load audio file
  const handleFileChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      setIsLoading(true);
      setAudioFile(file);
      setSelectedGroup(null);

      const url = URL.createObjectURL(file);
      setAudioUrl(url);

      if (wavesurferRef.current) {
        wavesurferRef.current.load(url);
      }
    }
  }, []);

  // Load from URL
  const handleUrlLoad = useCallback(() => {
    if (audioUrl && audioUrl.startsWith('http')) {
      setIsLoading(true);
      setAudioFile({ name: audioUrl.split('/').pop() });
      setSelectedGroup(null);

      if (wavesurferRef.current) {
        wavesurferRef.current.load(audioUrl);
      }
    }
  }, [audioUrl]);

  // Play/Pause toggle
  const togglePlayPause = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  }, []);

  // Stop and reset
  const handleStop = useCallback(() => {
    if (wavesurferRef.current) {
      wavesurferRef.current.stop();
      setCurrentTime(0);
    }
  }, []);

  // Label the current audio with a group
  const handleLabel = useCallback(async (group) => {
    if (!audioFile) {
      setStatus('No audio file loaded');
      return;
    }

    setSelectedGroup(group);

    const label = {
      filename: audioFile.name,
      group: group,
      timestamp: new Date().toISOString(),
    };

    // Add to local labels list
    setLabels(prev => [...prev.filter(l => l.filename !== audioFile.name), label]);

    // Save to backend
    try {
      const response = await fetch('/api/labels', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(label),
      });

      if (response.ok) {
        setStatus(`Labeled as "${group}" - saved!`);
      } else {
        setStatus(`Labeled as "${group}" - save failed`);
      }
    } catch (err) {
      setStatus(`Labeled as "${group}" (offline)`);
      console.error('Save error:', err);
    }

    setTimeout(() => setStatus(''), 3000);
  }, [audioFile]);

  // Format time as MM:SS
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.code === 'Space' && e.target.tagName !== 'INPUT') {
        e.preventDefault();
        togglePlayPause();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [togglePlayPause]);

  return (
    <div className="audio-labeler">
      <div className="labeler-header">
        <h1>Audio Labeler</h1>
        <p className="subtitle">Load audio, visualize waveform, click to seek & play, label by group</p>
      </div>

      {/* File Input Section */}
      <div className="file-input-section">
        <div className="file-upload">
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
          >
            Choose Audio File
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          {audioFile && <span className="filename">{audioFile.name}</span>}
        </div>

        <div className="url-input">
          <input
            type="text"
            placeholder="Or enter audio URL..."
            value={audioUrl}
            onChange={(e) => setAudioUrl(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleUrlLoad()}
          />
          <button onClick={handleUrlLoad}>Load URL</button>
        </div>
      </div>

      {/* Waveform Display */}
      <div className="waveform-container">
        {isLoading && <div className="loading-overlay">Loading...</div>}
        <div ref={waveformRef} className="waveform" />
      </div>

      {/* Playback Controls */}
      <div className="playback-controls">
        <button
          className={`control-btn ${isPlaying ? 'playing' : ''}`}
          onClick={togglePlayPause}
          disabled={!audioFile}
        >
          {isPlaying ? '⏸ Pause' : '▶ Play'}
        </button>
        <button
          className="control-btn"
          onClick={handleStop}
          disabled={!audioFile}
        >
          ⏹ Stop
        </button>
        <div className="time-display">
          <span>{formatTime(currentTime)}</span>
          <span> / </span>
          <span>{formatTime(duration)}</span>
        </div>
      </div>

      {/* Label Buttons */}
      <div className="label-section">
        <h3>Label this audio:</h3>
        <div className="label-buttons">
          {GROUPS.map(group => (
            <button
              key={group}
              className={`label-btn ${selectedGroup === group ? 'selected' : ''}`}
              onClick={() => handleLabel(group)}
              disabled={!audioFile}
            >
              {group}
            </button>
          ))}
        </div>
      </div>

      {/* Status Message */}
      {status && <div className="status-message">{status}</div>}

      {/* Recent Labels */}
      {labels.length > 0 && (
        <div className="recent-labels">
          <h3>Recent Labels ({labels.length})</h3>
          <div className="labels-list">
            {labels.slice(-10).reverse().map((label, idx) => (
              <div key={idx} className="label-item">
                <span className="label-filename">{label.filename}</span>
                <span className={`label-group group-${label.group}`}>{label.group}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Instructions */}
      <div className="instructions">
        <h4>How to use:</h4>
        <ul>
          <li>Upload an audio file or enter a URL</li>
          <li>Click anywhere on the waveform to seek and play from that position</li>
          <li>Press Space to play/pause</li>
          <li>Click a group button to label the audio</li>
        </ul>
      </div>
    </div>
  );
};

export default AudioLabeler;
