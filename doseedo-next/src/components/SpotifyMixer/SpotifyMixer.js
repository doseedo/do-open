import React, { useState, useEffect, useRef } from 'react';
import './SpotifyMixer.css';

/**
 * SpotifyMixer Component
 * Real-time stem mixing and manipulation for Spotify tracks
 *
 * The "OpenDAW for Spotify" UI
 */
function SpotifyMixer() {
  const [connected, setConnected] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [trackLoaded, setTrackLoaded] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [currentTrack, setCurrentTrack] = useState(null);

  // Playback info
  const [position, setPosition] = useState(0);
  const [duration, setDuration] = useState(0);
  const [tempo, setTempo] = useState(0);
  const [key, setKey] = useState(0);
  const [timeSignature, setTimeSignature] = useState(4);

  // Stem levels
  const [vocalsLevel, setVocalsLevel] = useState(1.0);
  const [drumsLevel, setDrumsLevel] = useState(1.0);
  const [bassLevel, setBassLevel] = useState(1.0);
  const [otherLevel, setOtherLevel] = useState(1.0);

  // Stem panning
  const [vocalsPan, setVocalsPan] = useState(0);
  const [drumsPan, setDrumsPan] = useState(0);
  const [bassPan, setBassPan] = useState(0);
  const [otherPan, setOtherPan] = useState(0);

  // Tempo control
  const [tempoFactor, setTempoFactor] = useState(1.0);

  const wsRef = useRef(null);
  const API_BASE = 'https://doseedo.com/spotify';
  const WS_URL = 'wss://doseedo.com/spotify/ws';

  // WebSocket connection
  useEffect(() => {
    const connectWebSocket = () => {
      const ws = new WebSocket(WS_URL);

      ws.onopen = () => {
        console.log('✓ Connected to Spotify processor');
        setConnected(true);
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        // Update playback info
        if (data.position !== undefined) setPosition(data.position);
        if (data.duration !== undefined) setDuration(data.duration);
        if (data.tempo !== undefined) setTempo(data.tempo);
        if (data.key !== undefined) setKey(data.key);
        if (data.time_signature !== undefined) setTimeSignature(data.time_signature);

        // Update stem levels from server
        if (data.stem_levels) {
          setVocalsLevel(data.stem_levels.vocals);
          setDrumsLevel(data.stem_levels.drums);
          setBassLevel(data.stem_levels.bass);
          setOtherLevel(data.stem_levels.other);
        }
      };

      ws.onclose = () => {
        console.log('✗ Disconnected from Spotify processor');
        setConnected(false);
        // Reconnect after 2 seconds
        setTimeout(connectWebSocket, 2000);
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };

      wsRef.current = ws;
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  // Search for tracks
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;

    try {
      const response = await fetch(`${API_BASE}/search?q=${encodeURIComponent(searchQuery)}`);
      const data = await response.json();
      setSearchResults(data.tracks);
    } catch (error) {
      console.error('Search error:', error);
    }
  };

  // Load a track
  const handleLoadTrack = async (track) => {
    try {
      const response = await fetch(`${API_BASE}/load`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ track_id: track.id })
      });

      const data = await response.json();

      if (data.status === 'success') {
        setCurrentTrack(track);
        setTrackLoaded(true);
        setDuration(data.duration);
        setTempo(data.tempo);
        setKey(data.key);
        setTimeSignature(data.time_signature);
        console.log('✓ Track loaded:', track.name);
      }
    } catch (error) {
      console.error('Load error:', error);
    }
  };

  // Start processing
  const handleStart = async () => {
    try {
      const response = await fetch(`${API_BASE}/start`, { method: 'POST' });
      const data = await response.json();

      if (data.status === 'processing') {
        setProcessing(true);
        console.log('✓ Processing started');
      }
    } catch (error) {
      console.error('Start error:', error);
    }
  };

  // Stop processing
  const handleStop = async () => {
    try {
      await fetch(`${API_BASE}/stop`, { method: 'POST' });
      setProcessing(false);
      console.log('✓ Processing stopped');
    } catch (error) {
      console.error('Stop error:', error);
    }
  };

  // Update stem level
  const updateStemLevel = async (stem, level) => {
    // Update local state immediately for responsiveness
    switch(stem) {
      case 'vocals': setVocalsLevel(level); break;
      case 'drums': setDrumsLevel(level); break;
      case 'bass': setBassLevel(level); break;
      case 'other': setOtherLevel(level); break;
    }

    // Send to server
    try {
      // Send via WebSocket for low latency
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          action: 'set_stem_level',
          stem,
          level
        }));
      } else {
        // Fallback to HTTP
        await fetch(`${API_BASE}/stem/level`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stem, level })
        });
      }
    } catch (error) {
      console.error('Update error:', error);
    }
  };

  // Update stem panning
  const updateStemPan = async (stem, pan) => {
    // Update local state
    switch(stem) {
      case 'vocals': setVocalsPan(pan); break;
      case 'drums': setDrumsPan(pan); break;
      case 'bass': setBassPan(pan); break;
      case 'other': setOtherPan(pan); break;
    }

    // Send to server
    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          action: 'set_stem_pan',
          stem,
          pan
        }));
      } else {
        await fetch(`${API_BASE}/stem/pan`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ stem, pan })
        });
      }
    } catch (error) {
      console.error('Pan error:', error);
    }
  };

  // Update tempo
  const updateTempo = async (factor) => {
    setTempoFactor(factor);

    try {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({
          action: 'set_tempo',
          factor
        }));
      } else {
        await fetch(`${API_BASE}/tempo`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ factor })
        });
      }
    } catch (error) {
      console.error('Tempo error:', error);
    }
  };

  // Format time display
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Key names
  const keyNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  return (
    <div className="spotify-mixer">
      <div className="mixer-header">
        <h2>Spotify Live Mixer</h2>
        <div className="connection-status">
          <span className={`status-dot ${connected ? 'connected' : 'disconnected'}`}></span>
          {connected ? 'Connected' : 'Disconnected'}
        </div>
      </div>

      {/* Search Section */}
      <div className="search-section">
        <div className="search-bar">
          <input
            type="text"
            placeholder="Search Spotify..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          />
          <button onClick={handleSearch}>Search</button>
        </div>

        {searchResults.length > 0 && (
          <div className="search-results">
            {searchResults.map(track => (
              <div
                key={track.id}
                className={`track-item ${currentTrack?.id === track.id ? 'selected' : ''}`}
                onClick={() => handleLoadTrack(track)}
              >
                <div className="track-name">{track.name}</div>
                <div className="track-artist">{track.artist}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Current Track Info */}
      {currentTrack && (
        <div className="current-track">
          <div className="track-info">
            <h3>{currentTrack.name}</h3>
            <p>{currentTrack.artist}</p>
          </div>
          <div className="track-meta">
            <span>{formatTime(position)} / {formatTime(duration)}</span>
            <span>{tempo.toFixed(1)} BPM</span>
            <span>{keyNames[key]} {timeSignature}/4</span>
          </div>
        </div>
      )}

      {/* Transport Controls */}
      <div className="transport-controls">
        <button
          onClick={processing ? handleStop : handleStart}
          disabled={!trackLoaded}
          className={processing ? 'stop-button' : 'start-button'}
        >
          {processing ? 'Stop' : 'Start Processing'}
        </button>
      </div>

      {/* Stem Mixers */}
      <div className="stem-mixers">
        {/* Vocals */}
        <div className="stem-channel">
          <h4>Vocals</h4>
          <div className="level-slider">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={vocalsLevel}
              onChange={(e) => updateStemLevel('vocals', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>{(vocalsLevel * 100).toFixed(0)}%</span>
          </div>
          <div className="pan-slider">
            <span>L</span>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={vocalsPan}
              onChange={(e) => updateStemPan('vocals', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>R</span>
          </div>
        </div>

        {/* Drums */}
        <div className="stem-channel">
          <h4>Drums</h4>
          <div className="level-slider">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={drumsLevel}
              onChange={(e) => updateStemLevel('drums', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>{(drumsLevel * 100).toFixed(0)}%</span>
          </div>
          <div className="pan-slider">
            <span>L</span>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={drumsPan}
              onChange={(e) => updateStemPan('drums', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>R</span>
          </div>
        </div>

        {/* Bass */}
        <div className="stem-channel">
          <h4>Bass</h4>
          <div className="level-slider">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={bassLevel}
              onChange={(e) => updateStemLevel('bass', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>{(bassLevel * 100).toFixed(0)}%</span>
          </div>
          <div className="pan-slider">
            <span>L</span>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={bassPan}
              onChange={(e) => updateStemPan('bass', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>R</span>
          </div>
        </div>

        {/* Other */}
        <div className="stem-channel">
          <h4>Other</h4>
          <div className="level-slider">
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={otherLevel}
              onChange={(e) => updateStemLevel('other', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>{(otherLevel * 100).toFixed(0)}%</span>
          </div>
          <div className="pan-slider">
            <span>L</span>
            <input
              type="range"
              min="-1"
              max="1"
              step="0.01"
              value={otherPan}
              onChange={(e) => updateStemPan('other', parseFloat(e.target.value))}
              disabled={!processing}
            />
            <span>R</span>
          </div>
        </div>
      </div>

      {/* Tempo Control */}
      <div className="tempo-control">
        <h4>Tempo</h4>
        <div className="tempo-slider">
          <input
            type="range"
            min="0.5"
            max="2.0"
            step="0.01"
            value={tempoFactor}
            onChange={(e) => updateTempo(parseFloat(e.target.value))}
            disabled={!processing}
          />
          <span>{(tempoFactor * 100).toFixed(0)}%</span>
        </div>
        <div className="tempo-presets">
          <button onClick={() => updateTempo(0.5)} disabled={!processing}>0.5x</button>
          <button onClick={() => updateTempo(0.75)} disabled={!processing}>0.75x</button>
          <button onClick={() => updateTempo(1.0)} disabled={!processing}>1.0x</button>
          <button onClick={() => updateTempo(1.25)} disabled={!processing}>1.25x</button>
          <button onClick={() => updateTempo(1.5)} disabled={!processing}>1.5x</button>
        </div>
      </div>

      {/* Setup Instructions */}
      {!processing && (
        <div className="setup-instructions">
          <h4>Setup Instructions:</h4>
          <ol>
            <li>Install BlackHole virtual audio device: <code>brew install blackhole-2ch</code></li>
            <li>Set Spotify's output device to BlackHole in Spotify settings</li>
            <li>Search for a track above and click to load it</li>
            <li>Start playing the track in Spotify</li>
            <li>Click "Start Processing" to begin mixing</li>
          </ol>
        </div>
      )}
    </div>
  );
}

export default SpotifyMixer;
