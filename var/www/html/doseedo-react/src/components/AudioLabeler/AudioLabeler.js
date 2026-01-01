import React, { useState, useEffect, useCallback, useRef } from 'react';
import './AudioLabeler.css';

// Available instrument groups
const INSTRUMENT_GROUPS = [
  'drums', 'voice', 'guitar', 'piano', 'bass', 'room',
  'strings', 'winds', 'brass', 'synth', 'percussion',
  'fx', 'plucked', 'organ', 'mallets', 'click', 'undefined'
];

// Subgroups per group
const SUBGROUPS = {
  brass: ['trumpet', 'trombone', 'french_horn', 'tuba', 'undefined'],
  guitar: ['electric_guitar', 'acoustic_guitar', 'undefined'],
  piano: ['acoustic_piano', 'keys', 'electric_piano', 'undefined'],
  strings: ['violin', 'cello', 'viola', 'undefined'],
  winds: ['sax', 'clarinet', 'flute', 'oboe', 'bassoon', 'undefined'],
  bass: ['electric_bass', 'upright_bass', 'synth_bass', 'undefined'],
};

/**
 * Audio Labeler Component
 * Review classifier predictions and correct labels
 */
const AudioLabeler = () => {
  // Manifest state
  const [manifests, setManifests] = useState([]);
  const [selectedManifest, setSelectedManifest] = useState(null);
  const [manifestData, setManifestData] = useState(null);

  // Entry state
  const [entries, setEntries] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [corrections, setCorrections] = useState({});

  // Filter state
  const [groupFilter, setGroupFilter] = useState('all');
  const [subgroupFilter, setSubgroupFilter] = useState('all');
  const [confidenceFilter, setConfidenceFilter] = useState('all'); // all, high, medium, low

  // Available filter options (from manifest)
  const [availableGroups, setAvailableGroups] = useState([]);
  const [availableSubgroups, setAvailableSubgroups] = useState([]);

  // UI state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState(null);
  const [playingPath, setPlayingPath] = useState(null);

  const audioRef = useRef(null);

  // Load manifests list
  const loadManifests = useCallback(async () => {
    try {
      const response = await fetch('/api/monitor/manifests');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setManifests(data.manifests || []);

      // Auto-select first classifier result manifest
      const classifierManifest = data.manifests.find(m => m.is_classifier_result);
      if (classifierManifest) {
        setSelectedManifest(classifierManifest.filename);
      } else if (data.manifests.length > 0) {
        setSelectedManifest(data.manifests[0].filename);
      }
    } catch (err) {
      console.error('Failed to load manifests:', err);
      setError(`Failed to load manifests: ${err.message}`);
    }
  }, []);

  // Load manifest entries
  const loadManifestEntries = useCallback(async () => {
    if (!selectedManifest) return;

    try {
      setLoading(true);

      // Build query params
      const params = new URLSearchParams();
      if (groupFilter !== 'all') params.append('group', groupFilter);
      if (subgroupFilter !== 'all') params.append('subgroup', subgroupFilter);

      // Confidence filter
      if (confidenceFilter === 'high') params.append('confidence_min', '0.85');
      if (confidenceFilter === 'medium') {
        params.append('confidence_min', '0.65');
        params.append('confidence_max', '0.85');
      }
      if (confidenceFilter === 'low') params.append('confidence_max', '0.65');

      params.append('limit', '5000');

      const url = `/api/monitor/manifest/${selectedManifest}?${params.toString()}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      setManifestData(data);
      setEntries(data.entries || []);
      setAvailableGroups(data.available_groups || []);
      setAvailableSubgroups(data.available_subgroups || []);
      setCurrentIndex(0);
      setError(null);
    } catch (err) {
      console.error('Failed to load manifest:', err);
      setError(`Failed to load manifest: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedManifest, groupFilter, subgroupFilter, confidenceFilter]);

  // Load corrections
  const loadCorrections = useCallback(async () => {
    try {
      const response = await fetch('/api/monitor/corrections');
      if (response.ok) {
        const data = await response.json();
        setCorrections(data.corrections || {});
      }
    } catch (err) {
      console.error('Failed to load corrections:', err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadManifests();
    loadCorrections();
  }, [loadManifests, loadCorrections]);

  // Load entries when manifest or filters change
  useEffect(() => {
    if (selectedManifest) {
      loadManifestEntries();
    }
  }, [selectedManifest, loadManifestEntries]);

  const currentEntry = entries[currentIndex];

  // Get correction for current item
  const getCurrentCorrection = () => {
    if (!currentEntry) return null;
    return corrections[currentEntry.path];
  };

  // Save correction
  const saveCorrection = async (path, group, subgroup = 'undefined') => {
    const newCorrections = {
      ...corrections,
      [path]: { group, subgroup, corrected_at: new Date().toISOString() }
    };
    setCorrections(newCorrections);

    try {
      setSaving(true);
      const response = await fetch('/api/monitor/corrections', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path, group, subgroup })
      });
      if (response.ok) {
        setSaveMessage('Saved!');
        setTimeout(() => setSaveMessage(null), 1500);
      }
    } catch (err) {
      console.error('Failed to save correction:', err);
    } finally {
      setSaving(false);
    }
  };

  // Confirm prediction as correct
  const confirmPrediction = () => {
    if (!currentEntry) return;
    saveCorrection(
      currentEntry.path,
      currentEntry.group,
      currentEntry.subgroup || 'undefined'
    );
    goNext();
  };

  // Navigation
  const goNext = () => {
    if (currentIndex < entries.length - 1) {
      setCurrentIndex(currentIndex + 1);
    }
  };

  const goPrev = () => {
    if (currentIndex > 0) {
      setCurrentIndex(currentIndex - 1);
    }
  };

  // Play audio
  const playAudio = (path) => {
    if (audioRef.current) {
      const audioUrl = `/api/monitor/audio?path=${encodeURIComponent(path)}`;
      audioRef.current.src = audioUrl;
      audioRef.current.play();
      setPlayingPath(path);
    }
  };

  // Update audio source when entry changes
  useEffect(() => {
    if (currentEntry && audioRef.current) {
      const audioUrl = `/api/monitor/audio?path=${encodeURIComponent(currentEntry.path)}`;
      audioRef.current.src = audioUrl;
      audioRef.current.load();
      setPlayingPath(null);
    }
  }, [currentEntry?.path]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      switch (e.key) {
        case 'ArrowRight':
        case 'n':
          goNext();
          break;
        case 'ArrowLeft':
        case 'p':
          goPrev();
          break;
        case ' ':
          e.preventDefault();
          if (currentEntry) playAudio(currentEntry.path);
          break;
        case 'Enter':
          e.preventDefault();
          confirmPrediction();
          break;
        default:
          break;
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [currentIndex, entries, currentEntry]);

  // Export corrections
  const exportCorrections = async () => {
    try {
      const response = await fetch('/api/monitor/corrections/export');
      if (response.ok) {
        const data = await response.json();
        setSaveMessage(`Exported ${data.count} corrections to ${data.path}`);
        setTimeout(() => setSaveMessage(null), 3000);
      }
    } catch (err) {
      console.error('Export failed:', err);
    }
  };

  // Get filename from path
  const getFilename = (path) => {
    return path ? path.split('/').pop() : '';
  };

  // Get session from path
  const getSession = (path) => {
    if (!path) return '';
    const parts = path.split('/');
    const audioIdx = parts.indexOf('Audio Files');
    if (audioIdx > 0) {
      return parts[audioIdx - 1];
    }
    return parts[parts.length - 2] || '';
  };

  // Handle manifest change
  const handleManifestChange = (e) => {
    setSelectedManifest(e.target.value);
    setGroupFilter('all');
    setSubgroupFilter('all');
    setConfidenceFilter('all');
  };

  if (loading && manifests.length === 0) {
    return (
      <div className="audio-labeler">
        <div className="loading">Loading manifests...</div>
      </div>
    );
  }

  if (error && manifests.length === 0) {
    return (
      <div className="audio-labeler">
        <div className="error">
          <h2>Error</h2>
          <p>{error}</p>
          <button onClick={loadManifests}>Retry</button>
        </div>
      </div>
    );
  }

  const correction = getCurrentCorrection();
  const effectiveGroup = correction?.group || currentEntry?.group;
  const allSubgroups = SUBGROUPS[effectiveGroup] || ['undefined'];

  return (
    <div className="audio-labeler">
      {/* Header */}
      <div className="labeler-header">
        <h1>Audio Labeler</h1>
        <p className="subtitle">Review and correct classifier predictions</p>

        <div className="header-controls">
          {/* Manifest Selector */}
          <div className="filter-group manifest-selector">
            <label>Manifest:</label>
            <select value={selectedManifest || ''} onChange={handleManifestChange}>
              <option value="">Select manifest...</option>
              {manifests.map(m => (
                <option key={m.filename} value={m.filename}>
                  {m.filename} ({m.entries} entries)
                </option>
              ))}
            </select>
          </div>

          {/* Group Filter */}
          <div className="filter-group">
            <label>Group:</label>
            <select
              value={groupFilter}
              onChange={(e) => { setGroupFilter(e.target.value); setSubgroupFilter('all'); }}
            >
              <option value="all">All Groups</option>
              {availableGroups.map(g => (
                <option key={g} value={g}>{g}</option>
              ))}
            </select>
          </div>

          {/* Subgroup Filter */}
          {availableSubgroups.length > 0 && (
            <div className="filter-group">
              <label>Subgroup:</label>
              <select
                value={subgroupFilter}
                onChange={(e) => setSubgroupFilter(e.target.value)}
              >
                <option value="all">All Subgroups</option>
                {availableSubgroups.map(s => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>
          )}

          {/* Confidence Filter */}
          <div className="filter-group">
            <label>Confidence:</label>
            <select
              value={confidenceFilter}
              onChange={(e) => setConfidenceFilter(e.target.value)}
            >
              <option value="all">All</option>
              <option value="high">High (85%+)</option>
              <option value="medium">Medium (65-85%)</option>
              <option value="low">Low (&lt;65%)</option>
            </select>
          </div>

          <button className="export-btn" onClick={exportCorrections}>
            Export Corrections
          </button>
        </div>

        <div className="stats">
          <span>Showing {entries.length} entries</span>
          <span>|</span>
          <span>{Object.keys(corrections).length} corrected</span>
          {manifestData?.total > entries.length && (
            <>
              <span>|</span>
              <span>{manifestData.total} total in manifest</span>
            </>
          )}
        </div>
      </div>

      {/* Main Content */}
      {loading ? (
        <div className="loading">Loading entries...</div>
      ) : entries.length === 0 ? (
        <div className="no-results">
          <p>No entries match the current filter.</p>
          {selectedManifest && <p>Try changing the filters or selecting a different manifest.</p>}
        </div>
      ) : (
        <div className="labeler-content">
          {/* Navigation */}
          <div className="navigation">
            <button onClick={goPrev} disabled={currentIndex === 0}>
              Prev (P)
            </button>
            <span className="position">
              {currentIndex + 1} / {entries.length}
            </span>
            <button onClick={goNext} disabled={currentIndex >= entries.length - 1}>
              Next (N)
            </button>
          </div>

          {/* Current Item */}
          {currentEntry && (
            <div className="current-item">
              {/* File Info */}
              <div className="file-info">
                <div className="filename">{getFilename(currentEntry.path)}</div>
                <div className="session">{getSession(currentEntry.path)}</div>
                <div className="path">{currentEntry.path}</div>
              </div>

              {/* Audio Player */}
              <div className="audio-section">
                <button
                  className="play-btn"
                  onClick={() => playAudio(currentEntry.path)}
                >
                  {playingPath === currentEntry.path ? 'Playing...' : 'Play (Space)'}
                </button>
                <audio ref={audioRef} controls onEnded={() => setPlayingPath(null)} />
              </div>

              {/* Prediction Info */}
              <div className="prediction-info">
                <div className="prediction-row">
                  <span className="label">Group:</span>
                  <span className={`value group-${currentEntry.group}`}>
                    {currentEntry.group}
                  </span>
                </div>
                <div className="prediction-row">
                  <span className="label">Subgroup:</span>
                  <span className="value">
                    {currentEntry.subgroup || 'undefined'}
                  </span>
                </div>
                {currentEntry.confidence !== undefined && currentEntry.confidence !== 1.0 && (
                  <div className="prediction-row">
                    <span className="label">Confidence:</span>
                    <span className={`value confidence-${
                      currentEntry.confidence >= 0.85 ? 'high' :
                      currentEntry.confidence >= 0.65 ? 'medium' : 'low'
                    }`}>
                      {(currentEntry.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                )}
                {currentEntry.all_probabilities && Object.keys(currentEntry.all_probabilities).length > 0 && (
                  <div className="all-probs">
                    <span className="label">All predictions:</span>
                    <div className="prob-bars">
                      {Object.entries(currentEntry.all_probabilities)
                        .sort((a, b) => b[1] - a[1])
                        .map(([label, prob]) => (
                          <div key={label} className="prob-bar">
                            <span className="prob-label">{label}</span>
                            <div className="prob-fill" style={{ width: `${prob * 100}%` }} />
                            <span className="prob-value">{(prob * 100).toFixed(0)}%</span>
                          </div>
                        ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Correction Status */}
              {correction && (
                <div className="correction-status">
                  Corrected to: <strong>{correction.group}</strong>
                  {correction.subgroup !== 'undefined' && ` / ${correction.subgroup}`}
                </div>
              )}

              {/* Label Selection */}
              <div className="label-section">
                <h3>Correct Label</h3>

                <div className="group-selector">
                  <label>Group:</label>
                  <div className="group-buttons">
                    {INSTRUMENT_GROUPS.map(group => (
                      <button
                        key={group}
                        className={`group-btn ${effectiveGroup === group ? 'selected' : ''}`}
                        onClick={() => saveCorrection(currentEntry.path, group, 'undefined')}
                      >
                        {group}
                      </button>
                    ))}
                  </div>
                </div>

                {allSubgroups.length > 1 && (
                  <div className="subgroup-selector">
                    <label>Subgroup:</label>
                    <div className="subgroup-buttons">
                      {allSubgroups.map(subgroup => (
                        <button
                          key={subgroup}
                          className={`subgroup-btn ${(correction?.subgroup || currentEntry.subgroup) === subgroup ? 'selected' : ''}`}
                          onClick={() => saveCorrection(currentEntry.path, effectiveGroup, subgroup)}
                        >
                          {subgroup}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                <div className="action-buttons">
                  <button
                    className="confirm-btn"
                    onClick={confirmPrediction}
                  >
                    Confirm & Next (Enter)
                  </button>
                </div>
              </div>

              {/* Save Message */}
              {saveMessage && (
                <div className="save-message">{saveMessage}</div>
              )}
            </div>
          )}

          {/* Keyboard Shortcuts */}
          <div className="shortcuts">
            <h4>Keyboard Shortcuts</h4>
            <ul>
              <li><kbd>Space</kbd> - Play audio</li>
              <li><kbd>P</kbd> - Previous</li>
              <li><kbd>N</kbd> - Next</li>
              <li><kbd>Enter</kbd> - Confirm & Next</li>
            </ul>
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioLabeler;
