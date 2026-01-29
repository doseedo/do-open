import React, { useState, useEffect, useCallback, useRef } from 'react';
import './AudioLabeler.css';

// Available instrument groups
const INSTRUMENT_GROUPS = [
  'drums', 'voice', 'guitar', 'piano', 'bass',
  'strings', 'winds', 'brass', 'synth', 'percussion',
  'plucked', 'organ', 'mallets', 'undefined'
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
  // View mode: 'predictions' or 'flagged'
  const [viewMode, setViewMode] = useState('predictions');

  // Classifier state
  const [classifiers, setClassifiers] = useState([]);
  const [selectedClassifier, setSelectedClassifier] = useState('instrument');

  // Entry state
  const [entries, setEntries] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [corrections, setCorrections] = useState({});
  const [summary, setSummary] = useState(null);

  // Filter state
  const [confidenceFilter, setConfidenceFilter] = useState('low'); // Start with low confidence (needs review)
  const [groupFilter, setGroupFilter] = useState('all');
  const [flagFilter, setFlagFilter] = useState('all');
  const [availableGroups, setAvailableGroups] = useState([]);

  // UI state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState(null);
  const [playingPath, setPlayingPath] = useState(null);

  const audioRef = useRef(null);

  // Load classifiers list
  const loadClassifiers = useCallback(async () => {
    try {
      const response = await fetch('/api/monitor/classifiers');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const data = await response.json();
      setClassifiers(data.classifiers || []);

      // Auto-select first classifier with predictions
      const withPredictions = data.classifiers.find(c => c.has_predictions);
      if (withPredictions) {
        setSelectedClassifier(withPredictions.type);
      }
    } catch (err) {
      console.error('Failed to load classifiers:', err);
      setError(`Failed to load classifiers: ${err.message}`);
    }
  }, []);

  // Load predictions
  const loadPredictions = useCallback(async () => {
    if (!selectedClassifier) return;

    try {
      setLoading(true);
      const params = new URLSearchParams();
      params.append('confidence', confidenceFilter);
      if (groupFilter !== 'all') params.append('group', groupFilter);
      params.append('limit', '1000');

      const url = `/api/monitor/classifier/${selectedClassifier}/predictions?${params.toString()}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      setEntries(data.entries || []);
      setAvailableGroups(data.available_groups || []);
      setSummary(data.summary || null);
      setCurrentIndex(0);
      setError(null);
    } catch (err) {
      console.error('Failed to load predictions:', err);
      setError(`Failed to load predictions: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedClassifier, confidenceFilter, groupFilter]);

  // Load flagged entries
  const loadFlagged = useCallback(async () => {
    if (!selectedClassifier) return;

    try {
      setLoading(true);
      const params = new URLSearchParams();
      params.append('flag_type', flagFilter);
      params.append('limit', '1000');

      const url = `/api/monitor/classifier/${selectedClassifier}/flagged?${params.toString()}`;
      const response = await fetch(url);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const data = await response.json();
      setEntries(data.entries || []);
      setSummary(data.summary || null);
      setCurrentIndex(0);
      setError(null);
    } catch (err) {
      console.error('Failed to load flagged:', err);
      setError(`Failed to load flagged entries: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [selectedClassifier, flagFilter]);

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
    loadClassifiers();
    loadCorrections();
  }, [loadClassifiers, loadCorrections]);

  // Load entries when classifier or filters change
  useEffect(() => {
    if (selectedClassifier) {
      if (viewMode === 'predictions') {
        loadPredictions();
      } else {
        loadFlagged();
      }
    }
  }, [selectedClassifier, viewMode, loadPredictions, loadFlagged]);

  const currentEntry = entries[currentIndex];
  const currentClassifier = classifiers.find(c => c.type === selectedClassifier);

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
    const group = currentEntry.predicted_group || currentEntry.true_label;
    saveCorrection(currentEntry.path, group, 'undefined');
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
        setSaveMessage(`Exported ${data.count} corrections`);
        setTimeout(() => setSaveMessage(null), 3000);
      }
    } catch (err) {
      console.error('Export failed:', err);
    }
  };

  // Get filename from path
  const getFilename = (path) => path ? path.split('/').pop() : '';

  // Get session from path
  const getSession = (path) => {
    if (!path) return '';
    const parts = path.split('/');
    const audioIdx = parts.indexOf('Audio Files');
    if (audioIdx > 0) return parts[audioIdx - 1];
    return parts[parts.length - 2] || '';
  };

  // Confidence color
  const getConfidenceColor = (conf) => {
    if (conf >= 0.85) return '#4ade80';
    if (conf >= 0.65) return '#fbbf24';
    return '#f87171';
  };

  const correction = getCurrentCorrection();
  const effectiveGroup = correction?.group || currentEntry?.predicted_group || currentEntry?.true_label;
  const allSubgroups = SUBGROUPS[effectiveGroup] || ['undefined'];

  if (loading && classifiers.length === 0) {
    return (
      <div className="audio-labeler">
        <div className="loading">Loading classifiers...</div>
      </div>
    );
  }

  return (
    <div className="audio-labeler">
      {/* Header */}
      <div className="labeler-header">
        <h1>Classifier Review</h1>

        {/* Classifier Status Cards */}
        <div className="classifier-cards">
          {classifiers.map(c => (
            <div
              key={c.type}
              className={`classifier-card ${selectedClassifier === c.type ? 'selected' : ''} ${!c.has_model ? 'no-model' : ''}`}
              onClick={() => c.has_model && setSelectedClassifier(c.type)}
            >
              <div className="card-title">{c.name}</div>
              {c.has_model ? (
                <>
                  <div className="card-stats">
                    {c.has_predictions && (
                      <span className="stat predictions">{c.predictions_count.toLocaleString()} predictions</span>
                    )}
                    {c.has_validation && c.flagged_count > 0 && (
                      <span className="stat flagged">{c.flagged_count.toLocaleString()} flagged</span>
                    )}
                  </div>
                  {!c.has_predictions && <div className="card-hint">Run classifier to get predictions</div>}
                </>
              ) : (
                <div className="card-hint">Model not trained yet</div>
              )}
            </div>
          ))}
        </div>

        {/* View Mode Tabs */}
        {currentClassifier?.has_model && (
          <div className="view-tabs">
            <button
              className={`tab ${viewMode === 'predictions' ? 'active' : ''}`}
              onClick={() => setViewMode('predictions')}
              disabled={!currentClassifier?.has_predictions}
            >
              Predictions {currentClassifier?.predictions_count ? `(${currentClassifier.predictions_count.toLocaleString()})` : ''}
            </button>
            <button
              className={`tab ${viewMode === 'flagged' ? 'active' : ''}`}
              onClick={() => setViewMode('flagged')}
              disabled={!currentClassifier?.has_validation}
            >
              Flagged {currentClassifier?.flagged_count ? `(${currentClassifier.flagged_count.toLocaleString()})` : ''}
            </button>
          </div>
        )}

        {/* Filters */}
        <div className="filters">
          {viewMode === 'predictions' ? (
            <>
              <div className="filter-group">
                <label>Confidence:</label>
                <select value={confidenceFilter} onChange={(e) => setConfidenceFilter(e.target.value)}>
                  <option value="all">All</option>
                  <option value="low">Low (&lt;65%) - Review First</option>
                  <option value="medium">Medium (65-85%)</option>
                  <option value="high">High (85%+)</option>
                </select>
              </div>
              <div className="filter-group">
                <label>Group:</label>
                <select value={groupFilter} onChange={(e) => setGroupFilter(e.target.value)}>
                  <option value="all">All Groups</option>
                  {availableGroups.map(g => <option key={g} value={g}>{g}</option>)}
                </select>
              </div>
            </>
          ) : (
            <div className="filter-group">
              <label>Flag Type:</label>
              <select value={flagFilter} onChange={(e) => setFlagFilter(e.target.value)}>
                <option value="all">All Flags</option>
                <option value="disagreement">Disagreement (wrong label?)</option>
                <option value="uncertain">Uncertain (ambiguous)</option>
                <option value="outlier">Outlier (unusual)</option>
              </select>
            </div>
          )}

          <button className="export-btn" onClick={exportCorrections}>
            Export ({Object.keys(corrections).length})
          </button>
        </div>

        <div className="stats-bar">
          <span>{entries.length} entries to review</span>
          <span>|</span>
          <span>{Object.keys(corrections).length} corrected this session</span>
        </div>
      </div>

      {/* Main Content */}
      {loading ? (
        <div className="loading">Loading entries...</div>
      ) : !currentClassifier?.has_model ? (
        <div className="no-results">
          <h2>No Model Available</h2>
          <p>Train the {selectedClassifier} classifier first:</p>
          <code>python latent_instrument_classifier.py --mode train ...</code>
        </div>
      ) : entries.length === 0 ? (
        <div className="no-results">
          <h2>No Entries</h2>
          <p>{viewMode === 'predictions' ? 'Run the classifier to generate predictions.' : 'Run validation to find flagged entries.'}</p>
        </div>
      ) : (
        <div className="labeler-content">
          {/* Navigation */}
          <div className="navigation">
            <button onClick={goPrev} disabled={currentIndex === 0}>← Prev (P)</button>
            <span className="position">{currentIndex + 1} / {entries.length}</span>
            <button onClick={goNext} disabled={currentIndex >= entries.length - 1}>Next (N) →</button>
          </div>

          {/* Current Item */}
          {currentEntry && (
            <div className="current-item">
              {/* File Info */}
              <div className="file-info">
                <div className="filename">{getFilename(currentEntry.path)}</div>
                <div className="session">{getSession(currentEntry.path)}</div>
              </div>

              {/* Audio Player */}
              <div className="audio-section">
                <button className="play-btn" onClick={() => playAudio(currentEntry.path)}>
                  {playingPath === currentEntry.path ? '▶ Playing...' : '▶ Play (Space)'}
                </button>
                <audio ref={audioRef} controls onEnded={() => setPlayingPath(null)} />
              </div>

              {/* Prediction/Flag Info */}
              <div className="prediction-info">
                {viewMode === 'predictions' ? (
                  <>
                    <div className="prediction-main">
                      <span className="predicted-label">{currentEntry.predicted_group}</span>
                      <span
                        className="confidence-badge"
                        style={{ backgroundColor: getConfidenceColor(currentEntry.confidence) }}
                      >
                        {(currentEntry.confidence * 100).toFixed(0)}%
                      </span>
                    </div>

                    {currentEntry.all_probabilities && Object.keys(currentEntry.all_probabilities).length > 0 && (
                      <div className="prob-bars">
                        {Object.entries(currentEntry.all_probabilities)
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 5)
                          .map(([label, prob]) => (
                            <div key={label} className="prob-bar">
                              <span className="prob-label">{label}</span>
                              <div className="prob-track">
                                <div className="prob-fill" style={{ width: `${prob * 100}%` }} />
                              </div>
                              <span className="prob-value">{(prob * 100).toFixed(0)}%</span>
                            </div>
                          ))}
                      </div>
                    )}
                  </>
                ) : (
                  <div className="flag-info">
                    <div className={`flag-badge ${currentEntry.flag_type}`}>
                      {currentEntry.flag_type}
                    </div>
                    <div className="flag-details">
                      <div>Current label: <strong>{currentEntry.true_label}</strong></div>
                      {currentEntry.predicted_label && (
                        <div>Classifier says: <strong>{currentEntry.predicted_label}</strong> ({(currentEntry.confidence * 100).toFixed(0)}%)</div>
                      )}
                      {currentEntry.entropy && (
                        <div>Entropy: {currentEntry.entropy.toFixed(2)}</div>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Correction Status */}
              {correction && (
                <div className="correction-status">
                  ✓ Corrected to: <strong>{correction.group}</strong>
                  {correction.subgroup !== 'undefined' && ` / ${correction.subgroup}`}
                </div>
              )}

              {/* Quick Actions */}
              <div className="quick-actions">
                <button className="confirm-btn" onClick={confirmPrediction}>
                  ✓ Correct - Next (Enter)
                </button>
              </div>

              {/* Label Selection */}
              <div className="label-section">
                <h3>Change Label</h3>
                <div className="group-buttons">
                  {INSTRUMENT_GROUPS.map(group => (
                    <button
                      key={group}
                      className={`group-btn ${effectiveGroup === group ? 'selected' : ''}`}
                      onClick={() => { saveCorrection(currentEntry.path, group, 'undefined'); goNext(); }}
                    >
                      {group}
                    </button>
                  ))}
                </div>

                {allSubgroups.length > 1 && (
                  <div className="subgroup-section">
                    <h4>Subgroup</h4>
                    <div className="subgroup-buttons">
                      {allSubgroups.map(subgroup => (
                        <button
                          key={subgroup}
                          className={`subgroup-btn ${(correction?.subgroup || 'undefined') === subgroup ? 'selected' : ''}`}
                          onClick={() => saveCorrection(currentEntry.path, effectiveGroup, subgroup)}
                        >
                          {subgroup}
                        </button>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Save Message */}
              {saveMessage && <div className="save-message">{saveMessage}</div>}
            </div>
          )}

          {/* Keyboard Shortcuts */}
          <div className="shortcuts">
            <kbd>Space</kbd> Play &nbsp;
            <kbd>P</kbd> Prev &nbsp;
            <kbd>N</kbd> Next &nbsp;
            <kbd>Enter</kbd> Confirm
          </div>
        </div>
      )}
    </div>
  );
};

export default AudioLabeler;
