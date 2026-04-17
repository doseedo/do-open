import React, { useState, useEffect, useCallback, useRef } from 'react';
import './DataMonitor.css';

// AES-GCM decryption using Web Crypto API
const decryptAESGCM = async (ciphertextB64, nonceB64, keyB64) => {
  try {
    const ciphertext = Uint8Array.from(atob(ciphertextB64), c => c.charCodeAt(0));
    const nonce = Uint8Array.from(atob(nonceB64), c => c.charCodeAt(0));
    const keyBytes = Uint8Array.from(atob(keyB64), c => c.charCodeAt(0));

    const key = await crypto.subtle.importKey(
      'raw', keyBytes, { name: 'AES-GCM' }, false, ['decrypt']
    );

    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: nonce }, key, ciphertext
    );

    return new TextDecoder().decode(decrypted);
  } catch (e) {
    console.error('Decryption error:', e);
    throw e;
  }
};

const DataMonitor = () => {
  const [stats, setStats] = useState(null);
  const [folderDetails, setFolderDetails] = useState({});
  const [selectedFolder, setSelectedFolder] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [dateSortBy, setDateSortBy] = useState('date'); // 'date' or 'size'
  const [typeSortBy, setTypeSortBy] = useState('size'); // 'name' or 'size'
  const [showInstrumentModal, setShowInstrumentModal] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState({});
  const decryptKeyRef = useRef(null);

  // Fetch decryption key (cached)
  const getDecryptKey = useCallback(async () => {
    if (decryptKeyRef.current) return decryptKeyRef.current;

    const keyResponse = await fetch('/api/monitor/decrypt-key');
    if (!keyResponse.ok) {
      throw new Error('Failed to get decryption key');
    }
    const keyData = await keyResponse.json();
    decryptKeyRef.current = keyData.key;
    return keyData.key;
  }, []);

  // Fetch overall stats
  const fetchStats = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);

      // First get decryption key
      const decryptKey = await getDecryptKey();

      const response = await fetch('/api/monitor/stats');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const rawData = await response.json();

      // Decrypt if encrypted
      let data;
      if (rawData.encrypted) {
        const decryptedJson = await decryptAESGCM(rawData.data, rawData.nonce, decryptKey);
        data = JSON.parse(decryptedJson);
      } else {
        data = rawData;
      }

      setStats(data);
      setLastUpdated(new Date());
    } catch (err) {
      setError(`Failed to fetch stats: ${err.message}`);
      console.error('Stats fetch error:', err);
    } finally {
      setIsLoading(false);
    }
  }, [getDecryptKey]);

  // Fetch details for a specific folder
  const fetchFolderDetails = useCallback(async (folderId) => {
    try {
      const response = await fetch(`/api/monitor/folder/${folderId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setFolderDetails(prev => ({ ...prev, [folderId]: data }));
    } catch (err) {
      console.error(`Folder details fetch error for ${folderId}:`, err);
    }
  }, []);

  // Initial load
  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  // Auto-refresh every 30 seconds if enabled
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchStats]);

  // Fetch folder details when selected
  useEffect(() => {
    if (selectedFolder && !folderDetails[selectedFolder]) {
      fetchFolderDetails(selectedFolder);
    }
  }, [selectedFolder, folderDetails, fetchFolderDetails]);

  // Format bytes to human readable
  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  // Format number with commas
  const formatNumber = (num) => {
    return num?.toLocaleString() || '0';
  };

  // Toggle instrument group expansion
  const toggleGroup = (group) => {
    setExpandedGroups(prev => ({
      ...prev,
      [group]: !prev[group]
    }));
  };

  // Get instrument icon
  const getInstrumentIcon = (group) => {
    const icons = {
      guitar: '🎸',
      piano: '🎹',
      bass: '🎸',
      strings: '🎻',
      winds: '🎷',
      brass: '🎺',
      undefined: '❓'
    };
    return icons[group] || '🎵';
  };

  return (
    <div className="data-monitor">
      <div className="monitor-header">
        <h1>Data Monitor</h1>
        <p className="subtitle">GCS Bucket Statistics & Data Overview</p>

        <div className="header-controls">
          <button
            className="refresh-btn"
            onClick={fetchStats}
            disabled={isLoading}
          >
            {isLoading ? 'Loading...' : '🔄 Refresh'}
          </button>

          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (30s)
          </label>

          {lastUpdated && (
            <span className="last-updated">
              Last updated: {lastUpdated.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* Summary Cards */}
      <div className="summary-section">
        <div className="summary-card total">
          <div className="card-icon">📦</div>
          <div className="card-content">
            <div className="card-value">{formatNumber(stats?.total_files)}</div>
            <div className="card-label">Total Files</div>
          </div>
        </div>

        <div className="summary-card size">
          <div className="card-icon">💾</div>
          <div className="card-content">
            <div className="card-value">{stats?.total_size_formatted || formatBytes(stats?.total_size || 0)}</div>
            <div className="card-label">Total Size</div>
          </div>
        </div>

        <div
          className="summary-card audio clickable"
          onClick={() => setShowInstrumentModal(true)}
          title="Click to see breakdown by instrument"
        >
          <div className="card-icon">🎵</div>
          <div className="card-content">
            <div className="card-value">{stats?.audio_hours_formatted || '0 hrs'}</div>
            <div className="card-label">Audio Duration</div>
          </div>
          <div className="card-expand-hint">▶</div>
        </div>

        <div className="summary-card recent">
          <div className="card-icon">📅</div>
          <div className="card-content">
            <div className="card-value">{formatNumber(stats?.modified_today || 0)}</div>
            <div className="card-label">Modified Today</div>
          </div>
        </div>
      </div>

      {/* Folder Grid */}
      <div className="folders-section">
        <h2>Data Folders</h2>
        <div className="folder-grid">
          {(stats?.folders || []).map(folder => {
            const isSelected = selectedFolder === folder.id;

            return (
              <div
                key={folder.id}
                className={`folder-card ${isSelected ? 'selected' : ''}`}
                onClick={() => setSelectedFolder(isSelected ? null : folder.id)}
              >
                <div className="folder-icon">{folder.icon || '📁'}</div>
                <div className="folder-name">{folder.name}</div>
                <div className="folder-stats">
                  <div className="stat">
                    <span className="stat-value">{formatNumber(folder.files)}</span>
                    <span className="stat-label">files</span>
                  </div>
                  <div className="stat">
                    <span className="stat-value">{folder.size_formatted || formatBytes(folder.size)}</span>
                    <span className="stat-label">size</span>
                  </div>
                  {folder.subdirs > 0 && (
                    <div className="stat">
                      <span className="stat-value">{formatNumber(folder.subdirs)}</span>
                      <span className="stat-label">sessions</span>
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Folder Details Panel */}
      {selectedFolder && (
        <div className="details-panel">
          <div className="details-header">
            <h3>
              {stats?.folders?.find(f => f.id === selectedFolder)?.icon || '📁'}{' '}
              {stats?.folders?.find(f => f.id === selectedFolder)?.name}
            </h3>
            <button
              className="close-btn"
              onClick={() => setSelectedFolder(null)}
            >
              ✕
            </button>
          </div>

          {folderDetails[selectedFolder] ? (
            <div className="details-content">
              {/* Date breakdown for dated folders */}
              {folderDetails[selectedFolder].byDate && Object.keys(folderDetails[selectedFolder].byDate).length > 0 && (
                <div className="date-breakdown">
                  <div className="breakdown-header">
                    <h4>By Date ({Object.keys(folderDetails[selectedFolder].byDate).length})</h4>
                    <select
                      className="sort-select"
                      value={dateSortBy}
                      onChange={(e) => setDateSortBy(e.target.value)}
                    >
                      <option value="date">Sort by Date</option>
                      <option value="size">Sort by Size</option>
                    </select>
                  </div>
                  <div className="date-list">
                    {Object.entries(folderDetails[selectedFolder].byDate)
                      .sort(([aKey, aData], [bKey, bData]) =>
                        dateSortBy === 'date'
                          ? bKey.localeCompare(aKey)
                          : bData.size - aData.size
                      )
                      .map(([date, data]) => (
                        <div key={date} className="date-row">
                          <span className="date">{date}</span>
                          <span className="count">{formatNumber(data.count)} files</span>
                          <span className="size">{formatBytes(data.size)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* File type breakdown */}
              {folderDetails[selectedFolder].byType && (
                <div className="type-breakdown">
                  <div className="breakdown-header">
                    <h4>By File Type</h4>
                    <select
                      className="sort-select"
                      value={typeSortBy}
                      onChange={(e) => setTypeSortBy(e.target.value)}
                    >
                      <option value="size">Sort by Size</option>
                      <option value="name">Sort by Name</option>
                    </select>
                  </div>
                  <div className="type-list">
                    {Object.entries(folderDetails[selectedFolder].byType)
                      .sort(([aKey, aData], [bKey, bData]) =>
                        typeSortBy === 'size'
                          ? bData.size - aData.size
                          : aKey.localeCompare(bKey)
                      )
                      .map(([type, data]) => (
                        <div key={type} className="type-row">
                          <span className="type">{type || 'unknown'}</span>
                          <span className="count">{formatNumber(data.count)}</span>
                          <span className="size">{formatBytes(data.size)}</span>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="loading-details">Loading details...</div>
          )}
        </div>
      )}

      {/* Instructions */}
      <div className="instructions">
        <h4>Data Monitor Info</h4>
        <ul>
          <li>Click on a folder card to view detailed breakdown</li>
          <li>Click on Audio Duration to see instrument breakdown</li>
          <li>Enable auto-refresh to keep stats updated</li>
        </ul>
      </div>

      {/* Instrument Hours Modal */}
      {showInstrumentModal && (
        <div className="modal-overlay" onClick={() => setShowInstrumentModal(false)}>
          <div className="instrument-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>🎵 Audio Hours by Instrument</h3>
              <button className="close-btn" onClick={() => setShowInstrumentModal(false)}>✕</button>
            </div>
            <div className="modal-content">
              <div className="total-hours">
                Total: <strong>{stats?.audio_hours_formatted || '0 hrs'}</strong>
              </div>
              <div className="instrument-list">
                {stats?.instrument_hours && Object.entries(stats.instrument_hours)
                  .sort(([, a], [, b]) => b.seconds - a.seconds)
                  .map(([group, data]) => (
                    <div key={group} className="instrument-group">
                      <div
                        className={`group-header ${expandedGroups[group] ? 'expanded' : ''}`}
                        onClick={() => toggleGroup(group)}
                      >
                        <span className="expand-icon">{expandedGroups[group] ? '▼' : '▶'}</span>
                        <span className="group-icon">{getInstrumentIcon(group)}</span>
                        <span className="group-name">{group}</span>
                        <span className="group-hours">{data.hours_formatted}</span>
                      </div>
                      {expandedGroups[group] && data.subgroups && (
                        <div className="subgroup-list">
                          {Object.entries(data.subgroups)
                            .sort(([, a], [, b]) => b.seconds - a.seconds)
                            .map(([subgroup, subData]) => (
                              <div key={subgroup} className="subgroup-row">
                                <span className="subgroup-name">{subgroup}</span>
                                <span className="subgroup-hours">{subData.hours_formatted}</span>
                              </div>
                            ))}
                        </div>
                      )}
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataMonitor;
