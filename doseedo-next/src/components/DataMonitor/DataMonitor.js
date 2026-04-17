import React, { useState, useEffect, useCallback, useRef } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
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
  const [formats, setFormats] = useState(null);
  const [folderDetails, setFolderDetails] = useState({});
  const [selectedFolder, setSelectedFolder] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [dateSortBy, setDateSortBy] = useState('date'); // 'date' or 'size'
  const [typeSortBy, setTypeSortBy] = useState('size'); // 'name' or 'size'
  const [showInstrumentModal, setShowInstrumentModal] = useState(false);
  const [showFileTypeModal, setShowFileTypeModal] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState({});
  const [timeline, setTimeline] = useState(null);
  const [timelineGranularity, setTimelineGranularity] = useState('month'); // 'day' or 'month'
  const [timelineMetric, setTimelineMetric] = useState('hours'); // 'files', 'hours', or 'gb'
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

  // Fetch format coverage stats
  const fetchFormats = useCallback(async () => {
    try {
      const response = await fetch('/api/monitor/formats');
      if (!response.ok) return;
      const data = await response.json();
      if (data.status === 'ok') {
        setFormats(data);
      }
    } catch (err) {
      console.error('Formats fetch error:', err);
    }
  }, []);

  // Fetch timeline data
  const fetchTimeline = useCallback(async (granularity = timelineGranularity, metric = timelineMetric) => {
    try {
      const response = await fetch(`/api/monitor/stats/timeline?granularity=${granularity}&metric=${metric}`);
      if (!response.ok) return;
      const data = await response.json();
      if (data.status === 'ok') {
        setTimeline(data);
      }
    } catch (err) {
      console.error('Timeline fetch error:', err);
    }
  }, [timelineGranularity, timelineMetric]);

  // Initial load
  useEffect(() => {
    fetchStats();
    fetchFormats();
    fetchTimeline();
  }, [fetchStats, fetchFormats, fetchTimeline]);

  // Auto-refresh every 30 seconds if enabled
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      fetchStats();
      fetchFormats();
      fetchTimeline();
    }, 30000);
    return () => clearInterval(interval);
  }, [autoRefresh, fetchStats, fetchFormats, fetchTimeline]);

  // Refetch timeline when granularity or metric changes
  useEffect(() => {
    fetchTimeline(timelineGranularity, timelineMetric);
  }, [timelineGranularity, timelineMetric, fetchTimeline]);

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
        <div
          className="summary-card total clickable"
          onClick={() => setShowFileTypeModal(true)}
          title="Click to see breakdown by file type"
        >
          <div className="card-icon">📦</div>
          <div className="card-content">
            <div className="card-value">{formatNumber(stats?.total_files)}</div>
            <div className="card-label">Total Files</div>
          </div>
          <div className="card-expand-hint">▶</div>
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

        <div className="summary-card sessions">
          <div className="card-icon">🎚️</div>
          <div className="card-content">
            <div className="card-value">{formatNumber(stats?.total_sessions || 0)}</div>
            <div className="card-label">Total Sessions</div>
          </div>
        </div>

        <div className="summary-card recent">
          <div className="card-icon">📅</div>
          <div className="card-content">
            <div className="card-value">{formatNumber(stats?.modified_today || 0)}</div>
            <div className="card-label">Modified Today</div>
          </div>
        </div>
      </div>

      {/* Format Coverage Section */}
      {formats && (
        <div className="formats-section">
          <h2>Format Coverage</h2>
          <div className="formats-grid">
            <div className="format-card">
              <div className="format-header">
                <span className="format-icon">🧠</span>
                <span className="format-name">Latent</span>
                <span className="format-pct">{formats.pct_latent}%</span>
              </div>
              <div className="format-bar">
                <div className="format-bar-fill latent" style={{ width: `${formats.pct_latent}%` }} />
              </div>
              <div className="format-stats">
                <span>{formatNumber(formats.with_latent)} have</span>
                <span className="needs">{formatNumber(formats.needs_latent)} need</span>
              </div>
            </div>

            <div className="format-card">
              <div className="format-header">
                <span className="format-icon">🎛️</span>
                <span className="format-name">Conditioning</span>
                <span className="format-pct">{formats.pct_conditioning}%</span>
              </div>
              <div className="format-bar">
                <div className="format-bar-fill conditioning" style={{ width: `${formats.pct_conditioning}%` }} />
              </div>
              <div className="format-stats">
                <span>{formatNumber(formats.with_conditioning)} have</span>
                <span className="needs">{formatNumber(formats.needs_conditioning)} need</span>
              </div>
            </div>

            <div className="format-card">
              <div className="format-header">
                <span className="format-icon">🎹</span>
                <span className="format-name">MIDI</span>
                <span className="format-pct">{formats.pct_midi}%</span>
              </div>
              <div className="format-bar">
                <div className="format-bar-fill midi" style={{ width: `${formats.pct_midi}%` }} />
              </div>
              <div className="format-stats">
                <span>{formatNumber(formats.with_midi)} have</span>
                <span className="needs">{formatNumber(formats.needs_midi)} need</span>
              </div>
            </div>
          </div>

          {/* Completeness Breakdown */}
          <div className="completeness-section">
            <h3>Completeness</h3>
            <div className="completeness-grid">
              {formats.breakdown?.map((item) => (
                <div key={item.key} className={`completeness-item ${item.key}`}>
                  <span className="completeness-short">{item.short}</span>
                  <span className="completeness-count">{formatNumber(item.count)}</span>
                  <span className="completeness-pct">{item.pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Timeline Chart */}
      {timeline && timeline.data && timeline.data.length > 0 && (
        <div className="timeline-section">
          <div className="timeline-header">
            <h2>Collection Timeline</h2>
            <div className="timeline-controls">
              <select
                className="timeline-select"
                value={timelineGranularity}
                onChange={(e) => setTimelineGranularity(e.target.value)}
              >
                <option value="day">By Day</option>
                <option value="month">By Month</option>
              </select>
              <select
                className="timeline-select"
                value={timelineMetric}
                onChange={(e) => setTimelineMetric(e.target.value)}
              >
                <option value="hours">Hours</option>
                <option value="files">Files</option>
                <option value="gb">GB</option>
              </select>
            </div>
          </div>

          {/* Recharts Stacked Bar Chart */}
          <div className="timeline-chart-container">
            <ResponsiveContainer width="100%" height={300}>
              <BarChart
                data={timeline.data.map(d => ({
                  date: timelineGranularity === 'month' ? d.date.slice(2) : d.date.slice(5),
                  ...d.sources
                }))}
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <XAxis
                  dataKey="date"
                  tick={{ fill: '#888', fontSize: 11 }}
                  angle={-45}
                  textAnchor="end"
                  interval={timelineGranularity === 'month' ? 0 : Math.floor(timeline.data.length / 20)}
                />
                <YAxis
                  tick={{ fill: '#888', fontSize: 12 }}
                  label={{
                    value: timelineMetric === 'hours' ? 'Hours' : timelineMetric === 'gb' ? 'GB' : 'Files',
                    angle: -90,
                    position: 'insideLeft',
                    fill: '#888'
                  }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1a1a2e',
                    border: '1px solid #333',
                    borderRadius: '8px'
                  }}
                  labelStyle={{ color: '#fff' }}
                  formatter={(value, name) => {
                    const source = timeline.sources?.find(s => s.id === name);
                    return [value.toFixed(1), source?.name || name];
                  }}
                />
                <Legend
                  formatter={(value) => {
                    const source = timeline.sources?.find(s => s.id === value);
                    return source?.name || value;
                  }}
                  wrapperStyle={{ paddingTop: '10px' }}
                />
                {timeline.sources?.map(source => (
                  <Bar
                    key={source.id}
                    dataKey={source.id}
                    stackId="a"
                    fill={source.color}
                    name={source.id}
                  />
                ))}
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="timeline-summary">
            <span className="summary-item">
              {timeline.total_dates} {timelineGranularity === 'month' ? 'months' : 'days'}
            </span>
            <span className="summary-item">
              {timeline.data.reduce((sum, d) => sum + d.total, 0).toFixed(1)} total {timelineMetric}
            </span>
          </div>
        </div>
      )}

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

      {/* File Type Breakdown Modal */}
      {showFileTypeModal && (
        <div className="modal-overlay" onClick={() => setShowFileTypeModal(false)}>
          <div className="instrument-modal" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📦 Files by Type</h3>
              <button className="close-btn" onClick={() => setShowFileTypeModal(false)}>✕</button>
            </div>
            <div className="modal-content">
              <div className="total-hours">
                Total: <strong>{formatNumber(stats?.total_files)} files</strong>
              </div>
              <div className="file-type-list">
                {stats?.file_type_breakdown && Object.entries(stats.file_type_breakdown)
                  .sort(([, a], [, b]) => b.count - a.count)
                  .map(([type, data]) => (
                    <div key={type} className="file-type-row">
                      <span className="file-type-icon">
                        {type === 'audio' ? '🎵' : type === 'latent' ? '🧠' : type === 'midi' ? '🎹' : '⚙️'}
                      </span>
                      <span className="file-type-name">{data.label}</span>
                      <span className="file-type-count">{formatNumber(data.count)}</span>
                      <span className="file-type-size">{data.size_formatted || formatBytes(data.size)}</span>
                    </div>
                  ))}
              </div>
              <div className="file-type-note">
                <small>Conditioning count = raw files / 6 (6 conditioning files per audio file)</small>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DataMonitor;
