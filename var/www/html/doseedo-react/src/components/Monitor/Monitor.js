import React, { useState, useEffect, useCallback, useRef } from 'react';
import './Monitor.css';

/**
 * Decrypt AES-GCM encrypted response using Web Crypto API
 */
async function decryptResponse(encryptedData, keyBase64) {
  try {
    const keyBytes = Uint8Array.from(atob(keyBase64), c => c.charCodeAt(0));
    const nonce = Uint8Array.from(atob(encryptedData.nonce), c => c.charCodeAt(0));
    const ciphertext = Uint8Array.from(atob(encryptedData.data), c => c.charCodeAt(0));

    const cryptoKey = await window.crypto.subtle.importKey(
      'raw',
      keyBytes,
      { name: 'AES-GCM' },
      false,
      ['decrypt']
    );

    const decrypted = await window.crypto.subtle.decrypt(
      { name: 'AES-GCM', iv: nonce },
      cryptoKey,
      ciphertext
    );

    const decoder = new TextDecoder();
    return JSON.parse(decoder.decode(decrypted));
  } catch (err) {
    console.error('Decryption failed:', err);
    throw new Error('Failed to decrypt response');
  }
}

/**
 * Data Monitor Component
 * GCS Bucket statistics and data overview
 */
const Monitor = () => {
  const [stats, setStats] = useState(null);
  const [formats, setFormats] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);
  const decryptKeyRef = useRef(null);

  // Fetch decrypt key once
  const fetchDecryptKey = useCallback(async () => {
    if (decryptKeyRef.current) return decryptKeyRef.current;
    const response = await fetch('/api/monitor/decrypt-key');
    if (!response.ok) {
      // Backend not deployed — caller will surface a friendly message
      throw new Error(`Monitor backend offline (HTTP ${response.status})`);
    }
    // Some 404 fallbacks return HTML; detect that
    const ct = response.headers.get('content-type') || '';
    if (!ct.includes('application/json')) {
      throw new Error('Monitor backend offline');
    }
    const data = await response.json();
    decryptKeyRef.current = data.key;
    return data.key;
  }, []);

  // Fetch stats from API
  const fetchStats = useCallback(async () => {
    try {
      // Get decrypt key first
      const decryptKey = await fetchDecryptKey();

      const response = await fetch('/api/monitor/stats');
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();

      // Decrypt if encrypted
      let statsData;
      if (data.encrypted) {
        statsData = await decryptResponse(data, decryptKey);
      } else {
        statsData = data;
      }

      setStats(statsData);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      console.error('Stats fetch error:', err);
      setError(`Failed to fetch stats: ${err.message}`);
    } finally {
      setLoading(false);
    }
  }, [fetchDecryptKey]);

  // Fetch formats data
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

  // Initial fetch and auto-refresh
  useEffect(() => {
    fetchStats();
    fetchFormats();
  }, [fetchStats, fetchFormats]);

  useEffect(() => {
    let interval;
    if (autoRefresh) {
      interval = setInterval(() => {
        fetchStats();
        fetchFormats();
      }, 30000); // 30 seconds
    }
    return () => clearInterval(interval);
  }, [autoRefresh, fetchStats, fetchFormats]);

  // Format time for display
  const formatTime = (date) => {
    if (!date) return '';
    return date.toLocaleTimeString();
  };

  // Format size for display
  const formatSize = (bytes) => {
    if (!bytes || bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let size = bytes;
    let unitIndex = 0;
    while (size >= 1024 && unitIndex < units.length - 1) {
      size /= 1024;
      unitIndex++;
    }
    return `${size.toFixed(1)} ${units[unitIndex]}`;
  };

  return (
    <div className="data-monitor">
      {/* Header */}
      <div className="monitor-header">
        <h1>Data Monitor</h1>
        <p className="subtitle">GCS Bucket Statistics &amp; Data Overview</p>
        <div className="header-controls">
          <button className="refresh-btn" onClick={fetchStats} disabled={loading}>
            🔄 Refresh
          </button>
          <label className="auto-refresh-toggle">
            <input
              type="checkbox"
              checked={autoRefresh}
              onChange={(e) => setAutoRefresh(e.target.checked)}
            />
            Auto-refresh (30s)
          </label>
          <span className="last-updated">
            Last updated: {formatTime(lastUpdated)}
          </span>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="error-message">
          <span>⚠️ {error}</span>
        </div>
      )}

      {/* Loading state */}
      {loading && !stats && (
        <div className="loading-state">
          <span>Loading statistics...</span>
        </div>
      )}

      {/* No cache warning */}
      {stats?.status === 'no_cache' && (
        <div className="warning-message">
          <span>⚠️ No cache found. Run: python3 gcs_cache_builder.py</span>
        </div>
      )}

      {/* Summary Cards */}
      <div className="summary-section">
        <div className="summary-card total">
          <div className="card-icon">📦</div>
          <div className="card-content">
            <div className="card-value">{stats?.total_files?.toLocaleString() || 0}</div>
            <div className="card-label">Total Files</div>
          </div>
        </div>

        <div className="summary-card size">
          <div className="card-icon">💾</div>
          <div className="card-content">
            <div className="card-value">{stats?.total_size_formatted || '0 B'}</div>
            <div className="card-label">Total Size</div>
          </div>
        </div>

        <div className="summary-card folders">
          <div className="card-icon">📁</div>
          <div className="card-content">
            <div className="card-value">{stats?.folder_count?.toLocaleString() || 0}</div>
            <div className="card-label">Folders</div>
          </div>
        </div>

        <div className="summary-card recent">
          <div className="card-icon">📅</div>
          <div className="card-content">
            <div className="card-value">{stats?.modified_today?.toLocaleString() || 0}</div>
            <div className="card-label">Modified Today</div>
          </div>
        </div>
      </div>

      {/* Formats Coverage Section */}
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
                <span>{formats.with_latent?.toLocaleString()} have</span>
                <span className="needs">{formats.needs_latent?.toLocaleString()} need</span>
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
                <span>{formats.with_conditioning?.toLocaleString()} have</span>
                <span className="needs">{formats.needs_conditioning?.toLocaleString()} need</span>
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
                <span>{formats.with_midi?.toLocaleString()} have</span>
                <span className="needs">{formats.needs_midi?.toLocaleString()} need</span>
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
                  <span className="completeness-count">{item.count?.toLocaleString()}</span>
                  <span className="completeness-pct">{item.pct}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Folders Section */}
      <div className="folders-section">
        <h2>Data Folders</h2>
        <div className="folder-grid">
          {stats?.folders?.map((folder) => (
            <div key={folder.id} className="folder-card">
              <div className="folder-icon">{folder.icon || '📂'}</div>
              <div className="folder-name">{folder.name}</div>
              <div className="folder-stats">
                <div className="stat">
                  <span className="stat-value">{folder.files?.toLocaleString() || 0}</span>
                  <span className="stat-label">files</span>
                </div>
                <div className="stat">
                  <span className="stat-value">{folder.size_formatted || '0 B'}</span>
                  <span className="stat-label">size</span>
                </div>
              </div>
            </div>
          )) || (
            <>
              <FolderPlaceholder icon="🎛️" name="ProTools Sessions" />
              <FolderPlaceholder icon="🎚️" name="ProTools A" />
              <FolderPlaceholder icon="🥁" name="Drum Bus" />
              <FolderPlaceholder icon="🎹" name="Drum MIDI" />
              <FolderPlaceholder icon="🎵" name="Basic Pitch" />
              <FolderPlaceholder icon="📊" name="Mel Spectrograms" />
            </>
          )}
        </div>
      </div>

      {/* Instructions */}
      <div className="instructions">
        <h4>Data Monitor Info</h4>
        <ul>
          <li>Click on a folder card to view detailed breakdown</li>
          <li>Enable auto-refresh to keep stats updated</li>
          <li>Data is pulled from the GCS bucket mount at /home/arlo/gcs-bucket</li>
          <li>Run <code>python3 gcs_cache_builder.py</code> to rebuild cache</li>
        </ul>
      </div>

      {/* Cache info */}
      {stats?.cache_age && (
        <div className="cache-info">
          Cache age: {Math.round(stats.cache_age / 60)} minutes
        </div>
      )}
    </div>
  );
};

/**
 * Folder placeholder component for when no data is loaded
 */
const FolderPlaceholder = ({ icon, name }) => (
  <div className="folder-card placeholder">
    <div className="folder-icon">{icon}</div>
    <div className="folder-name">{name}</div>
    <div className="folder-stats">
      <div className="stat">
        <span className="stat-value">0</span>
        <span className="stat-label">files</span>
      </div>
      <div className="stat">
        <span className="stat-value">0 B</span>
        <span className="stat-label">size</span>
      </div>
    </div>
  </div>
);

export default Monitor;
