import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';

/**
 * StemsSidebar Component - Right sidebar with track info and controls
 * Matches original doseedo2.html .stems-sidebar structure
 */
function StemsSidebar() {
  const { state, dispatch } = useApp();
  const [isCollapsed, setIsCollapsed] = useState(true);

  const selectedTrack = state.selectedTrack || null;
  const hasTrackSelected = !!selectedTrack;

  const toggleSidebar = () => {
    setIsCollapsed(!isCollapsed);
    dispatch({
      type: 'TOGGLE_STEMS_SIDEBAR'
    });
  };

  const handleDownloadTrack = () => {
    if (!selectedTrack) return;
    dispatch({
      type: 'DOWNLOAD_TRACK',
      payload: { trackId: selectedTrack.id }
    });
  };

  const handleRegenerateTrack = () => {
    if (!selectedTrack) return;
    dispatch({
      type: 'REGENERATE_TRACK',
      payload: { trackId: selectedTrack.id }
    });
  };

  const handleSeparateStems = () => {
    if (!selectedTrack) return;
    dispatch({
      type: 'SEPARATE_STEMS',
      payload: { trackId: selectedTrack.id }
    });
  };

  return (
    <div id="stems-sidebar" className={`stems-sidebar ${isCollapsed ? 'collapsed' : ''}`}>
      <div className="stems-toggle" onClick={toggleSidebar}>
        <i className="fa-solid fa-chevron-left stems-toggle-icon"></i>
      </div>

      <div className="stems-content">
        <div className="stems-header">
          <h3>Track Info</h3>
          <button className="stems-close" onClick={toggleSidebar}>
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>

        {/* Track Info Display */}
        <div id="track-info-display" className="track-info-display">
          {!hasTrackSelected ? (
            <div className="track-info-empty">
              <i className="fa-solid fa-music" style={{ fontSize: '32px', color: '#666', marginBottom: '10px' }}></i>
              <p>No track selected</p>
              <p style={{ fontSize: '12px', color: '#888' }}>Select a track to view its info</p>
            </div>
          ) : (
            <div id="track-info-content" className="track-info-content">
              {/* Instrument Info */}
              <div className="track-info-section">
                <h4><i className="fa-solid fa-guitar"></i> Instrument</h4>
                <div className="track-info-item">
                  <span className="track-info-label">Group:</span>
                  <span className="track-info-value" id="track-instrument-group">
                    {selectedTrack.instrumentGroup || '-'}
                  </span>
                </div>
                <div className="track-info-item">
                  <span className="track-info-label">Subgroup:</span>
                  <span className="track-info-value" id="track-instrument-subgroup">
                    {selectedTrack.instrumentSubgroup || '-'}
                  </span>
                </div>
              </div>

              {/* Source File */}
              <div className="track-info-section">
                <h4><i className="fa-solid fa-file-audio"></i> Source File</h4>
                <div className="track-info-item">
                  <span className="track-info-value" id="track-source-file" style={{ wordBreak: 'break-all', fontSize: '11px' }}>
                    {selectedTrack.sourceFile || '-'}
                  </span>
                </div>
              </div>

              {/* Download Button */}
              <div className="track-info-section">
                <button className="track-download-btn" onClick={handleDownloadTrack}>
                  <i className="fa-solid fa-download"></i> Download Track
                </button>
              </div>

              {/* Regenerate Button */}
              {selectedTrack.canRegenerate && (
                <div className="track-info-section" id="track-regenerate-section">
                  <button className="track-regenerate-btn" onClick={handleRegenerateTrack}>
                    <i className="fa-solid fa-rotate"></i> Regenerate
                  </button>
                </div>
              )}

              {/* Stem Separation */}
              {selectedTrack.canSeparateStems && (
                <div className="track-info-section" id="track-stem-separation-section">
                  <h4><i className="fa-solid fa-layer-group"></i> Stem Separation</h4>
                  <button className="track-stem-btn" onClick={handleSeparateStems}>
                    <i className="fa-solid fa-microphone"></i> Remove Music
                  </button>
                </div>
              )}

              {/* Version Control */}
              {selectedTrack.versions && selectedTrack.versions.length > 1 && (
                <div className="track-info-section" id="track-version-section">
                  <div className="track-version-control">
                    <button className="track-version-btn" title="Previous version">
                      <i className="fa-solid fa-chevron-left"></i>
                    </button>
                    <span className="track-version-label" id="track-version-label">
                      Version {selectedTrack.currentVersion || 1} of {selectedTrack.versions.length}
                    </span>
                    <button className="track-version-btn" title="Next version">
                      <i className="fa-solid fa-chevron-right"></i>
                    </button>
                  </div>
                </div>
              )}

              {/* Track Settings */}
              <div className="track-info-section">
                <button className="track-settings-toggle">
                  <i className="fa-solid fa-cog"></i> Generation Settings
                  <i className="fa-solid fa-chevron-down track-settings-icon"></i>
                </button>
                <div id="track-settings-content" className="track-settings-content">
                  {/* Settings will be populated dynamically */}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Upload Section */}
        <div className="stems-upload-section">
          <input
            type="file"
            id="stems-audio-input"
            accept="audio/*"
            style={{ display: 'none' }}
          />
          <button
            className="stems-upload-btn"
            onClick={() => document.getElementById('stems-audio-input')?.click()}
          >
            <i className="fa-solid fa-cloud-arrow-up"></i>
            <span>Upload Audio to Separate</span>
          </button>
        </div>

        {/* Stems Container */}
        <div id="stems-container" className="stems-list">
          <div className="stems-empty">
            <i className="fa-solid fa-layer-group" style={{ fontSize: '48px', color: '#666', marginBottom: '10px' }}></i>
            <p>No stems loaded</p>
            <p style={{ fontSize: '12px', color: '#888' }}>Upload audio above to separate into stems</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default StemsSidebar;
