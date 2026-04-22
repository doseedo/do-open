import React, { useState } from 'react';
import * as sessionService from '../../services/sessionService';
import * as sessionAPI from '../../services/sessionAPI';
import * as sessionExportService from '../../services/sessionExportService';
import styles from './Projects.module.css';
import modalStyles from '../MySessions/MySessions.module.css';

/**
 * ProjectCard Component
 * Displays individual project as a Spotify-style row
 */
const ProjectCard = ({ projectName, index, onLoad, onDelete, onRename }) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editedName, setEditedName] = useState(projectName);
  const [isHovered, setIsHovered] = useState(false);
  const [showMenu, setShowMenu] = useState(false);
  const [showUploadModal, setShowUploadModal] = useState(false);

  // Load session metadata for display
  const sessionData = sessionService.loadSession(projectName);
  const lastModified = sessionData?.timestamp
    ? new Date(sessionData.timestamp).toLocaleDateString()
    : 'Unknown';

  const trackCount = sessionData?.state?.buses?.reduce((count, bus) => {
    return count + (bus.tracks?.length || 0);
  }, 0) || 0;

  // Generate random duration for placeholder
  const duration = '3:24';

  const handleEditClick = (e) => {
    e.stopPropagation();
    setIsEditing(true);
    setShowMenu(false);
  };

  const handleSaveEdit = () => {
    if (editedName.trim() && editedName !== projectName) {
      onRename(projectName, editedName.trim());
    }
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setEditedName(projectName);
    setIsEditing(false);
  };

  const handleDeleteClick = (e) => {
    e.stopPropagation();
    onDelete(projectName);
    setShowMenu(false);
  };

  const handlePlayClick = (e) => {
    e.stopPropagation();
    if (!isEditing) {
      onLoad(projectName);
    }
  };

  const handleRowClick = () => {
    if (!isEditing) {
      onLoad(projectName);
    }
  };

  const handleMenuClick = (e) => {
    e.stopPropagation();
    setShowMenu(!showMenu);
  };

  const handlePostClick = (e) => {
    e.stopPropagation();
    setShowUploadModal(true);
  };

  return (
    <div
      className={styles.sessionRow}
      onClick={handleRowClick}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Index / Play Button */}
      <div className={styles.sessionNumber}>
        {isHovered ? (
          <button className={styles.playBtn} onClick={handlePlayClick}>
            <i className="fa-solid fa-play"></i>
          </button>
        ) : (
          <span>{index}</span>
        )}
      </div>

      {/* Title with Waveform */}
      <div className={styles.sessionTitle}>
        <div className={styles.waveformPreview}>
          {/* Placeholder waveform bars */}
          {[...Array(40)].map((_, i) => (
            <div
              key={i}
              className={styles.waveformBar}
              style={{
                height: `${Math.random() * 60 + 20}%`,
                opacity: 0.6
              }}
            />
          ))}
        </div>
        {isEditing ? (
          <input
            type="text"
            value={editedName}
            onChange={(e) => setEditedName(e.target.value)}
            onBlur={handleSaveEdit}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleSaveEdit();
              if (e.key === 'Escape') handleCancelEdit();
            }}
            onClick={(e) => e.stopPropagation()}
            className={styles.sessionNameInput}
            autoFocus
          />
        ) : (
          <span className={styles.sessionName}>{projectName}</span>
        )}
      </div>

      {/* Track Count */}
      <div className={styles.sessionTracks}>{trackCount}</div>

      {/* Date Modified */}
      <div className={styles.sessionDate}>{lastModified}</div>

      {/* Duration with Post Button */}
      <div className={styles.sessionDuration}>
        {duration}
        <button
          className={styles.postBtn}
          onClick={handlePostClick}
          title="Post Session"
        >
          <i className="fa-solid fa-cloud-arrow-up"></i>
        </button>
      </div>

      {/* Menu Button */}
      <div className={styles.sessionMenu}>
        <button className={styles.menuBtn} onClick={handleMenuClick}>
          <i className="fa-solid fa-ellipsis"></i>
        </button>

        {/* Dropdown Menu */}
        {showMenu && (
          <div className={styles.menuDropdown}>
            <button onClick={handleEditClick}>
              <i className="fa-solid fa-pen"></i>
              Rename
            </button>
            <button onClick={handleDeleteClick} className={styles.deleteOption}>
              <i className="fa-solid fa-trash"></i>
              Delete
            </button>
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUploadModal && (
        <UploadModal
          sessionName={projectName}
          onClose={() => setShowUploadModal(false)}
        />
      )}
    </div>
  );
};

/**
 * Upload Modal Component
 * Auto-uploads session data and files to GCS
 */
const UploadModal = ({ sessionName, onClose }) => {
  const [formData, setFormData] = useState({
    name: sessionName || '',
    description: '',
    type: 'project',
    isPublic: false
  });
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [sessionInfo, setSessionInfo] = useState(null);

  // Load session data to show info
  React.useEffect(() => {
    const session = sessionService.loadSession(sessionName);
    if (session) {
      const sizeInfo = sessionExportService.calculateSessionSize(session.state);
      setSessionInfo(sizeInfo);
    }
  }, [sessionName]);

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      setUploading(true);
      setUploadProgress(5);

      // Load the complete session data
      console.log('Loading session data...');
      const session = sessionService.loadSession(sessionName);
      if (!session) {
        throw new Error('Session not found');
      }
      setUploadProgress(10);

      // Export session to GCS
      const exportResult = await sessionExportService.exportSessionToGCS(
        session.state,
        formData.name || sessionName
      );
      setUploadProgress(70);

      // Create session record in backend
      console.log('Creating session record...');
      const sessionData = {
        name: formData.name || sessionName,
        description: formData.description,
        type: formData.type,
        is_public: formData.isPublic,
        session_id: exportResult.sessionId,
        gcs_base_path: exportResult.basePath,
        metadata: {
          ...exportResult.metadata,
          fileCount: exportResult.files.length,
          files: exportResult.files.map(f => ({
            path: f.path,
            gcs_url: f.gcsUrl,
            file_name: f.fileName,
            file_size: f.fileSize,
            type: f.type
          }))
        }
      };

      await sessionAPI.createSession(sessionData);
      setUploadProgress(100);

      alert(`Session uploaded successfully!\n\n${exportResult.files.length} files uploaded\nSession ID: ${exportResult.sessionId}`);
      onClose();
    } catch (error) {
      // Only log in development mode
      if (process.env.NODE_ENV === 'development') {
        console.warn('Upload error:', error.message);
      }
      alert('Upload failed: ' + error.message);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  return (
    <div className={modalStyles.modalOverlay} onClick={onClose}>
      <div className={modalStyles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={modalStyles.modalHeader}>
          <h2>Upload Session</h2>
          <button className={modalStyles.closeButton} onClick={onClose}>
            <i className="fa-solid fa-times"></i>
          </button>
        </div>

        <form onSubmit={handleSubmit} className={modalStyles.uploadForm}>
          <div className={modalStyles.formGroup}>
            <label>Session Name *</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
              placeholder="My Awesome Session"
            />
          </div>

          <div className={modalStyles.formGroup}>
            <label>Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Describe your session..."
              rows="3"
            />
          </div>

          <div className={modalStyles.formGroup}>
            <label>Type *</label>
            <select
              value={formData.type}
              onChange={(e) => setFormData({ ...formData, type: e.target.value })}
              required
            >
              <option value="project">Project</option>
              <option value="loop">Loop</option>
              <option value="preset">Preset</option>
              <option value="midi">MIDI</option>
            </select>
          </div>

          <div className={modalStyles.formGroup}>
            <label className={modalStyles.checkboxLabel}>
              <input
                type="checkbox"
                checked={formData.isPublic}
                onChange={(e) => setFormData({ ...formData, isPublic: e.target.checked })}
              />
              <span>Make this session public</span>
            </label>
          </div>

          {/* Session Info */}
          {sessionInfo && (
            <div className={modalStyles.formGroup}>
              <label>Session Contents</label>
              <div className={modalStyles.sessionInfoBox}>
                <div className={modalStyles.sessionInfoItem}>
                  <i className="fa-solid fa-music"></i>
                  <span>{sessionInfo.audioTracks} Audio Tracks</span>
                </div>
                <div className={modalStyles.sessionInfoItem}>
                  <i className="fa-solid fa-piano"></i>
                  <span>{sessionInfo.midiTracks} MIDI Tracks</span>
                </div>
                {sessionInfo.hasVideo && (
                  <div className={modalStyles.sessionInfoItem}>
                    <i className="fa-solid fa-video"></i>
                    <span>Video Included</span>
                  </div>
                )}
                <div className={modalStyles.sessionInfoItem}>
                  <i className="fa-solid fa-database"></i>
                  <span>~{sessionInfo.totalMB} MB</span>
                </div>
              </div>
              <p className={modalStyles.infoText}>
                <i className="fa-solid fa-info-circle"></i>
                All session data, tracks, and files will be automatically uploaded
              </p>
            </div>
          )}

          {uploading && (
            <div className={modalStyles.progressBar}>
              <div
                className={modalStyles.progressFill}
                style={{ width: `${uploadProgress}%` }}
              ></div>
              <span className={modalStyles.progressText}>{uploadProgress}%</span>
            </div>
          )}

          <div className={modalStyles.modalActions}>
            <button
              type="button"
              className={modalStyles.cancelButton}
              onClick={onClose}
              disabled={uploading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className={modalStyles.submitButton}
              disabled={uploading}
            >
              {uploading ? (
                <>
                  <i className="fa-solid fa-spinner fa-spin"></i>
                  Uploading...
                </>
              ) : (
                <>
                  <i className="fa-solid fa-cloud-arrow-up"></i>
                  Upload
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ProjectCard;
