import React, { useState, useEffect } from 'react';
import styles from './MySessions.module.css';
import * as sessionAPI from '../../services/sessionAPI';
import * as r2UploadService from '../../services/r2UploadService';

/**
 * My Sessions Component
 * Displays and manages user's uploaded sessions
 */
const MySessions = () => {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [showUploadModal, setShowUploadModal] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Load user sessions
  useEffect(() => {
    loadSessions();
  }, [selectedCategory]);

  const loadSessions = async () => {
    try {
      setLoading(true);
      const filters = selectedCategory !== 'all' ? { type: selectedCategory } : {};
      const data = await sessionAPI.getUserSessions(filters);
      setSessions(data.sessions || []);
    } catch (error) {
      // Silently fail - user sessions feature is optional and backend may not be available
      // Only log in development mode
      if (process.env.NODE_ENV === 'development') {
        console.warn('Sessions API not available:', error.message);
      }
      setSessions([]);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSession = async (sessionId) => {
    if (!window.confirm('Are you sure you want to delete this session?')) return;

    try {
      await sessionAPI.deleteSession(sessionId);
      setSessions(sessions.filter(s => s.id !== sessionId));
    } catch (error) {
      console.error('Failed to delete session:', error);
      alert('Failed to delete session: ' + error.message);
    }
  };

  const handleUploadSession = () => {
    setShowUploadModal(true);
  };

  const categories = [
    { id: 'all', label: 'All', icon: 'fa-grid-2' },
    { id: 'project', label: 'Projects', icon: 'fa-folder' },
    { id: 'loop', label: 'Loops', icon: 'fa-repeat' },
    { id: 'preset', label: 'Presets', icon: 'fa-sliders' },
    { id: 'midi', label: 'MIDI', icon: 'fa-music' }
  ];

  return (
    <div className={styles.sessionsContainer}>
      <div className={styles.sessionsHeader}>
        <div>
          <h1 className={styles.sessionsTitle}>My Sessions</h1>
          <p className={styles.sessionsSubtitle}>Manage your uploaded content and projects</p>
        </div>
        <button className={styles.uploadButton} onClick={handleUploadSession}>
          <i className="fa-solid fa-cloud-arrow-up"></i>
          Upload Session
        </button>
      </div>

      {/* Category Filters */}
      <div className={styles.categoriesSection}>
        {categories.map((category) => (
          <button
            key={category.id}
            className={`${styles.categoryBox} ${selectedCategory === category.id ? styles.active : ''}`}
            onClick={() => setSelectedCategory(category.id)}
          >
            <i className={`fa-solid ${category.icon}`}></i>
            <span>{category.label}</span>
          </button>
        ))}
      </div>

      {/* Sessions Grid */}
      {loading ? (
        <div className={styles.loadingState}>
          <i className="fa-solid fa-spinner fa-spin"></i>
          <p>Loading sessions...</p>
        </div>
      ) : sessions.length === 0 ? (
        <div className={styles.emptyState}>
          <i className="fa-solid fa-folder-open"></i>
          <h3>No sessions yet</h3>
          <p>Upload your first session to get started</p>
          <button className={styles.uploadButtonLarge} onClick={handleUploadSession}>
            <i className="fa-solid fa-cloud-arrow-up"></i>
            Upload Session
          </button>
        </div>
      ) : (
        <div className={styles.sessionsGrid}>
          {sessions.map((session) => (
            <div key={session.id} className={styles.sessionCard}>
              <div className={styles.sessionThumbnail}>
                {session.thumbnail_url ? (
                  <img src={session.thumbnail_url} alt={session.name} />
                ) : (
                  <i className={`fa-solid ${
                    session.type === 'project' ? 'fa-folder' :
                    session.type === 'loop' ? 'fa-repeat' :
                    session.type === 'preset' ? 'fa-sliders' :
                    'fa-music'
                  }`}></i>
                )}
              </div>
              <div className={styles.sessionInfo}>
                <h3 className={styles.sessionName}>{session.name}</h3>
                <p className={styles.sessionMeta}>
                  <span className={styles.sessionType}>{session.type?.toUpperCase()}</span>
                  <span className={styles.sessionDate}>
                    {new Date(session.created_at).toLocaleDateString()}
                  </span>
                </p>
                {session.description && (
                  <p className={styles.sessionDescription}>{session.description}</p>
                )}
              </div>
              <div className={styles.sessionActions}>
                <button className={styles.actionButton} title="Download">
                  <i className="fa-solid fa-download"></i>
                </button>
                <button className={styles.actionButton} title="Share">
                  <i className="fa-solid fa-share"></i>
                </button>
                <button
                  className={`${styles.actionButton} ${styles.deleteButton}`}
                  onClick={() => handleDeleteSession(session.id)}
                  title="Delete"
                >
                  <i className="fa-solid fa-trash"></i>
                </button>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Upload Modal */}
      {showUploadModal && (
        <UploadModal
          onClose={() => setShowUploadModal(false)}
          onUploadComplete={loadSessions}
        />
      )}
    </div>
  );
};

/**
 * Upload Modal Component
 */
const UploadModal = ({ onClose, onUploadComplete }) => {
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    type: 'project',
    isPublic: false
  });
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (e) => {
    setFiles(Array.from(e.target.files));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (files.length === 0) {
      alert('Please select at least one file');
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(10);

      // Upload files to GCS
      const uploadResults = await r2UploadService.uploadMultipleToR2(
        files,
        formData.type,
        { sessionName: formData.name }
      );
      setUploadProgress(60);

      // Create session with uploaded file URLs
      console.log('Creating session...');
      const sessionData = {
        name: formData.name,
        description: formData.description,
        type: formData.type,
        is_public: formData.isPublic,
        files: uploadResults.map(result => ({
          gcs_url: result.gcsUrl,
          file_name: result.fileName,
          file_size: result.fileSize
        }))
      };

      await sessionAPI.createSession(sessionData);
      setUploadProgress(100);

      alert('Session uploaded successfully!');
      onUploadComplete();
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
    <div className={styles.modalOverlay} onClick={onClose}>
      <div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
        <div className={styles.modalHeader}>
          <h2>Upload Session</h2>
          <button className={styles.closeButton} onClick={onClose}>
            <i className="fa-solid fa-times"></i>
          </button>
        </div>

        <form onSubmit={handleSubmit} className={styles.uploadForm}>
          <div className={styles.formGroup}>
            <label>Session Name *</label>
            <input
              type="text"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              required
              placeholder="My Awesome Session"
            />
          </div>

          <div className={styles.formGroup}>
            <label>Description</label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              placeholder="Describe your session..."
              rows="3"
            />
          </div>

          <div className={styles.formGroup}>
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

          <div className={styles.formGroup}>
            <label className={styles.checkboxLabel}>
              <input
                type="checkbox"
                checked={formData.isPublic}
                onChange={(e) => setFormData({ ...formData, isPublic: e.target.checked })}
              />
              <span>Make this session public</span>
            </label>
          </div>

          <div className={styles.formGroup}>
            <label>Files *</label>
            <div className={styles.fileUploadArea}>
              <input
                type="file"
                multiple
                onChange={handleFileChange}
                id="fileInput"
                className={styles.fileInput}
              />
              <label htmlFor="fileInput" className={styles.fileInputLabel}>
                <i className="fa-solid fa-cloud-arrow-up"></i>
                <span>
                  {files.length === 0
                    ? 'Click to select files or drag and drop'
                    : `${files.length} file(s) selected`}
                </span>
              </label>
            </div>
            {files.length > 0 && (
              <div className={styles.fileList}>
                {files.map((file, index) => (
                  <div key={index} className={styles.fileItem}>
                    <i className="fa-solid fa-file"></i>
                    <span>{file.name}</span>
                    <span className={styles.fileSize}>
                      ({(file.size / 1024 / 1024).toFixed(2)} MB)
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {uploading && (
            <div className={styles.progressBar}>
              <div
                className={styles.progressFill}
                style={{ width: `${uploadProgress}%` }}
              ></div>
              <span className={styles.progressText}>{uploadProgress}%</span>
            </div>
          )}

          <div className={styles.modalActions}>
            <button
              type="button"
              className={styles.cancelButton}
              onClick={onClose}
              disabled={uploading}
            >
              Cancel
            </button>
            <button
              type="submit"
              className={styles.submitButton}
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

export default MySessions;
