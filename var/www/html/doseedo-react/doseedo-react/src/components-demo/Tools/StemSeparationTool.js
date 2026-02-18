import React, { useState, useRef, useCallback } from 'react';
import { separateStems } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * Stem Separation Tool
 * Separate audio into individual stems (vocals, drums, bass, other)
 */
const StemSeparationTool = ({ tool }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [stems, setStems] = useState(null);
  const [selectedStem, setSelectedStem] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const fileInputRef = useRef(null);

  const stemTypes = [
    { id: 'vocals', label: 'Vocals', icon: 'fa-microphone', color: '#ef4444' },
    { id: 'drums', label: 'Drums', icon: 'fa-drum', color: '#f59e0b' },
    { id: 'bass', label: 'Bass', icon: 'fa-guitar', color: '#10b981' },
    { id: 'other', label: 'Other', icon: 'fa-music', color: '#667eea' },
    { id: 'accompaniment', label: 'Instrumental', icon: 'fa-sliders', color: '#9c82c8' }
  ];

  // Handle file upload
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setUploadedUrl(url);
      setStems(null);
      setSelectedStem(null);
      setStatusMessage(`Loaded: ${file.name}`);
    }
  }, []);

  // Handle drag and drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('audio/')) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setUploadedUrl(url);
      setStems(null);
      setSelectedStem(null);
      setStatusMessage(`Loaded: ${file.name}`);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  // Separate stems
  const handleSeparate = useCallback(async () => {
    if (!uploadedFile) {
      setStatusMessage('Please upload an audio file first.');
      return;
    }

    setIsProcessing(true);
    setStatusMessage('Separating stems... This may take a few minutes.');
    setStems(null);

    try {
      // For separation, we need to upload the file first or use a URL
      // The backend expects an audio URL, so we'll create a temporary upload
      const formData = new FormData();
      formData.append('audioFile', uploadedFile);

      // First, upload the file to get a URL
      const uploadResponse = await fetch('/api/upload-audio', {
        method: 'POST',
        body: formData
      });

      let audioUrl;
      if (uploadResponse.ok) {
        const uploadResult = await uploadResponse.json();
        audioUrl = uploadResult.url;
      } else {
        // Fallback: try using the blob URL directly (may not work with all backends)
        audioUrl = uploadedUrl;
      }

      // Call stem separation endpoint
      const result = await separateStems(audioUrl);

      if (result.stems) {
        setStems(result.stems);
        setStatusMessage('Separation complete! Click a stem to preview.');
      } else {
        setStatusMessage('Separation completed but no stems returned.');
      }

    } catch (error) {
      console.error('Stem separation failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
    }
  }, [uploadedFile, uploadedUrl]);

  // Download a stem
  const handleDownloadStem = useCallback((stemType) => {
    if (stems && stems[stemType]) {
      const a = document.createElement('a');
      a.href = stems[stemType];
      a.download = `${uploadedFile?.name?.replace(/\.[^/.]+$/, '') || 'audio'}_${stemType}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [stems, uploadedFile]);

  // Download all stems as zip (placeholder - would need backend support)
  const handleDownloadAll = useCallback(() => {
    if (!stems) return;

    // Download each stem individually
    Object.keys(stems).forEach(stemType => {
      setTimeout(() => {
        handleDownloadStem(stemType);
      }, 500 * Object.keys(stems).indexOf(stemType));
    });
  }, [stems, handleDownloadStem]);

  // Select a stem for preview
  const handleSelectStem = useCallback((stemType) => {
    setSelectedStem(stemType);
  }, []);

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Tool Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div className={styles.toolGeneratorIcon} style={{ background: 'linear-gradient(135deg, rgba(156, 130, 200, 0.4), rgba(156, 130, 200, 0.2))' }}>
            <i className="fa-solid fa-layer-group" style={{ color: '#9c82c8' }}></i>
          </div>
          <div className={styles.toolGeneratorTitleText}>
            <h2 className={styles.toolGeneratorTitle}>{tool.name}</h2>
            <p className={styles.toolGeneratorDescription}>{tool.description}</p>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-upload"></i>
          Upload Audio
        </label>
        <div
          className={styles.toolFileUpload}
          onClick={() => fileInputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="audio/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          {uploadedFile ? (
            <div className={styles.uploadedFileInfo}>
              <i className="fa-solid fa-file-audio"></i>
              <span>{uploadedFile.name}</span>
              <button
                className={styles.removeFileBtn}
                onClick={(e) => {
                  e.stopPropagation();
                  setUploadedFile(null);
                  setUploadedUrl(null);
                  setStems(null);
                  setSelectedStem(null);
                }}
              >
                <i className="fa-solid fa-xmark"></i>
              </button>
            </div>
          ) : (
            <>
              <i className="fa-solid fa-cloud-arrow-up"></i>
              <span>Click or drag audio file here</span>
              <span className={styles.uploadHint}>MP3, WAV, FLAC supported</span>
            </>
          )}
        </div>
      </div>

      {/* Original Audio Waveform */}
      {uploadedUrl && (
        <div className={styles.toolWaveformSection}>
          <div className={styles.toolWaveformHeader}>
            <span className={styles.toolWaveformLabel}>
              <i className="fa-solid fa-waveform-lines"></i>
              Original Audio
            </span>
          </div>
          <ToolWaveform
            audioUrl={uploadedUrl}
            height={80}
            color="#666"
          />
        </div>
      )}

      {/* Stems Grid */}
      {stems && (
        <div className={styles.toolSection}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-layer-group"></i>
            Separated Stems
          </label>
          <div className={styles.stemsGrid}>
            {stemTypes.map(stem => {
              const hasStems = stems[stem.id];
              return (
                <button
                  key={stem.id}
                  className={`${styles.stemCard} ${selectedStem === stem.id ? styles.active : ''} ${!hasStems ? styles.disabled : ''}`}
                  onClick={() => hasStems && handleSelectStem(stem.id)}
                  disabled={!hasStems}
                >
                  <i className={`fa-solid ${stem.icon}`} style={{ color: stem.color }}></i>
                  <span>{stem.label}</span>
                  {hasStems && (
                    <button
                      className={styles.stemDownloadBtn}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDownloadStem(stem.id);
                      }}
                    >
                      <i className="fa-solid fa-download"></i>
                    </button>
                  )}
                </button>
              );
            })}
          </div>
        </div>
      )}

      {/* Selected Stem Waveform */}
      {selectedStem && stems && stems[selectedStem] && (
        <div className={styles.toolWaveformSection}>
          <div className={styles.toolWaveformHeader}>
            <span className={styles.toolWaveformLabel}>
              <i className={`fa-solid ${stemTypes.find(s => s.id === selectedStem)?.icon}`}></i>
              {stemTypes.find(s => s.id === selectedStem)?.label} Stem
            </span>
            <div className={styles.toolWaveformActions}>
              <button
                className={styles.toolActionBtn}
                onClick={() => handleDownloadStem(selectedStem)}
                title="Download"
              >
                <i className="fa-solid fa-download"></i>
              </button>
            </div>
          </div>
          <ToolWaveform
            audioUrl={stems[selectedStem]}
            height={100}
            color={stemTypes.find(s => s.id === selectedStem)?.color || '#667eea'}
          />
        </div>
      )}

      {/* Status Message */}
      {statusMessage && (
        <div className={styles.statusMessage}>{statusMessage}</div>
      )}

      {/* Action Buttons */}
      <div className={styles.toolControlSection}>
        <div className={styles.toolControlRow}>
          {stems && (
            <button
              className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
              onClick={handleDownloadAll}
            >
              <i className="fa-solid fa-download"></i>
              Download All
            </button>
          )}
        </div>
        <button
          className={`${styles.toolControlBtn} ${styles.toolControlBtnPrimary}`}
          onClick={handleSeparate}
          disabled={isProcessing || !uploadedFile}
        >
          {isProcessing ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Separating...
            </>
          ) : (
            <>
              <i className="fa-solid fa-scissors"></i>
              Separate Stems
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default StemSeparationTool;
