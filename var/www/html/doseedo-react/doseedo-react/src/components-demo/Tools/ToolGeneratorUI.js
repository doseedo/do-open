import React, { useState, useRef } from 'react';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * ToolGeneratorUI - Generator interface for each tool
 * Displays tool-specific header, controls, and waveform output
 */
const ToolGeneratorUI = ({ tool, onBack }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [inputText, setInputText] = useState('');
  const fileInputRef = useRef(null);
  const [uploadedFile, setUploadedFile] = useState(null);

  // Handle generate action (placeholder for now)
  const handleGenerate = async () => {
    setIsGenerating(true);
    // Placeholder - simulate generation
    await new Promise(resolve => setTimeout(resolve, 2000));
    setIsGenerating(false);
    // In the future, this would call the actual API and set the generated audio URL
  };

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      // Create object URL for preview
      const url = URL.createObjectURL(file);
      setGeneratedAudioUrl(url);
    }
  };

  // Handle download
  const handleDownload = () => {
    if (generatedAudioUrl) {
      const a = document.createElement('a');
      a.href = generatedAudioUrl;
      a.download = `${tool.name.toLowerCase().replace(/\s+/g, '_')}_output.mp3`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  // Get tool-specific accent color
  const getToolColor = () => {
    const colors = {
      'Video to Music': '#667eea',
      'Lyric Edit': '#9c82c8',
      'Voice to Instrument': '#667eea',
      'Sample Regenerator': '#ba9cff',
      'Stem Separation': '#9c82c8',
      'Beat Generator': '#667eea'
    };
    return colors[tool.name] || '#667eea';
  };

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Tool Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div
            className={styles.toolGeneratorIcon}
            style={{ background: `linear-gradient(135deg, ${getToolColor()}40, ${getToolColor()}20)` }}
          >
            <i className={`fa-solid ${tool.icon}`} style={{ color: getToolColor() }}></i>
          </div>
          <div className={styles.toolGeneratorTitleText}>
            <h2 className={styles.toolGeneratorTitle}>{tool.name}</h2>
            <p className={styles.toolGeneratorDescription}>{tool.description}</p>
          </div>
        </div>
      </div>

      {/* Input Section */}
      <div className={styles.toolInputSection}>
        {/* Text/Prompt Input */}
        <div className={styles.toolInputGroup}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-pen"></i>
            Input Prompt
          </label>
          <textarea
            className={styles.toolTextarea}
            placeholder="Describe what you want to generate..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            rows={3}
          />
        </div>

        {/* File Upload */}
        <div className={styles.toolInputGroup}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-upload"></i>
            Upload File (Optional)
          </label>
          <div
            className={styles.toolFileUpload}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*,video/*"
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
                    setGeneratedAudioUrl(null);
                  }}
                >
                  <i className="fa-solid fa-xmark"></i>
                </button>
              </div>
            ) : (
              <>
                <i className="fa-solid fa-cloud-arrow-up"></i>
                <span>Click to upload or drag and drop</span>
                <span className={styles.uploadHint}>Audio or Video files</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Waveform Display */}
      <div className={styles.toolWaveformSection}>
        <div className={styles.toolWaveformHeader}>
          <span className={styles.toolWaveformLabel}>
            <i className="fa-solid fa-waveform-lines"></i>
            Output
          </span>
          {generatedAudioUrl && (
            <div className={styles.toolWaveformActions}>
              <button className={styles.toolActionBtn} title="Play">
                <i className="fa-solid fa-play"></i>
              </button>
              <button className={styles.toolActionBtn} onClick={handleDownload} title="Download">
                <i className="fa-solid fa-download"></i>
              </button>
            </div>
          )}
        </div>
        <ToolWaveform
          audioUrl={generatedAudioUrl}
          height={120}
          color={getToolColor()}
        />
      </div>

      {/* Control Buttons - Placeholder buttons same for all tools */}
      <div className={styles.toolControlSection}>
        <div className={styles.toolControlRow}>
          <button
            className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
            disabled={isGenerating}
          >
            <i className="fa-solid fa-sliders"></i>
            Settings
          </button>
          <button
            className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
            disabled={isGenerating}
          >
            <i className="fa-solid fa-wand-magic-sparkles"></i>
            Presets
          </button>
          <button
            className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
            disabled={isGenerating || !generatedAudioUrl}
          >
            <i className="fa-solid fa-rotate"></i>
            Regenerate
          </button>
        </div>
        <button
          className={`${styles.toolControlBtn} ${styles.toolControlBtnPrimary}`}
          onClick={handleGenerate}
          disabled={isGenerating}
        >
          {isGenerating ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Generating...
            </>
          ) : (
            <>
              <i className="fa-solid fa-bolt"></i>
              Generate
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default ToolGeneratorUI;
