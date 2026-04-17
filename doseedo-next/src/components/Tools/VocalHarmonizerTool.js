import React, { useState, useRef, useCallback } from 'react';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

const API_BASE = '';

// Voicing style options
const VOICING_STYLES = [
  { id: 'thirds', name: 'Thirds (Traditional)', description: 'Classic harmony using thirds and fifths' },
  { id: 'sixths', name: 'Sixths', description: 'Sweet sounding sixths harmony' },
  { id: 'power', name: 'Power (Rock)', description: 'Simple power chord style harmonies' },
  { id: 'close', name: 'Close Harmony', description: 'Tight, close harmonies' },
  { id: 'wide', name: 'Wide Spread', description: 'Wide, spacious harmonies' },
  { id: 'jazz', name: 'Jazz Voicing', description: 'Jazz-style extended harmonies' },
  { id: 'minor', name: 'Minor Thirds', description: 'Darker minor harmonies' },
  { id: 'barbershop', name: 'Barbershop', description: 'Four-part barbershop style' }
];

// Musical keys
const KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'chromatic'];

/**
 * Vocal Harmonizer Tool
 * Generate harmony tracks from vocal audio
 */
const VocalHarmonizerTool = ({ tool }) => {
  // State
  const [audioFile, setAudioFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState(null);

  // Settings
  const [numHarmonies, setNumHarmonies] = useState(2);
  const [voicingStyle, setVoicingStyle] = useState('thirds');
  const [musicalKey, setMusicalKey] = useState('C');
  const [mode, setMode] = useState('major');
  const [noiseLevel, setNoiseLevel] = useState(0.3);
  const [useAceStep, setUseAceStep] = useState(false);

  // Results
  const [harmonyAudios, setHarmonyAudios] = useState([]);
  const [harmonyMidi, setHarmonyMidi] = useState(null);
  const [extractedLyrics, setExtractedLyrics] = useState(null);

  const fileInputRef = useRef(null);

  // Handle file upload
  const handleFileUpload = useCallback((file) => {
    if (file && file.type.startsWith('audio/')) {
      setAudioFile(file);
      setError(null);
      setStatusMessage(`File loaded: ${file.name}`);
      // Clear previous results
      setHarmonyAudios([]);
      setHarmonyMidi(null);
      setExtractedLyrics(null);
    } else {
      setError('Please upload an audio file (WAV, MP3, etc.)');
    }
  }, []);

  // Drag and drop handlers
  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    handleFileUpload(file);
  }, [handleFileUpload]);

  const handleFileInputChange = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      handleFileUpload(file);
    }
  }, [handleFileUpload]);

  // Generate harmonies
  const handleGenerate = useCallback(async () => {
    if (!audioFile) {
      setError('Please upload an audio file first');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setError(null);
    setStatusMessage('Starting harmony generation...');

    try {
      const formData = new FormData();
      formData.append('audioFile', audioFile);
      formData.append('numHarmonies', numHarmonies.toString());
      formData.append('voicingStyle', voicingStyle);
      formData.append('key', musicalKey);
      formData.append('mode', mode);
      formData.append('noiseLevel', noiseLevel.toString());
      formData.append('useAceStep', useAceStep.toString());

      setStatusMessage('Uploading audio and extracting pitch...');
      setProgress(0.1);

      const response = await fetch(`${API_BASE}/api/vocal-harmonizer`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      setProgress(0.5);
      setStatusMessage('Generating harmony tracks...');

      const result = await response.json();

      setProgress(1.0);
      setStatusMessage('Harmonies generated successfully!');

      // Set results
      if (result.harmony_audio && result.harmony_audio.length > 0) {
        setHarmonyAudios(result.harmony_audio);
      }
      if (result.harmony_midi) {
        setHarmonyMidi(result.harmony_midi);
      }
      if (result.lyrics) {
        setExtractedLyrics(result.lyrics);
      }

    } catch (err) {
      console.error('Harmony generation failed:', err);
      setError(err.message || 'Failed to generate harmonies');
      setStatusMessage('');
    } finally {
      setIsGenerating(false);
    }
  }, [audioFile, numHarmonies, voicingStyle, musicalKey, mode, noiseLevel, useAceStep]);

  // Download MIDI
  const handleDownloadMidi = useCallback(() => {
    if (harmonyMidi) {
      const link = document.createElement('a');
      link.href = harmonyMidi;
      link.download = 'harmonies.mid';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  }, [harmonyMidi]);

  // Remove file
  const handleRemoveFile = useCallback(() => {
    setAudioFile(null);
    setHarmonyAudios([]);
    setHarmonyMidi(null);
    setExtractedLyrics(null);
    setStatusMessage('');
    setError(null);
  }, []);

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div
            className={styles.toolGeneratorIcon}
            style={{ background: 'linear-gradient(135deg, rgba(76, 175, 80, 0.4), rgba(76, 175, 80, 0.2))' }}
          >
            <i className="fa-solid fa-music" style={{ color: '#4CAF50' }}></i>
          </div>
          <div className={styles.toolGeneratorTitleText}>
            <h2 className={styles.toolGeneratorTitle}>{tool?.name || 'Vocal Harmonizer'}</h2>
            <p className={styles.toolGeneratorDescription}>
              {tool?.description || 'Generate beautiful harmony tracks from your vocals'}
            </p>
          </div>
        </div>
        <span className={styles.availableBadge}>Available</span>
      </div>

      {/* File Upload Section */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-upload"></i>
          Upload Vocal Audio
        </label>

        {!audioFile ? (
          <div
            className={`${styles.toolFileUpload} ${isDragging ? styles.toolFileUploadActive : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <i className="fa-solid fa-cloud-arrow-up"></i>
            <span>Drag & drop your vocal audio here</span>
            <span className={styles.uploadHint}>or click to browse (WAV, MP3, FLAC)</span>
          </div>
        ) : (
          <div className={styles.uploadedFileInfo}>
            <i className="fa-solid fa-file-audio"></i>
            <span>{audioFile.name}</span>
            <button className={styles.removeFileBtn} onClick={handleRemoveFile}>
              <i className="fa-solid fa-xmark"></i>
            </button>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="audio/*"
          onChange={handleFileInputChange}
          style={{ display: 'none' }}
        />
      </div>

      {/* Settings Section */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-sliders"></i>
          Harmony Settings
        </label>

        <div className={styles.harmonizerSettings}>
          {/* Number of Harmonies */}
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>Number of Harmony Voices</label>
            <div className={styles.harmonyCountBtns}>
              {[1, 2, 3, 4].map(n => (
                <button
                  key={n}
                  className={`${styles.harmonyCountBtn} ${numHarmonies === n ? styles.active : ''}`}
                  onClick={() => setNumHarmonies(n)}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Voicing Style */}
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>Voicing Style</label>
            <select
              className={styles.toolSelect}
              value={voicingStyle}
              onChange={(e) => setVoicingStyle(e.target.value)}
            >
              {VOICING_STYLES.map(style => (
                <option key={style.id} value={style.id}>
                  {style.name}
                </option>
              ))}
            </select>
            <span className={styles.settingHint}>
              {VOICING_STYLES.find(s => s.id === voicingStyle)?.description}
            </span>
          </div>

          {/* Musical Key */}
          <div className={styles.settingRowInline}>
            <div className={styles.settingCol}>
              <label className={styles.settingLabel}>Key</label>
              <select
                className={styles.toolSelect}
                value={musicalKey}
                onChange={(e) => setMusicalKey(e.target.value)}
              >
                {KEYS.map(key => (
                  <option key={key} value={key}>{key}</option>
                ))}
              </select>
            </div>
            <div className={styles.settingCol}>
              <label className={styles.settingLabel}>Mode</label>
              <select
                className={styles.toolSelect}
                value={mode}
                onChange={(e) => setMode(e.target.value)}
              >
                <option value="major">Major</option>
                <option value="minor">Minor</option>
              </select>
            </div>
          </div>

          {/* Noise Level */}
          <div className={styles.settingRow}>
            <label className={styles.settingLabel}>
              Noise Reduction: {Math.round(noiseLevel * 100)}%
            </label>
            <input
              type="range"
              className={styles.toolSlider}
              min="0"
              max="1"
              step="0.1"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
            />
            <div className={styles.sliderLabels}>
              <span>None</span>
              <span>Maximum</span>
            </div>
          </div>

          {/* ACE-Step Option */}
          <div className={styles.toolOptionsRow}>
            <label className={styles.toolCheckbox}>
              <input
                type="checkbox"
                checked={useAceStep}
                onChange={(e) => setUseAceStep(e.target.checked)}
              />
              <span>Use ACE-Step for natural vocal rendering (slower)</span>
            </label>
          </div>
        </div>
      </div>

      {/* Progress/Status */}
      {(isGenerating || statusMessage) && (
        <div className={styles.toolSection}>
          {isGenerating && (
            <div className={styles.progressContainer}>
              <div
                className={styles.progressBar}
                style={{ width: `${progress * 100}%` }}
              ></div>
            </div>
          )}
          {statusMessage && (
            <div className={styles.statusMessage}>{statusMessage}</div>
          )}
        </div>
      )}

      {/* Error Message */}
      {error && (
        <div className={styles.errorMessage}>
          <i className="fa-solid fa-circle-exclamation"></i>
          {error}
        </div>
      )}

      {/* Results Section */}
      {harmonyAudios.length > 0 && (
        <div className={styles.toolSection}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-music"></i>
            Generated Harmonies
          </label>

          <div className={styles.harmonyResults}>
            {harmonyAudios.map((audioUrl, index) => (
              <div key={index} className={styles.harmonyTrack}>
                <div className={styles.harmonyTrackHeader}>
                  <span className={styles.harmonyTrackLabel}>
                    <i className="fa-solid fa-waveform-lines"></i>
                    Harmony Voice {index + 1}
                  </span>
                  <a
                    href={audioUrl}
                    download={`harmony_${index + 1}.wav`}
                    className={styles.downloadBtn}
                  >
                    <i className="fa-solid fa-download"></i>
                  </a>
                </div>
                <ToolWaveform
                  audioUrl={audioUrl}
                  height={80}
                  color={`hsl(${120 + index * 30}, 70%, 50%)`}
                />
              </div>
            ))}
          </div>

          {/* MIDI Download */}
          {harmonyMidi && (
            <button
              className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
              onClick={handleDownloadMidi}
            >
              <i className="fa-solid fa-file-audio"></i>
              Download Harmony MIDI
            </button>
          )}
        </div>
      )}

      {/* Extracted Lyrics */}
      {extractedLyrics && extractedLyrics.text && (
        <div className={styles.toolSection}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-closed-captioning"></i>
            Extracted Lyrics
          </label>
          <div className={styles.lyricsDisplay}>
            {extractedLyrics.text}
          </div>
        </div>
      )}

      {/* Generate Button */}
      <div className={styles.toolControlSection}>
        <button
          className={`${styles.toolControlBtn} ${styles.toolControlBtnPrimary}`}
          onClick={handleGenerate}
          disabled={isGenerating || !audioFile}
        >
          {isGenerating ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Generating Harmonies...
            </>
          ) : (
            <>
              <i className="fa-solid fa-wand-magic-sparkles"></i>
              Generate Harmonies
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default VocalHarmonizerTool;
