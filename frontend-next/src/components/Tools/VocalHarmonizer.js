import React, { useState, useCallback, useRef } from 'react';
import styles from './VocalHarmonizer.module.css';

/**
 * VocalHarmonizer Component
 * Generate harmonies from vocal audio input
 */
const VocalHarmonizer = ({ onBack }) => {
  // State
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [progress, setProgress] = useState('');
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Settings
  const [numHarmonies, setNumHarmonies] = useState(2);
  const [voicing, setVoicing] = useState('close');
  const [musicalKey, setMusicalKey] = useState('C');
  const [noiseLevel, setNoiseLevel] = useState(0.8);
  const [useAceStep, setUseAceStep] = useState(false);
  const [extractLyrics, setExtractLyrics] = useState(true);

  const fileInputRef = useRef(null);

  // Voicing options
  const voicingOptions = [
    { value: 'close', label: 'Close Harmony', description: 'Tight voicings (3rds & 5ths)' },
    { value: 'thirds', label: 'Thirds', description: 'Major/minor thirds' },
    { value: 'fifths', label: 'Fifths', description: 'Perfect fifths' },
    { value: 'gospel', label: 'Gospel', description: 'Rich gospel style' },
    { value: 'barbershop', label: 'Barbershop', description: 'Classic quartet style' },
    { value: 'octaves', label: 'Octaves', description: 'Octave doubling' },
  ];

  // Key options
  const keyOptions = [
    'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B',
    'Cm', 'C#m', 'Dm', 'D#m', 'Em', 'Fm', 'F#m', 'Gm', 'G#m', 'Am', 'A#m', 'Bm'
  ];

  // Handle file selection
  const handleFileSelect = useCallback((event) => {
    const file = event.target.files?.[0];
    if (file) {
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  // Handle drag and drop
  const handleDrop = useCallback((event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      setAudioFile(file);
      setAudioUrl(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    }
  }, []);

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
  }, []);

  // Generate harmonies
  const handleGenerate = useCallback(async () => {
    if (!audioFile) {
      setError('Please select an audio file first');
      return;
    }

    setIsProcessing(true);
    setProgress('Uploading audio...');
    setError(null);
    setResult(null);

    try {
      // Create form data
      const formData = new FormData();
      formData.append('input_audio', audioFile);
      formData.append('num_harmonies', numHarmonies.toString());
      formData.append('voicing', voicing);
      formData.append('key', musicalKey);
      formData.append('noise_level', noiseLevel.toString());
      formData.append('use_ace_step', useAceStep.toString());
      formData.append('extract_lyrics', extractLyrics.toString());
      formData.append('tempo', '120');

      setProgress('Processing vocals...');

      // Call API
      const response = await fetch('/api/vocal-harmonizer', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Harmonization failed');
      }

      const data = await response.json();

      if (data.success) {
        setResult(data);
        setProgress('');
      } else {
        throw new Error('Generation failed');
      }

    } catch (err) {
      console.error('Harmonization error:', err);
      setError(err.message || 'An error occurred during harmonization');
    } finally {
      setIsProcessing(false);
    }
  }, [audioFile, numHarmonies, voicing, musicalKey, noiseLevel, useAceStep, extractLyrics]);

  // Clear all
  const handleClear = useCallback(() => {
    setAudioFile(null);
    setAudioUrl(null);
    setResult(null);
    setError(null);
    setProgress('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  }, []);

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <button className={styles.backButton} onClick={onBack}>
          <i className="fa-solid fa-arrow-left"></i>
          Back to Tools
        </button>
        <h1 className={`${styles.title} page-title`}>
          <i className="fa-solid fa-users"></i>
          Vocal Harmonizer
        </h1>
        <p className={styles.subtitle}>
          Upload your vocal audio and generate harmonies automatically
        </p>
      </div>

      <div className={styles.content}>
        {/* Left Panel - Settings */}
        <div className={styles.settingsPanel}>
          <h2 className={styles.sectionTitle}>Settings</h2>

          {/* Number of Harmonies */}
          <div className={styles.settingGroup}>
            <label className={styles.label}>Number of Harmonies</label>
            <div className={styles.buttonGroup}>
              {[1, 2, 3, 4].map(num => (
                <button
                  key={num}
                  className={`${styles.optionButton} ${numHarmonies === num ? styles.active : ''}`}
                  onClick={() => setNumHarmonies(num)}
                >
                  {num}
                </button>
              ))}
            </div>
          </div>

          {/* Voicing Style */}
          <div className={styles.settingGroup}>
            <label className={styles.label}>Voicing Style</label>
            <select
              className={styles.select}
              value={voicing}
              onChange={(e) => setVoicing(e.target.value)}
            >
              {voicingOptions.map(opt => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
            <span className={styles.description}>
              {voicingOptions.find(v => v.value === voicing)?.description}
            </span>
          </div>

          {/* Musical Key */}
          <div className={styles.settingGroup}>
            <label className={styles.label}>Musical Key</label>
            <select
              className={styles.select}
              value={musicalKey}
              onChange={(e) => setMusicalKey(e.target.value)}
            >
              {keyOptions.map(key => (
                <option key={key} value={key}>{key}</option>
              ))}
            </select>
          </div>

          {/* Noise Level */}
          <div className={styles.settingGroup}>
            <label className={styles.label}>
              Noise Level: {(noiseLevel * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              className={styles.slider}
              min="0"
              max="1"
              step="0.05"
              value={noiseLevel}
              onChange={(e) => setNoiseLevel(parseFloat(e.target.value))}
            />
            <div className={styles.sliderLabels}>
              <span>Original</span>
              <span>Creative</span>
            </div>
          </div>

          {/* Toggle Options */}
          <div className={styles.settingGroup}>
            <label className={styles.toggleLabel}>
              <input
                type="checkbox"
                checked={extractLyrics}
                onChange={(e) => setExtractLyrics(e.target.checked)}
              />
              <span className={styles.toggleText}>Extract Lyrics (Whisper)</span>
            </label>

            <label className={styles.toggleLabel}>
              <input
                type="checkbox"
                checked={useAceStep}
                onChange={(e) => setUseAceStep(e.target.checked)}
              />
              <span className={styles.toggleText}>ACE-Step Rendering</span>
              <span className={styles.betaTag}>Beta</span>
            </label>
          </div>
        </div>

        {/* Right Panel - Upload & Results */}
        <div className={styles.mainPanel}>
          {/* Upload Area */}
          <div
            className={`${styles.uploadArea} ${audioFile ? styles.hasFile : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDrop={handleDrop}
            onDragOver={handleDragOver}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={handleFileSelect}
              style={{ display: 'none' }}
            />

            {audioFile ? (
              <div className={styles.fileInfo}>
                <i className="fa-solid fa-file-audio"></i>
                <span className={styles.fileName}>{audioFile.name}</span>
                <span className={styles.fileSize}>
                  {(audioFile.size / (1024 * 1024)).toFixed(2)} MB
                </span>
              </div>
            ) : (
              <div className={styles.uploadPrompt}>
                <i className="fa-solid fa-cloud-arrow-up"></i>
                <span>Drop your vocal audio here</span>
                <span className={styles.uploadHint}>or click to browse</span>
                <span className={styles.supportedFormats}>WAV, MP3, FLAC supported</span>
              </div>
            )}
          </div>

          {/* Original Audio Player */}
          {audioUrl && (
            <div className={styles.audioPlayer}>
              <label>Original Vocal</label>
              <audio controls src={audioUrl} className={styles.audio} />
            </div>
          )}

          {/* Action Buttons */}
          <div className={styles.actions}>
            <button
              className={styles.generateButton}
              onClick={handleGenerate}
              disabled={!audioFile || isProcessing}
            >
              {isProcessing ? (
                <>
                  <i className="fa-solid fa-spinner fa-spin"></i>
                  Processing...
                </>
              ) : (
                <>
                  <i className="fa-solid fa-wand-magic-sparkles"></i>
                  Generate Harmonies
                </>
              )}
            </button>

            {audioFile && (
              <button className={styles.clearButton} onClick={handleClear}>
                <i className="fa-solid fa-xmark"></i>
                Clear
              </button>
            )}
          </div>

          {/* Progress */}
          {progress && (
            <div className={styles.progress}>
              <i className="fa-solid fa-circle-notch fa-spin"></i>
              {progress}
            </div>
          )}

          {/* Error */}
          {error && (
            <div className={styles.error}>
              <i className="fa-solid fa-triangle-exclamation"></i>
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div className={styles.results}>
              <h3 className={styles.resultsTitle}>
                <i className="fa-solid fa-check-circle"></i>
                Harmonies Generated
              </h3>

              {/* Extracted Lyrics */}
              {result.lyrics && (
                <div className={styles.lyricsSection}>
                  <label>Extracted Lyrics</label>
                  <div className={styles.lyricsText}>{result.lyrics}</div>
                </div>
              )}

              {/* Harmony Audio Players */}
              <div className={styles.harmonyPlayers}>
                {result.harmony_audio?.map((harmony, idx) => (
                  <div key={idx} className={styles.harmonyPlayer}>
                    <div className={styles.harmonyHeader}>
                      <span className={styles.harmonyLabel}>
                        Harmony {harmony.voice}
                      </span>
                      <a
                        href={harmony.url}
                        download={harmony.filename}
                        className={styles.downloadLink}
                      >
                        <i className="fa-solid fa-download"></i>
                      </a>
                    </div>
                    <audio controls src={harmony.url} className={styles.audio} />
                  </div>
                ))}
              </div>

              {/* ACE-Step Results */}
              {result.ace_step_audio?.length > 0 && (
                <div className={styles.aceStepSection}>
                  <h4>ACE-Step Enhanced</h4>
                  <div className={styles.harmonyPlayers}>
                    {result.ace_step_audio.map((harmony, idx) => (
                      <div key={idx} className={styles.harmonyPlayer}>
                        <div className={styles.harmonyHeader}>
                          <span className={styles.harmonyLabel}>
                            Enhanced Voice {harmony.voice}
                          </span>
                          <a
                            href={harmony.url}
                            download={harmony.filename}
                            className={styles.downloadLink}
                          >
                            <i className="fa-solid fa-download"></i>
                          </a>
                        </div>
                        <audio controls src={harmony.url} className={styles.audio} />
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* MIDI Download */}
              {result.midi_url && (
                <div className={styles.midiSection}>
                  <a
                    href={result.midi_url}
                    download="harmonies.mid"
                    className={styles.midiButton}
                  >
                    <i className="fa-solid fa-file-audio"></i>
                    Download MIDI
                  </a>
                </div>
              )}

              {/* Stats */}
              <div className={styles.stats}>
                <span>{result.num_pitch_events} notes detected</span>
                {result.word_count > 0 && (
                  <span>{result.word_count} words extracted</span>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default VocalHarmonizer;
