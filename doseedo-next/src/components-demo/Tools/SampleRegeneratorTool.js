import React, { useState, useRef, useCallback } from 'react';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * Sample Regenerator Tool
 * Regenerate and enhance audio samples with AI
 */
const SampleRegeneratorTool = ({ tool }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);

  // Regeneration parameters
  const [denoiseStrength, setDenoiseStrength] = useState(0.5);
  const [steps, setSteps] = useState(100);
  const [cfgWeight, setCfgWeight] = useState(2.5);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1000000));

  // Handle file upload
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setUploadedUrl(url);
      setGeneratedAudioUrl(null);
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
      setGeneratedAudioUrl(null);
      setStatusMessage(`Loaded: ${file.name}`);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  // Regenerate sample
  const handleRegenerate = useCallback(async () => {
    if (!uploadedFile) {
      setStatusMessage('Please upload an audio file first.');
      return;
    }

    setIsGenerating(true);
    setStatusMessage('Starting regeneration...');
    setProgress(0);

    try {
      const params = {
        t0: denoiseStrength,
        steps: steps,
        cfgWeight: cfgWeight,
        seed: seed,
        noiseLevel: 1 - denoiseStrength, // Higher denoise = lower noise
        adapterScale: 0.5,
        monophonicMode: false
      };

      const startResult = await startGeneration(params, uploadedFile);
      const taskId = startResult.task_id;

      setStatusMessage('Regenerating audio...');

      const result = await pollUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.progress || 0);
          setStatusMessage(`Processing... ${Math.round((progressData.progress || 0) * 100)}%`);
        },
        null,
        1800
      );

      if (result.file_paths && result.file_paths.length > 0) {
        setGeneratedAudioUrl(result.file_paths[0]);
        setStatusMessage('Regeneration complete!');
      } else {
        setStatusMessage('Regeneration completed but no audio returned.');
      }

    } catch (error) {
      console.error('Regeneration failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  }, [uploadedFile, denoiseStrength, steps, cfgWeight, seed]);

  // Randomize seed
  const handleRandomizeSeed = useCallback(() => {
    setSeed(Math.floor(Math.random() * 1000000));
  }, []);

  // Download generated audio
  const handleDownload = useCallback(() => {
    if (generatedAudioUrl) {
      const a = document.createElement('a');
      a.href = generatedAudioUrl;
      a.download = `regenerated_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [generatedAudioUrl]);

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Tool Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div className={styles.toolGeneratorIcon} style={{ background: 'linear-gradient(135deg, rgba(186, 156, 255, 0.4), rgba(186, 156, 255, 0.2))' }}>
            <i className="fa-solid fa-rotate" style={{ color: '#ba9cff' }}></i>
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
          Upload Sample
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
                  setGeneratedAudioUrl(null);
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
              Original Sample
            </span>
          </div>
          <ToolWaveform
            audioUrl={uploadedUrl}
            height={80}
            color="#666"
          />
        </div>
      )}

      {/* Parameters */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-sliders"></i>
          Regeneration Settings
        </label>

        {/* Denoise Strength */}
        <div className={styles.parameterRow}>
          <span className={styles.paramLabel}>Denoise Strength: {Math.round(denoiseStrength * 100)}%</span>
          <input
            type="range"
            className={styles.toolSlider}
            min="0"
            max="1"
            step="0.05"
            value={denoiseStrength}
            onChange={(e) => setDenoiseStrength(parseFloat(e.target.value))}
          />
          <div className={styles.sliderLabels}>
            <span>Subtle</span>
            <span>Strong</span>
          </div>
        </div>

        {/* Steps */}
        <div className={styles.parameterRow}>
          <span className={styles.paramLabel}>Quality (Steps): {steps}</span>
          <input
            type="range"
            className={styles.toolSlider}
            min="50"
            max="200"
            step="10"
            value={steps}
            onChange={(e) => setSteps(parseInt(e.target.value))}
          />
          <div className={styles.sliderLabels}>
            <span>Fast</span>
            <span>Quality</span>
          </div>
        </div>

        {/* CFG Weight */}
        <div className={styles.parameterRow}>
          <span className={styles.paramLabel}>CFG Weight: {cfgWeight.toFixed(1)}</span>
          <input
            type="range"
            className={styles.toolSlider}
            min="1"
            max="5"
            step="0.1"
            value={cfgWeight}
            onChange={(e) => setCfgWeight(parseFloat(e.target.value))}
          />
          <div className={styles.sliderLabels}>
            <span>Creative</span>
            <span>Faithful</span>
          </div>
        </div>

        {/* Seed */}
        <div className={styles.parameterRow}>
          <span className={styles.paramLabel}>Seed:</span>
          <div className={styles.seedInput}>
            <input
              type="number"
              className={styles.toolNumberInput}
              value={seed}
              onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
            />
            <button
              className={styles.seedRandomBtn}
              onClick={handleRandomizeSeed}
              title="Randomize seed"
            >
              <i className="fa-solid fa-dice"></i>
            </button>
          </div>
        </div>
      </div>

      {/* Output Waveform */}
      <div className={styles.toolWaveformSection}>
        <div className={styles.toolWaveformHeader}>
          <span className={styles.toolWaveformLabel}>
            <i className="fa-solid fa-waveform-lines"></i>
            Regenerated Sample
          </span>
          {generatedAudioUrl && (
            <div className={styles.toolWaveformActions}>
              <button className={styles.toolActionBtn} onClick={handleDownload} title="Download">
                <i className="fa-solid fa-download"></i>
              </button>
            </div>
          )}
        </div>
        <ToolWaveform
          audioUrl={generatedAudioUrl}
          height={100}
          color="#ba9cff"
        />
        {statusMessage && (
          <div className={styles.statusMessage}>{statusMessage}</div>
        )}
      </div>

      {/* Progress Bar */}
      {isGenerating && (
        <div className={styles.progressContainer}>
          <div className={styles.progressBar} style={{ width: `${progress * 100}%` }}></div>
        </div>
      )}

      {/* Generate Button */}
      <div className={styles.toolControlSection}>
        <button
          className={`${styles.toolControlBtn} ${styles.toolControlBtnPrimary}`}
          onClick={handleRegenerate}
          disabled={isGenerating || !uploadedFile}
        >
          {isGenerating ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Regenerating...
            </>
          ) : (
            <>
              <i className="fa-solid fa-rotate"></i>
              Regenerate Sample
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default SampleRegeneratorTool;
