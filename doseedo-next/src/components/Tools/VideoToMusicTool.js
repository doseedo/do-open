import React, { useState, useRef, useCallback } from 'react';
import { uploadVideo, pollVideoUntilComplete, collapseSceneChanges, computeBestTempos } from '../../services/videoAPI';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * Video to Music Tool
 * Generate music from video with scene detection
 */
const VideoToMusicTool = ({ tool }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [videoPreviewUrl, setVideoPreviewUrl] = useState(null);
  const [sceneData, setSceneData] = useState(null);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [progress, setProgress] = useState(0);
  const fileInputRef = useRef(null);
  const videoRef = useRef(null);

  // Generation options
  const [useWhisper, setUseWhisper] = useState(false);
  const [instrumentGroup, setInstrumentGroup] = useState('strings');

  const instrumentGroups = [
    { id: 'piano', label: 'Piano', icon: 'fa-piano' },
    { id: 'guitar', label: 'Guitar', icon: 'fa-guitar' },
    { id: 'strings', label: 'Strings', icon: 'fa-violin' },
    { id: 'brass', label: 'Brass', icon: 'fa-trumpet' },
    { id: 'electronic', label: 'Electronic', icon: 'fa-wave-square' }
  ];

  // Handle file upload
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setVideoPreviewUrl(url);
      setSceneData(null);
      setGeneratedAudioUrl(null);
      setStatusMessage(`Loaded: ${file.name}`);
    }
  }, []);

  // Handle drag and drop
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('video/')) {
      setUploadedFile(file);
      const url = URL.createObjectURL(file);
      setVideoPreviewUrl(url);
      setSceneData(null);
      setGeneratedAudioUrl(null);
      setStatusMessage(`Loaded: ${file.name}`);
    }
  }, []);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
  }, []);

  // Analyze video for scene detection
  const handleAnalyzeVideo = useCallback(async () => {
    if (!uploadedFile) {
      setStatusMessage('Please upload a video first.');
      return;
    }

    setIsProcessing(true);
    setStatusMessage('Uploading video for analysis...');
    setProgress(0);

    try {
      // Upload video
      const uploadResult = await uploadVideo(uploadedFile, useWhisper);
      const taskId = uploadResult.task_id;

      setStatusMessage('Detecting scenes...');

      // Poll for completion
      const result = await pollVideoUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.attempts / 200);
          setStatusMessage(`Analyzing video... (${progressData.attempts}s)`);
        },
        3000,
        600
      );

      if (result.scene_changes) {
        // Process scene changes
        const collapsedScenes = collapseSceneChanges(result.scene_changes, 3);
        const tempos = computeBestTempos(collapsedScenes);

        setSceneData({
          sceneChanges: collapsedScenes,
          tempos: tempos,
          duration: result.video_duration,
          audioUrl: result.audio_url
        });

        setStatusMessage(`Found ${collapsedScenes.length} scenes. Ready to generate music.`);
      } else {
        setStatusMessage('Analysis completed but no scenes detected.');
      }

    } catch (error) {
      console.error('Video analysis failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsProcessing(false);
      setProgress(0);
    }
  }, [uploadedFile, useWhisper]);

  // Generate music for the video
  const handleGenerateMusic = useCallback(async () => {
    if (!sceneData) {
      setStatusMessage('Please analyze the video first.');
      return;
    }

    setIsGenerating(true);
    setStatusMessage('Generating music for video...');
    setProgress(0);

    try {
      // Build generation params with scene data
      const params = {
        instrumentGroup: instrumentGroup,
        instrumentSubgroup: instrumentGroup === 'strings' ? 'ensemble_strings' : `${instrumentGroup}_main`,
        monophonicMode: true,
        arrangeMode: true,
        sceneDurations: sceneData.sceneChanges,
        sceneTempos: sceneData.tempos,
        steps: 100,
        seed: Math.floor(Math.random() * 1000000)
      };

      // If we have extracted audio, use it as conditioning
      let audioFile = null;
      if (sceneData.audioUrl) {
        const response = await fetch(sceneData.audioUrl);
        const blob = await response.blob();
        audioFile = new File([blob], 'video_audio.wav', { type: 'audio/wav' });
      }

      const startResult = await startGeneration(params, audioFile);
      const taskId = startResult.task_id;

      const result = await pollUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.progress || 0);
          setStatusMessage(`Generating... ${Math.round((progressData.progress || 0) * 100)}%`);
        },
        null,
        1800
      );

      if (result.file_paths && result.file_paths.length > 0) {
        setGeneratedAudioUrl(result.file_paths[0]);
        setStatusMessage('Music generation complete!');
      } else {
        setStatusMessage('Generation completed but no audio returned.');
      }

    } catch (error) {
      console.error('Music generation failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  }, [sceneData, instrumentGroup]);

  // Download generated audio
  const handleDownload = useCallback(() => {
    if (generatedAudioUrl) {
      const a = document.createElement('a');
      a.href = generatedAudioUrl;
      a.download = `video_music_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [generatedAudioUrl]);

  // Format time
  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Tool Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div className={styles.toolGeneratorIcon} style={{ background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(102, 126, 234, 0.2))' }}>
            <i className="fa-solid fa-video" style={{ color: '#667eea' }}></i>
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
          <i className="fa-solid fa-film"></i>
          Upload Video
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
            accept="video/*"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          {uploadedFile ? (
            <div className={styles.uploadedFileInfo}>
              <i className="fa-solid fa-file-video"></i>
              <span>{uploadedFile.name}</span>
              <button
                className={styles.removeFileBtn}
                onClick={(e) => {
                  e.stopPropagation();
                  setUploadedFile(null);
                  setVideoPreviewUrl(null);
                  setSceneData(null);
                  setGeneratedAudioUrl(null);
                }}
              >
                <i className="fa-solid fa-xmark"></i>
              </button>
            </div>
          ) : (
            <>
              <i className="fa-solid fa-cloud-arrow-up"></i>
              <span>Click or drag video file here</span>
              <span className={styles.uploadHint}>MP4, MOV, AVI supported</span>
            </>
          )}
        </div>
      </div>

      {/* Video Preview */}
      {videoPreviewUrl && (
        <div className={styles.videoPreviewContainer}>
          <video
            ref={videoRef}
            src={videoPreviewUrl}
            className={styles.videoPreview}
            controls
            muted
          />
        </div>
      )}

      {/* Options */}
      <div className={styles.toolOptionsRow}>
        <label className={styles.toolCheckbox}>
          <input
            type="checkbox"
            checked={useWhisper}
            onChange={(e) => setUseWhisper(e.target.checked)}
          />
          <span>Use Whisper for speech transcription</span>
        </label>
      </div>

      {/* Scene Analysis Results */}
      {sceneData && (
        <div className={styles.toolSection}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-film"></i>
            Detected Scenes ({sceneData.sceneChanges.length})
          </label>
          <div className={styles.sceneTimeline}>
            {sceneData.sceneChanges.map((time, index) => (
              <div
                key={index}
                className={styles.sceneMarker}
                style={{ left: `${(time / sceneData.duration) * 100}%` }}
                title={`Scene ${index + 1}: ${formatTime(time)} (${sceneData.tempos[index] || 120} BPM)`}
              />
            ))}
          </div>
          <div className={styles.sceneInfo}>
            <span>Duration: {formatTime(sceneData.duration)}</span>
            <span>Avg Tempo: {Math.round(sceneData.tempos.reduce((a, b) => a + b, 0) / sceneData.tempos.length)} BPM</span>
          </div>
        </div>
      )}

      {/* Instrument Selection */}
      {sceneData && (
        <div className={styles.toolSection}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-music"></i>
            Music Style
          </label>
          <div className={styles.instrumentGroupGrid}>
            {instrumentGroups.map(group => (
              <button
                key={group.id}
                className={`${styles.instrumentGroupBtn} ${instrumentGroup === group.id ? styles.active : ''}`}
                onClick={() => setInstrumentGroup(group.id)}
              >
                <i className={`fa-solid ${group.icon}`}></i>
                <span>{group.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Waveform Display */}
      <div className={styles.toolWaveformSection}>
        <div className={styles.toolWaveformHeader}>
          <span className={styles.toolWaveformLabel}>
            <i className="fa-solid fa-waveform-lines"></i>
            Generated Music
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
          height={120}
          color="#667eea"
        />
        {statusMessage && (
          <div className={styles.statusMessage}>{statusMessage}</div>
        )}
      </div>

      {/* Progress Bar */}
      {(isProcessing || isGenerating) && (
        <div className={styles.progressContainer}>
          <div className={styles.progressBar} style={{ width: `${progress * 100}%` }}></div>
        </div>
      )}

      {/* Action Buttons */}
      <div className={styles.toolControlSection}>
        <div className={styles.toolControlRow}>
          <button
            className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
            onClick={handleAnalyzeVideo}
            disabled={isProcessing || isGenerating || !uploadedFile}
          >
            {isProcessing ? (
              <>
                <i className="fa-solid fa-spinner fa-spin"></i>
                Analyzing...
              </>
            ) : (
              <>
                <i className="fa-solid fa-magnifying-glass"></i>
                Analyze Video
              </>
            )}
          </button>
        </div>
        <button
          className={`${styles.toolControlBtn} ${styles.toolControlBtnPrimary}`}
          onClick={handleGenerateMusic}
          disabled={isProcessing || isGenerating || !sceneData}
        >
          {isGenerating ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Generating...
            </>
          ) : (
            <>
              <i className="fa-solid fa-bolt"></i>
              Generate Music
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default VideoToMusicTool;
