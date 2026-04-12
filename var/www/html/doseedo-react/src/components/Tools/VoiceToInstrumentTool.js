import React, { useState, useRef, useCallback, useMemo } from 'react';
import { useAudioRecorder } from '../../hooks/useAudioRecorder';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * Voice to Instrument Tool
 * Record voice/humming and convert to instrument using AI
 */
const VoiceToInstrumentTool = ({ tool }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [recordedFile, setRecordedFile] = useState(null);
  const fileInputRef = useRef(null);

  // Instrument selection state
  const [instrumentGroup, setInstrumentGroup] = useState('strings');
  const [instrumentSubgroup, setInstrumentSubgroup] = useState('violin');

  // Generation parameters
  // eslint-disable-next-line no-unused-vars
  const [extractMidi, setExtractMidi] = useState(true);

  // Audio recorder hook
  const {
    isRecording,
    recordingDuration,
    startRecording,
    stopRecording,
    cancelRecording
  } = useAudioRecorder({ extractMidi: false, convertToSine: false });

  // Instrument configuration
  const instrumentSubgroups = useMemo(() => ({
    piano: ['acoustic_piano', 'keys'],
    guitar: ['acoustic_guitar', 'electric_guitar'],
    bass: ['electric_bass', 'upright_bass'],
    strings: ['ensemble_strings', 'violin', 'cello'],
    brass: ['ensemble_brass', 'trumpet', 'trombone'],
    winds: ['ensemble_winds', 'flute', 'sax']
  }), []);

  const instrumentGroups = useMemo(() => [
    { id: 'piano',   label: 'Piano',   iconImg: '/assets/icons/piano.png' },
    { id: 'guitar',  label: 'Guitar',  iconImg: '/assets/icons/acguitar.png' },
    { id: 'bass',    label: 'Bass',    iconImg: '/assets/icons/elecbass.png' },
    { id: 'strings', label: 'Strings', iconImg: '/assets/icons/violin.png' },
    { id: 'brass',   label: 'Brass',   iconImg: '/assets/icons/trumpetens.png' },
    { id: 'winds',   label: 'Winds',   iconImg: '/assets/icons/sax.png' }
  ], []);

  const subgroupIcons = useMemo(() => ({
    'acoustic_piano': '/assets/icons/piano.png',
    'keys': '/assets/icons/keyboard.png',
    'acoustic_guitar': '/assets/icons/acguitar.png',
    'electric_guitar': '/assets/icons/elecgtr.png',
    'electric_bass': '/assets/icons/elecbass.png',
    'upright_bass': '/assets/icons/elecbass.png',
    'violin': '/assets/icons/violin.png',
    'cello': '/assets/icons/cello.png',
    'ensemble_strings': '/assets/icons/viollinensemble.png',
    'trumpet': '/assets/icons/tpt.png',
    'trombone': '/assets/icons/tbn.png',
    'ensemble_brass': '/assets/icons/trumpetens.png',
    'flute': '/assets/icons/flute.png',
    'sax': '/assets/icons/sax.png',
    'ensemble_winds': '/assets/icons/sax.png'
  }), []);

  // Handle group change
  const handleGroupChange = useCallback((newGroup) => {
    setInstrumentGroup(newGroup);
    const newSubgroups = instrumentSubgroups[newGroup] || instrumentSubgroups.strings;
    setInstrumentSubgroup(newSubgroups[0]);
  }, [instrumentSubgroups]);

  // Handle recording stop
  const handleStopRecording = useCallback(async () => {
    const file = await stopRecording();
    if (file) {
      setRecordedFile(file);
      setStatusMessage('Recording saved. Click Generate to convert to instrument.');
    }
  }, [stopRecording]);

  // Handle file upload
  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setRecordedFile(file);
      setStatusMessage(`File loaded: ${file.name}`);
    }
  };

  // Generate instrument from voice
  const handleGenerate = useCallback(async () => {
    if (!recordedFile) {
      setStatusMessage('Please record or upload audio first.');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setStatusMessage('Starting generation...');

    try {
      // Build generation parameters
      const params = {
        instrumentGroup,
        instrumentSubgroup,
        monophonicMode: instrumentSubgroup.startsWith('ensemble_'),
        arrangeMode: instrumentSubgroup.startsWith('ensemble_'),
        steps: 100,
        adapterScale: 0.5,
        cfgWeight: 2.5,
        t0: 0.95,
        seed: Math.floor(Math.random() * 1000000)
      };

      // Start generation
      const startResult = await startGeneration(params, recordedFile);
      const taskId = startResult.task_id;

      setStatusMessage('Processing audio...');

      // Poll for completion
      const result = await pollUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.progress || 0);
          setStatusMessage(`Processing... ${Math.round((progressData.progress || 0) * 100)}%`);
        },
        null,
        1800
      );

      // Get generated audio URL
      if (result.file_paths && result.file_paths.length > 0) {
        const audioPath = result.file_paths[0];
        setGeneratedAudioUrl(audioPath);
        setStatusMessage('Generation complete!');
      } else {
        setStatusMessage('Generation completed but no audio returned.');
      }

    } catch (error) {
      console.error('Generation failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  }, [recordedFile, instrumentGroup, instrumentSubgroup]);

  // Download generated audio
  const handleDownload = useCallback(() => {
    if (generatedAudioUrl) {
      const a = document.createElement('a');
      a.href = generatedAudioUrl;
      a.download = `${instrumentSubgroup}_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [generatedAudioUrl, instrumentSubgroup]);

  // Format recording duration
  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const availableSubgroups = instrumentSubgroups[instrumentGroup] || [];

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Tool Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div className={styles.toolGeneratorIcon} style={{ background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(102, 126, 234, 0.2))' }}>
            <i className="fa-solid fa-microphone-lines" style={{ color: '#667eea' }}></i>
          </div>
          <div className={styles.toolGeneratorTitleText}>
            <h2 className={styles.toolGeneratorTitle}>{tool.name}</h2>
            <p className={styles.toolGeneratorDescription}>{tool.description}</p>
          </div>
        </div>
      </div>

      {/* Instrument Selection */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-music"></i>
          Select Instrument
        </label>
        <div className={styles.instrumentGroupGrid}>
          {instrumentGroups.map(group => (
            <button
              key={group.id}
              type="button"
              className={`${styles.instrumentGroupBtn} ${instrumentGroup === group.id ? styles.active : ''}`}
              onClick={() => handleGroupChange(group.id)}
            >
              <img
                src={group.iconImg}
                alt={group.label}
                className={styles.instrumentIcon}
                style={{ width: 40, height: 40, objectFit: 'contain', opacity: 0.9 }}
                aria-hidden="true"
              />
              <span>{group.label}</span>
            </button>
          ))}
        </div>

        {/* Subgroup Selection */}
        <div className={styles.instrumentSubgroupGrid}>
          {availableSubgroups.map(subgroup => {
            let displayLabel = subgroup.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            if (subgroup.startsWith('ensemble_')) displayLabel = 'Ensemble';

            return (
              <button
                key={subgroup}
                type="button"
                className={`${styles.instrumentSubgroupBtn} ${instrumentSubgroup === subgroup ? styles.active : ''}`}
                onClick={() => setInstrumentSubgroup(subgroup)}
              >
                {subgroupIcons[subgroup] && (
                  <img
                    src={subgroupIcons[subgroup]}
                    alt={subgroup}
                    className={styles.instrumentIcon}
                    style={{ width: 36, height: 36, objectFit: 'contain', opacity: 0.9 }}
                    aria-hidden="true"
                  />
                )}
                <span>{displayLabel}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Recording Section */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-microphone"></i>
          Record Voice / Humming
        </label>

        <div className={styles.recordingControls}>
          {!isRecording ? (
            <>
              <button
                className={`${styles.toolControlBtn} ${styles.recordBtn}`}
                onClick={startRecording}
                disabled={isGenerating}
              >
                <i className="fa-solid fa-circle" style={{ color: '#ef4444' }}></i>
                Start Recording
              </button>
              <span className={styles.orDivider}>or</span>
              <button
                className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
                onClick={() => fileInputRef.current?.click()}
                disabled={isGenerating}
              >
                <i className="fa-solid fa-upload"></i>
                Upload Audio
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </>
          ) : (
            <>
              <div className={styles.recordingIndicator}>
                <i className="fa-solid fa-circle fa-beat" style={{ color: '#ef4444' }}></i>
                <span>Recording: {formatDuration(recordingDuration)}</span>
              </div>
              <button
                className={`${styles.toolControlBtn} ${styles.stopBtn}`}
                onClick={handleStopRecording}
              >
                <i className="fa-solid fa-stop"></i>
                Stop
              </button>
              <button
                className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
                onClick={cancelRecording}
              >
                <i className="fa-solid fa-xmark"></i>
                Cancel
              </button>
            </>
          )}
        </div>

        {recordedFile && (
          <div className={styles.recordedFileInfo}>
            <i className="fa-solid fa-file-audio"></i>
            <span>{recordedFile.name}</span>
            <button
              className={styles.removeFileBtn}
              onClick={() => {
                setRecordedFile(null);
              }}
            >
              <i className="fa-solid fa-xmark"></i>
            </button>
          </div>
        )}
      </div>

      {/* Options */}
      <div className={styles.toolOptionsRow}>
        <label className={styles.toolCheckbox}>
          <input
            type="checkbox"
            checked={extractMidi}
            onChange={(e) => setExtractMidi(e.target.checked)}
          />
          <span>Extract MIDI first (better pitch accuracy)</span>
        </label>
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
              <button className={styles.toolActionBtn} title="Download" onClick={handleDownload}>
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
      {isGenerating && (
        <div className={styles.progressContainer}>
          <div className={styles.progressBar} style={{ width: `${progress * 100}%` }}></div>
        </div>
      )}

      {/* Generate Button */}
      <div className={styles.toolControlSection}>
        <button
          className={`${styles.toolControlBtn} ${styles.toolControlBtnPrimary}`}
          onClick={handleGenerate}
          disabled={isGenerating || isRecording || !recordedFile}
        >
          {isGenerating ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Generating...
            </>
          ) : (
            <>
              <i className="fa-solid fa-bolt"></i>
              Generate Instrument
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default VoiceToInstrumentTool;
