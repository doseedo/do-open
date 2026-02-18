import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { generateAudio, createGenerationPayload } from '../../utils/api';
import { isMidiFile, isAudioFile, createAudioBlobUrl } from '../../utils/audioUtils';
import './GenerationPanel.css';

/**
 * GenerationPanel Component
 * Main control panel for audio generation parameters
 * Based on the input sections from doseedo2.html
 */
function GenerationPanel() {
  const { state, dispatch } = useApp();
  const [collapsedSections, setCollapsedSections] = useState({
    instrument: false,
    mode: true,
    processing: true,
    generation: true,
    advanced: true,
    testTime: true
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [generationError, setGenerationError] = useState(null);
  const [generationProgress, setGenerationProgress] = useState(0);

  const updateParam = (param, value) => {
    dispatch({
      type: 'UPDATE_GENERATION_PARAMS',
      payload: { [param]: value }
    });
  };

  const toggleSection = (section) => {
    setCollapsedSections(prev => ({
      ...prev,
      [section]: !prev[section]
    }));
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Determine file type
    const fileType = isMidiFile(file) ? 'midi' : isAudioFile(file) ? 'audio' : 'unknown';

    if (fileType === 'unknown') {
      alert('Unsupported file type. Please upload an audio or MIDI file.');
      return;
    }

    // Create preview URL for audio files
    let previewUrl = null;
    if (fileType === 'audio') {
      previewUrl = URL.createObjectURL(file);
    }

    dispatch({
      type: 'SET_UPLOADED_FILE',
      payload: { file, fileType, previewUrl }
    });

    // Auto-enable MIDI mode if MIDI file is uploaded
    if (fileType === 'midi' && !state.generationParams.midiMode) {
      updateParam('midiMode', true);
    }

    console.log('File uploaded:', {
      name: file.name,
      type: fileType,
      size: (file.size / 1024).toFixed(2) + ' KB'
    });
  };

  const handleClearFile = () => {
    // Revoke preview URL if exists
    if (state.uploadedFile?.previewUrl) {
      URL.revokeObjectURL(state.uploadedFile.previewUrl);
    }

    dispatch({
      type: 'SET_UPLOADED_FILE',
      payload: { file: null, fileType: null, previewUrl: null }
    });
  };

  const handleGenerate = async () => {
    setIsGenerating(true);
    setGenerationError(null);
    setGenerationProgress(0);

    try {
      console.log('Starting generation with params:', state.generationParams);

      // Create form data payload
      const formData = createGenerationPayload(state, state.uploadedFile);

      // Simulate progress (in real app, use WebSocket or polling)
      const progressInterval = setInterval(() => {
        setGenerationProgress(prev => Math.min(prev + 10, 90));
      }, 500);

      // Call API
      const result = await generateAudio(formData);

      clearInterval(progressInterval);
      setGenerationProgress(100);

      console.log('Generation result:', result);

      // Handle generated audio
      if (result.audio_files && result.audio_files.length > 0) {
        // Add generated tracks to workspace
        result.audio_files.forEach((audioUrl, index) => {
          const trackId = Date.now() + index;
          const trackName = `Generated Track ${state.audioTracks.length + index + 1}`;

          dispatch({
            type: 'ADD_AUDIO_TRACK',
            payload: {
              id: trackId,
              name: trackName,
              url: audioUrl,
              duration: null // Will be set when WaveSurfer loads it
            }
          });
        });

        // Set first generated track as current
        if (result.audio_files.length > 0) {
          dispatch({
            type: 'SET_CURRENT_TRACK',
            payload: {
              id: Date.now(),
              name: `Generated Track ${state.audioTracks.length + 1}`,
              url: result.audio_files[0]
            }
          });
        }

        alert(`✅ Generation complete! Created ${result.audio_files.length} track(s).`);
      } else {
        throw new Error('No audio files generated');
      }

    } catch (error) {
      console.error('Generation error:', error);
      setGenerationError(error.message || 'Generation failed. Please try again.');
      alert(`❌ Generation failed: ${error.message || 'Unknown error'}`);
    } finally {
      setIsGenerating(false);
      setGenerationProgress(0);
    }
  };

  return (
    <div className="generation-panel">
      <h4>Generation Controls</h4>

      {/* 1. Instrument Selection */}
      <div className={`input-section ${collapsedSections.instrument ? 'collapsed' : ''}`}>
        <h5 onClick={() => toggleSection('instrument')}>
          1. Instrument Selection
          <i className={`fa-solid ${collapsedSections.instrument ? 'fa-chevron-down' : 'fa-chevron-up'}`}></i>
        </h5>
        {!collapsedSections.instrument && (
          <div className="section-content">
            <div className="param-row">
              <label>
                Group:
                <select
                  value={state.generationParams.instrumentGroup}
                  onChange={(e) => updateParam('instrumentGroup', e.target.value)}
                >
                  <option value="strings">Strings</option>
                  <option value="winds">Winds</option>
                  <option value="brass">Brass</option>
                  <option value="keys">Keys</option>
                </select>
              </label>
            </div>
            <div className="param-row">
              <label>
                Subgroup:
                <select
                  value={state.generationParams.instrumentSubgroup}
                  onChange={(e) => updateParam('instrumentSubgroup', e.target.value)}
                >
                  <option value="violin">Violin</option>
                  <option value="cello">Cello</option>
                  <option value="flute">Flute</option>
                  <option value="trumpet">Trumpet</option>
                  <option value="muted_trumpet">Muted Trumpet</option>
                </select>
              </label>
            </div>
            <div className="param-row">
              <label>
                Key:
                <select
                  value={state.generationParams.generationKey}
                  onChange={(e) => updateParam('generationKey', e.target.value)}
                >
                  {['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'].map(key => (
                    <option key={key} value={key}>{key}</option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        )}
      </div>

      {/* 2. Mode Selection */}
      <div className={`input-section ${collapsedSections.mode ? 'collapsed' : ''}`}>
        <h5 onClick={() => toggleSection('mode')}>
          2. Mode Selection
          <i className={`fa-solid ${collapsedSections.mode ? 'fa-chevron-down' : 'fa-chevron-up'}`}></i>
        </h5>
        {!collapsedSections.mode && (
          <div className="section-content">
            <div className="checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={state.generationParams.midiMode}
                  onChange={(e) => updateParam('midiMode', e.target.checked)}
                />
                MIDI Mode
              </label>
            </div>
            <div className="checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={state.generationParams.monophonicMode}
                  onChange={(e) => updateParam('monophonicMode', e.target.checked)}
                />
                Monophonic Mode
              </label>
            </div>
            <div className="checkbox-row">
              <label>
                <input
                  type="checkbox"
                  checked={state.generationParams.fattenMode}
                  onChange={(e) => updateParam('fattenMode', e.target.checked)}
                />
                Fatten Mode
              </label>
            </div>
          </div>
        )}
      </div>

      {/* 3. Generation Parameters */}
      <div className={`input-section ${collapsedSections.generation ? 'collapsed' : ''}`}>
        <h5 onClick={() => toggleSection('generation')}>
          3. Generation Parameters
          <i className={`fa-solid ${collapsedSections.generation ? 'fa-chevron-down' : 'fa-chevron-up'}`}></i>
        </h5>
        {!collapsedSections.generation && (
          <div className="section-content">
            <div className="param-row">
              <label>
                Seed: {state.generationParams.seed}
                <input
                  type="range"
                  min="0"
                  max="10000"
                  value={state.generationParams.seed}
                  onChange={(e) => updateParam('seed', parseInt(e.target.value))}
                />
              </label>
            </div>
            <div className="param-row">
              <label>
                Steps: {state.generationParams.steps}
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={state.generationParams.steps}
                  onChange={(e) => updateParam('steps', parseInt(e.target.value))}
                />
              </label>
            </div>
            <div className="param-row">
              <label>
                Noise Level: {state.generationParams.noiseLevel}
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={state.generationParams.noiseLevel}
                  onChange={(e) => updateParam('noiseLevel', parseFloat(e.target.value))}
                />
              </label>
            </div>
          </div>
        )}
      </div>

      {/* File Upload */}
      <div className="file-upload-section">
        <label htmlFor="file-upload" className="upload-button">
          <i className="fa-solid fa-upload"></i> Upload Audio/MIDI
        </label>
        <input
          id="file-upload"
          type="file"
          accept="audio/*,.mid,.midi"
          onChange={handleFileUpload}
          style={{ display: 'none' }}
          disabled={isGenerating}
        />

        {state.uploadedFile && (
          <div className="uploaded-file-info">
            <div className="file-info-header">
              <div className="file-details">
                <i className={`fa-solid ${state.fileType === 'midi' ? 'fa-file-audio' : 'fa-file-waveform'}`}></i>
                <div>
                  <div className="file-name">{state.uploadedFile.name}</div>
                  <div className="file-type">{state.fileType?.toUpperCase()} File</div>
                </div>
              </div>
              <button
                className="clear-file-btn"
                onClick={handleClearFile}
                title="Remove file"
                disabled={isGenerating}
              >
                <i className="fa-solid fa-times"></i>
              </button>
            </div>

            {/* Audio preview for audio files */}
            {state.fileType === 'audio' && state.uploadedFile.previewUrl && (
              <div className="audio-preview">
                <audio controls src={state.uploadedFile.previewUrl}>
                  Your browser does not support the audio element.
                </audio>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Error Display */}
      {generationError && (
        <div className="generation-error">
          <i className="fa-solid fa-triangle-exclamation"></i>
          <span>{generationError}</span>
        </div>
      )}

      {/* Generation Progress */}
      {isGenerating && (
        <div className="generation-progress">
          <div className="progress-header">
            <i className="fa-solid fa-spinner fa-spin"></i>
            <span>Generating audio... {generationProgress}%</span>
          </div>
          <div className="progress-bar">
            <div
              className="progress-fill"
              style={{ width: `${generationProgress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Generate Button */}
      <button
        className="generate-button"
        onClick={handleGenerate}
        disabled={isGenerating}
      >
        {isGenerating ? (
          <>
            <i className="fa-solid fa-spinner fa-spin"></i> Generating...
          </>
        ) : (
          <>
            <i className="fa-solid fa-wand-magic-sparkles"></i> Generate
          </>
        )}
      </button>
    </div>
  );
}

export default GenerationPanel;
