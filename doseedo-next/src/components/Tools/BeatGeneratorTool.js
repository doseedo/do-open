import React, { useState, useCallback } from 'react';
import { generateAndDownloadDrumSamples } from '../../services/drumSamplerAPI';
import { generateDrums, generateRisers } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * Beat Generator Tool
 * Generate custom beats and drum patterns with AI
 */
const BeatGeneratorTool = ({ tool }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [drumKit, setDrumKit] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [midiFile, setMidiFile] = useState(null);

  // Generation parameters
  const [bpm, setBpm] = useState(120);
  const [pattern, setPattern] = useState(4); // 1, 2, or 4 bars
  const [generationType, setGenerationType] = useState('random'); // 'random' or 'orchestral'
  const [includeRisers, setIncludeRisers] = useState(false);

  const patternOptions = [
    { value: 1, label: '1 Bar' },
    { value: 2, label: '2 Bars' },
    { value: 4, label: '4 Bars' }
  ];

  // Generate random drum pattern
  const handleGenerateRandom = useCallback(async () => {
    setIsGenerating(true);
    setStatusMessage('Generating random drum pattern...');

    try {
      const result = await generateAndDownloadDrumSamples(bpm);

      if (result.audioUrl) {
        setGeneratedAudioUrl(result.audioUrl);
        setDrumKit(result.drumKit);
        setMidiFile(result.midiFile);
        setStatusMessage(`Generated: ${result.midiFile} at ${bpm} BPM`);
      } else {
        setStatusMessage('Generation completed but no audio returned.');
      }

    } catch (error) {
      console.error('Drum generation failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  }, [bpm]);

  // Generate orchestral drums
  const handleGenerateOrchestral = useCallback(async () => {
    setIsGenerating(true);
    setStatusMessage('Generating orchestral drums...');

    try {
      const params = {
        tempo: bpm,
        pattern: pattern
      };

      const result = await generateDrums(params);

      if (result.audio_url) {
        setGeneratedAudioUrl(result.audio_url);
        setStatusMessage(`Generated orchestral drums at ${bpm} BPM`);

        // Generate risers if enabled
        if (includeRisers) {
          setStatusMessage('Generating risers...');
          const riserResult = await generateRisers(params);
          console.log('Risers generated:', riserResult);
        }
      } else {
        setStatusMessage('Generation completed but no audio returned.');
      }

    } catch (error) {
      console.error('Orchestral drum generation failed:', error);
      setStatusMessage(`Error: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  }, [bpm, pattern, includeRisers]);

  // Main generate handler
  const handleGenerate = useCallback(async () => {
    if (generationType === 'random') {
      await handleGenerateRandom();
    } else {
      await handleGenerateOrchestral();
    }
  }, [generationType, handleGenerateRandom, handleGenerateOrchestral]);

  // Download generated audio
  const handleDownload = useCallback(() => {
    if (generatedAudioUrl) {
      const a = document.createElement('a');
      a.href = generatedAudioUrl;
      a.download = `drums_${bpm}bpm_${Date.now()}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  }, [generatedAudioUrl, bpm]);

  return (
    <div className={styles.toolGeneratorContainer}>
      {/* Tool Header */}
      <div className={styles.toolGeneratorHeader}>
        <div className={styles.toolGeneratorTitleSection}>
          <div className={styles.toolGeneratorIcon} style={{ background: 'linear-gradient(135deg, rgba(102, 126, 234, 0.4), rgba(102, 126, 234, 0.2))' }}>
            <i className="fa-solid fa-drum" style={{ color: '#667eea' }}></i>
          </div>
          <div className={styles.toolGeneratorTitleText}>
            <h2 className={styles.toolGeneratorTitle}>{tool.name}</h2>
            <p className={styles.toolGeneratorDescription}>{tool.description}</p>
          </div>
        </div>
      </div>

      {/* Generation Type */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-wand-magic-sparkles"></i>
          Generation Type
        </label>
        <div className={styles.generationTypeGrid}>
          <button
            className={`${styles.generationTypeBtn} ${generationType === 'random' ? styles.active : ''}`}
            onClick={() => setGenerationType('random')}
          >
            <i className="fa-solid fa-dice"></i>
            <span>Random Pattern</span>
            <span className={styles.typeDescription}>AI-generated drum samples</span>
          </button>
          <button
            className={`${styles.generationTypeBtn} ${generationType === 'orchestral' ? styles.active : ''}`}
            onClick={() => setGenerationType('orchestral')}
          >
            <i className="fa-solid fa-drum-steelpan"></i>
            <span>Orchestral</span>
            <span className={styles.typeDescription}>Cinematic percussion</span>
          </button>
        </div>
      </div>

      {/* BPM Input */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-gauge-high"></i>
          Tempo (BPM)
        </label>
        <div className={styles.bpmInput}>
          <button
            className={styles.bpmBtn}
            onClick={() => setBpm(Math.max(60, bpm - 5))}
          >
            <i className="fa-solid fa-minus"></i>
          </button>
          <input
            type="number"
            className={styles.bpmValue}
            value={bpm}
            onChange={(e) => setBpm(Math.max(60, Math.min(200, parseInt(e.target.value) || 120)))}
            min="60"
            max="200"
          />
          <button
            className={styles.bpmBtn}
            onClick={() => setBpm(Math.min(200, bpm + 5))}
          >
            <i className="fa-solid fa-plus"></i>
          </button>
        </div>
        <input
          type="range"
          className={styles.toolSlider}
          min="60"
          max="200"
          value={bpm}
          onChange={(e) => setBpm(parseInt(e.target.value))}
        />
        <div className={styles.sliderLabels}>
          <span>60</span>
          <span>120</span>
          <span>200</span>
        </div>
      </div>

      {/* Pattern Length (for orchestral) */}
      {generationType === 'orchestral' && (
        <div className={styles.toolSection}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-bars-staggered"></i>
            Pattern Length
          </label>
          <div className={styles.patternGrid}>
            {patternOptions.map(opt => (
              <button
                key={opt.value}
                className={`${styles.patternBtn} ${pattern === opt.value ? styles.active : ''}`}
                onClick={() => setPattern(opt.value)}
              >
                {opt.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Options */}
      {generationType === 'orchestral' && (
        <div className={styles.toolOptionsRow}>
          <label className={styles.toolCheckbox}>
            <input
              type="checkbox"
              checked={includeRisers}
              onChange={(e) => setIncludeRisers(e.target.checked)}
            />
            <span>Include risers and transitions</span>
          </label>
        </div>
      )}

      {/* Drum Kit Info */}
      {drumKit && (
        <div className={styles.drumKitInfo}>
          <i className="fa-solid fa-drum"></i>
          <span>Kit: {drumKit}</span>
        </div>
      )}

      {/* Waveform Display */}
      <div className={styles.toolWaveformSection}>
        <div className={styles.toolWaveformHeader}>
          <span className={styles.toolWaveformLabel}>
            <i className="fa-solid fa-waveform-lines"></i>
            Generated Beat
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

      {/* Generate Button */}
      <div className={styles.toolControlSection}>
        <div className={styles.toolControlRow}>
          <button
            className={`${styles.toolControlBtn} ${styles.toolControlBtnSecondary}`}
            onClick={handleGenerate}
            disabled={isGenerating}
          >
            <i className="fa-solid fa-dice"></i>
            Randomize
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
              Generate Beat
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default BeatGeneratorTool;
