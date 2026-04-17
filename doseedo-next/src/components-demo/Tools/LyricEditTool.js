import React, { useState, useCallback } from 'react';
import { generateACEStep, pollACEStepUntilComplete } from '../../services/generationAPI';
import { sendChatMessage } from '../../services/chatAPI';
import ToolWaveform from './ToolWaveform';
import styles from './Tools.module.css';

/**
 * Lyric Edit Tool
 * Edit and generate vocals with AI-assisted lyrics
 */
const LyricEditTool = ({ tool }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  // Input state
  const [prompt, setPrompt] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [aceKey, setAceKey] = useState('C');
  const [aceSteps, setAceSteps] = useState(100);

  const languages = [
    { value: 'english', label: 'English' },
    { value: 'spanish', label: 'Spanish' },
    { value: 'french', label: 'French' },
    { value: 'german', label: 'German' },
    { value: 'mandarin', label: 'Mandarin' },
    { value: 'japanese', label: 'Japanese' }
  ];

  const keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

  // Count syllables helper
  const countSyllables = (word) => {
    word = word.toLowerCase().trim();
    if (word.length === 0) return 0;
    if (word.length <= 3) return 1;
    word = word.replace(/[^a-záéíóúñü'-]/gi, '');
    word = word.replace(/(?:[^laeiouyáéíóúü]es|ed|[^laeiouyáéíóúü]e)$/i, '');
    word = word.replace(/^y/i, '');
    const syllables = word.match(/[aeiouyáéíóúü]{1,2}/gi);
    return syllables ? Math.max(syllables.length, 1) : 1;
  };

  const countLineSyllables = (line) => {
    const cleanLine = line.trim();
    if (cleanLine === '') return 0;
    const words = cleanLine.replace(/[^\w\s'-áéíóúñü]/gi, '').split(/\s+/);
    return words.reduce((count, word) => word.length === 0 ? count : count + countSyllables(word), 0);
  };

  // Handle language translation
  const handleLanguageChange = useCallback(async (languageValue) => {
    setSelectedLanguage(languageValue);

    if (!lyrics || lyrics.trim() === '') return;

    setIsTranslating(true);
    setStatusMessage('Translating lyrics...');

    try {
      const language = languages.find(l => l.value === languageValue)?.label || languageValue;

      // Count syllables for each line
      const lines = lyrics.split('\n');
      const linesWithCounts = lines.map(line => {
        const cleanLine = line.trim();
        if (cleanLine === '') return '';
        const syllableCount = countLineSyllables(cleanLine);
        return `[${syllableCount} syllables] ${cleanLine}`;
      }).join('\n');

      const payload = {
        system_prompt: 'You are a professional lyricist assistant. Translate lyrics while preserving syllable counts and rhythm.',
        daw_context: { bpm: 120, key: aceKey },
        message: `TASK: LYRIC CHANGE\nPAYLOAD:\nLanguage: ${language}\nOriginal Lyrics (with syllable counts per line):\n${linesWithCounts}`,
        conversation_history: []
      };

      const response = await sendChatMessage(payload);

      let translatedLyrics = response?.response || response?.message || response?.content || (typeof response === 'string' ? response : null);

      if (translatedLyrics) {
        // Strip syllable count numbers from end of lines
        const cleanedLines = translatedLyrics.split('\n').map(line => line.replace(/\s*\(\d+\)\s*$/, '').trim());
        setLyrics(cleanedLines.join('\n'));
        setStatusMessage(`Lyrics translated to ${language}`);
      } else {
        setStatusMessage('Translation failed - could not extract lyrics');
      }
    } catch (error) {
      console.error('Translation failed:', error);
      setStatusMessage(`Translation error: ${error.message}`);
    } finally {
      setIsTranslating(false);
    }
  }, [lyrics, aceKey]);

  // Generate vocals with ACE-Step
  const handleGenerate = useCallback(async () => {
    if (!lyrics.trim() && !prompt.trim()) {
      setStatusMessage('Please enter lyrics or a prompt.');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setStatusMessage('Starting ACE-Step generation...');

    try {
      const params = {
        acePrompt: prompt,
        aceLyrics: lyrics,
        aceKey: aceKey,
        aceSteps: aceSteps,
        seed: Math.floor(Math.random() * 1000000),
        t0: 0.8
      };

      const startResult = await generateACEStep(params, null);
      const taskId = startResult.task_id;

      setStatusMessage('Generating vocals...');

      const result = await pollACEStepUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.attempts / 100);
          setStatusMessage(`Generating... Step ${progressData.attempts}`);
        },
        1800
      );

      if (result.file_paths && result.file_paths.length > 0) {
        setGeneratedAudioUrl(result.file_paths[0]);
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
  }, [prompt, lyrics, aceKey, aceSteps]);

  // Download generated audio
  const handleDownload = useCallback(() => {
    if (generatedAudioUrl) {
      const a = document.createElement('a');
      a.href = generatedAudioUrl;
      a.download = `vocals_${Date.now()}.wav`;
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
          <div className={styles.toolGeneratorIcon} style={{ background: 'linear-gradient(135deg, rgba(156, 130, 200, 0.4), rgba(156, 130, 200, 0.2))' }}>
            <i className="fa-solid fa-pen-to-square" style={{ color: '#9c82c8' }}></i>
          </div>
          <div className={styles.toolGeneratorTitleText}>
            <h2 className={styles.toolGeneratorTitle}>{tool.name}</h2>
            <p className={styles.toolGeneratorDescription}>{tool.description}</p>
          </div>
        </div>
      </div>

      {/* Prompt Input */}
      <div className={styles.toolInputSection}>
        <div className={styles.toolInputGroup}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-wand-magic-sparkles"></i>
            Style Prompt
          </label>
          <input
            type="text"
            className={styles.toolTextInput}
            placeholder="Describe the music style... (e.g., 'emotional ballad, female voice')"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
          />
        </div>

        <div className={styles.toolInputGroup}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-music"></i>
            Key
          </label>
          <select
            className={styles.toolSelect}
            value={aceKey}
            onChange={(e) => setAceKey(e.target.value)}
          >
            {keys.map(key => (
              <option key={key} value={key}>{key}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Lyrics Input */}
      <div className={styles.toolSection}>
        <div className={styles.lyricsHeader}>
          <label className={styles.toolInputLabel}>
            <i className="fa-solid fa-align-left"></i>
            Lyrics
          </label>
          <div className={styles.languageSelect}>
            <select
              className={styles.toolSelect}
              value={selectedLanguage}
              onChange={(e) => handleLanguageChange(e.target.value)}
              disabled={isTranslating}
            >
              {languages.map(lang => (
                <option key={lang.value} value={lang.value}>{lang.label}</option>
              ))}
            </select>
            {isTranslating && <i className="fa-solid fa-spinner fa-spin"></i>}
          </div>
        </div>
        <textarea
          className={styles.toolTextarea}
          placeholder="Enter your lyrics here...&#10;Each line will be sung as written."
          value={lyrics}
          onChange={(e) => setLyrics(e.target.value)}
          rows={8}
        />
        <div className={styles.lyricStats}>
          <span>{lyrics.split('\n').filter(l => l.trim()).length} lines</span>
          <span>{countLineSyllables(lyrics)} total syllables</span>
        </div>
      </div>

      {/* Steps Slider */}
      <div className={styles.toolSection}>
        <label className={styles.toolInputLabel}>
          <i className="fa-solid fa-sliders"></i>
          Quality (Steps): {aceSteps}
        </label>
        <input
          type="range"
          className={styles.toolSlider}
          min="50"
          max="200"
          value={aceSteps}
          onChange={(e) => setAceSteps(parseInt(e.target.value))}
        />
        <div className={styles.sliderLabels}>
          <span>Fast (50)</span>
          <span>Balanced (100)</span>
          <span>Quality (200)</span>
        </div>
      </div>

      {/* Waveform Display */}
      <div className={styles.toolWaveformSection}>
        <div className={styles.toolWaveformHeader}>
          <span className={styles.toolWaveformLabel}>
            <i className="fa-solid fa-waveform-lines"></i>
            Generated Vocals
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
          color="#9c82c8"
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
          disabled={isGenerating || isTranslating}
        >
          {isGenerating ? (
            <>
              <i className="fa-solid fa-spinner fa-spin"></i>
              Generating...
            </>
          ) : (
            <>
              <i className="fa-solid fa-bolt"></i>
              Generate Vocals
            </>
          )}
        </button>
      </div>
    </div>
  );
};

export default LyricEditTool;
