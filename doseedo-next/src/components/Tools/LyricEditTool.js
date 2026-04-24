import React, { useState, useCallback } from 'react';
import { generateACEStep, pollACEStepUntilComplete } from '../../services/generationAPI';
import { sendChatMessage } from '../../services/chatAPI';
import ToolWaveform from './ToolWaveform';
import { C, P, Ic, ToolShell, Panel, FieldLabel, Slider, Select, ChipRow } from './toolShell';

const LANGUAGES = [
  { value: 'english', label: 'English' },
  { value: 'spanish', label: 'Spanish' },
  { value: 'french', label: 'French' },
  { value: 'german', label: 'German' },
  { value: 'mandarin', label: 'Mandarin' },
  { value: 'japanese', label: 'Japanese' },
];

const KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function countSyllables(word) {
  word = word.toLowerCase().trim();
  if (word.length === 0) return 0;
  if (word.length <= 3) return 1;
  word = word.replace(/[^a-záéíóúñü'-]/gi, '');
  word = word.replace(/(?:[^laeiouyáéíóúü]es|ed|[^laeiouyáéíóúü]e)$/i, '');
  word = word.replace(/^y/i, '');
  const syllables = word.match(/[aeiouyáéíóúü]{1,2}/gi);
  return syllables ? Math.max(syllables.length, 1) : 1;
}

function countLineSyllables(line) {
  const cleanLine = line.trim();
  if (cleanLine === '') return 0;
  const words = cleanLine.replace(/[^\w\s'-áéíóúñü]/gi, '').split(/\s+/);
  return words.reduce((count, word) => (word.length === 0 ? count : count + countSyllables(word)), 0);
}

/**
 * Lyric Edit — text-in / text-out lyric tool, plus ACE-Step vocal render.
 * Backend: generateACEStep + pollACEStepUntilComplete (unchanged).
 */
const LyricEditTool = ({ tool, onBack }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [isTranslating, setIsTranslating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');

  const [prompt, setPrompt] = useState('');
  const [lyrics, setLyrics] = useState('');
  const [selectedLanguage, setSelectedLanguage] = useState('english');
  const [aceKey, setAceKey] = useState('C');
  const [aceSteps, setAceSteps] = useState(100);

  const handleLanguageChange = useCallback(
    async (languageValue) => {
      setSelectedLanguage(languageValue);
      if (!lyrics || lyrics.trim() === '') return;
      setIsTranslating(true);
      setStatusMessage('Translating lyrics…');
      try {
        const language = LANGUAGES.find((l) => l.value === languageValue)?.label || languageValue;
        const lines = lyrics.split('\n');
        const linesWithCounts = lines
          .map((line) => {
            const cleanLine = line.trim();
            if (cleanLine === '') return '';
            return `[${countLineSyllables(cleanLine)} syllables] ${cleanLine}`;
          })
          .join('\n');
        const payload = {
          system_prompt:
            'You are a professional lyricist assistant. Translate lyrics while preserving syllable counts and rhythm.',
          daw_context: { bpm: 120, key: aceKey },
          message: `TASK: LYRIC CHANGE\nPAYLOAD:\nLanguage: ${language}\nOriginal Lyrics (with syllable counts per line):\n${linesWithCounts}`,
          conversation_history: [],
        };
        const response = await sendChatMessage(payload);
        let translatedLyrics =
          response?.response || response?.message || response?.content ||
          (typeof response === 'string' ? response : null);
        if (translatedLyrics) {
          const cleanedLines = translatedLyrics
            .split('\n')
            .map((line) => line.replace(/\s*\(\d+\)\s*$/, '').trim());
          setLyrics(cleanedLines.join('\n'));
          setStatusMessage(`Lyrics translated to ${language}`);
        } else {
          setStatusMessage('Translation failed — could not extract lyrics.');
        }
      } catch (err) {
        console.error('Translation failed:', err);
        setStatusMessage(`Translation error: ${err.message}`);
      } finally {
        setIsTranslating(false);
      }
    },
    [lyrics, aceKey],
  );

  const handleGenerate = useCallback(async () => {
    if (!lyrics.trim() && !prompt.trim()) {
      setStatusMessage('Enter lyrics or a prompt.');
      return;
    }
    setIsGenerating(true);
    setProgress(0);
    setStatusMessage('Starting ACE-Step generation…');
    try {
      const params = {
        acePrompt: prompt,
        aceLyrics: lyrics,
        aceKey,
        aceSteps,
        seed: Math.floor(Math.random() * 1000000),
        t0: 0.8,
      };
      const startResult = await generateACEStep(params, null);
      const taskId = startResult.task_id;
      setStatusMessage('Generating vocals…');
      const result = await pollACEStepUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.attempts / 100);
          setStatusMessage(`Generating… Step ${progressData.attempts}`);
        },
        1800,
      );
      if (result.file_paths && result.file_paths.length > 0) {
        setGeneratedAudioUrl(result.file_paths[0]);
        setStatusMessage('Generation complete.');
      } else {
        setStatusMessage('Completed but no audio returned.');
      }
    } catch (err) {
      console.error('Generation failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  }, [prompt, lyrics, aceKey, aceSteps]);

  const handleDownload = useCallback(() => {
    if (!generatedAudioUrl) return;
    const a = document.createElement('a');
    a.href = generatedAudioUrl;
    a.download = `vocals_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [generatedAudioUrl]);

  const lineCount = lyrics.split('\n').filter((l) => l.trim()).length;
  const totalSyllables = countLineSyllables(lyrics.replace(/\n/g, ' '));

  // ---------- Wide body: lyrics editor + side controls ----------
  const body = (
    <div
      style={{
        display: 'grid',
        gridTemplateColumns: 'minmax(0, 2fr) minmax(260px, 1fr)',
        gap: 12,
        minHeight: 400,
      }}
    >
      {/* Left: lyrics editor */}
      <Panel
        title="Lyrics · editor"
        marker="◆"
        status={
          <span style={{ color: C.inkMute }}>
            {lineCount} lines · {totalSyllables} syllables
          </span>
        }
      >
        <div style={{ padding: '12px 16px', display: 'flex', gap: 10, alignItems: 'center' }}>
          <FieldLabel style={{ flexShrink: 0 }}>Language</FieldLabel>
          <Select
            value={selectedLanguage}
            onChange={handleLanguageChange}
            options={LANGUAGES}
            style={{ flex: 1 }}
          />
          {isTranslating && (
            <span
              style={{
                fontFamily: C.mono,
                fontSize: 10,
                color: C.warm,
                letterSpacing: 0.5,
                textTransform: 'uppercase',
              }}
            >
              translating…
            </span>
          )}
        </div>
        <div style={{ padding: '0 16px 16px', flex: 1, display: 'flex', flexDirection: 'column' }}>
          <textarea
            placeholder="Enter your lyrics here… each line will be sung as written."
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            style={{
              flex: 1,
              minHeight: 280,
              padding: '14px 16px',
              background: C.bg,
              border: `1px solid ${C.rule}`,
              color: C.ink,
              fontFamily: C.mono,
              fontSize: 13,
              lineHeight: 1.65,
              letterSpacing: 0.2,
              resize: 'vertical',
              outline: 'none',
            }}
          />
        </div>
      </Panel>

      {/* Right: prompt + key + quality + output */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12, minWidth: 0 }}>
        <Panel title="Controls · vocal render" marker="◇">
          <div style={{ padding: '12px 16px' }}>
            <FieldLabel style={{ marginBottom: 6 }}>Style prompt</FieldLabel>
            <input
              type="text"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="e.g. emotional ballad, female voice"
              style={{
                width: '100%',
                padding: '8px 10px',
                background: C.bg,
                border: `1px solid ${C.rule}`,
                color: C.ink,
                fontFamily: C.mono,
                fontSize: 11,
                letterSpacing: 0.2,
              }}
            />
          </div>
          <div style={{ padding: '0 16px 4px' }}>
            <FieldLabel style={{ marginBottom: 6 }}>Key</FieldLabel>
            <ChipRow
              options={KEYS.map((k) => ({ value: k, label: k }))}
              value={aceKey}
              onChange={setAceKey}
            />
          </div>
          <Slider
            label="Quality (steps)"
            value={aceSteps}
            min={50}
            max={200}
            step={10}
            onChange={setAceSteps}
            leftLabel="Fast"
            rightLabel="Quality"
          />
        </Panel>

        <Panel
          title="Output · vocals"
          marker="●"
          status={generatedAudioUrl ? <span style={{ color: C.ok }}>ready</span> : null}
        >
          <div style={{ padding: 14 }}>
            <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
              <ToolWaveform audioUrl={generatedAudioUrl} height={60} color={C.purple} />
            </div>
            {generatedAudioUrl && (
              <button
                type="button"
                onClick={handleDownload}
                style={{
                  width: '100%',
                  marginTop: 10,
                  padding: '9px 12px',
                  background: C.ink,
                  color: C.bg,
                  border: 'none',
                  fontFamily: C.mono,
                  fontSize: 10,
                  fontWeight: 600,
                  letterSpacing: 0.8,
                  textTransform: 'uppercase',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: 8,
                  cursor: 'pointer',
                }}
              >
                <Ic d={P.dl} size={12} color={C.bg} /> Download WAV
              </button>
            )}
          </div>
        </Panel>
      </div>
    </div>
  );

  return (
    <ToolShell
      tool={{ ...tool, sku: 'T-03', category: 'Lyrics tool', version: 'v1.0.0' }}
      subtitle="ACE-Step vocal render"
      description="Edit lyrics, translate while preserving syllable count, then render with the ACE-Step vocal engine."
      meta={[{ k: 'Avg time', v: '~5s translate · ~90s render' }]}
      running={isGenerating}
      progress={progress}
      statusMessage={statusMessage}
      primaryLabel="Render vocals"
      onPrimary={handleGenerate}
      primaryDisabled={isTranslating || (!lyrics.trim() && !prompt.trim())}
      onBack={onBack}
      layout="wide"
      body={body}
    />
  );
};

export default LyricEditTool;
