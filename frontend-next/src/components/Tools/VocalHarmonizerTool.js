import React, { useState, useRef, useCallback } from 'react';
import ToolWaveform from './ToolWaveform';
import {
  C,
  P,
  Ic,
  ToolShell,
  Panel,
  DropZone,
  ChipRow,
  FieldLabel,
  Slider,
  Select,
} from './toolShell';

const API_BASE = '';

const VOICING_STYLES = [
  { value: 'thirds', label: 'Thirds (Traditional)' },
  { value: 'sixths', label: 'Sixths' },
  { value: 'power', label: 'Power (Rock)' },
  { value: 'close', label: 'Close Harmony' },
  { value: 'wide', label: 'Wide Spread' },
  { value: 'jazz', label: 'Jazz Voicing' },
  { value: 'minor', label: 'Minor Thirds' },
  { value: 'barbershop', label: 'Barbershop' },
];

const KEYS = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B', 'chromatic'];

/**
 * Vocal Harmonizer — generates N harmony voices from a dry vocal.
 * Backend wiring (POST /api/vocal-harmonizer) preserved verbatim from the
 * pre-refactor implementation; only the presentation layer has changed.
 */
const VocalHarmonizerTool = ({ tool, onBack }) => {
  const [audioFile, setAudioFile] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [error, setError] = useState(null);

  const [numHarmonies, setNumHarmonies] = useState(2);
  const [voicingStyle, setVoicingStyle] = useState('thirds');
  const [musicalKey, setMusicalKey] = useState('C');
  const [mode, setMode] = useState('major');
  const [noiseLevel, setNoiseLevel] = useState(0.3);
  const [useAceStep, setUseAceStep] = useState(false);

  const [harmonyAudios, setHarmonyAudios] = useState([]);
  const [harmonyMidi, setHarmonyMidi] = useState(null);
  const abortRef = useRef(null);

  const resetResults = () => {
    setHarmonyAudios([]);
    setHarmonyMidi(null);
  };

  const handleFileUpload = useCallback((file) => {
    if (file && file.type.startsWith('audio/')) {
      setAudioFile(file);
      setError(null);
      setStatusMessage(`File loaded: ${file.name}`);
      resetResults();
    } else {
      setError('Please upload an audio file (WAV, MP3, etc.)');
    }
  }, []);

  const handleRemoveFile = useCallback(() => {
    setAudioFile(null);
    resetResults();
    setStatusMessage('');
    setError(null);
  }, []);

  const handleGenerate = useCallback(async () => {
    if (!audioFile) {
      setError('Please upload an audio file first');
      return;
    }

    setIsGenerating(true);
    setProgress(0);
    setError(null);
    setStatusMessage('Uploading audio and extracting pitch…');

    try {
      const formData = new FormData();
      formData.append('audioFile', audioFile);
      formData.append('numHarmonies', numHarmonies.toString());
      formData.append('voicingStyle', voicingStyle);
      formData.append('key', musicalKey);
      formData.append('mode', mode);
      formData.append('noiseLevel', noiseLevel.toString());
      formData.append('useAceStep', useAceStep.toString());

      setProgress(0.1);
      const controller = new AbortController();
      abortRef.current = controller;
      const response = await fetch(`${API_BASE}/api/vocal-harmonizer`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
      }

      setProgress(0.5);
      setStatusMessage('Generating harmony tracks…');
      const result = await response.json();
      setProgress(1);
      setStatusMessage('Harmonies generated successfully.');

      if (result.harmony_audio && result.harmony_audio.length > 0) {
        setHarmonyAudios(result.harmony_audio);
      }
      if (result.harmony_midi) {
        setHarmonyMidi(result.harmony_midi);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error('Harmony generation failed:', err);
        setError(err.message || 'Failed to generate harmonies');
      }
      setStatusMessage('');
    } finally {
      abortRef.current = null;
      setIsGenerating(false);
    }
  }, [audioFile, numHarmonies, voicingStyle, musicalKey, mode, noiseLevel, useAceStep]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setStatusMessage('Cancelled.');
  }, []);

  const handleDownloadMidi = useCallback(() => {
    if (!harmonyMidi) return;
    const link = document.createElement('a');
    link.href = harmonyMidi;
    link.download = 'harmonies.mid';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [harmonyMidi]);

  // ---------- Panel: Input ----------
  const InputPanel = (
    <Panel
      title="Input · dry vocal"
      marker="◆"
      status={
        audioFile ? (
          <span style={{ color: C.ok, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.ok }} />
            loaded
          </span>
        ) : (
          <span style={{ color: C.inkMute }}>empty</span>
        )
      }
    >
      <DropZone
        file={audioFile}
        onPick={handleFileUpload}
        onRemove={handleRemoveFile}
        accept="audio/*"
        hint="WAV · MP3 · FLAC"
        icon="WAV"
      />
      {audioFile && (
        <div style={{ padding: '0 16px 14px' }}>
          <FieldLabel style={{ marginBottom: 8 }}>Preview · dry</FieldLabel>
          <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
            <ToolWaveform audioUrl={URL.createObjectURL(audioFile)} height={56} color={C.ink} />
          </div>
        </div>
      )}
      {error && (
        <div
          style={{
            margin: '0 14px 14px',
            padding: '8px 12px',
            border: `1px solid ${C.warm}66`,
            background: 'rgba(201,79,44,0.08)',
            color: C.warm,
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.4,
          }}
        >
          {error}
        </div>
      )}
    </Panel>
  );

  // ---------- Panel: Controls ----------
  const ControlsPanel = (
    <Panel title="Controls · harmony engine" marker="◇">
      <div style={{ padding: '12px 16px 0' }}>
        <FieldLabel style={{ marginBottom: 8 }}>Voices</FieldLabel>
        <ChipRow
          options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))}
          value={numHarmonies}
          onChange={setNumHarmonies}
        />
      </div>

      <div style={{ padding: '14px 16px 0' }}>
        <FieldLabel style={{ marginBottom: 8 }}>Voicing style</FieldLabel>
        <Select value={voicingStyle} onChange={setVoicingStyle} options={VOICING_STYLES} />
      </div>

      <div style={{ padding: '14px 16px 0', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        <div>
          <FieldLabel style={{ marginBottom: 6 }}>Key</FieldLabel>
          <Select
            value={musicalKey}
            onChange={setMusicalKey}
            options={KEYS.map((k) => ({ value: k, label: k }))}
          />
        </div>
        <div>
          <FieldLabel style={{ marginBottom: 6 }}>Mode</FieldLabel>
          <Select
            value={mode}
            onChange={setMode}
            options={[
              { value: 'major', label: 'Major' },
              { value: 'minor', label: 'Minor' },
            ]}
          />
        </div>
      </div>

      <Slider
        label="Noise reduction"
        value={Math.round(noiseLevel * 100)}
        min={0}
        max={100}
        onChange={(v) => setNoiseLevel(v / 100)}
        unit="%"
        leftLabel="None"
        rightLabel="Max"
      />

      <div
        style={{
          padding: '10px 16px 16px',
          borderTop: `1px solid ${C.rule}`,
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}
      >
        <input
          id="useAceStep"
          type="checkbox"
          checked={useAceStep}
          onChange={(e) => setUseAceStep(e.target.checked)}
        />
        <label
          htmlFor="useAceStep"
          style={{
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.5,
            color: C.inkSoft,
            textTransform: 'uppercase',
          }}
        >
          ACE-Step natural vocals (slower)
        </label>
      </div>
    </Panel>
  );

  // ---------- Panel: Output ----------
  const OutputPanel = (
    <Panel
      title="Output · harmonies"
      marker="●"
      status={
        harmonyAudios.length > 0 ? (
          <span style={{ color: C.ok }}>{harmonyAudios.length} voice{harmonyAudios.length === 1 ? '' : 's'}</span>
        ) : null
      }
    >
      <div style={{ padding: '6px 16px', flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {harmonyAudios.length === 0 && (
          <div
            style={{
              padding: '40px 16px',
              textAlign: 'center',
              fontFamily: C.mono,
              fontSize: 10,
              letterSpacing: 0.5,
              color: C.inkMute,
              textTransform: 'uppercase',
            }}
          >
            Render a take to see harmony voices here.
          </div>
        )}
        {harmonyAudios.map((audioUrl, index) => (
          <div
            key={index}
            style={{
              display: 'grid',
              gridTemplateColumns: '64px 1fr 24px',
              alignItems: 'center',
              gap: 10,
              padding: '8px 0',
              borderBottom: `1px solid ${C.rule}`,
            }}
          >
            <div>
              <div
                style={{
                  fontFamily: C.mono,
                  fontSize: 10,
                  fontWeight: 500,
                  letterSpacing: 0.5,
                  textTransform: 'uppercase',
                  color: index === 0 ? C.ink : index === 1 ? C.purple : index === 2 ? C.accent : C.warm,
                }}
              >
                V{index + 1}
              </div>
              <div style={{ fontFamily: C.mono, fontSize: 9, letterSpacing: 0.5, color: C.inkMute, marginTop: 1 }}>
                harmony
              </div>
            </div>
            <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 4 }}>
              <ToolWaveform audioUrl={audioUrl} height={28} color={C.ink} />
            </div>
            <a
              href={audioUrl}
              download={`harmony_${index + 1}.wav`}
              title="Download"
              style={{
                width: 24,
                height: 24,
                border: `1px solid ${C.rule}`,
                display: 'grid',
                placeItems: 'center',
                color: C.inkSoft,
              }}
            >
              <Ic d={P.dl} size={11} />
            </a>
          </div>
        ))}
      </div>
      {harmonyMidi && (
        <div style={{ padding: 14, borderTop: `1px solid ${C.rule}` }}>
          <button
            type="button"
            onClick={handleDownloadMidi}
            style={{
              width: '100%',
              padding: '10px 12px',
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
            <Ic d={P.dl} size={12} color={C.bg} /> Download harmony MIDI
          </button>
        </div>
      )}
    </Panel>
  );

  return (
    <ToolShell
      tool={{ ...tool, sku: 'T-01', category: 'Audio tool', version: 'v2.4.1' }}
      subtitle="single-take mode"
      description="Drop a dry vocal. We’ll analyze pitch, key, and timing, then render up to four harmony voices with selectable intervals and voicings."
      meta={[
        { k: 'Runs today', v: '2 / 3 free' },
        { k: 'Avg time', v: '~42s' },
      ]}
      running={isGenerating}
      progress={progress}
      statusMessage={statusMessage}
      primaryLabel="Render take"
      onPrimary={handleGenerate}
      onCancel={handleCancel}
      primaryDisabled={!audioFile}
      onBack={onBack}
      left={InputPanel}
      center={ControlsPanel}
      right={OutputPanel}
    />
  );
};

export default VocalHarmonizerTool;
