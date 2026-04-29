import React, { useState, useRef, useCallback, useMemo } from 'react';
import { useAudioRecorder } from '../../hooks/useAudioRecorder';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import { C, P, Ic, ToolShell, Panel, DropZone, FieldLabel } from './toolShell';

/**
 * Voice to Instrument — hums / voice memos → instrumental audio.
 * Backend: startGeneration + pollUntilComplete (unchanged).
 */
const VoiceToInstrumentTool = ({ tool, onBack }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [progress, setProgress] = useState(0);
  const [statusMessage, setStatusMessage] = useState('');
  const [recordedFile, setRecordedFile] = useState(null);
  const fileInputRef = useRef(null);

  const [instrumentGroup, setInstrumentGroup] = useState('strings');
  const [instrumentSubgroup, setInstrumentSubgroup] = useState('violin');

  const {
    isRecording,
    recordingDuration,
    startRecording,
    stopRecording,
    cancelRecording,
  } = useAudioRecorder({ extractMidi: false, convertToSine: false });

  const instrumentSubgroups = useMemo(
    () => ({
      piano: ['acoustic_piano', 'keys'],
      guitar: ['acoustic_guitar', 'electric_guitar'],
      bass: ['electric_bass', 'upright_bass'],
      strings: ['ensemble_strings', 'violin', 'cello'],
      brass: ['ensemble_brass', 'trumpet', 'trombone'],
      winds: ['ensemble_winds', 'flute', 'sax'],
    }),
    [],
  );

  const instrumentGroups = useMemo(
    () => [
      { id: 'piano', label: 'Piano' },
      { id: 'guitar', label: 'Guitar' },
      { id: 'bass', label: 'Bass' },
      { id: 'strings', label: 'Strings' },
      { id: 'brass', label: 'Brass' },
      { id: 'winds', label: 'Winds' },
    ],
    [],
  );

  const handleGroupChange = useCallback(
    (newGroup) => {
      setInstrumentGroup(newGroup);
      const subs = instrumentSubgroups[newGroup] || instrumentSubgroups.strings;
      setInstrumentSubgroup(subs[0]);
    },
    [instrumentSubgroups],
  );

  const handleStopRecording = useCallback(async () => {
    const file = await stopRecording();
    if (file) {
      setRecordedFile(file);
      setStatusMessage('Recording saved. Press render to convert.');
    }
  }, [stopRecording]);

  const handleFileUpload = (file) => {
    if (!file) return;
    setRecordedFile(file);
    setStatusMessage(`File loaded: ${file.name}`);
  };

  const handleGenerate = useCallback(async () => {
    if (!recordedFile) {
      setStatusMessage('Record or upload audio first.');
      return;
    }
    setIsGenerating(true);
    setProgress(0);
    setStatusMessage('Starting generation…');
    try {
      const params = {
        instrumentGroup,
        instrumentSubgroup,
        monophonicMode: instrumentSubgroup.startsWith('ensemble_'),
        arrangeMode: instrumentSubgroup.startsWith('ensemble_'),
        steps: 100,
        adapterScale: 0.5,
        cfgWeight: 2.5,
        t0: 0.95,
        seed: Math.floor(Math.random() * 1000000),
      };
      const startResult = await startGeneration(params, recordedFile);
      const taskId = startResult.task_id;
      setStatusMessage('Processing audio…');
      const result = await pollUntilComplete(
        taskId,
        (progressData) => {
          setProgress(progressData.progress || 0);
          setStatusMessage(`Processing… ${Math.round((progressData.progress || 0) * 100)}%`);
        },
        null,
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
  }, [recordedFile, instrumentGroup, instrumentSubgroup]);

  const handleDownload = useCallback(() => {
    if (!generatedAudioUrl) return;
    const a = document.createElement('a');
    a.href = generatedAudioUrl;
    a.download = `${instrumentSubgroup}_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [generatedAudioUrl, instrumentSubgroup]);

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const availableSubgroups = instrumentSubgroups[instrumentGroup] || [];

  // ---------- Input panel ----------
  const InputPanel = (
    <Panel
      title="Input · voice / hum"
      marker="◆"
      status={
        recordedFile ? (
          <span style={{ color: C.ok, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span style={{ width: 6, height: 6, borderRadius: '50%', background: C.ok }} />
            loaded
          </span>
        ) : isRecording ? (
          <span style={{ color: C.warm, display: 'inline-flex', alignItems: 'center', gap: 5 }}>
            <span
              style={{
                width: 6,
                height: 6,
                borderRadius: '50%',
                background: C.warm,
                boxShadow: `0 0 6px ${C.warm}`,
              }}
            />
            recording {formatDuration(recordingDuration)}
          </span>
        ) : (
          <span style={{ color: C.inkMute }}>empty</span>
        )
      }
    >
      <div style={{ padding: 14, display: 'grid', gap: 10 }}>
        {!isRecording ? (
          <>
            <button
              type="button"
              onClick={startRecording}
              disabled={isGenerating}
              style={{
                padding: '14px 12px',
                background: C.ink,
                color: C.bg,
                border: 'none',
                fontFamily: C.mono,
                fontSize: 11,
                fontWeight: 600,
                letterSpacing: 0.8,
                textTransform: 'uppercase',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10,
              }}
            >
              <span
                style={{
                  width: 10,
                  height: 10,
                  borderRadius: '50%',
                  background: C.warm,
                  boxShadow: `0 0 6px ${C.warm}`,
                }}
              />
              Start recording
            </button>
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isGenerating}
              style={{
                padding: '10px 12px',
                background: 'transparent',
                color: C.inkSoft,
                border: `1px dashed ${C.ruleStrong}`,
                fontFamily: C.mono,
                fontSize: 10,
                letterSpacing: 0.8,
                textTransform: 'uppercase',
                cursor: 'pointer',
              }}
            >
              Or upload a file
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="audio/*"
              onChange={(e) => handleFileUpload(e.target.files?.[0])}
              style={{ display: 'none' }}
            />
          </>
        ) : (
          <>
            <div
              style={{
                padding: '14px 12px',
                background: C.ink,
                color: C.bg,
                fontFamily: C.mono,
                fontSize: 12,
                letterSpacing: 0.5,
                textAlign: 'center',
              }}
            >
              <div style={{ fontSize: 10, color: C.warm, letterSpacing: 0.8, textTransform: 'uppercase' }}>
                Recording
              </div>
              <div style={{ fontSize: 24, fontWeight: 500, marginTop: 4 }}>
                {formatDuration(recordingDuration)}
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <button
                type="button"
                onClick={handleStopRecording}
                style={{
                  padding: '10px 12px',
                  background: C.ink,
                  color: C.bg,
                  border: 'none',
                  fontFamily: C.mono,
                  fontSize: 10,
                  letterSpacing: 0.8,
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                }}
              >
                Stop
              </button>
              <button
                type="button"
                onClick={cancelRecording}
                style={{
                  padding: '10px 12px',
                  background: 'transparent',
                  color: C.inkSoft,
                  border: `1px solid ${C.rule}`,
                  fontFamily: C.mono,
                  fontSize: 10,
                  letterSpacing: 0.8,
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                }}
              >
                Cancel
              </button>
            </div>
          </>
        )}
      </div>
      {recordedFile && !isRecording && (
        <div style={{ borderTop: `1px solid ${C.rule}` }}>
          <DropZone
            file={recordedFile}
            onPick={handleFileUpload}
            onRemove={() => setRecordedFile(null)}
            accept="audio/*"
            icon="WAV"
          />
        </div>
      )}
    </Panel>
  );

  // ---------- Controls panel ----------
  const ControlsPanel = (
    <Panel title="Controls · instrument" marker="◇">
      <div style={{ padding: '12px 12px 0' }}>
        <FieldLabel style={{ padding: '0 4px 8px' }}>Group</FieldLabel>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2,1fr)', gap: 4 }}>
          {instrumentGroups.map((g) => {
            const active = instrumentGroup === g.id;
            return (
              <button
                key={g.id}
                type="button"
                onClick={() => handleGroupChange(g.id)}
                style={{
                  padding: '8px 10px',
                  background: active ? C.ink : C.bg,
                  color: active ? C.bg : C.ink,
                  border: `1px solid ${active ? C.ink : C.rule}`,
                  fontFamily: C.mono,
                  fontSize: 10,
                  letterSpacing: 0.6,
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                }}
              >
                {g.label}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ padding: '14px 12px 0' }}>
        <FieldLabel style={{ padding: '0 4px 8px' }}>Subgroup</FieldLabel>
        <div style={{ display: 'grid', gap: 4 }}>
          {availableSubgroups.map((sg) => {
            const active = instrumentSubgroup === sg;
            let label = sg
              .split('_')
              .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
              .join(' ');
            if (sg.startsWith('ensemble_')) label = 'Ensemble';
            return (
              <button
                key={sg}
                type="button"
                onClick={() => setInstrumentSubgroup(sg)}
                style={{
                  padding: '9px 12px',
                  background: active ? C.ink : C.bg,
                  color: active ? C.bg : C.ink,
                  border: `1px solid ${active ? C.ink : C.rule}`,
                  textAlign: 'left',
                  fontFamily: C.mono,
                  fontSize: 11,
                  letterSpacing: 0.3,
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                }}
              >
                {label}
              </button>
            );
          })}
        </div>
      </div>

      <div style={{ flex: 1 }} />
      <div
        style={{
          padding: '12px 16px',
          borderTop: `1px solid ${C.rule}`,
          fontFamily: C.mono,
          fontSize: 9,
          letterSpacing: 0.5,
          color: C.inkMute,
          textTransform: 'uppercase',
          lineHeight: 1.6,
        }}
      >
        {instrumentSubgroup.startsWith('ensemble_')
          ? 'Ensemble · monophonic arrange mode'
          : 'Solo · polyphonic mode'}
      </div>
    </Panel>
  );

  // ---------- Output panel ----------
  const OutputPanel = (
    <Panel
      title="Output · rendered instrument"
      marker="●"
      status={generatedAudioUrl ? <span style={{ color: C.ok }}>ready</span> : null}
    >
      <div style={{ padding: 16 }}>
        <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 10, marginBottom: 10 }}>
          <FieldLabel>Preview</FieldLabel>
          <div style={{ marginTop: 8 }}>
            <ToolWaveform audioUrl={generatedAudioUrl} height={100} color={C.accent} />
          </div>
        </div>
      </div>
      <div style={{ flex: 1 }} />
      {generatedAudioUrl && (
        <div style={{ padding: 14, borderTop: `1px solid ${C.rule}` }}>
          <button
            type="button"
            onClick={handleDownload}
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
            <Ic d={P.dl} size={12} color={C.bg} /> Download WAV
          </button>
        </div>
      )}
    </Panel>
  );

  return (
    <ToolShell
      tool={{ ...tool, sku: 'T-04', category: 'Audio tool', version: 'v1.2.0' }}
      subtitle={`${instrumentSubgroup.replace(/_/g, ' ')}`}
      description="Hum or speak a melody. We extract the contour, align it to the scale, and render it with the instrument you picked."
      meta={[{ k: 'Avg time', v: '~30s' }]}
      running={isGenerating}
      progress={progress}
      statusMessage={statusMessage}
      primaryLabel="Render instrument"
      onPrimary={handleGenerate}
      primaryDisabled={!recordedFile || isRecording}
      onBack={onBack}
      left={InputPanel}
      center={ControlsPanel}
      right={OutputPanel}
    />
  );
};

export default VoiceToInstrumentTool;
