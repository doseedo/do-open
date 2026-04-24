import React, { useState, useCallback } from 'react';
import { separateStems } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import { C, P, Ic, ToolShell, Panel, DropZone, FieldLabel } from './toolShell';

const STEM_TYPES = [
  { id: 'vocals', label: 'Vocals', color: C.warm },
  { id: 'drums', label: 'Drums', color: '#d58a2c' },
  { id: 'bass', label: 'Bass', color: C.ok },
  { id: 'other', label: 'Other', color: C.accent },
  { id: 'accompaniment', label: 'Instrumental', color: C.purple },
];

/**
 * Stem Separation — uploads audio, POSTs /api/upload-audio, then calls
 * separateStems. Result shape: { stems: { vocals, drums, bass, other,
 * accompaniment } } — preserved verbatim.
 */
const StemSeparationTool = ({ tool, onBack }) => {
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [stems, setStems] = useState(null);
  const [selectedStem, setSelectedStem] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');

  const handleFileUpload = useCallback((file) => {
    if (!file) return;
    setUploadedFile(file);
    setUploadedUrl(URL.createObjectURL(file));
    setStems(null);
    setSelectedStem(null);
    setStatusMessage(`Loaded: ${file.name}`);
  }, []);

  const handleRemove = useCallback(() => {
    setUploadedFile(null);
    setUploadedUrl(null);
    setStems(null);
    setSelectedStem(null);
    setStatusMessage('');
  }, []);

  const handleSeparate = useCallback(async () => {
    if (!uploadedFile) {
      setStatusMessage('Upload an audio file first.');
      return;
    }
    setIsProcessing(true);
    setStatusMessage('Separating stems… this may take a few minutes.');
    setStems(null);
    try {
      const formData = new FormData();
      formData.append('audioFile', uploadedFile);
      const uploadResponse = await fetch('/api/upload-audio', { method: 'POST', body: formData });
      let audioUrl;
      if (uploadResponse.ok) {
        const uploadResult = await uploadResponse.json();
        audioUrl = uploadResult.url;
      } else {
        audioUrl = uploadedUrl;
      }
      const result = await separateStems(audioUrl);
      if (result.stems) {
        setStems(result.stems);
        setStatusMessage('Separation complete. Click a stem to preview.');
      } else {
        setStatusMessage('Separation completed but no stems returned.');
      }
    } catch (err) {
      console.error('Stem separation failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsProcessing(false);
    }
  }, [uploadedFile, uploadedUrl]);

  const handleDownloadStem = useCallback(
    (stemType) => {
      if (!stems || !stems[stemType]) return;
      const a = document.createElement('a');
      a.href = stems[stemType];
      a.download = `${uploadedFile?.name?.replace(/\.[^/.]+$/, '') || 'audio'}_${stemType}.wav`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    },
    [stems, uploadedFile],
  );

  const handleDownloadAll = useCallback(() => {
    if (!stems) return;
    Object.keys(stems).forEach((st, idx) => {
      setTimeout(() => handleDownloadStem(st), 400 * idx);
    });
  }, [stems, handleDownloadStem]);

  // ---------- Input ----------
  const InputPanel = (
    <Panel
      title="Input · mix"
      marker="◆"
      status={
        uploadedFile ? (
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
        file={uploadedFile}
        onPick={handleFileUpload}
        onRemove={handleRemove}
        accept="audio/*"
        hint="MP3 · WAV · FLAC"
        icon="MIX"
      />
      {uploadedUrl && (
        <div style={{ padding: '0 16px 14px' }}>
          <FieldLabel style={{ marginBottom: 8 }}>Source waveform</FieldLabel>
          <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
            <ToolWaveform audioUrl={uploadedUrl} height={48} color={C.ink} />
          </div>
        </div>
      )}
    </Panel>
  );

  // ---------- Controls ----------
  const ControlsPanel = (
    <Panel title="Controls · separation" marker="◇">
      <div style={{ padding: '14px 16px' }}>
        <FieldLabel style={{ marginBottom: 8 }}>Model</FieldLabel>
        <div
          style={{
            padding: '9px 12px',
            background: C.ink,
            color: C.bg,
            border: `1px solid ${C.ink}`,
            fontFamily: C.mono,
            fontSize: 11,
            letterSpacing: 0.4,
            textTransform: 'uppercase',
          }}
        >
          Demucs v4 (default)
        </div>
        <div
          style={{
            fontFamily: C.mono,
            fontSize: 9,
            letterSpacing: 0.4,
            color: C.inkMute,
            marginTop: 6,
            textTransform: 'uppercase',
          }}
        >
          Separates into vocals · drums · bass · other · instrumental
        </div>
      </div>
      <div style={{ padding: '0 16px 16px' }}>
        <FieldLabel style={{ marginBottom: 8 }}>Expected stems</FieldLabel>
        <div style={{ display: 'grid', gap: 6 }}>
          {STEM_TYPES.map((s) => (
            <div
              key={s.id}
              style={{
                display: 'grid',
                gridTemplateColumns: '12px 1fr',
                alignItems: 'center',
                gap: 10,
                fontFamily: C.mono,
                fontSize: 10,
                letterSpacing: 0.4,
                color: C.inkSoft,
                textTransform: 'uppercase',
              }}
            >
              <span style={{ width: 8, height: 8, background: s.color, borderRadius: '50%' }} />
              {s.label}
            </div>
          ))}
        </div>
      </div>
    </Panel>
  );

  // ---------- Output ----------
  const OutputPanel = (
    <Panel
      title="Output · stems"
      marker="●"
      status={stems ? <span style={{ color: C.ok }}>{Object.keys(stems).length} stems</span> : null}
    >
      <div style={{ padding: '6px 16px', flex: 1, overflowY: 'auto', minHeight: 0 }}>
        {!stems && (
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
            Press separate to extract stems.
          </div>
        )}
        {stems &&
          STEM_TYPES.map((stem) => {
            const url = stems[stem.id];
            if (!url) return null;
            const isSel = selectedStem === stem.id;
            return (
              <div
                key={stem.id}
                style={{
                  display: 'grid',
                  gridTemplateColumns: '72px 1fr 24px',
                  alignItems: 'center',
                  gap: 10,
                  padding: '8px 0',
                  borderBottom: `1px solid ${C.rule}`,
                  cursor: 'pointer',
                  background: isSel ? 'rgba(170,176,238,.08)' : 'transparent',
                }}
                onClick={() => setSelectedStem(stem.id)}
              >
                <div>
                  <div
                    style={{
                      fontFamily: C.mono,
                      fontSize: 10,
                      fontWeight: 500,
                      letterSpacing: 0.5,
                      textTransform: 'uppercase',
                      color: stem.color,
                    }}
                  >
                    {stem.label}
                  </div>
                  <div
                    style={{
                      fontFamily: C.mono,
                      fontSize: 9,
                      letterSpacing: 0.5,
                      color: C.inkMute,
                      marginTop: 1,
                    }}
                  >
                    {stem.id}
                  </div>
                </div>
                <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 4 }}>
                  <ToolWaveform audioUrl={url} height={24} color={stem.color} />
                </div>
                <button
                  type="button"
                  onClick={(e) => {
                    e.stopPropagation();
                    handleDownloadStem(stem.id);
                  }}
                  title="Download"
                  style={{
                    width: 24,
                    height: 24,
                    border: `1px solid ${C.rule}`,
                    background: 'transparent',
                    display: 'grid',
                    placeItems: 'center',
                    color: C.inkSoft,
                    cursor: 'pointer',
                  }}
                >
                  <Ic d={P.dl} size={11} />
                </button>
              </div>
            );
          })}
      </div>
      {stems && (
        <div style={{ padding: 14, borderTop: `1px solid ${C.rule}` }}>
          <button
            type="button"
            onClick={handleDownloadAll}
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
            <Ic d={P.dl} size={12} color={C.bg} /> Download all stems
          </button>
        </div>
      )}
    </Panel>
  );

  return (
    <ToolShell
      tool={{ ...tool, sku: 'T-06', category: 'Audio tool', version: 'v4.0.0' }}
      subtitle="Demucs v4"
      description="Drop any audio file. We’ll split it into vocals, drums, bass, other, and an instrumental mix. Studio-grade separation."
      meta={[{ k: 'Avg time', v: '~60s per minute' }]}
      running={isProcessing}
      progress={isProcessing ? 0.55 : 0}
      statusMessage={statusMessage}
      primaryLabel="Separate stems"
      onPrimary={handleSeparate}
      primaryDisabled={!uploadedFile}
      onBack={onBack}
      left={InputPanel}
      center={ControlsPanel}
      right={OutputPanel}
    />
  );
};

export default StemSeparationTool;
