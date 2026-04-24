import React, { useState, useCallback } from 'react';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import { C, P, Ic, ToolShell, Panel, DropZone, FieldLabel, Slider } from './toolShell';

/**
 * Sample Regenerator — audio-in, audio-out: re-roll a sample with varying
 * denoise strength. Backend: startGeneration + pollUntilComplete.
 */
const SampleRegeneratorTool = ({ tool, onBack }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedUrl, setUploadedUrl] = useState(null);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [progress, setProgress] = useState(0);

  const [denoiseStrength, setDenoiseStrength] = useState(0.5);
  const [steps, setSteps] = useState(100);
  const [cfgWeight, setCfgWeight] = useState(2.5);
  const [seed, setSeed] = useState(() => Math.floor(Math.random() * 1000000));

  const handleFileUpload = useCallback((file) => {
    if (!file) return;
    setUploadedFile(file);
    setUploadedUrl(URL.createObjectURL(file));
    setGeneratedAudioUrl(null);
    setStatusMessage(`Loaded: ${file.name}`);
  }, []);

  const handleRemove = useCallback(() => {
    setUploadedFile(null);
    setUploadedUrl(null);
    setGeneratedAudioUrl(null);
    setStatusMessage('');
  }, []);

  const handleRegenerate = useCallback(async () => {
    if (!uploadedFile) {
      setStatusMessage('Upload an audio file first.');
      return;
    }
    setIsGenerating(true);
    setStatusMessage('Starting regeneration…');
    setProgress(0);
    try {
      const params = {
        t0: denoiseStrength,
        steps,
        cfgWeight,
        seed,
        noiseLevel: 1 - denoiseStrength,
        adapterScale: 0.5,
        monophonicMode: false,
      };
      const startResult = await startGeneration(params, uploadedFile);
      const taskId = startResult.task_id;
      setStatusMessage('Regenerating audio…');
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
        setStatusMessage('Regeneration complete.');
      } else {
        setStatusMessage('Completed but no audio returned.');
      }
    } catch (err) {
      console.error('Regeneration failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsGenerating(false);
      setProgress(0);
    }
  }, [uploadedFile, denoiseStrength, steps, cfgWeight, seed]);

  const handleRandomizeSeed = useCallback(() => {
    setSeed(Math.floor(Math.random() * 1000000));
  }, []);

  const handleDownload = useCallback(() => {
    if (!generatedAudioUrl) return;
    const a = document.createElement('a');
    a.href = generatedAudioUrl;
    a.download = `regenerated_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [generatedAudioUrl]);

  // ---------- Input ----------
  const InputPanel = (
    <Panel
      title="Input · original"
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
        icon="SMP"
      />
      {uploadedUrl && (
        <div style={{ padding: '0 16px 14px' }}>
          <FieldLabel style={{ marginBottom: 8 }}>Source</FieldLabel>
          <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
            <ToolWaveform audioUrl={uploadedUrl} height={56} color={C.ink} />
          </div>
        </div>
      )}
    </Panel>
  );

  // ---------- Controls ----------
  const ControlsPanel = (
    <Panel title="Controls · regen engine" marker="◇">
      <Slider
        label="Denoise strength"
        value={Math.round(denoiseStrength * 100)}
        min={0}
        max={100}
        step={5}
        onChange={(v) => setDenoiseStrength(v / 100)}
        unit="%"
        leftLabel="Subtle"
        rightLabel="Strong"
      />
      <Slider
        label="Quality (steps)"
        value={steps}
        min={50}
        max={200}
        step={10}
        onChange={setSteps}
        leftLabel="Fast"
        rightLabel="Quality"
      />
      <Slider
        label="CFG weight"
        value={Number(cfgWeight.toFixed(1))}
        min={1}
        max={5}
        step={0.1}
        onChange={setCfgWeight}
        leftLabel="Creative"
        rightLabel="Faithful"
      />
      <div
        style={{
          padding: '14px 16px',
          borderTop: `1px solid ${C.rule}`,
          display: 'grid',
          gridTemplateColumns: '1fr auto',
          gap: 8,
          alignItems: 'end',
        }}
      >
        <div>
          <FieldLabel style={{ marginBottom: 6 }}>Seed</FieldLabel>
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(parseInt(e.target.value) || 0)}
            style={{
              width: '100%',
              padding: '7px 10px',
              background: C.bg,
              border: `1px solid ${C.rule}`,
              color: C.ink,
              fontFamily: C.mono,
              fontSize: 11,
              letterSpacing: 0.3,
            }}
          />
        </div>
        <button
          type="button"
          onClick={handleRandomizeSeed}
          title="Randomize seed"
          style={{
            padding: '8px 10px',
            border: `1px solid ${C.rule}`,
            background: C.bg,
            color: C.inkSoft,
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.8,
            textTransform: 'uppercase',
            cursor: 'pointer',
          }}
        >
          Random
        </button>
      </div>
    </Panel>
  );

  // ---------- Output ----------
  const OutputPanel = (
    <Panel
      title="Output · regenerated"
      marker="●"
      status={generatedAudioUrl ? <span style={{ color: C.ok }}>ready</span> : null}
    >
      <div style={{ padding: 16 }}>
        <FieldLabel style={{ marginBottom: 8 }}>A · Original</FieldLabel>
        <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8, marginBottom: 12 }}>
          <ToolWaveform audioUrl={uploadedUrl} height={40} color={C.inkMute} />
        </div>
        <FieldLabel style={{ marginBottom: 8 }}>B · Regen</FieldLabel>
        <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
          <ToolWaveform audioUrl={generatedAudioUrl} height={60} color={C.purple} />
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
      tool={{ ...tool, sku: 'T-05', category: 'Audio tool · Beta', version: 'v0.4.0' }}
      subtitle="re-roll a sample"
      description="Upload a sample. Pick how faithful vs. creative you want the re-roll, and how many steps of denoising to apply."
      meta={[
        { k: 'Status', v: 'Beta' },
        { k: 'Avg time', v: '~25s' },
      ]}
      running={isGenerating}
      progress={progress}
      statusMessage={statusMessage}
      primaryLabel="Regenerate"
      onPrimary={handleRegenerate}
      primaryDisabled={!uploadedFile}
      onBack={onBack}
      left={InputPanel}
      center={ControlsPanel}
      right={OutputPanel}
    />
  );
};

export default SampleRegeneratorTool;
