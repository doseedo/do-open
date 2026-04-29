import React, { useState, useCallback } from 'react';
import { generateAndDownloadDrumSamples } from '../../services/drumSamplerAPI';
import { generateDrums, generateRisers } from '../../services/generationAPI';
import ToolWaveform from './ToolWaveform';
import { C, P, Ic, ToolShell, Panel, FieldLabel, Slider, ChipRow } from './toolShell';

/**
 * Beat Generator — no file input. Picks a generation style + BPM, calls
 * drumSamplerAPI (random pattern) or generateDrums (orchestral). Backend
 * wiring untouched.
 */
const BeatGeneratorTool = ({ tool, onBack }) => {
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedAudioUrl, setGeneratedAudioUrl] = useState(null);
  const [statusMessage, setStatusMessage] = useState('');
  const [drumKit, setDrumKit] = useState(null);
  // eslint-disable-next-line no-unused-vars
  const [midiFile, setMidiFile] = useState(null);

  const [bpm, setBpm] = useState(120);
  const [pattern, setPattern] = useState(4);
  const [generationType, setGenerationType] = useState('random');
  const [includeRisers, setIncludeRisers] = useState(false);

  const handleGenerateRandom = useCallback(async () => {
    setIsGenerating(true);
    setStatusMessage('Generating random drum pattern…');
    try {
      const result = await generateAndDownloadDrumSamples(bpm);
      if (result.audioUrl) {
        setGeneratedAudioUrl(result.audioUrl);
        setDrumKit(result.drumKit);
        setMidiFile(result.midiFile);
        setStatusMessage(`Generated · ${result.midiFile} · ${bpm} BPM`);
      } else {
        setStatusMessage('Completed but no audio returned.');
      }
    } catch (err) {
      console.error('Drum generation failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsGenerating(false);
    }
  }, [bpm]);

  const handleGenerateOrchestral = useCallback(async () => {
    setIsGenerating(true);
    setStatusMessage('Generating orchestral drums…');
    try {
      const params = { tempo: bpm, pattern };
      const result = await generateDrums(params);
      if (result.audio_url) {
        setGeneratedAudioUrl(result.audio_url);
        setStatusMessage(`Generated orchestral drums · ${bpm} BPM`);
        if (includeRisers) {
          setStatusMessage('Generating risers…');
          await generateRisers(params);
        }
      } else {
        setStatusMessage('Completed but no audio returned.');
      }
    } catch (err) {
      console.error('Orchestral drum generation failed:', err);
      setStatusMessage(`Error: ${err.message}`);
    } finally {
      setIsGenerating(false);
    }
  }, [bpm, pattern, includeRisers]);

  const handleGenerate = useCallback(async () => {
    if (generationType === 'random') {
      await handleGenerateRandom();
    } else {
      await handleGenerateOrchestral();
    }
  }, [generationType, handleGenerateRandom, handleGenerateOrchestral]);

  const handleDownload = useCallback(() => {
    if (!generatedAudioUrl) return;
    const a = document.createElement('a');
    a.href = generatedAudioUrl;
    a.download = `drums_${bpm}bpm_${Date.now()}.wav`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }, [generatedAudioUrl, bpm]);

  // ---------- Input (style picker) ----------
  const InputPanel = (
    <Panel title="Input · generation style" marker="◆">
      <div style={{ padding: 14, display: 'grid', gap: 10 }}>
        {[
          {
            id: 'random',
            label: 'Random pattern',
            desc: 'AI-generated drum samples over a rolled pattern.',
          },
          {
            id: 'orchestral',
            label: 'Orchestral',
            desc: 'Cinematic percussion — toms, timpani, ensemble hits.',
          },
        ].map((opt) => {
          const active = generationType === opt.id;
          return (
            <button
              key={opt.id}
              type="button"
              onClick={() => setGenerationType(opt.id)}
              style={{
                padding: '14px 14px',
                background: active ? C.ink : C.bg,
                color: active ? C.bg : C.ink,
                border: `1px solid ${active ? C.ink : C.rule}`,
                textAlign: 'left',
                cursor: 'pointer',
              }}
            >
              <div
                style={{
                  fontFamily: C.mono,
                  fontSize: 11,
                  fontWeight: 500,
                  letterSpacing: 0.3,
                  textTransform: 'uppercase',
                }}
              >
                {opt.label}
              </div>
              <div
                style={{
                  fontFamily: C.sans,
                  fontSize: 11,
                  color: active ? 'rgba(232,230,225,.7)' : C.inkSoft,
                  marginTop: 4,
                  lineHeight: 1.5,
                }}
              >
                {opt.desc}
              </div>
            </button>
          );
        })}
      </div>
      {drumKit && (
        <div
          style={{
            margin: '0 14px 14px',
            padding: '10px 12px',
            background: C.ink,
            color: C.lcd,
            fontFamily: C.mono,
            fontSize: 10,
            letterSpacing: 0.6,
            display: 'flex',
            justifyContent: 'space-between',
          }}
        >
          <span>Last kit</span>
          <span>{drumKit}</span>
        </div>
      )}
    </Panel>
  );

  // ---------- Controls ----------
  const ControlsPanel = (
    <Panel title="Controls · tempo & pattern" marker="◇">
      <div style={{ padding: '14px 16px' }}>
        <FieldLabel style={{ marginBottom: 6 }}>Tempo</FieldLabel>
        <div style={{ display: 'grid', gridTemplateColumns: '36px 1fr 36px', gap: 6, alignItems: 'center' }}>
          <button
            type="button"
            onClick={() => setBpm(Math.max(60, bpm - 5))}
            style={{
              padding: '6px 0',
              background: C.bg,
              border: `1px solid ${C.rule}`,
              color: C.ink,
              fontFamily: C.mono,
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            −
          </button>
          <input
            type="number"
            value={bpm}
            onChange={(e) =>
              setBpm(Math.max(60, Math.min(200, parseInt(e.target.value) || 120)))
            }
            min="60"
            max="200"
            style={{
              width: '100%',
              padding: '6px 10px',
              background: C.ink,
              color: C.lcd,
              border: `1px solid ${C.ink}`,
              textAlign: 'center',
              fontFamily: C.mono,
              fontSize: 16,
              fontWeight: 500,
              letterSpacing: 0.4,
            }}
          />
          <button
            type="button"
            onClick={() => setBpm(Math.min(200, bpm + 5))}
            style={{
              padding: '6px 0',
              background: C.bg,
              border: `1px solid ${C.rule}`,
              color: C.ink,
              fontFamily: C.mono,
              fontSize: 12,
              cursor: 'pointer',
            }}
          >
            +
          </button>
        </div>
      </div>
      <Slider
        label="BPM"
        value={bpm}
        min={60}
        max={200}
        onChange={setBpm}
        leftLabel="60"
        rightLabel="200"
      />

      {generationType === 'orchestral' && (
        <>
          <div style={{ padding: '6px 16px 14px' }}>
            <FieldLabel style={{ marginBottom: 8 }}>Pattern length</FieldLabel>
            <ChipRow
              options={[
                { value: 1, label: '1 bar' },
                { value: 2, label: '2 bars' },
                { value: 4, label: '4 bars' },
              ]}
              value={pattern}
              onChange={setPattern}
            />
          </div>
          <div
            style={{
              padding: '10px 16px 14px',
              borderTop: `1px solid ${C.rule}`,
              display: 'flex',
              alignItems: 'center',
              gap: 8,
            }}
          >
            <input
              id="includeRisers"
              type="checkbox"
              checked={includeRisers}
              onChange={(e) => setIncludeRisers(e.target.checked)}
            />
            <label
              htmlFor="includeRisers"
              style={{
                fontFamily: C.mono,
                fontSize: 10,
                letterSpacing: 0.5,
                color: C.inkSoft,
                textTransform: 'uppercase',
              }}
            >
              Include risers + transitions
            </label>
          </div>
        </>
      )}
    </Panel>
  );

  // ---------- Output ----------
  const OutputPanel = (
    <Panel
      title="Output · beat"
      marker="●"
      status={generatedAudioUrl ? <span style={{ color: C.ok }}>ready</span> : null}
    >
      <div style={{ padding: 16 }}>
        <div style={{ background: C.bg, border: `1px solid ${C.rule}`, padding: 8 }}>
          <FieldLabel style={{ padding: '0 0 8px' }}>Rendered beat</FieldLabel>
          <ToolWaveform audioUrl={generatedAudioUrl} height={100} color={C.warm} />
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
      tool={{ ...tool, sku: 'T-07', category: 'Audio tool · Beta', version: 'v0.3.0' }}
      subtitle={generationType === 'random' ? 'random pattern' : 'orchestral'}
      description="Pick a style and a tempo. Random rolls an AI drum pattern; Orchestral builds cinematic percussion arranged over bars."
      meta={[
        { k: 'Status', v: 'Beta' },
        { k: 'Avg time', v: '~10s' },
      ]}
      running={isGenerating}
      progress={isGenerating ? 0.6 : 0}
      statusMessage={statusMessage}
      primaryLabel="Generate beat"
      secondaryLabel="Randomize"
      onSecondary={handleGenerate}
      onPrimary={handleGenerate}
      onBack={onBack}
      left={InputPanel}
      center={ControlsPanel}
      right={OutputPanel}
    />
  );
};

export default BeatGeneratorTool;
