/**
 * RefRendererDevPanel — a minimal floating dev UI that sits on top of the
 * normal PluginCreator editor. Lets you:
 *   - preview the golden Helix / Strata DSLs
 *   - type a brief and have Qwen generate a new PluginDSL
 *   - (NEW) drop in a reference image → Moondream analyzes it → fed into
 *     the DSL prompt the same way the designer agent uses reference pics
 *   - inspect vision answers + raw JSON
 *   - swap between the rendered preview and the generated DSL JSON
 *
 * Toggle with ?refRenderer=1 or localStorage.setItem('doseedo.refRenderer','1').
 */

import React, { useState, useEffect, useMemo, useRef } from 'react';
import {
  RenderDSLSafe,
  generatePluginDSL,
  warmupQwen,
  helixDSL,
  strataDSL,
  fileToDataUrl,
} from './index';
import './refRenderer.css';

export function isRefRendererEnabled() {
  if (typeof window === 'undefined') return false;
  try {
    const params = new URLSearchParams(window.location.search);
    if (params.get('refRenderer') === '1') return true;
    return window.localStorage?.getItem('doseedo.refRenderer') === '1';
  } catch {
    return false;
  }
}

export function RefRendererDevPanel() {
  const enabled = useMemo(isRefRendererEnabled, []);
  const [open, setOpen] = useState(false);
  const [brief, setBrief] = useState(
    'A warm analog-modeled polysynth with two wavetable oscillators, sub osc, ' +
    'noise, a resonant ladder filter, amplitude envelope, mod LFO, 4 macros, ' +
    'and a velocity curve. Dark chassis, cyan accent. Hero preset: Glacier Pad.',
  );
  const [refImage, setRefImage] = useState(null);       // { dataUrl, name }
  const [visionAnswers, setVisionAnswers] = useState(null);
  const [visionLatency, setVisionLatency] = useState(null);
  const [stage, setStage] = useState(null);             // 'vision:start'|'dsl:start'|...

  const [dsl, setDsl] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [attempts, setAttempts] = useState(0);
  const [rawText, setRawText] = useState('');
  const [tab, setTab] = useState('preview'); // preview | json | vision

  const fileInputRef = useRef(null);

  useEffect(() => {
    if (enabled) warmupQwen();
  }, [enabled]);

  if (!enabled) return null;

  const pickImage = async (file) => {
    if (!file) return;
    const dataUrl = await fileToDataUrl(file);
    setRefImage({ dataUrl, name: file.name, size: file.size });
    setVisionAnswers(null);
  };

  const onDrop = async (e) => {
    e.preventDefault();
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) await pickImage(file);
  };

  const runGenerate = async () => {
    setBusy(true);
    setError(null);
    setStage(null);
    setVisionAnswers(null);
    setVisionLatency(null);
    try {
      const res = await generatePluginDSL({
        brief,
        referenceImage: refImage?.dataUrl || undefined,
        onStage: (name, info) => {
          setStage(name);
          if (name === 'vision:done') {
            setVisionAnswers(info?.answers || null);
            setVisionLatency(info?.latencyMs ?? null);
          }
        },
      });
      setDsl(res.dsl);
      setAttempts(res.attempts);
      setRawText(res.rawText || '');
      if (res.visionAnswers) setVisionAnswers(res.visionAnswers);
      if (refImage) setTab('vision');
    } catch (e) {
      setError(String(e.message || e));
      setRawText(e.lastText || '');
    } finally {
      setBusy(false);
      setStage(null);
    }
  };

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        style={{
          position: 'fixed',
          right: 16,
          bottom: 16,
          zIndex: 10000,
          padding: '10px 16px',
          borderRadius: 8,
          background: '#1a1816',
          color: '#ffb84d',
          border: '1px solid #4a3a28',
          fontSize: 12,
          fontFamily: 'ui-monospace, monospace',
          cursor: 'pointer',
          letterSpacing: '0.08em',
          boxShadow: '0 6px 18px rgba(0,0,0,0.5)',
        }}
      >
        {open ? 'CLOSE ▾' : 'REF RENDERER ▴'}
      </button>

      {open && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 9999,
            background: 'rgba(5, 4, 3, 0.88)',
            display: 'flex',
            flexDirection: 'column',
            padding: 20,
            gap: 12,
            fontFamily: 'ui-monospace, monospace',
            fontSize: 12,
            color: '#d9d4cd',
          }}
        >
          <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
            <span style={{ fontSize: 13, letterSpacing: '0.14em', color: '#ffb84d' }}>
              REF-RENDERER DEV PANEL
            </span>
            <span style={{ color: '#6a665f' }}>
              · qwen3-14b + moondream2 · goldens: helix, strata · attempts {attempts || 0}
            </span>
            {stage && (
              <span style={{ color: '#ffb84d', fontWeight: 600 }}>
                · {stage.replace(':', ' → ')}
              </span>
            )}
            {visionLatency != null && (
              <span style={{ color: '#6a665f' }}>
                · vision {Math.round(visionLatency / 100) / 10}s
              </span>
            )}
            <span style={{ flex: 1 }} />
            <button onClick={() => setDsl(helixDSL)}   style={btnStyle}>Load HELIX</button>
            <button onClick={() => setDsl(strataDSL)}  style={btnStyle}>Load STRATA</button>
            <button onClick={() => setOpen(false)}     style={btnStyle}>✕</button>
          </div>

          <div style={{ display: 'flex', gap: 12, flex: 1, minHeight: 0 }}>
            <div style={{ width: 400, display: 'flex', flexDirection: 'column', gap: 8, minHeight: 0 }}>
              <textarea
                value={brief}
                onChange={(e) => setBrief(e.target.value)}
                placeholder="Describe the plugin (archetype + aesthetic + control inventory). If you upload a reference image below, vision will supplement this text."
                style={{
                  minHeight: 140,
                  padding: 10,
                  background: '#0f0e0d',
                  color: '#d9d4cd',
                  border: '1px solid #3a3632',
                  borderRadius: 4,
                  fontFamily: 'ui-monospace, monospace',
                  fontSize: 12,
                  resize: 'vertical',
                }}
              />

              {/* Reference image drop-zone */}
              <div
                onDrop={onDrop}
                onDragOver={(e) => e.preventDefault()}
                onClick={() => fileInputRef.current?.click()}
                style={{
                  padding: 10,
                  background: refImage ? '#1a1816' : '#0f0e0d',
                  border: `1px dashed ${refImage ? '#ffb84d' : '#3a3632'}`,
                  borderRadius: 4,
                  cursor: 'pointer',
                  display: 'flex',
                  gap: 10,
                  alignItems: 'center',
                  minHeight: 70,
                }}
              >
                {refImage ? (
                  <>
                    <img
                      src={refImage.dataUrl}
                      alt={refImage.name}
                      style={{
                        width: 80, height: 50, objectFit: 'cover',
                        borderRadius: 3, border: '1px solid #3a3632',
                      }}
                    />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ color: '#ffb84d', fontSize: 11, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {refImage.name}
                      </div>
                      <div style={{ color: '#6a665f', fontSize: 10 }}>
                        {(refImage.size / 1024).toFixed(1)} KB · vision will analyze this
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => { e.stopPropagation(); setRefImage(null); setVisionAnswers(null); }}
                      style={{ ...btnStyle, padding: '4px 8px' }}
                    >
                      ✕
                    </button>
                  </>
                ) : (
                  <div style={{ color: '#6a665f', fontSize: 11 }}>
                    📷 drop or click — optional reference image (Moondream 2 vision)
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  style={{ display: 'none' }}
                  onChange={(e) => pickImage(e.target.files?.[0])}
                />
              </div>

              <button
                type="button"
                onClick={runGenerate}
                disabled={busy || (!brief.trim() && !refImage)}
                style={{
                  padding: '10px 14px',
                  background: busy ? '#3a3632' : '#ffb84d',
                  color: busy ? '#6a665f' : '#1a1816',
                  border: 'none',
                  borderRadius: 4,
                  fontWeight: 600,
                  letterSpacing: '0.08em',
                  cursor: busy ? 'wait' : 'pointer',
                }}
              >
                {busy
                  ? (stage?.startsWith('vision') ? 'ANALYZING REFERENCE IMAGE…'
                  :  stage?.startsWith('dsl')    ? 'GENERATING DSL…'
                  :  'WORKING…')
                  : refImage
                    ? 'ANALYZE + GENERATE'
                    : 'GENERATE PLUGIN DSL'}
              </button>

              {error && (
                <div style={{ padding: 10, background: '#1a0e0e', color: '#ff8a8a', borderRadius: 4, maxHeight: 120, overflow: 'auto' }}>
                  {error}
                </div>
              )}
              <div style={{ display: 'flex', gap: 6 }}>
                <button onClick={() => setTab('preview')} style={{ ...btnStyle, background: tab === 'preview' ? '#4a3a28' : '#1a1816' }}>
                  preview
                </button>
                <button onClick={() => setTab('vision')} disabled={!visionAnswers} style={{ ...btnStyle, background: tab === 'vision' ? '#4a3a28' : '#1a1816', opacity: visionAnswers ? 1 : 0.4 }}>
                  vision
                </button>
                <button onClick={() => setTab('json')} style={{ ...btnStyle, background: tab === 'json' ? '#4a3a28' : '#1a1816' }}>
                  json
                </button>
              </div>
            </div>

            <div style={{
              flex: 1, minWidth: 0, overflow: 'auto',
              background: '#050403', border: '1px solid #3a3632',
              borderRadius: 4, padding: 20,
              display: 'flex',
              alignItems: tab === 'preview' ? 'flex-start' : 'stretch',
              justifyContent: tab === 'preview' ? 'center' : 'flex-start',
            }}>
              {tab === 'preview' ? (
                dsl ? (
                  <div style={{ transformOrigin: 'top center' }}>
                    <RenderDSLSafe dsl={dsl} />
                  </div>
                ) : (
                  <div style={{ color: '#6a665f', marginTop: 40 }}>
                    Load a golden or generate from a brief to preview.
                  </div>
                )
              ) : tab === 'vision' ? (
                visionAnswers ? (
                  <div style={{ color: '#d9d4cd', fontSize: 12, fontFamily: 'ui-monospace, monospace', lineHeight: 1.5, width: '100%' }}>
                    <div style={{ color: '#ffb84d', letterSpacing: '0.12em', marginBottom: 10 }}>
                      MOONDREAM 2 — REFERENCE ANALYSIS
                    </div>
                    {Object.entries(visionAnswers).map(([k, v]) => (
                      <div key={k} style={{ marginBottom: 10 }}>
                        <div style={{ color: '#ffb84d', fontSize: 10, letterSpacing: '0.14em' }}>{k.toUpperCase()}</div>
                        <div style={{ whiteSpace: 'pre-wrap', color: '#d9d4cd' }}>{v}</div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: '#6a665f', marginTop: 40 }}>
                    No vision analysis yet. Drop in a reference image and generate.
                  </div>
                )
              ) : (
                <pre style={{ color: '#d9d4cd', margin: 0, whiteSpace: 'pre-wrap', wordBreak: 'break-word', fontSize: 11 }}>
                  {dsl ? JSON.stringify(dsl, null, 2) : (rawText || '—')}
                </pre>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

const btnStyle = {
  padding: '6px 10px',
  background: '#1a1816',
  color: '#d9d4cd',
  border: '1px solid #3a3632',
  borderRadius: 3,
  fontSize: 11,
  fontFamily: 'ui-monospace, monospace',
  cursor: 'pointer',
};

export default RefRendererDevPanel;
