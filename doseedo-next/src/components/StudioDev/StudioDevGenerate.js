/*
 * StudioDevGenerate — themed generation panel for /studio-dev.
 *
 * Replaces components/GenerationPanelOptimized. Minimal but functional:
 * prompt + style + mood + key + bars, plus a chord-aware "Fit my chords"
 * toggle that injects state.chords into the params. Calls the same
 * /api/generate-do endpoint the production panel does, then polls the
 * task until complete and drops the result as a new track.
 */
import React, { useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { startGeneration, pollUntilComplete } from '../../services/generationAPI';

// Sliders laid out in the advanced panel. Matches the production set in
// components/GenerationPanel/GenerationPanelOptimized.js → GenerationParameters.
const ADV_SLIDERS = [
  { k: 'seed',         label: 'Seed',         min: 0,   max: 10000, step: 1,    def: 0 },
  { k: 'steps',        label: 'Steps',        min: 10,  max: 100,   step: 1,    def: 20 },
  { k: 'adapterScale', label: 'Adapter',      min: 0,   max: 2,     step: 0.05, def: 1.0 },
  { k: 'cfgWeight',    label: 'CFG Weight',   min: 0,   max: 10,    step: 0.1,  def: 1.0 },
  { k: 't0',           label: 'T0',           min: 0,   max: 1,     step: 0.05, def: 0.95 },
  { k: 'noiseLevel',   label: 'Noise',        min: 0,   max: 1,     step: 0.05, def: 1.0 },
  { k: 'temperature',  label: 'Temperature',  min: 0.1, max: 2.0,   step: 0.05, def: 1.0 },
  { k: 'coverNoiseStrength', label: 'Cover noise', min: 0, max: 1, step: 0.05, def: 0.2 },
  { k: 'audioCoverStrength', label: 'Audio cover', min: 0, max: 1, step: 0.05, def: 0.5 },
  { k: 'tapeSpeed',    label: 'Tape speed',   min: 0.25, max: 2, step: 0.05, def: 1.0 },
];

const STYLES = ['cinematic', 'pop', 'r&b', 'folk', 'electronic', 'jazz', 'hip-hop', 'rock', 'ambient'];
const MOODS  = ['uplifting', 'dark', 'dreamy', 'aggressive', 'melancholy', 'warm', 'energetic', 'peaceful'];
const KEYS   = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const MODES  = ['major', 'minor'];

export default function StudioDevGenerate({ onClose, embedded = false }) {
  const { state, dispatch } = useApp();
  const [prompt, setPrompt] = useState('');
  const [style,  setStyle]  = useState(STYLES[0]);
  const [mood,   setMood]   = useState(MOODS[0]);
  const [key,    setKey]    = useState('C');
  const [mode,   setMode]   = useState('major');
  const [bars,   setBars]   = useState(8);
  const [useChords, setUseChords] = useState(true);
  const [monophonic, setMonophonic] = useState(false);
  const [arrange, setArrange] = useState(false);
  const [running, setRunning] = useState(false);
  const [status,  setStatus]  = useState('');
  const [error,   setError]   = useState(null);

  // Advanced params — all mirror the production GenerationParameters pane.
  const [advOpen, setAdvOpen] = useState(false);
  const [advParams, setAdvParams] = useState(() => {
    const init = {};
    for (const p of ADV_SLIDERS) init[p.k] = p.def;
    return init;
  });
  const setAdv = (k, v) => setAdvParams((a) => ({ ...a, [k]: v }));

  // Processing-mode toggles — fatten / slowdown / upsample, all on/off.
  const [fattenMode, setFattenMode] = useState(true);
  const [fattenType, setFattenType] = useState('real');
  const [slowdownMethod, setSlowdownMethod] = useState('stretch');
  const [upsampleMode, setUpsampleMode] = useState(false);
  const [upsampleNoise, setUpsampleNoise] = useState(0.3);
  const [upsampleSteps, setUpsampleSteps] = useState(20);

  const hasChords = useMemo(() =>
    state.chords && Object.keys(state.chords).length > 0, [state.chords]);

  const run = async () => {
    setRunning(true); setStatus('starting…'); setError(null);
    try {
      const params = {
        prompt, style, mood, key, mode, bars,
        bpm: state.bpm || 120,
        monophonicMode: monophonic,
        arrangeMode: arrange,
        fattenMode,
        fattenType,
        slowdownMethod,
        upsampleMode,
        upsampleNoiseLevel: upsampleNoise,
        upsampleSteps,
        ...advParams,
        ...(useChords && hasChords ? { chords: state.chords } : {}),
      };
      const { task_id } = await startGeneration(params);
      setStatus(`task ${task_id.slice(0, 8)}… running`);
      const final = await pollUntilComplete(task_id, (s) => {
        setStatus(s?.progress != null ? `${Math.round(s.progress * 100)}%` : (s?.status || '…'));
      });
      setStatus('done');
      // Add the result as a new track.
      if (final?.audio_url) {
        const busId = `bus-gen-${Date.now()}`;
        const trackId = `t-gen-${Date.now()}`;
        dispatch({
          type: 'CREATE_BUS',
          payload: { id: busId, type: 'INSTRUMENT', name: `Gen · ${style}`, expanded: true },
        });
        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId,
            track: {
              id: trackId, name: `Gen · ${prompt.slice(0, 24) || style}`,
              audioUrl: final.audio_url, duration: final.duration || 0, startPosition: 0,
              gain: 1, isMuted: false, isSolo: false,
              fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
              metadata: { type: 'generated', prompt, style, mood, key, mode },
            },
          },
        });
        dispatch({ type: 'SELECT_TRACK', payload: { trackId, busId } });
        onClose?.();
      }
    } catch (e) {
      setError(e?.message || String(e));
      setStatus('');
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className={`sd-gen ${embedded ? 'sd-gen-embedded' : ''}`}>
      {!embedded && (
        <div className="sd-midi-toolbar">
          <div className="sd-midi-title">
            <span className="sd-midi-meta">GENERATE</span>
            <span className="sd-midi-name" style={{ marginLeft: 6 }}>Fresh part</span>
          </div>
          <div className="sd-midi-spacer" />
          {onClose && <button className="sd-midi-btn" onClick={onClose}>Close</button>}
        </div>
      )}

      <div className="sd-gen-body">
        <div className="sd-label">Prompt</div>
        <textarea className="sd-gen-textarea" rows={3}
                  placeholder="e.g. warm rhodes, relaxed shuffle, 90 bpm…"
                  value={prompt} onChange={(e) => setPrompt(e.target.value)} />


        {/* ============= ADVANCED ============= */}
        <button className="sd-adv-toggle" onClick={() => setAdvOpen((v) => !v)}>
          <i className={`fa-solid fa-${advOpen ? 'caret-down' : 'caret-right'}`} />
          <span>Advanced parameters</span>
        </button>

        {advOpen && (
          <div className="sd-adv">
            <div className="sd-label">Diffusion params</div>
            <div className="sd-adv-grid">
              {ADV_SLIDERS.map((p) => (
                <div key={p.k} className="sd-adv-slider">
                  <div className="sd-adv-head">
                    <span className="sd-adv-label">{p.label}</span>
                    <span className="sd-adv-value">
                      {typeof advParams[p.k] === 'number' ? advParams[p.k].toFixed(p.step < 1 ? 2 : 0) : p.def}
                    </span>
                  </div>
                  <input type="range" min={p.min} max={p.max} step={p.step}
                         value={advParams[p.k] ?? p.def}
                         onChange={(e) => setAdv(p.k, parseFloat(e.target.value))} />
                </div>
              ))}
            </div>

            <div className="sd-label" style={{ marginTop: 14 }}>Processing</div>
            <div className="sd-adv-rows">
              <label className="sd-gen-check">
                <input type="checkbox" checked={fattenMode}
                       onChange={(e) => setFattenMode(e.target.checked)} />
                <span>Fatten mode</span>
              </label>
              {fattenMode && (
                <div className="sd-adv-inline">
                  <span className="sd-adv-label">Type</span>
                  <select className="sd-gen-select" style={{ width: 120 }}
                          value={fattenType} onChange={(e) => setFattenType(e.target.value)}>
                    <option value="real">Real</option>
                    <option value="synth">Synth</option>
                  </select>
                </div>
              )}

              <div className="sd-adv-inline">
                <span className="sd-adv-label">Slowdown</span>
                <label style={{ marginLeft: 8 }}>
                  <input type="radio" name="sdmethod" value="stretch"
                         checked={slowdownMethod === 'stretch'}
                         onChange={() => setSlowdownMethod('stretch')} /> Stretch
                </label>
                <label style={{ marginLeft: 8 }}>
                  <input type="radio" name="sdmethod" value="tape"
                         checked={slowdownMethod === 'tape'}
                         onChange={() => setSlowdownMethod('tape')} /> Tape
                </label>
              </div>

              <label className="sd-gen-check">
                <input type="checkbox" checked={upsampleMode}
                       onChange={(e) => setUpsampleMode(e.target.checked)} />
                <span>Upsample after generation</span>
              </label>
              {upsampleMode && (
                <>
                  <div className="sd-adv-slider">
                    <div className="sd-adv-head">
                      <span className="sd-adv-label">Upsample noise</span>
                      <span className="sd-adv-value">{upsampleNoise.toFixed(2)}</span>
                    </div>
                    <input type="range" min={0} max={1} step={0.05}
                           value={upsampleNoise}
                           onChange={(e) => setUpsampleNoise(parseFloat(e.target.value))} />
                  </div>
                  <div className="sd-adv-slider">
                    <div className="sd-adv-head">
                      <span className="sd-adv-label">Upsample steps</span>
                      <span className="sd-adv-value">{upsampleSteps}</span>
                    </div>
                    <input type="range" min={5} max={80} step={1}
                           value={upsampleSteps}
                           onChange={(e) => setUpsampleSteps(parseInt(e.target.value, 10))} />
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {error && <div className="sd-gen-error">{error}</div>}

        <div className="sd-gen-actions">
          <button className="wb-btn wb-btn--primary sd-gen-submit" disabled={running} onClick={run}>
            {running ? `> Generating · ${status}` : '> Generate'}
          </button>
        </div>
      </div>
    </div>
  );
}
