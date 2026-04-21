/*
 * StudioDevGenerate — themed generation panel for /studio-dev.
 *
 * Wired to the stemphonic 130k Modal backend (/api/generate-stemphonic +
 * generate-stemphonic/task/<id>). Replaces the legacy /api/generate-do
 * call. Params exposed match the stage-2d-130k checkpoint's real
 * controls: prompt + instrument + duration + diffusion knobs (steps,
 * cfg, seed) + conditioning strengths (cover_noise, audio_cover).
 *
 * The currently selected track is auto-loaded as the generation input
 * — MIDI tracks go to midiFile, audio tracks go to refAudio (BasicPitch
 * runs on the server in parallel for audio).
 */
import React, { useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { generateStemphonic, pollStemphonicUntilComplete } from '../../services/generationAPI';

// Stemphonic 130k diffusion params. Ranges + defaults mirror the
// production GenerationPanelOptimized call to generateStemphonic().
const ADV_SLIDERS = [
  { k: 'steps',               label: 'Steps',        min: 10,  max: 100,  step: 1,    def: 50 },
  { k: 'cfg',                 label: 'CFG',          min: 1,   max: 15,   step: 0.1,  def: 7.0 },
  { k: 'seed',                label: 'Seed',         min: -1,  max: 99999, step: 1,   def: -1 },
  { k: 'cover_noise_strength', label: 'Cover noise',  min: 0,   max: 1,    step: 0.01, def: 0.20 },
  { k: 'audio_cover_strength', label: 'Audio cover',  min: 0,   max: 1,    step: 0.01, def: 0.50 },
];

// Instrument subgroups that stemphonic 130k recognizes. Keep the list
// aligned with the backend's APPROVED_SUBGROUPS (stemphonic_server.py).
const INSTRUMENT_CHOICES = [
  'piano', 'rhodes', 'guitar', 'acoustic_guitar', 'electric_guitar',
  'bass', 'upright_bass', 'synth_bass',
  'drums', 'drum_kit',
  'strings', 'violin', 'cello', 'pad',
  'brass', 'trumpet', 'saxophone', 'flute', 'clarinet',
  'vocals', 'choir',
];

export default function StudioDevGenerate({ onClose, embedded = false }) {
  const { state, dispatch } = useApp();
  const [prompt, setPrompt] = useState('');
  const [instrument, setInstrument] = useState('piano');
  const [timbrePreset, setTimbrePreset] = useState('');
  const [duration, setDuration] = useState(16);
  const [drumMode, setDrumMode] = useState(false);
  const [voxMode, setVoxMode] = useState(false);
  const [lyrics, setLyrics] = useState('');
  const [running, setRunning] = useState(false);
  const [status,  setStatus]  = useState('');
  const [error,   setError]   = useState(null);
  const [advOpen, setAdvOpen] = useState(false);
  const [advParams, setAdvParams] = useState(() => {
    const init = {};
    for (const p of ADV_SLIDERS) init[p.k] = p.def;
    return init;
  });
  const setAdv = (k, v) => setAdvParams((a) => ({ ...a, [k]: v }));

  // Pull the currently selected track as conditioning input. MIDI track
  // → midiFile, audio track → refAudio. The user's "selected source"
  // in the left sidebar doubles as the generation input.
  const selectedTrack = state.selectedTrack;
  const inputInfo = useMemo(() => {
    if (!selectedTrack) return { kind: 'none' };
    if (selectedTrack.midiData?.notes?.length || selectedTrack.metadata?.midiData?.notes?.length) {
      return { kind: 'midi', name: selectedTrack.name || selectedTrack.id };
    }
    if (selectedTrack.audioUrl || selectedTrack.audioFile) {
      return { kind: 'audio', name: selectedTrack.name || selectedTrack.id };
    }
    return { kind: 'none' };
  }, [selectedTrack]);

  // Convert selectedTrack → File for upload. MIDI: serialize notes to a
  // minimal .mid blob is expensive; server accepts JSON under the same
  // midiFile field, so pass a JSON blob with the notes array. Audio:
  // fetch the blob from audioUrl.
  const buildInputFile = async () => {
    if (inputInfo.kind === 'midi') {
      const md = selectedTrack.midiData || selectedTrack.metadata?.midiData;
      const notes = md?.notes || [];
      const json = JSON.stringify({ notes, duration: md?.duration || duration, tempo: md?.tempo || state.bpm || 120 });
      return new File([json], 'input.midi.json', { type: 'application/json' });
    }
    if (inputInfo.kind === 'audio') {
      if (selectedTrack.audioFile instanceof File) return selectedTrack.audioFile;
      const r = await fetch(selectedTrack.audioUrl);
      const b = await r.blob();
      return new File([b], (selectedTrack.name || 'input') + '.wav', { type: b.type || 'audio/wav' });
    }
    return null;
  };

  const run = async () => {
    setRunning(true); setStatus('starting…'); setError(null);
    try {
      const params = {
        prompt: prompt || instrument.replace(/_/g, ' '),
        instrument,
        timbre_preset: timbrePreset || `${instrument}:default`,
        duration,
        steps: advParams.steps,
        cfg: advParams.cfg,
        seed: advParams.seed,
        cover_noise_strength: advParams.cover_noise_strength,
        audio_cover_strength: advParams.audio_cover_strength,
        drum_mode: drumMode ? 'true' : 'false',
        vox_mode: voxMode ? 'true' : 'false',
      };
      if (voxMode && lyrics.trim()) params.lyrics = lyrics.trim();

      const inputFile = await buildInputFile();
      const midiFile = inputInfo.kind === 'midi' ? inputFile : null;
      const refAudio = inputInfo.kind === 'audio' ? inputFile : null;

      const start = await generateStemphonic(params, midiFile, refAudio);
      if (!start.task_id) throw new Error('No task_id returned');
      setStatus(`task ${start.task_id.slice(0, 8)}… running`);

      const result = await pollStemphonicUntilComplete(start.task_id, (p) => {
        setStatus(`${p.status || '…'}${p.attempts ? ` · ${p.attempts}s` : ''}`);
      });
      setStatus('done');

      const filePaths = result?.file_paths || [];
      if (filePaths.length) {
        const firstUrl = filePaths[0];
        const busId = `bus-gen-${Date.now()}`;
        const trackId = `t-gen-${Date.now()}`;
        dispatch({
          type: 'CREATE_BUS',
          payload: { id: busId, type: 'INSTRUMENT', name: `Gen · ${instrument}`, expanded: true },
        });
        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId,
            track: {
              id: trackId,
              name: `Gen · ${prompt.slice(0, 24) || instrument}`,
              audioUrl: firstUrl,
              duration: result.duration || duration,
              startPosition: 0,
              gain: 1, isMuted: false, isSolo: false,
              fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
              metadata: {
                type: 'generated', source: 'stemphonic',
                instrument, timbre_preset: params.timbre_preset,
                prompt: params.prompt, params,
                versions: filePaths.map((url, i) => ({
                  audioUrl: url, timestamp: Date.now(), type: 'generated',
                  name: `Candidate ${i + 1}`, params,
                })),
                currentVersionIndex: 0,
              },
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
            <span className="sd-midi-name" style={{ marginLeft: 6 }}>Stemphonic 130k</span>
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

        <div className="sd-adv-grid" style={{ marginTop: 10 }}>
          <div className="sd-adv-slider">
            <div className="sd-adv-head">
              <span className="sd-adv-label">Instrument</span>
            </div>
            <select className="sd-gen-select"
                    value={instrument} onChange={(e) => setInstrument(e.target.value)}>
              {INSTRUMENT_CHOICES.map((i) => (
                <option key={i} value={i}>{i.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>
          <div className="sd-adv-slider">
            <div className="sd-adv-head">
              <span className="sd-adv-label">Duration</span>
              <span className="sd-adv-value">{duration}s</span>
            </div>
            <input type="range" min={4} max={60} step={1}
                   value={duration}
                   onChange={(e) => setDuration(parseInt(e.target.value, 10))} />
          </div>
        </div>

        <div className="sd-adv-rows" style={{ marginTop: 8 }}>
          <label className="sd-gen-check">
            <input type="checkbox" checked={drumMode}
                   onChange={(e) => setDrumMode(e.target.checked)} />
            <span>Drum mode</span>
          </label>
          <label className="sd-gen-check">
            <input type="checkbox" checked={voxMode}
                   onChange={(e) => setVoxMode(e.target.checked)} />
            <span>Vocal mode</span>
          </label>
        </div>

        {voxMode && (
          <>
            <div className="sd-label" style={{ marginTop: 8 }}>Lyrics</div>
            <textarea className="sd-gen-textarea" rows={2}
                      placeholder="one line per phrase"
                      value={lyrics} onChange={(e) => setLyrics(e.target.value)} />
          </>
        )}

        {inputInfo.kind !== 'none' && (
          <div className="sd-gen-input-info">
            <span>Conditioning {inputInfo.kind === 'midi' ? 'MIDI' : 'audio'}</span>
            <strong>{inputInfo.name}</strong>
          </div>
        )}

        {/* ============= ADVANCED ============= */}
        <button className="sd-adv-toggle" onClick={() => setAdvOpen((v) => !v)}>
          <i className={`fa-solid fa-${advOpen ? 'caret-down' : 'caret-right'}`} />
          <span>Advanced parameters</span>
        </button>

        {advOpen && (
          <div className="sd-adv">
            <div className="sd-label">Diffusion params · stemphonic 130k</div>
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

            <div className="sd-label" style={{ marginTop: 14 }}>Timbre preset</div>
            <input type="text" className="sd-gen-textarea" style={{ minHeight: 0, height: 28 }}
                   placeholder="e.g. piano:rhodes_mkii  (leave blank for default)"
                   value={timbrePreset}
                   onChange={(e) => setTimbrePreset(e.target.value)} />
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
