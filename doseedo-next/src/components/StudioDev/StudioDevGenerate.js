/*
 * StudioDevGenerate — themed generation panel for /studio-dev.
 *
 * Wired to the stemphonic 130k Modal backend (/api/generate-stemphonic).
 * Instrument is picked from the left-sidebar palette (selectedInstrument
 * prop — group/subgroup). Drum / vocal mode derives from that. This
 * panel just handles prompt + advanced diffusion params.
 *
 * The currently selected track is auto-loaded as the generation input
 * (MIDI track → midiFile, audio track → refAudio — server runs
 * BasicPitch in parallel for audio).
 */
import React, { useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';
import { generateStemphonic, pollStemphonicUntilComplete } from '../../services/generationAPI';

// Stemphonic 130k diffusion params. Ranges + defaults mirror the
// production GenerationPanelOptimized call to generateStemphonic().
const ADV_SLIDERS = [
  { k: 'steps',                label: 'Steps',       min: 10,  max: 100,   step: 1,    def: 50 },
  { k: 'cfg',                  label: 'CFG',         min: 1,   max: 15,    step: 0.1,  def: 7.0 },
  { k: 'seed',                 label: 'Seed',        min: -1,  max: 99999, step: 1,    def: -1 },
  { k: 'cover_noise_strength', label: 'Cover noise', min: 0,   max: 1,     step: 0.01, def: 0.20 },
  { k: 'audio_cover_strength', label: 'Audio cover', min: 0,   max: 1,     step: 0.01, def: 0.50 },
];

export default function StudioDevGenerate({ onClose, embedded = false, selectedInstrument = null }) {
  const { state, dispatch } = useApp();
  const [prompt, setPrompt] = useState('');
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

  // Selected track = conditioning input. Priority:
  //   midi   — notes in either track.midiData or track.metadata.midiData.
  //            Covers: (a) new tracks where the user drew notes in the
  //            MIDI window, and (b) uploaded tracks whose BasicPitch Tier 1
  //            transcription already populated metadata.midiData.notes.
  //            Server renders via latent-soundfont and drives the per-layer
  //            MIDI hooks directly — no LatentPitch call needed.
  //   latent — uploaded tracks with a cached VAE latent (metadata.latentId).
  //            Skips the server-side audio_to_fsq_tokens re-encode. Used
  //            only when no MIDI is available.
  //   audio  — last resort: raw audio, server re-encodes + runs LatentPitch.
  //   none   — text-only generation.
  const selectedTrack = state.selectedTrack;
  const inputInfo = useMemo(() => {
    if (!selectedTrack) return { kind: 'none' };
    const md = selectedTrack.midiData || selectedTrack.metadata?.midiData;
    if (md?.notes?.length) {
      return { kind: 'midi', name: selectedTrack.name || selectedTrack.id };
    }
    const latentId = selectedTrack.metadata?.latentId;
    if (latentId) {
      return { kind: 'latent', name: selectedTrack.name || selectedTrack.id, latentId };
    }
    if (selectedTrack.audioFile instanceof File || selectedTrack.audioUrl) {
      return { kind: 'audio', name: selectedTrack.name || selectedTrack.id };
    }
    return { kind: 'none' };
  }, [selectedTrack]);

  const buildInputFile = async () => {
    if (inputInfo.kind === 'midi') {
      // Serialize the notes to a real .mid binary so the server's
      // `Path(midi_path).suffix in ('.mid','.midi')` gate at
      // stemphonic_server.py:2001 accepts it. A JSON file here silently
      // fell through to the "no input" branch and generated from text alone.
      const md = selectedTrack.midiData || selectedTrack.metadata?.midiData;
      const notes = md?.notes || [];
      const tempo = md?.tempo || state.bpm || 120;
      const { Midi } = await import('@tonejs/midi');
      const midi = new Midi();
      midi.header.setTempo(tempo);
      const track = midi.addTrack();
      for (const n of notes) {
        track.addNote({
          midi: Math.round(n.note),
          time: Math.max(0, n.time || 0),
          duration: Math.max(0.01, n.duration || 0.25),
          velocity: Math.max(0.01, Math.min(1, (n.velocity ?? 100) / 127)),
        });
      }
      const bytes = midi.toArray();
      return new File([bytes], 'input.mid', { type: 'audio/midi' });
    }
    if (inputInfo.kind === 'latent') {
      // latent_id rides as a form param, not a file.
      return null;
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
    // Instrument must come from the left-sidebar pick. No default — the
    // panel just tells the user to pick one if missing.
    if (!selectedInstrument) {
      setError('Pick an instrument in the left sidebar first.');
      return;
    }
    setRunning(true); setStatus('starting…'); setError(null);
    try {
      const sub = selectedInstrument.subgroup || selectedInstrument.id;
      const grp = selectedInstrument.group || '';
      const isDrum = grp === 'drums';
      const isVox  = grp === 'vocals';
      const params = {
        prompt: prompt || sub.replace(/_/g, ' '),
        instrument: sub,
        timbre_preset: `${sub}:${selectedInstrument.sub || 'default'}`,
        steps: advParams.steps,
        cfg: advParams.cfg,
        seed: advParams.seed,
        cover_noise_strength: advParams.cover_noise_strength,
        audio_cover_strength: advParams.audio_cover_strength,
        drum_mode: isDrum ? 'true' : 'false',
        vox_mode: isVox ? 'true' : 'false',
      };
      // Duration: follow the conditioning track's length when present;
      // fall back to 16s. Stemphonic caps internally anyway.
      params.duration = selectedTrack?.duration || 16;

      const inputFile = await buildInputFile();
      const midiFile = inputInfo.kind === 'midi' ? inputFile : null;
      const refAudio = inputInfo.kind === 'audio' ? inputFile : null;
      if (inputInfo.kind === 'latent') {
        params.latent_id = inputInfo.latentId;
      }

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
          payload: { id: busId, type: 'INSTRUMENT', name: `Gen · ${selectedInstrument.label}`, expanded: true },
        });
        dispatch({
          type: 'ADD_TRACK',
          payload: {
            busId,
            track: {
              id: trackId,
              name: `Gen · ${prompt.slice(0, 24) || selectedInstrument.label}`,
              audioUrl: firstUrl,
              duration: result.duration || params.duration,
              startPosition: 0,
              gain: 1, isMuted: false, isSolo: false,
              fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
              metadata: {
                type: 'generated', source: 'stemphonic',
                instrument: sub, timbre_preset: params.timbre_preset,
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

        {selectedInstrument && (
          <div className="sd-gen-input-info">
            <span>Instrument</span>
            <strong>{selectedInstrument.label}</strong>
          </div>
        )}
        {inputInfo.kind !== 'none' && (
          <div className="sd-gen-input-info">
            <span>Conditioning {
              inputInfo.kind === 'midi' ? 'MIDI' :
              inputInfo.kind === 'latent' ? 'latent' : 'audio'
            }</span>
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
