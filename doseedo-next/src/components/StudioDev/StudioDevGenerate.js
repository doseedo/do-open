/*
 * StudioDevGenerate — themed generation panel for /studio-dev.
 *
 * Wired to the stemphonic 130k Modal backend (/api/generate-stemphonic).
 * Instrument is picked from the left-sidebar palette (selectedInstrument
 * prop — group/subgroup). Drum / vocal mode derives from that. This
 * panel just handles advanced diffusion params.
 *
 * The currently selected track is auto-loaded as the generation input
 * (MIDI track → midiFile, audio track → refAudio — server runs
 * BasicPitch in parallel for audio).
 */
import React, { useMemo, useState } from 'react';
import { DotLottieReact } from '@lottiefiles/dotlottie-react';
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
  //   audio  — last resort: raw audio, server re-encodes + runs LatentPitch.
  //   none   — text-only generation.
  const selectedTrack = state.selectedTrack;
  const selectedBus = state.selectedBus;

  // Pure helper: derive {kind, latentId} for any track. Same priority as
  // the original inputInfo (midi → latent → audio → none). Used both for
  // the single-track header readout and per-track in the bus loop so
  // bus generation respects each track's actual conditioning content.
  const describeTrackInput = (track) => {
    if (!track) return { kind: 'none' };
    const md = track.midiData || track.metadata?.midiData;
    if (md?.notes?.length) return { kind: 'midi' };
    const latentId = track.metadata?.latentId;
    if (latentId) return { kind: 'latent', latentId };
    if (track.audioFile instanceof File || track.audioUrl) return { kind: 'audio' };
    return { kind: 'none' };
  };

  const inputInfo = useMemo(() => {
    const info = describeTrackInput(selectedTrack);
    if (selectedTrack && info.kind !== 'none') {
      info.name = selectedTrack.name || selectedTrack.id;
    }
    return info;
  }, [selectedTrack]);

  // Bus-target preview: when a bus is selected without a per-track focus,
  // count the eligible tracks so the panel header can read e.g.
  // "MIDI bus · 4 tracks" before the user clicks Generate.
  const busTargetInfo = useMemo(() => {
    if (selectedTrack || !selectedBus) return null;
    const tracks = (selectedBus.tracks || []).filter((t) => !t.metadata?.isBusMaster);
    const eligible = tracks.filter((t) => describeTrackInput(t).kind !== 'none');
    return { name: selectedBus.name || selectedBus.type || 'Bus', total: tracks.length, eligible: eligible.length };
  }, [selectedBus, selectedTrack]);

  const buildInputFileFor = async (track, info) => {
    if (info.kind === 'midi') {
      // Emit a real .mid binary so the server's
      // `Path(midi_path).suffix in ('.mid','.midi')` gate accepts it.
      // A .json file here silently drops through to text-only generation.
      const md = track.midiData || track.metadata?.midiData;
      const notes = md?.notes || [];
      const tempo = md?.tempo || state.bpm || 120;
      const { Midi } = await import('@tonejs/midi');
      const midi = new Midi();
      midi.header.setTempo(tempo);
      const trk = midi.addTrack();
      for (const n of notes) {
        trk.addNote({
          midi: Math.round(n.note),
          time: Math.max(0, n.time || 0),
          duration: Math.max(0.01, n.duration || 0.25),
          velocity: Math.max(0.01, Math.min(1, (n.velocity ?? 100) / 127)),
        });
      }
      return new File([midi.toArray()], 'input.mid', { type: 'audio/midi' });
    }
    if (info.kind === 'latent') {
      // latent_id rides as a form param, no file body.
      return null;
    }
    if (info.kind === 'audio') {
      if (track.audioFile instanceof File) return track.audioFile;
      const r = await fetch(track.audioUrl);
      const b = await r.blob();
      return new File([b], (track.name || 'input') + '.wav', { type: b.type || 'audio/wav' });
    }
    return null;
  };

  // Per-track generation pass — one stemphonic round-trip + the existing
  // UPDATE_TRACK dispatch shape (preserves prior audioUrl as a version
  // so revert works). Lifted out of run() so the bus-target loop below
  // can call it once per child track without reimplementing the
  // version-merge logic.
  const generateForTrack = async ({ track, busId, baseParams, label }) => {
    const info = describeTrackInput(track);
    const params = {
      ...baseParams,
      duration: track?.duration || baseParams.duration || 16,
    };
    const inputFile = await buildInputFileFor(track, info);
    const midiFile = info.kind === 'midi' ? inputFile : null;
    const refAudio = info.kind === 'audio' ? inputFile : null;
    if (info.kind === 'latent') params.latent_id = info.latentId;

    const start = await generateStemphonic(params, midiFile, refAudio);
    if (!start.task_id) throw new Error('No task_id returned');
    setStatus(`${label} · task ${start.task_id.slice(0, 8)}…`);

    const result = await pollStemphonicUntilComplete(start.task_id, (p) => {
      setStatus(`${label} · ${p.status || '…'}${p.attempts ? ` · ${p.attempts}s` : ''}`);
    });

    const filePaths = result?.file_paths || [];
    if (!filePaths.length) return false;
    const firstUrl = filePaths[0];
    const now = Date.now();
    const newCandidateVersions = filePaths.map((url, i) => ({
      audioUrl: url,
      timestamp: now,
      type: 'generated',
      name: filePaths.length > 1 ? `Gen · ${label} · c${i + 1}` : `Gen · ${label}`,
      params,
    }));

    const existingVersions = track.metadata?.versions || [];
    const prevAudioUrl = track.audioUrl;
    const prevVersionAlreadyLogged = existingVersions.some((v) => v.audioUrl === prevAudioUrl);
    const preservedPrev = prevAudioUrl && !prevVersionAlreadyLogged
      ? [{
          audioUrl: prevAudioUrl,
          timestamp: track.metadata?.timestamp || (now - 1),
          type: track.metadata?.type || 'previous',
          name: existingVersions.length
            ? `Previous (v${existingVersions.length + 1})`
            : 'Previous',
          params: track.metadata?.params || null,
          midiData: track.metadata?.midiData || track.midiData || null,
        }]
      : [];

    const newVersions = [...existingVersions, ...preservedPrev, ...newCandidateVersions];
    const firstNewIndex = existingVersions.length + preservedPrev.length;

    dispatch({
      type: 'UPDATE_TRACK',
      payload: {
        busId,
        trackId: track.id,
        updates: {
          audioUrl: firstUrl,
          duration: result.duration || track.duration || params.duration,
          metadata: {
            ...(track.metadata || {}),
            type: 'generated',
            source: 'stemphonic',
            instrument: baseParams.instrument,
            timbre_preset: baseParams.timbre_preset,
            prompt: baseParams.prompt,
            params,
            timestamp: now,
            versions: newVersions,
            currentVersionIndex: firstNewIndex,
          },
        },
      },
    });
    return true;
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
      const baseParams = {
        // No user-facing text prompt — server derives the caption from
        // `instrument` via TRAINING_CAPTIONS. We still send the subgroup as
        // `prompt` for the no-match fallback path.
        prompt: sub.replace(/_/g, ' '),
        instrument: sub,
        timbre_preset: `${sub}:${selectedInstrument.sub || 'default'}`,
        steps: advParams.steps,
        cfg: advParams.cfg,
        seed: advParams.seed,
        cover_noise_strength: advParams.cover_noise_strength,
        audio_cover_strength: advParams.audio_cover_strength,
        drum_mode: isDrum ? 'true' : 'false',
        vox_mode: isVox ? 'true' : 'false',
        duration: 16,
      };

      // Bus-target path: iterate eligible child tracks and run one pass
      // per track. Each call writes a new version onto its own track via
      // generateForTrack, so per-track + bus-wide revert in the right
      // sidebar work without any extra plumbing.
      if (!selectedTrack && selectedBus) {
        const childTracks = (selectedBus.tracks || []).filter((t) => !t.metadata?.isBusMaster);
        const eligible = childTracks.filter((t) => describeTrackInput(t).kind !== 'none');
        if (eligible.length === 0) {
          setError('No tracks with audio or MIDI to generate from in this bus.');
          return;
        }
        let okCount = 0;
        for (let i = 0; i < eligible.length; i++) {
          const t = eligible[i];
          const label = `${selectedInstrument.label} · ${t.name || t.id} (${i + 1}/${eligible.length})`;
          try {
            const ok = await generateForTrack({
              track: t,
              busId: selectedBus.id,
              baseParams,
              label,
            });
            if (ok) okCount++;
          } catch (perr) {
            // Surface but keep going — partial success beats aborting the
            // whole bus run on one bad track.
            console.warn('[bus-gen] track failed:', t.id, perr);
            setError(`${t.name || t.id}: ${perr?.message || perr}`);
          }
        }
        setStatus(`done · ${okCount}/${eligible.length} tracks`);
        // Re-select the bus so the right sidebar's Versions section
        // picks up the new currentVersionIndex on each track.
        dispatch({ type: 'SELECT_BUS', payload: { busId: selectedBus.id } });
        if (okCount > 0) onClose?.();
        return;
      }

      // Single-track path (the original flow). Wraps generateForTrack so
      // both branches share the same version-merge + dispatch shape.
      const ownerBus = selectedTrack
        ? state.buses.find((b) => b.tracks?.some((t) => t.id === selectedTrack.id))
        : null;
      if (ownerBus && selectedTrack) {
        const ok = await generateForTrack({
          track: selectedTrack,
          busId: ownerBus.id,
          baseParams,
          label: selectedInstrument.label,
        });
        if (ok) {
          setStatus('done');
          dispatch({ type: 'SELECT_TRACK', payload: { trackId: selectedTrack.id, busId: ownerBus.id } });
          onClose?.();
        }
        return;
      }

      // Fallback (no selected track AND no selected bus — text-only): run
      // a single pass and create a fresh bus + track from the result.
      const params = { ...baseParams, duration: 16 };
      const start = await generateStemphonic(params, null, null);
      if (!start.task_id) throw new Error('No task_id returned');
      setStatus(`task ${start.task_id.slice(0, 8)}… running`);
      const result = await pollStemphonicUntilComplete(start.task_id, (p) => {
        setStatus(`${p.status || '…'}${p.attempts ? ` · ${p.attempts}s` : ''}`);
      });
      setStatus('done');
      const filePaths = result?.file_paths || [];
      if (filePaths.length) {
        const firstUrl = filePaths[0];
        const now = Date.now();
        const newCandidateVersions = filePaths.map((url, i) => ({
          audioUrl: url,
          timestamp: now,
          type: 'generated',
          name: filePaths.length > 1 ? `Gen · ${selectedInstrument.label} · c${i + 1}` : `Gen · ${selectedInstrument.label}`,
          params,
        }));
        const busId = `bus-gen-${now}`;
        const trackId = `t-gen-${now}`;
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
              name: `Gen · ${selectedInstrument.label}`,
              audioUrl: firstUrl,
              duration: result.duration || params.duration,
              startPosition: 0,
              gain: 1, isMuted: false, isSolo: false,
              fx: { reverb: 0, fadeIn: 0, fadeOut: 0 },
              metadata: {
                type: 'generated', source: 'stemphonic',
                instrument: baseParams.instrument, timbre_preset: params.timbre_preset,
                prompt: params.prompt, params, timestamp: now,
                versions: newCandidateVersions,
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
        {selectedInstrument && (
          <div className="sd-gen-input-info">
            <span>Instrument</span>
            <strong>{selectedInstrument.label}</strong>
          </div>
        )}
        {inputInfo.kind !== 'none' && (
          <div className="sd-gen-input-info">
            <span>Conditioning {inputInfo.kind === 'midi' ? 'MIDI' : 'audio'}</span>
            <strong>{inputInfo.name}</strong>
          </div>
        )}
        {/* Bus target indicator — only when a bus is selected without a
         * focused track. Tells the user generation will fan out to every
         * eligible child track in the bus (one pass each, replacing
         * audio + appending to per-track versions). */}
        {busTargetInfo && (
          <div className="sd-gen-input-info">
            <span>Bus target</span>
            <strong>
              {busTargetInfo.name} · {busTargetInfo.eligible}/{busTargetInfo.total} track{busTargetInfo.total === 1 ? '' : 's'}
            </strong>
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
        {running && (
          <div className="sd-gen-loading" aria-hidden="true">
            <DotLottieReact
              src="/assets/doloading.lottie"
              loop
              autoplay
              style={{ width: 80, height: 80 }}
            />
          </div>
        )}
      </div>
    </div>
  );
}
