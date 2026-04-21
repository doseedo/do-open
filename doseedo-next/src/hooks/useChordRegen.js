import { useCallback } from 'react';
import { useApp } from '../context/AppContext';
import { regenStemForChord } from '../services/trackAnalysisAPI';

/**
 * useChordRegen — smart per-stem regeneration when a chord changes.
 *
 * Extracted from components/ChordWindow/ChordWindow.js so /studio-dev
 * (StudioDevChords.js) can fire the same pipeline when a user edits a
 * chord cell. Both routes then stay in lockstep with identical behavior:
 *   • Find every track overlapping the bar for the edited beat
 *   • Pick ONE harmonic track to own chord extensions (9/11/13)
 *   • Send triad versions to the other harmonic tracks
 *   • Poll each regen task and swap audioUrl on success
 */
export default function useChordRegen() {
  const { state, dispatch } = useApp();

  return useCallback(async ({ beatIndex, oldChord, newChord }) => {
    if (!newChord || newChord === oldChord) return;

    const bpm = state.bpm || 120;
    const beatsPerBar = state.beatsPerBar || 4;
    const secondsPerBeat = 60 / bpm;
    const barIndex = Math.floor(beatIndex / beatsPerBar);
    const regionStart = barIndex * beatsPerBar * secondsPerBeat;
    const regionEnd = regionStart + beatsPerBar * secondsPerBeat;

    const candidates = [];
    (state.buses || []).forEach((bus) => {
      (bus.tracks || []).forEach((tr) => {
        if (!tr.audioUrl || !tr.metadata?.instrument) return;
        const sp = tr.startPosition || 0;
        const dur = tr.duration || 0;
        if (sp + dur < regionStart || sp > regionEnd) return;
        candidates.push({ bus, track: tr });
      });
    });
    if (!candidates.length) return;

    // Only ONE harmonic instrument owns extensions for a given chord.
    const HARMONIC = new Set(['guitar', 'piano', 'keys', 'synth', 'harmony']);
    const harmonics = candidates.filter(({ track }) => HARMONIC.has(track.metadata.instrument));
    let extensionOwnerId = null;
    if (harmonics.length) {
      const PRIORITY = { piano: 5, keys: 4, guitar: 3, synth: 2, harmony: 1 };
      harmonics.sort((a, b) =>
        (PRIORITY[b.track.metadata.instrument] || 0) - (PRIORITY[a.track.metadata.instrument] || 0)
      );
      extensionOwnerId = harmonics[0].track.id;
    }

    for (const { track } of candidates) {
      const role = track.metadata.instrument;
      let chordForThisTrack = newChord;
      if (HARMONIC.has(role) && track.id !== extensionOwnerId) {
        chordForThisTrack = newChord.replace(/(9|11|13|maj9|m9|maj11|m11|maj13|m13|add9)$/i, '');
      }
      try {
        const audioBlob = await (await fetch(track.audioUrl)).blob();
        const audioFile = new File([audioBlob], (track.name || 'stem') + '.wav', { type: 'audio/wav' });
        let midiFile = null;
        if (track.metadata?.midi) {
          try {
            const mb = await (await fetch(track.metadata.midi)).blob();
            midiFile = new File([mb], 'input.mid', { type: 'audio/midi' });
          } catch (_) { /* best effort */ }
        }
        const result = await regenStemForChord({
          audioFile, midiFile, role,
          oldChord, newChord: chordForThisTrack,
          regionStart, regionEnd,
          coverNoise: 0.7,
          duration: track.duration,
        });
        if (result.skipped || !result.task_id) continue;

        const taskId = result.task_id;
        (async () => {
          for (let i = 0; i < 300; i++) {
            await new Promise((r) => setTimeout(r, 2000));
            const r2 = await fetch(`/api/generate-stemphonic/task/${taskId}`);
            if (!r2.ok) continue;
            const tr2 = await r2.json();
            if (tr2.state === 'SUCCESS' && tr2.result?.file_paths?.[0]) {
              const newUrl = tr2.result.file_paths[0];
              const trackBusId = (state.buses || []).find((b) => (b.tracks || []).some((t) => t.id === track.id))?.id;
              dispatch({
                type: 'UPDATE_TRACK',
                payload: {
                  busId: trackBusId,
                  trackId: track.id,
                  updates: {
                    audioUrl: newUrl,
                    metadata: {
                      ...track.metadata,
                      lastChordRegen: { from: oldChord, to: chordForThisTrack, at: Date.now() },
                    },
                  },
                },
              });
              return;
            }
            if (tr2.state === 'FAILURE') return;
          }
        })();
      } catch (err) {
        console.error(`[chord-regen] ${role} failed:`, err?.message || err);
      }
    }
  }, [state.buses, state.bpm, state.beatsPerBar, dispatch]);
}
