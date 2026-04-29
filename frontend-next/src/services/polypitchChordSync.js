/**
 * polypitchChordSync — orchestrate polypitch resynth when the chord row
 * changes. Called by StudioDev's chord-diff effect.
 *
 * Scope: for each (stem track, beatIndex) pair whose chord moved, pick the
 * notes in that beat window, voice-lead them against the new chord, render
 * new audio with polypitchService, swap the track's audioUrl. Skips drum
 * stems entirely — pitch-shifting a kick doesn't help anybody.
 *
 * Keeps the latest per-track render promise so rapid chord edits don't pile
 * up N concurrent renders — a later edit during an in-flight render wins.
 *
 * This runs ON the main thread for now; the polypitch Pipeline itself does
 * the WebGPU + WASM heavy lifting and is already singleton-guarded inside
 * polypitchService.getPipeline().
 */

import { logPipeline } from './pipelineStatus';
import { notesFromMidiData, renderWithNewPitches } from './polypitchService';
import { voiceLeadForChordChange } from './polypitchVoicing';

/**
 * Resolve a beatIndex to a [startSec, endSec] window — the full BAR
 * containing `beatIndex`, not just the single beat. At 112 BPM a beat
 * window is ~0.54 s, and most pop stems (especially "other") only have
 * one note every ~1.5–2 s after BasicPitch refinement, so a single-beat
 * window often caught zero notes on the tonal stems that actually carry
 * the chord. Widening to one bar reliably captures 2–4 notes per stem
 * and, musically, a chord change is anchored at bar boundaries anyway.
 */
function beatWindowSec(beatIndex, { beatMap, bpm, beatsPerBar = 4 }) {
  const i = Math.max(0, beatIndex | 0);
  const barStartBeat = Math.floor(i / beatsPerBar) * beatsPerBar;
  const barEndBeat = barStartBeat + beatsPerBar;
  if (Array.isArray(beatMap) && beatMap.length > 0) {
    const t0 = beatMap[barStartBeat]?.t ?? beatMap[beatMap.length - 1].t;
    const t1 = beatMap[barEndBeat]?.t
      ?? (beatMap[beatMap.length - 1].t + (bpm > 0 ? 60 / bpm : 0.5));
    return [t0, t1];
  }
  const sec = bpm > 0 ? 60 / bpm : 0.5;
  return [barStartBeat * sec, barEndBeat * sec];
}

const _renderVersionByTrackId = new Map();
const _inFlightVersionByTrackId = new Map();

/**
 * Apply a newPitches map (noteId → new MIDI) to a midiData.notes array,
 * returning a new array with the shifted pitches. Ids are computed with
 * the same formula notesFromMidiData uses (pitch + time), so a lookup
 * here matches what the voicing layer emitted.
 */
function applyPitchesToMidiNotes(notes, newPitches) {
  if (!Array.isArray(notes) || newPitches.size === 0) return notes;
  let changed = false;
  const out = notes.map((n) => {
    const id = `${n.note | 0}@${n.time.toFixed(3)}`;
    const target = newPitches.get(id);
    if (typeof target !== 'number' || target === (n.note | 0)) return n;
    changed = true;
    return { ...n, note: target };
  });
  return changed ? out : notes;
}

/**
 * @param {object} args
 * @param {number} args.beatIndex — chord row beat position whose label changed
 * @param {string|null} args.oldChord — previous chord label (may be null on first set)
 * @param {string|null} args.newChord — new chord label ('' or null = erase)
 * @param {Array} args.pitchedStemTracks — [{id, busId, audioUrl, metadata?, midiData? ...}]
 * @param {{beatMap: Array, bpm: number}} args.tempo
 * @param {(trackId:string, busId:string, newUrl:string) => void} args.onTrackAudioReady
 * @param {(trackId:string, busId:string, newMidiData:object) => void} [args.onTrackMidiReady]
 *   Optional: called with the updated midiData (same shape as input, with
 *   shifted note.note values) so the MIDI-window view can re-render.
 */
export async function applyChordChange({
  beatIndex, oldChord, newChord,
  pitchedStemTracks, tempo, onTrackAudioReady, onTrackMidiReady,
}) {
  if (!newChord || oldChord === newChord) return;
  if (!Array.isArray(pitchedStemTracks) || pitchedStemTracks.length === 0) return;

  const [t0, t1] = beatWindowSec(beatIndex, tempo);

  for (const track of pitchedStemTracks) {
    const midiData = track.midiData || track.metadata?.midiData;
    if (!midiData || !Array.isArray(midiData.notes) || midiData.notes.length === 0) continue;
    if (!track.audioUrl) continue;

    // Pick notes whose ONSET falls in the beat window. Using onset (not
    // overlap) keeps sustained notes that ring across multiple chords on
    // the earlier chord's voicing, which matches how a human would hear it.
    const notesInWindow = midiData.notes.filter((n) => n.time >= t0 && n.time < t1);
    if (notesInWindow.length === 0) continue;

    const poly = notesFromMidiData({ notes: notesInWindow });
    const newPitches = voiceLeadForChordChange(poly, oldChord, newChord);
    if (newPitches.size === 0) continue;

    const version = (_renderVersionByTrackId.get(track.id) || 0) + 1;
    _renderVersionByTrackId.set(track.id, version);

    logPipeline(
      'polypitch',
      `${track.name || track.id}: ${newPitches.size} note(s) for ${oldChord || '—'} → ${newChord}`,
    );

    // Kick off the render without awaiting the whole batch — per-track
    // pipeline is a singleton so multiple stems serialize through it anyway,
    // but they don't block each other's scheduling.
    (async () => {
      try {
        _inFlightVersionByTrackId.set(track.id, version);
        const blob = await renderWithNewPitches({
          audioUrl: track.audioUrl,
          notes: notesFromMidiData(midiData),
          newPitches,
        });
        if (!blob) return;
        if (_renderVersionByTrackId.get(track.id) !== version) {
          // A newer chord edit landed while we were rendering; discard.
          logPipeline(
            'polypitch',
            `${track.name || track.id}: discarded stale render (newer edit pending)`,
            'warn',
          );
          return;
        }
        const url = URL.createObjectURL(blob);
        logPipeline(
          'polypitch',
          `${track.name || track.id}: swapping audioUrl → ${url.slice(0, 40)}…`,
          'ok',
        );
        onTrackAudioReady(track.id, track.busId, url);

        // Mirror the same pitch edits into midiData so the MIDI window
        // reflects the new voicing. Without this the audio moves but the
        // piano roll keeps showing the original pitches.
        if (typeof onTrackMidiReady === 'function') {
          const newNotes = applyPitchesToMidiNotes(midiData.notes, newPitches);
          if (newNotes !== midiData.notes) {
            onTrackMidiReady(track.id, track.busId, { ...midiData, notes: newNotes });
          }
        }
      } catch (err) {
        logPipeline('polypitch', `${track.id}: ${err?.message || err}`, 'error');
      } finally {
        if (_inFlightVersionByTrackId.get(track.id) === version) {
          _inFlightVersionByTrackId.delete(track.id);
        }
      }
    })();
  }
}
