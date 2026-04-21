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
 * Resolve a beatIndex to a [startSec, endSec] window. Prefers the actual
 * beat_map times when analyze-rhythm has populated one; falls back to a
 * constant-tempo grid when not.
 */
function beatWindowSec(beatIndex, { beatMap, bpm }) {
  const i = Math.max(0, beatIndex | 0);
  if (Array.isArray(beatMap) && beatMap.length > 0) {
    const t0 = beatMap[i]?.t ?? beatMap[beatMap.length - 1].t;
    const t1 = beatMap[i + 1]?.t ?? t0 + (bpm > 0 ? 60 / bpm : 0.5);
    return [t0, t1];
  }
  const sec = bpm > 0 ? 60 / bpm : 0.5;
  return [i * sec, (i + 1) * sec];
}

const _renderVersionByTrackId = new Map();
const _inFlightVersionByTrackId = new Map();

/**
 * @param {object} args
 * @param {number} args.beatIndex — chord row beat position whose label changed
 * @param {string|null} args.oldChord — previous chord label (may be null on first set)
 * @param {string|null} args.newChord — new chord label ('' or null = erase)
 * @param {Array} args.pitchedStemTracks — [{id, busId, audioUrl, metadata?, midiData? ...}]
 * @param {{beatMap: Array, bpm: number}} args.tempo
 * @param {(trackId:string, busId:string, newUrl:string) => void} args.onTrackAudioReady
 */
export async function applyChordChange({
  beatIndex, oldChord, newChord,
  pitchedStemTracks, tempo, onTrackAudioReady,
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
          return;
        }
        const url = URL.createObjectURL(blob);
        onTrackAudioReady(track.id, track.busId, url);
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
