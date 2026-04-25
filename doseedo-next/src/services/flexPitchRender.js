/**
 * Flex-pitch render — feed Logic's per-note pitch edits into polypitch.
 *
 * Logic emits `session.flexPitchNotes` as `[{ pitch, formant, time_s, dur_s,
 * auflGid? }]`. Each entry is the EDITED pitch at the given time/duration.
 * To shift audio, polypitchService needs both the ORIGINAL pitches (so it
 * knows what's already in the source) and the new targets. We pull
 * originals from `track.midiData.notes` (populated by latentPitch /
 * basic-pitch upstream), match flexPitchNotes onto those by time, build
 * a `newPitches` Map<noteId, newMidi>, and call
 * `renderWithNewPitches({ audioUrl, notes, newPitches })` — same offline
 * WebGPU pipeline polypitchChordSync uses for chord re-renders.
 *
 * Triggered once per track per (track.audioUrl, flexPitch fingerprint).
 * The rendered blob is cached so a second play() doesn't re-render.
 *
 * Out-of-scope here:
 *   - Per-hop detune (EAFP tags 0x02/0x03) — needs bytes the backend
 *     hasn't bound to specific clips. We render at note granularity.
 *   - Formant shifts — polypitch shifts pitch only; Logic's formant
 *     value is recorded but not applied. Audible only on extreme shifts.
 *   - Tracks without midiData — basic-pitch detection isn't run here;
 *     a follow-on can call basicPitchOnnx first if the data is missing.
 */

import { notesFromMidiData, renderWithNewPitches } from './polypitchService';
import { logPipeline } from './pipelineStatus';

// Cache of rendered blob URLs keyed by `${track.audioUrl}|${fingerprint}`.
// Survives across play() cycles in a single page session.
const _renderCache = new Map();
// Tracks renders currently in flight to dedupe concurrent triggers.
const _inFlight = new Map();

function _fingerprint(flexPitchNotes) {
  if (!Array.isArray(flexPitchNotes) || flexPitchNotes.length === 0) return '0';
  // Round to ms to keep tiny float jitter from busting the cache.
  const parts = flexPitchNotes.map((n) =>
    `${Math.round((Number(n.time_s) || 0) * 1000)}:${Math.round((Number(n.dur_s) || 0) * 1000)}:${n.pitch}`);
  return parts.join('|');
}

/**
 * Filter session.flexPitchNotes to those overlapping a track's timeline
 * window. Tracks have `startPosition` and `duration` on the timeline; flex
 * notes carry absolute `time_s` (per the schema). A note is "in window"
 * if its onset falls within the track. Multiple tracks overlapping in
 * time will both pick up the same notes — the audible effect is the same
 * (whichever track's source contains that pitch gets shifted), and the
 * non-overlapping track's render is a no-op since no detected note will
 * match a non-existent pitch.
 */
function _notesInTrackWindow(flexPitchNotes, track) {
  if (!Array.isArray(flexPitchNotes)) return [];
  const t0 = Number(track.startPosition) || 0;
  const t1 = t0 + (Number(track.duration) || 0);
  return flexPitchNotes.filter((n) => {
    const t = Number(n?.time_s);
    return Number.isFinite(t) && t >= t0 && t < t1;
  });
}

/** Find the basic-pitch note nearest a target time, within tolerance. */
function _matchNoteAtTime(notes, targetTime, toleranceSec = 0.05) {
  let best = null;
  let bestDist = Infinity;
  for (const n of notes) {
    const t = Number(n.startSec);
    const dist = Math.abs(t - targetTime);
    if (dist < bestDist && dist <= toleranceSec) {
      best = n;
      bestDist = dist;
    }
  }
  return best;
}

/**
 * Render a track with flex-pitch edits applied, if applicable.
 *
 * @param {object} args
 * @param {object} args.track                  Track object (must have audioUrl + midiData).
 * @param {Array}  args.flexPitchNotes         Filtered list (or full session list — we filter).
 * @param {(url: string) => void} [args.onRendered]   Called with the rendered blob URL.
 * @returns {Promise<string | null>}           Resolves to URL or null when nothing to do.
 */
export async function renderTrackFlexPitch({ track, flexPitchNotes, onRendered }) {
  if (!track || !track.audioUrl) return null;
  const midi = track.midiData || track.metadata?.midiData;
  if (!midi || !Array.isArray(midi.notes) || midi.notes.length === 0) {
    // Without basic-pitch / midi notes we can't drive polypitch at the
    // note level. This is a known gap (see file header).
    return null;
  }
  const inWindow = _notesInTrackWindow(flexPitchNotes, track);
  if (inWindow.length === 0) return null;

  const fp = _fingerprint(inWindow);
  const cacheKey = `${track.audioUrl}|${fp}`;

  const cached = _renderCache.get(cacheKey);
  if (cached) {
    if (typeof onRendered === 'function') onRendered(cached);
    return cached;
  }
  if (_inFlight.has(cacheKey)) return _inFlight.get(cacheKey);

  const job = (async () => {
    const polyNotes = notesFromMidiData(midi);
    const newPitches = new Map();
    for (const fpNote of inWindow) {
      const target = _matchNoteAtTime(polyNotes, Number(fpNote.time_s));
      if (!target) continue;
      const newMidi = Number(fpNote.pitch);
      if (!Number.isFinite(newMidi) || newMidi === target.pitchMidi) continue;
      newPitches.set(target.id, newMidi);
    }
    if (newPitches.size === 0) {
      logPipeline('polypitch',
        `${track.name || track.id}: no flex-pitch notes matched detected pitches; skipping render`,
        'warn');
      return null;
    }
    logPipeline('polypitch',
      `${track.name || track.id}: rendering ${newPitches.size}/${inWindow.length} flex-pitch edits`);
    try {
      const blob = await renderWithNewPitches({
        audioUrl: track.audioUrl, notes: polyNotes, newPitches,
      });
      if (!blob) return null;
      const url = URL.createObjectURL(blob);
      _renderCache.set(cacheKey, url);
      if (typeof onRendered === 'function') onRendered(url);
      return url;
    } catch (err) {
      logPipeline('polypitch',
        `${track.name || track.id}: flex-pitch render failed: ${err?.message || err}`,
        'error');
      return null;
    }
  })();
  _inFlight.set(cacheKey, job);
  job.finally(() => _inFlight.delete(cacheKey));
  return job;
}

/** Test/debug only: clear the render cache. */
export function _resetFlexPitchCache() {
  _renderCache.clear();
  _inFlight.clear();
}
