/*
 * detectChordsFromMIDI — JS port of stemphonic_server.py's symbolic
 * chord detector (_detect_chords_from_midi + _match_chord_from_pcs +
 * _CHORD_TEMPLATES). Runs entirely client-side on the MIDI notes that
 * LatentPitch already produced per stem — zero backend roundtrip.
 *
 * Usage:
 *   import { detectChordsFromNotes } from './detectChordsFromMIDI';
 *   const chords = detectChordsFromNotes(notes, beatTimes);
 *   // chords = { '0': 'Am7', '4': 'G7', ... }  — 1 entry per beat
 *   //   where the chord CHANGES; unchanged beats are omitted.
 *
 * Notes shape (matches the app's internal MIDI note model):
 *   [{ start: seconds, end: seconds, note|midi|pitch: int, velocity: 0..1|0..127 }, ...]
 *
 * Skip notes with `isDrum: true` or notes from stems whose stemType is
 * 'drums' / 'drum_kit' — drums would produce nonsense chord matches.
 *
 * The matcher tries all 12 roots × 12 templates and picks the highest
 * (matched - missing - 0.5 × extra) score, with a small bonus for
 * fuller chord forms. Root must be present (no rootless voicings).
 * Bass is the lowest active MIDI pitch in the beat window — if it
 * differs from the root, the label becomes "C/E" style slash chord.
 */

const CHORD_TEMPLATES = [
  ['maj7',  [0, 4, 7, 11]],
  ['m7',    [0, 3, 7, 10]],
  ['7',     [0, 4, 7, 10]],
  ['dim7',  [0, 3, 6, 9]],
  ['m7b5',  [0, 3, 6, 10]],
  ['',      [0, 4, 7]],       // major triad
  ['m',     [0, 3, 7]],       // minor triad
  ['dim',   [0, 3, 6]],
  ['aug',   [0, 4, 8]],
  ['sus4',  [0, 5, 7]],
  ['sus2',  [0, 2, 7]],
  ['5',     [0, 7]],          // power chord fallback
];

const PITCH_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'];

function chordScore(activePcs, templatePcs) {
  const act = new Set(activePcs);
  const tmpl = new Set(templatePcs);
  let matched = 0, missing = 0, extra = 0;
  for (const p of tmpl) { if (act.has(p)) matched++; else missing++; }
  for (const p of act) { if (!tmpl.has(p)) extra++; }
  return { matched, missing, extra };
}

/**
 * Pick the best chord label for a pitch-class set + optional bass pc.
 * Returns null if no template scores above the minimum threshold.
 */
export function matchChordFromPcs(activePcs, bassPc = null) {
  if (!activePcs || activePcs.size === 0) return null;
  let best = null;
  let bestScore = -Infinity;
  for (let root = 0; root < 12; root++) {
    if (!activePcs.has(root)) continue;          // no rootless voicings
    for (const [suffix, offsets] of CHORD_TEMPLATES) {
      const tmpl = offsets.map((o) => (root + o) % 12);
      const { matched, missing, extra } = chordScore(activePcs, tmpl);
      let score = matched - missing - 0.5 * extra;
      // Prefer fuller chords on ties.
      score += 0.01 * offsets.length;
      if (score > bestScore) {
        bestScore = score;
        best = [root, suffix];
      }
    }
  }
  if (best === null || bestScore < 1.5) return null;
  const [root, suffix] = best;
  let label = `${PITCH_NAMES[root]}${suffix}`;
  if (bassPc !== null && bassPc !== undefined && bassPc !== root) {
    label = `${label}/${PITCH_NAMES[bassPc]}`;
  }
  return label;
}

/**
 * Given a pool of notes + a beat-time array, return {beat_idx_string:
 * chord_label} for every beat whose chord differs from the prior one.
 *
 * @param {Array} notes - [{start, end, pitch?|note?|midi?, velocity?, isDrum?}]
 * @param {number[]|Float32Array} beatTimes - seconds of each beat downbeat
 * @returns {Object} {'0': 'C', '4': 'G', ...} — sparse by-beat map
 */
export function detectChordsFromNotes(notes, beatTimes) {
  if (!Array.isArray(notes) || notes.length === 0 || !beatTimes || beatTimes.length === 0) {
    return {};
  }
  // Normalize: pitch from any of .pitch/.note/.midi; skip drum notes.
  const flat = [];
  for (const n of notes) {
    if (n.isDrum) continue;
    const pitch = Number.isFinite(n.pitch) ? n.pitch
                : Number.isFinite(n.note) ? n.note
                : Number.isFinite(n.midi) ? n.midi
                : null;
    if (pitch === null) continue;
    const start = Number.isFinite(n.start) ? n.start
                : Number.isFinite(n.time) ? n.time
                : null;
    if (start === null) continue;
    const dur = Number.isFinite(n.duration) ? n.duration : null;
    const end = Number.isFinite(n.end) ? n.end
              : dur !== null ? start + dur
              : start + 0.25;
    flat.push({ start, end, pitch });
  }
  if (flat.length === 0) return {};
  flat.sort((a, b) => a.start - b.start);

  const chords = {};
  let lastLabel = null;
  const nBeats = beatTimes.length;
  for (let i = 0; i < nBeats; i++) {
    const t0 = beatTimes[i];
    const t1 = (i + 1 < nBeats) ? beatTimes[i + 1] : t0 + 0.5;
    // Active notes: start < t1 AND end > t0. Binary-search friendly
    // since flat is sorted by start, but linear scan is fine for the
    // note counts we see (typically ~hundreds per stem).
    const active = flat.filter((n) => n.start < t1 && n.end > t0);
    if (active.length === 0) continue;
    const pcs = new Set(active.map((n) => ((n.pitch % 12) + 12) % 12));
    let bassMidi = Infinity;
    for (const n of active) if (n.pitch < bassMidi) bassMidi = n.pitch;
    const bassPc = Number.isFinite(bassMidi) ? ((bassMidi % 12) + 12) % 12 : null;
    const label = matchChordFromPcs(pcs, bassPc);
    if (label === null) continue;
    if (label !== lastLabel) {
      chords[String(i)] = label;
      lastLabel = label;
    }
  }
  return chords;
}

/**
 * Re-run chord detection from a map of stem-name → notes array and a
 * beat_map (or a constant bpm fallback). Returns the sparse chord map
 * ready to dispatch via SET_CHORDS (keys as numbers).
 *
 * Use this when BasicPitch (the high-quality server-side pass) finishes
 * a stem and replaces the latent-pitch placeholder — call it from the
 * onMidiReady handler after updating track metadata. Debounce on the
 * caller side if you want one re-detection per burst of swaps.
 */
export function rerunChordDetection(notesByStem, { beatMap, bpm }) {
  const pooled = [];
  for (const [stem, notes] of Object.entries(notesByStem || {})) {
    const s = (stem || '').toLowerCase();
    if (s === 'drums' || s === 'drum_kit' || s === 'percussion') continue;
    if (!Array.isArray(notes)) continue;
    for (const n of notes) {
      const start = Number.isFinite(n.start) ? n.start
                  : Number.isFinite(n.time) ? n.time : null;
      if (start === null) continue;
      const dur = Number.isFinite(n.duration) ? n.duration : 0.25;
      const pitch = Number.isFinite(n.pitch) ? n.pitch
                  : Number.isFinite(n.note) ? n.note
                  : Number.isFinite(n.midi) ? n.midi : null;
      if (pitch === null) continue;
      pooled.push({ start, end: start + dur, pitch });
    }
  }
  if (pooled.length === 0) return {};

  let beatTimes;
  if (Array.isArray(beatMap) && beatMap.length > 0) {
    beatTimes = beatMap.map((b) => b.t);
  } else {
    const spb = 60 / (bpm || 120);
    const maxEnd = pooled.reduce((m, n) => Math.max(m, n.end), 0);
    const nBeats = Math.ceil(maxEnd / spb) + 1;
    beatTimes = Array.from({ length: nBeats }, (_, i) => i * spb);
  }
  const chords = detectChordsFromNotes(pooled, beatTimes);
  const out = {};
  for (const [k, v] of Object.entries(chords)) out[parseInt(k, 10)] = v;
  return out;
}

/**
 * Pool notes from a set of stem tracks (excluding drums), offset each
 * note by its track.startPosition so every note lives on the project
 * timeline, and return the combined note list ready for
 * detectChordsFromNotes.
 *
 * Each track expected to have `metadata.midiData.notes` populated by
 * LatentPitch (see StudioDev.js upload flow).
 */
export function poolMidiFromStems(stemTracks) {
  const notes = [];
  for (const t of stemTracks || []) {
    const stemType = (t?.metadata?.stemType || t?.metadata?.instrument || '').toLowerCase();
    if (stemType === 'drums' || stemType === 'drum_kit' || stemType === 'percussion') continue;
    const md = t?.metadata?.midiData;
    if (!md || !Array.isArray(md.notes)) continue;
    const offset = t.startPosition || 0;
    for (const n of md.notes) {
      const start = Number.isFinite(n.start) ? n.start
                  : Number.isFinite(n.time) ? n.time : null;
      if (start === null) continue;
      const dur = Number.isFinite(n.duration) ? n.duration : 0.25;
      const pitch = Number.isFinite(n.pitch) ? n.pitch
                  : Number.isFinite(n.note) ? n.note
                  : Number.isFinite(n.midi) ? n.midi : null;
      if (pitch === null) continue;
      notes.push({ start: start + offset, end: start + offset + dur, pitch });
    }
  }
  return notes;
}
