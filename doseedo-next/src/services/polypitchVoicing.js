/**
 * polypitchVoicing — voice-lead notes from an old chord to a new one.
 *
 * Strategy is deliberately simple: for each note in the time window, pick the
 * chord tone in the NEW chord that's nearest (modulo octave) to the note's
 * current pitch, then preserve the note's original octave. Outliers outside
 * the old chord fall through to the "closest new chord tone" fallback anyway,
 * so non-chord tones (passing tones, melody leaps) still end up musical.
 *
 * We can upgrade to a DP-based voice-leader (Levine / Tymoczko-style) once
 * this nearest-tone path is proven end-to-end with polypitch.
 *
 * Chord parsing: reuses detectChordsFromMIDI's CHORD_TEMPLATES so the round-
 * trip (midi → chord label → midi) matches what the detector emits.
 */

import { CHORD_TEMPLATES, PITCH_NAMES } from './detectChordsFromMIDI';

const PITCH_CLASS_BY_NAME = (() => {
  const m = new Map();
  PITCH_NAMES.forEach((n, i) => m.set(n, i));
  // Accept enharmonic spellings we don't emit but users might type.
  const extras = [
    ['Db', 1], ['D#', 3], ['Gb', 6], ['G#', 8], ['A#', 10],
    ['Cb', 11], ['B#', 0], ['Fb', 4], ['E#', 5],
  ];
  for (const [n, i] of extras) m.set(n, i);
  return m;
})();

const TEMPLATE_BY_QUALITY = (() => {
  const m = new Map();
  for (const [q, pcs] of CHORD_TEMPLATES) m.set(q, pcs);
  // Common synonyms the detector doesn't output but users type.
  m.set('M7', CHORD_TEMPLATES.find(([q]) => q === 'maj7')[1]);
  m.set('min7', CHORD_TEMPLATES.find(([q]) => q === 'm7')[1]);
  m.set('dom7', CHORD_TEMPLATES.find(([q]) => q === '7')[1]);
  m.set('°7', CHORD_TEMPLATES.find(([q]) => q === 'dim7')[1]);
  m.set('ø7', CHORD_TEMPLATES.find(([q]) => q === 'm7b5')[1]);
  m.set('+', CHORD_TEMPLATES.find(([q]) => q === 'aug')[1]);
  m.set('°', CHORD_TEMPLATES.find(([q]) => q === 'dim')[1]);
  m.set('M', CHORD_TEMPLATES.find(([q]) => q === '')[1]);
  m.set('min', CHORD_TEMPLATES.find(([q]) => q === 'm')[1]);
  return m;
})();

/**
 * Parse a chord symbol like "Cmaj7", "Am7/E", "G#dim" into its pitch-class set.
 *
 * @param {string} symbol
 * @returns {{rootPc:number, pitchClasses:Set<number>, bassPc:number|null} | null}
 */
export function parseChord(symbol) {
  if (!symbol || typeof symbol !== 'string') return null;
  const trimmed = symbol.trim();
  if (!trimmed || trimmed.toUpperCase() === 'N.C.' || trimmed === '—') return null;

  // Split slash-chord bass.
  let bassPc = null;
  let core = trimmed;
  const slash = trimmed.indexOf('/');
  if (slash > 0) {
    core = trimmed.slice(0, slash);
    const bassName = trimmed.slice(slash + 1).trim();
    if (PITCH_CLASS_BY_NAME.has(bassName)) bassPc = PITCH_CLASS_BY_NAME.get(bassName);
    else if (PITCH_CLASS_BY_NAME.has(bassName[0])) bassPc = PITCH_CLASS_BY_NAME.get(bassName[0]);
  }

  // Root note = longest matching prefix (try 2-char accidentals first).
  let rootPc = null;
  let quality = '';
  if (core.length >= 2 && PITCH_CLASS_BY_NAME.has(core.slice(0, 2))) {
    rootPc = PITCH_CLASS_BY_NAME.get(core.slice(0, 2));
    quality = core.slice(2);
  } else if (PITCH_CLASS_BY_NAME.has(core[0])) {
    rootPc = PITCH_CLASS_BY_NAME.get(core[0]);
    quality = core.slice(1);
  } else {
    return null;
  }

  const intervals = TEMPLATE_BY_QUALITY.get(quality) ?? TEMPLATE_BY_QUALITY.get('');
  const pitchClasses = new Set(intervals.map((iv) => (rootPc + iv) % 12));
  return { rootPc, pitchClasses, bassPc };
}

/** Nearest target pitch class to `pc`, modulo 12. Returns the target pc + delta. */
function nearestPitchClass(pc, targets) {
  let bestPc = pc;
  let bestDelta = 0;
  let bestAbs = Infinity;
  for (const t of targets) {
    // Signed minimum-distance delta in [-6, 6].
    let d = t - pc;
    if (d > 6) d -= 12;
    else if (d < -6) d += 12;
    if (Math.abs(d) < bestAbs) {
      bestAbs = Math.abs(d);
      bestDelta = d;
      bestPc = t;
    }
  }
  return { pc: bestPc, delta: bestDelta };
}

/**
 * Produce a per-note target MIDI pitch for `notes` given that the chord in
 * the surrounding window is moving from `oldChord` to `newChord`.
 *
 * @param {Array<{id:string, pitchMidi:number}>} notes
 * @param {string} oldChord  e.g. "Cmaj7"
 * @param {string} newChord  e.g. "Am7"
 * @returns {Map<string, number>} noteId → new MIDI pitch (integer). Notes that
 *   stay unchanged are omitted so the caller can trivially check if any work
 *   is needed.
 */
export function voiceLeadForChordChange(notes, oldChord, newChord) {
  const out = new Map();
  const from = parseChord(oldChord);
  const to = parseChord(newChord);
  if (!to) return out;
  const targets = [...to.pitchClasses];
  if (targets.length === 0) return out;

  for (const note of notes) {
    if (typeof note.pitchMidi !== 'number') continue;
    const midi = note.pitchMidi;
    const pc = ((midi % 12) + 12) % 12;

    // If the note is already a tone of the new chord, leave it alone. This
    // keeps common tones stable (parsimonious voice leading) and avoids any
    // polypitch render when the chord extension adds notes that overlap.
    if (to.pitchClasses.has(pc)) continue;

    // If we know the old chord AND this note was a chord tone in it, map it
    // through the closest new chord tone. If we DON'T know the old chord (or
    // the note was non-chord), fall through to the same nearest-tone pick —
    // the behaviour matches either way, the `from` parse is advisory.
    const { delta } = nearestPitchClass(pc, targets);
    if (delta === 0) continue;
    out.set(note.id, midi + delta);
  }
  return out;
}
