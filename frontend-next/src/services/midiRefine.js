/*
 * midiRefine — hybrid master-vs-stem MIDI refinement.
 *
 * Problem: on upload we produce several transcriptions at different
 * tiers of quality, arriving at different times:
 *   1. Master BasicPitch (in-browser, WebGPU, ~5–10s after upload).
 *      High note accuracy on pitched content, but attributed to the
 *      ENTIRE MIX — no stem separation.
 *   2. Per-stem latentPitch (in-browser, after stem-separation latents
 *      are available, ~3–5s after upload). Attributes to a specific
 *      stem but noisy (octave ghosts, harmonic fires, false positives).
 *   3. Per-stem backend BasicPitch (server-side, ~15–30s after upload).
 *      Ground truth: accurate notes AND correct stem attribution.
 *
 * This module fuses (1) + (2) into a REFINED per-stem transcription
 * while (3) is still running:
 *
 *   refineStemWithMaster(stemNotes, masterNotes):
 *     For each stem note at (t, pitch):
 *       - If master has a note at (t ± TOL, pitch) → KEEP it, snap
 *         timing to master (BasicPitch's onset timing is better than
 *         latentPitch's).
 *       - Else → DROP (likely a false positive from lossy stem path).
 *
 *   splitMasterByStems(masterNotes, stemNotesByName):
 *     For every master note, find which stem's latentPitch had the
 *     strongest presence at that (t, pitch). Assign the master note
 *     to that stem. Master notes with no stem backing stay un-assigned
 *     (they could be reverb tails, overtones, etc.). This recovers
 *     master notes that the stem's noisy transcription missed.
 *
 * The combination of the two gives per-stem MIDI that's AT LEAST as
 * complete as latentPitch alone AND AT LEAST as clean as master alone.
 * When backend BasicPitch arrives it replaces the refined output
 * wholesale — no re-merge needed.
 */

// Default tolerances — all times in seconds.
const TIME_TOL = 0.05;        // 50ms: BasicPitch onset precision
const SPLIT_TIME_TOL = 0.10;  // 100ms: looser window for splitMasterByStems
const MIN_STEM_VEL = 0.05;    // below this, stem detection is "not present"

/**
 * Normalize a note's pitch field — our codebase uses `note` in some
 * places, `pitch` or `midi` in others. This helper lets refinement
 * consume notes from any shape.
 */
function pitchOf(n) {
  if (n == null) return null;
  if (Number.isFinite(n.note)) return n.note;
  if (Number.isFinite(n.pitch)) return n.pitch;
  if (Number.isFinite(n.midi)) return n.midi;
  return null;
}
function timeOf(n) {
  if (Number.isFinite(n.time)) return n.time;
  if (Number.isFinite(n.start)) return n.start;
  return null;
}

/**
 * Index master notes by pitch for O(log N) lookup during stem refinement.
 * Returns Map<pitch, sortedArray<{time, duration, velocity, ...orig}>>.
 */
function indexByPitch(notes) {
  const map = new Map();
  for (const n of notes || []) {
    const p = pitchOf(n);
    const t = timeOf(n);
    if (p === null || t === null) continue;
    if (!map.has(p)) map.set(p, []);
    map.get(p).push({
      time: t,
      duration: Number.isFinite(n.duration) ? n.duration : 0.25,
      velocity: Number.isFinite(n.velocity) ? n.velocity : 96,
      note: p,
    });
  }
  for (const arr of map.values()) arr.sort((a, b) => a.time - b.time);
  return map;
}

/** Binary search: index of first element with time >= target. */
function lowerBound(sortedArr, target) {
  let lo = 0, hi = sortedArr.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (sortedArr[mid].time < target) lo = mid + 1; else hi = mid;
  }
  return lo;
}

/** Find closest master note to (t, pitch) within `tol` seconds. */
function findNearest(masterByPitch, pitch, t, tol) {
  const arr = masterByPitch.get(pitch);
  if (!arr || arr.length === 0) return null;
  const i = lowerBound(arr, t - tol);
  let best = null, bestD = Infinity;
  for (let k = i; k < arr.length; k++) {
    const m = arr[k];
    if (m.time > t + tol) break;
    const d = Math.abs(m.time - t);
    if (d < bestD) { bestD = d; best = m; }
  }
  return best;
}

/**
 * Refine a lossy per-stem transcription using high-quality master notes.
 *
 * @param {Array} stemNotes - [{note|pitch|midi, time|start, duration, velocity}]
 * @param {Array} masterNotes - same shape, from BasicPitch on the master mix
 * @param {Object} [opts]
 * @param {number} [opts.timeTol=0.05] seconds — window for master↔stem match
 * @returns {Array} refined notes in {note, time, duration, velocity} shape
 */
export function refineStemWithMaster(stemNotes, masterNotes, opts = {}) {
  const tol = opts.timeTol ?? TIME_TOL;
  const masterIdx = indexByPitch(masterNotes);
  const out = [];
  for (const sn of stemNotes || []) {
    const p = pitchOf(sn);
    const t = timeOf(sn);
    if (p === null || t === null) continue;
    const m = findNearest(masterIdx, p, t, tol);
    if (!m) continue;     // drop: stem thinks there's a note here, master doesn't
    out.push({
      note: p,
      time: m.time,
      duration: m.duration,
      velocity: m.velocity,
    });
  }
  // Deduplicate overlapping same-pitch notes (can happen if two stem
  // onsets snap to the same master note).
  out.sort((a, b) => a.time - b.time || a.note - b.note);
  const dedup = [];
  for (const n of out) {
    const last = dedup[dedup.length - 1];
    if (last && last.note === n.note && Math.abs(last.time - n.time) < 1e-3) continue;
    dedup.push(n);
  }
  return dedup;
}

/**
 * Assign each master note to the stem whose transcription had the
 * strongest presence at that (pitch, time).
 *
 * Strategy: per master note, find a candidate in each stem's note list
 * at the same pitch within SPLIT_TIME_TOL. Score the candidate by
 * velocity / 127 (stem's confidence proxy). Pick the highest-scoring
 * stem. If no stem has any candidate, mark as 'unassigned' (usually
 * overtones, reverb, or a stem that needs backend BasicPitch to land).
 *
 * @returns {Object} { [stemName]: notes[], _unassigned: notes[] }
 */
export function splitMasterByStems(masterNotes, stemNotesByName, opts = {}) {
  const tol = opts.timeTol ?? SPLIT_TIME_TOL;
  const minVel = opts.minVel ?? MIN_STEM_VEL;
  const out = { _unassigned: [] };
  const stemIdx = {};
  for (const [name, notes] of Object.entries(stemNotesByName || {})) {
    out[name] = [];
    stemIdx[name] = indexByPitch(notes);
  }

  for (const mn of masterNotes || []) {
    const p = pitchOf(mn);
    const t = timeOf(mn);
    if (p === null || t === null) continue;
    let bestStem = null;
    let bestScore = 0;
    for (const stem of Object.keys(stemIdx)) {
      const cand = findNearest(stemIdx[stem], p, t, tol);
      if (!cand) continue;
      const score = (cand.velocity || 0) / 127;
      if (score > bestScore && score >= minVel) { bestScore = score; bestStem = stem; }
    }
    const assigned = {
      note: p,
      time: t,
      duration: Number.isFinite(mn.duration) ? mn.duration : 0.25,
      velocity: Number.isFinite(mn.velocity) ? mn.velocity : 96,
    };
    if (bestStem) out[bestStem].push(assigned);
    else out._unassigned.push(assigned);
  }
  return out;
}

/**
 * High-level combine: given master notes + a dict of stem names →
 * stem notes, produce a refined per-stem dict where each stem has:
 *   - All stem notes that match a master note (kept + master-timed)
 *   - PLUS master notes assigned to that stem via splitMasterByStems
 *     that weren't already in the refined set.
 *
 * This is the output you pass to dispatchers — upgrades both stem
 * quality (drop false positives) and recall (add missing master notes).
 */
export function combineForStems(masterNotes, stemNotesByName, opts = {}) {
  const refined = {};
  for (const [name, notes] of Object.entries(stemNotesByName || {})) {
    refined[name] = refineStemWithMaster(notes, masterNotes, opts);
  }
  const split = splitMasterByStems(masterNotes, stemNotesByName, opts);
  for (const [name, extras] of Object.entries(split)) {
    if (name === '_unassigned') continue;
    const have = new Set(refined[name].map((n) => `${n.note}@${n.time.toFixed(3)}`));
    for (const e of extras) {
      const key = `${e.note}@${e.time.toFixed(3)}`;
      if (!have.has(key)) {
        refined[name].push(e);
        have.add(key);
      }
    }
    refined[name].sort((a, b) => a.time - b.time || a.note - b.note);
  }
  return refined;
}
