/*
 * virtualTrackEdit — per-track playback schedules, rearrange-only.
 *
 * The source audio buffer is immutable. A Schedule describes how the
 * engine should read from it: each Segment maps a contiguous src window
 * to a contiguous dst (timeline) window. Meter changes become a schedule
 * rebuild — no backend, no decode, no latent touch.
 *
 * Design rules (STRICT):
 *   • rate === 1 on every segment. NO pitch shift, ever. Meter conversions
 *     are expressed as keep/drop/duplicate of whole eighth-note slices,
 *     never as playbackRate stretches on AudioBufferSourceNode.
 *   • No silence. Every dst second has real source audio — we never pad
 *     with zeros to fill a gap. Drops simply end a segment earlier and
 *     the next bar's audio starts immediately.
 *   • The `stretch` kind is reserved for situations that truly demand
 *     time-stretch (e.g. a future BPM ramp, not meter math). When a
 *     `stretch` appears, the engine MUST route it through a pitch-
 *     preserving WSOLA AudioWorklet — never through playbackRate.
 *
 * Types (JSDoc):
 *   Segment = {
 *     srcStart, srcEnd,     // seconds into the source buffer
 *     dstStart, dstEnd,     // seconds on the track timeline (clip-local)
 *     rate,                 // always 1 for rearrange rules (kept in schema
 *                           // for WSOLA-routed stretch segments only)
 *     fadeIn, fadeOut,      // seconds of linear gain ramp at seg edges
 *     kind,                 // 'identity' | 'keep' | 'duplicate' |
 *                           // 'preroll' | 'stretch'
 *   }
 *
 * INVARIANTS
 *   - Segments are sorted by dstStart and non-overlapping.
 *   - Adjacent segments are contiguous in dst (prev.dstEnd === next.dstStart)
 *     — no gaps, ever.
 *   - For rearrange segs: srcEnd - srcStart === dstEnd - dstStart, rate=1.
 *   - An identity schedule has exactly one Segment covering [0, duration].
 */

const EPS = 1e-6;

/**
 * Identity schedule: play the source straight through.
 * Every track starts with this; meter change only replaces it when needed.
 */
export function identitySchedule(duration, cropStart = 0) {
  const dur = Math.max(0, duration);
  return [{
    srcStart: cropStart,
    srcEnd: cropStart + dur,
    dstStart: 0,
    dstEnd: dur,
    rate: 1,
    fadeIn: 0,
    fadeOut: 0,
    kind: 'identity',
  }];
}

/** Total dst-timeline duration of a schedule. */
export function scheduleDuration(schedule) {
  if (!schedule || schedule.length === 0) return 0;
  let end = 0;
  for (const s of schedule) if (s.dstEnd > end) end = s.dstEnd;
  return end;
}

/**
 * Synthesize bar-start times from constant bpm + meter.
 *
 * Used when we don't have per-song downbeat detection cached on the
 * track. Tempo drift isn't handled — for drifting songs, swap in real
 * bar_starts (e.g. from beat_this) via the `barStarts` input.
 */
function synthBarStarts(duration, bpm, srcMeter, downbeatOffset = 0) {
  const [n, d] = srcMeter;
  const secPerBeat = 60 / bpm;
  const barSec = n * (4 / d) * secPerBeat;
  if (!(barSec > 0)) return [0, duration];
  const starts = [];
  let pos = Math.max(0, downbeatOffset);
  // Include the pre-roll as "bar 0" so the initial pickup isn't dropped.
  if (pos > EPS) starts.push(0);
  while (pos < duration - EPS) {
    starts.push(pos);
    pos += barSec;
  }
  starts.push(duration);
  return starts;
}

/**
 * Remap one source bar to rearrange-only segments in the target meter.
 *
 * Every segment plays at rate 1 — no pitch shift, ever. Meter changes are
 * expressed as keep/drop/duplicate of whole eighth-note slices, so every
 * bit of output audio is real source content at its original pitch.
 *
 * Returns { segs: Segment[], dstAdvance: number }.
 */
function remapBar({ srcBarStart, srcBarEnd, srcMeter, tgtMeter, dstCursor }) {
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcSig = `${sn}/${sd}`;
  const tgtSig = `${tn}/${td}`;
  const barLen = srcBarEnd - srcBarStart;

  // No change: identity passthrough.
  if (srcSig === tgtSig) {
    return {
      segs: [{
        srcStart: srcBarStart, srcEnd: srcBarEnd,
        dstStart: dstCursor, dstEnd: dstCursor + barLen,
        rate: 1, fadeIn: 0, fadeOut: 0, kind: 'identity',
      }],
      dstAdvance: barLen,
    };
  }

  // Convenience: src bar length expressed in its own eighth-notes.
  const srcEighthsPerBar = sn * (sd === 4 ? 2 : 1);
  const tgtEighthsPerBar = tn * (td === 4 ? 2 : 1);
  const eighth = barLen / srcEighthsPerBar;

  // "keep" = play a contiguous src slice at rate 1.
  // "duplicate" = same as keep, but conceptually re-reads a slice already
  // used (so the engine can tag a stronger fade-in). Both are rate 1.
  const slice = (kind, fromSrcOff, lenSrc) => ({
    srcStart: srcBarStart + fromSrcOff,
    srcEnd:   srcBarStart + fromSrcOff + lenSrc,
    dstStart: dstCursor,   // caller patches before returning
    dstEnd:   dstCursor,
    rate: 1,
    fadeIn: 0, fadeOut: 0,
    kind,
  });
  const keep = (off, len) => slice('keep', off, len);
  const dup  = (off, len) => slice('duplicate', off, len);

  let segs;

  // 4/4 -> 7/8: drop the last eighth ("and" of 4). 7 src eighths kept, 1
  // dropped. Preserves the downbeat of every bar; the "and"-of-4 goes away
  // instead of being compressed. Reads as a truncated 4/4 bar, no pitch shift.
  if (srcSig === '4/4' && tgtSig === '7/8') {
    segs = [keep(0, eighth * 7)];
  }
  // 7/8 -> 4/4: 7 src eighths play straight, then the last eighth plays
  // once more to fill beat 4-and.
  else if (srcSig === '7/8' && tgtSig === '4/4') {
    segs = [keep(0, eighth * 7), dup(eighth * 6, eighth * 1)];
  }
  // 4/4 -> 6/8: drop the last 2 eighths (beat 4).
  else if (srcSig === '4/4' && tgtSig === '6/8') {
    segs = [keep(0, eighth * 6)];
  }
  // 6/8 -> 4/4: duplicate the last 2 eighths to fill beat 4.
  else if (srcSig === '6/8' && tgtSig === '4/4') {
    segs = [keep(0, eighth * 6), dup(eighth * 4, eighth * 2)];
  }
  // 4/4 -> 3/4: drop beat 4 (last 2 eighths).
  else if (srcSig === '4/4' && tgtSig === '3/4') {
    segs = [keep(0, eighth * 6)];
  }
  // 3/4 -> 4/4: play whole bar (6 eighths), then duplicate beat 3 as the
  // new beat 4.
  else if (srcSig === '3/4' && tgtSig === '4/4') {
    segs = [keep(0, barLen), dup(eighth * 4, eighth * 2)];
  }
  // 4/4 -> 5/4: play whole bar, then duplicate beat 4 as the new beat 5.
  else if (srcSig === '4/4' && tgtSig === '5/4') {
    segs = [keep(0, barLen), dup(eighth * 6, eighth * 2)];
  }
  // 5/4 -> 4/4: drop beat 5.
  else if (srcSig === '5/4' && tgtSig === '4/4') {
    segs = [keep(0, eighth * 8)];
  }
  // 3/4 -> 7/8: 6 src eighths, need 7. Duplicate the last eighth.
  else if (srcSig === '3/4' && tgtSig === '7/8') {
    segs = [keep(0, eighth * 6), dup(eighth * 5, eighth * 1)];
  }
  // 7/8 -> 3/4: 7 src eighths, need 6. Drop the last eighth.
  else if (srcSig === '7/8' && tgtSig === '3/4') {
    segs = [keep(0, eighth * 6)];
  }
  // Generic rearrange fallback.
  //   - tgt has FEWER eighths than src → drop the tail diff.
  //   - tgt has MORE eighths than src → play whole bar, then duplicate the
  //     last (diff) eighths to fill.
  // No stretch segment is ever emitted — every output eighth is real source
  // audio at rate 1.
  else {
    const diff = tgtEighthsPerBar - srcEighthsPerBar;
    if (diff === 0) {
      segs = [keep(0, barLen)];
    } else if (diff < 0) {
      segs = [keep(0, eighth * tgtEighthsPerBar)];
    } else {
      const fromOff = Math.max(0, eighth * (srcEighthsPerBar - diff));
      segs = [keep(0, barLen), dup(fromOff, eighth * diff)];
    }
  }

  // Patch dst coords + short fades at every cut to hide clicks and the
  // transient discontinuity at a duplicate's attack. 3ms is inaudible on
  // its own but kills splice clicks. Fades are gain ramps applied by the
  // engine — they do NOT introduce silence, just a brief slope.
  const FADE = 0.003;
  let cursor = dstCursor;
  for (let i = 0; i < segs.length; i++) {
    const s = segs[i];
    if (s.rate !== 1) {
      console.warn('[virtualTrackEdit] rearrange rule emitted rate !== 1 — bug:', s);
    }
    const dstLen = s.srcEnd - s.srcStart; // rate is 1
    s.dstStart = cursor;
    s.dstEnd = cursor + dstLen;
    s.fadeIn  = (i === 0) ? FADE : FADE;
    s.fadeOut = (i === segs.length - 1) ? FADE : FADE;
    cursor = s.dstEnd;
  }

  return { segs, dstAdvance: cursor - dstCursor };
}

/**
 * Build a meter-change schedule for a single clip.
 *
 * @param {Object}   opts
 * @param {number}   opts.duration       source duration (seconds)
 * @param {number[]} [opts.barStarts]    optional real downbeats (seconds);
 *                                       synthesized from bpm+meter if omitted
 * @param {number[]} opts.srcMeter       [n, d] e.g. [4, 4]
 * @param {number[]} opts.tgtMeter       [n, d] e.g. [7, 8]
 * @param {number}   opts.bpm            source bpm (for bar synth)
 * @param {number}   [opts.downbeatOffset] pre-roll before the first downbeat
 * @param {number}   [opts.cropStart]    offset into the buffer for the clip
 * @returns {Segment[]}
 */
export function buildMeterSchedule({
  duration, barStarts, srcMeter, tgtMeter, bpm,
  downbeatOffset = 0, cropStart = 0,
}) {
  if (!(duration > 0)) return [];
  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }
  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  const out = [];
  let dstCursor = 0;

  // Pre-roll (audio before bar 1) — keep it straight, don't meter-edit.
  if (starts[0] > EPS) {
    const preLen = starts[0];
    out.push({
      srcStart: cropStart,
      srcEnd:   cropStart + preLen,
      dstStart: 0,
      dstEnd:   preLen,
      rate: 1,
      fadeIn: 0, fadeOut: 0,
      kind: 'preroll',
    });
    dstCursor = preLen;
  }

  // Remap each complete source bar.
  for (let b = 0; b < starts.length - 1; b++) {
    const sbs = starts[b];
    const sbe = starts[b + 1];
    if (sbe - sbs < EPS) continue;
    const { segs, dstAdvance } = remapBar({
      srcBarStart: cropStart + sbs,
      srcBarEnd:   cropStart + sbe,
      srcMeter, tgtMeter,
      dstCursor,
    });
    out.push(...segs);
    dstCursor += dstAdvance;
  }
  return out;
}

/**
 * Produce the schedule for a given track under the current project meter.
 *
 * Precedence:
 *   1. track.editSchedule — user-authored override (future manual trims)
 *   2. meter edit — if track has a detected source meter and the project
 *      meter differs, chop accordingly
 *   3. identity — straight-through playback
 *
 * "Source BPM" is track.metadata.detectedBpm (set during ImportAudioModal
 * detection). When absent we fall back to the project BPM — equivalent to
 * "track was recorded in project tempo", which is a safe identity.
 *
 * DRUM SUBSTEMS (NOT YET WIRED): a drum track is currently handled the
 * same as any other stem — one bar-level rearrange across the full mix.
 * The robust path (split into kick/snare/toms/hh/ride/crash, per-substem
 * beat-snap) lives in services/latentMeterChange.js:meterChangeDrumStem
 * but is reached only via the (gated-off) backend repaint. To make
 * virtual-edit substem-aware without the latent hop, a track would hold
 * an optional `track.substems = [{ audioUrl, kind }]` array; the engine
 * would build one schedule per substem and mix through the per-track
 * gain. Open follow-up.
 */
export function getTrackSchedule(track, project) {
  if (!track) return [];
  const duration = track.duration || track.length || 0;
  const cropStart = track.cropStart || 0;
  if (duration <= 0) return identitySchedule(0);

  if (Array.isArray(track.editSchedule) && track.editSchedule.length > 0) {
    return track.editSchedule;
  }

  const meta = track.metadata || {};
  const srcBpm = meta.detectedBpm || project.bpm || 120;
  const srcMeter = [meta.detectedMeter || project.beatsPerBar || 4, 4];
  const tgtMeter = [project.beatsPerBar || 4, project.meterDenominator || 4];
  const downbeatOffset = typeof meta.downbeatOffset === 'number' ? meta.downbeatOffset : 0;

  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }

  return buildMeterSchedule({
    duration, srcMeter, tgtMeter, bpm: srcBpm, downbeatOffset, cropStart,
    barStarts: meta.barStarts,
  });
}

/* ------------------------------------------------------------------
 * Drum substem support — robust, musical, rate-1 only.
 *
 * When a drum stem track has metadata.drumSubstems (the kick/snare/toms/
 * hh/ride/crash WAV URLs from MDX23C-DrumSep on the backend) plus
 * metadata.drumSubstemOnsets (librosa onset times per substem), we
 * rearrange each substem independently with substem-aware musical
 * placement on the new meter grid.
 *
 * Mirrors time-sig-editor/server.py:_process_stem_pattern_aware +
 * _requantize_stem (the "robust" path), but emits rate-1 keep/duplicate
 * slices instead of time-stretches. Asymmetric grouping is preserved:
 *
 *   4/4 → 7/8 grouped 4+3 ("the 3" felt as 1.5+1.5 = two dotted eighths)
 *     • first half (src eighths 0-3) → tgt eighths 0-3, identity
 *     • second half (src eighths 4-7) → tgt eighths 4-7 (3 eighths total):
 *         - KICK / TOMS  : proportional remap into the 3-eighth window
 *                          (groove preserved, no hit dropped)
 *         - SNARE         : snap to tgt eighth 5.5 — start of the second
 *                          1.5-group, the standard 7/8 snare placement
 *         - HH (sustain) : bar-rearrange (drop last eighth)
 *
 *   7/8 → 4/4: first half identity, second half (3 eighths) maps to
 *              4 eighths via proportional remap; snare lands on tgt
 *              eighth 6 (beat 4).
 *
 *   4/4 ↔ 3/4, 4/4 ↔ 5/4, 4/4 ↔ 6/8, 3/4 ↔ 7/8, 5/4 ↔ 4/4, etc.: same
 *   musical-aware mapping, rate-1.
 *
 * Per-substem PATTERN DETECTION (triplet vs eighth vs 16th) is computed
 * from onset density (hits per src beat). Triplet hits in a non-triplet-
 * compatible target meter (7/8, 5/4, 3/4) are flagged for later
 * re-quantization to 16ths/8ths — for now they take the proportional path.
 *
 * All substems mix into one shared per-track gain so solo/mute on the
 * parent drum stem track keeps working.
 * ------------------------------------------------------------------ */

const PERCUSSIVE_SUBSTEMS = new Set(['kick', 'snare', 'toms']);
const SUSTAIN_SUBSTEMS    = new Set(['hh', 'ride', 'crash']);

/**
 * Substem-aware musical snap: given a src eighth position (fractional)
 * and the meter conversion + substem name, return the list of tgt eighth
 * positions where this hit should land. Empty list = drop the hit.
 *
 * Returns FRACTIONAL eighth positions (e.g. 5.5 means the snare lands at
 * the boundary of the two dotted-eighths in 7/8's "3" group). The caller
 * multiplies by tgt-eighth length to get dst seconds.
 *
 * Mirrors the musical intent of time-sig-editor's _requantize_stem (4+3
 * grouping etc.) but emits rate-1 placements instead of time-stretches.
 */
function snapEighthInBar(srcEighthFloat, srcMeter, tgtMeter, substemName) {
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const sSig = `${sn}/${sd}`;
  const tSig = `${tn}/${td}`;
  if (sSig === tSig) return [srcEighthFloat];

  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);

  // 4/4 → 7/8: 4+3 grouping, with substem-aware placement in the 3-group.
  if (sSig === '4/4' && tSig === '7/8') {
    if (srcEighthFloat < 4) return [srcEighthFloat];           // first half identity
    if (substemName === 'snare') return [5.5];                 // start of 2nd 1.5-group
    return [4 + (srcEighthFloat - 4) * 3 / 4];                 // proportional in 3-group
  }
  // 7/8 → 4/4: first half identity, 3-eighth tail → 4-eighth tail.
  if (sSig === '7/8' && tSig === '4/4') {
    if (srcEighthFloat < 4) return [srcEighthFloat];
    if (substemName === 'snare') return [6];                   // beat 4 snare
    return [4 + (srcEighthFloat - 4) * 4 / 3];
  }
  // 4/4 → 6/8: drop last 2 eighths (beat 4); 6/8 felt as 3+3.
  if (sSig === '4/4' && tSig === '6/8') {
    if (srcEighthFloat < 6) return [srcEighthFloat];
    return [];
  }
  // 6/8 → 4/4: keep first 6 eighths identity, append a duplicate of
  // beat-3 (eighths 4-5) onto eighths 6-7 to give the new beat 4.
  if (sSig === '6/8' && tSig === '4/4') {
    if (srcEighthFloat < 6) return [srcEighthFloat];
    return [];                                                 // shouldn't happen — src has only 6
  }
  // 4/4 → 3/4: drop beat 4 (last 2 eighths).
  if (sSig === '4/4' && tSig === '3/4') {
    if (srcEighthFloat < 6) return [srcEighthFloat];
    return [];
  }
  // 3/4 → 4/4: identity for 6 eighths; the duplicate covering eighths 6-7
  // is emitted by the per-bar duplicate pass below (snapHitInBar can
  // return identity here; the duplicate insertion happens in the bar
  // loop, which adds a second hit at e_t = e_s + 2 for src e_s in [4,6)).
  if (sSig === '3/4' && tSig === '4/4') {
    return [srcEighthFloat];                                   // identity in src window
  }
  // 4/4 → 5/4: identity for 8 eighths; the duplicate covering eighths 8-9
  // is emitted by the per-bar duplicate pass.
  if (sSig === '4/4' && tSig === '5/4') {
    return [srcEighthFloat];
  }
  // 5/4 → 4/4: drop beat 5.
  if (sSig === '5/4' && tSig === '4/4') {
    if (srcEighthFloat < 8) return [srcEighthFloat];
    return [];
  }
  // 3/4 → 7/8: 3+3+1 — keep first 4 eighths, then last 2 src eighths
  // map proportionally into 3 tgt eighths (snare on the new beat).
  if (sSig === '3/4' && tSig === '7/8') {
    if (srcEighthFloat < 4) return [srcEighthFloat];
    if (substemName === 'snare') return [5.5];
    return [4 + (srcEighthFloat - 4) * 3 / 2];
  }
  // 7/8 → 3/4: drop last eighth.
  if (sSig === '7/8' && tSig === '3/4') {
    if (srcEighthFloat < 6) return [srcEighthFloat];
    return [];
  }

  // Generic proportional fallback: scale to tgt. Hits past tgt are dropped.
  const tgt = srcEighthFloat * tgtEighths / srcEighths;
  if (tgt >= tgtEighths - EPS) return [];
  return [tgt];
}

/**
 * Per-bar "duplicate insertion" pass: for tgt eighths beyond src eighths
 * (when extending the meter), figure out which src eighths to duplicate.
 *
 * For 3/4 → 4/4: duplicate the LAST BEAT (src eighths 4-5) at tgt 6-7.
 *   So a hit at src e=4 fires twice in dst: at tgt 4 (identity) and tgt 6 (dup).
 *   A hit at src e=5 fires at tgt 5 and tgt 7.
 * For 4/4 → 5/4: duplicate beat 4 (src eighths 6-7) at tgt 8-9.
 *
 * Returns extra tgt eighth positions a src hit should also fire at.
 */
function duplicatedTgtsForSrcEighth(srcEighthFloat, srcMeter, tgtMeter) {
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const sSig = `${sn}/${sd}`;
  const tSig = `${tn}/${td}`;
  if (sSig === '3/4' && tSig === '4/4') {
    if (srcEighthFloat >= 4 && srcEighthFloat < 6) return [srcEighthFloat + 2];
    return [];
  }
  if (sSig === '4/4' && tSig === '5/4') {
    if (srcEighthFloat >= 6 && srcEighthFloat < 8) return [srcEighthFloat + 2];
    return [];
  }
  return [];
}

/**
 * Detect approximate hits-per-beat from an onset list within a window
 * (mirrors time-sig-editor's _detect_drum_pattern but onset-based, since
 * we already have librosa onsets cached). Used to flag triplets that
 * won't fit the target meter (they'd want re-quantization to 16ths —
 * follow-up work; for now we just log).
 */
function detectHitsPerBeat(onsets, durSec, bpm) {
  if (!onsets || onsets.length < 4 || durSec <= 0 || bpm <= 0) return 0;
  const beats = (durSec / 60) * bpm;
  if (beats <= 0) return 0;
  const hpb = onsets.length / beats;
  const candidates = [1, 2, 3, 4];
  let best = 1, bestErr = Infinity;
  for (const c of candidates) {
    const err = Math.abs(hpb - c) / c;
    if (err < bestErr) { bestErr = err; best = c; }
  }
  return bestErr < 0.25 ? best : 0;     // require reasonable confidence
}

/**
 * Pattern fits target meter? Triplets in 7/8, 5/4, 3/4 don't groove.
 */
function patternFitsTargetMeter(hpb, tgtMeter) {
  if (hpb !== 3) return true;
  const [tn, td] = tgtMeter;
  if (td === 8) return false;
  if (tn === 5 || tn === 7) return false;
  return true;
}

/**
 * Per-substem schedule builder: musical-aware hit snap with rate-1
 * keep + duplicate slices.
 *
 * Per src bar:
 *   1. Find onsets in [bar_start, bar_end].
 *   2. For each onset, compute fractional src eighth.
 *   3. snapEighthInBar(srcE, src, tgt, substemName) → list of tgt eighth
 *      positions to place this hit at.
 *   4. duplicatedTgtsForSrcEighth(srcE, src, tgt) → extra tgt positions
 *      for the duplicate-tail rules (3/4→4/4, 4/4→5/4).
 *   5. Each placement emits a rate-1 segment with HIT_PRE/HIT_MAX window
 *      and short fades. Sub-eighth offset is preserved (groove).
 *
 * Segments may overlap on dst (a duplicated hit, or a long tail running
 * into the next hit). The engine schedules each as its own
 * AudioBufferSourceNode → segGain → trackGain so they sum naturally.
 */
function buildPercussiveSubstemSchedule({
  duration, onsets, srcMeter, tgtMeter, bpm, substemName,
  downbeatOffset = 0, cropStart = 0, barStarts,
}) {
  if (!(duration > 0)) return [];
  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }
  const HIT_PRE  = 0.005;
  const HIT_MAX  = 0.5;
  const FADE_IN  = 0.002;
  const FADE_OUT = 0.020;

  // Triplet flag — if src has triplets and tgt doesn't fit them, log.
  // Proper re-quantize-to-16ths is a follow-up (needs sample synthesis).
  const hpb = detectHitsPerBeat(onsets, duration, bpm);
  if (hpb === 3 && !patternFitsTargetMeter(hpb, tgtMeter)) {
    console.warn(`[virtualTrackEdit] ${substemName}: triplet pattern detected, target meter ${tgtMeter[0]}/${tgtMeter[1]} doesn't fit triplets — using proportional snap (re-quantize-to-16ths is a follow-up)`);
  }

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);

  const segs = [];
  let dstCursor = 0;

  // Pre-roll: copy through (drum substem during pickup is usually below
  // noise floor; if there's a pickup hit we'd rather hear it than chop it).
  if (starts[0] > EPS) {
    const preLen = starts[0];
    segs.push({
      srcStart: cropStart, srcEnd: cropStart + preLen,
      dstStart: 0, dstEnd: preLen,
      rate: 1, fadeIn: 0, fadeOut: 0, kind: 'preroll',
    });
    dstCursor = preLen;
  }

  for (let b = 0; b < starts.length - 1; b++) {
    const sbs = starts[b];
    const sbe = starts[b + 1];
    const srcBarLen = sbe - sbs;
    if (srcBarLen <= EPS) continue;
    const srcEighthLen = srcBarLen / srcEighths;
    const tgtEighthLen = srcEighthLen;                // rate 1
    const tgtBarLen = tgtEighthLen * tgtEighths;

    const inBar = (Array.isArray(onsets) ? onsets : [])
      .filter((o) => o >= sbs && o < sbe)
      .sort((a, b) => a - b);

    for (let i = 0; i < inBar.length; i++) {
      const onset = inBar[i];
      const srcEighthFloat = (onset - sbs) / srcEighthLen;
      const primary = snapEighthInBar(srcEighthFloat, srcMeter, tgtMeter, substemName);
      const dups = duplicatedTgtsForSrcEighth(srcEighthFloat, srcMeter, tgtMeter);
      const tgts = [...primary, ...dups];
      if (tgts.length === 0) continue;

      const next = (i + 1 < inBar.length) ? inBar[i + 1] : sbe;
      const winSrc = Math.min(next - onset + HIT_PRE, HIT_MAX);
      const srcStart = cropStart + Math.max(sbs, onset - HIT_PRE);
      const srcEnd = srcStart + winSrc;

      for (const tgtEighthFloat of tgts) {
        const dstHitStart = Math.max(dstCursor, dstCursor + tgtEighthFloat * tgtEighthLen - HIT_PRE);
        segs.push({
          srcStart, srcEnd,
          dstStart: dstHitStart,
          dstEnd: dstHitStart + winSrc,
          rate: 1, fadeIn: FADE_IN, fadeOut: FADE_OUT, kind: 'hit',
        });
      }
    }

    dstCursor += tgtBarLen;
  }

  segs.sort((a, b) => a.dstStart - b.dstStart);
  return segs;
}

/**
 * Build per-substem schedules for a drum stem track.
 *
 * Returns null if the track doesn't have substem metadata; otherwise
 * returns { [substemName]: { audioUrl, schedule, kind } }. The engine
 * schedules each substem through one shared per-track gain so the parent
 * track's gain/solo/mute still controls the whole drum mix.
 *
 * Substem strategy:
 *   - PERCUSSIVE (kick/snare/toms) with onsets: musical-aware hit snap
 *     (4+3 grouping for 7/8, snare on second 1.5-group, kick proportional,
 *     duplicate for extending meters). See buildPercussiveSubstemSchedule.
 *   - SUSTAIN (hh/ride/crash): bar-rearrange (so wash/decay carries
 *     through naturally; snapping a crash's attack drops the sustain).
 *   - PERCUSSIVE without onsets: fall through to bar-rearrange so it
 *     still plays — slightly less accurate but never silent.
 */
export function getTrackSubstemSchedules(track, project) {
  const meta = track?.metadata || {};
  const subUrls = meta.drumSubstems;
  if (!subUrls || typeof subUrls !== 'object') return null;
  const subOnsets = meta.drumSubstemOnsets || {};
  const duration = track.duration || track.length || 0;
  if (duration <= 0) return null;
  const cropStart = track.cropStart || 0;

  const srcBpm = meta.detectedBpm || project.bpm || 120;
  const srcMeter = [meta.detectedMeter || project.beatsPerBar || 4, 4];
  const tgtMeter = [project.beatsPerBar || 4, project.meterDenominator || 4];
  const downbeatOffset = typeof meta.downbeatOffset === 'number' ? meta.downbeatOffset : 0;
  const identity = srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1];

  const out = {};
  for (const [name, audioUrl] of Object.entries(subUrls)) {
    if (!audioUrl) continue;
    let schedule;
    let kind;
    if (identity) {
      schedule = identitySchedule(duration, cropStart);
      kind = 'identity';
    } else if (PERCUSSIVE_SUBSTEMS.has(name) && Array.isArray(subOnsets[name]) && subOnsets[name].length > 0) {
      schedule = buildPercussiveSubstemSchedule({
        duration, onsets: subOnsets[name],
        srcMeter, tgtMeter, bpm: srcBpm, substemName: name,
        downbeatOffset, cropStart, barStarts: meta.barStarts,
      });
      kind = 'snap';
    } else {
      // Sustain substem (hh/ride/crash) OR percussive substem with no
      // onsets cached — fall through to the bar-rearrange path.
      schedule = buildMeterSchedule({
        duration, srcMeter, tgtMeter, bpm: srcBpm,
        downbeatOffset, cropStart, barStarts: meta.barStarts,
      });
      kind = 'rearrange';
    }
    out[name] = { audioUrl, schedule, kind };
  }
  return Object.keys(out).length > 0 ? out : null;
}

/**
 * Live-resume helper — given a schedule and the current dst-timeline
 * playhead (clip-local), compute, for each segment that still needs to
 * play, the wall-clock offset and src offset to start from.
 *
 * Returns entries in schedule order: { seg, dstOffsetIntoSeg, srcOffsetIntoSeg, srcDuration }.
 * A `dstOffsetIntoSeg` of 0 means "seg hasn't started yet" (schedule it
 * in the future); a positive value means "seg is already in flight, cut
 * in mid-way and play the remainder."
 */
export function resumeFromPlayhead(schedule, t) {
  const out = [];
  for (const seg of schedule) {
    if (seg.dstEnd <= t + EPS) continue;              // already finished
    const dstOffsetIntoSeg = Math.max(0, t - seg.dstStart);
    // rate is 1 on every rearrange seg, but keep the formula general so the
    // same function works once WSOLA-routed stretch segments appear.
    const srcOffsetIntoSeg = dstOffsetIntoSeg * seg.rate;
    const srcDuration = (seg.srcEnd - seg.srcStart) - srcOffsetIntoSeg;
    if (srcDuration <= EPS) continue;
    out.push({ seg, dstOffsetIntoSeg, srcOffsetIntoSeg, srcDuration });
  }
  return out;
}
