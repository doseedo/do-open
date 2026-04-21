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

/* ------------------------------------------------------------------
 * METER_RULES — single source of truth for every meter conversion.
 *
 * Each rule is a list of `sections` walked in order, expressed in
 * src-eighth and tgt-eighth coordinates. Both interpreters consume it:
 *
 *   • barRearrangeFromRule (no-onset path) emits keep/drop/duplicate
 *     segments per section to build a rate-1 bar-rearrange schedule.
 *   • placementsForSrcEighth (per-substem snap path) maps a single
 *     onset's src-eighth position to one or more tgt-eighth positions
 *     based on the section it lands in.
 *
 * Section types:
 *   { keep: [srcEighthA, srcEighthB] }
 *     Play that src window straight (identity). Per-onset: identity.
 *   { drop: [srcEighthA, srcEighthB] }
 *     Discard that src range. Per-onset: dropped.
 *   { warp: { srcRange, tgtRange }, snareSnap?: tgtEighthFloat }
 *     Asymmetric remap (the 4+3 grouping case). Bar-rearrange converts
 *     to "keep first min(src,tgt) + drop OR duplicate the difference".
 *     Per-onset: snare snaps to snareSnap if given (musical placement),
 *     other substems use proportional remap into the tgt window.
 *   { duplicate: { srcRange, tgtRange }, snareSnap?: tgtEighthFloat }
 *     Explicit duplicate (covers tgt eighths beyond src). Per-onset:
 *     onsets in srcRange ALSO fire at the proportional tgt position
 *     (or snareSnap for snare). Combined with a `keep` section over
 *     the same src eighths, this produces "onset fires at identity AND
 *     at the duplicated position" — e.g. snare on beat 3 of 3/4 also
 *     fires on the new beat 4 of 4/4.
 *
 * Snare snap targets reflect standard backbeat placement in the new
 * meter (e.g. tgt eighth 5.5 in 7/8 = start of the second 1.5-group of
 * a 4+3 grouping; tgt eighth 7 in 4/4-extended = the new beat 4 down).
 * ------------------------------------------------------------------ */
const METER_RULES = {
  '4/4->7/8': { srcEighths: 8, tgtEighths: 7, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 8], tgtRange: [4, 7] }, snareSnap: 5.5 },
  ]},
  '7/8->4/4': { srcEighths: 7, tgtEighths: 8, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 7], tgtRange: [4, 8] }, snareSnap: 6 },
  ]},
  '4/4->6/8': { srcEighths: 8, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 8] },
  ]},
  '6/8->4/4': { srcEighths: 6, tgtEighths: 8, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [4, 6], tgtRange: [6, 8] }, snareSnap: 7 },
  ]},
  '4/4->3/4': { srcEighths: 8, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 8] },
  ]},
  '3/4->4/4': { srcEighths: 6, tgtEighths: 8, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [4, 6], tgtRange: [6, 8] }, snareSnap: 7 },
  ]},
  '4/4->5/4': { srcEighths: 8, tgtEighths: 10, sections: [
    { keep: [0, 8] },
    { duplicate: { srcRange: [6, 8], tgtRange: [8, 10] }, snareSnap: 9 },
  ]},
  '5/4->4/4': { srcEighths: 10, tgtEighths: 8, sections: [
    { keep: [0, 8] }, { drop: [8, 10] },
  ]},
  '3/4->7/8': { srcEighths: 6, tgtEighths: 7, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 6], tgtRange: [4, 7] }, snareSnap: 5.5 },
  ]},
  '7/8->3/4': { srcEighths: 7, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 7] },
  ]},
  '5/4->7/8': { srcEighths: 10, tgtEighths: 7, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 10], tgtRange: [4, 7] }, snareSnap: 5.5 },
  ]},
  '7/8->5/4': { srcEighths: 7, tgtEighths: 10, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 7], tgtRange: [4, 10] }, snareSnap: 7 },
  ]},
  '5/4->3/4': { srcEighths: 10, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 10] },
  ]},
  '3/4->5/4': { srcEighths: 6, tgtEighths: 10, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [2, 6], tgtRange: [6, 10] }, snareSnap: 7 },
  ]},
  '5/4->6/8': { srcEighths: 10, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 10] },
  ]},
  '6/8->5/4': { srcEighths: 6, tgtEighths: 10, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [2, 6], tgtRange: [6, 10] }, snareSnap: 7 },
  ]},
  '6/8->3/4': { srcEighths: 6, tgtEighths: 6, sections: [{ keep: [0, 6] }] },
  '3/4->6/8': { srcEighths: 6, tgtEighths: 6, sections: [{ keep: [0, 6] }] },
  '6/8->7/8': { srcEighths: 6, tgtEighths: 7, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [5, 6], tgtRange: [6, 7] } },
  ]},
  '7/8->6/8': { srcEighths: 7, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 7] },
  ]},
  '5/4->5/4': { srcEighths: 10, tgtEighths: 10, sections: [{ keep: [0, 10] }] },
};

function ruleFor(srcMeter, tgtMeter) {
  return METER_RULES[`${srcMeter[0]}/${srcMeter[1]}->${tgtMeter[0]}/${tgtMeter[1]}`] || null;
}

/**
 * Per-onset placement: src-eighth (fractional) → list of tgt-eighth
 * positions for this substem. Walks the rule sections in order and
 * collects every section that the onset lands in.
 *
 * For a substem's onset to fire in dst, AT LEAST ONE section must
 * accept it. Empty list = dropped (intentional silence at that hit).
 */
function placementsForSrcEighth(srcEighthFloat, rule, substemName) {
  const out = [];
  for (const sec of rule.sections) {
    if (sec.keep) {
      const [a, b] = sec.keep;
      if (srcEighthFloat >= a - EPS && srcEighthFloat < b - EPS) {
        out.push(srcEighthFloat);
      }
    } else if (sec.drop) {
      // dropped — no entry
    } else if (sec.warp) {
      const [sa, sb] = sec.warp.srcRange;
      const [ta, tb] = sec.warp.tgtRange;
      if (srcEighthFloat >= sa - EPS && srcEighthFloat < sb - EPS) {
        if (substemName === 'snare' && typeof sec.snareSnap === 'number') {
          out.push(sec.snareSnap);
        } else {
          const tgt = ta + ((srcEighthFloat - sa) / (sb - sa)) * (tb - ta);
          out.push(tgt);
        }
      }
    } else if (sec.duplicate) {
      const [sa, sb] = sec.duplicate.srcRange;
      const [ta, tb] = sec.duplicate.tgtRange;
      if (srcEighthFloat >= sa - EPS && srcEighthFloat < sb - EPS) {
        if (substemName === 'snare' && typeof sec.snareSnap === 'number') {
          out.push(sec.snareSnap);
        } else {
          const tgt = ta + ((srcEighthFloat - sa) / (sb - sa)) * (tb - ta);
          out.push(tgt);
        }
      }
    }
  }
  return out;
}

/**
 * Bar-rearrange interpreter: walks rule sections in order, emits one
 * rate-1 segment per non-drop section.
 *
 * warp section: converted to keep+drop (when src window > tgt window)
 *   or keep+duplicate (when src window < tgt window) so the resulting
 *   schedule is rate-1 throughout. The substem snap path uses warp
 *   directly via proportional placement; this path collapses warp into
 *   simpler rearrange ops because there are no per-onset positions to
 *   place — we just have to fill the dst window with src content.
 */
function barRearrangeFromRule(rule, srcBarStart, srcEighthLen, dstCursor) {
  const segs = [];
  let cursor = dstCursor;

  const emit = (kind, srcOffEighths, lenEighths) => {
    const len = lenEighths * srcEighthLen;
    segs.push({
      srcStart: srcBarStart + srcOffEighths * srcEighthLen,
      srcEnd:   srcBarStart + srcOffEighths * srcEighthLen + len,
      dstStart: cursor,
      dstEnd:   cursor + len,
      rate: 1, fadeIn: 0, fadeOut: 0, kind,
    });
    cursor += len;
  };

  for (const sec of rule.sections) {
    if (sec.keep) {
      emit('keep', sec.keep[0], sec.keep[1] - sec.keep[0]);
    } else if (sec.drop) {
      // No segment, no cursor advance.
    } else if (sec.warp) {
      const [sa, sb] = sec.warp.srcRange;
      const [ta, tb] = sec.warp.tgtRange;
      const srcWin = sb - sa;
      const tgtWin = tb - ta;
      if (tgtWin <= srcWin) {
        // Drop tail: keep the first `tgtWin` src eighths in this window.
        emit('keep', sa, tgtWin);
      } else {
        // Keep all + duplicate the last (tgtWin - srcWin) src eighths.
        emit('keep', sa, srcWin);
        const dupEighths = tgtWin - srcWin;
        emit('duplicate', sb - dupEighths, dupEighths);
      }
    } else if (sec.duplicate) {
      // Explicit duplicate ALWAYS appears alongside a sibling keep
      // section that covers the source identity. Here we only emit
      // the duplicate placement at the dup tgt range.
      const [sa, sb] = sec.duplicate.srcRange;
      emit('duplicate', sa, sb - sa);
    }
  }

  // 3ms fades on every segment edge.
  const FADE = 0.003;
  for (const s of segs) { s.fadeIn = FADE; s.fadeOut = FADE; }
  return { segs, dstAdvance: cursor - dstCursor };
}

/**
 * Remap one source bar to rearrange-only segments in the target meter.
 *
 * Looks up METER_RULES first; falls back to generic proportional drop/
 * duplicate when an explicit rule isn't defined.
 *
 * Every segment plays at rate 1 — no pitch shift, ever.
 */
function remapBar({ srcBarStart, srcBarEnd, srcMeter, tgtMeter, dstCursor }) {
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const sSig = `${sn}/${sd}`;
  const tSig = `${tn}/${td}`;
  const barLen = srcBarEnd - srcBarStart;

  // No change: identity passthrough.
  if (sSig === tSig) {
    return {
      segs: [{
        srcStart: srcBarStart, srcEnd: srcBarEnd,
        dstStart: dstCursor, dstEnd: dstCursor + barLen,
        rate: 1, fadeIn: 0, fadeOut: 0, kind: 'identity',
      }],
      dstAdvance: barLen,
    };
  }

  const rule = ruleFor(srcMeter, tgtMeter);
  if (rule) {
    const srcEighthLen = barLen / rule.srcEighths;
    return barRearrangeFromRule(rule, srcBarStart, srcEighthLen, dstCursor);
  }

  // Generic fallback for combos not in the rules table.
  const srcEighthsPerBar = sn * (sd === 4 ? 2 : 1);
  const tgtEighthsPerBar = tn * (td === 4 ? 2 : 1);
  const eighth = barLen / srcEighthsPerBar;
  const FADE = 0.003;
  const segs = [];
  let cursor = dstCursor;
  const push = (off, len, kind) => {
    segs.push({
      srcStart: srcBarStart + off, srcEnd: srcBarStart + off + len,
      dstStart: cursor, dstEnd: cursor + len,
      rate: 1, fadeIn: FADE, fadeOut: FADE, kind,
    });
    cursor += len;
  };
  const diff = tgtEighthsPerBar - srcEighthsPerBar;
  if (diff === 0) push(0, barLen, 'keep');
  else if (diff < 0) push(0, eighth * tgtEighthsPerBar, 'keep');
  else {
    push(0, barLen, 'keep');
    push(eighth * Math.max(0, srcEighthsPerBar - diff), eighth * diff, 'duplicate');
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
  const srcMeter = [
    meta.detectedMeter || project.beatsPerBar || 4,
    meta.detectedMeterDenominator || 4,
  ];
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
 * Detect approximate hits-per-beat from an onset list (onset-based
 * triplet detector). Mirrors the intent of time-sig-editor's
 * _detect_drum_pattern but uses cached librosa onsets, not ACF.
 * Returns 0 if no confident match.
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
  return bestErr < 0.25 ? best : 0;
}

/** Triplet content fits target meter? Triplets in /8, 5/4, 3/4 do not. */
function patternFitsTargetMeter(hpb, tgtMeter) {
  if (hpb !== 3) return true;
  const [tn, td] = tgtMeter;
  if (td === 8) return false;
  if (tn === 5 || tn === 7) return false;
  return true;
}

/**
 * Choose target subdivision when re-quantizing triplets:
 * 16ths under 135 BPM (more energy), 8ths above (avoid clutter).
 * Returns subdivisions PER tgt eighth (so 16ths = 2/eighth, 8ths = 1/eighth).
 */
function chooseTargetSubdivPerEighth(bpm) {
  return bpm < 135 ? 2 : 1;
}

/**
 * Triplet re-quantize: when src has triplet hits (hpb=3) and the target
 * meter doesn't fit triplets (7/8, 5/4, 3/4), replace the triplet pattern
 * with a steady 16th- or 8th-note pattern in the new bar — pluck the
 * closest src onset's audio window onto each tgt subdivision position.
 *
 * Mirrors time-sig-editor's _requantize_stem ("8 src triplet hits → 8
 * sixteenths in 4 eighths") but uses rate-1 placement instead of
 * time-stretch. Each tgt subdivision plays a real src hit (closest in
 * time within the bar) so dynamics are preserved.
 */
function buildTripletReQuantizeSchedule({
  duration, onsets, srcMeter, tgtMeter, bpm,
  downbeatOffset = 0, cropStart = 0, barStarts,
}) {
  const HIT_PRE  = 0.005;
  const FADE_IN  = 0.001;
  const FADE_OUT = 0.015;

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);
  const subdivPerEighth = chooseTargetSubdivPerEighth(bpm);
  const tgtSubdivPerBar = tgtEighths * subdivPerEighth;

  const segs = [];
  let dstCursor = 0;

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
    const tgtBarLen = srcEighthLen * tgtEighths;             // rate 1
    const tgtSubdivLen = tgtBarLen / tgtSubdivPerBar;

    const inBar = (Array.isArray(onsets) ? onsets : [])
      .filter((o) => o >= sbs && o < sbe)
      .sort((a, b) => a - b);

    if (inBar.length === 0) {
      dstCursor += tgtBarLen;
      continue;
    }

    for (let k = 0; k < tgtSubdivPerBar; k++) {
      // Map this tgt subdivision back into src time at proportional
      // position within the src bar — finds the equivalent triplet hit.
      const relPos = k / tgtSubdivPerBar;
      const srcTimeApprox = sbs + relPos * srcBarLen;
      let best = inBar[0], bestDist = Math.abs(inBar[0] - srcTimeApprox);
      for (const o of inBar) {
        const d = Math.abs(o - srcTimeApprox);
        if (d < bestDist) { bestDist = d; best = o; }
      }
      const idx = inBar.indexOf(best);
      const next = (idx + 1 < inBar.length) ? inBar[idx + 1] : sbe;
      const winSrc = Math.min(next - best + HIT_PRE, tgtSubdivLen + HIT_PRE);
      const srcStart = cropStart + Math.max(sbs, best - HIT_PRE);
      const srcEnd = srcStart + winSrc;
      const dstHitStart = dstCursor + k * tgtSubdivLen;
      segs.push({
        srcStart, srcEnd,
        dstStart: dstHitStart,
        dstEnd: dstHitStart + winSrc,
        rate: 1, fadeIn: FADE_IN, fadeOut: FADE_OUT, kind: 'tripletQuant',
      });
    }

    dstCursor += tgtBarLen;
  }

  segs.sort((a, b) => a.dstStart - b.dstStart);
  return segs;
}

/**
 * Per-substem schedule builder: rules-table-driven musical snap with
 * rate-1 keep + duplicate slices. Supports every meter combo in
 * METER_RULES with substem-aware snare placement.
 *
 * Triplet flow: if onset density indicates triplets and the target meter
 * doesn't fit triplets, hand off to buildTripletReQuantizeSchedule which
 * pluck-and-places src triplet hits onto the new 16th/8th grid.
 *
 * Otherwise per src bar:
 *   1. Find onsets in [bar_start, bar_end).
 *   2. For each onset, compute fractional src eighth.
 *   3. placementsForSrcEighth(srcE, rule, substemName) → list of tgt eighth
 *      positions (handles keep / warp / duplicate / drop sections + snare
 *      snap musical placement in one pass).
 *   4. Each placement emits a rate-1 segment with HIT_PRE/HIT_MAX window
 *      and 2/20ms fades. Sub-eighth offset preserved (groove).
 *
 * Segments may overlap on dst (a duplicated hit fires multiple times,
 * a long tail can extend past the next hit). The engine schedules each
 * as its own AudioBufferSourceNode → segGain → trackGain, mixing naturally.
 */
function buildPercussiveSubstemSchedule({
  duration, onsets, srcMeter, tgtMeter, bpm, substemName,
  downbeatOffset = 0, cropStart = 0, barStarts,
}) {
  if (!(duration > 0)) return [];
  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }

  // Triplet check: route to re-quantize if src is triplet and tgt doesn't fit.
  const hpb = detectHitsPerBeat(onsets, duration, bpm);
  if (hpb === 3 && !patternFitsTargetMeter(hpb, tgtMeter)) {
    console.log(`[virtualTrackEdit] ${substemName}: triplets detected → re-quantize to ${chooseTargetSubdivPerEighth(bpm) === 2 ? '16ths' : '8ths'}`);
    return buildTripletReQuantizeSchedule({
      duration, onsets, srcMeter, tgtMeter, bpm,
      downbeatOffset, cropStart, barStarts,
    });
  }

  const HIT_PRE  = 0.005;
  const HIT_MAX  = 0.5;
  const FADE_IN  = 0.002;
  const FADE_OUT = 0.020;

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  const rule = ruleFor(srcMeter, tgtMeter);
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);

  const segs = [];
  let dstCursor = 0;

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
    const tgtEighthLen = srcEighthLen;                       // rate 1
    const tgtBarLen = tgtEighthLen * tgtEighths;

    const inBar = (Array.isArray(onsets) ? onsets : [])
      .filter((o) => o >= sbs && o < sbe)
      .sort((a, b) => a - b);

    for (let i = 0; i < inBar.length; i++) {
      const onset = inBar[i];
      const srcEighthFloat = (onset - sbs) / srcEighthLen;
      const tgts = rule
        ? placementsForSrcEighth(srcEighthFloat, rule, substemName)
        // No rule → generic proportional snap. Keep hits inside tgt only.
        : (() => {
            const t = srcEighthFloat * tgtEighths / srcEighths;
            return t < tgtEighths - EPS ? [t] : [];
          })();
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
  // detectedMeterDenominator defaults to 4 since the current
  // `detectChordsAndTempo` only returns `beats_per_bar` (numerator).
  // Once the detector reports denominator, set it on the track metadata
  // and per-/8 source conversions slot in for free.
  const srcMeter = [
    meta.detectedMeter || project.beatsPerBar || 4,
    meta.detectedMeterDenominator || 4,
  ];
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
