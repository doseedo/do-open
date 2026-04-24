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

import { barStartsFromTempoMap } from './tempoMap.js';

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
    { duplicate: { srcRange: [4, 6], tgtRange: [6, 8] } },
  ], impliedSnareTgtEighths: [6] },     // backbeat on the new beat 4
  '4/4->3/4': { srcEighths: 8, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 8] },
  ]},
  '3/4->4/4': { srcEighths: 6, tgtEighths: 8, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [4, 6], tgtRange: [6, 8] } },
  ], impliedSnareTgtEighths: [6] },
  '4/4->5/4': { srcEighths: 8, tgtEighths: 10, sections: [
    { keep: [0, 8] },
    { duplicate: { srcRange: [6, 8], tgtRange: [8, 10] } },
  ]},                                    // no implied — beat 5 is upbeat
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

  // ─── 5/4 grouping variants (2+3 vs 3+2) ─────────────────────────────
  // Default keys above assume 3+2 grouping (beat 4 is the start of the
  // 2-group → backbeat snare). For 2+3 grouped songs (Mission:Impossible,
  // Take Five's B section), the backbeat sits on beat 3 (start of the
  // 3-group). These variant keys replace the default when a track or
  // project has grouping='2+3'.
  '5/4(2+3)->4/4': { srcEighths: 10, tgtEighths: 8, sections: [
    // Keep the 2-group (beats 1-2) identity, then the 3-group (beats 3-5)
    // compresses to 2 beats via keep+drop inside the warp.
    { keep: [0, 4] },
    { warp: { srcRange: [4, 10], tgtRange: [4, 8] }, snareSnap: 4 },
  ], impliedSnareTgtEighths: [4] },
  '4/4->5/4(2+3)': { srcEighths: 8, tgtEighths: 10, sections: [
    // Keep beats 1-2 identity, then stretch beats 3-4 to fill the 3-group.
    { keep: [0, 4] },
    { warp: { srcRange: [4, 8], tgtRange: [4, 10] }, snareSnap: 4 },
  ], impliedSnareTgtEighths: [4] },
  '5/4(2+3)->7/8': { srcEighths: 10, tgtEighths: 7, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 10], tgtRange: [4, 7] }, snareSnap: 4 },
  ]},
  '7/8->5/4(2+3)': { srcEighths: 7, tgtEighths: 10, sections: [
    { keep: [0, 4] },
    { warp: { srcRange: [4, 7], tgtRange: [4, 10] }, snareSnap: 4 },
  ]},
  '3/4->5/4(2+3)': { srcEighths: 6, tgtEighths: 10, sections: [
    // 3/4 has 3 beats. 2+3 5/4 puts the accent on beat 3. Keep bars 1-2
    // identity, duplicate beat 3 over eighths 4-9 to fill.
    { keep: [0, 4] },
    { duplicate: { srcRange: [4, 6], tgtRange: [4, 10] }, snareSnap: 4 },
  ]},

  // ─── 9/8 rules (grouping 2+2+2+3 = dominant, 3+3+3 = alt) ────────────
  '4/4->9/8': { srcEighths: 8, tgtEighths: 9, sections: [
    // Keep first 4 eighths, duplicate the last eighth to fill 9.
    { keep: [0, 8] },
    { duplicate: { srcRange: [6, 8], tgtRange: [8, 9] } },
  ]},
  '9/8->4/4': { srcEighths: 9, tgtEighths: 8, sections: [
    { keep: [0, 8] }, { drop: [8, 9] },
  ]},
  '3/4->9/8': { srcEighths: 6, tgtEighths: 9, sections: [
    // 3/4 → 9/8 compound: duplicate beats 2-3 to fill eighths 6-8.
    { keep: [0, 6] },
    { duplicate: { srcRange: [3, 6], tgtRange: [6, 9] } },
  ]},
  '9/8->3/4': { srcEighths: 9, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 9] },
  ]},
  '6/8->9/8': { srcEighths: 6, tgtEighths: 9, sections: [
    { keep: [0, 6] },
    { duplicate: { srcRange: [3, 6], tgtRange: [6, 9] } },
  ]},
  '9/8->6/8': { srcEighths: 9, tgtEighths: 6, sections: [
    { keep: [0, 6] }, { drop: [6, 9] },
  ]},
  '9/8->9/8': { srcEighths: 9, tgtEighths: 9, sections: [{ keep: [0, 9] }] },

  // ─── 11/8 rules (dominant grouping 4+3+4 or 3+4+4) ───────────────────
  '4/4->11/8': { srcEighths: 8, tgtEighths: 11, sections: [
    { keep: [0, 8] },
    { duplicate: { srcRange: [5, 8], tgtRange: [8, 11] } },
  ]},
  '11/8->4/4': { srcEighths: 11, tgtEighths: 8, sections: [
    { keep: [0, 8] }, { drop: [8, 11] },
  ]},
  '11/8->7/8': { srcEighths: 11, tgtEighths: 7, sections: [
    { keep: [0, 7] }, { drop: [7, 11] },
  ]},
  '7/8->11/8': { srcEighths: 7, tgtEighths: 11, sections: [
    { keep: [0, 7] },
    { duplicate: { srcRange: [3, 7], tgtRange: [7, 11] } },
  ]},
  '11/8->11/8': { srcEighths: 11, tgtEighths: 11, sections: [{ keep: [0, 11] }] },
};

/**
 * Look up a meter rule with optional grouping awareness.
 *
 * Grouping is a per-meter annotation:
 *   5/4: '3+2' (default) or '2+3'   — affects backbeat placement
 *   7/8: '4+3' (default) or '3+4'   — affects which group is the longer one
 *   9/8: '2+2+2+3' (default) or '3+3+3'
 *
 * Call sites supply grouping via:
 *   - track.metadata.detectedGrouping (from /api/analyze-rhythm)
 *   - project.grouping (if the project-level map has grouping on the
 *     relevant entry)
 *   - explicit srcGrouping/tgtGrouping args to this function
 *
 * The lookup first tries '{src(Grp)}->{tgt(Grp)}', then falls back to
 * '{src(Grp)}->{tgt}', '{src}->{tgt(Grp)}', and finally '{src}->{tgt}'.
 * Any miss returns null and the caller falls back to the generic path.
 */
function ruleFor(srcMeter, tgtMeter, srcGrouping = null, tgtGrouping = null) {
  const s = `${srcMeter[0]}/${srcMeter[1]}`;
  const t = `${tgtMeter[0]}/${tgtMeter[1]}`;
  const sG = srcGrouping ? `${s}(${srcGrouping})` : s;
  const tG = tgtGrouping ? `${t}(${tgtGrouping})` : t;
  return METER_RULES[`${sG}->${tG}`]
      || METER_RULES[`${sG}->${t}`]
      || METER_RULES[`${s}->${tG}`]
      || METER_RULES[`${s}->${t}`]
      || null;
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
function remapBar({ srcBarStart, srcBarEnd, srcMeter, tgtMeter, dstCursor, srcGrouping, tgtGrouping }) {
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

  const rule = ruleFor(srcMeter, tgtMeter, srcGrouping, tgtGrouping);
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
  tempoMap = null,        // optional per-bar src tempo+meter map
  tgtTempoMap = null,     // optional per-bar tgt tempo+meter map
  srcGrouping = null,     // whole-track grouping fallback (tempoMap entries
  tgtGrouping = null,     // still override per-bar when present)
}) {
  if (!(duration > 0)) return [];

  // Use the tempoMap's real downbeat times + per-bar local meter when
  // available. This is what makes tempo drift AND in-song meter changes
  // line up on the actual bar boundaries instead of a synthesized grid.
  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : (tempoMap && tempoMap.length > 0)
      ? barStartsFromTempoMap(tempoMap, duration, downbeatOffset)
      : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  // Per-bar src meter lookup: each bar uses the tempoMap entry that
  // contains it. When tempoMap is null, every bar uses the constant
  // srcMeter passed by the caller.
  const localSrcMeterAtBar = (barIdx, barTime) => {
    if (!tempoMap || tempoMap.length === 0) return srcMeter;
    // Locate entry by bar number (1-indexed). First entry covers bar 1.
    let e = tempoMap[0];
    for (const cand of tempoMap) {
      if (cand.bar <= barIdx + 1) e = cand;
      else break;
    }
    return e.meter || srcMeter;
  };
  // Per-bar src grouping lookup (same shape as meter — tempoMap entries
  // can carry a grouping hint: '3+2' / '2+3' for 5/4, '4+3' / '3+4' for 7/8).
  const localSrcGroupingAtBar = (barIdx) => {
    if (!tempoMap || tempoMap.length === 0) return srcGrouping;
    let e = tempoMap[0];
    for (const cand of tempoMap) {
      if (cand.bar <= barIdx + 1) e = cand;
      else break;
    }
    return e.grouping || srcGrouping;
  };
  // Per-bar tgt meter lookup: when the project has its own tempoMap, the
  // target meter can vary per bar too (e.g. a song that starts 4/4 and
  // flips to 7/8 at bar 17 under a different project-level meter map).
  const localTgtMeterAtBar = (barIdx, barTime) => {
    if (!tgtTempoMap || tgtTempoMap.length === 0) return tgtMeter;
    let e = tgtTempoMap[0];
    for (const cand of tgtTempoMap) {
      if (cand.bar <= barIdx + 1) e = cand;
      else break;
    }
    return e.meter || tgtMeter;
  };
  const localTgtGroupingAtBar = (barIdx) => {
    if (!tgtTempoMap || tgtTempoMap.length === 0) return tgtGrouping;
    let e = tgtTempoMap[0];
    for (const cand of tgtTempoMap) {
      if (cand.bar <= barIdx + 1) e = cand;
      else break;
    }
    return e.grouping || tgtGrouping;
  };

  // Fast path: identity meter across the whole song → single identity segment.
  const everyBarIdentity =
    !tempoMap && !tgtTempoMap &&
    srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1];
  if (everyBarIdentity) return identitySchedule(duration, cropStart);

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

  // Remap each complete source bar using its OWN local meter conversion.
  for (let b = 0; b < starts.length - 1; b++) {
    const sbs = starts[b];
    const sbe = starts[b + 1];
    if (sbe - sbs < EPS) continue;
    const barSrcMeter = localSrcMeterAtBar(b, sbs);
    const barTgtMeter = localTgtMeterAtBar(b, sbs);
    const barSrcGrouping = localSrcGroupingAtBar(b);
    const barTgtGrouping = localTgtGroupingAtBar(b);
    const { segs, dstAdvance } = remapBar({
      srcBarStart: cropStart + sbs,
      srcBarEnd:   cropStart + sbe,
      srcMeter: barSrcMeter,
      tgtMeter: barTgtMeter,
      srcGrouping: barSrcGrouping,
      tgtGrouping: barTgtGrouping,
      dstCursor,
    });
    out.push(...segs);
    dstCursor += dstAdvance;
  }
  return out;
}

/* ------------------------------------------------------------------
 * Onset-protected schedule builder.
 *
 * Converts bar-level meter edits into segments BUT refuses to cut across
 * a protected region (a sustained word, bass note, etc.). When a drop/
 * duplicate would land inside a protected region, the whole affected
 * bar is emitted as a rate-!=1 stretch segment instead — the engine
 * routes that through WSOLA (services/wsolaStretch.js) so pitch is
 * preserved and no lyric/note is cut.
 *
 * Used by buildVocalProtectedSchedule (with whisper word timings as
 * protected regions) and buildMelodicProtectedSchedule (with per-stem
 * onsets + note-length estimates).
 *
 * A "protected region" is any interval where cutting is unacceptable.
 * The rule: for a bar [sbs, sbe), if any section boundary proposed by
 * the meter rule falls INSIDE such an interval, stretch the bar.
 * Otherwise fall through to the regular rate-1 rearrange path.
 * ------------------------------------------------------------------ */

function buildOnsetProtectedSchedule({
  duration, protectedRegions, srcMeter, tgtMeter, bpm,
  downbeatOffset = 0, cropStart = 0, barStarts, tempoMap, tgtTempoMap,
  stemLabel = 'stem',
  forbidDuplicates = false,     // vocals: true (repeating a word is audible)
                                // bass:   false (duplicate bass notes are fine)
  srcGrouping = null,           // e.g. '3+2' / '2+3' for 5/4
  tgtGrouping = null,
}) {
  if (!(duration > 0)) return [];
  if (!Array.isArray(protectedRegions)) protectedRegions = [];

  // No meter change → identity for the whole clip.
  const sameMeter = srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1];
  if (sameMeter && (!tempoMap || tempoMap.length === 0)
               && (!tgtTempoMap || tgtTempoMap.length === 0)) {
    return identitySchedule(duration, cropStart);
  }

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : (tempoMap && tempoMap.length > 0)
      ? barStartsFromTempoMap(tempoMap, duration, downbeatOffset)
      : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  // Binary-search-friendly region test: does [a, b) overlap any protected
  // region? Regions are sorted by .start for performance when big.
  const regions = [...protectedRegions].sort((a, b) => a.start - b.start);
  const overlapsProtected = (a, b) => {
    for (const r of regions) {
      if (r.end <= a) continue;
      if (r.start >= b) return false;
      return true;
    }
    return false;
  };

  const FADE = 0.003;
  const FADE_STRETCH = 0.020;
  const out = [];
  let dstCursor = 0;

  if (starts[0] > EPS) {
    out.push({
      srcStart: cropStart,
      srcEnd:   cropStart + starts[0],
      dstStart: 0, dstEnd: starts[0],
      rate: 1, fadeIn: 0, fadeOut: 0, kind: 'preroll',
    });
    dstCursor = starts[0];
  }

  for (let b = 0; b < starts.length - 1; b++) {
    const sbs = starts[b];
    const sbe = starts[b + 1];
    const srcBarLen = sbe - sbs;
    if (srcBarLen < EPS) continue;

    // Per-bar local src meter (tempoMap may vary across the clip).
    const barSrcMeter = (tempoMap && tempoMap.length > 0)
      ? (tempoMap.reduce((acc, e) => (e.bar <= b + 1 ? e : acc), tempoMap[0]).meter || srcMeter)
      : srcMeter;
    const barTgtMeter = (tgtTempoMap && tgtTempoMap.length > 0)
      ? (tgtTempoMap.reduce((acc, e) => (e.bar <= b + 1 ? e : acc), tgtTempoMap[0]).meter || tgtMeter)
      : tgtMeter;

    if (barSrcMeter[0] === barTgtMeter[0] && barSrcMeter[1] === barTgtMeter[1]) {
      // Identity bar.
      out.push({
        srcStart: cropStart + sbs, srcEnd: cropStart + sbe,
        dstStart: dstCursor, dstEnd: dstCursor + srcBarLen,
        rate: 1, fadeIn: FADE, fadeOut: FADE, kind: 'identity',
      });
      dstCursor += srcBarLen;
      continue;
    }

    // Determine which src ranges the rule actually drops (or duplicates
    // for forbidDuplicates stems). If any protected region overlaps a
    // discarded/repeated range, this bar must stretch instead.
    const rule = ruleFor(barSrcMeter, barTgtMeter, srcGrouping, tgtGrouping);
    const srcEighths = barSrcMeter[0] * (barSrcMeter[1] === 4 ? 2 : 1);
    const tgtEighths = barTgtMeter[0] * (barTgtMeter[1] === 4 ? 2 : 1);
    const srcEighthLen = srcBarLen / srcEighths;
    const tgtBarLen = srcEighthLen * tgtEighths;

    const protectedIncompatibleRanges = [];   // src-time seconds [a, b)
    if (rule) {
      for (const sec of rule.sections) {
        if (sec.drop) {
          // Explicit drop: these src eighths are discarded entirely.
          protectedIncompatibleRanges.push({
            a: sbs + sec.drop[0] * srcEighthLen,
            b: sbs + sec.drop[1] * srcEighthLen,
          });
        } else if (sec.warp) {
          const [sa, sb] = sec.warp.srcRange;
          const [ta, tb] = sec.warp.tgtRange;
          const srcWin = sb - sa;
          const tgtWin = tb - ta;
          if (tgtWin < srcWin) {
            // Drop-path warp: last (srcWin - tgtWin) src eighths are dropped.
            protectedIncompatibleRanges.push({
              a: sbs + (sa + tgtWin) * srcEighthLen,
              b: sbs + sb * srcEighthLen,
            });
          }
          // Duplicate-path warp (tgtWin > srcWin) doesn't drop src, it
          // duplicates. For forbidDuplicates stems, mark the duplicated
          // tail as incompatible too.
          if (forbidDuplicates && tgtWin > srcWin) {
            const dupEighths = tgtWin - srcWin;
            protectedIncompatibleRanges.push({
              a: sbs + (sb - dupEighths) * srcEighthLen,
              b: sbs + sb * srcEighthLen,
            });
          }
        } else if (sec.duplicate && forbidDuplicates) {
          // Explicit duplicate section: src range gets re-read. For vocals
          // this means hearing a word twice, which the protection forbids.
          protectedIncompatibleRanges.push({
            a: sbs + sec.duplicate.srcRange[0] * srcEighthLen,
            b: sbs + sec.duplicate.srcRange[1] * srcEighthLen,
          });
        }
      }
    }
    // Test: does any protected region overlap any incompatible range?
    const cutHitsProtected = protectedIncompatibleRanges.some(({ a, b }) =>
      overlapsProtected(a, b));

    if (!cutHitsProtected && rule) {
      // Safe to rearrange at rate 1 — no protected region overlaps any cut.
      const srcEighthLenAtBar = srcEighthLen;
      const { segs, dstAdvance } = (function barRearrange() {
        const segs = [];
        let cursor = dstCursor;
        const emit = (kind, srcOffEighths, lenEighths) => {
          const len = lenEighths * srcEighthLenAtBar;
          segs.push({
            srcStart: cropStart + sbs + srcOffEighths * srcEighthLenAtBar,
            srcEnd:   cropStart + sbs + srcOffEighths * srcEighthLenAtBar + len,
            dstStart: cursor, dstEnd: cursor + len,
            rate: 1, fadeIn: FADE, fadeOut: FADE, kind,
          });
          cursor += len;
        };
        for (const sec of rule.sections) {
          if (sec.keep) emit('keep', sec.keep[0], sec.keep[1] - sec.keep[0]);
          else if (sec.drop) { /* skip */ }
          else if (sec.warp) {
            const [sa, sb] = sec.warp.srcRange;
            const [ta, tb] = sec.warp.tgtRange;
            if (tb - ta <= sb - sa) emit('keep', sa, tb - ta);
            else {
              emit('keep', sa, sb - sa);
              const dupEighths = (tb - ta) - (sb - sa);
              emit('duplicate', sb - dupEighths, dupEighths);
            }
          } else if (sec.duplicate) {
            const [sa, sb] = sec.duplicate.srcRange;
            emit('duplicate', sa, sb - sa);
          }
        }
        return { segs, dstAdvance: cursor - dstCursor };
      })();
      out.push(...segs);
      dstCursor += dstAdvance;
    } else {
      // Protected region OR no explicit rule → stretch the whole bar via
      // WSOLA. srcLen stays the full bar, dstLen = what the rule wants the
      // tgt bar to be at the local meter. rate = srcLen / dstLen.
      const dstLen = tgtBarLen;
      out.push({
        srcStart: cropStart + sbs, srcEnd: cropStart + sbe,
        dstStart: dstCursor, dstEnd: dstCursor + dstLen,
        rate: srcBarLen / dstLen,
        fadeIn: FADE_STRETCH, fadeOut: FADE_STRETCH,
        kind: 'stretch',
      });
      if (cutHitsProtected) {
        console.log(`[virtualTrackEdit] ${stemLabel}: bar ${b + 1} cut would hit a protected region → WSOLA stretch (${(srcBarLen * 1000).toFixed(0)}ms → ${(dstLen * 1000).toFixed(0)}ms)`);
      }
      dstCursor += dstLen;
    }
  }

  return out;
}

/**
 * Vocal-protected schedule.
 *
 * Protected regions = every whisper word interval [word.start, word.end].
 * Per bar: if the meter rule's cut points would fall inside a word, the
 * bar stretches via WSOLA instead of dropping/duplicating. Otherwise we
 * rearrange like any other stem.
 *
 * Without lyrics metadata, falls through to the regular bar-rearrange
 * (so vocals without whisper data still play correctly, just without the
 * protection guarantee).
 */
export function buildVocalProtectedSchedule({ track, project }) {
  const meta = track?.metadata || {};
  const duration = track.duration || track.length || 0;
  if (duration <= 0) return identitySchedule(0);
  const cropStart = track.cropStart || 0;
  const srcBpm = meta.detectedBpm || project.bpm || 120;
  const srcN = meta.detectedMeter || project.beatsPerBar || 4;
  const srcD = meta.detectedMeterDenominator || inferMeterDenominator(srcN, null, srcBpm);
  const srcMeter = [srcN, srcD];
  const tgtMeter = [project.beatsPerBar || 4, project.meterDenominator || 4];
  const downbeatOffset = typeof meta.downbeatOffset === 'number' ? meta.downbeatOffset : 0;

  const words = Array.isArray(meta.vocalsLyrics) ? meta.vocalsLyrics : [];
  const regions = words
    .filter((w) => typeof w.start === 'number' && typeof w.end === 'number' && w.end > w.start)
    .map((w) => ({ start: w.start, end: w.end }));

  return buildOnsetProtectedSchedule({
    duration, protectedRegions: regions,
    srcMeter, tgtMeter, bpm: srcBpm,
    downbeatOffset, cropStart,
    barStarts: meta.barStarts,
    tempoMap: meta.tempoMap || null,
    tgtTempoMap: project.tempoMap || null,
    stemLabel: 'vocals',
    forbidDuplicates: true,       // never repeat a word
    srcGrouping: meta.detectedGrouping || null,
    tgtGrouping: project.grouping || null,
  });
}

/**
 * Melodic-protected schedule (bass and sustained pitched stems).
 *
 * Protected regions = onset-to-next-onset intervals from per-stem librosa
 * onsets (meta.stemOnsets). Treats each onset as a note that sustains
 * until the next onset or bar boundary. Cutting mid-note is audible on
 * pitched content; this gates the drop/duplicate path the same way vocals
 * are gated.
 *
 * Without stem onsets, falls through to regular bar-rearrange.
 */
export function buildMelodicProtectedSchedule({ track, project }) {
  const meta = track?.metadata || {};
  const duration = track.duration || track.length || 0;
  if (duration <= 0) return identitySchedule(0);
  const cropStart = track.cropStart || 0;
  const srcBpm = meta.detectedBpm || project.bpm || 120;
  const srcN = meta.detectedMeter || project.beatsPerBar || 4;
  const srcD = meta.detectedMeterDenominator || inferMeterDenominator(srcN, null, srcBpm);
  const srcMeter = [srcN, srcD];
  const tgtMeter = [project.beatsPerBar || 4, project.meterDenominator || 4];
  const downbeatOffset = typeof meta.downbeatOffset === 'number' ? meta.downbeatOffset : 0;

  const onsets = Array.isArray(meta.stemOnsets) ? meta.stemOnsets : [];
  const MAX_NOTE_LEN = 2.0;
  const regions = [];
  for (let i = 0; i < onsets.length; i++) {
    const start = onsets[i];
    const next = i + 1 < onsets.length ? onsets[i + 1] : start + MAX_NOTE_LEN;
    const end = Math.min(next, start + MAX_NOTE_LEN);
    if (end > start) regions.push({ start, end });
  }

  return buildOnsetProtectedSchedule({
    duration, protectedRegions: regions,
    srcMeter, tgtMeter, bpm: srcBpm,
    downbeatOffset, cropStart,
    barStarts: meta.barStarts,
    tempoMap: meta.tempoMap || null,
    tgtTempoMap: project.tempoMap || null,
    stemLabel: meta.stemType || 'melodic',
    srcGrouping: meta.detectedGrouping || null,
    tgtGrouping: project.grouping || null,
  });
}

/**
 * Per-stem-type strategy dispatcher.
 *
 * Replaces the monolithic getTrackSchedule with a router that picks the
 * right schedule builder based on track.metadata.stemType:
 *
 *   vocals          → buildVocalProtectedSchedule   (never cut lyrics)
 *   drums/drum_kit  → drumSubstems path handles this — dispatcher skips
 *                     drum tracks and lets useAudioPlayback call
 *                     getTrackSubstemSchedules directly.
 *   bass            → buildMelodicProtectedSchedule (never cut sustained notes)
 *   other pitched   → buildMeterSchedule            (bar rearrange)
 *   unknown/no meta → buildMeterSchedule
 */
export function dispatchStrategy(track, project) {
  const stemType = track?.metadata?.stemType || track?.metadata?.instrument || '';
  const st = stemType.toLowerCase();
  if (st === 'vocals' || st === 'lead_vox' || st === 'bg_vox' || st === 'choir') {
    return buildVocalProtectedSchedule({ track, project });
  }
  if (st === 'bass') {
    return buildMelodicProtectedSchedule({ track, project });
  }
  // Everything else (piano, guitar, synth, pads, etc.) goes through the
  // same bar-rearrange path getTrackSchedule was using. No protection
  // yet — these stems' onsets aren't cached by the backend. When we add
  // per-stem onset extraction to separate-stems, we can route pitched
  // non-bass content through buildMelodicProtectedSchedule too.
  return getTrackSchedule(track, project);
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
  const srcN = meta.detectedMeter || project.beatsPerBar || 4;
  const srcD = meta.detectedMeterDenominator
    || inferMeterDenominator(srcN, meta.drumSubstemOnsets, srcBpm);
  const srcMeter = [srcN, srcD];
  const tgtMeter = [project.beatsPerBar || 4, project.meterDenominator || 4];
  const downbeatOffset = typeof meta.downbeatOffset === 'number' ? meta.downbeatOffset : 0;

  // Identity fast path: if there's no tempoMap (= constant tempo/meter)
  // AND the src meter equals tgt meter, just play straight through.
  const hasTempoMap = Array.isArray(meta.tempoMap) && meta.tempoMap.length > 0;
  const hasTgtMap = Array.isArray(project.tempoMap) && project.tempoMap.length > 0;
  if (!hasTempoMap && !hasTgtMap &&
      srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }

  return buildMeterSchedule({
    duration, srcMeter, tgtMeter, bpm: srcBpm, downbeatOffset, cropStart,
    barStarts: meta.barStarts,
    tempoMap: meta.tempoMap || null,
    tgtTempoMap: project.tempoMap || null,
    srcGrouping: meta.detectedGrouping || null,
    tgtGrouping: project.grouping || null,
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
const HIHAT_SUBSTEMS = new Set(['hh', 'hat', 'hihat']);

/* ------------------------------------------------------------------
 * TRIPLET_METER_RULES — triplet-source meter conversions.
 *
 * Separate from METER_RULES because the latter is eighth-coordinate and
 * can't express triplet→16th re-quantization. When src is detected as
 * triplet AND tgt can't fit triplets (any /8 or odd-numerator meter),
 * we use these rules instead.
 *
 * Each rule defines:
 *   srcTripletsPerBar     — always 12 for /4 sources, 9 for 3/4 etc.
 *   tripletTo16th[k]      — src triplet k → tgt 16th idx (or -1 if dropped)
 *   hh:    { path, srcTripletsInBar }  — hi-hat routing + WSOLA window
 *   snare: { tripletSet }              — allowed src triplet positions
 *   kick:  { boundary, interior, phraseLen, boundaryAt }
 *                                       — phrase-aware triplet sets
 *   general: { tripletSet }            — toms/ride/crash
 * ------------------------------------------------------------------ */
const TRIPLET_METER_RULES = {
  '4/4->7/8': {
    srcTripletsPerBar: 12,
    // 4+1.5+1.5 grouping: src triplets 0-5 → tgt 16ths [0,1,2,4,5,6] (skip 3,7),
    // src triplets 6-11 → tgt 16ths [8,9,10,11,12,13] (1:1).
    tripletTo16th: [0, 1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13],
    hh: { path: 'wsolaStretch', srcTripletsInBar: 13 },
    snare: { tripletSet: [4, 7, 10] },       // backbeat "e" positions
    kick: {
      boundary: [0, 1, 3, 4, 6, 7, 9, 10],   // 8 hits incl. downbeat
      interior: [1, 3, 4, 6, 7, 9, 10],      // 7 hits no downbeat
      phraseLen: 4,                           // 4-bar phrase
      boundaryAt: [0, 3],                     // phrase-mod-4 positions with downbeat
    },
    general: { tripletSet: [0, 1, 3, 4, 6, 7, 9, 10] },
  },
  // Add other triplet-aware meter pairs here (4/4↔3/4, 4/4↔5/4 etc.)
  // as they're empirically tuned against hand-corrected references.
};

function tripletRuleFor(srcMeter, tgtMeter) {
  const key = `${srcMeter[0]}/${srcMeter[1]}->${tgtMeter[0]}/${tgtMeter[1]}`;
  return TRIPLET_METER_RULES[key] || null;
}

/**
 * Triplet-source meter conversion. Reads TRIPLET_METER_RULES for the
 * current meter pair and emits WSOLA stretch segments.
 *
 *   HI-HAT (rule.hh.path === 'wsolaStretch'): per bar emit ONE rate≠1
 *     segment stretching `srcTripletsInBar * srcTripletLen` → tgt bar.
 *     User's method: "take 14 triplets, stretch to fit 7/8, offset to grid."
 *
 *   KICK / SNARE / TOMS / RIDE / CRASH: per bar, for each allowed src
 *     triplet position, find nearest src onset; emit WSOLA stretch
 *     between consecutive (srcT, dstT) pairs. Rate varies per segment
 *     to match local src/tgt spacing.
 */
function buildOnsetStretchSchedule({
  duration, onsets, srcMeter, tgtMeter, bpm, substemName,
  downbeatOffset = 0, cropStart = 0, barStarts,
}) {
  if (!(duration > 0)) return [];
  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }
  const rule = tripletRuleFor(srcMeter, tgtMeter);
  if (!rule) return [];   // no triplet handling for this meter pair

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);
  const subdivPerEighth = chooseTargetSubdivPerEighth(bpm);
  const nTgtSlots = tgtEighths * subdivPerEighth;
  const isHiHat = HIHAT_SUBSTEMS.has(substemName);
  const onsetsArr = Array.isArray(onsets) ? onsets : [];

  // Rule-driven triplet sets per substem role.
  const SNARE_EXP   = rule.snare?.tripletSet   || [];
  const GENERAL_EXP = rule.general?.tripletSet || [];

  let firstActiveBarIdx = -1;
  for (let b = 0; b < starts.length - 1; b++) {
    if (onsetsArr.some((o) => o >= starts[b] && o < starts[b + 1])) {
      firstActiveBarIdx = b; break;
    }
  }

  const segs = [];
  let dstCursor = 0;
  if (starts[0] > EPS) {
    // Preroll plays raw source [0, starts[0]) at dst [0, starts[0]) so the
    // pickup survives. It used to extend 500ms PAST starts[0] to preserve
    // the acoustic tail of pickup hits — but that overlaps the mapped
    // segments (which also begin at dst=starts[0]) and re-fires the first
    // downbeat hit that's already in the raw source. With the preroll +
    // mapped layer playing the same hit 500ms apart the downbeat smeared,
    // and the per-substem overlap even shifted perceptually per play. Cap
    // strictly at starts[0] — bar 1 onwards is owned by the mapped
    // schedule. The mapped layer's first segment gets a short fadeIn to
    // hide the pickup-tail chop.
    segs.push({
      srcStart: cropStart, srcEnd: cropStart + starts[0],
      dstStart: 0, dstEnd: starts[0],
      rate: 1, fadeIn: 0, fadeOut: 0.02, kind: 'preroll',
    });
    dstCursor = starts[0];
  }

  if (isHiHat && rule.hh?.path === 'wsolaStretch') {
    // ONE stretch segment per bar: src window = srcTripletsInBar * trp.
    // Pulls that span of src triplets into the tgt bar. Rule-configured
    // srcTripletsInBar (e.g. 13 for 4/4→7/8) controls how far we borrow
    // from the next src bar — shorter = softer last-16th.
    const tripletsInBar = rule.hh.srcTripletsInBar;
    for (let b = 0; b < starts.length - 1; b++) {
      const sbs = starts[b], sbe = starts[b + 1];
      const srcBarLen = sbe - sbs;
      if (srcBarLen <= EPS) continue;
      const srcEighthLen = srcBarLen / srcEighths;
      const tgtBarLen = srcEighthLen * tgtEighths;
      const srcTripletLen = srcBarLen / rule.srcTripletsPerBar;
      const srcEnd = Math.min(duration, sbs + tripletsInBar * srcTripletLen);
      if (srcEnd <= sbs + EPS) { dstCursor += tgtBarLen; continue; }
      const rateB = (srcEnd - sbs) / tgtBarLen;
      segs.push({
        srcStart: cropStart + sbs, srcEnd: cropStart + srcEnd,
        dstStart: dstCursor, dstEnd: dstCursor + tgtBarLen,
        rate: rateB, fadeIn: 0.002, fadeOut: 0.01, kind: 'hhStretch',
      });
      dstCursor += tgtBarLen;
    }
    return segs;
  }

  // Kick / snare / toms / ride / crash — per-onset placement w/ stretch.
  // Each substem's allowed triplet positions come from the rule; kick's
  // per-phrase filter picks boundary-bar vs interior-bar sets based on
  // bar-mod-`phraseLen`.
  const kickRule = rule.kick || { boundary: [], interior: [], phraseLen: 4, boundaryAt: [] };
  const kickBoundarySet = new Set(kickRule.boundaryAt || [0]);
  const tripletTo16th = rule.tripletTo16th || [];
  const mappings = [];
  let dstC = starts[0] > EPS ? starts[0] : 0;
  for (let b = 0; b < starts.length - 1; b++) {
    const sbs = starts[b], sbe = starts[b + 1];
    const srcBarLen = sbe - sbs;
    if (srcBarLen <= EPS) continue;
    const srcEighthLen = srcBarLen / srcEighths;
    const tgtBarLen = srcEighthLen * tgtEighths;
    const srcTripletLen = srcBarLen / rule.srcTripletsPerBar;
    const tgtSlotLen = tgtBarLen / nTgtSlots;
    let expected;
    if (substemName === 'snare') expected = SNARE_EXP;
    else if (substemName === 'kick') {
      const phrasePos = firstActiveBarIdx >= 0
        ? (b - firstActiveBarIdx) % (kickRule.phraseLen || 4)
        : 0;
      expected = kickBoundarySet.has(phrasePos) ? kickRule.boundary : kickRule.interior;
    } else expected = GENERAL_EXP;
    const inBar = onsetsArr.filter((o) => o >= sbs && o < sbe);
    const used = new Set();
    for (const trpIdx of expected) {
      const expectedSrcT = sbs + trpIdx * srcTripletLen;
      let nearest = null, minDist = srcTripletLen;
      for (const o of inBar) {
        if (used.has(o)) continue;
        const d = Math.abs(o - expectedSrcT);
        if (d < minDist) { minDist = d; nearest = o; }
      }
      if (nearest === null) continue;
      used.add(nearest);
      const tgt16th = tripletTo16th[trpIdx];
      if (tgt16th === undefined || tgt16th < 0 || tgt16th >= nTgtSlots) continue;
      mappings.push({ srcT: nearest, dstT: dstC + tgt16th * tgtSlotLen });
    }
    dstC += tgtBarLen;
  }
  mappings.sort((a, b) => a.srcT - b.srcT);
  for (let i = 0; i < mappings.length; i++) {
    const srcStart = mappings[i].srcT;
    const dstStart = mappings[i].dstT;
    const hasNext = i + 1 < mappings.length;
    const srcEnd = hasNext ? mappings[i + 1].srcT : Math.min(duration, srcStart + 0.3);
    const dstEnd = hasNext ? mappings[i + 1].dstT : dstStart + (srcEnd - srcStart);
    if (srcEnd <= srcStart + EPS || dstEnd <= dstStart + EPS) continue;
    const rate = (srcEnd - srcStart) / (dstEnd - dstStart);
    // First mapped segment gets a slightly longer fadeIn to mask the
    // preroll→mapped handoff at dst=starts[0]. Subsequent segments are
    // onset-to-onset contiguous so 2ms is fine.
    const fadeIn = (i === 0 && starts[0] > EPS) ? 0.015 : 0.002;
    segs.push({
      srcStart: cropStart + srcStart, srcEnd: cropStart + srcEnd,
      dstStart, dstEnd,
      rate, fadeIn, fadeOut: 0.01, kind: 'onsetStretch',
    });
  }
  return segs;
}

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

/**
 * Per-bar hits-per-beat detection. Returns one value per bar interval
 * (barStarts[i] → barStarts[i+1]). A bar classified as 0 = sparse/silent.
 *
 * Per-bar granularity catches the common case where a song has a busy
 * 16th-note groove for most bars and a triplet fill bar every 8 bars —
 * the fill bar alone routes to triplet re-quantize, the groove bars use
 * the snap path.
 */
function detectHitsPerBeatPerBar(onsets, barStarts, bpm) {
  const out = [];
  if (!Array.isArray(onsets) || onsets.length === 0) {
    for (let i = 0; i < barStarts.length - 1; i++) out.push(0);
    return out;
  }
  for (let i = 0; i < barStarts.length - 1; i++) {
    const sbs = barStarts[i];
    const sbe = barStarts[i + 1];
    const dur = sbe - sbs;
    const inBar = onsets.filter((o) => o >= sbs && o < sbe);
    out.push(detectHitsPerBeat(inBar, dur, bpm));
  }
  return out;
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
 * Compatibility score (0..1) for a src rhythmic subdivision under a
 * given tgt meter. Higher = the subdivision's native pulse aligns with
 * the tgt grid; lower = re-quantize recommended.
 *
 * Rationale:
 *   quarter / eighth / 16th → always fit (they subdivide any pulse)
 *   triplet → ok in plain /4 (feels like "swing"); poor in /8 and odd meters
 *   sparse → treat as identity (nothing to quantize)
 */
function compatibilityScore(hpb, tgtMeter) {
  if (hpb === 0) return 1;          // sparse/silent — leave alone
  const [tn, td] = tgtMeter;
  if (hpb === 1) return 1;          // quarter fits everything
  if (hpb === 2) return 1;          // eighth fits everything
  if (hpb === 4) return td === 8 ? 0.6 : 1;   // 16ths in /8 get busy
  if (hpb === 3) {
    if (td === 8) return 0.2;       // triplets in /8 don't groove
    if (tn === 5 || tn === 7) return 0.3;
    return 0.7;                     // triplets in /4 → swing feel
  }
  return 0.5;
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
 * Client-side meter-denominator inference from substem onset patterns.
 *
 * Used when the backend detector hasn't supplied `meter_denominator`.
 * Numerator-based fast path covers 95% of cases:
 *   3, 4, 5  → /4   (slow waltz, 4/4, 5/4 prog)
 *   7, 9, 11 → /8   (asymmetric meters are almost always /8)
 *   6        → ambiguous (3/4 simple-triple vs 6/8 compound-duple).
 *              Disambiguate via kick interval pattern: 6/8 typically has
 *              kicks every 3 eighths (downbeats of two 3-groups), 3/4 has
 *              kicks every 2 eighths (every quarter beat).
 */
export function inferMeterDenominator(numerator, onsetsBySubstem, bpm) {
  if (numerator === 7 || numerator === 9 || numerator === 11) return 8;
  if (numerator !== 6) return 4;
  const kicks = onsetsBySubstem?.kick;
  if (!kicks || kicks.length < 4) return bpm < 80 ? 8 : 4;
  // Median inter-kick interval in src-eighth units (assuming /4 for the calc).
  const eighthSec = (60 / bpm) / 2;
  const intervals = [];
  for (let i = 1; i < kicks.length; i++) intervals.push(kicks[i] - kicks[i - 1]);
  intervals.sort((a, b) => a - b);
  const median = intervals[Math.floor(intervals.length / 2)];
  const eighthsBetweenKicks = median / eighthSec;
  // 6/8 if kicks pulse every ~3 eighths (compound feel), else 3/4.
  if (eighthsBetweenKicks > 2.5 && eighthsBetweenKicks < 3.5) return 8;
  return 4;
}

/**
 * Client-side 5/4 grouping inference (3+2 vs 2+3) from snare onset
 * positions + strengths. The grouping that gets the most snare emphasis
 * on its "second group's downbeat" wins.
 *
 *   3+2: snare lands on beat 4 (start of the "2") → returns '3+2'
 *   2+3: snare lands on beat 3 (start of the "3") → returns '2+3'
 */
export function inferFiveFourGrouping(onsetsBySubstem, strengthsBySubstem, bpm, downbeatOffset = 0) {
  const snares = onsetsBySubstem?.snare || [];
  const strs   = strengthsBySubstem?.snare || [];
  if (snares.length < 4 || bpm <= 0) return '3+2';
  const beatSec = 60 / bpm;
  const barSec = 5 * beatSec;
  const sumByBeat = [0, 0, 0, 0, 0];
  for (let i = 0; i < snares.length; i++) {
    const t = snares[i];
    if (t < downbeatOffset) continue;
    const inBar = ((t - downbeatOffset) % barSec) / beatSec;
    const beat = Math.round(inBar) % 5;
    const w = strs[i] !== undefined ? strs[i] : 1;
    sumByBeat[beat] += w;
  }
  // beat[3] = beat 4 (1-indexed), beat[2] = beat 3.
  return sumByBeat[3] > sumByBeat[2] ? '3+2' : '2+3';
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
  duration, onsets, strengths, srcMeter, tgtMeter, bpm,
  downbeatOffset = 0, cropStart = 0, barStarts,
}) {
  const HIT_PRE  = 0.005;
  const FADE_IN  = 0.001;
  const FADE_OUT = 0.015;
  // Accent-aware selection: balance proximity (closer = better) against
  // onset strength (stronger = better) when picking which src hit
  // populates each tgt subdivision. Without strengths, falls back to
  // pure proximity.
  const PROXIMITY_PENALTY = 5; // onset-strength units per second of distance
  const haveStrengths = Array.isArray(strengths) && strengths.length === (onsets?.length || -1);

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

    // Keep onsets + strengths as parallel arrays so accent weighting can
    // look up the per-onset strength. Sort by time.
    const idxAll = (Array.isArray(onsets) ? onsets : []).map((_, i) => i);
    const inBarIdx = idxAll
      .filter((i) => onsets[i] >= sbs && onsets[i] < sbe)
      .sort((a, b) => onsets[a] - onsets[b]);

    if (inBarIdx.length === 0) {
      dstCursor += tgtBarLen;
      continue;
    }

    for (let k = 0; k < tgtSubdivPerBar; k++) {
      // Map this tgt subdivision back into src time at proportional
      // position within the src bar.
      const relPos = k / tgtSubdivPerBar;
      const srcTimeApprox = sbs + relPos * srcBarLen;

      // Score each in-bar onset: HIGHER strength is better, FARTHER is
      // worse. The proximity penalty is in onset-strength units per
      // second, so 5 means a 200ms drift is worth ~1 strength unit.
      // Without strengths, every onset gets strength 1 → pure proximity.
      let bestI = inBarIdx[0];
      let bestScore = -Infinity;
      for (const i of inBarIdx) {
        const dist = Math.abs(onsets[i] - srcTimeApprox);
        const str = haveStrengths ? strengths[i] : 1;
        const score = str - dist * PROXIMITY_PENALTY;
        if (score > bestScore) { bestScore = score; bestI = i; }
      }
      const best = onsets[bestI];
      const localPos = inBarIdx.indexOf(bestI);
      const nextOnset = (localPos + 1 < inBarIdx.length)
        ? onsets[inBarIdx[localPos + 1]]
        : sbe;
      const winSrc = Math.min(nextOnset - best + HIT_PRE, tgtSubdivLen + HIT_PRE);
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
  duration, onsets, strengths, srcMeter, tgtMeter, bpm, substemName,
  downbeatOffset = 0, cropStart = 0, barStarts,
  srcGrouping = null, tgtGrouping = null,
}) {
  if (!(duration > 0)) return [];
  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }

  const HIT_PRE  = 0.005;
  const HIT_MAX  = 0.5;
  const FADE_IN  = 0.002;
  const FADE_OUT = 0.020;

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  const rule = ruleFor(srcMeter, tgtMeter, srcGrouping, tgtGrouping);
  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);
  const haveStrengths = Array.isArray(strengths) && strengths.length === (onsets?.length || -1);

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

  // Per-bar subdivision classification. Each bar gets its own hpb so a
  // song that groove-rides 16ths with occasional triplet fills can route
  // the fills through re-quantize while the groove bars stay on the
  // snap path.
  const perBarHpb = detectHitsPerBeatPerBar(onsets, starts, bpm);

  // Track per-bar dst window so the implied-snare pass below knows where
  // to look for "uncovered" snareSnap positions.
  const barDstWindows = [];           // { dstBarStart, sbs, sbe }

  for (let b = 0; b < starts.length - 1; b++) {
    const sbs = starts[b];
    const sbe = starts[b + 1];
    const srcBarLen = sbe - sbs;
    if (srcBarLen <= EPS) continue;
    const srcEighthLen = srcBarLen / srcEighths;
    const tgtEighthLen = srcEighthLen;                       // rate 1
    const tgtBarLen = tgtEighthLen * tgtEighths;
    barDstWindows.push({ dstBarStart: dstCursor, sbs, sbe, tgtEighthLen });

    // Parallel idx arrays so we can preserve onset ↔ strength alignment.
    const idxAll = (Array.isArray(onsets) ? onsets : []).map((_, i) => i);
    const inBarIdx = idxAll
      .filter((i) => onsets[i] >= sbs && onsets[i] < sbe)
      .sort((a, b) => onsets[a] - onsets[b]);

    // Per-bar compatibility check. If this bar's local subdivision won't
    // groove in the target meter (e.g. triplets → 7/8), re-quantize just
    // this bar. Other bars continue on the snap path.
    const barHpb = perBarHpb[b] || 0;
    const score = compatibilityScore(barHpb, tgtMeter);
    if (score < 0.5 && inBarIdx.length >= 3) {
      // Local re-quantize: treat this one bar like buildTripletReQuantize-
      // Schedule, but bounded to [sbs, sbe). Rest of the track stays snap.
      const subdivPerEighth = chooseTargetSubdivPerEighth(bpm);
      const tgtSubdivPerBar = tgtEighths * subdivPerEighth;
      const tgtSubdivLen = tgtBarLen / tgtSubdivPerBar;
      const PROXIMITY_PENALTY = 5;
      for (let k = 0; k < tgtSubdivPerBar; k++) {
        const relPos = k / tgtSubdivPerBar;
        const srcTimeApprox = sbs + relPos * srcBarLen;
        let bestI = inBarIdx[0], bestScore = -Infinity;
        for (const i of inBarIdx) {
          const dist = Math.abs(onsets[i] - srcTimeApprox);
          const str = haveStrengths ? strengths[i] : 1;
          const s = str - dist * PROXIMITY_PENALTY;
          if (s > bestScore) { bestScore = s; bestI = i; }
        }
        const best = onsets[bestI];
        const localPos = inBarIdx.indexOf(bestI);
        const nextOnset = (localPos + 1 < inBarIdx.length)
          ? onsets[inBarIdx[localPos + 1]] : sbe;
        const winSrc = Math.min(nextOnset - best + HIT_PRE, tgtSubdivLen + HIT_PRE);
        const srcStart = cropStart + Math.max(sbs, best - HIT_PRE);
        const srcEnd = srcStart + winSrc;
        const dstHitStart = dstCursor + k * tgtSubdivLen;
        segs.push({
          srcStart, srcEnd,
          dstStart: dstHitStart, dstEnd: dstHitStart + winSrc,
          rate: 1, fadeIn: 0.001, fadeOut: 0.015, kind: 'tripletQuant',
        });
      }
      console.log(`[virtualTrackEdit] ${substemName} bar ${b + 1}: hpb=${barHpb} score=${score.toFixed(2)} → local re-quantize`);
      dstCursor += tgtBarLen;
      continue;
    }

    for (let li = 0; li < inBarIdx.length; li++) {
      const i = inBarIdx[li];
      const onset = onsets[i];
      const srcEighthFloat = (onset - sbs) / srcEighthLen;
      const tgts = rule
        ? placementsForSrcEighth(srcEighthFloat, rule, substemName)
        : (() => {
            const t = srcEighthFloat * tgtEighths / srcEighths;
            return t < tgtEighths - EPS ? [t] : [];
          })();
      if (tgts.length === 0) continue;

      const next = (li + 1 < inBarIdx.length) ? onsets[inBarIdx[li + 1]] : sbe;
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

  // ── Implied snare insertion ────────────────────────────────────────
  // Some extending-meter rules (3/4→4/4, 6/8→4/4) define
  // `impliedSnareTgtEighths` — positions in the new bar that should have
  // a snare hit even when the source had no snare onset in the duplicated
  // src range. Without this, the new beat 4 of the extended bar feels
  // "empty" because there's no source material to copy from.
  //
  // Strategy: per bar, for each implied position, check whether any
  // already-placed snare segment lands within ±¼ tgt eighth of it. If
  // not, find the strongest in-bar snare onset elsewhere and emit a
  // duplicate at the implied position. The duplicated audio keeps the
  // original snare's timbre/dynamics.
  if (substemName === 'snare' && rule?.impliedSnareTgtEighths?.length) {
    for (let b = 0; b < barDstWindows.length; b++) {
      const { dstBarStart, sbs, sbe, tgtEighthLen } = barDstWindows[b];
      const idxAll = (Array.isArray(onsets) ? onsets : []).map((_, i) => i);
      const inBarIdx = idxAll
        .filter((i) => onsets[i] >= sbs && onsets[i] < sbe)
        .sort((a, b) => onsets[a] - onsets[b]);
      if (inBarIdx.length === 0) continue;

      // Pick the strongest in-bar snare as the donor for any implied hits.
      let donorI = inBarIdx[0];
      let donorScore = haveStrengths ? strengths[donorI] : 1;
      for (const i of inBarIdx) {
        const s = haveStrengths ? strengths[i] : 1;
        if (s > donorScore) { donorScore = s; donorI = i; }
      }
      const donor = onsets[donorI];
      const donorPos = inBarIdx.indexOf(donorI);
      const donorNext = (donorPos + 1 < inBarIdx.length) ? onsets[inBarIdx[donorPos + 1]] : sbe;
      const donorWin = Math.min(donorNext - donor + HIT_PRE, HIT_MAX);
      const donorSrcStart = cropStart + Math.max(sbs, donor - HIT_PRE);
      const donorSrcEnd = donorSrcStart + donorWin;

      for (const impliedTgtE of rule.impliedSnareTgtEighths) {
        const desiredDst = dstBarStart + impliedTgtE * tgtEighthLen - HIT_PRE;
        const tolerance = tgtEighthLen * 0.25;
        const alreadyCovered = segs.some((s) =>
          Math.abs(s.dstStart - desiredDst) < tolerance);
        if (alreadyCovered) continue;
        segs.push({
          srcStart: donorSrcStart, srcEnd: donorSrcEnd,
          dstStart: Math.max(dstBarStart, desiredDst),
          dstEnd: Math.max(dstBarStart, desiredDst) + donorWin,
          rate: 1, fadeIn: FADE_IN, fadeOut: FADE_OUT, kind: 'impliedSnare',
        });
      }
    }
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
  const subStrengths = meta.drumSubstemOnsetStrengths || {};
  const duration = track.duration || track.length || 0;
  if (duration <= 0) return null;
  const cropStart = track.cropStart || 0;

  const srcBpm = meta.detectedBpm || project.bpm || 120;
  const srcN = meta.detectedMeter || project.beatsPerBar || 4;
  // Source meter denominator: prefer explicit detection, else infer
  // client-side from substem onset patterns (handles 6/8 vs 3/4 etc.).
  const srcD = meta.detectedMeterDenominator
    || inferMeterDenominator(srcN, subOnsets, srcBpm);
  const srcMeter = [srcN, srcD];
  const tgtMeter = [project.beatsPerBar || 4, project.meterDenominator || 4];
  const downbeatOffset = typeof meta.downbeatOffset === 'number' ? meta.downbeatOffset : 0;
  const identity = srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1];

  // 5/4 grouping (3+2 vs 2+3): used by the implied-snare pass to pick
  // which beat to imply a backbeat at when extending another meter to 5/4.
  // Inferred from snare emphasis when not explicitly set.
  const fiveFourGrouping = (srcN === 5 || tgtMeter[0] === 5)
    ? (meta.detectedGrouping || inferFiveFourGrouping(subOnsets, subStrengths, srcBpm, downbeatOffset))
    : null;

  // Fall back to project.barStarts when the stem metadata didn't inherit
  // the master's analyzed rhythm.
  const resolvedBarStarts = (Array.isArray(meta.barStarts) && meta.barStarts.length >= 2)
    ? meta.barStarts
    : ((Array.isArray(project.barStarts) && project.barStarts.length >= 2) ? project.barStarts : null);
  const resolvedDownbeat = typeof meta.downbeatOffset === 'number'
    ? meta.downbeatOffset
    : (typeof project.downbeatOffset === 'number' ? project.downbeatOffset : downbeatOffset);
  const tgtFitsTriplet = patternFitsTargetMeter(3, tgtMeter);
  const anyTriplet = ['kick', 'snare', 'toms', 'hh', 'ride', 'crash']
    .some((n) => Array.isArray(subOnsets[n]) && detectHitsPerBeat(subOnsets[n], duration, srcBpm) === 3);
  // Triplet path only fires when a rule is defined for this meter pair.
  // Other meter pairs fall through to the existing METER_RULES /
  // buildPercussiveSubstemSchedule path — which works fine for
  // non-triplet sources.
  const hasTripletRule = !!tripletRuleFor(srcMeter, tgtMeter);

  const out = {};
  for (const [name, audioUrl] of Object.entries(subUrls)) {
    if (!audioUrl) continue;
    let schedule;
    let kind;
    const onsetsHere = subOnsets[name];
    const hasOnsets = Array.isArray(onsetsHere) && onsetsHere.length > 0;
    if (identity) {
      schedule = identitySchedule(duration, cropStart);
      kind = 'identity';
    } else if (hasOnsets && !tgtFitsTriplet && anyTriplet && hasTripletRule) {
      // Triplet source + odd target (7/8 etc.) + rule defined in
      // TRIPLET_METER_RULES for this meter pair. User's manual method —
      // hi-hat = whole-bar WSOLA stretch, others = per-onset stretch
      // at expected tgt 16th positions.
      schedule = buildOnsetStretchSchedule({
        duration, onsets: onsetsHere,
        srcMeter, tgtMeter, bpm: srcBpm, substemName: name,
        downbeatOffset: resolvedDownbeat, cropStart, barStarts: resolvedBarStarts,
      });
      kind = 'onsetStretch';
    } else if (PERCUSSIVE_SUBSTEMS.has(name) && hasOnsets) {
      schedule = buildPercussiveSubstemSchedule({
        duration, onsets: onsetsHere, strengths: subStrengths[name],
        srcMeter, tgtMeter, bpm: srcBpm, substemName: name,
        downbeatOffset: resolvedDownbeat, cropStart, barStarts: resolvedBarStarts,
        srcGrouping: meta.detectedGrouping || fiveFourGrouping || null,
        tgtGrouping: project.grouping || null,
      });
      kind = 'snap';
    } else {
      schedule = buildMeterSchedule({
        duration, srcMeter, tgtMeter, bpm: srcBpm,
        downbeatOffset: resolvedDownbeat, cropStart, barStarts: resolvedBarStarts,
        tempoMap: meta.tempoMap || project.tempoMap || null,
        tgtTempoMap: project.tempoMap || null,
        srcGrouping: meta.detectedGrouping || fiveFourGrouping || null,
        tgtGrouping: project.grouping || null,
      });
      kind = 'rearrange';
    }
    out[name] = { audioUrl, schedule, kind, ...(fiveFourGrouping ? { fiveFourGrouping } : {}) };
  }
  if (Object.keys(out).length === 0) return null;
  // Visibility: log the per-substem schedule breakdown whenever the drum
  // substem path is taken, so it's obvious in the console which branch
  // fired (per-substem snap vs the bar-rearrange fallback). Kept under
  // the meter-change emoji so it groups with the other meter logs.
  const breakdown = Object.entries(out)
    .map(([name, { schedule, kind }]) => `${name}:${kind}(${schedule.length})`)
    .join(' ');
  const tag = identity ? 'identity' : `${srcMeter[0]}/${srcMeter[1]} → ${tgtMeter[0]}/${tgtMeter[1]}`;
  console.log(`🥁 [meter] drums per-substem schedule (${tag}): ${breakdown}`);
  return out;
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
