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
 * Drum substem support.
 *
 * When a drum stem track has metadata.drumSubstems (the kick/snare/toms/
 * hh/ride/crash WAV URLs from MDX23C-DrumSep on the backend) and
 * metadata.drumSubstemOnsets (librosa onset times per substem), we can
 * rearrange each substem independently for a meter change:
 *   - PERCUSSIVE substems (kick/snare/toms) use buildPercussiveSubstemSchedule
 *     below: snap each onset to the corresponding target-meter eighth grid
 *     position, drop hits in dropped eighths, duplicate hits for added
 *     eighths. Each hit becomes a tiny rate-1 segment with a 20ms tail
 *     fade. Between hits the substem is naturally silent in the source,
 *     so silence in the dst is correct (the OTHER substems carry audio
 *     during those moments, summed at the per-track gain).
 *   - SUSTAIN substems (hh/ride/crash) keep the bar-rearrange path so
 *     wash/decay carries through naturally.
 *
 * The engine sums all substems through one shared per-track gain, so
 * solo/mute on the parent drum stem track keeps working.
 * ------------------------------------------------------------------ */

const PERCUSSIVE_SUBSTEMS = new Set(['kick', 'snare', 'toms']);

/**
 * Per-substem beat-snap schedule for a percussive drum substem.
 *
 * Per src bar:
 *   1. Find onsets falling in the bar.
 *   2. For each onset, determine its src eighth index e_src.
 *   3. Use the meter-rule forward map to find every tgt eighth e_t that
 *      receives e_src (a list — duplicate cases produce multiple targets).
 *   4. Emit a tiny rate-1 segment per (onset, tgt eighth) at the snapped
 *      tgt eighth boundary, preserving the onset's sub-eighth offset
 *      (groove). Hit window = [onset-5ms, next_onset_in_bar] capped at 0.5s.
 *   5. Tail fade 20ms to keep cymbal-style decay sounding natural even
 *      when the next snapped hit overlaps it.
 *
 * Segments may overlap in dst (a duplicated hit fires multiple times,
 * one hit's tail can extend past the next hit's start). The engine
 * schedules each as its own AudioBufferSourceNode → segGain → trackGain
 * so they sum naturally.
 */
function buildPercussiveSubstemSchedule({
  duration, onsets, srcMeter, tgtMeter, bpm,
  downbeatOffset = 0, cropStart = 0, barStarts,
}) {
  if (!(duration > 0)) return [];
  if (srcMeter[0] === tgtMeter[0] && srcMeter[1] === tgtMeter[1]) {
    return identitySchedule(duration, cropStart);
  }
  const HIT_PRE  = 0.005;   // 5ms grab before the onset for transient
  const HIT_MAX  = 0.5;     // cap per-hit tail (cymbal-friendly)
  const FADE_IN  = 0.002;
  const FADE_OUT = 0.020;

  const starts = (barStarts && barStarts.length >= 2)
    ? barStarts
    : synthBarStarts(duration, bpm, srcMeter, downbeatOffset);

  const [sn, sd] = srcMeter;
  const [tn, td] = tgtMeter;
  const srcEighths = sn * (sd === 4 ? 2 : 1);
  const tgtEighths = tn * (td === 4 ? 2 : 1);
  const diff = tgtEighths - srcEighths;

  // Forward map: src eighth e_src → list of tgt eighths receiving it.
  // Default: e_t < srcEighths → e_s = e_t. For e_t >= srcEighths (only
  // possible when tgt > src), e_s = e_t - diff (duplicate-tail rule).
  // Eighths in [tgtEighths, srcEighths) when src > tgt have no entry →
  // their hits are dropped.
  const fwdMap = Array.from({ length: srcEighths }, () => []);
  for (let e_t = 0; e_t < tgtEighths; e_t++) {
    const e_s = (e_t < srcEighths) ? e_t : (e_t - diff);
    if (e_s >= 0 && e_s < srcEighths) fwdMap[e_s].push(e_t);
  }

  const segs = [];
  let dstCursor = 0;

  // Pre-roll: copy through as one contiguous slice (drum substem during
  // pickup is usually below noise floor; if there's a pickup hit, we'd
  // rather hear it than chop it).
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
    const tgtBarLen = srcEighthLen * tgtEighths;     // rate 1 in dst
    const tgtEighthLen = tgtBarLen / tgtEighths;

    const inBar = (Array.isArray(onsets) ? onsets : [])
      .filter((o) => o >= sbs && o < sbe)
      .sort((a, b) => a - b);

    for (let i = 0; i < inBar.length; i++) {
      const onset = inBar[i];
      const e_src = Math.min(srcEighths - 1, Math.max(0, Math.floor((onset - sbs) / srcEighthLen)));
      const tgts = fwdMap[e_src];
      if (!tgts || tgts.length === 0) continue;       // dropped

      const next = (i + 1 < inBar.length) ? inBar[i + 1] : sbe;
      const winSrc = Math.min(next - onset + HIT_PRE, HIT_MAX);
      const srcStart = cropStart + Math.max(sbs, onset - HIT_PRE);
      const srcEnd = srcStart + winSrc;

      // Sub-eighth offset preserved so hits that were slightly behind
      // the beat in src stay slightly behind in dst (groove).
      const subEighthOff = Math.max(0, (onset - HIT_PRE) - (sbs + e_src * srcEighthLen));

      for (const e_t of tgts) {
        const dstHitStart = dstCursor + e_t * tgtEighthLen + subEighthOff;
        const dstHitEnd = dstHitStart + winSrc;
        segs.push({
          srcStart, srcEnd,
          dstStart: dstHitStart, dstEnd: dstHitEnd,
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
 * If onsets are missing for a percussive substem, that substem falls
 * back to the bar-rearrange path so it still plays — slightly less
 * accurate but never silent.
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
        srcMeter, tgtMeter, bpm: srcBpm,
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
