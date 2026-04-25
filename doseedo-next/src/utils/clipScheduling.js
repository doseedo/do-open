/**
 * Pure per-clip playback math — no Web Audio, no I/O. Feed in one Clip
 * (Logic region descriptor) plus playhead state, and get back a
 * `ClipPlayPlan` ready for the scheduler to wire into Web Audio nodes,
 * or `null` if the clip isn't playable (no buffer cached, already past).
 *
 * Keeping the decision logic pure means the scheduler test suite can
 * verify the buffer-time math (`bufOffset`, `bufDur` under playbackRate),
 * slice expansion, and fade-curve plans without spinning up an
 * AudioContext.
 *
 * Clip shape (per `docs/session_json_schema.md` and `logicx_format.md`):
 *   auflGid, subIndex,
 *   startPosition, duration,                // timeline seconds
 *   sourceDuration, sourceOffsetSamples,    // full source length + slice origin
 *   sourceOffsetSeconds,                    // offset in the chosen-source timebase
 *   url,                                    // bit-exact variant wav
 *   rootUrl?, rootAuflGid?, stretchRatio?,  // informational for 'variant' mode;
 *                                           // directive for 'root_stretch' mode
 *   flexDeltaSeconds?,                      // Logic flex onset shift
 *   playbackPath?: 'variant' | 'root_stretch',
 *   fadeIn?: { ms: number, curve?: string },
 *   fadeOut?: { ms: number, curve?: string },
 *   sliceMarkers?: Array<{ slice_index, source_sample_offset,
 *                          position_ticks, position_s,
 *                          slice_type }>,
 *
 * Playback mode selection:
 *   1. `playbackPath === 'variant'`     — variant wav at rate=1. REQUIRED when
 *      Logic destructively rendered the stretch (SPEED mode): the root
 *      contains unrelated audio at the given offset, so only the variant
 *      is faithful. This is Project 1's case.
 *   2. `playbackPath === 'root_stretch'` — root wav at `playbackRate =
 *      1 / stretchRatio` (per `docs/logicx_playback_model.md` §1.1). Only
 *      valid when Logic stored the stretch non-destructively.
 *   3. No `playbackPath` (legacy) — heuristic: root+stretch when the root
 *      buffer is cached AND stretchRatio is usable; else variant.
 *
 * Slicing:
 *   When `clip.sliceMarkers` is present, each marker is an independently-
 *   scheduled source: a slice of the chosen buffer at `source_sample_offset
 *   / sampleRate`, placed on the timeline at `position_s` (absolute), and
 *   running until the next slice's `position_s` or the clip's end. Slices
 *   always play at rate=1 (slicing is just re-triggering — no time-stretch
 *   in the slice itself).
 *
 * Buffer-time math (`source.start(when, bufOffset, bufDur)`):
 *   - `bufOffset` is in buffer-seconds (rate=1 of the chosen buffer).
 *   - `bufDur` is in buffer-seconds; with playbackRate r, wall-clock
 *     duration = bufDur / r.
 *   - To get wall-clock duration D at rate r: bufDur = D × r.
 */

/**
 * @typedef {Object} SourceSpec
 * @property {string} playUrl
 * @property {number} playbackRate
 * @property {boolean} useRoot
 * @property {number} when                AudioContext time to start
 * @property {number} bufOffset           offset into buffer (seconds, rate=1 space)
 * @property {number} bufDur              duration to consume (buffer-seconds)
 * @property {number} bufferDuration      total buffer duration (diagnostic)
 * @property {number} clipStart           absolute timeline start (post flex shift)
 * @property {number} clipEnd             absolute timeline end
 * @property {boolean} [isSlice]          true when emitted from sliceMarkers
 */

/**
 * @typedef {Object} ClipPlayPlan
 * @property {SourceSpec[]} sources       one entry, or N for sliced clips
 * @property {number} fadeInSec           clip-level fade-in (seconds, 0 = none)
 * @property {number} fadeOutSec          clip-level fade-out (seconds, 0 = none)
 * @property {number} clipStart           timeline start (post-flex)
 * @property {number} clipEnd             timeline end
 */

const DEFAULT_SAMPLE_RATE = 44100;

/**
 * @param {object} clip
 * @param {number} currentPlayheadTime
 * @param {number} schedulingStartTime
 * @param {(url: string) => ({duration: number, sampleRate?: number} | null | undefined)} bufferLookup
 * @param {object} [opts]
 * @param {number} [opts.sampleRate]   session sample rate, for sliceMarker.source_sample_offset → seconds
 * @returns {ClipPlayPlan | null}
 */
export function computeClipPlayback(
  clip,
  currentPlayheadTime,
  schedulingStartTime,
  bufferLookup,
  opts = {},
) {
  if (!clip || typeof clip !== 'object') return null;

  // ── 1. Pick source (variant vs root) ─────────────────────────────────
  const stretchRatio = Number(clip.stretchRatio);
  const hasStretch = Number.isFinite(stretchRatio) && stretchRatio > 0 && stretchRatio !== 1;
  const rootBuf = clip.rootUrl ? bufferLookup(clip.rootUrl) : null;
  const variantBuf = clip.url ? bufferLookup(clip.url) : null;

  let useRoot;
  if (clip.playbackPath === 'variant') {
    useRoot = false;
  } else if (clip.playbackPath === 'root_stretch') {
    useRoot = !!(rootBuf && hasStretch);
    if (!useRoot) return null;   // can't fall back: offset timebase differs
  } else {
    useRoot = !!(rootBuf && hasStretch);
  }

  const playUrl = useRoot ? clip.rootUrl : clip.url;
  const buf = useRoot ? rootBuf : variantBuf;
  if (!buf || !playUrl) return null;

  // playbackRate per docs/logicx_playback_model.md §1.1: when playing the
  // root through a stretch, the rate slows the buffer down to fit the
  // longer destination region (or speeds it up for ratios < 1).
  const playbackRate = useRoot ? (1 / stretchRatio) : 1;

  // ── 2. Timeline window ───────────────────────────────────────────────
  const clipStart = (Number(clip.startPosition) || 0)
                  + (Number(clip.flexDeltaSeconds) || 0);
  const clipDuration = Number(clip.duration) || 0;
  if (clipDuration <= 0) return null;
  const clipEnd = clipStart + clipDuration;
  if (currentPlayheadTime >= clipEnd) return null;

  // ── 3. Fades ─────────────────────────────────────────────────────────
  // Logic emits fadeIn/fadeOut as { ms, curve }. Curve string is
  // documented as "Logic-internal" in playback-model §10 — we use a
  // linear ramp until the precise mapping is reverse-engineered. That's
  // imperceptible for short fades (<50ms typical) and audibly close for
  // longer ones.
  const fadeInSec  = _fadeSeconds(clip.fadeIn);
  const fadeOutSec = _fadeSeconds(clip.fadeOut);

  // ── 4. Source spec(s) ────────────────────────────────────────────────
  const bufferDuration = Number(buf.duration) || 0;
  const sampleRate = Number(opts.sampleRate)
                  || Number(buf.sampleRate)
                  || DEFAULT_SAMPLE_RATE;

  let sources;
  if (Array.isArray(clip.sliceMarkers) && clip.sliceMarkers.length > 0) {
    sources = _expandSlices(
      clip.sliceMarkers,
      { clipStart, clipEnd, currentPlayheadTime, schedulingStartTime,
        bufferDuration, playUrl, useRoot, sampleRate },
    );
  } else {
    const single = _singleSourceSpec({
      clipStart, clipEnd, currentPlayheadTime, schedulingStartTime,
      sourceOffsetSeconds: Number(clip.sourceOffsetSeconds) || 0,
      playbackRate, bufferDuration, playUrl, useRoot,
    });
    sources = single ? [single] : [];
  }
  if (sources.length === 0) return null;

  return {
    sources,
    fadeInSec,
    fadeOutSec,
    clipStart,
    clipEnd,
  };
}

function _fadeSeconds(fade) {
  if (!fade || typeof fade !== 'object') return 0;
  const ms = Number(fade.ms);
  if (!Number.isFinite(ms) || ms <= 0) return 0;
  return ms / 1000;
}

function _singleSourceSpec({
  clipStart, clipEnd, currentPlayheadTime, schedulingStartTime,
  sourceOffsetSeconds, playbackRate, bufferDuration, playUrl, useRoot,
}) {
  const clipDuration = clipEnd - clipStart;
  let when, bufOffset, bufDur;
  if (currentPlayheadTime < clipStart) {
    when = schedulingStartTime + (clipStart - currentPlayheadTime);
    bufOffset = sourceOffsetSeconds;
    bufDur = clipDuration * playbackRate;
  } else {
    when = schedulingStartTime;
    const timeIntoClip = currentPlayheadTime - clipStart;
    bufOffset = sourceOffsetSeconds + timeIntoClip * playbackRate;
    bufDur = (clipDuration - timeIntoClip) * playbackRate;
  }
  if (bufOffset >= bufferDuration) return null;
  bufDur = Math.min(bufDur, bufferDuration - bufOffset);
  if (bufDur <= 0) return null;
  return {
    playUrl, playbackRate, useRoot,
    when, bufOffset, bufDur, bufferDuration,
    clipStart, clipEnd,
  };
}

/**
 * Expand sliceMarkers into N source specs. Each slice plays at rate=1 from
 * its source_sample_offset, lasting until the next slice's position_s (or
 * the clip's end). Slices already past the playhead are dropped; mid-slice
 * resume is handled the same way as a single clip mid-resume.
 */
function _expandSlices(markers, ctx) {
  const {
    clipStart, clipEnd, currentPlayheadTime, schedulingStartTime,
    bufferDuration, playUrl, useRoot, sampleRate,
  } = ctx;
  // Normalize: ensure ordered by position_s, drop markers outside the clip.
  const sorted = markers
    .filter((m) => m && Number.isFinite(Number(m.position_s)))
    .map((m) => ({
      sourceOffsetSec: (Number(m.source_sample_offset) || 0) / sampleRate,
      pos: Number(m.position_s),
    }))
    .filter((m) => m.pos >= clipStart - 1e-6 && m.pos < clipEnd + 1e-6)
    .sort((a, b) => a.pos - b.pos);
  if (sorted.length === 0) return [];

  const out = [];
  for (let i = 0; i < sorted.length; i++) {
    const m = sorted[i];
    const sliceStart = Math.max(m.pos, clipStart);
    const sliceEnd   = i + 1 < sorted.length ? sorted[i + 1].pos : clipEnd;
    if (sliceEnd <= sliceStart) continue;
    if (currentPlayheadTime >= sliceEnd) continue;

    let when, bufOffset, bufDur;
    if (currentPlayheadTime < sliceStart) {
      when = schedulingStartTime + (sliceStart - currentPlayheadTime);
      bufOffset = m.sourceOffsetSec;
      bufDur = sliceEnd - sliceStart;
    } else {
      // Mid-slice resume — slices play at rate 1, so wall-clock and
      // buffer-clock advance equally.
      when = schedulingStartTime;
      const timeIntoSlice = currentPlayheadTime - sliceStart;
      bufOffset = m.sourceOffsetSec + timeIntoSlice;
      bufDur = (sliceEnd - sliceStart) - timeIntoSlice;
    }
    if (bufOffset >= bufferDuration) continue;
    bufDur = Math.min(bufDur, bufferDuration - bufOffset);
    if (bufDur <= 0) continue;

    out.push({
      playUrl,
      playbackRate: 1,
      useRoot,
      when,
      bufOffset,
      bufDur,
      bufferDuration,
      clipStart,
      clipEnd,
      isSlice: true,
    });
  }
  return out;
}
