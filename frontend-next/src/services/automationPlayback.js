/**
 * Automation lane playback — schedule a sorted list of breakpoints onto a
 * Web Audio AudioParam (e.g. trackGain.gain or trackPan.pan).
 *
 * Lane shape (normalized): Array<{ time: seconds, value: number, curve?:
 * 'linear' | 'hold' }>. The AutomationWindow editor writes points with a
 * `volume` field; Logic's desktop emit uses `value`. Both are accepted —
 * `_pointValue` reads `value ?? volume ?? gain ?? 0`.
 *
 * Anchor model: at `play()` time, useAudioPlayback captures
 *   schedulingStartTime = audioContext.currentTime
 *   anchorTimelineTime  = currentPlayheadTime  (transport seconds)
 * For a point at timeline time `pt.time`:
 *   ctxTime = schedulingStartTime + (pt.time - anchorTimelineTime)
 * Points before the playhead don't schedule events; instead we set the
 * AudioParam's initial value to the linearly-interpolated lane value
 * AT the playhead so playback starts on-curve.
 *
 * Curves:
 *   'hold'  → setValueAtTime (step)
 *   'linear' (default) → linearRampToValueAtTime
 *   The exact dB-vs-linear mapping for Logic's encoded curve byte is
 *   documented as Logic-internal (gap #7 in playback-model). We treat
 *   any non-'hold' curve as linear; that's perceptually close for short
 *   ramps and a starting point for finer mapping later.
 */

/** Pull a numeric value off a point in either schema. */
function _pointValue(p) {
  if (!p) return 0;
  const v = p.value ?? p.volume ?? p.gain;
  return Number.isFinite(Number(v)) ? Number(v) : 0;
}

function _pointTime(p) {
  return Number(p?.time) || 0;
}

function _pointCurve(p) {
  return p?.curve === 'hold' ? 'hold' : 'linear';
}

/**
 * Linearly interpolate the lane value at the given timeline time.
 * Outside the lane, clamps to the first / last value (Logic's behavior:
 * the lane holds its endpoints).
 */
export function laneValueAt(lane, timelineTime) {
  if (!Array.isArray(lane) || lane.length === 0) return null;
  const sorted = lane.slice().sort((a, b) => _pointTime(a) - _pointTime(b));
  if (timelineTime <= _pointTime(sorted[0])) return _pointValue(sorted[0]);
  if (timelineTime >= _pointTime(sorted[sorted.length - 1])) {
    return _pointValue(sorted[sorted.length - 1]);
  }
  for (let i = 0; i < sorted.length - 1; i++) {
    const a = sorted[i];
    const b = sorted[i + 1];
    const at = _pointTime(a);
    const bt = _pointTime(b);
    if (timelineTime >= at && timelineTime <= bt) {
      if (_pointCurve(b) === 'hold') return _pointValue(a);
      const span = bt - at;
      if (span <= 0) return _pointValue(b);
      const t = (timelineTime - at) / span;
      return _pointValue(a) + (_pointValue(b) - _pointValue(a)) * t;
    }
  }
  return _pointValue(sorted[sorted.length - 1]);
}

/**
 * Schedule a lane onto an AudioParam.
 *
 * @param {AudioParam} param
 * @param {Array} lane
 * @param {object} ctx
 * @param {number} ctx.audioContextCurrentTime  audioContext.currentTime now
 * @param {number} ctx.schedulingStartTime      ctx time at play start
 * @param {number} ctx.anchorTimelineTime       transport time at play start
 * @param {number} [ctx.fallbackValue=1]        used when lane is empty / all points past anchor
 *
 * @returns {{ scheduledCount: number, paramAtAnchor: number }}
 */
export function scheduleAutomation(param, lane, ctx) {
  if (!param || typeof param.setValueAtTime !== 'function') {
    return { scheduledCount: 0, paramAtAnchor: 1 };
  }
  const {
    audioContextCurrentTime, schedulingStartTime,
    anchorTimelineTime, fallbackValue = 1,
  } = ctx;

  const safeNow = Math.max(audioContextCurrentTime, schedulingStartTime);
  // Wipe any prior automation events from this point forward. Without
  // this, replaying from a different transport position layers stale
  // ramps on top of new ones.
  try { param.cancelScheduledValues(safeNow); } catch (_) {}

  const initial = (Array.isArray(lane) && lane.length > 0)
    ? laneValueAt(lane, anchorTimelineTime)
    : fallbackValue;

  // Anchor the param exactly at play-start with the value the lane
  // implies at this transport position. Keeps gain/pan continuous
  // even if the user seeks into the middle of a ramp.
  param.setValueAtTime(initial, schedulingStartTime);

  if (!Array.isArray(lane) || lane.length === 0) {
    return { scheduledCount: 0, paramAtAnchor: initial };
  }

  const sorted = lane.slice().sort((a, b) => _pointTime(a) - _pointTime(b));
  let scheduled = 0;
  for (const p of sorted) {
    const ptTime = _pointTime(p);
    if (ptTime <= anchorTimelineTime) continue;   // already past — handled by anchor
    const ctxTime = schedulingStartTime + (ptTime - anchorTimelineTime);
    const val = _pointValue(p);
    try {
      if (_pointCurve(p) === 'hold') {
        param.setValueAtTime(val, ctxTime);
      } else {
        param.linearRampToValueAtTime(val, ctxTime);
      }
      scheduled++;
    } catch (err) {
      // setValueAtTime throws if ctxTime < currentTime; treat as no-op.
    }
  }
  return { scheduledCount: scheduled, paramAtAnchor: initial };
}

/**
 * Clear any pending automation on a param and pin it to a fallback value.
 * Used on pause and seek. Use a small delay (currentTime + 1ms) to avoid
 * Web Audio's "no scheduling in the past" exception.
 */
export function clearAutomation(param, audioContextCurrentTime, fallbackValue = 1) {
  if (!param || typeof param.cancelScheduledValues !== 'function') return;
  try { param.cancelScheduledValues(audioContextCurrentTime); } catch (_) {}
  try { param.setValueAtTime(fallbackValue, audioContextCurrentTime + 0.001); } catch (_) {}
}
