/*
 * tempoMap — helpers for reading a per-bar tempo/meter map.
 *
 * Schema: Array<TempoEntry> where each entry is:
 *   {
 *     bar:       1-indexed bar number this entry activates at
 *     t:         start time in seconds (bar 1 downbeat = 0)
 *     bpm:       local BPM for bars [this entry .. next entry]
 *     meter:     [n, d]   e.g. [4,4] or [7,8]
 *     grouping?: string   e.g. '4+3' / '3+4' for 7/8, '3+2' / '2+3' for 5/4
 *   }
 *
 * Entries are sorted by bar ascending. Entry 0 covers bar 1 onwards; each
 * subsequent entry applies from its `bar` onwards, overriding the previous
 * local tempo/meter. The final entry is open-ended (extends to end of clip).
 *
 * All helpers are pure. No React / AudioContext dependencies.
 */

const EPS = 1e-6;

/** Returns the entry covering bar B (1-indexed), or null if map empty. */
export function entryAtBar(tempoMap, bar) {
  if (!tempoMap || tempoMap.length === 0) return null;
  let result = tempoMap[0];
  for (const e of tempoMap) {
    if (e.bar <= bar) result = e;
    else break;
  }
  return result;
}

/** Returns the entry covering time T in seconds, or null if map empty. */
export function entryAtTime(tempoMap, t) {
  if (!tempoMap || tempoMap.length === 0) return null;
  let result = tempoMap[0];
  for (const e of tempoMap) {
    if ((e.t ?? 0) <= t + EPS) result = e;
    else break;
  }
  return result;
}

/**
 * Given a tempo map and a clip duration, compute every bar-start time in
 * seconds (so the caller can slice audio at real bar boundaries). Mirrors
 * synthBarStarts() but honors in-song tempo/meter changes.
 *
 * Returns [t0, t1, t2, ..., duration] — a sentinel duration entry at the
 * end so per-bar loops can use `starts[i+1]` as the bar end.
 *
 * When a tempo map entry has `t` but no BPM/meter for subsequent bars,
 * we extrapolate the entry's local BPM/meter until the next entry or
 * the clip's end.
 */
export function barStartsFromTempoMap(tempoMap, duration, downbeatOffset = 0) {
  if (!tempoMap || tempoMap.length === 0) return [0, duration];
  const starts = [];
  // Pre-roll: audio before bar 1. Keep as-is in the schedule.
  if (downbeatOffset > EPS) starts.push(0);

  let cursor = Math.max(0, downbeatOffset);
  for (let i = 0; i < tempoMap.length; i++) {
    const entry = tempoMap[i];
    const next = tempoMap[i + 1];
    // Use `t` as the authoritative start time when present; otherwise
    // advance from the cursor using `bpm` + `meter`.
    const entryStart = typeof entry.t === 'number' ? entry.t : cursor;
    if (entryStart > cursor + EPS) cursor = entryStart;
    const nextStart = next ? (typeof next.t === 'number' ? next.t : null) : duration;

    const [n, d] = entry.meter || [4, 4];
    const barSec = n * (4 / d) * (60 / (entry.bpm || 120));
    if (!(barSec > 0)) continue;

    // Emit bar starts up to the next entry's start time (or duration).
    const limit = nextStart != null ? nextStart : duration;
    while (cursor < limit - EPS) {
      starts.push(cursor);
      cursor += barSec;
    }
  }
  if (cursor < duration - EPS) starts.push(cursor);
  starts.push(duration);
  return starts;
}

/**
 * Identify which tempoMap entry owns a given bar-start time. Returns the
 * entry object plus the 1-indexed `barWithinEntry` offset (useful for
 * musical reasoning like "which bar of this meter-section are we in?").
 */
export function locateBar(tempoMap, t) {
  const entry = entryAtTime(tempoMap, t);
  if (!entry) return null;
  const [n, d] = entry.meter || [4, 4];
  const barSec = n * (4 / d) * (60 / (entry.bpm || 120));
  const offsetSec = t - (entry.t ?? 0);
  const barWithinEntry = Math.floor(offsetSec / Math.max(EPS, barSec)) + 1;
  return { entry, barWithinEntry };
}

/**
 * Pretty-print a tempo map for logs.
 *
 *   bar  1 @   0.00s   120 BPM   4/4
 *   bar  9 @  16.00s   124 BPM   4/4
 *   bar 17 @  32.00s   124 BPM   7/8 (4+3)
 */
export function formatTempoMap(tempoMap) {
  if (!tempoMap || tempoMap.length === 0) return '(empty)';
  return tempoMap.map((e) => {
    const bar = String(e.bar ?? '?').padStart(3);
    const t = typeof e.t === 'number' ? e.t.toFixed(2).padStart(6) : '  -  ';
    const bpm = (e.bpm ?? '?').toString().padStart(3);
    const meter = e.meter ? `${e.meter[0]}/${e.meter[1]}` : '?/?';
    const group = e.grouping ? ` (${e.grouping})` : '';
    return `  bar ${bar} @ ${t}s   ${bpm} BPM   ${meter}${group}`;
  }).join('\n');
}

/**
 * Build a constant-meter single-entry map from legacy detection output.
 * Used as a shim until the backend returns a full tempoMap.
 */
export function tempoMapFromConstant({ bpm, beatsPerBar, meterDenominator, grouping }) {
  return [{
    bar: 1, t: 0,
    bpm: bpm || 120,
    meter: [beatsPerBar || 4, meterDenominator || 4],
    ...(grouping ? { grouping } : {}),
  }];
}
