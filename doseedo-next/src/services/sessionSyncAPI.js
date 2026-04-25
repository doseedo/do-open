/**
 * Session sync API — reads the auth-service `/api/sessions/*` endpoints
 * that the Logic desktop app writes into. Companion to saveService.js,
 * which owns the web-origin WRITE path; this module owns the READ path
 * so /studio?session=<uuid> can hydrate from a desktop-synced project.
 *
 * Shapes:
 *   GET /api/sessions              → { items: SessionRow[], total, next_offset }
 *   GET /api/sessions/{id}         → SessionRow
 *   GET /api/sessions/{id}/state   → { state, version, updated_at }
 *
 * `state` is the desktop-produced JSON written by
 * logic_engine/logic_sync.py::_build_session_json — top-level keys
 * projectName/bpm/timeSignature/tempoMap/markers/buses[].  Track shape
 * uses Logic-era names (`url`, `volume`, `mute`, `solo`) which the web
 * reducer doesn't speak; adaptDesktopStateToContext() normalizes to the
 * AppContext keys (audioUrl, gain, isMuted, isSolo) before LOAD_SESSION.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || '';

async function _authToken() {
  if (typeof window === 'undefined') return null;
  if (typeof window.__clerkGetToken === 'function') {
    try { return await window.__clerkGetToken(); } catch { /* fall through */ }
  }
  try { return window.localStorage?.getItem('token'); } catch { return null; }
}

async function _authHeaders(extra = {}) {
  const token = await _authToken();
  return {
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
    ...extra,
  };
}

async function _get(path, { shareToken } = {}) {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (shareToken) url.searchParams.set('share_token', shareToken);
  const res = await fetch(url.toString(), {
    method: 'GET',
    headers: await _authHeaders(),
    credentials: 'include',
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return res.json();
}

export async function listMySessions({ limit = 50, offset = 0 } = {}) {
  const qs = new URLSearchParams({ limit, offset }).toString();
  return _get(`/api/sessions?${qs}`);
}

export async function fetchSessionMeta(sessionId, opts = {}) {
  return _get(`/api/sessions/${sessionId}`, opts);
}

export async function fetchSessionState(sessionId, opts = {}) {
  return _get(`/api/sessions/${sessionId}/state`, opts);
}

export async function deleteSession(sessionId) {
  const url = `${API_BASE}/api/sessions/${sessionId}`;
  const res = await fetch(url, {
    method: 'DELETE',
    headers: await _authHeaders(),
    credentials: 'include',
  });
  if (!res.ok && res.status !== 204) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  return true;
}

/**
 * Convert the desktop `state.json` shape (see logic_sync._build_session_json)
 * into the AppContext reducer's LOAD_SESSION payload. Pure; testable.
 *
 * The desktop blob never carries web-only analysis fields (midiData.onsets,
 * latents, envelopeData, waveformBuffer) — those get recomputed lazily by
 * the studio when the audioUrl is hit. Keeping this mapper tight means we
 * don't have to coordinate desktop and web schemas beyond the stable
 * subset (buses → tracks → {url, position, duration, midi notes}).
 */
export function adaptDesktopStateToContext(desktopState) {
  if (!desktopState || typeof desktopState !== 'object') return null;

  const buses = Array.isArray(desktopState.buses) ? desktopState.buses : [];

  // Logic flex-time edits: per-transient onset deltas keyed by the audio
  // region's auflGid. Index once so every track can match in O(1).
  const flexByAuflGid = _indexFlexOnsetEdits(desktopState.flexOnsetEdits);

  const adaptedBuses = buses.map((bus) => ({
    id: bus.id,
    type: _capitalize(bus.type) || 'Music',
    name: bus.name || _capitalize(bus.type) || 'Bus',
    gain: typeof bus.gain === 'number' ? bus.gain : 1.0,
    pan: typeof bus.pan === 'number' ? bus.pan : 0,
    reverbSend: typeof bus.reverbSend === 'number' ? bus.reverbSend : 0,
    mute: !!bus.mute,
    solo: !!bus.solo,
    expanded: false,
    metadata: {},
    tracks: (Array.isArray(bus.tracks) ? bus.tracks : []).map(
      (t) => _adaptTrack(t, flexByAuflGid),
    ),
  }));

  let totalDuration = 10;
  for (const bus of adaptedBuses) {
    for (const t of bus.tracks) {
      const end = (t.startPosition || 0) + (t.duration || 0);
      if (end > totalDuration) totalDuration = end;
    }
  }

  const ts = desktopState.timeSignature || {};
  const beatsPerBar = Number(ts.numerator) > 0 ? Number(ts.numerator) : 4;
  const meterDenominator = Number(ts.denominator) > 0 ? Number(ts.denominator) : 4;

  // Pass through every session-level field the desktop sync now emits
  // (commit 248239c). Most of these are metadata-only at this point — the
  // playback engine consumes a subset (cycleRegion, tempoMap, meterMap),
  // the rest is preserved for UI components and future DSP work without
  // requiring a re-sync.
  const passthrough = {};
  const SESSION_PASS_FIELDS = [
    'tempoMap', 'meterMap', 'keySignatureMap',
    'cycleRegion', 'autopunchRegion',
    'metronomeCountIn', 'metronomeClickEnabled',
    'sessionLengthSamples', 'sampleRate', 'recordingFormat',
    'projectLengthTicks', 'markers', 'flexMode', 'trackUuids',
    'appleLoops', 'drummerState', 'smartTempoResults',
    'articulationSets', 'flexPitchNotes', 'flexOnsetEdits',
    'sessionId', 'syncId', 'syncedAt', 'sourceApp',
  ];
  for (const k of SESSION_PASS_FIELDS) {
    if (desktopState[k] !== undefined) passthrough[k] = desktopState[k];
  }

  return {
    projectName: desktopState.projectName || 'Synced Session',
    bpm: Number(desktopState.bpm) > 0 ? Number(desktopState.bpm) : 120,
    beatsPerBar,
    meterDenominator,
    buses: adaptedBuses,
    totalDuration,
    ...passthrough,
    // Blank the undo stack — a just-hydrated session should not "undo" back
    // into its pre-hydration empty state.
    history: { past: [], future: [] },
  };
}

// Track-level fields stuffed into metadata (preserved verbatim from the
// desktop emit). Anything in this list lives under track.metadata.<key>
// and is read either by the playback engine (sends, automationMetadata)
// or by UI components (instrumentPatch, samplerRefs, freeze, etc.).
const TRACK_PASS_FIELDS = [
  'uuid', 'output', 'input', 'sends', 'recordMonitor', 'channelFormat',
  'automationMetadata', 'enviHeader', 'instrumentPatch', 'samplerRefs',
  'midiTransform', 'freeze', 'activeTake', 'group',
];

function _adaptTrack(t, flexByAuflGid = new Map()) {
  if (!t) return t;
  // Desktop emits Logic-era field names. Translate to the AppContext
  // reducer's vocabulary. Missing values fall back to sensible defaults
  // so a sparsely-populated track still plays (muted/solo'd is benign;
  // missing audioUrl renders as an empty clip the user can re-add).
  const flexTime = _buildFlexTime(t, flexByAuflGid);
  const clips = _enrichClipsWithFlex(t.clips, flexByAuflGid);

  const passthrough = {};
  for (const k of TRACK_PASS_FIELDS) {
    if (t[k] !== undefined) passthrough[k] = t[k];
  }

  const out = {
    id: t.id,
    name: t.name || 'Track',
    type: t.type || 'audio',
    audioUrl: t.url || t.audioUrl || null,
    startPosition: Number(t.startPosition) || 0,
    duration: Number(t.duration) || 0,
    gain: typeof t.volume === 'number' ? t.volume : (typeof t.gain === 'number' ? t.gain : 1.0),
    pan: Number(t.pan) || 0,
    reverbSend: typeof t.reverbSend === 'number' ? t.reverbSend : 0.15,
    isMuted: !!t.mute || !!t.isMuted,
    isSolo: !!t.solo || !!t.isSolo,
    color: t.color,
    cropStart: Number(t.cropStart) || 0,
    cropEnd: Number(t.cropEnd) || 0,
    metadata: {
      source: 'desktop-sync',
      ...(t.logicPlugins ? { logicPlugins: t.logicPlugins } : {}),
      ...(clips.length > 0 ? { clips } : {}),
      ...(flexTime ? { flexTime } : {}),
      ...passthrough,
    },
  };
  if (t.midiData && typeof t.midiData === 'object') {
    out.midiData = t.midiData;  // {notes, tempo, duration, cc, pitchBend}
    out.type = 'midi';
  }
  if (t.automation) out.automation = t.automation;
  return out;
}

function _indexFlexOnsetEdits(flexOnsetEdits) {
  const map = new Map();
  if (!Array.isArray(flexOnsetEdits)) return map;
  for (const e of flexOnsetEdits) {
    if (!e || typeof e !== 'object') continue;
    const gid = typeof e.auflGid === 'number' ? e.auflGid : null;
    if (gid == null) continue;
    const list = map.get(gid) || [];
    list.push(e);
    map.set(gid, list);
  }
  return map;
}

/**
 * Attach the matching flex onset delta (if any) to each clip as
 * `flexDeltaSeconds`. Per-clip lets the scheduler shift each region
 * independently — what Logic's Slicing algorithm actually does.
 *
 * Matching: prefer an exact (auflGid, subIndex) hit; fall back to any
 * edit for the region if the clip has no subIndex. The backend already
 * filters out zero deltas, so any match is a real user edit.
 */
function _enrichClipsWithFlex(clips, flexByAuflGid) {
  if (!Array.isArray(clips) || clips.length === 0) return [];
  if (!(flexByAuflGid instanceof Map) || flexByAuflGid.size === 0) {
    return clips.slice();
  }
  return clips.map((c) => {
    if (!c || typeof c !== 'object') return c;
    const gid = typeof c.auflGid === 'number' ? c.auflGid : null;
    if (gid == null) return c;
    const edits = flexByAuflGid.get(gid);
    if (!edits || edits.length === 0) return c;
    const sub = typeof c.subIndex === 'number' ? c.subIndex : null;
    const match = sub != null
      ? (edits.find((e) => e.subIndex === sub) || null)
      : edits[0];
    if (!match) return c;
    const delta = Number(match.deltaSeconds) || 0;
    if (delta === 0) return c;
    return { ...c, flexDeltaSeconds: delta };
  });
}

/**
 * Build an internal "flex time" model on a track.
 *
 * Closest match to Logic's Flex Time "Slicing" algorithm: the region holds
 * an ordered list of detected transients, and each edited transient stores
 * a per-slot delta from its detected position. Logic renders a slice per
 * transient and plays each slice at (detected_position + delta).
 *
 * Shape:
 *   track.metadata.flexTime = {
 *     algorithm:      'slicing',              // the mode we emulate
 *     sampleRate:     44100,                  // from the source audio
 *     transients: [
 *       { auflGid, subIndex, regionName,
 *         deltaSamples, deltaSeconds, flags },
 *       ...
 *     ],
 *     offsetSeconds:  <mean of non-zero deltaSeconds>,  // scalar for MVP
 *   }
 *
 * **Why a scalar `offsetSeconds`?** Until the scheduler handles per-slice
 * playback, each track plays as one buffer. We collapse all per-transient
 * deltas to one number — the average shift — and apply that to the whole
 * clip's start time. It's an approximation; a single-delta region (the
 * common case produced by the backend) is faithful, and a multi-delta
 * region gets "net region drift". When per-slice scheduling lands, the
 * `transients[]` array is already the right data to drive it.
 *
 * Matching: flex edits are keyed by `auflGid` (the Logic audio region's
 * group id). A track has:
 *   - `t.auflGid` — its primary playable region (what `audioUrl` points at)
 *   - `t.clips[i].auflGid` — every region placed on the timeline lane
 * We pull edits for all of them so the model stays complete even if the
 * web only plays one of the regions today.
 */
function _buildFlexTime(t, flexByAuflGid) {
  if (!t || !(flexByAuflGid instanceof Map) || flexByAuflGid.size === 0) return null;
  const gids = new Set();
  if (typeof t.auflGid === 'number') gids.add(t.auflGid);
  if (Array.isArray(t.clips)) {
    for (const c of t.clips) {
      if (c && typeof c.auflGid === 'number') gids.add(c.auflGid);
    }
  }
  if (gids.size === 0) return null;

  const transients = [];
  let sampleRate = null;
  for (const gid of gids) {
    const list = flexByAuflGid.get(gid);
    if (!list) continue;
    for (const e of list) {
      transients.push({
        auflGid: e.auflGid,
        subIndex: Number(e.subIndex) || 0,
        regionName: e.regionName || null,
        deltaSamples: Number(e.deltaSamples) || 0,
        deltaSeconds: Number(e.deltaSeconds) || 0,
        flags: Number(e.flags) || 0,
      });
      if (typeof e.sampleRate === 'number' && e.sampleRate > 0) {
        sampleRate = e.sampleRate;
      }
    }
  }
  if (transients.length === 0) return null;

  const offsetSeconds = transients.reduce(
    (sum, tr) => sum + tr.deltaSeconds, 0,
  ) / transients.length;

  return {
    algorithm: 'slicing',
    sampleRate,
    transients,
    offsetSeconds,
  };
}

function _capitalize(s) {
  if (!s || typeof s !== 'string') return s;
  return s.charAt(0).toUpperCase() + s.slice(1);
}

const api = {
  listMySessions,
  fetchSessionMeta,
  fetchSessionState,
  deleteSession,
  adaptDesktopStateToContext,
};

export default api;
