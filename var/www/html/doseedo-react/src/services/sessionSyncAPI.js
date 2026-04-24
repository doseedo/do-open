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

const API_BASE = process.env.REACT_APP_API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || '';

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
    tracks: (Array.isArray(bus.tracks) ? bus.tracks : []).map(_adaptTrack),
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

  return {
    projectName: desktopState.projectName || 'Synced Session',
    bpm: Number(desktopState.bpm) > 0 ? Number(desktopState.bpm) : 120,
    beatsPerBar,
    meterDenominator,
    buses: adaptedBuses,
    totalDuration,
    // Blank the undo stack — a just-hydrated session should not "undo" back
    // into its pre-hydration empty state.
    history: { past: [], future: [] },
  };
}

function _adaptTrack(t) {
  if (!t) return t;
  // Desktop emits Logic-era field names. Translate to the AppContext
  // reducer's vocabulary. Missing values fall back to sensible defaults
  // so a sparsely-populated track still plays (muted/solo'd is benign;
  // missing audioUrl renders as an empty clip the user can re-add).
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
      ...(Array.isArray(t.clips) && t.clips.length > 0 ? { clips: t.clips } : {}),
    },
  };
  if (t.midiData && typeof t.midiData === 'object') {
    out.midiData = t.midiData;  // {notes, tempo, duration, cc, pitchBend}
    out.type = 'midi';
  }
  if (t.automation) out.automation = t.automation;
  return out;
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
