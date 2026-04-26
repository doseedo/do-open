/**
 * Session edits producer — emits semantic edits the desktop replays into Logic.
 *
 * Companion to sessionSyncAPI.js. Where sessionSyncAPI.js owns the READ
 * path for hydrating /studio?session=<uuid> from the desktop's last upload,
 * this module owns the WRITE path: every change a user makes in the web
 * DAW (volume, pan, plugin params, …) gets POSTed to
 *   POST /api/sessions/{sid}/edits
 * The desktop's edit_consumer polls that log and replays each edit into
 * Logic via doo_hook live mode.
 *
 * Design:
 *   - One in-memory queue per session, keyed by sessionId.
 *   - Queue is debounced: rapid slider drags coalesce into a single POST
 *     ~250ms after the user stops moving (and a hard cap of 1s in flight).
 *   - Each enqueued edit gets a uuid `client_op_id` so a network retry
 *     doesn't apply twice (server enforces (session_id, client_op_id)
 *     uniqueness — see auth-service alembic 008).
 *   - Failures fall through silently for now; a real outbox / retry is a
 *     follow-up. This stage is fire-and-forget — the worst case is a
 *     dropped edit on a flaky network, which the user will notice and
 *     correct manually. (When we wire offline support, add IndexedDB.)
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';
const FLUSH_DELAY_MS = 250;
const MAX_INFLIGHT_DELAY_MS = 1000;
const MAX_BATCH = 200;

// ── Auth token helper (same convention as sessionSyncAPI.js) ─────────────────

async function _authToken() {
  if (typeof window === 'undefined') return null;
  if (typeof window.__clerkGetToken === 'function') {
    try { return await window.__clerkGetToken(); } catch { /* fall through */ }
  }
  try { return window.localStorage?.getItem('token'); } catch { return null; }
}

async function _authHeaders() {
  const token = await _authToken();
  return {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  };
}

function _uuid() {
  // Cheap RFC4122 v4 — fine for client_op_id (server doesn't trust it for crypto)
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

// ── Per-session queue ────────────────────────────────────────────────────────

const _queues = new Map(); // sessionId → { pending: [], flushTimer, firstEnqueuedAt }

function _ensureQueue(sessionId) {
  let q = _queues.get(sessionId);
  if (!q) {
    q = { pending: [], flushTimer: null, firstEnqueuedAt: null };
    _queues.set(sessionId, q);
  }
  return q;
}

async function _flush(sessionId) {
  const q = _queues.get(sessionId);
  if (!q || q.pending.length === 0) return;
  const batch = q.pending.splice(0, MAX_BATCH);
  q.firstEnqueuedAt = q.pending.length > 0 ? performance.now() : null;
  if (q.flushTimer) {
    clearTimeout(q.flushTimer);
    q.flushTimer = null;
  }
  try {
    const headers = await _authHeaders();
    await fetch(`${API_BASE}/api/sessions/${sessionId}/edits`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ edits: batch }),
      keepalive: true, // survive page-unload mid-flight
    });
  } catch (e) {
    // Fire-and-forget — real outbox lands in a follow-up commit.
    console.warn('[sessionEditsAPI] flush failed:', e?.message || e);
  }
  // If more edits arrived during flight, schedule another flush.
  if (q.pending.length > 0) _scheduleFlush(sessionId);
}

function _scheduleFlush(sessionId) {
  const q = _ensureQueue(sessionId);
  const now = performance.now();
  if (q.firstEnqueuedAt == null) q.firstEnqueuedAt = now;

  const elapsed = now - q.firstEnqueuedAt;
  const delay = Math.max(0, Math.min(FLUSH_DELAY_MS, MAX_INFLIGHT_DELAY_MS - elapsed));

  if (q.flushTimer) clearTimeout(q.flushTimer);
  q.flushTimer = setTimeout(() => _flush(sessionId), delay);
}

// ── Public API ───────────────────────────────────────────────────────────────

/**
 * Enqueue a semantic edit. Returns immediately — the actual POST happens
 * after debounce. Pass a stable `dedupKey` (e.g. "volume:track-7") and
 * later edits with the same key replace earlier ones in the queue, so
 * dragging a slider 100×/sec only sends the final value.
 */
export function enqueueEdit(sessionId, op, args, { dedupKey = null } = {}) {
  if (!sessionId) {
    console.warn('[sessionEditsAPI] enqueueEdit: no sessionId — dropping', op);
    return;
  }
  const q = _ensureQueue(sessionId);
  if (dedupKey) {
    const idx = q.pending.findIndex((e) => e.dedupKey === dedupKey);
    if (idx >= 0) {
      q.pending[idx] = { op, args, client_op_id: _uuid(), dedupKey };
      _scheduleFlush(sessionId);
      return;
    }
  }
  q.pending.push({ op, args, client_op_id: _uuid(), dedupKey });
  _scheduleFlush(sessionId);
}

/** Convenience for the volume slider. `channel` is the 1-based Logic mixer
 *  strip index; `value` is 0..1 (matches both the `<input type="range">`
 *  scale and what doo_hook.set_volume expects). */
export function enqueueVolumeEdit(sessionId, channel, value) {
  if (typeof channel !== 'number' || channel < 1) return;
  enqueueEdit(
    sessionId,
    'set_volume',
    { channel, value: Math.max(0, Math.min(1, value)) },
    { dedupKey: `set_volume:${channel}` }
  );
}

/** Force-flush before navigation / unmount. */
export async function flushSession(sessionId) {
  await _flush(sessionId);
}

/** Tear down all queues (e.g. on signout). Drops any pending edits. */
export function resetAll() {
  for (const q of _queues.values()) {
    if (q.flushTimer) clearTimeout(q.flushTimer);
  }
  _queues.clear();
}
