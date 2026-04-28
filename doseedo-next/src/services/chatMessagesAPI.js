/**
 * chatMessagesAPI — server-side chat history per session.
 *
 * Endpoints (auth-service, alembic 014):
 *   GET  /api/sessions/{sid}/chat-messages?limit&before
 *   POST /api/sessions/{sid}/chat-messages   { role, content, client_op_id, metadata }
 *
 * Both web and desktop write here (Phase B/C of the chat unification),
 * so a thread started on either side surfaces on the other. POST is
 * idempotent on `client_op_id` — clients pre-mint a UUID and retry
 * the same value if a network blip drops the response.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';

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

async function _request(path, { method = 'GET', body, query } = {}) {
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
  if (query) {
    for (const [k, v] of Object.entries(query)) {
      if (v !== undefined && v !== null) url.searchParams.set(k, String(v));
    }
  }
  const res = await fetch(url.toString(), {
    method,
    headers: await _authHeaders(body ? { 'Content-Type': 'application/json' } : {}),
    body: body ? JSON.stringify(body) : undefined,
    credentials: 'include',
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try { detail = (await res.json()).detail || detail; } catch {}
    const err = new Error(detail);
    err.status = res.status;
    throw err;
  }
  if (res.status === 204) return null;
  return res.json();
}

/**
 * Fetch the latest N messages for a session, newest-first. To page
 * back, pass the oldest message's `created_at` as `before`.
 *
 * @returns {Promise<Array<{id, session_id, role, content, author_user_id,
 *   author_origin, client_op_id, metadata, created_at}>>}
 */
export async function listChatMessages(sessionId, { limit = 100, before = null } = {}) {
  if (!sessionId) return [];
  const query = { limit };
  if (before) query.before = before;
  return _request(`/api/sessions/${sessionId}/chat-messages`, { query });
}

/**
 * Append a chat message. `client_op_id` should be a UUID the client
 * pre-mints — repeated POSTs with the same value return the same row,
 * not a duplicate. Use this to make the chat panel's flush loop safe
 * across network blips.
 */
export async function appendChatMessage(sessionId, { role, content, clientOpId, metadata, authorOrigin = 'web' }) {
  if (!sessionId) throw new Error('appendChatMessage requires sessionId');
  if (!role || !content) throw new Error('appendChatMessage requires role + content');
  return _request(`/api/sessions/${sessionId}/chat-messages`, {
    method: 'POST',
    body: {
      role,
      content,
      client_op_id: clientOpId || null,
      author_origin: authorOrigin,
      metadata: metadata || null,
    },
  });
}

/**
 * Generate a UUID for use as `client_op_id`. Falls back to a
 * timestamp-randomness composite when crypto.randomUUID isn't
 * available (very old browsers).
 */
export function newClientOpId() {
  try {
    if (typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function') {
      return crypto.randomUUID();
    }
  } catch { /* ignore */ }
  return `op-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
