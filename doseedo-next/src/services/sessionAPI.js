/**
 * Session Management API
 *
 * Wraps the doseedo-api `/api/sessions` routes (rewritten through
 * `next.config.js` to `NEXT_PUBLIC_AUTH_ORIGIN`).
 *
 *   POST   /api/sessions                     -> SessionResponse
 *   GET    /api/sessions?limit=&offset=      -> { items, total, next_offset }
 *   GET    /api/sessions/search?q=...        -> { items, total, next_offset }
 *   GET    /api/sessions/{id}                -> SessionResponse
 *   PATCH  /api/sessions/{id}                -> SessionResponse (metadata only)
 *   DELETE /api/sessions/{id}                -> 204
 *   PUT    /api/sessions/{id}/state          -> writes R2 `{gcs_base_path}state.json`
 *   GET    /api/sessions/{id}/state          -> reads R2 state blob
 *   POST   /api/sessions/{id}/import[?share_token=...] -> SessionResponse
 *
 * Auth is handled by httpClient: the `.doseedo.com` cookie is sent via
 * `credentials: 'include'`, plus a Clerk JWT Bearer fallback when present.
 */

import { apiFetch } from './httpClient';

// ── Core CRUD ────────────────────────────────────────────────────────────

export const createSession = async (sessionData) =>
  apiFetch('/api/sessions', {
    method: 'POST',
    body: JSON.stringify(sessionData),
  });

/**
 * New paged listing. Returns `{items, total, next_offset}`.
 */
export const listSessions = async (params = {}) => {
  const qs = new URLSearchParams();
  qs.set('limit', String(params.limit ?? 50));
  qs.set('offset', String(params.offset ?? 0));
  for (const [k, v] of Object.entries(params)) {
    if (k === 'limit' || k === 'offset') continue;
    if (v === undefined || v === null || v === '') continue;
    qs.set(k, String(v));
  }
  return apiFetch(`/api/sessions?${qs.toString()}`);
};

/**
 * Legacy-shape wrapper kept for unmigrated callers. Always returns
 * `{sessions, items, total, next_offset}` so callers that treat the result
 * as either an array or the old `.sessions` field keep working.
 */
export const getUserSessions = async (filters = {}) => {
  try {
    const data = await listSessions(filters);
    if (Array.isArray(data)) {
      return { sessions: data, items: data, total: data.length, next_offset: null };
    }
    return {
      sessions: data?.items || [],
      items: data?.items || [],
      total: data?.total ?? (data?.items?.length ?? 0),
      next_offset: data?.next_offset ?? null,
    };
  } catch (error) {
    if (process.env.NODE_ENV === 'development') {
      console.warn('Get sessions error:', error.message);
    }
    throw error;
  }
};

export const getSession = async (sessionId) =>
  apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`);

export const updateSession = async (sessionId, updates) =>
  apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: 'PATCH',
    body: JSON.stringify(updates),
  });

export const deleteSession = async (sessionId) => {
  await apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}`, { method: 'DELETE' });
  return true;
};

// ── Project-state blob (R2-backed) ───────────────────────────────────────

/**
 * Write the project state to R2. Backend stores at `{gcs_base_path}state.json`
 * and bumps `gcs_object_version`.
 * @returns {Promise<{version: number, updated_at: string, key: string, size_bytes: number}>}
 */
export const putSessionState = async (sessionId, state) =>
  apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/state`, {
    method: 'PUT',
    body: JSON.stringify(state ?? {}),
  });

/**
 * Read the project state from R2 via the backend.
 * Returns `{state, version, updated_at}`. `state` is `null` if nothing saved yet.
 */
export const getSessionState = async (sessionId) =>
  apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/state`);

// ── Search / public / import ─────────────────────────────────────────────

export const searchSessions = async (query, filters = {}) => {
  const qs = new URLSearchParams({ q: query || '' });
  qs.set('limit', String(filters.limit ?? 50));
  qs.set('offset', String(filters.offset ?? 0));
  for (const [k, v] of Object.entries(filters)) {
    if (k === 'limit' || k === 'offset') continue;
    if (v === undefined || v === null || v === '') continue;
    qs.set(k, String(v));
  }
  return apiFetch(`/api/sessions/search?${qs.toString()}`);
};

export const getPublicSessions = async (filters = {}) => {
  const qs = new URLSearchParams(filters).toString();
  return apiFetch(`/api/sessions/public${qs ? `?${qs}` : ''}`, {}, { skipAuth: true });
};

export const importSession = async (sessionId, shareToken = null) => {
  const qs = shareToken ? `?share_token=${encodeURIComponent(shareToken)}` : '';
  return apiFetch(
    `/api/sessions/${encodeURIComponent(sessionId)}/import${qs}`,
    { method: 'POST' }
  );
};

// Kept for back-compat with the old multipart upload path.
export const uploadSessionContent = async (sessionId, contentData) =>
  apiFetch(`/api/sessions/${encodeURIComponent(sessionId)}/upload`, {
    method: 'POST',
    body: JSON.stringify(contentData),
  });

export default {
  createSession,
  listSessions,
  getUserSessions,
  getSession,
  updateSession,
  deleteSession,
  putSessionState,
  getSessionState,
  searchSessions,
  getPublicSessions,
  importSession,
  uploadSessionContent,
};
