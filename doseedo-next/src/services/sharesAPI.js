/**
 * Session share-token API — owner-only token mint/list/revoke for collab invites.
 *
 *   GET    /api/sessions/{sid}/shares             → SessionShareResponse[]
 *   POST   /api/sessions/{sid}/share              → SessionShareResponse
 *                  body: { role: "view" | "edit", expires_in_hours?: int }
 *   DELETE /api/sessions/{sid}/share/{token}      → 204
 *
 * The token returned by POST is a 64-char hex string. Build invite URLs as:
 *   `${origin}/studio?session=${sid}&share_token=${token}`
 * The /studio route's useSessionSync hook already reads `?share_token=` on
 * mount so guests landing on the URL get scoped read/edit access according
 * to the role baked into the token.
 *
 * Auth pattern matches commitsAPI.js / sessionEditsAPI.js — Clerk JWT via
 * `window.__clerkGetToken()` plus the legacy `localStorage.token` fallback.
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

async function _request(path, { method = 'GET', body } = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, {
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
 * List all share tokens for a session (owner only). Returns 403 for non-owners.
 * Includes revoked + expired rows — caller filters to actionable set.
 */
export async function listShares(sessionId) {
  if (!sessionId) return [];
  return _request(`/api/sessions/${sessionId}/shares`);
}

/**
 * Mint a new share token. Owner only.
 *   role: 'view' (read-only) | 'edit' (full collab)
 *   expiresInHours: 1..8760 (1y); null = never expires
 */
export async function createShare(sessionId, { role = 'view', expiresInHours = null } = {}) {
  if (!sessionId) throw new Error('createShare: sessionId required');
  const body = { role };
  if (expiresInHours != null) body.expires_in_hours = expiresInHours;
  return _request(`/api/sessions/${sessionId}/share`, { method: 'POST', body });
}

/**
 * Revoke a share token by its hex value. Owner only. Returns null on success.
 */
export async function revokeShare(sessionId, token) {
  if (!sessionId || !token) throw new Error('revokeShare: sessionId + token required');
  return _request(`/api/sessions/${sessionId}/share/${encodeURIComponent(token)}`, {
    method: 'DELETE',
  });
}

/**
 * Build the user-facing invite URL for a share token. Centralised so the UI
 * and the eventual server-rendered share preview stay in sync.
 */
export function buildInviteUrl(sessionId, token) {
  if (!sessionId || !token) return null;
  const origin = (typeof window !== 'undefined' && window.location?.origin) || 'https://doseedo.com';
  return `${origin}/studio?session=${encodeURIComponent(sessionId)}&share_token=${encodeURIComponent(token)}`;
}

/**
 * Whether the current viewer can manage shares (owner-only). The server is
 * the source of truth — non-owners get a 403 from `listShares`. Use that
 * signal rather than deriving from a Clerk id we'd have to compare across
 * Clerk-id → internal user_id mapping. Returns:
 *   { canManage: bool, shares: SessionShareResponse[] | [] }
 */
export async function probeOwnerAccess(sessionId) {
  try {
    const shares = await listShares(sessionId);
    return { canManage: true, shares: Array.isArray(shares) ? shares : [] };
  } catch (e) {
    if (e?.status === 403 || e?.status === 401 || e?.status === 404) {
      return { canManage: false, shares: [] };
    }
    // Network / 5xx — surface as not-manageable but don't swallow silently.
    console.warn('[sharesAPI] probeOwnerAccess failed:', e?.message || e);
    return { canManage: false, shares: [] };
  }
}
