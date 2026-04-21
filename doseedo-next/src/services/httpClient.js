/**
 * httpClient — shared fetch wrapper for backend `/api/*` calls.
 *
 * Auth strategy (matches `app/after-signin/page.tsx` + `auth-service/app/deps.py`):
 *
 *   1. Primary: the clerk-bridge endpoint sets an HTTP-only `access_token`
 *      cookie scoped to `.doseedo.com`, so every `/api/*` call must be made
 *      with `credentials: 'include'` to carry it.
 *
 *   2. Fallback: in dev (localhost) the cookie won't cross; if a Clerk token
 *      is exposed via `window.__clerkGetToken()` or a `clerk_token`/`token`
 *      localStorage snapshot, attach it as `Authorization: Bearer`.
 */

// Relative base — Next.js rewrites `/api/*` to NEXT_PUBLIC_AUTH_ORIGIN.
const API_BASE = '';

export async function getAuthToken() {
  try {
    if (typeof window === 'undefined') return null;
    const clerkGetter =
      typeof window.__clerkGetToken === 'function' ? window.__clerkGetToken : null;
    if (clerkGetter) {
      try {
        const t = await clerkGetter();
        if (t) return t;
      } catch {
        /* fall through to localStorage */
      }
    }
    const snap =
      window.localStorage?.getItem?.('clerk_token') ||
      window.localStorage?.getItem?.('token');
    return snap || null;
  } catch {
    return null;
  }
}

export async function fetchWithAuth(path, init = {}, opts = {}) {
  const headers = new Headers(init.headers || {});
  if (!opts.skipAuth) {
    const token = await getAuthToken();
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
  }
  if (init.body && typeof init.body === 'string' && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json');
  }
  const url = path.startsWith('http') ? path : `${API_BASE}${path}`;
  return fetch(url, { ...init, headers, credentials: init.credentials || 'include' });
}

export async function apiFetch(path, init = {}, opts = {}) {
  const res = await fetchWithAuth(path, init, opts);
  const ct = res.headers.get('content-type') || '';
  if (ct.includes('text/html')) {
    throw new Error(`API endpoint not available at ${path} (got HTML — backend may be offline)`);
  }
  if (res.status === 204) return null;
  const data = ct.includes('application/json') ? await res.json() : await res.text();
  if (!res.ok) {
    const msg =
      (data && (data.error || data.detail || data.message)) ||
      (typeof data === 'string' && data) ||
      `HTTP ${res.status}`;
    const err = new Error(msg);
    err.status = res.status;
    err.body = data;
    throw err;
  }
  return data;
}

export default { fetchWithAuth, apiFetch, getAuthToken };
