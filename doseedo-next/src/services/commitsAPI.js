/**
 * Server-side commit DAG API — read path for the History tab.
 *
 * Companion to sessionHistory.js (localStorage). The localStorage DAG is
 * still useful as an offline cache, but the server commit log is now the
 * source of truth: it carries authorship, persists across devices, and
 * is the substrate for rights metadata + .logicx archive recovery.
 *
 *   GET  /api/sessions/{sid}/commits          → CommitResponse[]
 *   GET  /api/sessions/{sid}/commits/{cid}    → CommitDetail (incl. tree_body)
 *   GET  /api/sessions/{sid}/refs             → RefResponse[]
 *   PUT  /api/sessions/{sid}/refs/{name}      → CAS ref update
 *   POST /api/sessions/{sid}/commits          → mint commit (write path; not used yet here)
 *   GET  /api/sessions/{sid}/blobs/{sha}      → signed download URL
 *
 * All authenticated reads go through _get; no public flavor — commits
 * are owner-only on the server.
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

// ── Commits ─────────────────────────────────────────────────────────────────

export async function listCommits(sessionId, { limit = 50, before = null } = {}) {
  if (!sessionId) return [];
  const query = { limit };
  if (before) query.before = before;
  return _request(`/api/sessions/${sessionId}/commits`, { query });
}

export async function getCommit(sessionId, commitId) {
  if (!sessionId || !commitId) return null;
  return _request(`/api/sessions/${sessionId}/commits/${commitId}`);
}

// ── Refs ────────────────────────────────────────────────────────────────────

export async function listRefs(sessionId) {
  if (!sessionId) return [];
  return _request(`/api/sessions/${sessionId}/refs`);
}

export async function putRef(sessionId, name, commitId, expectedCommitId) {
  if (!sessionId || !name || !commitId) {
    throw new Error('putRef requires sessionId, name, commitId');
  }
  return _request(`/api/sessions/${sessionId}/refs/${encodeURIComponent(name)}`, {
    method: 'PUT',
    body: { commit_id: commitId, expected_commit_id: expectedCommitId ?? null },
  });
}

// ── Blobs ───────────────────────────────────────────────────────────────────

export async function blobDownloadUrl(sessionId, sha256) {
  if (!sessionId || !sha256) return null;
  return _request(`/api/sessions/${sessionId}/blobs/${sha256}`);
}

// ── Aggregates ──────────────────────────────────────────────────────────────

/**
 * Fetch commits + refs in parallel — what the History tab actually needs
 * to render. Returns { commits, refs, head: { ref, commit_id } | null,
 * original: commit_id | null }.
 *
 * `head` is the current `main` tip. `original` is the
 * `protected/original` ref minted by commit 0; the History UI uses it to
 * mark the "Original" row distinctly so users can always see where the
 * unmodified bytes live.
 */
export async function fetchHistory(sessionId) {
  if (!sessionId) return { commits: [], refs: [], head: null, original: null };
  const [commits, refs] = await Promise.all([
    listCommits(sessionId, { limit: 200 }),
    listRefs(sessionId),
  ]);
  const refByName = Object.fromEntries((refs || []).map((r) => [r.name, r.commit_id]));
  return {
    commits: commits || [],
    refs: refs || [],
    head: refByName.main ? { ref: 'main', commit_id: refByName.main } : null,
    original: refByName['protected/original'] || null,
  };
}
