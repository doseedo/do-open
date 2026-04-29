/**
 * attestationsAPI — request / confirm / dispute attestations on a commit.
 *
 * Level 2 (the on-chain Polygon publish gate) is reached when every named
 * contributor confirms their attribution AND zero have disputed. Disputes
 * are revocable: a contributor who later changes their mind calls
 * /confirm to clear the dispute.
 *
 * The list endpoint is owner-scoped (commit author sees all attestations
 * on their session). Confirm/dispute endpoints are contributor-scoped
 * (only the named contributor can act on an attestation).
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
  const url = new URL(`${API_BASE}${path}`, window.location.origin);
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

export async function listAttestations(sessionId, commitId) {
  if (!sessionId || !commitId) return [];
  return _request(`/api/sessions/${sessionId}/commits/${commitId}/attestations`);
}

export async function requestAttestation(sessionId, commitId, contributorUsername, contributorRole = null) {
  if (!sessionId || !commitId || !contributorUsername) {
    throw new Error('requestAttestation requires sessionId, commitId, contributorUsername');
  }
  return _request(`/api/sessions/${sessionId}/commits/${commitId}/attestations`, {
    method: 'POST',
    body: { contributor_username: contributorUsername, contributor_role: contributorRole },
  });
}

export async function confirmAttestation(sessionId, attestationId) {
  return _request(`/api/sessions/${sessionId}/attestations/${attestationId}/confirm`, {
    method: 'POST',
  });
}

export async function disputeAttestation(sessionId, attestationId, reason) {
  if (!reason || !reason.trim()) throw new Error('disputeAttestation requires a reason');
  return _request(`/api/sessions/${sessionId}/attestations/${attestationId}/dispute`, {
    method: 'POST',
    body: { reason: reason.trim() },
  });
}

export async function getAttestationLevel(sessionId, commitId) {
  return _request(`/api/sessions/${sessionId}/commits/${commitId}/attestation_level`);
}
