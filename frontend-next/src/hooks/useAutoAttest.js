/**
 * useAutoAttest — when enabled, automatically self-attest every server
 * commit on this session that has zero attestations yet.
 *
 * Per-commit visual states the History tab renders:
 *   • attesting (grey)  — request+confirm in flight, OR confirmed by
 *                         this user but the publisher hasn't anchored
 *                         on Polygon yet (polygon_status != 'confirmed')
 *   • attested ✓ (green) — polygon_status === 'confirmed'
 *
 * Flow per commit: requestAttestation(self) → confirmAttestation. The
 * commit's attestation_total moves 0 → 1 and attestation_confirmed
 * moves to 1, which trips L2 (single contributor) and the server-side
 * publisher takes it to chain. We refresh server-history after each
 * successful confirm so the row re-renders without user action.
 *
 * No-ops when:
 *   • disabled (checkbox off)
 *   • currentUsername is empty (no Clerk user)
 *   • commit already has attestations (attestation_total > 0)
 *   • commit is already on chain (polygon_status === 'confirmed')
 *   • the request is already in flight for this commit
 */
import { useEffect, useRef, useState, useCallback } from 'react';
import {
  requestAttestation,
  confirmAttestation,
} from '../services/attestationsAPI';
import { fetchMyProfile } from '../services/sessionSyncAPI';

export function useAutoAttest({ sessionId, commits, currentUsername, enabled, refresh }) {
  const inflightRef = useRef(new Set());
  const failedRef = useRef(new Set());     // don't retry forever on hard errors
  const queueRef = useRef(Promise.resolve());
  const [tick, setTick] = useState(0);     // bumps to re-render `attesting` set

  // Resolve the auth-service's canonical username once. Clerk's
  // `user.username` lives in a different namespace; the confirm
  // endpoint 403s on `att.contributor_username != user.username`.
  // Cached in state so we don't re-fetch on every render.
  const [serverUsername, setServerUsername] = useState(null);
  const meFetchedRef = useRef(false);
  useEffect(() => {
    if (!enabled || !sessionId || meFetchedRef.current) return;
    meFetchedRef.current = true;
    fetchMyProfile()
      .then((me) => { if (me?.username) setServerUsername(me.username); })
      .catch((err) => {
        meFetchedRef.current = false;
        console.warn('[auto-attest] could not fetch /api/profiles/me:', err?.message || err);
      });
  }, [enabled, sessionId]);

  // Prefer the server-resolved username; fall back to whatever the caller
  // passed (still useful if /me is unavailable but the two happen to match).
  const effectiveUsername = serverUsername || currentUsername || '';

  // Poll-while-pending: once a commit has attestation_total > 0 but
  // polygon_status !== 'confirmed', the publisher is the only thing that
  // can flip it. Refetch server history every 15s so the pill turns green
  // when the on-chain anchor lands, instead of staying grey forever.
  const pendingAnchor = Array.isArray(commits)
    ? commits.some((c) => (c.attestation_total || 0) > 0 && c.polygon_status !== 'confirmed' && c.polygon_status !== 'final')
    : false;
  useEffect(() => {
    if (!enabled || !sessionId || !pendingAnchor || typeof refresh !== 'function') return;
    const id = setInterval(() => { refresh().catch(() => { /* ignore */ }); }, 15_000);
    return () => clearInterval(id);
  }, [enabled, sessionId, pendingAnchor, refresh]);

  const isAttesting = useCallback((commitId) => {
    return inflightRef.current.has(commitId);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tick]);

  useEffect(() => {
    if (!enabled || !sessionId || !effectiveUsername) return;
    if (!Array.isArray(commits) || commits.length === 0) return;

    const targets = commits.filter((c) => {
      if (!c?.id) return false;
      if (failedRef.current.has(c.id)) return false;
      if (inflightRef.current.has(c.id)) return false;
      if ((c.attestation_total || 0) > 0) return false;          // someone already attested
      if ((c.polygon_status === 'confirmed' || c.polygon_status === 'final')) return false;        // already on chain
      return true;
    });
    if (targets.length === 0) return;

    for (const c of targets) inflightRef.current.add(c.id);
    setTick((t) => t + 1);

    queueRef.current = queueRef.current.then(async () => {
      let didAny = false;
      for (const c of targets) {
        try {
          const att = await requestAttestation(sessionId, c.id, effectiveUsername, null);
          if (att?.id) {
            await confirmAttestation(sessionId, att.id);
            didAny = true;
          }
        } catch (err) {
          if (err?.status !== 409) {
            failedRef.current.add(c.id);
            console.warn(
              `[auto-attest] commit ${c.id} (${c.label}):`,
              err?.message || err,
            );
          }
        } finally {
          inflightRef.current.delete(c.id);
        }
      }
      setTick((t) => t + 1);
      if (didAny && typeof refresh === 'function') {
        try { await refresh(); } catch { /* swallow */ }
      }
    });
  }, [sessionId, effectiveUsername, enabled, commits, refresh]);

  return { isAttesting };
}

export default useAutoAttest;
