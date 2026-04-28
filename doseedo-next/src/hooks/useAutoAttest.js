/**
 * useAutoAttest — when enabled, automatically self-attest every server
 * commit on this session that has zero attestations yet. Each commit
 * cycles through the visual states the History tab renders:
 *
 *   • attesting (grey)      — request+confirm in flight, OR confirmed by
 *                             this user but the publisher hasn't anchored
 *                             on Polygon yet (polygon_status != 'confirmed')
 *   • attested ✓ (green)    — polygon_status === 'confirmed'
 *
 * Flow per commit: requestAttestation(self) → confirmAttestation. The
 * commit's attestation_total moves from 0 → 1 and attestation_confirmed
 * moves to 1, which trips L2 (single-contributor) and the server-side
 * publisher takes it to chain. We refresh the server-history view after
 * each successful confirm so the row re-renders without the user clicking
 * anything.
 *
 * No-ops when:
 *   • disabled (checkbox unchecked)
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

export function useAutoAttest({ sessionId, commits, currentUsername, enabled, refresh }) {
  const inflightRef = useRef(new Set());
  const failedRef = useRef(new Set()); // don't retry forever on hard errors
  const queueRef = useRef(Promise.resolve());
  const [tick, setTick] = useState(0);  // bump to re-render `attesting` set

  const isAttesting = useCallback((commitId) => {
    return inflightRef.current.has(commitId);
  }, [tick]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!enabled || !sessionId || !currentUsername) return;
    if (!Array.isArray(commits) || commits.length === 0) return;

    const targets = commits.filter((c) => {
      if (!c?.id) return false;
      if (failedRef.current.has(c.id)) return false;
      if (inflightRef.current.has(c.id)) return false;
      // Already attested by someone — skip; only auto-attest the empty ones.
      if ((c.attestation_total || 0) > 0) return false;
      // Already on chain — definitely skip.
      if (c.polygon_status === 'confirmed') return false;
      return true;
    });

    if (targets.length === 0) return;

    for (const c of targets) inflightRef.current.add(c.id);
    setTick((t) => t + 1);

    queueRef.current = queueRef.current.then(async () => {
      let didAny = false;
      for (const c of targets) {
        try {
          const att = await requestAttestation(sessionId, c.id, currentUsername, null);
          if (att?.id) {
            await confirmAttestation(sessionId, att.id);
            didAny = true;
          }
        } catch (err) {
          // 409 (already exists) is benign — refresh will pick it up. Hard
          // failures (4xx from auth, 5xx, network) get parked so we don't
          // retry-storm on a known-bad commit.
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
  }, [sessionId, currentUsername, enabled, commits, refresh]);

  return { isAttesting };
}

export default useAutoAttest;
