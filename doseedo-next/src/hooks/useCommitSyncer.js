/**
 * useCommitSyncer — push local sessionHistory commits up to the server
 * commit DAG so the History tab flips into useServer-true mode and the
 * Attest button on each row can reach a real server commit.
 *
 * Local commits live in `state.sessionHistory` (services/sessionHistory.js,
 * AppContext reducer). Server commits live behind
 * `POST /api/sessions/{sid}/commits` (commitsAPI.mintCommit).
 *
 * Behavior:
 *   • On first tick after mount, if the server already has commits for
 *     this session, we treat all known local commits as already-synced
 *     and only push commits made AFTER mount. This avoids duplicating
 *     desktop-pushed history when the web reopens an already-synced
 *     session.
 *   • If the server has no commits yet, all local commits get pushed in
 *     chronological (parent-first) order, each `parent_ids: [serverHead]`,
 *     with an inline `ref_update: { name: 'main', expected_commit_id: ... }`
 *     so the server-side `main` ref advances atomically with each commit.
 *   • POSTs are serialized through one Promise queue so parents resolve
 *     before children. On any failure the chain stops; the next local
 *     dispatch retries from where it left off.
 *   • After a successful push round, calls serverHistory.refresh() so the
 *     History tab re-renders with the new server rows.
 */
import { useEffect, useRef } from 'react';
import { mintCommit } from '../services/commitsAPI';

export function useCommitSyncer(sessionId, sessionHistory, serverHistory) {
  const pushedRef = useRef(new Set());      // local commit IDs already POSTed
  const headRef = useRef(null);             // server-side HEAD commit_id
  const queueRef = useRef(Promise.resolve());
  const initializedRef = useRef(false);

  // Keep server-head ref in sync with the latest read.
  const serverHeadId = serverHistory?.head?.commit_id || null;
  useEffect(() => { headRef.current = serverHeadId; }, [serverHeadId]);

  const refresh = serverHistory?.refresh;
  const serverStatus = serverHistory?.status;
  const serverCommitCount = serverHistory?.commits?.length || 0;

  useEffect(() => {
    if (!sessionId || !sessionHistory) return;
    // Wait for the initial server-side fetch to settle so we know whether
    // to seed the pushed-set or push-everything.
    if (serverStatus === 'loading' || serverStatus === 'idle') return;

    const commits = sessionHistory.commits || {};
    const branchHead = sessionHistory.refs?.[sessionHistory.currentBranch || 'main'];
    if (!branchHead) return;

    if (!initializedRef.current) {
      initializedRef.current = true;
      if (serverCommitCount > 0) {
        // Server already has history for this session — assume it
        // corresponds to current local commits (we can't dedup by id
        // since local + server use different namespaces). Mark all
        // currently-known local commits as already-pushed; only NEW
        // commits made after this mount get synced upward.
        Object.keys(commits).forEach((id) => pushedRef.current.add(id));
        return;
      }
      // Server is empty — fall through and push all local commits.
    }

    // Walk parents from local HEAD until we hit a pushed commit, reverse
    // so the chain is parent-first.
    const chain = [];
    let cur = branchHead;
    const seen = new Set();
    while (cur && !pushedRef.current.has(cur) && !seen.has(cur)) {
      seen.add(cur);
      const c = commits[cur];
      if (!c) break;
      chain.push(c);
      cur = (c.parentIds && c.parentIds[0]) || null;
      if (chain.length > 500) break;        // safety cap
    }
    chain.reverse();
    if (chain.length === 0) return;

    queueRef.current = queueRef.current.then(async () => {
      let pushedAny = false;
      for (const localCommit of chain) {
        if (pushedRef.current.has(localCommit.id)) continue;
        const expected = headRef.current;
        const parent_ids = expected ? [expected] : [];
        try {
          const row = await mintCommit(sessionId, {
            tree: localCommit.snapshot || {},
            parent_ids,
            label: localCommit.label || null,
            action_type: localCommit.actionType || null,
            author_origin: 'web',
            ref_update: { name: 'main', expected_commit_id: expected },
          });
          pushedRef.current.add(localCommit.id);
          headRef.current = row.id;
          pushedAny = true;
        } catch (err) {
          console.warn(
            `[commit-syncer] push failed for ${localCommit.id} (${localCommit.label}):`,
            err?.message || err,
          );
          break;                            // retry on next dispatch
        }
      }
      if (pushedAny && typeof refresh === 'function') {
        try { await refresh(); } catch { /* swallow */ }
      }
    });
  }, [sessionId, sessionHistory, serverStatus, serverCommitCount, refresh]);
}

export default useCommitSyncer;
