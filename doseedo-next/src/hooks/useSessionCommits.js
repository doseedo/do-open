/**
 * useSessionCommits — read server-side commit DAG for a session.
 *
 * Fetches commits + refs in parallel on mount and whenever sessionId
 * changes. Refetch is also exposed on the hook return so callers can
 * invalidate after minting a commit (or after a poll tick).
 *
 * Returns:
 *   { commits, refs, head, original, status, error, refresh }
 *
 * Status values:
 *   'idle'     — no sessionId provided
 *   'loading'  — initial fetch in flight
 *   'ok'      — last fetch succeeded
 *   'error'    — last fetch threw; `error` carries the message
 *
 * The hook deliberately does NOT cache to localStorage — that's the
 * existing sessionHistory module's job, and it carries different
 * (heavier) snapshots. This hook is the read-through to the server.
 */
import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchHistory } from '../services/commitsAPI';

export function useSessionCommits(sessionId) {
  const [state, setState] = useState({
    commits: [],
    refs: [],
    head: null,
    original: null,
    status: sessionId ? 'loading' : 'idle',
    error: null,
  });
  const inflightRef = useRef(null);

  const refresh = useCallback(async () => {
    if (!sessionId) {
      setState({ commits: [], refs: [], head: null, original: null, status: 'idle', error: null });
      return;
    }
    const ticket = Symbol('fetch');
    inflightRef.current = ticket;
    setState((s) => ({ ...s, status: 'loading', error: null }));
    try {
      const data = await fetchHistory(sessionId);
      if (inflightRef.current !== ticket) return; // raced; drop result
      setState({
        commits: data.commits || [],
        refs: data.refs || [],
        head: data.head || null,
        original: data.original || null,
        status: 'ok',
        error: null,
      });
    } catch (err) {
      if (inflightRef.current !== ticket) return;
      setState((s) => ({ ...s, status: 'error', error: err?.message || String(err) }));
    }
  }, [sessionId]);

  useEffect(() => { refresh(); }, [refresh]);

  return { ...state, refresh };
}
