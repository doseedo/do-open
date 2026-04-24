/**
 * useSessionSync — the web half of the desktop → web session-sync loop.
 *
 * Responsibilities:
 *   1. On /studio?session=<uuid>, fetch the session's state.json from the
 *      auth-service and LOAD_SESSION into the AppContext reducer. Then
 *      strip the ?session= query param so this effect doesn't re-fire on
 *      every subsequent location change within /studio.
 *   2. Poll GET /api/sessions/{id} every 30s while /studio is active and a
 *      session is loaded; if the server-side `gcs_object_version` advances
 *      past what we loaded, prompt the user to reload. Dedup by version so
 *      rapid upstream writes only nag once.
 *
 * This lives as a hook (not inline in AppContent) so the App.js diff for
 * enabling the feature is a single import + call site, and the sync.sh
 * mirror on every dev-server start doesn't regress the effect's internals.
 */
import { useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import * as sessionService from '../services/sessionService';
import * as sessionSyncAPI from '../services/sessionSyncAPI';

export function useSessionSync(dispatch) {
  const location = useLocation();
  const navigate = useNavigate();

  const hydratedSessionIdRef = useRef(null);
  const hydratedSessionRef = useRef(null); // {id, version}
  const lastStalePromptRef = useRef(null);

  useEffect(() => {
    if (location.pathname !== '/studio') return;
    const params = new URLSearchParams(location.search);
    let sid = params.get('session');
    const shareToken = params.get('share_token') || params.get('t') || undefined;
    // Fallback: if the early-access gate redirected /studio?session=<uuid>
    // to /studio and dropped the query param, use the last-hydrated session
    // id from localStorage. Only applies on the FIRST mount (guarded by
    // hydratedSessionIdRef) so subsequent in-app navigation to /studio
    // without a param doesn't re-hydrate. Skip the fallback when a
    // shareToken is in the URL — share-token URLs are always explicit.
    if (!sid && !shareToken && hydratedSessionIdRef.current == null) {
      sid = sessionService.getLastSyncedSessionId();
    }
    if (!sid || hydratedSessionIdRef.current === sid) return;

    hydratedSessionIdRef.current = sid;
    let cancelled = false;

    (async () => {
      try {
        const res = await sessionSyncAPI.fetchSessionState(sid, { shareToken });
        if (cancelled) return;

        if (!res || res.state == null) {
          console.warn(`[session-sync] session ${sid} has no state yet — opening blank`);
        } else {
          const payload = sessionSyncAPI.adaptDesktopStateToContext(res.state);
          if (payload) {
            dispatch({ type: 'LOAD_SESSION', payload });
            console.log(
              `✅ Hydrated session ${sid} — ${payload.buses?.length || 0} buses, ` +
              `version ${res.version ?? 'unset'}`
            );
          } else {
            console.warn(`[session-sync] session ${sid} state had no buses — opening blank`);
          }
        }

        hydratedSessionRef.current = { id: sid, version: res?.version ?? null };
        lastStalePromptRef.current = null;
        sessionService.setActiveProject(sid);
        // Remember the last successfully hydrated remote session so the
        // effect above can fall back to it when the early-access gate
        // eats the ?session= query param.
        sessionService.setLastSyncedSessionId(sid);
        navigate('/studio', { replace: true });
      } catch (err) {
        if (cancelled) return;
        console.error(`[session-sync] failed to load session ${sid}:`, err);
        if (err?.status === 401 || err?.status === 403) {
          alert('Sign in to open this session.');
        } else if (err?.status === 404) {
          alert('Session not found or has been removed.');
          // The fallback key pointed at a deleted session — clear it so
          // the next /studio mount opens a blank editor instead of
          // alerting again.
          sessionService.clearLastSyncedSessionId();
          navigate('/studio', { replace: true });
        } else {
          alert(`Could not load session: ${err?.message || 'network error'}`);
        }
      }
    })();

    return () => { cancelled = true; };
  }, [location.pathname, location.search, dispatch, navigate]);

  useEffect(() => {
    if (location.pathname !== '/studio') return;

    const tick = async () => {
      const cur = hydratedSessionRef.current;
      if (!cur?.id) return;
      try {
        const meta = await sessionSyncAPI.fetchSessionMeta(cur.id);
        const remote = meta?.gcs_object_version ?? null;
        const known = cur.version ?? null;
        if (remote != null && known != null && remote > known &&
            lastStalePromptRef.current !== remote) {
          lastStalePromptRef.current = remote;
          const reload = window.confirm(
            'This session was updated elsewhere. Reload to pick up the latest changes?\n\n' +
            'OK = reload  (discards unsaved local edits)\n' +
            'Cancel = keep your local version'
          );
          if (reload) {
            hydratedSessionIdRef.current = null;
            navigate(`/studio?session=${encodeURIComponent(cur.id)}`);
          } else {
            hydratedSessionRef.current = { id: cur.id, version: remote };
          }
        }
      } catch (err) {
        if (process.env.NODE_ENV === 'development') {
          console.debug('[session-sync] poll failed:', err?.message || err);
        }
      }
    };

    const intervalId = setInterval(tick, 30000);
    return () => clearInterval(intervalId);
  }, [location.pathname, navigate]);
}

export default useSessionSync;
