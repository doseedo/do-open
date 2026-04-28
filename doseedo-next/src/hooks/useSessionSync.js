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

    // Tier A — localStorage hydration. Autosave (AppContext.js:1920) writes
    // the live project state to `session-{sid}` every 3s, so on hard refresh
    // we usually have a usable snapshot already. Dispatch it synchronously
    // before the server fetch so the studio paints instantly and never sits
    // blank during the network round-trip. The server load below still runs
    // as a freshness check and replaces this payload only when it's newer
    // (or the only source).
    let cached = null;
    try { cached = sessionService.loadSession(sid); } catch { /* ignore */ }
    const cachedHasBuses =
      cached?.state && Array.isArray(cached.state.buses) && cached.state.buses.length > 0;
    if (cachedHasBuses) {
      dispatch({
        type: 'LOAD_SESSION',
        payload: { ...cached.state, activeSessionId: sid },
      });
      console.log(
        `✅ Hydrated session ${sid} from localStorage — ` +
        `${cached.state.buses.length} buses (server refresh pending)`
      );
      // Keep the active-project pointer + last-synced key in sync with
      // the cache hit so a follow-up tab close before the server returns
      // doesn't lose the binding the next refresh needs.
      sessionService.setActiveProject(sid);
      sessionService.setLastSyncedSessionId(sid);
    }

    (async () => {
      try {
        const res = await sessionSyncAPI.fetchSessionState(sid, { shareToken });
        if (cancelled) return;

        const serverPayload = res?.state
          ? sessionSyncAPI.adaptDesktopStateToContext(res.state)
          : null;
        const serverHasBuses =
          !!serverPayload && Array.isArray(serverPayload.buses) && serverPayload.buses.length > 0;

        // Replace with server data when:
        //  - we have nothing on screen yet (no cache hit), OR
        //  - the server snapshot is strictly newer than the cached one.
        // Otherwise keep the local cache — protects offline edits whose
        // autosave landed locally before the cloud-save tier flushed.
        const serverUpdated = res?.updated_at ? Date.parse(res.updated_at) : NaN;
        const cachedAt = Number(cached?.timestamp) || 0;
        const serverIsNewer = Number.isFinite(serverUpdated) && serverUpdated > cachedAt;
        const shouldReplace = serverHasBuses && (!cachedHasBuses || serverIsNewer);

        if (shouldReplace) {
          // Attach the session UUID so the reducer can store it for the
          // edits producer to read (sessionEditsAPI routes by sessionId).
          serverPayload.activeSessionId = sid;

          // Hydrate persisted source-audio rows. Keys: track.metadata
          // .sourceAudioId → SessionAudio.id. We swap each track's
          // audioUrl from a (now-dead) blob: URL to a fresh presigned R2
          // GET so playback works after reload. Best-effort — if listing
          // fails the session still loads, just without source audio.
          try {
            const { listSessionAudio } = await import('../services/sessionAudioAPI');
            const rows = await listSessionAudio({ sessionId: sid, shareToken });
            if (Array.isArray(rows) && rows.length > 0) {
              const byId = new Map(rows.map((r) => [r.id, r]));
              for (const bus of serverPayload.buses || []) {
                for (const track of bus.tracks || []) {
                  const sid2 = track?.metadata?.sourceAudioId;
                  if (sid2 && byId.has(sid2)) {
                    const row = byId.get(sid2);
                    track.audioUrl = row.audio?.download_url || row.download_url;
                    track.metadata = track.metadata || {};
                    if (row.latent) {
                      track.metadata.latentUrl = row.latent.download_url;
                      track.metadata.sourceLatentSha = row.latent.sha256;
                      track.metadata.sourceLatentFrames = row.latent.n_frames;
                      track.metadata.sourceLatentVae = row.latent.vae_version;
                    }
                    if (row.midi) {
                      track.metadata.midiUrl = row.midi.download_url;
                      track.metadata.sourceMidiSha = row.midi.sha256;
                      track.metadata.sourceMidiNotes = row.midi.n_notes;
                    }
                  }
                }
              }
              console.log(`[session-sync] rehydrated ${rows.length} audio asset(s)`);
            }
          } catch (e) {
            console.warn('[session-sync] audio rehydrate failed:', e?.message || e);
          }

          dispatch({ type: 'LOAD_SESSION', payload: serverPayload });
          console.log(
            `✅ Hydrated session ${sid} from server — ${serverPayload.buses.length} buses, ` +
            `version ${res.version ?? 'unset'}`
          );
        } else if (!cachedHasBuses && !serverHasBuses) {
          console.warn(`[session-sync] session ${sid} has no state on server or in cache — opening blank`);
        } else if (cachedHasBuses && !shouldReplace) {
          console.log(`[session-sync] keeping localStorage cache for ${sid} (server empty or older)`);
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
        // Soft failure when the cache already painted: keep what's on
        // screen, drop the alert, let the 30s poll retry meta freshness.
        if (cachedHasBuses) {
          console.warn(
            `[session-sync] server fetch failed for ${sid} — keeping localStorage cache: ` +
            (err?.message || err)
          );
          hydratedSessionRef.current = { id: sid, version: null };
          // Strip the ?session= query param if it's still on the URL so
          // the effect doesn't re-fire on every subsequent navigation.
          if (location.search) navigate('/studio', { replace: true });
          return;
        }
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
