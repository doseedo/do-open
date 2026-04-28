/**
 * useCollabPresence — WebSocket-driven presence tracker for the studio.
 *
 * Connects to the auth-service collab relay at:
 *   wss://doseedo-api.fly.dev/ws/collab/{sessionId}
 *     ?api_key=dsk_live_...&source_id=<uuid>&username=<display>&share_token=<optional>
 *
 * The relay (auth-service/app/routers/collab.py) sends:
 *   - presence_init  → full peer snapshot at connect
 *   - peer_joined    → someone arrived
 *   - peer_left      → someone disconnected
 *   - state_sync, awareness, cursor — IGNORED here (handled by other channels)
 *
 * This hook is INTENTIONALLY split from useEditStream (SSE, edits-only).
 * Two channels, two concerns: presence is bidirectional and ephemeral;
 * edits are unidirectional (server → client) and authoritative-replayed.
 *
 * Source-of-identity rules:
 *   - source_id is per-TAB (sessionStorage) so two tabs of one user appear
 *     as two peers in the avatar row, matching real-world expectation.
 *   - api_key is per-USER (localStorage) — minted once via the existing
 *     POST /api/keys flow, then reused. The key is named "Web Collab" and
 *     has a 1-day expiry so the user's key list doesn't grow unbounded.
 *     If a cached key is rejected (deleted from another device, expired),
 *     we mint a fresh one transparently.
 *
 * Reconnect: 1s, 2s, 5s, 10s capped. Resets on a clean (1000) close.
 */
import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE   = process.env.NEXT_PUBLIC_API_BASE_URL  || '';
const AUTH_ORIGIN = process.env.NEXT_PUBLIC_AUTH_ORIGIN || 'https://doseedo-api.fly.dev';
// localStorage key — caches the dsk_live_ token across tabs/reloads. The
// suffix is intentionally not user-scoped because a single-user client
// reuses one key; on Clerk sign-out the AppShell clears localStorage.
const _LS_KEY = 'doo_collab_api_key_v1';
const _SS_SRC_ID_PREFIX = 'doo_collab_src_'; // per-session source_id


function _uuid() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID();
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

async function _clerkBearer() {
  if (typeof window === 'undefined') return null;
  if (typeof window.__clerkGetToken === 'function') {
    try { return await window.__clerkGetToken(); } catch { return null; }
  }
  try { return window.localStorage?.getItem('token'); } catch { return null; }
}

/**
 * Get a usable dsk_live_ API key for the current Clerk user.
 *
 * Strategy:
 *   1. Look in localStorage for a cached key.
 *   2. If absent, POST /api/keys with the Clerk JWT to mint one. The
 *      response includes the full key ONCE — stash it.
 *   3. We can't validate the cached key without a side effect, so we trust
 *      it until the WebSocket close code 4001 says otherwise; then we
 *      clear the cache and let the next call re-mint.
 *
 * Returns null if Clerk hasn't loaded a token yet, or the mint failed.
 */
async function _getOrMintApiKey({ forceRefresh = false } = {}) {
  if (typeof window === 'undefined') return null;
  if (!forceRefresh) {
    try {
      const cached = window.localStorage?.getItem(_LS_KEY);
      if (cached && cached.startsWith('dsk_')) return cached;
    } catch { /* ignore — fall through to mint */ }
  } else {
    try { window.localStorage?.removeItem(_LS_KEY); } catch {}
  }
  const bearer = await _clerkBearer();
  if (!bearer) return null;
  try {
    const res = await fetch(`${API_BASE}/api/keys`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
        Authorization: `Bearer ${bearer}`,
      },
      credentials: 'include',
      body: JSON.stringify({
        name: 'Web Collab',
        scopes: ['doo'],
        expires_in_days: 365,
      }),
    });
    if (!res.ok) {
      console.warn('[useCollabPresence] mint api key failed:', res.status, res.statusText);
      return null;
    }
    const data = await res.json();
    const key = data?.key;
    if (typeof key === 'string' && key.startsWith('dsk_')) {
      try { window.localStorage?.setItem(_LS_KEY, key); } catch {}
      return key;
    }
    return null;
  } catch (e) {
    console.warn('[useCollabPresence] mint api key error:', e?.message || e);
    return null;
  }
}

function _getSourceId(sessionId) {
  if (typeof window === 'undefined') return _uuid();
  const k = `${_SS_SRC_ID_PREFIX}${sessionId}`;
  try {
    const existing = window.sessionStorage?.getItem(k);
    if (existing) return existing;
  } catch { /* fall through */ }
  const fresh = _uuid();
  try { window.sessionStorage?.setItem(k, fresh); } catch {}
  return fresh;
}

function _wsBase() {
  // Auth-service base; .fly.dev fallback. Convert https→wss, http→ws.
  const origin = AUTH_ORIGIN.replace(/^http:/, 'ws:').replace(/^https:/, 'wss:');
  return origin;
}

function _readShareTokenFromUrl() {
  if (typeof window === 'undefined') return '';
  try {
    const p = new URLSearchParams(window.location.search);
    return p.get('share_token') || p.get('t') || '';
  } catch { return ''; }
}

/**
 * @param {string|null} sessionId — UUID PK of the active session, or null/undefined
 *                                  to disable the hook.
 * @param {string} username       — display name shown to peers.
 * @returns {{ peers: Array, connected: boolean, selfSourceId: string|null }}
 *   peers — [{ source_id, username, user_id, role, joined_at }, ...] (excludes self)
 */
export function useCollabPresence(sessionId, username) {
  const [peers, setPeers] = useState([]);   // Map<source_id, peerInfo> as array
  const [connected, setConnected] = useState(false);
  const peersMapRef = useRef(new Map());
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const cancelledRef = useRef(false);
  const selfSourceIdRef = useRef(null);

  const flushPeers = useCallback(() => {
    setPeers(Array.from(peersMapRef.current.values()));
  }, []);

  useEffect(() => {
    if (!sessionId) return undefined;
    if (typeof WebSocket === 'undefined') return undefined;

    cancelledRef.current = false;
    let attempt = 0;
    const backoffSchedule = [1000, 2000, 5000, 10000]; // last value sticky

    const sourceId = _getSourceId(sessionId);
    selfSourceIdRef.current = sourceId;

    const open = async () => {
      if (cancelledRef.current) return;

      const apiKey = await _getOrMintApiKey({
        forceRefresh: attempt > 0 && attempt % 2 === 0,
      });
      if (cancelledRef.current) return;
      if (!apiKey) {
        // No Clerk token yet, or mint failed. Try again later.
        scheduleReconnect();
        return;
      }

      const shareToken = _readShareTokenFromUrl();
      const params = new URLSearchParams({
        api_key: apiKey,
        source_id: sourceId,
        username: (username || '').slice(0, 64),
      });
      if (shareToken) params.set('share_token', shareToken);

      const url = `${_wsBase()}/ws/collab/${encodeURIComponent(sessionId)}?${params.toString()}`;
      let ws;
      try {
        ws = new WebSocket(url);
      } catch (e) {
        console.warn('[useCollabPresence] WebSocket ctor failed:', e?.message || e);
        scheduleReconnect();
        return;
      }
      wsRef.current = ws;

      ws.onopen = () => {
        if (cancelledRef.current) return;
        attempt = 0;
        setConnected(true);
      };

      ws.onmessage = (evt) => {
        if (cancelledRef.current) return;
        let msg;
        try { msg = JSON.parse(evt.data); } catch { return; }
        const t = msg?.type;
        if (t === 'presence_init') {
          const next = new Map();
          const peers = msg.peers || {};
          for (const [pid, pdata] of Object.entries(peers)) {
            if (pid === sourceId) continue;
            next.set(pid, { source_id: pid, ...(pdata || {}) });
          }
          peersMapRef.current = next;
          flushPeers();
        } else if (t === 'peer_joined') {
          const pid = msg.source_id;
          if (!pid || pid === sourceId) return;
          peersMapRef.current.set(pid, { source_id: pid, ...(msg.peer || {}) });
          flushPeers();
        } else if (t === 'peer_left') {
          const pid = msg.source_id;
          if (!pid) return;
          if (peersMapRef.current.delete(pid)) flushPeers();
        }
        // state_sync / awareness / etc: ignored — not our channel.
      };

      ws.onerror = () => {
        // Don't tear down here — wait for `close` so we have the close code.
      };

      ws.onclose = (ev) => {
        if (cancelledRef.current) return;
        setConnected(false);
        if (ev?.code === 4001) {
          // Invalid API key — likely revoked. Force a re-mint on next attempt.
          try { window.localStorage?.removeItem(_LS_KEY); } catch {}
        } else if (ev?.code === 4003) {
          // Live collab disabled feature flag. Don't spam reconnects.
          console.warn('[useCollabPresence] live collab disabled by feature flag');
          return;
        } else if (ev?.code === 4004) {
          // Room full (8 peers). Reconnect with longer delay; someone may leave.
          attempt = Math.max(attempt, 3);
        }
        scheduleReconnect();
      };
    };

    const scheduleReconnect = () => {
      if (cancelledRef.current) return;
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      const delay = backoffSchedule[Math.min(attempt, backoffSchedule.length - 1)];
      attempt += 1;
      reconnectTimerRef.current = setTimeout(open, delay);
    };

    open();

    return () => {
      cancelledRef.current = true;
      if (reconnectTimerRef.current) {
        clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      try { wsRef.current?.close(1000, 'unmount'); } catch {}
      wsRef.current = null;
      peersMapRef.current = new Map();
      selfSourceIdRef.current = null;
      setPeers([]);
      setConnected(false);
    };
  }, [sessionId, username, flushPeers]);

  return { peers, connected, selfSourceId: selfSourceIdRef.current };
}

export default useCollabPresence;
