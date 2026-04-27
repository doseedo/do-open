/**
 * useEditStream — subscribe to /api/sessions/{sid}/edits/stream and apply
 * inbound semantic edits to AppContext state.
 *
 * Symmetric to the desktop's edit_consumer: each `edit` event from the
 * server is an op the OTHER client (or this client's other tab/desktop
 * pair) just wrote. We translate the semantic op into a reducer dispatch
 * so the UI reflects it sub-second.
 *
 * Echo suppression
 * ----------------
 * The producer (sessionEditsAPI.js) stamps each outbound edit with a
 * `client_op_id`. We track the last N ids in a Set; if the SSE stream
 * emits an event with one we just produced, we drop it. The server
 * already does the unique-per-(session, client_op_id) check, so the
 * incoming event is necessarily our own write.
 *
 * The hook deliberately uses the browser's native EventSource. It
 * reconnects automatically and tracks `since` so a brief disconnect
 * doesn't drop events.
 */
import { useEffect, useRef } from 'react';
import { useApp } from '../context/AppContext';
import { recordOutboundOpId, isOutboundOpId } from '../services/sessionEditsAPI';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';

export function useEditStream(sessionId) {
  const { state, dispatch } = useApp();
  // Latest cursor we've applied. Survives reconnects so the catch-up
  // replay only sends rows we actually missed.
  const cursorRef = useRef(0);
  const stateRef = useRef(state);
  useEffect(() => { stateRef.current = state; }, [state]);

  useEffect(() => {
    if (!sessionId) return undefined;
    if (typeof EventSource === 'undefined') return undefined;

    let es = null;
    let cancelled = false;
    let backoffMs = 1000;

    const open = () => {
      if (cancelled) return;
      const url = new URL(`${API_BASE}/api/sessions/${sessionId}/edits/stream`, window.location.origin);
      url.searchParams.set('since', String(cursorRef.current));
      es = new EventSource(url.toString(), { withCredentials: true });

      es.addEventListener('edit', (ev) => {
        let data;
        try { data = JSON.parse(ev.data); } catch { return; }
        if (typeof data.cursor === 'number' && data.cursor > cursorRef.current) {
          cursorRef.current = data.cursor;
        }
        // Drop our own writes — the producer already updated local state
        // when the user moved the slider.
        if (data.client_op_id && isOutboundOpId(data.client_op_id)) return;
        applyOp(data, dispatch, stateRef.current);
      });
      es.addEventListener('ping', () => { /* keepalive */ });
      es.onopen = () => { backoffMs = 1000; };
      es.onerror = () => {
        es?.close(); es = null;
        if (cancelled) return;
        // Reconnect with exponential backoff capped at 30s. EventSource
        // also auto-reconnects, but doing it ourselves means we control
        // the cursor query param so the catch-up replay is tight.
        setTimeout(open, backoffMs);
        backoffMs = Math.min(backoffMs * 2, 30000);
      };
    };

    open();
    return () => {
      cancelled = true;
      try { es?.close(); } catch {}
      es = null;
    };
  }, [sessionId, dispatch]);
}

// ── Op → reducer translation ────────────────────────────────────────────────

function applyOp(edit, dispatch, state) {
  const { op, args } = edit || {};
  if (!op) return;

  if (op === 'set_volume_v2') {
    const trackUuid = args?.track_uuid;
    const value = args?.value;
    if (typeof trackUuid !== 'string' || typeof value !== 'number') return;
    const track = findTrackByUuid(state, trackUuid);
    if (!track) return;
    dispatch({
      type: 'UPDATE_TRACK',
      payload: { trackId: track.id, busId: track._busId || null, updates: { gain: value }, skipHistory: true },
    });
    return;
  }

  if (op === 'set_volume') {
    // v1 fallback — channel is positional. Best-effort: use logicTrackIndex
    // matching, which is also positional.
    const channel = args?.channel;
    const value = args?.value;
    if (typeof channel !== 'number' || typeof value !== 'number') return;
    const track = findTrackByLogicIndex(state, channel - 1);
    if (!track) return;
    dispatch({
      type: 'UPDATE_TRACK',
      payload: { trackId: track.id, busId: track._busId || null, updates: { gain: value }, skipHistory: true },
    });
    return;
  }

  // Unknown op — ignore. New ops land here as we add handlers.
}

function findTrackByUuid(state, uuid) {
  const norm = (uuid || '').toLowerCase();
  for (const bus of state.buses || []) {
    for (const t of bus.tracks || []) {
      if ((t.uuid || '').toLowerCase() === norm) return { ...t, _busId: bus.id };
    }
  }
  return null;
}

function findTrackByLogicIndex(state, idx) {
  for (const bus of state.buses || []) {
    for (const t of bus.tracks || []) {
      if (t.logicTrackIndex === idx) return { ...t, _busId: bus.id };
    }
  }
  return null;
}
