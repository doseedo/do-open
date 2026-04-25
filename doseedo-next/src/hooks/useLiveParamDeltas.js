/**
 * useLiveParamDeltas — A5 frontend subscriber.
 *
 * Listens for `param_delta` messages on the existing Doseedo chat WebSocket
 * (`/_chat/ws`, managed by useAgentWebSocket and friends — but we DON'T
 * couple to that hook directly, since the registry pattern lets us run on
 * any page that has a chat WS open). On each delta, looks up the matching
 * live PluginAdapter slot in `liveTrackChainRegistry` and forwards the
 * value to `slot.setLogicParam(param_id, value)`. PluginAdapter handles
 * the curve fit + normalization internally — we are pure plumbing.
 *
 * Wire shape from chat_server.py (relayed verbatim from doo_hook):
 *   {
 *     type: "param_delta",
 *     track_index: number,   // 0-based; -1 if doo_hook couldn't resolve
 *     ginstid:     number,   // Logic GInstID
 *     plugin_slot: number,   // 0-based insert slot
 *     param_id:    number,   // AudioUnitParameterID (Logic-native)
 *     value:       number,   // raw AU param value (Logic-native)
 *     ts:          number    // unix-ms when doo_hook emitted
 *   }
 *
 * Fallback story:
 *   - dylib not injected / older dylib without delta channel →
 *     chat_server.py's relay loop never connects, no `param_delta`
 *     messages arrive, this hook is a no-op. The next regular full-sync
 *     brings the project state up to date.
 *   - delta arrives for a track that hasn't been spliced into a live
 *     chain (A2 hasn't run, feature flag off, mapping missing) →
 *     `registry.get` returns null, we log once-per-(track,slot) and ignore.
 *   - delta arrives mid-sync (chain disposed, new chain not yet registered)
 *     → same null-path; the next delta or sync will land in the new chain.
 *
 * Throttling is owned by doo_hook (≤60Hz per param) — this hook does no
 * additional throttling so an automation lane being bounced back from the
 * dylib still hits the engine at 60Hz.
 */

import { useEffect, useRef } from 'react';
import liveTrackChainRegistry from '../lib/liveTrackChainRegistry';

// Module-level dedup so React StrictMode double-invocations don't spam logs.
const _droppedKeys = new Set();

function _logDropped(trackKey, slotIndex, paramId) {
  const key = `${trackKey}:${slotIndex}:${paramId}`;
  if (_droppedKeys.has(key)) return;
  _droppedKeys.add(key);
  if (typeof console !== 'undefined' && console.debug) {
    console.debug(
      `[liveParamDelta] dropped — no live chain for track=${trackKey} slot=${slotIndex} param=${paramId} ` +
      '(falling back to next full sync)'
    );
  }
}

/**
 * Resolve a delta payload to a live slot. Tries (in order):
 *   1. ginstid → registry.getByGinstid
 *   2. track_index → registry.getByTrackIndex
 *   3. legacy: registry.get(trackKey, slotIndex) — for callers that
 *      register by trackId matching one of these wires.
 * Returns the slot (with setLogicParam) or null if no live chain.
 */
function _resolveSlot(delta) {
  const slotIndex = delta.plugin_slot;
  if (typeof slotIndex !== 'number') return null;

  if (typeof delta.ginstid === 'number') {
    const entry = liveTrackChainRegistry.getByGinstid(delta.ginstid);
    if (entry?.slots && slotIndex >= 0 && slotIndex < entry.slots.length) {
      return entry.slots[slotIndex] || null;
    }
  }
  if (typeof delta.track_index === 'number' && delta.track_index >= 0) {
    const entry = liveTrackChainRegistry.getByTrackIndex(delta.track_index);
    if (entry?.slots && slotIndex >= 0 && slotIndex < entry.slots.length) {
      return entry.slots[slotIndex] || null;
    }
  }
  return null;
}

/**
 * Apply a single param_delta to its registered live slot, if one exists.
 * Exported for unit tests + the `wsRef` path below; either is callable
 * standalone with no React context.
 */
export function applyParamDelta(delta) {
  if (!delta || delta.type !== 'param_delta') return false;
  const slot = _resolveSlot(delta);
  if (!slot || typeof slot.setLogicParam !== 'function') {
    const trackKey = (typeof delta.ginstid === 'number')
      ? `gid:${delta.ginstid}`
      : `idx:${delta.track_index}`;
    _logDropped(trackKey, delta.plugin_slot, delta.param_id);
    return false;
  }
  try {
    slot.setLogicParam(delta.param_id, delta.value);
    return true;
  } catch (e) {
    if (typeof console !== 'undefined' && console.warn) {
      console.warn('[liveParamDelta] setLogicParam failed:', e?.message || e);
    }
    return false;
  }
}

/**
 * Hook: attach a `param_delta` listener to a WebSocket. Pass a ref or a
 * direct WebSocket. Returns nothing — the hook is fire-and-forget.
 *
 * @param {React.MutableRefObject<WebSocket|null> | WebSocket | null | undefined} wsOrRef
 *        The shared chat WS (e.g. wsRef from useAgentWebSocket). Hook
 *        re-attaches when the underlying ws.current changes.
 */
export function useLiveParamDeltas(wsOrRef) {
  const handlerRef = useRef(null);

  useEffect(() => {
    const ws = (wsOrRef && typeof wsOrRef === 'object' && 'current' in wsOrRef)
      ? wsOrRef.current
      : wsOrRef;
    if (!ws || typeof ws.addEventListener !== 'function') return undefined;

    // Don't replace any existing onmessage handler — chain via
    // addEventListener so multiple consumers (chat, deltas, future
    // collab) coexist on the same socket.
    const handler = (event) => {
      if (typeof event?.data !== 'string') return;
      let payload;
      try { payload = JSON.parse(event.data); } catch { return; }
      if (!payload || payload.type !== 'param_delta') return;
      applyParamDelta(payload);
    };
    handlerRef.current = handler;
    ws.addEventListener('message', handler);

    return () => {
      ws.removeEventListener('message', handler);
      handlerRef.current = null;
    };
    // wsOrRef.current changes don't trigger re-runs (refs are stable);
    // pass the ws/ref itself as a dep — useEffect identity-compares.
  }, [wsOrRef]);
}

export default useLiveParamDeltas;
