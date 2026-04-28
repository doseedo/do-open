/**
 * liveTrackChainRegistry — singleton registry that bridges A2's orchestrator
 * splice (where live PluginAdapter chains are built and connected to the
 * Web Audio graph) and A5's live param-delta hook (which needs to look up
 * the running chain by track id + slot index to mutate plugin params in
 * real time, without re-synthesizing the whole project).
 *
 * Why a top-level singleton (not React Context):
 *   - The doo_hook delta WS messages arrive *outside* React's render cycle —
 *     we want O(1) lookup from any layer (hooks, services, devtools shims),
 *     without forcing every consumer to be a child of a Provider.
 *   - A2's chain owner (`useAudioPlayback`) constructs chains during play()
 *     and disposes them on stop()/seek(); the registry mirrors that lifetime
 *     1:1 — register on splice, clear on dispose. No stale chains survive a
 *     transport reset because A2 calls `clear()` before re-registering.
 *   - The registry stores ONLY references to slots that A2 already created
 *     — it never instantiates anything. If A2 hasn't run (no live chains
 *     for the project, feature-flag off, etc.), `get()` returns null and
 *     callers fall back to the next full-sync.
 *
 * Wire shape (A2's orchestrator splice should call):
 *   register(trackId, {
 *     trackIndex,            // 0-based index in the bus's track list
 *     ginstid,               // Logic GInstID for direct lookup by delta msg
 *     slots: [pluginSlot0, pluginSlot1, ...],  // each slot has setLogicParam
 *     dispose,               // optional — invoked from clear() if present
 *   })
 *
 * The slots array indexes positionally, matching `plugin_slot` in delta
 * messages from doo_hook. PluginAdapter.buildTrackChain returns an object
 * already shaped like this; A2 just needs to forward it.
 */

let _registry = new Map(); // key: trackId (any) → entry
let _byGinstid = new Map(); // key: ginstid (number) → trackId
let _byTrackIndex = new Map(); // key: trackIndex (number) → trackId
let _byUuid = new Map(); // key: trackUuid (lowercased hex) → trackId

const _listeners = new Set();

function _emitChange() {
  for (const fn of _listeners) {
    try { fn(); } catch (_) { /* never let a listener crash a register */ }
  }
}

/**
 * Register a live track chain. Idempotent — re-registering the same trackId
 * replaces the previous entry without disposing it (A2 owns disposal).
 */
export function register(trackId, chain) {
  if (trackId == null || !chain) return;
  _registry.set(trackId, chain);
  if (typeof chain.ginstid === 'number') _byGinstid.set(chain.ginstid, trackId);
  if (typeof chain.trackIndex === 'number') _byTrackIndex.set(chain.trackIndex, trackId);
  if (typeof chain.trackUuid === 'string' && chain.trackUuid) {
    _byUuid.set(chain.trackUuid.toLowerCase(), trackId);
  }
  _emitChange();
}

/**
 * Look up a live chain entry by stable Logic UUID. Used by
 * useEditStream's inbound plugin-op handlers — those address tracks by
 * uuid (matching the producer wire shape) and need the chain to apply
 * the audio-side update. Returns null when the uuid wasn't seen at
 * register time (chain-less tracks).
 */
export function getByUuid(trackUuid) {
  if (typeof trackUuid !== 'string' || !trackUuid) return null;
  const tid = _byUuid.get(trackUuid.toLowerCase());
  if (tid == null) return null;
  return _registry.get(tid) || null;
}

/**
 * Look up a live PluginAdapter slot by (trackId|trackIndex|ginstid) +
 * slotIndex. Returns the slot (with setLogicParam) or null if any layer
 * is missing — callers ignore null and let the next sync correct things.
 */
export function get(trackKey, slotIndex) {
  let entry = _registry.get(trackKey);
  if (!entry && typeof trackKey === 'number') {
    // Try gid first (matches the doo_hook delta wire shape), then
    // fall back to index — gid is more specific than index.
    const tIdByGid = _byGinstid.get(trackKey);
    if (tIdByGid != null) entry = _registry.get(tIdByGid);
    if (!entry) {
      const tIdByIdx = _byTrackIndex.get(trackKey);
      if (tIdByIdx != null) entry = _registry.get(tIdByIdx);
    }
  }
  if (!entry || !Array.isArray(entry.slots)) return null;
  // No slotIndex → caller wants the chain entry, not a specific slot.
  // (LogicPluginRack and useEditStream's inbound handlers both need this.)
  if (slotIndex === undefined || slotIndex === null) return entry;
  if (slotIndex < 0 || slotIndex >= entry.slots.length) return null;
  return entry.slots[slotIndex] || null;
}

/**
 * Convenience alias: return the chain entry (or null) without the
 * slot-index overload. Equivalent to `get(trackKey)` but reads more
 * obviously at the call site when callers want the whole entry.
 */
export function getChain(trackKey) {
  return get(trackKey);
}

/**
 * Look up an entry by ginstid. Used by the delta hook before it knows the
 * trackId. Returns the registered chain or null.
 */
export function getByGinstid(ginstid) {
  if (typeof ginstid !== 'number') return null;
  const trackId = _byGinstid.get(ginstid);
  if (trackId == null) return null;
  return _registry.get(trackId) || null;
}

/**
 * Look up an entry by track_index. Mirror of getByGinstid for the case
 * where the doo_hook can't resolve a ginstid → trackIndex (returns -1).
 */
export function getByTrackIndex(trackIndex) {
  if (typeof trackIndex !== 'number') return null;
  const trackId = _byTrackIndex.get(trackIndex);
  if (trackId == null) return null;
  return _registry.get(trackId) || null;
}

/**
 * Drop all chains. A2's orchestrator calls this on stop()/seek()/unmount
 * — the registry never owns disposal, but we forward to dispose() if the
 * caller embedded one in the chain (handy for unit tests).
 */
export function clear() {
  for (const entry of _registry.values()) {
    if (typeof entry?.dispose === 'function') {
      try { entry.dispose(); } catch (_) { /* noop */ }
    }
  }
  _registry = new Map();
  _byGinstid = new Map();
  _byTrackIndex = new Map();
  _byUuid = new Map();
  _emitChange();
}

/**
 * Drop a single track's chain — called when A2 disposes a chain mid-flight
 * (e.g. user toggles solo and the lane goes silent). Does NOT call dispose;
 * A2 owns lifetime.
 */
export function unregister(trackId) {
  const entry = _registry.get(trackId);
  if (!entry) return;
  _registry.delete(trackId);
  if (typeof entry.ginstid === 'number') _byGinstid.delete(entry.ginstid);
  if (typeof entry.trackIndex === 'number') _byTrackIndex.delete(entry.trackIndex);
  if (typeof entry.trackUuid === 'string' && entry.trackUuid) {
    _byUuid.delete(entry.trackUuid.toLowerCase());
  }
  _emitChange();
}

/**
 * Subscribe to registry changes. Used by useLiveParamDeltas so the hook
 * can flush any deltas it queued before the chain was registered (rare —
 * race between the doo_hook firing pre-register and A2 finishing the
 * splice). Returns an unsubscribe fn.
 */
export function subscribe(fn) {
  _listeners.add(fn);
  return () => _listeners.delete(fn);
}

/** Test helper — never call in app code. */
export function _debugSnapshot() {
  return {
    size: _registry.size,
    trackIds: Array.from(_registry.keys()),
    byGinstid: Array.from(_byGinstid.entries()),
    byTrackIndex: Array.from(_byTrackIndex.entries()),
  };
}

export default {
  register,
  unregister,
  get,
  getChain,
  getByGinstid,
  getByTrackIndex,
  getByUuid,
  clear,
  subscribe,
  _debugSnapshot,
};
