/**
 * editStreamApply.js — pure op-translation layer for inbound peer
 * edits. Mounted by useEditStream.js but kept React-free so unit tests
 * can drive `applyOp` directly with synthetic state + a mock dispatch.
 *
 * Each branch handles one inbound op. The handler:
 *   1. Mutates AppContext state via `dispatch` so UI reflects the edit.
 *   2. Mutates the live PluginAdapter slot (when one exists in the
 *      liveTrackChainRegistry) so audio updates immediately.
 *   3. Suppresses the broadcast on the slot (`broadcast:false`) so
 *      applying an inbound peer edit doesn't bounce back through the
 *      producer.
 *
 * Tracks are addressed by stable Logic UUID — the producer wire shape
 * — so concurrent reorders on the other client don't address the wrong
 * track.
 */

import liveTrackChainRegistry from '../lib/liveTrackChainRegistry.js';

export function applyOp(edit, dispatch, state) {
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

  if (op === 'set_plugin_param') {
    const trackUuid = args?.track_uuid;
    const slot = args?.slot;
    const paramId = args?.param_id;
    const value = args?.value;
    if (typeof trackUuid !== 'string' || typeof slot !== 'number'
        || typeof paramId !== 'number' || typeof value !== 'number') return;
    _updateLogicPluginParam(state, dispatch, trackUuid, slot, paramId, value);
    _applyToLiveSlot(trackUuid, slot, (s) => {
      s.setLogicParam?.(paramId, value, { broadcast: false });
    });
    return;
  }

  if (op === 'set_plugin_params_batch') {
    const trackUuid = args?.track_uuid;
    const slot = args?.slot;
    const params = args?.params;
    if (typeof trackUuid !== 'string' || typeof slot !== 'number'
        || !Array.isArray(params)) return;
    _updateLogicPluginParamBatch(state, dispatch, trackUuid, slot, params);
    _applyToLiveSlot(trackUuid, slot, (s) => {
      for (const p of params) {
        if (typeof p?.param_id === 'number' && typeof p?.value === 'number') {
          s.setLogicParam?.(p.param_id, p.value, { broadcast: false });
        }
      }
    });
    return;
  }

  if (op === 'set_plugin_bypass') {
    const trackUuid = args?.track_uuid;
    const slot = args?.slot;
    const bypassed = args?.bypassed;
    if (typeof trackUuid !== 'string' || typeof slot !== 'number'
        || typeof bypassed !== 'boolean') return;
    _setLogicPluginBypass(state, dispatch, trackUuid, slot, bypassed);
    _applyToLiveSlot(trackUuid, slot, (s) => {
      s.setBypassed?.(bypassed, { broadcast: false });
    });
    return;
  }

  if (op === 'add_plugin') {
    const trackUuid = args?.track_uuid;
    const pluginName = args?.plugin_name;
    if (typeof trackUuid !== 'string' || typeof pluginName !== 'string'
        || !pluginName) return;
    _addLogicPlugin(state, dispatch, trackUuid, pluginName);
    // The live chain doesn't rebuild mid-playback — the new plugin
    // appears in state, and on the next play() invocation
    // useAudioPlayback's chain-build path picks it up. Documented in
    // the rack UX.
    return;
  }

  if (op === 'remove_plugin') {
    const trackUuid = args?.track_uuid;
    const slot = args?.slot;
    if (typeof trackUuid !== 'string' || typeof slot !== 'number') return;
    _removeLogicPlugin(state, dispatch, trackUuid, slot);
    return;
  }

  // Unknown op — ignore. New ops land here as we add handlers.
}

// ── State lookup ───────────────────────────────────────────────────────

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

function _findTrackAndPlugins(state, trackUuid) {
  const t = findTrackByUuid(state, trackUuid);
  if (!t) return null;
  const plugins = Array.isArray(t.logicPlugins) ? t.logicPlugins
                  : Array.isArray(t.metadata?.logicPlugins) ? t.metadata.logicPlugins
                  : [];
  return { track: t, plugins };
}

// ── State mutations ────────────────────────────────────────────────────

function _updateLogicPluginParam(state, dispatch, trackUuid, slot, paramId, value) {
  const found = _findTrackAndPlugins(state, trackUuid);
  if (!found) return;
  const newPlugins = found.plugins.map((p, i) => {
    if (i !== slot) return p;
    const oldParams = Array.isArray(p?.parameters) ? p.parameters : [];
    let touched = false;
    const newParams = oldParams.map((q) => {
      if (Number(q?.id) !== paramId) return q;
      touched = true;
      return { ...q, value };
    });
    if (!touched) newParams.push({ id: paramId, value });
    return { ...p, parameters: newParams };
  });
  dispatch({
    type: 'UPDATE_TRACK_LOGIC_PLUGINS',
    payload: { trackUuid, logicPlugins: newPlugins },
  });
}

function _updateLogicPluginParamBatch(state, dispatch, trackUuid, slot, params) {
  const found = _findTrackAndPlugins(state, trackUuid);
  if (!found) return;
  const map = new Map();
  for (const p of params) {
    if (typeof p?.param_id === 'number' && typeof p?.value === 'number') {
      map.set(p.param_id, p.value);
    }
  }
  if (map.size === 0) return;
  const newPlugins = found.plugins.map((p, i) => {
    if (i !== slot) return p;
    const old = Array.isArray(p?.parameters) ? p.parameters : [];
    const seen = new Set();
    const merged = old.map((q) => {
      const id = Number(q?.id);
      if (map.has(id)) {
        seen.add(id);
        return { ...q, value: map.get(id) };
      }
      return q;
    });
    for (const [id, value] of map.entries()) {
      if (!seen.has(id)) merged.push({ id, value });
    }
    return { ...p, parameters: merged };
  });
  dispatch({
    type: 'UPDATE_TRACK_LOGIC_PLUGINS',
    payload: { trackUuid, logicPlugins: newPlugins },
  });
}

function _setLogicPluginBypass(state, dispatch, trackUuid, slot, bypassed) {
  const found = _findTrackAndPlugins(state, trackUuid);
  if (!found) return;
  const newPlugins = found.plugins.map((p, i) =>
    i === slot ? { ...p, bypassed } : p
  );
  dispatch({
    type: 'UPDATE_TRACK_LOGIC_PLUGINS',
    payload: { trackUuid, logicPlugins: newPlugins },
  });
}

function _addLogicPlugin(state, dispatch, trackUuid, pluginName) {
  const found = _findTrackAndPlugins(state, trackUuid);
  if (!found) return;
  // Stub entry — desktop dispatcher will replay the insert into Logic
  // and the next session sync (or live observer if shipped) fills in
  // the real plugin_id + parameters. The web UI shows the entry
  // immediately so the operator sees what the peer just did.
  const stub = {
    plugin_id: 0,
    plugin_name: pluginName,
    parameters: [],
    _pending: true,
  };
  dispatch({
    type: 'UPDATE_TRACK_LOGIC_PLUGINS',
    payload: { trackUuid, logicPlugins: [...found.plugins, stub] },
  });
}

function _removeLogicPlugin(state, dispatch, trackUuid, slot) {
  const found = _findTrackAndPlugins(state, trackUuid);
  if (!found) return;
  if (slot < 0 || slot >= found.plugins.length) return;
  const newPlugins = found.plugins.filter((_, i) => i !== slot);
  dispatch({
    type: 'UPDATE_TRACK_LOGIC_PLUGINS',
    payload: { trackUuid, logicPlugins: newPlugins },
  });
}

// ── Live slot dispatch ─────────────────────────────────────────────────

function _applyToLiveSlot(trackUuid, slotIndex, fn) {
  // Direct uuid lookup via the registry's _byUuid index. Returns
  // silently when no chain exists — chain-less tracks just update
  // React state and audio reflects on next playback.
  try {
    const entry = liveTrackChainRegistry.getByUuid?.(trackUuid);
    if (!entry || !Array.isArray(entry.slots)) return;
    if (slotIndex < 0 || slotIndex >= entry.slots.length) return;
    const slot = entry.slots[slotIndex];
    if (!slot) return;
    try { fn(slot); } catch (_) { /* swallow per-slot errors */ }
  } catch (_) { /* registry missing — bail */ }
}
