import React, { useEffect, useMemo, useState, useCallback } from 'react';
import liveTrackChainRegistry from '../../lib/liveTrackChainRegistry';
import { useApp } from '../../context/AppContext';
import {
  enqueueAddPlugin,
  enqueueRemovePlugin,
} from '../../services/sessionEditsAPI';
import { LOGIC_STOCK_PLUGINS, uniqueNames } from '../../lib/logicStockPlugins';
import styles from './LogicPluginRack.module.css';

/**
 * LogicPluginRack — bypass / Mix / add / remove UI for the Logic
 * plugins on a synced track.
 *
 * For each plugin in `track.logicPlugins`:
 *   - Plugin name + Bypass toggle (drives slot.setBypassed →
 *     PluginAdapter.editCallbacks → enqueueSetPluginBypass).
 *   - Mix slider when the mapping declares `wet_param_id`. Drives
 *     slot.setLogicParam(wetId, value) → enqueueSetPluginParam.
 *   - Remove button → enqueueRemovePlugin (server replays into Logic
 *     via the desktop's edit-dispatcher).
 * Plus a footer "Add plugin" picker → enqueueAddPlugin.
 *
 * The component is a thin presentation layer; all the state of record
 * lives on the slot (engine + mapping). Renders nothing when the track
 * has no logicPlugins or no live chain is registered (i.e. mappings
 * are missing and the bounce-cache fallback is in effect — surfacing
 * UI controls would be misleading because they'd have no audio effect).
 */
function LogicPluginRack({ track }) {
  const { state } = useApp();
  const [chain, setChain] = useState(null);
  const [, setTick] = useState(0);
  const [adding, setAdding] = useState(false);
  const [pluginToAdd, setPluginToAdd] = useState('');

  // The chain lives in the singleton registry (registered by
  // useAudioPlayback when it builds a live chain). We poll the
  // registry because chains are wired up async from playback start; a
  // one-shot useEffect would miss them. Polling at 500ms is cheap (the
  // registry lookup is a Map.get) and stops once we resolve a chain.
  useEffect(() => {
    if (!track?.id) {
      setChain(null);
      return undefined;
    }
    let cancelled = false;
    const probe = () => {
      if (cancelled) return;
      const c = liveTrackChainRegistry.get?.(track.id);
      if (c?.slots?.length) {
        setChain(c);
      } else if (!cancelled) {
        setTimeout(probe, 500);
      }
    };
    probe();
    return () => { cancelled = true; };
  }, [track?.id]);

  const logicPlugins = track?.logicPlugins || track?.metadata?.logicPlugins || [];
  const trackUuid = track?.uuid || track?.metadata?.uuid || null;
  const sessionId = state?.activeSessionId || null;

  // Autocomplete options: union of (a) Logic stock-plugin names from
  // logicStockPlugins.js — the canonical vocabulary the desktop
  // calibration harness understands — and (b) any plugin name the
  // adapter has actively loaded a mapping for. The free-text input
  // accepts anything outside this list too, so plugins we haven't
  // catalogued (3rd-party AUs, future Logic releases) still broadcast
  // correctly when typed.
  const autocompleteNames = useMemo(() => {
    const fromMappings = [];
    if (chain?.slots) {
      for (const s of chain.slots) {
        const m = s?.getMapping?.();
        const n = m?.plugin_name || m?.logic_plugin_name;
        if (n) fromMappings.push(n);
      }
    }
    return uniqueNames(LOGIC_STOCK_PLUGINS, fromMappings).sort(
      (a, b) => a.localeCompare(b)
    );
  }, [chain]);

  const handleAdd = useCallback(async () => {
    const name = (pluginToAdd || '').trim();
    if (!name || !sessionId || !trackUuid) return;
    setAdding(true);
    try {
      enqueueAddPlugin(sessionId, trackUuid, name);
      setPluginToAdd('');
    } finally {
      setAdding(false);
    }
  }, [pluginToAdd, sessionId, trackUuid]);

  const handleRemove = useCallback((slotIndex) => {
    if (!sessionId || !trackUuid || typeof slotIndex !== 'number') return;
    enqueueRemovePlugin(sessionId, trackUuid, slotIndex);
  }, [sessionId, trackUuid]);

  const hasPlugins = Array.isArray(logicPlugins) && logicPlugins.length > 0;
  const showRack = hasPlugins && chain?.slots?.length;

  // Even when no live chain is registered (mappings missing) we still
  // surface the Add/Remove panel so the user can drop a plugin and
  // have the desktop wire it up. We just hide the per-slot bypass /
  // Mix rows because they wouldn't drive anything.
  if (!hasPlugins && (!sessionId || !trackUuid)) return null;

  return (
    <div className={styles.rack}>
      <div className={styles.rackHeader}>
        <span><i className={`fa-solid fa-plug ${styles.icon}`}></i>Logic Plugins</span>
        {hasPlugins && !showRack && (
          <span className={styles.empty}>fallback (no mapping)</span>
        )}
      </div>

      {showRack && chain.slots.map((slot, idx) => {
        const lp = logicPlugins[idx] || {};
        return (
          <PluginSlotRow
            key={idx}
            slotIndex={idx}
            slot={slot}
            logicPlugin={lp}
            onChange={() => setTick((n) => n + 1)}
            onRemove={trackUuid && sessionId ? () => handleRemove(idx) : null}
          />
        );
      })}

      {trackUuid && sessionId && (
        <div className={styles.addRow}>
          <input
            type="text"
            list="logic-stock-plugins"
            value={pluginToAdd}
            onChange={(e) => setPluginToAdd(e.target.value)}
            placeholder="Add plugin…"
            disabled={adding}
            spellCheck={false}
            autoComplete="off"
            // Hitting Enter on the autocomplete row should add the plugin
            // (mirrors the Add button) — saves a click for keyboard users.
            onKeyDown={(e) => {
              if (e.key === 'Enter' && pluginToAdd && !adding) {
                e.preventDefault();
                handleAdd();
              }
            }}
          />
          <datalist id="logic-stock-plugins">
            {autocompleteNames.map((n) => (
              <option key={n} value={n} />
            ))}
          </datalist>
          <button
            type="button"
            className={styles.addButton}
            onClick={handleAdd}
            disabled={adding || !pluginToAdd}
          >
            Add
          </button>
        </div>
      )}
    </div>
  );
}

function PluginSlotRow({ slotIndex, slot, logicPlugin, onChange, onRemove }) {
  // Local state so the UI re-renders on toggle without subscribing to
  // every slot's internal state. The slot's setBypassed / setLogicParam
  // are the source of truth; this state is just a render hint.
  const [bypassed, setBypassed] = useState(() => {
    try { return slot?.isBypassed?.() || false; } catch { return false; }
  });

  const mapping = slot?.getMapping?.();
  const wetParamId = typeof mapping?.wet_param_id === 'number'
    ? mapping.wet_param_id : null;

  const initialWet = useMemo(() => {
    if (wetParamId == null) return null;
    const lpRow = (logicPlugin?.parameters || [])
      .find((p) => Number(p?.id) === wetParamId);
    if (lpRow == null) return 1;
    return Number(lpRow.value);
  }, [wetParamId, logicPlugin]);
  const [wet, setWet] = useState(initialWet);

  const handleBypassToggle = useCallback(() => {
    if (!slot?.setBypassed) return;
    const next = !bypassed;
    slot.setBypassed(next);
    setBypassed(next);
    onChange?.();
  }, [slot, bypassed, onChange]);

  const handleWetChange = useCallback((e) => {
    const v = Number(e.target.value);
    setWet(v);
    if (slot?.setLogicParam && wetParamId != null) {
      slot.setLogicParam(wetParamId, v);
    }
    onChange?.();
  }, [slot, wetParamId, onChange]);

  return (
    <div className={styles.row}>
      <div className={styles.rowHeader}>
        <span className={styles.pluginName}>
          {slot?.pluginName || logicPlugin?.plugin_name || `Plugin ${slotIndex + 1}`}
        </span>
        {slot?.bypassSupported !== false && (
          <label className={styles.bypassToggle}>
            <input type="checkbox" checked={bypassed} onChange={handleBypassToggle} />
            <span>Bypass</span>
          </label>
        )}
        {onRemove && (
          <button
            type="button"
            className={styles.removeButton}
            onClick={onRemove}
            title="Remove plugin"
          >
            ✕
          </button>
        )}
      </div>
      {wetParamId != null && (
        <div className={styles.mixRow}>
          <span>Mix</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.001}
            value={wet ?? 1}
            onChange={handleWetChange}
          />
          <span>{wet != null ? wet.toFixed(2) : '—'}</span>
        </div>
      )}
    </div>
  );
}

export default LogicPluginRack;
