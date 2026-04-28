import React, { useEffect, useState, useCallback } from 'react';
import liveTrackChainRegistry from '../../lib/liveTrackChainRegistry';

/**
 * LogicPluginRack — minimal UI for the Logic plugins on a synced track.
 *
 * For each plugin in `track.logicPlugins`, renders:
 *   - Plugin name + bypass toggle (calls slot.setBypassed → broadcasts
 *     via PluginAdapter.editCallbacks → enqueueSetPluginBypass).
 *   - Mix slider when the mapping declared `wet_param_id`. Drives
 *     slot.setLogicParam(wetId, value) → enqueueSetPluginParam.
 *
 * The component is intentionally small and unstyled — the bigger value
 * is wiring the controls into the running PluginAdapter slots so any
 * UI we add later can drive the same setBypassed/setLogicParam calls
 * and get sync-broadcast for free. Renders nothing when:
 *   - no track is selected
 *   - the track has no logicPlugins (or feature flag is off)
 *   - no live chain is registered for the track (i.e. mappings are
 *     missing and the bounce-cache fallback is in effect — bypass /
 *     wet sliders would be UI-only with no audio effect)
 */
function LogicPluginRack({ track }) {
  const [chain, setChain] = useState(null);
  const [, setTick] = useState(0);

  // The chain lives in the singleton registry (registered by
  // useAudioPlayback when it builds a live chain). We poll the registry
  // because chains are wired up async from playback start; a one-shot
  // useEffect would miss them. Polling at 500ms is cheap — the registry
  // lookup is a Map.get — and stops once we resolve a chain.
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
  if (!Array.isArray(logicPlugins) || logicPlugins.length === 0) return null;
  if (!chain || !chain.slots) return null;

  return (
    <div style={{ marginTop: 12, padding: '8px 0', borderTop: '1px solid #333' }}>
      <div style={{ fontWeight: 600, marginBottom: 6 }}>Logic Plugins</div>
      {chain.slots.map((slot, idx) => {
        const lp = logicPlugins[idx] || {};
        return (
          <PluginSlotRow
            key={idx}
            slotIndex={idx}
            slot={slot}
            logicPlugin={lp}
            onChange={() => setTick((n) => n + 1)}
          />
        );
      })}
    </div>
  );
}

function PluginSlotRow({ slotIndex, slot, logicPlugin, onChange }) {
  // Local state so the UI re-renders on toggle without subscribing to
  // every slot's internal state. The slot's setBypassed / setLogicParam
  // are the source of truth; this state is just a render hint.
  const [bypassed, setBypassed] = useState(() => {
    try { return slot?.isBypassed?.() || false; } catch { return false; }
  });

  const mapping = slot?.getMapping?.();
  const wetParamId = typeof mapping?.wet_param_id === 'number'
    ? mapping.wet_param_id : null;
  const initialWet = (() => {
    if (wetParamId == null) return null;
    const lpRow = (logicPlugin?.parameters || [])
      .find((p) => Number(p?.id) === wetParamId);
    if (lpRow == null) return null;
    return Number(lpRow.value);
  })();
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
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ flex: 1, fontSize: 13 }}>
          {slot?.pluginName || logicPlugin?.plugin_name || `Plugin ${slotIndex + 1}`}
        </span>
        {slot?.bypassSupported !== false && (
          <label style={{ display: 'inline-flex', alignItems: 'center',
                          gap: 4, fontSize: 12, cursor: 'pointer' }}>
            <input type="checkbox" checked={bypassed} onChange={handleBypassToggle} />
            <span>Bypass</span>
          </label>
        )}
      </div>
      {wetParamId != null && (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8,
                      paddingLeft: 12, marginTop: 4 }}>
          <span style={{ fontSize: 11, opacity: 0.8 }}>Mix</span>
          <input
            type="range"
            min={0} max={1} step={0.001}
            value={wet ?? 1}
            onChange={handleWetChange}
            style={{ flex: 1 }}
          />
          <span style={{ fontSize: 11, opacity: 0.8, minWidth: 32, textAlign: 'right' }}>
            {wet != null ? wet.toFixed(2) : '—'}
          </span>
        </div>
      )}
    </div>
  );
}

export default LogicPluginRack;
