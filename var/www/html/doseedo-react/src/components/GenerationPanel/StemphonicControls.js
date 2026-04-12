import React, { useState, useEffect, useCallback, useMemo } from 'react';
import * as generationAPI from '../../services/generationAPI';

/**
 * Two pieces of stemphonic UI that live inside the regular Instruments
 * panel — placed between the instrument grid and the Input section:
 *
 *   <CheckpointDropdown />  — picks the active model. "DoPerformer" is the
 *                             default and means "use the original do-v1
 *                             instruments flow". Picking any other ckpt
 *                             routes generation through the stemphonic
 *                             backend.
 *
 *   <TimbreVariantCycler /> — given the current instrumentSubgroup, lets
 *                             the user audition + cycle through the 10
 *                             timbre variants for that instrument. Only
 *                             rendered when stemphonic mode is active.
 */

export const DOPERFORMER_ID = 'doperformer';
const DOPERFORMER_OPTION = {
  id: DOPERFORMER_ID,
  label: 'DoPerformer (original)',
  has_midi: true,
  downloaded: true,
};

export function CheckpointDropdown({ value, onChange, disabled }) {
  const [options, setOptions] = useState([DOPERFORMER_OPTION]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await generationAPI.listStemphonicCheckpoints();
        if (cancelled) return;
        setOptions([DOPERFORMER_OPTION, ...(data.checkpoints || [])]);
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const handleChange = useCallback(async (e) => {
    const id = e.target.value;
    onChange(id);
    if (id === DOPERFORMER_ID) return;
    setLoading(true);
    setError(null);
    try {
      await generationAPI.switchStemphonicCheckpoint(id);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, [onChange]);

  return (
    <div style={{ marginTop: '12px' }}>
      <label style={{ display: 'block', fontSize: '11px', opacity: 0.75, marginBottom: '4px' }}>
        Model checkpoint
      </label>
      <select
        value={value || DOPERFORMER_ID}
        onChange={handleChange}
        disabled={disabled || loading}
        style={{
          width: '100%',
          padding: '6px',
          background: '#222',
          color: '#fff',
          border: '1px solid #444',
          borderRadius: '4px',
          fontSize: '12px',
        }}
      >
        {options.map((c) => (
          <option key={c.id} value={c.id}>
            {c.label}
            {!c.downloaded && c.id !== DOPERFORMER_ID ? ' ⬇' : ''}
            {c.has_midi === false ? ' · no MIDI hook' : ''}
          </option>
        ))}
      </select>
      {loading && <div style={{ fontSize: '11px', opacity: 0.7, marginTop: '4px' }}>Loading checkpoint…</div>}
      {error && <div style={{ fontSize: '11px', color: '#f66', marginTop: '4px' }}>{error}</div>}
    </div>
  );
}

export function TimbreVariantCycler({ instrumentSubgroup, value, onChange, disabled }) {
  const [timbres, setTimbres] = useState([]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const data = await generationAPI.listStemphonicTimbres();
        if (!cancelled) setTimbres(data.timbres || []);
      } catch (e) {
        // Silent — variants are optional sugar.
      }
    })();
    return () => { cancelled = true; };
  }, []);

  // Try exact match, then fuzzy fallback (e.g. "violin" matches "violin",
  // "ensemble_strings" → "violin" etc.)
  const current = useMemo(() => {
    if (!instrumentSubgroup) return null;
    const exact = timbres.find((t) => t.key === instrumentSubgroup);
    if (exact) return exact;
    return timbres.find((t) => instrumentSubgroup.includes(t.key) || t.key.includes(instrumentSubgroup)) || null;
  }, [timbres, instrumentSubgroup]);

  const nVariants = current?.variants?.length || 0;
  const variantIdx = Math.max(0, Math.min(value ?? 0, Math.max(0, nVariants - 1)));
  const variant = current?.variants?.[variantIdx];

  if (!current) return null;

  const cycle = (delta) => {
    if (nVariants === 0) return;
    onChange((variantIdx + delta + nVariants) % nVariants);
  };

  return (
    <div style={{ marginTop: '8px' }}>
      <label style={{ display: 'block', fontSize: '11px', opacity: 0.75, marginBottom: '4px' }}>
        Timbre · {current.label} · variant {variantIdx + 1}/{nVariants}
      </label>
      <div style={{ display: 'flex', alignItems: 'center', gap: '4px', padding: '4px', background: '#1a1a1a', border: '1px solid #333', borderRadius: '3px' }}>
        <button
          type="button"
          onClick={() => cycle(-1)}
          disabled={disabled || nVariants <= 1}
          title="Previous variant"
          style={{ padding: '4px 10px', background: '#333', color: '#fff', border: '1px solid #555', borderRadius: '3px', cursor: 'pointer', fontSize: '14px' }}
        >
          ‹
        </button>
        {variant && (
          <audio
            key={`${current.key}-${variantIdx}`}
            src={variant.audio_url}
            controls
            preload="none"
            style={{ flex: 1, height: '28px' }}
          />
        )}
        <button
          type="button"
          onClick={() => cycle(1)}
          disabled={disabled || nVariants <= 1}
          title="Next variant"
          style={{ padding: '4px 10px', background: '#333', color: '#fff', border: '1px solid #555', borderRadius: '3px', cursor: 'pointer', fontSize: '14px' }}
        >
          ›
        </button>
      </div>
    </div>
  );
}
