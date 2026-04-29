/*
 * StudioDevFX — themed FX-chain editor for /studio-dev.
 *
 * Replaces the original components/FXView. A horizontal strip of FX
 * modules (reverb, delay, EQ, compressor, fades) backed by the selected
 * track's fx object. Values written via UPDATE_TRACK_PROPS so the
 * production audio engine (tunaFX via useAudioPlayback) picks them up.
 */
import React, { useMemo } from 'react';
import { useApp } from '../../context/AppContext';

const MODULES = [
  { key: 'reverb', title: 'Reverb', params: [
      { k: 'reverb',     label: 'Wet',  min: 0, max: 1,  step: 0.01, def: 0 },
      { k: 'reverbSize', label: 'Size', min: 0, max: 1,  step: 0.01, def: 0.5 },
      { k: 'reverbDamp', label: 'Damp', min: 0, max: 1,  step: 0.01, def: 0.5 },
    ] },
  { key: 'delay', title: 'Delay', params: [
      { k: 'delay',     label: 'Mix',  min: 0, max: 1, step: 0.01, def: 0 },
      { k: 'delayTime', label: 'Time', min: 0, max: 1, step: 0.01, def: 0.25 },
      { k: 'delayFeedback', label: 'Fbk', min: 0, max: 0.95, step: 0.01, def: 0.3 },
    ] },
  { key: 'eq', title: 'EQ', params: [
      { k: 'eqLow',  label: 'Low',  min: -12, max: 12, step: 0.5, def: 0 },
      { k: 'eqMid',  label: 'Mid',  min: -12, max: 12, step: 0.5, def: 0 },
      { k: 'eqHigh', label: 'High', min: -12, max: 12, step: 0.5, def: 0 },
    ] },
  { key: 'comp', title: 'Comp', params: [
      { k: 'compThresh', label: 'Thr',  min: -48, max: 0,  step: 0.5, def: -12 },
      { k: 'compRatio',  label: 'Ratio', min: 1,  max: 20, step: 0.1, def: 2 },
      { k: 'compMakeup', label: 'Gain',  min: 0,  max: 12, step: 0.1, def: 0 },
    ] },
  { key: 'fade', title: 'Fades', params: [
      { k: 'fadeIn',  label: 'In',  min: 0, max: 4, step: 0.05, def: 0 },
      { k: 'fadeOut', label: 'Out', min: 0, max: 4, step: 0.05, def: 0 },
    ] },
];

function Knob({ param, value, onChange }) {
  const pct = param.max === param.min ? 0 : ((value ?? param.def) - param.min) / (param.max - param.min);
  const hasNegRange = param.min < 0;
  const centered = hasNegRange ? (value ?? param.def) === 0 : false;
  return (
    <label className="sd-fx-knob">
      <span className="sd-fx-knob-label">{param.label}</span>
      <div className="sd-fx-knob-outer" data-centered={centered}>
        <div className="sd-fx-knob-fill"
             style={{ background: `conic-gradient(var(--hifi-accent) ${pct * 360}deg, var(--hifi-surface-2) ${pct * 360}deg)` }} />
        <div className="sd-fx-knob-value">
          {(value ?? param.def).toFixed(param.step < 0.5 ? 2 : 1)}
        </div>
      </div>
      <input type="range" min={param.min} max={param.max} step={param.step}
             value={value ?? param.def}
             onChange={(e) => onChange(parseFloat(e.target.value))}
             className="sd-fx-knob-range" />
    </label>
  );
}

export default function StudioDevFX() {
  const { state, dispatch } = useApp();
  const selectedTrack = state.selectedTrack;
  const busId = useMemo(() => {
    if (!selectedTrack) return null;
    for (const bus of state.buses || []) {
      if ((bus.tracks || []).some((t) => t.id === selectedTrack.id)) return bus.id;
    }
    return null;
  }, [state.buses, selectedTrack]);

  const fx = selectedTrack?.fx || {};
  const setFxParam = (k, v) => {
    if (!selectedTrack) return;
    const nextFx = { ...fx, [k]: v };
    dispatch({
      type: 'UPDATE_TRACK_PROPS',
      payload: { trackId: selectedTrack.id, fx: nextFx },
    });
  };

  if (!selectedTrack) {
    return (
      <div className="sd-fx-empty">
        <div className="sd-wave-empty-eyebrow">— the fx chain —</div>
        <div className="sd-wave-empty-title">Pick a track.</div>
        <div className="sd-wave-empty-body">
          Select a track to see its reverb, delay, EQ, compression, and fades.
        </div>
      </div>
    );
  }

  return (
    <div className="sd-fx">
      <div className="sd-midi-toolbar">
        <div className="sd-midi-title">
          <span className="sd-midi-meta">FX CHAIN</span>
          <span className="sd-midi-name" style={{ marginLeft: 6 }}>{selectedTrack.name || selectedTrack.id}</span>
        </div>
        <div className="sd-midi-spacer" />
        <div className="sd-midi-group">
          <button className="sd-midi-btn" onClick={() => {
            // Reset all knobs on this track.
            const cleared = {};
            MODULES.forEach((m) => m.params.forEach((p) => { cleared[p.k] = p.def; }));
            dispatch({ type: 'UPDATE_TRACK_PROPS', payload: { trackId: selectedTrack.id, fx: cleared } });
          }}>Reset</button>
        </div>
      </div>

      <div className="sd-fx-strip">
        {MODULES.map((m) => {
          const bypassKey = `${m.key}_bypass`;
          const bypassed = !!fx[bypassKey];
          return (
            <div key={m.key} className={`sd-fx-module ${bypassed ? 'bypassed' : ''}`}>
              <div className="sd-fx-module-title">
                <span>{m.title}</span>
                <button
                  className={`sd-fx-bypass ${bypassed ? 'on' : ''}`}
                  onClick={() => setFxParam(bypassKey, !bypassed)}
                  title={bypassed ? 'Enable module' : 'Bypass module'}
                >{bypassed ? 'Off' : 'On'}</button>
              </div>
              <div className="sd-fx-module-knobs">
                {m.params.map((p) => (
                  <Knob key={p.k} param={p} value={fx[p.k]} onChange={(v) => setFxParam(p.k, v)} />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
