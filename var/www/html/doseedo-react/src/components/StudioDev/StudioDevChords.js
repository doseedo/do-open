/*
 * StudioDevChords — themed chord-grid overlay for /studio-dev.
 *
 * Pinned inside the canvas area (not shown when mode = 'fx'). Shows
 * state.chords as a row of per-beat cells with editable chord symbols.
 * Click empty cell → add chord (opens inline picker). Click existing →
 * edit. Right-click → delete. Dispatches UPDATE_CHORD / DELETE_CHORD to
 * match the production reducer.
 */
import React, { useMemo, useState } from 'react';
import { useApp } from '../../context/AppContext';
import useChordRegen from '../../hooks/useChordRegen';

const ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const QUALS = ['', 'm', '7', 'maj7', 'm7', 'sus4', 'sus2', 'dim', 'aug', '9'];
const BARS_VISIBLE = 8;

export default function StudioDevChords() {
  const { state, dispatch } = useApp();
  const [editingBeat, setEditingBeat] = useState(null);
  const chords = state.chords || {};
  const runChordRegen = useChordRegen();
  // Helper: dispatch a chord edit + kick off smart per-stem regen so any
  // analyzed stems overlapping the bar get repitched to match the new chord.
  const setChord = (beatIndex, newChord) => {
    const oldChord = chords[beatIndex] || '';
    dispatch({ type: 'UPDATE_CHORD', payload: { beatIndex, chord: newChord } });
    runChordRegen({ beatIndex, oldChord, newChord }).catch((e) =>
      console.warn('[studio-dev] chord regen failed:', e?.message || e));
  };

  const currentBeat = useMemo(() => {
    const bpm = state.bpm || 120;
    const spb = 60 / bpm;
    return Math.floor((state.playheadPosition || 0) / spb);
  }, [state.playheadPosition, state.bpm]);

  const BEATS_PER_BAR = state.beatsPerBar || 4;
  const cells = Array.from({ length: BARS_VISIBLE * BEATS_PER_BAR }, (_, i) => i);

  return (
    <div className="sd-chords">
      <div className="sd-chords-head">
        <span className="sd-midi-kv-k">Chords</span>
        <span className="sd-midi-meta">{state.beatsPerBar || 4}/{state.meterDenominator || 4} · {BARS_VISIBLE} bars</span>
        <div className="sd-midi-spacer" />
        <button className="sd-midi-btn" onClick={() => dispatch({ type: 'CLEAR_CHORDS' })}>Clear</button>
      </div>
      <div className="sd-chords-grid">
        {cells.map((b) => {
          const chord = chords[b];
          const bar = Math.floor(b / BEATS_PER_BAR);
          const beatInBar = b % BEATS_PER_BAR;
          const isBarStart = beatInBar === 0;
          const isPlaying = b === currentBeat;
          return (
            <div
              key={b}
              className={`sd-chord-cell ${isBarStart ? 'bar-start' : ''} ${isPlaying ? 'playing' : ''} ${chord ? 'filled' : ''}`}
              onClick={() => setEditingBeat(b)}
              onContextMenu={(e) => {
                e.preventDefault();
                if (chord) dispatch({ type: 'DELETE_CHORD', payload: { beatIndex: b } });
              }}
            >
              <span className="sd-chord-barlabel">{isBarStart ? bar + 1 : ''}</span>
              <span className="sd-chord-symbol">{chord || '·'}</span>
            </div>
          );
        })}
      </div>

      {/* Inline picker */}
      {editingBeat != null && (
        <div className="sd-chords-picker" onClick={(e) => e.stopPropagation()}>
          <div className="sd-chord-picker-head">
            <span className="sd-midi-kv-k">Beat {editingBeat + 1}</span>
            <div className="sd-midi-spacer" />
            <button className="sd-midi-btn" onClick={() => setEditingBeat(null)}>Done</button>
          </div>
          <div className="sd-chord-picker-row">
            {ROOTS.map((r) => (
              <button key={r} className="sd-chord-picker-btn"
                      onClick={() => {
                        const q = QUALS[0];
                        setChord(editingBeat, `${r}${q}`);
                      }}>{r}</button>
            ))}
          </div>
          <div className="sd-chord-picker-row">
            {QUALS.map((q) => (
              <button key={q || 'maj'} className="sd-chord-picker-btn"
                      onClick={() => {
                        const cur = chords[editingBeat] || 'C';
                        const root = cur.match(/^[A-G]#?/)?.[0] || 'C';
                        setChord(editingBeat, `${root}${q}`);
                      }}>{q || 'maj'}</button>
            ))}
          </div>
          <div className="sd-chord-picker-row">
            <button className="sd-midi-btn sd-midi-danger"
                    onClick={() => {
                      dispatch({ type: 'DELETE_CHORD', payload: { beatIndex: editingBeat } });
                      setEditingBeat(null);
                    }}>Delete chord</button>
          </div>
        </div>
      )}
    </div>
  );
}
