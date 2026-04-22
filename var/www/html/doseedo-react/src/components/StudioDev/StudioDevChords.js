/*
 * StudioDevChords — themed chord grid for /studio-dev.
 *
 * Reads and writes state.chordTrack.chords (the same field the tier-1/2/3
 * detection populates + the polypitch chord-sync effect watches), so
 * auto-detected chords after upload show up here and manual edits flow
 * back to the resynth pipeline.
 *
 * Click empty cell → open inline picker. Click filled cell → edit. Right-
 * click → delete. Delete via right-click or picker fires SET_CHORD_FOR_BEAT
 * with a falsy payload, which the reducer treats as a removal.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useApp } from '../../context/AppContext';
import useChordRegen from '../../hooks/useChordRegen';

const ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const QUALS = ['', 'm', '7', 'maj7', 'm7', 'sus4', 'sus2', 'dim', 'aug', '9'];
const BARS_VISIBLE = 8;

export default function StudioDevChords() {
  const { state, dispatch } = useApp();
  const [editingBeat, setEditingBeat] = useState(null);
  const [pageStartBar, setPageStartBar] = useState(0);
  const chords = state.chordTrack?.chords || {};
  const runChordRegen = useChordRegen();

  const setChord = (beatIndex, newChord) => {
    const oldChord = chords[beatIndex] || '';
    dispatch({ type: 'SET_CHORD_FOR_BEAT', payload: { beatIndex, chord: newChord } });
    runChordRegen({ beatIndex, oldChord, newChord }).catch((e) =>
      console.warn('[studio-dev] chord regen failed:', e?.message || e));
  };

  const deleteChord = (beatIndex) => {
    dispatch({ type: 'SET_CHORD_FOR_BEAT', payload: { beatIndex, chord: null } });
  };

  const BEATS_PER_BAR = state.beatsPerBar || 4;

  const currentBeat = useMemo(() => {
    const bpm = state.bpm || 120;
    const spb = 60 / bpm;
    return Math.floor((state.playheadPosition || 0) / spb);
  }, [state.playheadPosition, state.bpm]);

  // Detector emits a sparse map keyed on chord-change beats; every beat in
  // between inherits the prior label. Carry it forward so sustained chords
  // actually fill the grid — otherwise 9 chord changes across 60 beats
  // render as 9 dots surrounded by `·` and the bar looks empty.
  //
  // Scan from beat 0 up to the last chord change OR the last visible beat,
  // whichever is larger, so paging forward keeps working once the detector
  // assigns labels beyond the current window.
  const lastChordBeat = useMemo(() => {
    let m = 0;
    for (const k of Object.keys(chords)) {
      const n = parseInt(k, 10);
      if (Number.isFinite(n) && n > m) m = n;
    }
    return m;
  }, [chords]);

  const filledChords = useMemo(() => {
    const out = {};
    const totalBeats = Math.max(
      lastChordBeat + 1,
      (pageStartBar + BARS_VISIBLE) * BEATS_PER_BAR
    );
    let current = null;
    for (let i = 0; i < totalBeats; i++) {
      if (chords[i] != null) current = chords[i];
      if (current != null) out[i] = current;
    }
    return out;
  }, [chords, lastChordBeat, pageStartBar, BEATS_PER_BAR]);

  // Auto-advance the page when playback crosses into the next 8-bar window,
  // but only while playing — otherwise manual paging would get fought every
  // render.
  useEffect(() => {
    if (!state.isPlaying) return;
    const currentBar = Math.floor(currentBeat / BEATS_PER_BAR);
    const pageEndBar = pageStartBar + BARS_VISIBLE;
    if (currentBar >= pageEndBar || currentBar < pageStartBar) {
      const nextStart = Math.floor(currentBar / BARS_VISIBLE) * BARS_VISIBLE;
      setPageStartBar(nextStart);
    }
  }, [currentBeat, state.isPlaying, pageStartBar, BEATS_PER_BAR]);

  const totalBars = useMemo(() => {
    const lastBar = Math.floor(lastChordBeat / BEATS_PER_BAR);
    return Math.max(BARS_VISIBLE, lastBar + 1);
  }, [lastChordBeat, BEATS_PER_BAR]);

  const cells = Array.from(
    { length: BARS_VISIBLE * BEATS_PER_BAR },
    (_, i) => pageStartBar * BEATS_PER_BAR + i
  );

  const canPrev = pageStartBar > 0;
  const canNext = pageStartBar + BARS_VISIBLE < totalBars;

  return (
    <div className="sd-chords">
      <div className="sd-chords-head">
        <span className="sd-midi-kv-k">Chords</span>
        <span className="sd-midi-meta">
          {state.beatsPerBar || 4}/{state.meterDenominator || 4} · bars {pageStartBar + 1}–{pageStartBar + BARS_VISIBLE}
        </span>
        <button
          className="sd-midi-btn"
          disabled={!canPrev}
          onClick={() => setPageStartBar(Math.max(0, pageStartBar - BARS_VISIBLE))}
        >‹ Prev</button>
        <button
          className="sd-midi-btn"
          disabled={!canNext}
          onClick={() => setPageStartBar(pageStartBar + BARS_VISIBLE)}
        >Next ›</button>
        <div className="sd-midi-spacer" />
        <button className="sd-midi-btn" onClick={() => { dispatch({ type: 'CLEAR_CHORDS' }); setPageStartBar(0); }}>Clear</button>
      </div>
      <div className="sd-chords-grid">
        {cells.map((b) => {
          const chord = filledChords[b];
          const bar = Math.floor(b / BEATS_PER_BAR);
          const beatInBar = b % BEATS_PER_BAR;
          const isBarStart = beatInBar === 0;
          const isPlaying = b === currentBeat;
          // Only show the chord symbol on the beat where it *changes* or at
          // the start of a bar — otherwise every beat would repeat the same
          // label and the grid becomes noisy.
          const showSymbol = chord && (chords[b] != null || isBarStart);
          return (
            <div
              key={b}
              className={`sd-chord-cell ${isBarStart ? 'bar-start' : ''} ${isPlaying ? 'playing' : ''} ${chord ? 'filled' : ''}`}
              onClick={() => setEditingBeat(b)}
              onContextMenu={(e) => {
                e.preventDefault();
                if (chords[b]) deleteChord(b);
              }}
            >
              <span className="sd-chord-barlabel">{isBarStart ? bar + 1 : ''}</span>
              <span className="sd-chord-symbol">{showSymbol ? chord : ''}</span>
            </div>
          );
        })}
      </div>

      {/* Inline picker */}
      {editingBeat != null && (() => {
        const activeChord = filledChords[editingBeat] || null;
        const activeRoot = activeChord ? (activeChord.match(/^[A-G]#?/)?.[0] || null) : null;
        const activeQual = activeChord && activeRoot
          ? activeChord.slice(activeRoot.length).split('/')[0]
          : null;
        return (
          <div className="sd-chords-picker" onClick={(e) => e.stopPropagation()}>
            <div className="sd-chord-picker-head">
              <span className="sd-midi-kv-k">Beat {editingBeat + 1}</span>
              <span className="sd-chord-picker-current">
                {activeChord || '—'}
              </span>
              <div className="sd-midi-spacer" />
              <button className="sd-midi-btn" onClick={() => setEditingBeat(null)}>Done</button>
            </div>
            <div className="sd-chord-picker-row">
              {ROOTS.map((r) => (
                <button
                  key={r}
                  className={`sd-chord-picker-btn ${r === activeRoot ? 'active' : ''}`}
                  onClick={() => {
                    const q = activeQual ?? QUALS[0];
                    setChord(editingBeat, `${r}${q}`);
                  }}
                >{r}</button>
              ))}
            </div>
            <div className="sd-chord-picker-row">
              {QUALS.map((q) => (
                <button
                  key={q || 'maj'}
                  className={`sd-chord-picker-btn ${q === activeQual ? 'active' : ''}`}
                  onClick={() => {
                    const root = activeRoot || 'C';
                    setChord(editingBeat, `${root}${q}`);
                  }}
                >{q || 'maj'}</button>
              ))}
            </div>
            <div className="sd-chord-picker-row">
              <button className="sd-midi-btn sd-midi-danger"
                      onClick={() => {
                        deleteChord(editingBeat);
                        setEditingBeat(null);
                      }}>Delete chord</button>
            </div>
          </div>
        );
      })()}
    </div>
  );
}
