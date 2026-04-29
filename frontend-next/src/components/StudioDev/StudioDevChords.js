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

const ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const QUALS = ['', 'm', '7', 'maj7', 'm7', 'sus4', 'sus2', 'dim', 'aug', '9'];
const BARS_VISIBLE = 8;

export default function StudioDevChords() {
  const { state, dispatch } = useApp();
  const [editingBeat, setEditingBeat] = useState(null);
  const [pageStartBar, setPageStartBar] = useState(0);
  const chords = state.chordTrack?.chords || {};

  // SET_CHORD_FOR_BEAT is picked up by StudioDev.js's polypitch chord-diff
  // effect, which re-voices pitched stems on the client via the WebGPU
  // polypitch pipeline. The legacy /api/regen-stem-for-chord path (A100
  // stemphonic runtime) is not deployed in the current Modal setup — it
  // used to be wired here via useChordRegen but consistently returned
  // HTTP 500. Polypitch is the current source of truth for chord edits.
  const setChord = (beatIndex, newChord) => {
    dispatch({ type: 'SET_CHORD_FOR_BEAT', payload: { beatIndex, chord: newChord } });
  };

  const deleteChord = (beatIndex) => {
    dispatch({ type: 'SET_CHORD_FOR_BEAT', payload: { beatIndex, chord: null } });
  };

  const BEATS_PER_BAR = state.beatsPerBar || 4;
  const meterDen = state.meterDenominator || 4;
  // One beat = quarter for /4 meters, eighth for /8. Affects synthetic
  // (non-beatMap) time math for seconds-per-beat and seconds-per-bar.
  const beatUnitFactor = meterDen === 8 ? 0.5 : 1;

  // Cell grouping per bar. For 7/8 default 4+1.5+1.5 feel we render 4
  // chord cells at eighth offsets [0, 2, 4, 6] with fr widths 2+2+1.5+1.5
  // — playhead moves faster through the trailing pair. All other meters
  // keep one cell per beat (the historical behavior).
  const cellStartsPerBar = useMemo(
    () => (meterDen === 8 && BEATS_PER_BAR === 7
      ? [0, 2, 4, 6]
      : Array.from({ length: BEATS_PER_BAR }, (_, i) => i)),
    [meterDen, BEATS_PER_BAR],
  );
  const cellWidthsPerBar = useMemo(
    () => (meterDen === 8 && BEATS_PER_BAR === 7
      ? [2, 2, 1.5, 1.5]
      : Array.from({ length: BEATS_PER_BAR }, () => 1)),
    [meterDen, BEATS_PER_BAR],
  );
  const cellsPerBar = cellStartsPerBar.length;

  // First real downbeat in seconds. The rhythm analyzer writes a tempoMap
  // whose first entry (`tempoMap[0].t`) is the onset of bar 1 — for most
  // songs this is 0.2–1.5 s after t=0 (count-in, intro noise, leading
  // silence). beatMap[0].t is the same value at a per-beat granularity.
  // When absent (rhythm analysis still running, or pure constant-BPM
  // fallback), pickupSec = 0 and bar 1 starts at t=0.
  const pickupSec = useMemo(() => {
    if (Array.isArray(state.beatMap) && state.beatMap.length > 0) {
      return Math.max(0, state.beatMap[0]?.t || 0);
    }
    if (Array.isArray(state.tempoMap) && state.tempoMap.length > 0) {
      return Math.max(0, state.tempoMap[0]?.t || 0);
    }
    return 0;
  }, [state.beatMap, state.tempoMap]);

  // Cell index under the playhead. When beatMap is available we snap to
  // real beat times (handles tempo drift, pickup bars, rit/accel). The
  // returned value is -1 while the playhead sits before the first
  // downbeat — callers use that to light up the pickup cell instead of
  // a real bar-1 cell.
  const currentBeat = useMemo(() => {
    const pos = state.playheadPosition || 0;
    const bm = state.beatMap;
    if (Array.isArray(bm) && bm.length > 0) {
      if (pos < bm[0].t) return -1;  // pre-downbeat → pickup region
      // Binary-search for the largest i where bm[i].t <= pos.
      let lo = 0, hi = bm.length - 1;
      while (lo < hi) {
        const mid = (lo + hi + 1) >> 1;
        if (bm[mid].t <= pos) lo = mid;
        else hi = mid - 1;
      }
      return lo;
    }
    const bpm = state.bpm || 120;
    // Synthetic fallback — eighth-note beats for /8 meters.
    const spb = (60 / bpm) * beatUnitFactor;
    return Math.floor(pos / spb);
  }, [state.playheadPosition, state.beatMap, state.bpm, beatUnitFactor]);

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
  // render. currentBeat === -1 means pre-downbeat (pickup region): stay on
  // the first page so the pickup cell stays visible.
  useEffect(() => {
    if (!state.isPlaying) return;
    if (currentBeat < 0) {
      if (pageStartBar !== 0) setPageStartBar(0);
      return;
    }
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

  // Per-bar cells. Each cell owns a beat range [beatStart, beatEnd) in
  // the same per-eighth index space that state.chordTrack.chords uses,
  // so chord storage + carry-forward still works untouched. For 7/8 the
  // 4 cells are bar*7 + [0, 2, 4, 6] with ends [2, 4, 6, 7].
  const cells = useMemo(() => {
    const out = [];
    for (let barOffset = 0; barOffset < BARS_VISIBLE; barOffset++) {
      const bar = pageStartBar + barOffset;
      for (let ci = 0; ci < cellsPerBar; ci++) {
        const beatStart = bar * BEATS_PER_BAR + cellStartsPerBar[ci];
        const beatEnd = ci + 1 < cellsPerBar
          ? bar * BEATS_PER_BAR + cellStartsPerBar[ci + 1]
          : (bar + 1) * BEATS_PER_BAR;
        out.push({ bar, ci, beatStart, beatEnd });
      }
    }
    return out;
  }, [pageStartBar, cellsPerBar, cellStartsPerBar, BEATS_PER_BAR]);

  const canPrev = pageStartBar > 0;
  const canNext = pageStartBar + BARS_VISIBLE < totalBars;

  // Pickup bar: one extra "bar 0" column before the normal bar grid,
  // width-proportional to the pre-downbeat offset. Only shown on the
  // first page (bar 1 is always page 0), otherwise bars 9..N don't
  // need a pickup prefix. The grid uses fr units — pickupFr is the
  // pickup's share of one bar, capped so a wildly long intro doesn't
  // squash the real bars to invisibility.
  const showPickup = pickupSec > 0 && pageStartBar === 0;
  const barSec = (state.beatMap && state.beatMap.length > BEATS_PER_BAR)
    ? Math.max(0.01, state.beatMap[BEATS_PER_BAR].t - state.beatMap[0].t)
    : (60 / (state.bpm || 120)) * beatUnitFactor * BEATS_PER_BAR;
  const pickupFr = showPickup
    ? Math.min(BEATS_PER_BAR, Math.max(0.5, (pickupSec / barSec) * BEATS_PER_BAR))
    : 0;
  // Per-bar column spec — 4fr × 1 for /4, "2fr 2fr 1.5fr 1.5fr" for 7/8.
  // We always emit an explicit template now (the stylesheet's fixed
  // repeat(32, 1fr) would misrender any non-/4 meter).
  const barColsSpec = cellWidthsPerBar.map((w) => `${w}fr`).join(' ');
  const gridTemplate = showPickup
    ? `${pickupFr}fr repeat(${BARS_VISIBLE}, ${barColsSpec})`
    : `repeat(${BARS_VISIBLE}, ${barColsSpec})`;

  return (
    <div className="sd-chords">
      {/* Page nav lives ON the chord row — left + right edge icons —
       * so the entire header bar (title / meter readout / Clear) can
       * disappear and the vertical resize handle sits flush against
       * the chord cells, saving a row of vertical space. */}
      <button
        className="sd-chords-page sd-chords-page-prev"
        disabled={!canPrev}
        onClick={() => setPageStartBar(Math.max(0, pageStartBar - BARS_VISIBLE))}
        title={`Previous ${BARS_VISIBLE} bars`}
        aria-label="Previous bars"
      >
        <i className="fa-solid fa-chevron-left" />
      </button>
      <button
        className="sd-chords-page sd-chords-page-next"
        disabled={!canNext}
        onClick={() => setPageStartBar(pageStartBar + BARS_VISIBLE)}
        title={`Next ${BARS_VISIBLE} bars`}
        aria-label="Next bars"
      >
        <i className="fa-solid fa-chevron-right" />
      </button>
      <div className="sd-chords-grid" style={gridTemplate ? { gridTemplateColumns: gridTemplate } : undefined}>
        {showPickup && (
          <div
            className={`sd-chord-cell pickup ${currentBeat === -1 ? 'playing' : ''}`}
            title={`Pickup — ${pickupSec.toFixed(2)}s before bar 1`}
          >
            <span className="sd-chord-barlabel">0</span>
            <span className="sd-chord-symbol">pickup</span>
          </div>
        )}
        {cells.map((cell) => {
          const b = cell.beatStart;
          const chord = filledChords[b];
          const isBarStart = cell.ci === 0;
          const isPlaying = currentBeat >= cell.beatStart && currentBeat < cell.beatEnd;
          // In 7/8 mode each of the 4 cells is a distinct chord slot, so
          // all 4 show their (carried-forward) chord symbol. In /4 we keep
          // the historical rule: only show at bar start or where the chord
          // explicitly changes, otherwise the grid gets noisy with every
          // beat repeating the same label.
          const showSymbol = chord && (
            meterDen === 8 || chords[b] != null || isBarStart
          );
          return (
            <div
              key={`${cell.bar}-${cell.ci}`}
              className={`sd-chord-cell ${isBarStart ? 'bar-start' : ''} ${isPlaying ? 'playing' : ''} ${chord ? 'filled' : ''}`}
              onClick={() => setEditingBeat(b)}
              onContextMenu={(e) => {
                e.preventDefault();
                if (chords[b]) deleteChord(b);
              }}
            >
              <span className="sd-chord-barlabel">{isBarStart ? cell.bar + 1 : ''}</span>
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
