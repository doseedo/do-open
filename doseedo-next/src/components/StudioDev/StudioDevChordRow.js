import React, { useMemo, useCallback } from 'react';
import { useApp } from '../../context/AppContext';

/**
 * StudioDev-native chord row. Sits above the lanes strip and mirrors the
 * ruler's pct-of-visible-span layout so positions stay glued to beats as
 * the user zooms or scrolls. The chord labels come from
 * state.chordTrack.chords (populated by tier-1/2/3 detection) — keyed by
 * beat index, using state.beatMap when available, otherwise synthesising
 * a constant-bpm beat grid.
 *
 * Click a cell to open an inline input; committing it dispatches
 * UPDATE_CHORD so the polypitch chord-sync effect can resynth the audio.
 * Escape / blur on empty cancels.
 *
 * Props
 *   visibleSec    — how many seconds are painted across the row (matches
 *                   the parent lane's `TIMELINE_SECONDS / timelineZoom`)
 *   timelineOffsetSec — scroll offset in seconds (parent's scrollLeft
 *                   translated to the same units)
 */
export default function StudioDevChordRow({ visibleSec, timelineOffsetSec = 0 }) {
  const { state, dispatch } = useApp();
  const [editing, setEditing] = React.useState(null); // beatIndex being edited
  const [draft, setDraft] = React.useState('');

  const bpm = state.bpm || 120;
  const beatsPerBar = state.beatsPerBar || 4;
  const beatMap = state.beatMap;
  const chords = state.chordTrack?.chords || {};

  // Beats visible in the current viewport, with their left-% and width-%.
  const cells = useMemo(() => {
    const cellArray = [];
    const startSec = Math.max(0, timelineOffsetSec);
    const endSec = startSec + visibleSec;

    if (Array.isArray(beatMap) && beatMap.length > 0) {
      for (let i = 0; i < beatMap.length; i++) {
        const t0 = beatMap[i].t;
        const t1 = (i + 1 < beatMap.length ? beatMap[i + 1].t : t0 + 60 / bpm);
        if (t1 < startSec) continue;
        if (t0 > endSec) break;
        const leftPct = ((t0 - startSec) / visibleSec) * 100;
        const widthPct = ((t1 - t0) / visibleSec) * 100;
        cellArray.push({
          beatIndex: i,
          pos: i + 1,
          chord: chords[i] || null,
          leftPct, widthPct,
        });
      }
    } else {
      const secPerBeat = 60 / bpm;
      const firstBeat = Math.floor(startSec / secPerBeat);
      const lastBeat = Math.ceil(endSec / secPerBeat);
      for (let b = firstBeat; b <= lastBeat; b++) {
        const t0 = b * secPerBeat;
        const t1 = t0 + secPerBeat;
        const leftPct = ((t0 - startSec) / visibleSec) * 100;
        const widthPct = (secPerBeat / visibleSec) * 100;
        cellArray.push({
          beatIndex: b,
          pos: (b % beatsPerBar) + 1,
          chord: chords[b] || null,
          leftPct, widthPct,
        });
      }
    }
    return cellArray;
  }, [beatMap, chords, visibleSec, timelineOffsetSec, bpm, beatsPerBar]);

  const commitEdit = useCallback((beatIndex, value) => {
    const v = (value || '').trim();
    if (!v) {
      // Erase the cell.
      dispatch({ type: 'UPDATE_CHORD', payload: { beatIndex, chord: null } });
    } else {
      dispatch({ type: 'UPDATE_CHORD', payload: { beatIndex, chord: v } });
    }
    setEditing(null);
    setDraft('');
  }, [dispatch]);

  return (
    <div className="sd-chord-row" role="row">
      {cells.map((cell) => {
        const isBarStart = (cell.pos - 1) % beatsPerBar === 0;
        const isEditing = editing === cell.beatIndex;
        return (
          <div
            key={`c-${cell.beatIndex}`}
            className={`sd-chord-cell${isBarStart ? ' bar-start' : ''}`}
            style={{ left: `${cell.leftPct}%`, width: `${cell.widthPct}%` }}
            onClick={(e) => {
              e.stopPropagation();
              setEditing(cell.beatIndex);
              setDraft(cell.chord || '');
            }}
            title={cell.chord ? `Beat ${cell.beatIndex + 1}: ${cell.chord}` : `Beat ${cell.beatIndex + 1}`}
          >
            {isEditing ? (
              <input
                autoFocus
                className="sd-chord-input"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onBlur={() => commitEdit(cell.beatIndex, draft)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') commitEdit(cell.beatIndex, draft);
                  else if (e.key === 'Escape') { setEditing(null); setDraft(''); }
                }}
                onClick={(e) => e.stopPropagation()}
              />
            ) : cell.chord ? (
              <span className="sd-chord-label">{cell.chord}</span>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}
