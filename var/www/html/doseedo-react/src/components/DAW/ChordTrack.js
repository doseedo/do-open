import { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { useApp } from '../../context/AppContext';

/**
 * ChordTrack Component - Matches Timeline structure EXACTLY
 *
 * Displays chord selection cells that align with BPM bars in the timeline.
 * Each cell represents one bar (4 beats in 4/4 time).
 *
 * CRITICAL: This component must have the EXACT same DOM structure as Timeline
 * to ensure proper width measurement and mouse coordinate alignment.
 */
const ChordTrack = ({ totalDuration, zoomLevel, onBeatSelect }) => {
  const { state } = useApp();
  const [hoveredBeat, setHoveredBeat] = useState(null);
  const chordTrackRef = useRef(null);

  const bpm = state.bpm || 120;
  const beatsPerBar = state.beatsPerBar || 4;
  const meterDenominator = state.meterDenominator || 4;
  const sceneTempos = state.video?.sceneTempos || [];
  const sceneChanges = state.video?.sceneChanges || [];

  // Chord cells per bar for the current meter. Expressed in beat-unit
  // subdivisions (eighths for /8, quarters for /4) as [[start, end], ...]
  // where each pair is one cell. For 7/8 default 4+1.5+1.5 grouping we
  // render 4 cells at 2+2+1.5+1.5 eighths — playback naturally moves
  // faster through the skinny trailing pair.
  const cellGrouping = useMemo(() => {
    if (meterDenominator === 8 && beatsPerBar === 7) {
      return [[0, 2], [2, 4], [4, 5.5], [5.5, 7]];
    }
    const g = [];
    for (let i = 0; i < beatsPerBar; i++) g.push([i, i + 1]);
    return g;
  }, [beatsPerBar, meterDenominator]);

  // CRITICAL: Use the SAME containerWidth as Timeline from global state
  // This ensures perfect sync between Timeline and ChordTrack
  const containerWidth = state.timelineWidth || 700;
  const chordTrackContentWidth = containerWidth * zoomLevel;

  // Force width using ref - bypasses CSS constraints
  useEffect(() => {
    if (chordTrackRef.current) {
      chordTrackRef.current.style.width = `${chordTrackContentWidth}px`;
      chordTrackRef.current.style.minWidth = `${chordTrackContentWidth}px`;
      chordTrackRef.current.style.maxWidth = `${chordTrackContentWidth}px`;
    }
  }, [chordTrackContentWidth]);

  // Generate BPM beat cells - 4 beats per bar
  const beatCells = useMemo(() => {
    const cellArray = [];
    const width = containerWidth * zoomLevel;
    const useSceneTempos = sceneTempos.length > 0 && sceneChanges.length > 1;

    if (useSceneTempos) {
      // Render beats with tempo changes per scene
      let accumulatedBeats = 0;
      let beatNumber = 1;

      for (let sceneIdx = 0; sceneIdx < sceneTempos.length; sceneIdx++) {
        const sceneBPM = sceneTempos[sceneIdx];
        const sceneStart = sceneChanges[sceneIdx];
        const sceneEnd = sceneChanges[sceneIdx + 1] || totalDuration;
        const secondsPerBeat = 60 / sceneBPM;

        // Calculate where the next beat should start
        const beatsIntoFirstBeat = accumulatedBeats % 1;
        const beatsUntilNextBeat = beatsIntoFirstBeat === 0 ? 0 : (1 - beatsIntoFirstBeat);
        const firstBeatTime = sceneStart + (beatsUntilNextBeat * secondsPerBeat);

        // Render beats starting from first beat boundary in this scene
        let beatTime = firstBeatTime;
        while (beatTime < sceneEnd && beatTime <= totalDuration) {
          if (beatTime >= sceneStart) {
            const beatPosition = (beatTime / totalDuration) * width;
            const nextBeatTime = beatTime + secondsPerBeat;
            const nextBeatPosition = Math.min((nextBeatTime / totalDuration) * width, width);

            cellArray.push({
              id: `beat-${beatNumber}`,
              beatNumber,
              time: beatTime,
              position: beatPosition,
              width: nextBeatPosition - beatPosition,
              chord: state.chords?.[beatNumber - 1] || null // Chord per beat (0-indexed)
            });

            beatNumber++;
          }
          beatTime += secondsPerBeat;
        }

        // Update accumulated beats for next scene
        const sceneDuration = sceneEnd - sceneStart;
        accumulatedBeats += sceneDuration / secondsPerBeat;
      }
    } else {
      // Bar-grouped rendering. Bar starts come from beatMap pos=1 entries
      // when the detected meter still matches state.beatsPerBar; otherwise
      // fall back to synthetic bars at constant tempo + timelineOffset.
      // Within each bar we emit one cell per entry in cellGrouping — for
      // 7/8 that's 4 cells at 2+2+1.5+1.5 eighths.
      const beatUnitFactor = meterDenominator === 8 ? 0.5 : 1;
      const secondsPerUnit = (60 / bpm) * beatUnitFactor;
      const secondsPerBar = secondsPerUnit * beatsPerBar;
      const unitsInBar = beatsPerBar;

      const barStarts = [];
      const bm = state.beatMap;
      const bmMatches = bm && bm.length > 0 &&
        bm.reduce((m, b) => Math.max(m, b.pos || 0), 0) === beatsPerBar;
      if (bmMatches) {
        for (const b of bm) {
          if (b.pos === 1 && b.t <= totalDuration) barStarts.push(b.t);
        }
      }
      if (barStarts.length === 0) {
        const tlOffset = state.timelineOffset || 0;
        for (let t = tlOffset; t <= totalDuration; t += secondsPerBar) barStarts.push(t);
      }

      for (let b = 0; b < barStarts.length; b++) {
        const barStart = barStarts[b];
        const nextBarStart = (b + 1 < barStarts.length)
          ? barStarts[b + 1]
          : Math.min(barStart + secondsPerBar, totalDuration);
        const barDur = Math.max(0, nextBarStart - barStart);
        if (barDur <= 0) continue;

        cellGrouping.forEach(([u0, u1], ci) => {
          const cellStart = barStart + (u0 / unitsInBar) * barDur;
          const cellEnd = barStart + (u1 / unitsInBar) * barDur;
          if (cellStart >= totalDuration) return;
          const pos = (cellStart / totalDuration) * width;
          const right = Math.min((cellEnd / totalDuration) * width, width);
          const globalIdx = cellArray.length;
          cellArray.push({
            id: `cell-${b + 1}-${ci}`,
            beatNumber: globalIdx + 1,
            barNumber: b + 1,
            cellInBar: ci,
            isBarStart: ci === 0,
            time: cellStart,
            position: pos,
            width: Math.max(0, right - pos),
            chord: state.chordTrack?.chords?.[globalIdx] || null,
          });
        });
      }
    }

    return cellArray;
  }, [containerWidth, zoomLevel, totalDuration, bpm, beatsPerBar, meterDenominator, cellGrouping, sceneTempos, sceneChanges, state.chordTrack, state.timelineOffset, state.beatMap]);

  // Mouse move handler
  const handleMouseMove = useCallback((e) => {
    if (!chordTrackRef?.current) return;

    // Get the bounding rect of the actual content div with the cells
    const rect = chordTrackRef.current.getBoundingClientRect();

    // Get mouse position relative to the chord track
    const mouseX = e.clientX - rect.left;

    // Find which beat cell the mouse is in
    let hoveredBeatNumber = null;
    for (const cell of beatCells) {
      if (mouseX >= cell.position && mouseX < cell.position + cell.width) {
        hoveredBeatNumber = cell.beatNumber;
        break;
      }
    }

    setHoveredBeat(hoveredBeatNumber);
  }, [beatCells, containerWidth, zoomLevel]);

  const handleMouseLeave = useCallback(() => {
    setHoveredBeat(null);
  }, []);

  // Click handler
  const handleClick = useCallback((e) => {
    if (!chordTrackRef?.current) return;

    const rect = chordTrackRef.current.getBoundingClientRect();
    // Get click position relative to the chord track
    const clickX = e.clientX - rect.left;

    // Find which beat was clicked
    for (const cell of beatCells) {
      if (clickX >= cell.position && clickX < cell.position + cell.width) {
        onBeatSelect?.(cell.beatNumber);
        return;
      }
    }
  }, [beatCells, onBeatSelect]);

  return (
    <div
      ref={chordTrackRef}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
      style={{
        position: 'absolute', // CRITICAL: Escape flex/grid constraints
        left: '0px', // Align with timeline ticks
        top: 0,
        width: `${chordTrackContentWidth}px`,
        height: '100%',
        cursor: 'pointer',
        overflow: 'visible',
        pointerEvents: 'auto' // Ensure mouse events work
      }}
    >
      {/* Render cells — grouping-driven: one per beat in /4, 4 per 7/8 bar. */}
      {beatCells.map((cell) => {
        const isBarStart = cell.isBarStart ?? ((cell.beatNumber - 1) % beatsPerBar === 0);

        return (
          <div
            key={cell.id}
            style={{
              position: 'absolute',
              left: `${cell.position}px`,
              top: 0,
              width: `${cell.width}px`,
              height: '100%',
              borderLeft: isBarStart ? '2px solid rgba(255, 255, 255, 0.4)' : '1px solid rgba(255, 255, 255, 0.1)',
              borderRight: '1px solid rgba(255, 255, 255, 0.1)',
              borderTop: '1px solid rgba(255, 255, 255, 0.1)',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              backgroundColor: hoveredBeat === cell.beatNumber
                ? 'rgba(102, 126, 234, 0.3)'
                : 'transparent',
              transition: 'background-color 0.1s ease',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              pointerEvents: 'none',
              boxSizing: 'border-box'
            }}
          >
          {/* Chord label - show on every beat */}
          {cell.chord && (
            <div
              style={{
                fontSize: '11px',
                fontWeight: 600,
                color: '#c5cae9',
                whiteSpace: 'nowrap',
                textShadow: '0 1px 2px rgba(0,0,0,0.5)'
              }}
            >
              {cell.chord}
            </div>
          )}
        </div>
        );
      })}
    </div>
  );
};

export default ChordTrack;
