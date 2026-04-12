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
  const sceneTempos = state.video?.sceneTempos || [];
  const sceneChanges = state.video?.sceneChanges || [];

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
    } else if (state.beatMap && state.beatMap.length > 0) {
      const bm = state.beatMap;
      for (let i = 0; i < bm.length; i++) {
        const time = bm[i].t;
        if (time > totalDuration) break;
        const nextTime = (i + 1 < bm.length) ? bm[i + 1].t : (time + 60 / bpm);
        const beatPosition = (time / totalDuration) * width;
        const nextBeatPosition = Math.min((nextTime / totalDuration) * width, width);
        cellArray.push({
          id: `beat-${i + 1}`,
          beatNumber: i + 1,
          time,
          position: beatPosition,
          width: nextBeatPosition - beatPosition,
          chord: state.chordTrack?.chords?.[i] || null,
        });
      }
    } else {
      // Render with constant BPM - 4 beats per bar
      const secondsPerBeat = 60 / bpm;
      const tlOffset = state.timelineOffset || 0;
      let beatNumber = 1;

      for (let time = tlOffset; time <= totalDuration; time += secondsPerBeat) {
        const beatPosition = (time / totalDuration) * width;
        const nextBeatTime = time + secondsPerBeat;
        const nextBeatPosition = Math.min((nextBeatTime / totalDuration) * width, width);

        cellArray.push({
          id: `beat-${beatNumber}`,
          beatNumber,
          time,
          position: beatPosition,
          width: nextBeatPosition - beatPosition,
          chord: state.chordTrack?.chords?.[beatNumber - 1] || null // Chord per beat (0-indexed)
        });

        beatNumber++;
      }
    }

    return cellArray;
  }, [containerWidth, zoomLevel, totalDuration, bpm, beatsPerBar, sceneTempos, sceneChanges, state.chordTrack, state.timelineOffset, state.beatMap]);

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
      {/* Render beat cells - 4 beats per bar */}
      {beatCells.map((cell) => {
        // Beat 1 of each bar (every Nth beat per current meter)
        const isBarStart = (cell.beatNumber - 1) % beatsPerBar === 0;

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
