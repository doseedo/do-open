import React, { useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import styles from './DAW.module.css';

/**
 * TimelineGrid - Renders vertical and horizontal grid lines across the entire DAW
 * Synchronizes with Timeline tick marks (seconds or BPM beats)
 * Also renders horizontal lines at track height boundaries
 */
const TimelineGrid = React.memo(({
  totalDuration,
  zoomLevel,
  containerWidth = 700,
  isBPMMode = false,
  bpm = 120,
  sceneTempos = [],
  sceneChanges = [],
  buses = [],
  trackHeight = 64,
  onSeek = null
}) => {
  const { state } = useApp();
  const beatsPerBar = state.beatsPerBar || 4;
  const meterDenominator = state.meterDenominator || 4;
  const beatUnitFactor = meterDenominator === 8 ? 0.5 : 1;
  // Calculate pixels per second
  const pixelsPerSecond = useMemo(() => {
    const width = containerWidth * zoomLevel;
    return totalDuration > 0 ? width / totalDuration : 0;
  }, [containerWidth, zoomLevel, totalDuration]);

  // Determine optimal tick interval based on zoom level (matches Timeline.js logic)
  const tickInterval = useMemo(() => {
    if (pixelsPerSecond < 10) return 10;  // Very zoomed out - 10s intervals
    if (pixelsPerSecond < 20) return 5;   // Zoomed out - 5s intervals
    if (pixelsPerSecond < 40) return 2;   // Medium zoom - 2s intervals
    return 1;  // Zoomed in - 1s intervals
  }, [pixelsPerSecond]);

  // Determine subdivision level for BPM mode (matches Timeline.js logic)
  const subdivisionLevel = useMemo(() => {
    if (!isBPMMode) return 1;

    const avgBPM = sceneTempos.length > 0 ? sceneTempos.reduce((a, b) => a + b) / sceneTempos.length : bpm;
    const secondsPerBeat = 60 / avgBPM;
    const pixelsPerBeat = pixelsPerSecond * secondsPerBeat;

    if (pixelsPerBeat < 20) return 1;  // Too tight, only show quarter notes
    if (pixelsPerBeat < 60) return 2;  // Show 8th notes
    return 4; // Show 16th notes
  }, [isBPMMode, bpm, sceneTempos, pixelsPerSecond]);

  // Generate BPM tick marks (bars, beats, and sub-beats)
  const bpmTicks = useMemo(() => {
    if (!isBPMMode) return null;

    const tickArray = [];
    const width = containerWidth * zoomLevel;
    const useSceneTempos = sceneTempos.length > 0 && sceneChanges.length > 1;

    if (useSceneTempos) {
      // Render beats/bars with tempo changes per scene
      let accumulatedBeats = 0;
      let barNumber = 1;

      for (let sceneIdx = 0; sceneIdx < sceneTempos.length; sceneIdx++) {
        const sceneBPM = sceneTempos[sceneIdx];
        const sceneStart = sceneChanges[sceneIdx];
        const sceneEnd = sceneChanges[sceneIdx + 1] || totalDuration;
        const secondsPerBeat = (60 / sceneBPM) * beatUnitFactor;
        const secondsPerBar = secondsPerBeat * beatsPerBar;

        // Calculate where the next bar should start based on accumulated beats
        const beatsIntoFirstBar = accumulatedBeats % beatsPerBar;
        const beatsUntilNextBar = beatsIntoFirstBar === 0 ? 0 : (beatsPerBar - beatsIntoFirstBar);
        const firstBarTime = sceneStart + (beatsUntilNextBar * secondsPerBeat);

        // Render bars starting from first bar boundary in this scene
        let barTime = firstBarTime;
        while (barTime < sceneEnd && barTime <= totalDuration) {
          if (barTime >= sceneStart) {
            const barPosition = (barTime / totalDuration) * width;

            // Add bar marker
            tickArray.push({
              id: `bar-${barNumber}`,
              time: barTime,
              position: barPosition,
              isMajor: true,
              isBar: true,
              subdivision: 1
            });

            // Add beat and sub-beat subdivisions
            const totalSubdivisions = beatsPerBar * subdivisionLevel;
            for (let sub = 1; sub < totalSubdivisions; sub++) {
              const subTime = barTime + (sub * secondsPerBeat / subdivisionLevel);
              if (subTime >= sceneEnd || subTime > totalDuration) break;

              const subPosition = (subTime / totalDuration) * width;
              const isBeat = (sub % subdivisionLevel) === 0;

              tickArray.push({
                id: `bar-${barNumber}-sub-${sub}`,
                time: subTime,
                position: subPosition,
                isMajor: false,
                isBeat: isBeat,
                isSubBeat: !isBeat,
                subdivision: subdivisionLevel
              });
            }

            barNumber++;
          }
          barTime += secondsPerBar;
        }

        // Update accumulated beats for next scene
        const sceneDuration = sceneEnd - sceneStart;
        accumulatedBeats += sceneDuration / secondsPerBeat;
      }

    } else if (
      state.beatMap && state.beatMap.length >= beatsPerBar &&
      meterDenominator === 4 &&
      state.beatMap.reduce((m, b) => Math.max(m, b.pos), 0) === beatsPerBar
    ) {
      const bm = state.beatMap;
      let barNumber = 1;
      for (let i = 0; i < bm.length; i++) {
        if (bm[i].pos !== 1) continue;
        const time = bm[i].t;
        if (time > totalDuration) break;
        tickArray.push({
          id: `bar-${barNumber}`,
          time,
          position: (time / totalDuration) * width,
          isMajor: true, isBar: true, subdivision: 1,
        });
        for (let j = 1; j < beatsPerBar && (i + j) < bm.length; j++) {
          const subTime = bm[i + j].t;
          if (subTime > totalDuration) break;
          tickArray.push({
            id: `bar-${barNumber}-sub-${j}`,
            time: subTime,
            position: (subTime / totalDuration) * width,
            isMajor: false, isBeat: true, isSubBeat: false, subdivision: 1,
          });
        }
        barNumber++;
      }
    } else {
      // Render with constant BPM
      const secondsPerBeat = (60 / bpm) * beatUnitFactor;
      const secondsPerBar = secondsPerBeat * beatsPerBar;
      const tlOffset = state.timelineOffset || 0;
      let barNumber = 1;

      for (let time = tlOffset; time <= totalDuration; time += secondsPerBar) {
        const barPosition = (time / totalDuration) * width;

        // Add bar marker
        tickArray.push({
          id: `bar-${barNumber}`,
          time,
          position: barPosition,
          isMajor: true,
          isBar: true,
          subdivision: 1
        });

        // Add beat and sub-beat subdivisions
        const totalSubdivisions = beatsPerBar * subdivisionLevel;
        for (let sub = 1; sub < totalSubdivisions; sub++) {
          const subTime = time + (sub * secondsPerBeat / subdivisionLevel);
          if (subTime > totalDuration) break;

          const subPosition = (subTime / totalDuration) * width;
          const isBeat = (sub % subdivisionLevel) === 0;

          tickArray.push({
            id: `bar-${barNumber}-sub-${sub}`,
            time: subTime,
            position: subPosition,
            isMajor: false,
            isBeat: isBeat,
            isSubBeat: !isBeat,
            subdivision: subdivisionLevel
          });
        }

        barNumber++;
      }
    }

    return tickArray;
  }, [isBPMMode, bpm, beatsPerBar, beatUnitFactor, sceneTempos, sceneChanges, totalDuration, containerWidth, zoomLevel, subdivisionLevel, state.timelineOffset, state.beatMap]);

  // Generate time tick marks (seconds)
  const timeTicks = useMemo(() => {
    if (isBPMMode) return null;

    const tickArray = [];
    const width = containerWidth * zoomLevel;

    for (let time = 0; time <= totalDuration; time += tickInterval) {
      const position = (time / totalDuration) * width;

      tickArray.push({
        id: `tick-${time}`,
        time,
        position,
        isMajor: time % (tickInterval * 5) === 0 || tickInterval >= 5,
        isBar: false,
        isBeat: false,
        isSubBeat: false
      });
    }

    return tickArray;
  }, [isBPMMode, totalDuration, tickInterval, containerWidth, zoomLevel]);

  // Select which ticks to use
  const ticks = isBPMMode ? bpmTicks : timeTicks;

  // Calculate horizontal grid line positions based on track heights
  const horizontalLines = useMemo(() => {
    const lines = [];
    let currentY = 0;

    buses.forEach((bus, busIndex) => {
      // Add line at the top of each bus
      lines.push({
        id: `bus-${busIndex}-top`,
        y: currentY,
        isBusTop: true
      });

      // Move down by bus height
      currentY += trackHeight;

      // Add lines for each track in the bus
      if (bus.expanded && bus.tracks) {
        bus.tracks.forEach((track, trackIndex) => {
          lines.push({
            id: `bus-${busIndex}-track-${trackIndex}`,
            y: currentY,
            isBusTop: false
          });
          currentY += trackHeight;
        });
      }
    });

    return lines;
  }, [buses, trackHeight]);

  const timelineContentWidth = containerWidth * zoomLevel;

  // Handle click to seek
  const handleClick = (e) => {
    if (!onSeek) return;

    // Don't seek if clicking on a track or track-related element
    // Check if the click target or any parent is a track, bus, or waveform element
    const clickedElement = e.target;
    const isTrackElement = clickedElement.closest('[data-track-id]') ||
                          clickedElement.closest('[data-bus-id]') ||
                          clickedElement.closest('[class*="track"]') ||
                          clickedElement.closest('[class*="bus"]') ||
                          clickedElement.closest('[class*="waveform"]') ||
                          clickedElement.closest('canvas');

    // Only seek if clicking on empty grid space
    if (isTrackElement) {
      console.log('🚫 TimelineGrid click ignored - clicked on track/bus element');
      return;
    }

    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickedTime = (clickX / timelineContentWidth) * totalDuration;
    const clampedTime = Math.max(0, Math.min(clickedTime, totalDuration));

    console.log('🎯 TimelineGrid seeking to:', clampedTime.toFixed(2) + 's');
    onSeek(clampedTime);
  };

  return (
    <div
      className={styles.timelineGrid}
      style={{
        width: `${timelineContentWidth}px`,
        pointerEvents: 'none',
        cursor: 'default'
      }}
    >
      {/* Vertical grid lines */}
      {ticks && ticks.map((tick) => (
        <div
          key={tick.id}
          className={styles.gridLine}
          style={{
            left: `${tick.position}px`,
            width: tick.isBar ? '1px' : '0.5px',
            backgroundColor: tick.isBar ? 'rgba(200, 200, 220, 0.12)' :
                           tick.isBeat ? 'rgba(180, 180, 200, 0.08)' :
                           tick.isSubBeat ? 'rgba(160, 160, 180, 0.04)' :
                           tick.isMajor ? 'rgba(190, 190, 210, 0.1)' : 'rgba(170, 170, 190, 0.06)'
          }}
        />
      ))}

      {/* Horizontal grid lines at track boundaries */}
      {horizontalLines.map((line) => (
        <div
          key={line.id}
          className={styles.horizontalGridLine}
          style={{
            top: `${line.y}px`,
            width: '100%',
            height: '1px',
            position: 'absolute',
            left: 0,
            backgroundColor: 'rgba(100, 100, 120, 0.08)',
            pointerEvents: 'none'
          }}
        />
      ))}
    </div>
  );
});

TimelineGrid.displayName = 'TimelineGrid';

export default TimelineGrid;
