import React, { useMemo, useState, useCallback } from 'react';
import { useApp } from '../../context/AppContext';
import { useThemeColor } from '../../hooks/useThemeColor';
import styles from './SceneMarkers.module.css';

/**
 * SceneMarkers Component - Renders scene change markers on timeline
 * Shows colored range bars with arrows and tempo labels
 * Uses same positioning formula as Timeline ticks for perfect alignment
 * Expands on hover to reveal chord cells for editing
 */
const SceneMarkers = React.memo(({ totalDuration, width }) => {
  const { state, dispatch } = useApp();
  const [hoveredScene, setHoveredScene] = useState(null);
  const [selectedBeat, setSelectedBeat] = useState(null);

  // Get theme colors for reactive scene markers
  const primaryBlue = useThemeColor('--color-primary-blue', '#667eea');
  const primaryPurple = useThemeColor('--color-primary-purple', '#8b5cf6');
  const primaryPurpleDark = useThemeColor('--color-primary-purple-dark', '#764ba2');

  // Calculate scene ranges from scene changes and their chord cells
  const sceneRanges = useMemo(() => {
    if (!state.video?.sceneChanges || state.video.sceneChanges.length === 0) {
      return [];
    }

    const ranges = [];
    const sceneChanges = state.video.sceneChanges;
    const sceneTempos = state.video.sceneTempos || [];
    const bpm = state.bpm || 120;
    // Use the last scene change as the end if no duration is available
    const videoDuration = state.video.duration || totalDuration || sceneChanges[sceneChanges.length - 1];

    // CRITICAL: Use the actual measured timeline width * zoomLevel (same as Timeline.js)
    // The 'width' prop should be state.timelineWidth * zoomLevel
    const timelineWidth = width;

    // Convert hex to RGB for interpolation
    const hexToRgb = (hex) => {
      const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
      return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
      } : { r: 102, g: 126, b: 234 };
    };

    const startColor = hexToRgb(primaryBlue);
    const endColor = hexToRgb(primaryPurpleDark);

    let globalBeatNumber = 1;

    for (let i = 0; i < sceneChanges.length; i++) {
      const startTime = sceneChanges[i];
      const endTime = sceneChanges[i + 1] ?? videoDuration;
      // Use same formula as Timeline ticks: (time / totalDuration) * width
      const startX = (startTime / totalDuration) * timelineWidth;
      const rangeWidth = ((endTime - startTime) / totalDuration) * timelineWidth;

      // Generate gradient colors using theme colors (blue → purple)
      const progress = sceneChanges.length > 1 ? i / (sceneChanges.length - 1) : 0;
      const r = Math.round(startColor.r + (endColor.r - startColor.r) * progress);
      const g = Math.round(startColor.g + (endColor.g - startColor.g) * progress);
      const b = Math.round(startColor.b + (endColor.b - startColor.b) * progress);

      // Generate chord cells for this scene
      const sceneBPM = sceneTempos[i] || bpm;
      const secondsPerBeat = 60 / sceneBPM;
      const chordCells = [];

      let beatTime = startTime;
      while (beatTime < endTime && beatTime <= totalDuration) {
        const beatPosition = (beatTime / totalDuration) * timelineWidth;
        const nextBeatTime = Math.min(beatTime + secondsPerBeat, endTime);
        const nextBeatPosition = (nextBeatTime / totalDuration) * timelineWidth;
        const cellWidth = nextBeatPosition - beatPosition;

        // Position relative to scene start
        const relativePosition = beatPosition - startX;

        chordCells.push({
          id: `beat-${globalBeatNumber}`,
          beatNumber: globalBeatNumber,
          time: beatTime,
          position: relativePosition,
          width: cellWidth,
          chord: state.chordTrack?.chords?.[globalBeatNumber - 1] || null
        });

        globalBeatNumber++;
        beatTime += secondsPerBeat;
      }

      ranges.push({
        id: `scene-${i}`,
        index: i,
        startTime,
        endTime,
        startX,
        width: rangeWidth,
        color: `rgba(${r}, ${g}, ${b}, 0.8)`,
        tempo: sceneTempos[i] || null,
        chordCells
      });
    }

    return ranges;
  }, [state.video?.sceneChanges, state.video?.sceneTempos, state.video?.duration, state.bpm, state.chordTrack, totalDuration, width, primaryBlue, primaryPurpleDark]);

  // Handlers for scene interaction
  const handleSceneMouseEnter = useCallback((sceneIndex) => {
    setHoveredScene(sceneIndex);
  }, []);

  const handleSceneMouseLeave = useCallback(() => {
    setHoveredScene(null);
  }, []);

  const handleChordClick = useCallback((beatNumber) => {
    setSelectedBeat(beatNumber);
    // Open chord window for this beat
    dispatch({ type: 'SET_CHORD_WINDOW_BEAT', payload: beatNumber });
    console.log('Opening chord editor for beat', beatNumber);
  }, [dispatch]);

  if (sceneRanges.length === 0) {
    return null;
  }

  return (
    <>
      {sceneRanges.map((scene) => {
        const isHovered = hoveredScene === scene.index;

        return (
          <React.Fragment key={scene.id}>
            {/* Scene range bar - expands on hover */}
            <div
              className={`${styles.sceneRange} ${isHovered ? styles.sceneRangeExpanded : ''}`}
              style={{
                left: `${scene.startX}px`,
                width: `${scene.width}px`,
                background: scene.color,
                height: isHovered ? '40px' : '8px',
                pointerEvents: 'auto'
              }}
              onMouseEnter={() => handleSceneMouseEnter(scene.index)}
              onMouseLeave={handleSceneMouseLeave}
              title={`Scene ${scene.index + 1}`}
            >
              {/* Chord cells - only visible when hovered */}
              {isHovered && scene.chordCells && (
                <div className={styles.chordCellsContainer}>
                  {scene.chordCells.map((cell) => (
                    <div
                      key={cell.id}
                      className={styles.chordCell}
                      style={{
                        left: `${cell.position}px`,
                        width: `${cell.width}px`
                      }}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleChordClick(cell.beatNumber);
                      }}
                    >
                      <span className={styles.chordLabel}>
                        {cell.chord || '+'}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Scene arrow at start */}
            <div
              className={styles.sceneArrow}
              style={{
                left: `${scene.startX}px`,
                pointerEvents: 'none'
              }}
              title={`Scene ${scene.index + 1}`}
            />

            {/* Tempo label */}
            {scene.tempo && (
              <div
                className={styles.sceneTempoLabel}
                style={{
                  left: `${scene.startX}px`,
                  pointerEvents: 'none'
                }}
              >
                {scene.tempo} BPM
              </div>
            )}
          </React.Fragment>
        );
      })}
    </>
  );
});

SceneMarkers.displayName = 'SceneMarkers';

export default SceneMarkers;
