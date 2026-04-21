import React, { useEffect, useCallback, useRef, useState, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import { useAudioPlayback } from '../../hooks/useAudioPlayback';
import TimelineTick from './TimelineTick';
import PlayheadCursor from './PlayheadCursor';
import SceneMarkers from './SceneMarkers';

/**
 * TimelineWrapper Component - Declarative timeline with React patterns
 * Uses custom hooks for calculations and renders ticks via JSX
 */
const TimelineWrapper = React.memo(() => {
  const { state, dispatch } = useApp();
  const timelineRef = useRef(null);

  // Custom hook for timeline calculations (memoized)
  const { ticks, timelineWidth } = useTimeline(
    state.totalDuration || 10,
    state.zoomLevel || 1.0,
    800
  );

  // Measure ACTUAL rendered width (use this for all calculations)
  const [actualWidth, setActualWidth] = useState(timelineWidth);

  useEffect(() => {
    if (!timelineRef.current) return;

    const measureWidth = () => {
      // Use requestAnimationFrame to ensure measurement happens after render
      requestAnimationFrame(() => {
        const rect = timelineRef.current.getBoundingClientRect();
        if (rect.width > 1) { // Only update if we have a valid width
          setActualWidth(rect.width);
          console.log('📏 Measured actual timeline width:', rect.width, 'px');
        }
      });
    };

    measureWidth();
    window.addEventListener('resize', measureWidth);
    return () => window.removeEventListener('resize', measureWidth);
  }, [timelineWidth, state.zoomLevel]);

  // Calculate actual pixels per second from actual width
  const actualPixelsPerSecond = useMemo(() => {
    return actualWidth / (state.totalDuration || 10);
  }, [actualWidth, state.totalDuration]);

  // Recalculate ticks based on actual rendered width (same as Timeline.js does)
  const adjustedTicks = useMemo(() => {
    const pixelsPerSecond = actualWidth / (state.totalDuration || 10);
    return ticks.map(tick => ({
      ...tick,
      position: tick.time * pixelsPerSecond
    }));
  }, [ticks, actualWidth, state.totalDuration]);

  // Custom hook for audio playback (connects to global state via dispatch)
  const { seek } = useAudioPlayback(
    state.tracks,
    state.isPlaying,
    dispatch,
    state.totalDuration || 10,
    state.playheadPosition || 0,
    state.bpm || 120,
    state.masterGain || 0.8,  // Master gain (default 80% to prevent clipping)
    state.beatsPerBar || 4,
    state.meterDenominator || 4,
  );

  // Handle timeline click to seek - use actual width
  const handleTimelineClick = useCallback((e) => {
    if (!timelineRef.current) return;

    const rect = timelineRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const timeInSeconds = clickX / actualPixelsPerSecond;

    // Clamp to valid range
    const clampedTime = Math.max(0, Math.min(timeInSeconds, state.totalDuration || 10));

    console.log('🖱️ Timeline click:', {
      clickX,
      actualWidth,
      actualPixelsPerSecond,
      calculatedTime: timeInSeconds,
      clampedTime
    });

    seek(clampedTime);
  }, [actualPixelsPerSecond, actualWidth, state.totalDuration, seek]);

  // Log state changes for debugging
  useEffect(() => {
    console.log('🎵 Playback state:', state.isPlaying ? 'PLAYING' : 'PAUSED', 'Position:', state.playheadPosition?.toFixed(2) || 0);
  }, [state.isPlaying, state.playheadPosition]);

  return (
    <div className="timeline-wrapper">
      <div id="scene-bar-overlay">
        <SceneMarkers />
      </div>
      <div
        id="timeline-bar"
        ref={timelineRef}
        onClick={handleTimelineClick}
        style={{
          width: `${actualWidth}px`,
          cursor: 'pointer'
        }}
      >
        {/* Render ticks declaratively - using adjusted positions based on actual width */}
        {adjustedTicks.map(tick => (
          <TimelineTick
            key={tick.id}
            time={tick.time}
            position={tick.position}
            label={tick.label}
            isMajor={true}
          />
        ))}

        {/* Playhead cursor (synced with audio playback) */}
        <PlayheadCursor
          position={state.playheadPosition || 0}
          totalDuration={state.totalDuration || 10}
          actualWidth={actualWidth}
        />
      </div>
    </div>
  );
});

TimelineWrapper.displayName = 'TimelineWrapper';

export default TimelineWrapper;
