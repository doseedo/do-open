import { useMemo } from 'react';

/**
 * Custom hook for timeline calculations
 * Returns ticks, width, and pixels per second based on duration and zoom
 */
export function useTimeline(totalDuration, zoomLevel, containerWidth = 800) {
  // Calculate pixels per second based on zoom
  // containerWidth is now the actual timeline width, no need to subtract
  const pixelsPerSecond = useMemo(() => {
    const zoomedWidth = containerWidth * zoomLevel;
    return totalDuration > 0 ? zoomedWidth / totalDuration : 0;
  }, [containerWidth, zoomLevel, totalDuration]);

  // Calculate timeline width
  const timelineWidth = useMemo(() => {
    return containerWidth * zoomLevel;
  }, [containerWidth, zoomLevel]);

  // Calculate tick interval based on zoom level (adaptive)
  const tickInterval = useMemo(() => {
    if (pixelsPerSecond < 2) return 20;
    if (pixelsPerSecond < 5) return 10;
    if (pixelsPerSecond < 10) return 5;
    if (pixelsPerSecond < 20) return 2;
    return 1;
  }, [pixelsPerSecond]);

  // Generate tick array for rendering
  const ticks = useMemo(() => {
    const tickArray = [];
    for (let t = 0; t <= totalDuration; t += tickInterval) {
      tickArray.push({
        id: `tick-${t}`,
        time: t,
        position: t * pixelsPerSecond,
        label: `${t.toFixed(0)}s`
      });
    }
    return tickArray;
  }, [totalDuration, tickInterval, pixelsPerSecond]);

  return {
    ticks,
    timelineWidth,
    pixelsPerSecond,
    tickInterval
  };
}
