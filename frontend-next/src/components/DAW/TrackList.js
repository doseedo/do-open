import React, { useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import DraggableTrack from './DraggableTrack';

/**
 * TrackList Component - Renders list of draggable tracks for a bus
 */
const TrackList = React.memo(({ bus, busId, mode, tracks = [] }) => {
  const { state } = useApp();

  // Debug: Log when component renders
  React.useEffect(() => {
    console.log(`🎵 TrackList render for bus ${busId}: ${tracks.length} tracks`, tracks.map(t => t.name));
  }, [busId, tracks.length]);

  // Get pixels per second for track positioning
  const { pixelsPerSecond } = useTimeline(
    state.totalDuration || 10,
    state.zoomLevel || 1.0,
    800
  );

  // Get expansion state from bus
  const isExpanded = bus?.expanded ?? true;

  // Calculate container height based on number of tracks and expansion state
  const containerHeight = useMemo(() => {
    return isExpanded ? tracks.length * 60 : 0; // 60px per track, 0 when collapsed
  }, [tracks.length, isExpanded]);

  return (
    <div
      id={busId}
      className="download-list"
      style={{
        position: 'relative',
        width: '100%',
        height: `${containerHeight}px`,
        minHeight: isExpanded ? '60px' : '0',
        background: '#0a0a0a',
        borderRadius: '4px',
        marginTop: '10px',
        overflow: 'visible', // Allow selected track border to show
        paddingBottom: '4px', // Add space for bottom border highlight
        transition: 'height 0.3s ease'
      }}
    >
      {tracks.map((track, index) => (
        <DraggableTrack
          key={track.id}
          track={track}
          busId={busId}
          mode={mode}
          index={index}
          isSelected={state.selectedTrack?.id === track.id}
          pixelsPerSecond={pixelsPerSecond}
        />
      ))}
    </div>
  );
});

TrackList.displayName = 'TrackList';

export default TrackList;
