import React from 'react';
import { useApp } from '../../context/AppContext';
import MoreControls from './MoreControls';
import TimelineWrapper from './TimelineWrapper';
import TrackList from './TrackList';

/**
 * Downloads Component - Main timeline and track display area
 * Dynamically renders track lists for all buses
 */
function Downloads() {
  const { state } = useApp();

  return (
    <div className="downloads">
      <MoreControls />
      <TimelineWrapper />

      {/* Render track lists for each bus from state */}
      {state.buses.map((bus) => (
        <TrackList
          key={bus.id}
          bus={bus}
          busId={bus.id}
          mode={bus.type}
          tracks={bus.tracks}
        />
      ))}
    </div>
  );
}

export default Downloads;
