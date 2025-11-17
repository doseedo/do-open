import React from 'react';
import { useApp } from '../../context/AppContext';
import TrackBus from './TrackBus';

/**
 * TrackBox Component - Container for all track busses
 * Dynamically renders buses from state.buses array
 */
function TrackBox() {
  const { state } = useApp();

  // Icon mapping for bus types
  const busIcons = {
    'VO': 'fa-video',
    'Music': 'fa-music',
    'SFX': 'fa-volume-high'
  };

  return (
    <div className="trackbox">
      {/* Render all buses from state */}
      {state.buses.map((bus) => (
        <TrackBus
          key={bus.id}
          bus={bus}
          icon={busIcons[bus.type] || 'fa-music'}
        />
      ))}
    </div>
  );
}

export default TrackBox;
