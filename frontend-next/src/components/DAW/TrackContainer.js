import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import TrackBox from './TrackBox';
import TempoControls from './TempoControls';
import MasterTrack from './MasterTrack';
import MasterFXPanels from './MasterFXPanels';

/**
 * TrackContainer Component - Main container for all tracks
 * Matches original doseedo2.html .trackcontainer structure
 */
function TrackContainer() {
  const { state } = useApp();

  return (
    <div className="trackcontainer">
      <TempoControls />
      <TrackBox />
      <MasterFXPanels />
      <MasterTrack />
      <div className="horizontal-resizer"></div>
    </div>
  );
}

export default TrackContainer;
