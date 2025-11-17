import React, { useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import { useTimeline } from '../../hooks/useTimeline';
import MIDITrackVisualization from './MIDITrackVisualization';
import styles from './DAW.module.css';

/**
 * CompositeMIDIView - Shows all MIDI tracks combined in one visualization
 * Used when the bus is collapsed to show a master view
 */
const CompositeMIDIView = ({ tracks, busId, trackHeight = 72 }) => {
  const { state } = useApp();
  // Get first MIDI track to extract common data - use useMemo to avoid recalculation
  const firstMidiTrack = useMemo(() =>
    tracks.find(t => t.type === 'midi'),
    [tracks]
  );

  // Combine all notes from all tracks (no voice colors for timeline view)
  const combinedMidiData = useMemo(() => {
    if (!firstMidiTrack) return null;

    const allNotes = [];
    let maxDuration = 0;

    tracks.forEach((track, trackIndex) => {
      if (track.type === 'midi' && track.midiData?.notes) {
        // Add notes from this track (no voiceIndex for timeline view)
        track.midiData.notes.forEach(note => {
          allNotes.push(note);
          const endTime = note.time + note.duration;
          if (endTime > maxDuration) maxDuration = endTime;
        });
      }
    });

    return {
      notes: allNotes,
      duration: maxDuration || firstMidiTrack.midiData.duration,
      tempo: firstMidiTrack.midiData.tempo
      // No voiceColors for timeline view
    };
  }, [tracks, firstMidiTrack]);

  // Calculate timeline metrics
  const { pixelsPerSecond } = useTimeline(
    combinedMidiData?.duration || 10,
    1.0, // Use state zoom if available
    950
  );

  // Now check for early return after all hooks are called
  if (!firstMidiTrack || !combinedMidiData) return null;

  // No tempo scaling - notes are already converted to timeline BPM
  const fullWidth = combinedMidiData.duration * pixelsPerSecond;

  return (
    <div
      className={styles.track}
      style={{
        transform: 'translate3d(0px, 0px, 0)',
        width: `${fullWidth}px`,
        height: `${trackHeight}px`
      }}
    >
      <MIDITrackVisualization
        midiData={combinedMidiData}
        width={fullWidth}
        height={trackHeight}
        pixelsPerSecond={pixelsPerSecond}
        startTime={0}
        endTime={combinedMidiData.duration}
        timelineBpm={state.bpm || 120}
      />
    </div>
  );
};

export default CompositeMIDIView;
