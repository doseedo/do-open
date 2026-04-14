import React, { useCallback, useMemo } from 'react';
import { useApp } from '../../context/AppContext';
import OptimizedTrack from './OptimizedTrack';
import CompositeBusWaveform from './CompositeBusWaveform';
import CompositeMIDIView from './CompositeMIDIView';
import LevelMeter from './LevelMeter';
import PanKnob from './PanKnob';
import ReverbSlider from './ReverbSlider';
import styles from './DAW.module.css';

/**
 * BusRow Component
 * Represents a single audio bus with tracks
 *
 * @param {Object} bus - Bus object containing id, type, name, tracks, gain, mute, solo, expanded
 * @param {string} icon - Font Awesome icon class (fallback)
 * @param {number} trackHeight - Dynamic track height in pixels
 * @param {Map} gainNodes - Map of trackId -> gainNode for audio metering
 * @param {boolean} draggable - Whether the bus can be dragged
 * @param {function} onDragStart - Drag start handler
 * @param {function} onDragOver - Drag over handler
 * @param {function} onDragLeave - Drag leave handler
 * @param {function} onDrop - Drop handler
 * @param {function} onDragEnd - Drag end handler
 * @param {boolean} isDragOver - Whether another bus is being dragged over this one
 */
const BusRow = React.memo(({
  bus,
  icon,
  trackHeight = 72,
  gainNodes,
  draggable = false,
  onDragStart,
  onDragOver,
  onDragLeave,
  onDrop,
  onDragEnd,
  isDragOver = false,
  pluginMode = false
}) => {
  const { state, dispatch } = useApp();
  const [isFileDragOver, setIsFileDragOver] = React.useState(false);

  // Determine which instrument icon to use based on track metadata
  const getInstrumentIcon = useCallback(() => {
    // Resolve an instrument icon path (white PNG) for the bus
    if (bus.tracks.length > 0) {
      const firstTrack = bus.tracks[0];
      const subgroup = firstTrack.instrumentSubgroup?.toLowerCase();
      const SUB = {
        'violin': '/assets/icons/violin.png', 'viola': '/assets/icons/violin.png', 'cello': '/assets/icons/cello.png',
        'piano': '/assets/icons/piano.png', 'acoustic_piano': '/assets/icons/piano.png', 'keys': '/assets/icons/keyboard.png',
        'acoustic guitar': '/assets/icons/acguitar.png', 'acoustic_guitar': '/assets/icons/acguitar.png',
        'electric guitar': '/assets/icons/elecgtr.png', 'electric_guitar': '/assets/icons/elecgtr.png',
        'electric_bass': '/assets/icons/elecbass.png', 'upright_bass': '/assets/icons/elecbass.png',
        'trumpet': '/assets/icons/tpt.png', 'trombone': '/assets/icons/tbn.png', 'tuba': '/assets/icons/tuba.png',
        'french_horn': '/assets/icons/tuba.png', 'ensemble_brass': '/assets/icons/trumpetens.png',
        'sax': '/assets/icons/sax.png', 'clarinet': '/assets/icons/sax.png', 'bassoon': '/assets/icons/sax.png',
        'flute': '/assets/icons/flute.png', 'oboe': '/assets/icons/flute.png',
        'drum_kit': '/assets/icons/drumkit.png', 'percussion': '/assets/icons/drumkit.png', 'electronic': '/assets/icons/elecdrums.png',
      };
      if (subgroup && SUB[subgroup]) return SUB[subgroup];

      const group = firstTrack.instrumentGroup?.toLowerCase();
      const GROUP = {
        'strings': '/assets/icons/violin.png', 'brass': '/assets/icons/trumpetens.png', 'winds': '/assets/icons/sax.png',
        'piano': '/assets/icons/piano.png', 'guitar': '/assets/icons/acguitar.png', 'bass': '/assets/icons/elecbass.png',
        'drums': '/assets/icons/drumkit.png',
      };
      if (group && GROUP[group]) return GROUP[group];

      // PANNs classifier may have set metadata.instrument as a coarse type
      const classified = firstTrack.metadata?.instrument;
      if (classified && GROUP[classified]) return GROUP[classified];
    }

    // Check bus name as a final fallback
    const n = bus.name?.toLowerCase() || '';
    if (n.includes('brass')) return '/assets/icons/trumpetens.png';
    if (n.includes('drum')) return '/assets/icons/drumkit.png';
    if (n.includes('string')) return '/assets/icons/violin.png';
    if (n.includes('guitar')) return '/assets/icons/acguitar.png';
    if (n.includes('piano') || n.includes('key')) return '/assets/icons/piano.png';
    return null;
  }, [bus.tracks, bus.name]);

  // Removed debug logging for performance

  // hasStems / hasPendingUpload need to be declared BEFORE
  // handleExpandToggle so the useCallback's dependency array can read
  // the latter without hitting a TDZ ReferenceError at init time.
  const hasStems = useMemo(() => {
    return bus.tracks.some(t => t.metadata?.type === 'stem');
  }, [bus.tracks]);

  const hasPendingUpload = useMemo(() => {
    return bus.tracks.some(t => t.metadata?.type === 'uploaded') && !hasStems;
  }, [bus.tracks, hasStems]);

  const handleExpandToggle = useCallback(() => {
    // Lock expansion while stems are still separating — the bus is
    // effectively a single track until then, and expanding would
    // reveal only the uploaded master (which will be hidden anyway
    // once stems arrive), creating a visual flicker.
    if (hasPendingUpload) return;
    dispatch({ type: 'TOGGLE_BUS_EXPANDED', payload: { busId: bus.id } });
  }, [dispatch, bus.id, hasPendingUpload]);

  const handleGainChange = useCallback((e) => {
    const value = parseFloat(e.target.value);
    dispatch({ type: 'UPDATE_BUS_GAIN', payload: { busId: bus.id, gain: value } });
  }, [dispatch, bus.id]);

  const handlePanChange = useCallback((e) => {
    const value = parseFloat(e.target.value);
    dispatch({ type: 'UPDATE_BUS_PAN', payload: { busId: bus.id, pan: value } });
  }, [dispatch, bus.id]);

  const handleReverbChange = useCallback((e) => {
    const value = parseFloat(e.target.value);
    dispatch({ type: 'UPDATE_BUS_REVERB', payload: { busId: bus.id, reverbSend: value } });
  }, [dispatch, bus.id]);

  const handleMute = useCallback(() => {
    dispatch({ type: 'TOGGLE_BUS_MUTE', payload: { busId: bus.id } });
  }, [dispatch, bus.id]);

  const handleSolo = useCallback(() => {
    dispatch({ type: 'TOGGLE_BUS_SOLO', payload: { busId: bus.id } });
  }, [dispatch, bus.id]);

  const handleTrackSelect = useCallback((trackId) => {
    dispatch({ type: 'SELECT_TRACK', payload: { trackId } });
  }, [dispatch]);

  const handleTrackGainChange = useCallback((trackId, e) => {
    const value = parseFloat(e.target.value);
    dispatch({ type: 'UPDATE_TRACK_GAIN', payload: { busId: bus.id, trackId, gain: value } });
  }, [dispatch, bus.id]);

  const handleTrackPanChange = useCallback((trackId, e) => {
    const value = parseFloat(e.target.value);
    dispatch({ type: 'UPDATE_TRACK_PAN', payload: { busId: bus.id, trackId, pan: value } });
  }, [dispatch, bus.id]);

  const handleTrackReverbChange = useCallback((trackId, e) => {
    const value = parseFloat(e.target.value);
    dispatch({ type: 'UPDATE_TRACK_REVERB', payload: { busId: bus.id, trackId, reverbSend: value } });
  }, [dispatch, bus.id]);

  const handleTrackMute = useCallback((trackId, e) => {
    e.stopPropagation(); // Prevent track selection
    dispatch({ type: 'TOGGLE_TRACK_MUTE', payload: { busId: bus.id, trackId } });
  }, [dispatch, bus.id]);

  const handleTrackSolo = useCallback((trackId, e) => {
    e.stopPropagation(); // Prevent track selection
    dispatch({ type: 'TOGGLE_TRACK_SOLO', payload: { busId: bus.id, trackId } });
  }, [dispatch, bus.id]);

  // Check if this bus has multitrack MIDI (must be defined before handleBusClick)
  const hasMultitrackMIDI = useMemo(() => {
    const midiTracks = bus.tracks.filter(t => t.type === 'midi');
    return midiTracks.length > 1;
  }, [bus.tracks]);

  // Check if this bus has multiple tracks (audio or MIDI)
  const hasMultipleTracks = useMemo(() => {
    return bus.tracks.length > 1;
  }, [bus.tracks]);

  // Tracks visible in the expanded list: stems when we have them,
  // otherwise everything. The uploaded master stays in bus.tracks (for
  // analysis + mask playback) but is filtered out of the UI.
  // (hasStems and hasPendingUpload are defined higher up so the
  // expand-toggle callback can reference them without TDZ errors.)
  const visibleTracks = useMemo(() => {
    if (!hasStems) return bus.tracks;
    return bus.tracks.filter(t => t.metadata?.type !== 'uploaded');
  }, [bus.tracks, hasStems]);

  const handleBusClick = useCallback(() => {
    console.log(`🖱️ Bus clicked: ${bus.name}, hasMultitrackMIDI: ${hasMultitrackMIDI}, hasMultipleTracks: ${hasMultipleTracks}`);

    // If this is a multitrack MIDI bus, create a composite track for MIDI editing
    if (hasMultitrackMIDI) {
      console.log(`🎼 Creating composite track for multitrack MIDI bus`);
      // Combine all MIDI tracks into one composite track
      const allNotes = [];
      let maxDuration = 0;
      let combinedTempo = 120;
      const voiceColors = [];

      // Generate distinct colors for each voice
      const generateVoiceColor = (index, total) => {
        const hue = (index * 360 / total) % 360;
        return { hue, saturation: 70, lightness: 55 };
      };

      bus.tracks.forEach((track, trackIndex) => {
        if (track.type === 'midi' && track.midiData?.notes) {
          // Generate color for this voice
          const color = generateVoiceColor(trackIndex, bus.tracks.length);
          voiceColors.push(color);

          // Use tempo from first track
          if (trackIndex === 0) {
            combinedTempo = track.midiData.tempo || 120;
          }

          // Add notes with voice index
          track.midiData.notes.forEach(note => {
            allNotes.push({
              ...note,
              voiceIndex: trackIndex
            });
            const endTime = note.time + note.duration;
            if (endTime > maxDuration) maxDuration = endTime;
          });

          // Update maxDuration from track duration
          if (track.duration > maxDuration) maxDuration = track.duration;
        }
      });

      // Preserve the original MIDI file from first track (all tracks share the same file)
      const firstMidiTrack = bus.tracks.find(t => t.type === 'midi');
      const originalMidiFile = firstMidiTrack?.file;

      // Create composite track object
      const compositeTrack = {
        id: `composite-${bus.id}`,
        name: `${bus.name} (All Voices)`,
        type: 'midi',
        midiData: {
          notes: allNotes,
          tempo: combinedTempo,
          duration: maxDuration,
          voiceColors: voiceColors,
          isMultitrack: true,
          trackCount: bus.tracks.filter(t => t.type === 'midi').length
        },
        duration: maxDuration,
        isComposite: true,
        compositeBusId: bus.id,
        file: originalMidiFile // Preserve original MIDI file for generation
      };

      // Select the composite track for MIDI editing AND the bus (for showing both MIDI window and Bus Info)
      console.log(`✅ Dispatching SELECT_TRACK for composite: ${compositeTrack.id} with busId: ${bus.id}`);
      dispatch({ type: 'SELECT_TRACK', payload: { trackId: compositeTrack.id, compositeTrack, busId: bus.id } });
    } else if (hasMultipleTracks) {
      // For multi-track audio buses, select the bus (shows Bus Info)
      console.log(`🎵 Selecting bus for multi-track audio: ${bus.id}`);
      dispatch({ type: 'SELECT_BUS', payload: { busId: bus.id } });
    } else {
      // For single-track buses, select the bus normally
      console.log(`✅ Dispatching SELECT_BUS for: ${bus.id}`);
      dispatch({ type: 'SELECT_BUS', payload: { busId: bus.id } });
    }
  }, [dispatch, bus.id, bus.name, bus.tracks, hasMultitrackMIDI, hasMultipleTracks]);

  // Drag and drop handlers for audio files
  const handleFileDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsFileDragOver(true);
  }, []);

  const handleFileDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsFileDragOver(false);
  }, []);

  const handleFileDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsFileDragOver(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      const file = files[0];

      // Check if it's a MIDI file
      const isMidiFile = file.name.toLowerCase().endsWith('.mid') ||
                        file.name.toLowerCase().endsWith('.midi') ||
                        file.type === 'audio/midi' ||
                        file.type === 'audio/mid';

      // Check if it's an audio file
      if (!file.type.startsWith('audio/') && !isMidiFile) {
        alert('Please drop an audio or MIDI file');
        return;
      }

      console.log(`🎵 ${isMidiFile ? 'MIDI' : 'Audio'} file dropped on ${bus.name}:`, file.name);

      if (isMidiFile) {
        // Handle MIDI file
        const reader = new FileReader();
        reader.onload = async (e) => {
          try {
            // Parse MIDI file using Tone.js
            const Midi = (await import('@tonejs/midi')).Midi;
            const midi = new Midi(e.target.result);

            console.log(`🎹 Parsed MIDI: ${midi.tracks.length} tracks, duration: ${midi.duration}s`);

            // Filter out empty tracks
            const validTracks = midi.tracks.filter(track => track.notes.length > 0);

            if (validTracks.length === 0) {
              alert('MIDI file has no notes');
              return;
            }

            // If this is a multi-track MIDI, store the original file in bus metadata
            if (validTracks.length > 1) {
              const originalMidiBlob = new Blob([await file.arrayBuffer()], { type: 'audio/midi' });
              dispatch({
                type: 'UPDATE_BUS_METADATA',
                payload: {
                  busId: bus.id,
                  metadata: {
                    originalMultitrackMidi: originalMidiBlob,
                    originalMidiFilename: file.name
                  }
                }
              });
              console.log(`💾 Stored original multi-track MIDI file for bus ${bus.id}`);
            }

            // Create separate track for each MIDI track
            validTracks.forEach(async (midiTrack, trackIndex) => {
              const trackId = `track-${Date.now()}-${trackIndex}`;

              // Create a single-track MIDI file for this track
              const { Midi: ToneMidi } = await import('@tonejs/midi');
              const singleTrackMidi = new ToneMidi();
              singleTrackMidi.header.setTempo(midi.header.tempos[0]?.bpm || 120);
              // Note: ppq is readonly in Tone.js, it defaults to 480
              midi.header.timeSignatures.forEach(ts => {
                singleTrackMidi.header.timeSignatures.push(ts);
              });
              const newTrack = singleTrackMidi.addTrack();
              newTrack.name = midiTrack.name;
              midiTrack.notes.forEach(note => {
                newTrack.addNote({
                  midi: note.midi,
                  time: note.time,
                  duration: note.duration,
                  velocity: note.velocity
                });
              });

              // Convert to binary and create blob
              const singleTrackBlob = new Blob([singleTrackMidi.toArray()], { type: 'audio/midi' });

              const track = {
                id: trackId,
                name: midiTrack.name || `${file.name} - Track ${trackIndex + 1}`,
                type: 'midi',
                midiData: {
                  duration: midi.duration,
                  notes: midiTrack.notes.map(note => ({
                    midi: note.midi,
                    note: note.midi,
                    time: note.time,
                    duration: note.duration,
                    velocity: note.velocity,
                    name: note.name
                  })),
                  tempo: midi.header.tempos[0]?.bpm || 120,
                  tempos: midi.header.tempos.map(t => ({ time: t.time, bpm: t.bpm })),
                  timeSignatures: midi.header.timeSignatures.map(ts => ({
                    time: ts.time,
                    numerator: ts.timeSignature[0],
                    denominator: ts.timeSignature[1]
                  })),
                  ppq: midi.header.ppq,
                  isMultitrack: validTracks.length > 1
                },
                duration: midi.duration,
                startPosition: 0,
                gain: 1.0,
                isMuted: false,
                isSolo: false,
                fx: {
                  reverb: 0,
                  fadeIn: 0,
                  fadeOut: 0
                },
                metadata: {
                  type: 'uploaded',
                  midiBlob: singleTrackBlob,
                  midiFilename: `${file.name.replace(/\.mid$/i, '')}_track_${trackIndex + 1}.mid`,
                  instrument: midiTrack.instrument?.name,
                  isMultitrackSource: validTracks.length > 1
                }
              };

              // Add track to this bus
              dispatch({
                type: 'ADD_TRACK',
                payload: { busId: bus.id, track }
              });
            });

            // Expand the bus if it's collapsed
            if (!bus.expanded) {
              dispatch({ type: 'TOGGLE_BUS_EXPANDED', payload: { busId: bus.id } });
            }

            console.log(`✅ MIDI file added to ${bus.name} with ${validTracks.length} tracks`);
          } catch (error) {
            console.error('❌ Error parsing MIDI file:', error);
            alert(`Failed to load MIDI file: ${error.message}`);
          }
        };
        reader.readAsArrayBuffer(file);
      } else {
        // Handle audio file
        // Create a blob URL for the audio file
        const audioUrl = URL.createObjectURL(file);

        // Get audio duration
        const audio = new Audio();
        audio.src = audioUrl;

        audio.addEventListener('loadedmetadata', () => {
          const duration = audio.duration;

          // Create the track
          const track = {
            id: `track-${Date.now()}`,
            name: file.name,
            audioUrl: audioUrl,
            duration: duration,
            startPosition: 0, // Always add at the beginning when dropping on a bus
            gain: 1.0,
            isMuted: false,
            isSolo: false,
            cropStart: 0,
            cropEnd: 0,
            fx: {
              reverb: 0,
              fadeIn: 0.2,
              fadeOut: 1.0
            }
          };

          // Add track to this bus
          dispatch({
            type: 'ADD_TRACK',
            payload: { busId: bus.id, track }
          });

          // Expand the bus if it's collapsed
          if (!bus.expanded) {
            dispatch({ type: 'TOGGLE_BUS_EXPANDED', payload: { busId: bus.id } });
          }

          console.log(`✅ Audio file added to ${bus.name}`);
        });
      }
    }
  }, [bus.id, bus.name, bus.expanded, dispatch]);

  // Handle double-click to create empty MIDI track
  const handleDoubleClick = useCallback((e) => {
    // Only create MIDI track if bus is empty
    if (bus.tracks.length > 0) {
      return;
    }

    e.preventDefault();
    e.stopPropagation();

    console.log(`🎹 Creating empty MIDI track for ${bus.name}`);

    // Count existing MIDI tracks to generate a unique name
    const midiTrackCount = bus.tracks.filter(t => t.type === 'midi').length;
    const trackNumber = midiTrackCount + 1;

    // Create an empty MIDI track
    const trackId = `track-${Date.now()}`;
    const newTrack = {
      id: trackId,
      name: `MIDI ${trackNumber}`,
      type: 'midi',
      midiData: {
        duration: 30, // Default 30 seconds
        tracks: [],
        notes: [],
        tempo: 120, // Set tempo for playback synchronization
        tempos: [{ time: 0, bpm: 120 }],
        timeSignatures: [{ time: 0, numerator: 4, denominator: 4 }],
        ppq: 480
      },
      duration: 30,
      startPosition: 0,
      color: `hsl(${Math.random() * 360}, 70%, 60%)`,
      solo: false,
      mute: false,
      volume: 1.0,
      pan: 0,
      cropStart: 0,
      cropEnd: 0
    };

    dispatch({
      type: 'ADD_TRACK',
      payload: {
        busId: bus.id,
        track: newTrack
      }
    });

    // Expand the bus if it's collapsed
    if (!bus.expanded) {
      dispatch({ type: 'TOGGLE_BUS_EXPANDED', payload: { busId: bus.id } });
    }

    console.log(`✅ Empty MIDI track created for ${bus.name}`);
  }, [bus.id, bus.name, bus.tracks.length, bus.expanded, dispatch]);

  // Calculate heights for smooth transitions. Use visibleTracks
  // (which excludes the hidden uploaded master) so the expanded bus
  // doesn't reserve an empty row where the master used to be.
  const tracksHeight = useMemo(() => {
    return bus.expanded ? visibleTracks.length * trackHeight : 0;
  }, [bus.expanded, visibleTracks.length, trackHeight]);

  const instrumentIcon = getInstrumentIcon();

  return (
    <div
      data-bus-id={bus.id}
      className={`${styles.busRow} ${isDragOver ? styles.dragOver : ''}`}
      draggable={draggable}
      onDragStart={onDragStart}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      onDragEnd={onDragEnd}
    >
      {/* Bus Header Container - unified icon + label */}
      <div className={`${styles.busHeaderContainer} ${
        (state.selectedBus?.id === bus.id) ? styles.selected : ''
      } ${bus.expanded ? styles.expanded : ''}`} style={{ height: `${trackHeight}px` }}>

        {/* Bus Icon */}
        <div className={`${styles.busIconContainer} ${bus.expanded ? styles.expanded : ''}`} onClick={handleExpandToggle}>
          <i className={`fa-solid fa-caret-right ${styles.busExpandCaret} ${bus.expanded ? styles.expanded : ''}`}></i>
          {instrumentIcon ? (
            <img
              src={instrumentIcon}
              alt="Instrument"
              className={styles.busIconImage}
              style={{ width: 28, height: 28, objectFit: 'contain', opacity: 0.9 }}
            />
          ) : (
            <i
              className={`fa-solid ${icon} ${styles.busIcon}`}
              style={{ color: 'rgba(255,255,255,0.85)' }}
            />
          )}
        </div>

        {/* Bus Label & Controls */}
        <div className={`${styles.busLabelRow} ${bus.expanded ? styles.expanded : ''}`} onClick={handleExpandToggle}>
        <div className={styles.busControls} onClick={(e) => e.stopPropagation()}>
          {pluginMode ? (
            /* Simplified Plugin Mode - stacked layout: name above volume icon */
            <div className={styles.busControlsPluginMode}>
              <div className={styles.busNameLabel} title={bus.name}>
                {bus.name.length > 12 ? bus.name.substring(0, 12) + '...' : bus.name}
              </div>
              <div className={styles.pluginMuteContainer}>
                <button
                  className={`${styles.pluginMuteButton} ${bus.mute ? styles.muted : ''}`}
                  onClick={handleMute}
                  title={bus.mute ? "Unmute" : "Mute"}
                >
                  <i className={`fa-solid ${bus.mute ? 'fa-volume-xmark' : 'fa-volume-high'}`}></i>
                </button>
                <div className={styles.pluginLevelSlider}>
                  <LevelMeter
                    gain={bus.gain}
                    onGainChange={handleGainChange}
                    audioNode={gainNodes && bus.tracks.length > 0 ? gainNodes.get(bus.tracks[0].id) : null}
                  />
                </div>
              </div>
            </div>
          ) : (
            /* Normal Mode - full controls */
            <>
              <div className={styles.busControlsLayout}>
                <div className={styles.busNameRow}>
                  <div className={styles.busNameLabel} title={bus.name}>
                    {bus.name.length > 25 ? bus.name.substring(0, 25) + '...' : bus.name}
                  </div>
                </div>
                <div className={styles.busSliders}>
                  <LevelMeter
                    gain={bus.gain}
                    onGainChange={handleGainChange}
                    audioNode={gainNodes && bus.tracks.length > 0 ? gainNodes.get(bus.tracks[0].id) : null}
                  />
                  <PanKnob
                    pan={bus.pan || 0}
                    onPanChange={handlePanChange}
                  />
                </div>
              </div>
              <div className={styles.busButtons}>
                <button
                  className={`${styles.busButton} ${styles.muteButton} ${bus.mute ? styles.activeMute : ''}`}
                  onClick={handleMute}
                  title="Mute"
                >
                  M
                </button>
                <button
                  className={`${styles.busButton} ${bus.solo ? styles.active : ''}`}
                  onClick={handleSolo}
                  title="Solo"
                >
                  S
                </button>
              </div>
            </>
          )}
        </div>
      </div>
      </div>

      {/* Waveforms - must be in same row as icon and label */}
      <div
        className={`${styles.busTracks} ${!bus.expanded ? styles.collapsed : ''} ${bus.expanded ? styles.expanded : ''} ${isDragOver ? styles.dragOver : ''}`}
        style={{
          height: bus.expanded ? `${trackHeight + tracksHeight}px` : `${trackHeight}px`, // header + tracks (no gap)
          minHeight: `${trackHeight}px`,
          paddingTop: bus.expanded ? `${trackHeight}px` : '0'
        }}
        onDragOver={handleFileDragOver}
        onDragLeave={handleFileDragLeave}
        onDrop={handleFileDrop}
        onDoubleClick={handleDoubleClick}
      >
        {/* The bus-row master waveform lives in the TOP strip (always
            at y=0, height=trackHeight) regardless of expanded state.
            When collapsed it's the only thing in this container. When
            expanded, child stem tracks sit below it (pushed down by the
            paddingTop: trackHeight on the parent). Clicking it toggles
            the bus expansion so the interaction stays identical. */}
        {hasStems || hasMultipleTracks || hasMultitrackMIDI ? (
          <div
            onClick={handleBusClick}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              height: `${trackHeight}px`,
              cursor: 'pointer',
              // No width / no right:0 — the wrapper sizes to its child
              // canvas (which has pixel width = busDuration×pixelsPerSecond),
              // matching the horizontal footprint of a stem track exactly.
              // The canvas protrudes beyond the grid column the same way
              // stem tracks do (busTracks has overflow: visible).
            }}
            className={`${styles.masterTrackView} ${
              (state.selectedBus?.id === bus.id ||
               (state.selectedTrack?.isComposite && state.selectedTrack?.compositeBusId === bus.id))
              ? styles.selected : ''
            }`}
          >
            {hasMultitrackMIDI ? (
              <CompositeMIDIView tracks={bus.tracks} busId={bus.id} trackHeight={trackHeight} />
            ) : (
              /* Composite waveform = Σ per-stem envelopes × stem.gain
                 × bus.gain, normalized to a CONSTANT baseline (Σ at
                 gain=1) so turning a stem down shrinks the master. */
              <CompositeBusWaveform
                bus={bus}
                height={trackHeight}
                color={state.selectedBus?.id === bus.id ? '#8b5cf6' : '#667eea'}
              />
            )}
          </div>
        ) : null}

        {/* Expanded: render the visible child tracks (stems only once
            separation is done; master is hidden but still in bus.tracks
            for analysis + mask playback). Collapsed with a single
            pre-separation track: render that one track here so the
            user still sees a waveform before stems arrive. */}
        {bus.expanded ? (
          visibleTracks.length > 0 ? (
            visibleTracks
              .filter(track => track.expanded !== false)
              .map((track, index) => (
                <OptimizedTrack
                  key={track.id}
                  track={track}
                  busId={bus.id}
                  index={index}
                  isExpanded={true}
                  isSelected={state.selectedTrack?.id === track.id}
                  trackHeight={trackHeight}
                />
              ))
          ) : (
            <div className={styles.emptyBusHint}>
              Double-click to create MIDI track
            </div>
          )
        ) : (
          /* Collapsed + no stems + single track → the single-track
             preview replaces the composite (that branch returned null
             above). */
          !hasStems && !hasMultipleTracks && !hasMultitrackMIDI && bus.tracks.length > 0 ? (
            <OptimizedTrack
              key={bus.tracks[0].id}
              track={bus.tracks[0]}
              busId={bus.id}
              index={0}
              isExpanded={false}
              isSelected={state.selectedTrack?.id === bus.tracks[0].id}
              isMasterView={true}
              trackHeight={trackHeight}
            />
          ) : bus.tracks.length === 0 ? (
            <div className={styles.emptyBusHint}>
              Double-click to create MIDI track
            </div>
          ) : null
        )}
      </div>

      {/* Track Labels (only render when expanded). Uses visibleTracks
          so the hidden uploaded master doesn't get a label row. */}
      {bus.expanded && (
        <div className={styles.trackLabelsColumn}>
          {visibleTracks.map((track, index) => {
            // Resolve a white-PNG icon path for the track from any
            // metadata source (PANNs classification, generation params,
            // stem type, instrument group).
            const getTrackIconImg = () => {
              const GROUP = {
                piano: '/assets/icons/piano.png', guitar: '/assets/icons/acguitar.png', bass: '/assets/icons/elecbass.png',
                strings: '/assets/icons/violin.png', brass: '/assets/icons/trumpetens.png', winds: '/assets/icons/sax.png',
                woodwind: '/assets/icons/sax.png', keys: '/assets/icons/keyboard.png',
                drums: '/assets/icons/drumkit.png', cello: '/assets/icons/cello.png',
                trumpet: '/assets/icons/tpt.png', trombone: '/assets/icons/tbn.png', tuba: '/assets/icons/tuba.png',
              };
              if (track.metadata?.type === 'stem' && track.metadata?.stemType) {
                return GROUP[track.metadata.stemType] || null;
              }
              const ig = track.metadata?.params?.instrumentGroup || track.metadata?.instrumentGroup;
              if (ig && GROUP[ig]) return GROUP[ig];
              if (track.metadata?.instrument && GROUP[track.metadata.instrument]) {
                return GROUP[track.metadata.instrument];
              }
              return null;
            };
            const trackIconImg = getTrackIconImg();

            return (
              <div key={track.id} className={styles.trackRow} style={{ height: `${trackHeight}px` }}>
                {/* Track Header Container - unified icon + label */}
                <div className={`${styles.trackHeaderContainer} ${state.selectedTrack?.id === track.id ? styles.selected : ''}`} style={{ height: `${trackHeight}px` }}>

                  {/* Track Icon Container */}
                  <div className={styles.trackIconContainer}>
                    <i className={`fa-solid fa-caret-right ${styles.trackExpandCaret}`}></i>
                    {trackIconImg ? (
                      <img
                        src={trackIconImg}
                        alt="Instrument"
                        className={styles.trackIconImage}
                        style={{ width: 24, height: 24, objectFit: 'contain', opacity: 0.9 }}
                      />
                    ) : (
                      <i
                        className={`fa-solid ${track.type === 'midi' ? 'fa-music' : 'fa-waveform-lines'} ${styles.trackIcon}`}
                        style={{ color: 'rgba(255,255,255,0.85)' }}
                      />
                    )}
                  </div>

                  {/* Track Label & Controls */}
                  <div
                    className={styles.trackLabel}
                    onClick={() => handleTrackSelect(track.id)}
                  >
                  <div className={styles.trackControls} onClick={(e) => e.stopPropagation()}>
                    {pluginMode ? (
                      /* Simplified Plugin Mode - stacked layout: name above volume icon */
                      <div className={styles.trackControlsPluginMode}>
                        <div className={styles.trackNameLabel} title={track.name || track.audioUrl?.split('/').pop() || 'Untitled'}>
                          {(() => {
                            const displayName = track.name || track.audioUrl?.split('/').pop() || 'Untitled';
                            return displayName.length > 12 ? displayName.substring(0, 12) + '...' : displayName;
                          })()}
                        </div>
                        <div className={styles.pluginMuteContainer}>
                          <button
                            className={`${styles.pluginMuteButton} ${track.isMuted ? styles.muted : ''}`}
                            onClick={(e) => handleTrackMute(track.id, e)}
                            title={track.isMuted ? "Unmute" : "Mute"}
                          >
                            <i className={`fa-solid ${track.isMuted ? 'fa-volume-xmark' : 'fa-volume-high'}`}></i>
                          </button>
                          <div className={styles.pluginLevelSlider}>
                            <LevelMeter
                              gain={track.gain || 1.0}
                              onGainChange={(e) => handleTrackGainChange(track.id, e)}
                              audioNode={gainNodes ? gainNodes.get(track.id) : null}
                            />
                          </div>
                        </div>
                      </div>
                    ) : (
                      /* Normal Mode - full controls */
                      <>
                        <div className={styles.trackControlsLayout}>
                          <div className={styles.trackNameRow}>
                            {track.metadata?.icon && (
                              <i
                                className={`fa-solid ${track.metadata.icon}`}
                                title={track.metadata.instrumentLabel || track.metadata.instrument || 'analyzing'}
                                style={{ marginRight: 6, opacity: 0.7, color: 'rgba(255,255,255,0.9)' }}
                              />
                            )}
                            <div className={styles.trackNameLabel} title={track.name || track.audioUrl?.split('/').pop() || 'Untitled'}>
                              {(() => {
                                const displayName = track.name || track.audioUrl?.split('/').pop() || 'Untitled';
                                return displayName.length > 25 ? displayName.substring(0, 25) + '...' : displayName;
                              })()}
                            </div>
                          </div>
                          <div className={styles.trackSliders}>
                            <LevelMeter
                              gain={track.gain || 1.0}
                              onGainChange={(e) => handleTrackGainChange(track.id, e)}
                              audioNode={gainNodes ? gainNodes.get(track.id) : null}
                            />
                            <PanKnob
                              pan={track.pan || 0}
                              onPanChange={(e) => handleTrackPanChange(track.id, e)}
                            />
                          </div>
                        </div>
                        <div className={styles.trackButtons}>
                          <button
                            className={`${styles.trackButton} ${styles.muteButton} ${track.isMuted ? styles.activeMute : ''}`}
                            onClick={(e) => handleTrackMute(track.id, e)}
                            title="Mute Track"
                          >
                            M
                          </button>
                          <button
                            className={`${styles.trackButton} ${styles.soloButton} ${track.isSolo ? styles.activeSolo : ''}`}
                            onClick={(e) => handleTrackSolo(track.id, e)}
                            title="Solo Track"
                          >
                            S
                          </button>
                        </div>
                      </>
                    )}
                  </div>
                </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
});

BusRow.displayName = 'BusRow';

export default BusRow;
