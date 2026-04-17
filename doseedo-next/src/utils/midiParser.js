/**
 * MIDI Parser Utility
 * Parses MIDI files and extracts note data for visualization
 */

export const parseMIDIFile = async (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const arrayBuffer = e.target.result;
        const midiData = parseMIDIData(new Uint8Array(arrayBuffer));
        resolve(midiData);
      } catch (error) {
        reject(error);
      }
    };

    reader.onerror = () => reject(new Error('Failed to read MIDI file'));
    reader.readAsArrayBuffer(file);
  });
};

/**
 * Parse MIDI from ArrayBuffer
 * @param {ArrayBuffer} arrayBuffer - MIDI file data
 * @returns {Object} - Parsed MIDI data with notes array
 */
export const parseMIDI = (arrayBuffer) => {
  return parseMIDIData(new Uint8Array(arrayBuffer));
};

const parseMIDIData = (data) => {
  let offset = 0;

  // Parse header chunk
  const headerChunk = parseChunk(data, offset);
  if (headerChunk.type !== 'MThd') {
    throw new Error('Invalid MIDI file: Missing header');
  }

  offset += 8 + headerChunk.length;

  const format = (headerChunk.data[0] << 8) | headerChunk.data[1];
  const numTracks = (headerChunk.data[2] << 8) | headerChunk.data[3];
  const division = (headerChunk.data[4] << 8) | headerChunk.data[5];

  const ticksPerBeat = division & 0x7FFF;

  // Parse track chunks
  const tracks = [];
  let globalTempo = 500000; // Will be updated from first track with tempo

  for (let i = 0; i < numTracks; i++) {
    const trackChunk = parseChunk(data, offset);
    if (trackChunk.type !== 'MTrk') {
      throw new Error('Invalid MIDI file: Missing track chunk');
    }

    // For first track, get the tempo. For subsequent tracks, use the global tempo.
    const track = parseTrack(trackChunk.data, ticksPerBeat, globalTempo);
    tracks.push(track);

    // Update global tempo from first track's tempo changes
    if (i === 0 && track.tempoMicroseconds) {
      globalTempo = track.tempoMicroseconds;
      console.log(`🎵 Using tempo from first track: ${track.tempo} BPM for all tracks`);
    }

    offset += 8 + trackChunk.length;
  }

  // Combine all tracks and extract notes
  const allNotes = [];
  let maxTime = 0;
  const maxNotes = 10000; // Limit for very large files
  let noteCount = 0;

  tracks.forEach(track => {
    track.notes.forEach(note => {
      if (noteCount < maxNotes) {
        allNotes.push(note);
        noteCount++;
      }
      const endTime = note.time + note.duration;
      if (endTime > maxTime) maxTime = endTime;
    });
  });

  // Sort notes by time (only if not too many notes)
  if (allNotes.length < 5000) {
    allNotes.sort((a, b) => a.time - b.time);
  }

  console.log(`🎹 MIDI parsed: ${numTracks} tracks, ${allNotes.length} notes, ${maxTime.toFixed(2)}s duration`);

  return {
    format,
    numTracks,
    ticksPerBeat,
    tracks,
    notes: allNotes,
    duration: maxTime,
    tempo: tracks[0]?.tempo || 120 // Default tempo
  };
};

const parseChunk = (data, offset) => {
  const type = String.fromCharCode(
    data[offset],
    data[offset + 1],
    data[offset + 2],
    data[offset + 3]
  );

  const length = (data[offset + 4] << 24) |
                 (data[offset + 5] << 16) |
                 (data[offset + 6] << 8) |
                 data[offset + 7];

  const chunkData = data.slice(offset + 8, offset + 8 + length);

  return { type, length, data: chunkData };
};

const parseTrack = (data, ticksPerBeat, initialTempo = 500000) => {
  let offset = 0;
  let time = 0;
  let tempo = Math.round(60000000 / initialTempo); // BPM
  let microsecondsPerBeat = initialTempo; // Use provided tempo or default

  const notes = [];
  const noteOnMap = new Map(); // Track note-on events

  let lastStatus = 0;

  while (offset < data.length) {
    // Parse delta time
    const { value: deltaTime, bytesRead } = parseVariableLength(data, offset);
    offset += bytesRead;
    time += deltaTime;

    if (offset >= data.length) break;

    // Parse event
    let status = data[offset];

    // Handle running status
    if ((status & 0x80) === 0) {
      status = lastStatus;
    } else {
      offset++;
      lastStatus = status;
    }

    const eventType = status & 0xF0;
    const channel = status & 0x0F;

    if (eventType === 0x90) {
      // Note On
      const noteNumber = data[offset++];
      const velocity = data[offset++];

      if (velocity === 0) {
        // Note off (velocity 0)
        const noteOn = noteOnMap.get(noteNumber);
        if (noteOn) {
          const timeInSeconds = ticksToSeconds(time, ticksPerBeat, microsecondsPerBeat);
          const durationInSeconds = timeInSeconds - noteOn.timeInSeconds;

          notes.push({
            note: noteNumber,
            time: noteOn.timeInSeconds,
            duration: durationInSeconds,
            velocity: noteOn.velocity
          });

          noteOnMap.delete(noteNumber);
        }
      } else {
        // Real note on
        const timeInSeconds = ticksToSeconds(time, ticksPerBeat, microsecondsPerBeat);
        noteOnMap.set(noteNumber, {
          tick: time,
          timeInSeconds,
          velocity
        });
      }
    } else if (eventType === 0x80) {
      // Note Off
      const noteNumber = data[offset++];
      const velocity = data[offset++];

      const noteOn = noteOnMap.get(noteNumber);
      if (noteOn) {
        const timeInSeconds = ticksToSeconds(time, ticksPerBeat, microsecondsPerBeat);
        const durationInSeconds = timeInSeconds - noteOn.timeInSeconds;

        notes.push({
          note: noteNumber,
          time: noteOn.timeInSeconds,
          duration: durationInSeconds,
          velocity: noteOn.velocity
        });

        noteOnMap.delete(noteNumber);
      }
    } else if (eventType === 0xB0) {
      // Control Change
      offset += 2;
    } else if (eventType === 0xC0) {
      // Program Change
      offset += 1;
    } else if (eventType === 0xE0) {
      // Pitch Bend
      offset += 2;
    } else if (status === 0xFF) {
      // Meta event
      const metaType = data[offset++];
      const { value: metaLength, bytesRead: metaBytesRead } = parseVariableLength(data, offset);
      offset += metaBytesRead;

      if (metaType === 0x51 && metaLength === 3) {
        // Set Tempo
        microsecondsPerBeat = (data[offset] << 16) | (data[offset + 1] << 8) | data[offset + 2];
        tempo = Math.round(60000000 / microsecondsPerBeat);
        console.log(`   Found tempo in track: ${tempo} BPM (${microsecondsPerBeat} µs/beat)`);
      }

      offset += metaLength;
    } else if (status === 0xF0 || status === 0xF7) {
      // SysEx event
      const { value: sysexLength, bytesRead: sysexBytesRead } = parseVariableLength(data, offset);
      offset += sysexBytesRead + sysexLength;
    } else {
      // Unknown event, skip
      break;
    }
  }

  return { notes, tempo, tempoMicroseconds: microsecondsPerBeat };
};

const parseVariableLength = (data, offset) => {
  let value = 0;
  let bytesRead = 0;
  let byte;

  do {
    byte = data[offset + bytesRead];
    value = (value << 7) | (byte & 0x7F);
    bytesRead++;
  } while ((byte & 0x80) !== 0 && bytesRead < 4);

  return { value, bytesRead };
};

const ticksToSeconds = (ticks, ticksPerBeat, microsecondsPerBeat) => {
  return (ticks / ticksPerBeat) * (microsecondsPerBeat / 1000000);
};

export const getNoteNameFromMidi = (midiNote) => {
  const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
  const octave = Math.floor(midiNote / 12) - 1;
  const noteName = noteNames[midiNote % 12];
  return `${noteName}${octave}`;
};
