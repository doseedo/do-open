/**
 * Frontend pitch detection and MIDI conversion using Web Audio API
 * No backend calls required!
 */

import { Midi } from '@tonejs/midi';

/**
 * Detect pitch using autocorrelation method (YIN algorithm)
 * @param {Float32Array} buffer - Audio buffer
 * @param {number} sampleRate - Sample rate in Hz
 * @returns {number|null} - Frequency in Hz or null if no pitch detected
 */
function detectPitch(buffer, sampleRate) {
  // Autocorrelation implementation
  const SIZE = buffer.length;
  const MAX_SAMPLES = Math.floor(SIZE / 2);
  const r1 = 0;
  const r2 = SIZE - 1;
  const rms = Math.sqrt(buffer.reduce((sum, val) => sum + val * val, 0) / SIZE);

  // Silence threshold - LOWERED for better sensitivity
  if (rms < 0.001) return null;  // Changed from 0.01 to 0.001

  // Autocorrelation
  let maxCorr = 0;
  let foundGoodPitch = false;
  let bestOffset = -1;

  for (let offset = 1; offset < MAX_SAMPLES; offset++) {
    let correlation = 0;
    for (let i = 0; i < MAX_SAMPLES; i++) {
      correlation += Math.abs(buffer[i] - buffer[i + offset]);
    }
    correlation = 1 - correlation / MAX_SAMPLES;

    // LOWERED threshold for better pitch detection
    if (correlation > 0.7 && correlation > maxCorr) {  // Changed from 0.9 to 0.7
      maxCorr = correlation;
      bestOffset = offset;
      foundGoodPitch = true;
    }
  }

  if (foundGoodPitch && bestOffset > 0) {
    const frequency = sampleRate / bestOffset;
    // Only return frequencies in musical range (C2 to C8)
    if (frequency >= 65 && frequency <= 4200) {
      return frequency;
    }
  }

  return null;
}

/**
 * Convert frequency to MIDI note number
 * @param {number} frequency - Frequency in Hz
 * @returns {number} - MIDI note number (0-127)
 */
function frequencyToMidi(frequency) {
  return Math.round(69 + 12 * Math.log2(frequency / 440));
}

/**
 * Analyze audio and extract pitch over time
 * @param {AudioBuffer} audioBuffer - Web Audio API AudioBuffer
 * @param {number} hopSize - Analysis hop size in samples (default: 256)
 * @returns {Array<{time: number, pitch: number|null, frequency: number|null}>}
 */
function analyzePitch(audioBuffer, hopSize = 256) {  // Changed from 512 to 256 for better resolution
  const channelData = audioBuffer.getChannelData(0); // Use first channel (mono)
  const sampleRate = audioBuffer.sampleRate;
  const pitchData = [];
  const windowSize = hopSize * 4; // Use 4x hop size for analysis window (larger = more accurate)

  // Analyze in windows
  for (let i = 0; i < channelData.length - windowSize; i += hopSize) {
    const buffer = channelData.slice(i, i + windowSize);
    const frequency = detectPitch(buffer, sampleRate);
    const time = i / sampleRate;

    pitchData.push({
      time,
      frequency,
      pitch: frequency ? frequencyToMidi(frequency) : null
    });
  }

  return pitchData;
}

/**
 * Convert pitch data to MIDI notes with note segmentation
 * @param {Array} pitchData - Array of pitch analysis frames
 * @param {number} minNoteDuration - Minimum note duration in seconds (default: 0.05)
 * @returns {Array<{pitch: number, startTime: number, endTime: number, velocity: number}>}
 */
function pitchToNotes(pitchData, minNoteDuration = 0.05) {  // Changed from 0.1 to 0.05
  const notes = [];
  let currentNote = null;

  for (let i = 0; i < pitchData.length; i++) {
    const frame = pitchData[i];

    if (frame.pitch !== null) {
      // Start new note or continue current
      if (currentNote === null) {
        // Start new note
        currentNote = {
          pitch: frame.pitch,
          startTime: frame.time,
          endTime: frame.time,
          velocity: 80
        };
      } else if (Math.abs(currentNote.pitch - frame.pitch) <= 1) {
        // Same note (allow 1 semitone variation for stability)
        currentNote.endTime = frame.time;
      } else {
        // Different note - save current and start new
        const duration = currentNote.endTime - currentNote.startTime;
        if (duration >= minNoteDuration) {
          notes.push(currentNote);
        }
        currentNote = {
          pitch: frame.pitch,
          startTime: frame.time,
          endTime: frame.time,
          velocity: 80
        };
      }
    } else {
      // Silence - end current note if exists
      if (currentNote !== null) {
        const duration = currentNote.endTime - currentNote.startTime;
        if (duration >= minNoteDuration) {
          notes.push(currentNote);
        }
        currentNote = null;
      }
    }
  }

  // Add last note if exists
  if (currentNote !== null) {
    const duration = currentNote.endTime - currentNote.startTime;
    if (duration >= minNoteDuration) {
      notes.push(currentNote);
    }
  }

  return notes;
}

/**
 * Create MIDI file from audio file
 * @param {File} audioFile - Audio file (WAV, MP3, etc.)
 * @param {Function} onProgress - Optional progress callback (0-1)
 * @returns {Promise<{midiBlob: Blob, metadata: Object}>}
 */
export async function audioToMidi(audioFile, onProgress = null) {
  console.log('🎵 Converting audio to MIDI (frontend-only)...');

  // Decode audio file
  if (onProgress) onProgress(0.1);
  const arrayBuffer = await audioFile.arrayBuffer();
  const audioContext = new (window.AudioContext || window.webkitAudioContext)();
  const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

  console.log(`   Audio: ${audioBuffer.duration.toFixed(2)}s, ${audioBuffer.sampleRate}Hz`);

  // Analyze pitch
  if (onProgress) onProgress(0.3);
  const pitchData = analyzePitch(audioBuffer);
  const framesWithPitch = pitchData.filter(f => f.pitch !== null).length;
  console.log(`   Analyzed ${pitchData.length} frames, ${framesWithPitch} with detected pitch (${(framesWithPitch/pitchData.length*100).toFixed(1)}%)`);

  // Convert to notes
  if (onProgress) onProgress(0.6);
  const notes = pitchToNotes(pitchData);
  console.log(`   Found ${notes.length} notes`);

  // Debug: Show sample of detected pitches
  if (framesWithPitch > 0 && notes.length === 0) {
    const samplePitches = pitchData.filter(f => f.pitch !== null).slice(0, 10);
    console.log(`   ⚠️ Detected pitches but no notes formed. Sample pitches:`, samplePitches.map(p => `${p.pitch}@${p.time.toFixed(2)}s`));
  }

  if (notes.length === 0) {
    throw new Error('No notes detected in audio. Try recording with a clearer tone.');
  }

  // Create MIDI file using @tonejs/midi
  if (onProgress) onProgress(0.8);
  const midi = new Midi();
  const track = midi.addTrack();
  track.name = 'Recorded Voice';

  // Add notes to track
  notes.forEach(note => {
    track.addNote({
      midi: note.pitch,
      time: note.startTime,
      duration: note.endTime - note.startTime,
      velocity: note.velocity / 127 // Tone.js uses 0-1 range
    });
  });

  // Set tempo (estimate from note density or use default)
  midi.header.setTempo(0, 120);

  // Convert to binary MIDI
  const midiArray = midi.toArray();
  const midiBlob = new Blob([midiArray], { type: 'audio/midi' });

  console.log('✅ MIDI file created');

  if (onProgress) onProgress(1.0);

  // Calculate metadata
  const pitches = notes.map(n => n.pitch);
  const metadata = {
    duration: audioBuffer.duration,
    total_notes: notes.length,
    tempo: 120,
    min_pitch: Math.min(...pitches),
    max_pitch: Math.max(...pitches)
  };

  return {
    midiBlob,
    metadata,
    notes // Return notes for visualization if needed
  };
}

/**
 * Quick test if audio has detectable pitch
 * @param {File} audioFile - Audio file
 * @returns {Promise<boolean>} - True if pitch detected
 */
export async function hasDetectablePitch(audioFile) {
  try {
    const arrayBuffer = await audioFile.arrayBuffer();
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Only analyze first second
    const sampleSize = Math.min(audioBuffer.length, audioBuffer.sampleRate);
    const buffer = audioBuffer.getChannelData(0).slice(0, sampleSize);
    const frequency = detectPitch(buffer, audioBuffer.sampleRate);

    return frequency !== null;
  } catch (error) {
    console.error('Error testing pitch detection:', error);
    return false;
  }
}
