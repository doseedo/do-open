/**
 * Chord Rendering API Service
 * Handles chord progression rendering to MIDI
 */

/**
 * Render chord progression as MIDI file
 * @param {Object} chordMap - Map of beat index to chord name (e.g., { 0: 'Cmaj7', 4: 'Am7' })
 * @param {number} bpm - BPM for the rendered MIDI
 * @param {number} duration - Total duration in beats
 * @param {Object} settings - Voicing, rhythm, and style settings { voicing, rhythm, style }
 * @returns {Promise<Object>} - Response with file path and metadata
 */
export async function renderChordProgression(chordMap, bpm = 120, duration = 16, settings = {}) {
  console.log('🎹 Rendering chord progression:', {
    chordMap,
    bpm,
    duration,
    settings
  });

  const response = await fetch('/api/render-chords', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      chords: chordMap,
      bpm: bpm,
      duration: duration,
      voicing: settings.voicing || 'random',
      rhythm: settings.rhythm || 'random',
      style: settings.style || 'random'
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP error! status: ${response.status}`);
  }

  const result = await response.json();
  console.log('✅ Chord progression rendered:', result);
  return result;
}

/**
 * Download rendered chord MIDI file
 * @param {string} filePath - File path returned from renderChordProgression (e.g., /download-chord-midi/abc123/chord_progression.mid)
 * @returns {Promise<Blob>} - MIDI file as blob
 */
export async function downloadChordMidi(filePath) {
  console.log('📥 Downloading chord MIDI from:', filePath);

  const response = await fetch(filePath);

  if (!response.ok) {
    throw new Error(`Failed to download chord MIDI: ${response.status}`);
  }

  const blob = await response.blob();
  console.log('✅ Chord MIDI downloaded, size:', blob.size, 'bytes');
  return blob;
}
