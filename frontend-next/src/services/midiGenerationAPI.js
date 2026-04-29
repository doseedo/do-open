/**
 * MIDI Generation API Service
 * Unified interface for all MIDI generation modes
 */

// Generation modes
export const GENERATION_MODES = {
  CHORDS: 'chords',           // Chord progression rendering (existing)
  MELODY_BASIC: 'melody_basic',    // ProperMelodyGenerator (target-note technique)
  MELODY_GENRE: 'melody_genre',    // Genre-specific generation
  MELODY_CONTEXT: 'melody_context', // Context-aware generation
};

// Available genres for genre mode
export const AVAILABLE_GENRES = [
  'jazz', 'blues', 'funk', 'pop', 'rock', 'electronic',
  'classical', 'hip-hop', 'r&b', 'country', 'reggae',
  'gospel', 'metal', 'singer-songwriter',
  // World music
  'arabic', 'indian', 'turkish', 'african', 'latin'
];

// Melody style presets
export const MELODY_STYLES = {
  bebop: { chromatic: 0.4, leapiness: 0.3, density: 'high' },
  ballad: { chromatic: 0.1, leapiness: 0.2, density: 'low' },
  funk: { chromatic: 0.2, leapiness: 0.4, density: 'medium' },
  classical: { chromatic: 0.05, leapiness: 0.5, density: 'medium' },
};

/**
 * Generate melody using basic algorithm (ProperMelodyGenerator)
 * Uses target-note technique with chord-scale theory
 *
 * @param {Object} params - Generation parameters
 * @returns {Promise<Object>} - Generated notes and metadata
 */
export async function generateMelodyBasic(params) {
  const {
    key = 'C minor',
    chords = 'Cm7,G7,Cm7,G7',
    bars = 4,
    minNote = 60,
    maxNote = 76,
    chromatic = 0.3,
    tempo = 120,
    seed = null,
  } = params;

  console.log('🎵 Generating melody (basic mode):', { key, chords, bars });

  const response = await fetch('/api/generate-melody', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode: 'basic',
      key,
      chords,
      bars,
      minNote,
      maxNote,
      chromatic,
      tempo,
      seed,
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Generate melody using genre-specific algorithms
 * Uses midi_generator's UnifiedMusicGenerator
 *
 * @param {Object} params - Generation parameters
 * @returns {Promise<Object>} - Generated notes and metadata
 */
export async function generateMelodyGenre(params) {
  const {
    key = 'C minor',
    chords = 'Cm7,G7,Cm7,G7',
    bars = 4,
    minNote = 60,
    maxNote = 76,
    tempo = 120,
    genre = 'jazz',
    style = 'bebop',
    seed = null,
  } = params;

  console.log('🎵 Generating melody (genre mode):', { key, chords, bars, genre, style });

  const response = await fetch('/api/generate-melody', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      mode: 'genre',
      key,
      chords,
      bars,
      minNote,
      maxNote,
      tempo,
      genre,
      style,
      seed,
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Generate melody with context-awareness
 * Analyzes existing MIDI/tracks and generates complementary melody
 *
 * @param {Object} params - Generation parameters
 * @param {File} existingMidi - Optional existing MIDI file for context
 * @returns {Promise<Object>} - Generated notes and metadata
 */
export async function generateMelodyContext(params, existingMidi = null) {
  const {
    key = 'C minor',
    chords = 'Cm7,G7,Cm7,G7',
    bars = 4,
    minNote = 60,
    maxNote = 76,
    tempo = 120,
    role = 'melody', // 'melody', 'bass', 'harmony', 'countermelody'
    matchDensity = true,
    matchStyle = true,
    seed = null,
  } = params;

  console.log('🎵 Generating melody (context mode):', { key, chords, bars, role });

  const formData = new FormData();
  formData.append('params', JSON.stringify({
    mode: 'context',
    key,
    chords,
    bars,
    minNote,
    maxNote,
    tempo,
    role,
    matchDensity,
    matchStyle,
    seed,
  }));

  if (existingMidi) {
    formData.append('existingMidi', existingMidi);
  }

  const response = await fetch('/api/generate-melody', {
    method: 'POST',
    body: formData
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Render chord progression as MIDI (existing functionality)
 *
 * @param {Object} params - Chord rendering parameters
 * @returns {Promise<Object>} - Rendered MIDI file info
 */
export async function renderChords(params) {
  const {
    chords = {},
    bpm = 120,
    duration = 16,
    voicing = 'random',
    rhythm = 'random',
    style = 'random',
  } = params;

  console.log('🎹 Rendering chords:', { chords, bpm, duration });

  const response = await fetch('/api/render-chords', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      chords,
      bpm,
      duration,
      voicing,
      rhythm,
      style,
    })
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Unified generation function - routes to appropriate generator
 *
 * @param {string} mode - Generation mode (GENERATION_MODES)
 * @param {Object} params - Generation parameters
 * @param {File} existingMidi - Optional context MIDI file
 * @returns {Promise<Object>} - Generated/rendered result
 */
export async function generateMIDI(mode, params, existingMidi = null) {
  switch (mode) {
    case GENERATION_MODES.CHORDS:
      return renderChords(params);
    case GENERATION_MODES.MELODY_BASIC:
      return generateMelodyBasic(params);
    case GENERATION_MODES.MELODY_GENRE:
      return generateMelodyGenre(params);
    case GENERATION_MODES.MELODY_CONTEXT:
      return generateMelodyContext(params, existingMidi);
    default:
      throw new Error(`Unknown generation mode: ${mode}`);
  }
}
