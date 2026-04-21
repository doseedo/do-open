import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import useChordRegen from '../../hooks/useChordRegen';
import styles from './ChordWindow.module.css';

/**
 * ChordWindow Component
 * Modal window for selecting chords for a specific beat
 */
const ChordWindow = () => {
  const { state, dispatch } = useApp();
  const [selectedChord, setSelectedChord] = useState(null);
  const runChordRegen = useChordRegen();

  // Common chords organized by key
  const chordsByKey = {
    'C': ['C', 'Dm', 'Em', 'F', 'G', 'Am', 'Bdim', 'Cmaj7', 'Dm7', 'Em7', 'Fmaj7', 'G7', 'Am7'],
    'D': ['D', 'Em', 'F#m', 'G', 'A', 'Bm', 'C#dim', 'Dmaj7', 'Em7', 'F#m7', 'Gmaj7', 'A7', 'Bm7'],
    'E': ['E', 'F#m', 'G#m', 'A', 'B', 'C#m', 'D#dim', 'Emaj7', 'F#m7', 'G#m7', 'Amaj7', 'B7', 'C#m7'],
    'F': ['F', 'Gm', 'Am', 'Bb', 'C', 'Dm', 'Edim', 'Fmaj7', 'Gm7', 'Am7', 'Bbmaj7', 'C7', 'Dm7'],
    'G': ['G', 'Am', 'Bm', 'C', 'D', 'Em', 'F#dim', 'Gmaj7', 'Am7', 'Bm7', 'Cmaj7', 'D7', 'Em7'],
    'A': ['A', 'Bm', 'C#m', 'D', 'E', 'F#m', 'G#dim', 'Amaj7', 'Bm7', 'C#m7', 'Dmaj7', 'E7', 'F#m7'],
    'B': ['B', 'C#m', 'D#m', 'E', 'F#', 'G#m', 'A#dim', 'Bmaj7', 'C#m7', 'D#m7', 'Emaj7', 'F#7', 'G#m7']
  };

  // Additional common chords (not key-specific)
  const additionalChords = ['Csus4', 'Dsus4', 'Esus4', 'Fsus4', 'Gsus4', 'Asus4', 'Bsus4', 'C7', 'D7', 'E7', 'F7', 'G7', 'A7', 'B7'];

  // Get key from state or default to C
  const currentKey = state.generationParams?.aceKey || 'C';
  const availableChords = chordsByKey[currentKey] || chordsByKey['C'];

  const handleChordSelect = async (chord) => {
    if (state.chordWindow.beatIndex === null) return;
    const beatIndex = state.chordWindow.beatIndex;
    const oldChord = state.chordTrack?.chords?.[beatIndex] || '';

    dispatch({
      type: 'SET_CHORD_FOR_BEAT',
      payload: { beatIndex, chord },
    });
    handleClose();

    // Smart per-stem regen — identical logic used by /studio-dev.
    runChordRegen({ beatIndex, oldChord, newChord: chord }).catch((err) =>
      console.error('[chord-window] regen failed:', err?.message || err));
  };

  const handleClose = () => {
    dispatch({ type: 'CLOSE_CHORD_WINDOW' });
  };

  if (!state.chordWindow.isVisible) {
    return null;
  }

  return (
    <div className={styles.chordWindowOverlay}>
      <div className={styles.chordWindow}>
        <div className={styles.header}>
          <h3>Select Chord for Beat {(state.chordWindow.beatIndex || 0) + 1}</h3>
          <button className={styles.closeButton} onClick={handleClose}>
            <i className="fa-solid fa-times"></i>
          </button>
        </div>

        <div className={styles.content}>
          <div className={styles.keyInfo}>
            <span>Key: {currentKey}</span>
          </div>

          <div className={styles.section}>
            <h4>Diatonic Chords</h4>
            <div className={styles.chordGrid}>
              {availableChords.map((chord) => (
                <button
                  key={chord}
                  className={`${styles.chordButton} ${selectedChord === chord ? styles.selected : ''}`}
                  onClick={() => handleChordSelect(chord)}
                  onMouseEnter={() => setSelectedChord(chord)}
                >
                  {chord}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.section}>
            <h4>Additional Chords</h4>
            <div className={styles.chordGrid}>
              {additionalChords.map((chord) => (
                <button
                  key={chord}
                  className={`${styles.chordButton} ${selectedChord === chord ? styles.selected : ''}`}
                  onClick={() => handleChordSelect(chord)}
                  onMouseEnter={() => setSelectedChord(chord)}
                >
                  {chord}
                </button>
              ))}
            </div>
          </div>

          <div className={styles.actions}>
            <button className={styles.clearButton} onClick={() => handleChordSelect(null)}>
              Clear Chord
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ChordWindow;
