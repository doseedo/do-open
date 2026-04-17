import React, { useState } from 'react';
import { useApp } from '../../context/AppContext';
import { regenStemForChord } from '../../services/trackAnalysisAPI';
import styles from './ChordWindow.module.css';

/**
 * ChordWindow Component
 * Modal window for selecting chords for a specific beat
 */
const ChordWindow = () => {
  const { state, dispatch } = useApp();
  const [selectedChord, setSelectedChord] = useState(null);

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

    // ----- Phase B+C: smart per-stem regen -----
    if (!chord || chord === oldChord) return;

    // Convert beat index → time range for the affected bar
    const bpm = state.bpm || 120;
    const beatsPerBar = state.beatsPerBar || 4;
    const secondsPerBeat = 60 / bpm;
    const barIndex = Math.floor(beatIndex / beatsPerBar);
    const regionStart = barIndex * beatsPerBar * secondsPerBeat;
    const regionEnd = regionStart + beatsPerBar * secondsPerBeat;

    // Find tracks overlapping this bar with analyzed metadata
    const candidates = [];
    state.buses.forEach((bus) => {
      bus.tracks?.forEach((tr) => {
        if (!tr.audioUrl || !tr.metadata?.instrument) return;
        const sp = tr.startPosition || 0;
        const dur = tr.duration || 0;
        if (sp + dur < regionStart || sp > regionEnd) return;
        candidates.push({ bus, track: tr });
      });
    });
    if (candidates.length === 0) {
      console.log('🎼 No analyzed tracks overlap bar', barIndex);
      return;
    }

    // Arrangement decision: only ONE harmonic instrument owns extensions.
    // Pick the highest-register harmonic candidate to receive 9/11/13.
    const HARMONIC = new Set(['guitar', 'piano', 'keys', 'synth', 'harmony']);
    const harmonics = candidates.filter(({ track }) => HARMONIC.has(track.metadata.instrument));
    let extensionOwnerId = null;
    if (harmonics.length > 0) {
      // Heuristic: piano > keys > guitar > synth > harmony
      const PRIORITY = { piano: 5, keys: 4, guitar: 3, synth: 2, harmony: 1 };
      harmonics.sort((a, b) =>
        (PRIORITY[b.track.metadata.instrument] || 0) - (PRIORITY[a.track.metadata.instrument] || 0)
      );
      extensionOwnerId = harmonics[0].track.id;
    }

    console.log(`🎼 Chord ${oldChord || '∅'} → ${chord} @ bar ${barIndex+1}, ${candidates.length} candidate tracks, extension owner: ${extensionOwnerId}`);

    for (const { track } of candidates) {
      const role = track.metadata.instrument;
      // Strip extensions from non-owner harmonic instruments by sending them a triad version
      let chordForThisTrack = chord;
      if (HARMONIC.has(role) && track.id !== extensionOwnerId) {
        chordForThisTrack = chord.replace(/(9|11|13|maj9|m9|maj11|m11|maj13|m13|add9)$/i, '');
      }
      try {
        const audioBlob = await (await fetch(track.audioUrl)).blob();
        const audioFile = new File([audioBlob], (track.name || 'stem') + '.wav', { type: 'audio/wav' });
        let midiFile = null;
        if (track.metadata?.midi) {
          try {
            const mb = await (await fetch(track.metadata.midi)).blob();
            midiFile = new File([mb], 'input.mid', { type: 'audio/midi' });
          } catch {}
        }
        const result = await regenStemForChord({
          audioFile, midiFile, role,
          oldChord, newChord: chordForThisTrack,
          regionStart, regionEnd,
          coverNoise: 0.7,
          duration: track.duration,
        });
        if (result.skipped) {
          console.log(`  ⏩ ${role} skipped: ${result.reason}`);
          continue;
        }
        // Poll the task and replace the track audio when done
        const taskId = result.task_id;
        console.log(`  🎛️ ${role} regenerating, task ${taskId}`);
        const poll = async () => {
          for (let i = 0; i < 300; i++) {
            await new Promise((r) => setTimeout(r, 2000));
            const r2 = await fetch(`/api/generate-stemphonic/task/${taskId}`);
            if (!r2.ok) continue;
            const tr2 = await r2.json();
            if (tr2.state === 'SUCCESS' && tr2.result?.file_paths?.[0]) {
              const newUrl = tr2.result.file_paths[0];
              dispatch({
                type: 'UPDATE_TRACK',
                payload: {
                  busId: state.buses.find((b) => b.tracks.some((t) => t.id === track.id))?.id,
                  trackId: track.id,
                  updates: {
                    audioUrl: newUrl,
                    metadata: {
                      ...track.metadata,
                      lastChordRegen: { from: oldChord, to: chordForThisTrack, at: Date.now() },
                    },
                  },
                },
              });
              console.log(`  ✅ ${role} replaced with ${newUrl}`);
              return;
            }
            if (tr2.state === 'FAILURE') {
              console.error(`  ❌ ${role} regen failed:`, tr2.error);
              return;
            }
          }
        };
        poll();
      } catch (err) {
        console.error(`  ❌ ${role} regen error:`, err);
      }
    }
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
