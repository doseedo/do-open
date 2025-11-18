#!/usr/bin/env python3
"""
Chord Progression Generator for Dø Music Generation System

Takes a chord beat map (beat index -> chord name) and generates MIDI progressions
with multiple voicing presets, rhythm patterns, and arrangement styles.

Features:
- Extended chord voicings (9th, 11th, 13th chords)
- Scale context awareness (diatonic vs modal interchange)
- Automatic tension resolution (b9 on V7 in minor, natural 9 in major)

This module is designed to be called from genfrominterface.py to generate
conditioning MIDI from user-specified chord progressions.

Author: Adapted from free-midi-chords project
"""

import mido
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
import random
import re

# ============================================================================
# SCALE CONTEXT SYSTEM
# ============================================================================

class ScaleContext:
    """
    Defines the scale/key context for chord progressions.
    Used to determine diatonic extensions vs alterations.
    """

    # Scale degrees to semitones (chromatic)
    CHROMATIC = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    # Major scale intervals from root
    MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]  # 1, 2, 3, 4, 5, 6, 7

    # Natural minor scale intervals from root
    MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]  # 1, 2, b3, 4, 5, b6, b7

    # Harmonic minor (raised 7th for dominant function)
    HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]  # 1, 2, b3, 4, 5, b6, 7

    def __init__(self, root_note: int, scale_type: str = 'major'):
        """
        Args:
            root_note: MIDI note number for tonic (e.g., 60 for C)
            scale_type: 'major', 'minor', or 'harmonic_minor'
        """
        self.root_note = root_note
        self.scale_type = scale_type

        if scale_type == 'major':
            self.scale_intervals = self.MAJOR_SCALE
        elif scale_type == 'harmonic_minor':
            self.scale_intervals = self.HARMONIC_MINOR
        else:  # minor or natural_minor
            self.scale_intervals = self.MINOR_SCALE

    def is_diatonic(self, note: int) -> bool:
        """Check if a note is diatonic to the scale"""
        interval = (note - self.root_note) % 12
        return interval in self.scale_intervals

    def get_scale_degree(self, note: int) -> Optional[int]:
        """Get scale degree (1-7) of a note, or None if not diatonic"""
        interval = (note - self.root_note) % 12
        if interval in self.scale_intervals:
            return self.scale_intervals.index(interval) + 1
        return None

    def get_diatonic_9th(self, chord_root: int) -> int:
        """
        Get the diatonic 9th for a chord root in this scale context.
        Returns the MIDI note number for the 9th.
        """
        # 9th is 2 scale degrees up (an octave + a 2nd)
        base_9th = chord_root + 14  # Octave (12) + major 2nd (2)

        # Check if natural 9 (major 2nd up) is in scale
        second_degree = (chord_root + 2) % 12
        root_scale_position = (chord_root - self.root_note) % 12

        # In minor scales, V chord (dominant) typically uses b9 for tension
        if self.scale_type in ['minor', 'harmonic_minor']:
            # Detect if this is the V chord (dominant)
            if root_scale_position == 7:  # V chord is 7 semitones up
                return chord_root + 13  # b9 (minor 9th)

        # Default: use natural 9 (major 9th)
        return base_9th

    def __repr__(self):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        root_name = note_names[self.root_note % 12]
        return f"ScaleContext({root_name} {self.scale_type})"


def detect_scale_context(chord_beat_map: Dict[int, str]) -> ScaleContext:
    """
    Analyze chord progression and detect the most likely scale context.

    Simple heuristic:
    - Look at first and last chords to determine tonic
    - Check for minor vs major quality
    - Default to C major if unclear
    """
    if not chord_beat_map:
        return ScaleContext(60, 'major')  # Default C major

    # Get first chord to determine root
    sorted_chords = sorted(chord_beat_map.items())
    first_chord = sorted_chords[0][1]

    # Parse root note from chord name
    root_note = _parse_chord_root(first_chord)

    # Determine major vs minor
    is_minor = 'm' in first_chord.lower() and 'maj' not in first_chord.lower()
    scale_type = 'minor' if is_minor else 'major'

    return ScaleContext(root_note, scale_type)


def _parse_chord_root(chord_name: str) -> int:
    """Extract root note MIDI number from chord name"""
    NOTE_MAP = {
        'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
        'Db': 61, 'D#': 63, 'Eb': 63, 'E#': 65,
        'Gb': 66, 'F#': 66, 'G#': 68, 'Ab': 68, 'A#': 70, 'Bb': 70,
        'C#': 61
    }

    # Match root note (with possible accidental)
    match = re.match(r'^([A-G][b#]?)', chord_name)
    if match:
        root = match.group(1)
        return NOTE_MAP.get(root, 60)

    return 60  # Default to C


# ============================================================================
# CHORD DEFINITIONS (C Major/Minor Reference)
# ============================================================================

# Extended chord library supporting triads, 7ths, 9ths, and extended chords
CHORD_LIBRARY = {
    # Major triads
    'C': [60, 64, 67], 'Cmaj': [60, 64, 67],
    'D': [62, 66, 69], 'Dmaj': [62, 66, 69],
    'E': [64, 68, 71], 'Emaj': [64, 68, 71],
    'F': [65, 69, 72], 'Fmaj': [65, 69, 72],
    'G': [67, 71, 74], 'Gmaj': [67, 71, 74],
    'A': [69, 73, 76], 'Amaj': [69, 73, 76],
    'B': [71, 75, 78], 'Bmaj': [71, 75, 78],

    # Flat major triads
    'Db': [61, 65, 68], 'Eb': [63, 67, 70], 'Gb': [66, 70, 73],
    'Ab': [68, 72, 75], 'Bb': [70, 74, 77],

    # Sharp major triads
    'C#': [61, 65, 68], 'D#': [63, 67, 70], 'F#': [66, 70, 73],
    'G#': [68, 72, 75], 'A#': [70, 74, 77],

    # Minor triads
    'Cm': [60, 63, 67], 'Cmin': [60, 63, 67],
    'Dm': [62, 65, 69], 'Dmin': [62, 65, 69],
    'Em': [64, 67, 71], 'Emin': [64, 67, 71],
    'Fm': [65, 68, 72], 'Fmin': [65, 68, 72],
    'Gm': [67, 70, 74], 'Gmin': [67, 70, 74],
    'Am': [69, 72, 76], 'Amin': [69, 72, 76],
    'Bm': [71, 74, 78], 'Bmin': [71, 74, 78],

    # Flat minor triads
    'Dbm': [61, 64, 68], 'Ebm': [63, 66, 70], 'Gbm': [66, 69, 73],
    'Abm': [68, 71, 75], 'Bbm': [70, 73, 77],

    # Sharp minor triads
    'C#m': [61, 64, 68], 'D#m': [63, 66, 70], 'F#m': [66, 69, 73],
    'G#m': [68, 71, 75], 'A#m': [70, 73, 77],

    # Major 7th chords
    'Cmaj7': [60, 64, 67, 71], 'Dmaj7': [62, 66, 69, 73],
    'Emaj7': [64, 68, 71, 75], 'Fmaj7': [65, 69, 72, 76],
    'Gmaj7': [67, 71, 74, 78], 'Amaj7': [69, 73, 76, 80],
    'Bmaj7': [71, 75, 78, 82],

    # Minor 7th chords
    'Cm7': [60, 63, 67, 70], 'Dm7': [62, 65, 69, 72],
    'Em7': [64, 67, 71, 74], 'Fm7': [65, 68, 72, 75],
    'Gm7': [67, 70, 74, 77], 'Am7': [69, 72, 76, 79],
    'Bm7': [71, 74, 78, 81],

    # Dominant 7th chords
    'C7': [60, 64, 67, 70], 'D7': [62, 66, 69, 72],
    'E7': [64, 68, 71, 74], 'F7': [65, 69, 72, 75],
    'G7': [67, 71, 74, 77], 'A7': [69, 73, 76, 79],
    'B7': [71, 75, 78, 81],

    # Major 9th chords (Root-3-5-7-9)
    'Cmaj9': [60, 64, 67, 71, 74], 'Dmaj9': [62, 66, 69, 73, 76],
    'Emaj9': [64, 68, 71, 75, 78], 'Fmaj9': [65, 69, 72, 76, 79],
    'Gmaj9': [67, 71, 74, 78, 81], 'Amaj9': [69, 73, 76, 80, 83],

    # Minor 9th chords
    'Cm9': [60, 63, 67, 70, 74], 'Dm9': [62, 65, 69, 72, 76],
    'Em9': [64, 67, 71, 74, 78], 'Fm9': [65, 68, 72, 75, 79],
    'Gm9': [67, 70, 74, 77, 81], 'Am9': [69, 72, 76, 79, 83],

    # Dominant 9th chords (natural 9)
    'C9': [60, 64, 67, 70, 74], 'D9': [62, 66, 69, 72, 76],
    'E9': [64, 68, 71, 74, 78], 'F9': [65, 69, 72, 75, 79],
    'G9': [67, 71, 74, 77, 81], 'A9': [69, 73, 76, 79, 83],
    'B9': [71, 75, 78, 81, 85],

    # Dominant 7b9 chords (altered - flat 9)
    'C7b9': [60, 64, 67, 70, 73], 'D7b9': [62, 66, 69, 72, 75],
    'E7b9': [64, 68, 71, 74, 77], 'F7b9': [65, 69, 72, 75, 78],
    'G7b9': [67, 71, 74, 77, 80], 'A7b9': [69, 73, 76, 79, 82],
    'B7b9': [71, 75, 78, 81, 84],

    # Dominant 7#9 chords (altered - sharp 9, aka "Hendrix chord")
    'C7#9': [60, 64, 67, 70, 75], 'D7#9': [62, 66, 69, 72, 77],
    'E7#9': [64, 68, 71, 74, 79], 'F7#9': [65, 69, 72, 75, 80],
    'G7#9': [67, 71, 74, 77, 82], 'A7#9': [69, 73, 76, 79, 84],

    # Diminished and augmented
    'Cdim': [60, 63, 66], 'Ddim': [62, 65, 68], 'Edim': [64, 67, 70],
    'Fdim': [65, 68, 71], 'Gdim': [67, 70, 73], 'Adim': [69, 72, 75],
    'Bdim': [71, 74, 77],

    'Caug': [60, 64, 68], 'Daug': [62, 66, 70], 'Eaug': [64, 68, 72],
    'Faug': [65, 69, 73], 'Gaug': [67, 71, 75], 'Aaug': [69, 73, 77],

    # Suspended chords
    'Csus2': [60, 62, 67], 'Csus4': [60, 65, 67],
    'Dsus2': [62, 64, 69], 'Dsus4': [62, 67, 69],
    'Gsus2': [67, 69, 74], 'Gsus4': [67, 72, 74],
}


# ============================================================================
# INVERSION SYSTEM
# ============================================================================

class ChordInversion:
    """
    Automatic chord inversion calculator.
    Works with any chord, including extended voicings (9ths, 11ths, 13ths).
    """

    @staticmethod
    def get_inversion(notes: List[int], inversion: int) -> List[int]:
        """
        Calculate chord inversion by rotating notes.

        Args:
            notes: List of MIDI note numbers in root position
            inversion: Inversion number (0=root, 1=first, 2=second, etc.)

        Returns:
            List of MIDI notes in the requested inversion

        Examples:
            Cmaj7 (root): C(60), E(64), G(67), B(71)
            Cmaj7 (1st):  E(64), G(67), B(71), C(72)  <- C moves up octave
            Cmaj7 (2nd):  G(67), B(71), C(72), E(76)  <- E also up
            Cmaj7 (3rd):  B(71), C(72), E(76), G(79)  <- G also up
        """
        if not notes:
            return notes

        if inversion == 0:
            return notes[:]  # Root position

        # Limit inversion to number of notes in chord
        num_notes = len(notes)
        inversion = inversion % num_notes  # Wrap around if too large

        # Rotate notes and move lower ones up an octave
        result = notes[:]
        for i in range(inversion):
            # Take the lowest note and move it up an octave
            lowest = result.pop(0)
            result.append(lowest + 12)

        return result

    @staticmethod
    def get_all_inversions(notes: List[int]) -> Dict[str, List[int]]:
        """
        Generate all possible inversions for a chord.

        Returns:
            Dict mapping inversion name to note list
        """
        num_notes = len(notes)
        inversions = {}

        inversion_names = {
            0: 'root',
            1: '1st',
            2: '2nd',
            3: '3rd',
            4: '4th',
            5: '5th',
            6: '6th',
            7: '7th'
        }

        for inv_num in range(num_notes):
            inv_name = inversion_names.get(inv_num, f'{inv_num}th')
            inversions[inv_name] = ChordInversion.get_inversion(notes, inv_num)

        return inversions


# ============================================================================
# VOICING PRESETS
# ============================================================================

class VoicingPreset:
    """Defines how to voice a chord across different octaves and registers"""

    @staticmethod
    def close_position(notes: List[int]) -> List[int]:
        """All notes in close position (1-2 octaves)"""
        return notes[:]

    @staticmethod
    def open_position(notes: List[int]) -> List[int]:
        """Spread notes across wider range"""
        if len(notes) < 3:
            return notes
        result = [notes[0]]  # Root
        if len(notes) > 1:
            result.append(notes[1] + 12)  # 3rd up an octave
        if len(notes) > 2:
            result.append(notes[2])  # 5th in original octave
        if len(notes) > 3:
            result.extend(notes[3:])  # Extensions
        return sorted(result)

    @staticmethod
    def drop_2(notes: List[int]) -> List[int]:
        """Drop second note from top down an octave"""
        if len(notes) < 3:
            return notes
        result = notes[:]
        if len(result) >= 3:
            result[-2] -= 12
        return sorted(result)

    @staticmethod
    def drop_3(notes: List[int]) -> List[int]:
        """Drop third note from top down an octave"""
        if len(notes) < 4:
            return notes
        result = notes[:]
        if len(result) >= 4:
            result[-3] -= 12
        return sorted(result)

    @staticmethod
    def root_position_bass(notes: List[int]) -> List[int]:
        """Add bass root below chord"""
        if not notes:
            return notes
        bass_note = notes[0] - 12
        return [bass_note] + notes

    @staticmethod
    def shell_voicing(notes: List[int]) -> List[int]:
        """Just root, 3rd, and 7th (for jazz comping)"""
        if len(notes) < 4:
            return notes[:3] if len(notes) >= 3 else notes
        return [notes[0], notes[1], notes[3]]  # R, 3, 7

    @staticmethod
    def spread_voicing(notes: List[int]) -> List[int]:
        """Wide spread across 2+ octaves"""
        if len(notes) < 3:
            return notes
        result = [notes[0]]  # Root low
        if len(notes) > 2:
            result.append(notes[2] + 12)  # 5th up octave
        if len(notes) > 1:
            result.append(notes[1] + 24)  # 3rd up 2 octaves
        if len(notes) > 3:
            result.extend([n + 12 for n in notes[3:]])  # Extensions up
        return sorted(result)


VOICING_PRESETS = {
    'close': VoicingPreset.close_position,
    'open': VoicingPreset.open_position,
    'drop2': VoicingPreset.drop_2,
    'drop3': VoicingPreset.drop_3,
    'bass': VoicingPreset.root_position_bass,
    'shell': VoicingPreset.shell_voicing,
    'spread': VoicingPreset.spread_voicing,
}


# ============================================================================
# RHYTHM PRESETS
# ============================================================================

class RhythmPattern:
    """Defines when and how long notes play within a beat/bar"""

    @staticmethod
    def whole_notes(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Single sustained note/chord for entire duration"""
        return [(0.0, float(beats_per_chord))]

    @staticmethod
    def half_notes(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Two half notes"""
        half = beats_per_chord / 2
        return [(0.0, half), (half, half)]

    @staticmethod
    def quarter_notes(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Quarter note rhythm"""
        return [(float(i), 1.0) for i in range(beats_per_chord)]

    @staticmethod
    def eighth_notes(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Eighth note rhythm"""
        return [(i * 0.5, 0.5) for i in range(beats_per_chord * 2)]

    @staticmethod
    def sixteenth_notes(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Sixteenth note rhythm"""
        return [(i * 0.25, 0.25) for i in range(beats_per_chord * 4)]

    @staticmethod
    def syncopated(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Syncopated rhythm (off-beat emphasis)"""
        return [(0.0, 0.75), (1.0, 0.5), (1.75, 0.75), (2.75, 1.25)]

    @staticmethod
    def arpeggio_up(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Arpeggiated pattern (individual notes)"""
        # This needs special handling - returns note indices instead
        return [(i * 0.5, 0.5) for i in range(beats_per_chord * 2)]

    @staticmethod
    def dotted_rhythm(beats_per_chord: int = 4) -> List[Tuple[float, float]]:
        """Dotted quarter + eighth pattern"""
        pattern = []
        pos = 0.0
        while pos < beats_per_chord:
            pattern.append((pos, 1.5))  # Dotted quarter
            pos += 1.5
            if pos < beats_per_chord:
                pattern.append((pos, 0.5))  # Eighth
                pos += 0.5
        return pattern


RHYTHM_PRESETS = {
    'whole': RhythmPattern.whole_notes,
    'half': RhythmPattern.half_notes,
    'quarter': RhythmPattern.quarter_notes,
    'eighth': RhythmPattern.eighth_notes,
    'sixteenth': RhythmPattern.sixteenth_notes,
    'syncopated': RhythmPattern.syncopated,
    'arpeggio': RhythmPattern.arpeggio_up,
    'dotted': RhythmPattern.dotted_rhythm,
}


# ============================================================================
# CHORD PROGRESSION GENERATOR
# ============================================================================

class ChordProgressionGenerator:
    """
    Main generator class that converts chord beat maps to MIDI files
    with scale context awareness for diatonic extensions.
    """

    def __init__(self,
                 bpm: int = 120,
                 ticks_per_beat: int = 480,
                 beats_per_bar: int = 4,
                 scale_context: Optional[ScaleContext] = None):
        self.bpm = bpm
        self.ticks_per_beat = ticks_per_beat
        self.beats_per_bar = beats_per_bar
        self.tempo_microseconds = int(60_000_000 / bpm)
        self.scale_context = scale_context

    def parse_chord_name(self, chord_name: str, use_context: bool = True) -> List[int]:
        """
        Parse chord name and return MIDI notes with optional scale context.
        Handles chord names from user input (e.g., 'Cmaj7', 'Am', 'G7')

        Args:
            chord_name: Chord symbol (e.g., 'Cm9', 'G7', 'Fmaj9')
            use_context: If True and scale_context is set, apply diatonic extensions

        Returns:
            List of MIDI note numbers
        """
        chord_name = chord_name.strip()

        # IMPORTANT: Check for contextual 9th chords BEFORE direct lookup
        # This allows us to override library definitions with context-aware ones
        if use_context and self.scale_context:
            # Check for basic 9th chord patterns (not already altered)
            if chord_name.endswith('9') and not any(alt in chord_name for alt in ['b9', '#9', '7b9', '7#9']):
                # Try to build the chord with context-aware 9th
                base_chord = self._build_contextual_9th_chord(chord_name)
                if base_chord:
                    return base_chord

        # Direct lookup (for explicit alterations or when no context)
        if chord_name in CHORD_LIBRARY:
            return CHORD_LIBRARY[chord_name].copy()

        # Try case variations
        for key in CHORD_LIBRARY.keys():
            if key.lower() == chord_name.lower():
                return CHORD_LIBRARY[key].copy()

        # Fallback: return C major and warn
        print(f"⚠️  Unknown chord '{chord_name}', using C major")
        return [60, 64, 67]

    def _build_contextual_9th_chord(self, chord_name: str) -> Optional[List[int]]:
        """
        Build a 9th chord with context-aware extension.
        For example, in C minor: G9 becomes G7b9 automatically.
        """
        if not self.scale_context:
            return None

        # Parse the base chord (without the 9)
        base_name = chord_name[:-1]  # Remove '9'

        # Determine chord quality
        is_minor = 'm' in base_name.lower() and 'maj' not in base_name.lower()
        is_dominant = not is_minor and not 'maj' in base_name.lower()

        # Build base 7th chord first
        if is_minor:
            base_7th_name = base_name + ('7' if not base_name.endswith('7') else '')
        elif is_dominant:
            base_7th_name = base_name + ('7' if not base_name.endswith('7') else '')
        else:
            base_7th_name = base_name + 'maj7'

        # Get base chord notes
        if base_7th_name not in CHORD_LIBRARY:
            return None

        base_notes = CHORD_LIBRARY[base_7th_name].copy()
        chord_root = base_notes[0]

        # Get contextual 9th
        ninth_note = self.scale_context.get_diatonic_9th(chord_root)

        # Add 9th to chord
        base_notes.append(ninth_note)

        print(f"  🎼 Context: {chord_name} -> {base_notes} (9th: {ninth_note}, scale: {self.scale_context})")

        return base_notes

    def apply_inversion(self, notes: List[int], inversion: int = 0) -> List[int]:
        """Apply chord inversion before voicing"""
        return ChordInversion.get_inversion(notes, inversion)

    def apply_voicing(self, notes: List[int], voicing: str = 'close') -> List[int]:
        """Apply voicing preset to chord notes"""
        if voicing not in VOICING_PRESETS:
            print(f"⚠️  Unknown voicing '{voicing}', using close position")
            voicing = 'close'

        voicing_func = VOICING_PRESETS[voicing]
        return voicing_func(notes)

    def generate_from_chord_map(self,
                                chord_beat_map: Dict[int, str],
                                voicing: str = 'close',
                                rhythm: str = 'whole',
                                output_path: Optional[str] = None,
                                velocity: int = 80,
                                arrangement_style: str = 'block',
                                inversion: int = 0) -> str:
        """
        Generate MIDI file from chord beat map

        Args:
            chord_beat_map: Dict mapping beat index to chord name
                           e.g., {0: 'Cmaj7', 4: 'Am7', 8: 'Fmaj7', 12: 'G7'}
            voicing: Voicing preset name ('close', 'open', 'drop2', etc.)
            rhythm: Rhythm preset name ('whole', 'quarter', 'eighth', etc.)
            output_path: Optional output path for MIDI file
            velocity: MIDI velocity (0-127)
            arrangement_style: 'block' (chords) or 'arpeggio' (individual notes)
            inversion: Chord inversion (0=root, 1=1st, 2=2nd, 3=3rd, etc.)

        Returns:
            Path to generated MIDI file
        """

        print(f"\n{'='*60}")
        print(f"🎹 GENERATING CHORD PROGRESSION")
        print(f"{'='*60}")
        print(f"Chord map: {chord_beat_map}")
        print(f"Voicing: {voicing}")
        print(f"Rhythm: {rhythm}")
        print(f"Style: {arrangement_style}")
        print(f"Inversion: {inversion}")
        print(f"BPM: {self.bpm}")

        # Sort chord map by beat position
        sorted_chords = sorted(chord_beat_map.items())

        # Determine total duration
        if sorted_chords:
            last_beat = max(chord_beat_map.keys())
            total_beats = last_beat + self.beats_per_bar  # Add one more bar
        else:
            total_beats = self.beats_per_bar * 4  # Default 4 bars

        print(f"Total duration: {total_beats} beats ({total_beats / self.beats_per_bar} bars)")

        # Create MIDI file
        mid = mido.MidiFile(ticks_per_beat=self.ticks_per_beat)
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=self.tempo_microseconds, time=0))
        track.append(mido.MetaMessage('time_signature',
                                     numerator=self.beats_per_bar,
                                     denominator=4,
                                     clocks_per_click=24,
                                     notated_32nd_notes_per_beat=8,
                                     time=0))

        # Get rhythm pattern
        if rhythm not in RHYTHM_PRESETS:
            print(f"⚠️  Unknown rhythm '{rhythm}', using whole notes")
            rhythm = 'whole'

        rhythm_func = RHYTHM_PRESETS[rhythm]

        # Track for delta time calculation
        current_tick = 0

        # Process each chord
        for i, (beat_pos, chord_name) in enumerate(sorted_chords):
            # Get base chord notes
            base_notes = self.parse_chord_name(chord_name)

            # Apply inversion FIRST (before voicing)
            inverted_notes = self.apply_inversion(base_notes, inversion)

            # Then apply voicing
            voiced_notes = self.apply_voicing(inverted_notes, voicing)

            inv_label = ['root', '1st', '2nd', '3rd', '4th', '5th', '6th', '7th'][inversion] if inversion < 8 else f'{inversion}th'
            print(f"  Beat {beat_pos}: {chord_name} ({inv_label} inv) -> notes {voiced_notes}")

            # Calculate chord duration (to next chord or end)
            if i < len(sorted_chords) - 1:
                next_beat = sorted_chords[i + 1][0]
                chord_duration_beats = next_beat - beat_pos
            else:
                chord_duration_beats = total_beats - beat_pos

            # Get rhythm pattern for this chord
            rhythm_pattern = rhythm_func(int(chord_duration_beats))

            # Generate notes based on arrangement style
            if arrangement_style == 'arpeggio' or rhythm == 'arpeggio':
                # Arpeggiated: play notes individually
                current_tick = self._add_arpeggio_notes(track, voiced_notes, beat_pos,
                                        rhythm_pattern, velocity, current_tick)
            else:
                # Block chords: play all notes together
                current_tick = self._add_block_chord_notes(track, voiced_notes, beat_pos,
                                           rhythm_pattern, velocity, current_tick)

        # Generate output path
        if output_path is None:
            temp_dir = Path(tempfile.gettempdir())
            output_path = str(temp_dir / f"chord_progression_{random.randint(1000, 9999)}.mid")

        # Save MIDI file
        mid.save(output_path)
        print(f"✅ MIDI file saved: {output_path}")
        print(f"{'='*60}\n")

        return output_path

    def _add_block_chord_notes(self, track, notes, start_beat, rhythm_pattern, velocity, current_tick):
        """Add block chord notes (all notes play together)"""
        for onset_beat, duration_beat in rhythm_pattern:
            abs_start_beat = start_beat + onset_beat
            abs_start_tick = int(abs_start_beat * self.ticks_per_beat)
            duration_ticks = int(duration_beat * self.ticks_per_beat)

            # Note on messages (all simultaneous)
            for i, note in enumerate(notes):
                delta = abs_start_tick - current_tick if i == 0 else 0
                track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
                if i == 0:
                    current_tick = abs_start_tick

            # Note off messages
            abs_end_tick = abs_start_tick + duration_ticks
            for i, note in enumerate(notes):
                delta = abs_end_tick - current_tick if i == 0 else 0
                track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
                if i == 0:
                    current_tick = abs_end_tick

        return current_tick

    def _add_arpeggio_notes(self, track, notes, start_beat, rhythm_pattern, velocity, current_tick):
        """Add arpeggiated notes (notes play one at a time)"""
        for pattern_idx, (onset_beat, duration_beat) in enumerate(rhythm_pattern):
            abs_start_beat = start_beat + onset_beat
            abs_start_tick = int(abs_start_beat * self.ticks_per_beat)
            duration_ticks = int(duration_beat * self.ticks_per_beat)

            # Cycle through chord notes
            note = notes[pattern_idx % len(notes)]

            # Note on
            delta = abs_start_tick - current_tick
            track.append(mido.Message('note_on', note=note, velocity=velocity, time=delta))
            current_tick = abs_start_tick

            # Note off
            abs_end_tick = abs_start_tick + duration_ticks
            delta = abs_end_tick - current_tick
            track.append(mido.Message('note_off', note=note, velocity=0, time=delta))
            current_tick = abs_end_tick

        return current_tick


# ============================================================================
# HIGH-LEVEL API FOR GENFROMINTERFACE.PY
# ============================================================================

def generate_chord_progression_midi(chord_beat_map: Dict[int, str],
                                   bpm: int = 120,
                                   voicing: str = 'close',
                                   rhythm: str = 'whole',
                                   style: str = 'block',
                                   output_path: Optional[str] = None,
                                   scale_context: Optional[ScaleContext] = None,
                                   auto_detect_scale: bool = True,
                                   inversion: int = 0) -> str:
    """
    High-level function to generate MIDI from chord beat map with scale context awareness.

    This is the main entry point called from genfrominterface.py

    Args:
        chord_beat_map: Dict mapping beat positions to chord names
                       e.g., {0: 'Cmaj7', 4: 'Am7', 8: 'Fmaj7', 12: 'G7'}
        bpm: Tempo in beats per minute
        voicing: Voicing preset - 'close', 'open', 'drop2', 'drop3', 'bass', 'shell', 'spread'
        rhythm: Rhythm preset - 'whole', 'half', 'quarter', 'eighth', 'sixteenth',
                               'syncopated', 'arpeggio', 'dotted'
        style: Arrangement style - 'block' or 'arpeggio'
        output_path: Optional output path for MIDI file
        scale_context: Optional ScaleContext object. If None and auto_detect_scale=True,
                      will auto-detect from progression
        auto_detect_scale: If True, automatically detect scale context from first chord

    Returns:
        Path to generated MIDI file

    Example:
        >>> # C minor progression with auto-detected context
        >>> chord_map = {0: 'Cm9', 4: 'Fm9', 8: 'G9', 12: 'Cm9'}
        >>> midi_path = generate_chord_progression_midi(
        ...     chord_map,
        ...     bpm=120,
        ...     voicing='drop2',
        ...     rhythm='whole',
        ...     style='block',
        ...     auto_detect_scale=True  # Will detect C minor, G9 becomes G7b9
        ... )

        >>> # Explicit scale context
        >>> from chord_progression_generator import ScaleContext
        >>> context = ScaleContext(60, 'minor')  # C minor
        >>> midi_path = generate_chord_progression_midi(
        ...     chord_map,
        ...     scale_context=context
        ... )
    """

    # Auto-detect scale context if requested and not provided
    if scale_context is None and auto_detect_scale:
        scale_context = detect_scale_context(chord_beat_map)
        print(f"🎵 Auto-detected scale context: {scale_context}")

    generator = ChordProgressionGenerator(bpm=bpm, scale_context=scale_context)
    return generator.generate_from_chord_map(
        chord_beat_map=chord_beat_map,
        voicing=voicing,
        rhythm=rhythm,
        output_path=output_path,
        arrangement_style=style,
        inversion=inversion
    )


def get_available_presets() -> Dict[str, List[str]]:
    """
    Return all available preset options

    Returns:
        Dict with keys 'voicings', 'rhythms', 'styles'
    """
    return {
        'voicings': list(VOICING_PRESETS.keys()),
        'rhythms': list(RHYTHM_PRESETS.keys()),
        'styles': ['block', 'arpeggio'],
        'example_chords': [
            'C', 'Cmaj7', 'Cm7', 'C7', 'Cmaj9',
            'Dm', 'Dm7', 'D7', 'Dmaj7',
            'Em', 'Em7', 'E7', 'Emaj7',
            'F', 'Fmaj7', 'Fm7', 'F7',
            'G', 'Gmaj7', 'Gm7', 'G7', 'G9',
            'Am', 'Am7', 'A7', 'Amaj7',
            'Bm', 'Bm7', 'B7', 'Bdim'
        ]
    }


# ============================================================================
# COMMAND LINE INTERFACE (for testing)
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate MIDI from chord progressions')
    parser.add_argument('--chords', type=str,
                       help='Comma-separated chord:beat pairs (e.g., "Cmaj7:0,Am7:4,Fmaj7:8,G7:12")')
    parser.add_argument('--bpm', type=int, default=120, help='Tempo (default: 120)')
    parser.add_argument('--voicing', type=str, default='close',
                       choices=list(VOICING_PRESETS.keys()),
                       help='Voicing preset')
    parser.add_argument('--rhythm', type=str, default='whole',
                       choices=list(RHYTHM_PRESETS.keys()),
                       help='Rhythm preset')
    parser.add_argument('--style', type=str, default='block',
                       choices=['block', 'arpeggio'],
                       help='Arrangement style')
    parser.add_argument('--output', type=str, help='Output MIDI file path')
    parser.add_argument('--list-presets', action='store_true',
                       help='List all available presets and exit')

    args = parser.parse_args()

    if args.list_presets:
        presets = get_available_presets()
        print("\n=== AVAILABLE PRESETS ===\n")
        print(f"Voicings: {', '.join(presets['voicings'])}")
        print(f"Rhythms: {', '.join(presets['rhythms'])}")
        print(f"Styles: {', '.join(presets['styles'])}")
        print(f"\nExample chords:")
        for i in range(0, len(presets['example_chords']), 6):
            print(f"  {', '.join(presets['example_chords'][i:i+6])}")
        exit(0)

    if not args.chords:
        # Demo progression
        print("No chords specified, using demo progression (I-vi-IV-V in C)")
        chord_map = {
            0: 'Cmaj7',
            4: 'Am7',
            8: 'Fmaj7',
            12: 'G7'
        }
    else:
        # Parse chord:beat pairs
        chord_map = {}
        for pair in args.chords.split(','):
            chord, beat = pair.split(':')
            chord_map[int(beat)] = chord.strip()

    # Generate MIDI
    output = generate_chord_progression_midi(
        chord_beat_map=chord_map,
        bpm=args.bpm,
        voicing=args.voicing,
        rhythm=args.rhythm,
        style=args.style,
        output_path=args.output
    )

    print(f"\n✅ Success! Generated MIDI file:")
    print(f"   {output}")
