#!/usr/bin/env python3
"""
Melody Harmonizer - Arranges melodies with intelligent voicings

Takes a melody MIDI and harmonizes it using:
- 4-way closed voicing (4 voices in close position)
- 5-way closed voicing (5 voices in close position)
- Block chords (melody doubled with block voicing below)

Uses chord and scale context to choose diatonic extensions intelligently.
"""

import mido
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import tempfile
import random
from chord_progression_generator import ScaleContext, _parse_chord_root

# ============================================================================
# MELODY ANALYSIS
# ============================================================================

class MelodyAnalyzer:
    """Analyzes a melody MIDI file and extracts note events"""

    @staticmethod
    def extract_melody(midi_path: str) -> List[Tuple[int, int, int, int]]:
        """
        Extract melody notes from MIDI file.

        Returns:
            List of (start_tick, duration_ticks, note, velocity) tuples
        """
        mid = mido.MidiFile(midi_path)

        # Find melody track (first track with notes)
        melody_events = []
        for track in mid.tracks:
            note_ons = {}  # Track note_on events waiting for note_off
            time_accum = 0

            for msg in track:
                time_accum += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    note_ons[msg.note] = (time_accum, msg.velocity)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in note_ons:
                        start_time, velocity = note_ons[msg.note]
                        duration = time_accum - start_time
                        melody_events.append((start_time, duration, msg.note, velocity))
                        del note_ons[msg.note]

            if melody_events:
                break  # Found melody track

        return sorted(melody_events)  # Sort by start time


# ============================================================================
# HARMONIC ANALYSIS
# ============================================================================

class HarmonicAnalyzer:
    """Analyzes melody notes and determines appropriate chords"""

    def __init__(self, scale_context: ScaleContext):
        self.scale_context = scale_context

    def analyze_melody_note(self, melody_note: int) -> Dict[str, any]:
        """
        Analyze a melody note and determine its harmonic function.

        Returns:
            Dict with scale_degree, chord_tones, available_extensions
        """
        # Get scale degree of melody note
        interval = (melody_note - self.scale_context.root_note) % 12

        # Determine if diatonic
        is_diatonic = interval in self.scale_context.scale_intervals

        # Get scale degree (1-7)
        scale_degree = None
        if is_diatonic:
            scale_degree = self.scale_context.scale_intervals.index(interval) + 1

        # Determine likely chord based on melody note
        # For simplicity, we'll use a chord where the melody note is a chord tone
        chord_info = self._get_chord_for_melody_note(melody_note, scale_degree)

        return {
            'melody_note': melody_note,
            'scale_degree': scale_degree,
            'is_diatonic': is_diatonic,
            'chord_root': chord_info['root'],
            'chord_quality': chord_info['quality'],
            'chord_tones': chord_info['tones'],
            'extensions': chord_info['extensions']
        }

    def _get_chord_for_melody_note(self, melody_note: int, scale_degree: Optional[int]) -> Dict:
        """
        Determine the best chord where this melody note is a chord tone.

        Uses tertian harmony (stacking thirds) to build chords.
        """
        if scale_degree is None:
            # Non-diatonic note - treat as passing tone, use closest chord
            scale_degree = 1

        # Build a chord from the scale degree
        # For each scale degree, we can build a chord using that note
        # Let's use the melody note as a chord tone (root, 3rd, 5th, 7th, or 9th)

        # Simple approach: build a triad from the scale degree below the melody
        # Then extend with diatonic tones

        # Get the root note for a chord where melody is 3rd or 5th
        # (melody as root would be too low for voicings below)

        # Strategy: Build I chord (tonic) if melody is scale degrees 1, 3, 5
        #           Build IV chord if melody is 1, 4, 6
        #           Build V chord if melody is 2, 5, 7

        chord_root = self.scale_context.root_note
        chord_quality = 'major' if self.scale_context.scale_type == 'major' else 'minor'

        # Build diatonic chord tones
        chord_tones = self._build_diatonic_chord(chord_root, chord_quality, melody_note)

        # Get available extensions
        extensions = self._get_diatonic_extensions(chord_root, melody_note)

        return {
            'root': chord_root,
            'quality': chord_quality,
            'tones': chord_tones,
            'extensions': extensions
        }

    def _build_diatonic_chord(self, root: int, quality: str, melody_note: int) -> List[int]:
        """Build a diatonic chord (triad + 7th) from root"""
        chord = []

        # Start from root, go up in thirds diatonically
        current = root
        for i in range(4):  # Root, 3rd, 5th, 7th
            chord.append(current)
            # Go up a diatonic third
            current = self._diatonic_third_above(current)

        return chord

    def _diatonic_third_above(self, note: int) -> int:
        """Get the diatonic third above a note in the current scale"""
        # Move up 2 scale degrees
        intervals = self.scale_context.scale_intervals
        note_interval = (note - self.scale_context.root_note) % 12

        if note_interval in intervals:
            idx = intervals.index(note_interval)
            # Move up 2 scale degrees (wrapping around)
            next_idx = (idx + 2) % len(intervals)
            next_interval = intervals[next_idx]

            # Calculate actual note
            octave_offset = 12 if next_idx < idx else 0
            return self.scale_context.root_note + next_interval + octave_offset
        else:
            # Not in scale, just go up 4 semitones (major third)
            return note + 4

    def _get_diatonic_extensions(self, chord_root: int, melody_note: int) -> List[int]:
        """Get available diatonic extensions (9th, 11th, 13th)"""
        extensions = []

        # 9th (2nd an octave up)
        ninth = chord_root + 14  # Major 9th
        if self.scale_context.is_diatonic(ninth):
            extensions.append(ninth)

        # 11th (4th an octave up)
        eleventh = chord_root + 17  # Perfect 11th
        if self.scale_context.is_diatonic(eleventh):
            extensions.append(eleventh)

        # 13th (6th an octave up)
        thirteenth = chord_root + 21  # Major 13th
        if self.scale_context.is_diatonic(thirteenth):
            extensions.append(thirteenth)

        return extensions


# ============================================================================
# VOICING STRATEGIES
# ============================================================================

class MelodyVoicing:
    """Generates voicings with melody as top note"""

    @staticmethod
    def four_way_closed(melody_note: int, chord_tones: List[int], extensions: List[int] = None) -> List[int]:
        """
        4-way closed voicing: 4 voices in close position with melody on top.

        Strategy: Take melody note, then fill in 3 notes below from chord tones
        in close position (within an octave).
        """
        if extensions is None:
            extensions = []

        # Available notes: chord tones + extensions
        available = sorted(set(chord_tones + extensions))

        # Find all notes below melody within ~2 octaves
        notes_below = []
        for note in available:
            # Try this note in multiple octaves below melody
            for oct in range(-3, 0):  # Up to 3 octaves below
                candidate = note + (oct * 12)
                if 24 <= candidate < melody_note:  # Valid MIDI range and below melody
                    notes_below.append(candidate)

        notes_below = sorted(set(notes_below), reverse=True)  # Highest first

        # Pick 3 notes below melody in close position
        # Strategy: Start from just below melody, pick notes descending
        selected = []
        last_note = melody_note

        for candidate in notes_below:
            # Only add if it's close enough (within a couple semitones of last note)
            if last_note - candidate <= 7:  # Within a 5th
                selected.append(candidate)
                last_note = candidate
                if len(selected) >= 3:
                    break

        # If we don't have enough notes, fill in with chord tones
        while len(selected) < 3 and notes_below:
            for note in notes_below:
                if note not in selected:
                    selected.append(note)
                    if len(selected) >= 3:
                        break

        # Combine and return: bottom 3 + melody on top
        result = sorted(selected[:3]) + [melody_note]
        return result

    @staticmethod
    def five_way_closed(melody_note: int, chord_tones: List[int], extensions: List[int] = None) -> List[int]:
        """
        5-way closed voicing: 5 voices in close position with melody on top.

        Adds more harmonic richness with 5th voice.
        """
        if extensions is None:
            extensions = []

        # Available notes
        available = sorted(set(chord_tones + extensions))

        # Find all notes below melody
        notes_below = []
        for note in available:
            for oct in range(-3, 0):
                candidate = note + (oct * 12)
                if 24 <= candidate < melody_note:
                    notes_below.append(candidate)

        notes_below = sorted(set(notes_below), reverse=True)

        # Pick 4 notes below melody
        selected = []
        last_note = melody_note

        for candidate in notes_below:
            if last_note - candidate <= 6:  # Keep it close
                selected.append(candidate)
                last_note = candidate
                if len(selected) >= 4:
                    break

        # Fill if needed
        while len(selected) < 4 and notes_below:
            for note in notes_below:
                if note not in selected:
                    selected.append(note)
                    if len(selected) >= 4:
                        break

        result = sorted(selected[:4]) + [melody_note]
        return result

    @staticmethod
    def block_chords(melody_note: int, chord_tones: List[int], extensions: List[int] = None) -> List[int]:
        """
        Block chords: Melody doubled, with block chord below.

        Creates a powerful, full sound.
        """
        if extensions is None:
            extensions = []

        # Use 4-way closed for the block
        four_way = MelodyVoicing.four_way_closed(melody_note, chord_tones, extensions)

        # Double the melody an octave below
        melody_double = melody_note - 12 if melody_note - 12 >= 24 else melody_note

        # Combine: block + melody + melody double
        result = sorted(set(four_way + [melody_double]))
        return result


# ============================================================================
# MELODY HARMONIZER
# ============================================================================

def harmonize_melody(
    input_midi_path: str,
    output_midi_path: str,
    voicing_type: str = '5-way-closed',
    scale_context: Optional[ScaleContext] = None,
    chord_progression: Optional[Dict[int, str]] = None
) -> str:
    """
    Harmonize a melody MIDI file.

    Args:
        input_midi_path: Path to input melody MIDI
        voicing_type: '4-way-closed', '5-way-closed', or 'block'
        scale_context: Scale context for harmonic analysis
        chord_progression: Optional chord progression {beat: chord_name}

    Returns:
        Path to output harmonized MIDI file
    """
    print(f"\n{'='*80}")
    print(f"🎵 MELODY HARMONIZER")
    print(f"{'='*80}")
    print(f"Input: {input_midi_path}")
    print(f"Voicing: {voicing_type}")

    # Extract melody
    melody_events = MelodyAnalyzer.extract_melody(input_midi_path)
    print(f"Extracted {len(melody_events)} melody notes")

    # Auto-detect scale if not provided
    if scale_context is None:
        # Analyze first note to guess scale
        first_note = melody_events[0][2]
        # Assume C major for now
        scale_context = ScaleContext(60, 'major')
        print(f"Auto-detected scale: {scale_context}")

    # Create harmonic analyzer
    analyzer = HarmonicAnalyzer(scale_context)

    # Load input MIDI to get tempo and setup
    input_mid = mido.MidiFile(input_midi_path)
    ticks_per_beat = input_mid.ticks_per_beat

    # Create output MIDI
    output_mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)
    output_track = mido.MidiTrack()
    output_mid.tracks.append(output_track)

    # Copy tempo from input
    for msg in input_mid.tracks[0]:
        if msg.type in ['set_tempo', 'time_signature']:
            output_track.append(msg.copy())

    # Harmonize each melody note
    current_tick = 0

    for start_tick, duration, melody_note, velocity in melody_events:
        # Analyze melody note
        analysis = analyzer.analyze_melody_note(melody_note)

        print(f"\n  Note {melody_note}: scale degree {analysis['scale_degree']}")
        print(f"    Chord tones: {analysis['chord_tones']}")
        print(f"    Extensions: {analysis['extensions']}")

        # Generate voicing
        if voicing_type == '4-way-closed':
            voicing = MelodyVoicing.four_way_closed(
                melody_note,
                analysis['chord_tones'],
                analysis['extensions']
            )
        elif voicing_type == '5-way-closed':
            voicing = MelodyVoicing.five_way_closed(
                melody_note,
                analysis['chord_tones'],
                analysis['extensions']
            )
        elif voicing_type == 'block':
            voicing = MelodyVoicing.block_chords(
                melody_note,
                analysis['chord_tones'],
                analysis['extensions']
            )
        else:
            voicing = [melody_note]  # Fallback

        print(f"    Voicing: {voicing}")

        # Add note_on messages
        delta = start_tick - current_tick
        for i, note in enumerate(voicing):
            output_track.append(mido.Message(
                'note_on',
                note=note,
                velocity=velocity,
                time=delta if i == 0 else 0
            ))

        current_tick = start_tick

        # Add note_off messages
        end_tick = start_tick + duration
        delta = end_tick - current_tick
        for i, note in enumerate(voicing):
            output_track.append(mido.Message(
                'note_off',
                note=note,
                velocity=0,
                time=delta if i == 0 else 0
            ))

        current_tick = end_tick

    # Save output
    output_track.append(mido.MetaMessage('end_of_track', time=0))
    output_mid.save(output_midi_path)

    print(f"\n✅ Harmonized MIDI saved: {output_midi_path}")
    print(f"{'='*80}\n")

    return output_midi_path


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Harmonize melody MIDI files')
    parser.add_argument('input', help='Input melody MIDI file')
    parser.add_argument('--output', '-o', help='Output MIDI file')
    parser.add_argument('--voicing', '-v',
                       choices=['4-way-closed', '5-way-closed', 'block'],
                       default='5-way-closed',
                       help='Voicing type')
    parser.add_argument('--key', '-k', help='Key (e.g., C, Dm, Eb)')

    args = parser.parse_args()

    # Parse key if provided
    scale_context = None
    if args.key:
        is_minor = 'm' in args.key.lower()
        root_note = _parse_chord_root(args.key)
        scale_type = 'minor' if is_minor else 'major'
        scale_context = ScaleContext(root_note, scale_type)

    # Generate output path if not provided
    output_path = args.output
    if not output_path:
        input_path = Path(args.input)
        output_path = input_path.parent / f"{input_path.stem}_harmonized_{args.voicing}.mid"

    # Harmonize
    harmonize_melody(
        args.input,
        str(output_path),
        voicing_type=args.voicing,
        scale_context=scale_context
    )

    print(f"✅ Done! Output: {output_path}")
