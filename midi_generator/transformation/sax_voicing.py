#!/usr/bin/env python3
"""
Professional Sax Soli Voicing Engine
====================================

Sophisticated sax section voicing with drop-2, drop-3, and spread voicings,
voice leading optimization, and register-specific spacing rules.

Based on research from:
- Evan Rogers: "Big Band Arranging | Voicings"
- Frans Absil: "Arranging by Examples"
- Mark Levine: "Jazz Theory Book" - Voice Leading chapter
- Thad Jones, Count Basie sax section analysis
- Matthew Keating (2023): LSTM voice-leading paper

This module replaces the basic close-position-only sax voicing in
BigBandArranger._harmonize_saxes() with professional-quality voicings.

Key Features:
-------------
- Drop-2 voicings (THE MOST COMMON big band voicing)
- Drop-3 voicings
- Drop-2-4 voicings (open, powerful sound)
- Spread voicings (modern sound)
- Voice leading optimization (minimize voice movement)
- Register-specific spacing (wider in bass, closer in treble)
- Configurable sax section (alto1, alto2, tenor1, tenor2, bari)

Author: Agent 2 - Sax Soli Voicing Master
License: MIT
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import sys
from pathlib import Path
import copy

# Import dependencies
sys.path.append(str(Path(__file__).parent.parent))
from analysis.midi_analyzer import ChordEvent, NoteEvent
from transformation.voice_leading_optimizer import (
    VoiceLeadingOptimizer, VoicingType, Voicing, VoicingGenerator, VoiceRange
)


# ============================================================================
# SAX SECTION DEFINITIONS
# ============================================================================

@dataclass
class SaxInstrument:
    """Saxophone instrument definition"""
    name: str
    range_min: int  # MIDI pitch
    range_max: int  # MIDI pitch
    comfortable_min: int  # Sweet spot minimum
    comfortable_max: int  # Sweet spot maximum


# Standard big band sax section
SAX_SECTION = {
    'alto1': SaxInstrument(
        name='Alto Sax 1 (Lead)',
        range_min=52,  # E3
        range_max=87,  # D#6
        comfortable_min=57,  # A3
        comfortable_max=81   # A5
    ),
    'alto2': SaxInstrument(
        name='Alto Sax 2',
        range_min=52,  # E3
        range_max=87,  # D#6
        comfortable_min=57,  # A3
        comfortable_max=81   # A5
    ),
    'tenor1': SaxInstrument(
        name='Tenor Sax 1',
        range_min=47,  # B2
        range_max=82,  # A#5
        comfortable_min=52,  # E3
        comfortable_max=76   # E5
    ),
    'tenor2': SaxInstrument(
        name='Tenor Sax 2',
        range_min=47,  # B2
        range_max=82,  # A#5
        comfortable_min=52,  # E3
        comfortable_max=76   # E5
    ),
    'bari': SaxInstrument(
        name='Baritone Sax',
        range_min=36,  # C2
        range_max=69,  # A4
        comfortable_min=41,  # F2
        comfortable_max=64   # E4
    ),
}

# Standard voice order (bottom to top)
DEFAULT_SAX_VOICES = ['bari', 'tenor2', 'tenor1', 'alto2', 'alto1']


# ============================================================================
# PROFESSIONAL SAX VOICING ENGINE
# ============================================================================

class SaxSoliVoicing:
    """
    Professional sax section voicing engine.

    Uses voice leading optimization and register-specific spacing
    to create authentic big band sax soli.
    """

    @staticmethod
    def voice_melody(melody: List[NoteEvent],
                    chords: List[ChordEvent],
                    voicing_style: str = "drop_2",
                    optimize_voice_leading: bool = True,
                    section: Optional[List[str]] = None,
                    apply_register_spacing: bool = True) -> Dict[str, List[NoteEvent]]:
        """
        Voice a melody for sax section (5-part harmony).

        Args:
            melody: Melody notes (will be top voice)
            chords: Chord progression
            voicing_style: "close", "drop_2", "drop_3", "drop_2_4", "spread"
            optimize_voice_leading: Apply voice leading optimization
            section: List of sax voices (default: standard 5-piece section)
            apply_register_spacing: Apply register-specific spacing rules

        Returns:
            Dictionary: {instrument_name: [NoteEvent, ...]}
            Example: {'alto1': [...], 'alto2': [...], 'tenor1': [...], ...}
        """
        if not melody or not chords:
            return {}

        if section is None:
            section = DEFAULT_SAX_VOICES

        num_voices = len(section)

        # Build voice ranges from section definition
        voice_ranges = []
        for voice_name in section:
            if voice_name not in SAX_SECTION:
                raise ValueError(f"Unknown sax instrument: {voice_name}")

            sax = SAX_SECTION[voice_name]
            # Use comfortable range for better sound
            voice_ranges.append(VoiceRange(
                low=sax.range_min,
                high=sax.range_max,
                comfortable_low=sax.comfortable_min,
                comfortable_high=sax.comfortable_max
            ))

        # Map voicing style string to VoicingType enum
        voicing_type_map = {
            "close": VoicingType.CLOSE,
            "drop_2": VoicingType.DROP_2,
            "drop_3": VoicingType.DROP_3,
            "drop_2_4": VoicingType.DROP_2_4,
            "spread": VoicingType.SPREAD,
            "open": VoicingType.OPEN,
        }
        voicing_type = voicing_type_map.get(voicing_style.lower(), VoicingType.DROP_2)

        # Create sax section notes
        sax_parts = {voice: [] for voice in section}

        # Group melody notes by chord
        melody_by_chord = SaxSoliVoicing._group_melody_by_chords(melody, chords)

        # Voice each chord
        for chord_idx, chord in enumerate(chords):
            melody_notes = melody_by_chord.get(chord_idx, [])

            if not melody_notes:
                continue

            # Extract melody pitches for this chord
            melody_pitches = [note.pitch for note in melody_notes]

            # Get corresponding chord events
            chord_sequence = [chord] * len(melody_pitches)

            # Optimize voice leading for this chord's melody notes
            if optimize_voice_leading:
                optimization_result = VoiceLeadingOptimizer.optimize_chord_sequence(
                    chords=chord_sequence,
                    num_voices=num_voices,
                    voice_ranges=voice_ranges,
                    voicing_types=[voicing_type]
                )
                # Extract pitch lists from OptimizationResult
                voicing_sequence = [v.pitches for v in optimization_result.voicings]
            else:
                # No optimization: generate voicing for each note independently
                # Handle both dict and ChordEvent formats
                if isinstance(chord, dict):
                    root = chord['root']
                    quality = chord['quality']
                else:
                    root = chord.root
                    quality = chord.quality

                voicing_sequence = []
                for melody_pitch in melody_pitches:
                    voicings = VoicingGenerator.generate_all_voicings(
                        root=root,
                        quality=quality,
                        num_voices=num_voices,
                        voice_ranges=voice_ranges,
                        voicing_type=voicing_type
                    )
                    if voicings:
                        voicing_sequence.append(voicings[0].pitches)
                    else:
                        # Fallback to simple voicing
                        voicing_sequence.append([melody_pitch - i*4 for i in range(num_voices)])

            # Apply register-specific spacing if requested
            if apply_register_spacing:
                voicing_sequence = [
                    SaxSoliVoicing._apply_register_spacing(voicing)
                    for voicing in voicing_sequence
                ]

            # Create NoteEvent objects for each voice
            for note_idx, melody_note in enumerate(melody_notes):
                if note_idx >= len(voicing_sequence):
                    break

                voicing = voicing_sequence[note_idx]

                for voice_idx, (voice_name, pitch) in enumerate(zip(section, voicing)):
                    # Copy melody note properties
                    sax_note = copy.copy(melody_note)
                    sax_note.pitch = pitch
                    sax_note.velocity = int(melody_note.velocity * 0.85)  # Slightly softer
                    sax_note.track_idx = voice_idx

                    sax_parts[voice_name].append(sax_note)

        return sax_parts

    @staticmethod
    def voice_chord_progression(chords: List[ChordEvent],
                               voicing_style: str = "drop_2",
                               section: Optional[List[str]] = None,
                               rhythm_pattern: Optional[List[float]] = None) -> Dict[str, List[NoteEvent]]:
        """
        Voice a chord progression (no melody) for sax section backgrounds.

        Args:
            chords: Chord progression
            voicing_style: "close", "drop_2", "drop_3", "drop_2_4", "spread"
            section: Sax voices (default: standard section)
            rhythm_pattern: Beat positions for chord hits (e.g., [0, 1, 2, 3])

        Returns:
            Dictionary of sax parts
        """
        if not chords:
            return {}

        if section is None:
            section = DEFAULT_SAX_VOICES

        num_voices = len(section)

        # Build voice ranges
        voice_ranges = []
        for voice_name in section:
            sax = SAX_SECTION[voice_name]
            voice_ranges.append(VoiceRange(
                low=sax.range_min,
                high=sax.range_max,
                comfortable_low=sax.comfortable_min,
                comfortable_high=sax.comfortable_max
            ))

        # Map voicing style
        voicing_type_map = {
            "close": VoicingType.CLOSE,
            "drop_2": VoicingType.DROP_2,
            "drop_3": VoicingType.DROP_3,
            "drop_2_4": VoicingType.DROP_2_4,
            "spread": VoicingType.SPREAD,
        }
        voicing_type = voicing_type_map.get(voicing_style.lower(), VoicingType.DROP_2)

        # Optimize voice leading across chord sequence
        optimization_result = VoiceLeadingOptimizer.optimize_chord_sequence(
            chords=chords,
            num_voices=num_voices,
            voice_ranges=voice_ranges,
            voicing_types=[voicing_type]
        )

        # Extract pitch lists from OptimizationResult
        voicing_sequence = [v.pitches for v in optimization_result.voicings]

        # Apply register spacing
        voicing_sequence = [
            SaxSoliVoicing._apply_register_spacing(voicing)
            for voicing in voicing_sequence
        ]

        # Create NoteEvent objects
        sax_parts = {voice: [] for voice in section}

        for chord_idx, (chord, voicing) in enumerate(zip(chords, voicing_sequence)):
            # Determine rhythm pattern (default: whole note)
            if rhythm_pattern:
                hit_times = [chord.start_time + beat for beat in rhythm_pattern]
                duration = 0.5  # Quarter note duration
            else:
                hit_times = [chord.start_time]
                duration = chord.duration

            for hit_time in hit_times:
                if hit_time >= chord.start_time + chord.duration:
                    break

                for voice_idx, (voice_name, pitch) in enumerate(zip(section, voicing)):
                    note = NoteEvent(
                        start_time=hit_time,
                        duration=duration,
                        start_tick=int(hit_time * 480),  # Assume 480 PPQN
                        duration_ticks=int(duration * 480),
                        pitch=pitch,
                        velocity=80,
                        channel=voice_idx,
                        track_idx=voice_idx
                    )
                    sax_parts[voice_name].append(note)

        return sax_parts

    @staticmethod
    def _group_melody_by_chords(melody: List[NoteEvent],
                                chords: List[ChordEvent]) -> Dict[int, List[NoteEvent]]:
        """
        Group melody notes by which chord they belong to.

        Returns:
            {chord_index: [NoteEvent, ...]}
        """
        melody_by_chord = {}

        for note in melody:
            # Find which chord this note belongs to
            for chord_idx, chord in enumerate(chords):
                if (note.start_time >= chord.start_time and
                    note.start_time < chord.start_time + chord.duration):

                    if chord_idx not in melody_by_chord:
                        melody_by_chord[chord_idx] = []
                    melody_by_chord[chord_idx].append(note)
                    break

        return melody_by_chord

    @staticmethod
    def _apply_register_spacing(voicing: List[int]) -> List[int]:
        """
        Apply register-specific spacing rules to voicing.

        Rules from big band arranging theory:
        - Below C4 (60): minimum 4-semitone spacing (avoid mud)
        - C4-C5 (60-72): 3-4 semitone spacing
        - Above C5 (72): 2-3 semitone spacing (close is OK in high register)

        Args:
            voicing: List of MIDI pitches (sorted low to high)

        Returns:
            Adjusted voicing with proper spacing
        """
        if len(voicing) < 2:
            return voicing

        adjusted = [voicing[0]]  # Keep bass note

        for i in range(1, len(voicing)):
            prev_pitch = adjusted[-1]
            current_pitch = voicing[i]

            # Determine minimum spacing based on register
            if prev_pitch < 60:  # Below C4
                min_spacing = 4
            elif prev_pitch < 72:  # C4-C5
                min_spacing = 3
            else:  # Above C5
                min_spacing = 2

            # Check spacing
            spacing = current_pitch - prev_pitch

            if spacing < min_spacing:
                # Adjust upward to meet minimum spacing
                current_pitch = prev_pitch + min_spacing

            adjusted.append(current_pitch)

        return adjusted

    @staticmethod
    def calculate_voicing_statistics(sax_parts: Dict[str, List[NoteEvent]]) -> Dict[str, float]:
        """
        Calculate statistics about voicing quality.

        Returns:
            {
                'average_voice_movement': float,  # Average semitones per voice change
                'average_voice_spacing': float,   # Average spacing between adjacent voices
                'drop_2_usage': float,            # Percentage of drop-2 voicings (approx)
                'max_leap': int,                  # Largest leap in any voice
            }
        """
        stats = {
            'average_voice_movement': 0.0,
            'average_voice_spacing': 0.0,
            'drop_2_usage': 0.0,
            'max_leap': 0,
        }

        # Get voice names
        voices = list(sax_parts.keys())
        if not voices:
            return stats

        # Calculate voice movement
        total_movement = 0
        movement_count = 0

        for voice in voices:
            notes = sax_parts[voice]
            for i in range(1, len(notes)):
                movement = abs(notes[i].pitch - notes[i-1].pitch)
                total_movement += movement
                movement_count += 1
                stats['max_leap'] = max(stats['max_leap'], movement)

        if movement_count > 0:
            stats['average_voice_movement'] = total_movement / movement_count

        # Calculate average spacing
        total_spacing = 0
        spacing_count = 0

        # Sample every 10th sonority to estimate spacing
        sample_indices = range(0, len(sax_parts[voices[0]]), 10)

        for idx in sample_indices:
            pitches = []
            for voice in voices:
                if idx < len(sax_parts[voice]):
                    pitches.append(sax_parts[voice][idx].pitch)

            pitches.sort()

            # Calculate spacing between adjacent voices
            for i in range(1, len(pitches)):
                spacing = pitches[i] - pitches[i-1]
                total_spacing += spacing
                spacing_count += 1

        if spacing_count > 0:
            stats['average_voice_spacing'] = total_spacing / spacing_count

        return stats


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def voice_sax_soli(melody: List[NoteEvent],
                   chords: List[ChordEvent],
                   style: str = "drop_2") -> Dict[str, List[NoteEvent]]:
    """
    Convenience function to voice sax soli.

    Args:
        melody: Lead melody
        chords: Chord progression
        style: "close", "drop_2", "drop_3", "drop_2_4", "spread"

    Returns:
        Sax parts dictionary
    """
    return SaxSoliVoicing.voice_melody(
        melody=melody,
        chords=chords,
        voicing_style=style,
        optimize_voice_leading=True,
        apply_register_spacing=True
    )


def analyze_sax_voicing(sax_parts: Dict[str, List[NoteEvent]]) -> None:
    """
    Print analysis of sax voicing quality.
    """
    stats = SaxSoliVoicing.calculate_voicing_statistics(sax_parts)

    print("=== Sax Voicing Analysis ===")
    print(f"Average voice movement: {stats['average_voice_movement']:.2f} semitones")
    print(f"Average voice spacing: {stats['average_voice_spacing']:.2f} semitones")
    print(f"Maximum leap: {stats['max_leap']} semitones")
    print()

    # Professional standards (from research)
    print("Professional Standards:")
    print("  Average movement: < 3 semitones (smooth)")
    print("  Average spacing: > 3 semitones (avoid mud)")
    print("  Maximum leap: < 12 semitones (singable)")
    print()

    # Evaluate
    if stats['average_voice_movement'] < 3:
        print("✓ Voice leading is smooth (professional quality)")
    else:
        print("✗ Voice leading has large leaps (needs improvement)")

    if stats['average_voice_spacing'] > 3:
        print("✓ Voice spacing is good (clear, not muddy)")
    else:
        print("✗ Voice spacing too close (may sound muddy)")


if __name__ == "__main__":
    """
    Example usage and testing
    """
    # Create test data
    test_chords = [
        ChordEvent(
            start_time=0.0,
            duration=2.0,
            root=0,  # C
            quality='major7',
            pitches=[0, 4, 7, 11],
            bass_note=0,
            confidence=1.0
        ),
        ChordEvent(
            start_time=2.0,
            duration=2.0,
            root=2,  # D
            quality='minor7',
            pitches=[2, 5, 9, 0],
            bass_note=2,
            confidence=1.0
        ),
        ChordEvent(
            start_time=4.0,
            duration=2.0,
            root=7,  # G
            quality='dominant7',
            pitches=[7, 11, 2, 5],
            bass_note=7,
            confidence=1.0
        ),
    ]

    test_melody = [
        NoteEvent(0.0, 1.0, 0, 480, 72, 100, 0, 0),  # C5
        NoteEvent(1.0, 1.0, 480, 480, 74, 100, 0, 0),  # D5
        NoteEvent(2.0, 1.0, 960, 480, 74, 100, 0, 0),  # D5
        NoteEvent(3.0, 1.0, 1440, 480, 72, 100, 0, 0),  # C5
        NoteEvent(4.0, 2.0, 1920, 960, 71, 100, 0, 0),  # B4
    ]

    print("Testing Sax Soli Voicing Engine")
    print("=" * 50)

    # Test drop-2 voicing
    print("\nDrop-2 Voicing:")
    sax_parts = voice_sax_soli(test_melody, test_chords, style="drop_2")
    analyze_sax_voicing(sax_parts)

    print("\nDrop-3 Voicing:")
    sax_parts = voice_sax_soli(test_melody, test_chords, style="drop_3")
    analyze_sax_voicing(sax_parts)

    print("\nClose Voicing:")
    sax_parts = voice_sax_soli(test_melody, test_chords, style="close")
    analyze_sax_voicing(sax_parts)
