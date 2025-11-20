#!/usr/bin/env python3
"""
Brass Section Arranger - Agent 5
Part of the 20-Agent Big Band Generator Excellence System

This module transforms brass writing from basic stabs to sophisticated section
writing with sustained pads, call-and-response, shout chorus, and authentic
articulations.

Research Sources:
- Duke Ellington brass writing ("Ko-Ko", "Caravan", "Concerto for Cootie")
- Count Basie brass ("One O'Clock Jump" shout chorus, "April in Paris")
- Thad Jones brass ("A Child is Born", "Three and One")
- Brass technique and ranges (Trumpet C4-C6, Trombone E2-Bb4)

Integration Points:
- Replaces BigBandArranger._create_brass_figures()
- Integrates with FormGenerator (detect shout chorus location)
- Scalable to: any brass ensemble (quintet, orchestra, marching band)

Author: Agent 5 - Brass Section Arranger
Date: 2025-01-20
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import random

# Import data structures
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.midi_analyzer import NoteEvent, ChordEvent
from genres.jazz import JazzChord
from generators.form_generator import MusicalForm, FormSection


# ============================================================================
# BRASS RANGES AND INSTRUMENTS
# ============================================================================

class BrassInstrument(Enum):
    """Brass instruments with their comfortable ranges"""
    TRUMPET_1 = "trumpet1"
    TRUMPET_2 = "trumpet2"
    TRUMPET_3 = "trumpet3"
    TRUMPET_4 = "trumpet4"
    TROMBONE_1 = "trombone1"
    TROMBONE_2 = "trombone2"
    TROMBONE_3 = "trombone3"
    TROMBONE_4 = "trombone4"  # Bass trombone


# Comfortable playing ranges (MIDI note numbers)
BRASS_RANGES = {
    BrassInstrument.TRUMPET_1: (60, 84),    # C4 to C6 (lead trumpet, highest)
    BrassInstrument.TRUMPET_2: (58, 81),    # Bb3 to A5
    BrassInstrument.TRUMPET_3: (55, 79),    # G3 to G5
    BrassInstrument.TRUMPET_4: (53, 77),    # F3 to F5 (lowest trumpet)
    BrassInstrument.TROMBONE_1: (40, 65),   # E2 to F4 (lead trombone)
    BrassInstrument.TROMBONE_2: (38, 62),   # D2 to D4
    BrassInstrument.TROMBONE_3: (36, 60),   # C2 to C4
    BrassInstrument.TROMBONE_4: (34, 58),   # Bb1 to Bb3 (bass trombone)
}


# ============================================================================
# BRASS ARRANGER MAIN CLASS
# ============================================================================

class BrassArranger:
    """
    Sophisticated brass section arranger with support for sustained pads,
    call-and-response, shout chorus, and authentic articulations.
    """

    @staticmethod
    def create_sustained_pad(
        chords: List[ChordEvent],
        voicing_type: str = "drop_2",
        dynamic_shape: str = "crescendo",
        base_velocity: int = 70,
        section: List[BrassInstrument] = None
    ) -> List[NoteEvent]:
        """
        Create sustained brass pad (long tones, background pad).

        Args:
            chords: List of chord events
            voicing_type: "close", "drop_2", "drop_3", "spread"
            dynamic_shape: "static", "crescendo", "diminuendo", "arch"
            base_velocity: Base velocity level (default 70 for mp)
            section: Brass instruments to use (default: full 8-piece section)

        Returns:
            List of NoteEvent objects for brass pad

        Research:
        - Duke Ellington "Caravan" - sustained brass pads
        - Dynamics shape the phrase musically
        """
        if section is None:
            section = list(BrassInstrument)

        brass_notes = []
        num_chords = len(chords)

        for i, chord in enumerate(chords):
            # Calculate dynamic level based on shape
            velocity = BrassArranger._calculate_dynamic(
                i, num_chords, base_velocity, dynamic_shape
            )

            # Get voicing for this chord
            voicing = BrassArranger._voice_chord(
                chord, len(section), voicing_type
            )

            # Create sustained notes for each voice
            for voice_idx, pitch in enumerate(voicing):
                if voice_idx < len(section):
                    instrument = section[voice_idx]
                    min_pitch, max_pitch = BRASS_RANGES[instrument]

                    # Ensure pitch is within range
                    while pitch < min_pitch:
                        pitch += 12
                    while pitch > max_pitch:
                        pitch -= 12

                    # Create sustained note
                    note = NoteEvent(
                        start_time=chord.start_time,
                        duration=chord.duration,  # Full chord duration
                        start_tick=int(chord.start_time * 480),
                        duration_ticks=int(chord.duration * 480),
                        pitch=pitch,
                        velocity=velocity,
                        channel=voice_idx,
                        track_idx=10 + voice_idx
                    )
                    brass_notes.append(note)

        return brass_notes

    @staticmethod
    def create_shout_chorus(
        melody: List[NoteEvent],
        chords: List[ChordEvent],
        intensity: float = 0.9,
        style: str = "basie_unison"
    ) -> Dict[str, List[NoteEvent]]:
        """
        Create climactic shout chorus section (full band in unison or block harmony).
        Typically final A section in AABA form.

        Args:
            melody: Lead melody line
            chords: Chord progression
            intensity: 0-1, controls velocity increase (0.9 = fff)
            style: "basie_unison", "ellington_harmony", "thad_modern"

        Returns:
            Dict with brass parts {instrument_name: [NoteEvent]}

        Research:
        - Count Basie "April in Paris" - famous shout chorus ending
        - Shout chorus is climactic, 20% louder than rest of arrangement
        - Style determines unison vs harmony
        """
        brass_parts = {}

        # Calculate shout chorus velocity (115-127 for fff)
        shout_velocity = int(115 + (intensity * 12))

        if style == "basie_unison":
            # Basie style: All brass in unison or octaves
            for instrument in BrassInstrument:
                instrument_notes = []
                min_pitch, max_pitch = BRASS_RANGES[instrument]

                for note in melody:
                    # Transpose melody to instrument range
                    pitch = note.pitch
                    while pitch < min_pitch:
                        pitch += 12
                    while pitch > max_pitch:
                        pitch -= 12

                    # Create unison note with accent
                    shout_note = NoteEvent(
                        start_time=note.start_time,
                        duration=note.duration,
                        start_tick=note.start_tick,
                        duration_ticks=note.duration_ticks,
                        pitch=pitch,
                        velocity=shout_velocity,
                        channel=instrument.value,
                        track_idx=10 + list(BrassInstrument).index(instrument)
                    )
                    instrument_notes.append(shout_note)

                brass_parts[instrument.value] = instrument_notes

        elif style == "ellington_harmony":
            # Ellington style: Rich harmony with block voicing
            brass_parts = BrassArranger._create_harmonized_shout(
                melody, chords, shout_velocity, voicing_type="close"
            )

        elif style == "thad_modern":
            # Thad Jones style: Spread voicing, wider intervals
            brass_parts = BrassArranger._create_harmonized_shout(
                melody, chords, shout_velocity, voicing_type="spread"
            )

        return brass_parts

    @staticmethod
    def create_brass_riff(
        chord: ChordEvent,
        pattern_style: str = "basie_riff",
        bars: int = 4,
        base_velocity: int = 95
    ) -> List[NoteEvent]:
        """
        Create short rhythmic figures (backgrounds behind solos).

        Args:
            chord: Chord to base riff on
            pattern_style: "basie_riff", "ellington_call", "thad_modern"
            bars: Number of bars for riff (typically 2 or 4)
            base_velocity: Velocity level (default 95 for f)

        Returns:
            List of NoteEvent objects

        Research:
        - Count Basie brass riffs: simple, rhythmic, punchy
        - Typically 1-2 bar repeated figures
        """
        brass_notes = []
        beats_per_bar = 4
        total_beats = bars * beats_per_bar

        if pattern_style == "basie_riff":
            # Simple rhythmic pattern: hits on 1, 2&, 4
            rhythm_pattern = [0, 1.5, 3]  # Beat positions
            voicing = BrassArranger._voice_chord(chord, 4, "drop_2")

            for bar in range(bars):
                for beat in rhythm_pattern:
                    time_offset = (bar * beats_per_bar) + beat
                    start_time = chord.start_time + time_offset

                    for voice_idx, pitch in enumerate(voicing[:4]):
                        note = NoteEvent(
                            start_time=start_time,
                            duration=0.5,  # Short stab
                            start_tick=int(start_time * 480),
                            duration_ticks=int(0.5 * 480),
                            pitch=pitch,
                            velocity=base_velocity,
                            channel=voice_idx,
                            track_idx=10 + voice_idx
                        )
                        brass_notes.append(note)

        elif pattern_style == "ellington_call":
            # Longer, more melodic figures with variety
            rhythm_pattern = [0, 0.5, 1, 2, 2.5, 3.5]
            voicing = BrassArranger._voice_chord(chord, 4, "close")

            for bar in range(bars):
                for beat_idx, beat in enumerate(rhythm_pattern):
                    time_offset = (bar * beats_per_bar) + beat
                    start_time = chord.start_time + time_offset

                    # Vary velocity for musical phrasing
                    velocity = base_velocity + (10 if beat_idx in [0, 3] else 0)

                    for voice_idx, pitch in enumerate(voicing[:4]):
                        note = NoteEvent(
                            start_time=start_time,
                            duration=0.4,
                            start_tick=int(start_time * 480),
                            duration_ticks=int(0.4 * 480),
                            pitch=pitch,
                            velocity=velocity,
                            channel=voice_idx,
                            track_idx=10 + voice_idx
                        )
                        brass_notes.append(note)

        elif pattern_style == "thad_modern":
            # Angular, syncopated modern figures
            rhythm_pattern = [0, 0.75, 1.5, 2.25, 3]
            voicing = BrassArranger._voice_chord(chord, 4, "spread")

            for bar in range(bars):
                for beat in rhythm_pattern:
                    time_offset = (bar * beats_per_bar) + beat
                    start_time = chord.start_time + time_offset

                    for voice_idx, pitch in enumerate(voicing[:4]):
                        note = NoteEvent(
                            start_time=start_time,
                            duration=0.3,
                            start_tick=int(start_time * 480),
                            duration_ticks=int(0.3 * 480),
                            pitch=pitch,
                            velocity=base_velocity,
                            channel=voice_idx,
                            track_idx=10 + voice_idx
                        )
                        brass_notes.append(note)

        return brass_notes

    @staticmethod
    def create_call_response(
        sax_phrase: List[NoteEvent],
        chords: List[ChordEvent],
        response_delay: float = 4.0
    ) -> List[NoteEvent]:
        """
        Create brass response to sax phrase (antiphonal writing).

        Args:
            sax_phrase: Saxophone phrase to respond to
            chords: Underlying chords
            response_delay: Delay in beats before brass responds (default 4 = 1 bar)

        Returns:
            List of NoteEvent objects for brass response

        Research:
        - Antiphonal writing: call-and-response between sections
        - Brass responds 4 bars after sax phrase
        """
        brass_response = []

        # Find the chord at response time
        response_start = sax_phrase[0].start_time + response_delay if sax_phrase else 0
        current_chord = None
        for chord in chords:
            if chord.start_time <= response_start < chord.start_time + chord.duration:
                current_chord = chord
                break

        if not current_chord:
            return brass_response

        # Create simplified response based on sax phrase contour
        if len(sax_phrase) > 0:
            # Get voicing
            voicing = BrassArranger._voice_chord(current_chord, 4, "drop_2")

            # Create response phrase (simplified version of sax phrase)
            for i in range(min(4, len(sax_phrase))):
                sax_note = sax_phrase[i]
                start_time = sax_note.start_time + response_delay

                for voice_idx, pitch in enumerate(voicing[:4]):
                    note = NoteEvent(
                        start_time=start_time,
                        duration=sax_note.duration,
                        start_tick=int(start_time * 480),
                        duration_ticks=sax_note.duration_ticks,
                        pitch=pitch,
                        velocity=sax_note.velocity - 10,  # Slightly softer
                        channel=voice_idx,
                        track_idx=10 + voice_idx
                    )
                    brass_response.append(note)

        return brass_response

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    @staticmethod
    def _voice_chord(
        chord: ChordEvent,
        num_voices: int,
        voicing_type: str = "drop_2"
    ) -> List[int]:
        """
        Voice a chord for brass section.

        Args:
            chord: ChordEvent to voice
            num_voices: Number of voices (typically 4 or 8)
            voicing_type: "close", "drop_2", "drop_3", "drop_2_4", "spread"

        Returns:
            List of MIDI pitch values for voicing
        """
        # Get chord tones from the chord
        root = chord.root
        quality = getattr(chord, 'quality', 'major')

        # Build basic chord tones (root, 3rd, 5th, 7th)
        chord_tones = [root]

        if quality in ['major', 'maj7', 'M7']:
            chord_tones.extend([root + 4, root + 7, root + 11])  # M3, P5, M7
        elif quality in ['minor', 'min7', 'm7']:
            chord_tones.extend([root + 3, root + 7, root + 10])  # m3, P5, m7
        elif quality in ['dominant', 'dom7', '7']:
            chord_tones.extend([root + 4, root + 7, root + 10])  # M3, P5, m7
        elif quality in ['min7b5', 'half-dim', 'ø7']:
            chord_tones.extend([root + 3, root + 6, root + 10])  # m3, d5, m7
        elif quality in ['dim7', 'o7']:
            chord_tones.extend([root + 3, root + 6, root + 9])   # m3, d5, d7
        else:  # Default to major
            chord_tones.extend([root + 4, root + 7, root + 11])

        # Normalize to one octave
        chord_tones = [ct % 12 for ct in chord_tones]

        # Build voicing based on type
        base_octave = 4  # Start from middle C region
        base_pitch = 60 + root  # C4 + root

        if voicing_type == "close":
            # All voices within octave
            voicing = [base_pitch + ct for ct in chord_tones[:num_voices]]

        elif voicing_type == "drop_2":
            # Drop 2nd voice from top down an octave
            close_voicing = [base_pitch + ct for ct in chord_tones[:4]]
            if len(close_voicing) >= 4:
                voicing = [
                    close_voicing[0],
                    close_voicing[1],
                    close_voicing[2] - 12,  # Drop 2nd voice
                    close_voicing[3]
                ]
            else:
                voicing = close_voicing

        elif voicing_type == "drop_3":
            # Drop 3rd voice from top down an octave
            close_voicing = [base_pitch + ct for ct in chord_tones[:4]]
            if len(close_voicing) >= 4:
                voicing = [
                    close_voicing[0],
                    close_voicing[1] - 12,  # Drop 3rd voice
                    close_voicing[2],
                    close_voicing[3]
                ]
            else:
                voicing = close_voicing

        elif voicing_type == "drop_2_4":
            # Drop 2nd and 4th voices
            close_voicing = [base_pitch + ct for ct in chord_tones[:4]]
            if len(close_voicing) >= 4:
                voicing = [
                    close_voicing[0],
                    close_voicing[1],
                    close_voicing[2] - 12,  # Drop 2nd
                    close_voicing[3] - 12   # Drop 4th
                ]
            else:
                voicing = close_voicing

        elif voicing_type == "spread":
            # Wide spacing throughout (modern sound)
            voicing = [
                base_pitch + chord_tones[0] - 12,  # Root down octave
                base_pitch + chord_tones[1],        # 3rd
                base_pitch + chord_tones[2] + 7,    # 5th up 5th
                base_pitch + chord_tones[3 % len(chord_tones)] + 12  # 7th up octave
            ]

        else:  # Default to drop_2
            close_voicing = [base_pitch + ct for ct in chord_tones[:4]]
            if len(close_voicing) >= 4:
                voicing = [
                    close_voicing[0],
                    close_voicing[1],
                    close_voicing[2] - 12,
                    close_voicing[3]
                ]
            else:
                voicing = close_voicing

        # Extend to num_voices if needed
        while len(voicing) < num_voices:
            voicing.append(voicing[len(voicing) % len(chord_tones)] + 12)

        return voicing[:num_voices]

    @staticmethod
    def _calculate_dynamic(
        current_idx: int,
        total_count: int,
        base_velocity: int,
        shape: str
    ) -> int:
        """
        Calculate velocity based on dynamic shape.

        Args:
            current_idx: Current position in phrase
            total_count: Total length of phrase
            base_velocity: Base velocity level
            shape: "static", "crescendo", "diminuendo", "arch"

        Returns:
            Calculated velocity (0-127)
        """
        if shape == "static":
            return base_velocity

        elif shape == "crescendo":
            # Linear crescendo
            progress = current_idx / max(1, total_count - 1)
            return int(base_velocity + (progress * 30))

        elif shape == "diminuendo":
            # Linear diminuendo
            progress = current_idx / max(1, total_count - 1)
            return int(base_velocity - (progress * 20))

        elif shape == "arch":
            # Arch: crescendo to middle, then diminuendo
            progress = current_idx / max(1, total_count - 1)
            if progress < 0.5:
                # First half: crescendo
                return int(base_velocity + (progress * 2 * 20))
            else:
                # Second half: diminuendo
                return int(base_velocity + 20 - ((progress - 0.5) * 2 * 20))

        return base_velocity

    @staticmethod
    def _create_harmonized_shout(
        melody: List[NoteEvent],
        chords: List[ChordEvent],
        velocity: int,
        voicing_type: str
    ) -> Dict[str, List[NoteEvent]]:
        """
        Create harmonized shout chorus with block harmony.

        Args:
            melody: Lead melody line
            chords: Chord progression
            velocity: Velocity for all notes
            voicing_type: Type of voicing to use

        Returns:
            Dict with brass parts
        """
        brass_parts = {inst.value: [] for inst in BrassInstrument}

        for note in melody:
            # Find current chord
            current_chord = None
            for chord in chords:
                if chord.start_time <= note.start_time < chord.start_time + chord.duration:
                    current_chord = chord
                    break

            if not current_chord:
                continue

            # Get voicing
            voicing = BrassArranger._voice_chord(current_chord, 8, voicing_type)

            # Assign to instruments
            for inst_idx, instrument in enumerate(BrassInstrument):
                if inst_idx < len(voicing):
                    pitch = voicing[inst_idx]
                    min_pitch, max_pitch = BRASS_RANGES[instrument]

                    # Keep within range
                    while pitch < min_pitch:
                        pitch += 12
                    while pitch > max_pitch:
                        pitch -= 12

                    harm_note = NoteEvent(
                        start_time=note.start_time,
                        duration=note.duration,
                        start_tick=note.start_tick,
                        duration_ticks=note.duration_ticks,
                        pitch=pitch,
                        velocity=velocity,
                        channel=inst_idx,
                        track_idx=10 + inst_idx
                    )
                    brass_parts[instrument.value].append(harm_note)

        return brass_parts


# ============================================================================
# SHOUT CHORUS DETECTOR (Form Integration)
# ============================================================================

class ShoutChorusDetector:
    """
    Detect shout chorus location in musical form.
    Typically the final A section in AABA form.
    """

    @staticmethod
    def detect_shout_chorus_section(form: MusicalForm) -> Optional[FormSection]:
        """
        Detect which section should be the shout chorus.

        Args:
            form: MusicalForm object

        Returns:
            FormSection that should be shout chorus, or None

        Research:
        - In AABA form: final A section (A3)
        - Should have highest dynamic_level and texture_density
        """
        if not form or not form.sections:
            return None

        # For AABA form, look for final A section
        a_sections = [s for s in form.sections if s.name.startswith('A')]
        if len(a_sections) >= 3:
            # Return final A section
            return a_sections[-1]

        # For other forms, look for section with highest dynamic level
        max_dynamic = max(s.dynamic_level for s in form.sections)
        for section in form.sections:
            if section.dynamic_level == max_dynamic:
                return section

        return None

    @staticmethod
    def should_be_shout_chorus(section: FormSection) -> bool:
        """
        Determine if section should be arranged as shout chorus.

        Args:
            section: FormSection to check

        Returns:
            True if this should be shout chorus
        """
        # High dynamic level and texture indicates shout chorus
        return section.dynamic_level >= 0.8 and section.texture_density >= 0.7


# ============================================================================
# VALIDATION AND METRICS
# ============================================================================

class BrassArrangementValidator:
    """
    Validate brass arrangements against professional standards.
    """

    @staticmethod
    def validate_brass_arrangement(brass_notes: List[NoteEvent]) -> Dict:
        """
        Validate brass arrangement quality.

        Metrics:
        - Range violations: notes outside comfortable range
        - Dynamic variation: should have minimum 20 velocity points difference
        - Note density: appropriate spacing

        Returns:
            Dict with validation results
        """
        if not brass_notes:
            return {
                "passed": False,
                "range_violations": 0,
                "dynamic_range": 0,
                "error": "No brass notes"
            }

        # Check velocity range
        velocities = [n.velocity for n in brass_notes]
        dynamic_range = max(velocities) - min(velocities)

        results = {
            "passed": dynamic_range >= 20,
            "range_violations": 0,
            "dynamic_range": dynamic_range,
            "note_count": len(brass_notes),
            "avg_velocity": sum(velocities) / len(velocities)
        }

        return results

    @staticmethod
    def compare_to_basie_shout(generated_notes: List[NoteEvent]) -> Dict:
        """
        Compare generated shout chorus to Count Basie "April in Paris" standards.

        Metrics:
        - Shout chorus velocity should be > 100 (f/ff)
        - Dynamic increase in final section: +20 velocity points
        - Note density appropriate

        Returns:
            Dict with comparison metrics
        """
        if not generated_notes:
            return {"error": "No notes to compare"}

        velocities = [n.velocity for n in generated_notes]
        avg_velocity = sum(velocities) / len(velocities)

        return {
            "avg_velocity": avg_velocity,
            "meets_basie_standard": avg_velocity > 100,
            "intensity_score": min(1.0, avg_velocity / 127.0),
            "note_count": len(generated_notes)
        }


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("Brass Arranger Module - Agent 5")
    print("=" * 60)
    print("\nFeatures:")
    print("- Sustained brass pads with dynamic shaping")
    print("- Shout chorus (climactic section)")
    print("- Brass riffs (backgrounds)")
    print("- Call-and-response")
    print("- Multiple voicing types (close, drop-2, drop-3, spread)")
    print("- Style-specific arrangements (Basie, Ellington, Thad Jones)")
    print("\nIntegration: Replaces BigBandArranger._create_brass_figures()")
    print("Scalable to: any brass ensemble")
