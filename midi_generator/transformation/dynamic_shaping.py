#!/usr/bin/env python3
"""
Dynamic Shaping & Phrasing Master - Agent 9
=============================================

Add musical phrasing with crescendo, diminuendo, accent patterns, and breath marks
to make arrangements sound human and musical.

This module provides comprehensive dynamic shaping for MIDI generation, transforming
static-velocity arrangements into musically expressive performances.

Features:
---------
- Phrase contour shaping (arch, ascending, descending, peak_early, terrace)
- Crescendo and diminuendo (linear and exponential curves)
- Accent patterns (strong-weak, syncopated, metric)
- Breath marks and phrase boundaries
- Form-based dynamic mapping (intro, verse, chorus, bridge, shout chorus)
- Section-level dynamics with smooth transitions
- MIDI velocity mapping (ppp to fff)

Research Sources:
-----------------
- Classical and jazz phrasing principles
- Big band dynamic conventions (shout chorus climaxes)
- MIDI velocity perception studies
- Professional score analysis (dynamic markings)

Author: Agent 9 - Dynamic Shaping & Phrasing Master
Date: 2025
Part of: 20-Agent Big Band Generator Excellence System
"""

import math
import copy
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# Import data structures from existing modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.midi_analyzer import NoteEvent
from genres.jazz import JazzNote
from generators.form_generator import MusicalForm, FormSection, FormType


# ============================================================================
# VELOCITY MAPPING CONSTANTS
# ============================================================================

class DynamicLevel(Enum):
    """Standard musical dynamic levels with MIDI velocity mappings"""
    PPP = ("ppp", 20, 30)    # Pianississimo - very very soft
    PP = ("pp", 30, 45)       # Pianissimo - very soft
    P = ("p", 45, 60)         # Piano - soft
    MP = ("mp", 60, 75)       # Mezzo-piano - medium soft
    MF = ("mf", 75, 90)       # Mezzo-forte - medium loud
    F = ("f", 90, 105)        # Forte - loud
    FF = ("ff", 105, 115)     # Fortissimo - very loud
    FFF = ("fff", 115, 127)   # Fortississimo - very very loud

    def __init__(self, symbol: str, min_vel: int, max_vel: int):
        self.symbol = symbol
        self.min_velocity = min_vel
        self.max_velocity = max_vel
        self.mid_velocity = (min_vel + max_vel) // 2

    @staticmethod
    def from_float(level: float) -> 'DynamicLevel':
        """Convert 0-1 float to DynamicLevel"""
        if level <= 0.125:
            return DynamicLevel.PPP
        elif level <= 0.25:
            return DynamicLevel.PP
        elif level <= 0.375:
            return DynamicLevel.P
        elif level <= 0.5:
            return DynamicLevel.MP
        elif level <= 0.625:
            return DynamicLevel.MF
        elif level <= 0.75:
            return DynamicLevel.F
        elif level <= 0.875:
            return DynamicLevel.FF
        else:
            return DynamicLevel.FFF

    @staticmethod
    def to_velocity(level: float) -> int:
        """Convert 0-1 float directly to MIDI velocity (linear mapping)"""
        # Map 0.0-1.0 to 20-127 (avoid velocities below 20 for realism)
        return int(20 + (level * 107))


# ============================================================================
# PHRASE CONTOUR TYPES
# ============================================================================

class PhraseContour(Enum):
    """Musical phrase shape types"""
    ARCH = "arch"                     # Start med, crescendo to mid, diminuendo to end
    ASCENDING = "ascending"           # Gradual build throughout
    DESCENDING = "descending"         # Gradual decay throughout
    PEAK_EARLY = "peak_early"         # Peak at 1/4, then decay
    PEAK_LATE = "peak_late"           # Build to 3/4, then release
    TERRACE = "terrace"               # Sudden dynamic shifts (Baroque)
    WAVE = "wave"                     # Multiple peaks and valleys
    FLAT = "flat"                     # Even dynamics throughout


# ============================================================================
# ACCENT PATTERN TYPES
# ============================================================================

class AccentPattern(Enum):
    """Rhythmic accent patterns"""
    STRONG_WEAK = "strong_weak"       # Beats 1,3 louder than 2,4
    SYNCOPATED = "syncopated"         # Offbeats accented
    DOWNBEAT = "downbeat"             # Strong downbeat only
    EVEN = "even"                     # No accents
    CUMULATIVE = "cumulative"         # Each beat louder than previous
    ALTERNATING = "alternating"       # Strong-weak-strong-weak on all beats


# ============================================================================
# MAIN DYNAMIC SHAPING ENGINE
# ============================================================================

class DynamicShaping:
    """
    Main dynamic shaping engine for musical phrasing

    This class provides methods to apply dynamic contours, accents, and
    phrasing to lists of NoteEvent or JazzNote objects.
    """

    @staticmethod
    def apply_phrase_contour(
        notes: List[Union[NoteEvent, JazzNote]],
        phrase_length_bars: int = 4,
        contour: PhraseContour = PhraseContour.ARCH,
        base_velocity: int = 75,
        variation_range: int = 25
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Apply dynamic contour to phrase

        Args:
            notes: List of notes to shape
            phrase_length_bars: Length of phrase in bars (for calculating positions)
            contour: Type of phrase contour to apply
            base_velocity: Base velocity level
            variation_range: How much velocity can vary (+/-)

        Returns:
            List of notes with modified velocities
        """
        if not notes:
            return notes

        shaped_notes = []
        total_duration = notes[-1].start_time - notes[0].start_time + notes[-1].duration

        for i, note in enumerate(notes):
            new_note = copy.copy(note)

            # Calculate position in phrase (0.0 to 1.0)
            if total_duration > 0:
                position = (note.start_time - notes[0].start_time) / total_duration
            else:
                position = 0.0

            # Calculate velocity multiplier based on contour
            multiplier = DynamicShaping._get_contour_multiplier(position, contour)

            # Apply multiplier to base velocity with variation range
            velocity_offset = int(multiplier * variation_range)
            new_velocity = base_velocity + velocity_offset

            # Clamp to MIDI range
            new_note.velocity = max(1, min(127, new_velocity))

            shaped_notes.append(new_note)

        return shaped_notes

    @staticmethod
    def _get_contour_multiplier(position: float, contour: PhraseContour) -> float:
        """
        Get velocity multiplier (-1.0 to 1.0) for given position and contour

        Args:
            position: Position in phrase (0.0 to 1.0)
            contour: Type of contour

        Returns:
            Multiplier value
        """
        if contour == PhraseContour.ARCH:
            # Arch: starts at 0, peaks at 0.5, ends at 0
            # Use sine wave: sin(π * position)
            return math.sin(math.pi * position)

        elif contour == PhraseContour.ASCENDING:
            # Linear crescendo
            return position * 2 - 1  # -1 to 1

        elif contour == PhraseContour.DESCENDING:
            # Linear diminuendo
            return 1 - position * 2  # 1 to -1

        elif contour == PhraseContour.PEAK_EARLY:
            # Peak at 1/4 (position 0.25)
            if position < 0.25:
                return position * 4  # 0 to 1
            else:
                return 1 - ((position - 0.25) / 0.75)  # 1 to 0

        elif contour == PhraseContour.PEAK_LATE:
            # Peak at 3/4 (position 0.75)
            if position < 0.75:
                return (position / 0.75)  # 0 to 1
            else:
                return 1 - ((position - 0.75) / 0.25)  # 1 to 0

        elif contour == PhraseContour.TERRACE:
            # Sudden shifts (Baroque terrace dynamics)
            if position < 0.25:
                return -0.5  # Soft
            elif position < 0.5:
                return 0.5   # Loud
            elif position < 0.75:
                return -0.5  # Soft
            else:
                return 0.5   # Loud

        elif contour == PhraseContour.WAVE:
            # Multiple peaks (2 waves)
            return math.sin(2 * math.pi * position)

        elif contour == PhraseContour.FLAT:
            # No contour
            return 0.0

        else:
            return 0.0

    @staticmethod
    def apply_crescendo(
        notes: List[Union[NoteEvent, JazzNote]],
        start_velocity: int = 60,
        end_velocity: int = 100,
        curve: str = "linear"
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Apply crescendo (gradual increase in volume)

        Args:
            notes: List of notes
            start_velocity: Starting velocity
            end_velocity: Ending velocity
            curve: Type of curve ("linear", "exponential", "logarithmic")

        Returns:
            List of notes with crescendo applied
        """
        if not notes:
            return notes

        shaped_notes = []
        num_notes = len(notes)

        for i, note in enumerate(notes):
            new_note = copy.copy(note)

            # Calculate position (0.0 to 1.0)
            position = i / max(1, num_notes - 1)

            # Apply curve
            if curve == "exponential":
                # Exponential: starts slow, accelerates
                position = position ** 2
            elif curve == "logarithmic":
                # Logarithmic: starts fast, slows down
                position = math.sqrt(position)
            # else: linear (no transformation)

            # Interpolate velocity
            new_velocity = int(start_velocity + (end_velocity - start_velocity) * position)
            new_note.velocity = max(1, min(127, new_velocity))

            shaped_notes.append(new_note)

        return shaped_notes

    @staticmethod
    def apply_diminuendo(
        notes: List[Union[NoteEvent, JazzNote]],
        start_velocity: int = 100,
        end_velocity: int = 60,
        curve: str = "linear"
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Apply diminuendo (gradual decrease in volume)

        This is just a crescendo in reverse
        """
        return DynamicShaping.apply_crescendo(notes, start_velocity, end_velocity, curve)

    @staticmethod
    def apply_accent_pattern(
        notes: List[Union[NoteEvent, JazzNote]],
        pattern: AccentPattern = AccentPattern.STRONG_WEAK,
        accent_amount: int = 15,
        beats_per_bar: int = 4
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Apply accent pattern based on metric position

        Args:
            notes: List of notes
            pattern: Type of accent pattern
            accent_amount: Velocity increase for accented notes
            beats_per_bar: Time signature numerator

        Returns:
            List of notes with accents applied
        """
        if not notes:
            return notes

        shaped_notes = []

        for note in notes:
            new_note = copy.copy(note)

            # Determine beat position (which beat of the bar)
            beat_position = note.start_time % beats_per_bar

            # Check if this note should be accented
            should_accent = False

            if pattern == AccentPattern.STRONG_WEAK:
                # Beats 1 and 3 are strong (in 4/4)
                should_accent = (int(beat_position) % 2 == 0)

            elif pattern == AccentPattern.SYNCOPATED:
                # Offbeats (0.5, 1.5, 2.5, 3.5) are accented
                fractional_part = beat_position - int(beat_position)
                should_accent = (0.4 < fractional_part < 0.6)

            elif pattern == AccentPattern.DOWNBEAT:
                # Only beat 1
                should_accent = (int(beat_position) == 0)

            elif pattern == AccentPattern.EVEN:
                # No accents
                should_accent = False

            elif pattern == AccentPattern.CUMULATIVE:
                # Each beat louder
                beat_num = int(beat_position)
                new_note.velocity = min(127, new_note.velocity + (beat_num * accent_amount // beats_per_bar))

            elif pattern == AccentPattern.ALTERNATING:
                # Every other note
                # Approximate by checking if near beat or offbeat
                should_accent = (int(beat_position * 2) % 2 == 0)

            # Apply accent if needed (for non-cumulative patterns)
            if should_accent and pattern != AccentPattern.CUMULATIVE:
                new_note.velocity = min(127, new_note.velocity + accent_amount)

            shaped_notes.append(new_note)

        return shaped_notes

    @staticmethod
    def mark_breath_points(
        notes: List[Union[NoteEvent, JazzNote]],
        phrase_length_bars: int = 4,
        breath_gap: float = 0.15,
        beats_per_bar: int = 4
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Add gaps for breath marks at phrase boundaries

        Args:
            notes: List of notes
            phrase_length_bars: Length of phrases in bars
            breath_gap: Gap to insert in beats
            beats_per_bar: Time signature numerator

        Returns:
            List of notes with shortened durations at phrase endings
        """
        if not notes:
            return notes

        shaped_notes = []
        phrase_length_beats = phrase_length_bars * beats_per_bar

        for i, note in enumerate(notes):
            new_note = copy.copy(note)

            # Check if this note is near a phrase boundary
            # Phrase boundaries occur at multiples of phrase_length_beats
            note_end_time = note.start_time + note.duration
            next_phrase_boundary = math.ceil(note_end_time / phrase_length_beats) * phrase_length_beats

            # If note ends within 0.5 beats of phrase boundary, shorten it
            if abs(note_end_time - next_phrase_boundary) < 0.5:
                # Shorten duration to create breath gap
                new_note.duration = max(0.1, note.duration - breath_gap)

            shaped_notes.append(new_note)

        return shaped_notes

    @staticmethod
    def apply_swell(
        notes: List[Union[NoteEvent, JazzNote]],
        swell_duration_beats: float = 2.0
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Apply swell (crescendo then diminuendo) to long notes

        Args:
            notes: List of notes
            swell_duration_beats: Minimum note duration to apply swell

        Returns:
            List of notes (note: true swell requires multiple MIDI messages,
                    this is a simplified version that averages the velocity)
        """
        shaped_notes = []

        for note in notes:
            new_note = copy.copy(note)

            # Apply slight boost to long notes to simulate swell
            if note.duration >= swell_duration_beats:
                # Increase velocity slightly for sustained notes
                new_note.velocity = min(127, int(note.velocity * 1.1))

            shaped_notes.append(new_note)

        return shaped_notes


# ============================================================================
# FORM-BASED DYNAMIC MAPPING
# ============================================================================

def generate_dynamic_map_for_form(form: MusicalForm) -> Dict[str, float]:
    """
    Generate dynamic levels for each section of a musical form

    Args:
        form: MusicalForm object

    Returns:
        Dictionary mapping section names to dynamic levels (0.0 to 1.0)
    """
    dynamic_map = {}

    # Use the dynamic_level from each FormSection if available
    for section in form.sections:
        dynamic_map[section.name] = section.dynamic_level

    # Apply form-specific overrides for common patterns
    if form.form_type == FormType.AABA:
        # Classic AABA dynamics
        for section in form.sections:
            if "A1" in section.name:
                dynamic_map[section.name] = 0.65  # mf - establishing
            elif "A2" in section.name:
                dynamic_map[section.name] = 0.70  # Slightly louder
            elif "B" in section.name or "Bridge" in section.name:
                dynamic_map[section.name] = 0.60  # mp - contrast
            elif "A3" in section.name or "Return" in section.name:
                dynamic_map[section.name] = 0.85  # f - shout chorus!

    elif form.form_type == FormType.VERSE_CHORUS:
        # Pop song dynamics
        for section in form.sections:
            if "Intro" in section.name:
                dynamic_map[section.name] = 0.50  # mp
            elif "Verse" in section.name:
                verse_num = 1
                if "2" in section.name:
                    verse_num = 2
                dynamic_map[section.name] = 0.55 + (verse_num - 1) * 0.05
            elif "Pre-Chorus" in section.name:
                dynamic_map[section.name] = 0.65  # Building
            elif "Chorus" in section.name:
                if "Final" in section.name or "Repeat" in section.name:
                    dynamic_map[section.name] = 0.90  # fff - climax
                else:
                    dynamic_map[section.name] = 0.80  # f
            elif "Bridge" in section.name:
                dynamic_map[section.name] = 0.60  # Contrast
            elif "Outro" in section.name:
                dynamic_map[section.name] = 0.45  # Fade

    elif form.form_type == FormType.SONATA:
        # Sonata form dynamics
        for section in form.sections:
            if "Introduction" in section.name:
                dynamic_map[section.name] = 0.40  # p - mysterious
            elif "Exposition - First Theme" in section.name:
                dynamic_map[section.name] = 0.75  # f - confident
            elif "Exposition - Transition" in section.name:
                dynamic_map[section.name] = 0.70
            elif "Exposition - Second Theme" in section.name:
                dynamic_map[section.name] = 0.60  # mf - lyrical
            elif "Development" in section.name:
                # Development builds intensity
                phase_num = 0
                if "Phase" in section.name:
                    try:
                        phase_num = int(section.name.split("Phase")[1].strip()[0])
                    except:
                        phase_num = 0
                dynamic_map[section.name] = 0.60 + (phase_num * 0.05)
            elif "Recapitulation - First Theme" in section.name:
                dynamic_map[section.name] = 0.85  # ff - triumphant return
            elif "Recapitulation - Second Theme" in section.name:
                dynamic_map[section.name] = 0.70  # f
            elif "Coda" in section.name:
                dynamic_map[section.name] = 0.95  # fff - grand finale

    return dynamic_map


def apply_dynamics_to_section(
    notes: List[Union[NoteEvent, JazzNote]],
    section: FormSection,
    form: MusicalForm
) -> List[Union[NoteEvent, JazzNote]]:
    """
    Apply appropriate dynamics to notes based on their section

    Args:
        notes: List of notes in this section
        section: FormSection object
        form: MusicalForm object (for context)

    Returns:
        List of notes with dynamics applied
    """
    if not notes:
        return notes

    # Get base velocity from section's dynamic_level
    base_velocity = DynamicLevel.to_velocity(section.dynamic_level)

    # Determine appropriate phrase contour based on section character
    contour = PhraseContour.ARCH  # Default

    if "intro" in section.character.lower() or "mysterious" in section.character.lower():
        contour = PhraseContour.ASCENDING
    elif "climactic" in section.character.lower() or "shout" in section.character.lower():
        contour = PhraseContour.PEAK_EARLY
    elif "ending" in section.character.lower() or "fade" in section.character.lower():
        contour = PhraseContour.DESCENDING
    elif "build" in section.character.lower():
        contour = PhraseContour.ASCENDING

    # Apply phrase contour
    shaped_notes = DynamicShaping.apply_phrase_contour(
        notes,
        phrase_length_bars=section.length_bars,
        contour=contour,
        base_velocity=base_velocity,
        variation_range=20
    )

    # Apply accents based on style
    if form.form_type in [FormType.AABA, FormType.TWELVE_BAR_BLUES]:
        # Jazz styles - use swing accents
        shaped_notes = DynamicShaping.apply_accent_pattern(
            shaped_notes,
            pattern=AccentPattern.SYNCOPATED,
            accent_amount=10
        )
    else:
        # Classical styles - use metric accents
        shaped_notes = DynamicShaping.apply_accent_pattern(
            shaped_notes,
            pattern=AccentPattern.STRONG_WEAK,
            accent_amount=12
        )

    # Add breath marks at phrase boundaries
    shaped_notes = DynamicShaping.mark_breath_points(
        shaped_notes,
        phrase_length_bars=min(4, section.length_bars // 2),
        breath_gap=0.1
    )

    return shaped_notes


# ============================================================================
# BIG BAND SPECIFIC DYNAMICS
# ============================================================================

class BigBandDynamics:
    """
    Big band specific dynamic shaping

    Implements conventions from Duke Ellington, Count Basie, Thad Jones
    """

    @staticmethod
    def apply_shout_chorus_dynamics(
        notes: List[Union[NoteEvent, JazzNote]],
        intensity: float = 0.9
    ) -> List[Union[NoteEvent, JazzNote]]:
        """
        Apply shout chorus dynamics (climactic final A section)

        Args:
            notes: Notes in shout chorus section
            intensity: Intensity level (0.0 to 1.0)

        Returns:
            Notes with shout chorus dynamics
        """
        # Shout chorus characteristics:
        # - Very loud (ff to fff)
        # - Strong accents on downbeats
        # - Building energy

        # Base velocity from intensity
        base_velocity = int(90 + (intensity * 35))  # 90-125 range

        # Apply ascending contour (build throughout)
        shaped_notes = DynamicShaping.apply_crescendo(
            notes,
            start_velocity=base_velocity,
            end_velocity=min(127, base_velocity + 15),
            curve="exponential"
        )

        # Strong downbeat accents
        shaped_notes = DynamicShaping.apply_accent_pattern(
            shaped_notes,
            pattern=AccentPattern.DOWNBEAT,
            accent_amount=20
        )

        return shaped_notes

    @staticmethod
    def apply_section_balance(
        arrangement: Dict[str, List[Union[NoteEvent, JazzNote]]],
        lead_boost: int = 10,
        brass_power: int = 5,
        sax_blend: int = 0,
        rhythm_reduction: int = -5
    ) -> Dict[str, List[Union[NoteEvent, JazzNote]]]:
        """
        Apply proper section balance to big band arrangement

        Args:
            arrangement: Dictionary of instrument sections to notes
            lead_boost: Velocity boost for lead melody
            brass_power: Velocity boost for brass
            sax_blend: Velocity adjustment for saxes
            rhythm_reduction: Velocity reduction for rhythm section

        Returns:
            Balanced arrangement
        """
        balanced = {}

        for section_name, notes in arrangement.items():
            adjustment = 0

            if "lead" in section_name.lower():
                adjustment = lead_boost
            elif "brass" in section_name.lower() or "trumpet" in section_name.lower() or "trombone" in section_name.lower():
                adjustment = brass_power
            elif "sax" in section_name.lower():
                adjustment = sax_blend
            elif any(word in section_name.lower() for word in ["piano", "bass", "drums", "guitar"]):
                adjustment = rhythm_reduction

            # Apply adjustment
            adjusted_notes = []
            for note in notes:
                new_note = copy.copy(note)
                new_note.velocity = max(1, min(127, note.velocity + adjustment))
                adjusted_notes.append(new_note)

            balanced[section_name] = adjusted_notes

        return balanced


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def velocity_to_dynamic_marking(velocity: int) -> str:
    """Convert MIDI velocity to dynamic marking string"""
    if velocity < 30:
        return "ppp"
    elif velocity < 45:
        return "pp"
    elif velocity < 60:
        return "p"
    elif velocity < 75:
        return "mp"
    elif velocity < 90:
        return "mf"
    elif velocity < 105:
        return "f"
    elif velocity < 115:
        return "ff"
    else:
        return "fff"


def analyze_dynamic_range(notes: List[Union[NoteEvent, JazzNote]]) -> Dict[str, any]:
    """
    Analyze the dynamic range of a list of notes

    Args:
        notes: List of notes to analyze

    Returns:
        Dictionary with dynamic statistics
    """
    if not notes:
        return {
            "min_velocity": 0,
            "max_velocity": 0,
            "avg_velocity": 0,
            "dynamic_range": 0,
            "has_variation": False
        }

    velocities = [note.velocity for note in notes]
    min_vel = min(velocities)
    max_vel = max(velocities)
    avg_vel = sum(velocities) / len(velocities)

    return {
        "min_velocity": min_vel,
        "max_velocity": max_vel,
        "avg_velocity": avg_vel,
        "dynamic_range": max_vel - min_vel,
        "has_variation": (max_vel - min_vel) > 10,
        "min_marking": velocity_to_dynamic_marking(min_vel),
        "max_marking": velocity_to_dynamic_marking(max_vel),
        "avg_marking": velocity_to_dynamic_marking(int(avg_vel))
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🎵 DYNAMIC SHAPING & PHRASING ENGINE - Agent 9\n")
    print("=" * 80)

    # Example 1: Create test notes
    print("\nExample 1: Applying phrase contours to test notes\n")

    # Create 16 test notes (4 bars of quarter notes)
    test_notes = []
    for i in range(16):
        note = NoteEvent(
            start_time=float(i),
            duration=0.9,
            start_tick=i * 480,
            duration_ticks=int(0.9 * 480),
            pitch=60 + (i % 8),
            velocity=75,  # Static velocity
            channel=0,
            track_idx=0
        )
        test_notes.append(note)

    print(f"Original notes: all velocity = 75")

    # Apply different contours
    contours_to_test = [
        PhraseContour.ARCH,
        PhraseContour.ASCENDING,
        PhraseContour.DESCENDING,
        PhraseContour.PEAK_EARLY
    ]

    for contour in contours_to_test:
        shaped = DynamicShaping.apply_phrase_contour(
            test_notes,
            phrase_length_bars=4,
            contour=contour,
            base_velocity=75,
            variation_range=25
        )

        velocities = [n.velocity for n in shaped]
        print(f"\n{contour.value.upper()}:")
        print(f"  Velocities: {velocities}")
        print(f"  Range: {min(velocities)} - {max(velocities)}")
        stats = analyze_dynamic_range(shaped)
        print(f"  Dynamics: {stats['min_marking']} to {stats['max_marking']}")

    # Example 2: Crescendo and Diminuendo
    print("\n" + "=" * 80)
    print("\nExample 2: Crescendo and Diminuendo\n")

    cresc = DynamicShaping.apply_crescendo(test_notes, 50, 110, "exponential")
    dim = DynamicShaping.apply_diminuendo(test_notes, 110, 50, "linear")

    print(f"Crescendo (50→110, exponential): {[n.velocity for n in cresc]}")
    print(f"Diminuendo (110→50, linear): {[n.velocity for n in dim]}")

    # Example 3: Accent Patterns
    print("\n" + "=" * 80)
    print("\nExample 3: Accent Patterns\n")

    patterns_to_test = [
        AccentPattern.STRONG_WEAK,
        AccentPattern.SYNCOPATED,
        AccentPattern.DOWNBEAT
    ]

    for pattern in patterns_to_test:
        accented = DynamicShaping.apply_accent_pattern(
            test_notes,
            pattern=pattern,
            accent_amount=20
        )
        velocities = [n.velocity for n in accented]
        print(f"{pattern.value.upper()}: {velocities}")

    # Example 4: Form-based dynamics
    print("\n" + "=" * 80)
    print("\nExample 4: Form-Based Dynamic Mapping\n")

    # Import and create AABA form
    from generators.form_generator import FormGenerator, FormType

    aaba = FormGenerator.generate_form(
        FormType.AABA,
        tonic_key=60,
        is_major=True,
        tempo=140
    )

    dynamic_map = generate_dynamic_map_for_form(aaba)

    print("AABA Form Dynamic Map:")
    for section_name, level in dynamic_map.items():
        velocity = DynamicLevel.to_velocity(level)
        marking = velocity_to_dynamic_marking(velocity)
        print(f"  {section_name:30} -> {level:.2f} ({marking}, vel={velocity})")

    print("\n" + "=" * 80)
    print("\n✅ Dynamic Shaping Engine Examples Complete!")
    print("\nKey Features Demonstrated:")
    print("  - Phrase contours (arch, ascending, descending, peak_early)")
    print("  - Crescendo/diminuendo with different curves")
    print("  - Accent patterns (strong-weak, syncopated, downbeat)")
    print("  - Form-based dynamic mapping for AABA")
    print("\nThis module makes arrangements sound human and musical!")
    print("=" * 80 + "\n")
