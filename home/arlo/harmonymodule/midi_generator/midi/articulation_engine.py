#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Articulation Engine for Realistic MIDI Performance

Handles articulation types for various instruments and converts them to MIDI
parameters (note length, velocity, CC messages, keyswitches, etc.).

Features:
- Comprehensive articulation database
- Note length modulation (staccato, legato, tenuto, etc.)
- Velocity curve adjustments
- MIDI CC automation for expression
- Keyswitch support for sample libraries
- UACC (Universal Articulation Control) support
- Instrument-specific techniques

Articulation Types Supported:
- Common: legato, staccato, staccatissimo, tenuto, marcato, accent
- Strings: arco, pizzicato, col legno, sul ponticello, sul tasto, tremolo, harmonics
- Brass: straight, muted, flutter tongue, fall-off, rip
- Woodwinds: tongued, double tongue, slap tongue, growl

Research References:
- Professional notation practice
- Sample library articulation standards (VSL, Spitfire, etc.)
- MIDI 1.0 and 2.0 specifications
- Instrument technique guides

Author: Claude (Sonnet 4.5)
Created: 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.instrument_library import Instrument, get_instrument


class ArticulationType(Enum):
    """Articulation types (expanded from instrument_library)"""
    # ========== COMMON ARTICULATIONS ==========
    LEGATO = "legato"
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    ACCENT = "accent"
    PORTATO = "portato"
    SFORZANDO = "sforzando"

    # ========== STRING ARTICULATIONS ==========
    ARCO = "arco"
    PIZZICATO = "pizzicato"
    COL_LEGNO = "col_legno"
    SUL_PONTICELLO = "sul_ponticello"
    SUL_TASTO = "sul_tasto"
    TREMOLO = "tremolo"
    HARMONICS = "harmonics"
    SPICCATO = "spiccato"
    RICOCHET = "ricochet"
    BARTOK_PIZZ = "bartok_pizz"

    # ========== BRASS ARTICULATIONS ==========
    STRAIGHT = "straight"
    MUTED = "muted"
    CUP_MUTE = "cup_mute"
    HARMON_MUTE = "harmon_mute"
    STRAIGHT_MUTE = "straight_mute"
    FLUTTER_TONGUE = "flutter_tongue"
    FALL_OFF = "fall_off"
    RIP = "rip"
    GLISSANDO = "glissando"
    DIP = "dip"

    # ========== WOODWIND ARTICULATIONS ==========
    TONGUED = "tongued"
    DOUBLE_TONGUE = "double_tongue"
    TRIPLE_TONGUE = "triple_tongue"
    SLAP_TONGUE = "slap_tongue"
    GROWL = "growl"
    MULTIPHONICS = "multiphonics"


@dataclass
class ArticulationSpec:
    """Specification for an articulation"""
    articulation: ArticulationType
    note_length_multiplier: float  # 1.0 = full length, 0.25 = 25% length
    velocity_offset: int  # Added to base velocity
    velocity_multiplier: float  # Multiplied with velocity
    attack_time_ms: int  # Attack time in milliseconds
    release_time_ms: int  # Release time in milliseconds
    keyswitch_note: Optional[int] = None  # MIDI note for keyswitch
    uacc_value: Optional[int] = None  # UACC CC value
    cc_modulations: Dict[int, int] = None  # CC number -> value
    pitch_bend: Optional[int] = None  # Pitch bend value (0-16383, 8192=center)
    description: str = ""


# ============================================================================
# ARTICULATION DATABASE
# ============================================================================

# Common articulations (work for most instruments)
COMMON_ARTICULATIONS: Dict[ArticulationType, ArticulationSpec] = {
    ArticulationType.LEGATO: ArticulationSpec(
        articulation=ArticulationType.LEGATO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        attack_time_ms=50,
        release_time_ms=50,
        uacc_value=20,
        description="Smooth, connected notes"
    ),

    ArticulationType.STACCATO: ArticulationSpec(
        articulation=ArticulationType.STACCATO,
        note_length_multiplier=0.5,
        velocity_offset=5,
        velocity_multiplier=1.1,
        attack_time_ms=10,
        release_time_ms=10,
        uacc_value=21,
        description="Short, detached notes (50% length)"
    ),

    ArticulationType.STACCATISSIMO: ArticulationSpec(
        articulation=ArticulationType.STACCATISSIMO,
        note_length_multiplier=0.25,
        velocity_offset=10,
        velocity_multiplier=1.2,
        attack_time_ms=5,
        release_time_ms=5,
        uacc_value=22,
        description="Very short, detached notes (25% length)"
    ),

    ArticulationType.TENUTO: ArticulationSpec(
        articulation=ArticulationType.TENUTO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        attack_time_ms=20,
        release_time_ms=20,
        uacc_value=23,
        description="Full length, slightly emphasized"
    ),

    ArticulationType.MARCATO: ArticulationSpec(
        articulation=ArticulationType.MARCATO,
        note_length_multiplier=0.75,
        velocity_offset=15,
        velocity_multiplier=1.3,
        attack_time_ms=10,
        release_time_ms=20,
        uacc_value=24,
        description="Stressed, accented notes"
    ),

    ArticulationType.ACCENT: ArticulationSpec(
        articulation=ArticulationType.ACCENT,
        note_length_multiplier=1.0,
        velocity_offset=20,
        velocity_multiplier=1.2,
        attack_time_ms=15,
        release_time_ms=30,
        uacc_value=25,
        description="Emphasized attack"
    ),

    ArticulationType.PORTATO: ArticulationSpec(
        articulation=ArticulationType.PORTATO,
        note_length_multiplier=0.8,
        velocity_offset=0,
        velocity_multiplier=1.0,
        attack_time_ms=30,
        release_time_ms=20,
        uacc_value=26,
        description="Semi-detached (between legato and staccato)"
    ),

    ArticulationType.SFORZANDO: ArticulationSpec(
        articulation=ArticulationType.SFORZANDO,
        note_length_multiplier=1.0,
        velocity_offset=30,
        velocity_multiplier=1.4,
        attack_time_ms=5,
        release_time_ms=40,
        uacc_value=27,
        description="Sudden, strong accent"
    ),
}

# String-specific articulations
STRING_ARTICULATIONS: Dict[ArticulationType, ArticulationSpec] = {
    ArticulationType.ARCO: ArticulationSpec(
        articulation=ArticulationType.ARCO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        attack_time_ms=50,
        release_time_ms=50,
        keyswitch_note=36,  # C1
        uacc_value=40,
        description="Normal bowing"
    ),

    ArticulationType.PIZZICATO: ArticulationSpec(
        articulation=ArticulationType.PIZZICATO,
        note_length_multiplier=0.3,
        velocity_offset=10,
        velocity_multiplier=1.1,
        attack_time_ms=5,
        release_time_ms=100,
        keyswitch_note=37,  # C#1
        uacc_value=41,
        description="Plucked strings"
    ),

    ArticulationType.COL_LEGNO: ArticulationSpec(
        articulation=ArticulationType.COL_LEGNO,
        note_length_multiplier=0.4,
        velocity_offset=5,
        velocity_multiplier=0.8,
        attack_time_ms=10,
        release_time_ms=50,
        keyswitch_note=38,  # D1
        uacc_value=42,
        cc_modulations={74: 30},  # Filter cutoff (darker)
        description="Bowing with wood of bow"
    ),

    ArticulationType.SUL_PONTICELLO: ArticulationSpec(
        articulation=ArticulationType.SUL_PONTICELLO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.9,
        attack_time_ms=40,
        release_time_ms=40,
        keyswitch_note=39,  # D#1
        uacc_value=43,
        cc_modulations={74: 100},  # Filter cutoff (brighter)
        description="Bowing near bridge (metallic, bright)"
    ),

    ArticulationType.SUL_TASTO: ArticulationSpec(
        articulation=ArticulationType.SUL_TASTO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.8,
        attack_time_ms=60,
        release_time_ms=60,
        keyswitch_note=40,  # E1
        uacc_value=44,
        cc_modulations={74: 20},  # Filter cutoff (darker)
        description="Bowing over fingerboard (soft, mellow)"
    ),

    ArticulationType.TREMOLO: ArticulationSpec(
        articulation=ArticulationType.TREMOLO,
        note_length_multiplier=1.0,
        velocity_offset=5,
        velocity_multiplier=1.0,
        attack_time_ms=30,
        release_time_ms=30,
        keyswitch_note=41,  # F1
        uacc_value=45,
        cc_modulations={1: 64},  # Modulation wheel
        description="Rapid repeated bowing"
    ),

    ArticulationType.HARMONICS: ArticulationSpec(
        articulation=ArticulationType.HARMONICS,
        note_length_multiplier=1.0,
        velocity_offset=-10,
        velocity_multiplier=0.7,
        attack_time_ms=70,
        release_time_ms=100,
        keyswitch_note=42,  # F#1
        uacc_value=46,
        cc_modulations={74: 127},  # Filter cutoff (very bright)
        description="Harmonic overtones (ethereal)"
    ),

    ArticulationType.SPICCATO: ArticulationSpec(
        articulation=ArticulationType.SPICCATO,
        note_length_multiplier=0.4,
        velocity_offset=10,
        velocity_multiplier=1.2,
        attack_time_ms=5,
        release_time_ms=20,
        keyswitch_note=43,  # G1
        uacc_value=47,
        description="Bouncing bow (short, light)"
    ),

    ArticulationType.BARTOK_PIZZ: ArticulationSpec(
        articulation=ArticulationType.BARTOK_PIZZ,
        note_length_multiplier=0.2,
        velocity_offset=20,
        velocity_multiplier=1.3,
        attack_time_ms=2,
        release_time_ms=50,
        keyswitch_note=44,  # G#1
        uacc_value=48,
        description="Snap pizzicato (string rebounds off fingerboard)"
    ),
}

# Brass-specific articulations
BRASS_ARTICULATIONS: Dict[ArticulationType, ArticulationSpec] = {
    ArticulationType.STRAIGHT: ArticulationSpec(
        articulation=ArticulationType.STRAIGHT,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        attack_time_ms=50,
        release_time_ms=50,
        keyswitch_note=36,  # C1
        uacc_value=60,
        description="Normal brass tone"
    ),

    ArticulationType.MUTED: ArticulationSpec(
        articulation=ArticulationType.MUTED,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.7,
        attack_time_ms=40,
        release_time_ms=40,
        keyswitch_note=37,  # C#1
        uacc_value=61,
        cc_modulations={74: 40},  # Filter cutoff (darker)
        description="With mute (generic)"
    ),

    ArticulationType.CUP_MUTE: ArticulationSpec(
        articulation=ArticulationType.CUP_MUTE,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.6,
        attack_time_ms=50,
        release_time_ms=50,
        keyswitch_note=38,  # D1
        uacc_value=62,
        cc_modulations={74: 30},
        description="Cup mute (soft, mellow)"
    ),

    ArticulationType.HARMON_MUTE: ArticulationSpec(
        articulation=ArticulationType.HARMON_MUTE,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.7,
        attack_time_ms=45,
        release_time_ms=45,
        keyswitch_note=39,  # D#1
        uacc_value=63,
        cc_modulations={74: 90},  # Nasal quality
        description="Harmon mute (Miles Davis sound)"
    ),

    ArticulationType.STRAIGHT_MUTE: ArticulationSpec(
        articulation=ArticulationType.STRAIGHT_MUTE,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.8,
        attack_time_ms=40,
        release_time_ms=40,
        keyswitch_note=40,  # E1
        uacc_value=64,
        cc_modulations={74: 70},
        description="Straight mute (bright, pinched)"
    ),

    ArticulationType.FLUTTER_TONGUE: ArticulationSpec(
        articulation=ArticulationType.FLUTTER_TONGUE,
        note_length_multiplier=1.0,
        velocity_offset=5,
        velocity_multiplier=1.1,
        attack_time_ms=40,
        release_time_ms=40,
        keyswitch_note=41,  # F1
        uacc_value=65,
        cc_modulations={1: 80},  # Modulation for flutter
        description="Rolled 'R' sound"
    ),

    ArticulationType.FALL_OFF: ArticulationSpec(
        articulation=ArticulationType.FALL_OFF,
        note_length_multiplier=0.8,
        velocity_offset=10,
        velocity_multiplier=1.2,
        attack_time_ms=30,
        release_time_ms=100,
        keyswitch_note=42,  # F#1
        uacc_value=66,
        pitch_bend=4096,  # Bend down
        description="Pitch falls at end of note"
    ),

    ArticulationType.RIP: ArticulationSpec(
        articulation=ArticulationType.RIP,
        note_length_multiplier=0.6,
        velocity_offset=15,
        velocity_multiplier=1.3,
        attack_time_ms=50,
        release_time_ms=30,
        keyswitch_note=43,  # G1
        uacc_value=67,
        pitch_bend=12288,  # Bend up
        description="Quick upward pitch sweep into note"
    ),

    ArticulationType.DIP: ArticulationSpec(
        articulation=ArticulationType.DIP,
        note_length_multiplier=0.7,
        velocity_offset=5,
        velocity_multiplier=1.1,
        attack_time_ms=40,
        release_time_ms=40,
        keyswitch_note=44,  # G#1
        uacc_value=68,
        pitch_bend=6144,  # Slight bend down
        description="Pitch briefly drops then returns"
    ),
}

# Woodwind-specific articulations
WOODWIND_ARTICULATIONS: Dict[ArticulationType, ArticulationSpec] = {
    ArticulationType.TONGUED: ArticulationSpec(
        articulation=ArticulationType.TONGUED,
        note_length_multiplier=0.9,
        velocity_offset=5,
        velocity_multiplier=1.1,
        attack_time_ms=20,
        release_time_ms=30,
        keyswitch_note=36,  # C1
        uacc_value=80,
        description="Normal tongued articulation"
    ),

    ArticulationType.DOUBLE_TONGUE: ArticulationSpec(
        articulation=ArticulationType.DOUBLE_TONGUE,
        note_length_multiplier=0.6,
        velocity_offset=10,
        velocity_multiplier=1.2,
        attack_time_ms=10,
        release_time_ms=20,
        keyswitch_note=37,  # C#1
        uacc_value=81,
        description="Fast repeated tonguing (ta-ka-ta-ka)"
    ),

    ArticulationType.TRIPLE_TONGUE: ArticulationSpec(
        articulation=ArticulationType.TRIPLE_TONGUE,
        note_length_multiplier=0.5,
        velocity_offset=10,
        velocity_multiplier=1.2,
        attack_time_ms=8,
        release_time_ms=15,
        keyswitch_note=38,  # D1
        uacc_value=82,
        description="Very fast tonguing (ta-ka-ta)"
    ),

    ArticulationType.SLAP_TONGUE: ArticulationSpec(
        articulation=ArticulationType.SLAP_TONGUE,
        note_length_multiplier=0.3,
        velocity_offset=20,
        velocity_multiplier=1.4,
        attack_time_ms=5,
        release_time_ms=50,
        keyswitch_note=39,  # D#1
        uacc_value=83,
        description="Percussive tongue slap (jazz technique)"
    ),

    ArticulationType.GROWL: ArticulationSpec(
        articulation=ArticulationType.GROWL,
        note_length_multiplier=1.0,
        velocity_offset=10,
        velocity_multiplier=1.2,
        attack_time_ms=40,
        release_time_ms=40,
        keyswitch_note=40,  # E1
        uacc_value=84,
        cc_modulations={1: 100},  # Modulation for growl
        description="Singing/humming while playing"
    ),

    ArticulationType.MULTIPHONICS: ArticulationSpec(
        articulation=ArticulationType.MULTIPHONICS,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=0.9,
        attack_time_ms=60,
        release_time_ms=60,
        keyswitch_note=41,  # F1
        uacc_value=85,
        description="Multiple pitches simultaneously"
    ),
}


# ============================================================================
# ARTICULATION ENGINE CLASS
# ============================================================================

class ArticulationEngine:
    """
    Engine for applying articulations to MIDI notes.

    Modulates note properties (length, velocity, timing) based on
    articulation type and instrument characteristics.
    """

    def __init__(self):
        """Initialize articulation engine"""
        self.articulation_db = {
            **COMMON_ARTICULATIONS,
            **STRING_ARTICULATIONS,
            **BRASS_ARTICULATIONS,
            **WOODWIND_ARTICULATIONS
        }

    def apply_articulation(
        self,
        notes: List[int],
        durations: List[float],
        velocities: List[int],
        articulation: ArticulationType,
        instrument: Optional[Instrument] = None
    ) -> Tuple[List[int], List[float], List[int]]:
        """
        Apply articulation to notes.

        Args:
            notes: MIDI note numbers
            durations: Note durations (in beats)
            velocities: MIDI velocities
            articulation: Articulation type to apply
            instrument: Optional instrument for instrument-specific handling

        Returns:
            Tuple of (notes, modified_durations, modified_velocities)
        """
        if articulation not in self.articulation_db:
            print(f"Warning: Unknown articulation {articulation}, using legato")
            articulation = ArticulationType.LEGATO

        spec = self.articulation_db[articulation]

        # Modify durations
        new_durations = [
            dur * spec.note_length_multiplier
            for dur in durations
        ]

        # Modify velocities
        new_velocities = [
            self._adjust_velocity(vel, spec)
            for vel in velocities
        ]

        return notes, new_durations, new_velocities

    def _adjust_velocity(
        self,
        original_velocity: int,
        spec: ArticulationSpec
    ) -> int:
        """Adjust velocity based on articulation spec"""
        adjusted = int(original_velocity * spec.velocity_multiplier) + spec.velocity_offset
        return max(1, min(127, adjusted))  # Clamp to MIDI range

    def get_keyswitch_note(
        self,
        articulation: ArticulationType
    ) -> Optional[int]:
        """
        Get keyswitch note for articulation.

        Used for sample libraries that use keyswitches to change articulation.

        Args:
            articulation: Articulation type

        Returns:
            MIDI note number for keyswitch, or None
        """
        if articulation in self.articulation_db:
            return self.articulation_db[articulation].keyswitch_note
        return None

    def get_uacc_value(
        self,
        articulation: ArticulationType
    ) -> Optional[int]:
        """
        Get UACC (Universal Articulation Control) value.

        UACC uses CC#32 to switch articulations in compatible libraries.

        Args:
            articulation: Articulation type

        Returns:
            UACC value (0-127), or None
        """
        if articulation in self.articulation_db:
            return self.articulation_db[articulation].uacc_value
        return None

    def get_cc_modulations(
        self,
        articulation: ArticulationType
    ) -> Dict[int, int]:
        """
        Get CC modulations for articulation.

        Some articulations require specific CC values (filter, modulation, etc.)

        Args:
            articulation: Articulation type

        Returns:
            Dict of CC number -> value
        """
        if articulation in self.articulation_db:
            spec = self.articulation_db[articulation]
            return spec.cc_modulations if spec.cc_modulations else {}
        return {}

    def suggest_articulation(
        self,
        context: str,
        instrument: Optional[Instrument] = None
    ) -> ArticulationType:
        """
        Suggest appropriate articulation based on musical context.

        Args:
            context: Musical context ("lyrical", "energetic", "soft", "accented", etc.)
            instrument: Optional instrument for specific suggestions

        Returns:
            Recommended ArticulationType
        """
        suggestions = {
            "lyrical": ArticulationType.LEGATO,
            "singing": ArticulationType.LEGATO,
            "smooth": ArticulationType.LEGATO,

            "energetic": ArticulationType.STACCATO,
            "bouncy": ArticulationType.SPICCATO,
            "light": ArticulationType.STACCATO,

            "accented": ArticulationType.MARCATO,
            "emphasized": ArticulationType.ACCENT,
            "powerful": ArticulationType.MARCATO,

            "soft": ArticulationType.TENUTO,
            "gentle": ArticulationType.PORTATO,

            "dramatic": ArticulationType.SFORZANDO,
            "intense": ArticulationType.SFORZANDO,
        }

        return suggestions.get(context.lower(), ArticulationType.LEGATO)

    def create_articulation_sequence(
        self,
        base_articulation: ArticulationType,
        accent_positions: List[int],
        total_notes: int
    ) -> List[ArticulationType]:
        """
        Create a sequence of articulations with accents at specific positions.

        Args:
            base_articulation: Base articulation for most notes
            accent_positions: Indices of notes to accent
            total_notes: Total number of notes

        Returns:
            List of ArticulationType for each note
        """
        sequence = [base_articulation] * total_notes

        for pos in accent_positions:
            if 0 <= pos < total_notes:
                sequence[pos] = ArticulationType.ACCENT

        return sequence


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_expressive_phrase(
    notes: List[int],
    durations: List[float],
    velocities: List[int],
    phrase_type: str = "default"
) -> Tuple[List[int], List[float], List[int]]:
    """
    Add expressive articulation to a phrase.

    Args:
        notes: MIDI notes
        durations: Durations
        velocities: Velocities
        phrase_type: "crescendo", "diminuendo", "arch", "valley"

    Returns:
        Modified (notes, durations, velocities)
    """
    new_velocities = velocities.copy()

    if phrase_type == "crescendo":
        # Gradually increase velocity
        for i in range(len(new_velocities)):
            factor = i / len(new_velocities)
            new_velocities[i] = int(new_velocities[i] * (0.7 + 0.6 * factor))

    elif phrase_type == "diminuendo":
        # Gradually decrease velocity
        for i in range(len(new_velocities)):
            factor = 1 - (i / len(new_velocities))
            new_velocities[i] = int(new_velocities[i] * (0.4 + 0.6 * factor))

    elif phrase_type == "arch":
        # Build to middle, then decrease
        for i in range(len(new_velocities)):
            factor = 1 - abs((i / len(new_velocities)) - 0.5) * 2
            new_velocities[i] = int(new_velocities[i] * (0.7 + 0.5 * factor))

    elif phrase_type == "valley":
        # Start and end loud, soft in middle
        for i in range(len(new_velocities)):
            factor = abs((i / len(new_velocities)) - 0.5) * 2
            new_velocities[i] = int(new_velocities[i] * (0.7 + 0.5 * factor))

    # Clamp to MIDI range
    new_velocities = [max(20, min(127, v)) for v in new_velocities]

    return notes, durations, new_velocities


# ============================================================================
# MAIN (EXAMPLES/TESTS)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ARTICULATION ENGINE - EXAMPLES")
    print("=" * 80)

    engine = ArticulationEngine()

    # Example notes
    notes = [60, 62, 64, 65, 67, 69, 71, 72]
    durations = [1.0] * 8
    velocities = [80] * 8

    print("\n1. Applying different articulations:")

    # Test common articulations
    articulations_to_test = [
        ArticulationType.LEGATO,
        ArticulationType.STACCATO,
        ArticulationType.MARCATO,
        ArticulationType.TENUTO
    ]

    for artic in articulations_to_test:
        _, new_durs, new_vels = engine.apply_articulation(
            notes.copy(), durations.copy(), velocities.copy(), artic
        )
        print(f"\n{artic.value}:")
        print(f"  Duration multiplier: {new_durs[0]:.2f} (original: 1.00)")
        print(f"  Velocity: {new_vels[0]} (original: 80)")

    # Test keyswitches
    print("\n2. Keyswitch notes:")
    ks_articulations = [
        ArticulationType.PIZZICATO,
        ArticulationType.TREMOLO,
        ArticulationType.MUTED
    ]

    for artic in ks_articulations:
        ks = engine.get_keyswitch_note(artic)
        if ks:
            print(f"  {artic.value}: Note {ks} (MIDI)")

    # Test UACC
    print("\n3. UACC values:")
    for artic in articulations_to_test:
        uacc = engine.get_uacc_value(artic)
        print(f"  {artic.value}: CC#32 = {uacc}")

    # Test articulation suggestions
    print("\n4. Articulation suggestions:")
    contexts = ["lyrical", "energetic", "accented", "soft", "dramatic"]
    for context in contexts:
        suggested = engine.suggest_articulation(context)
        print(f"  {context}: {suggested.value}")

    # Test expressive phrase
    print("\n5. Expressive phrase shaping:")
    phrase_types = ["crescendo", "diminuendo", "arch", "valley"]
    for phrase_type in phrase_types:
        _, _, new_vels = create_expressive_phrase(
            notes.copy(), durations.copy(), velocities.copy(), phrase_type
        )
        print(f"  {phrase_type}: {new_vels[:4]}... (first 4 velocities)")

    # Test articulation sequence
    print("\n6. Articulation sequence with accents:")
    sequence = engine.create_articulation_sequence(
        ArticulationType.LEGATO,
        accent_positions=[0, 4, 7],
        total_notes=8
    )
    print("  Sequence:")
    for i, artic in enumerate(sequence):
        print(f"    Note {i}: {artic.value}")

    print("\n" + "=" * 80)
    print("Articulation engine ready!")
    print("=" * 80)
