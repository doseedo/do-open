"""
Big Band Drum Pattern Library

Authentic big band drum patterns across styles with fills, dynamic variation,
and integration with groove template system.

Covers:
- Swing (Count Basie, Duke Ellington style)
- Bebop (Buddy Rich, Max Roach style)
- Latin Jazz (Afro-Cuban, Bossa Nova, Samba)
- Fills (2-bar, 4-bar, 8-bar phrase endings)
- Dynamic variation (intro, verse, bridge, shout chorus, ending)

Research References:
- Buddy Rich "West Side Story" - bebop drum approach
- Louie Bellson with Duke Ellington - classic swing
- Mel Lewis with Thad Jones - modern jazz orchestral
- "Manteca" (Dizzy Gillespie) - Afro-Cuban patterns
- "Samba de Uma Nota So" - Brazilian samba

Author: Agent 7 - Drum Pattern & Groove Specialist
Date: 2025
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import random

# Import from existing modules
from ..midi.midi_constants import GM_DRUM_MAP, PPQN_HIGH_RES
from ..algorithms.rhythm_engine import RhythmNote


# ============================================================================
# Big Band Drum Pattern Library
# ============================================================================

class BigBandDrumPatterns:
    """
    Library of authentic big band drum patterns.

    Each pattern includes:
    - Ride cymbal patterns (swing 8ths, bebop syncopation)
    - Hi-hat patterns (backbeat 2&4, all beats)
    - Bass drum patterns (feathered kick, bebop bombs)
    - Latin patterns (cowbell, surdo, bossa ride)
    - Fills (snare rolls, tom patterns)
    """

    # ========================================================================
    # SWING PATTERNS
    # ========================================================================

    @staticmethod
    def swing_ride(ppqn: int = PPQN_HIGH_RES,
                   bars: int = 4,
                   swing_ratio: float = 0.62) -> List[RhythmNote]:
        """
        Classic big band swing ride cymbal pattern.

        Characteristics:
        - Swing 8th notes on ride cymbal
        - Accent on beats 2 and 4
        - Swing ratio: 0.58-0.67 (0.62 is medium)

        Used by: Mel Lewis, Louie Bellson, Buddy Rich (medium tempo)

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars to generate
            swing_ratio: Swing amount (0.5=straight, 0.62=medium, 0.67=triplet)

        Returns:
            List of RhythmNote for ride cymbal
        """
        sixteenth = ppqn // 4
        bar_length = ppqn * 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # Swing 8th notes
            for i in range(8):  # 8th note grid
                tick = bar_offset + i * (ppqn // 2)

                # Apply swing to offbeats
                if i % 2 == 1:
                    swing_offset = int((swing_ratio - 0.5) * (ppqn // 2))
                    tick += swing_offset

                # Accent beats 2 and 4 (i=2, i=6)
                if i in [2, 6]:
                    vel = 85
                else:
                    vel = 70 if i % 2 == 0 else 60

                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth * 2,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['RIDE_CYMBAL_1']
                ))

        return pattern

    @staticmethod
    def bebop_ride(ppqn: int = PPQN_HIGH_RES,
                   bars: int = 4,
                   swing_ratio: float = 0.62) -> List[RhythmNote]:
        """
        Bebop ride cymbal pattern with more syncopation.

        Characteristics:
        - More complex than straight swing
        - Occasional skipped beats
        - Varied accents

        Used by: Max Roach, Art Blakey, Elvin Jones

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars
            swing_ratio: Swing amount

        Returns:
            List of RhythmNote for ride cymbal
        """
        sixteenth = ppqn // 4
        bar_length = ppqn * 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # Swing 8th notes with occasional variations
            for i in range(8):
                tick = bar_offset + i * (ppqn // 2)

                # Apply swing to offbeats
                if i % 2 == 1:
                    swing_offset = int((swing_ratio - 0.5) * (ppqn // 2))
                    tick += swing_offset

                # Occasionally skip offbeats for bebop feel (10% chance)
                if i % 2 == 1 and random.random() < 0.1:
                    continue

                # More varied accents
                if i in [2, 6]:
                    vel = 85
                elif i == 0:
                    vel = 75
                else:
                    vel = 65 if i % 2 == 0 else 58

                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth * 2,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['RIDE_CYMBAL_1']
                ))

        return pattern

    # ========================================================================
    # HI-HAT PATTERNS
    # ========================================================================

    @staticmethod
    def hihat_2_4(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Classic backbeat hi-hat on beats 2 and 4.

        Characteristics:
        - Hi-hat chick on 2 & 4
        - Strong, consistent
        - Count Basie style

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for hi-hat
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # Beats 2 and 4
            for beat in [1, 3]:
                pattern.append(RhythmNote(
                    tick=bar_offset + beat * ppqn,
                    duration=sixteenth,
                    velocity=95,
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
                ))

        return pattern

    @staticmethod
    def hihat_all_beats(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Hi-hat on all four beats (modern big band).

        Characteristics:
        - Four-on-the-floor hi-hat
        - More modern sound

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for hi-hat
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # All four beats
            for beat in range(4):
                vel = 95 if beat in [1, 3] else 85  # Accent 2 & 4
                pattern.append(RhythmNote(
                    tick=bar_offset + beat * ppqn,
                    duration=sixteenth,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
                ))

        return pattern

    # ========================================================================
    # BASS DRUM PATTERNS
    # ========================================================================

    @staticmethod
    def feathered_kick(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Feathered bass drum - all four beats, soft (Count Basie style).

        Characteristics:
        - Four on the floor
        - Very soft (velocity 40-50)
        - Constant pulse underneath

        Used by: Count Basie band, traditional swing

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for kick drum
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # All four beats, very soft
            for beat in range(4):
                pattern.append(RhythmNote(
                    tick=bar_offset + beat * ppqn,
                    duration=ppqn // 2,
                    velocity=45,  # Very soft, "feathered"
                    pitch=GM_DRUM_MAP['BASS_DRUM_1']
                ))

        return pattern

    @staticmethod
    def bebop_kick(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Bebop "bombs" - syncopated kick accents.

        Characteristics:
        - Syncopated accents ("bombs")
        - Not constant pulse
        - Louder than feathered kick

        Used by: Max Roach, Kenny Clarke

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for kick drum
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # Syncopated bombs - random placement
            # Typically on: beat 1, "and of 2", "and of 4"
            kick_positions = [0]  # Always start on 1

            # Add occasional syncopation
            if random.random() < 0.7:
                kick_positions.append(sixteenth * 6)  # "and of 2"
            if random.random() < 0.5:
                kick_positions.append(sixteenth * 14)  # "and of 4"
            if random.random() < 0.3:
                kick_positions.append(sixteenth * 10)  # "and of 3"

            for pos in kick_positions:
                pattern.append(RhythmNote(
                    tick=bar_offset + pos,
                    duration=sixteenth * 2,
                    velocity=random.randint(85, 100),  # Varied velocity
                    pitch=GM_DRUM_MAP['BASS_DRUM_1']
                ))

        return pattern

    # ========================================================================
    # LATIN PATTERNS
    # ========================================================================

    @staticmethod
    def afro_cuban_bell(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Afro-Cuban cowbell pattern (Manteca style).

        Characteristics:
        - Constant 8th note cowbell
        - Driving, insistent pattern

        Used in: "Manteca" (Dizzy Gillespie), Latin jazz

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for cowbell
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        for bar in range(bars):
            bar_offset = bar * bar_length

            # 8th note cowbell pattern
            for i in range(8):
                vel = 80 if i % 2 == 0 else 70  # Accent downbeats
                pattern.append(RhythmNote(
                    tick=bar_offset + i * (ppqn // 2),
                    duration=sixteenth,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['COWBELL']
                ))

        return pattern

    @staticmethod
    def samba_surdo(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Samba bass drum (surdo) pattern.

        Characteristics:
        - Low drum on 1, "and of 2", "and of 3"
        - Syncopated Brazilian feel

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for bass drum
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        # Samba surdo pattern: beats at specific 16th positions
        surdo_positions = [0, sixteenth * 6, sixteenth * 10]  # 1, "and of 2", "and of 3"

        for bar in range(bars):
            bar_offset = bar * bar_length

            for pos in surdo_positions:
                pattern.append(RhythmNote(
                    tick=bar_offset + pos,
                    duration=sixteenth * 2,
                    velocity=90,
                    pitch=GM_DRUM_MAP['BASS_DRUM_1']
                ))

        return pattern

    @staticmethod
    def bossa_ride(ppqn: int = PPQN_HIGH_RES, bars: int = 4) -> List[RhythmNote]:
        """
        Bossa nova ride cymbal pattern.

        Characteristics:
        - Subtle, syncopated
        - Light touch
        - Often with brushes

        Args:
            ppqn: Pulses per quarter note
            bars: Number of bars

        Returns:
            List of RhythmNote for ride cymbal
        """
        bar_length = ppqn * 4
        sixteenth = ppqn // 4
        pattern = []

        # Bossa ride: 16th positions 0, 4, 8, 10, 12
        bossa_positions = [0, 4, 8, 10, 12]  # In 16th notes

        for bar in range(bars):
            bar_offset = bar * bar_length

            for pos in bossa_positions:
                vel = 60 if pos in [0, 8] else 50  # Subtle accents
                pattern.append(RhythmNote(
                    tick=bar_offset + pos * sixteenth,
                    duration=sixteenth,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['RIDE_CYMBAL_1']
                ))

        return pattern

    # ========================================================================
    # FILLS
    # ========================================================================

    @staticmethod
    def generate_fill(ppqn: int = PPQN_HIGH_RES,
                     length_beats: int = 2,
                     intensity: float = 0.7,
                     target_instrument: str = "snare") -> List[RhythmNote]:
        """
        Generate drum fill at phrase ending.

        Characteristics:
        - Snare rolls, tom patterns
        - Builds tension before next section
        - Length: 1-4 beats typical

        Args:
            ppqn: Pulses per quarter note
            length_beats: Length of fill in beats (1, 2, or 4)
            intensity: Fill intensity (0.0-1.0)
            target_instrument: "snare", "toms", or "mixed"

        Returns:
            List of RhythmNote for fill
        """
        sixteenth = ppqn // 4
        pattern = []

        if target_instrument == "snare":
            # Snare roll fill
            num_hits = length_beats * 4  # 16th notes

            for i in range(num_hits):
                tick = i * sixteenth
                # Crescendo to end of fill
                vel = int(60 + (intensity * 40 * (i / num_hits)))
                vel = min(115, vel)

                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth // 2,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
                ))

        elif target_instrument == "toms":
            # Tom pattern (high to low)
            tom_pitches = [
                GM_DRUM_MAP['HIGH_TOM'],
                GM_DRUM_MAP['HI_MID_TOM'],
                GM_DRUM_MAP['LOW_MID_TOM'],
                GM_DRUM_MAP['LOW_TOM']
            ]

            num_hits = length_beats * 2  # 8th note triplets feel

            for i in range(num_hits):
                tick = i * (ppqn // 2)
                tom_index = min(i % 4, 3)
                vel = int(70 + (intensity * 35))

                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth * 2,
                    velocity=vel,
                    pitch=tom_pitches[tom_index]
                ))

        else:  # "mixed"
            # Combination of snare and toms
            eighth = ppqn // 2

            for i in range(length_beats * 2):
                tick = i * eighth

                if i % 2 == 0:
                    # Snare
                    pitch = GM_DRUM_MAP['ACOUSTIC_SNARE']
                else:
                    # Tom
                    pitch = GM_DRUM_MAP['LOW_TOM']

                vel = int(70 + (intensity * 40))
                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth,
                    velocity=vel,
                    pitch=pitch
                ))

        return pattern


# ============================================================================
# Example Usage & Tests
# ============================================================================

if __name__ == "__main__":
    """Example usage of Big Band Drum Patterns"""

    print("=" * 70)
    print("BIG BAND DRUM PATTERN LIBRARY - Examples")
    print("=" * 70)

    ppqn = 960

    # Example 1: Swing ride pattern
    print("\n1. Swing Ride Pattern (4 bars):")
    swing_ride = BigBandDrumPatterns.swing_ride(ppqn=ppqn, bars=4, swing_ratio=0.62)
    print(f"   Notes: {len(swing_ride)}")
    print(f"   Swing ratio: 0.62 (medium)")

    # Example 2: Bebop ride
    print("\n2. Bebop Ride Pattern (4 bars):")
    bebop_ride = BigBandDrumPatterns.bebop_ride(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(bebop_ride)}")

    # Example 3: Hi-hat 2&4
    print("\n3. Hi-Hat on 2 & 4 (Basie style):")
    hihat = BigBandDrumPatterns.hihat_2_4(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(hihat)}")

    # Example 4: Feathered kick
    print("\n4. Feathered Kick (4 bars):")
    feathered = BigBandDrumPatterns.feathered_kick(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(feathered)}")
    print(f"   Velocity: 45 (very soft)")

    # Example 5: Bebop kick
    print("\n5. Bebop Bombs (syncopated kick):")
    bebop_kick = BigBandDrumPatterns.bebop_kick(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(bebop_kick)}")

    # Example 6: Afro-Cuban cowbell
    print("\n6. Afro-Cuban Cowbell (Manteca style):")
    cowbell = BigBandDrumPatterns.afro_cuban_bell(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(cowbell)}")

    # Example 7: Samba surdo
    print("\n7. Samba Surdo:")
    surdo = BigBandDrumPatterns.samba_surdo(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(surdo)}")

    # Example 8: Bossa ride
    print("\n8. Bossa Nova Ride:")
    bossa = BigBandDrumPatterns.bossa_ride(ppqn=ppqn, bars=4)
    print(f"   Notes: {len(bossa)}")

    # Example 9: Fills
    print("\n9. Drum Fills:")
    snare_fill = BigBandDrumPatterns.generate_fill(ppqn=ppqn, length_beats=2,
                                                    intensity=0.7, target_instrument="snare")
    print(f"   Snare fill (2 beats): {len(snare_fill)} notes")

    tom_fill = BigBandDrumPatterns.generate_fill(ppqn=ppqn, length_beats=4,
                                                  intensity=0.9, target_instrument="toms")
    print(f"   Tom fill (4 beats): {len(tom_fill)} notes")

    print("\n" + "=" * 70)
    print("Big Band Drum Patterns library ready!")
    print("=" * 70)
    print("\nAvailable patterns:")
    print("  SWING:")
    print("    - swing_ride (classic big band ride cymbal)")
    print("    - bebop_ride (syncopated bebop style)")
    print("    - hihat_2_4 (backbeat on 2 & 4)")
    print("    - hihat_all_beats (modern four-on-floor)")
    print("    - feathered_kick (soft Basie-style)")
    print("    - bebop_kick (syncopated bombs)")
    print("  LATIN:")
    print("    - afro_cuban_bell (Manteca-style cowbell)")
    print("    - samba_surdo (Brazilian bass drum)")
    print("    - bossa_ride (subtle bossa nova)")
    print("  FILLS:")
    print("    - generate_fill (snare, toms, mixed)")
    print("=" * 70)
