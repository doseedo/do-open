"""
Groove Library - Famous Drum Grooves and Timing Profiles

Comprehensive collection of:
- Famous drum grooves (Purdie shuffle, Motown backbeat, Afrobeat, etc.)
- Genre-specific microtiming profiles
- Instrument-specific timing characteristics
- Professional drummer timing models

Research References:
- "The Beat Will Make You Confess" - Iyer (2002)
- "Timing in Music Performance" - Repp & Su (2013)
- "Groove and Synchronization" - Janata et al. (2012)
- "Microtiming Deviations" - Davies et al. (2013)
- "The Funky Drummer" - Pressing (2002)

Author: MIDI Generator Team
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np

# Import from rhythm_engine
from algorithms.rhythm_engine import GrooveTemplate, RhythmNote
from midi.midi_constants import (
    GM_DRUM_MAP,
    DEFAULT_PPQN,
    PPQN_HIGH_RES
)


# ============================================================================
# Famous Drum Grooves
# ============================================================================

class FamousGrooves:
    """
    Library of famous drum grooves from legendary drummers and tracks.

    Each groove includes:
    - Full drum pattern (kick, snare, hi-hat)
    - Characteristic timing deviations
    - Velocity patterns
    - Ghost notes and embellishments
    """

    @staticmethod
    def purdie_shuffle(ppqn: int = PPQN_HIGH_RES) -> List[RhythmNote]:
        """
        Bernard "Pretty" Purdie's famous shuffle groove.

        Characteristics:
        - Laid-back hi-hat
        - Strong backbeat
        - Ghost notes on snare
        - Triplet feel

        Used in: Steely Dan, Aretha Franklin recordings
        """
        ticks_per_beat = ppqn
        sixteenth = ticks_per_beat // 4

        pattern = []

        # Bar 1 (4 beats)
        for beat in range(4):
            base_tick = beat * ticks_per_beat

            # Kick on 1 and 3
            if beat in [0, 2]:
                pattern.append(RhythmNote(
                    tick=base_tick,
                    duration=sixteenth * 2,
                    velocity=95,
                    pitch=GM_DRUM_MAP['BASS_DRUM_1']
                ))

            # Snare on 2 and 4 (backbeat)
            if beat in [1, 3]:
                # Main snare hit
                pattern.append(RhythmNote(
                    tick=base_tick,
                    duration=sixteenth * 2,
                    velocity=100,
                    pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
                ))

                # Ghost note before backbeat (signature Purdie)
                pattern.append(RhythmNote(
                    tick=base_tick - sixteenth // 2,
                    duration=sixteenth,
                    velocity=40,  # Soft ghost note
                    pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
                ))

            # Hi-hat shuffle pattern (triplet feel)
            # Laid-back timing on offbeats
            for triplet in range(3):
                tick = base_tick + (triplet * ticks_per_beat * 2) // 3
                vel = 75 if triplet == 0 else 55  # Accent first of triplet

                # Slightly late on offbeats (laid-back feel)
                timing_adjust = 10 if triplet % 2 == 1 else 0

                pattern.append(RhythmNote(
                    tick=tick + timing_adjust,
                    duration=sixteenth,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
                ))

        return sorted(pattern, key=lambda n: n.tick)

    @staticmethod
    def motown_backbeat(ppqn: int = PPQN_HIGH_RES) -> List[RhythmNote]:
        """
        Classic Motown backbeat (Benny Benjamin, Uriel Jones).

        Characteristics:
        - Strong, consistent backbeat
        - Tambourine on quarters
        - Locked-in feel
        - Four-on-the-floor kick

        Used in: Supremes, Four Tops, Temptations
        """
        ticks_per_beat = ppqn
        sixteenth = ticks_per_beat // 4

        pattern = []

        for beat in range(4):
            base_tick = beat * ticks_per_beat

            # Kick on all four beats (four-on-the-floor)
            pattern.append(RhythmNote(
                tick=base_tick,
                duration=sixteenth * 2,
                velocity=90,
                pitch=GM_DRUM_MAP['BASS_DRUM_1']
            ))

            # Snare on 2 and 4 (strong backbeat)
            if beat in [1, 3]:
                pattern.append(RhythmNote(
                    tick=base_tick,
                    duration=sixteenth * 2,
                    velocity=105,  # Strong!
                    pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
                ))

            # Hi-hat on 8th notes
            for eighth in range(2):
                tick = base_tick + eighth * ticks_per_beat // 2
                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth,
                    velocity=70,
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
                ))

            # Tambourine on quarters (Motown signature)
            pattern.append(RhythmNote(
                tick=base_tick,
                duration=sixteenth * 3,
                velocity=75,
                pitch=GM_DRUM_MAP['TAMBOURINE']
            ))

        return sorted(pattern, key=lambda n: n.tick)

    @staticmethod
    def funky_drummer(ppqn: int = PPQN_HIGH_RES) -> List[RhythmNote]:
        """
        Clyde Stubblefield's "Funky Drummer" break.

        One of the most sampled drum breaks in history.

        Characteristics:
        - Syncopated kick pattern
        - Ghost notes on snare
        - Open hi-hat on upbeats
        - Laid-back 16th note feel

        Used in: James Brown, sampled in countless hip-hop tracks
        """
        ticks_per_beat = ppqn
        sixteenth = ticks_per_beat // 4

        pattern = []

        # Beat 1
        pattern.append(RhythmNote(tick=0, duration=sixteenth*2, velocity=100, pitch=GM_DRUM_MAP['BASS_DRUM_1']))
        pattern.append(RhythmNote(tick=0, duration=sixteenth, velocity=70, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))
        pattern.append(RhythmNote(tick=sixteenth*2, duration=sixteenth, velocity=60, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))

        # Beat 2 (backbeat + ghost notes)
        pattern.append(RhythmNote(tick=ticks_per_beat, duration=sixteenth*2, velocity=105, pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']))
        pattern.append(RhythmNote(tick=ticks_per_beat, duration=sixteenth, velocity=65, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))
        # Ghost note
        pattern.append(RhythmNote(tick=ticks_per_beat + sixteenth, duration=sixteenth, velocity=35, pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']))
        pattern.append(RhythmNote(tick=ticks_per_beat + sixteenth*2, duration=sixteenth, velocity=60, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))

        # Beat 3 (syncopated kick)
        pattern.append(RhythmNote(tick=ticks_per_beat*2 - sixteenth, duration=sixteenth*2, velocity=90, pitch=GM_DRUM_MAP['BASS_DRUM_1']))
        pattern.append(RhythmNote(tick=ticks_per_beat*2, duration=sixteenth, velocity=70, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))
        pattern.append(RhythmNote(tick=ticks_per_beat*2 + sixteenth, duration=sixteenth, velocity=95, pitch=GM_DRUM_MAP['BASS_DRUM_1']))
        pattern.append(RhythmNote(tick=ticks_per_beat*2 + sixteenth*2, duration=sixteenth, velocity=60, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))

        # Beat 4 (backbeat + open hi-hat)
        pattern.append(RhythmNote(tick=ticks_per_beat*3, duration=sixteenth*2, velocity=105, pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']))
        pattern.append(RhythmNote(tick=ticks_per_beat*3, duration=sixteenth, velocity=65, pitch=GM_DRUM_MAP['CLOSED_HI_HAT']))
        # Open hi-hat on offbeat
        pattern.append(RhythmNote(tick=ticks_per_beat*3 + sixteenth*2, duration=sixteenth*2, velocity=70, pitch=GM_DRUM_MAP['OPEN_HI_HAT']))

        return sorted(pattern, key=lambda n: n.tick)

    @staticmethod
    def afrobeat_pattern(ppqn: int = PPQN_HIGH_RES) -> List[RhythmNote]:
        """
        Afrobeat drum pattern (Tony Allen style).

        Characteristics:
        - Complex polyrhythmic kick and snare
        - Continuous 16th note hi-hat
        - Syncopated, interlocking patterns
        - Strong emphasis on groove pocket

        Used in: Fela Kuti, Tony Allen
        """
        ticks_per_beat = ppqn
        sixteenth = ticks_per_beat // 4

        pattern = []

        # Continuous 16th note hi-hat
        for i in range(16):
            tick = i * sixteenth
            # Accent on beats and offbeats
            vel = 75 if i % 4 == 0 else (65 if i % 2 == 0 else 50)
            pattern.append(RhythmNote(
                tick=tick,
                duration=sixteenth // 2,
                velocity=vel,
                pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
            ))

        # Complex kick pattern
        kick_positions = [0, 6, 10, 14]  # In 16th notes
        for pos in kick_positions:
            pattern.append(RhythmNote(
                tick=pos * sixteenth,
                duration=sixteenth * 2,
                velocity=95,
                pitch=GM_DRUM_MAP['BASS_DRUM_1']
            ))

        # Syncopated snare
        snare_positions = [4, 8, 13]  # In 16th notes
        for pos in snare_positions:
            vel = 100 if pos in [4, 12] else 85  # Accent on 2 and 4
            pattern.append(RhythmNote(
                tick=pos * sixteenth,
                duration=sixteenth * 2,
                velocity=vel,
                pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
            ))

        return sorted(pattern, key=lambda n: n.tick)

    @staticmethod
    def questlove_pocket(ppqn: int = PPQN_HIGH_RES) -> List[RhythmNote]:
        """
        Questlove's (The Roots) signature pocket groove.

        Characteristics:
        - Deep, behind-the-beat feel
        - Complex ghost note patterns
        - Subtle dynamics
        - Jazz-influenced hip-hop pocket
        """
        ticks_per_beat = ppqn
        sixteenth = ticks_per_beat // 4

        pattern = []

        # Slightly behind-the-beat timing offset
        laid_back = 8  # Ticks late

        for beat in range(4):
            base_tick = beat * ticks_per_beat

            # Kick on 1 and 3 (behind the beat)
            if beat in [0, 2]:
                pattern.append(RhythmNote(
                    tick=base_tick + laid_back,
                    duration=sixteenth * 2,
                    velocity=92,
                    pitch=GM_DRUM_MAP['BASS_DRUM_1']
                ))

            # Snare on 2 and 4 (also behind)
            if beat in [1, 3]:
                # Main snare
                pattern.append(RhythmNote(
                    tick=base_tick + laid_back,
                    duration=sixteenth * 2,
                    velocity=98,
                    pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
                ))

                # Complex ghost note pattern
                ghost_positions = [sixteenth, sixteenth * 3]
                for gp in ghost_positions:
                    pattern.append(RhythmNote(
                        tick=base_tick + gp,
                        duration=sixteenth,
                        velocity=30 + np.random.randint(-5, 5),  # Varied ghost notes
                        pitch=GM_DRUM_MAP['ACOUSTIC_SNARE']
                    ))

            # Hi-hat with subtle dynamics
            for eighth in range(2):
                tick = base_tick + eighth * ticks_per_beat // 2
                # Subtle velocity variation
                vel = 68 if eighth == 0 else 62
                pattern.append(RhythmNote(
                    tick=tick,
                    duration=sixteenth,
                    velocity=vel,
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT']
                ))

        return sorted(pattern, key=lambda n: n.tick)

    @staticmethod
    def d_n_b_amen_break(ppqn: int = PPQN_HIGH_RES) -> List[RhythmNote]:
        """
        Amen Break pattern (basis of drum and bass).

        Characteristics:
        - Syncopated kick and snare
        - Complex pattern that loops well
        - Foundation of jungle/D&B

        Original: "Amen, Brother" - The Winstons
        """
        ticks_per_beat = ppqn
        sixteenth = ticks_per_beat // 4

        pattern = []

        # The famous Amen break pattern (simplified)
        # Bar 1
        events = [
            # (tick_16ths, type, velocity)
            (0, 'kick', 100),
            (0, 'hat', 70),
            (2, 'hat', 60),
            (4, 'snare', 105),
            (4, 'hat', 65),
            (5, 'kick', 85),
            (6, 'hat', 60),
            (7, 'kick', 90),
            (8, 'kick', 95),
            (8, 'hat', 70),
            (10, 'hat', 60),
            (11, 'snare', 100),
            (12, 'snare', 105),
            (12, 'hat', 65),
            (13, 'kick', 90),
            (14, 'hat', 60),
            (15, 'snare', 95),
        ]

        drum_map = {
            'kick': GM_DRUM_MAP['BASS_DRUM_1'],
            'snare': GM_DRUM_MAP['ACOUSTIC_SNARE'],
            'hat': GM_DRUM_MAP['CLOSED_HI_HAT'],
        }

        for pos, drum_type, vel in events:
            pattern.append(RhythmNote(
                tick=pos * sixteenth,
                duration=sixteenth,
                velocity=vel,
                pitch=drum_map[drum_type]
            ))

        return sorted(pattern, key=lambda n: n.tick)


# ============================================================================
# Genre-Specific Timing Profiles
# ============================================================================

@dataclass
class TimingProfile:
    """Timing characteristics for a genre or style"""
    name: str
    description: str
    avg_deviation_ms: float         # Average timing deviation in milliseconds
    deviation_std_ms: float          # Standard deviation of timing
    early_late_bias: float           # Positive = late, negative = early, 0 = centered
    velocity_variation: float        # Coefficient of variation for velocity
    swing_ratio: float               # Typical swing amount (0.5 = straight, 0.67 = triplet)
    accent_strength: float           # How much stronger accented notes are (1.0-2.0)


class GenreTimingProfiles:
    """
    Collection of timing profiles for different musical genres.

    Based on research in systematic timing deviations across genres.
    """

    PROFILES = {
        'jazz_bebop': TimingProfile(
            name='Jazz Bebop',
            description='Fast, swinging, riding cymbal',
            avg_deviation_ms=8.0,
            deviation_std_ms=4.0,
            early_late_bias=0.0,  # Generally centered
            velocity_variation=0.25,
            swing_ratio=0.62,  # Strong swing
            accent_strength=1.4
        ),

        'jazz_ballad': TimingProfile(
            name='Jazz Ballad',
            description='Slow, brushes, gentle swing',
            avg_deviation_ms=15.0,
            deviation_std_ms=8.0,
            early_late_bias=5.0,  # Slightly laid back
            velocity_variation=0.3,
            swing_ratio=0.58,  # Lighter swing
            accent_strength=1.2
        ),

        'rock_straight': TimingProfile(
            name='Rock (Straight)',
            description='Driving, on-the-beat, loud',
            avg_deviation_ms=3.0,
            deviation_std_ms=2.0,
            early_late_bias=-1.0,  # Slightly pushing
            velocity_variation=0.15,
            swing_ratio=0.5,  # Straight
            accent_strength=1.6
        ),

        'funk': TimingProfile(
            name='Funk',
            description='Tight pocket, ghost notes, groove',
            avg_deviation_ms=5.0,
            deviation_std_ms=3.0,
            early_late_bias=2.0,  # Slightly behind
            velocity_variation=0.35,  # Wide dynamic range
            swing_ratio=0.52,  # Slight swing
            accent_strength=2.0  # Strong accents
        ),

        'hip_hop': TimingProfile(
            name='Hip Hop',
            description='Quantized but with feel, sampling influence',
            avg_deviation_ms=4.0,
            deviation_std_ms=2.0,
            early_late_bias=3.0,  # Laid back
            velocity_variation=0.2,
            swing_ratio=0.5,
            accent_strength=1.5
        ),

        'latin': TimingProfile(
            name='Latin',
            description='Clave-based, precise, energetic',
            avg_deviation_ms=3.0,
            deviation_std_ms=2.0,
            early_late_bias=-2.0,  # Slightly pushing
            velocity_variation=0.25,
            swing_ratio=0.5,
            accent_strength=1.7
        ),

        'reggae': TimingProfile(
            name='Reggae',
            description='One-drop, laid back, offbeat emphasis',
            avg_deviation_ms=12.0,
            deviation_std_ms=6.0,
            early_late_bias=10.0,  # Very laid back
            velocity_variation=0.2,
            swing_ratio=0.5,
            accent_strength=1.3
        ),

        'electronic_edm': TimingProfile(
            name='Electronic (EDM)',
            description='Quantized, tight, programmed',
            avg_deviation_ms=0.5,
            deviation_std_ms=0.3,
            early_late_bias=0.0,
            velocity_variation=0.1,
            swing_ratio=0.5,
            accent_strength=1.3
        ),

        'electronic_idm': TimingProfile(
            name='Electronic (IDM)',
            description='Intentional complexity, micro-variations',
            avg_deviation_ms=6.0,
            deviation_std_ms=5.0,
            early_late_bias=0.0,
            velocity_variation=0.4,
            swing_ratio=0.55,
            accent_strength=1.8
        ),

        'metal': TimingProfile(
            name='Metal',
            description='Precise, double bass, blast beats',
            avg_deviation_ms=2.0,
            deviation_std_ms=1.5,
            early_late_bias=-1.5,  # Pushing forward
            velocity_variation=0.12,
            swing_ratio=0.5,
            accent_strength=1.4
        ),

        'r_n_b': TimingProfile(
            name='R&B',
            description='Smooth, behind the beat, ghost notes',
            avg_deviation_ms=8.0,
            deviation_std_ms=4.0,
            early_late_bias=6.0,  # Behind the beat
            velocity_variation=0.3,
            swing_ratio=0.54,
            accent_strength=1.5
        ),

        'country': TimingProfile(
            name='Country',
            description='Train beat, steady, backbeat emphasis',
            avg_deviation_ms=4.0,
            deviation_std_ms=2.5,
            early_late_bias=0.0,
            velocity_variation=0.2,
            swing_ratio=0.5,
            accent_strength=1.6
        ),
    }

    @classmethod
    def get_profile(cls, genre: str) -> Optional[TimingProfile]:
        """Get timing profile for a genre"""
        return cls.PROFILES.get(genre.lower())

    @classmethod
    def list_genres(cls) -> List[str]:
        """List all available genre profiles"""
        return list(cls.PROFILES.keys())


# ============================================================================
# Instrument-Specific Timing Characteristics
# ============================================================================

@dataclass
class InstrumentTimingCharacteristics:
    """Timing behavior specific to an instrument"""
    name: str
    avg_offset_ms: float            # Average timing relative to beat (positive = late)
    attack_time_ms: float            # Attack envelope time
    natural_jitter_ms: float         # Natural timing variation
    velocity_sensitivity: float      # How much velocity affects timing (0-1)


class InstrumentTiming:
    """
    Instrument-specific timing characteristics.

    Based on research showing different instruments naturally have
    different timing relationships in ensemble playing.
    """

    CHARACTERISTICS = {
        'bass': InstrumentTimingCharacteristics(
            name='Bass',
            avg_offset_ms=5.0,      # Bass often slightly behind beat
            attack_time_ms=15.0,
            natural_jitter_ms=4.0,
            velocity_sensitivity=0.3
        ),

        'drums_kick': InstrumentTimingCharacteristics(
            name='Kick Drum',
            avg_offset_ms=0.0,      # Kick often defines the beat
            attack_time_ms=5.0,
            natural_jitter_ms=3.0,
            velocity_sensitivity=0.2
        ),

        'drums_snare': InstrumentTimingCharacteristics(
            name='Snare Drum',
            avg_offset_ms=1.0,      # Slightly ahead for cutting through
            attack_time_ms=3.0,
            natural_jitter_ms=3.0,
            velocity_sensitivity=0.4
        ),

        'drums_hihat': InstrumentTimingCharacteristics(
            name='Hi-Hat',
            avg_offset_ms=-2.0,     # Often slightly ahead
            attack_time_ms=2.0,
            natural_jitter_ms=2.0,
            velocity_sensitivity=0.5
        ),

        'guitar': InstrumentTimingCharacteristics(
            name='Guitar',
            avg_offset_ms=-3.0,     # Guitars often slightly ahead (strumming)
            attack_time_ms=8.0,
            natural_jitter_ms=5.0,
            velocity_sensitivity=0.4
        ),

        'piano': InstrumentTimingCharacteristics(
            name='Piano',
            avg_offset_ms=0.0,      # Piano often reference
            attack_time_ms=5.0,
            natural_jitter_ms=3.0,
            velocity_sensitivity=0.3
        ),

        'vocals': InstrumentTimingCharacteristics(
            name='Vocals',
            avg_offset_ms=8.0,      # Vocals often laid back
            attack_time_ms=20.0,    # Slower attack
            natural_jitter_ms=10.0, # More variation
            velocity_sensitivity=0.6
        ),
    }

    @classmethod
    def get_characteristics(cls, instrument: str) -> Optional[InstrumentTimingCharacteristics]:
        """Get timing characteristics for an instrument"""
        return cls.CHARACTERISTICS.get(instrument.lower())

    @classmethod
    def list_instruments(cls) -> List[str]:
        """List all instruments with timing profiles"""
        return list(cls.CHARACTERISTICS.keys())


# ============================================================================
# Groove Library Manager
# ============================================================================

class GrooveLibrary:
    """
    Main interface for accessing grooves and timing profiles.
    """

    def __init__(self, ppqn: int = PPQN_HIGH_RES):
        self.ppqn = ppqn
        self.famous_grooves = FamousGrooves()
        self.genre_profiles = GenreTimingProfiles()
        self.instrument_timing = InstrumentTiming()

    def get_groove(self, groove_name: str) -> List[RhythmNote]:
        """
        Get a famous groove by name.

        Available grooves:
        - purdie_shuffle
        - motown_backbeat
        - funky_drummer
        - afrobeat_pattern
        - questlove_pocket
        - d_n_b_amen_break
        """
        groove_methods = {
            'purdie_shuffle': self.famous_grooves.purdie_shuffle,
            'motown_backbeat': self.famous_grooves.motown_backbeat,
            'funky_drummer': self.famous_grooves.funky_drummer,
            'afrobeat_pattern': self.famous_grooves.afrobeat_pattern,
            'questlove_pocket': self.famous_grooves.questlove_pocket,
            'd_n_b_amen_break': self.famous_grooves.d_n_b_amen_break,
        }

        if groove_name not in groove_methods:
            raise ValueError(f"Unknown groove: {groove_name}")

        return groove_methods[groove_name](self.ppqn)

    def get_genre_profile(self, genre: str) -> Optional[TimingProfile]:
        """Get timing profile for a genre"""
        return self.genre_profiles.get_profile(genre)

    def get_instrument_timing(self, instrument: str) -> Optional[InstrumentTimingCharacteristics]:
        """Get timing characteristics for an instrument"""
        return self.instrument_timing.get_characteristics(instrument)

    def list_grooves(self) -> List[str]:
        """List all available famous grooves"""
        return [
            'purdie_shuffle',
            'motown_backbeat',
            'funky_drummer',
            'afrobeat_pattern',
            'questlove_pocket',
            'd_n_b_amen_break',
        ]

    def list_genre_profiles(self) -> List[str]:
        """List all genre timing profiles"""
        return self.genre_profiles.list_genres()

    def list_instrument_timings(self) -> List[str]:
        """List all instrument timing profiles"""
        return self.instrument_timing.list_instruments()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """Example usage of the Groove Library"""

    print("=" * 70)
    print("GROOVE LIBRARY - Example Usage")
    print("=" * 70)

    # Initialize library
    library = GrooveLibrary(ppqn=960)

    # Example 1: Get famous grooves
    print("\n1. Famous Grooves Available:")
    for groove_name in library.list_grooves():
        groove = library.get_groove(groove_name)
        print(f"   - {groove_name}: {len(groove)} notes")

    # Example 2: Get specific groove
    print("\n2. Purdie Shuffle Pattern:")
    purdie = library.get_groove('purdie_shuffle')
    print(f"   Notes: {len(purdie)}")
    print(f"   Duration: {max(n.tick for n in purdie)} ticks")

    # Example 3: Genre timing profiles
    print("\n3. Genre Timing Profiles:")
    for genre in library.list_genre_profiles()[:5]:  # Show first 5
        profile = library.get_genre_profile(genre)
        print(f"   - {profile.name}:")
        print(f"     Swing: {profile.swing_ratio:.2f}, Bias: {profile.early_late_bias:+.1f}ms")

    # Example 4: Instrument timing
    print("\n4. Instrument Timing Characteristics:")
    for instrument in library.list_instrument_timings():
        chars = library.get_instrument_timing(instrument)
        print(f"   - {chars.name}: offset {chars.avg_offset_ms:+.1f}ms")

    # Example 5: Apply genre feel
    print("\n5. Comparing Genre Feels:")
    genres_to_compare = ['jazz_bebop', 'funk', 'electronic_edm']
    for genre in genres_to_compare:
        profile = library.get_genre_profile(genre)
        print(f"   {profile.name}:")
        print(f"     Timing: {profile.avg_deviation_ms:.1f}±{profile.deviation_std_ms:.1f}ms")
        print(f"     Velocity var: {profile.velocity_variation:.2f}")
        print(f"     Accent: {profile.accent_strength:.1f}x")

    print("\n" + "=" * 70)
    print("Groove Library examples completed!")
    print("=" * 70)
