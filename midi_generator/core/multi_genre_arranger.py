#!/usr/bin/env python3
"""
Multi-Genre Arranger - Track-Level Genre Control System

This module enables seamless multi-genre arrangements where different tracks
can use different genres while maintaining harmonic compatibility, rhythmic
synchronization, and proper voice leading.

Research Foundation:
- Jazz Fusion harmonic techniques (quartal/quintal harmony, extensions)
- West African drum ensemble synchronization (role-optimized coupling)
- Genre-specific timing variability research (expertise modulates sync)
- Multi-modal synchronization in musical ensembles (audio/visual cues)
- Scale-free cross-correlations in ensemble timing
- Harmonic voicing and layering across genres

Features:
- Per-track genre assignment with automatic compatibility checking
- Harmonic unification across disparate genres
- Rhythmic synchronization with genre-appropriate timing variability
- Voice leading management across style boundaries
- Intelligent register allocation and texture balancing
- Dynamic orchestration suggestions
- Genre-specific articulation and phrasing
- Temporal alignment strategies (accompaniment vs. lead parts)

Examples:
- Track 1 (bass): Funk style with syncopation and slap technique
- Track 2 (piano): Jazz style with extended harmonies
- Track 3 (drums): Hip-hop style with laid-back timing
- Track 4 (strings): Classical style with smooth voice leading
- Result: Coherent fusion arrangement with each track in its native style

Author: Agent 9 - Track-Level Genre Control
Date: 2025
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import copy
from collections import defaultdict

# Import existing modules from the library
try:
    from midi_generator.generators.style_fusion import GenreFeatures, GENRE_PROFILES
except ImportError:
    # Fallback for testing
    from dataclasses import dataclass
    from typing import List, Tuple

    @dataclass
    class GenreFeatures:
        name: str
        tempo_range: Tuple[int, int]
        swing_factor: float
        syncopation: float
        rhythmic_complexity: float
        chord_types: List[str]
        harmonic_rhythm: float
        use_extensions: bool
        chromaticism: float
        interval_preference: str
        ornamentation: float
        melodic_range: Tuple[int, int]
        instruments: List[int]
        texture: str
        register_preference: str
        cultural_origin: str
        rhythmic_basis: str
        groove_type: str

    GENRE_PROFILES = {}


# ==============================================================================
# ENUMS AND DATA STRUCTURES
# ==============================================================================

class TrackRole(Enum):
    """Musical role of a track in the arrangement"""
    BASS = "bass"                       # Bass line foundation
    HARMONY = "harmony"                 # Chordal accompaniment
    MELODY = "melody"                   # Lead melodic line
    PERCUSSION = "percussion"           # Drums and percussion
    COUNTER_MELODY = "counter_melody"   # Secondary melodic line
    PAD = "pad"                         # Sustained harmonic texture
    RHYTHMIC_ACCENT = "rhythmic_accent" # Rhythmic hits (brass, stabs)
    ORNAMENT = "ornament"               # Decorative elements


class SyncStrategy(Enum):
    """Rhythmic synchronization strategies for multi-genre ensembles"""
    STRICT_GRID = "strict_grid"             # All parts align to strict tempo grid
    ACCOMPANIMENT_REFERENCE = "accompaniment_reference"  # Bass/drums lead, others follow
    GENRE_WEIGHTED_TIMING = "genre_weighted_timing"      # Each genre's natural timing variability
    LOOSE_POCKET = "loose_pocket"           # Intentional timing variations (jazz, hip-hop)
    POLYRHYTHMIC = "polyrhythmic"           # Independent rhythmic layers
    ADAPTIVE = "adaptive"                   # Dynamically adjust based on section


class VoiceLeadingPriority(Enum):
    """Priority levels for voice leading considerations"""
    STRICT = "strict"           # Classical-style voice leading (no parallel 5ths/8ves)
    MODERATE = "moderate"       # Prefer smooth motion, allow some parallels
    LOOSE = "loose"             # Genre-appropriate, don't enforce classical rules
    GENRE_SPECIFIC = "genre_specific"  # Apply genre's native voice leading rules


class RegisterRange(Enum):
    """Frequency registers for orchestration"""
    SUB_BASS = (20, 40)         # E0 to E1 (MIDI 16-28)
    BASS = (28, 55)             # E1 to G3 (MIDI 28-55)
    LOW_MID = (48, 67)          # C3 to G4 (MIDI 48-67)
    MID = (60, 79)              # C4 to G5 (MIDI 60-79)
    HIGH_MID = (72, 91)         # C5 to G6 (MIDI 72-91)
    HIGH = (84, 108)            # C6 to C8 (MIDI 84-108)


@dataclass
class TrackSpec:
    """
    Specification for a single track in multi-genre arrangement

    Defines genre, role, and generation parameters for one track
    """
    track_number: int
    genre: str                          # Genre name (must exist in GENRE_PROFILES)
    role: TrackRole
    instrument: int                     # MIDI program number

    # Optional parameters
    register: Optional[RegisterRange] = None
    velocity_range: Tuple[int, int] = (60, 100)
    sync_strategy: SyncStrategy = SyncStrategy.ACCOMPANIMENT_REFERENCE
    voice_leading_priority: VoiceLeadingPriority = VoiceLeadingPriority.MODERATE

    # Genre override parameters (override GenreFeatures for this track)
    custom_swing_factor: Optional[float] = None
    custom_syncopation: Optional[float] = None
    custom_articulation: Optional[str] = None

    # Mix parameters
    timing_offset_ms: float = 0.0       # Deliberate timing offset (negative = rush, positive = drag)
    humanize_amount: float = 0.05       # Random timing variation (0-1)

    def __post_init__(self):
        if self.genre not in GENRE_PROFILES and GENRE_PROFILES:
            raise ValueError(f"Unknown genre: {self.genre}. Available: {list(GENRE_PROFILES.keys())}")


@dataclass
class HarmonicContext:
    """
    Shared harmonic information for all tracks

    Ensures all tracks follow the same chord progression regardless of genre
    """
    chord_progression: List[str]        # Chord symbols (e.g., ['Cmaj7', 'Dm7', 'G7'])
    key: str                            # Key signature
    time_signature: Tuple[int, int]     # (numerator, denominator)
    tempo_bpm: int
    length_measures: int

    # Advanced harmonic features
    modal_center: Optional[str] = None  # For modal pieces
    harmonic_rhythm_pattern: Optional[List[int]] = None  # Measures per chord
    allow_reharmonization: bool = True  # Allow genre-specific chord substitutions

    def get_chord_at_measure(self, measure: int) -> str:
        """Get chord symbol for a specific measure"""
        if not self.chord_progression:
            return "C"  # Default

        if self.harmonic_rhythm_pattern:
            # Use custom harmonic rhythm
            cumulative = 0
            for i, duration in enumerate(self.harmonic_rhythm_pattern):
                cumulative += duration
                if measure < cumulative:
                    idx = i % len(self.chord_progression)
                    return self.chord_progression[idx]

        # Default: one chord per measure, cycling
        idx = measure % len(self.chord_progression)
        return self.chord_progression[idx]


@dataclass
class GeneratedTrack:
    """Output of track generation"""
    track_number: int
    spec: TrackSpec
    notes: List[Tuple[int, float, float, int]]  # (pitch, start_time, duration, velocity)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_pitch_range(self) -> Tuple[int, int]:
        """Get actual pitch range of generated notes"""
        if not self.notes:
            return (0, 0)
        pitches = [n[0] for n in self.notes]
        return (min(pitches), max(pitches))

    def get_time_range(self) -> Tuple[float, float]:
        """Get time range of track"""
        if not self.notes:
            return (0.0, 0.0)
        start = min(n[1] for n in self.notes)
        end = max(n[1] + n[2] for n in self.notes)
        return (start, end)


# ==============================================================================
# GENRE COMPATIBILITY ANALYSIS
# ==============================================================================

class GenreCompatibilityAnalyzer:
    """
    Analyzes compatibility between multiple genres for arrangement

    Based on research into harmonic compatibility, rhythmic synchronization,
    and successful fusion precedents
    """

    # Known successful fusion combinations (higher compatibility)
    FUSION_PRECEDENTS = {
        ('jazz', 'hiphop'): 0.8,        # Nu-jazz, jazz-hop
        ('jazz', 'latin'): 0.9,         # Afro-Cuban jazz
        ('funk', 'jazz'): 0.85,         # Fusion jazz
        ('electronic', 'jazz'): 0.75,   # Electro-jazz
        ('blues', 'jazz'): 0.9,         # Jazz blues
        ('funk', 'hiphop'): 0.85,       # G-funk
        ('latin', 'electronic'): 0.7,   # Latin house
        ('reggae', 'hiphop'): 0.75,     # Reggae fusion
    }

    @staticmethod
    def calculate_compatibility(genre_a: str, genre_b: str) -> Dict[str, float]:
        """
        Calculate multi-dimensional compatibility between two genres

        Args:
            genre_a, genre_b: Genre names

        Returns:
            {
                'overall': float (0-1),
                'rhythmic': float (0-1),
                'harmonic': float (0-1),
                'timbral': float (0-1),
                'cultural': float (0-1)
            }
        """
        if genre_a not in GENRE_PROFILES or genre_b not in GENRE_PROFILES:
            return {
                'overall': 0.5,
                'rhythmic': 0.5,
                'harmonic': 0.5,
                'timbral': 0.5,
                'cultural': 0.5
            }

        profile_a = GENRE_PROFILES[genre_a]
        profile_b = GENRE_PROFILES[genre_b]

        # Rhythmic compatibility
        rhythmic = GenreCompatibilityAnalyzer._calculate_rhythmic_compatibility(profile_a, profile_b)

        # Harmonic compatibility
        harmonic = GenreCompatibilityAnalyzer._calculate_harmonic_compatibility(profile_a, profile_b)

        # Timbral compatibility
        timbral = GenreCompatibilityAnalyzer._calculate_timbral_compatibility(profile_a, profile_b)

        # Cultural/historical fusion precedent
        cultural = GenreCompatibilityAnalyzer._calculate_cultural_compatibility(genre_a, genre_b)

        # Overall weighted average
        overall = (rhythmic * 0.3 + harmonic * 0.3 + timbral * 0.2 + cultural * 0.2)

        return {
            'overall': overall,
            'rhythmic': rhythmic,
            'harmonic': harmonic,
            'timbral': timbral,
            'cultural': cultural
        }

    @staticmethod
    def _calculate_rhythmic_compatibility(profile_a: GenreFeatures, profile_b: GenreFeatures) -> float:
        """Calculate rhythmic compatibility based on tempo, swing, and complexity"""
        # Tempo range overlap
        a_min, a_max = profile_a.tempo_range
        b_min, b_max = profile_b.tempo_range

        overlap_start = max(a_min, b_min)
        overlap_end = min(a_max, b_max)

        if overlap_end > overlap_start:
            overlap_size = overlap_end - overlap_start
            union_size = max(a_max, b_max) - min(a_min, b_min)
            tempo_compatibility = overlap_size / union_size
        else:
            tempo_compatibility = 0.0

        # Swing factor similarity (difference from 1.0, normalized)
        swing_diff = abs(profile_a.swing_factor - profile_b.swing_factor)
        swing_compatibility = 1.0 - min(swing_diff / 0.2, 1.0)  # 0.2 difference = 0 compatibility

        # Syncopation similarity
        syncopation_diff = abs(profile_a.syncopation - profile_b.syncopation)
        syncopation_compatibility = 1.0 - syncopation_diff

        # Weighted average
        return (tempo_compatibility * 0.4 + swing_compatibility * 0.3 + syncopation_compatibility * 0.3)

    @staticmethod
    def _calculate_harmonic_compatibility(profile_a: GenreFeatures, profile_b: GenreFeatures) -> float:
        """Calculate harmonic compatibility based on chord types and harmonic rhythm"""
        # Chord type overlap (Jaccard similarity)
        set_a = set(profile_a.chord_types)
        set_b = set(profile_b.chord_types)

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        if union > 0:
            chord_similarity = intersection / union
        else:
            chord_similarity = 0.0

        # Extension usage (both use or both don't = compatible)
        extension_compatibility = 1.0 if profile_a.use_extensions == profile_b.use_extensions else 0.5

        # Chromaticism similarity
        chromaticism_diff = abs(profile_a.chromaticism - profile_b.chromaticism)
        chromaticism_compatibility = 1.0 - chromaticism_diff

        return (chord_similarity * 0.5 + extension_compatibility * 0.25 + chromaticism_compatibility * 0.25)

    @staticmethod
    def _calculate_timbral_compatibility(profile_a: GenreFeatures, profile_b: GenreFeatures) -> float:
        """Calculate timbral compatibility based on instrumentation and texture"""
        # Instrument overlap
        set_a = set(profile_a.instruments)
        set_b = set(profile_b.instruments)

        if set_a or set_b:
            instrument_similarity = len(set_a & set_b) / len(set_a | set_b)
        else:
            instrument_similarity = 0.5

        # Texture compatibility (same texture = more compatible)
        texture_compatibility = 1.0 if profile_a.texture == profile_b.texture else 0.6

        return (instrument_similarity * 0.6 + texture_compatibility * 0.4)

    @staticmethod
    def _calculate_cultural_compatibility(genre_a: str, genre_b: str) -> float:
        """Check for historical fusion precedents"""
        # Check both orderings
        key1 = (genre_a, genre_b)
        key2 = (genre_b, genre_a)

        if key1 in GenreCompatibilityAnalyzer.FUSION_PRECEDENTS:
            return GenreCompatibilityAnalyzer.FUSION_PRECEDENTS[key1]
        elif key2 in GenreCompatibilityAnalyzer.FUSION_PRECEDENTS:
            return GenreCompatibilityAnalyzer.FUSION_PRECEDENTS[key2]
        else:
            # No known precedent, return neutral
            return 0.5

    @staticmethod
    def analyze_multi_genre_compatibility(genres: List[str]) -> Dict[str, Any]:
        """
        Analyze compatibility across multiple genres

        Returns:
            {
                'pairwise_scores': Dict[(genre_a, genre_b), Dict[str, float]],
                'overall_compatibility': float,
                'potential_issues': List[str],
                'recommendations': List[str]
            }
        """
        pairwise_scores = {}
        all_scores = []

        # Calculate pairwise compatibility
        for i, genre_a in enumerate(genres):
            for j, genre_b in enumerate(genres):
                if i < j:  # Avoid duplicates
                    scores = GenreCompatibilityAnalyzer.calculate_compatibility(genre_a, genre_b)
                    pairwise_scores[(genre_a, genre_b)] = scores
                    all_scores.append(scores['overall'])

        # Overall compatibility (average of all pairs)
        overall = sum(all_scores) / len(all_scores) if all_scores else 0.5

        # Identify potential issues
        issues = []
        recommendations = []

        for (genre_a, genre_b), scores in pairwise_scores.items():
            if scores['rhythmic'] < 0.4:
                issues.append(f"Low rhythmic compatibility between {genre_a} and {genre_b}")
                recommendations.append(f"Use STRICT_GRID sync for {genre_a} and {genre_b} tracks")

            if scores['harmonic'] < 0.3:
                issues.append(f"Low harmonic compatibility between {genre_a} and {genre_b}")
                recommendations.append(f"Consider shared chord progression with genre-specific voicings")

            if scores['overall'] > 0.7:
                recommendations.append(f"{genre_a} and {genre_b} blend well - emphasize this combination")

        return {
            'pairwise_scores': pairwise_scores,
            'overall_compatibility': overall,
            'potential_issues': issues,
            'recommendations': recommendations
        }


# ==============================================================================
# HARMONIC UNIFICATION
# ==============================================================================

class HarmonicUnifier:
    """
    Ensures harmonic consistency across tracks with different genres

    Allows genre-specific chord voicings and substitutions while maintaining
    underlying harmonic structure
    """

    @staticmethod
    def parse_chord_symbol(chord: str) -> Dict[str, Any]:
        """
        Parse chord symbol into components

        Args:
            chord: Chord symbol (e.g., 'Cmaj7', 'Dm7b5', 'G7#9')

        Returns:
            {
                'root': str,
                'quality': str,
                'extensions': List[str],
                'bass_note': Optional[str]
            }
        """
        # Simple parser (can be extended)
        chord = chord.strip()

        # Check for slash chord (bass note)
        bass_note = None
        if '/' in chord:
            chord, bass_note = chord.split('/')

        # Extract root (first 1-2 characters)
        if len(chord) > 1 and chord[1] in ['#', 'b']:
            root = chord[:2]
            remainder = chord[2:]
        else:
            root = chord[0]
            remainder = chord[1:]

        # Determine quality and extensions
        quality = 'maj'  # Default
        extensions = []

        if 'maj7' in remainder:
            quality = 'maj7'
            remainder = remainder.replace('maj7', '')
        elif 'm7b5' in remainder or 'ø' in remainder:
            quality = 'half-dim7'
            remainder = remainder.replace('m7b5', '').replace('ø', '')
        elif 'dim7' in remainder or 'o7' in remainder:
            quality = 'dim7'
            remainder = remainder.replace('dim7', '').replace('o7', '')
        elif 'm7' in remainder or 'min7' in remainder:
            quality = 'min7'
            remainder = remainder.replace('m7', '').replace('min7', '')
        elif '7' in remainder:
            quality = 'dom7'
            remainder = remainder.replace('7', '')
        elif 'm' in remainder or 'min' in remainder:
            quality = 'min'
            remainder = remainder.replace('m', '').replace('min', '')

        # Remaining are extensions
        if remainder:
            extensions.append(remainder)

        return {
            'root': root,
            'quality': quality,
            'extensions': extensions,
            'bass_note': bass_note
        }

    @staticmethod
    def get_chord_tones(chord: str, octave: int = 4) -> List[int]:
        """
        Get MIDI note numbers for chord tones

        Args:
            chord: Chord symbol
            octave: Base octave for root note

        Returns:
            List of MIDI note numbers
        """
        parsed = HarmonicUnifier.parse_chord_symbol(chord)
        root_name = parsed['root']
        quality = parsed['quality']

        # Note name to pitch class
        note_map = {
            'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
            'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
            'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
        }

        root_pc = note_map.get(root_name, 0)
        root_midi = root_pc + (octave * 12)

        # Build chord based on quality
        intervals = []
        if quality == 'maj':
            intervals = [0, 4, 7]
        elif quality == 'min':
            intervals = [0, 3, 7]
        elif quality == 'maj7':
            intervals = [0, 4, 7, 11]
        elif quality == 'min7':
            intervals = [0, 3, 7, 10]
        elif quality == 'dom7':
            intervals = [0, 4, 7, 10]
        elif quality == 'half-dim7':
            intervals = [0, 3, 6, 10]
        elif quality == 'dim7':
            intervals = [0, 3, 6, 9]
        else:
            intervals = [0, 4, 7]  # Default to major

        return [root_midi + interval for interval in intervals]

    @staticmethod
    def apply_genre_substitution(chord: str, genre: str) -> str:
        """
        Apply genre-appropriate chord substitution

        Args:
            chord: Original chord symbol
            genre: Genre for substitution style

        Returns:
            Substituted chord (may be same as input)
        """
        parsed = HarmonicUnifier.parse_chord_symbol(chord)

        if genre == 'jazz':
            # Jazz: Add extensions
            if parsed['quality'] == 'maj7':
                return f"{parsed['root']}maj9"
            elif parsed['quality'] == 'min7':
                return f"{parsed['root']}m9"
            elif parsed['quality'] == 'dom7':
                # Altered dominant
                return f"{parsed['root']}7#9"

        elif genre == 'blues':
            # Blues: Everything becomes dominant 7th
            if parsed['quality'] in ['maj', 'maj7']:
                return f"{parsed['root']}7"

        elif genre == 'electronic':
            # Electronic: Simplify to triads or sus chords
            if parsed['quality'] in ['maj7', 'maj']:
                return f"{parsed['root']}"
            elif parsed['quality'] == 'dom7':
                return f"{parsed['root']}sus4"

        # No substitution
        return chord

    @staticmethod
    def ensure_harmonic_compatibility(tracks: List[TrackSpec],
                                     harmonic_context: HarmonicContext) -> Dict[int, List[str]]:
        """
        Generate genre-specific chord progressions for each track

        All tracks follow the same underlying harmony but with genre-appropriate
        voicings and substitutions

        Returns:
            {track_number: [chord_symbols]}
        """
        track_progressions = {}

        for track in tracks:
            if harmonic_context.allow_reharmonization:
                # Apply genre-specific substitutions
                progression = [
                    HarmonicUnifier.apply_genre_substitution(chord, track.genre)
                    for chord in harmonic_context.chord_progression
                ]
            else:
                # Use original progression
                progression = harmonic_context.chord_progression.copy()

            track_progressions[track.track_number] = progression

        return track_progressions


# ==============================================================================
# RHYTHMIC SYNCHRONIZATION
# ==============================================================================

class RhythmicSynchronizer:
    """
    Manages rhythmic synchronization across tracks with different genres

    Based on research showing:
    - Accompaniment parts serve as timing reference
    - Genre-specific timing variability is natural
    - Scale-free cross-correlations in ensemble timing
    """

    @staticmethod
    def apply_genre_timing(notes: List[Tuple[int, float, float, int]],
                          genre: str,
                          sync_strategy: SyncStrategy,
                          is_reference_track: bool = False) -> List[Tuple[int, float, float, int]]:
        """
        Apply genre-specific timing characteristics

        Args:
            notes: List of (pitch, start_time, duration, velocity)
            genre: Genre name
            sync_strategy: Synchronization strategy
            is_reference_track: True if this track is the timing reference

        Returns:
            Modified notes with genre-appropriate timing
        """
        if not notes or genre not in GENRE_PROFILES:
            return notes

        profile = GENRE_PROFILES[genre]
        modified = []

        for pitch, start_time, duration, velocity in notes:
            new_start = start_time

            if sync_strategy == SyncStrategy.STRICT_GRID:
                # No timing modification - stay on grid
                pass

            elif sync_strategy == SyncStrategy.GENRE_WEIGHTED_TIMING:
                # Apply genre-specific timing variability
                if profile.groove_type == 'swing':
                    # Swing timing on off-beats
                    beat = start_time % 1.0
                    if 0.4 < beat < 0.6:  # Near off-beat
                        # Delay by swing amount
                        swing_offset = (profile.swing_factor - 0.5) * 0.5
                        new_start += swing_offset

                elif profile.groove_type == 'laid-back':
                    # Hip-hop laid-back feel (slightly behind beat)
                    new_start += random.uniform(0.01, 0.03)

            elif sync_strategy == SyncStrategy.LOOSE_POCKET:
                # Natural timing variations
                if not is_reference_track:
                    # Follower tracks have more variation
                    variation = random.gauss(0, 0.02)  # Gaussian with 20ms std dev
                    new_start += variation

            modified.append((pitch, new_start, duration, velocity))

        return modified

    @staticmethod
    def quantize_to_grid(notes: List[Tuple[int, float, float, int]],
                        grid_division: float = 0.25) -> List[Tuple[int, float, float, int]]:
        """
        Quantize notes to rhythmic grid

        Args:
            notes: Input notes
            grid_division: Grid resolution in beats (0.25 = 16th notes)

        Returns:
            Quantized notes
        """
        quantized = []

        for pitch, start_time, duration, velocity in notes:
            # Round to nearest grid point
            quantized_start = round(start_time / grid_division) * grid_division
            quantized.append((pitch, quantized_start, duration, velocity))

        return quantized

    @staticmethod
    def determine_reference_track(tracks: List[TrackSpec]) -> int:
        """
        Determine which track should be the timing reference

        Based on research: accompaniment parts (bass, drums) typically lead

        Returns:
            Track number of reference track
        """
        # Priority: percussion > bass > harmony > melody
        role_priority = {
            TrackRole.PERCUSSION: 0,
            TrackRole.BASS: 1,
            TrackRole.HARMONY: 2,
            TrackRole.RHYTHMIC_ACCENT: 3,
            TrackRole.PAD: 4,
            TrackRole.MELODY: 5,
            TrackRole.COUNTER_MELODY: 6,
            TrackRole.ORNAMENT: 7
        }

        reference = min(tracks, key=lambda t: role_priority.get(t.role, 99))
        return reference.track_number


# ==============================================================================
# VOICE LEADING MANAGER
# ==============================================================================

class VoiceLeadingManager:
    """
    Manages voice leading across tracks with different genres

    Applies appropriate voice leading rules based on genre and priority
    """

    @staticmethod
    def check_parallel_motion(voice1_notes: List[int],
                             voice2_notes: List[int],
                             interval_type: str = 'fifth') -> List[int]:
        """
        Check for parallel fifths or octaves

        Args:
            voice1_notes: Pitches in voice 1
            voice2_notes: Pitches in voice 2
            interval_type: 'fifth' or 'octave'

        Returns:
            List of indices where parallel motion occurs
        """
        parallels = []
        interval_size = 7 if interval_type == 'fifth' else 12

        for i in range(len(voice1_notes) - 1):
            if i >= len(voice2_notes) - 1:
                break

            # Current interval
            current_interval = abs(voice1_notes[i] - voice2_notes[i]) % 12
            # Next interval
            next_interval = abs(voice1_notes[i+1] - voice2_notes[i+1]) % 12

            # Check if both are perfect fifths/octaves and moving in same direction
            if current_interval == interval_size and next_interval == interval_size:
                # Check parallel motion
                voice1_direction = voice1_notes[i+1] - voice1_notes[i]
                voice2_direction = voice2_notes[i+1] - voice2_notes[i]

                if voice1_direction * voice2_direction > 0:  # Same direction
                    parallels.append(i)

        return parallels

    @staticmethod
    def smooth_voice_leading(notes: List[Tuple[int, float, float, int]],
                            chord_progression: List[str],
                            voice_leading_priority: VoiceLeadingPriority) -> List[Tuple[int, float, float, int]]:
        """
        Optimize voice leading for smooth motion

        Args:
            notes: Input notes
            chord_progression: Chord symbols
            voice_leading_priority: How strict to be

        Returns:
            Notes with improved voice leading
        """
        if voice_leading_priority == VoiceLeadingPriority.LOOSE:
            # Don't modify
            return notes

        # Group notes by chord (simplified - assumes one chord per measure)
        # This is a basic implementation - can be extended

        smoothed = []
        prev_pitch = None

        for pitch, start_time, duration, velocity in notes:
            if prev_pitch is not None:
                interval = abs(pitch - prev_pitch)

                if voice_leading_priority == VoiceLeadingPriority.STRICT:
                    # Prefer stepwise motion
                    if interval > 7:  # Larger than fifth
                        # Try to move by inversion (octave shift)
                        if pitch > prev_pitch:
                            new_pitch = pitch - 12
                        else:
                            new_pitch = pitch + 12

                        # Check if new pitch is better
                        new_interval = abs(new_pitch - prev_pitch)
                        if new_interval < interval:
                            pitch = new_pitch

            smoothed.append((pitch, start_time, duration, velocity))
            prev_pitch = pitch

        return smoothed


# ==============================================================================
# REGISTER ALLOCATOR
# ==============================================================================

class RegisterAllocator:
    """
    Intelligently allocate frequency registers to avoid muddiness

    Ensures tracks don't overlap too much in frequency space
    """

    @staticmethod
    def analyze_register_usage(tracks: List[GeneratedTrack]) -> Dict[RegisterRange, float]:
        """
        Analyze how much each register is being used

        Returns:
            {RegisterRange: density (0-1)}
        """
        register_usage = {reg: 0.0 for reg in RegisterRange}

        for track in tracks:
            low, high = track.get_pitch_range()

            # Check which registers this track occupies
            for reg in RegisterRange:
                reg_low, reg_high = reg.value

                # Calculate overlap
                overlap_low = max(low, reg_low)
                overlap_high = min(high, reg_high)

                if overlap_high > overlap_low:
                    overlap_size = overlap_high - overlap_low
                    register_usage[reg] += overlap_size / (reg_high - reg_low)

        # Normalize by number of tracks
        if tracks:
            for reg in register_usage:
                register_usage[reg] /= len(tracks)

        return register_usage

    @staticmethod
    def suggest_register(existing_tracks: List[GeneratedTrack],
                        role: TrackRole) -> RegisterRange:
        """
        Suggest optimal register for new track

        Args:
            existing_tracks: Already generated tracks
            role: Role of new track

        Returns:
            Suggested register
        """
        usage = RegisterAllocator.analyze_register_usage(existing_tracks)

        # Role-based preferences
        preferred_registers = {
            TrackRole.BASS: [RegisterRange.BASS, RegisterRange.LOW_MID],
            TrackRole.HARMONY: [RegisterRange.MID, RegisterRange.LOW_MID],
            TrackRole.MELODY: [RegisterRange.MID, RegisterRange.HIGH_MID],
            TrackRole.PAD: [RegisterRange.MID, RegisterRange.HIGH_MID, RegisterRange.HIGH],
            TrackRole.PERCUSSION: [RegisterRange.LOW_MID, RegisterRange.MID],
        }

        candidates = preferred_registers.get(role, list(RegisterRange))

        # Choose least used register from preferences
        best_register = min(candidates, key=lambda r: usage.get(r, 0))

        return best_register


# ==============================================================================
# MAIN MULTI-GENRE ARRANGER
# ==============================================================================

class MultiGenreArranger:
    """
    Main class for creating multi-genre arrangements

    Orchestrates all components: compatibility analysis, harmonic unification,
    rhythmic synchronization, voice leading, and register allocation

    Example usage:
        arranger = MultiGenreArranger()

        # Define harmonic context
        harmonic_context = HarmonicContext(
            chord_progression=['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],
            key='C',
            time_signature=(4, 4),
            tempo_bpm=120,
            length_measures=16
        )

        # Define tracks
        tracks = [
            TrackSpec(1, 'funk', TrackRole.BASS, 33),
            TrackSpec(2, 'jazz', TrackRole.HARMONY, 0),
            TrackSpec(3, 'hiphop', TrackRole.PERCUSSION, 128),
            TrackSpec(4, 'electronic', TrackRole.MELODY, 81)
        ]

        # Generate arrangement
        result = arranger.arrange(harmonic_context, tracks)
    """

    def __init__(self):
        self.compatibility_analyzer = GenreCompatibilityAnalyzer()
        self.harmonic_unifier = HarmonicUnifier()
        self.rhythmic_synchronizer = RhythmicSynchronizer()
        self.voice_leading_manager = VoiceLeadingManager()
        self.register_allocator = RegisterAllocator()

    def analyze_arrangement_compatibility(self, tracks: List[TrackSpec]) -> Dict[str, Any]:
        """
        Analyze compatibility of proposed arrangement

        Returns compatibility analysis and suggestions
        """
        genres = list(set(track.genre for track in tracks))
        return self.compatibility_analyzer.analyze_multi_genre_compatibility(genres)

    def arrange(self,
                harmonic_context: HarmonicContext,
                tracks: List[TrackSpec],
                auto_optimize: bool = True) -> Dict[str, Any]:
        """
        Generate complete multi-genre arrangement

        Args:
            harmonic_context: Shared harmonic structure
            tracks: Track specifications
            auto_optimize: Automatically optimize voice leading and registers

        Returns:
            {
                'tracks': List[GeneratedTrack],
                'harmonic_context': HarmonicContext,
                'compatibility_analysis': Dict,
                'metadata': Dict
            }
        """
        # Step 1: Analyze compatibility
        compatibility = self.analyze_arrangement_compatibility(tracks)

        # Step 2: Generate genre-specific chord progressions
        track_progressions = self.harmonic_unifier.ensure_harmonic_compatibility(
            tracks, harmonic_context
        )

        # Step 3: Determine timing reference track
        reference_track_num = self.rhythmic_synchronizer.determine_reference_track(tracks)

        # Step 4: Generate each track
        generated_tracks = []

        for track_spec in tracks:
            # Generate track content (delegated to genre-specific generators)
            notes = self._generate_track_content(
                track_spec,
                track_progressions[track_spec.track_number],
                harmonic_context
            )

            # Apply rhythmic synchronization
            is_reference = (track_spec.track_number == reference_track_num)
            notes = self.rhythmic_synchronizer.apply_genre_timing(
                notes,
                track_spec.genre,
                track_spec.sync_strategy,
                is_reference
            )

            # Apply voice leading optimization if requested
            if auto_optimize:
                notes = self.voice_leading_manager.smooth_voice_leading(
                    notes,
                    track_progressions[track_spec.track_number],
                    track_spec.voice_leading_priority
                )

            # Create GeneratedTrack
            gen_track = GeneratedTrack(
                track_number=track_spec.track_number,
                spec=track_spec,
                notes=notes,
                metadata={
                    'chord_progression': track_progressions[track_spec.track_number],
                    'is_reference_track': is_reference
                }
            )

            generated_tracks.append(gen_track)

        # Step 5: Analyze register usage
        register_usage = self.register_allocator.analyze_register_usage(generated_tracks)

        return {
            'tracks': generated_tracks,
            'harmonic_context': harmonic_context,
            'compatibility_analysis': compatibility,
            'metadata': {
                'reference_track': reference_track_num,
                'register_usage': register_usage,
                'track_count': len(tracks)
            }
        }

    def _generate_track_content(self,
                               track_spec: TrackSpec,
                               chord_progression: List[str],
                               harmonic_context: HarmonicContext) -> List[Tuple[int, float, float, int]]:
        """
        Generate actual note content for a track

        This is a placeholder - in a full implementation, this would delegate to
        the appropriate genre-specific generator (bass_engine, drum_patterns, etc.)

        For now, generates simple patterns based on role
        """
        notes = []
        beats_per_measure = harmonic_context.time_signature[0]
        total_beats = harmonic_context.length_measures * beats_per_measure

        if track_spec.role == TrackRole.BASS:
            # Generate walking bass pattern
            for measure in range(harmonic_context.length_measures):
                chord = harmonic_context.get_chord_at_measure(measure)
                chord_tones = self.harmonic_unifier.get_chord_tones(chord, octave=2)

                for beat in range(beats_per_measure):
                    time = measure * beats_per_measure + beat
                    pitch = chord_tones[beat % len(chord_tones)]
                    notes.append((pitch, float(time), 0.8, 80))

        elif track_spec.role == TrackRole.HARMONY:
            # Generate chords (whole notes)
            for measure in range(harmonic_context.length_measures):
                chord = harmonic_context.get_chord_at_measure(measure)
                chord_tones = self.harmonic_unifier.get_chord_tones(chord, octave=4)

                time = measure * beats_per_measure
                # Add each chord tone
                for pitch in chord_tones:
                    notes.append((pitch, float(time), float(beats_per_measure), 70))

        elif track_spec.role == TrackRole.MELODY:
            # Generate simple melody using chord tones
            for measure in range(harmonic_context.length_measures):
                chord = harmonic_context.get_chord_at_measure(measure)
                chord_tones = self.harmonic_unifier.get_chord_tones(chord, octave=5)

                # Create melodic rhythm
                for beat in range(beats_per_measure):
                    if random.random() < 0.7:  # 70% density
                        time = measure * beats_per_measure + beat
                        pitch = random.choice(chord_tones)
                        duration = random.choice([0.5, 1.0, 1.5])
                        notes.append((pitch, float(time), duration, 85))

        elif track_spec.role == TrackRole.PERCUSSION:
            # Generate basic drum pattern
            for measure in range(harmonic_context.length_measures):
                for beat in range(beats_per_measure):
                    time = measure * beats_per_measure + beat

                    # Kick on 1 and 3
                    if beat % 2 == 0:
                        notes.append((36, float(time), 0.2, 90))  # Kick

                    # Snare on 2 and 4
                    if beat % 2 == 1:
                        notes.append((38, float(time), 0.2, 85))  # Snare

                    # Hi-hat every beat
                    notes.append((42, float(time), 0.2, 70))  # Closed hi-hat

        return notes

    def export_to_midi(self, arrangement: Dict[str, Any], filename: str):
        """
        Export arrangement to MIDI file

        Args:
            arrangement: Output from arrange()
            filename: Output MIDI file path
        """
        try:
            import mido
            from mido import Message, MidiFile, MidiTrack
        except ImportError:
            print("Warning: mido library not available. Cannot export MIDI.")
            return

        mid = MidiFile()
        tempo = arrangement['harmonic_context'].tempo_bpm

        # Add tempo track
        tempo_track = MidiTrack()
        mid.tracks.append(tempo_track)

        # Calculate tempo in microseconds per beat
        tempo_microseconds = int(60000000 / tempo)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo_microseconds))

        # Add each track
        for gen_track in arrangement['tracks']:
            track = MidiTrack()
            mid.tracks.append(track)

            # Set program (instrument)
            track.append(Message('program_change', program=gen_track.spec.instrument, time=0))

            # Sort notes by start time
            sorted_notes = sorted(gen_track.notes, key=lambda n: n[1])

            # Convert notes to MIDI messages
            current_time = 0
            for pitch, start_time, duration, velocity in sorted_notes:
                # Note on
                delta_time = int((start_time - current_time) * 480)  # Convert to ticks
                track.append(Message('note_on', note=pitch, velocity=velocity, time=delta_time))
                current_time = start_time

                # Note off
                delta_time = int(duration * 480)
                track.append(Message('note_off', note=pitch, velocity=0, time=delta_time))
                current_time = start_time + duration

        mid.save(filename)
        print(f"Exported arrangement to {filename}")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def create_simple_arrangement(genres: List[str],
                             key: str = 'C',
                             tempo: int = 120,
                             measures: int = 8) -> Dict[str, Any]:
    """
    Quick helper to create a simple multi-genre arrangement

    Args:
        genres: List of genres (one per track)
        key: Key signature
        tempo: Tempo in BPM
        measures: Number of measures

    Returns:
        Arrangement dictionary
    """
    # Default chord progression
    chord_progression = ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']

    harmonic_context = HarmonicContext(
        chord_progression=chord_progression,
        key=key,
        time_signature=(4, 4),
        tempo_bpm=tempo,
        length_measures=measures
    )

    # Create track specs
    role_map = {
        0: TrackRole.BASS,
        1: TrackRole.HARMONY,
        2: TrackRole.MELODY,
        3: TrackRole.PERCUSSION
    }

    instrument_map = {
        TrackRole.BASS: 33,        # Electric bass
        TrackRole.HARMONY: 0,      # Piano
        TrackRole.MELODY: 65,      # Sax
        TrackRole.PERCUSSION: 128  # Drums
    }

    tracks = []
    for i, genre in enumerate(genres):
        role = role_map.get(i, TrackRole.HARMONY)
        instrument = instrument_map.get(role, 0)

        tracks.append(TrackSpec(
            track_number=i + 1,
            genre=genre,
            role=role,
            instrument=instrument
        ))

    # Generate arrangement
    arranger = MultiGenreArranger()
    return arranger.arrange(harmonic_context, tracks)


# ==============================================================================
# MAIN - EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Multi-Genre Arranger - Agent 9")
    print("=" * 60)

    # Example 1: Jazz piano + Funk bass + Hip-hop drums
    print("\nExample 1: Jazz + Funk + Hip-hop fusion")
    print("-" * 60)

    harmonic_context = HarmonicContext(
        chord_progression=['Cmaj7', 'Dm7', 'G7', 'Cmaj7'],
        key='C',
        time_signature=(4, 4),
        tempo_bpm=95,
        length_measures=8
    )

    tracks = [
        TrackSpec(1, 'funk', TrackRole.BASS, 33),
        TrackSpec(2, 'jazz', TrackRole.HARMONY, 0),
        TrackSpec(3, 'hiphop', TrackRole.PERCUSSION, 128)
    ]

    arranger = MultiGenreArranger()

    # Analyze compatibility first
    compatibility = arranger.analyze_arrangement_compatibility(tracks)
    print(f"Overall compatibility: {compatibility['overall_compatibility']:.2f}")

    if compatibility['recommendations']:
        print("Recommendations:")
        for rec in compatibility['recommendations']:
            print(f"  - {rec}")

    # Generate arrangement
    result = arranger.arrange(harmonic_context, tracks)

    print(f"\nGenerated {len(result['tracks'])} tracks:")
    for track in result['tracks']:
        print(f"  Track {track.track_number}: {track.spec.genre} {track.spec.role.value}")
        print(f"    Notes: {len(track.notes)}, Range: {track.get_pitch_range()}")

    print("\n" + "=" * 60)
    print("Multi-Genre Arranger ready for integration!")
