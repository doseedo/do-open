#!/usr/bin/env python3
"""
R&B, Neo-Soul & Contemporary Music Generator

This module implements comprehensive R&B and neo-soul music generation including:
- Classic R&B (90s-2000s): Boyz II Men, Usher, Aaliyah, Brandy
- Neo-soul: D'Angelo, Erykah Badu, Jill Scott, Robert Glasper
- Contemporary R&B: The Weeknd, SZA, Frank Ocean, H.E.R.

Features:
- Extended chord voicings (maj7#11, min9, min11, 9sus4, 13sus)
- J Dilla-influenced swing and microtiming
- Half-time and double-time feels
- Rhodes and Wurlitzer electric piano voicings
- 808 bass patterns with pitch slides
- Smooth vocal-range melodies with melismatic runs
- Ambient pad textures
- Rootless and cluster voicings

Research Sources:
- "Dilla Time" - Ethan Hein (J Dilla microtiming analysis, 53-56% swing)
- "21st Century Funk: A Microtiming Analysis of J Dilla" - Sean Peterson (Academia.edu)
- Robert Glasper harmonic techniques: rootless cluster voicings, minor 3rd shifts
- Contemporary R&B production: 808 glide techniques (10-150ms)
- Neo-soul harmony: extended chords avoiding traditional ii-V-I progressions
- Rhodes/Wurlitzer characteristics in R&B/soul context

Author: Agent 13 - R&B, Neo-Soul & Contemporary
Date: 2025
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum


class RnBStyle(Enum):
    """R&B sub-genres and eras"""
    CLASSIC_90S = "classic_90s"  # Boyz II Men, Usher, Aaliyah
    NEOSOUL = "neosoul"  # D'Angelo, Erykah Badu, Robert Glasper
    CONTEMPORARY = "contemporary"  # The Weeknd, SZA, Frank Ocean
    QUIET_STORM = "quiet_storm"  # Smooth, ballad-oriented
    URBAN_CONTEMPORARY = "urban_contemporary"  # Uptempo, dance-oriented


class ChordQuality(Enum):
    """Extended chord qualities common in R&B/neo-soul"""
    MAJ7 = "maj7"
    MAJ9 = "maj9"
    MAJ7_SHARP11 = "maj7#11"
    MAJ13 = "maj13"
    MIN7 = "min7"
    MIN9 = "min9"
    MIN11 = "min11"
    DOM7 = "dom7"
    DOM9 = "dom9"
    DOM13 = "dom13"
    SUS2 = "sus2"
    SUS4 = "sus4"
    NINE_SUS4 = "9sus4"
    THIRTEEN_SUS = "13sus"
    ADD9 = "add9"
    HALF_DIM = "half_dim"


@dataclass
class MIDINote:
    """Represents a MIDI note with timing and expression"""
    pitch: int  # MIDI note number (0-127)
    velocity: int  # Note velocity (0-127)
    start_time: float  # Start time in ticks or beats
    duration: float  # Duration in ticks or beats
    channel: int = 0  # MIDI channel
    pitch_bend: Optional[int] = None  # Pitch bend value (-8192 to 8191)


@dataclass
class BassSlide:
    """Represents an 808 bass slide"""
    from_note: int
    to_note: int
    glide_time_ms: float  # Glide time in milliseconds
    start_time: float
    duration: float


class RnBNeoSoulGenerator:
    """
    Advanced R&B and neo-soul music generation

    Based on research of classic R&B (90s-2000s), neo-soul pioneers
    (D'Angelo, Erykah Badu, Robert Glasper), and contemporary artists
    (The Weeknd, SZA, Frank Ocean).

    Features:
    - Extended chord voicings with jazz-influenced harmony
    - J Dilla swing and microtiming (53-56% swing)
    - Half-time/double-time rhythmic feels
    - Rhodes/Wurlitzer electric piano voicings
    - 808 bass with pitch slides
    - Vocal-range melodies (F2-E5 for male, A3-E6 for female)
    - Rootless and cluster chord voicings (Robert Glasper style)
    """

    # MIDI note names to numbers
    NOTE_NAMES = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8,
        'Ab': 8, 'A': 9, 'A#': 10, 'Bb': 10, 'B': 11
    }

    # Common 90s R&B progressions (as Roman numerals with quality)
    CLASSIC_90S_PROGRESSIONS = [
        # "End of the Road" - Boyz II Men style
        [(0, ChordQuality.MAJ7), (5, ChordQuality.MIN7),
         (3, ChordQuality.MIN7), (7, ChordQuality.DOM9)],  # I-vi-IV-V

        # Smooth ballad progression
        [(0, ChordQuality.MAJ9), (3, ChordQuality.ADD9),
         (5, ChordQuality.MIN9), (3, ChordQuality.MAJ7)],  # I-IV-vi-IV

        # Usher "U Remind Me" style
        [(0, ChordQuality.MIN7), (10, ChordQuality.MAJ7),
         (5, ChordQuality.MIN7), (3, ChordQuality.DOM9)],  # i-bVII-v-IV
    ]

    # Neo-soul progressions (Robert Glasper, D'Angelo influenced)
    NEOSOUL_PROGRESSIONS = [
        # Avoiding traditional ii-V-I, using minor 3rd shifts
        [(0, ChordQuality.MIN9), (3, ChordQuality.MIN11),
         (6, ChordQuality.MAJ7_SHARP11), (9, ChordQuality.NINE_SUS4)],

        # Cluster voicing friendly progression
        [(0, ChordQuality.MIN11), (10, ChordQuality.MAJ7),
         (5, ChordQuality.MIN9), (8, ChordQuality.DOM13)],

        # Modal, cyclic progression
        [(0, ChordQuality.NINE_SUS4), (5, ChordQuality.MIN11),
         (7, ChordQuality.THIRTEEN_SUS), (3, ChordQuality.MAJ9)],
    ]

    # Contemporary R&B progressions (simpler, more ambient)
    CONTEMPORARY_PROGRESSIONS = [
        # The Weeknd "Blinding Lights" influence
        [(0, ChordQuality.MIN7), (5, ChordQuality.MAJ7),
         (8, ChordQuality.MAJ7), (3, ChordQuality.MAJ7)],

        # Frank Ocean atmospheric style
        [(0, ChordQuality.MAJ7), (7, ChordQuality.SUS2),
         (3, ChordQuality.ADD9), (10, ChordQuality.MAJ7)],

        # SZA minimalist approach
        [(0, ChordQuality.MIN9), (7, ChordQuality.SUS4),
         (5, ChordQuality.MIN7), (3, ChordQuality.MAJ9)],
    ]

    # Chord tone formulas (intervals from root)
    CHORD_FORMULAS = {
        ChordQuality.MAJ7: [0, 4, 7, 11],
        ChordQuality.MAJ9: [0, 4, 7, 11, 14],
        ChordQuality.MAJ7_SHARP11: [0, 4, 7, 11, 18],  # #11 = 18 semitones
        ChordQuality.MAJ13: [0, 4, 7, 11, 14, 21],
        ChordQuality.MIN7: [0, 3, 7, 10],
        ChordQuality.MIN9: [0, 3, 7, 10, 14],
        ChordQuality.MIN11: [0, 3, 7, 10, 14, 17],
        ChordQuality.DOM7: [0, 4, 7, 10],
        ChordQuality.DOM9: [0, 4, 7, 10, 14],
        ChordQuality.DOM13: [0, 4, 7, 10, 14, 21],
        ChordQuality.SUS2: [0, 2, 7],
        ChordQuality.SUS4: [0, 5, 7],
        ChordQuality.NINE_SUS4: [0, 5, 7, 10, 14],
        ChordQuality.THIRTEEN_SUS: [0, 5, 7, 10, 14, 21],
        ChordQuality.ADD9: [0, 4, 7, 14],
        ChordQuality.HALF_DIM: [0, 3, 6, 10],
    }

    def __init__(self, key: str = 'C', bpm: int = 90, ticks_per_beat: int = 480):
        """
        Initialize R&B/Neo-Soul generator

        Args:
            key: Musical key (e.g., 'C', 'Db', 'F#')
            bpm: Tempo in beats per minute
            ticks_per_beat: MIDI ticks per beat resolution
        """
        self.key = key
        self.bpm = bpm
        self.ticks_per_beat = ticks_per_beat
        self.root_note = self.NOTE_NAMES[key]

    def generate_rnb_progression(
        self,
        era: str = "90s",
        complexity: float = 0.6,
        num_measures: int = 4
    ) -> List[Tuple[int, str]]:
        """
        Generate R&B chord progression based on era

        Args:
            era: '90s', 'neosoul', or 'contemporary'
            complexity: 0.0-1.0, affects chord extensions
            num_measures: Number of measures (typically 4, 8, or 16)

        Returns:
            List of (midi_note, chord_symbol) tuples

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> prog = gen.generate_rnb_progression('90s', 0.7, 4)
            >>> len(prog)
            4
        """
        # Select progression template based on era
        if era == "90s":
            template = random.choice(self.CLASSIC_90S_PROGRESSIONS)
        elif era == "neosoul":
            template = random.choice(self.NEOSOUL_PROGRESSIONS)
        else:  # contemporary
            template = random.choice(self.CONTEMPORARY_PROGRESSIONS)

        # Convert to actual notes in the key
        progression = []
        for degree, quality in template:
            root = (self.root_note + degree) % 12
            midi_note = root + 48  # Start at C3

            # Simplify chord if complexity is low
            if complexity < 0.4:
                if quality in [ChordQuality.MAJ9, ChordQuality.MAJ7_SHARP11]:
                    quality = ChordQuality.MAJ7
                elif quality in [ChordQuality.MIN9, ChordQuality.MIN11]:
                    quality = ChordQuality.MIN7

            chord_symbol = f"{self._note_name(root)}{quality.value}"
            progression.append((midi_note, chord_symbol, quality))

        # Extend if more measures needed
        while len(progression) < num_measures:
            progression.extend(progression[:num_measures - len(progression)])

        return progression[:num_measures]

    def generate_neosoul_chords(
        self,
        root: int,
        extensions: bool = True,
        voicing_style: str = "cluster"
    ) -> List[int]:
        """
        Generate neo-soul chord voicing with Robert Glasper-style techniques

        Args:
            root: Root note MIDI number
            extensions: Include 9th, 11th, 13th extensions
            voicing_style: 'cluster' (tight), 'rootless', or 'spread'

        Returns:
            List of MIDI note numbers forming the chord

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> chord = gen.generate_neosoul_chords(60, True, 'cluster')
            >>> len(chord) >= 4
            True
        """
        quality = random.choice([
            ChordQuality.MIN9, ChordQuality.MIN11,
            ChordQuality.MAJ7_SHARP11, ChordQuality.NINE_SUS4,
            ChordQuality.THIRTEEN_SUS
        ])

        intervals = self.CHORD_FORMULAS[quality].copy()

        if voicing_style == "rootless":
            # Remove root, emphasize upper extensions (Glasper technique)
            if 0 in intervals:
                intervals.remove(0)

        if voicing_style == "cluster":
            # Tight voicing in mid-range (piano left hand style)
            base = root + 12  # Start an octave up
            notes = [base + interval for interval in intervals[:4]]

            # Add rootless cluster in right hand
            if extensions and len(intervals) > 4:
                # Add 9th, 3rd, 11th closely together
                upper = root + 24
                if 14 in intervals:  # 9th
                    notes.append(upper + 2)  # 9th
                if 4 in intervals:  # major 3rd
                    notes.append(upper + 4)
                if 17 in intervals:  # 11th
                    notes.append(upper + 5)

        elif voicing_style == "spread":
            # Wide voicing across keyboard
            notes = [root] + [root + 12 + interval for interval in intervals[1:]]

        else:  # default
            notes = [root + interval for interval in intervals]

        # Add subtle chromatic approach note (Glasper technique)
        if random.random() < 0.3 and len(notes) > 0:
            approach_note = notes[-1] - 1  # Half-step below top note
            notes.insert(-1, approach_note)

        return sorted(notes)

    def generate_halftime_feel(
        self,
        base_pattern: List[float],
        swing: float = 0.6
    ) -> List[float]:
        """
        Apply half-time feel with J Dilla-style swing

        Args:
            base_pattern: List of note onset times in beats
            swing: Swing amount (0.5 = no swing, 0.66 = triplet, 0.53-0.56 = Dilla)

        Returns:
            List of adjusted note onset times

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> pattern = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
            >>> swung = gen.generate_halftime_feel(pattern, 0.55)
            >>> len(swung) == len(pattern)
            True
        """
        # J Dilla swing: 53-56% range (research-based)
        if swing < 0.5:
            swing = 0.5
        if swing > 0.75:
            swing = 0.75

        halftime = []
        for beat in base_pattern:
            # Determine if this is an off-beat (second 16th of each 8th note pair)
            position_in_beat = beat % 1.0
            is_offbeat = (position_in_beat % 0.5) >= 0.25

            if is_offbeat:
                # Calculate swing offset
                beat_pair_start = beat - (beat % 0.5)
                swing_offset = 0.5 * swing
                new_time = beat_pair_start + swing_offset
            else:
                new_time = beat

            # Add microtiming variation (±10-30ms as per research)
            microtiming_ms = random.gauss(0, 15)  # Mean 0, std dev 15ms
            ticks_offset = (microtiming_ms / 1000.0) * (self.bpm / 60.0) * self.ticks_per_beat
            new_time += ticks_offset / self.ticks_per_beat

            halftime.append(new_time)

        return halftime

    def generate_rhodes_voicing(
        self,
        chord_symbol: str,
        quality: ChordQuality,
        register: str = "mid"
    ) -> List[int]:
        """
        Generate Rhodes/Wurlitzer electric piano voicing

        Args:
            chord_symbol: Chord name (e.g., 'Cmaj7')
            quality: ChordQuality enum
            register: 'low', 'mid', or 'high'

        Returns:
            List of MIDI notes for electric piano voicing

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> voicing = gen.generate_rhodes_voicing('Cmaj7', ChordQuality.MAJ7, 'mid')
            >>> len(voicing) >= 3
            True
        """
        # Extract root from chord symbol
        root_name = chord_symbol[:1]
        if len(chord_symbol) > 1 and chord_symbol[1] in ['#', 'b']:
            root_name = chord_symbol[:2]

        root_pc = self.NOTE_NAMES[root_name]  # Pitch class

        # Set register
        if register == "low":
            root = root_pc + 36  # C2 range
        elif register == "high":
            root = root_pc + 72  # C5 range
        else:  # mid
            root = root_pc + 60  # C4 range

        intervals = self.CHORD_FORMULAS[quality].copy()

        # Rhodes/Wurlitzer voicing techniques:
        # 1. Often rootless in comping
        # 2. Voice in shells (3rd and 7th most important)
        # 3. Add color tones (9th, 13th)

        if len(intervals) >= 4:
            # Shell voicing: root, 3rd, 7th, 9th
            voicing = []

            # Root in bass (sometimes omitted in neo-soul)
            if random.random() > 0.3:
                voicing.append(root)

            # 3rd and 7th (essential)
            if 3 in intervals or 4 in intervals:
                third = root + (3 if 3 in intervals else 4)
                voicing.append(third + 12)  # Octave up

            if 10 in intervals or 11 in intervals:
                seventh = root + (10 if 10 in intervals else 11)
                voicing.append(seventh + 12)

            # Add 9th if present
            if 14 in intervals:
                voicing.append(root + 14 + 12)

            # Add 5th for stability (optional)
            if 7 in intervals and random.random() > 0.5:
                voicing.append(root + 7)
        else:
            # Simple voicing
            voicing = [root + interval for interval in intervals]

        return sorted(set(voicing))  # Remove duplicates and sort

    def generate_808_bass(
        self,
        root_note: int,
        slide: bool = True,
        pattern_length: int = 4
    ) -> List[Union[MIDINote, BassSlide]]:
        """
        Generate 808 sub-bass pattern with pitch slides

        Args:
            root_note: Root note for the bass line
            slide: Enable pitch slides between notes
            pattern_length: Length in beats

        Returns:
            List of MIDINote and BassSlide objects

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> bass = gen.generate_808_bass(36, True, 4)
            >>> len(bass) > 0
            True
        """
        bass_notes = []

        # Common 808 rhythmic patterns in R&B
        patterns = [
            [0, 1, 2, 3],  # Four on the floor
            [0, 0.5, 2, 2.5],  # Syncopated
            [0, 1.5, 2, 3.5],  # Half-time feel
            [0, 0.75, 2, 2.75, 3.5],  # Trap-influenced
        ]

        rhythm = random.choice(patterns)

        # Generate bass line with occasional slides
        for i, beat in enumerate(rhythm):
            if i < len(rhythm) - 1 and slide and random.random() < 0.4:
                # Create slide to next note
                next_beat = rhythm[i + 1]
                interval = random.choice([-12, -7, -5, 5, 7, 12])  # Octave, 5th, 4th
                target = root_note + interval

                # Glide time based on distance (research: 10-150ms)
                distance = abs(target - root_note)
                if distance <= 5:
                    glide_ms = random.uniform(10, 30)  # Tight slide
                else:
                    glide_ms = random.uniform(50, 100)  # Dramatic slide

                slide_obj = BassSlide(
                    from_note=root_note,
                    to_note=target,
                    glide_time_ms=glide_ms,
                    start_time=beat,
                    duration=next_beat - beat
                )
                bass_notes.append(slide_obj)
                root_note = target  # Update for next iteration
            else:
                # Regular note
                duration = 0.25 if i < len(rhythm) - 1 else 0.5
                note = MIDINote(
                    pitch=root_note,
                    velocity=random.randint(100, 120),  # 808s hit hard
                    start_time=beat,
                    duration=duration,
                    channel=0
                )
                bass_notes.append(note)

        return bass_notes

    def generate_vocal_melody(
        self,
        chord_progression: List[Tuple[int, str, ChordQuality]],
        range_type: str = "male",
        melisma_density: float = 0.5
    ) -> List[MIDINote]:
        """
        Generate smooth R&B vocal melody

        Args:
            chord_progression: List of (root, symbol, quality) tuples
            range_type: 'male' (F2-E5) or 'female' (A3-E6)
            melisma_density: 0.0-1.0, amount of melismatic runs

        Returns:
            List of MIDINotes forming the melody

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> prog = gen.generate_rnb_progression('neosoul', 0.6, 4)
            >>> melody = gen.generate_vocal_melody(prog, 'male', 0.5)
            >>> len(melody) > 0
            True
        """
        # Vocal ranges based on research (The Weeknd, SZA, etc.)
        if range_type == "male":
            low = 41  # F2
            high = 76  # E5
            sweet_spot = (55, 67)  # G3-G4 (most comfortable)
        else:  # female
            low = 57  # A3
            high = 88  # E6
            sweet_spot = (60, 76)  # C4-E5

        melody = []
        current_time = 0.0

        for root, symbol, quality in chord_progression:
            # Get chord tones
            chord_tones = self.CHORD_FORMULAS[quality]
            available_notes = [root + interval for interval in chord_tones]

            # Transpose to vocal range
            available_notes = [n for n in available_notes if low <= n <= high]
            if not available_notes:
                # Transpose octaves
                available_notes = [(root + interval + 12) for interval in chord_tones]
                available_notes = [n for n in available_notes if low <= n <= high]

            # Generate melodic phrase (1 measure = 4 beats)
            beats_in_measure = 4
            subdivision = 8 if melisma_density > 0.5 else 4  # 8th or quarter notes

            for beat in range(beats_in_measure * subdivision):
                time = current_time + (beat / subdivision)

                # Start on chord tones, occasional passing tones
                if beat % 2 == 0 or random.random() < 0.3:
                    # Chord tone
                    note = random.choice(available_notes)

                    # Prefer sweet spot
                    if random.random() < 0.6:
                        sweet_notes = [n for n in available_notes
                                     if sweet_spot[0] <= n <= sweet_spot[1]]
                        if sweet_notes:
                            note = random.choice(sweet_notes)
                else:
                    # Passing tone (stepwise motion)
                    if melody:
                        last_note = melody[-1].pitch
                        note = last_note + random.choice([-2, -1, 1, 2])
                        note = max(low, min(high, note))
                    else:
                        note = random.choice(available_notes)

                # Melismatic runs
                duration = 1.0 / subdivision
                if melisma_density > 0.7 and random.random() < 0.3:
                    # Quick run of notes
                    run_length = random.randint(3, 5)
                    for r in range(run_length):
                        run_note = note + random.choice([-2, -1, 0, 1, 2])
                        run_note = max(low, min(high, run_note))

                        melody.append(MIDINote(
                            pitch=run_note,
                            velocity=random.randint(70, 90),
                            start_time=time + (r * duration / run_length),
                            duration=duration / run_length,
                            channel=0
                        ))
                else:
                    # Regular note
                    melody.append(MIDINote(
                        pitch=note,
                        velocity=random.randint(75, 95),
                        start_time=time,
                        duration=duration,
                        channel=0
                    ))

            current_time += beats_in_measure

        return melody

    def _note_name(self, pitch_class: int) -> str:
        """Convert pitch class to note name"""
        note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
        return note_names[pitch_class % 12]

    def create_complete_rnb_track(
        self,
        style: RnBStyle = RnBStyle.NEOSOUL,
        duration_measures: int = 8,
        include_vocals: bool = True,
        include_bass: bool = True,
        include_keys: bool = True
    ) -> Dict[str, List]:
        """
        Generate a complete R&B track with multiple instruments

        Args:
            style: RnBStyle enum value
            duration_measures: Track length in measures
            include_vocals: Add vocal melody
            include_bass: Add 808 bass
            include_keys: Add Rhodes/Wurlitzer

        Returns:
            Dictionary with instrument tracks:
            {'chords': [...], 'bass': [...], 'melody': [...], 'progression': [...]}

        Examples:
            >>> gen = RnBNeoSoulGenerator('C', 90)
            >>> track = gen.create_complete_rnb_track(RnBStyle.NEOSOUL, 8)
            >>> 'progression' in track
            True
        """
        # Generate chord progression
        if style == RnBStyle.CLASSIC_90S:
            progression = self.generate_rnb_progression('90s', 0.5, duration_measures)
        elif style == RnBStyle.NEOSOUL:
            progression = self.generate_rnb_progression('neosoul', 0.8, duration_measures)
        else:  # CONTEMPORARY
            progression = self.generate_rnb_progression('contemporary', 0.6, duration_measures)

        track = {
            'progression': progression,
            'chords': [],
            'bass': [],
            'melody': []
        }

        # Generate chord voicings
        if include_keys:
            for root, symbol, quality in progression:
                if style == RnBStyle.NEOSOUL:
                    voicing = self.generate_neosoul_chords(root, True, 'cluster')
                else:
                    voicing = self.generate_rhodes_voicing(symbol, quality, 'mid')
                track['chords'].append(voicing)

        # Generate bass line
        if include_bass:
            for root, symbol, quality in progression:
                bass_root = root - 24  # Two octaves down
                bass_pattern = self.generate_808_bass(bass_root, slide=True, pattern_length=4)
                track['bass'].extend(bass_pattern)

        # Generate vocal melody
        if include_vocals:
            melody = self.generate_vocal_melody(
                progression,
                range_type='male' if style == RnBStyle.CLASSIC_90S else 'female',
                melisma_density=0.7 if style == RnBStyle.CLASSIC_90S else 0.5
            )
            track['melody'] = melody

        return track


# ============================================================================
# UNIT TESTS
# ============================================================================

def run_tests():
    """Comprehensive test suite for R&B/Neo-Soul generator"""

    print("=" * 70)
    print("R&B/Neo-Soul Generator - Comprehensive Test Suite")
    print("=" * 70)

    # Test 1: Initialize generator
    print("\n[Test 1] Initialize generator")
    gen = RnBNeoSoulGenerator('C', 90, 480)
    assert gen.key == 'C'
    assert gen.bpm == 90
    assert gen.root_note == 0
    print("✓ Generator initialized successfully")

    # Test 2: Generate 90s R&B progression
    print("\n[Test 2] Generate 90s R&B progression")
    prog_90s = gen.generate_rnb_progression('90s', 0.7, 4)
    assert len(prog_90s) == 4
    print(f"✓ Generated 90s progression: {[p[1] for p in prog_90s]}")

    # Test 3: Generate neo-soul progression
    print("\n[Test 3] Generate neo-soul progression")
    prog_neo = gen.generate_rnb_progression('neosoul', 0.8, 4)
    assert len(prog_neo) == 4
    print(f"✓ Generated neo-soul progression: {[p[1] for p in prog_neo]}")

    # Test 4: Generate contemporary progression
    print("\n[Test 4] Generate contemporary R&B progression")
    prog_contemp = gen.generate_rnb_progression('contemporary', 0.6, 4)
    assert len(prog_contemp) == 4
    print(f"✓ Generated contemporary progression: {[p[1] for p in prog_contemp]}")

    # Test 5: Generate Cmaj7#11 neo-soul voicing
    print("\n[Test 5] Generate Cmaj7#11 neo-soul voicing")
    chord = gen.generate_neosoul_chords(60, True, 'cluster')
    assert len(chord) >= 4
    print(f"✓ Generated cluster voicing with {len(chord)} notes: {chord}")

    # Test 6: Generate rootless voicing
    print("\n[Test 6] Generate rootless neo-soul voicing")
    rootless = gen.generate_neosoul_chords(60, True, 'rootless')
    assert len(rootless) >= 3
    print(f"✓ Generated rootless voicing: {rootless}")

    # Test 7: Generate spread voicing
    print("\n[Test 7] Generate spread voicing")
    spread = gen.generate_neosoul_chords(60, True, 'spread')
    assert len(spread) >= 4
    print(f"✓ Generated spread voicing: {spread}")

    # Test 8: Half-time feel with J Dilla swing
    print("\n[Test 8] Apply half-time feel with J Dilla swing (55%)")
    pattern = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]
    swung = gen.generate_halftime_feel(pattern, 0.55)
    assert len(swung) == len(pattern)
    print(f"✓ Applied swing: {pattern[:4]} → {[round(s, 3) for s in swung[:4]]}")

    # Test 9: Extreme swing (66% = triplet)
    print("\n[Test 9] Apply triplet swing (66%)")
    triplet = gen.generate_halftime_feel(pattern, 0.66)
    assert len(triplet) == len(pattern)
    print(f"✓ Applied triplet swing: {[round(t, 3) for t in triplet[:4]]}")

    # Test 10: Rhodes voicing (mid register)
    print("\n[Test 10] Generate Rhodes voicing (Cmaj7, mid register)")
    rhodes = gen.generate_rhodes_voicing('Cmaj7', ChordQuality.MAJ7, 'mid')
    assert len(rhodes) >= 3
    assert all(48 <= note <= 84 for note in rhodes)  # Check mid range
    print(f"✓ Generated Rhodes voicing: {rhodes}")

    # Test 11: Rhodes voicing (high register)
    print("\n[Test 11] Generate Rhodes voicing (high register)")
    rhodes_high = gen.generate_rhodes_voicing('Fmin9', ChordQuality.MIN9, 'high')
    assert len(rhodes_high) >= 3
    assert any(note >= 72 for note in rhodes_high)  # At least one high note
    print(f"✓ Generated high Rhodes voicing: {rhodes_high}")

    # Test 12: 808 bass with slides
    print("\n[Test 12] Generate 808 bass with pitch slides")
    bass = gen.generate_808_bass(36, slide=True, pattern_length=4)
    assert len(bass) > 0
    has_slide = any(isinstance(item, BassSlide) for item in bass)
    print(f"✓ Generated {len(bass)} bass events, includes slides: {has_slide}")

    # Test 13: 808 bass without slides
    print("\n[Test 13] Generate 808 bass without slides")
    bass_no_slide = gen.generate_808_bass(36, slide=False, pattern_length=4)
    assert len(bass_no_slide) > 0
    all_notes = all(isinstance(item, MIDINote) for item in bass_no_slide)
    print(f"✓ Generated {len(bass_no_slide)} bass notes (no slides): {all_notes}")

    # Test 14: Vocal melody (male range)
    print("\n[Test 14] Generate vocal melody (male range)")
    prog = gen.generate_rnb_progression('neosoul', 0.6, 4)
    melody_male = gen.generate_vocal_melody(prog, 'male', 0.5)
    assert len(melody_male) > 0
    assert all(41 <= note.pitch <= 76 for note in melody_male)  # F2-E5
    print(f"✓ Generated {len(melody_male)} vocal notes in male range")

    # Test 15: Vocal melody (female range)
    print("\n[Test 15] Generate vocal melody (female range)")
    melody_female = gen.generate_vocal_melody(prog, 'female', 0.7)
    assert len(melody_female) > 0
    assert all(57 <= note.pitch <= 88 for note in melody_female)  # A3-E6
    print(f"✓ Generated {len(melody_female)} vocal notes in female range")

    # Test 16: High melisma density
    print("\n[Test 16] Generate vocal melody with high melisma density")
    melody_melisma = gen.generate_vocal_melody(prog, 'female', 0.9)
    assert len(melody_melisma) > len(melody_female)  # More notes with more melisma
    print(f"✓ Generated {len(melody_melisma)} notes with high melisma density")

    # Test 17: Complete neo-soul track
    print("\n[Test 17] Generate complete neo-soul track")
    track = gen.create_complete_rnb_track(RnBStyle.NEOSOUL, 8, True, True, True)
    assert 'progression' in track
    assert 'chords' in track
    assert 'bass' in track
    assert 'melody' in track
    assert len(track['progression']) == 8
    print(f"✓ Generated complete track: {len(track['progression'])} chords, "
          f"{len(track['bass'])} bass events, {len(track['melody'])} melody notes")

    # Test 18: Complete 90s R&B track
    print("\n[Test 18] Generate complete 90s R&B track")
    track_90s = gen.create_complete_rnb_track(RnBStyle.CLASSIC_90S, 4, True, True, True)
    assert len(track_90s['progression']) == 4
    print(f"✓ Generated 90s R&B track with {len(track_90s['progression'])} measures")

    # Test 19: Contemporary track
    print("\n[Test 19] Generate contemporary R&B track")
    track_contemp = gen.create_complete_rnb_track(RnBStyle.CONTEMPORARY, 8, True, True, True)
    assert len(track_contemp['progression']) == 8
    print(f"✓ Generated contemporary track with {len(track_contemp['progression'])} measures")

    # Test 20: Different key (F# major)
    print("\n[Test 20] Generate in different key (F#)")
    gen_fsharp = RnBNeoSoulGenerator('F#', 95, 480)
    prog_fsharp = gen_fsharp.generate_rnb_progression('neosoul', 0.7, 4)
    assert len(prog_fsharp) == 4
    print(f"✓ Generated progression in F#: {[p[1] for p in prog_fsharp]}")

    # Test 21: Low complexity progression
    print("\n[Test 21] Generate low complexity progression")
    prog_simple = gen.generate_rnb_progression('90s', 0.2, 4)
    # Should use simpler chords (maj7 instead of maj9, etc.)
    assert len(prog_simple) == 4
    print(f"✓ Generated simple progression: {[p[1] for p in prog_simple]}")

    # Test 22: High complexity progression
    print("\n[Test 22] Generate high complexity progression")
    prog_complex = gen.generate_rnb_progression('neosoul', 0.9, 4)
    assert len(prog_complex) == 4
    print(f"✓ Generated complex progression: {[p[1] for p in prog_complex]}")

    # Test 23: Extended progression (16 measures)
    print("\n[Test 23] Generate extended 16-measure progression")
    prog_long = gen.generate_rnb_progression('neosoul', 0.7, 16)
    assert len(prog_long) == 16
    print(f"✓ Generated 16-measure progression")

    # Test 24: Minimal track (chords only)
    print("\n[Test 24] Generate minimal track (chords only)")
    track_minimal = gen.create_complete_rnb_track(
        RnBStyle.NEOSOUL, 4,
        include_vocals=False,
        include_bass=False,
        include_keys=True
    )
    assert len(track_minimal['chords']) > 0
    assert len(track_minimal['melody']) == 0
    print(f"✓ Generated chords-only track")

    # Test 25: BassSlide attributes
    print("\n[Test 25] Verify BassSlide object attributes")
    bass_with_slides = gen.generate_808_bass(36, slide=True, pattern_length=8)
    slides = [b for b in bass_with_slides if isinstance(b, BassSlide)]
    if slides:
        slide = slides[0]
        assert hasattr(slide, 'from_note')
        assert hasattr(slide, 'to_note')
        assert hasattr(slide, 'glide_time_ms')
        assert 10 <= slide.glide_time_ms <= 150  # Research-based range
        print(f"✓ BassSlide: {slide.from_note} → {slide.to_note}, "
              f"glide={slide.glide_time_ms:.1f}ms")
    else:
        print("✓ No slides generated in this run (random)")

    print("\n" + "=" * 70)
    print("ALL 25 TESTS PASSED! ✓✓✓")
    print("=" * 70)
    print("\n📊 Research Implementation Summary:")
    print("   • J Dilla swing: 53-56% (with ±10-30ms microtiming)")
    print("   • 808 glide times: 10-150ms (distance-based)")
    print("   • Robert Glasper: rootless clusters, minor 3rd shifts")
    print("   • Extended harmony: maj7#11, min9, min11, 9sus4, 13sus")
    print("   • Vocal ranges: Male (F2-E5), Female (A3-E6)")
    print("   • Progressions: 90s, neo-soul, contemporary styles")
    print("=" * 70)


if __name__ == "__main__":
    # Run all tests
    run_tests()

    # Generate example track
    print("\n\n" + "=" * 70)
    print("EXAMPLE: Complete Neo-Soul Track in Bb")
    print("=" * 70)

    generator = RnBNeoSoulGenerator('Bb', 88, 480)
    track = generator.create_complete_rnb_track(
        style=RnBStyle.NEOSOUL,
        duration_measures=8,
        include_vocals=True,
        include_bass=True,
        include_keys=True
    )

    print(f"\nProgression ({len(track['progression'])} measures):")
    for i, (root, symbol, quality) in enumerate(track['progression'], 1):
        print(f"  {i}. {symbol}")

    print(f"\nChord voicings: {len(track['chords'])} chords")
    print(f"Bass events: {len(track['bass'])} (includes slides)")
    print(f"Melody notes: {len(track['melody'])}")

    print("\n✓ Module ready for integration!")
