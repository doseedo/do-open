#!/usr/bin/env python3
"""
Intro/Outro Generator - Musical Introduction and Ending Generator
Part of the Ultimate MIDI Generation Library - Agent 10

This module provides comprehensive support for generating intros and outros
for big band arrangements and other musical forms.

Intro Styles:
- Vamp: Repeat I chord with rhythm section figures
- Last 4: Last 4 bars of the main progression
- Button: Short punchy hit (Count Basie style)
- Rubato: Free tempo, expressive introduction

Outro Styles:
- Tag: Repeat last 4 bars with ritardando
- Fermata: Sustained final chord
- Ritardando: Gradual slowdown
- Button: Short punchy ending (Count Basie style)

Research Sources:
- Count Basie "button" intros/endings
- Duke Ellington rubato introductions
- Standard big band arranging practices
- Mark Levine: The Jazz Theory Book

Author: Agent 10 - Form Structure Integrator
Date: 2025
"""

import random
import copy
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from genres.jazz import JazzChord, JazzNote
from analysis.midi_analyzer import NoteEvent, ChordEvent


# ============================================================================
# INTRO/OUTRO STYLES
# ============================================================================

class IntroStyle(Enum):
    """Introduction styles for big band arrangements"""
    VAMP = "vamp"                    # Repeat I chord with figures
    LAST_4 = "last_4"                # Last 4 bars of progression
    BUTTON = "button"                # Short punchy hit (Basie)
    RUBATO = "rubato"                # Free tempo, expressive


class OutroStyle(Enum):
    """Ending styles for big band arrangements"""
    TAG = "tag"                      # Repeat last 4 bars with ritardando
    FERMATA = "fermata"              # Sustained final chord
    RITARDANDO = "ritardando"        # Gradual slowdown
    BUTTON = "button"                # Short punchy ending (Basie)


# ============================================================================
# INTRO GENERATOR
# ============================================================================

class IntroOutroGenerator:
    """Generate introductions and endings for musical forms"""

    @staticmethod
    def generate_intro(
        progression: List[JazzChord],
        style: IntroStyle = IntroStyle.VAMP,
        length_bars: int = 4,
        tempo: int = 120,
        key: int = 0
    ) -> Dict:
        """
        Generate introduction section

        Args:
            progression: Main chord progression (for context)
            style: Type of intro to generate
            length_bars: Length of intro in bars
            tempo: Tempo in BPM
            key: Root key (0-11)

        Returns:
            Dictionary with:
                - intro_progression: List[JazzChord]
                - intro_notes: List[NoteEvent]
                - intro_rhythm: List[NoteEvent] (rhythm section)
                - duration_bars: int
        """
        if style == IntroStyle.VAMP:
            return IntroOutroGenerator._generate_vamp_intro(
                key, length_bars, tempo
            )
        elif style == IntroStyle.LAST_4:
            return IntroOutroGenerator._generate_last_4_intro(
                progression, tempo
            )
        elif style == IntroStyle.BUTTON:
            return IntroOutroGenerator._generate_button_intro(
                key, tempo
            )
        elif style == IntroStyle.RUBATO:
            return IntroOutroGenerator._generate_rubato_intro(
                key, length_bars
            )
        else:
            raise ValueError(f"Unknown intro style: {style}")

    @staticmethod
    def generate_ending(
        progression: List[JazzChord],
        style: OutroStyle = OutroStyle.TAG,
        length_bars: int = 4,
        tempo: int = 120
    ) -> Dict:
        """
        Generate ending section

        Args:
            progression: Main chord progression
            style: Type of ending to generate
            length_bars: Length of ending in bars
            tempo: Tempo in BPM

        Returns:
            Dictionary with:
                - outro_progression: List[JazzChord]
                - outro_notes: List[NoteEvent]
                - outro_rhythm: List[NoteEvent]
                - duration_bars: int
                - ritardando_factor: float (if applicable)
        """
        if style == OutroStyle.TAG:
            return IntroOutroGenerator._generate_tag_ending(
                progression, length_bars, tempo
            )
        elif style == OutroStyle.FERMATA:
            return IntroOutroGenerator._generate_fermata_ending(
                progression, tempo
            )
        elif style == OutroStyle.RITARDANDO:
            return IntroOutroGenerator._generate_ritardando_ending(
                progression, length_bars, tempo
            )
        elif style == OutroStyle.BUTTON:
            return IntroOutroGenerator._generate_button_ending(
                progression, tempo
            )
        else:
            raise ValueError(f"Unknown outro style: {style}")

    # ========================================================================
    # INTRO IMPLEMENTATIONS
    # ========================================================================

    @staticmethod
    def _generate_vamp_intro(
        key: int,
        length_bars: int,
        tempo: int
    ) -> Dict:
        """
        Generate vamp intro - repeat I chord with rhythm figures

        Classic intro style: rhythm section establishes groove on tonic
        """
        # Simple I chord vamp
        tonic_chord = JazzChord(root=key, quality="maj7")
        progression = [tonic_chord] * length_bars

        # Generate rhythmic figures
        notes = []
        current_time = 0.0

        for bar in range(length_bars):
            # Bass hits on 1 and 3
            bass_times = [bar * 4.0, bar * 4.0 + 2.0]
            for time in bass_times:
                note = NoteEvent(
                    start_time=time,
                    duration=0.5,
                    start_tick=int(time * 480),
                    duration_ticks=int(0.5 * 480),
                    pitch=key + 36,  # Bass register
                    velocity=80,
                    channel=1,
                    track_idx=0
                )
                notes.append(note)

            # Piano/brass hits on 2 and 4
            hit_times = [bar * 4.0 + 1.0, bar * 4.0 + 3.0]
            for time in hit_times:
                # Chord tones
                for pitch in [key + 48, key + 52, key + 55, key + 59]:
                    note = NoteEvent(
                        start_time=time,
                        duration=0.25,
                        start_tick=int(time * 480),
                        duration_ticks=int(0.25 * 480),
                        pitch=pitch,
                        velocity=85,
                        channel=0,
                        track_idx=1
                    )
                    notes.append(note)

        return {
            'intro_progression': progression,
            'intro_notes': notes,
            'intro_rhythm': notes,  # Same for vamp
            'duration_bars': length_bars,
            'style': IntroStyle.VAMP
        }

    @staticmethod
    def _generate_last_4_intro(
        progression: List[JazzChord],
        tempo: int
    ) -> Dict:
        """
        Generate intro using last 4 bars of progression

        Classic technique: intro = last 4 bars of tune
        """
        if len(progression) < 4:
            # If progression too short, repeat what we have
            intro_progression = progression * 2
            intro_progression = intro_progression[-4:]
        else:
            # Take last 4 chords
            intro_progression = progression[-4:]

        # Generate simple block chord hits
        notes = []
        for i, chord in enumerate(intro_progression):
            time = float(i * 4)  # 4 beats per bar

            # Whole note chord
            chord_tones = IntroOutroGenerator._get_chord_tones(chord, octave=4)
            for pitch in chord_tones:
                note = NoteEvent(
                    start_time=time,
                    duration=4.0,
                    start_tick=int(time * 480),
                    duration_ticks=int(4.0 * 480),
                    pitch=pitch,
                    velocity=75,
                    channel=0,
                    track_idx=0
                )
                notes.append(note)

        return {
            'intro_progression': intro_progression,
            'intro_notes': notes,
            'intro_rhythm': [],
            'duration_bars': 4,
            'style': IntroStyle.LAST_4
        }

    @staticmethod
    def _generate_button_intro(
        key: int,
        tempo: int
    ) -> Dict:
        """
        Generate Count Basie-style "button" intro

        Short, punchy hit to kick off the tune
        """
        # Single I chord hit
        tonic_chord = JazzChord(root=key, quality="maj7")
        progression = [tonic_chord]

        notes = []

        # Short punchy chord hit (1 beat)
        chord_tones = IntroOutroGenerator._get_chord_tones(tonic_chord, octave=4)
        for pitch in chord_tones:
            note = NoteEvent(
                start_time=0.0,
                duration=0.5,
                start_tick=0,
                duration_ticks=int(0.5 * 480),
                pitch=pitch,
                velocity=100,  # Loud and punchy
                channel=0,
                track_idx=0
            )
            notes.append(note)

        # Optional: add bass note
        bass_note = NoteEvent(
            start_time=0.0,
            duration=0.5,
            start_tick=0,
            duration_ticks=int(0.5 * 480),
            pitch=key + 36,
            velocity=100,
            channel=1,
            track_idx=1
        )
        notes.append(bass_note)

        return {
            'intro_progression': progression,
            'intro_notes': notes,
            'intro_rhythm': [],
            'duration_bars': 1,  # Just 1 beat, but count as 1 bar
            'style': IntroStyle.BUTTON
        }

    @staticmethod
    def _generate_rubato_intro(
        key: int,
        length_bars: int
    ) -> Dict:
        """
        Generate rubato (free tempo) intro

        Expressive, non-metered introduction (Duke Ellington style)
        """
        # Simple I chord
        tonic_chord = JazzChord(root=key, quality="maj7")
        progression = [tonic_chord]

        notes = []

        # Arpeggiate chord tones with expressive timing
        chord_tones = IntroOutroGenerator._get_chord_tones(tonic_chord, octave=4)

        # Rubato timing: irregular spacing
        timings = [0.0, 0.8, 1.8, 3.2]  # Irregular
        for i, pitch in enumerate(chord_tones[:4]):
            note = NoteEvent(
                start_time=timings[i],
                duration=2.0,  # Long sustain
                start_tick=int(timings[i] * 480),
                duration_ticks=int(2.0 * 480),
                pitch=pitch,
                velocity=60 + i * 5,  # Crescendo
                channel=0,
                track_idx=0
            )
            notes.append(note)

        return {
            'intro_progression': progression,
            'intro_notes': notes,
            'intro_rhythm': [],
            'duration_bars': length_bars,
            'style': IntroStyle.RUBATO,
            'is_rubato': True  # Flag for free tempo
        }

    # ========================================================================
    # OUTRO IMPLEMENTATIONS
    # ========================================================================

    @staticmethod
    def _generate_tag_ending(
        progression: List[JazzChord],
        length_bars: int,
        tempo: int
    ) -> Dict:
        """
        Generate tag ending - repeat last 4 bars with ritardando

        Classic ending: repeat final phrase with slowdown
        """
        if len(progression) < 4:
            tag_progression = progression * 2
            tag_progression = tag_progression[-4:]
        else:
            tag_progression = progression[-4:]

        notes = []

        # Generate notes with ritardando (tempo gradually slowing)
        for i, chord in enumerate(tag_progression):
            # Apply ritardando: each bar gets progressively slower
            ritardando_factor = 1.0 + (i * 0.15)  # 15% slower each bar
            time = sum([4.0 * (1.0 + j * 0.15) for j in range(i)])

            # Whole note chord
            chord_tones = IntroOutroGenerator._get_chord_tones(chord, octave=4)
            duration = 4.0 * ritardando_factor

            for pitch in chord_tones:
                note = NoteEvent(
                    start_time=time,
                    duration=duration,
                    start_tick=int(time * 480),
                    duration_ticks=int(duration * 480),
                    pitch=pitch,
                    velocity=70 - i * 5,  # Diminuendo
                    channel=0,
                    track_idx=0
                )
                notes.append(note)

        return {
            'outro_progression': tag_progression,
            'outro_notes': notes,
            'outro_rhythm': [],
            'duration_bars': 4,
            'ritardando_factor': 1.45,  # Final tempo factor
            'style': OutroStyle.TAG
        }

    @staticmethod
    def _generate_fermata_ending(
        progression: List[JazzChord],
        tempo: int
    ) -> Dict:
        """
        Generate fermata ending - sustained final chord

        Classic ending: hold the final chord
        """
        # Final chord (tonic)
        final_chord = progression[-1] if progression else JazzChord(root=0, quality="maj7")
        outro_progression = [final_chord]

        notes = []

        # Long sustained chord (fermata = hold indefinitely)
        chord_tones = IntroOutroGenerator._get_chord_tones(final_chord, octave=4)
        for pitch in chord_tones:
            note = NoteEvent(
                start_time=0.0,
                duration=8.0,  # Long sustain
                start_tick=0,
                duration_ticks=int(8.0 * 480),
                pitch=pitch,
                velocity=85,
                channel=0,
                track_idx=0
            )
            notes.append(note)

        return {
            'outro_progression': outro_progression,
            'outro_notes': notes,
            'outro_rhythm': [],
            'duration_bars': 2,  # Counts as 2 bars
            'is_fermata': True,
            'style': OutroStyle.FERMATA
        }

    @staticmethod
    def _generate_ritardando_ending(
        progression: List[JazzChord],
        length_bars: int,
        tempo: int
    ) -> Dict:
        """
        Generate ritardando ending - gradual slowdown

        Slow down over final bars
        """
        # Take last few chords
        ending_chords = progression[-length_bars:] if len(progression) >= length_bars else progression

        notes = []
        current_time = 0.0

        for i, chord in enumerate(ending_chords):
            # Progressive ritardando
            tempo_factor = 1.0 + (i * 0.2)  # 20% slower each bar
            bar_duration = 4.0 * tempo_factor

            # Chord voicing
            chord_tones = IntroOutroGenerator._get_chord_tones(chord, octave=4)
            for pitch in chord_tones:
                note = NoteEvent(
                    start_time=current_time,
                    duration=bar_duration,
                    start_tick=int(current_time * 480),
                    duration_ticks=int(bar_duration * 480),
                    pitch=pitch,
                    velocity=75 - i * 8,  # Diminuendo
                    channel=0,
                    track_idx=0
                )
                notes.append(note)

            current_time += bar_duration

        return {
            'outro_progression': ending_chords,
            'outro_notes': notes,
            'outro_rhythm': [],
            'duration_bars': length_bars,
            'ritardando_factor': 1.0 + (length_bars - 1) * 0.2,
            'style': OutroStyle.RITARDANDO
        }

    @staticmethod
    def _generate_button_ending(
        progression: List[JazzChord],
        tempo: int
    ) -> Dict:
        """
        Generate Count Basie-style "button" ending

        Short, punchy ending chord
        """
        # Final chord
        final_chord = progression[-1] if progression else JazzChord(root=0, quality="maj7")
        outro_progression = [final_chord]

        notes = []

        # Short punchy chord hit
        chord_tones = IntroOutroGenerator._get_chord_tones(final_chord, octave=4)
        for pitch in chord_tones:
            note = NoteEvent(
                start_time=0.0,
                duration=0.5,
                start_tick=0,
                duration_ticks=int(0.5 * 480),
                pitch=pitch,
                velocity=110,  # Very loud and punchy
                channel=0,
                track_idx=0
            )
            notes.append(note)

        # Bass note
        bass_note = NoteEvent(
            start_time=0.0,
            duration=0.5,
            start_tick=0,
            duration_ticks=int(0.5 * 480),
            pitch=final_chord.root + 36,
            velocity=110,
            channel=1,
            track_idx=1
        )
        notes.append(bass_note)

        return {
            'outro_progression': outro_progression,
            'outro_notes': notes,
            'outro_rhythm': [],
            'duration_bars': 1,
            'style': OutroStyle.BUTTON
        }

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    @staticmethod
    def _get_chord_tones(chord: JazzChord, octave: int = 4) -> List[int]:
        """
        Get MIDI note numbers for chord tones

        Args:
            chord: JazzChord object
            octave: Base octave (4 = middle C octave)

        Returns:
            List of MIDI note numbers
        """
        base = octave * 12 + chord.root
        tones = [base]  # Root

        # Add 3rd
        if 'min' in chord.quality or 'm' in chord.quality:
            tones.append(base + 3)  # Minor 3rd
        else:
            tones.append(base + 4)  # Major 3rd

        # Add 5th (unless diminished/augmented)
        if 'dim' in chord.quality or 'b5' in chord.quality:
            tones.append(base + 6)  # Diminished 5th
        elif 'aug' in chord.quality or '#5' in chord.quality:
            tones.append(base + 8)  # Augmented 5th
        else:
            tones.append(base + 7)  # Perfect 5th

        # Add 7th
        if 'maj7' in chord.quality or 'M7' in chord.quality:
            tones.append(base + 11)  # Major 7th
        elif '7' in chord.quality:
            tones.append(base + 10)  # Minor 7th
        elif '6' in chord.quality:
            tones.append(base + 9)  # Major 6th

        return sorted(tones)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🎵 INTRO/OUTRO GENERATOR - Big Band Intros and Endings\n")

    # Example progression (ii-V-I in C)
    test_progression = [
        JazzChord(root=2, quality="min7"),   # Dm7
        JazzChord(root=7, quality="dom7"),   # G7
        JazzChord(root=0, quality="maj7"),   # Cmaj7
    ]

    print("=" * 80)
    print("EXAMPLE 1: Vamp Intro")
    print("=" * 80)
    vamp_intro = IntroOutroGenerator.generate_intro(
        progression=test_progression,
        style=IntroStyle.VAMP,
        length_bars=4,
        tempo=140,
        key=0  # C
    )
    print(f"Style: {vamp_intro['style'].value}")
    print(f"Duration: {vamp_intro['duration_bars']} bars")
    print(f"Notes generated: {len(vamp_intro['intro_notes'])}")

    print("\n" + "=" * 80)
    print("EXAMPLE 2: Button Intro (Count Basie style)")
    print("=" * 80)
    button_intro = IntroOutroGenerator.generate_intro(
        progression=test_progression,
        style=IntroStyle.BUTTON,
        tempo=180,
        key=0
    )
    print(f"Style: {button_intro['style'].value}")
    print(f"Duration: {button_intro['duration_bars']} bar (short punch)")
    print(f"Notes generated: {len(button_intro['intro_notes'])}")

    print("\n" + "=" * 80)
    print("EXAMPLE 3: Tag Ending with Ritardando")
    print("=" * 80)
    tag_ending = IntroOutroGenerator.generate_ending(
        progression=test_progression,
        style=OutroStyle.TAG,
        length_bars=4,
        tempo=140
    )
    print(f"Style: {tag_ending['style'].value}")
    print(f"Duration: {tag_ending['duration_bars']} bars")
    print(f"Ritardando factor: {tag_ending.get('ritardando_factor', 1.0):.2f}")
    print(f"Notes generated: {len(tag_ending['outro_notes'])}")

    print("\n" + "=" * 80)
    print("EXAMPLE 4: Fermata Ending (Sustained)")
    print("=" * 80)
    fermata_ending = IntroOutroGenerator.generate_ending(
        progression=test_progression,
        style=OutroStyle.FERMATA,
        tempo=120
    )
    print(f"Style: {fermata_ending['style'].value}")
    print(f"Is fermata: {fermata_ending.get('is_fermata', False)}")
    print(f"Duration: {fermata_ending['duration_bars']} bars (sustained)")
    print(f"Notes generated: {len(fermata_ending['outro_notes'])}")

    print("\n✅ Intro/Outro Generator examples complete!")
    print("This module provides professional intros and endings for big band arrangements.\n")
