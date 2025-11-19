#!/usr/bin/env python3
"""
Form Generator - Musical Form & Structure Engine
Part of the Ultimate MIDI Generation Library

This module provides comprehensive support for generating complete musical forms:
- Classical Forms: Sonata, Rondo, Theme & Variations, Fugue
- Popular Forms: Verse-Chorus, AABA, 12-bar Blues
- Automatic section generation with proper key relationships

Author: Agent 5 - Form & Structure Engine
Research: Caplin's Classical Form, Hepokoski's Sonata Theory, Schoenberg's Fundamentals
"""

import random
from typing import List, Dict, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class FormType(Enum):
    """Enumeration of supported musical forms"""
    # Classical Forms
    SONATA = "sonata"
    RONDO = "rondo"
    THEME_AND_VARIATIONS = "theme_and_variations"
    FUGUE = "fugue"
    BINARY = "binary"
    TERNARY = "ternary"

    # Popular Forms
    VERSE_CHORUS = "verse_chorus"
    AABA = "aaba"
    TWELVE_BAR_BLUES = "twelve_bar_blues"
    VERSE_CHORUS_BRIDGE = "verse_chorus_bridge"
    THROUGH_COMPOSED = "through_composed"


class KeyRelationship(Enum):
    """Key relationships for modulation in forms"""
    TONIC = "tonic"
    DOMINANT = "dominant"
    SUBDOMINANT = "subdominant"
    RELATIVE_MAJOR = "relative_major"
    RELATIVE_MINOR = "relative_minor"
    PARALLEL_MAJOR = "parallel_major"
    PARALLEL_MINOR = "parallel_minor"
    MEDIANT = "mediant"
    SUBMEDIANT = "submediant"


@dataclass
class FormSection:
    """
    Represents a section within a musical form

    Attributes:
        name: Section identifier (e.g., 'exposition', 'verse', 'A')
        key_relationship: Relationship to tonic key
        length_bars: Number of bars in this section
        character: Musical character/mood
        thematic_material: Which theme(s) appear in this section
        development_level: 0-1 scale of how developed/fragmented the material is
        dynamic_level: Suggested dynamic level (0-1)
        texture_density: Suggested texture complexity (0-1)
    """
    name: str
    key_relationship: KeyRelationship
    length_bars: int
    character: str = "neutral"
    thematic_material: List[str] = None
    development_level: float = 0.0
    dynamic_level: float = 0.5
    texture_density: float = 0.5
    repeat: bool = False

    def __post_init__(self):
        if self.thematic_material is None:
            self.thematic_material = []


@dataclass
class MusicalForm:
    """
    Complete musical form structure

    Attributes:
        form_type: Type of form
        sections: List of FormSection objects in order
        tonic_key: Root note (MIDI number) of tonic key
        is_major: True for major, False for minor
        tempo: Tempo in BPM
        time_signature: Tuple of (numerator, denominator)
        total_bars: Total length in bars
    """
    form_type: FormType
    sections: List[FormSection]
    tonic_key: int = 60  # C
    is_major: bool = True
    tempo: int = 120
    time_signature: Tuple[int, int] = (4, 4)

    @property
    def total_bars(self) -> int:
        """Calculate total length from all sections"""
        total = 0
        for section in self.sections:
            multiplier = 2 if section.repeat else 1
            total += section.length_bars * multiplier
        return total

    def get_section_timeline(self) -> List[Tuple[int, int, FormSection]]:
        """
        Get timeline of sections with start and end bars

        Returns:
            List of (start_bar, end_bar, FormSection) tuples
        """
        timeline = []
        current_bar = 0

        for section in self.sections:
            start_bar = current_bar
            end_bar = current_bar + section.length_bars
            timeline.append((start_bar, end_bar, section))
            current_bar = end_bar

            # Handle repeats
            if section.repeat:
                start_bar = current_bar
                end_bar = current_bar + section.length_bars
                timeline.append((start_bar, end_bar, section))
                current_bar = end_bar

        return timeline


# ============================================================================
# KEY CALCULATION UTILITIES
# ============================================================================

def calculate_related_key(tonic: int, is_major: bool, relationship: KeyRelationship) -> Tuple[int, bool]:
    """
    Calculate a related key based on relationship

    Args:
        tonic: MIDI note number of tonic
        is_major: True if tonic key is major
        relationship: Desired key relationship

    Returns:
        Tuple of (new_tonic_midi_note, is_new_key_major)
    """
    if relationship == KeyRelationship.TONIC:
        return (tonic, is_major)

    elif relationship == KeyRelationship.DOMINANT:
        return (tonic + 7, is_major)  # Perfect 5th up

    elif relationship == KeyRelationship.SUBDOMINANT:
        return (tonic + 5, is_major)  # Perfect 4th up

    elif relationship == KeyRelationship.RELATIVE_MAJOR:
        if is_major:
            return (tonic, is_major)  # Already major
        else:
            return (tonic + 3, True)  # Minor 3rd up to relative major

    elif relationship == KeyRelationship.RELATIVE_MINOR:
        if not is_major:
            return (tonic, is_major)  # Already minor
        else:
            return (tonic - 3, False)  # Minor 3rd down to relative minor

    elif relationship == KeyRelationship.PARALLEL_MAJOR:
        return (tonic, True)

    elif relationship == KeyRelationship.PARALLEL_MINOR:
        return (tonic, False)

    elif relationship == KeyRelationship.MEDIANT:
        if is_major:
            return (tonic + 4, False)  # Major 3rd up, minor key
        else:
            return (tonic + 3, True)  # Minor 3rd up, major key

    elif relationship == KeyRelationship.SUBMEDIANT:
        if is_major:
            return (tonic + 9, False)  # Major 6th up, minor key
        else:
            return (tonic + 8, True)  # Minor 6th up, major key

    else:
        return (tonic, is_major)


# ============================================================================
# SONATA FORM GENERATOR
# ============================================================================

class SonataFormGenerator:
    """
    Generate Sonata Form structure with proper key relationships

    Classical sonata form:
    - EXPOSITION: 1st theme (tonic) → Transition → 2nd theme (dominant/relative major)
    - DEVELOPMENT: Fragmentation, modulation, sequence
    - RECAPITULATION: 1st theme (tonic) → 2nd theme (tonic)
    - Optional: Introduction, Coda
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 120,
        exposition_length: int = 32,
        development_length: int = 24,
        recapitulation_length: int = 32,
        include_introduction: bool = False,
        include_coda: bool = True
    ) -> MusicalForm:
        """
        Generate a complete sonata form structure

        Args:
            tonic_key: Tonic pitch (MIDI note)
            is_major: True for major key
            tempo: Tempo in BPM
            exposition_length: Bars for exposition
            development_length: Bars for development
            recapitulation_length: Bars for recapitulation
            include_introduction: Add slow introduction
            include_coda: Add coda at end

        Returns:
            MusicalForm object with complete structure
        """
        sections = []

        # Optional Introduction (slow, mysterious)
        if include_introduction:
            sections.append(FormSection(
                name="Introduction",
                key_relationship=KeyRelationship.TONIC,
                length_bars=8,
                character="mysterious, slow",
                thematic_material=["intro_motif"],
                development_level=0.0,
                dynamic_level=0.3,
                texture_density=0.3
            ))

        # EXPOSITION
        # First theme group (tonic)
        first_theme_bars = exposition_length // 3
        sections.append(FormSection(
            name="Exposition - First Theme",
            key_relationship=KeyRelationship.TONIC,
            length_bars=first_theme_bars,
            character="confident, assertive",
            thematic_material=["theme_1"],
            development_level=0.0,
            dynamic_level=0.7,
            texture_density=0.6,
            repeat=False
        ))

        # Transition (modulating)
        transition_bars = exposition_length // 6
        transition_key = KeyRelationship.DOMINANT if is_major else KeyRelationship.RELATIVE_MAJOR
        sections.append(FormSection(
            name="Exposition - Transition",
            key_relationship=transition_key,
            length_bars=transition_bars,
            character="unstable, modulatory",
            thematic_material=["theme_1", "transition_material"],
            development_level=0.3,
            dynamic_level=0.6,
            texture_density=0.7
        ))

        # Second theme group (dominant or relative major)
        second_theme_bars = exposition_length // 2
        sections.append(FormSection(
            name="Exposition - Second Theme",
            key_relationship=transition_key,
            length_bars=second_theme_bars,
            character="lyrical, contrasting",
            thematic_material=["theme_2"],
            development_level=0.0,
            dynamic_level=0.5,
            texture_density=0.5,
            repeat=False
        ))

        # Exposition repeat (traditional)
        # Note: In performance, the entire exposition is played twice
        # We mark this with a special section
        sections.append(FormSection(
            name="Exposition Repeat",
            key_relationship=KeyRelationship.TONIC,
            length_bars=0,  # Virtual marker
            character="repeat",
            thematic_material=[],
            development_level=0.0
        ))

        # DEVELOPMENT
        # Fragmentation and modulation through various keys
        development_sections = SonataFormGenerator._generate_development_sections(
            development_length, is_major
        )
        sections.extend(development_sections)

        # RECAPITULATION
        # First theme (tonic)
        sections.append(FormSection(
            name="Recapitulation - First Theme",
            key_relationship=KeyRelationship.TONIC,
            length_bars=first_theme_bars,
            character="confident, return home",
            thematic_material=["theme_1"],
            development_level=0.0,
            dynamic_level=0.8,
            texture_density=0.6
        ))

        # Transition (staying in tonic this time)
        sections.append(FormSection(
            name="Recapitulation - Transition",
            key_relationship=KeyRelationship.TONIC,
            length_bars=transition_bars,
            character="stable, leading to second theme",
            thematic_material=["theme_1", "transition_material"],
            development_level=0.2,
            dynamic_level=0.7,
            texture_density=0.7
        ))

        # Second theme (NOW IN TONIC - this is the key point!)
        sections.append(FormSection(
            name="Recapitulation - Second Theme",
            key_relationship=KeyRelationship.TONIC,
            length_bars=second_theme_bars,
            character="lyrical, resolved to tonic",
            thematic_material=["theme_2"],
            development_level=0.0,
            dynamic_level=0.6,
            texture_density=0.5
        ))

        # Optional Coda
        if include_coda:
            sections.append(FormSection(
                name="Coda",
                key_relationship=KeyRelationship.TONIC,
                length_bars=12,
                character="triumphant, conclusive",
                thematic_material=["theme_1", "theme_2"],
                development_level=0.4,
                dynamic_level=0.9,
                texture_density=0.8
            ))

        return MusicalForm(
            form_type=FormType.SONATA,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )

    @staticmethod
    def _generate_development_sections(length_bars: int, is_major: bool) -> List[FormSection]:
        """Generate development section with modulations"""
        sections = []

        # Development typically moves through several keys
        development_keys = [
            (KeyRelationship.SUBDOMINANT, "exploring subdominant"),
            (KeyRelationship.SUBMEDIANT, "dark, remote"),
            (KeyRelationship.MEDIANT, "unstable, seeking"),
            (KeyRelationship.DOMINANT, "building tension"),
        ]

        bars_per_subsection = length_bars // len(development_keys)

        for i, (key_rel, character) in enumerate(development_keys):
            sections.append(FormSection(
                name=f"Development - Phase {i+1}",
                key_relationship=key_rel,
                length_bars=bars_per_subsection,
                character=character,
                thematic_material=["theme_1_fragment", "theme_2_fragment"],
                development_level=0.7 + (i * 0.1),  # Increasing intensity
                dynamic_level=0.5 + (i * 0.1),
                texture_density=0.7 + (i * 0.05)
            ))

        return sections


# ============================================================================
# RONDO FORM GENERATOR
# ============================================================================

class RondoFormGenerator:
    """
    Generate Rondo Form: ABACA or ABACABA

    Rondo characteristics:
    - A section (refrain) returns in tonic
    - B, C sections (episodes) in contrasting keys
    - Creates a sense of return and journey
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 120,
        pattern: str = "ABACA",
        section_length: int = 8
    ) -> MusicalForm:
        """
        Generate Rondo form structure

        Args:
            tonic_key: Tonic pitch
            is_major: Major or minor
            tempo: Tempo in BPM
            pattern: Rondo pattern ('ABACA', 'ABACABA', etc.)
            section_length: Bars per section

        Returns:
            MusicalForm with rondo structure
        """
        sections = []

        # Define key relationships for episodes
        episode_keys = {
            'A': KeyRelationship.TONIC,  # Refrain always in tonic
            'B': KeyRelationship.DOMINANT,
            'C': KeyRelationship.SUBDOMINANT,
            'D': KeyRelationship.RELATIVE_MINOR if is_major else KeyRelationship.RELATIVE_MAJOR,
        }

        # Define characters
        characters = {
            'A': "joyful, main theme",
            'B': "contrasting, playful",
            'C': "lyrical, calm",
            'D': "dramatic, intense"
        }

        for i, letter in enumerate(pattern):
            key_rel = episode_keys.get(letter, KeyRelationship.TONIC)
            character = characters.get(letter, "neutral")

            section_type = "Refrain" if letter == 'A' else "Episode"

            sections.append(FormSection(
                name=f"{section_type} {letter}",
                key_relationship=key_rel,
                length_bars=section_length,
                character=character,
                thematic_material=[f"theme_{letter}"],
                development_level=0.0 if letter == 'A' else 0.3,
                dynamic_level=0.7 if letter == 'A' else 0.5,
                texture_density=0.6
            ))

        return MusicalForm(
            form_type=FormType.RONDO,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )


# ============================================================================
# THEME AND VARIATIONS GENERATOR
# ============================================================================

class ThemeAndVariationsGenerator:
    """
    Generate Theme and Variations form

    Structure:
    - Original theme (simple, clear)
    - Variation 1: Melodic ornamentation
    - Variation 2: Harmonic recoloring
    - Variation 3: Rhythmic transformation
    - Variation 4: Textural change
    - Variation 5: Mode change (major ↔ minor)
    - Variation 6: Tempo change
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 120,
        theme_length: int = 16,
        num_variations: int = 6
    ) -> MusicalForm:
        """
        Generate Theme and Variations structure

        Args:
            tonic_key: Tonic pitch
            is_major: Major or minor
            tempo: Base tempo
            theme_length: Bars for theme
            num_variations: Number of variations

        Returns:
            MusicalForm with theme and variations
        """
        sections = []

        # Original Theme
        sections.append(FormSection(
            name="Theme",
            key_relationship=KeyRelationship.TONIC,
            length_bars=theme_length,
            character="simple, clear, memorable",
            thematic_material=["original_theme"],
            development_level=0.0,
            dynamic_level=0.6,
            texture_density=0.4
        ))

        # Variations
        variation_types = [
            ("Melodic Ornamentation", "elaborate, decorated", 0.3, 0.6, 0.5),
            ("Harmonic Recoloring", "rich harmonies, reharmonized", 0.2, 0.6, 0.6),
            ("Rhythmic Transformation", "rhythmic energy, syncopated", 0.4, 0.7, 0.7),
            ("Textural Change", "complex texture, polyphonic", 0.3, 0.5, 0.8),
            ("Mode Change", "emotional shift, modal", 0.2, 0.5, 0.5),
            ("Tempo/Character", "different character, transformed", 0.5, 0.7, 0.6),
        ]

        for i in range(num_variations):
            if i < len(variation_types):
                var_name, character, dev_level, dyn_level, texture = variation_types[i]
            else:
                var_name = f"Variation {i+1}"
                character = "transformed"
                dev_level, dyn_level, texture = 0.4, 0.6, 0.6

            # Mode change variation goes to parallel key
            if i == 4:  # Mode change variation
                key_rel = KeyRelationship.PARALLEL_MINOR if is_major else KeyRelationship.PARALLEL_MAJOR
            else:
                key_rel = KeyRelationship.TONIC

            sections.append(FormSection(
                name=f"Variation {i+1}: {var_name}",
                key_relationship=key_rel,
                length_bars=theme_length,
                character=character,
                thematic_material=["original_theme", f"variation_{i+1}"],
                development_level=dev_level,
                dynamic_level=dyn_level,
                texture_density=texture
            ))

        # Optional final variation - grand coda
        sections.append(FormSection(
            name="Final Variation & Coda",
            key_relationship=KeyRelationship.TONIC,
            length_bars=theme_length + 8,
            character="brilliant, conclusive",
            thematic_material=["original_theme", "all_variations"],
            development_level=0.6,
            dynamic_level=0.9,
            texture_density=0.9
        ))

        return MusicalForm(
            form_type=FormType.THEME_AND_VARIATIONS,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )


# ============================================================================
# FUGUE GENERATOR
# ============================================================================

class FugueGenerator:
    """
    Generate Fugue structure

    Fugue components:
    - Exposition: Subject → Answer → Subject → Answer (voices enter successively)
    - Episodes: Sequential development, modulation
    - Middle Entries: Subject returns in related keys
    - Final Section: Stretto, augmentation, return to tonic
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 100,
        num_voices: int = 4,
        subject_length: int = 4
    ) -> MusicalForm:
        """
        Generate Fugue structure

        Args:
            tonic_key: Tonic pitch
            is_major: Major or minor
            tempo: Tempo (fugues typically moderate)
            num_voices: Number of voices (3-4 typical)
            subject_length: Bars for subject

        Returns:
            MusicalForm with fugue structure
        """
        sections = []

        # EXPOSITION - Voices enter one by one
        for i in range(num_voices):
            voice_num = i + 1
            # Alternate: Subject (tonic) → Answer (dominant) → Subject → Answer
            is_answer = (i % 2 == 1)
            key_rel = KeyRelationship.DOMINANT if is_answer else KeyRelationship.TONIC
            entry_type = "Answer" if is_answer else "Subject"

            sections.append(FormSection(
                name=f"Exposition - Voice {voice_num} ({entry_type})",
                key_relationship=key_rel,
                length_bars=subject_length,
                character=f"voice {voice_num} enters",
                thematic_material=["subject"],
                development_level=0.0,
                dynamic_level=0.4 + (i * 0.1),
                texture_density=0.2 + (i * 0.15)
            ))

        # EPISODE 1 - Sequential development
        sections.append(FormSection(
            name="Episode 1",
            key_relationship=KeyRelationship.SUBDOMINANT,
            length_bars=8,
            character="sequential development",
            thematic_material=["subject_fragment", "countersubject"],
            development_level=0.6,
            dynamic_level=0.5,
            texture_density=0.7
        ))

        # MIDDLE ENTRY 1
        sections.append(FormSection(
            name="Middle Entry 1",
            key_relationship=KeyRelationship.RELATIVE_MINOR if is_major else KeyRelationship.RELATIVE_MAJOR,
            length_bars=subject_length * 2,
            character="subject returns in relative key",
            thematic_material=["subject"],
            development_level=0.2,
            dynamic_level=0.6,
            texture_density=0.8
        ))

        # EPISODE 2
        sections.append(FormSection(
            name="Episode 2",
            key_relationship=KeyRelationship.SUBMEDIANT,
            length_bars=8,
            character="more intense development",
            thematic_material=["subject_fragment", "countersubject"],
            development_level=0.7,
            dynamic_level=0.6,
            texture_density=0.8
        ))

        # MIDDLE ENTRY 2
        sections.append(FormSection(
            name="Middle Entry 2",
            key_relationship=KeyRelationship.SUBDOMINANT,
            length_bars=subject_length * 2,
            character="building tension",
            thematic_material=["subject"],
            development_level=0.3,
            dynamic_level=0.7,
            texture_density=0.8
        ))

        # STRETTO - Overlapping entries
        sections.append(FormSection(
            name="Stretto",
            key_relationship=KeyRelationship.DOMINANT,
            length_bars=8,
            character="climactic overlapping entries",
            thematic_material=["subject", "subject_stretto"],
            development_level=0.8,
            dynamic_level=0.8,
            texture_density=0.9
        ))

        # FINAL ENTRY - Return to tonic
        sections.append(FormSection(
            name="Final Entry",
            key_relationship=KeyRelationship.TONIC,
            length_bars=subject_length * 2,
            character="triumphant return",
            thematic_material=["subject"],
            development_level=0.4,
            dynamic_level=0.9,
            texture_density=0.9
        ))

        # CODA - Optional augmentation
        sections.append(FormSection(
            name="Coda",
            key_relationship=KeyRelationship.TONIC,
            length_bars=8,
            character="conclusive, possibly augmented",
            thematic_material=["subject", "augmentation"],
            development_level=0.5,
            dynamic_level=0.9,
            texture_density=0.8
        ))

        return MusicalForm(
            form_type=FormType.FUGUE,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )


# ============================================================================
# POPULAR SONG FORMS
# ============================================================================

class VerseChorusGenerator:
    """
    Generate Verse-Chorus song form

    Common pop song structure:
    - Intro
    - Verse 1
    - Chorus
    - Verse 2
    - Chorus
    - Bridge
    - Chorus (final, often with variations)
    - Outro
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 120,
        verse_length: int = 8,
        chorus_length: int = 8,
        include_bridge: bool = True,
        num_verses: int = 2
    ) -> MusicalForm:
        """
        Generate Verse-Chorus form

        Args:
            tonic_key: Tonic pitch
            is_major: Major or minor
            tempo: Tempo in BPM
            verse_length: Bars per verse
            chorus_length: Bars per chorus
            include_bridge: Include bridge section
            num_verses: Number of verses (2-3 typical)

        Returns:
            MusicalForm with verse-chorus structure
        """
        sections = []

        # Intro
        sections.append(FormSection(
            name="Intro",
            key_relationship=KeyRelationship.TONIC,
            length_bars=4,
            character="establishing mood",
            thematic_material=["intro_hook"],
            development_level=0.0,
            dynamic_level=0.5,
            texture_density=0.4
        ))

        # Verses and Choruses
        for i in range(num_verses):
            # Verse
            sections.append(FormSection(
                name=f"Verse {i+1}",
                key_relationship=KeyRelationship.TONIC,
                length_bars=verse_length,
                character="storytelling, building",
                thematic_material=[f"verse"],
                development_level=0.1,
                dynamic_level=0.5 + (i * 0.1),
                texture_density=0.5
            ))

            # Pre-Chorus (optional, before chorus)
            if i == 0:  # Add pre-chorus before first chorus
                sections.append(FormSection(
                    name="Pre-Chorus",
                    key_relationship=KeyRelationship.TONIC,
                    length_bars=4,
                    character="building anticipation",
                    thematic_material=["pre_chorus"],
                    development_level=0.2,
                    dynamic_level=0.6,
                    texture_density=0.6
                ))

            # Chorus
            sections.append(FormSection(
                name=f"Chorus {i+1}",
                key_relationship=KeyRelationship.TONIC,
                length_bars=chorus_length,
                character="hook, memorable, emotional peak",
                thematic_material=["chorus"],
                development_level=0.0,
                dynamic_level=0.8,
                texture_density=0.7
            ))

        # Bridge (contrasting section)
        if include_bridge:
            bridge_key = KeyRelationship.SUBDOMINANT if is_major else KeyRelationship.RELATIVE_MAJOR
            sections.append(FormSection(
                name="Bridge",
                key_relationship=bridge_key,
                length_bars=8,
                character="contrast, new perspective",
                thematic_material=["bridge"],
                development_level=0.4,
                dynamic_level=0.6,
                texture_density=0.6
            ))

        # Final Chorus (often repeated or extended)
        sections.append(FormSection(
            name="Chorus (Final)",
            key_relationship=KeyRelationship.TONIC,
            length_bars=chorus_length,
            character="climactic, final statement",
            thematic_material=["chorus", "chorus_variation"],
            development_level=0.2,
            dynamic_level=0.9,
            texture_density=0.8
        ))

        # Optional repeat of final chorus
        sections.append(FormSection(
            name="Chorus (Repeat)",
            key_relationship=KeyRelationship.TONIC,
            length_bars=chorus_length,
            character="reinforcing hook",
            thematic_material=["chorus"],
            development_level=0.0,
            dynamic_level=0.9,
            texture_density=0.8
        ))

        # Outro
        sections.append(FormSection(
            name="Outro",
            key_relationship=KeyRelationship.TONIC,
            length_bars=4,
            character="fading, conclusive",
            thematic_material=["chorus_fragment", "intro_hook"],
            development_level=0.3,
            dynamic_level=0.5,
            texture_density=0.4
        ))

        return MusicalForm(
            form_type=FormType.VERSE_CHORUS,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )


class AABAGenerator:
    """
    Generate AABA form (32-bar song form)

    Classic jazz standard form:
    - A section (8 bars): Main theme in tonic
    - A section (8 bars): Repeat of main theme
    - B section (8 bars): Bridge in contrasting key
    - A section (8 bars): Return of main theme
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 120,
        section_length: int = 8
    ) -> MusicalForm:
        """
        Generate AABA (32-bar) form

        Args:
            tonic_key: Tonic pitch
            is_major: Major or minor
            tempo: Tempo in BPM
            section_length: Bars per section (typically 8)

        Returns:
            MusicalForm with AABA structure
        """
        sections = []

        # A section (first statement)
        sections.append(FormSection(
            name="A1",
            key_relationship=KeyRelationship.TONIC,
            length_bars=section_length,
            character="main theme, establishing",
            thematic_material=["theme_A"],
            development_level=0.0,
            dynamic_level=0.6,
            texture_density=0.5
        ))

        # A section (repeat)
        sections.append(FormSection(
            name="A2",
            key_relationship=KeyRelationship.TONIC,
            length_bars=section_length,
            character="main theme repeated",
            thematic_material=["theme_A"],
            development_level=0.0,
            dynamic_level=0.6,
            texture_density=0.5
        ))

        # B section (bridge/release)
        bridge_key = KeyRelationship.SUBDOMINANT
        sections.append(FormSection(
            name="B (Bridge)",
            key_relationship=bridge_key,
            length_bars=section_length,
            character="contrasting, release",
            thematic_material=["theme_B"],
            development_level=0.3,
            dynamic_level=0.7,
            texture_density=0.6
        ))

        # A section (return)
        sections.append(FormSection(
            name="A3 (Return)",
            key_relationship=KeyRelationship.TONIC,
            length_bars=section_length,
            character="return of theme, conclusive",
            thematic_material=["theme_A"],
            development_level=0.0,
            dynamic_level=0.7,
            texture_density=0.6
        ))

        return MusicalForm(
            form_type=FormType.AABA,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )


class TwelveBarBluesGenerator:
    """
    Generate 12-bar blues form

    Standard 12-bar blues progression structure:
    Bars 1-4: I chord (tonic)
    Bars 5-6: IV chord (subdominant)
    Bars 7-8: I chord (tonic)
    Bar 9: V chord (dominant)
    Bar 10: IV chord (subdominant)
    Bars 11-12: I chord (tonic) with turnaround
    """

    @staticmethod
    def generate(
        tonic_key: int = 60,
        is_major: bool = False,  # Blues typically minor-ish
        tempo: int = 120,
        num_choruses: int = 3
    ) -> MusicalForm:
        """
        Generate 12-bar blues form

        Args:
            tonic_key: Tonic pitch
            is_major: Major or minor (blues is usually minor-based)
            tempo: Tempo in BPM (shuffle feel)
            num_choruses: Number of 12-bar choruses

        Returns:
            MusicalForm with 12-bar blues structure
        """
        sections = []

        # Generate multiple choruses
        for chorus_num in range(num_choruses):
            # Tonic (bars 1-4)
            sections.append(FormSection(
                name=f"Chorus {chorus_num+1} - I (bars 1-4)",
                key_relationship=KeyRelationship.TONIC,
                length_bars=4,
                character="establishing blues feel",
                thematic_material=[f"blues_theme_{chorus_num+1}"],
                development_level=0.1 * chorus_num,
                dynamic_level=0.6 + (0.1 * chorus_num),
                texture_density=0.5
            ))

            # Subdominant (bars 5-6)
            sections.append(FormSection(
                name=f"Chorus {chorus_num+1} - IV (bars 5-6)",
                key_relationship=KeyRelationship.SUBDOMINANT,
                length_bars=2,
                character="subdominant lift",
                thematic_material=[f"blues_theme_{chorus_num+1}"],
                development_level=0.1 * chorus_num,
                dynamic_level=0.7 + (0.1 * chorus_num),
                texture_density=0.6
            ))

            # Back to Tonic (bars 7-8)
            sections.append(FormSection(
                name=f"Chorus {chorus_num+1} - I (bars 7-8)",
                key_relationship=KeyRelationship.TONIC,
                length_bars=2,
                character="back to tonic",
                thematic_material=[f"blues_theme_{chorus_num+1}"],
                development_level=0.1 * chorus_num,
                dynamic_level=0.7 + (0.1 * chorus_num),
                texture_density=0.6
            ))

            # Dominant (bar 9)
            sections.append(FormSection(
                name=f"Chorus {chorus_num+1} - V (bar 9)",
                key_relationship=KeyRelationship.DOMINANT,
                length_bars=1,
                character="tension, dominant",
                thematic_material=[f"blues_theme_{chorus_num+1}"],
                development_level=0.2 * chorus_num,
                dynamic_level=0.8 + (0.1 * chorus_num),
                texture_density=0.7
            ))

            # Subdominant (bar 10)
            sections.append(FormSection(
                name=f"Chorus {chorus_num+1} - IV (bar 10)",
                key_relationship=KeyRelationship.SUBDOMINANT,
                length_bars=1,
                character="subdominant",
                thematic_material=[f"blues_theme_{chorus_num+1}"],
                development_level=0.2 * chorus_num,
                dynamic_level=0.7 + (0.1 * chorus_num),
                texture_density=0.6
            ))

            # Turnaround (bars 11-12)
            sections.append(FormSection(
                name=f"Chorus {chorus_num+1} - I-V (bars 11-12)",
                key_relationship=KeyRelationship.TONIC,
                length_bars=2,
                character="turnaround to next chorus",
                thematic_material=[f"blues_theme_{chorus_num+1}", "turnaround"],
                development_level=0.2 * chorus_num,
                dynamic_level=0.6 + (0.1 * chorus_num),
                texture_density=0.5
            ))

        return MusicalForm(
            form_type=FormType.TWELVE_BAR_BLUES,
            sections=sections,
            tonic_key=tonic_key,
            is_major=is_major,
            tempo=tempo,
            time_signature=(4, 4)
        )


# ============================================================================
# MAIN FORM GENERATOR CLASS
# ============================================================================

class FormGenerator:
    """
    Main form generator class - high-level API for generating musical forms
    """

    @staticmethod
    def generate_form(
        form_type: FormType,
        tonic_key: int = 60,
        is_major: bool = True,
        tempo: int = 120,
        **kwargs
    ) -> MusicalForm:
        """
        Generate a musical form

        Args:
            form_type: Type of form to generate
            tonic_key: Tonic pitch (MIDI note)
            is_major: True for major, False for minor
            tempo: Tempo in BPM
            **kwargs: Additional arguments specific to form type

        Returns:
            MusicalForm object
        """
        if form_type == FormType.SONATA:
            return SonataFormGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        elif form_type == FormType.RONDO:
            return RondoFormGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        elif form_type == FormType.THEME_AND_VARIATIONS:
            return ThemeAndVariationsGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        elif form_type == FormType.FUGUE:
            return FugueGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        elif form_type == FormType.VERSE_CHORUS:
            return VerseChorusGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        elif form_type == FormType.AABA:
            return AABAGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        elif form_type == FormType.TWELVE_BAR_BLUES:
            return TwelveBarBluesGenerator.generate(tonic_key, is_major, tempo, **kwargs)
        else:
            raise ValueError(f"Unsupported form type: {form_type}")

    @staticmethod
    def print_form_analysis(form: MusicalForm) -> str:
        """
        Generate a human-readable analysis of a musical form

        Args:
            form: MusicalForm to analyze

        Returns:
            String with formatted analysis
        """
        output = []
        output.append("=" * 80)
        output.append(f"MUSICAL FORM ANALYSIS: {form.form_type.value.upper()}")
        output.append("=" * 80)
        output.append(f"Tonic Key: {_midi_to_note_name(form.tonic_key)} {'major' if form.is_major else 'minor'}")
        output.append(f"Tempo: {form.tempo} BPM")
        output.append(f"Time Signature: {form.time_signature[0]}/{form.time_signature[1]}")
        output.append(f"Total Length: {form.total_bars} bars")
        output.append("")
        output.append("STRUCTURE:")
        output.append("-" * 80)

        timeline = form.get_section_timeline()
        for start_bar, end_bar, section in timeline:
            # Calculate key for this section
            key_note, key_is_major = calculate_related_key(
                form.tonic_key, form.is_major, section.key_relationship
            )
            key_str = f"{_midi_to_note_name(key_note)} {'major' if key_is_major else 'minor'}"

            output.append(f"Bars {start_bar+1}-{end_bar}: {section.name}")
            output.append(f"  Key: {key_str} ({section.key_relationship.value})")
            output.append(f"  Character: {section.character}")
            output.append(f"  Themes: {', '.join(section.thematic_material)}")
            output.append(f"  Development: {section.development_level:.1f} | Dynamics: {section.dynamic_level:.1f} | Texture: {section.texture_density:.1f}")
            output.append("")

        output.append("=" * 80)
        return "\n".join(output)


def _midi_to_note_name(midi_note: int) -> str:
    """Convert MIDI note number to note name"""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[midi_note % 12]


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🎵 FORM GENERATOR - Musical Structure Engine\n")

    # Example 1: Sonata Form
    print("=" * 80)
    print("EXAMPLE 1: Sonata Form in C Major")
    print("=" * 80)
    sonata = FormGenerator.generate_form(
        FormType.SONATA,
        tonic_key=60,  # C
        is_major=True,
        tempo=140,
        exposition_length=32,
        development_length=24,
        include_introduction=True,
        include_coda=True
    )
    print(FormGenerator.print_form_analysis(sonata))

    # Example 2: Rondo Form
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Rondo Form ABACABA in G Major")
    print("=" * 80)
    rondo = FormGenerator.generate_form(
        FormType.RONDO,
        tonic_key=67,  # G
        is_major=True,
        tempo=120,
        pattern="ABACABA",
        section_length=8
    )
    print(FormGenerator.print_form_analysis(rondo))

    # Example 3: Theme and Variations
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Theme and Variations in D Minor")
    print("=" * 80)
    variations = FormGenerator.generate_form(
        FormType.THEME_AND_VARIATIONS,
        tonic_key=62,  # D
        is_major=False,
        tempo=100,
        theme_length=16,
        num_variations=6
    )
    print(FormGenerator.print_form_analysis(variations))

    # Example 4: Fugue
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Fugue in F Major, 4 voices")
    print("=" * 80)
    fugue = FormGenerator.generate_form(
        FormType.FUGUE,
        tonic_key=65,  # F
        is_major=True,
        tempo=96,
        num_voices=4,
        subject_length=4
    )
    print(FormGenerator.print_form_analysis(fugue))

    # Example 5: Verse-Chorus (Pop Song)
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Verse-Chorus Pop Song in A Minor")
    print("=" * 80)
    pop_song = FormGenerator.generate_form(
        FormType.VERSE_CHORUS,
        tonic_key=69,  # A
        is_major=False,
        tempo=128,
        verse_length=8,
        chorus_length=8,
        include_bridge=True,
        num_verses=2
    )
    print(FormGenerator.print_form_analysis(pop_song))

    # Example 6: 12-Bar Blues
    print("\n" + "=" * 80)
    print("EXAMPLE 6: 12-Bar Blues in E")
    print("=" * 80)
    blues = FormGenerator.generate_form(
        FormType.TWELVE_BAR_BLUES,
        tonic_key=64,  # E
        is_major=False,
        tempo=120,
        num_choruses=3
    )
    print(FormGenerator.print_form_analysis(blues))

    # Example 7: AABA (Jazz Standard)
    print("\n" + "=" * 80)
    print("EXAMPLE 7: AABA (32-Bar) Form in Bb Major")
    print("=" * 80)
    aaba = FormGenerator.generate_form(
        FormType.AABA,
        tonic_key=70,  # Bb
        is_major=True,
        tempo=140,
        section_length=8
    )
    print(FormGenerator.print_form_analysis(aaba))

    print("\n✅ Form Generator examples complete!")
    print("This module provides structural frameworks that can be filled with actual")
    print("musical content using melody, harmony, and rhythm generators.\n")
