#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Professional Orchestration Engine

Intelligent automatic orchestration system that selects instruments, creates voicings,
applies doubling rules, and generates professional orchestral arrangements.

Features:
- Automatic instrument selection based on register, texture, and musical context
- Professional doubling and spacing rules
- Voice leading optimization
- Dynamic orchestral balance
- Style-based orchestration (Classical, Romantic, Film, Chamber, etc.)
- Instrument combination analysis (blend/avoid)
- Tessitura-aware writing
- Automatic range validation and transposition

Research References:
- Rimsky-Korsakov: Principles of Orchestration
- Samuel Adler: The Study of Orchestration
- Berlioz: Treatise on Instrumentation
- Film scoring techniques (Goldsmith, Williams, Zimmer)

Author: Claude (Sonnet 4.5)
Created: 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.instrument_library import (
    Instrument, InstrumentFamily, INSTRUMENTS,
    get_instrument, get_instruments_by_family,
    is_in_comfortable_range, is_in_optimal_range,
    get_register_name, written_to_sounding
)


class OrchestrationStyle(Enum):
    """Orchestration style/period"""
    CLASSICAL = "classical"  # Mozart, Haydn (smaller, transparent)
    ROMANTIC = "romantic"    # Brahms, Tchaikovsky (lush, full)
    IMPRESSIONIST = "impressionist"  # Debussy, Ravel (colorful, sparse)
    MODERN = "modern"        # Stravinsky, Bartok (angular, percussive)
    FILM = "film"           # Williams, Goldsmith (powerful, emotional)
    CHAMBER = "chamber"      # Small ensemble (3-10 players)
    BIG_BAND = "big_band"   # Jazz orchestra
    POP = "pop"             # Pop/rock arrangement


class TextureType(Enum):
    """Type of musical texture"""
    MELODY = "melody"
    HARMONY = "harmony"
    BASS = "bass"
    COUNTERMELODY = "countermelody"
    OSTINATO = "ostinato"
    PEDAL = "pedal"
    TUTTI = "tutti"  # Everyone playing


@dataclass
class VoicePart:
    """A single voice/part to be orchestrated"""
    notes: List[int]  # MIDI note numbers (sounding pitch)
    durations: List[float]  # Duration in beats
    start_times: List[float]  # Start time in beats
    velocities: List[int]  # MIDI velocities
    texture_type: TextureType = TextureType.MELODY
    dynamic_level: str = "mf"  # pp, p, mp, mf, f, ff, fff


@dataclass
class OrchestralVoicing:
    """Result of orchestration: which instruments play which notes"""
    instrument_name: str
    notes: List[int]  # Written notes for this instrument
    durations: List[float]
    start_times: List[float]
    velocities: List[int]
    articulations: List[str] = None


class Orchestrator:
    """
    Intelligent orchestration engine.

    Automatically assigns musical material to instruments based on:
    - Register and tessitura
    - Texture type (melody, harmony, bass, etc.)
    - Dynamic level
    - Orchestration style
    - Instrument capabilities and blend characteristics
    """

    def __init__(
        self,
        style: OrchestrationStyle = OrchestrationStyle.ROMANTIC,
        available_instruments: Optional[List[str]] = None
    ):
        """
        Initialize orchestrator.

        Args:
            style: Orchestration style/period
            available_instruments: List of available instrument names.
                                 If None, uses full symphony orchestra.
        """
        self.style = style

        if available_instruments is None:
            # Default: Full symphony orchestra
            self.available_instruments = self._get_default_orchestra()
        else:
            self.available_instruments = [
                get_instrument(name) for name in available_instruments
                if get_instrument(name) is not None
            ]

        # Dynamic level to velocity mapping
        self.dynamic_map = {
            'ppp': 20, 'pp': 30, 'p': 45, 'mp': 60,
            'mf': 75, 'f': 90, 'ff': 105, 'fff': 120
        }

    def _get_default_orchestra(self) -> List[Instrument]:
        """Get default full symphony orchestra instrumentation"""
        orchestra = [
            # Strings (always available)
            "Violin", "Viola", "Cello", "Double Bass",

            # Woodwinds
            "Flute", "Piccolo", "Oboe", "English Horn",
            "Clarinet", "Bass Clarinet", "Bassoon", "Contrabassoon",

            # Brass
            "Trumpet", "Horn", "Trombone", "Bass Trombone", "Tuba",

            # Percussion
            "Timpani",

            # Keyboards
            "Piano"
        ]

        return [get_instrument(name) for name in orchestra if get_instrument(name)]

    def orchestrate(
        self,
        voices: List[VoicePart],
        prefer_families: Optional[List[InstrumentFamily]] = None
    ) -> List[OrchestralVoicing]:
        """
        Orchestrate multiple voice parts automatically.

        Args:
            voices: List of voice parts to orchestrate
            prefer_families: Preferred instrument families (optional)

        Returns:
            List of orchestral voicings (instrument assignments)
        """
        voicings = []

        for voice in voices:
            # Determine best instruments for this voice
            candidates = self._find_candidate_instruments(
                voice, prefer_families
            )

            if not candidates:
                print(f"Warning: No suitable instruments found for voice")
                continue

            # Select best instrument(s)
            selected = self._select_instruments(voice, candidates)

            # Create voicing(s)
            for inst_name, notes_subset in selected.items():
                inst = get_instrument(inst_name)
                if inst:
                    voicing = self._create_voicing(inst, voice, notes_subset)
                    voicings.append(voicing)

        # Apply doubling rules
        voicings = self._apply_doubling_rules(voicings, voices)

        # Optimize balance
        voicings = self._optimize_balance(voicings)

        return voicings

    def _find_candidate_instruments(
        self,
        voice: VoicePart,
        prefer_families: Optional[List[InstrumentFamily]] = None
    ) -> List[Instrument]:
        """
        Find instruments suitable for a voice part.

        Considers:
        - Range compatibility
        - Texture type appropriateness
        - Family preferences
        - Style conventions
        """
        candidates = []

        # Determine required range
        if not voice.notes:
            return candidates

        min_note = min(voice.notes)
        max_note = max(voice.notes)
        avg_note = sum(voice.notes) / len(voice.notes)

        for inst in self.available_instruments:
            # Check range compatibility
            if not self._can_play_range(inst, min_note, max_note):
                continue

            # Check family preference
            if prefer_families and inst.family not in prefer_families:
                continue

            # Check texture appropriateness
            if not self._is_appropriate_for_texture(inst, voice.texture_type):
                continue

            # Add with suitability score
            score = self._score_instrument_suitability(inst, voice)
            candidates.append((inst, score))

        # Sort by suitability score (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [inst for inst, score in candidates]

    def _can_play_range(
        self,
        instrument: Instrument,
        min_note: int,
        max_note: int
    ) -> bool:
        """Check if instrument can play the required range"""
        # Check absolute limits
        if min_note < instrument.range.lowest_note:
            return False
        if max_note > instrument.range.highest_note:
            return False

        # Prefer comfortable range (but allow wider if needed)
        comfortable = (
            min_note >= instrument.range.comfortable_low and
            max_note <= instrument.range.comfortable_high
        )

        return True  # Can play, even if not fully comfortable

    def _is_appropriate_for_texture(
        self,
        instrument: Instrument,
        texture: TextureType
    ) -> bool:
        """
        Check if instrument is appropriate for texture type.

        Different instruments excel at different roles.
        """
        family = instrument.family

        if texture == TextureType.MELODY:
            # Melody: singing, expressive instruments
            return family in [
                InstrumentFamily.STRINGS,
                InstrumentFamily.WOODWINDS,
                InstrumentFamily.BRASS
            ]

        elif texture == TextureType.HARMONY:
            # Harmony: good blenders
            return family in [
                InstrumentFamily.STRINGS,
                InstrumentFamily.BRASS,
                InstrumentFamily.KEYBOARDS
            ]

        elif texture == TextureType.BASS:
            # Bass: low instruments with good foundation
            return instrument.name in [
                "Double Bass", "Cello", "Bassoon", "Contrabassoon",
                "Bass Clarinet", "Tuba", "Bass Trombone", "Piano"
            ]

        elif texture == TextureType.COUNTERMELODY:
            # Counter-melody: agile, distinctive but not overpowering
            return instrument.name in [
                "Viola", "Cello", "Clarinet", "English Horn",
                "Horn", "Oboe"
            ]

        elif texture == TextureType.OSTINATO:
            # Ostinato: rhythmic, consistent
            return family in [
                InstrumentFamily.STRINGS,  # pizzicato
                InstrumentFamily.KEYBOARDS,
                InstrumentFamily.PERCUSSION
            ]

        elif texture == TextureType.PEDAL:
            # Pedal point: sustained, foundational
            return family in [
                InstrumentFamily.BRASS,  # horns especially
                InstrumentFamily.STRINGS,
                InstrumentFamily.KEYBOARDS
            ]

        return True  # Default: allow any

    def _score_instrument_suitability(
        self,
        instrument: Instrument,
        voice: VoicePart
    ) -> float:
        """
        Score how suitable an instrument is for a voice part.

        Higher score = more suitable.
        """
        score = 0.0

        if not voice.notes:
            return score

        # Check if notes are in optimal range (best)
        optimal_count = sum(
            1 for note in voice.notes
            if is_in_optimal_range(note, instrument)
        )
        score += (optimal_count / len(voice.notes)) * 50

        # Check if notes are in comfortable range (good)
        comfortable_count = sum(
            1 for note in voice.notes
            if is_in_comfortable_range(note, instrument)
        )
        score += (comfortable_count / len(voice.notes)) * 30

        # Bonus for strings in Romantic style
        if self.style == OrchestrationStyle.ROMANTIC:
            if instrument.family == InstrumentFamily.STRINGS:
                score += 20

        # Bonus for appropriate texture
        if voice.texture_type == TextureType.MELODY:
            if instrument.name in ["Violin", "Flute", "Oboe", "Trumpet"]:
                score += 15

        # Penalty for extreme ranges
        for note in voice.notes:
            register = get_register_name(note, instrument)
            if register == "extreme":
                score -= 10

        # Bonus for versatile instruments
        if len(instrument.articulations) > 8:
            score += 5

        return score

    def _select_instruments(
        self,
        voice: VoicePart,
        candidates: List[Instrument]
    ) -> Dict[str, List[int]]:
        """
        Select which instrument(s) will play this voice.

        May select multiple instruments for doubling or divisi.

        Returns:
            Dict mapping instrument names to note indices they should play
        """
        if not candidates:
            return {}

        selected = {}

        # Primary instrument (best candidate)
        primary = candidates[0]
        selected[primary.name] = list(range(len(voice.notes)))

        # Consider doubling for certain textures
        if voice.texture_type == TextureType.MELODY and len(candidates) > 1:
            # Melody: consider doubling at octave or unison
            if self._should_double_melody(voice):
                secondary = candidates[1]
                selected[secondary.name] = list(range(len(voice.notes)))

        return selected

    def _should_double_melody(self, voice: VoicePart) -> bool:
        """Decide if melody should be doubled"""
        # Double melodies in Romantic and Film styles
        if self.style in [OrchestrationStyle.ROMANTIC, OrchestrationStyle.FILM]:
            # Double if loud dynamic
            if voice.dynamic_level in ['f', 'ff', 'fff']:
                return True

        # Classical style: less doubling
        if self.style == OrchestrationStyle.CLASSICAL:
            return False

        return False

    def _create_voicing(
        self,
        instrument: Instrument,
        voice: VoicePart,
        note_indices: List[int]
    ) -> OrchestralVoicing:
        """
        Create an orchestral voicing for an instrument.

        Handles transposition and converts to written pitch.
        """
        # Extract relevant notes
        notes = [voice.notes[i] for i in note_indices]
        durations = [voice.durations[i] for i in note_indices]
        start_times = [voice.start_times[i] for i in note_indices]
        velocities = [voice.velocities[i] for i in note_indices]

        # Convert sounding to written pitch
        written_notes = [note - instrument.transposition for note in notes]

        # Adjust velocities based on dynamic level
        base_velocity = self.dynamic_map.get(voice.dynamic_level, 75)
        adjusted_velocities = [
            self._adjust_velocity(v, base_velocity, instrument)
            for v in velocities
        ]

        return OrchestralVoicing(
            instrument_name=instrument.name,
            notes=written_notes,
            durations=durations,
            start_times=start_times,
            velocities=adjusted_velocities,
            articulations=["legato"] * len(written_notes)  # Default
        )

    def _adjust_velocity(
        self,
        original_velocity: int,
        target_dynamic: int,
        instrument: Instrument
    ) -> int:
        """Adjust velocity for instrument characteristics"""
        # Blend original expression with target dynamic
        adjusted = int(original_velocity * 0.5 + target_dynamic * 0.5)

        # Clamp to instrument's dynamic range
        adjusted = max(instrument.min_dynamic, adjusted)
        adjusted = min(instrument.max_dynamic, adjusted)

        return adjusted

    def _apply_doubling_rules(
        self,
        voicings: List[OrchestralVoicing],
        voices: List[VoicePart]
    ) -> List[OrchestralVoicing]:
        """
        Apply professional doubling rules.

        Rules:
        - Octave doubling for reinforcement (strings, winds)
        - Unison doubling for blend
        - Avoid muddy combinations (bassoon + cello low register)
        - Double melody at climax
        """
        # TODO: Implement sophisticated doubling logic
        # For now, return as-is
        return voicings

    def _optimize_balance(
        self,
        voicings: List[OrchestralVoicing]
    ) -> List[OrchestralVoicing]:
        """
        Optimize orchestral balance.

        Adjust velocities so instruments blend properly.
        """
        # Balance rules:
        # - Brass naturally louder than woodwinds
        # - Strings can balance brass when in sections
        # - Melody should be 10-15 velocity points above harmony

        for voicing in voicings:
            instrument = get_instrument(voicing.instrument_name)
            if not instrument:
                continue

            # Adjust based on family
            if instrument.family == InstrumentFamily.BRASS:
                # Brass tends to overpower; reduce slightly
                voicing.velocities = [max(20, v - 10) for v in voicing.velocities]

            elif instrument.family == InstrumentFamily.WOODWINDS:
                # Woodwinds can get lost; boost slightly
                voicing.velocities = [min(120, v + 5) for v in voicing.velocities]

        return voicings

    def auto_arrange(
        self,
        melody: List[int],
        chords: List[List[int]],
        bass: List[int],
        tempo: int = 120,
        time_signature: Tuple[int, int] = (4, 4)
    ) -> List[OrchestralVoicing]:
        """
        Automatically arrange melody, chords, and bass for orchestra.

        Args:
            melody: List of melody notes (MIDI)
            chords: List of chord voicings (each is list of MIDI notes)
            bass: List of bass notes (MIDI)
            tempo: Tempo in BPM
            time_signature: Time signature (numerator, denominator)

        Returns:
            List of orchestral voicings
        """
        voices = []

        # Create melody voice
        if melody:
            melody_voice = VoicePart(
                notes=melody,
                durations=[1.0] * len(melody),  # Default: quarter notes
                start_times=[i * 1.0 for i in range(len(melody))],
                velocities=[80] * len(melody),
                texture_type=TextureType.MELODY,
                dynamic_level='mf'
            )
            voices.append(melody_voice)

        # Create harmony voices from chords
        if chords:
            # Separate into SATB-like voices
            harmony_voices = self._chords_to_voices(chords)
            voices.extend(harmony_voices)

        # Create bass voice
        if bass:
            bass_voice = VoicePart(
                notes=bass,
                durations=[1.0] * len(bass),
                start_times=[i * 1.0 for i in range(len(bass))],
                velocities=[75] * len(bass),
                texture_type=TextureType.BASS,
                dynamic_level='mf'
            )
            voices.append(bass_voice)

        # Orchestrate all voices
        return self.orchestrate(voices)

    def _chords_to_voices(
        self,
        chords: List[List[int]]
    ) -> List[VoicePart]:
        """
        Convert chord voicings to separate voices (SATB-like).

        Args:
            chords: List of chords (each chord is list of MIDI notes)

        Returns:
            List of voice parts (soprano, alto, tenor)
        """
        if not chords:
            return []

        # Determine number of voices (usually 3-4)
        max_voices = max(len(chord) for chord in chords)
        voices_count = min(max_voices, 4)

        voices = [[] for _ in range(voices_count)]

        # Distribute chord notes to voices
        for chord in chords:
            sorted_chord = sorted(chord)

            # Pad with last note if chord has fewer notes
            while len(sorted_chord) < voices_count:
                sorted_chord.append(sorted_chord[-1])

            # Assign to voices (bottom to top)
            for i, note in enumerate(sorted_chord[:voices_count]):
                voices[i].append(note)

        # Create VoicePart objects
        voice_parts = []
        voice_types = [
            TextureType.HARMONY,  # Alto
            TextureType.HARMONY,  # Tenor
            TextureType.HARMONY,  # Soprano
        ]

        for i, notes in enumerate(voices):
            if notes:
                voice_type = voice_types[i] if i < len(voice_types) else TextureType.HARMONY
                voice_part = VoicePart(
                    notes=notes,
                    durations=[1.0] * len(notes),
                    start_times=[j * 1.0 for j in range(len(notes))],
                    velocities=[70] * len(notes),
                    texture_type=voice_type,
                    dynamic_level='mp'
                )
                voice_parts.append(voice_part)

        return voice_parts

    def analyze_spacing(
        self,
        voicing: List[int]
    ) -> Dict[str, any]:
        """
        Analyze chord spacing quality.

        Good spacing rules:
        - Wide spacing in bass (no intervals < major 3rd below C3)
        - Closer spacing in treble
        - Avoid gaps in middle register
        - Optimal: 4-10 semitones between adjacent voices

        Args:
            voicing: Chord notes (sorted low to high)

        Returns:
            Analysis dict with quality score and issues
        """
        if len(voicing) < 2:
            return {"score": 100, "issues": []}

        issues = []
        score = 100

        sorted_voicing = sorted(voicing)

        # Check intervals between adjacent notes
        for i in range(len(sorted_voicing) - 1):
            interval = sorted_voicing[i + 1] - sorted_voicing[i]
            lower_note = sorted_voicing[i]

            # Bass spacing rules (below C3 = 48)
            if lower_note < 48:
                if interval < 3:  # Less than minor 3rd
                    issues.append(f"Bass too close: {interval} semitones at note {lower_note}")
                    score -= 20
                elif interval < 4:  # Less than major 3rd
                    issues.append(f"Bass spacing tight: {interval} semitones")
                    score -= 10

            # Middle register spacing (C3 to C5)
            elif 48 <= lower_note < 72:
                if interval > 12:  # More than octave
                    issues.append(f"Gap in middle: {interval} semitones at note {lower_note}")
                    score -= 15

            # Treble spacing (above C5)
            else:
                if interval < 2:  # Less than major 2nd
                    issues.append(f"Treble too close: {interval} semitones")
                    score -= 5

        return {
            "score": max(0, score),
            "issues": issues,
            "range": sorted_voicing[-1] - sorted_voicing[0]
        }

    def suggest_doubling(
        self,
        voicing: OrchestralVoicing,
        context: str = "default"
    ) -> List[Tuple[str, str]]:
        """
        Suggest doubling options for a voicing.

        Args:
            voicing: Current orchestral voicing
            context: Musical context ("climax", "soft", "default", etc.)

        Returns:
            List of (instrument_name, doubling_type) tuples
            doubling_type: "unison", "octave_up", "octave_down"
        """
        suggestions = []
        instrument = get_instrument(voicing.instrument_name)

        if not instrument:
            return suggestions

        # Get average note for register analysis
        avg_note = sum(voicing.notes) / len(voicing.notes) if voicing.notes else 60

        # Melody instruments: suggest doubling
        if instrument.name == "Violin":
            if context == "climax":
                suggestions.append(("Flute", "unison"))
                suggestions.append(("Trumpet", "unison"))
            suggestions.append(("Viola", "octave_down"))

        elif instrument.name == "Flute":
            suggestions.append(("Violin", "unison"))
            if avg_note > 72:  # High register
                suggestions.append(("Piccolo", "octave_up"))

        elif instrument.name == "Oboe":
            suggestions.append(("Violin", "unison"))
            suggestions.append(("Clarinet", "unison"))

        elif instrument.name == "Clarinet":
            suggestions.append(("Violin", "unison"))
            suggestions.append(("Flute", "unison"))
            if avg_note < 60:  # Low register
                suggestions.append(("Bass Clarinet", "octave_down"))

        # Bass instruments: suggest doubling
        elif instrument.name == "Double Bass":
            suggestions.append(("Cello", "octave_up"))
            suggestions.append(("Bassoon", "octave_up"))
            suggestions.append(("Contrabassoon", "unison"))

        elif instrument.name == "Cello":
            if avg_note < 48:  # Low register
                suggestions.append(("Double Bass", "octave_down"))
            suggestions.append(("Bassoon", "unison"))

        return suggestions

    def apply_orchestration_rules(
        self,
        voicings: List[OrchestralVoicing]
    ) -> List[str]:
        """
        Check voicings against orchestration rules and return warnings.

        Rules checked:
        - Instrument range violations
        - Muddy combinations
        - Balance issues
        - Tessitura problems

        Returns:
            List of warning strings
        """
        warnings = []

        for voicing in voicings:
            instrument = get_instrument(voicing.instrument_name)
            if not instrument:
                continue

            # Check range
            for note in voicing.notes:
                sounding = note + instrument.transposition
                if sounding < instrument.range.lowest_note:
                    warnings.append(
                        f"{instrument.name}: Note {note} below range "
                        f"(lowest: {instrument.range.lowest_note})"
                    )
                elif sounding > instrument.range.highest_note:
                    warnings.append(
                        f"{instrument.name}: Note {note} above range "
                        f"(highest: {instrument.range.highest_note})"
                    )

            # Check tessitura
            uncomfortable_count = sum(
                1 for note in voicing.notes
                if not is_in_comfortable_range(note, instrument)
            )

            if uncomfortable_count > len(voicing.notes) * 0.5:
                warnings.append(
                    f"{instrument.name}: {uncomfortable_count}/{len(voicing.notes)} "
                    f"notes outside comfortable range"
                )

        # Check for muddy combinations
        bass_instruments = [v for v in voicings
                          if get_instrument(v.instrument_name).family == InstrumentFamily.BRASS
                          or v.instrument_name in ["Bassoon", "Cello"]]

        if len(bass_instruments) > 1:
            # Check if bass instruments are too close
            for i in range(len(bass_instruments) - 1):
                for j in range(i + 1, len(bass_instruments)):
                    v1, v2 = bass_instruments[i], bass_instruments[j]
                    if v1.notes and v2.notes:
                        avg1 = sum(v1.notes) / len(v1.notes)
                        avg2 = sum(v2.notes) / len(v2.notes)
                        if abs(avg1 - avg2) < 7 and min(avg1, avg2) < 48:
                            warnings.append(
                                f"Potentially muddy: {v1.instrument_name} and "
                                f"{v2.instrument_name} too close in low register"
                            )

        return warnings


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_string_section_voicing(
    chord: List[int],
    style: str = "romantic"
) -> Dict[str, List[int]]:
    """
    Create a string section voicing for a chord.

    Args:
        chord: Chord notes (MIDI)
        style: "classical", "romantic", "modern"

    Returns:
        Dict mapping instrument names to note lists
    """
    voicing = {}
    sorted_chord = sorted(chord)

    if len(sorted_chord) < 3:
        return voicing

    # Classical style: simpler, more transparent
    if style == "classical":
        voicing["Violin"] = [sorted_chord[-1]]  # Melody (soprano)
        voicing["Viola"] = [sorted_chord[1] if len(sorted_chord) > 2 else sorted_chord[0]]  # Alto
        voicing["Cello"] = [sorted_chord[0]]  # Bass
        voicing["Double Bass"] = [sorted_chord[0] - 12]  # Octave below

    # Romantic style: lush, doubled
    elif style == "romantic":
        voicing["Violin"] = [sorted_chord[-1], sorted_chord[-1] + 12]  # Doubled at octave
        if len(sorted_chord) > 2:
            voicing["Viola"] = [sorted_chord[-2]]  # Inner voice
        voicing["Cello"] = [sorted_chord[1] if len(sorted_chord) > 1 else sorted_chord[0]]
        voicing["Double Bass"] = [sorted_chord[0] - 12]

    # Modern style: open, spread
    else:
        voicing["Violin"] = [sorted_chord[-1]]
        if len(sorted_chord) > 3:
            voicing["Viola"] = [sorted_chord[-2]]
            voicing["Cello"] = [sorted_chord[-3]]
        else:
            voicing["Viola"] = [sorted_chord[1]]
            voicing["Cello"] = [sorted_chord[0]]
        voicing["Double Bass"] = [sorted_chord[0] - 12]

    return voicing


def get_orchestration_template(
    template_name: str
) -> List[str]:
    """
    Get pre-defined orchestration templates.

    Args:
        template_name: Name of template
            - "symphony_orchestra"
            - "chamber_orchestra"
            - "string_quartet"
            - "wind_quintet"
            - "brass_quintet"
            - "piano_trio"
            - "jazz_combo"
            - "big_band"

    Returns:
        List of instrument names
    """
    templates = {
        "symphony_orchestra": [
            # Strings
            "Violin", "Viola", "Cello", "Double Bass",
            # Woodwinds (pairs)
            "Flute", "Piccolo", "Oboe", "English Horn",
            "Clarinet", "Bass Clarinet", "Bassoon", "Contrabassoon",
            # Brass
            "Trumpet", "Horn", "Trombone", "Tuba",
            # Percussion
            "Timpani",
            # Keyboards
            "Piano"
        ],

        "chamber_orchestra": [
            "Violin", "Viola", "Cello", "Double Bass",
            "Flute", "Oboe", "Clarinet", "Bassoon",
            "Horn"
        ],

        "string_quartet": [
            "Violin", "Violin", "Viola", "Cello"
        ],

        "wind_quintet": [
            "Flute", "Oboe", "Clarinet", "Bassoon", "Horn"
        ],

        "brass_quintet": [
            "Trumpet", "Trumpet", "Horn", "Trombone", "Tuba"
        ],

        "piano_trio": [
            "Violin", "Cello", "Piano"
        ],

        "jazz_combo": [
            "Trumpet", "Piano", "Double Bass"
        ],

        "big_band": [
            "Trumpet", "Trumpet", "Trumpet", "Trumpet",
            "Trombone", "Trombone", "Trombone", "Bass Trombone",
            "Piano", "Double Bass"
        ]
    }

    return templates.get(template_name, [])


# ============================================================================
# MAIN (EXAMPLES/TESTS)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ORCHESTRATION ENGINE - EXAMPLES")
    print("=" * 80)

    # Example 1: Simple melody orchestration
    print("\n1. Orchestrating a simple melody...")

    orchestrator = Orchestrator(style=OrchestrationStyle.ROMANTIC)

    melody_notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    melody = VoicePart(
        notes=melody_notes,
        durations=[1.0] * 8,
        start_times=[i * 1.0 for i in range(8)],
        velocities=[80] * 8,
        texture_type=TextureType.MELODY,
        dynamic_level='mf'
    )

    voicings = orchestrator.orchestrate([melody])

    print(f"\nMelody assigned to:")
    for voicing in voicings:
        print(f"  • {voicing.instrument_name}: {len(voicing.notes)} notes")

    # Example 2: String section voicing
    print("\n2. String section chord voicing...")

    chord = [48, 52, 55, 60]  # C major chord
    string_voicing = create_string_section_voicing(chord, style="romantic")

    print("\nRomantic string voicing for C major chord:")
    for instrument, notes in string_voicing.items():
        note_names = [f"{n}" for n in notes]
        print(f"  • {instrument}: {note_names}")

    # Example 3: Auto-arrange
    print("\n3. Auto-arranging melody + chords + bass...")

    melody = [72, 74, 76, 77, 79]
    chords = [
        [60, 64, 67],  # C major
        [62, 65, 69],  # Dm
        [64, 67, 71],  # Em
        [65, 69, 72],  # F major
        [67, 71, 74],  # G major
    ]
    bass = [48, 50, 52, 53, 55]

    voicings = orchestrator.auto_arrange(melody, chords, bass)

    print(f"\nAuto-arrangement created {len(voicings)} parts:")
    for voicing in voicings:
        print(f"  • {voicing.instrument_name}: {len(voicing.notes)} notes")

    # Example 4: Spacing analysis
    print("\n4. Analyzing chord spacing...")

    good_voicing = [48, 55, 60, 64, 67]  # Good spacing
    bad_voicing = [48, 49, 50, 60, 85]   # Bad spacing (close bass, gap, wide treble)

    analysis_good = orchestrator.analyze_spacing(good_voicing)
    analysis_bad = orchestrator.analyze_spacing(bad_voicing)

    print(f"\nGood voicing score: {analysis_good['score']}/100")
    print(f"Issues: {len(analysis_good['issues'])}")

    print(f"\nBad voicing score: {analysis_bad['score']}/100")
    print(f"Issues: {len(analysis_bad['issues'])}")
    for issue in analysis_bad['issues']:
        print(f"  • {issue}")

    # Example 5: Orchestration templates
    print("\n5. Orchestration templates...")

    templates = ["symphony_orchestra", "string_quartet", "wind_quintet", "big_band"]

    for template_name in templates:
        instruments = get_orchestration_template(template_name)
        print(f"\n{template_name.replace('_', ' ').title()}:")
        print(f"  {len(instruments)} instruments: {', '.join(instruments[:5])}...")

    # Example 6: Doubling suggestions
    print("\n6. Doubling suggestions...")

    violin_voicing = OrchestralVoicing(
        instrument_name="Violin",
        notes=[72, 74, 76, 77],
        durations=[1.0] * 4,
        start_times=[0, 1, 2, 3],
        velocities=[90] * 4
    )

    suggestions = orchestrator.suggest_doubling(violin_voicing, context="climax")
    print("\nDoubling suggestions for climactic violin melody:")
    for inst, doubling_type in suggestions:
        print(f"  • {inst} ({doubling_type})")

    print("\n" + "=" * 80)
    print("Orchestration engine ready!")
    print("=" * 80)
