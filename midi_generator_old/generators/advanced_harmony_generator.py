#!/usr/bin/env python3
"""
Advanced Harmony Generator
===========================

Unified interface for advanced harmonic generation using Neo-Riemannian transformations,
modal harmony systems, and microtonal scales.

This generator integrates:
- Neo-Riemannian transformations (PLR, hexatonic systems)
- Modal harmony (church modes, harmonic/melodic minor modes)
- Modal interchange and borrowing
- Microtonal systems (24-TET, 53-TET, just intonation)
- World music scales (Arabic maqam, Indian raga, Turkish makam, Persian dastgah)

Author: Agent 3 - Advanced Harmony & Modal Systems
License: MIT
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.neo_riemannian import (
    Triad, TriadQuality, NeoRiemannianTransformations,
    TransformationChain, HexatonicSystem, ChromaticMediant,
    VoiceLeadingAnalyzer, Tonnetz
)
from core.modal_harmony import (
    Mode, ModalScaleLibrary, ModalProgressionGenerator,
    ModalInterchange, ModalCadence, PedalPointGenerator,
    HarmonicMinorMode, MelodicMinorMode
)
from core.microtonality import (
    MicrotonalScale, EqualTemperament, CommonET, JustIntonation,
    MaqamSystem, ArabicMaqam, IndianRaga, TurkishMakam, PersianDastgah,
    MicrotonalMIDI
)

from typing import List, Dict, Tuple, Optional, Union
from enum import Enum


class HarmonyStyle(Enum):
    """Harmony generation styles"""
    NEO_RIEMANNIAN = "neo_riemannian"
    MODAL = "modal"
    CHROMATIC_MEDIANT = "chromatic_mediant"
    MICROTONAL = "microtonal"
    WORLD_MUSIC = "world_music"
    MODAL_INTERCHANGE = "modal_interchange"


class AdvancedHarmonyGenerator:
    """
    Comprehensive harmony generator with advanced techniques.

    Provides high-level interface for generating progressions using
    cutting-edge harmonic theory.
    """

    def __init__(self, root: int = 0, octave: int = 4):
        """
        Initialize generator.

        Args:
            root: Root pitch class (0-11, C=0)
            octave: Base MIDI octave for chord generation
        """
        self.root = root % 12
        self.octave = octave
        self.tonnetz = Tonnetz()

    # ========================================================================
    # NEO-RIEMANNIAN GENERATION
    # ========================================================================

    def generate_neo_riemannian(self,
                                transformation_sequence: str,
                                start_quality: str = "major",
                                voice_lead: bool = True) -> Dict:
        """
        Generate progression using Neo-Riemannian transformations.

        Args:
            transformation_sequence: String of transformations (e.g., "PLR", "P L R")
            start_quality: Starting triad quality ("major" or "minor")
            voice_lead: Apply smooth voice leading

        Returns:
            Dictionary with progression data
        """
        quality = TriadQuality.MAJOR if start_quality == "major" else TriadQuality.MINOR
        start_triad = Triad(self.root, quality)

        chain = TransformationChain(start_triad)
        chain.apply_sequence(transformation_sequence)

        midi_prog = chain.get_midi_progression(self.octave, voice_lead)
        triads = chain.get_progression()

        # Analyze voice leading
        analysis = VoiceLeadingAnalyzer.analyze_progression(midi_prog)

        return {
            "style": "Neo-Riemannian",
            "transformations": transformation_sequence,
            "triads": [str(t) for t in triads],
            "midi_notes": midi_prog,
            "voice_leading_analysis": analysis
        }

    def generate_hexatonic_cycle(self, pole: int = 0) -> Dict:
        """
        Generate hexatonic cycle progression.

        Args:
            pole: Hexatonic system (0=Northern, 1=Southern, 2=Eastern, 3=Western)

        Returns:
            Dictionary with progression data
        """
        hex_system = HexatonicSystem(pole)
        cycle = hex_system.get_cycle()

        # Convert to MIDI with voice leading
        midi_prog = []
        for triad in cycle:
            midi_prog.append(triad.to_midi_notes(self.octave))

        pole_names = ["Northern", "Southern", "Eastern", "Western"]

        return {
            "style": "Hexatonic Cycle",
            "pole": pole_names[pole],
            "triads": [str(t) for t in cycle],
            "midi_notes": midi_prog,
            "pitch_classes": sorted(hex_system.pitch_classes)
        }

    def generate_chromatic_mediant_prog(self, pattern: str = "UCM LCM") -> Dict:
        """
        Generate chromatic mediant progression.

        Args:
            pattern: Space-separated mediant types (UCM, LCM, UFM, LFM)

        Returns:
            Dictionary with progression data
        """
        start = Triad(self.root, TriadQuality.MAJOR)
        progression = ChromaticMediant.create_mediant_progression(start, pattern)

        midi_prog = [t.to_midi_notes(self.octave) for t in progression]

        return {
            "style": "Chromatic Mediant",
            "pattern": pattern,
            "triads": [str(t) for t in progression],
            "midi_notes": midi_prog
        }

    # ========================================================================
    # MODAL GENERATION
    # ========================================================================

    def generate_modal_progression(self,
                                   mode: Mode,
                                   progression_type: str = "characteristic",
                                   length: int = 4) -> Dict:
        """
        Generate modal progression.

        Args:
            mode: Modal scale
            progression_type: Type ("characteristic", "vamp", "plagal", "descending")
            length: Number of chords

        Returns:
            Dictionary with progression data
        """
        generator = ModalProgressionGenerator(self.root, mode)

        if progression_type == "characteristic":
            midi_prog = generator.generate_characteristic_progression()
        elif progression_type == "vamp":
            midi_prog = generator.generate_vamp(1, 5, bars=length)
        elif progression_type == "plagal":
            midi_prog = generator.generate_plagal_progression(length)
        elif progression_type == "descending":
            midi_prog = generator.generate_descending_progression(steps=length)
        else:
            midi_prog = generator.generate_characteristic_progression()

        mode_def = ModalScaleLibrary.get_mode(mode)

        return {
            "style": "Modal",
            "mode": mode.value,
            "progression_type": progression_type,
            "midi_notes": midi_prog,
            "scale_intervals": mode_def.intervals,
            "characteristic_degrees": mode_def.characteristic_degrees,
            "brightness": mode_def.brightness
        }

    def generate_modal_interchange(self,
                                   primary_mode: Mode,
                                   borrowed_mode: Mode,
                                   primary_degrees: List[int],
                                   insert_positions: List[int]) -> Dict:
        """
        Generate progression with modal interchange.

        Args:
            primary_mode: Primary mode
            borrowed_mode: Mode to borrow from
            primary_degrees: Scale degrees in primary mode
            insert_positions: Positions to insert borrowed chords

        Returns:
            Dictionary with progression data
        """
        interchange = ModalInterchange(self.root, primary_mode)
        midi_prog = interchange.create_modal_mixture_progression(
            primary_degrees, borrowed_mode, insert_positions
        )

        return {
            "style": "Modal Interchange",
            "primary_mode": primary_mode.value,
            "borrowed_mode": borrowed_mode.value,
            "midi_notes": midi_prog,
            "borrowed_positions": insert_positions
        }

    def generate_modal_cadence(self, mode: Mode, cadence_type: str) -> Dict:
        """
        Generate modal cadence.

        Args:
            mode: Modal scale
            cadence_type: Type ("plagal", "phrygian", "dorian", etc.)

        Returns:
            Dictionary with cadence data
        """
        midi_prog = ModalCadence.get_cadence(self.root, mode, cadence_type)

        return {
            "style": "Modal Cadence",
            "mode": mode.value,
            "cadence_type": cadence_type,
            "midi_notes": midi_prog
        }

    # ========================================================================
    # MICROTONAL GENERATION
    # ========================================================================

    def generate_microtonal_scale(self,
                                  system: str = "24-TET",
                                  steps: Optional[List[int]] = None) -> Dict:
        """
        Generate microtonal scale.

        Args:
            system: Equal temperament ("24-TET", "19-TET", "31-TET", "53-TET")
            steps: Step pattern (if None, uses chromatic)

        Returns:
            Dictionary with scale data
        """
        if system == "24-TET":
            et = CommonET.get_24tet()
        elif system == "19-TET":
            et = CommonET.get_19tet()
        elif system == "31-TET":
            et = CommonET.get_31tet()
        elif system == "53-TET":
            et = CommonET.get_53tet()
        else:
            et = EqualTemperament(24)

        if steps:
            scale = et.get_scale(steps)
        else:
            scale = et.get_chromatic_scale()

        # Convert to MIDI with pitch bends
        root_midi = 12 * self.octave + self.root
        pitches = scale.get_pitches(root_midi, octave_span=1, bend_range=2)

        return {
            "style": "Microtonal",
            "system": system,
            "scale_name": scale.name,
            "intervals_cents": scale.intervals,
            "midi_with_bends": pitches  # List of (note, bend_value) tuples
        }

    def generate_just_intonation_scale(self, scale_type: str = "major") -> Dict:
        """
        Generate just intonation scale.

        Args:
            scale_type: Scale type ("major", "harmonic_series")

        Returns:
            Dictionary with scale data
        """
        if scale_type == "major":
            scale = JustIntonation.get_major_scale()
        elif scale_type == "harmonic_series":
            scale = JustIntonation.get_harmonic_series(16)
        else:
            scale = JustIntonation.get_major_scale()

        root_midi = 12 * self.octave + self.root
        pitches = scale.get_pitches(root_midi, octave_span=1, bend_range=2)

        return {
            "style": "Just Intonation",
            "scale_type": scale_type,
            "scale_name": scale.name,
            "intervals_cents": scale.intervals,
            "midi_with_bends": pitches
        }

    # ========================================================================
    # WORLD MUSIC GENERATION
    # ========================================================================

    def generate_arabic_maqam(self, maqam: ArabicMaqam) -> Dict:
        """
        Generate Arabic maqam scale.

        Args:
            maqam: Maqam to generate

        Returns:
            Dictionary with maqam data
        """
        scale = MaqamSystem.get_maqam(maqam)
        root_midi = 12 * self.octave + self.root
        pitches = scale.get_pitches(root_midi, octave_span=1, bend_range=2)

        maqam_def = MaqamSystem.MAQAMAT.get(maqam, {})

        return {
            "style": "Arabic Maqam",
            "maqam": maqam.value,
            "description": maqam_def.get("description", ""),
            "intervals_cents": scale.intervals,
            "midi_with_bends": pitches,
            "lower_jins": maqam_def.get("lower", ""),
            "upper_jins": maqam_def.get("upper", "")
        }

    def generate_indian_raga(self, raga_name: str, ascending: bool = True) -> Dict:
        """
        Generate Indian raga scale.

        Args:
            raga_name: Name of raga
            ascending: True for arohana, False for avarohana

        Returns:
            Dictionary with raga data
        """
        scale = IndianRaga.get_raga_scale(raga_name, ascending)
        raga_info = IndianRaga.get_raga(raga_name)

        root_midi = 12 * self.octave + self.root
        pitches = scale.get_pitches(root_midi, octave_span=1, bend_range=2)

        return {
            "style": "Indian Raga",
            "raga": raga_name,
            "direction": "ascending" if ascending else "descending",
            "time": raga_info.get("time", ""),
            "rasa": raga_info.get("rasa", ""),
            "vadi": raga_info.get("vadi"),
            "intervals_cents": scale.intervals,
            "midi_with_bends": pitches
        }

    def generate_turkish_makam(self, makam_name: str) -> Dict:
        """
        Generate Turkish makam scale.

        Args:
            makam_name: Name of makam

        Returns:
            Dictionary with makam data
        """
        scale = TurkishMakam.get_makam(makam_name)
        makam_def = TurkishMakam.MAKAMLAR.get(makam_name, {})

        root_midi = 12 * self.octave + self.root
        pitches = scale.get_pitches(root_midi, octave_span=1, bend_range=2)

        return {
            "style": "Turkish Makam",
            "makam": makam_name,
            "description": makam_def.get("description", ""),
            "system": "53-TET",
            "intervals_cents": scale.intervals,
            "midi_with_bends": pitches
        }

    def generate_persian_dastgah(self, dastgah_name: str) -> Dict:
        """
        Generate Persian dastgah scale.

        Args:
            dastgah_name: Name of dastgah

        Returns:
            Dictionary with dastgah data
        """
        scale = PersianDastgah.get_dastgah(dastgah_name)
        dastgah_def = PersianDastgah.DASTGAH_HA.get(dastgah_name, {})

        root_midi = 12 * self.octave + self.root
        pitches = scale.get_pitches(root_midi, octave_span=1, bend_range=2)

        return {
            "style": "Persian Dastgah",
            "dastgah": dastgah_name,
            "description": dastgah_def.get("description", ""),
            "gooshe_count": dastgah_def.get("gooshe_count", 0),
            "intervals_cents": scale.intervals,
            "midi_with_bends": pitches
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED HARMONY GENERATOR - EXAMPLES")
    print("=" * 70)

    gen = AdvancedHarmonyGenerator(root=0, octave=4)  # C root

    # Example 1: Neo-Riemannian progression
    print("\n1. NEO-RIEMANNIAN PROGRESSION")
    print("-" * 70)
    nr_prog = gen.generate_neo_riemannian("P L R", voice_lead=True)
    print(f"Style: {nr_prog['style']}")
    print(f"Transformations: {nr_prog['transformations']}")
    print(f"Triads: {' → '.join(nr_prog['triads'])}")
    print(f"Voice leading: {nr_prog['voice_leading_analysis']['avg_motion']:.2f} semitones avg")

    # Example 2: Hexatonic cycle
    print("\n2. HEXATONIC CYCLE")
    print("-" * 70)
    hex_prog = gen.generate_hexatonic_cycle(pole=0)
    print(f"Style: {hex_prog['style']}")
    print(f"Pole: {hex_prog['pole']}")
    print(f"Cycle: {' → '.join(hex_prog['triads'])}")

    # Example 3: Dorian modal progression
    print("\n3. DORIAN MODAL PROGRESSION")
    print("-" * 70)
    modal_prog = gen.generate_modal_progression(Mode.DORIAN, "characteristic")
    print(f"Style: {modal_prog['style']}")
    print(f"Mode: {modal_prog['mode']}")
    print(f"Brightness: {modal_prog['brightness']}")
    print(f"Chords: {len(modal_prog['midi_notes'])}")

    # Example 4: Modal interchange
    print("\n4. MODAL INTERCHANGE (Borrowing from parallel minor)")
    print("-" * 70)
    interchange = gen.generate_modal_interchange(
        Mode.IONIAN, Mode.AEOLIAN,
        primary_degrees=[1, 4, 5, 1],
        insert_positions=[1, 3]
    )
    print(f"Primary mode: {interchange['primary_mode']}")
    print(f"Borrowed from: {interchange['borrowed_mode']}")
    print(f"Borrowed at positions: {interchange['borrowed_positions']}")

    # Example 5: Arabic Maqam Rast
    print("\n5. ARABIC MAQAM RAST")
    print("-" * 70)
    maqam = gen.generate_arabic_maqam(ArabicMaqam.RAST)
    print(f"Style: {maqam['style']}")
    print(f"Maqam: {maqam['maqam']}")
    print(f"Description: {maqam['description']}")
    print(f"Jins: {maqam['lower_jins']} + {maqam['upper_jins']}")

    # Example 6: Indian Raga Bhairav
    print("\n6. INDIAN RAGA BHAIRAV")
    print("-" * 70)
    raga = gen.generate_indian_raga("Bhairav")
    print(f"Style: {raga['style']}")
    print(f"Raga: {raga['raga']}")
    print(f"Time: {raga['time']}")
    print(f"Rasa: {raga['rasa']}")

    # Example 7: Turkish Makam Hicaz
    print("\n7. TURKISH MAKAM HICAZ")
    print("-" * 70)
    makam = gen.generate_turkish_makam("Hicaz")
    print(f"Style: {makam['style']}")
    print(f"Makam: {makam['makam']}")
    print(f"System: {makam['system']}")
    print(f"Description: {makam['description']}")

    # Example 8: Persian Dastgah Shur
    print("\n8. PERSIAN DASTGAH SHUR")
    print("-" * 70)
    dastgah = gen.generate_persian_dastgah("Shur")
    print(f"Style: {dastgah['style']}")
    print(f"Dastgah: {dastgah['dastgah']}")
    print(f"Description: {dastgah['description']}")

    # Example 9: 24-TET quarter-tone scale
    print("\n9. 24-TET QUARTER-TONE SCALE")
    print("-" * 70)
    microtonal = gen.generate_microtonal_scale("24-TET")
    print(f"Style: {microtonal['style']}")
    print(f"System: {microtonal['system']}")
    print(f"Scale: {microtonal['scale_name']}")

    # Example 10: Just intonation major scale
    print("\n10. JUST INTONATION MAJOR SCALE")
    print("-" * 70)
    just = gen.generate_just_intonation_scale("major")
    print(f"Style: {just['style']}")
    print(f"Scale: {just['scale_name']}")
    print(f"First 4 intervals: {[f'{c:.2f}' for c in just['intervals_cents'][:4]]}")

    print("\n" + "=" * 70)
