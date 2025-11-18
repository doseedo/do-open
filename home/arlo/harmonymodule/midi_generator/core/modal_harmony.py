#!/usr/bin/env python3
"""
Modal Harmony System
====================

Comprehensive implementation of modal harmony theory, covering church modes,
harmonic/melodic minor modes, modal interchange, and modal composition techniques.

Modal harmony differs from functional harmony by emphasizing horizontal melodic
motion and characteristic scale degrees rather than hierarchical chord progressions.
Each mode has a unique "color" or "brightness" determined by its interval structure.

Theory Background:
-----------------
The seven church modes derive from different rotations of the major scale:
- Ionian (major): Brightest, same as major scale
- Dorian: Minor with raised 6th, jazz/folk character
- Phrygian: Minor with lowered 2nd, Spanish/flamenco flavor
- Lydian: Major with raised 4th, dreamy/ethereal
- Mixolydian: Major with lowered 7th, rock/blues character
- Aeolian: Natural minor, melancholic
- Locrian: Darkest, diminished character (rarely used)

Modal interchange allows borrowing chords from parallel modes, creating
color and chromaticism while maintaining tonal center.

Applications:
------------
- Jazz improvisation (modal jazz, Miles Davis)
- Film scoring (modal ambiguity)
- Rock/metal (Dorian, Phrygian)
- Folk music (Dorian, Mixolydian)
- Contemporary classical (modal writing)

References:
----------
- George Russell: "Lydian Chromatic Concept of Tonal Organization" (1953)
- Jerry Coker: "Improvising Jazz" (modal jazz)
- Vincent Persichetti: "Twentieth Century Harmony" (modal techniques)
- Olivier Messiaen: "Modes of Limited Transposition"

Author: Agent 3 - Advanced Harmony & Modal Systems
License: MIT
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CORE MODAL STRUCTURES
# ============================================================================

class Mode(Enum):
    """Church modes (diatonic modes)"""
    IONIAN = "Ionian"
    DORIAN = "Dorian"
    PHRYGIAN = "Phrygian"
    LYDIAN = "Lydian"
    MIXOLYDIAN = "Mixolydian"
    AEOLIAN = "Aeolian"
    LOCRIAN = "Locrian"


class HarmonicMinorMode(Enum):
    """Modes of the harmonic minor scale"""
    HARMONIC_MINOR = "Harmonic Minor"
    LOCRIAN_NAT6 = "Locrian ♮6"
    IONIAN_SHARP5 = "Ionian ♯5"
    DORIAN_SHARP4 = "Dorian ♯4"
    PHRYGIAN_DOMINANT = "Phrygian Dominant"
    LYDIAN_SHARP2 = "Lydian ♯2"
    SUPER_LOCRIAN_BB7 = "Super Locrian 𝄫7"


class MelodicMinorMode(Enum):
    """Modes of the melodic minor scale (ascending)"""
    MELODIC_MINOR = "Melodic Minor"
    DORIAN_FLAT2 = "Dorian ♭2"
    LYDIAN_AUGMENTED = "Lydian Augmented"
    LYDIAN_DOMINANT = "Lydian Dominant"
    MIXOLYDIAN_FLAT6 = "Mixolydian ♭6"
    LOCRIAN_NAT2 = "Locrian ♮2"
    ALTERED = "Altered"


class SymmetricalScale(Enum):
    """Symmetrical/synthetic scales"""
    WHOLE_TONE = "Whole Tone"
    DIMINISHED_HALF_WHOLE = "Diminished (H-W)"
    DIMINISHED_WHOLE_HALF = "Diminished (W-H)"
    AUGMENTED = "Augmented"
    PROMETHEUS = "Prometheus"
    TRITONE = "Tritone"


@dataclass
class ModalScale:
    """
    Complete modal scale definition.

    Attributes:
        name: Mode name
        intervals: Semitone intervals from tonic
        characteristic_degrees: Scale degrees that define the mode's character
        avoid_notes: Notes to avoid in melody (typically one scale degree)
        chord_quality: Quality of tonic triad (major, minor, diminished)
        brightness: Relative brightness (higher = brighter)
    """
    name: str
    intervals: Tuple[int, ...]
    characteristic_degrees: Tuple[int, ...]
    avoid_notes: Tuple[int, ...]
    chord_quality: str
    brightness: int

    def get_scale_degrees(self, root: int) -> List[int]:
        """
        Get all scale degrees as MIDI pitch classes.

        Args:
            root: Root note (0-11)

        Returns:
            List of pitch classes
        """
        return [(root + interval) % 12 for interval in self.intervals]

    def get_modal_chord(self, root: int, degree: int, octave: int = 4) -> List[int]:
        """
        Build a diatonic chord on a scale degree.

        Args:
            root: Scale root (0-11)
            degree: Scale degree (1-7)
            octave: MIDI octave

        Returns:
            Triad as MIDI notes
        """
        scale_notes = self.get_scale_degrees(root)
        base = 12 * octave + root

        # Get chord tones (1, 3, 5 of the scale degree)
        degree_idx = degree - 1
        chord_indices = [degree_idx, (degree_idx + 2) % 7, (degree_idx + 4) % 7]

        chord = []
        for idx in chord_indices:
            note = scale_notes[idx]
            # Adjust octave if needed to maintain ascending order
            if chord and note < chord[-1]:
                note += 12
            chord.append(base + note)

        return chord


# ============================================================================
# MODAL SCALE LIBRARY
# ============================================================================

class ModalScaleLibrary:
    """
    Comprehensive library of modal scales.

    Provides definitions and methods for all common modal systems.
    """

    # Church Modes (diatonic)
    CHURCH_MODES: Dict[Mode, ModalScale] = {
        Mode.IONIAN: ModalScale(
            name="Ionian",
            intervals=(0, 2, 4, 5, 7, 9, 11),
            characteristic_degrees=(4, 7),  # Perfect 4th, Major 7th
            avoid_notes=(4,),  # Avoid 4th over major chord
            chord_quality="major",
            brightness=6
        ),
        Mode.DORIAN: ModalScale(
            name="Dorian",
            intervals=(0, 2, 3, 5, 7, 9, 10),
            characteristic_degrees=(6, 3),  # Major 6th, minor 3rd
            avoid_notes=(6,),  # Avoid 6th (context-dependent)
            chord_quality="minor",
            brightness=4
        ),
        Mode.PHRYGIAN: ModalScale(
            name="Phrygian",
            intervals=(0, 1, 3, 5, 7, 8, 10),
            characteristic_degrees=(2, 6),  # Minor 2nd, minor 6th
            avoid_notes=(2,),  # Avoid ♭2 over minor chord
            chord_quality="minor",
            brightness=2
        ),
        Mode.LYDIAN: ModalScale(
            name="Lydian",
            intervals=(0, 2, 4, 6, 7, 9, 11),
            characteristic_degrees=(4, 7),  # ♯4, Major 7th
            avoid_notes=(),  # No avoid notes
            chord_quality="major",
            brightness=7
        ),
        Mode.MIXOLYDIAN: ModalScale(
            name="Mixolydian",
            intervals=(0, 2, 4, 5, 7, 9, 10),
            characteristic_degrees=(7, 3),  # ♭7, Major 3rd
            avoid_notes=(4,),  # Avoid 4th
            chord_quality="major",
            brightness=5
        ),
        Mode.AEOLIAN: ModalScale(
            name="Aeolian",
            intervals=(0, 2, 3, 5, 7, 8, 10),
            characteristic_degrees=(6, 3),  # Minor 6th, minor 3rd
            avoid_notes=(6,),  # Avoid ♭6
            chord_quality="minor",
            brightness=3
        ),
        Mode.LOCRIAN: ModalScale(
            name="Locrian",
            intervals=(0, 1, 3, 5, 6, 8, 10),
            characteristic_degrees=(5, 2),  # ♭5, ♭2
            avoid_notes=(2,),  # Avoid ♭2
            chord_quality="diminished",
            brightness=1
        ),
    }

    # Harmonic Minor Modes
    HARMONIC_MINOR_MODES: Dict[HarmonicMinorMode, ModalScale] = {
        HarmonicMinorMode.HARMONIC_MINOR: ModalScale(
            name="Harmonic Minor",
            intervals=(0, 2, 3, 5, 7, 8, 11),
            characteristic_degrees=(6, 7),  # ♭6, Major 7
            avoid_notes=(6,),
            chord_quality="minor",
            brightness=3
        ),
        HarmonicMinorMode.LOCRIAN_NAT6: ModalScale(
            name="Locrian ♮6",
            intervals=(0, 1, 3, 5, 6, 9, 10),
            characteristic_degrees=(6, 5),  # Natural 6, ♭5
            avoid_notes=(2,),
            chord_quality="diminished",
            brightness=2
        ),
        HarmonicMinorMode.IONIAN_SHARP5: ModalScale(
            name="Ionian ♯5",
            intervals=(0, 2, 4, 5, 8, 9, 11),
            characteristic_degrees=(5,),  # ♯5
            avoid_notes=(4,),
            chord_quality="augmented",
            brightness=6
        ),
        HarmonicMinorMode.DORIAN_SHARP4: ModalScale(
            name="Dorian ♯4",
            intervals=(0, 2, 3, 6, 7, 9, 10),
            characteristic_degrees=(4,),  # ♯4
            avoid_notes=(),
            chord_quality="minor",
            brightness=4
        ),
        HarmonicMinorMode.PHRYGIAN_DOMINANT: ModalScale(
            name="Phrygian Dominant",
            intervals=(0, 1, 4, 5, 7, 8, 10),
            characteristic_degrees=(2, 3),  # ♭2, Major 3
            avoid_notes=(2,),
            chord_quality="major",
            brightness=4
        ),
        HarmonicMinorMode.LYDIAN_SHARP2: ModalScale(
            name="Lydian ♯2",
            intervals=(0, 3, 4, 6, 7, 9, 11),
            characteristic_degrees=(2, 4),  # ♯2, ♯4
            avoid_notes=(),
            chord_quality="major",
            brightness=7
        ),
        HarmonicMinorMode.SUPER_LOCRIAN_BB7: ModalScale(
            name="Super Locrian 𝄫7",
            intervals=(0, 1, 3, 5, 6, 8, 9),
            characteristic_degrees=(7,),  # 𝄫7
            avoid_notes=(2,),
            chord_quality="diminished",
            brightness=1
        ),
    }

    # Melodic Minor Modes
    MELODIC_MINOR_MODES: Dict[MelodicMinorMode, ModalScale] = {
        MelodicMinorMode.MELODIC_MINOR: ModalScale(
            name="Melodic Minor",
            intervals=(0, 2, 3, 5, 7, 9, 11),
            characteristic_degrees=(3, 6, 7),  # Minor 3, Major 6, Major 7
            avoid_notes=(6,),
            chord_quality="minor",
            brightness=5
        ),
        MelodicMinorMode.DORIAN_FLAT2: ModalScale(
            name="Dorian ♭2",
            intervals=(0, 1, 3, 5, 7, 9, 10),
            characteristic_degrees=(2,),  # ♭2
            avoid_notes=(2,),
            chord_quality="minor",
            brightness=3
        ),
        MelodicMinorMode.LYDIAN_AUGMENTED: ModalScale(
            name="Lydian Augmented",
            intervals=(0, 2, 4, 6, 8, 9, 11),
            characteristic_degrees=(4, 5),  # ♯4, ♯5
            avoid_notes=(),
            chord_quality="augmented",
            brightness=8
        ),
        MelodicMinorMode.LYDIAN_DOMINANT: ModalScale(
            name="Lydian Dominant",
            intervals=(0, 2, 4, 6, 7, 9, 10),
            characteristic_degrees=(4, 7),  # ♯4, ♭7
            avoid_notes=(),
            chord_quality="dominant",
            brightness=6
        ),
        MelodicMinorMode.MIXOLYDIAN_FLAT6: ModalScale(
            name="Mixolydian ♭6",
            intervals=(0, 2, 4, 5, 7, 8, 10),
            characteristic_degrees=(6,),  # ♭6
            avoid_notes=(4,),
            chord_quality="major",
            brightness=4
        ),
        MelodicMinorMode.LOCRIAN_NAT2: ModalScale(
            name="Locrian ♮2",
            intervals=(0, 2, 3, 5, 6, 8, 10),
            characteristic_degrees=(2, 5),  # Natural 2, ♭5
            avoid_notes=(),
            chord_quality="diminished",
            brightness=2
        ),
        MelodicMinorMode.ALTERED: ModalScale(
            name="Altered",
            intervals=(0, 1, 3, 4, 6, 8, 10),
            characteristic_degrees=(2, 4, 5, 6),  # ♭9, ♯9, ♭5, ♯5
            avoid_notes=(),
            chord_quality="altered_dominant",
            brightness=1
        ),
    }

    # Symmetrical Scales
    SYMMETRICAL_SCALES: Dict[SymmetricalScale, ModalScale] = {
        SymmetricalScale.WHOLE_TONE: ModalScale(
            name="Whole Tone",
            intervals=(0, 2, 4, 6, 8, 10),
            characteristic_degrees=(2, 4, 6),  # All whole steps
            avoid_notes=(),
            chord_quality="augmented",
            brightness=7
        ),
        SymmetricalScale.DIMINISHED_HALF_WHOLE: ModalScale(
            name="Diminished (H-W)",
            intervals=(0, 1, 3, 4, 6, 7, 9, 10),
            characteristic_degrees=(2, 4),  # ♭9, ♯9
            avoid_notes=(),
            chord_quality="diminished",
            brightness=2
        ),
        SymmetricalScale.DIMINISHED_WHOLE_HALF: ModalScale(
            name="Diminished (W-H)",
            intervals=(0, 2, 3, 5, 6, 8, 9, 11),
            characteristic_degrees=(3, 5),  # Minor 3, ♭5
            avoid_notes=(),
            chord_quality="diminished",
            brightness=3
        ),
        SymmetricalScale.AUGMENTED: ModalScale(
            name="Augmented",
            intervals=(0, 3, 4, 7, 8, 11),
            characteristic_degrees=(3, 5),  # Minor 3, ♯5
            avoid_notes=(),
            chord_quality="augmented",
            brightness=6
        ),
    }

    @classmethod
    def get_mode(cls, mode: Mode) -> ModalScale:
        """Get church mode definition"""
        return cls.CHURCH_MODES[mode]

    @classmethod
    def get_brightness_order(cls) -> List[Tuple[str, int]]:
        """
        Get modes ordered by brightness (darkest to brightest).

        Returns:
            List of (mode_name, brightness) tuples
        """
        modes = [(m.name, m.brightness) for m in cls.CHURCH_MODES.values()]
        return sorted(modes, key=lambda x: x[1])


# ============================================================================
# MODAL INTERCHANGE (BORROWED CHORDS)
# ============================================================================

class ModalInterchange:
    """
    Modal interchange system for borrowing chords from parallel modes.

    Allows chromatic enrichment while maintaining tonal center.
    """

    def __init__(self, tonic: int, primary_mode: Mode = Mode.IONIAN):
        """
        Initialize modal interchange system.

        Args:
            tonic: Tonic pitch class (0-11)
            primary_mode: Primary mode (usually Ionian or Aeolian)
        """
        self.tonic = tonic % 12
        self.primary_mode = primary_mode

    def get_parallel_mode_chords(self, target_mode: Mode) -> Dict[int, List[int]]:
        """
        Get all diatonic chords from a parallel mode.

        Args:
            target_mode: Mode to borrow from

        Returns:
            Dictionary mapping scale degree to chord (MIDI notes)
        """
        mode_def = ModalScaleLibrary.get_mode(target_mode)
        chords = {}

        for degree in range(1, 8):
            chords[degree] = mode_def.get_modal_chord(self.tonic, degree)

        return chords

    def get_common_borrowed_chords(self, major_key: bool = True) -> Dict[str, List[int]]:
        """
        Get commonly borrowed chords.

        Args:
            major_key: True for major key, False for minor

        Returns:
            Dictionary of chord name to MIDI notes
        """
        borrowed = {}

        if major_key:
            # Borrowing from parallel minor (Aeolian)
            minor_chords = self.get_parallel_mode_chords(Mode.AEOLIAN)
            borrowed["♭III"] = minor_chords[3]  # ♭III major
            borrowed["iv"] = minor_chords[4]    # iv minor (subdominant minor)
            borrowed["♭VI"] = minor_chords[6]   # ♭VI major
            borrowed["♭VII"] = minor_chords[7]  # ♭VII major

            # From Phrygian
            phrygian_chords = self.get_parallel_mode_chords(Mode.PHRYGIAN)
            borrowed["♭II"] = phrygian_chords[2]  # ♭II major (Neapolitan)

        else:
            # Borrowing from parallel major (Ionian)
            major_chords = self.get_parallel_mode_chords(Mode.IONIAN)
            borrowed["III"] = major_chords[3]   # III major
            borrowed["IV"] = major_chords[4]    # IV major
            borrowed["VI"] = major_chords[6]    # VI major
            borrowed["VII"] = major_chords[7]   # VII diminished

        return borrowed

    def create_modal_mixture_progression(self, primary_degrees: List[int],
                                          borrowed_mode: Mode,
                                          insert_positions: List[int]) -> List[List[int]]:
        """
        Create progression with modal mixture.

        Args:
            primary_degrees: Scale degrees in primary mode
            borrowed_mode: Mode to borrow from
            insert_positions: Positions to insert borrowed chords

        Returns:
            List of chords (MIDI notes)
        """
        primary_def = ModalScaleLibrary.get_mode(self.primary_mode)
        borrowed_def = ModalScaleLibrary.get_mode(borrowed_mode)

        progression = []
        for i, degree in enumerate(primary_degrees):
            if i in insert_positions:
                chord = borrowed_def.get_modal_chord(self.tonic, degree)
            else:
                chord = primary_def.get_modal_chord(self.tonic, degree)
            progression.append(chord)

        return progression


# ============================================================================
# MODAL PROGRESSION GENERATOR
# ============================================================================

class ModalProgressionGenerator:
    """
    Generate modal progressions using modal composition techniques.

    Modal progressions avoid functional harmony and instead emphasize:
    - Static harmony (vamps)
    - Plagal motion (subdominant emphasis)
    - Avoidance of V-I cadences
    - Pedal points and drones
    """

    def __init__(self, root: int, mode: Mode):
        """
        Initialize generator.

        Args:
            root: Root pitch class (0-11)
            mode: Modal scale
        """
        self.root = root % 12
        self.mode = mode
        self.mode_def = ModalScaleLibrary.get_mode(mode)

    def generate_vamp(self, degree1: int, degree2: int, bars: int = 4) -> List[List[int]]:
        """
        Generate a modal vamp (two-chord oscillation).

        Common in modal jazz and rock.

        Args:
            degree1: First scale degree
            degree2: Second scale degree
            bars: Number of bars (alternates each bar)

        Returns:
            List of chords
        """
        chord1 = self.mode_def.get_modal_chord(self.root, degree1)
        chord2 = self.mode_def.get_modal_chord(self.root, degree2)

        progression = []
        for i in range(bars):
            progression.append(chord1 if i % 2 == 0 else chord2)

        return progression

    def generate_plagal_progression(self, length: int = 4) -> List[List[int]]:
        """
        Generate plagal (subdominant-based) progression.

        Emphasizes IV-I motion, common in Mixolydian and Dorian.

        Args:
            length: Number of chords

        Returns:
            List of chords
        """
        # Common plagal degrees: IV, ♭VII, I
        if self.mode in [Mode.MIXOLYDIAN, Mode.DORIAN]:
            degrees = [4, 7, 1, 4]  # IV - ♭VII - I - IV
        elif self.mode == Mode.LYDIAN:
            degrees = [2, 4, 1, 2]  # ii - IV - I - ii
        else:
            degrees = [4, 1, 4, 1]  # Simple IV - I

        progression = []
        for i in range(length):
            degree = degrees[i % len(degrees)]
            chord = self.mode_def.get_modal_chord(self.root, degree)
            progression.append(chord)

        return progression

    def generate_descending_progression(self, start_degree: int = 7,
                                        steps: int = 4) -> List[List[int]]:
        """
        Generate descending modal progression.

        Common in Aeolian and Phrygian.

        Args:
            start_degree: Starting scale degree
            steps: Number of descending steps

        Returns:
            List of chords
        """
        progression = []
        for i in range(steps):
            degree = start_degree - i
            if degree < 1:
                degree += 7
            chord = self.mode_def.get_modal_chord(self.root, degree)
            progression.append(chord)

        return progression

    def generate_characteristic_progression(self) -> List[List[int]]:
        """
        Generate progression emphasizing mode's characteristic tones.

        Returns:
            List of chords showcasing the mode
        """
        progressions = {
            Mode.DORIAN: [1, 4, 5, 1],      # i - IV - v - i (major IV)
            Mode.PHRYGIAN: [1, 2, 1, 2],    # i - ♭II - i - ♭II (♭II emphasis)
            Mode.LYDIAN: [1, 2, 1, 2],      # I - II - I - II (II major)
            Mode.MIXOLYDIAN: [1, 7, 4, 1],  # I - ♭VII - IV - I
            Mode.AEOLIAN: [1, 6, 7, 1],     # i - ♭VI - ♭VII - i
            Mode.LOCRIAN: [1, 2, 1, 2],     # i° - ♭II - i° - ♭II
            Mode.IONIAN: [1, 4, 5, 1],      # I - IV - V - I (standard)
        }

        degrees = progressions.get(self.mode, [1, 4, 5, 1])
        return [self.mode_def.get_modal_chord(self.root, deg) for deg in degrees]


# ============================================================================
# MODAL CADENCES
# ============================================================================

class ModalCadence:
    """
    Modal cadence formulas.

    Modal cadences avoid dominant-tonic resolution, using instead:
    - Plagal cadences (IV-I)
    - Double plagal (♭VII-IV-I)
    - Phrygian cadence (♭II-I)
    - Dorian cadence (iv-I)
    """

    @staticmethod
    def get_cadence(root: int, mode: Mode, cadence_type: str) -> List[List[int]]:
        """
        Get a modal cadence.

        Args:
            root: Root pitch class
            mode: Mode
            cadence_type: 'plagal', 'double_plagal', 'phrygian', 'dorian'

        Returns:
            List of chords forming cadence
        """
        mode_def = ModalScaleLibrary.get_mode(mode)

        cadences = {
            'plagal': [4, 1],              # IV - I
            'double_plagal': [7, 4, 1],    # ♭VII - IV - I
            'phrygian': [2, 1],            # ♭II - I
            'dorian': [4, 1],              # iv - I (in minor context)
            'lydian': [2, 1],              # II - I
            'aeolian': [7, 6, 5, 1],       # ♭VII - ♭VI - v - i
        }

        degrees = cadences.get(cadence_type, [5, 1])
        return [mode_def.get_modal_chord(root, deg) for deg in degrees]


# ============================================================================
# PEDAL POINT GENERATOR
# ============================================================================

class PedalPointGenerator:
    """
    Generate harmony over pedal points (sustained bass notes).

    Essential for modal music, especially Phrygian and Dorian.
    """

    def __init__(self, root: int, mode: Mode):
        """Initialize generator"""
        self.root = root % 12
        self.mode = mode
        self.mode_def = ModalScaleLibrary.get_mode(mode)

    def generate_over_pedal(self, pedal_note: int, degrees: List[int],
                           octave: int = 3) -> Tuple[int, List[List[int]]]:
        """
        Generate chord progression over sustained pedal.

        Args:
            pedal_note: MIDI note for pedal (bass)
            degrees: Scale degrees for upper harmony
            octave: Octave for upper chords

        Returns:
            Tuple of (pedal_note, list of upper voicings)
        """
        upper_chords = []
        for degree in degrees:
            chord = self.mode_def.get_modal_chord(self.root, degree, octave)
            upper_chords.append(chord)

        return (pedal_note, upper_chords)

    def generate_modal_drone(self, drone_interval: int = 7) -> Tuple[int, int]:
        """
        Generate drone (two sustained notes).

        Args:
            drone_interval: Interval above root (7 = perfect fifth)

        Returns:
            Tuple of (root_note, fifth_note) in MIDI
        """
        root_note = 12 * 3 + self.root  # Octave 3
        drone_note = root_note + drone_interval
        return (root_note, drone_note)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MODAL HARMONY SYSTEM - EXAMPLES")
    print("=" * 70)

    # Example 1: Mode brightness comparison
    print("\n1. MODE BRIGHTNESS ORDER")
    print("-" * 70)
    brightness = ModalScaleLibrary.get_brightness_order()
    for mode_name, bright in brightness:
        print(f"{mode_name:20} Brightness: {bright}")

    # Example 2: Dorian progression
    print("\n2. DORIAN MODAL PROGRESSION")
    print("-" * 70)
    dorian_gen = ModalProgressionGenerator(2, Mode.DORIAN)  # D Dorian
    dorian_prog = dorian_gen.generate_characteristic_progression()
    print(f"D Dorian characteristic progression:")
    for i, chord in enumerate(dorian_prog, 1):
        print(f"  Chord {i}: {chord}")

    # Example 3: Modal interchange
    print("\n3. MODAL INTERCHANGE (Borrowed Chords)")
    print("-" * 70)
    interchange = ModalInterchange(0, Mode.IONIAN)  # C major
    borrowed = interchange.get_common_borrowed_chords(major_key=True)
    print("Chords borrowed from parallel minor:")
    for name, chord in borrowed.items():
        print(f"  {name}: {chord}")

    # Example 4: Lydian progression
    print("\n4. LYDIAN MODAL PROGRESSION")
    print("-" * 70)
    lydian_gen = ModalProgressionGenerator(5, Mode.LYDIAN)  # F Lydian
    lydian_prog = lydian_gen.generate_plagal_progression(4)
    print(f"F Lydian plagal progression:")
    for i, chord in enumerate(lydian_prog, 1):
        print(f"  Chord {i}: {chord}")

    # Example 5: Phrygian cadence
    print("\n5. PHRYGIAN CADENCE")
    print("-" * 70)
    phrygian_cadence = ModalCadence.get_cadence(4, Mode.PHRYGIAN, 'phrygian')
    print(f"E Phrygian cadence (♭II - I):")
    for i, chord in enumerate(phrygian_cadence, 1):
        print(f"  Chord {i}: {chord}")

    # Example 6: Harmonic minor modes
    print("\n6. HARMONIC MINOR MODES")
    print("-" * 70)
    hm_mode = ModalScaleLibrary.HARMONIC_MINOR_MODES[HarmonicMinorMode.PHRYGIAN_DOMINANT]
    print(f"{hm_mode.name}:")
    print(f"  Intervals: {hm_mode.intervals}")
    print(f"  Characteristic degrees: {hm_mode.characteristic_degrees}")
    print(f"  Chord quality: {hm_mode.chord_quality}")

    # Example 7: Melodic minor modes
    print("\n7. MELODIC MINOR MODES")
    print("-" * 70)
    mm_mode = ModalScaleLibrary.MELODIC_MINOR_MODES[MelodicMinorMode.LYDIAN_DOMINANT]
    print(f"{mm_mode.name}:")
    print(f"  Intervals: {mm_mode.intervals}")
    print(f"  Characteristic degrees: {mm_mode.characteristic_degrees}")
    print(f"  Use: Jazz dominant chords (7♯11)")

    # Example 8: Mixolydian vamp
    print("\n8. MIXOLYDIAN VAMP")
    print("-" * 70)
    mixo_gen = ModalProgressionGenerator(7, Mode.MIXOLYDIAN)  # G Mixolydian
    vamp = mixo_gen.generate_vamp(1, 7, bars=4)
    print(f"G Mixolydian vamp (I - ♭VII):")
    for i, chord in enumerate(vamp, 1):
        print(f"  Bar {i}: {chord}")

    # Example 9: Pedal point
    print("\n9. PEDAL POINT HARMONY")
    print("-" * 70)
    pedal_gen = PedalPointGenerator(9, Mode.DORIAN)  # A Dorian
    pedal, upper = pedal_gen.generate_over_pedal(33, [1, 4, 5, 1])  # A2 bass
    print(f"A Dorian over A pedal (bass: {pedal}):")
    for i, chord in enumerate(upper, 1):
        print(f"  Chord {i}: {chord}")

    # Example 10: Symmetrical scales
    print("\n10. SYMMETRICAL SCALES")
    print("-" * 70)
    whole_tone = ModalScaleLibrary.SYMMETRICAL_SCALES[SymmetricalScale.WHOLE_TONE]
    print(f"{whole_tone.name}:")
    print(f"  Intervals: {whole_tone.intervals}")
    print(f"  Pitch classes (C root): {whole_tone.get_scale_degrees(0)}")

    print("\n" + "=" * 70)
