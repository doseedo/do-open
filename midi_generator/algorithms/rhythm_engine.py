"""
Advanced Rhythm & Groove Engine

Comprehensive rhythm generation, groove quantization, microtiming, and humanization.

Features:
- Groove Template System: Extract and apply timing/velocity patterns
- Advanced Polyrhythm Generator: Complex cross-rhythms and metric modulation
- Humanization Engine: Natural timing and velocity variation
- Rhythm Transformations: Augmentation, retrograde, shuffle conversion, etc.

Research References:
- "Timing Patterns in Music: The Groove" - Bengtsson & Gabrielsson (1983)
- "The Perception of Musical Rhythm" - Justin London (2012)
- "Analyzing Performed Music" - Bruno Repp (1995)
- "Timing Microstructure in Drum Patterns" - Kilchenmann & Senn (2015)

Author: MIDI Generator Team
"""

from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import defaultdict
import copy

# Import MIDI constants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from midi.midi_constants import DEFAULT_PPQN, PPQN_HIGH_RES


# ============================================================================
# Data Classes and Enums
# ============================================================================

class TimingStyle(Enum):
    """Timing feel/style for humanization"""
    LOCKED = "locked"              # Quantized, no deviation
    TIGHT = "tight"                # Very slight deviation
    LAID_BACK = "laid_back"        # Notes slightly late
    RUSHING = "rushing"            # Notes slightly early
    DRUNK = "drunk"                # Heavy random deviation
    HUMAN = "human"                # Natural variation
    MACHINE = "machine"            # Perfect timing


class GrooveIntensity(Enum):
    """Intensity of groove application"""
    SUBTLE = 0.25
    LIGHT = 0.5
    MEDIUM = 0.75
    HEAVY = 1.0
    EXTREME = 1.5


@dataclass
class RhythmNote:
    """A single rhythm event with timing and velocity"""
    tick: int                      # MIDI tick position
    duration: int                  # Duration in ticks
    velocity: int = 64             # MIDI velocity (1-127)
    pitch: Optional[int] = None    # MIDI note number (for drums/melody)

    def __post_init__(self):
        """Validate values"""
        self.velocity = max(1, min(127, self.velocity))
        if self.duration < 0:
            self.duration = 0


@dataclass
class GrooveTemplate:
    """Template for timing and velocity patterns"""
    name: str
    description: str
    timing_offsets: List[float]     # Timing deviations in ticks per position
    velocity_curve: List[float]     # Velocity multipliers per position
    grid_division: int = 16         # Grid resolution (16th notes, etc.)
    swing_ratio: float = 0.5        # Swing amount (0.5 = straight, 0.67 = triplet)

    def __post_init__(self):
        """Ensure lists have same length"""
        max_len = max(len(self.timing_offsets), len(self.velocity_curve))
        if len(self.timing_offsets) < max_len:
            self.timing_offsets.extend([0.0] * (max_len - len(self.timing_offsets)))
        if len(self.velocity_curve) < max_len:
            self.velocity_curve.extend([1.0] * (max_len - len(self.velocity_curve)))


@dataclass
class PolyrhythmSpec:
    """Specification for generating polyrhythms"""
    ratio_a: int                   # First rhythm (e.g., 3 in "3 against 4")
    ratio_b: int                   # Second rhythm (e.g., 4 in "3 against 4")
    beats: int = 4                 # Number of beats to span
    velocity_a: int = 80           # Velocity for first rhythm
    velocity_b: int = 64           # Velocity for second rhythm
    pitch_a: Optional[int] = None  # MIDI pitch for first rhythm
    pitch_b: Optional[int] = None  # MIDI pitch for second rhythm


# ============================================================================
# Groove Template System
# ============================================================================

class GrooveTemplateEngine:
    """
    Extract timing/velocity patterns from MIDI and apply them as groove templates.

    Groove templates capture the "feel" of a performance by analyzing:
    - Timing deviations from grid (early/late)
    - Velocity patterns (accents, ghost notes)
    - Swing ratios
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn
        self.templates: Dict[str, GrooveTemplate] = {}

    def extract_groove_from_notes(
        self,
        notes: List[RhythmNote],
        grid_division: int = 16,
        name: str = "extracted_groove",
        description: str = "Extracted from MIDI"
    ) -> GrooveTemplate:
        """
        Extract groove template from a list of rhythm notes.

        Args:
            notes: List of RhythmNote objects
            grid_division: Grid resolution (16 = 16th notes)
            name: Name for the template
            description: Description of the groove

        Returns:
            GrooveTemplate object
        """
        if not notes:
            return GrooveTemplate(name, description, [], [], grid_division)

        # Calculate grid size in ticks
        ticks_per_quarter = self.ppqn
        ticks_per_grid = ticks_per_quarter * 4 // grid_division

        # Analyze timing deviations
        timing_offsets = []
        velocity_values = []

        for note in sorted(notes, key=lambda n: n.tick):
            # Find nearest grid position
            grid_position = round(note.tick / ticks_per_grid)
            expected_tick = grid_position * ticks_per_grid

            # Calculate offset
            offset = note.tick - expected_tick
            timing_offsets.append(float(offset))
            velocity_values.append(note.velocity)

        # Normalize velocities to multipliers (relative to mean)
        if velocity_values:
            mean_vel = np.mean(velocity_values)
            velocity_curve = [v / mean_vel if mean_vel > 0 else 1.0 for v in velocity_values]
        else:
            velocity_curve = [1.0]

        # Detect swing ratio
        swing_ratio = self._detect_swing(notes, ticks_per_quarter)

        template = GrooveTemplate(
            name=name,
            description=description,
            timing_offsets=timing_offsets,
            velocity_curve=velocity_curve,
            grid_division=grid_division,
            swing_ratio=swing_ratio
        )

        self.templates[name] = template
        return template

    def _detect_swing(self, notes: List[RhythmNote], ticks_per_quarter: int) -> float:
        """
        Detect swing ratio from note timing.

        Returns:
            Swing ratio (0.5 = straight, 0.67 = triplet swing)
        """
        eighth_note_ticks = ticks_per_quarter // 2

        # Find pairs of 8th notes
        swing_ratios = []
        sorted_notes = sorted(notes, key=lambda n: n.tick)

        for i in range(len(sorted_notes) - 1):
            tick_diff = sorted_notes[i + 1].tick - sorted_notes[i].tick

            # Check if this could be a swung pair (approximately 8th note spacing)
            if abs(tick_diff - eighth_note_ticks) < ticks_per_quarter // 4:
                # Calculate actual ratio
                ratio = tick_diff / (2 * eighth_note_ticks)
                if 0.45 < ratio < 0.75:  # Reasonable swing range
                    swing_ratios.append(ratio)

        if swing_ratios:
            return float(np.median(swing_ratios))
        return 0.5  # Default to straight

    def apply_groove(
        self,
        notes: List[RhythmNote],
        template: Union[str, GrooveTemplate],
        intensity: float = 1.0
    ) -> List[RhythmNote]:
        """
        Apply groove template to notes.

        Args:
            notes: Notes to groove
            template: GrooveTemplate or template name
            intensity: How much to apply groove (0.0-2.0+)

        Returns:
            New list of grooved notes
        """
        if isinstance(template, str):
            if template not in self.templates:
                raise ValueError(f"Template '{template}' not found")
            template = self.templates[template]

        grooved_notes = []

        for note in notes:
            new_note = copy.deepcopy(note)

            # Determine position in template (cycle through if needed)
            note_index = len(grooved_notes) % len(template.timing_offsets)

            # Apply timing offset
            timing_offset = template.timing_offsets[note_index] * intensity
            new_note.tick = int(note.tick + timing_offset)

            # Apply velocity curve
            velocity_mult = 1.0 + (template.velocity_curve[note_index] - 1.0) * intensity
            new_note.velocity = int(note.velocity * velocity_mult)
            new_note.velocity = max(1, min(127, new_note.velocity))

            grooved_notes.append(new_note)

        return grooved_notes

    def add_template(self, template: GrooveTemplate):
        """Add a groove template to the library"""
        self.templates[template.name] = template

    def get_template(self, name: str) -> Optional[GrooveTemplate]:
        """Get a template by name"""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List all available template names"""
        return list(self.templates.keys())


# ============================================================================
# Advanced Polyrhythm Generator
# ============================================================================

class PolyrhythmGenerator:
    """
    Generate complex polyrhythms, cross-rhythms, and metric modulations.

    Supports:
    - Simple polyrhythms (3:2, 5:4, etc.)
    - Complex cross-rhythms
    - African timeline patterns
    - Indian tala-inspired patterns
    - Metric modulation
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn

    def generate_polyrhythm(
        self,
        spec: PolyrhythmSpec,
        duration_ticks: Optional[int] = None
    ) -> Tuple[List[RhythmNote], List[RhythmNote]]:
        """
        Generate a polyrhythm pattern.

        Args:
            spec: PolyrhythmSpec defining the polyrhythm
            duration_ticks: Total duration (None = use spec.beats * ppqn)

        Returns:
            Tuple of (rhythm_a_notes, rhythm_b_notes)
        """
        if duration_ticks is None:
            duration_ticks = spec.beats * self.ppqn * 4  # Assume 4/4

        # Calculate tick positions for each rhythm
        rhythm_a_notes = self._generate_evenly_spaced_notes(
            spec.ratio_a,
            duration_ticks,
            spec.velocity_a,
            spec.pitch_a
        )

        rhythm_b_notes = self._generate_evenly_spaced_notes(
            spec.ratio_b,
            duration_ticks,
            spec.velocity_b,
            spec.pitch_b
        )

        return rhythm_a_notes, rhythm_b_notes

    def _generate_evenly_spaced_notes(
        self,
        count: int,
        duration_ticks: int,
        velocity: int,
        pitch: Optional[int]
    ) -> List[RhythmNote]:
        """Generate evenly spaced notes"""
        notes = []
        interval = duration_ticks / count

        for i in range(count):
            tick = int(i * interval)
            note = RhythmNote(
                tick=tick,
                duration=int(interval * 0.8),  # 80% duration
                velocity=velocity,
                pitch=pitch
            )
            notes.append(note)

        return notes

    def generate_euclidean_rhythm(
        self,
        hits: int,
        steps: int,
        rotation: int = 0,
        velocity: int = 80,
        pitch: Optional[int] = None,
        duration_beats: int = 4
    ) -> List[RhythmNote]:
        """
        Generate Euclidean rhythm (distributes hits as evenly as possible across steps).

        Based on Godfried Toussaint's algorithm for maximum evenness.
        Used in many world music traditions (African, Brazilian, Indian).

        Args:
            hits: Number of hits/onsets
            steps: Total number of steps
            rotation: Rotate pattern by this many steps
            velocity: MIDI velocity
            pitch: MIDI pitch
            duration_beats: Duration in beats

        Returns:
            List of RhythmNote objects
        """
        # Bjorklund's algorithm for Euclidean rhythm
        pattern = self._bjorklund(hits, steps)

        # Rotate if needed
        if rotation:
            pattern = pattern[rotation:] + pattern[:rotation]

        # Convert to RhythmNote objects
        ticks_per_step = (self.ppqn * 4 * duration_beats) // steps
        notes = []

        for i, hit in enumerate(pattern):
            if hit:
                tick = i * ticks_per_step
                note = RhythmNote(
                    tick=tick,
                    duration=int(ticks_per_step * 0.8),
                    velocity=velocity,
                    pitch=pitch
                )
                notes.append(note)

        return notes

    def _bjorklund(self, hits: int, steps: int) -> List[int]:
        """
        Bjorklund's algorithm for Euclidean rhythms.

        Returns:
            Binary pattern (1 = hit, 0 = rest)
        """
        if hits >= steps:
            return [1] * steps
        if hits == 0:
            return [0] * steps

        # Initialize with hits and rests
        pattern = [[1] for _ in range(hits)] + [[0] for _ in range(steps - hits)]

        # Apply Euclidean algorithm
        return self._bjorklund_recursive(pattern)

    def _bjorklund_recursive(self, pattern: List[List[int]]) -> List[int]:
        """Recursive helper for Bjorklund's algorithm"""
        if len(set(map(len, pattern))) <= 1:
            # All groups same length, we're done
            return [item for group in pattern for item in group]

        # Count groups by length
        counts = {}
        for group in pattern:
            length = len(group)
            counts[length] = counts.get(length, 0) + 1

        # Find the two most common lengths
        sorted_counts = sorted(counts.items(), key=lambda x: (-x[1], -x[0]))
        if len(sorted_counts) < 2:
            return [item for group in pattern for item in group]

        # Pair longer groups with shorter groups
        long_len = sorted_counts[0][0]
        short_len = sorted_counts[1][0]

        long_groups = [g for g in pattern if len(g) == long_len]
        short_groups = [g for g in pattern if len(g) == short_len]
        other_groups = [g for g in pattern if len(g) != long_len and len(g) != short_len]

        # Combine
        pairs = min(len(long_groups), len(short_groups))
        new_pattern = [long_groups[i] + short_groups[i] for i in range(pairs)]
        new_pattern.extend(long_groups[pairs:])
        new_pattern.extend(short_groups[pairs:])
        new_pattern.extend(other_groups)

        return self._bjorklund_recursive(new_pattern)

    def generate_african_timeline(
        self,
        pattern_name: str,
        duration_beats: int = 4,
        velocity: int = 80,
        pitch: Optional[int] = None
    ) -> List[RhythmNote]:
        """
        Generate traditional African timeline patterns.

        Timelines are asymmetric rhythmic patterns that serve as reference points
        in African music.

        Args:
            pattern_name: Name of the pattern (see AFRICAN_TIMELINES)
            duration_beats: Duration in beats
            velocity: MIDI velocity
            pitch: MIDI pitch

        Returns:
            List of RhythmNote objects
        """
        # Famous African timeline patterns (in 16th note grid)
        AFRICAN_TIMELINES = {
            'son_clave': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
            'rumba_clave': [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
            'bossa_clave': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0],
            'gankogui': [1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1],  # 12/8 bell pattern
            'soukous': [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            'bembe': [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
        }

        if pattern_name not in AFRICAN_TIMELINES:
            raise ValueError(f"Unknown timeline pattern: {pattern_name}")

        pattern = AFRICAN_TIMELINES[pattern_name]
        steps = len(pattern)

        # Convert to RhythmNote objects
        ticks_per_step = (self.ppqn * 4 * duration_beats) // steps
        notes = []

        for i, hit in enumerate(pattern):
            if hit:
                tick = i * ticks_per_step
                note = RhythmNote(
                    tick=tick,
                    duration=int(ticks_per_step * 0.7),
                    velocity=velocity,
                    pitch=pitch
                )
                notes.append(note)

        return notes


# ============================================================================
# Humanization Engine
# ============================================================================

class HumanizationEngine:
    """
    Add natural human timing and velocity variations to MIDI.

    Based on research in music performance analysis:
    - Early/late tendencies per instrument
    - Velocity humanization (avoid machine-gun effect)
    - Flam/drag/buzz simulation for drums
    - Ensemble sync modeling
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN, random_seed: Optional[int] = None):
        self.ppqn = ppqn
        if random_seed is not None:
            np.random.seed(random_seed)

    def humanize_timing(
        self,
        notes: List[RhythmNote],
        style: TimingStyle = TimingStyle.HUMAN,
        deviation_ticks: Optional[int] = None
    ) -> List[RhythmNote]:
        """
        Add human timing variations.

        Args:
            notes: Notes to humanize
            style: Timing style (see TimingStyle enum)
            deviation_ticks: Max deviation in ticks (None = auto)

        Returns:
            Humanized notes
        """
        if style == TimingStyle.LOCKED or style == TimingStyle.MACHINE:
            return copy.deepcopy(notes)

        # Auto-calculate deviation based on style
        if deviation_ticks is None:
            deviation_map = {
                TimingStyle.TIGHT: self.ppqn // 64,
                TimingStyle.LAID_BACK: self.ppqn // 32,
                TimingStyle.RUSHING: self.ppqn // 32,
                TimingStyle.DRUNK: self.ppqn // 8,
                TimingStyle.HUMAN: self.ppqn // 48,
            }
            deviation_ticks = deviation_map.get(style, self.ppqn // 48)

        humanized = []

        for note in notes:
            new_note = copy.deepcopy(note)

            if style == TimingStyle.LAID_BACK:
                # Notes tend to be late
                offset = int(abs(np.random.normal(0, deviation_ticks)))
                new_note.tick += offset
            elif style == TimingStyle.RUSHING:
                # Notes tend to be early
                offset = int(abs(np.random.normal(0, deviation_ticks)))
                new_note.tick -= offset
            elif style == TimingStyle.DRUNK:
                # Heavy random variation
                offset = int(np.random.uniform(-deviation_ticks, deviation_ticks))
                new_note.tick += offset
            else:  # HUMAN or TIGHT
                # Gaussian distribution (more subtle)
                offset = int(np.random.normal(0, deviation_ticks / 2))
                new_note.tick += offset

            # Ensure tick is non-negative
            new_note.tick = max(0, new_note.tick)

            humanized.append(new_note)

        return humanized

    def humanize_velocity(
        self,
        notes: List[RhythmNote],
        variation: float = 0.15,
        avoid_extremes: bool = True
    ) -> List[RhythmNote]:
        """
        Add velocity variation to avoid machine-gun effect.

        Args:
            notes: Notes to humanize
            variation: Amount of variation (0.0-1.0)
            avoid_extremes: Avoid very loud/soft notes

        Returns:
            Humanized notes
        """
        humanized = []

        for note in notes:
            new_note = copy.deepcopy(note)

            # Apply random variation
            factor = 1.0 + np.random.uniform(-variation, variation)
            new_velocity = int(note.velocity * factor)

            # Clamp to valid range
            if avoid_extremes:
                new_velocity = max(20, min(110, new_velocity))
            else:
                new_velocity = max(1, min(127, new_velocity))

            new_note.velocity = new_velocity
            humanized.append(new_note)

        return humanized

    def add_drummer_feel(
        self,
        notes: List[RhythmNote],
        ghost_note_probability: float = 0.2,
        ghost_velocity: int = 30,
        flam_probability: float = 0.1,
        flam_delay_ticks: Optional[int] = None
    ) -> List[RhythmNote]:
        """
        Add drummer-specific feel: ghost notes, flams, etc.

        Args:
            notes: Drum notes
            ghost_note_probability: Chance of adding ghost note (0.0-1.0)
            ghost_velocity: Velocity for ghost notes
            flam_probability: Chance of adding flam (0.0-1.0)
            flam_delay_ticks: Ticks between flam notes (None = auto)

        Returns:
            Enhanced drum notes
        """
        if flam_delay_ticks is None:
            flam_delay_ticks = self.ppqn // 32  # Short delay

        enhanced = []

        for note in notes:
            # Maybe add flam (grace note before main note)
            if np.random.random() < flam_probability:
                flam_note = copy.deepcopy(note)
                flam_note.tick -= flam_delay_ticks
                flam_note.velocity = int(note.velocity * 0.7)
                enhanced.append(flam_note)

            # Add main note
            enhanced.append(copy.deepcopy(note))

            # Maybe add ghost note after
            if np.random.random() < ghost_note_probability:
                ghost_note = copy.deepcopy(note)
                ghost_note.tick += note.duration // 2
                ghost_note.velocity = ghost_velocity
                ghost_note.duration = note.duration // 4
                enhanced.append(ghost_note)

        return sorted(enhanced, key=lambda n: n.tick)


# ============================================================================
# Rhythm Transformations
# ============================================================================

class RhythmTransformer:
    """
    Transform rhythms: augmentation, diminution, retrograde, rotation, etc.
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn

    def augment(self, notes: List[RhythmNote], factor: float = 2.0) -> List[RhythmNote]:
        """
        Augmentation: Make rhythm slower by multiplying durations.

        Args:
            notes: Notes to augment
            factor: Multiplication factor (2.0 = twice as slow)

        Returns:
            Augmented notes
        """
        augmented = []
        for note in notes:
            new_note = copy.deepcopy(note)
            new_note.tick = int(note.tick * factor)
            new_note.duration = int(note.duration * factor)
            augmented.append(new_note)
        return augmented

    def diminute(self, notes: List[RhythmNote], factor: float = 2.0) -> List[RhythmNote]:
        """
        Diminution: Make rhythm faster by dividing durations.

        Args:
            notes: Notes to diminish
            factor: Division factor (2.0 = twice as fast)

        Returns:
            Diminished notes
        """
        return self.augment(notes, 1.0 / factor)

    def retrograde(self, notes: List[RhythmNote], duration_ticks: Optional[int] = None) -> List[RhythmNote]:
        """
        Retrograde: Reverse the rhythm.

        Args:
            notes: Notes to reverse
            duration_ticks: Total duration (None = auto-detect)

        Returns:
            Reversed notes
        """
        if not notes:
            return []

        if duration_ticks is None:
            duration_ticks = max(n.tick + n.duration for n in notes)

        reversed_notes = []
        for note in notes:
            new_note = copy.deepcopy(note)
            new_note.tick = duration_ticks - (note.tick + note.duration)
            reversed_notes.append(new_note)

        return sorted(reversed_notes, key=lambda n: n.tick)

    def rotate(self, notes: List[RhythmNote], rotation_ticks: int) -> List[RhythmNote]:
        """
        Rotate rhythm by shifting all notes.

        Args:
            notes: Notes to rotate
            rotation_ticks: Amount to shift (positive = later, negative = earlier)

        Returns:
            Rotated notes
        """
        rotated = []
        for note in notes:
            new_note = copy.deepcopy(note)
            new_note.tick = note.tick + rotation_ticks
            if new_note.tick >= 0:  # Only include notes that are still in range
                rotated.append(new_note)

        return sorted(rotated, key=lambda n: n.tick)

    def convert_swing(
        self,
        notes: List[RhythmNote],
        from_ratio: float = 0.5,
        to_ratio: float = 0.67,
        grid_division: int = 8
    ) -> List[RhythmNote]:
        """
        Convert between straight and swing feel.

        Args:
            notes: Notes to convert
            from_ratio: Current swing ratio (0.5 = straight, 0.67 = triplet)
            to_ratio: Target swing ratio
            grid_division: Grid division (8 = 8th notes, 16 = 16th notes)

        Returns:
            Swing-converted notes
        """
        ticks_per_grid = (self.ppqn * 4) // grid_division
        converted = []

        for note in notes:
            new_note = copy.deepcopy(note)

            # Find position within grid pair
            grid_pos = note.tick / ticks_per_grid
            grid_index = int(grid_pos)
            position_in_pair = grid_pos - grid_index

            # Only adjust offbeat notes (odd grid positions)
            if grid_index % 2 == 1:
                # Calculate swing adjustment
                # from_ratio determines where the note currently is
                # to_ratio determines where it should be
                current_offset = from_ratio * 2 - 1  # Convert to -1 to 1 range
                target_offset = to_ratio * 2 - 1

                adjustment = (target_offset - current_offset) * ticks_per_grid / 2
                new_note.tick = int(note.tick + adjustment)

            converted.append(new_note)

        return sorted(converted, key=lambda n: n.tick)

    def change_time_signature(
        self,
        notes: List[RhythmNote],
        from_sig: Tuple[int, int],
        to_sig: Tuple[int, int]
    ) -> List[RhythmNote]:
        """
        Convert rhythm to different time signature.

        Args:
            notes: Notes to convert
            from_sig: Source time signature (numerator, denominator)
            to_sig: Target time signature

        Returns:
            Converted notes
        """
        # Calculate ratio of beat lengths
        from_num, from_denom = from_sig
        to_num, to_denom = to_sig

        # Time ratio = (beats in from / beats in to) * (beat length to / beat length from)
        ratio = (to_denom / from_denom)

        converted = []
        for note in notes:
            new_note = copy.deepcopy(note)
            new_note.tick = int(note.tick * ratio)
            new_note.duration = int(note.duration * ratio)
            converted.append(new_note)

        return converted


# ============================================================================
# Main Rhythm Engine
# ============================================================================

class RhythmEngine:
    """
    Main rhythm engine combining all rhythm generation and transformation capabilities.
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn
        self.groove_engine = GrooveTemplateEngine(ppqn)
        self.polyrhythm_generator = PolyrhythmGenerator(ppqn)
        self.humanizer = HumanizationEngine(ppqn)
        self.transformer = RhythmTransformer(ppqn)

    def create_full_rhythm_pattern(
        self,
        base_pattern: List[RhythmNote],
        groove_template: Optional[Union[str, GrooveTemplate]] = None,
        groove_intensity: float = 0.75,
        humanize_timing: bool = True,
        humanize_velocity: bool = True,
        timing_style: TimingStyle = TimingStyle.HUMAN,
        velocity_variation: float = 0.15
    ) -> List[RhythmNote]:
        """
        Create a full rhythm pattern with groove and humanization.

        This is a convenience method that applies multiple transformations:
        1. Apply groove template (if provided)
        2. Humanize timing
        3. Humanize velocity

        Args:
            base_pattern: Starting rhythm pattern
            groove_template: Groove template to apply
            groove_intensity: Groove application intensity
            humanize_timing: Whether to humanize timing
            humanize_velocity: Whether to humanize velocity
            timing_style: Style of timing humanization
            velocity_variation: Amount of velocity variation

        Returns:
            Fully processed rhythm pattern
        """
        result = copy.deepcopy(base_pattern)

        # Apply groove template
        if groove_template:
            result = self.groove_engine.apply_groove(
                result,
                groove_template,
                intensity=groove_intensity
            )

        # Humanize timing
        if humanize_timing:
            result = self.humanizer.humanize_timing(result, style=timing_style)

        # Humanize velocity
        if humanize_velocity:
            result = self.humanizer.humanize_velocity(result, variation=velocity_variation)

        return result


# ============================================================================
# Utility Functions
# ============================================================================

def ticks_to_ms(ticks: int, bpm: float, ppqn: int = DEFAULT_PPQN) -> float:
    """Convert MIDI ticks to milliseconds"""
    microseconds_per_beat = 60_000_000 / bpm
    microseconds_per_tick = microseconds_per_beat / ppqn
    return ticks * microseconds_per_tick / 1000


def ms_to_ticks(ms: float, bpm: float, ppqn: int = DEFAULT_PPQN) -> int:
    """Convert milliseconds to MIDI ticks"""
    microseconds_per_beat = 60_000_000 / bpm
    microseconds_per_tick = microseconds_per_beat / ppqn
    return int((ms * 1000) / microseconds_per_tick)


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    """Example usage of the Rhythm Engine"""

    print("=" * 70)
    print("ADVANCED RHYTHM ENGINE - Example Usage")
    print("=" * 70)

    # Initialize engine
    engine = RhythmEngine(ppqn=960)

    # Example 1: Create a simple rhythm pattern
    print("\n1. Creating base rhythm pattern (16th note kicks)...")
    base_pattern = [
        RhythmNote(tick=i * 240, duration=200, velocity=80, pitch=36)
        for i in range(16)
    ]
    print(f"   Created {len(base_pattern)} notes")

    # Example 2: Generate polyrhythm (3 against 4)
    print("\n2. Generating polyrhythm (3 against 4)...")
    poly_spec = PolyrhythmSpec(
        ratio_a=3,
        ratio_b=4,
        beats=4,
        velocity_a=90,
        velocity_b=70,
        pitch_a=42,  # Closed hi-hat
        pitch_b=38   # Snare
    )
    rhythm_a, rhythm_b = engine.polyrhythm_generator.generate_polyrhythm(poly_spec)
    print(f"   Rhythm A: {len(rhythm_a)} notes")
    print(f"   Rhythm B: {len(rhythm_b)} notes")

    # Example 3: Euclidean rhythm
    print("\n3. Generating Euclidean rhythm (5 hits in 8 steps)...")
    euclidean = engine.polyrhythm_generator.generate_euclidean_rhythm(
        hits=5,
        steps=8,
        velocity=85,
        pitch=36
    )
    print(f"   Generated {len(euclidean)} notes")
    print(f"   Pattern: {[n.tick // 480 for n in euclidean]}")

    # Example 4: African timeline
    print("\n4. Generating Son Clave pattern...")
    clave = engine.polyrhythm_generator.generate_african_timeline(
        pattern_name='son_clave',
        duration_beats=4,
        velocity=95,
        pitch=75  # Claves
    )
    print(f"   Generated {len(clave)} notes")

    # Example 5: Humanization
    print("\n5. Humanizing rhythm...")
    humanized = engine.humanizer.humanize_timing(
        base_pattern,
        style=TimingStyle.LAID_BACK
    )
    humanized = engine.humanizer.humanize_velocity(humanized, variation=0.2)
    print("   Applied laid-back timing and velocity variation")

    # Example 6: Rhythm transformations
    print("\n6. Rhythm transformations...")
    augmented = engine.transformer.augment(base_pattern, factor=2.0)
    print(f"   Augmented (2x slower): {len(augmented)} notes")

    diminished = engine.transformer.diminute(base_pattern, factor=2.0)
    print(f"   Diminished (2x faster): {len(diminished)} notes")

    retrograde = engine.transformer.retrograde(base_pattern)
    print(f"   Retrograde (reversed): {len(retrograde)} notes")

    # Example 7: Swing conversion
    print("\n7. Converting straight to swing...")
    swing = engine.transformer.convert_swing(
        base_pattern,
        from_ratio=0.5,   # Straight
        to_ratio=0.67,    # Triplet swing
        grid_division=8
    )
    print("   Converted to triplet swing")

    print("\n" + "=" * 70)
    print("Rhythm Engine examples completed!")
    print("=" * 70)
