"""
Agent 22: Dynamics Specialist
==============================

Advanced dynamics control system for the Musical Program Synthesis framework.

This specialist provides:
1. ADSR Envelope Control - Attack, Decay, Sustain, Release shaping
2. Dynamic Curves - Crescendo, diminuendo, custom curves
3. Humanization - Natural velocity variation and micro-dynamics
4. Voice Balancing - Multi-voice dynamic balance
5. Articulation-Dynamics Coupling - Context-aware dynamics

ARCHITECTURAL ROLE:
    parameters → DynamicsSpecialist → apply_dynamics() → MIDI with expressive dynamics
    MIDI → DynamicsSpecialist.analyze() → dynamics features → XGBoost models

CAPABILITIES:
    - Apply ADSR envelopes to individual notes or phrases
    - Generate crescendo/diminuendo curves with various shapes
    - Humanize velocities with controllable randomness
    - Balance dynamics across multiple voices/layers
    - Couple dynamics with articulation for natural expression
    - Extract dynamics features for inverse learning

Author: Agent 22 - Dynamics Specialist
Part of: 35-Agent Self-Expanding Inverse Music Generation System
License: MIT
"""

import numpy as np
import mido
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# ENUMERATIONS
# ============================================================================

class DynamicCurveType(Enum):
    """Types of dynamic curves"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    PARABOLIC = "parabolic"
    CUSTOM = "custom"


class DynamicDirection(Enum):
    """Direction of dynamic change"""
    CRESCENDO = "crescendo"
    DIMINUENDO = "diminuendo"
    STABLE = "stable"


class ArticulationType(Enum):
    """Articulation types that affect dynamics"""
    LEGATO = "legato"
    STACCATO = "staccato"
    MARCATO = "marcato"
    TENUTO = "tenuto"
    ACCENT = "accent"
    SFORZANDO = "sforzando"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class ADSREnvelope:
    """ADSR envelope parameters"""
    attack_time: float = 0.05  # seconds
    decay_time: float = 0.1    # seconds
    sustain_level: float = 0.7 # 0.0-1.0
    release_time: float = 0.2  # seconds

    def validate(self) -> bool:
        """Validate envelope parameters"""
        return (
            0.0 <= self.attack_time <= 2.0 and
            0.0 <= self.decay_time <= 2.0 and
            0.0 <= self.sustain_level <= 1.0 and
            0.0 <= self.release_time <= 5.0
        )


@dataclass
class DynamicCurve:
    """Dynamic curve specification"""
    curve_type: DynamicCurveType
    direction: DynamicDirection
    start_level: float  # 0.0-1.0
    end_level: float    # 0.0-1.0
    duration: float     # seconds
    shape_factor: float = 1.0  # For exponential/logarithmic curves

    def validate(self) -> bool:
        """Validate curve parameters"""
        return (
            0.0 <= self.start_level <= 1.0 and
            0.0 <= self.end_level <= 1.0 and
            self.duration > 0.0
        )


@dataclass
class DynamicsProfile:
    """Complete dynamics profile for a musical passage"""
    overall_level: float = 0.7       # Master volume (0.0-1.0)
    dynamic_range: float = 0.7       # Range from pp to ff (0.1-1.0)
    accent_intensity: float = 0.6    # Accent strength (0.0-1.0)
    humanization_amount: float = 0.3 # Randomness (0.0-1.0)
    micro_timing_variance: float = 0.02  # Timing humanization
    layer_balance: List[float] = field(default_factory=lambda: [1.0, 0.7, 0.8, 0.9])
    adsr_envelope: Optional[ADSREnvelope] = None
    dynamic_curve: Optional[DynamicCurve] = None


@dataclass
class Note:
    """Note representation with dynamics"""
    pitch: int
    velocity: int
    start_time: float
    end_time: float
    duration: float
    channel: int = 0
    articulation: Optional[ArticulationType] = None


@dataclass
class DynamicsAnalysisResult:
    """Results from dynamics analysis"""
    # Global metrics
    mean_velocity: float
    std_velocity: float
    velocity_range: Tuple[int, int]
    dynamic_contrast: float  # Range normalized

    # Temporal dynamics
    velocity_trajectory: np.ndarray
    crescendo_count: int
    diminuendo_count: int
    dynamic_changes: List[Tuple[float, str]]  # (time, direction)

    # Articulation
    accent_frequency: float
    ghost_note_frequency: float
    articulation_distribution: Dict[str, int]

    # Humanization
    velocity_consistency: float  # 0=random, 1=mechanical
    micro_timing_variance: float
    natural_variation_score: float

    # Layer balancing
    layer_balance_ratios: Optional[Dict[int, float]] = None

    # ADSR characteristics
    average_attack_profile: Optional[np.ndarray] = None
    average_release_profile: Optional[np.ndarray] = None


# ============================================================================
# DYNAMICS SPECIALIST CLASS
# ============================================================================

class DynamicsSpecialist:
    """
    Advanced dynamics control and analysis for musical expression.

    This class provides comprehensive dynamics manipulation including:
    - ADSR envelope generation and application
    - Dynamic curve generation (crescendo, diminuendo, etc.)
    - Humanization with natural velocity variation
    - Multi-voice dynamic balancing
    - Articulation-dynamics coupling
    - Inverse dynamics analysis for feature extraction
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize Dynamics Specialist.

        Args:
            seed: Random seed for reproducible humanization
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        # Velocity constraints (MIDI range)
        self.min_velocity = 1
        self.max_velocity = 127

        # Humanization parameters
        self.humanization_factor = 0.15  # Default humanization strength
        self.micro_timing_variance = 0.02  # seconds

        # Articulation-dynamics coupling
        self.articulation_velocity_modifiers = {
            ArticulationType.LEGATO: 0.95,
            ArticulationType.STACCATO: 0.85,
            ArticulationType.MARCATO: 1.15,
            ArticulationType.TENUTO: 1.0,
            ArticulationType.ACCENT: 1.25,
            ArticulationType.SFORZANDO: 1.4,
        }

    # ========================================================================
    # ADSR ENVELOPE METHODS
    # ========================================================================

    def generate_adsr_envelope(
        self,
        envelope: ADSREnvelope,
        note_duration: float,
        sample_rate: int = 100
    ) -> np.ndarray:
        """
        Generate ADSR envelope curve.

        Args:
            envelope: ADSR parameters
            note_duration: Total duration of note in seconds
            sample_rate: Samples per second for envelope curve

        Returns:
            Array of envelope values (0.0-1.0) over note duration
        """
        if not envelope.validate():
            raise ValueError("Invalid ADSR envelope parameters")

        total_samples = int(note_duration * sample_rate)
        envelope_curve = np.zeros(total_samples)

        attack_samples = int(envelope.attack_time * sample_rate)
        decay_samples = int(envelope.decay_time * sample_rate)
        release_samples = int(envelope.release_time * sample_rate)

        # Sustain duration is remainder
        sustain_samples = max(0, total_samples - attack_samples - decay_samples - release_samples)

        current_idx = 0

        # Attack phase (0 → 1.0)
        if attack_samples > 0:
            envelope_curve[current_idx:current_idx + attack_samples] = np.linspace(
                0.0, 1.0, attack_samples
            )
            current_idx += attack_samples

        # Decay phase (1.0 → sustain_level)
        if decay_samples > 0 and current_idx < total_samples:
            end_idx = min(current_idx + decay_samples, total_samples)
            actual_decay = end_idx - current_idx
            envelope_curve[current_idx:end_idx] = np.linspace(
                1.0, envelope.sustain_level, actual_decay
            )
            current_idx = end_idx

        # Sustain phase (constant at sustain_level)
        if sustain_samples > 0 and current_idx < total_samples:
            end_idx = min(current_idx + sustain_samples, total_samples)
            envelope_curve[current_idx:end_idx] = envelope.sustain_level
            current_idx = end_idx

        # Release phase (sustain_level → 0)
        if release_samples > 0 and current_idx < total_samples:
            end_idx = min(current_idx + release_samples, total_samples)
            actual_release = end_idx - current_idx
            start_level = envelope_curve[current_idx - 1] if current_idx > 0 else envelope.sustain_level
            envelope_curve[current_idx:end_idx] = np.linspace(
                start_level, 0.0, actual_release
            )

        return envelope_curve

    def apply_adsr_to_note(
        self,
        note: Note,
        envelope: ADSREnvelope,
        base_velocity: Optional[int] = None
    ) -> Note:
        """
        Apply ADSR envelope to a single note's velocity.

        Args:
            note: Input note
            envelope: ADSR parameters
            base_velocity: Base velocity (uses note.velocity if None)

        Returns:
            New note with ADSR-modified velocity
        """
        if base_velocity is None:
            base_velocity = note.velocity

        # Generate envelope curve
        envelope_curve = self.generate_adsr_envelope(envelope, note.duration)

        # Use envelope value at note center for velocity scaling
        center_idx = len(envelope_curve) // 2
        envelope_value = envelope_curve[center_idx]

        # Apply envelope to velocity
        new_velocity = int(base_velocity * envelope_value)
        new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

        return Note(
            pitch=note.pitch,
            velocity=new_velocity,
            start_time=note.start_time,
            end_time=note.end_time,
            duration=note.duration,
            channel=note.channel,
            articulation=note.articulation
        )

    def apply_adsr_to_notes(
        self,
        notes: List[Note],
        envelope: ADSREnvelope
    ) -> List[Note]:
        """
        Apply ADSR envelope to multiple notes.

        Args:
            notes: List of notes
            envelope: ADSR parameters

        Returns:
            List of notes with ADSR-modified velocities
        """
        return [self.apply_adsr_to_note(note, envelope) for note in notes]

    # ========================================================================
    # DYNAMIC CURVE METHODS
    # ========================================================================

    def generate_dynamic_curve(
        self,
        curve: DynamicCurve,
        num_points: int = 100
    ) -> np.ndarray:
        """
        Generate dynamic curve values.

        Args:
            curve: Curve specification
            num_points: Number of points in curve

        Returns:
            Array of dynamic levels (0.0-1.0)
        """
        if not curve.validate():
            raise ValueError("Invalid dynamic curve parameters")

        t = np.linspace(0, 1, num_points)

        if curve.curve_type == DynamicCurveType.LINEAR:
            values = curve.start_level + (curve.end_level - curve.start_level) * t

        elif curve.curve_type == DynamicCurveType.EXPONENTIAL:
            # Exponential growth/decay
            values = curve.start_level + (curve.end_level - curve.start_level) * (
                np.exp(curve.shape_factor * t) - 1
            ) / (np.exp(curve.shape_factor) - 1)

        elif curve.curve_type == DynamicCurveType.LOGARITHMIC:
            # Logarithmic curve
            values = curve.start_level + (curve.end_level - curve.start_level) * (
                np.log(1 + curve.shape_factor * t) / np.log(1 + curve.shape_factor)
            )

        elif curve.curve_type == DynamicCurveType.SIGMOID:
            # S-shaped curve
            sigmoid = 1 / (1 + np.exp(-curve.shape_factor * (t - 0.5)))
            sigmoid_normalized = (sigmoid - sigmoid[0]) / (sigmoid[-1] - sigmoid[0])
            values = curve.start_level + (curve.end_level - curve.start_level) * sigmoid_normalized

        elif curve.curve_type == DynamicCurveType.PARABOLIC:
            # Parabolic curve (accelerating/decelerating)
            values = curve.start_level + (curve.end_level - curve.start_level) * (t ** curve.shape_factor)

        else:
            # Default to linear
            values = curve.start_level + (curve.end_level - curve.start_level) * t

        return np.clip(values, 0.0, 1.0)

    def apply_dynamic_curve(
        self,
        notes: List[Note],
        curve: DynamicCurve,
        time_window: Optional[Tuple[float, float]] = None
    ) -> List[Note]:
        """
        Apply dynamic curve to notes within a time window.

        Args:
            notes: List of notes
            curve: Curve specification
            time_window: (start_time, end_time), uses full range if None

        Returns:
            List of notes with curve-modified velocities
        """
        if not notes:
            return []

        # Determine time window
        if time_window is None:
            start_time = min(n.start_time for n in notes)
            end_time = max(n.end_time for n in notes)
        else:
            start_time, end_time = time_window

        duration = end_time - start_time
        if duration <= 0:
            return notes

        # Generate curve
        curve_values = self.generate_dynamic_curve(curve)

        # Apply curve to each note based on its position in time
        modified_notes = []
        for note in notes:
            # Calculate position in curve (0.0-1.0)
            note_position = (note.start_time - start_time) / duration
            note_position = np.clip(note_position, 0.0, 1.0)

            # Get curve value at this position
            curve_idx = int(note_position * (len(curve_values) - 1))
            curve_value = curve_values[curve_idx]

            # Apply to velocity
            new_velocity = int(note.velocity * curve_value)
            new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

            modified_notes.append(Note(
                pitch=note.pitch,
                velocity=new_velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                duration=note.duration,
                channel=note.channel,
                articulation=note.articulation
            ))

        return modified_notes

    def apply_crescendo(
        self,
        notes: List[Note],
        start_level: float = 0.5,
        end_level: float = 1.0,
        curve_type: DynamicCurveType = DynamicCurveType.LINEAR
    ) -> List[Note]:
        """
        Apply crescendo to notes.

        Args:
            notes: List of notes
            start_level: Starting dynamic level (0.0-1.0)
            end_level: Ending dynamic level (0.0-1.0)
            curve_type: Type of curve

        Returns:
            Notes with crescendo applied
        """
        curve = DynamicCurve(
            curve_type=curve_type,
            direction=DynamicDirection.CRESCENDO,
            start_level=start_level,
            end_level=end_level,
            duration=0.0  # Will be calculated
        )
        return self.apply_dynamic_curve(notes, curve)

    def apply_diminuendo(
        self,
        notes: List[Note],
        start_level: float = 1.0,
        end_level: float = 0.5,
        curve_type: DynamicCurveType = DynamicCurveType.LINEAR
    ) -> List[Note]:
        """
        Apply diminuendo to notes.

        Args:
            notes: List of notes
            start_level: Starting dynamic level (0.0-1.0)
            end_level: Ending dynamic level (0.0-1.0)
            curve_type: Type of curve

        Returns:
            Notes with diminuendo applied
        """
        curve = DynamicCurve(
            curve_type=curve_type,
            direction=DynamicDirection.DIMINUENDO,
            start_level=start_level,
            end_level=end_level,
            duration=0.0
        )
        return self.apply_dynamic_curve(notes, curve)

    # ========================================================================
    # HUMANIZATION METHODS
    # ========================================================================

    def humanize_velocities(
        self,
        notes: List[Note],
        amount: float = 0.15,
        preserve_accents: bool = True
    ) -> List[Note]:
        """
        Add natural velocity variation (humanization).

        Args:
            notes: List of notes
            amount: Humanization strength (0.0-1.0)
            preserve_accents: Keep loud notes loud

        Returns:
            Notes with humanized velocities
        """
        if amount <= 0.0:
            return notes

        humanized = []
        for note in notes:
            # Add random variation
            variation = np.random.normal(0, amount * 20)  # ~20 velocity units max
            new_velocity = int(note.velocity + variation)

            # Preserve accents (don't reduce very loud notes as much)
            if preserve_accents and note.velocity > 100:
                # Less reduction for loud notes
                if new_velocity < note.velocity:
                    reduction = note.velocity - new_velocity
                    new_velocity = int(note.velocity - reduction * 0.5)

            new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

            humanized.append(Note(
                pitch=note.pitch,
                velocity=new_velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                duration=note.duration,
                channel=note.channel,
                articulation=note.articulation
            ))

        return humanized

    def add_micro_dynamics(
        self,
        notes: List[Note],
        variance: float = 0.1,
        phrase_length: int = 4
    ) -> List[Note]:
        """
        Add subtle phrase-level micro-dynamics.

        Creates natural rise and fall within phrases.

        Args:
            notes: List of notes
            variance: Amount of micro-variation (0.0-1.0)
            phrase_length: Notes per phrase for dynamics shaping

        Returns:
            Notes with micro-dynamics
        """
        if variance <= 0.0 or not notes:
            return notes

        modified = []
        for i, note in enumerate(notes):
            # Position within phrase
            phrase_position = (i % phrase_length) / phrase_length

            # Slight crescendo then diminuendo within phrase
            micro_curve = np.sin(phrase_position * np.pi)  # 0 → 1 → 0

            # Add small random component
            random_component = np.random.normal(0, variance * 0.5)

            # Combine
            dynamics_factor = 1.0 + (micro_curve * variance * 0.3) + random_component
            dynamics_factor = np.clip(dynamics_factor, 0.7, 1.3)

            new_velocity = int(note.velocity * dynamics_factor)
            new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

            modified.append(Note(
                pitch=note.pitch,
                velocity=new_velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                duration=note.duration,
                channel=note.channel,
                articulation=note.articulation
            ))

        return modified

    def humanize_timing(
        self,
        notes: List[Note],
        variance: float = 0.02
    ) -> List[Note]:
        """
        Add subtle timing variations (micro-timing).

        Args:
            notes: List of notes
            variance: Timing variance in seconds

        Returns:
            Notes with humanized timing
        """
        if variance <= 0.0:
            return notes

        humanized = []
        for note in notes:
            # Add small timing offset
            time_offset = np.random.normal(0, variance)

            new_start = note.start_time + time_offset
            new_end = note.end_time + time_offset

            # Don't let notes become negative or overlap badly
            new_start = max(0.0, new_start)
            new_end = max(new_start + 0.01, new_end)  # Minimum 10ms duration

            humanized.append(Note(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=new_start,
                end_time=new_end,
                duration=new_end - new_start,
                channel=note.channel,
                articulation=note.articulation
            ))

        return humanized

    # ========================================================================
    # VOICE BALANCING METHODS
    # ========================================================================

    def balance_voices(
        self,
        voice_notes: Dict[int, List[Note]],
        balance_ratios: Optional[List[float]] = None
    ) -> Dict[int, List[Note]]:
        """
        Balance dynamics across multiple voices/layers.

        Args:
            voice_notes: Dictionary mapping voice_id to list of notes
            balance_ratios: Relative volume for each voice (None = equal)

        Returns:
            Dictionary with balanced voice notes
        """
        if not voice_notes:
            return voice_notes

        num_voices = len(voice_notes)

        # Default to equal balance
        if balance_ratios is None:
            balance_ratios = [1.0] * num_voices

        # Ensure we have enough ratios
        while len(balance_ratios) < num_voices:
            balance_ratios.append(1.0)

        # Apply balance to each voice
        balanced = {}
        for i, (voice_id, notes) in enumerate(voice_notes.items()):
            ratio = balance_ratios[i]

            balanced_notes = []
            for note in notes:
                new_velocity = int(note.velocity * ratio)
                new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

                balanced_notes.append(Note(
                    pitch=note.pitch,
                    velocity=new_velocity,
                    start_time=note.start_time,
                    end_time=note.end_time,
                    duration=note.duration,
                    channel=note.channel,
                    articulation=note.articulation
                ))

            balanced[voice_id] = balanced_notes

        return balanced

    def emphasize_melody(
        self,
        notes: List[Note],
        emphasis: float = 0.2,
        melody_range: Tuple[int, int] = (60, 84)
    ) -> List[Note]:
        """
        Emphasize melody notes (typically highest notes in melody range).

        Args:
            notes: List of notes
            emphasis: Amount to boost melody (0.0-1.0)
            melody_range: MIDI pitch range for melody

        Returns:
            Notes with emphasized melody
        """
        if emphasis <= 0.0:
            return notes

        emphasized = []
        for note in notes:
            # Check if note is in melody range
            if melody_range[0] <= note.pitch <= melody_range[1]:
                # Boost velocity
                boost = 1.0 + emphasis
                new_velocity = int(note.velocity * boost)
            else:
                new_velocity = note.velocity

            new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)

            emphasized.append(Note(
                pitch=note.pitch,
                velocity=new_velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                duration=note.duration,
                channel=note.channel,
                articulation=note.articulation
            ))

        return emphasized

    # ========================================================================
    # ARTICULATION-DYNAMICS COUPLING
    # ========================================================================

    def apply_articulation_dynamics(
        self,
        notes: List[Note]
    ) -> List[Note]:
        """
        Apply dynamics based on articulation type.

        Args:
            notes: List of notes with articulation set

        Returns:
            Notes with articulation-appropriate dynamics
        """
        modified = []
        for note in notes:
            if note.articulation and note.articulation in self.articulation_velocity_modifiers:
                modifier = self.articulation_velocity_modifiers[note.articulation]
                new_velocity = int(note.velocity * modifier)
                new_velocity = np.clip(new_velocity, self.min_velocity, self.max_velocity)
            else:
                new_velocity = note.velocity

            modified.append(Note(
                pitch=note.pitch,
                velocity=new_velocity,
                start_time=note.start_time,
                end_time=note.end_time,
                duration=note.duration,
                channel=note.channel,
                articulation=note.articulation
            ))

        return modified

    # ========================================================================
    # COMPREHENSIVE DYNAMICS APPLICATION
    # ========================================================================

    def apply_dynamics_profile(
        self,
        notes: List[Note],
        profile: DynamicsProfile
    ) -> List[Note]:
        """
        Apply complete dynamics profile to notes.

        Applies in order:
        1. Overall level scaling
        2. Dynamic range adjustment
        3. ADSR envelope (if specified)
        4. Dynamic curve (if specified)
        5. Articulation coupling
        6. Humanization
        7. Micro-dynamics

        Args:
            notes: List of notes
            profile: Complete dynamics profile

        Returns:
            Notes with full dynamics profile applied
        """
        if not notes:
            return notes

        result = notes.copy()

        # 1. Overall level
        if profile.overall_level != 1.0:
            result = [
                Note(
                    pitch=n.pitch,
                    velocity=int(np.clip(n.velocity * profile.overall_level, self.min_velocity, self.max_velocity)),
                    start_time=n.start_time,
                    end_time=n.end_time,
                    duration=n.duration,
                    channel=n.channel,
                    articulation=n.articulation
                )
                for n in result
            ]

        # 2. Dynamic range adjustment
        if profile.dynamic_range != 1.0:
            # Compress or expand dynamic range
            velocities = [n.velocity for n in result]
            mean_vel = np.mean(velocities)
            result = [
                Note(
                    pitch=n.pitch,
                    velocity=int(np.clip(
                        mean_vel + (n.velocity - mean_vel) * profile.dynamic_range,
                        self.min_velocity,
                        self.max_velocity
                    )),
                    start_time=n.start_time,
                    end_time=n.end_time,
                    duration=n.duration,
                    channel=n.channel,
                    articulation=n.articulation
                )
                for n in result
            ]

        # 3. ADSR envelope
        if profile.adsr_envelope:
            result = self.apply_adsr_to_notes(result, profile.adsr_envelope)

        # 4. Dynamic curve
        if profile.dynamic_curve:
            result = self.apply_dynamic_curve(result, profile.dynamic_curve)

        # 5. Articulation coupling
        result = self.apply_articulation_dynamics(result)

        # 6. Humanization
        if profile.humanization_amount > 0:
            result = self.humanize_velocities(result, profile.humanization_amount)

        # 7. Micro-dynamics
        result = self.add_micro_dynamics(result, variance=0.1)

        # 8. Micro-timing
        if profile.micro_timing_variance > 0:
            result = self.humanize_timing(result, profile.micro_timing_variance)

        return result

    # ========================================================================
    # MIDI INTEGRATION
    # ========================================================================

    def parse_midi_notes(self, midi: mido.MidiFile) -> List[Note]:
        """
        Parse MIDI file into Note objects.

        Args:
            midi: MIDI file

        Returns:
            List of Note objects
        """
        notes = []
        current_notes = {}  # key: (pitch, channel), value: (velocity, start_time)

        for track in midi.tracks:
            track_time = 0.0
            for msg in track:
                track_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    current_notes[key] = (msg.velocity, track_time)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in current_notes:
                        velocity, start_time = current_notes.pop(key)
                        duration = track_time - start_time
                        if duration > 0:
                            notes.append(Note(
                                pitch=msg.note,
                                velocity=velocity,
                                start_time=start_time,
                                end_time=track_time,
                                duration=duration,
                                channel=msg.channel
                            ))

        notes.sort(key=lambda n: n.start_time)
        return notes

    def apply_dynamics_to_midi(
        self,
        midi_path: Path,
        profile: DynamicsProfile,
        output_path: Path
    ):
        """
        Apply dynamics profile to MIDI file.

        Args:
            midi_path: Input MIDI file
            profile: Dynamics profile to apply
            output_path: Output MIDI file
        """
        # Load MIDI
        midi = mido.MidiFile(str(midi_path))

        # Parse notes
        notes = self.parse_midi_notes(midi)

        # Apply dynamics
        modified_notes = self.apply_dynamics_profile(notes, profile)

        # Create note lookup for updating
        note_updates = {}
        for orig, mod in zip(notes, modified_notes):
            key = (orig.pitch, orig.start_time, orig.channel)
            note_updates[key] = mod.velocity

        # Update MIDI messages
        for track in midi.tracks:
            track_time = 0.0
            for msg in track:
                track_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, track_time, msg.channel)
                    if key in note_updates:
                        msg.velocity = note_updates[key]

        # Save
        midi.save(str(output_path))

    # ========================================================================
    # ANALYSIS METHODS (for inverse learning)
    # ========================================================================

    def analyze_dynamics(
        self,
        notes: List[Note]
    ) -> DynamicsAnalysisResult:
        """
        Analyze dynamics characteristics of notes.

        Args:
            notes: List of notes to analyze

        Returns:
            Complete dynamics analysis
        """
        if not notes:
            return self._get_empty_analysis()

        velocities = np.array([n.velocity for n in notes])
        times = np.array([n.start_time for n in notes])

        # Global metrics
        mean_velocity = float(np.mean(velocities))
        std_velocity = float(np.std(velocities))
        velocity_range = (int(np.min(velocities)), int(np.max(velocities)))
        dynamic_contrast = (velocity_range[1] - velocity_range[0]) / 127.0

        # Velocity trajectory (binned over time)
        num_bins = min(50, len(notes) // 10 + 1)
        if len(notes) > 0:
            bins = np.linspace(times[0], times[-1], num_bins)
            bin_indices = np.digitize(times, bins)
            velocity_trajectory = np.array([
                np.mean(velocities[bin_indices == i]) if np.any(bin_indices == i) else mean_velocity
                for i in range(1, len(bins))
            ])
        else:
            velocity_trajectory = np.array([mean_velocity])

        # Detect crescendos and diminuendos
        crescendo_count = 0
        diminuendo_count = 0
        dynamic_changes = []

        window_size = 5
        for i in range(len(velocity_trajectory) - window_size):
            window = velocity_trajectory[i:i+window_size]
            trend = window[-1] - window[0]

            if trend > 10:  # Crescendo
                crescendo_count += 1
                time_pos = bins[i] if i < len(bins) else times[-1]
                dynamic_changes.append((float(time_pos), "crescendo"))
            elif trend < -10:  # Diminuendo
                diminuendo_count += 1
                time_pos = bins[i] if i < len(bins) else times[-1]
                dynamic_changes.append((float(time_pos), "diminuendo"))

        # Articulation analysis
        accent_frequency = np.sum(velocities > 100) / len(velocities)
        ghost_note_frequency = np.sum(velocities < 40) / len(velocities)

        # Humanization metrics
        # High consistency = mechanical, low consistency = human
        velocity_consistency = 1.0 / (1.0 + std_velocity / 30.0)

        # Calculate inter-onset intervals for timing analysis
        if len(times) > 1:
            iois = np.diff(times)
            micro_timing_variance = float(np.std(iois)) if len(iois) > 0 else 0.0
        else:
            micro_timing_variance = 0.0

        # Natural variation score (combination of velocity and timing variance)
        natural_variation_score = 1.0 - velocity_consistency

        return DynamicsAnalysisResult(
            mean_velocity=mean_velocity,
            std_velocity=std_velocity,
            velocity_range=velocity_range,
            dynamic_contrast=dynamic_contrast,
            velocity_trajectory=velocity_trajectory,
            crescendo_count=crescendo_count,
            diminuendo_count=diminuendo_count,
            dynamic_changes=dynamic_changes,
            accent_frequency=accent_frequency,
            ghost_note_frequency=ghost_note_frequency,
            articulation_distribution={},
            velocity_consistency=velocity_consistency,
            micro_timing_variance=micro_timing_variance,
            natural_variation_score=natural_variation_score
        )

    def _get_empty_analysis(self) -> DynamicsAnalysisResult:
        """Return empty analysis result"""
        return DynamicsAnalysisResult(
            mean_velocity=64.0,
            std_velocity=0.0,
            velocity_range=(64, 64),
            dynamic_contrast=0.0,
            velocity_trajectory=np.array([64.0]),
            crescendo_count=0,
            diminuendo_count=0,
            dynamic_changes=[],
            accent_frequency=0.0,
            ghost_note_frequency=0.0,
            articulation_distribution={},
            velocity_consistency=1.0,
            micro_timing_variance=0.0,
            natural_variation_score=0.0
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_default_profile() -> DynamicsProfile:
    """Create a default dynamics profile"""
    return DynamicsProfile(
        overall_level=0.7,
        dynamic_range=0.7,
        accent_intensity=0.6,
        humanization_amount=0.3,
        micro_timing_variance=0.02,
        layer_balance=[1.0, 0.7, 0.8, 0.9]
    )


def create_expressive_profile() -> DynamicsProfile:
    """Create an expressive dynamics profile with wide range"""
    return DynamicsProfile(
        overall_level=0.75,
        dynamic_range=0.9,
        accent_intensity=0.8,
        humanization_amount=0.4,
        micro_timing_variance=0.03,
        layer_balance=[1.0, 0.6, 0.75, 0.85],
        adsr_envelope=ADSREnvelope(
            attack_time=0.02,
            decay_time=0.1,
            sustain_level=0.75,
            release_time=0.15
        )
    )


def create_mechanical_profile() -> DynamicsProfile:
    """Create mechanical (non-humanized) profile"""
    return DynamicsProfile(
        overall_level=0.7,
        dynamic_range=0.5,
        accent_intensity=0.3,
        humanization_amount=0.0,
        micro_timing_variance=0.0,
        layer_balance=[1.0, 1.0, 1.0, 1.0]
    )


# ============================================================================
# MAIN / DEMO
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("AGENT 22: DYNAMICS SPECIALIST")
    print("=" * 80)
    print()
    print("Advanced Dynamics Control System for Musical Program Synthesis")
    print()
    print("Capabilities:")
    print("  ✓ ADSR Envelope Generation and Application")
    print("  ✓ Dynamic Curves (Crescendo, Diminuendo, Custom)")
    print("  ✓ Humanization (Velocity & Timing Variation)")
    print("  ✓ Voice Balancing")
    print("  ✓ Articulation-Dynamics Coupling")
    print("  ✓ Inverse Dynamics Analysis")
    print()
    print("=" * 80)
    print()

    # Demo: Create specialist
    specialist = DynamicsSpecialist(seed=42)
    print("✅ Dynamics Specialist initialized")

    # Demo: ADSR envelope
    envelope = ADSREnvelope(
        attack_time=0.05,
        decay_time=0.1,
        sustain_level=0.7,
        release_time=0.2
    )
    curve = specialist.generate_adsr_envelope(envelope, note_duration=1.0)
    print(f"✅ Generated ADSR envelope: {len(curve)} samples")

    # Demo: Dynamic curves
    crescendo = DynamicCurve(
        curve_type=DynamicCurveType.EXPONENTIAL,
        direction=DynamicDirection.CRESCENDO,
        start_level=0.3,
        end_level=1.0,
        duration=4.0,
        shape_factor=2.0
    )
    curve_values = specialist.generate_dynamic_curve(crescendo, num_points=100)
    print(f"✅ Generated crescendo curve: {len(curve_values)} points")

    # Demo: Create sample notes
    sample_notes = [
        Note(pitch=60, velocity=80, start_time=0.0, end_time=0.5, duration=0.5),
        Note(pitch=64, velocity=85, start_time=0.5, end_time=1.0, duration=0.5),
        Note(pitch=67, velocity=90, start_time=1.0, end_time=1.5, duration=0.5),
        Note(pitch=72, velocity=95, start_time=1.5, end_time=2.0, duration=0.5),
    ]
    print(f"✅ Created {len(sample_notes)} sample notes")

    # Demo: Apply dynamics
    profile = create_expressive_profile()
    modified_notes = specialist.apply_dynamics_profile(sample_notes, profile)
    print(f"✅ Applied dynamics profile to notes")

    # Demo: Analysis
    analysis = specialist.analyze_dynamics(sample_notes)
    print(f"✅ Analyzed dynamics: mean_vel={analysis.mean_velocity:.1f}, contrast={analysis.dynamic_contrast:.2f}")

    print()
    print("=" * 80)
    print("✅ AGENT 22 DEMONSTRATION COMPLETE")
    print("=" * 80)
