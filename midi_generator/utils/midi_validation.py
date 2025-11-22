"""
MIDI Validation and Quality Metrics - Agent 2
=============================================

MIDIValidator: Validate generated MIDI and compute quality metrics.

This module provides:
- MIDI file validation (format, ranges, validity)
- Quality metrics (note density, polyphony, rhythmic regularity)
- Distance metrics for reconstruction loss (pitch, rhythm, harmony)

Author: Agent 2 - Differentiable MIDI & Utilities Support
Date: November 22, 2025
License: MIT
"""

import numpy as np
import pretty_midi
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from collections import Counter
import warnings


# ============================================================================
# MIDI Validator
# ============================================================================

class MIDIValidator:
    """
    Validate MIDI files and compute quality metrics.

    Provides three main functions:
    1. validate() - Check MIDI validity and detect errors
    2. get_quality_metrics() - Compute musical quality metrics
    3. compute_distance() - Measure distance between two MIDI files (for reconstruction loss)

    Usage:
        # Validate
        validation = MIDIValidator.validate(midi)
        if not validation['is_valid']:
            print(f"Errors: {validation['errors']}")

        # Quality metrics
        quality = MIDIValidator.get_quality_metrics(midi)
        print(f"Note density: {quality['note_density']:.2f} notes/sec")

        # Distance (reconstruction loss)
        distance = MIDIValidator.compute_distance(original, reconstructed)
        loss = distance['overall_distance']
    """

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def validate(midi: pretty_midi.PrettyMIDI) -> Dict[str, Any]:
        """
        Validate MIDI file and return diagnostics.

        Checks:
        - Basic format validity
        - At least one instrument and note
        - Pitch/velocity ranges
        - Time/duration validity
        - No overlapping notes on same pitch

        Args:
            midi: MIDI file to validate

        Returns:
            Dict with keys:
            - 'is_valid': bool
            - 'num_notes': int
            - 'num_tracks': int
            - 'duration': float (seconds)
            - 'errors': List[str] (validation errors)
            - 'warnings': List[str] (potential issues)
        """
        errors = []
        warnings_list = []

        # Basic checks
        if not isinstance(midi, pretty_midi.PrettyMIDI):
            errors.append("Not a PrettyMIDI object")
            return {
                'is_valid': False,
                'num_notes': 0,
                'num_tracks': 0,
                'duration': 0.0,
                'errors': errors,
                'warnings': warnings_list
            }

        num_instruments = len(midi.instruments)
        if num_instruments == 0:
            errors.append("No instruments found")

        # Collect all notes
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)

        num_notes = len(all_notes)
        if num_notes == 0:
            errors.append("No notes found")

        # Validate each note
        for i, note in enumerate(all_notes):
            # Pitch range
            if not (0 <= note.pitch <= 127):
                errors.append(f"Note {i}: Invalid pitch {note.pitch} (must be 0-127)")

            # Velocity range
            if not (1 <= note.velocity <= 127):
                errors.append(f"Note {i}: Invalid velocity {note.velocity} (must be 1-127)")

            # Time validity
            if note.start < 0:
                errors.append(f"Note {i}: Negative start time {note.start}")

            if note.end < 0:
                errors.append(f"Note {i}: Negative end time {note.end}")

            if note.end <= note.start:
                errors.append(f"Note {i}: End time {note.end} <= start time {note.start}")

            # Duration checks
            duration = note.end - note.start
            if duration > 60.0:
                warnings_list.append(f"Note {i}: Very long duration {duration:.2f}s (possible stuck note)")

            if duration < 0.01:
                warnings_list.append(f"Note {i}: Very short duration {duration:.4f}s")

        # Check for overlapping notes
        for instrument in midi.instruments:
            overlaps = MIDIValidator._find_overlapping_notes(instrument.notes)
            if overlaps > 0:
                warnings_list.append(f"Instrument {instrument.program}: {overlaps} overlapping notes")

        # Duration check
        try:
            duration = midi.get_end_time()
            if duration <= 0:
                errors.append("Duration is zero or negative")
            elif duration > 600:  # 10 minutes
                warnings_list.append(f"Very long MIDI file: {duration:.2f}s")
        except:
            duration = 0.0
            errors.append("Failed to get duration")

        is_valid = len(errors) == 0

        return {
            'is_valid': is_valid,
            'num_notes': num_notes,
            'num_tracks': num_instruments,
            'duration': duration,
            'errors': errors,
            'warnings': warnings_list
        }

    @staticmethod
    def _find_overlapping_notes(notes: List[pretty_midi.Note]) -> int:
        """
        Count overlapping notes on the same pitch.

        Args:
            notes: List of notes

        Returns:
            Number of overlapping note pairs
        """
        # Group by pitch
        pitch_groups = {}
        for note in notes:
            if note.pitch not in pitch_groups:
                pitch_groups[note.pitch] = []
            pitch_groups[note.pitch].append(note)

        overlap_count = 0

        for pitch, pitch_notes in pitch_groups.items():
            # Sort by start time
            pitch_notes.sort(key=lambda n: n.start)

            # Check for overlaps
            for i in range(len(pitch_notes) - 1):
                if pitch_notes[i].end > pitch_notes[i + 1].start:
                    overlap_count += 1

        return overlap_count

    # ========================================================================
    # Quality Metrics
    # ========================================================================

    @staticmethod
    def get_quality_metrics(midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Compute quality metrics for generated MIDI.

        Metrics:
        - note_density: Notes per second
        - pitch_range: Max pitch - min pitch
        - avg_velocity: Average velocity
        - polyphony: Average simultaneous notes
        - rhythmic_regularity: Measure of rhythmic structure [0, 1]
        - harmonic_consistency: Measure of harmonic coherence [0, 1]

        Args:
            midi: MIDI file

        Returns:
            Dict with quality metrics
        """
        # Collect all notes
        all_notes = []
        for instrument in midi.instruments:
            all_notes.extend(instrument.notes)

        if len(all_notes) == 0:
            return {
                'num_notes': 0,
                'note_density': 0.0,
                'pitch_range': 0,
                'avg_pitch': 0.0,
                'avg_velocity': 0.0,
                'avg_duration': 0.0,
                'polyphony_avg': 0.0,
                'polyphony_max': 0,
                'rhythmic_regularity': 0.0,
                'harmonic_consistency': 0.0
            }

        duration = midi.get_end_time()

        # Note-level metrics
        pitches = [n.pitch for n in all_notes]
        velocities = [n.velocity for n in all_notes]
        durations = [n.end - n.start for n in all_notes]

        # Polyphony
        polyphony_avg, polyphony_max = MIDIValidator._compute_polyphony(all_notes, duration)

        # Rhythmic regularity
        rhythmic_regularity = MIDIValidator._compute_rhythmic_regularity(all_notes)

        # Harmonic consistency
        harmonic_consistency = MIDIValidator._compute_harmonic_consistency(all_notes)

        return {
            'num_notes': len(all_notes),
            'note_density': len(all_notes) / duration if duration > 0 else 0.0,
            'pitch_range': max(pitches) - min(pitches),
            'avg_pitch': float(np.mean(pitches)),
            'avg_velocity': float(np.mean(velocities)),
            'avg_duration': float(np.mean(durations)),
            'polyphony_avg': polyphony_avg,
            'polyphony_max': polyphony_max,
            'rhythmic_regularity': rhythmic_regularity,
            'harmonic_consistency': harmonic_consistency
        }

    @staticmethod
    def _compute_polyphony(
        notes: List[pretty_midi.Note],
        duration: float,
        time_resolution: float = 0.01
    ) -> Tuple[float, int]:
        """
        Compute average and max polyphony.

        Args:
            notes: List of all notes
            duration: Total duration
            time_resolution: Sampling resolution

        Returns:
            (avg_polyphony, max_polyphony)
        """
        if duration <= 0:
            return 0.0, 0

        time_steps = int(duration / time_resolution)
        active_notes = np.zeros(time_steps, dtype=int)

        for note in notes:
            start_idx = int(note.start / time_resolution)
            end_idx = int(note.end / time_resolution)
            start_idx = max(0, min(start_idx, time_steps - 1))
            end_idx = max(0, min(end_idx, time_steps))

            active_notes[start_idx:end_idx] += 1

        avg_polyphony = float(active_notes.mean())
        max_polyphony = int(active_notes.max())

        return avg_polyphony, max_polyphony

    @staticmethod
    def _compute_rhythmic_regularity(notes: List[pretty_midi.Note]) -> float:
        """
        Measure rhythmic regularity [0, 1].

        Computed as inverse of inter-onset interval variance.
        Higher value = more regular rhythm.

        Args:
            notes: List of notes

        Returns:
            Rhythmic regularity score [0, 1]
        """
        if len(notes) < 2:
            return 0.0

        # Extract onset times
        onsets = sorted([n.start for n in notes])

        # Inter-onset intervals
        iois = np.diff(onsets)

        if len(iois) == 0:
            return 0.0

        # Regularity = 1 / (1 + std)
        # Low std = high regularity
        regularity = 1.0 / (1.0 + float(np.std(iois)))

        return regularity

    @staticmethod
    def _compute_harmonic_consistency(notes: List[pretty_midi.Note]) -> float:
        """
        Measure harmonic consistency [0, 1].

        Computed using pitch class distribution entropy.
        Lower entropy = more consistent harmony.

        Args:
            notes: List of notes

        Returns:
            Harmonic consistency score [0, 1]
        """
        if len(notes) == 0:
            return 0.0

        # Pitch class distribution
        pitch_classes = [n.pitch % 12 for n in notes]
        pitch_class_counts = Counter(pitch_classes)

        # Normalize to probabilities
        total = len(pitch_classes)
        probs = np.array([pitch_class_counts.get(i, 0) / total for i in range(12)])

        # Entropy
        # High entropy = uniform distribution (inconsistent)
        # Low entropy = concentrated distribution (consistent)
        probs = probs[probs > 0]  # Remove zeros
        entropy = -np.sum(probs * np.log2(probs))

        # Max entropy = log2(12) ≈ 3.58
        max_entropy = np.log2(12)

        # Consistency = 1 - (entropy / max_entropy)
        consistency = 1.0 - (entropy / max_entropy)

        return float(consistency)

    # ========================================================================
    # Distance Metrics (for Reconstruction Loss)
    # ========================================================================

    @staticmethod
    def compute_distance(
        midi1: pretty_midi.PrettyMIDI,
        midi2: pretty_midi.PrettyMIDI,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute distance between two MIDI files.

        Used by Agent 5 for reconstruction loss.

        Args:
            midi1: First MIDI (original)
            midi2: Second MIDI (reconstructed)
            metrics: Which metrics to compute (default: all)

        Returns:
            Dict with distances:
            - 'pitch_distance': Pitch content similarity [0, 1]
            - 'rhythm_distance': Rhythmic similarity [0, 1]
            - 'note_f1': Note-level F1 score [0, 1]
            - 'overall_distance': Weighted combination [0, 1]
        """
        metrics = metrics or ['pitch', 'rhythm', 'note_f1']

        distances = {}

        # Pitch distance
        if 'pitch' in metrics:
            distances['pitch_distance'] = MIDIValidator._pitch_distance(midi1, midi2)

        # Rhythm distance
        if 'rhythm' in metrics:
            distances['rhythm_distance'] = MIDIValidator._rhythm_distance(midi1, midi2)

        # Note-level F1
        if 'note_f1' in metrics:
            f1_scores = MIDIValidator._note_f1_score(midi1, midi2)
            distances['note_f1'] = f1_scores['f1']
            distances['note_precision'] = f1_scores['precision']
            distances['note_recall'] = f1_scores['recall']

        # Overall distance (weighted combination)
        overall = 0.0
        weights = {'pitch_distance': 0.4, 'rhythm_distance': 0.4, 'note_f1': 0.2}

        for key, weight in weights.items():
            if key in distances:
                if key == 'note_f1':
                    # F1 is a similarity measure, convert to distance
                    overall += weight * (1.0 - distances[key])
                else:
                    overall += weight * distances[key]

        distances['overall_distance'] = overall

        return distances

    @staticmethod
    def _pitch_distance(midi1: pretty_midi.PrettyMIDI, midi2: pretty_midi.PrettyMIDI) -> float:
        """
        Compare pitch content using pitch class histogram.

        Returns normalized distance [0, 1].
        """
        hist1 = MIDIValidator._pitch_class_histogram(midi1)
        hist2 = MIDIValidator._pitch_class_histogram(midi2)

        # L1 distance (Manhattan)
        distance = np.sum(np.abs(hist1 - hist2)) / 2.0  # Divide by 2 to normalize to [0, 1]

        return float(distance)

    @staticmethod
    def _pitch_class_histogram(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Compute normalized pitch class histogram"""
        notes = [n for inst in midi.instruments for n in inst.notes]
        if len(notes) == 0:
            return np.zeros(12)

        pitch_classes = [n.pitch % 12 for n in notes]
        hist = np.array([pitch_classes.count(i) for i in range(12)], dtype=float)
        hist /= hist.sum()  # Normalize

        return hist

    @staticmethod
    def _rhythm_distance(midi1: pretty_midi.PrettyMIDI, midi2: pretty_midi.PrettyMIDI) -> float:
        """
        Compare rhythmic patterns using inter-onset interval distribution.

        Returns normalized distance [0, 1].
        """
        ioi1 = MIDIValidator._get_ioi_distribution(midi1)
        ioi2 = MIDIValidator._get_ioi_distribution(midi2)

        if len(ioi1) == 0 or len(ioi2) == 0:
            return 1.0  # Maximum distance

        # Compare distributions using histogram
        bins = np.linspace(0, 2.0, 20)  # 0-2 seconds, 20 bins
        hist1, _ = np.histogram(ioi1, bins=bins, density=True)
        hist2, _ = np.histogram(ioi2, bins=bins, density=True)

        # Normalize
        hist1 = hist1 / (hist1.sum() + 1e-10)
        hist2 = hist2 / (hist2.sum() + 1e-10)

        # L1 distance
        distance = np.sum(np.abs(hist1 - hist2)) / 2.0

        return float(distance)

    @staticmethod
    def _get_ioi_distribution(midi: pretty_midi.PrettyMIDI) -> np.ndarray:
        """Get inter-onset intervals"""
        notes = [n for inst in midi.instruments for n in inst.notes]
        if len(notes) < 2:
            return np.array([])

        onsets = sorted([n.start for n in notes])
        iois = np.diff(onsets)

        return iois

    @staticmethod
    def _note_f1_score(
        midi1: pretty_midi.PrettyMIDI,
        midi2: pretty_midi.PrettyMIDI,
        onset_tolerance: float = 0.05,
        offset_tolerance: float = 0.05
    ) -> Dict[str, float]:
        """
        Compute note-level F1 score.

        Two notes match if:
        - Same pitch
        - Onset within tolerance
        - Offset within tolerance

        Args:
            midi1: Original MIDI
            midi2: Reconstructed MIDI
            onset_tolerance: Onset matching tolerance (seconds)
            offset_tolerance: Offset matching tolerance (seconds)

        Returns:
            Dict with precision, recall, f1
        """
        notes1 = [n for inst in midi1.instruments for n in inst.notes]
        notes2 = [n for inst in midi2.instruments for n in inst.notes]

        if len(notes1) == 0 or len(notes2) == 0:
            return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

        # Find matches
        matches = MIDIValidator._find_note_matches(notes1, notes2, onset_tolerance, offset_tolerance)

        precision = len(matches) / len(notes2) if len(notes2) > 0 else 0.0
        recall = len(matches) / len(notes1) if len(notes1) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    @staticmethod
    def _find_note_matches(
        notes1: List[pretty_midi.Note],
        notes2: List[pretty_midi.Note],
        onset_tolerance: float,
        offset_tolerance: float
    ) -> List[Tuple[int, int]]:
        """Find matching note pairs"""
        matches = []
        used_notes2 = set()

        for i, note1 in enumerate(notes1):
            for j, note2 in enumerate(notes2):
                if j in used_notes2:
                    continue

                # Check match
                if (note1.pitch == note2.pitch and
                    abs(note1.start - note2.start) <= onset_tolerance and
                    abs(note1.end - note2.end) <= offset_tolerance):

                    matches.append((i, j))
                    used_notes2.add(j)
                    break

        return matches


# ============================================================================
# Demo
# ============================================================================

def demo_validation():
    """Demonstrate MIDI validation and quality metrics"""
    print("MIDI Validation Demo")
    print("=" * 60)

    # Create a simple test MIDI
    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    # Add some notes
    for i in range(16):
        note = pretty_midi.Note(
            velocity=80,
            pitch=60 + (i % 12),
            start=i * 0.5,
            end=i * 0.5 + 0.4
        )
        piano.notes.append(note)

    midi.instruments.append(piano)

    # Validate
    validation = MIDIValidator.validate(midi)
    print(f"\nValidation Results:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Notes: {validation['num_notes']}")
    print(f"  Duration: {validation['duration']:.2f}s")
    print(f"  Errors: {validation['errors']}")
    print(f"  Warnings: {validation['warnings']}")

    # Quality metrics
    quality = MIDIValidator.get_quality_metrics(midi)
    print(f"\nQuality Metrics:")
    for key, value in quality.items():
        print(f"  {key}: {value:.3f}")

    # Distance (compare to itself)
    distance = MIDIValidator.compute_distance(midi, midi)
    print(f"\nDistance (self-comparison):")
    for key, value in distance.items():
        print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    demo_validation()
