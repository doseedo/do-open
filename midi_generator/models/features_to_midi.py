"""
Features to MIDI - Base Interface
==================================

Abstract base class for converting 1150D feature vectors to MIDI files.

This module defines the interface that all Features→MIDI implementations
must follow, whether rule-based, neural, or hybrid.

Author: Agent 1 - MIDI Decoder Architecture Lead
Date: November 22, 2025
"""

from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import mido

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class FeaturesToMIDI(ABC):
    """
    Abstract base class for Features → MIDI conversion.

    All implementations must provide:
    1. features_to_midi() - Main conversion method
    2. features_to_parameters() - Feature interpretation
    3. validate_output() - MIDI validation

    Usage:
        >>> converter = RuleBasedFeaturesToMIDI()
        >>> features = np.random.randn(1150)
        >>> midi = converter.features_to_midi(features)
        >>> midi.save('output.mid')
    """

    @abstractmethod
    def features_to_midi(
        self,
        features: Union[np.ndarray, 'torch.Tensor'],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> mido.MidiFile:
        """
        Convert feature vector to MIDI file.

        This is the main method that all implementations must provide.
        It should convert a 1150D feature vector into a playable MIDI file.

        Args:
            features: Feature vector [1150] or batch [B, 1150]
            output_path: Optional path to save MIDI file
            **kwargs: Implementation-specific parameters

        Returns:
            mido.MidiFile object

        Raises:
            ValueError: If features are invalid
            RuntimeError: If MIDI generation fails
        """
        pass

    @abstractmethod
    def features_to_parameters(
        self,
        features: Union[np.ndarray, 'torch.Tensor']
    ) -> Dict[str, Any]:
        """
        Extract musical parameters from features.

        This method interprets the feature vector and extracts
        high-level musical parameters such as key, tempo, chord
        progression, etc.

        Args:
            features: Feature vector [1150]

        Returns:
            Dictionary of musical parameters, e.g.:
            {
                'key': 'C',
                'mode': 'major',
                'tempo_bpm': 120,
                'time_signature': '4/4',
                'chord_progression': ['Cmaj7', 'Dm7', 'G7'],
                'melodic_range': 24,  # semitones
                'voice_count': 4,
                'num_bars': 16,
                ...
            }
        """
        pass

    def validate_output(self, midi: mido.MidiFile) -> bool:
        """
        Validate that generated MIDI is valid and playable.

        Checks:
        - Has at least one track
        - Has at least one note event
        - No negative times
        - Valid MIDI messages

        Args:
            midi: MIDI file to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check has tracks
            if len(midi.tracks) == 0:
                return False

            # Check has notes
            has_notes = False
            for track in midi.tracks:
                for msg in track:
                    # Check for negative times
                    if hasattr(msg, 'time') and msg.time < 0:
                        return False

                    # Check for note events
                    if msg.type in ('note_on', 'note_off'):
                        has_notes = True

                        # Validate MIDI note range
                        if hasattr(msg, 'note'):
                            if not (0 <= msg.note <= 127):
                                return False

                        # Validate velocity range
                        if hasattr(msg, 'velocity'):
                            if not (0 <= msg.velocity <= 127):
                                return False

            return has_notes

        except Exception as e:
            print(f"Validation error: {e}")
            return False

    def get_feature_breakdown(self) -> Dict[str, tuple]:
        """
        Get the breakdown of feature dimensions.

        Returns:
            Dictionary mapping feature category to (start, end) indices:
            {
                'harmony': (0, 250),
                'rhythm': (250, 500),
                'melody': (500, 700),
                'dynamics': (700, 850),
                'texture': (850, 950),
                'structure': (950, 1000),
                'orchestration': (1000, 1150)
            }
        """
        return {
            'harmony': (0, 250),
            'rhythm': (250, 500),
            'melody': (500, 700),
            'dynamics': (700, 850),
            'texture': (850, 950),
            'structure': (950, 1000),
            'orchestration': (1000, 1150)
        }

    def extract_feature_slice(
        self,
        features: Union[np.ndarray, 'torch.Tensor'],
        category: str
    ) -> Union[np.ndarray, 'torch.Tensor']:
        """
        Extract a specific category of features.

        Args:
            features: Full feature vector [1150]
            category: Category name ('harmony', 'rhythm', etc.)

        Returns:
            Slice of features for that category

        Example:
            >>> harmony_features = converter.extract_feature_slice(
            ...     features, 'harmony'
            ... )
            >>> print(harmony_features.shape)  # (250,)
        """
        breakdown = self.get_feature_breakdown()

        if category not in breakdown:
            raise ValueError(f"Unknown category: {category}")

        start, end = breakdown[category]
        return features[start:end]

    def compute_reconstruction_quality(
        self,
        original_midi: mido.MidiFile,
        reconstructed_midi: mido.MidiFile
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.

        Compares original and reconstructed MIDI to measure quality.

        Args:
            original_midi: Original MIDI file
            reconstructed_midi: Reconstructed MIDI file

        Returns:
            Dictionary of quality metrics:
            {
                'note_precision': 0.85,
                'note_recall': 0.82,
                'note_f1': 0.83,
                'pitch_accuracy': 0.90,
                'rhythm_similarity': 0.75,
                'overall_similarity': 0.80
            }
        """
        metrics = {}

        # Extract note events from both
        orig_notes = self._extract_notes(original_midi)
        recon_notes = self._extract_notes(reconstructed_midi)

        # Note-level metrics
        if len(recon_notes) > 0:
            metrics['note_precision'] = self._compute_note_precision(
                orig_notes, recon_notes
            )
        else:
            metrics['note_precision'] = 0.0

        if len(orig_notes) > 0:
            metrics['note_recall'] = self._compute_note_recall(
                orig_notes, recon_notes
            )
        else:
            metrics['note_recall'] = 0.0

        # F1 score
        if metrics['note_precision'] + metrics['note_recall'] > 0:
            metrics['note_f1'] = (
                2 * metrics['note_precision'] * metrics['note_recall'] /
                (metrics['note_precision'] + metrics['note_recall'])
            )
        else:
            metrics['note_f1'] = 0.0

        # Pitch accuracy
        metrics['pitch_accuracy'] = self._compute_pitch_accuracy(
            orig_notes, recon_notes
        )

        # Rhythm similarity
        metrics['rhythm_similarity'] = self._compute_rhythm_similarity(
            orig_notes, recon_notes
        )

        # Overall similarity (weighted average)
        metrics['overall_similarity'] = (
            0.4 * metrics['note_f1'] +
            0.3 * metrics['pitch_accuracy'] +
            0.3 * metrics['rhythm_similarity']
        )

        return metrics

    def _extract_notes(self, midi: mido.MidiFile) -> list:
        """Extract all notes from MIDI file."""
        notes = []
        current_time = 0

        for track in midi.tracks:
            track_time = 0
            active_notes = {}

            for msg in track:
                track_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.note, msg.channel)
                    active_notes[key] = (msg.velocity, track_time)

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.note, msg.channel)
                    if key in active_notes:
                        velocity, start_time = active_notes.pop(key)
                        notes.append({
                            'pitch': msg.note,
                            'start': start_time,
                            'duration': track_time - start_time,
                            'velocity': velocity,
                            'channel': msg.channel
                        })

        return notes

    def _compute_note_precision(self, orig_notes: list, recon_notes: list) -> float:
        """Compute note precision (TP / (TP + FP))."""
        if len(recon_notes) == 0:
            return 0.0

        matches = 0
        for recon in recon_notes:
            if self._find_matching_note(recon, orig_notes):
                matches += 1

        return matches / len(recon_notes)

    def _compute_note_recall(self, orig_notes: list, recon_notes: list) -> float:
        """Compute note recall (TP / (TP + FN))."""
        if len(orig_notes) == 0:
            return 0.0

        matches = 0
        for orig in orig_notes:
            if self._find_matching_note(orig, recon_notes):
                matches += 1

        return matches / len(orig_notes)

    def _find_matching_note(self, note: dict, note_list: list, tolerance: float = 0.1) -> bool:
        """Find if a matching note exists in list."""
        for candidate in note_list:
            # Match if pitch same and time close
            if (candidate['pitch'] == note['pitch'] and
                abs(candidate['start'] - note['start']) < tolerance):
                return True
        return False

    def _compute_pitch_accuracy(self, orig_notes: list, recon_notes: list) -> float:
        """Compute pitch accuracy."""
        if len(recon_notes) == 0 or len(orig_notes) == 0:
            return 0.0

        # Simple: count how many pitches match
        orig_pitches = sorted([n['pitch'] for n in orig_notes])
        recon_pitches = sorted([n['pitch'] for n in recon_notes])

        matches = sum(1 for o, r in zip(orig_pitches, recon_pitches) if o == r)
        return matches / max(len(orig_pitches), len(recon_pitches))

    def _compute_rhythm_similarity(self, orig_notes: list, recon_notes: list) -> float:
        """Compute rhythm similarity."""
        if len(recon_notes) == 0 or len(orig_notes) == 0:
            return 0.0

        # Compare inter-onset intervals
        orig_onsets = sorted([n['start'] for n in orig_notes])
        recon_onsets = sorted([n['start'] for n in recon_notes])

        orig_ioi = np.diff(orig_onsets) if len(orig_onsets) > 1 else np.array([])
        recon_ioi = np.diff(recon_onsets) if len(recon_onsets) > 1 else np.array([])

        if len(orig_ioi) == 0 or len(recon_ioi) == 0:
            return 0.0

        # Compute correlation
        min_len = min(len(orig_ioi), len(recon_ioi))
        if min_len == 0:
            return 0.0

        correlation = np.corrcoef(orig_ioi[:min_len], recon_ioi[:min_len])[0, 1]
        return max(0.0, correlation)  # Clip to [0, 1]


# ============================================================================
# Utility Functions
# ============================================================================

def validate_features(features: Union[np.ndarray, 'torch.Tensor']) -> bool:
    """
    Validate that features have correct shape and values.

    Args:
        features: Feature vector to validate

    Returns:
        True if valid
    """
    # Check shape
    if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
        shape = features.shape
        if len(shape) == 1:
            if shape[0] != 1150:
                return False
        elif len(shape) == 2:
            if shape[1] != 1150:
                return False
        else:
            return False
    elif isinstance(features, np.ndarray):
        shape = features.shape
        if len(shape) == 1:
            if shape[0] != 1150:
                return False
        elif len(shape) == 2:
            if shape[1] != 1150:
                return False
        else:
            return False
    else:
        return False

    # Check for NaN/Inf
    if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
        if torch.isnan(features).any() or torch.isinf(features).any():
            return False
    elif isinstance(features, np.ndarray):
        if np.isnan(features).any() or np.isinf(features).any():
            return False

    return True


if __name__ == "__main__":
    print("="*70)
    print("Features to MIDI - Base Interface")
    print("="*70)

    print("\n✅ Base interface module loaded")
    print("   Feature dimensions: 1150D")
    print("   Breakdown:")
    print("     - Harmony: 250D (0-250)")
    print("     - Rhythm: 250D (250-500)")
    print("     - Melody: 200D (500-700)")
    print("     - Dynamics: 150D (700-850)")
    print("     - Texture: 100D (850-950)")
    print("     - Structure: 50D (950-1000)")
    print("     - Orchestration: 150D (1000-1150)")

    print("\n📝 Abstract methods required:")
    print("   - features_to_midi()")
    print("   - features_to_parameters()")

    print("\n🔧 Utility methods provided:")
    print("   - validate_output()")
    print("   - compute_reconstruction_quality()")
    print("   - extract_feature_slice()")

    print("\n" + "="*70)
