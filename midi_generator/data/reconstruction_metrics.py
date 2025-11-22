"""
MIDI Reconstruction Metrics - Agent 8
======================================

Metrics for evaluating MIDI reconstruction quality.

This module provides:
- Pitch accuracy metrics
- Timing similarity metrics
- Note density metrics
- Harmonic similarity metrics
- Rhythmic similarity metrics
- Composite reconstruction scores

Author: Agent 8 - Data Pipeline & Preprocessing
"""

import warnings
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from collections import Counter
from dataclasses import dataclass, asdict

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class ReconstructionMetrics:
    """
    Comprehensive metrics for MIDI reconstruction quality.
    """
    # Pitch metrics
    pitch_accuracy: float  # Fraction of correct pitches (0-1)
    pitch_precision: float  # TP / (TP + FP)
    pitch_recall: float  # TP / (TP + FN)
    pitch_f1: float  # Harmonic mean of precision and recall

    # Timing metrics
    onset_mae: float  # Mean absolute error for note onsets (seconds)
    duration_mae: float  # Mean absolute error for note durations
    timing_correlation: float  # Correlation of onset times (-1 to 1)

    # Note density metrics
    note_count_ratio: float  # reconstructed / original
    note_density_similarity: float  # Similarity of notes per second (0-1)

    # Harmonic metrics (simplified)
    pitch_class_similarity: float  # Similarity of pitch class distribution (0-1)
    chord_similarity: float  # Similarity of simultaneous notes (0-1)

    # Rhythmic metrics
    ioi_similarity: float  # Inter-onset interval similarity (0-1)
    rhythm_pattern_similarity: float  # Rhythm pattern similarity (0-1)

    # Composite metrics
    overall_score: float  # Weighted average (0-1)

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return asdict(self)

    def print_summary(self, prefix: str = ""):
        """Print metrics summary"""
        print(f"\n{prefix}MIDI Reconstruction Metrics:")
        print(f"{prefix}{'='*60}")
        print(f"{prefix}Overall Score: {self.overall_score:.3f}")
        print(f"{prefix}")
        print(f"{prefix}Pitch Metrics:")
        print(f"{prefix}  Accuracy: {self.pitch_accuracy:.3f}")
        print(f"{prefix}  Precision: {self.pitch_precision:.3f}")
        print(f"{prefix}  Recall: {self.pitch_recall:.3f}")
        print(f"{prefix}  F1: {self.pitch_f1:.3f}")
        print(f"{prefix}")
        print(f"{prefix}Timing Metrics:")
        print(f"{prefix}  Onset MAE: {self.onset_mae:.4f}s")
        print(f"{prefix}  Duration MAE: {self.duration_mae:.4f}s")
        print(f"{prefix}  Correlation: {self.timing_correlation:.3f}")
        print(f"{prefix}")
        print(f"{prefix}Density Metrics:")
        print(f"{prefix}  Note count ratio: {self.note_count_ratio:.3f}")
        print(f"{prefix}  Density similarity: {self.note_density_similarity:.3f}")
        print(f"{prefix}")
        print(f"{prefix}Musical Metrics:")
        print(f"{prefix}  Pitch class similarity: {self.pitch_class_similarity:.3f}")
        print(f"{prefix}  Chord similarity: {self.chord_similarity:.3f}")
        print(f"{prefix}  IOI similarity: {self.ioi_similarity:.3f}")
        print(f"{prefix}{'='*60}\n")


# ============================================================================
# Pitch Metrics
# ============================================================================

def compute_pitch_accuracy(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]],
    time_tolerance: float = 0.1  # 100ms tolerance for matching
) -> Tuple[float, float, float, float]:
    """
    Compute pitch accuracy metrics.

    Matches notes within time_tolerance and computes precision/recall/F1.

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes
        time_tolerance: Time tolerance for matching notes (seconds)

    Returns:
        (accuracy, precision, recall, f1)
    """
    if len(original_notes) == 0:
        return (0.0, 0.0, 0.0, 0.0) if len(reconstructed_notes) == 0 else (0.0, 0.0, 1.0, 0.0)

    if len(reconstructed_notes) == 0:
        return (0.0, 0.0, 0.0, 0.0)

    # Build time-pitch pairs
    original_pairs = [(n['start_time'], n['pitch']) for n in original_notes]
    reconstructed_pairs = [(n['start_time'], n['pitch']) for n in reconstructed_notes]

    # Match notes (greedy matching within tolerance)
    matched = 0
    used_reconstructed = set()

    for orig_time, orig_pitch in original_pairs:
        # Find closest reconstructed note
        best_match = None
        best_distance = float('inf')

        for i, (recon_time, recon_pitch) in enumerate(reconstructed_pairs):
            if i in used_reconstructed:
                continue

            time_diff = abs(recon_time - orig_time)
            if time_diff <= time_tolerance and recon_pitch == orig_pitch:
                if time_diff < best_distance:
                    best_distance = time_diff
                    best_match = i

        if best_match is not None:
            matched += 1
            used_reconstructed.add(best_match)

    # Compute metrics
    true_positives = matched
    false_positives = len(reconstructed_notes) - matched
    false_negatives = len(original_notes) - matched

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = matched / max(len(original_notes), len(reconstructed_notes))

    return (accuracy, precision, recall, f1)


# ============================================================================
# Timing Metrics
# ============================================================================

def compute_timing_metrics(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]]
) -> Tuple[float, float, float]:
    """
    Compute timing similarity metrics.

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes

    Returns:
        (onset_mae, duration_mae, timing_correlation)
    """
    if len(original_notes) == 0 or len(reconstructed_notes) == 0:
        return (0.0, 0.0, 0.0)

    # Match notes by pitch and approximate time
    orig_onsets = []
    recon_onsets = []
    orig_durations = []
    recon_durations = []

    for orig_note in original_notes:
        # Find closest reconstructed note with same pitch
        orig_pitch = orig_note['pitch']
        orig_onset = orig_note['start_time']
        orig_duration = orig_note.get('duration', 0.1)

        best_match = None
        best_time_diff = float('inf')

        for recon_note in reconstructed_notes:
            if recon_note['pitch'] == orig_pitch:
                time_diff = abs(recon_note['start_time'] - orig_onset)
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_match = recon_note

        if best_match is not None:
            orig_onsets.append(orig_onset)
            recon_onsets.append(best_match['start_time'])
            orig_durations.append(orig_duration)
            recon_durations.append(best_match.get('duration', 0.1))

    if len(orig_onsets) == 0:
        return (0.0, 0.0, 0.0)

    # Compute MAE
    onset_mae = np.mean(np.abs(np.array(orig_onsets) - np.array(recon_onsets)))
    duration_mae = np.mean(np.abs(np.array(orig_durations) - np.array(recon_durations)))

    # Compute correlation
    if len(orig_onsets) > 1:
        timing_correlation = float(np.corrcoef(orig_onsets, recon_onsets)[0, 1])
    else:
        timing_correlation = 1.0

    return (onset_mae, duration_mae, timing_correlation)


# ============================================================================
# Note Density Metrics
# ============================================================================

def compute_note_density_metrics(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]],
    total_duration: float
) -> Tuple[float, float]:
    """
    Compute note density similarity.

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes
        total_duration: Total duration in seconds

    Returns:
        (note_count_ratio, density_similarity)
    """
    # Note count ratio
    if len(original_notes) == 0:
        note_count_ratio = 1.0 if len(reconstructed_notes) == 0 else 0.0
    else:
        note_count_ratio = len(reconstructed_notes) / len(original_notes)

    # Density similarity (notes per second, measured in bins)
    if total_duration <= 0:
        return (note_count_ratio, 1.0)

    # Divide into 1-second bins
    n_bins = int(np.ceil(total_duration))
    orig_density = np.zeros(n_bins)
    recon_density = np.zeros(n_bins)

    for note in original_notes:
        bin_idx = int(note['start_time'])
        if 0 <= bin_idx < n_bins:
            orig_density[bin_idx] += 1

    for note in reconstructed_notes:
        bin_idx = int(note['start_time'])
        if 0 <= bin_idx < n_bins:
            recon_density[bin_idx] += 1

    # Compute similarity (1 - normalized MAE)
    if n_bins > 0:
        max_density = max(orig_density.max(), recon_density.max(), 1)
        density_mae = np.mean(np.abs(orig_density - recon_density)) / max_density
        density_similarity = 1.0 - min(density_mae, 1.0)
    else:
        density_similarity = 1.0

    return (note_count_ratio, density_similarity)


# ============================================================================
# Harmonic Metrics
# ============================================================================

def compute_pitch_class_distribution(notes: List[Dict[str, Any]]) -> np.ndarray:
    """
    Compute pitch class distribution (12 pitch classes).

    Args:
        notes: MIDI notes

    Returns:
        12D array with pitch class counts (normalized)
    """
    pitch_classes = [note['pitch'] % 12 for note in notes]
    distribution = np.array([pitch_classes.count(pc) for pc in range(12)])

    # Normalize
    if distribution.sum() > 0:
        distribution = distribution / distribution.sum()

    return distribution


def compute_pitch_class_similarity(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]]
) -> float:
    """
    Compute similarity of pitch class distributions (cosine similarity).

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes

    Returns:
        Similarity score (0-1)
    """
    orig_dist = compute_pitch_class_distribution(original_notes)
    recon_dist = compute_pitch_class_distribution(reconstructed_notes)

    # Cosine similarity
    if orig_dist.sum() > 0 and recon_dist.sum() > 0:
        similarity = np.dot(orig_dist, recon_dist) / (
            np.linalg.norm(orig_dist) * np.linalg.norm(recon_dist)
        )
        return float(similarity)
    else:
        return 0.0


def compute_chord_similarity(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]],
    time_resolution: float = 0.1
) -> float:
    """
    Compute similarity of simultaneous notes (chord similarity).

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes
        time_resolution: Time resolution for considering notes simultaneous

    Returns:
        Similarity score (0-1)
    """
    if len(original_notes) == 0 or len(reconstructed_notes) == 0:
        return 0.0

    # Find max time
    max_time = max(
        max(n['start_time'] for n in original_notes),
        max(n['start_time'] for n in reconstructed_notes)
    )

    # Divide into time bins
    n_bins = int(np.ceil(max_time / time_resolution))

    similarities = []

    for bin_idx in range(n_bins):
        bin_start = bin_idx * time_resolution
        bin_end = (bin_idx + 1) * time_resolution

        # Get pitches active in this bin
        orig_pitches = set()
        for note in original_notes:
            if bin_start <= note['start_time'] < bin_end:
                orig_pitches.add(note['pitch'])

        recon_pitches = set()
        for note in reconstructed_notes:
            if bin_start <= note['start_time'] < bin_end:
                recon_pitches.add(note['pitch'])

        # Compute Jaccard similarity for this bin
        if len(orig_pitches) > 0 or len(recon_pitches) > 0:
            intersection = len(orig_pitches & recon_pitches)
            union = len(orig_pitches | recon_pitches)
            bin_similarity = intersection / union if union > 0 else 0.0
            similarities.append(bin_similarity)

    # Average similarity across bins
    return float(np.mean(similarities)) if similarities else 0.0


# ============================================================================
# Rhythmic Metrics
# ============================================================================

def compute_inter_onset_intervals(notes: List[Dict[str, Any]]) -> np.ndarray:
    """
    Compute inter-onset intervals (IOIs).

    Args:
        notes: MIDI notes (must be sorted by start_time)

    Returns:
        Array of IOIs in seconds
    """
    if len(notes) < 2:
        return np.array([])

    sorted_notes = sorted(notes, key=lambda n: n['start_time'])
    onsets = [n['start_time'] for n in sorted_notes]
    iois = np.diff(onsets)

    return iois


def compute_ioi_similarity(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]]
) -> float:
    """
    Compute similarity of inter-onset interval distributions.

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes

    Returns:
        Similarity score (0-1)
    """
    orig_iois = compute_inter_onset_intervals(original_notes)
    recon_iois = compute_inter_onset_intervals(reconstructed_notes)

    if len(orig_iois) == 0 or len(recon_iois) == 0:
        return 0.0

    # Compute IOI histograms (log scale)
    bins = np.logspace(-2, 1, 50)  # 0.01s to 10s

    orig_hist, _ = np.histogram(orig_iois, bins=bins, density=True)
    recon_hist, _ = np.histogram(recon_iois, bins=bins, density=True)

    # Normalize
    if orig_hist.sum() > 0:
        orig_hist = orig_hist / orig_hist.sum()
    if recon_hist.sum() > 0:
        recon_hist = recon_hist / recon_hist.sum()

    # Compute similarity (1 - Jensen-Shannon divergence)
    m = 0.5 * (orig_hist + recon_hist)

    # KL divergence (with smoothing)
    def kl_div(p, q):
        p = p + 1e-10
        q = q + 1e-10
        return np.sum(p * np.log(p / q))

    js_divergence = 0.5 * kl_div(orig_hist, m) + 0.5 * kl_div(recon_hist, m)
    similarity = 1.0 - min(js_divergence, 1.0)

    return float(similarity)


def compute_rhythm_pattern_similarity(
    original_notes: List[Dict[str, Any]],
    reconstructed_notes: List[Dict[str, Any]],
    quantize_grid: float = 0.125  # 16th note at 120 BPM
) -> float:
    """
    Compute rhythmic pattern similarity (quantized rhythm).

    Args:
        original_notes: Original MIDI notes
        reconstructed_notes: Reconstructed MIDI notes
        quantize_grid: Quantization grid in seconds

    Returns:
        Similarity score (0-1)
    """
    if len(original_notes) == 0 or len(reconstructed_notes) == 0:
        return 0.0

    # Quantize onsets
    def quantize_onsets(notes):
        return [int(n['start_time'] / quantize_grid) for n in notes]

    orig_quantized = Counter(quantize_onsets(original_notes))
    recon_quantized = Counter(quantize_onsets(reconstructed_notes))

    # Compute Jaccard similarity
    all_positions = set(orig_quantized.keys()) | set(recon_quantized.keys())

    intersection = sum(min(orig_quantized.get(pos, 0), recon_quantized.get(pos, 0))
                      for pos in all_positions)
    union = sum(max(orig_quantized.get(pos, 0), recon_quantized.get(pos, 0))
               for pos in all_positions)

    similarity = intersection / union if union > 0 else 0.0

    return float(similarity)


# ============================================================================
# Composite Metrics
# ============================================================================

class MIDIReconstructionMetrics:
    """
    Comprehensive MIDI reconstruction metrics calculator.

    Example:
        metrics_calc = MIDIReconstructionMetrics()
        metrics = metrics_calc.compute(
            original_notes=original_notes,
            reconstructed_notes=reconstructed_notes,
            total_duration=30.0
        )
        metrics.print_summary()
    """

    def __init__(
        self,
        time_tolerance: float = 0.1,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize metrics calculator.

        Args:
            time_tolerance: Time tolerance for note matching (seconds)
            weights: Custom weights for overall score (optional)
        """
        self.time_tolerance = time_tolerance

        # Default weights for overall score
        if weights is None:
            self.weights = {
                'pitch': 0.3,
                'timing': 0.2,
                'density': 0.1,
                'harmonic': 0.2,
                'rhythmic': 0.2
            }
        else:
            self.weights = weights

    def compute(
        self,
        original_notes: List[Dict[str, Any]],
        reconstructed_notes: List[Dict[str, Any]],
        total_duration: float
    ) -> ReconstructionMetrics:
        """
        Compute all reconstruction metrics.

        Args:
            original_notes: Original MIDI notes
            reconstructed_notes: Reconstructed MIDI notes
            total_duration: Total duration in seconds

        Returns:
            ReconstructionMetrics object
        """
        # Pitch metrics
        pitch_acc, pitch_prec, pitch_rec, pitch_f1 = compute_pitch_accuracy(
            original_notes, reconstructed_notes, self.time_tolerance
        )

        # Timing metrics
        onset_mae, duration_mae, timing_corr = compute_timing_metrics(
            original_notes, reconstructed_notes
        )

        # Density metrics
        note_count_ratio, density_sim = compute_note_density_metrics(
            original_notes, reconstructed_notes, total_duration
        )

        # Harmonic metrics
        pitch_class_sim = compute_pitch_class_similarity(
            original_notes, reconstructed_notes
        )
        chord_sim = compute_chord_similarity(
            original_notes, reconstructed_notes
        )

        # Rhythmic metrics
        ioi_sim = compute_ioi_similarity(
            original_notes, reconstructed_notes
        )
        rhythm_sim = compute_rhythm_pattern_similarity(
            original_notes, reconstructed_notes
        )

        # Compute overall score (weighted average)
        pitch_score = pitch_f1
        timing_score = max(0, 1.0 - onset_mae)  # Penalize high MAE
        density_score = density_sim
        harmonic_score = 0.5 * pitch_class_sim + 0.5 * chord_sim
        rhythmic_score = 0.5 * ioi_sim + 0.5 * rhythm_sim

        overall_score = (
            self.weights['pitch'] * pitch_score +
            self.weights['timing'] * timing_score +
            self.weights['density'] * density_score +
            self.weights['harmonic'] * harmonic_score +
            self.weights['rhythmic'] * rhythmic_score
        )

        return ReconstructionMetrics(
            pitch_accuracy=pitch_acc,
            pitch_precision=pitch_prec,
            pitch_recall=pitch_rec,
            pitch_f1=pitch_f1,
            onset_mae=onset_mae,
            duration_mae=duration_mae,
            timing_correlation=timing_corr,
            note_count_ratio=note_count_ratio,
            note_density_similarity=density_sim,
            pitch_class_similarity=pitch_class_sim,
            chord_similarity=chord_sim,
            ioi_similarity=ioi_sim,
            rhythm_pattern_similarity=rhythm_sim,
            overall_score=overall_score
        )

    def compute_batch(
        self,
        original_notes_list: List[List[Dict[str, Any]]],
        reconstructed_notes_list: List[List[Dict[str, Any]]],
        durations: List[float]
    ) -> List[ReconstructionMetrics]:
        """
        Compute metrics for a batch of MIDI files.

        Args:
            original_notes_list: List of original note sequences
            reconstructed_notes_list: List of reconstructed note sequences
            durations: List of durations

        Returns:
            List of ReconstructionMetrics
        """
        assert len(original_notes_list) == len(reconstructed_notes_list) == len(durations)

        metrics_list = []
        for orig, recon, dur in zip(original_notes_list, reconstructed_notes_list, durations):
            metrics = self.compute(orig, recon, dur)
            metrics_list.append(metrics)

        return metrics_list

    def compute_batch_average(
        self,
        original_notes_list: List[List[Dict[str, Any]]],
        reconstructed_notes_list: List[List[Dict[str, Any]]],
        durations: List[float]
    ) -> ReconstructionMetrics:
        """
        Compute average metrics across a batch.

        Args:
            original_notes_list: List of original note sequences
            reconstructed_notes_list: List of reconstructed note sequences
            durations: List of durations

        Returns:
            Average ReconstructionMetrics
        """
        metrics_list = self.compute_batch(
            original_notes_list,
            reconstructed_notes_list,
            durations
        )

        # Average all metrics
        avg_metrics = ReconstructionMetrics(
            pitch_accuracy=np.mean([m.pitch_accuracy for m in metrics_list]),
            pitch_precision=np.mean([m.pitch_precision for m in metrics_list]),
            pitch_recall=np.mean([m.pitch_recall for m in metrics_list]),
            pitch_f1=np.mean([m.pitch_f1 for m in metrics_list]),
            onset_mae=np.mean([m.onset_mae for m in metrics_list]),
            duration_mae=np.mean([m.duration_mae for m in metrics_list]),
            timing_correlation=np.mean([m.timing_correlation for m in metrics_list]),
            note_count_ratio=np.mean([m.note_count_ratio for m in metrics_list]),
            note_density_similarity=np.mean([m.note_density_similarity for m in metrics_list]),
            pitch_class_similarity=np.mean([m.pitch_class_similarity for m in metrics_list]),
            chord_similarity=np.mean([m.chord_similarity for m in metrics_list]),
            ioi_similarity=np.mean([m.ioi_similarity for m in metrics_list]),
            rhythm_pattern_similarity=np.mean([m.rhythm_pattern_similarity for m in metrics_list]),
            overall_score=np.mean([m.overall_score for m in metrics_list])
        )

        return avg_metrics
