#!/usr/bin/env python3
"""
Agent 08: Validation Utilities
===============================

Utility functions for validation including:
1. MIDI comparison functions
2. Statistical tests (KS test, chi-squared, t-test)
3. Musical distance metrics
4. Error aggregation and analysis
5. Distribution similarity measures

Author: Agent 08 - Validation Framework Builder
Date: 2025-11-20
License: MIT
"""

import math
import statistics
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import Counter, defaultdict
import numpy as np


# ==============================================================================
# STATISTICAL TESTS
# ==============================================================================

def calculate_mae(predictions: List[float], ground_truths: List[float]) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        predictions: List of predicted values
        ground_truths: List of ground truth values

    Returns:
        MAE value
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if not predictions:
        return 0.0

    errors = [abs(p - gt) for p, gt in zip(predictions, ground_truths)]
    return statistics.mean(errors)


def calculate_rmse(predictions: List[float], ground_truths: List[float]) -> float:
    """
    Calculate Root Mean Square Error.

    Args:
        predictions: List of predicted values
        ground_truths: List of ground truth values

    Returns:
        RMSE value
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if not predictions:
        return 0.0

    squared_errors = [(p - gt) ** 2 for p, gt in zip(predictions, ground_truths)]
    mse = statistics.mean(squared_errors)
    return math.sqrt(mse)


def calculate_r2_score(predictions: List[float], ground_truths: List[float]) -> float:
    """
    Calculate R² (coefficient of determination).

    Args:
        predictions: List of predicted values
        ground_truths: List of ground truth values

    Returns:
        R² score (can be negative if model is worse than mean baseline)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if not ground_truths:
        return 0.0

    mean_truth = statistics.mean(ground_truths)

    # Total sum of squares
    ss_tot = sum((gt - mean_truth) ** 2 for gt in ground_truths)

    # Residual sum of squares
    ss_res = sum((gt - pred) ** 2 for gt, pred in zip(ground_truths, predictions))

    if ss_tot == 0:
        return 0.0

    return 1 - (ss_res / ss_tot)


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Pearson correlation coefficient.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Correlation coefficient (-1 to 1)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    mean_x = statistics.mean(x)
    mean_y = statistics.mean(y)

    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / len(x)

    std_x = statistics.stdev(x)
    std_y = statistics.stdev(y)

    if std_x == 0 or std_y == 0:
        return 0.0

    return covariance / (std_x * std_y)


def calculate_spearman_correlation(x: List[float], y: List[float]) -> float:
    """
    Calculate Spearman rank correlation coefficient.

    Args:
        x: First variable
        y: Second variable

    Returns:
        Spearman correlation coefficient (-1 to 1)
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    # Rank the data
    def rank_data(data: List[float]) -> List[float]:
        sorted_with_idx = sorted(enumerate(data), key=lambda x: x[1])
        ranks = [0] * len(data)
        for rank, (idx, _) in enumerate(sorted_with_idx, 1):
            ranks[idx] = rank
        return ranks

    ranks_x = rank_data(x)
    ranks_y = rank_data(y)

    # Calculate Pearson correlation on ranks
    return calculate_correlation(ranks_x, ranks_y)


def kolmogorov_smirnov_test(sample1: List[float], sample2: List[float]) -> Tuple[float, float]:
    """
    Perform Kolmogorov-Smirnov test for distribution similarity.

    Args:
        sample1: First sample
        sample2: Second sample

    Returns:
        (statistic, approximate_p_value)
    """
    if not sample1 or not sample2:
        return (1.0, 0.0)

    # Sort samples
    sorted1 = sorted(sample1)
    sorted2 = sorted(sample2)

    # Combine and sort all values
    all_values = sorted(set(sorted1 + sorted2))

    # Calculate empirical CDFs
    n1 = len(sample1)
    n2 = len(sample2)

    max_diff = 0.0

    for value in all_values:
        cdf1 = sum(1 for x in sorted1 if x <= value) / n1
        cdf2 = sum(1 for x in sorted2 if x <= value) / n2
        diff = abs(cdf1 - cdf2)
        max_diff = max(max_diff, diff)

    # Approximate p-value (simplified)
    effective_n = (n1 * n2) / (n1 + n2)
    lambda_ks = max_diff * math.sqrt(effective_n)

    # Very rough p-value approximation
    p_value = math.exp(-2 * lambda_ks ** 2)

    return (max_diff, p_value)


def chi_squared_test(observed: List[int], expected: List[int]) -> Tuple[float, int]:
    """
    Perform chi-squared goodness-of-fit test.

    Args:
        observed: Observed frequencies
        expected: Expected frequencies

    Returns:
        (chi_squared_statistic, degrees_of_freedom)
    """
    if len(observed) != len(expected):
        raise ValueError("Observed and expected must have same length")

    chi_squared = sum(
        (obs - exp) ** 2 / exp if exp > 0 else 0
        for obs, exp in zip(observed, expected)
    )

    df = len(observed) - 1

    return (chi_squared, df)


def calculate_accuracy(predictions: List, ground_truths: List) -> float:
    """
    Calculate classification accuracy.

    Args:
        predictions: List of predicted labels
        ground_truths: List of ground truth labels

    Returns:
        Accuracy (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if not predictions:
        return 0.0

    correct = sum(1 for p, gt in zip(predictions, ground_truths) if p == gt)
    return correct / len(predictions)


def calculate_f1_score(predictions: List, ground_truths: List, positive_label: Any = 1) -> float:
    """
    Calculate F1 score for binary classification.

    Args:
        predictions: List of predicted labels
        ground_truths: List of ground truth labels
        positive_label: Label considered as positive class

    Returns:
        F1 score (0-1)
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")

    if not predictions:
        return 0.0

    tp = sum(1 for p, gt in zip(predictions, ground_truths)
             if p == positive_label and gt == positive_label)
    fp = sum(1 for p, gt in zip(predictions, ground_truths)
             if p == positive_label and gt != positive_label)
    fn = sum(1 for p, gt in zip(predictions, ground_truths)
             if p != positive_label and gt == positive_label)

    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        return 0.0

    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_confusion_matrix(predictions: List, ground_truths: List, labels: Optional[List] = None) -> Dict:
    """
    Calculate confusion matrix.

    Args:
        predictions: List of predicted labels
        ground_truths: List of ground truth labels
        labels: Optional list of all possible labels

    Returns:
        Dictionary with confusion matrix data
    """
    if labels is None:
        labels = sorted(set(predictions + ground_truths))

    matrix = defaultdict(lambda: defaultdict(int))

    for pred, truth in zip(predictions, ground_truths):
        matrix[truth][pred] += 1

    return {
        'matrix': dict(matrix),
        'labels': labels,
        'total': len(predictions)
    }


# ==============================================================================
# DISTRIBUTION SIMILARITY
# ==============================================================================

def cosine_similarity(dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
    """
    Calculate cosine similarity between two distributions.

    Args:
        dist1: First distribution (dict of key: probability)
        dist2: Second distribution

    Returns:
        Cosine similarity (0-1)
    """
    # Get all keys
    all_keys = set(dist1.keys()) | set(dist2.keys())

    # Build vectors
    vec1 = [dist1.get(k, 0.0) for k in all_keys]
    vec2 = [dist2.get(k, 0.0) for k in all_keys]

    # Calculate cosine similarity
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))

    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def kl_divergence(dist1: Dict[Any, float], dist2: Dict[Any, float], epsilon: float = 1e-10) -> float:
    """
    Calculate Kullback-Leibler divergence D(dist1 || dist2).

    Args:
        dist1: First distribution (P)
        dist2: Second distribution (Q)
        epsilon: Small constant to avoid log(0)

    Returns:
        KL divergence (0 to infinity)
    """
    all_keys = set(dist1.keys()) | set(dist2.keys())

    kl = 0.0
    for key in all_keys:
        p = dist1.get(key, 0.0)
        q = dist2.get(key, epsilon)  # Avoid log(0)

        if p > 0:
            kl += p * math.log(p / q)

    return kl


def jensen_shannon_divergence(dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
    """
    Calculate Jensen-Shannon divergence (symmetric version of KL divergence).

    Args:
        dist1: First distribution
        dist2: Second distribution

    Returns:
        JS divergence (0-1)
    """
    all_keys = set(dist1.keys()) | set(dist2.keys())

    # Calculate average distribution
    m = {k: (dist1.get(k, 0.0) + dist2.get(k, 0.0)) / 2 for k in all_keys}

    # JS divergence
    js = 0.5 * kl_divergence(dist1, m) + 0.5 * kl_divergence(dist2, m)

    # Normalize to [0, 1]
    return math.sqrt(js / math.log(2))


def histogram_intersection(dist1: Dict[Any, float], dist2: Dict[Any, float]) -> float:
    """
    Calculate histogram intersection (similarity metric).

    Args:
        dist1: First distribution
        dist2: Second distribution

    Returns:
        Intersection score (0-1)
    """
    all_keys = set(dist1.keys()) | set(dist2.keys())

    intersection = sum(min(dist1.get(k, 0.0), dist2.get(k, 0.0)) for k in all_keys)

    return intersection


# ==============================================================================
# MUSICAL DISTANCE METRICS
# ==============================================================================

def interval_distribution(pitches: List[int]) -> Dict[int, float]:
    """
    Calculate distribution of melodic intervals.

    Args:
        pitches: List of MIDI pitch values

    Returns:
        Dictionary of {interval: probability}
    """
    if len(pitches) < 2:
        return {}

    intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]

    # Count intervals
    counts = Counter(intervals)
    total = len(intervals)

    # Convert to probabilities
    return {interval: count / total for interval, count in counts.items()}


def pitch_class_distribution(pitches: List[int]) -> Dict[int, float]:
    """
    Calculate distribution of pitch classes (0-11).

    Args:
        pitches: List of MIDI pitch values

    Returns:
        Dictionary of {pitch_class: probability}
    """
    if not pitches:
        return {}

    pitch_classes = [p % 12 for p in pitches]

    # Count pitch classes
    counts = Counter(pitch_classes)
    total = len(pitch_classes)

    # Convert to probabilities
    return {pc: count / total for pc, count in counts.items()}


def rhythm_complexity(note_durations: List[float]) -> float:
    """
    Calculate rhythm complexity based on duration variety.

    Args:
        note_durations: List of note durations

    Returns:
        Complexity score (0-1)
    """
    if not note_durations:
        return 0.0

    # Calculate entropy of duration distribution
    counts = Counter(note_durations)
    total = len(note_durations)

    probabilities = [count / total for count in counts.values()]

    # Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)

    # Normalize by max possible entropy
    max_entropy = math.log2(len(counts)) if len(counts) > 1 else 1.0

    return entropy / max_entropy if max_entropy > 0 else 0.0


def harmonic_complexity(chords: List[List[int]]) -> float:
    """
    Calculate harmonic complexity based on chord diversity.

    Args:
        chords: List of chords (each chord is a list of pitch classes)

    Returns:
        Complexity score (0-1)
    """
    if not chords:
        return 0.0

    # Convert chords to tuples for hashing
    chord_tuples = [tuple(sorted(chord)) for chord in chords]

    # Calculate unique chords
    unique_chords = len(set(chord_tuples))
    total_chords = len(chord_tuples)

    return unique_chords / total_chords if total_chords > 0 else 0.0


def voice_leading_cost(chord1: List[int], chord2: List[int]) -> float:
    """
    Calculate voice leading cost between two chords.

    Uses minimum total voice movement.

    Args:
        chord1: First chord (list of pitches)
        chord2: Second chord (list of pitches)

    Returns:
        Total voice movement in semitones
    """
    if not chord1 or not chord2:
        return 0.0

    # Sort chords
    sorted1 = sorted(chord1)
    sorted2 = sorted(chord2)

    # Pad shorter chord
    while len(sorted1) < len(sorted2):
        sorted1.append(sorted1[-1])
    while len(sorted2) < len(sorted1):
        sorted2.append(sorted2[-1])

    # Calculate minimum total movement
    total_movement = sum(abs(p1 - p2) for p1, p2 in zip(sorted1, sorted2))

    return total_movement


def swing_ratio(onset_times: List[float], subdivision: str = '8th') -> float:
    """
    Calculate swing ratio from onset times.

    Args:
        onset_times: List of note onset times in beats
        subdivision: Subdivision level ('8th' or '16th')

    Returns:
        Swing ratio (1.0 = straight, 2.0 = triplet feel, 3.0 = extreme swing)
    """
    if len(onset_times) < 3:
        return 1.0

    # Find pairs of consecutive notes on beats
    pairs = []

    subdivision_duration = 0.5 if subdivision == '8th' else 0.25

    for i in range(len(onset_times) - 1):
        t1 = onset_times[i]
        t2 = onset_times[i + 1]

        # Check if this is a subdivision pair (on beat, off beat)
        if abs((t2 - t1) - subdivision_duration) < 0.1:  # Tolerance
            # Calculate actual ratio
            beat = round(t1)
            expected_offbeat = beat + subdivision_duration
            actual_offbeat = t2

            if abs(actual_offbeat - expected_offbeat) < 0.2:
                # Measure swing amount
                deviation = actual_offbeat - expected_offbeat
                pairs.append(deviation)

    if not pairs:
        return 1.0

    # Average deviation
    avg_deviation = statistics.mean(pairs)

    # Convert to swing ratio
    # Positive deviation = more swing
    swing = 1.0 + avg_deviation / subdivision_duration

    return max(1.0, swing)


# ==============================================================================
# MIDI COMPARISON
# ==============================================================================

def compare_midi_note_sequences(seq1: List[Tuple[float, int, float]],
                               seq2: List[Tuple[float, int, float]],
                               time_tolerance: float = 0.05,
                               duration_tolerance: float = 0.05) -> Dict[str, float]:
    """
    Compare two MIDI note sequences.

    Args:
        seq1: First sequence [(start_time, pitch, duration), ...]
        seq2: Second sequence
        time_tolerance: Tolerance for onset time matching (beats)
        duration_tolerance: Tolerance for duration matching (beats)

    Returns:
        Dictionary with comparison metrics
    """
    if not seq1 or not seq2:
        return {
            'note_accuracy': 0.0,
            'pitch_accuracy': 0.0,
            'rhythm_accuracy': 0.0,
            'total_notes_1': len(seq1),
            'total_notes_2': len(seq2),
            'matched_notes': 0
        }

    matched = 0
    pitch_matches = 0
    rhythm_matches = 0

    # For each note in seq1, find closest match in seq2
    for start1, pitch1, dur1 in seq1:
        best_match_distance = float('inf')
        best_match = None

        for start2, pitch2, dur2 in seq2:
            # Calculate distance
            time_dist = abs(start1 - start2)
            pitch_dist = abs(pitch1 - pitch2)
            dur_dist = abs(dur1 - dur2)

            # Check if within tolerances
            if time_dist <= time_tolerance:
                # Potential match
                total_dist = time_dist + pitch_dist * 0.1 + dur_dist

                if total_dist < best_match_distance:
                    best_match_distance = total_dist
                    best_match = (start2, pitch2, dur2)

        if best_match:
            matched += 1
            _, pitch2, dur2 = best_match

            if pitch1 == pitch2:
                pitch_matches += 1

            if abs(dur1 - dur2) <= duration_tolerance:
                rhythm_matches += 1

    total = len(seq1)

    return {
        'note_accuracy': matched / total if total > 0 else 0.0,
        'pitch_accuracy': pitch_matches / total if total > 0 else 0.0,
        'rhythm_accuracy': rhythm_matches / total if total > 0 else 0.0,
        'total_notes_1': len(seq1),
        'total_notes_2': len(seq2),
        'matched_notes': matched
    }


# ==============================================================================
# ERROR AGGREGATION
# ==============================================================================

def aggregate_errors(errors: List[float], method: str = 'mean') -> float:
    """
    Aggregate a list of errors.

    Args:
        errors: List of error values
        method: Aggregation method ('mean', 'median', 'max', 'min', 'std')

    Returns:
        Aggregated error value
    """
    if not errors:
        return 0.0

    if method == 'mean':
        return statistics.mean(errors)
    elif method == 'median':
        return statistics.median(errors)
    elif method == 'max':
        return max(errors)
    elif method == 'min':
        return min(errors)
    elif method == 'std':
        return statistics.stdev(errors) if len(errors) > 1 else 0.0
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def calculate_error_percentiles(errors: List[float], percentiles: List[int] = [50, 75, 90, 95, 99]) -> Dict[int, float]:
    """
    Calculate error percentiles.

    Args:
        errors: List of error values
        percentiles: List of percentiles to calculate

    Returns:
        Dictionary of {percentile: value}
    """
    if not errors:
        return {p: 0.0 for p in percentiles}

    sorted_errors = sorted(errors)
    n = len(sorted_errors)

    result = {}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[p] = sorted_errors[idx]

    return result


# ==============================================================================
# MAIN - EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("Agent 08: Validation Utilities")
    print("=" * 60)
    print("\nExample usage:")

    # Example 1: Statistical metrics
    predictions = [120, 125, 118, 130, 122]
    ground_truths = [120, 120, 120, 120, 120]

    print("\n1. Statistical Metrics:")
    print(f"   MAE: {calculate_mae(predictions, ground_truths):.2f}")
    print(f"   RMSE: {calculate_rmse(predictions, ground_truths):.2f}")
    print(f"   R²: {calculate_r2_score(predictions, ground_truths):.3f}")
    print(f"   Correlation: {calculate_correlation(predictions, ground_truths):.3f}")

    # Example 2: Distribution similarity
    dist1 = {0: 0.2, 1: 0.3, 2: 0.5}
    dist2 = {0: 0.25, 1: 0.25, 2: 0.5}

    print("\n2. Distribution Similarity:")
    print(f"   Cosine Similarity: {cosine_similarity(dist1, dist2):.3f}")
    print(f"   JS Divergence: {jensen_shannon_divergence(dist1, dist2):.3f}")

    # Example 3: Musical metrics
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
    interval_dist = interval_distribution(pitches)

    print("\n3. Musical Metrics:")
    print(f"   Interval Distribution: {interval_dist}")
    print(f"   Rhythm Complexity: {rhythm_complexity([1.0, 0.5, 0.5, 1.0, 0.25]):.3f}")

    print("\n" + "=" * 60)
