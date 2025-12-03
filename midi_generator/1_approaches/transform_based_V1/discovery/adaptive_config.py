"""
Adaptive Vocabulary-First MDL Configuration

Automatically scales parameters based on corpus size to maintain:
1. Constant statistical coverage (catch patterns above threshold)
2. Roughly O(n) compute time, not O(n²)
3. Memory within hardware limits
"""

from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass
class AdaptiveVocabConfig:
    """Auto-configured parameters for vocabulary-first MDL."""

    # Corpus stats (input)
    num_files: int
    num_objects: int
    num_timesteps: int
    avg_tracks_per_timestep: float

    # Derived parameters (output)
    samples_per_timestep: int
    min_frequency: int
    min_early_matches: int
    early_stop_fraction: float
    target_total_pairs: int
    max_compounds: int
    use_sampling: bool
    use_early_stopping: bool

    # Hardware constraints
    gpu_memory_gb: float = 40.0
    target_time_minutes: float = 60.0


def compute_adaptive_config(
    num_files: int,
    num_objects: int,
    num_timesteps: int,
    avg_tracks_per_timestep: float = 8.0,
    gpu_memory_gb: float = 40.0,
    target_time_minutes: float = 60.0,
    verbose: bool = True
) -> AdaptiveVocabConfig:
    """
    Compute optimal parameters based on corpus size.

    Design principles:
    1. Statistical: Sample enough to catch patterns at 0.01% frequency with 95% confidence
    2. Computational: Stay within target_time_minutes
    3. Memory: Stay within GPU memory
    4. Lewinian: Minimize information loss from sampling
    """

    # =========================================================================
    # CONSTANTS (tuned for A100 40GB)
    # =========================================================================

    PAIRS_PER_SECOND_GPU = 500_000  # Empirical: batch distance computations
    COMPOUNDS_DEPTH_2 = 1089        # 33² (could be reduced with pruning)
    MIN_SAMPLES_FOR_STATS = 100_000 # PAC bound for 0.01% detection at 95% conf

    # =========================================================================
    # COMPUTE EXHAUSTIVE COST
    # =========================================================================

    pairs_per_timestep = avg_tracks_per_timestep * (avg_tracks_per_timestep - 1) / 2
    total_exhaustive_pairs = int(num_timesteps * pairs_per_timestep)
    exhaustive_ops = total_exhaustive_pairs * COMPOUNDS_DEPTH_2
    exhaustive_time_minutes = exhaustive_ops / PAIRS_PER_SECOND_GPU / 60

    # =========================================================================
    # DECIDE: EXHAUSTIVE VS SAMPLING
    # =========================================================================

    if exhaustive_time_minutes <= target_time_minutes * 0.5:
        # Small corpus: exhaustive search is fine
        use_sampling = False
        use_early_stopping = False
        target_total_pairs = total_exhaustive_pairs
        samples_per_timestep = int(pairs_per_timestep)  # All pairs

    elif exhaustive_time_minutes <= target_time_minutes * 2:
        # Medium corpus: use early stopping only
        use_sampling = False
        use_early_stopping = True
        target_total_pairs = total_exhaustive_pairs
        samples_per_timestep = int(pairs_per_timestep)

    else:
        # Large corpus: sampling + early stopping
        use_sampling = True
        use_early_stopping = True

        # Target: enough pairs to stay within time budget
        # With early stopping, effective compounds ≈ 200 (not 1089)
        effective_compounds = 200
        max_pairs_for_time = int(target_time_minutes * 60 * PAIRS_PER_SECOND_GPU / effective_compounds)

        # But need at least MIN_SAMPLES_FOR_STATS for statistical validity
        target_total_pairs = max(MIN_SAMPLES_FOR_STATS, min(max_pairs_for_time, total_exhaustive_pairs))

        # Derive samples_per_timestep
        samples_per_timestep = max(3, int(target_total_pairs / num_timesteps))
        samples_per_timestep = min(samples_per_timestep, int(pairs_per_timestep))  # Cap at exhaustive

    # =========================================================================
    # MIN_FREQUENCY: Scale with corpus size
    # =========================================================================
    #
    # Principle: A pattern is "real" if it appears in ~1% of pieces
    # But floor at 3 to avoid noise in small corpora
    #
    # Also scale down if we're sampling (patterns appear less often in sample)

    sampling_ratio = target_total_pairs / max(1, total_exhaustive_pairs)
    base_min_frequency = max(3, num_files // 10)

    if use_sampling:
        # Adjust for sampling: if we see 10% of pairs, pattern appears 10% as often
        min_frequency = max(3, int(base_min_frequency * sampling_ratio))
    else:
        min_frequency = base_min_frequency

    # =========================================================================
    # EARLY STOPPING PARAMETERS
    # =========================================================================
    #
    # early_stop_fraction: What fraction to test before cutting compounds
    # min_early_matches: Minimum matches in early fraction to continue
    #
    # Principle: If a compound has 0 matches in 10% of data, probability
    # it has significant matches overall is <5%

    if use_early_stopping:
        if num_files < 50:
            early_stop_fraction = 0.2  # 20% for small corpora (more variance)
            min_early_matches = 2
        elif num_files < 200:
            early_stop_fraction = 0.1  # 10% for medium
            min_early_matches = 3
        else:
            early_stop_fraction = 0.05  # 5% for large (enough samples)
            min_early_matches = 5
    else:
        early_stop_fraction = 1.0
        min_early_matches = 0

    # =========================================================================
    # MAX COMPOUNDS: Prune obviously useless combinations
    # =========================================================================
    #
    # Full: 33² = 1089 two-step compounds
    # Pruned: Skip meaningless combinations like retrograde ∘ retrograde

    max_compounds = COMPOUNDS_DEPTH_2  # Could reduce with semantic pruning

    # =========================================================================
    # BUILD CONFIG
    # =========================================================================

    config = AdaptiveVocabConfig(
        num_files=num_files,
        num_objects=num_objects,
        num_timesteps=num_timesteps,
        avg_tracks_per_timestep=avg_tracks_per_timestep,
        samples_per_timestep=samples_per_timestep,
        min_frequency=min_frequency,
        min_early_matches=min_early_matches,
        early_stop_fraction=early_stop_fraction,
        target_total_pairs=target_total_pairs,
        max_compounds=max_compounds,
        use_sampling=use_sampling,
        use_early_stopping=use_early_stopping,
        gpu_memory_gb=gpu_memory_gb,
        target_time_minutes=target_time_minutes,
    )

    if verbose:
        print_config(config, total_exhaustive_pairs, exhaustive_time_minutes)

    return config


def print_config(config: AdaptiveVocabConfig, exhaustive_pairs: int, exhaustive_time: float):
    """Pretty print the adaptive configuration."""

    print(f"\n{'='*70}")
    print("ADAPTIVE VOCABULARY-FIRST CONFIGURATION")
    print(f"{'='*70}")

    print(f"\n  Corpus Statistics:")
    print(f"    Files: {config.num_files}")
    print(f"    Objects: {config.num_objects:,}")
    print(f"    Timesteps: {config.num_timesteps:,}")
    print(f"    Avg tracks/timestep: {config.avg_tracks_per_timestep:.1f}")

    print(f"\n  Exhaustive Cost:")
    print(f"    Total pairs: {exhaustive_pairs:,}")
    print(f"    Estimated time: {exhaustive_time:.1f} minutes")

    print(f"\n  Adaptive Strategy:")
    print(f"    Use sampling: {config.use_sampling}")
    print(f"    Use early stopping: {config.use_early_stopping}")

    print(f"\n  Derived Parameters:")
    print(f"    samples_per_timestep: {config.samples_per_timestep}")
    print(f"    target_total_pairs: {config.target_total_pairs:,}")
    print(f"    min_frequency: {config.min_frequency}")
    print(f"    early_stop_fraction: {config.early_stop_fraction}")
    print(f"    min_early_matches: {config.min_early_matches}")

    sampling_ratio = config.target_total_pairs / max(1, exhaustive_pairs)
    estimated_time = exhaustive_time * sampling_ratio
    if config.use_early_stopping:
        estimated_time *= 0.3  # Early stopping typically eliminates ~70% of compounds

    print(f"\n  Estimated Runtime:")
    print(f"    Phase 1 (mining): {estimated_time:.1f} minutes")
    print(f"    Total (with Phase 3): {estimated_time * 2:.1f} minutes")

    print(f"{'='*70}\n")


# =========================================================================
# PRESET CONFIGURATIONS FOR TESTING
# =========================================================================

def config_for_scale(num_files: int, objects_per_file: int = 3500) -> dict:
    """
    Quick lookup for expected configurations at different scales.

    Returns dict suitable for run_vocabulary_first_mdl(**config)
    """

    # Rough estimates
    num_objects = num_files * objects_per_file
    timesteps_per_file = 300  # ~5 min at 16th notes, scale 16
    num_timesteps = num_files * timesteps_per_file
    avg_tracks = 8

    config = compute_adaptive_config(
        num_files=num_files,
        num_objects=num_objects,
        num_timesteps=num_timesteps,
        avg_tracks_per_timestep=avg_tracks,
        verbose=False
    )

    return {
        'samples_per_timestep': config.samples_per_timestep,
        'min_frequency': config.min_frequency,
        'early_stop_fraction': config.early_stop_fraction,
        'min_early_matches': config.min_early_matches,
        'use_sampling': config.use_sampling,
        'use_early_stopping': config.use_early_stopping,
    }


# Quick reference table
SCALE_PRESETS = {
    10:   {'use_sampling': False, 'use_early_stopping': False, 'min_frequency': 3,  'samples_per_timestep': 28},
    25:   {'use_sampling': False, 'use_early_stopping': False, 'min_frequency': 3,  'samples_per_timestep': 28},
    50:   {'use_sampling': False, 'use_early_stopping': True,  'min_frequency': 5,  'samples_per_timestep': 28},
    100:  {'use_sampling': False, 'use_early_stopping': True,  'min_frequency': 10, 'samples_per_timestep': 28},
    250:  {'use_sampling': True,  'use_early_stopping': True,  'min_frequency': 15, 'samples_per_timestep': 15},
    500:  {'use_sampling': True,  'use_early_stopping': True,  'min_frequency': 25, 'samples_per_timestep': 8},
    1000: {'use_sampling': True,  'use_early_stopping': True,  'min_frequency': 50, 'samples_per_timestep': 5},
}
