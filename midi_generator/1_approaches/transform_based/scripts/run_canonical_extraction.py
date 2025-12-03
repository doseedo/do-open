#!/usr/bin/env python3
"""
Canonical Pattern Extraction Pipeline (GPU-Optimized)
======================================================

This script runs the full pattern extraction pipeline with:
1. Improved drum detection (pitch-based heuristics, not blind channel 9 filter)
2. T-normalization (patterns normalized to start on pitch-class 0)
3. Cross-track deduplication (same pattern in different keys = one pattern)
4. Transform edges linking transposed occurrences
5. GPU-accelerated pattern extraction and hashing

The output is a v26 checkpoint with canonical patterns and proper occurrence tracking.

Usage:
    python scripts/run_canonical_extraction.py /path/to/midi/folder --output checkpoint_v26.npz
"""

import argparse
import json
import time
import gzip
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_factored_pipeline import load_midi_factored, FactoredTrack
from core.canonical_patterns import CanonicalPatternRegistry, CanonicalPattern

# GPU device
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class ExtractionStats:
    """Statistics for the extraction pipeline."""
    n_files_processed: int = 0
    n_files_failed: int = 0
    n_tracks: int = 0
    n_notes: int = 0
    n_patterns_raw: int = 0
    n_patterns_canonical: int = 0
    n_occurrences: int = 0
    n_drum_tracks_filtered: int = 0
    n_melodic_ch9_tracks_kept: int = 0
    compression_ratio: float = 0.0
    processing_time: float = 0.0


def extract_patterns_gpu_batch(
    tracks: List[FactoredTrack],
    min_length: int = 3,
    max_length: int = 16,
    device: str = DEVICE,
) -> Dict[str, List[Tuple[Tuple[int, ...], int, int, int]]]:
    """
    GPU-accelerated batch pattern extraction from multiple tracks.

    Uses vectorized operations for T-normalization and hash computation.

    Returns:
        Dict mapping track_id -> list of (canonical_pc_tuple, t_offset, onset, track_idx)
    """
    results = {}

    # Process tracks in batches for GPU efficiency
    for track in tracks:
        pc = track.pitch_classes
        onsets = track.onsets
        n = len(pc)

        if n < min_length:
            results[track.track_id] = []
            continue

        # Convert to tensor for GPU operations
        pc_tensor = torch.tensor(pc, dtype=torch.int64, device=device)
        onsets_tensor = torch.tensor(onsets, dtype=torch.int64, device=device)

        track_patterns = []

        # Vectorized extraction for each window size
        for length in range(min_length, min(max_length, n) + 1):
            if n < length:
                break

            # Create all windows of this length using unfold
            # Shape: (n_windows, length)
            windows = pc_tensor.unfold(0, length, 1)
            window_onsets = onsets_tensor[:n - length + 1]

            # Vectorized T-normalization: subtract first element from each window
            # Shape: (n_windows,)
            base_notes = windows[:, 0]
            # Shape: (n_windows, length)
            normalized = (windows - base_notes.unsqueeze(1)) % 12

            # Move to CPU for dict operations (hashing still needs Python)
            normalized_cpu = normalized.cpu().numpy()
            onsets_cpu = window_onsets.cpu().numpy()
            bases_cpu = base_notes.cpu().numpy()

            # Convert to tuples for hashing
            for i in range(len(normalized_cpu)):
                canonical = tuple(normalized_cpu[i].tolist())
                t_offset = int(bases_cpu[i])
                onset = int(onsets_cpu[i])
                track_patterns.append((canonical, t_offset, onset, length))

        results[track.track_id] = track_patterns

    return results


def extract_patterns_from_track(
    track: FactoredTrack,
    min_length: int = 3,
    max_length: int = 32,
) -> List[Tuple[List[int], int, int]]:
    """
    Extract patterns from a single track using sliding window.
    Falls back to CPU for compatibility.

    Returns list of (pitch_classes, onset_time, length) tuples.
    """
    pc = track.pitch_classes.tolist()
    onsets = track.onsets.tolist()

    patterns = []

    # Use multiple window sizes for multi-scale patterns
    for length in range(min_length, min(max_length, len(pc)) + 1):
        for i in range(len(pc) - length + 1):
            pattern_pcs = pc[i:i+length]
            onset = onsets[i]
            patterns.append((pattern_pcs, onset, length))

    return patterns


def run_extraction_pipeline(
    midi_folder: str,
    output_path: str,
    max_files: Optional[int] = None,
    min_pattern_length: int = 3,
    max_pattern_length: int = 16,
    verbose: bool = True,
    use_gpu: bool = True,
) -> ExtractionStats:
    """
    Run the canonical pattern extraction pipeline with GPU acceleration.

    Args:
        midi_folder: Path to folder containing MIDI files
        output_path: Where to save the checkpoint
        max_files: Maximum files to process (None = all)
        min_pattern_length: Minimum pattern length
        max_pattern_length: Maximum pattern length
        verbose: Print progress
        use_gpu: Use GPU-accelerated extraction

    Returns:
        ExtractionStats with pipeline statistics
    """
    start_time = time.time()
    stats = ExtractionStats()

    device = DEVICE if use_gpu else 'cpu'
    if verbose:
        print(f"Using device: {device}")

    # Find all MIDI files
    midi_folder = Path(midi_folder)
    midi_files = list(midi_folder.glob("**/*.mid")) + list(midi_folder.glob("**/*.midi"))

    if max_files:
        midi_files = midi_files[:max_files]

    if verbose:
        print(f"Found {len(midi_files)} MIDI files")

    # Initialize canonical pattern registry with GPU-backed hash table
    registry = CanonicalPatternRegistry()

    # Track occurrences by piece
    piece_occurrences = defaultdict(list)

    # Process each MIDI file
    for i, midi_path in enumerate(midi_files):
        try:
            tracks = load_midi_factored(str(midi_path))

            if tracks is None:
                stats.n_files_failed += 1
                continue

            stats.n_files_processed += 1
            piece_id = midi_path.stem

            # GPU batch extraction - process all tracks at once
            if use_gpu and len(tracks) > 0:
                gpu_results = extract_patterns_gpu_batch(
                    tracks,
                    min_length=min_pattern_length,
                    max_length=max_pattern_length,
                    device=device,
                )

                for track in tracks:
                    stats.n_tracks += 1
                    stats.n_notes += len(track.notes)

                    track_patterns = gpu_results.get(track.track_id, [])
                    stats.n_patterns_raw += len(track_patterns)

                    # Add pre-normalized patterns to registry
                    for canonical, t_offset, onset, length in track_patterns:
                        # Canonical is already normalized, get pattern ID directly
                        pattern_id = registry.get_pattern_id(canonical)

                        # Add to registry if new
                        if canonical not in registry._patterns:
                            registry._patterns[canonical] = CanonicalPattern(
                                id=pattern_id,
                                pitch_classes=canonical,
                                length=length,
                            )

                        pattern = registry._patterns[canonical]
                        pattern.occurrence_count += 1

                        # Record occurrence
                        from core.canonical_patterns import PatternOccurrence
                        occ = PatternOccurrence(
                            pattern_id=pattern_id,
                            piece_id=piece_id,
                            track_id=track.track_id,
                            onset_time=onset,
                            t_offset=t_offset,
                        )
                        occ_idx = len(registry._occurrences)
                        registry._occurrences.append(occ)
                        registry._occurrences_by_pattern[pattern_id].append(occ_idx)

                        piece_occurrences[piece_id].append({
                            'pattern_id': pattern_id,
                            'track_id': track.track_id,
                            'onset': onset,
                            't_offset': t_offset,
                        })
            else:
                # CPU fallback
                for track in tracks:
                    stats.n_tracks += 1
                    stats.n_notes += len(track.notes)

                    patterns = extract_patterns_from_track(
                        track,
                        min_length=min_pattern_length,
                        max_length=max_pattern_length,
                    )

                    stats.n_patterns_raw += len(patterns)

                    for pattern_pcs, onset, length in patterns:
                        canonical_id, t_offset = registry.add_pattern(
                            pitch_classes=pattern_pcs,
                            piece_id=piece_id,
                            track_id=track.track_id,
                            onset_time=onset,
                        )

                        piece_occurrences[piece_id].append({
                            'pattern_id': canonical_id,
                            'track_id': track.track_id,
                            'onset': onset,
                            't_offset': t_offset,
                        })

            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(midi_files)} files, "
                      f"{registry.num_patterns} canonical patterns")

        except Exception as e:
            stats.n_files_failed += 1
            if verbose:
                print(f"  Failed: {midi_path.name}: {e}")

    # Update stats
    stats.n_patterns_canonical = registry.num_patterns
    stats.n_occurrences = registry.num_occurrences
    stats.compression_ratio = stats.n_patterns_raw / stats.n_patterns_canonical if stats.n_patterns_canonical > 0 else 0
    stats.processing_time = time.time() - start_time

    if verbose:
        print(f"\n=== Extraction Complete ===")
        print(f"Files: {stats.n_files_processed} processed, {stats.n_files_failed} failed")
        print(f"Tracks: {stats.n_tracks}")
        print(f"Notes: {stats.n_notes}")
        print(f"Raw patterns: {stats.n_patterns_raw}")
        print(f"Canonical patterns: {stats.n_patterns_canonical}")
        print(f"Compression ratio: {stats.compression_ratio:.1f}x")
        print(f"Occurrences: {stats.n_occurrences}")
        print(f"Time: {stats.processing_time:.2f}s")

    if verbose:
        print(f"\nBuilding save data...")

    # Build pattern ID to index mapping for efficient storage
    pattern_id_to_idx = {}
    pattern_list = []
    for i, (normalized, pattern) in enumerate(registry._patterns.items()):
        pattern_id_to_idx[pattern.id] = i
        pattern_list.append({
            'id': pattern.id,
            'pitch_classes': list(pattern.pitch_classes),
            'length': pattern.length,
            'occurrence_count': pattern.occurrence_count,
        })

    # Build piece ID to index mapping
    piece_id_to_idx = {}
    piece_list = list(piece_occurrences.keys())
    for i, pid in enumerate(piece_list):
        piece_id_to_idx[pid] = i

    # Convert occurrences to columnar numpy arrays (MUCH faster than JSON)
    n_occ = len(registry._occurrences)
    if verbose:
        print(f"  Converting {n_occ} occurrences to numpy arrays...")

    occ_pattern_idx = np.zeros(n_occ, dtype=np.int32)
    occ_piece_idx = np.zeros(n_occ, dtype=np.int32)
    occ_track_id = np.zeros(n_occ, dtype=np.int16)
    occ_onset = np.zeros(n_occ, dtype=np.int32)
    occ_t_offset = np.zeros(n_occ, dtype=np.int8)

    for i, occ in enumerate(registry._occurrences):
        occ_pattern_idx[i] = pattern_id_to_idx.get(occ.pattern_id, 0)
        occ_piece_idx[i] = piece_id_to_idx.get(occ.piece_id, 0)
        occ_track_id[i] = occ.track_id
        occ_onset[i] = occ.onset_time
        occ_t_offset[i] = occ.t_offset

    # Build transform edges (T-relationships between occurrences)
    if verbose:
        print(f"  Building transform edges...")

    transform_edges = []
    for pattern_id, occs in registry._occurrences_by_pattern.items():
        occ_list = [registry._occurrences[i] for i in occs]

        # Group by piece
        by_piece = defaultdict(list)
        for occ in occ_list:
            by_piece[occ.piece_id].append(occ)

        # Create T-edges between different transpositions in same piece
        for piece_id, piece_occs in by_piece.items():
            t_offsets = set(occ.t_offset for occ in piece_occs)
            if len(t_offsets) > 1:
                # Multiple transpositions of same pattern in same piece
                offsets = sorted(t_offsets)
                for j in range(len(offsets)):
                    for k in range(j + 1, len(offsets)):
                        t_diff = (offsets[k] - offsets[j]) % 12
                        transform_edges.append((
                            pattern_id_to_idx.get(pattern_id, 0),
                            piece_id_to_idx.get(piece_id, 0),
                            offsets[j],
                            offsets[k],
                            t_diff,
                        ))

    # Convert transform edges to numpy
    if transform_edges:
        transform_edges_arr = np.array(transform_edges, dtype=np.int32)
    else:
        transform_edges_arr = np.zeros((0, 5), dtype=np.int32)

    if verbose:
        print(f"  Saving to {output_path}...")

    # Save everything to a single compressed npz (MUCH faster than JSON)
    np.savez_compressed(
        output_path,
        # Metadata
        version=np.array(['v27_canonical_numpy']),
        stats_json=np.array([json.dumps({
            'n_files_processed': stats.n_files_processed,
            'n_tracks': stats.n_tracks,
            'n_notes': stats.n_notes,
            'n_patterns_raw': stats.n_patterns_raw,
            'n_patterns_canonical': stats.n_patterns_canonical,
            'n_occurrences': stats.n_occurrences,
            'compression_ratio': stats.compression_ratio,
        })]),

        # Pattern data (small, use JSON)
        patterns_json=np.array([json.dumps(pattern_list)]),
        piece_ids_json=np.array([json.dumps(piece_list)]),

        # Occurrence data as columnar arrays (large, use numpy)
        occ_pattern_idx=occ_pattern_idx,
        occ_piece_idx=occ_piece_idx,
        occ_track_id=occ_track_id,
        occ_onset=occ_onset,
        occ_t_offset=occ_t_offset,

        # Transform edges as numpy array
        transform_edges=transform_edges_arr,
    )

    if verbose:
        print(f"\nSaved checkpoint to {output_path}")
        print(f"Transform edges: {len(transform_edges)}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Run canonical pattern extraction pipeline"
    )
    parser.add_argument(
        "midi_folder",
        help="Path to folder containing MIDI files"
    )
    parser.add_argument(
        "--output", "-o",
        default="checkpoint_v26_canonical.npz",
        help="Output checkpoint path (default: checkpoint_v26_canonical.npz)"
    )
    parser.add_argument(
        "--max-files", "-n",
        type=int,
        default=None,
        help="Maximum number of files to process"
    )
    parser.add_argument(
        "--min-pattern-length",
        type=int,
        default=3,
        help="Minimum pattern length (default: 3)"
    )
    parser.add_argument(
        "--max-pattern-length",
        type=int,
        default=16,
        help="Maximum pattern length (default: 16)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    args = parser.parse_args()

    stats = run_extraction_pipeline(
        midi_folder=args.midi_folder,
        output_path=args.output,
        max_files=args.max_files,
        min_pattern_length=args.min_pattern_length,
        max_pattern_length=args.max_pattern_length,
        verbose=not args.quiet,
    )

    print(f"\nDone! Canonical patterns: {stats.n_patterns_canonical}, "
          f"Compression: {stats.compression_ratio:.1f}x")


if __name__ == '__main__':
    main()
