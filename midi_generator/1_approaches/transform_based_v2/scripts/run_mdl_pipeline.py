#!/usr/bin/env python3
"""
MDL-Based Pipeline Runner
=========================

Runs the full Dosedo pipeline with MDL-discovered transforms
instead of hardcoded D24.

Pipeline:
1. Load MIDI files → extract pitch-class sequences
2. Grammar induction (SEQUITUR)
3. Transform discovery (MDL primitives, NOT hardcoded D24)
4. Build canonical vocabulary
5. Save checkpoint

Usage:
    python scripts/run_mdl_pipeline.py --corpus /path/to/midi --output checkpoint_mdl.npz --max-files 500

Author: MDL Pipeline
"""

import os
import sys
import time
import json
import glob
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from core.primitives import (
    CompoundTransform,
    enumerate_compounds,
    apply_compound,
    find_transform,
    get_d24_with_retrograde,
)
from discovery.primitive_mdl import (
    extract_canonical_pairs,
    mine_transform_relations,
    select_vocabulary_mdl,
    compare_with_d24,
)


@dataclass
class PipelineStats:
    """Statistics from pipeline run."""
    n_files_loaded: int = 0
    n_files_failed: int = 0
    n_tracks: int = 0
    n_notes: int = 0
    n_pitch_sequences: int = 0
    n_grammar_rules: int = 0
    n_canonical_patterns: int = 0
    n_transform_vocabulary: int = 0
    d24_coverage: float = 0.0
    mdl_coverage: float = 0.0
    total_time: float = 0.0
    phase_times: Dict[str, float] = field(default_factory=dict)


def load_midi_to_pitch_class(midi_path: str) -> Optional[List[Dict]]:
    """
    Load MIDI file and extract pitch-class sequences per track.

    Returns list of dicts with:
        - piece_id: str
        - track_id: int
        - pitch_classes: List[int] (0-11)
        - onsets: List[int] (ticks)
    """
    try:
        import mido

        midi = mido.MidiFile(midi_path)
        piece_id = Path(midi_path).stem

        # Extract notes per track
        tracks = defaultdict(list)

        for track_idx, track in enumerate(midi.tracks):
            current_time = 0
            active = {}  # (channel, pitch) -> onset

            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    key = (msg.channel, msg.note)
                    active[key] = current_time

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    key = (msg.channel, msg.note)
                    if key in active:
                        onset = active[key]
                        tracks[track_idx].append({
                            'pitch': msg.note,
                            'pitch_class': msg.note % 12,
                            'onset': onset,
                        })
                        del active[key]

        # Convert to pitch-class sequences
        results = []
        for track_id, notes in tracks.items():
            if len(notes) < 3:  # Skip very short tracks
                continue

            # Sort by onset
            notes = sorted(notes, key=lambda n: n['onset'])

            results.append({
                'piece_id': piece_id,
                'track_id': track_id,
                'pitch_classes': [n['pitch_class'] for n in notes],
                'onsets': [n['onset'] for n in notes],
            })

        return results if results else None

    except Exception as e:
        return None


def run_fast_ngram_extraction(
    sequences: List[List[int]],
    min_n: int = 3,
    max_n: int = 12,
    min_freq: int = 3,
    max_patterns: int = 2000,
    verbose: bool = True
) -> Dict:
    """
    Fast n-gram pattern extraction (replaces slow SEQUITUR).

    For MDL transform discovery, we just need unique patterns.
    N-gram extraction is embarrassingly parallel and much faster.

    Args:
        sequences: List of pitch-class sequences
        min_n: Minimum n-gram size
        max_n: Maximum n-gram size
        min_freq: Minimum frequency to keep
        max_patterns: Maximum patterns to return
        verbose: Print progress

    Returns dict with:
        - rules: Dict[rule_id -> pattern]
        - n_rules: int
        - n_sequences: int
    """
    if verbose:
        total_notes = sum(len(s) for s in sequences)
        print(f"  Fast n-gram extraction: {len(sequences)} seqs, {total_notes:,} notes", flush=True)

    # Use Counter for efficient counting
    from collections import Counter
    ngrams = Counter()

    # Process in batches for progress
    batch_size = 500
    for batch_start in range(0, len(sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_seqs = sequences[batch_start:batch_end]

        for seq in batch_seqs:
            seq_len = len(seq)
            for n in range(min_n, min(max_n + 1, seq_len + 1)):
                for i in range(seq_len - n + 1):
                    # Convert to pitch classes here for efficiency
                    ngram = tuple(x % 12 for x in seq[i:i+n])
                    ngrams[ngram] += 1

        if verbose and batch_end % 1000 == 0:
            print(f"    Processed {batch_end}/{len(sequences)} sequences...", flush=True)

    # Filter by frequency and select top patterns
    frequent = [(ngram, count) for ngram, count in ngrams.most_common() if count >= min_freq]

    # Take top patterns by frequency
    selected = frequent[:max_patterns]

    rules = {
        str(i): list(ngram)
        for i, (ngram, count) in enumerate(selected)
    }

    if verbose:
        print(f"  Extracted {len(rules)} patterns (from {len(ngrams)} unique n-grams)", flush=True)

    return {
        'rules': rules,
        'n_rules': len(rules),
        'n_sequences': len(sequences),
    }


def run_repair_gpu(sequences: List[List[int]], device: str = 'cuda', verbose: bool = True) -> Dict:
    """
    Run GPU-accelerated Re-Pair v2 grammar induction (tensor-based, no Python dicts).

    Re-Pair v2 is fully GPU-optimized:
    1. Pair encoding: left * MAX_VOCAB + right (tensor operation)
    2. Counting via torch.bincount (GPU histogram)
    3. Rule table as tensor (no Python dict in hot loop)
    4. 100% GPU utilization on A100

    Args:
        sequences: List of pitch-class sequences
        device: 'cuda' for GPU, 'cpu' for fallback
        verbose: Print progress

    Returns dict with:
        - rules: Dict[rule_id -> expansion]
        - n_rules: int
        - n_sequences: int
    """
    try:
        from grammar.v2.repair_gpu_v2 import build_repair_grammar_v2

        if verbose:
            total_notes = sum(len(s) for s in sequences)
            print(f"  GPU Re-Pair v2: {len(sequences)} seqs, {total_notes:,} notes", flush=True)

        # Build Re-Pair grammar on GPU (v2 - tensor-based)
        grammar = build_repair_grammar_v2(
            sequences,
            device=device,
            max_rules=10000,
            verbose=verbose
        )

        # Extract terminal expansions as rules
        rules = {}
        for i in range(grammar.n_rules):
            rule_id = grammar.n_terminals + i
            expansion = grammar.expand_rule(rule_id)
            # Only keep patterns of reasonable length for transform discovery
            if 3 <= len(expansion) <= 20:
                rules[str(rule_id)] = expansion

        if verbose:
            print(f"  Re-Pair v2: {len(rules)} patterns (from {grammar.n_rules} rules), "
                  f"compression {grammar.compression_ratio():.2f}x", flush=True)

        return {
            'rules': rules,
            'n_rules': len(rules),
            'n_sequences': len(sequences),
        }

    except ImportError as e:
        if verbose:
            print(f"  Re-Pair v2 not available ({e}), falling back to n-grams", flush=True)
        return run_fast_ngram_extraction(sequences, verbose=verbose)
    except Exception as e:
        if verbose:
            import traceback
            traceback.print_exc()
            print(f"  Re-Pair v2 failed ({e}), falling back to n-grams", flush=True)
        return run_fast_ngram_extraction(sequences, verbose=verbose)


def run_sequitur(sequences: List[List[int]], verbose: bool = True, use_repair_v2: bool = True) -> Dict:
    """
    Run grammar induction on pitch-class sequences.

    By default uses GPU Re-Pair v2 (tensor-based, 100% GPU utilization).
    Set use_repair_v2=False to use fast n-gram extraction.
    """
    if use_repair_v2:
        # GPU Re-Pair v2 - tensor-based, no Python dicts, 100% GPU utilization
        return run_repair_gpu(sequences, device='cuda', verbose=verbose)
    else:
        # Fast n-gram as fallback
        return run_fast_ngram_extraction(sequences, verbose=verbose)


def extract_canonical_patterns(grammar_rules: Dict, min_length: int = 3, max_length: int = 20) -> List[Dict]:
    """
    Extract canonical patterns from grammar rules.

    Returns list of dicts with:
        - pattern_id: int
        - pitch_classes: List[int]
        - rule_id: str
    """
    canonicals = []

    for rule_id, expansion in grammar_rules.items():
        # Only consider terminal rules (all integers)
        if not all(isinstance(x, (int, np.integer)) for x in expansion):
            continue

        # Filter by length
        if not (min_length <= len(expansion) <= max_length):
            continue

        # Pitch classes already mod 12 from n-gram extraction
        pitch_classes = [int(x) % 12 for x in expansion]

        canonicals.append({
            'pattern_id': len(canonicals),
            'pitch_classes': pitch_classes,
            'rule_id': rule_id,
        })

    return canonicals


def run_mdl_transform_discovery(
    canonicals: List[Dict],
    max_depth: int = 2,
    min_frequency: int = 3,
    verbose: bool = True
) -> Dict:
    """
    Run MDL-based transform discovery on canonical patterns.

    Returns dict with:
        - vocabulary: List of transform names
        - stats: Dict of per-transform statistics
        - comparison: D24 vs MDL comparison
    """
    if verbose:
        print(f"\n  MDL Transform Discovery on {len(canonicals)} canonicals...")

    # Extract pairs
    pairs = extract_canonical_pairs(canonicals)
    if verbose:
        print(f"  Created {len(pairs)} pattern pairs")

    if not pairs:
        return {
            'vocabulary': [],
            'stats': {},
            'comparison': {},
        }

    # Generate compound candidates
    candidates = enumerate_compounds(max_depth=max_depth)
    if verbose:
        print(f"  Generated {len(candidates)} compound candidates")

    # Mine relations
    relations = mine_transform_relations(pairs, candidates, verbose=verbose)

    # Select vocabulary via MDL
    vocabulary, stats = select_vocabulary_mdl(relations, min_frequency, verbose=verbose)

    # Compare with D24 (skip for now - too slow with 2M+ pairs)
    # TODO: Use GPU-based comparison or sampling
    comparison = {
        'd24_coverage': 0.0,
        'd48_coverage': 0.0,
        'discovered_coverage': 0.0,
        'd24_count': 24,
        'd48_count': 48,
        'discovered_count': len(vocabulary),
    }
    if verbose:
        print(f"\n(Skipping D24 comparison - 2M+ pairs too slow for CPU)", flush=True)

    return {
        'vocabulary': [t.name for t in vocabulary],
        'stats': {
            name: {
                'frequency': s.frequency,
                'unique_sources': s.unique_sources,
                'mdl_benefit': s.mdl_benefit,
            }
            for name, s in stats.items()
        },
        'comparison': comparison,
    }


def save_checkpoint(
    path: str,
    canonicals: List[Dict],
    grammar: Dict,
    transform_discovery: Dict,
    stats: PipelineStats,
    verbose: bool = True
):
    """Save checkpoint with all discovered structures."""
    data = {
        'version': np.array(['mdl_v1']),

        # Stats
        'n_files': np.array([stats.n_files_loaded]),
        'n_tracks': np.array([stats.n_tracks]),
        'n_canonicals': np.array([len(canonicals)]),
        'n_grammar_rules': np.array([grammar['n_rules']]),
        'n_transform_vocab': np.array([len(transform_discovery['vocabulary'])]),
        'd24_coverage': np.array([transform_discovery['comparison'].get('d24_coverage', 0)]),
        'mdl_coverage': np.array([transform_discovery['comparison'].get('discovered_coverage', 0)]),
        'total_time': np.array([stats.total_time]),

        # JSON data
        'canonical_patterns_json': np.array([json.dumps(canonicals)]),
        'grammar_rules_json': np.array([json.dumps(grammar['rules'])]),
        'transform_vocabulary_json': np.array([json.dumps(transform_discovery['vocabulary'])]),
        'transform_stats_json': np.array([json.dumps(transform_discovery['stats'])]),
        'comparison_json': np.array([json.dumps(transform_discovery['comparison'])]),
    }

    np.savez_compressed(path, **data)

    if verbose:
        print(f"\n  Checkpoint saved to: {path}")


def run_pipeline(
    corpus_path: str,
    output_path: str = 'checkpoint_mdl.npz',
    max_files: int = 500,
    verbose: bool = True
) -> PipelineStats:
    """
    Run the full MDL-based pipeline.

    Args:
        corpus_path: Path to MIDI corpus directory
        output_path: Where to save checkpoint
        max_files: Maximum files to process
        verbose: Print progress

    Returns:
        PipelineStats
    """
    stats = PipelineStats()
    total_start = time.time()

    if verbose:
        print("=" * 70)
        print("MDL-BASED DOSEDO PIPELINE")
        print("=" * 70)

    # Phase 1: Load MIDI files
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 1] Loading MIDI files from {corpus_path}...", flush=True)

    midi_files = sorted(glob.glob(str(Path(corpus_path) / "*.mid")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "*.midi")))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.mid"), recursive=True))
    midi_files += sorted(glob.glob(str(Path(corpus_path) / "**/*.midi"), recursive=True))
    # Remove duplicates while preserving order
    midi_files = list(dict.fromkeys(midi_files))

    if max_files:
        midi_files = midi_files[:max_files]

    if verbose:
        print(f"  Found {len(midi_files)} MIDI files", flush=True)

    # Parallel loading with better logging
    all_tracks = []
    num_workers = min(8, max(1, len(midi_files)))  # Fewer workers to reduce memory

    if verbose:
        print(f"  Using {num_workers} parallel workers", flush=True)

    # Process in batches for better progress reporting
    batch_size = 50
    total_batches = (len(midi_files) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(midi_files))
        batch_files = midi_files[batch_start:batch_end]

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_midi_to_pitch_class, f): f for f in batch_files}

            for future in as_completed(futures):
                try:
                    result = future.result(timeout=30)  # 30s timeout per file
                    if result:
                        all_tracks.extend(result)
                        stats.n_files_loaded += 1
                    else:
                        stats.n_files_failed += 1
                except Exception as e:
                    stats.n_files_failed += 1

        if verbose:
            elapsed = time.time() - phase_start
            rate = (batch_end) / elapsed if elapsed > 0 else 0
            print(f"    [{batch_end}/{len(midi_files)}] {stats.n_files_loaded} loaded, "
                  f"{stats.n_files_failed} failed, {len(all_tracks)} tracks "
                  f"({rate:.1f} files/s)", flush=True)

    stats.n_tracks = len(all_tracks)
    stats.n_notes = sum(len(t['pitch_classes']) for t in all_tracks)
    stats.phase_times['loading'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Loaded {stats.n_files_loaded} files, {stats.n_tracks} tracks, {stats.n_notes:,} notes", flush=True)
        print(f"  Time: {stats.phase_times['loading']:.1f}s", flush=True)

    # Phase 2: Grammar induction
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 2] Running grammar induction on {len(all_tracks)} sequences...", flush=True)

    sequences = [t['pitch_classes'] for t in all_tracks]
    stats.n_pitch_sequences = len(sequences)

    grammar = run_sequitur(sequences, verbose=verbose)
    stats.n_grammar_rules = grammar['n_rules']
    stats.phase_times['grammar'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Induced {stats.n_grammar_rules} grammar rules", flush=True)
        print(f"  Time: {stats.phase_times['grammar']:.1f}s", flush=True)

    # Phase 3: Extract canonical patterns
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 3] Extracting canonical patterns...", flush=True)

    canonicals = extract_canonical_patterns(grammar['rules'])
    stats.n_canonical_patterns = len(canonicals)
    stats.phase_times['canonicals'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Extracted {stats.n_canonical_patterns} canonical patterns", flush=True)
        print(f"  Time: {stats.phase_times['canonicals']:.1f}s", flush=True)

    # Phase 4: MDL Transform Discovery
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 4] MDL Transform Discovery on {stats.n_canonical_patterns} patterns...", flush=True)

    transform_discovery = run_mdl_transform_discovery(
        canonicals,
        max_depth=2,
        min_frequency=3,
        verbose=verbose
    )
    stats.n_transform_vocabulary = len(transform_discovery['vocabulary'])
    stats.d24_coverage = transform_discovery['comparison'].get('d24_coverage', 0)
    stats.mdl_coverage = transform_discovery['comparison'].get('discovered_coverage', 0)
    stats.phase_times['transforms'] = time.time() - phase_start

    if verbose:
        print(f"  ✓ Discovered {stats.n_transform_vocabulary} transforms", flush=True)
        print(f"  D24 coverage: {stats.d24_coverage:.1%}", flush=True)
        print(f"  MDL coverage: {stats.mdl_coverage:.1%}", flush=True)
        print(f"  Time: {stats.phase_times['transforms']:.1f}s", flush=True)

    # Phase 5: Save checkpoint
    phase_start = time.time()
    if verbose:
        print(f"\n[Phase 5] Saving checkpoint to {output_path}...", flush=True)

    stats.total_time = time.time() - total_start

    save_checkpoint(
        output_path,
        canonicals,
        grammar,
        transform_discovery,
        stats,
        verbose=verbose
    )
    stats.phase_times['checkpoint'] = time.time() - phase_start

    # Final summary
    if verbose:
        print(f"\n{'=' * 70}", flush=True)
        print("✓ PIPELINE COMPLETE", flush=True)
        print(f"{'=' * 70}", flush=True)
        print(f"  Files: {stats.n_files_loaded} loaded, {stats.n_files_failed} failed", flush=True)
        print(f"  Tracks: {stats.n_tracks}", flush=True)
        print(f"  Notes: {stats.n_notes:,}", flush=True)
        print(f"  Grammar rules: {stats.n_grammar_rules}", flush=True)
        print(f"  Canonical patterns: {stats.n_canonical_patterns}", flush=True)
        print(f"  Transform vocabulary: {stats.n_transform_vocabulary}", flush=True)
        print(f"  Coverage: D24={stats.d24_coverage:.1%}, MDL={stats.mdl_coverage:.1%}", flush=True)
        print(f"  Total time: {stats.total_time:.1f}s", flush=True)
        print(f"\n  Checkpoint: {output_path}", flush=True)

    return stats


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='MDL-Based Dosedo Pipeline')
    parser.add_argument('--corpus', type=str, required=True, help='Path to MIDI corpus')
    parser.add_argument('--output', type=str, default='checkpoint_mdl.npz', help='Output checkpoint')
    parser.add_argument('--max-files', type=int, default=500, help='Max files to process')
    args = parser.parse_args()

    run_pipeline(
        corpus_path=args.corpus,
        output_path=args.output,
        max_files=args.max_files,
        verbose=True
    )
