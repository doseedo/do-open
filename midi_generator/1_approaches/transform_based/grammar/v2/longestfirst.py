"""
Step 3: LONGESTFIRST Grammar Variant (GPU-Optimized, T-Normalized)
===================================================================

Specifically designed for musical phrase segmentation with best F1 scores
on folk/hymn corpora.

Key insight: "combining LONGESTFIRST with duration-aware encoding dramatically
improves phrase detection"

What this means:
1. Prioritize longer patterns before shorter ones
2. Include duration information in pattern matching (not just pitch)
3. Use factorized representation (pitch_class + duration)
4. T-NORMALIZATION: Patterns normalized so first note = pitch-class 0
   - [0,4,7], [3,7,10], [7,11,2] all map to canonical [0,4,7] (major triad)
   - Dramatically reduces pattern count (12x compression for pitch-only)

GPU Optimizations:
- Batch pattern matching via tensor operations
- Parallel candidate scoring
- Memory-efficient sliding window on A100
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time


def t_normalize(pitch_classes: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    """
    T-normalize a pattern: transpose so first note = 0.

    Args:
        pitch_classes: Tuple of pitch classes (0-11)

    Returns:
        (canonical, t_offset) where:
        - canonical: pitch classes with first = 0
        - t_offset: transposition to apply to get original
    """
    if not pitch_classes:
        return tuple(), 0
    base = pitch_classes[0]
    canonical = tuple((pc - base) % 12 for pc in pitch_classes)
    return canonical, base


@dataclass
class LFRule:
    """A LONGESTFIRST grammar rule with T-normalization support."""
    rule_id: int
    pattern: List[int]  # Terminal sequence (pitch or pitch+duration encoded)
    count: int = 0      # Usage count
    score: float = 0.0  # MDL or frequency score

    # Duration info if available
    duration_pattern: Optional[List[int]] = None

    # T-normalization: canonical form (if pitch-only)
    canonical_pitches: Optional[Tuple[int, ...]] = None
    # Map: t_offset -> count of occurrences at that transposition
    transposition_counts: Optional[Dict[int, int]] = None

    def __len__(self):
        return len(self.pattern)

    def __repr__(self):
        pat_str = str(self.pattern[:8])
        if len(self.pattern) > 8:
            pat_str = pat_str[:-1] + ", ...]"
        if self.canonical_pitches:
            return f"R{self.rule_id}[len={len(self.pattern)}, count={self.count}, T-norm]: {pat_str}"
        return f"R{self.rule_id}[len={len(self.pattern)}, count={self.count}]: {pat_str}"


@dataclass
class LFGrammar:
    """Grammar produced by LONGESTFIRST algorithm."""
    rules: Dict[int, LFRule] = field(default_factory=dict)
    start_sequence: List[int] = field(default_factory=list)
    terminal_count: int = 0

    # Length statistics
    avg_rule_length: float = 0.0
    min_rule_length: int = 0
    max_rule_length: int = 0

    # T-normalization info
    t_normalized: bool = False
    canonical_to_rule_id: Dict[Tuple[int, ...], int] = field(default_factory=dict)

    def get_vocabulary_size(self) -> int:
        return len(self.rules)

    def get_rule_stats(self) -> Dict:
        stats = {}
        for rid, rule in self.rules.items():
            stats[str(rid)] = {
                'length': len(rule.pattern),
                'usage_count': rule.count,
                'expansion_length': len(rule.pattern),
                'depth': 1,
                'canonical_pitches': list(rule.canonical_pitches) if rule.canonical_pitches else None,
                'transposition_counts': dict(rule.transposition_counts) if rule.transposition_counts else None,
            }
        return stats

    def get_canonical_pattern(self, pattern: Tuple[int, ...]) -> Optional[int]:
        """Get rule_id for a canonical pattern, if it exists."""
        return self.canonical_to_rule_id.get(pattern)


class LongestFirstGPU:
    """
    GPU-accelerated LONGESTFIRST grammar induction with T-normalization.

    Algorithm:
    1. Find all repeated substrings of length >= min_length
    2. If t_normalize: group patterns that differ only by transposition
    3. Score by (length * frequency) - encoding cost
    4. Select highest-scoring pattern
    5. Replace all non-overlapping occurrences (longest match)
    6. Repeat until no pattern scores positive

    T-Normalization:
    - Patterns [0,4,7], [3,7,10], [7,11,2] all normalize to [0,4,7]
    - Dramatically reduces grammar size (up to 12x for pitch-only)
    - Stores transposition offset per occurrence for reconstruction

    GPU optimizations:
    - Suffix array construction on GPU (or CPU with numba)
    - Batch pattern counting via hash tables
    - Parallel MDL scoring
    """

    def __init__(
        self,
        device: str = 'cuda',
        min_length: int = 4,        # Minimum pattern length
        max_length: int = 64,       # Maximum pattern length
        min_frequency: int = 2,     # Minimum occurrences
        max_rules: int = 1000,      # Maximum number of rules
        include_duration: bool = True,  # Include duration in patterns
        t_normalize: bool = True,   # Apply T-normalization (normalize to first PC = 0)
        verbose: bool = False,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.min_length = min_length
        self.max_length = max_length
        self.min_frequency = min_frequency
        self.max_rules = max_rules
        self.include_duration = include_duration
        self.t_normalize = t_normalize
        self.verbose = verbose

        self.next_rule_id = 0
        self.max_terminal = 0

        # For T-normalized patterns: track canonical -> (rule_id, transposition_counts)
        self._canonical_patterns: Dict[Tuple[int, ...], Dict] = {}

    def induce(
        self,
        pitch_sequences: List[List[int]],
        duration_sequences: Optional[List[List[int]]] = None,
    ) -> LFGrammar:
        """
        Run LONGESTFIRST on pitch (and optionally duration) sequences.

        Args:
            pitch_sequences: List of pitch class sequences
            duration_sequences: Optional matching duration sequences

        Returns:
            LFGrammar with induced rules
        """
        start_time = time.time()

        # Combine pitch and duration if provided
        if self.include_duration and duration_sequences:
            sequences = self._combine_pitch_duration(pitch_sequences, duration_sequences)
        else:
            sequences = pitch_sequences

        # Find max terminal
        self.max_terminal = max(max(seq) for seq in sequences if seq) + 1
        self.next_rule_id = self.max_terminal

        # Concatenate with separators
        separator = -1
        combined = []
        seq_boundaries = [0]
        for seq in sequences:
            combined.extend(seq)
            combined.append(separator)
            seq_boundaries.append(len(combined))

        if self.verbose:
            print(f"[LONGESTFIRST] Starting with {len(sequences)} sequences, {len(combined)} tokens")
            print(f"[LONGESTFIRST] Pattern length range: {self.min_length}-{self.max_length}")

        grammar = LFGrammar()
        grammar.terminal_count = self.max_terminal

        # Convert to tensor for GPU operations
        seq_tensor = torch.tensor(combined, dtype=torch.int64, device=self.device)

        # Main loop: find and replace longest patterns
        iteration = 0
        rule_lengths = []

        while iteration < self.max_rules:
            # Find best pattern
            best_pattern, best_score, best_count = self._find_best_pattern(
                seq_tensor, separator
            )

            if best_pattern is None or best_score <= 0:
                break

            if len(best_pattern) < self.min_length:
                break

            # Create rule
            rule_id = self.next_rule_id
            self.next_rule_id += 1

            # Get transposition info if T-normalized
            trans_counts = None
            canonical_pcs = None
            if self.t_normalize and not self.include_duration:
                canonical_pcs = tuple(best_pattern)
                trans_counts = dict(self._transposition_info.get(canonical_pcs, {}))
                grammar.canonical_to_rule_id[canonical_pcs] = rule_id

            rule = LFRule(
                rule_id=rule_id,
                pattern=best_pattern,
                count=best_count,
                score=best_score,
                canonical_pitches=canonical_pcs,
                transposition_counts=trans_counts,
            )
            grammar.rules[rule_id] = rule
            rule_lengths.append(len(best_pattern))

            if self.verbose and iteration % 50 == 0:
                trans_info = ""
                if trans_counts:
                    trans_info = f", T-variants={len(trans_counts)}"
                print(f"[LONGESTFIRST] Iter {iteration}: R{rule_id} len={len(best_pattern)}, "
                      f"count={best_count}, score={best_score:.2f}{trans_info}")

            # Replace pattern with rule ID
            seq_tensor = self._replace_pattern_gpu(
                seq_tensor, best_pattern, rule_id, separator
            )

            iteration += 1

        # Extract final sequence
        final_seq = seq_tensor[seq_tensor != separator].cpu().tolist()
        grammar.start_sequence = final_seq

        # Set T-normalization flag
        grammar.t_normalized = self.t_normalize and not self.include_duration

        # Compute statistics
        if rule_lengths:
            grammar.avg_rule_length = np.mean(rule_lengths)
            grammar.min_rule_length = min(rule_lengths)
            grammar.max_rule_length = max(rule_lengths)

        elapsed = time.time() - start_time

        if self.verbose:
            tnorm_str = " (T-normalized)" if grammar.t_normalized else ""
            print(f"[LONGESTFIRST] Complete: {len(grammar.rules)} rules{tnorm_str}")
            print(f"[LONGESTFIRST] Avg rule length: {grammar.avg_rule_length:.1f}")
            if grammar.t_normalized:
                # Count total unique transpositions
                total_variants = sum(
                    len(r.transposition_counts) for r in grammar.rules.values()
                    if r.transposition_counts
                )
                print(f"[LONGESTFIRST] T-variants: {total_variants} across {len(grammar.rules)} canonical patterns")
            print(f"[LONGESTFIRST] Time: {elapsed:.2f}s")

        return grammar

    def _combine_pitch_duration(
        self,
        pitch_seqs: List[List[int]],
        duration_seqs: List[List[int]],
    ) -> List[List[int]]:
        """
        Combine pitch and duration into single tokens.

        Encoding: token = pitch * N_DURATIONS + duration_bucket
        """
        N_PITCHES = 12
        N_DURATIONS = 8  # Quantized duration buckets

        combined = []
        for pitches, durations in zip(pitch_seqs, duration_seqs):
            seq = []
            for p, d in zip(pitches, durations):
                # Quantize duration to bucket
                d_bucket = min(d // 120, N_DURATIONS - 1)  # 120 ticks per bucket
                token = p * N_DURATIONS + d_bucket
                seq.append(token)
            combined.append(seq)

        return combined

    def _find_best_pattern(
        self,
        seq: torch.Tensor,
        separator: int,
    ) -> Tuple[Optional[List[int]], float, int]:
        """
        Find the best (highest-scoring) repeating pattern.

        Uses sliding window + hash-based counting for efficiency.
        """
        seq_cpu = seq.cpu().numpy()
        n = len(seq_cpu)

        # Track best pattern
        best_pattern = None
        best_score = 0
        best_count = 0

        # Use hash-based counting for different lengths
        # Start from longest and work down (LONGESTFIRST principle)
        for length in range(min(self.max_length, n // 2), self.min_length - 1, -1):
            patterns = self._count_patterns_of_length(seq_cpu, length, separator)

            for pattern, count in patterns.items():
                if count < self.min_frequency:
                    continue

                # MDL-style scoring: benefit - cost
                # Benefit: length * count (characters saved)
                # Cost: length + log(count) (encoding the rule)
                benefit = length * count
                cost = length + np.log2(max(count, 1)) * 2
                score = benefit - cost

                if score > best_score:
                    best_score = score
                    best_pattern = list(pattern)
                    best_count = count

            # Early termination: if we found a good long pattern, use it
            # (LONGESTFIRST principle)
            if best_pattern is not None and len(best_pattern) >= length:
                break

        return best_pattern, best_score, best_count

    def _count_patterns_of_length(
        self,
        seq: np.ndarray,
        length: int,
        separator: int,
    ) -> Dict[Tuple[int, ...], int]:
        """
        Count all patterns of given length.

        If t_normalize is enabled, patterns are normalized so first element = 0,
        and all transpositions are counted as the same canonical pattern.

        GPU-optimized using integer hashing on GPU, then bincount for counting.
        """
        counts = defaultdict(int)
        transposition_info = defaultdict(lambda: defaultdict(int))
        n = len(seq)

        if n < length:
            return counts

        # Convert to tensor for GPU operations
        seq_t = torch.tensor(seq, dtype=torch.int64, device=self.device)

        # Create sliding windows using unfold (all on GPU)
        windows = seq_t.unfold(0, length, 1)
        n_windows = windows.shape[0]

        # Check for separator in windows
        has_separator = (windows == separator).any(dim=1)
        valid_mask = ~has_separator

        if self.t_normalize and not self.include_duration:
            # GPU-accelerated T-normalization
            bases = windows[:, 0]
            normalized = (windows - bases.unsqueeze(1)) % 12

            # GPU HASH: Convert pattern to single integer for fast comparison
            # Each position is 0-11, so base-12 encoding works
            # hash = p[0]*12^(n-1) + p[1]*12^(n-2) + ... + p[n-1]
            powers = torch.pow(12, torch.arange(length - 1, -1, -1, device=self.device, dtype=torch.int64))
            pattern_hashes = (normalized * powers).sum(dim=1)

            # Apply mask - set invalid hashes to -1
            pattern_hashes = torch.where(valid_mask, pattern_hashes, torch.tensor(-1, device=self.device))

            # GPU BINCOUNT: Count occurrences of each hash
            valid_hashes = pattern_hashes[valid_mask]
            if len(valid_hashes) == 0:
                self._transposition_info = {}
                return counts

            # Get unique hashes and their counts using GPU
            unique_hashes, inverse, hash_counts = torch.unique(
                valid_hashes, return_inverse=True, return_counts=True
            )

            # Now convert only the UNIQUE patterns back to tuples (much smaller!)
            unique_hashes_cpu = unique_hashes.cpu().numpy()
            hash_counts_cpu = hash_counts.cpu().numpy()

            # Decode hashes back to patterns
            for i, (h, c) in enumerate(zip(unique_hashes_cpu, hash_counts_cpu)):
                if c < self.min_frequency:
                    continue  # Skip infrequent patterns early
                # Decode hash to pattern
                pattern = []
                val = h
                for _ in range(length):
                    pattern.insert(0, int(val % 12))
                    val //= 12
                canonical = tuple(pattern)
                counts[canonical] = int(c)

            # For transposition info, we need to be smarter
            # Group by hash and collect base notes
            if len(counts) > 0:
                # Only compute transposition info for patterns that meet frequency threshold
                normalized_cpu = normalized[valid_mask].cpu().numpy()
                bases_cpu = bases[valid_mask].cpu().numpy()
                inverse_cpu = inverse.cpu().numpy()

                # Build transposition info for qualifying patterns only
                qualifying_indices = set()
                for i, (h, c) in enumerate(zip(unique_hashes_cpu, hash_counts_cpu)):
                    if c >= self.min_frequency:
                        qualifying_indices.add(i)

                for i in range(len(inverse_cpu)):
                    hash_idx = inverse_cpu[i]
                    if hash_idx in qualifying_indices:
                        canonical = tuple(normalized_cpu[i].tolist())
                        t_offset = int(bases_cpu[i])
                        transposition_info[canonical][t_offset] += 1

        else:
            # Non-normalized counting - similar GPU hash approach
            # For non-T-norm, values can be larger, need bigger hash space
            max_val = seq.max() + 1
            powers = torch.pow(max_val, torch.arange(length - 1, -1, -1, device=self.device, dtype=torch.int64))

            # Check for overflow
            if length > 8 or max_val > 100:
                # Fall back to CPU for large patterns
                windows_cpu = windows[valid_mask].cpu().numpy()
                for i in range(len(windows_cpu)):
                    pattern = tuple(windows_cpu[i].tolist())
                    counts[pattern] += 1
            else:
                pattern_hashes = (windows * powers).sum(dim=1)
                valid_hashes = pattern_hashes[valid_mask]

                if len(valid_hashes) > 0:
                    unique_hashes, hash_counts = torch.unique(valid_hashes, return_counts=True)
                    unique_hashes_cpu = unique_hashes.cpu().numpy()
                    hash_counts_cpu = hash_counts.cpu().numpy()
                    windows_valid_cpu = windows[valid_mask].cpu().numpy()

                    # Find one example of each unique hash
                    seen_hashes = {}
                    for i, h in enumerate(pattern_hashes[valid_mask].cpu().numpy()):
                        if h not in seen_hashes:
                            seen_hashes[h] = tuple(windows_valid_cpu[i].tolist())

                    for h, c in zip(unique_hashes_cpu, hash_counts_cpu):
                        if c >= self.min_frequency and h in seen_hashes:
                            counts[seen_hashes[h]] = int(c)

        self._transposition_info = dict(transposition_info)
        return counts

    def _replace_pattern_gpu(
        self,
        seq: torch.Tensor,
        pattern: List[int],
        new_symbol: int,
        separator: int,
    ) -> torch.Tensor:
        """
        Replace all non-overlapping occurrences of pattern with new_symbol.

        If T-normalization is enabled, matches ALL transpositions of the
        canonical pattern (pattern is already in canonical form with first = 0).

        GPU-optimized using vectorized operations for pattern matching.
        """
        pattern_len = len(pattern)
        n = len(seq)
        if n < pattern_len:
            return seq

        # Create pattern tensor on GPU
        pattern_t = torch.tensor(pattern, dtype=seq.dtype, device=self.device)

        # Create sliding windows using unfold (all on GPU)
        # Shape: (n_windows, pattern_len)
        windows = seq.unfold(0, pattern_len, 1)
        n_windows = windows.shape[0]

        if self.t_normalize and not self.include_duration:
            # GPU-accelerated T-normalized matching
            # For each window, compute: (window - window[0]) % 12
            # Then compare with canonical pattern

            # Get first element of each window (the transposition base)
            bases = windows[:, 0]  # Shape: (n_windows,)

            # Normalize all windows in parallel: (window - base) % 12
            # Shape: (n_windows, pattern_len)
            normalized = (windows - bases.unsqueeze(1)) % 12

            # Compare normalized windows to canonical pattern
            # Shape: (n_windows, pattern_len) -> (n_windows,)
            matches = (normalized == pattern_t).all(dim=1)
        else:
            # Exact matching (all on GPU)
            matches = (windows == pattern_t).all(dim=1)

        # Check for separators - mark windows containing separator as non-matching
        has_separator = (windows == separator).any(dim=1)
        matches = matches & ~has_separator

        # Get match positions
        match_positions = torch.nonzero(matches, as_tuple=True)[0]

        if len(match_positions) == 0:
            return seq

        # GPU-accelerated non-overlapping filter
        # Key insight: a match at position i is valid if no matches exist in [i-pattern_len+1, i-1]
        # We can compute this using a sliding window check

        n_matches = len(match_positions)

        if n_matches <= 1:
            non_overlapping = match_positions
        else:
            # Compute gaps between consecutive matches
            # If gap >= pattern_len, both can be kept
            gaps = match_positions[1:] - match_positions[:-1]

            # A match is "blocked" if the previous kept match overlaps it
            # Use a greedy left-to-right strategy on GPU:
            # Keep match i if gap from last kept match >= pattern_len

            # Fast path: use vectorized approach for most cases
            # Mark positions where gap < pattern_len (potential overlap)
            overlap_with_prev = gaps < pattern_len

            # Build keep mask using cumulative logic on GPU
            # Start by keeping first match, then keep subsequent if no overlap
            keep_mask = torch.ones(n_matches, dtype=torch.bool, device=self.device)

            # For positions with potential overlap, we need sequential logic
            # But we can batch-process stretches of non-overlapping matches
            overlap_indices = torch.nonzero(overlap_with_prev, as_tuple=True)[0]

            if len(overlap_indices) > 0:
                # Only need CPU for overlapping regions (usually much smaller)
                keep_mask_cpu = keep_mask.cpu().numpy()
                match_pos_cpu = match_positions.cpu().numpy()
                overlap_idx_cpu = overlap_indices.cpu().numpy()

                # Only iterate through overlapping regions
                last_kept_end = match_pos_cpu[0] + pattern_len
                for i in overlap_idx_cpu:
                    idx = int(i) + 1  # overlap_with_prev[i] means gap between i and i+1 is small
                    if match_pos_cpu[idx] < last_kept_end:
                        keep_mask_cpu[idx] = False
                    else:
                        last_kept_end = match_pos_cpu[idx] + pattern_len

                keep_mask = torch.tensor(keep_mask_cpu, dtype=torch.bool, device=self.device)

            non_overlapping = match_positions[keep_mask]

        if len(non_overlapping) == 0:
            return seq

        # Build replacement mask (GPU)
        seq_keep_mask = torch.ones(n, dtype=torch.bool, device=self.device)
        new_seq = seq.clone()

        # non_overlapping is already a tensor on GPU
        # Set new symbol at match positions
        new_seq[non_overlapping] = new_symbol

        # Create indices for positions to remove (all but first in each match)
        # For each match at pos, remove positions pos+1, pos+2, ..., pos+pattern_len-1
        remove_offsets = torch.arange(1, pattern_len, device=self.device)
        # Shape: (n_matches, pattern_len-1)
        remove_indices = non_overlapping.unsqueeze(1) + remove_offsets.unsqueeze(0)
        # Flatten and filter valid indices
        remove_indices = remove_indices.flatten()
        remove_indices = remove_indices[remove_indices < n]

        seq_keep_mask[remove_indices] = False

        return new_seq[seq_keep_mask]


def build_longestfirst_grammar(
    pitch_sequences: List[List[int]],
    duration_sequences: Optional[List[List[int]]] = None,
    device: str = 'cuda',
    min_length: int = 4,
    max_length: int = 64,
    min_frequency: int = 2,
    max_rules: int = 1000,
    include_duration: bool = True,
    t_normalize: bool = True,
    verbose: bool = False,
) -> LFGrammar:
    """
    Build grammar using LONGESTFIRST algorithm with T-normalization.

    Args:
        pitch_sequences: List of pitch class sequences
        duration_sequences: Optional duration sequences (for duration-aware matching)
        device: 'cuda' or 'cpu'
        min_length: Minimum pattern length (default 4)
        max_length: Maximum pattern length (default 64)
        min_frequency: Minimum occurrences (default 2)
        max_rules: Maximum rules to create (default 1000)
        include_duration: Whether to include duration in patterns
        t_normalize: Apply T-normalization (patterns with first PC = 0, default True)
        verbose: Print progress

    Returns:
        LFGrammar with T-normalized patterns
    """
    lf = LongestFirstGPU(
        device=device,
        min_length=min_length,
        max_length=max_length,
        min_frequency=min_frequency,
        max_rules=max_rules,
        include_duration=include_duration,
        t_normalize=t_normalize,
        verbose=verbose,
    )

    return lf.induce(pitch_sequences, duration_sequences)


def build_longestfirst_from_corpus(
    factored_objects: List,
    device: str = 'cuda',
    min_length: int = 4,
    max_length: int = 64,
    min_frequency: int = 2,
    max_rules: int = 1000,
    include_duration: bool = True,
    t_normalize: bool = True,
    verbose: bool = False,
) -> LFGrammar:
    """
    Build LONGESTFIRST grammar from factored MIDI objects with T-normalization.

    Args:
        factored_objects: List of factored objects with pitch_class and duration arrays
        t_normalize: Apply T-normalization (default True)

    Returns:
        LFGrammar with T-normalized patterns
    """
    pitch_sequences = []
    duration_sequences = []

    for obj in factored_objects:
        if hasattr(obj, 'pitch_class') and len(obj.pitch_class) > 0:
            pc = obj.pitch_class
            if hasattr(pc, 'tolist'):
                pc = pc.tolist()
            pitch_sequences.append(pc)

            # Get duration if available
            if hasattr(obj, 'duration') and len(obj.duration) > 0:
                dur = obj.duration
                if hasattr(dur, 'tolist'):
                    dur = dur.tolist()
                duration_sequences.append(dur)
            else:
                duration_sequences.append([120] * len(pc))  # Default duration

    if verbose:
        print(f"[LONGESTFIRST] Extracted {len(pitch_sequences)} sequences")

    return build_longestfirst_grammar(
        pitch_sequences,
        duration_sequences if include_duration else None,
        device=device,
        min_length=min_length,
        max_length=max_length,
        min_frequency=min_frequency,
        max_rules=max_rules,
        include_duration=include_duration,
        t_normalize=t_normalize,
        verbose=verbose,
    )


if __name__ == '__main__':
    print("Testing LONGESTFIRST GPU implementation with T-normalization...")

    # Test T-normalization: same pattern in different keys should be ONE pattern
    print("\n=== Test 1: T-Normalization ===")
    print("C major triad [0,4,7], Eb major [3,7,10], G major [7,11,2] should all be ONE pattern")

    test_t_norm = [
        [0, 4, 7, 0, 4, 7, 0, 4, 7],      # C major triad repeated
        [3, 7, 10, 3, 7, 10, 3, 7, 10],   # Eb major triad repeated (T3)
        [7, 11, 2, 7, 11, 2, 7, 11, 2],   # G major triad repeated (T7)
        [5, 9, 0, 5, 9, 0, 5, 9, 0],      # F major triad repeated (T5)
    ]

    # Test WITHOUT T-normalization
    grammar_no_tnorm = build_longestfirst_grammar(
        test_t_norm,
        None,  # No duration
        device='cuda',
        min_length=3,
        include_duration=False,
        t_normalize=False,
        verbose=True,
    )
    print(f"\nWithout T-norm: {len(grammar_no_tnorm.rules)} rules (should be ~4)")

    # Test WITH T-normalization
    grammar_tnorm = build_longestfirst_grammar(
        test_t_norm,
        None,  # No duration
        device='cuda',
        min_length=3,
        include_duration=False,
        t_normalize=True,
        verbose=True,
    )
    print(f"\nWith T-norm: {len(grammar_tnorm.rules)} rules (should be ~1)")

    if grammar_tnorm.rules:
        for rid, rule in grammar_tnorm.rules.items():
            print(f"  {rule}")
            if rule.transposition_counts:
                print(f"    Transpositions: {rule.transposition_counts}")

    # Test with musical-like patterns (original test)
    print("\n=== Test 2: Full Musical Example ===")
    test_pitch = [
        [0, 2, 4, 5, 7, 0, 2, 4, 5, 7, 0, 2, 4, 5, 7],  # C major scale pattern
        [0, 4, 7, 0, 4, 7, 0, 4, 7],  # C major arpeggio
        [0, 2, 4, 5, 7, 9, 11, 0, 2, 4, 5, 7, 9, 11],  # Full scale
    ]

    test_duration = [
        [120, 120, 120, 120, 240, 120, 120, 120, 120, 240, 120, 120, 120, 120, 240],
        [240, 240, 480, 240, 240, 480, 240, 240, 480],
        [120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120],
    ]

    grammar = build_longestfirst_grammar(
        test_pitch,
        test_duration,
        device='cuda',
        min_length=3,
        include_duration=True,  # Duration included, so T-norm won't apply
        t_normalize=True,
        verbose=True,
    )

    print(f"\nGrammar rules:")
    for rid, rule in grammar.rules.items():
        print(f"  {rule}")

    print(f"\nStatistics:")
    print(f"  Avg rule length: {grammar.avg_rule_length:.1f}")
    print(f"  Min/Max length: {grammar.min_rule_length}/{grammar.max_rule_length}")
    print(f"  T-normalized: {grammar.t_normalized}")
