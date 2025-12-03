"""
T+τ+v Factored Hierarchical GPU Re-Pair Implementation
=======================================================

Extends T-norm Re-Pair to normalize by all three factors:
- T (Transposition): Pitch interval between consecutive notes
- τ (Tempo): Rhythm ratio between consecutive IOIs
- v (Velocity): Velocity ratio between consecutive notes

Each pair is characterized by its normalized triple: (pitch_interval, rhythm_bucket, velocity_bucket)
All transpositions, tempo scalings, and dynamic scalings are treated as the same pattern.

Example of factored hierarchical patterns:
- R12 = (interval=3, rhythm=1.0, velocity=same) - minor 3rd at same tempo
- R13 = (interval=4, rhythm=0.5, velocity=louder) - major 3rd accelerating and crescendo
- R14 = (R12, R13) - compound pattern matching ANY transposition/tempo/velocity

This enables discovering:
- Melodic motifs at any transposition AND any tempo
- Rhythmic patterns at any tempo AND any pitch
- Dynamic shapes at any velocity level
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import time


# Rhythm bucket constants (16 buckets for τ-normalization)
RHYTHM_BUCKETS = 16
VELOCITY_BUCKETS = 8


def compute_rhythm_bucket(ioi1: int, ioi2: int) -> int:
    """
    Compute rhythm ratio bucket for τ-normalization.

    Returns bucket 0-15 based on IOI ratio:
    - Buckets 0-7: Subdivision ratios (0.125x to 1.0x)
    - Buckets 8-15: Multiple ratios (1.0x to 8.0x+)
    """
    if ioi1 <= 0:
        return 7  # Default to 1.0 ratio

    ratio = ioi2 / ioi1

    if ratio < 1.0:
        # Subdivisions: 0.125, 0.25, 0.33, 0.5, 0.67, 0.75, 0.875, 1.0
        thresholds = [0.1875, 0.29, 0.42, 0.585, 0.71, 0.8125, 0.9375]
        for i, thresh in enumerate(thresholds):
            if ratio < thresh:
                return i
        return 7  # ~1.0
    else:
        # Multiples: 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 8.0+
        thresholds = [1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 8.5]
        for i, thresh in enumerate(thresholds):
            if ratio < thresh:
                return 8 + i
        return 15  # 8.0+


def compute_velocity_bucket(v1: int, v2: int) -> int:
    """
    Compute velocity ratio bucket for v-normalization.

    Returns bucket 0-7 based on velocity difference:
    - 0-2: Getting softer (large, medium, small decrease)
    - 3-4: Same (within threshold)
    - 5-7: Getting louder (small, medium, large increase)
    """
    diff = v2 - v1  # Positive means louder

    if diff < -30:
        return 0  # Much softer
    elif diff < -15:
        return 1  # Moderately softer
    elif diff < -5:
        return 2  # Slightly softer
    elif diff <= 5:
        return 3  # Same (centered)
    elif diff <= 15:
        return 5  # Slightly louder
    elif diff <= 30:
        return 6  # Moderately louder
    else:
        return 7  # Much louder


@dataclass
class FactoredHierarchicalGrammar:
    """Grammar produced by T+τ+v factored hierarchical Re-Pair."""

    # Rule table: rule_id -> (left, right)
    rule_table: torch.Tensor  # Shape: [n_rules, 2]
    rule_counts: torch.Tensor  # Shape: [n_rules]

    # Normalized factors for each rule
    rule_pitch_intervals: torch.Tensor  # Shape: [n_rules] - pitch interval (0-11)
    rule_rhythm_buckets: torch.Tensor   # Shape: [n_rules] - rhythm bucket (0-15)
    rule_velocity_buckets: torch.Tensor # Shape: [n_rules] - velocity bucket (0-7)

    # Final compressed sequence (with track boundaries)
    final_sequence: torch.Tensor
    track_boundaries: List[int]  # Indices where tracks end

    # Metadata
    n_terminals: int
    n_rules: int
    original_length: int
    compressed_length: int
    device: str

    # Track info for piece mapping
    track_info: List[Dict]  # piece_id, track_id per track

    # Occurrence data captured DURING compression (key fix for timing accuracy)
    # Dict: rule_id -> list of {track_idx, orig_pos, ioi, duration, velocity, onset}
    rule_occurrences: Dict[int, List[Dict]] = None

    def compression_ratio(self) -> float:
        if self.compressed_length == 0:
            return 1.0
        return self.original_length / self.compressed_length

    def get_rule(self, rule_id: int) -> Tuple[int, int]:
        """Get left/right symbols for a rule."""
        idx = rule_id - self.n_terminals
        if idx < 0 or idx >= len(self.rule_table):
            return (rule_id, -1)
        row = self.rule_table[idx]
        return (row[0].item(), row[1].item())

    def expand_rule(self, rule_id: int, memo: Optional[Dict] = None) -> List[int]:
        """Recursively expand rule to terminals."""
        if memo is None:
            memo = {}
        if rule_id in memo:
            return memo[rule_id]

        if rule_id < self.n_terminals:
            return [rule_id]

        left, right = self.get_rule(rule_id)
        if right == -1:
            return [left % self.n_terminals]

        result = self.expand_rule(left, memo) + self.expand_rule(right, memo)
        memo[rule_id] = result
        return result


class RePairFactoredHierarchical:
    """
    T+τ+v Factored Hierarchical GPU Re-Pair.

    Normalizes pairs by:
    1. Pitch interval (0-11) - T-normalization
    2. Rhythm ratio bucket (0-15) - τ-normalization
    3. Velocity ratio bucket (0-7) - v-normalization

    Pairs are counted and matched by their normalized triple.
    """

    def __init__(
        self,
        device: str = 'cuda',
        min_pair_count: int = 2,
        max_rules: int = 10000,
        pitch_range: int = 12,
        rhythm_buckets: int = RHYTHM_BUCKETS,
        velocity_buckets: int = VELOCITY_BUCKETS,
        verbose: bool = True,
        progress_every: int = 100,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.min_pair_count = min_pair_count
        self.max_rules = max_rules
        self.pitch_range = pitch_range
        self.rhythm_buckets = rhythm_buckets
        self.velocity_buckets = velocity_buckets
        self.verbose = verbose
        self.progress_every = progress_every

    def induce(
        self,
        pitch_sequences: List[List[int]],
        ioi_sequences: List[List[int]],
        velocity_sequences: List[List[int]],
        track_info: List[Dict] = None,
        duration_sequences: List[List[int]] = None,
        onset_sequences: List[List[int]] = None,
        octave_sequences: List[List[int]] = None,
    ) -> FactoredHierarchicalGrammar:
        """
        Run T+τ+v factored hierarchical Re-Pair.

        Args:
            pitch_sequences: List of pitch class sequences (0-11)
            ioi_sequences: List of IOI sequences (ticks)
            velocity_sequences: List of velocity sequences (0-127)
            track_info: Optional list of {piece_id, track_id} per track
            duration_sequences: Optional list of note duration sequences
            onset_sequences: Optional list of onset time sequences
            octave_sequences: Optional list of octave sequences (0-9) for pitch reconstruction

        Returns:
            FactoredHierarchicalGrammar
        """
        start_time = time.time()

        n_terminals = self.pitch_range  # 12 for pitch classes
        separator = -1

        # Concatenate sequences with separators
        combined_pitch = []
        combined_ioi = []
        combined_velocity = []
        combined_track_idx = []  # Track which track each element belongs to
        combined_orig_pos = []   # Original position within the track
        track_boundaries = []

        # Also store raw duration, onset, and octave for timing/pitch extraction
        combined_duration = []
        combined_onset = []
        combined_octave = []

        for i, (pitch_seq, ioi_seq, vel_seq) in enumerate(
            zip(pitch_sequences, ioi_sequences, velocity_sequences)
        ):
            seq_len = len(pitch_seq)

            # Get durations, onsets, and octaves if provided
            dur_seq = duration_sequences[i] if duration_sequences else [480] * seq_len
            onset_seq = onset_sequences[i] if onset_sequences else [j * 480 for j in range(seq_len)]
            oct_seq = octave_sequences[i] if octave_sequences else [4] * seq_len  # Default to octave 4

            combined_pitch.extend(pitch_seq)
            combined_ioi.extend(ioi_seq if len(ioi_seq) == len(pitch_seq) else ioi_seq + [480])
            combined_velocity.extend(vel_seq)
            combined_duration.extend(dur_seq if len(dur_seq) == seq_len else dur_seq + [480] * (seq_len - len(dur_seq)))
            combined_onset.extend(onset_seq if len(onset_seq) == seq_len else onset_seq + [0] * (seq_len - len(onset_seq)))
            combined_octave.extend(oct_seq if len(oct_seq) == seq_len else oct_seq + [4] * (seq_len - len(oct_seq)))
            # Use the actual track index from track_info, not the loop index
            # This preserves the original MIDI track numbering for instrument mapping
            combined_track_idx.extend([i] * seq_len)  # i = index into our track list (for piece mapping)
            combined_orig_pos.extend(range(seq_len))

            # Add separator
            combined_pitch.append(separator)
            combined_ioi.append(0)
            combined_velocity.append(0)
            combined_duration.append(0)
            combined_onset.append(0)
            combined_octave.append(0)
            combined_track_idx.append(-1)
            combined_orig_pos.append(-1)
            track_boundaries.append(len(combined_pitch) - 1)

        original_length = len(combined_pitch)

        # Move to GPU
        seq_pitch = torch.tensor(combined_pitch, dtype=torch.int64, device=self.device)
        seq_ioi = torch.tensor(combined_ioi, dtype=torch.int64, device=self.device)
        seq_velocity = torch.tensor(combined_velocity, dtype=torch.int64, device=self.device)
        seq_duration = torch.tensor(combined_duration, dtype=torch.int64, device=self.device)
        seq_onset = torch.tensor(combined_onset, dtype=torch.int64, device=self.device)
        seq_octave = torch.tensor(combined_octave, dtype=torch.int64, device=self.device)
        seq_track_idx = torch.tensor(combined_track_idx, dtype=torch.int64, device=self.device)
        seq_orig_pos = torch.tensor(combined_orig_pos, dtype=torch.int64, device=self.device)

        # Dictionary to collect occurrences during compression
        rule_occurrences: Dict[int, List[Dict]] = {}

        if self.verbose:
            print(f"[RePair-Factored] {len(pitch_sequences)} seqs, {original_length:,} tokens", flush=True)

        # Pre-allocate rule tables
        rule_table = torch.zeros((self.max_rules, 2), dtype=torch.int64, device=self.device)
        rule_counts = torch.zeros(self.max_rules, dtype=torch.int64, device=self.device)
        rule_pitch_intervals = torch.full((self.max_rules,), -1, dtype=torch.int64, device=self.device)
        rule_rhythm_buckets = torch.full((self.max_rules,), -1, dtype=torch.int64, device=self.device)
        rule_velocity_buckets = torch.full((self.max_rules,), -1, dtype=torch.int64, device=self.device)

        # Symbol base lookups (for T-normalization of compound patterns)
        max_symbols = n_terminals + self.max_rules
        symbol_base_pitch = torch.zeros(max_symbols, dtype=torch.int64, device=self.device)
        symbol_base_pitch[:n_terminals] = torch.arange(n_terminals, device=self.device)

        # Also track base IOI and velocity for compound patterns
        symbol_base_ioi = torch.full((max_symbols,), 480, dtype=torch.int64, device=self.device)
        symbol_base_velocity = torch.full((max_symbols,), 64, dtype=torch.int64, device=self.device)

        next_rule_id = n_terminals
        n_rules = 0

        # Main Re-Pair loop
        iteration = 0
        while iteration < self.max_rules:
            # ===== STEP 1: Find most frequent normalized triple =====

            # Get consecutive pairs
            left_pitch = seq_pitch[:-1]
            right_pitch = seq_pitch[1:]
            left_ioi = seq_ioi[:-1]
            right_ioi = seq_ioi[1:]
            left_vel = seq_velocity[:-1]
            right_vel = seq_velocity[1:]

            # Mask out pairs with separator
            valid = (left_pitch != separator) & (right_pitch != separator)

            if not valid.any():
                break

            left_p = left_pitch[valid]
            right_p = right_pitch[valid]
            left_i = left_ioi[valid]
            right_i = right_ioi[valid]
            left_v = left_vel[valid]
            right_v = right_vel[valid]

            # Get base pitches for T-normalization
            left_base_p = symbol_base_pitch[left_p]
            right_base_p = symbol_base_pitch[right_p]

            # Compute normalized pitch interval
            pitch_intervals = (right_base_p - left_base_p) % n_terminals

            # Compute rhythm buckets (τ-normalization) - CPU for now
            left_base_ioi = symbol_base_ioi[left_p]
            right_base_ioi = symbol_base_ioi[right_p]
            rhythm_ratios = right_base_ioi.float() / (left_base_ioi.float() + 1e-6)
            rhythm_buckets_tensor = torch.zeros_like(left_p)

            # Vectorized rhythm bucket computation
            rhythm_buckets_tensor = self._compute_rhythm_buckets_vectorized(
                left_base_ioi, right_base_ioi
            )

            # Compute velocity buckets (v-normalization)
            left_base_v = symbol_base_velocity[left_p]
            right_base_v = symbol_base_velocity[right_p]
            velocity_buckets_tensor = self._compute_velocity_buckets_vectorized(
                left_base_v, right_base_v
            )

            # Skip unison pitch intervals (interval=0)
            non_unison = pitch_intervals != 0
            if non_unison.any():
                pitch_intervals_f = pitch_intervals[non_unison]
                rhythm_buckets_f = rhythm_buckets_tensor[non_unison]
                velocity_buckets_f = velocity_buckets_tensor[non_unison]
                left_p_f = left_p[non_unison]
                right_p_f = right_p[non_unison]
            else:
                pitch_intervals_f = pitch_intervals
                rhythm_buckets_f = rhythm_buckets_tensor
                velocity_buckets_f = velocity_buckets_tensor
                left_p_f = left_p
                right_p_f = right_p

            if len(pitch_intervals_f) == 0:
                break

            # Encode normalized triple as single value for counting
            # triple_id = pitch * 16 * 8 + rhythm * 8 + velocity
            triple_ids = (pitch_intervals_f * self.rhythm_buckets * self.velocity_buckets +
                         rhythm_buckets_f * self.velocity_buckets +
                         velocity_buckets_f)

            # Find most common triple
            unique_triples, triple_counts = torch.unique(triple_ids, return_counts=True)
            max_triple_idx = triple_counts.argmax()
            max_triple_count = triple_counts[max_triple_idx].item()
            best_triple = unique_triples[max_triple_idx].item()

            if max_triple_count < self.min_pair_count:
                break

            # Decode best triple
            best_pitch_interval = best_triple // (self.rhythm_buckets * self.velocity_buckets)
            remainder = best_triple % (self.rhythm_buckets * self.velocity_buckets)
            best_rhythm_bucket = remainder // self.velocity_buckets
            best_velocity_bucket = remainder % self.velocity_buckets

            # Find most common actual pair with this normalized triple
            triple_mask = triple_ids == best_triple

            # Encode actual pairs
            max_vocab = next_rule_id + self.max_rules
            pair_ids = left_p_f[triple_mask] * max_vocab + right_p_f[triple_mask]

            unique_pairs, pair_counts = torch.unique(pair_ids, return_counts=True)
            max_pair_idx = pair_counts.argmax()
            best_pair_id = unique_pairs[max_pair_idx].item()
            actual_count = pair_counts[max_pair_idx].item()

            best_left = best_pair_id // max_vocab
            best_right = best_pair_id % max_vocab

            # ===== STEP 2: Create new rule =====
            rule_id = next_rule_id
            next_rule_id += 1

            # Store rule
            rule_table[n_rules, 0] = best_left
            rule_table[n_rules, 1] = best_right
            rule_counts[n_rules] = actual_count
            rule_pitch_intervals[n_rules] = best_pitch_interval
            rule_rhythm_buckets[n_rules] = best_rhythm_bucket
            rule_velocity_buckets[n_rules] = best_velocity_bucket

            # Compute bases for new rule
            symbol_base_pitch[rule_id] = symbol_base_pitch[best_left]
            symbol_base_ioi[rule_id] = symbol_base_ioi[best_left]
            symbol_base_velocity[rule_id] = symbol_base_velocity[best_left]

            n_rules += 1

            # ===== STEP 3: Replace all occurrences and capture timing data =====
            (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
             seq_track_idx, seq_orig_pos, occurrences) = self._replace_pair_gpu_with_tracking(
                seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                seq_track_idx, seq_orig_pos,
                best_left, best_right, rule_id, separator
            )

            # Store occurrences for this rule
            rule_occurrences[rule_id] = occurrences

            # Progress
            if self.verbose and (iteration % self.progress_every == 0 or iteration < 10):
                elapsed = time.time() - start_time
                interval_name = self._interval_name(best_pitch_interval)
                is_hierarchical = best_left >= n_terminals or best_right >= n_terminals
                hier_marker = " [HIER]" if is_hierarchical else ""
                print(f"  [iter {iteration}] R{rule_id}=({best_left},{best_right}) "
                      f"interval={best_pitch_interval}({interval_name}) "
                      f"rhythm={best_rhythm_bucket} vel={best_velocity_bucket}{hier_marker} "
                      f"count={actual_count:,}, len={len(seq_pitch):,}, time={elapsed:.1f}s", flush=True)

            iteration += 1

        # Remove separators from final sequence
        mask = seq_pitch != separator
        final_seq = seq_pitch[mask]

        elapsed = time.time() - start_time

        if self.verbose:
            ratio = original_length / len(final_seq) if len(final_seq) > 0 else 1.0
            hier_count = sum(
                1 for i in range(n_rules)
                if rule_table[i, 0].item() >= n_terminals or rule_table[i, 1].item() >= n_terminals
            )
            print(f"[RePair-Factored] Done: {n_rules} rules ({hier_count} hierarchical), "
                  f"compression {ratio:.2f}x, time {elapsed:.1f}s", flush=True)

        return FactoredHierarchicalGrammar(
            rule_table=rule_table[:n_rules],
            rule_counts=rule_counts[:n_rules],
            rule_pitch_intervals=rule_pitch_intervals[:n_rules],
            rule_rhythm_buckets=rule_rhythm_buckets[:n_rules],
            rule_velocity_buckets=rule_velocity_buckets[:n_rules],
            final_sequence=final_seq,
            track_boundaries=track_boundaries,
            n_terminals=n_terminals,
            n_rules=n_rules,
            original_length=original_length,
            compressed_length=len(final_seq),
            device=self.device,
            track_info=track_info or [],
            rule_occurrences=rule_occurrences,
        )

    def _compute_rhythm_buckets_vectorized(
        self, left_ioi: torch.Tensor, right_ioi: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized rhythm bucket computation."""
        ratios = right_ioi.float() / (left_ioi.float() + 1e-6)
        buckets = torch.zeros_like(left_ioi)

        # Subdivision thresholds (buckets 0-7)
        sub_thresh = torch.tensor([0.1875, 0.29, 0.42, 0.585, 0.71, 0.8125, 0.9375, 1.0],
                                  device=self.device)
        # Multiple thresholds (buckets 8-15)
        mul_thresh = torch.tensor([1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 8.5, float('inf')],
                                  device=self.device)

        # Subdivision case (ratio < 1.0)
        sub_mask = ratios < 1.0
        if sub_mask.any():
            sub_ratios = ratios[sub_mask].unsqueeze(1)
            sub_buckets = (sub_ratios >= sub_thresh).sum(dim=1).clamp(max=7)
            buckets[sub_mask] = sub_buckets

        # Multiple case (ratio >= 1.0)
        mul_mask = ratios >= 1.0
        if mul_mask.any():
            mul_ratios = ratios[mul_mask].unsqueeze(1)
            mul_buckets = 8 + (mul_ratios >= mul_thresh).sum(dim=1).clamp(max=7)
            buckets[mul_mask] = mul_buckets

        return buckets

    def _compute_velocity_buckets_vectorized(
        self, left_vel: torch.Tensor, right_vel: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized velocity bucket computation."""
        diffs = right_vel.float() - left_vel.float()
        buckets = torch.zeros_like(left_vel)

        # Thresholds for velocity difference
        thresholds = torch.tensor([-30, -15, -5, 5, 15, 30], device=self.device)
        bucket_values = torch.tensor([0, 1, 2, 3, 5, 6, 7], device=self.device)

        # Find bucket for each difference
        for i, (thresh, bval) in enumerate(zip([-float('inf')] + thresholds.tolist(), bucket_values.tolist())):
            if i < len(thresholds):
                next_thresh = thresholds[i].item()
                mask = (diffs >= thresh) & (diffs < next_thresh)
            else:
                mask = diffs >= thresh
            buckets[mask] = bval

        return buckets

    def _replace_pair_gpu(
        self,
        seq_pitch: torch.Tensor,
        seq_ioi: torch.Tensor,
        seq_velocity: torch.Tensor,
        left: int,
        right: int,
        new_symbol: int,
        separator: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace all (left, right) pairs with new_symbol in all sequences."""
        n = len(seq_pitch)
        if n < 2:
            return seq_pitch, seq_ioi, seq_velocity

        # Find pair positions
        is_left = seq_pitch[:-1] == left
        is_right = seq_pitch[1:] == right
        not_sep_left = seq_pitch[:-1] != separator
        not_sep_right = seq_pitch[1:] != separator

        pair_mask = is_left & is_right & not_sep_left & not_sep_right
        pair_indices = torch.where(pair_mask)[0]

        if len(pair_indices) == 0:
            return seq_pitch, seq_ioi, seq_velocity

        # Remove overlapping: greedy left-to-right
        if len(pair_indices) > 1:
            diffs = pair_indices[1:] - pair_indices[:-1]
            keep = torch.ones(len(pair_indices), dtype=torch.bool, device=seq_pitch.device)
            keep[1:] = diffs > 1
            pair_indices = pair_indices[keep]

        # Build output
        keep_mask = torch.ones(n, dtype=torch.bool, device=seq_pitch.device)
        keep_mask[pair_indices + 1] = False

        # Replace left elements with new symbol
        result_pitch = seq_pitch.clone()
        result_pitch[pair_indices] = new_symbol

        # Keep original IOI and velocity for the new symbol (from left element)
        result_ioi = seq_ioi.clone()
        result_velocity = seq_velocity.clone()

        return result_pitch[keep_mask], result_ioi[keep_mask], result_velocity[keep_mask]

    def _replace_pair_gpu_with_tracking(
        self,
        seq_pitch: torch.Tensor,
        seq_ioi: torch.Tensor,
        seq_velocity: torch.Tensor,
        seq_duration: torch.Tensor,
        seq_onset: torch.Tensor,
        seq_octave: torch.Tensor,
        seq_track_idx: torch.Tensor,
        seq_orig_pos: torch.Tensor,
        left: int,
        right: int,
        new_symbol: int,
        separator: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Replace all (left, right) pairs with new_symbol, tracking occurrences.

        Returns updated sequences plus a list of occurrence dictionaries with
        timing data extracted at the moment of replacement.
        """
        n = len(seq_pitch)
        if n < 2:
            return (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                    seq_track_idx, seq_orig_pos, [])

        # Find pair positions
        is_left = seq_pitch[:-1] == left
        is_right = seq_pitch[1:] == right
        not_sep_left = seq_pitch[:-1] != separator
        not_sep_right = seq_pitch[1:] != separator

        pair_mask = is_left & is_right & not_sep_left & not_sep_right
        pair_indices = torch.where(pair_mask)[0]

        if len(pair_indices) == 0:
            return (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                    seq_track_idx, seq_orig_pos, [])

        # Remove overlapping: greedy left-to-right
        if len(pair_indices) > 1:
            diffs = pair_indices[1:] - pair_indices[:-1]
            keep = torch.ones(len(pair_indices), dtype=torch.bool, device=seq_pitch.device)
            keep[1:] = diffs > 1
            pair_indices = pair_indices[keep]

        # === Extract occurrence data BEFORE modifying sequences ===
        occurrences = []
        pair_indices_cpu = pair_indices.cpu().numpy()

        for idx in pair_indices_cpu:
            # Get track and position info
            track_idx = seq_track_idx[idx].item()
            orig_pos = seq_orig_pos[idx].item()

            if track_idx >= 0 and orig_pos >= 0:
                # Extract timing data for the pair (left and right elements)
                occ = {
                    'track_idx': track_idx,
                    'orig_pos': orig_pos,
                    'ioi': seq_ioi[idx].item(),  # IOI from left element
                    'duration': seq_duration[idx].item(),  # Duration of left element
                    'velocity': seq_velocity[idx].item(),  # Velocity of left element
                    'onset': seq_onset[idx].item(),  # Onset of left element
                    'octave': seq_octave[idx].item(),  # Octave of left element (0-9)
                }
                # Also capture right element data for 2-note patterns
                if idx + 1 < n:
                    occ['ioi_right'] = seq_ioi[idx + 1].item()
                    occ['duration_right'] = seq_duration[idx + 1].item()
                    occ['velocity_right'] = seq_velocity[idx + 1].item()
                    occ['onset_right'] = seq_onset[idx + 1].item()
                    occ['octave_right'] = seq_octave[idx + 1].item()  # Octave of right element

                occurrences.append(occ)

        # === Now perform the replacement ===
        # Build output mask
        keep_mask = torch.ones(n, dtype=torch.bool, device=seq_pitch.device)
        keep_mask[pair_indices + 1] = False  # Remove right elements

        # Replace left elements with new symbol
        result_pitch = seq_pitch.clone()
        result_pitch[pair_indices] = new_symbol

        # Keep track/position from left element (which now represents the pattern)
        result_ioi = seq_ioi.clone()
        result_velocity = seq_velocity.clone()
        result_duration = seq_duration.clone()
        result_onset = seq_onset.clone()
        result_octave = seq_octave.clone()
        result_track_idx = seq_track_idx.clone()
        result_orig_pos = seq_orig_pos.clone()

        return (result_pitch[keep_mask], result_ioi[keep_mask], result_velocity[keep_mask],
                result_duration[keep_mask], result_onset[keep_mask], result_octave[keep_mask],
                result_track_idx[keep_mask], result_orig_pos[keep_mask], occurrences)

    def _interval_name(self, interval: int) -> str:
        """Convert interval to musical name."""
        names = {
            0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
            5: "P4", 6: "tritone", 7: "P5", 8: "m6", 9: "M6",
            10: "m7", 11: "M7"
        }
        return names.get(interval % 12, f"{interval}")


def build_factored_hierarchical_grammar(
    pitch_sequences: List[List[int]],
    ioi_sequences: List[List[int]],
    velocity_sequences: List[List[int]],
    track_info: List[Dict] = None,
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules: int = 10000,
    verbose: bool = True,
    duration_sequences: List[List[int]] = None,
    onset_sequences: List[List[int]] = None,
    octave_sequences: List[List[int]] = None,
) -> FactoredHierarchicalGrammar:
    """
    Build grammar using T+τ+v factored hierarchical Re-Pair.

    Args:
        pitch_sequences: List of pitch class sequences (0-11)
        ioi_sequences: List of IOI sequences (ticks between notes)
        velocity_sequences: List of velocity sequences (0-127)
        track_info: Optional list of {piece_id, track_id} per track
        device: 'cuda' or 'cpu'
        min_pair_count: Minimum pair frequency
        max_rules: Maximum rules to create
        verbose: Print progress
        duration_sequences: Optional list of note duration sequences
        onset_sequences: Optional list of onset time sequences
        octave_sequences: Optional list of octave sequences (0-9) for pitch reconstruction

    Returns:
        FactoredHierarchicalGrammar with occurrence data captured during compression
    """
    repair = RePairFactoredHierarchical(
        device=device,
        min_pair_count=min_pair_count,
        max_rules=max_rules,
        verbose=verbose,
    )
    return repair.induce(
        pitch_sequences, ioi_sequences, velocity_sequences, track_info,
        duration_sequences, onset_sequences, octave_sequences
    )


if __name__ == '__main__':
    print("Testing T+τ+v Factored Hierarchical GPU Re-Pair...")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Test with musical data
    np.random.seed(42)
    n_sequences = 500
    pitch_sequences = []
    ioi_sequences = []
    velocity_sequences = []

    # Create sequences with TRANSPOSED and TEMPO-VARIED patterns
    base_patterns = [
        ([0, 4, 7], [480, 480], [80, 80, 80]),       # Major triad, quarter notes
        ([0, 3, 7], [240, 240], [90, 80, 70]),       # Minor triad, eighth notes, decrescendo
        ([0, 2, 4, 5, 7], [480, 480, 480, 480], [70, 75, 80, 85, 90]),  # Scale, crescendo
    ]

    for _ in range(n_sequences):
        length = np.random.randint(30, 100)
        pitch_seq = []
        ioi_seq = []
        vel_seq = []

        while len(pitch_seq) < length:
            # Pick a random pattern
            pattern_idx = np.random.randint(len(base_patterns))
            pitches, iois, vels = base_patterns[pattern_idx]

            # Transpose randomly (mod 12 for pitch classes)
            transpose = np.random.randint(0, 12)
            transposed = [(p + transpose) % 12 for p in pitches]

            # Scale tempo randomly (0.5x to 2x)
            tempo_scale = np.random.choice([0.5, 1.0, 1.5, 2.0])
            scaled_iois = [int(i * tempo_scale) for i in iois]

            # Scale velocity randomly
            vel_offset = np.random.randint(-20, 20)
            scaled_vels = [max(1, min(127, v + vel_offset)) for v in vels]

            pitch_seq.extend(transposed)
            ioi_seq.extend(scaled_iois + [480])  # Add default IOI at end
            vel_seq.extend(scaled_vels)

        pitch_sequences.append(pitch_seq[:length])
        ioi_sequences.append(ioi_seq[:length])
        velocity_sequences.append(vel_seq[:length])

    total_notes = sum(len(s) for s in pitch_sequences)
    print(f"\nTest data: {n_sequences} sequences, {total_notes:,} notes")
    print("Patterns: Major/minor triads, scale fragment (transposed, tempo-varied, velocity-varied)")

    # Run T+τ+v factored hierarchical Re-Pair
    t0 = time.time()
    grammar = build_factored_hierarchical_grammar(
        pitch_sequences, ioi_sequences, velocity_sequences,
        device='cuda', verbose=True
    )
    elapsed = time.time() - t0

    print(f"\nResults:")
    print(f"  Rules: {grammar.n_rules}")
    print(f"  Compression: {grammar.compression_ratio():.2f}x")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Speed: {total_notes/elapsed:,.0f} notes/sec")

    # Show top rules
    print("\nTop 10 rules by count:")
    sorted_idx = grammar.rule_counts.argsort(descending=True)[:10]
    for idx in sorted_idx:
        rule_id = grammar.n_terminals + idx.item()
        left, right = grammar.get_rule(rule_id)
        count = grammar.rule_counts[idx].item()
        pitch_int = grammar.rule_pitch_intervals[idx].item()
        rhythm_b = grammar.rule_rhythm_buckets[idx].item()
        vel_b = grammar.rule_velocity_buckets[idx].item()

        interval_name = {
            0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3",
            5: "P4", 6: "tritone", 7: "P5", 8: "m6", 9: "M6",
            10: "m7", 11: "M7"
        }.get(pitch_int % 12 if pitch_int >= 0 else -1, f"{pitch_int}")

        is_hier = left >= 12 or right >= 12
        hier_mark = " [HIER]" if is_hier else ""

        print(f"  R{rule_id}: ({left},{right}) pitch={interval_name} "
              f"rhythm={rhythm_b} vel={vel_b}{hier_mark} (count={count:,})")
