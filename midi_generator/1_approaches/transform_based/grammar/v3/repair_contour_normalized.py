"""
Contour-Normalized Hierarchical GPU Re-Pair (v3)
=================================================

TRUE T+τ+v normalization where patterns ARE contours.

Key difference from v2:
- v2: Pattern = (pitch_class_A, pitch_class_B), stores interval
      Different pitch-class pairs are different patterns
- v3: Pattern = (interval, rhythm_bucket, velocity_bucket)
      All pairs with same contour are the SAME pattern
      Each occurrence stores its pitch_offset

Philosophy:
  "The pattern is the contour. The pitch is the transformation."

Example:
  - (C,D) and (E,F#) both have interval +2
  - They are INSTANCES of the same pattern [+2]
  - (C,D) has pitch_offset=0, (E,F#) has pitch_offset=4

This achieves true T-normalization: patterns are transposition-invariant.
The compression is optimal because we don't waste rules on redundant
patterns that differ only in starting pitch.

Equation:
  M = Pattern(contour) × T(pitch_offset) × O(octave_offset)

Where:
  - Pattern = (intervals, rhythm_ratios, velocity_ratios)
  - T(pitch_offset) = transposition from canonical (first) occurrence
  - O(octave_offset) = octave transposition from canonical
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time


# Rhythm bucket constants (16 buckets for τ-normalization)
RHYTHM_BUCKETS = 16
VELOCITY_BUCKETS = 8
INTERVAL_OFFSET = 128  # Offset for encoding signed intervals as positive


def compute_rhythm_bucket(ioi1: int, ioi2: int) -> int:
    """Compute rhythm ratio bucket for τ-normalization."""
    if ioi1 <= 0:
        return 7  # Default to 1.0 ratio

    ratio = ioi2 / ioi1

    if ratio < 1.0:
        thresholds = [0.1875, 0.29, 0.42, 0.585, 0.71, 0.8125, 0.9375]
        for i, thresh in enumerate(thresholds):
            if ratio < thresh:
                return i
        return 7
    else:
        thresholds = [1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 8.5]
        for i, thresh in enumerate(thresholds):
            if ratio < thresh:
                return 8 + i
        return 15


def compute_velocity_bucket(v1: int, v2: int) -> int:
    """Compute velocity ratio bucket for v-normalization."""
    diff = v2 - v1

    if diff < -30:
        return 0
    elif diff < -15:
        return 1
    elif diff < -5:
        return 2
    elif diff <= 5:
        return 3
    elif diff <= 15:
        return 5
    elif diff <= 30:
        return 6
    else:
        return 7


@dataclass
class ContourNormalizedGrammar:
    """Grammar where patterns ARE contours (interval sequences)."""

    # Rule definitions by normalized triple (not by pitch-class pair!)
    # rule_id -> (interval, rhythm_bucket, velocity_bucket)
    rule_contours: torch.Tensor  # Shape: [n_rules, 3] - (interval, rhythm, vel)
    rule_counts: torch.Tensor    # Shape: [n_rules] - total occurrences

    # For hierarchical rules: which sub-rules they combine
    # For terminal rules: (-1, -1)
    rule_children: torch.Tensor  # Shape: [n_rules, 2] - (left_rule, right_rule)

    # Final compressed sequence
    final_sequence: torch.Tensor
    track_boundaries: List[int]

    # Metadata
    n_terminals: int  # Number of pitch classes (12)
    n_rules: int
    original_length: int
    compressed_length: int
    device: str

    # Track info
    track_info: List[Dict]

    # Occurrences: rule_id -> list of {track_idx, orig_pos, pitch_offset, octave, ...}
    # pitch_offset is the starting pitch class (0-11) for this occurrence
    rule_occurrences: Dict[int, List[Dict]] = None

    def compression_ratio(self) -> float:
        if self.compressed_length == 0:
            return 1.0
        return self.original_length / self.compressed_length

    def get_rule_contour(self, rule_id: int) -> Tuple[int, int, int]:
        """Get (interval, rhythm_bucket, velocity_bucket) for a rule."""
        # Rules start at n_terminals + 1 (after separator), so idx = rule_id - (n_terminals + 1)
        idx = rule_id - self.n_terminals - 1
        if idx < 0 or idx >= len(self.rule_contours):
            return (0, 0, 0)
        row = self.rule_contours[idx]
        return (row[0].item(), row[1].item(), row[2].item())

    def get_rule_children(self, rule_id: int) -> Tuple[int, int]:
        """Get (left_child, right_child) for hierarchical rule, or (-1,-1) for terminal."""
        # Rules start at n_terminals + 1 (after separator), so idx = rule_id - (n_terminals + 1)
        idx = rule_id - self.n_terminals - 1
        if idx < 0 or idx >= len(self.rule_children):
            return (-1, -1)
        row = self.rule_children[idx]
        return (row[0].item(), row[1].item())

    def expand_rule(self, rule_id: int, memo: Optional[Dict] = None) -> List[int]:
        """Recursively expand rule to get interval sequence."""
        if memo is None:
            memo = {}
        if rule_id in memo:
            return memo[rule_id]

        if rule_id < self.n_terminals:
            # Terminal: no interval (single pitch class)
            return []

        left, right = self.get_rule_children(rule_id)
        interval, _, _ = self.get_rule_contour(rule_id)

        if left < 0:
            # Base 2-note pattern
            result = [interval]
        else:
            # Hierarchical: combine child intervals + connector
            left_intervals = self.expand_rule(left, memo)
            right_intervals = self.expand_rule(right, memo)
            result = left_intervals + [interval] + right_intervals

        memo[rule_id] = result
        return result


class RePairContourNormalized:
    """
    True contour-normalized Re-Pair.

    Patterns are defined by (interval, rhythm, velocity), not by pitch classes.
    All pairs with the same normalized triple are instances of ONE pattern.
    """

    def __init__(
        self,
        device: str = 'cuda',
        min_pair_count: int = 2,
        max_rules: int = 10000,
        rhythm_buckets: int = RHYTHM_BUCKETS,
        velocity_buckets: int = VELOCITY_BUCKETS,
        verbose: bool = True,
        progress_every: int = 100,
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.min_pair_count = min_pair_count
        self.max_rules = max_rules
        self.rhythm_buckets = rhythm_buckets
        self.velocity_buckets = velocity_buckets
        self.verbose = verbose
        self.progress_every = progress_every

    def _compute_rhythm_buckets_vectorized(self, left_ioi: torch.Tensor, right_ioi: torch.Tensor) -> torch.Tensor:
        """Vectorized rhythm bucket computation."""
        ratios = right_ioi.float() / (left_ioi.float() + 1e-6)
        buckets = torch.zeros_like(left_ioi)

        # Subdivision ratios (< 1.0)
        sub_thresholds = torch.tensor([0.1875, 0.29, 0.42, 0.585, 0.71, 0.8125, 0.9375], device=self.device)
        for i, thresh in enumerate(sub_thresholds):
            buckets[(ratios < thresh) & (buckets == 0)] = i
        buckets[(ratios < 1.0) & (buckets == 0)] = 7

        # Multiple ratios (>= 1.0)
        mult_thresholds = torch.tensor([1.25, 1.75, 2.5, 3.5, 5.0, 7.0, 8.5], device=self.device)
        for i, thresh in enumerate(mult_thresholds):
            buckets[(ratios >= 1.0) & (ratios < thresh) & (buckets == 0)] = 8 + i
        buckets[(ratios >= 8.5)] = 15

        return buckets

    def _compute_velocity_buckets_vectorized(self, left_v: torch.Tensor, right_v: torch.Tensor) -> torch.Tensor:
        """Vectorized velocity bucket computation."""
        diff = right_v - left_v
        buckets = torch.full_like(diff, 3)  # Default to "same"

        buckets[diff < -30] = 0
        buckets[(diff >= -30) & (diff < -15)] = 1
        buckets[(diff >= -15) & (diff < -5)] = 2
        buckets[(diff > 5) & (diff <= 15)] = 5
        buckets[(diff > 15) & (diff <= 30)] = 6
        buckets[diff > 30] = 7

        return buckets

    def compress(
        self,
        pitch_sequences: List[List[int]],
        ioi_sequences: List[List[int]],
        velocity_sequences: List[List[int]],
        track_info: List[Dict] = None,
        duration_sequences: List[List[int]] = None,
        onset_sequences: List[List[int]] = None,
        octave_sequences: List[List[int]] = None,
    ) -> ContourNormalizedGrammar:
        """
        Compress sequences using contour-normalized Re-Pair.

        Key difference: ALL pairs matching a normalized triple are replaced together,
        with each occurrence storing its pitch_offset.
        """
        start_time = time.time()

        if self.verbose:
            total_notes = sum(len(s) for s in pitch_sequences)
            print(f"[RePair-Contour] Starting: {len(pitch_sequences)} tracks, {total_notes:,} notes", flush=True)

        n_terminals = 12  # Pitch classes
        separator = n_terminals  # Use 12 as track separator

        # Concatenate all sequences with separators
        all_pitch = []
        all_ioi = []
        all_velocity = []
        all_duration = []
        all_onset = []
        all_octave = []
        all_track_idx = []
        all_orig_pos = []
        track_boundaries = []

        for track_idx, (pitches, iois, vels) in enumerate(zip(pitch_sequences, ioi_sequences, velocity_sequences)):
            if len(pitches) == 0:
                continue

            durs = duration_sequences[track_idx] if duration_sequences else [480] * len(pitches)
            onsets = onset_sequences[track_idx] if onset_sequences else list(range(0, len(pitches) * 480, 480))
            octs = octave_sequences[track_idx] if octave_sequences else [4] * len(pitches)

            # Add notes
            for i, (p, ioi, v, d, o, octave) in enumerate(zip(pitches, iois, vels, durs, onsets, octs)):
                all_pitch.append(p % 12)  # Pitch class
                all_ioi.append(ioi)
                all_velocity.append(v)
                all_duration.append(d)
                all_onset.append(o)
                all_octave.append(octave)
                all_track_idx.append(track_idx)
                all_orig_pos.append(i)

            # Add separator
            track_boundaries.append(len(all_pitch))
            all_pitch.append(separator)
            all_ioi.append(0)
            all_velocity.append(0)
            all_duration.append(0)
            all_onset.append(0)
            all_octave.append(0)
            all_track_idx.append(-1)
            all_orig_pos.append(-1)

        # Convert to tensors
        seq_pitch = torch.tensor(all_pitch, dtype=torch.int64, device=self.device)
        seq_ioi = torch.tensor(all_ioi, dtype=torch.int64, device=self.device)
        seq_velocity = torch.tensor(all_velocity, dtype=torch.int64, device=self.device)
        seq_duration = torch.tensor(all_duration, dtype=torch.int64, device=self.device)
        seq_onset = torch.tensor(all_onset, dtype=torch.int64, device=self.device)
        seq_octave = torch.tensor(all_octave, dtype=torch.int64, device=self.device)
        seq_track_idx = torch.tensor(all_track_idx, dtype=torch.int64, device=self.device)
        seq_orig_pos = torch.tensor(all_orig_pos, dtype=torch.int64, device=self.device)
        # Track the last original position for each symbol (for multi-note symbols after compression)
        seq_last_orig_pos = torch.tensor(all_orig_pos, dtype=torch.int64, device=self.device)

        original_length = len(seq_pitch) - len(pitch_sequences)  # Exclude separators

        # For tracking actual MIDI pitches (for interval computation)
        seq_first_pitch = seq_pitch.clone()
        seq_last_pitch = seq_pitch.clone()
        terminal_mask = seq_pitch < n_terminals
        seq_first_pitch[terminal_mask] = seq_pitch[terminal_mask] + seq_octave[terminal_mask] * 12
        seq_last_pitch[terminal_mask] = seq_pitch[terminal_mask] + seq_octave[terminal_mask] * 12

        # Rule storage
        rule_contours = torch.zeros((self.max_rules, 3), dtype=torch.int64, device=self.device)
        rule_children = torch.full((self.max_rules, 2), -1, dtype=torch.int64, device=self.device)
        rule_counts = torch.zeros(self.max_rules, dtype=torch.int64, device=self.device)
        rule_occurrences = {}

        # Symbol properties (for hierarchical patterns)
        max_symbols = n_terminals + 1 + self.max_rules
        symbol_base_ioi = torch.full((max_symbols,), 480, dtype=torch.int64, device=self.device)
        symbol_base_velocity = torch.full((max_symbols,), 64, dtype=torch.int64, device=self.device)

        # Track first/last pitch for each symbol position
        # For hierarchical: first_pitch of combined pattern, last_pitch of combined pattern

        next_rule_id = n_terminals + 1  # Start after separator
        n_rules = 0

        # Main Re-Pair loop
        iteration = 0
        while iteration < self.max_rules:
            # ===== STEP 1: Find most frequent normalized triple =====
            n = len(seq_pitch)
            if n < 2:
                break

            # Get consecutive pairs
            left_pitch = seq_pitch[:-1]
            right_pitch = seq_pitch[1:]
            left_ioi = seq_ioi[:-1]
            right_ioi = seq_ioi[1:]
            left_vel = seq_velocity[:-1]
            right_vel = seq_velocity[1:]

            # Valid pairs (not crossing separators)
            valid = (left_pitch != separator) & (right_pitch != separator)

            if not valid.any():
                break

            # Get actual MIDI pitches for interval computation
            left_last_pitch = seq_last_pitch[:-1][valid]
            right_first_pitch = seq_first_pitch[1:][valid]

            # Compute SIGNED pitch interval
            pitch_intervals = right_first_pitch - left_last_pitch

            # Get symbol base IOI/velocity for rhythm/velocity buckets
            left_p = left_pitch[valid]
            right_p = right_pitch[valid]
            left_base_ioi = symbol_base_ioi[left_p]
            right_base_ioi = symbol_base_ioi[right_p]
            left_base_v = symbol_base_velocity[left_p]
            right_base_v = symbol_base_velocity[right_p]

            # Compute normalized buckets
            rhythm_buckets = self._compute_rhythm_buckets_vectorized(left_base_ioi, right_base_ioi)
            velocity_buckets = self._compute_velocity_buckets_vectorized(left_base_v, right_base_v)

            # Encode normalized triple
            # triple_id = (interval+OFFSET) * rhythm_buckets * velocity_buckets + rhythm * velocity_buckets + velocity
            pitch_offset = pitch_intervals + INTERVAL_OFFSET
            triple_ids = (pitch_offset * self.rhythm_buckets * self.velocity_buckets +
                         rhythm_buckets * self.velocity_buckets +
                         velocity_buckets)

            # Find most common triple
            unique_triples, triple_counts = torch.unique(triple_ids, return_counts=True)
            max_triple_idx = triple_counts.argmax()
            max_triple_count = triple_counts[max_triple_idx].item()
            best_triple = unique_triples[max_triple_idx].item()

            if max_triple_count < self.min_pair_count:
                break

            # Decode best triple
            best_interval_offset = best_triple // (self.rhythm_buckets * self.velocity_buckets)
            best_interval = best_interval_offset - INTERVAL_OFFSET
            remainder = best_triple % (self.rhythm_buckets * self.velocity_buckets)
            best_rhythm = remainder // self.velocity_buckets
            best_velocity = remainder % self.velocity_buckets

            # ===== STEP 2: Create new rule for this contour =====
            rule_id = next_rule_id
            next_rule_id += 1

            # Store rule definition (contour, not pitch-class pair!)
            rule_contours[n_rules, 0] = best_interval
            rule_contours[n_rules, 1] = best_rhythm
            rule_contours[n_rules, 2] = best_velocity
            rule_counts[n_rules] = max_triple_count

            # Check if this is a hierarchical pattern
            triple_mask = triple_ids == best_triple
            valid_indices = torch.where(valid)[0]
            pair_positions = valid_indices[triple_mask]

            # Determine left/right symbol types for this rule
            # CRITICAL: For hierarchical patterns, we MUST match by exact symbol type!
            # For non-hierarchical (terminal, terminal) patterns, we allow any terminals.
            expected_left = -1
            expected_right = -1
            is_hierarchical = False

            if len(pair_positions) > 0:
                # Get first pair to determine symbol types
                first_left = seq_pitch[pair_positions[0]].item()
                first_right = seq_pitch[pair_positions[0] + 1].item()
                is_hierarchical = (first_left > n_terminals) or (first_right > n_terminals)

                if is_hierarchical:
                    # For hierarchical: store which sub-rules we're combining
                    # AND use this to filter matches (ensures exact symbol type matching)
                    rule_children[n_rules, 0] = first_left
                    rule_children[n_rules, 1] = first_right
                    expected_left = first_left
                    expected_right = first_right
                else:
                    # For non-hierarchical (terminal, terminal) patterns:
                    # We need to ensure we don't accidentally match (rule, terminal) pairs
                    # by requiring BOTH symbols to be terminals (< n_terminals)
                    # We use n_terminals as the threshold: any symbol >= n_terminals is a rule
                    # Setting expected to n_terminals signals "must be terminal"
                    expected_left = -2  # Special: require terminal (< n_terminals)
                    expected_right = -2  # Special: require terminal (< n_terminals)

            n_rules += 1

            # ===== STEP 3: Replace ALL pairs matching this triple =====
            # For hierarchical patterns, also require matching symbol types
            (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
             seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, occurrences) = self._replace_triple_gpu(
                seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch,
                symbol_base_ioi, symbol_base_velocity,
                best_interval, best_rhythm, best_velocity,
                rule_id, separator, n_terminals,
                expected_left=expected_left, expected_right=expected_right
            )

            # Store occurrences (each has pitch_offset!)
            rule_occurrences[rule_id] = occurrences

            # Set base properties for new symbol
            symbol_base_ioi[rule_id] = 480  # Default
            symbol_base_velocity[rule_id] = 64

            # Progress
            if self.verbose and (iteration % self.progress_every == 0 or iteration < 10):
                elapsed = time.time() - start_time
                interval_name = self._interval_name(best_interval)
                hier_marker = " [HIER]" if (n_rules > 0 and rule_children[n_rules-1, 0].item() >= 0) else ""
                print(f"  [iter {iteration}] R{rule_id}=contour({best_interval}:{interval_name}, r{best_rhythm}, v{best_velocity}){hier_marker} "
                      f"count={max_triple_count:,}, len={len(seq_pitch):,}, time={elapsed:.1f}s", flush=True)

            iteration += 1

        # Remove separators from final sequence
        mask = seq_pitch != separator
        final_seq = seq_pitch[mask]

        elapsed = time.time() - start_time

        if self.verbose:
            ratio = original_length / len(final_seq) if len(final_seq) > 0 else 1.0
            hier_count = sum(1 for i in range(n_rules) if rule_children[i, 0].item() >= 0)
            print(f"[RePair-Contour] Done: {n_rules} rules ({hier_count} hierarchical), "
                  f"compression {ratio:.2f}x, time {elapsed:.1f}s", flush=True)

        return ContourNormalizedGrammar(
            rule_contours=rule_contours[:n_rules],
            rule_counts=rule_counts[:n_rules],
            rule_children=rule_children[:n_rules],
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

    def _replace_triple_gpu(
        self,
        seq_pitch: torch.Tensor,
        seq_ioi: torch.Tensor,
        seq_velocity: torch.Tensor,
        seq_duration: torch.Tensor,
        seq_onset: torch.Tensor,
        seq_octave: torch.Tensor,
        seq_track_idx: torch.Tensor,
        seq_orig_pos: torch.Tensor,
        seq_last_orig_pos: torch.Tensor,  # NEW: tracks the LAST note's position for multi-note symbols
        seq_first_pitch: torch.Tensor,
        seq_last_pitch: torch.Tensor,
        symbol_base_ioi: torch.Tensor,
        symbol_base_velocity: torch.Tensor,
        target_interval: int,
        target_rhythm: int,
        target_velocity: int,
        new_symbol: int,
        separator: int,
        n_terminals: int,
        expected_left: int = -1,
        expected_right: int = -1,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Replace ALL pairs matching the normalized triple with new_symbol.

        KEY DIFFERENCE from v2: We match by (interval, rhythm, velocity), not by (left_pitch, right_pitch).
        This means (C,D) and (E,F#) both get replaced if they have the same interval.
        """
        n = len(seq_pitch)
        if n < 2:
            return (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                    seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, [])

        # Compute intervals and buckets for all pairs
        left_pitch = seq_pitch[:-1]
        right_pitch = seq_pitch[1:]

        valid = (left_pitch != separator) & (right_pitch != separator)

        if not valid.any():
            return (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                    seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, [])

        # Compute intervals
        left_last = seq_last_pitch[:-1]
        right_first = seq_first_pitch[1:]
        intervals = right_first - left_last

        # Compute rhythm/velocity buckets
        left_base_ioi = symbol_base_ioi[left_pitch]
        right_base_ioi = symbol_base_ioi[right_pitch]
        left_base_v = symbol_base_velocity[left_pitch]
        right_base_v = symbol_base_velocity[right_pitch]

        rhythm_buckets = self._compute_rhythm_buckets_vectorized(left_base_ioi, right_base_ioi)
        velocity_buckets = self._compute_velocity_buckets_vectorized(left_base_v, right_base_v)

        # Find pairs matching the target triple
        match_mask = (valid &
                     (intervals == target_interval) &
                     (rhythm_buckets == target_rhythm) &
                     (velocity_buckets == target_velocity))

        # Symbol type filtering to ensure we don't mix different pattern structures
        # -2 = must be terminal (< n_terminals)
        # >= 0 = must be exact match (for hierarchical patterns)
        # -1 = no filtering (unused now)
        if expected_left == -2:
            # Require left to be a terminal (not a rule)
            match_mask = match_mask & (left_pitch < n_terminals)
        elif expected_left >= 0:
            match_mask = match_mask & (left_pitch == expected_left)

        if expected_right == -2:
            # Require right to be a terminal (not a rule)
            match_mask = match_mask & (right_pitch < n_terminals)
        elif expected_right >= 0:
            match_mask = match_mask & (right_pitch == expected_right)

        pair_indices = torch.where(match_mask)[0]

        if len(pair_indices) == 0:
            return (seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset, seq_octave,
                    seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, [])

        # Remove overlapping (greedy left-to-right)
        if len(pair_indices) > 1:
            diffs = pair_indices[1:] - pair_indices[:-1]
            keep = torch.ones(len(pair_indices), dtype=torch.bool, device=seq_pitch.device)
            keep[1:] = diffs > 1
            pair_indices = pair_indices[keep]

        # === Extract occurrences with pitch_offset ===
        occurrences = []
        pair_indices_cpu = pair_indices.cpu().numpy()

        for idx in pair_indices_cpu:
            track_idx = seq_track_idx[idx].item()
            orig_pos = seq_orig_pos[idx].item()

            if track_idx >= 0 and orig_pos >= 0:
                # The pitch_offset is the starting pitch class (0-11)
                pitch_offset = seq_pitch[idx].item()
                if pitch_offset >= n_terminals:
                    # For hierarchical, use the first_pitch's pitch class
                    pitch_offset = seq_first_pitch[idx].item() % 12

                occ = {
                    'track_idx': track_idx,
                    'orig_pos': orig_pos,
                    'pitch_offset': pitch_offset,  # KEY: Starting pitch class
                    'octave': seq_octave[idx].item(),
                    'ioi': seq_ioi[idx].item(),
                    'duration': seq_duration[idx].item(),
                    'velocity': seq_velocity[idx].item(),
                    'onset': seq_onset[idx].item(),
                    'first_pitch': seq_first_pitch[idx].item(),  # Actual MIDI pitch
                }

                if idx + 1 < n:
                    occ['ioi_right'] = seq_ioi[idx + 1].item()
                    occ['duration_right'] = seq_duration[idx + 1].item()
                    occ['velocity_right'] = seq_velocity[idx + 1].item()
                    occ['octave_right'] = seq_octave[idx + 1].item()
                    occ['last_pitch'] = seq_last_pitch[idx + 1].item()  # Actual MIDI pitch

                occurrences.append(occ)

        # === Perform replacement ===
        keep_mask = torch.ones(n, dtype=torch.bool, device=seq_pitch.device)
        keep_mask[pair_indices + 1] = False

        result_pitch = seq_pitch.clone()
        result_pitch[pair_indices] = new_symbol

        result_ioi = seq_ioi.clone()
        result_velocity = seq_velocity.clone()
        result_duration = seq_duration.clone()
        result_onset = seq_onset.clone()
        result_octave = seq_octave.clone()
        result_track_idx = seq_track_idx.clone()
        result_orig_pos = seq_orig_pos.clone()

        # Update last_orig_pos: the combined symbol's last position is the right symbol's last position
        result_last_orig_pos = seq_last_orig_pos.clone()
        result_last_orig_pos[pair_indices] = seq_last_orig_pos[pair_indices + 1]

        result_first_pitch = seq_first_pitch.clone()
        result_last_pitch = seq_last_pitch.clone()
        result_last_pitch[pair_indices] = seq_last_pitch[pair_indices + 1]

        return (result_pitch[keep_mask], result_ioi[keep_mask], result_velocity[keep_mask],
                result_duration[keep_mask], result_onset[keep_mask], result_octave[keep_mask],
                result_track_idx[keep_mask], result_orig_pos[keep_mask],
                result_last_orig_pos[keep_mask], result_first_pitch[keep_mask],
                result_last_pitch[keep_mask], occurrences)

    def _interval_name(self, interval: int) -> str:
        """Convert signed interval to musical name."""
        if interval == 0:
            return "unison"

        direction = "+" if interval > 0 else "-"
        semitones = abs(interval)

        octaves = semitones // 12
        remainder = semitones % 12

        names = ['P1', 'm2', 'M2', 'm3', 'M3', 'P4', 'TT', 'P5', 'm6', 'M6', 'm7', 'M7']

        if octaves == 0:
            return f"{direction}{names[remainder]}"
        elif remainder == 0:
            return f"{direction}{octaves}oct"
        else:
            return f"{direction}{names[remainder]}+{octaves}oct"


def build_contour_normalized_grammar(
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
) -> ContourNormalizedGrammar:
    """Build grammar using contour-normalized Re-Pair."""
    compressor = RePairContourNormalized(
        device=device,
        min_pair_count=min_pair_count,
        max_rules=max_rules,
        verbose=verbose,
    )

    return compressor.compress(
        pitch_sequences=pitch_sequences,
        ioi_sequences=ioi_sequences,
        velocity_sequences=velocity_sequences,
        track_info=track_info,
        duration_sequences=duration_sequences,
        onset_sequences=onset_sequences,
        octave_sequences=octave_sequences,
    )
