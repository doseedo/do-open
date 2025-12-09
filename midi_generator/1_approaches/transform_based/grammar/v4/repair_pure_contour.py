"""
Pure Contour Re-Pair (v4)
=========================

TRUE pitch-agnostic normalization: terminals have NO pitch information.

Key difference from v3:
- v3: Terminal = pitch_class (0-11), pattern = (interval, rhythm, velocity)
      Hierarchical patterns still inherit pitch-class specificity from children
- v4: Terminal = (rhythm_bucket, velocity_bucket) ONLY - no pitch at all
      Every pattern is purely contour-based, even hierarchical ones

Philosophy:
  "Pitch is always instance-specific. The pattern IS the shape."

Example:
  - Note (C4, quarter, mf) and (E4, quarter, mf) are the SAME terminal
  - (C4→D4) and (E4→F#4) are instances of the same [+2] pattern
  - Even hierarchical pattern R21(R13, R14) merges across ALL transpositions

This achieves optimal compression because pitch is purely a transform parameter.

Equation:
  M = Pattern(contour) × T(first_pitch)

Where:
  - Pattern = (interval_sequence, rhythm_ratios, velocity_ratios)
  - T(first_pitch) = absolute starting MIDI pitch (0-127)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time


# Bucket constants
RHYTHM_BUCKETS = 16
VELOCITY_BUCKETS = 8
N_TERMINALS = RHYTHM_BUCKETS * VELOCITY_BUCKETS  # 128 pitch-agnostic terminals
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


def encode_terminal(rhythm_bucket: int, velocity_bucket: int) -> int:
    """Encode (rhythm, velocity) as single terminal ID."""
    return rhythm_bucket * VELOCITY_BUCKETS + velocity_bucket


def decode_terminal(terminal_id: int) -> Tuple[int, int]:
    """Decode terminal ID to (rhythm_bucket, velocity_bucket)."""
    rhythm = terminal_id // VELOCITY_BUCKETS
    velocity = terminal_id % VELOCITY_BUCKETS
    return rhythm, velocity


@dataclass
class PureContourGrammar:
    """Grammar where terminals have NO pitch - purely rhythm/velocity based."""

    # Rule definitions by normalized triple
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
    n_terminals: int  # 128 = RHYTHM_BUCKETS * VELOCITY_BUCKETS
    n_rules: int
    original_length: int
    compressed_length: int
    device: str

    # Track info
    track_info: List[Dict]

    # Occurrences: rule_id -> list of {track_idx, orig_pos, first_pitch, ...}
    # first_pitch is the absolute MIDI pitch (0-127) for this occurrence
    rule_occurrences: Dict[int, List[Dict]] = None

    def compression_ratio(self) -> float:
        if self.compressed_length == 0:
            return 1.0
        return self.original_length / self.compressed_length

    def get_rule_contour(self, rule_id: int) -> Tuple[int, int, int]:
        """Get (interval, rhythm_bucket, velocity_bucket) for a rule."""
        idx = rule_id - self.n_terminals - 1
        if idx < 0 or idx >= len(self.rule_contours):
            return (0, 0, 0)
        row = self.rule_contours[idx]
        return (row[0].item(), row[1].item(), row[2].item())

    def get_rule_children(self, rule_id: int) -> Tuple[int, int]:
        """Get (left_child, right_child) for hierarchical rule, or (-1,-1) for terminal."""
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
            # Terminal: no interval (single note)
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


class RePairPureContour:
    """
    Pure contour Re-Pair with pitch-agnostic terminals.

    Terminals are (rhythm_bucket, velocity_bucket) pairs.
    Pitch is purely occurrence-specific.
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
        self.n_terminals = rhythm_buckets * velocity_buckets
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
        pitch_sequences: List[List[int]],      # MIDI pitch (0-127)
        ioi_sequences: List[List[int]],
        velocity_sequences: List[List[int]],
        track_info: List[Dict] = None,
        duration_sequences: List[List[int]] = None,
        onset_sequences: List[List[int]] = None,
    ) -> PureContourGrammar:
        """
        Compress sequences using pure contour Re-Pair.

        Key difference: Terminals are (rhythm, velocity) pairs, NOT pitch classes.
        Pitch is stored per-occurrence as first_pitch.
        """
        start_time = time.time()

        if self.verbose:
            total_notes = sum(len(s) for s in pitch_sequences)
            print(f"[RePair-PureContour] Starting: {len(pitch_sequences)} tracks, {total_notes:,} notes", flush=True)

        n_terminals = self.n_terminals  # 128 = 16 * 8
        separator = n_terminals  # Use 128 as track separator

        # Store original track data for per-note ratio computation in occurrences
        self.original_tracks = {}
        for track_idx, (pitches, iois, vels) in enumerate(zip(pitch_sequences, ioi_sequences, velocity_sequences)):
            durs = duration_sequences[track_idx] if duration_sequences else [480] * len(pitches)
            self.original_tracks[track_idx] = {
                'pitches': list(pitches),
                'iois': list(iois),
                'velocities': list(vels),
                'durations': list(durs),
            }

        # Concatenate all sequences with separators
        all_terminal = []      # Terminal = (rhythm_bucket, velocity_bucket) encoded
        all_pitch = []         # MIDI pitch for reconstruction
        all_ioi = []
        all_velocity = []
        all_duration = []
        all_onset = []
        all_track_idx = []
        all_orig_pos = []
        track_boundaries = []

        for track_idx, (pitches, iois, vels) in enumerate(zip(pitch_sequences, ioi_sequences, velocity_sequences)):
            if len(pitches) == 0:
                continue

            durs = duration_sequences[track_idx] if duration_sequences else [480] * len(pitches)
            onsets = onset_sequences[track_idx] if onset_sequences else list(range(0, len(pitches) * 480, 480))

            # Compute rhythm bucket for each note (relative to previous)
            for i, (p, ioi, v, d, o) in enumerate(zip(pitches, iois, vels, durs, onsets)):
                # Rhythm bucket: ratio of this IOI to previous (or default for first)
                if i == 0:
                    rhythm_bucket = 7  # Default to 1.0 ratio for first note
                else:
                    rhythm_bucket = compute_rhythm_bucket(iois[i-1], ioi)

                # Velocity bucket: relative to previous (or default for first)
                if i == 0:
                    velocity_bucket = 3  # Default to "same" for first note
                else:
                    velocity_bucket = compute_velocity_bucket(vels[i-1], v)

                terminal_id = encode_terminal(rhythm_bucket, velocity_bucket)

                all_terminal.append(terminal_id)
                all_pitch.append(p)  # Full MIDI pitch, not pitch class!
                all_ioi.append(ioi)
                all_velocity.append(v)
                all_duration.append(d)
                all_onset.append(o)
                all_track_idx.append(track_idx)
                all_orig_pos.append(i)

            # Add separator
            track_boundaries.append(len(all_terminal))
            all_terminal.append(separator)
            all_pitch.append(0)
            all_ioi.append(0)
            all_velocity.append(0)
            all_duration.append(0)
            all_onset.append(0)
            all_track_idx.append(-1)
            all_orig_pos.append(-1)

        # Convert to tensors
        seq_terminal = torch.tensor(all_terminal, dtype=torch.int64, device=self.device)
        seq_pitch = torch.tensor(all_pitch, dtype=torch.int64, device=self.device)
        seq_ioi = torch.tensor(all_ioi, dtype=torch.int64, device=self.device)
        seq_velocity = torch.tensor(all_velocity, dtype=torch.int64, device=self.device)
        seq_duration = torch.tensor(all_duration, dtype=torch.int64, device=self.device)
        seq_onset = torch.tensor(all_onset, dtype=torch.int64, device=self.device)
        seq_track_idx = torch.tensor(all_track_idx, dtype=torch.int64, device=self.device)
        seq_orig_pos = torch.tensor(all_orig_pos, dtype=torch.int64, device=self.device)
        seq_last_orig_pos = torch.tensor(all_orig_pos, dtype=torch.int64, device=self.device)

        original_length = len(seq_terminal) - len(pitch_sequences)  # Exclude separators

        # For tracking MIDI pitches (for interval computation)
        seq_first_pitch = seq_pitch.clone()  # First pitch of symbol
        seq_last_pitch = seq_pitch.clone()   # Last pitch of symbol

        # Rule storage
        rule_contours = torch.zeros((self.max_rules, 3), dtype=torch.int64, device=self.device)
        rule_children = torch.full((self.max_rules, 2), -1, dtype=torch.int64, device=self.device)
        rule_counts = torch.zeros(self.max_rules, dtype=torch.int64, device=self.device)
        rule_occurrences = {}

        # Symbol properties (for hierarchical patterns)
        max_symbols = n_terminals + 1 + self.max_rules
        symbol_base_ioi = torch.full((max_symbols,), 480, dtype=torch.int64, device=self.device)
        symbol_base_velocity = torch.full((max_symbols,), 64, dtype=torch.int64, device=self.device)

        next_rule_id = n_terminals + 1  # Start after separator
        n_rules = 0

        # Main Re-Pair loop
        iteration = 0
        while iteration < self.max_rules:
            # ===== STEP 1: Find most frequent normalized triple =====
            n = len(seq_terminal)
            if n < 2:
                break

            # Get consecutive pairs
            left_terminal = seq_terminal[:-1]
            right_terminal = seq_terminal[1:]

            # Valid pairs (not crossing separators)
            valid = (left_terminal != separator) & (right_terminal != separator)

            if not valid.any():
                break

            # Get MIDI pitches for interval computation
            left_last_pitch = seq_last_pitch[:-1][valid]
            right_first_pitch = seq_first_pitch[1:][valid]

            # Compute SIGNED pitch interval
            pitch_intervals = right_first_pitch - left_last_pitch

            # Get symbol base IOI/velocity for rhythm/velocity buckets
            left_t = left_terminal[valid]
            right_t = right_terminal[valid]
            left_base_ioi = symbol_base_ioi[left_t]
            right_base_ioi = symbol_base_ioi[right_t]
            left_base_v = symbol_base_velocity[left_t]
            right_base_v = symbol_base_velocity[right_t]

            # Compute normalized buckets
            rhythm_buckets = self._compute_rhythm_buckets_vectorized(left_base_ioi, right_base_ioi)
            velocity_buckets = self._compute_velocity_buckets_vectorized(left_base_v, right_base_v)

            # Encode normalized triple
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

            # Store rule definition
            rule_contours[n_rules, 0] = best_interval
            rule_contours[n_rules, 1] = best_rhythm
            rule_contours[n_rules, 2] = best_velocity
            rule_counts[n_rules] = max_triple_count

            # Check if this is a hierarchical pattern
            triple_mask = triple_ids == best_triple
            valid_indices = torch.where(valid)[0]
            pair_positions = valid_indices[triple_mask]

            # Determine left/right symbol types for this rule
            expected_left = -1
            expected_right = -1
            is_hierarchical = False

            if len(pair_positions) > 0:
                first_left = seq_terminal[pair_positions[0]].item()
                first_right = seq_terminal[pair_positions[0] + 1].item()
                is_hierarchical = (first_left > n_terminals) or (first_right > n_terminals)

                if is_hierarchical:
                    rule_children[n_rules, 0] = first_left
                    rule_children[n_rules, 1] = first_right
                    expected_left = first_left
                    expected_right = first_right
                else:
                    expected_left = -2  # Require terminal
                    expected_right = -2

            n_rules += 1

            # ===== STEP 3: Replace ALL pairs matching this triple =====
            (seq_terminal, seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset,
             seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch,
             occurrences) = self._replace_triple_gpu(
                seq_terminal, seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset,
                seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch,
                symbol_base_ioi, symbol_base_velocity,
                best_interval, best_rhythm, best_velocity,
                rule_id, separator, n_terminals,
                expected_left=expected_left, expected_right=expected_right
            )

            # Store occurrences (each has first_pitch!)
            rule_occurrences[rule_id] = occurrences

            # Set base properties for new symbol
            symbol_base_ioi[rule_id] = 480
            symbol_base_velocity[rule_id] = 64

            # Progress
            if self.verbose and (iteration % self.progress_every == 0 or iteration < 10):
                elapsed = time.time() - start_time
                interval_name = self._interval_name(best_interval)
                hier_marker = " [HIER]" if is_hierarchical else ""
                print(f"  [iter {iteration}] R{rule_id}=contour({best_interval}:{interval_name}, r{best_rhythm}, v{best_velocity}){hier_marker} "
                      f"count={max_triple_count:,}, len={len(seq_terminal):,}, time={elapsed:.1f}s", flush=True)

            iteration += 1

        # Remove separators from final sequence
        mask = seq_terminal != separator
        final_seq = seq_terminal[mask]

        elapsed = time.time() - start_time

        if self.verbose:
            ratio = original_length / len(final_seq) if len(final_seq) > 0 else 1.0
            hier_count = sum(1 for i in range(n_rules) if rule_children[i, 0].item() >= 0)
            print(f"[RePair-PureContour] Done: {n_rules} rules ({hier_count} hierarchical), "
                  f"compression {ratio:.2f}x, time {elapsed:.1f}s", flush=True)

        return PureContourGrammar(
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
        seq_terminal: torch.Tensor,
        seq_pitch: torch.Tensor,
        seq_ioi: torch.Tensor,
        seq_velocity: torch.Tensor,
        seq_duration: torch.Tensor,
        seq_onset: torch.Tensor,
        seq_track_idx: torch.Tensor,
        seq_orig_pos: torch.Tensor,
        seq_last_orig_pos: torch.Tensor,
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
        """
        n = len(seq_terminal)
        if n < 2:
            return (seq_terminal, seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset,
                    seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, [])

        # Compute intervals and buckets for all pairs
        left_terminal = seq_terminal[:-1]
        right_terminal = seq_terminal[1:]

        valid = (left_terminal != separator) & (right_terminal != separator)

        if not valid.any():
            return (seq_terminal, seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset,
                    seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, [])

        # Compute intervals
        left_last = seq_last_pitch[:-1]
        right_first = seq_first_pitch[1:]
        intervals = right_first - left_last

        # Compute rhythm/velocity buckets
        left_t = left_terminal[:-1]
        right_t = right_terminal[:-1]
        left_base_ioi = symbol_base_ioi[left_t[valid[:-1]]] if len(valid) > 1 else symbol_base_ioi[left_t]
        right_base_ioi = symbol_base_ioi[right_t[valid[:-1]]] if len(valid) > 1 else symbol_base_ioi[right_t]

        # For simplicity, recompute buckets for valid pairs
        left_base_ioi = symbol_base_ioi[left_terminal[valid]]
        right_base_ioi = symbol_base_ioi[right_terminal[valid]]
        left_base_v = symbol_base_velocity[left_terminal[valid]]
        right_base_v = symbol_base_velocity[right_terminal[valid]]

        rhythm_buckets = self._compute_rhythm_buckets_vectorized(left_base_ioi, right_base_ioi)
        velocity_buckets = self._compute_velocity_buckets_vectorized(left_base_v, right_base_v)

        # Match target triple
        interval_match = intervals[valid] == target_interval
        rhythm_match = rhythm_buckets == target_rhythm
        velocity_match = velocity_buckets == target_velocity
        triple_match = interval_match & rhythm_match & velocity_match

        # Apply symbol type constraints
        valid_indices = torch.where(valid)[0]
        if expected_left >= 0:  # Exact hierarchical match
            left_type_match = left_terminal[valid_indices] == expected_left
            right_type_match = right_terminal[valid_indices] == expected_right
            triple_match = triple_match & left_type_match & right_type_match
        elif expected_left == -2:  # Both must be terminals
            left_is_terminal = left_terminal[valid_indices] < n_terminals
            right_is_terminal = right_terminal[valid_indices] < n_terminals
            triple_match = triple_match & left_is_terminal & right_is_terminal

        if not triple_match.any():
            return (seq_terminal, seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset,
                    seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch, [])

        # Get positions to replace
        match_positions = valid_indices[triple_match]

        # Greedy non-overlapping selection
        keep_mask = torch.ones(len(match_positions), dtype=torch.bool, device=self.device)
        positions_cpu = match_positions.cpu().numpy()
        for i in range(1, len(positions_cpu)):
            if positions_cpu[i] <= positions_cpu[i-1] + 1:
                if keep_mask[i-1]:
                    keep_mask[i] = False

        final_positions = match_positions[keep_mask]

        # Collect occurrences with first_pitch, timing data, and per-note ratios
        occurrences = []
        for pos in final_positions.cpu().numpy():
            track_idx = seq_track_idx[pos].item()
            orig_pos_start = seq_orig_pos[pos].item()
            orig_pos_end = seq_last_orig_pos[pos + 1].item()

            occ = {
                'track_idx': track_idx,
                'orig_pos': orig_pos_start,
                'last_orig_pos': orig_pos_end,
                'first_pitch': seq_first_pitch[pos].item(),  # MIDI pitch 0-127
                'last_pitch': seq_last_pitch[pos + 1].item(),
                'onset_time': seq_onset[pos].item(),
                'first_ioi': seq_ioi[pos].item(),  # IOI of first note (for playback)
            }

            # Compute per-note ratios from original track data if available
            if hasattr(self, 'original_tracks') and track_idx >= 0:
                track_data = self.original_tracks.get(track_idx, {})
                if track_data and orig_pos_start >= 0 and orig_pos_end >= orig_pos_start:
                    # Extract the note range for this pattern
                    n_notes = orig_pos_end - orig_pos_start + 1

                    # Get original IOIs, velocities, durations for this range
                    track_iois = track_data.get('iois', [])
                    track_vels = track_data.get('velocities', [])
                    track_durs = track_data.get('durations', [])

                    # Compute rhythm ratios (IOI[i+1] / IOI[i]) - successive ratios
                    # This is scale-invariant: same pattern shape at any tempo
                    if len(track_iois) > orig_pos_end:
                        iois_slice = track_iois[orig_pos_start:orig_pos_end + 1]
                        if len(iois_slice) > 1:
                            rhythm_ratios = []
                            for i in range(len(iois_slice) - 1):
                                if iois_slice[i] > 0:
                                    rhythm_ratios.append(iois_slice[i + 1] / iois_slice[i])
                                else:
                                    rhythm_ratios.append(1.0)
                            occ['rhythm_ratios'] = rhythm_ratios

                    # Compute velocity ratios (relative to first note)
                    if len(track_vels) > orig_pos_end:
                        vels_slice = track_vels[orig_pos_start:orig_pos_end + 1]
                        if len(vels_slice) > 0 and vels_slice[0] > 0:
                            velocity_ratios = [v / vels_slice[0] for v in vels_slice]
                            occ['velocity_ratios'] = velocity_ratios

                    # Compute duration ratios (relative to first note)
                    if len(track_durs) > orig_pos_end:
                        durs_slice = track_durs[orig_pos_start:orig_pos_end + 1]
                        if len(durs_slice) > 0 and durs_slice[0] > 0:
                            duration_ratios = [d / durs_slice[0] for d in durs_slice]
                            occ['duration_ratios'] = duration_ratios

            occurrences.append(occ)

        # Create new sequence with replacements
        remove_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        remove_mask[final_positions + 1] = True  # Remove right element of each pair

        # Build replacement values
        new_terminal = seq_terminal.clone()
        new_terminal[final_positions] = new_symbol

        # Update first/last pitch for combined symbols
        new_first_pitch = seq_first_pitch.clone()
        new_last_pitch = seq_last_pitch.clone()
        new_last_pitch[final_positions] = seq_last_pitch[final_positions + 1]

        # Update last_orig_pos for combined symbols
        new_last_orig_pos = seq_last_orig_pos.clone()
        new_last_orig_pos[final_positions] = seq_last_orig_pos[final_positions + 1]

        # Apply removal mask
        keep_mask = ~remove_mask
        seq_terminal = new_terminal[keep_mask]
        seq_pitch = seq_pitch[keep_mask]
        seq_ioi = seq_ioi[keep_mask]
        seq_velocity = seq_velocity[keep_mask]
        seq_duration = seq_duration[keep_mask]
        seq_onset = seq_onset[keep_mask]
        seq_track_idx = seq_track_idx[keep_mask]
        seq_orig_pos = seq_orig_pos[keep_mask]
        seq_last_orig_pos = new_last_orig_pos[keep_mask]
        seq_first_pitch = new_first_pitch[keep_mask]
        seq_last_pitch = new_last_pitch[keep_mask]

        return (seq_terminal, seq_pitch, seq_ioi, seq_velocity, seq_duration, seq_onset,
                seq_track_idx, seq_orig_pos, seq_last_orig_pos, seq_first_pitch, seq_last_pitch,
                occurrences)

    def _interval_name(self, interval: int) -> str:
        """Human-readable interval name."""
        names = {
            0: "unison", 1: "m2", 2: "M2", 3: "m3", 4: "M3", 5: "P4",
            6: "tritone", 7: "P5", 8: "m6", 9: "M6", 10: "m7", 11: "M7", 12: "octave"
        }
        abs_interval = abs(interval)
        if abs_interval in names:
            name = names[abs_interval]
        elif abs_interval > 12:
            octaves = abs_interval // 12
            remainder = abs_interval % 12
            name = f"{octaves}oct+{names.get(remainder, str(remainder))}"
        else:
            name = str(abs_interval)
        return f"-{name}" if interval < 0 else f"+{name}"


def build_pure_contour_grammar(
    tracks: list,
    device: str = 'cuda',
    min_pair_count: int = 2,
    max_rules: int = 10000,
    verbose: bool = True,
) -> PureContourGrammar:
    """
    Build pure contour grammar from FactoredTrack list.
    """
    pitch_sequences = []
    ioi_sequences = []
    velocity_sequences = []
    duration_sequences = []
    onset_sequences = []
    track_info = []

    for track in tracks:
        # Use full MIDI pitch, not pitch class
        midi_pitches = [pc + oct * 12 for pc, oct in zip(track.pitch_classes, track.octaves)]
        pitch_sequences.append(midi_pitches)
        ioi_sequences.append(list(track.rhythm_ioi))
        velocity_sequences.append(list(track.velocities))
        duration_sequences.append(list(track.durations))
        onset_sequences.append(list(track.onsets))
        track_info.append({
            'piece_id': track.piece_id,
            'track_id': track.track_id,
            'gm_program': track.gm_program,
            'is_drum': getattr(track, 'is_drum', False),  # From MIDI channel 9
        })

    compressor = RePairPureContour(
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
    )
