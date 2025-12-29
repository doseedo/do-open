#!/usr/bin/env python3
"""
V4 Dataset - SIMPLE (No Conditioning)

Mute-style dataset for artifact correction:
- Source: Precomputed DSP-shifted latent (has sox artifacts)
- Target: Real latent at destination pitch (clean)

Just returns (source, target) pairs. No shift conditioning.

FIXES:
1. Use round() not floor division for group mapping
2. Match targets by median_midi (within tolerance)
3. shift=0 uses identity pairs (same segment)
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
from torch.utils.data import Dataset


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    replacements = [
        ('/mnt/msdd/', '/mnt/msdd2/'),
        ('/home/arlo/gcs-bucket/', '/mnt/gcs-bucket/'),
    ]
    for old, new in replacements:
        if old in path:
            path = path.replace(old, new)
    return path


def normalize_path(path: str) -> str:
    """Normalize path for comparison (apply fix_path and get basename pattern)."""
    path = fix_path(path)
    basename = os.path.basename(path)
    for ext in ['.pt', '.wav', '.f0.npy', '.amp.npy']:
        if basename.endswith(ext):
            basename = basename[:-len(ext)]
            break
    return basename


class PitchShiftCorrectionDatasetSimple(Dataset):
    """
    Simple dataset for pitch-shift artifact correction.

    Uses precomputed DSP-shifted latents as source (has artifacts)
    and finds real latents at destination pitch as target (clean).

    Key behaviors:
    - shift=0: Returns identity pairs (same segment as source and target)
    - shift!=0: Matches target by median_midi (within tolerance)
    - Uses round() for group mapping, not floor division
    """

    def __init__(
        self,
        shifted_manifest: str,
        segments_json: str,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
        preload_latents: bool = True,
        flagged_json: str = None,
        pitch_tolerance: float = 1.0,  # Semitones tolerance for pitch matching
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.pitch_tolerance = pitch_tolerance

        # Load flagged recordings to exclude
        self.flagged_latents: set = set()
        if flagged_json and os.path.exists(flagged_json):
            print(f"Loading flagged recordings from: {flagged_json}")
            with open(flagged_json, 'r') as f:
                flagged_data = json.load(f)
            for rec in flagged_data.get('flagged_recordings', []):
                latent_path = rec.get('latent_path', '')
                if latent_path:
                    self.flagged_latents.add(normalize_path(latent_path))
            print(f"  Loaded {len(self.flagged_latents)} flagged recordings to exclude")

        # Load precomputed shifted latents manifest
        print(f"Loading shifted manifest from: {shifted_manifest}")
        with open(shifted_manifest, 'r') as f:
            shifted_data = json.load(f)

        self.available_shifts = shifted_data.get('shifts', [])
        all_entries = shifted_data.get('entries', [])

        # Filter out flagged entries and store with full metadata
        self.entries = []
        filtered_count = 0
        for entry in all_entries:
            source_path = entry.get('source_latent_path', '')
            if normalize_path(source_path) in self.flagged_latents:
                filtered_count += 1
                continue
            self.entries.append(entry)

        print(f"  Loaded {len(self.entries)} entries with shifts: {self.available_shifts}")
        if filtered_count > 0:
            print(f"  Filtered out {filtered_count} flagged entries")

        # Load original segments for target lookups
        print(f"Loading segments from: {segments_json}")
        with open(segments_json, 'r') as f:
            data = json.load(f)

        self.config = data['config']
        self.group_size = self.config['group_size']
        self.base_pitch = self.config['base_pitch']
        self.num_groups = self.config['num_groups']

        # Organize segments by group for target lookup (excluding flagged)
        self.segments_by_group: Dict[int, List[Dict]] = defaultdict(list)
        self.all_segments: List[Dict] = []  # Flat list for pitch-based lookup
        target_filtered = 0

        for group_id, segments in data['segments_by_group'].items():
            group_int = int(group_id)
            for seg in segments:
                if normalize_path(seg['latent_path']) in self.flagged_latents:
                    target_filtered += 1
                    continue
                seg['group'] = group_int
                self.segments_by_group[group_int].append(seg)
                self.all_segments.append(seg)

        if target_filtered > 0:
            print(f"  Filtered out {target_filtered} flagged target segments")

        self.valid_groups = sorted(self.segments_by_group.keys())
        print(f"  {len(self.valid_groups)} valid groups, {len(self.all_segments)} total segments")

        # Preload latents
        self.latent_cache: Dict[str, torch.Tensor] = {}
        if preload_latents:
            self._preload_latents()

        # Build list of valid pairs with full metadata
        self.valid_pairs = []
        self.identity_pairs = []  # Separate list for shift=0

        for entry in self.entries:
            source_group = entry.get('group', 0)
            source_midi = entry.get('median_midi', 0)
            source_latent_path = fix_path(entry.get('source_latent_path', ''))
            source_start = entry.get('start_frame', 0)
            source_end = entry.get('end_frame', 0)

            for shift_str, shifted_path in entry.get('shifted_latents', {}).items():
                shift = int(shift_str)
                shifted_path = fix_path(shifted_path)

                if not os.path.exists(shifted_path):
                    continue

                if shift == 0:
                    # Identity pair: source and target are the same segment
                    self.identity_pairs.append({
                        'shifted_path': shifted_path,
                        'shift': 0,
                        'source_midi': source_midi,
                        'source_latent_path': source_latent_path,
                        'source_start': source_start,
                        'source_end': source_end,
                        'is_identity': True,
                    })
                else:
                    # Non-identity: need to find matching target
                    # FIX #1: Use round() not floor division
                    delta_groups = int(round(shift / self.group_size))
                    target_group = source_group + delta_groups
                    target_group = max(0, min(target_group, self.num_groups - 1))

                    # Target pitch after shift
                    target_midi = source_midi + shift

                    if target_group in self.segments_by_group:
                        self.valid_pairs.append({
                            'shifted_path': shifted_path,
                            'shift': shift,
                            'source_midi': source_midi,
                            'target_midi': target_midi,
                            'target_group': target_group,
                            'is_identity': False,
                        })

        print(f"  {len(self.identity_pairs)} identity pairs (shift=0)")
        print(f"  {len(self.valid_pairs)} non-identity pairs")

    def _preload_latents(self):
        """Preload latents into memory."""
        print("Preloading latents...")

        # Preload shifted latents
        for entry in self.entries:
            for shift_str, path in entry.get('shifted_latents', {}).items():
                path = fix_path(path)
                if os.path.exists(path) and path not in self.latent_cache:
                    try:
                        data = torch.load(path, map_location='cpu', weights_only=True)
                        if isinstance(data, dict):
                            latent = data.get('latent', data.get('latents', data.get('z', None)))
                        else:
                            latent = data
                        if latent is not None:
                            if latent.dim() == 4:
                                latent = latent.squeeze(0)
                            self.latent_cache[path] = latent
                    except:
                        pass

        # Preload original latents for targets
        for seg in self.all_segments:
            path = fix_path(seg['latent_path'])
            if os.path.exists(path) and path not in self.latent_cache:
                try:
                    latent = torch.load(path, map_location='cpu', weights_only=True)
                    if isinstance(latent, dict):
                        latent = latent.get('latents', latent.get('latent', latent.get('z', None)))
                    if latent is not None:
                        if latent.dim() == 4:
                            latent = latent.squeeze(0)
                        self.latent_cache[path] = latent
                except:
                    pass

        print(f"  Cached {len(self.latent_cache)} latents")

    def _load_latent(self, path: str) -> Optional[torch.Tensor]:
        """Load a latent tensor."""
        path = fix_path(path)

        if path in self.latent_cache:
            return self.latent_cache[path]

        if not os.path.exists(path):
            return None

        try:
            data = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(data, dict):
                latent = data.get('latent', data.get('latents', data.get('z', None)))
            else:
                latent = data
            if latent is not None and latent.dim() == 4:
                latent = latent.squeeze(0)
            return latent
        except:
            return None

    def _get_window_at_position(
        self,
        latent: torch.Tensor,
        start: int,
        end: int,
        window_start: int = None,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """
        Extract a window from latent within [start, end] bounds.

        If window_start is provided, use that position (for aligned windows).
        Otherwise pick random position.

        Returns (window, actual_window_start) or (None, -1) on failure.
        """
        T = latent.shape[-1]
        available = end - start

        if available < self.window_frames:
            return None, -1

        if window_start is not None:
            # Use specified position (clamped to valid range)
            ws = max(start, min(window_start, end - self.window_frames))
        else:
            # Random position within segment bounds
            max_start = end - self.window_frames
            ws = random.randint(start, max_start) if max_start > start else start

        if ws + self.window_frames > T:
            return None, -1

        return latent[:, :, ws:ws + self.window_frames], ws

    def _find_pitch_matched_target(
        self,
        target_midi: float,
        target_group: int,
    ) -> Optional[Dict]:
        """
        Find a target segment that matches the target pitch within tolerance.

        FIX #2: Match by median_midi, not just group.
        """
        candidates = self.segments_by_group.get(target_group, [])
        if not candidates:
            return None

        # Filter by pitch tolerance
        matched = [
            seg for seg in candidates
            if abs(seg.get('median_midi', 0) - target_midi) <= self.pitch_tolerance
        ]

        if matched:
            return random.choice(matched)

        # Fallback: if no exact match, find closest
        closest = min(candidates, key=lambda s: abs(s.get('median_midi', 0) - target_midi))
        return closest

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            source_latent: [C, H, T] - DSP-shifted (has artifacts)
            target_latent: [C, H, T] - Real clean reference
            valid: bool
        """
        max_attempts = 50

        # Mix identity and non-identity pairs
        # Use ~10% identity pairs to teach "don't change clean input"
        use_identity = random.random() < 0.1 and len(self.identity_pairs) > 0

        for _ in range(max_attempts):
            if use_identity:
                # FIX #3: shift=0 uses identity pairs
                pair = random.choice(self.identity_pairs)

                # Load shifted latent (segment only, length ≈ source_end - source_start)
                shifted_latent = self._load_latent(pair['shifted_path'])
                if shifted_latent is None:
                    continue

                # Load source latent (full file)
                source_latent = self._load_latent(pair['source_latent_path'])
                if source_latent is None:
                    continue

                # Shifted latent length (segment-only)
                shifted_len = shifted_latent.shape[-1]
                seg_start = pair['source_start']
                seg_end = pair['source_end']
                seg_len = seg_end - seg_start

                # Both should be similar length, but shifted might differ slightly due to encoding
                # Use the shorter of the two for safety
                usable_len = min(shifted_len, seg_len)
                if usable_len < self.window_frames:
                    continue

                # Sample RELATIVE position within segment bounds
                max_rel = usable_len - self.window_frames
                ws_rel = random.randint(0, max_rel) if max_rel > 0 else 0

                # Extract aligned windows:
                # - shifted_latent is segment-only, so use ws_rel directly
                # - source_latent is full file, so offset by seg_start
                source_window = shifted_latent[:, :, ws_rel:ws_rel + self.window_frames]
                target_window = source_latent[:, :, seg_start + ws_rel:seg_start + ws_rel + self.window_frames]

                if source_window.shape[-1] != self.window_frames:
                    continue
                if target_window.shape[-1] != self.window_frames:
                    continue

            else:
                # Non-identity: find pitch-matched target
                pair = random.choice(self.valid_pairs)

                # Load shifted latent (source with artifacts)
                shifted_latent = self._load_latent(pair['shifted_path'])
                if shifted_latent is None:
                    continue

                source_window, _ = self._get_window_at_position(
                    shifted_latent,
                    start=0,
                    end=shifted_latent.shape[-1],
                )
                if source_window is None:
                    continue

                # FIX #2: Find pitch-matched target
                target_seg = self._find_pitch_matched_target(
                    pair['target_midi'],
                    pair['target_group'],
                )
                if target_seg is None:
                    continue

                target_latent = self._load_latent(target_seg['latent_path'])
                if target_latent is None:
                    continue

                target_window, _ = self._get_window_at_position(
                    target_latent,
                    start=target_seg['start_frame'],
                    end=target_seg['end_frame'],
                )
                if target_window is None:
                    continue

            # Validate shapes match
            if source_window.shape != target_window.shape:
                continue

            # Get shift value and pitch info for conditioning
            shift_val = pair.get('shift', 0)
            source_midi = pair.get('source_midi', 60.0)
            target_midi = source_midi + shift_val  # Target pitch = source + shift

            return {
                'source_latent': source_window,
                'target_latent': target_window,
                'shift': torch.tensor(shift_val, dtype=torch.float32),
                'source_pitch': torch.tensor(source_midi, dtype=torch.float32),
                'target_pitch': torch.tensor(target_midi, dtype=torch.float32),
                'valid': torch.tensor(True),
            }

        # Failed to find valid pair
        return {
            'source_latent': torch.zeros(8, 16, self.window_frames),
            'target_latent': torch.zeros(8, 16, self.window_frames),
            'shift': torch.tensor(0, dtype=torch.float32),
            'source_pitch': torch.tensor(60.0, dtype=torch.float32),
            'target_pitch': torch.tensor(60.0, dtype=torch.float32),
            'valid': torch.tensor(False),
        }
