"""
Range-Group Based Pitch Shift Dataset V3

Like mute_translator but with range groups instead of muted/dry categories.

Training approach:
- Source: random clip from ANY range group
- Target: random clip from TARGET range group (different recording!)
- Conditioning: target group ID

The model learns: f(source, target_group) → characteristics of target_group
Just like mute_translator learned: f(dry) → muted characteristics
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm


def fix_mount_path(path: str) -> str:
    """Fix paths that might have wrong mount points."""
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


class RangeGroupDataset(Dataset):
    """
    Dataset for V3 range-group based pitch shift.

    Like mute_translator:
    - Source: random clip from any group
    - Target: random clip from target group (different recording!)
    - Conditioning: target group ID

    The model learns the characteristics of each range group.
    """

    def __init__(
        self,
        segments_json: str,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
        min_segments_per_group: int = 10,
        preload_latents: bool = True,
        attack_focus_ratio: float = 0.3,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.min_segments_per_group = min_segments_per_group
        self.attack_focus_ratio = attack_focus_ratio

        # Load segmented data
        print(f"Loading segments: {segments_json}")
        with open(segments_json, 'r') as f:
            data = json.load(f)

        self.config = data['config']
        self.num_groups = self.config['num_groups']
        self.group_size = self.config['group_size']
        self.base_pitch = self.config['base_pitch']

        print(f"Config: {self.num_groups} groups, size={self.group_size}, base={self.base_pitch}")

        # Load segments by group
        self.segments_by_group: Dict[int, List[Dict]] = {}
        for group_str, segments in data['segments_by_group'].items():
            group = int(group_str)
            if len(segments) >= min_segments_per_group:
                self.segments_by_group[group] = segments

        self.valid_groups = sorted(self.segments_by_group.keys())
        print(f"Valid groups (>={min_segments_per_group} segments): {self.valid_groups}")

        for g in self.valid_groups:
            pitch_start = self.base_pitch + g * self.group_size
            pitch_end = pitch_start + self.group_size
            print(f"  Group {g} (MIDI {pitch_start}-{pitch_end}): {len(self.segments_by_group[g])} segments")

        # All segments flat list (for source sampling)
        self.all_segments = []
        for group, segments in self.segments_by_group.items():
            for seg in segments:
                seg['_group'] = group
                self.all_segments.append(seg)

        print(f"Total segments: {len(self.all_segments)}")

        # Latent cache
        self._latent_cache: Dict[str, torch.Tensor] = {}

        if preload_latents:
            self._preload_latents()

    def _preload_latents(self):
        """Preload all latents to RAM."""
        print("Preloading latents...")
        unique_paths = set(seg['latent_path'] for seg in self.all_segments)
        loaded = 0

        for path in tqdm(unique_paths, desc="Loading latents"):
            if path in self._latent_cache:
                continue
            path = fix_mount_path(path)
            if os.path.exists(path):
                try:
                    latent = torch.load(path, map_location='cpu', weights_only=True)
                    if isinstance(latent, dict):
                        latent = latent.get('latents', latent.get('latent', latent.get('z')))
                    if latent is not None:
                        if latent.dim() == 4:
                            latent = latent.squeeze(0)
                        self._latent_cache[path] = latent
                        loaded += 1
                except Exception as e:
                    pass

        print(f"Preloaded {loaded} latents")

    def _load_segment_latent(self, segment: Dict) -> Optional[torch.Tensor]:
        """Load and extract segment from latent."""
        path = fix_mount_path(segment['latent_path'])
        latent = self._latent_cache.get(path)

        if latent is None:
            return None

        start = segment['start_frame']
        end = segment['end_frame']

        # Extract segment
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape

        # Clamp to valid range
        start = max(0, min(start, T - 1))
        end = max(start + 1, min(end, T))

        segment_latent = latent[:, :, start:end]

        return segment_latent

    def _random_window(
        self,
        latent: torch.Tensor,
        attack_focused: bool = False,
    ) -> torch.Tensor:
        """Extract random window from segment latent."""
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape

        if T <= self.window_frames:
            pad_amount = self.window_frames - T
            latent = F.pad(latent, (0, pad_amount))
            return latent

        if attack_focused:
            # Focus on beginning (attack region)
            max_start = min(T // 4, T - self.window_frames)
            start = random.randint(0, max(0, max_start))
        else:
            start = random.randint(0, T - self.window_frames)

        return latent[:, :, start:start + self.window_frames]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample.

        Returns:
            - source_latent: Clip from any range group [C, H, T]
            - target_latent: Clip from target range group [C, H, T] (different recording!)
            - source_group: Source group ID
            - target_group: Target group ID (the conditioning!)
            - valid: Whether sample is valid
        """
        if len(self.valid_groups) < 2:
            return self._fallback_sample()

        # Sample source segment (from any group)
        source_segment = random.choice(self.all_segments)
        source_group = source_segment['_group']

        # Sample target group (can be same or different)
        target_group = random.choice(self.valid_groups)

        # Sample target segment from target group (DIFFERENT recording!)
        target_segment = random.choice(self.segments_by_group[target_group])

        # Ensure different recordings if same group
        retries = 0
        while (target_segment['latent_path'] == source_segment['latent_path'] and
               retries < 10 and len(self.segments_by_group[target_group]) > 1):
            target_segment = random.choice(self.segments_by_group[target_group])
            retries += 1

        # Load latents
        source_latent = self._load_segment_latent(source_segment)
        target_latent = self._load_segment_latent(target_segment)

        # Retry if loading failed
        retries = 0
        while (source_latent is None or target_latent is None) and retries < 10:
            if source_latent is None:
                source_segment = random.choice(self.all_segments)
                source_group = source_segment['_group']
                source_latent = self._load_segment_latent(source_segment)

            if target_latent is None:
                target_segment = random.choice(self.segments_by_group[target_group])
                target_latent = self._load_segment_latent(target_segment)

            retries += 1

        if source_latent is None or target_latent is None:
            return self._fallback_sample()

        # Decide attack focus
        attack_focused = random.random() < self.attack_focus_ratio

        # Get random windows
        source_window = self._random_window(source_latent, attack_focused)
        target_window = self._random_window(target_latent, attack_focused)

        return {
            'source_latent': source_window,
            'target_latent': target_window,
            'source_group': torch.tensor(source_group, dtype=torch.long),
            'target_group': torch.tensor(target_group, dtype=torch.long),
            'valid': torch.tensor(True),
        }

    def _fallback_sample(self) -> Dict[str, torch.Tensor]:
        """Return fallback sample when loading fails."""
        C, H, T = 8, 16, self.window_frames
        return {
            'source_latent': torch.zeros(C, H, T),
            'target_latent': torch.zeros(C, H, T),
            'source_group': torch.tensor(0, dtype=torch.long),
            'target_group': torch.tensor(0, dtype=torch.long),
            'valid': torch.tensor(False),
        }

    def get_group_pitch_range(self, group: int) -> Tuple[int, int]:
        """Get MIDI pitch range for a group."""
        start = self.base_pitch + group * self.group_size
        end = start + self.group_size
        return start, end


if __name__ == "__main__":
    # Test dataset
    import sys

    if len(sys.argv) < 2:
        print("Usage: python dataset_v3.py <segments.json>")
        sys.exit(1)

    dataset = RangeGroupDataset(
        segments_json=sys.argv[1],
        samples_per_epoch=100,
        preload_latents=True,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Valid groups: {dataset.valid_groups}")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Source latent shape: {sample['source_latent'].shape}")
    print(f"Target latent shape: {sample['target_latent'].shape}")
    print(f"Source group: {sample['source_group'].item()}")
    print(f"Target group: {sample['target_group'].item()}")
    print(f"Valid: {sample['valid'].item()}")
