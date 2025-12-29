#!/usr/bin/env python3
"""
v9 Dataset: ACTUAL Paired Formant-Shifted Data + Sox-Shifted Unpaired

Two training modes mixed 50/50:
1. Formant-shifted PAIRS: Same recording corrupted/natural → L1 loss
2. Sox-shifted UNPAIRED: Different recordings → distribution matching

The formant pairs teach: "preserve this exact content while fixing formants"
The sox data teaches: "this is what sox artifacts look like, push toward natural"
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


class FormantCorrectorDatasetPaired(Dataset):
    """
    Dataset with ACTUAL paired formant-shifted data.

    Each formant pair contains:
    - shifted_latent: formant-shifted version of recording
    - natural_latent: original natural recording (SAME recording!)

    Since they're the same recording, we use L1 loss.
    """

    def __init__(
        self,
        paired_manifest_path: str,
        sox_manifest_path: str = '/mnt/msdd2/pitchshift_v7_overlapped/manifest.json',
        window_frames: int = 16,
        samples_per_epoch: int = 10000,
        paired_ratio: float = 0.5,  # 50% paired, 50% sox
        pitch_tolerance: float = 3.0,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.paired_ratio = paired_ratio
        self.pitch_tolerance = pitch_tolerance

        # Load paired formant-shifted data
        print(f"Loading paired manifest: {paired_manifest_path}")
        with open(paired_manifest_path) as f:
            paired_manifest = json.load(f)
        self.paired_entries = paired_manifest.get('pairs', [])

        # Load sox-shifted unpaired data
        self.sox_shifted = []
        self.sox_targets = []
        if os.path.exists(sox_manifest_path):
            print(f"Loading sox manifest: {sox_manifest_path}")
            with open(sox_manifest_path) as f:
                sox_manifest = json.load(f)
            self.sox_shifted = sox_manifest.get('shifted_entries', [])
            self.sox_targets = sox_manifest.get('low_entries', [])

        # Build pitch lookup for sox targets
        self.sox_targets_by_pitch = {}
        for entry in self.sox_targets:
            midi = int(round(entry.get('median_midi', 0)))
            if midi not in self.sox_targets_by_pitch:
                self.sox_targets_by_pitch[midi] = []
            self.sox_targets_by_pitch[midi].append(entry)

        print(f"Loaded:")
        print(f"  Paired entries: {len(self.paired_entries)}")
        print(f"  Sox shifted: {len(self.sox_shifted)}")
        print(f"  Sox targets: {len(self.sox_targets)}")

    def _load_latent(self, path: str) -> Optional[torch.Tensor]:
        """Load a latent tensor from path."""
        try:
            path = fix_path(path)
            if not os.path.exists(path):
                return None

            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latent', data.get('latents'))
            else:
                latent = data

            if latent is None:
                return None

            if latent.dim() == 4:
                latent = latent.squeeze(0)

            return latent.float()
        except Exception:
            return None

    def _load_pair(self, path: str) -> Optional[Dict]:
        """Load a paired entry (contains both shifted and natural)."""
        try:
            path = fix_path(path)
            if not os.path.exists(path):
                return None

            data = torch.load(path, map_location='cpu', weights_only=False)
            return data
        except Exception:
            return None

    def _crop_window(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        """Crop a random window from latent."""
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape
        if T < self.window_frames:
            return None

        start = random.randint(0, T - self.window_frames)
        return latent[:, :, start:start + self.window_frames]

    def _crop_window_aligned(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
    ) -> Optional[tuple]:
        """Crop aligned windows from two latents (for paired data)."""
        if latent1.dim() == 4:
            latent1 = latent1.squeeze(0)
        if latent2.dim() == 4:
            latent2 = latent2.squeeze(0)

        T1 = latent1.shape[-1]
        T2 = latent2.shape[-1]
        T = min(T1, T2)

        if T < self.window_frames:
            return None

        start = random.randint(0, T - self.window_frames)
        crop1 = latent1[:, :, start:start + self.window_frames]
        crop2 = latent2[:, :, start:start + self.window_frames]
        return crop1, crop2

    def _find_sox_target(self, target_midi: float) -> Optional[Dict]:
        """Find a sox target with similar pitch."""
        target_int = int(round(target_midi))

        for offset in range(int(self.pitch_tolerance) + 1):
            for sign in [0, 1, -1]:
                pitch = target_int + sign * offset
                if pitch in self.sox_targets_by_pitch and self.sox_targets_by_pitch[pitch]:
                    return random.choice(self.sox_targets_by_pitch[pitch])
        return None

    def _get_paired_sample(self) -> Optional[Dict]:
        """Get a formant-shifted paired sample (L1 loss)."""
        if not self.paired_entries:
            return None

        entry = random.choice(self.paired_entries)
        pair_data = self._load_pair(entry['pair_path'])

        if pair_data is None:
            return None

        # Support both old format (shifted_latent/natural_latent) and new format (shifted/natural)
        shifted = pair_data.get('shifted')
        if shifted is None:
            shifted = pair_data.get('shifted_latent')
        natural = pair_data.get('natural')
        if natural is None:
            natural = pair_data.get('natural_latent')

        if shifted is None or natural is None:
            return None

        # Crop aligned windows (SAME position in both!)
        crops = self._crop_window_aligned(shifted, natural)
        if crops is None:
            return None

        shifted_crop, natural_crop = crops

        # Get direction from pair_data or entry (new format stores in pair_data)
        direction = pair_data.get('direction', entry.get('direction', 0))

        return {
            'input': shifted_crop,
            'target': natural_crop,
            'source_group': 0,  # Not using groups for now
            'direction': direction,
            'is_paired': True,  # Use L1 loss
            'valid': True,
        }

    def _get_sox_sample(self) -> Optional[Dict]:
        """Get a sox-shifted unpaired sample (distribution matching)."""
        if not self.sox_shifted or not self.sox_targets:
            return None

        entry = random.choice(self.sox_shifted)

        input_latent = self._load_latent(entry['latent_path'])
        if input_latent is None:
            return None

        # Find target with matching pitch (different recording!)
        target_midi = entry.get('target_midi', 50)
        target_entry = self._find_sox_target(target_midi)
        if target_entry is None:
            return None

        target_latent = self._load_latent(target_entry['latent_path'])
        if target_latent is None:
            return None

        # Crop windows independently (not aligned - different recordings)
        input_crop = self._crop_window(input_latent)
        target_crop = self._crop_window(target_latent)

        if input_crop is None or target_crop is None:
            return None

        # Determine source group from original pitch
        original_midi = entry.get('original_midi', 65)
        if original_midi >= 77:
            source_group = 2  # Group 3 → index 2
        elif original_midi >= 65:
            source_group = 1  # Group 2 → index 1
        else:
            source_group = 0  # Group 1 → index 0

        return {
            'input': input_crop,
            'target': target_crop,
            'source_group': source_group,
            'direction': 0,  # Sox always shifts down
            'is_paired': False,  # Use distribution matching
            'valid': True,
        }

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        for _ in range(50):
            # Decide sample type based on ratio
            use_paired = random.random() < self.paired_ratio

            if use_paired:
                sample = self._get_paired_sample()
            else:
                sample = self._get_sox_sample()

            if sample is not None:
                return sample

        # Return invalid sample
        return {
            'input': torch.zeros(8, 16, self.window_frames),
            'target': torch.zeros(8, 16, self.window_frames),
            'source_group': 0,
            'direction': 0,
            'is_paired': False,
            'valid': False,
        }


if __name__ == "__main__":
    # Test dataset
    ds = FormantCorrectorDatasetPaired(
        paired_manifest_path='/mnt/msdd2/pitchshift_v9_paired/manifest.json',
        samples_per_epoch=100,
    )

    valid_paired = 0
    valid_sox = 0
    for i in range(20):
        sample = ds[i]
        if sample['valid']:
            if sample['is_paired']:
                valid_paired += 1
            else:
                valid_sox += 1
        print(f"{i}: valid={sample['valid']}, paired={sample['is_paired']}, "
              f"group={sample['source_group']}, dir={sample['direction']}")

    print(f"\nValid: {valid_paired} paired, {valid_sox} sox")
