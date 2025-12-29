"""
Register-Aware Pitch Shift Dataset V2

DISTRIBUTION MATCHING approach (like mute translator).

1. Group latents by their dominant MIDI pitch
2. For training: sample source from pitch A, target from pitch B
3. Model learns to match DISTRIBUTION of pitch B while preserving content from source

This is fundamentally different from additive offsets - we sample REAL latents
at the target pitch as targets, not synthetic offsets.
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


class RegisterTransferDatasetV2(Dataset):
    """
    Dataset for V2 register timbre conversion using DISTRIBUTION MATCHING.

    Like mute_translator: the target is a REAL sample from the target pitch,
    NOT a synthetic offset. The model learns to match the distribution of
    "what pitch B sounds like" while preserving content from the source.

    Training pair:
    - source_latent: Real latent at source pitch A
    - target_latent: Real latent at target pitch B (DIFFERENT recording!)
    - source_pitch, target_pitch: For conditioning

    The model learns: f(source, src_pitch, tgt_pitch) → matches distribution of tgt_pitch
    """

    def __init__(
        self,
        manifest_path: str,
        instrument: str = 'trumpet',
        window_frames: int = 64,
        shift_range: Tuple[int, int] = (-12, 12),
        samples_per_epoch: int = 10000,
        min_samples_per_pitch: int = 5,
        preload_latents: bool = True,
    ):
        self.manifest_path = manifest_path
        self.instrument = instrument.lower()
        self.window_frames = window_frames
        self.shift_range = shift_range
        self.samples_per_epoch = samples_per_epoch
        self.min_samples_per_pitch = min_samples_per_pitch

        # Load manifest and filter entries
        print(f"Loading manifest: {manifest_path}")
        with open(manifest_path, 'r') as f:
            data = json.load(f)

        # Filter for instrument and non-muted only
        self.entries = []
        for entry in data:
            # Skip muted entries
            if entry.get('is_muted', False):
                continue

            # Check instrument via sub_group or path
            sub_group = entry.get('sub_group', '').lower()
            path = entry.get('latent_path', '').lower()
            if sub_group == self.instrument or self.instrument in path:
                if 'latent_path' in entry:
                    entry['latent_path'] = fix_mount_path(entry['latent_path'])
                if 'conditioning_paths' in entry:
                    for k, v in entry['conditioning_paths'].items():
                        entry['conditioning_paths'][k] = fix_mount_path(v)
                self.entries.append(entry)

        print(f"Loaded {len(self.entries)} {instrument} entries")

        # Latent cache
        self._latent_cache: Dict[str, torch.Tensor] = {}

        # Build pitch index - map each entry to its dominant pitch
        self.pitch_to_entries = self._build_pitch_index()

        # Get valid pitches (those with enough samples)
        self.valid_pitches = [
            p for p, entries in self.pitch_to_entries.items()
            if len(entries) >= min_samples_per_pitch
        ]
        print(f"Valid pitches with >={min_samples_per_pitch} samples: {len(self.valid_pitches)}")
        if self.valid_pitches:
            print(f"Pitch range: {min(self.valid_pitches)} - {max(self.valid_pitches)}")

        # Preload latents
        if preload_latents:
            self._preload_latents()

    def _build_pitch_index(self) -> Dict[int, List[Dict]]:
        """Build index mapping MIDI pitch to entries at that pitch."""
        pitch_to_entries = defaultdict(list)

        for entry in tqdm(self.entries, desc="Building pitch index"):
            cond = entry.get('conditioning_paths', {})
            f0_path = cond.get('f0', '') or ''
            f0_path = fix_mount_path(f0_path)

            dominant_pitch = self._get_dominant_pitch(f0_path)
            if dominant_pitch is not None:
                pitch_to_entries[dominant_pitch].append(entry)
                entry['_dominant_pitch'] = dominant_pitch

        return dict(pitch_to_entries)

    def _get_dominant_pitch(self, f0_path: str) -> Optional[int]:
        """Get dominant MIDI pitch from f0 file."""
        if not f0_path or not os.path.exists(f0_path):
            return None
        try:
            f0 = np.load(f0_path)
            valid_f0 = f0[(f0 > 0) & ~np.isnan(f0)]
            if len(valid_f0) == 0:
                return None
            # Convert to MIDI and get median (dominant pitch)
            midi = 69 + 12 * np.log2(valid_f0 / 440.0)
            dominant = int(np.median(midi).round())
            if 48 <= dominant <= 96:
                return dominant
            return None
        except Exception:
            return None

    def _preload_latents(self):
        """Preload all latents to RAM."""
        print(f"Preloading latents to RAM...")
        loaded = 0
        for entry in tqdm(self.entries, desc="Loading latents"):
            latent_path = entry.get('latent_path', '')
            if not latent_path or latent_path in self._latent_cache:
                continue
            if os.path.exists(latent_path):
                try:
                    latent = torch.load(latent_path, map_location='cpu', weights_only=True)
                    if isinstance(latent, dict):
                        latent = latent.get('latents', latent.get('latent', latent.get('z')))
                    if latent is not None:
                        if latent.dim() == 4:
                            latent = latent.squeeze(0)
                        self._latent_cache[latent_path] = latent
                        loaded += 1
                except Exception:
                    pass
        print(f"Preloaded {loaded} latents")

    def _load_latent(self, entry: Dict) -> Optional[torch.Tensor]:
        """Load latent from cache."""
        latent_path = entry.get('latent_path', '')
        return self._latent_cache.get(latent_path)

    def _random_window(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract random window from latent."""
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape

        if T <= self.window_frames:
            pad_amount = self.window_frames - T
            latent = F.pad(latent, (0, pad_amount))
            return latent

        start = random.randint(0, T - self.window_frames)
        return latent[:, :, start:start + self.window_frames]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get training sample using DISTRIBUTION MATCHING.

        Returns:
            - source_latent: Real latent at source pitch [C, H, T]
            - target_latent: Real latent at target pitch [C, H, T] (DIFFERENT recording!)
            - source_pitch: Source MIDI pitch
            - target_pitch: Target MIDI pitch
            - valid: Whether sample is valid
        """
        if len(self.valid_pitches) < 2:
            return self._fallback_sample()

        # Sample source pitch
        source_pitch = random.choice(self.valid_pitches)

        # Sample target pitch (different from source, within shift range)
        valid_targets = [
            p for p in self.valid_pitches
            if p != source_pitch and self.shift_range[0] <= (p - source_pitch) <= self.shift_range[1]
        ]

        if not valid_targets:
            # Fallback: just pick any different pitch
            valid_targets = [p for p in self.valid_pitches if p != source_pitch]

        if not valid_targets:
            return self._fallback_sample()

        target_pitch = random.choice(valid_targets)

        # Sample random entry at source pitch
        source_entry = random.choice(self.pitch_to_entries[source_pitch])
        source_latent = self._load_latent(source_entry)

        # Sample random entry at target pitch (DIFFERENT recording - distribution matching!)
        target_entry = random.choice(self.pitch_to_entries[target_pitch])
        target_latent = self._load_latent(target_entry)

        # Retry if loading failed
        retries = 0
        while (source_latent is None or target_latent is None) and retries < 10:
            if source_latent is None:
                source_entry = random.choice(self.pitch_to_entries[source_pitch])
                source_latent = self._load_latent(source_entry)
            if target_latent is None:
                target_entry = random.choice(self.pitch_to_entries[target_pitch])
                target_latent = self._load_latent(target_entry)
            retries += 1

        if source_latent is None or target_latent is None:
            return self._fallback_sample()

        # Get random windows
        source_window = self._random_window(source_latent)
        target_window = self._random_window(target_latent)

        return {
            'source_latent': source_window,  # Real latent at source pitch
            'target_latent': target_window,  # Real latent at target pitch (distribution target!)
            'source_pitch': torch.tensor(source_pitch, dtype=torch.long),
            'target_pitch': torch.tensor(target_pitch, dtype=torch.long),
            'valid': torch.tensor(True),
        }

    def _fallback_sample(self) -> Dict[str, torch.Tensor]:
        """Return a fallback sample when loading fails."""
        C, H, T = 8, 16, self.window_frames
        return {
            'source_latent': torch.zeros(C, H, T),
            'target_latent': torch.zeros(C, H, T),
            'source_pitch': torch.tensor(60, dtype=torch.long),
            'target_pitch': torch.tensor(60, dtype=torch.long),
            'valid': torch.tensor(False),
        }


if __name__ == "__main__":
    # Test dataset
    dataset = RegisterTransferDatasetV2(
        manifest_path="/home/arlo/Data.backup/final_training_manifest_final.json",
        instrument="trumpet",
        samples_per_epoch=100,
    )

    print(f"\nDataset size: {len(dataset)}")

    sample = dataset[0]
    print(f"\nSample keys: {sample.keys()}")
    print(f"Source latent shape: {sample['source_latent'].shape}")
    print(f"Target latent shape: {sample['target_latent'].shape}")
    print(f"Source pitch: {sample['source_pitch'].item()}")
    print(f"Target pitch: {sample['target_pitch'].item()}")
    print(f"Valid: {sample['valid'].item()}")
