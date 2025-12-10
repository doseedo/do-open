"""
Dataset for Mute Translator Training

Handles loading of dry and muted trumpet latents from manifest.
"""

import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path


# Verified muted trumpet file patterns
VERIFIED_MUTED_PATTERNS = [
    'TPTMUTE',
    'Trumpets 2_03.wav', 'Trumpets 2_04.wav', 'Trumpets 2_05.wav',
    'Trumpets 2.01_07.wav', 'Trumpets 2.02_08.wav', 'Trumpets 2_01.wav',
    'trumpet 4_01.wav', 'trumpet 4.01_02.wav', 'trumpet 4.02_03.wav',
    'Trumpet.29_45.wav', 'Trumpet.30_46.wav',
    'Trumpet 3_01.wav', 'Trumpet 3.01_02.wav',
    'Trumpet.37_40.wav', 'Trumpet.38_41.wav', 'Trumpet.39_42.wav',
    'Trumpet.26_31.wav', 'Trumpet.28_32.wav',
    '414 trumpet.02_05.wav', '414 trumpet.03_06.wav',
    'Trumpet2 - Yunho.01_04.wav',
    'Trumpets 2.35_83.wav', 'Trumpets 2.36_84.wav',
]

# Files to exclude
EXCLUDE_FILES = ['GUNMech_Tikka', 'TPT 1_02.wav']


def is_muted_file(audio_path: str) -> bool:
    """Check if audio path is a verified muted trumpet file."""
    return any(pattern in audio_path for pattern in VERIFIED_MUTED_PATTERNS)


def is_excluded_file(audio_path: str) -> bool:
    """Check if file should be excluded."""
    return any(excl in audio_path for excl in EXCLUDE_FILES)


def load_manifest(manifest_path: str) -> Tuple[List[Dict], List[Dict]]:
    """
    Load manifest and split into dry and muted entries.

    Returns:
        (dry_entries, muted_entries)
    """
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    dry_entries = []
    muted_entries = []

    for entry in manifest:
        if entry.get('sub_group') != 'trumpet':
            continue

        audio_path = entry.get('audio_path', '')
        if not audio_path or is_excluded_file(audio_path):
            continue

        if is_muted_file(audio_path):
            muted_entries.append(entry)
        else:
            dry_entries.append(entry)

    return dry_entries, muted_entries


class MuteTranslatorDataset(Dataset):
    """
    Dataset for training the mute translator.

    For each sample, returns:
    - dry_latent: A random dry trumpet latent
    - muted_latent: A random muted trumpet latent (for distribution matching)

    The translator learns to map dry → muted distribution.
    """

    def __init__(
        self,
        manifest_path: str,
        window_frames: int = 128,  # ~3 seconds at DCAE rate
        min_frames: int = 32,
        samples_per_epoch: int = 10000,
        extract_latents_fn=None,  # Function to extract latents if not on disk
    ):
        self.window_frames = window_frames
        self.min_frames = min_frames
        self.samples_per_epoch = samples_per_epoch
        self.extract_latents_fn = extract_latents_fn

        # Load manifest
        self.dry_entries, self.muted_entries = load_manifest(manifest_path)

        print(f"Loaded {len(self.dry_entries)} dry entries")
        print(f"Loaded {len(self.muted_entries)} muted entries")

        # Filter to entries with valid latent paths
        self.dry_entries = self._filter_valid_entries(self.dry_entries)
        self.muted_entries = self._filter_valid_entries(self.muted_entries)

        print(f"Valid dry entries: {len(self.dry_entries)}")
        print(f"Valid muted entries: {len(self.muted_entries)}")

        if len(self.muted_entries) == 0:
            raise ValueError("No valid muted entries found!")

        # Cache for loaded latents
        self._latent_cache = {}

    def _filter_valid_entries(self, entries: List[Dict]) -> List[Dict]:
        """Filter to entries with valid latent paths."""
        valid = []
        for entry in entries:
            latent_path = entry.get('latent_path', '')
            if latent_path and os.path.exists(latent_path):
                valid.append(entry)
            elif self.extract_latents_fn is not None:
                # We can extract on the fly
                valid.append(entry)
        return valid

    def _load_latent(self, entry: Dict) -> Optional[torch.Tensor]:
        """Load latent from disk or extract it."""
        latent_path = entry.get('latent_path', '')

        # Check cache
        if latent_path in self._latent_cache:
            return self._latent_cache[latent_path]

        if os.path.exists(latent_path):
            try:
                latent = torch.load(latent_path, map_location='cpu')
                # Handle different storage formats
                if isinstance(latent, dict):
                    latent = latent.get('latent', latent.get('z', None))
                if latent is not None:
                    self._latent_cache[latent_path] = latent
                return latent
            except Exception as e:
                print(f"Error loading {latent_path}: {e}")
                return None
        elif self.extract_latents_fn is not None:
            # Extract on the fly
            audio_path = entry.get('audio_path', '')
            if os.path.exists(audio_path):
                latent = self.extract_latents_fn(audio_path)
                self._latent_cache[latent_path] = latent
                return latent

        return None

    def _random_window(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract a random window from the latent."""
        # latent shape: [C, H, T] or [1, C, H, T]
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape

        if T <= self.window_frames:
            # Pad if too short
            pad_amount = self.window_frames - T
            latent = torch.nn.functional.pad(latent, (0, pad_amount))
            return latent

        # Random start
        start = random.randint(0, T - self.window_frames)
        return latent[:, :, start:start + self.window_frames]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample random dry and muted entries
        dry_entry = random.choice(self.dry_entries)
        muted_entry = random.choice(self.muted_entries)

        # Load latents
        dry_latent = self._load_latent(dry_entry)
        muted_latent = self._load_latent(muted_entry)

        # Retry if loading failed
        retries = 0
        while dry_latent is None and retries < 10:
            dry_entry = random.choice(self.dry_entries)
            dry_latent = self._load_latent(dry_entry)
            retries += 1

        retries = 0
        while muted_latent is None and retries < 10:
            muted_entry = random.choice(self.muted_entries)
            muted_latent = self._load_latent(muted_entry)
            retries += 1

        if dry_latent is None or muted_latent is None:
            # Return zeros as fallback
            return {
                'dry_latent': torch.zeros(8, 16, self.window_frames),
                'muted_latent': torch.zeros(8, 16, self.window_frames),
                'valid': torch.tensor(False),
            }

        # Extract random windows
        dry_window = self._random_window(dry_latent)
        muted_window = self._random_window(muted_latent)

        return {
            'dry_latent': dry_window,
            'muted_latent': muted_window,
            'valid': torch.tensor(True),
        }


class MutedLatentDataset(Dataset):
    """
    Dataset that only loads muted latents.
    Used for computing muted distribution statistics.
    """

    def __init__(self, manifest_path: str, window_frames: int = 128):
        self.window_frames = window_frames
        _, self.muted_entries = load_manifest(manifest_path)

        # Filter to valid
        self.muted_entries = [
            e for e in self.muted_entries
            if os.path.exists(e.get('latent_path', ''))
        ]

        print(f"MutedLatentDataset: {len(self.muted_entries)} entries")

    def __len__(self) -> int:
        return len(self.muted_entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.muted_entries[idx]
        latent_path = entry['latent_path']

        latent = torch.load(latent_path, map_location='cpu')
        if isinstance(latent, dict):
            latent = latent.get('latent', latent.get('z'))

        if latent.dim() == 4:
            latent = latent.squeeze(0)

        return {
            'latent': latent,
            'path': latent_path,
        }


def create_dataloader(
    manifest_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    window_frames: int = 128,
    samples_per_epoch: int = 10000,
) -> DataLoader:
    """Create training dataloader."""
    dataset = MuteTranslatorDataset(
        manifest_path=manifest_path,
        window_frames=window_frames,
        samples_per_epoch=samples_per_epoch,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Test dataset
    manifest = "/home/arlo/Data.backup/final_training_manifest_brass_only.json"

    print("Loading manifest...")
    dry, muted = load_manifest(manifest)
    print(f"Dry: {len(dry)}, Muted: {len(muted)}")

    print("\nMuted files:")
    for e in muted[:5]:
        print(f"  {os.path.basename(e['audio_path'])}")
