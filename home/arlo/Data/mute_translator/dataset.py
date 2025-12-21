"""
Dataset for Mute Translator Training

Handles loading of dry and muted trumpet latents from manifest.

v3: Attack-focused sampling with onset/amp conditioning for envelope learning.
"""

import os
import json
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple
from pathlib import Path


def fix_mount_path(path: str) -> str:
    """Fix mount path from msdd to msdd2 if needed."""
    if path and '/mnt/msdd/' in path:
        return path.replace('/mnt/msdd/', '/mnt/msdd2/')
    return path


# Verified muted trumpet file patterns
# NOTE: Use session-specific paths to avoid false positives from files with same basename
VERIFIED_MUTED_PATTERNS = [
    'TPTMUTE',  # Original GT
    # RyotaSasaki_Friends session - harmon mutes
    'Trumpets 2_03.wav', 'Trumpets 2_04.wav', 'Trumpets 2_05.wav',
    'Trumpets 2.01_07.wav', 'Trumpets 2.02_08.wav', 'Trumpets 2_01.wav',
    'trumpet 4_01.wav', 'trumpet 4.01_02.wav', 'trumpet 4.02_03.wav',
    'Trumpet.29_45.wav', 'Trumpet.30_46.wav',
    'Trumpet 3_01.wav', 'Trumpet 3.01_02.wav',
    # Tiril Jackson session - use full session path to avoid false positives
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.37_40.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.38_41.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.39_42.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.26_31.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.28_32.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.09_09.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.25_30.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.02_02.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.04_04.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.05_05.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.07_07.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.10_10.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.12_14.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.13_15.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.15_17.wav',
    'Tiril Jackson_HORNS OD_6.15.2025/Audio Files/Trumpet.16_18.wav',
    # Joav session
    '414 trumpet.02_05.wav', '414 trumpet.03_06.wav',
    # UnderTheSakuraSky session
    'Trumpet2 - Yunho.01_04.wav',
    'UnderTheSakuraSky_48bit/Audio Files/trumpet 3.05_11.wav',
    'UnderTheSakuraSky_48bit/Audio Files/trumpet 3.07_13.wav',
    # RyotaSasaki additional muted
    'Trumpets 2.35_83.wav', 'Trumpets 2.36_84.wav',
    # StephenGuerra_Spiraling_TPTDub session - muted dubs
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_15.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_16.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_18.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_36.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_38.wav',
    'StephenGuerra_Spiraling_TPTDub/Audio Files/Trumpet 1_40.wav',
    # DevonGates_DaveThePotter session
    'DevonGates_DaveThePotter/Audio Files/Trumpet.03_10.wav',
    # MP340_Experimental2_JadeFaria session
    'MP340_Experimental2_JadeFaria/Audio Files/Trumpet(2).00_02.wav',
]

# Files to exclude (not trumpet or bad quality)
EXCLUDE_FILES = [
    'GUNMech_Tikka',  # Gun sound effects mislabeled as trumpet
    'TPT 1_02.wav',   # Background audio only
    'TPT .45_47.wav', # Vocal file mislabeled as trumpet
]


def is_muted_file(audio_path: str) -> bool:
    """Check if audio path is a verified muted trumpet file."""
    return any(pattern in audio_path for pattern in VERIFIED_MUTED_PATTERNS)


def is_excluded_file(audio_path: str) -> bool:
    """Check if file should be excluded."""
    return any(excl in audio_path for excl in EXCLUDE_FILES)


def detect_onsets_in_latent(latent: torch.Tensor, threshold: float = 0.5) -> List[int]:
    """
    Detect onset positions in latent space using energy derivative.

    Args:
        latent: [C, H, T] tensor
        threshold: relative threshold for onset detection

    Returns:
        List of frame indices where onsets are detected
    """
    if latent.dim() == 4:
        latent = latent.squeeze(0)

    C, H, T = latent.shape

    # Compute energy per frame (sum over channels and height)
    energy = (latent ** 2).sum(dim=(0, 1))  # [T]

    # Compute energy derivative (positive = getting louder = onset)
    energy_diff = torch.diff(energy, prepend=energy[:1])

    # Normalize
    max_diff = energy_diff.max()
    if max_diff > 0:
        energy_diff = energy_diff / max_diff

    # Find peaks above threshold
    onsets = []
    for i in range(1, T - 1):
        if energy_diff[i] > threshold:
            # Local maximum check
            if energy_diff[i] >= energy_diff[i-1] and energy_diff[i] >= energy_diff[i+1]:
                onsets.append(i)

    return onsets


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
        attack_focus_ratio: float = 0.5,  # Fraction of samples to focus on attacks
    ):
        self.window_frames = window_frames
        self.min_frames = min_frames
        self.samples_per_epoch = samples_per_epoch
        self.extract_latents_fn = extract_latents_fn
        self.attack_focus_ratio = attack_focus_ratio

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
            latent_path = fix_mount_path(entry.get('latent_path', ''))
            if latent_path and os.path.exists(latent_path):
                valid.append(entry)
            elif self.extract_latents_fn is not None:
                # We can extract on the fly
                valid.append(entry)
        return valid

    def _load_latent(self, entry: Dict) -> Optional[torch.Tensor]:
        """Load latent from disk or extract it."""
        latent_path = fix_mount_path(entry.get('latent_path', ''))

        # Check cache
        if latent_path in self._latent_cache:
            return self._latent_cache[latent_path]

        if os.path.exists(latent_path):
            try:
                latent = torch.load(latent_path, map_location='cpu')
                # Handle different storage formats
                if isinstance(latent, dict):
                    latent = latent.get('latents', latent.get('latent', latent.get('z', None)))
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

    def _load_conditioning(self, entry: Dict, cond_type: str) -> Optional[np.ndarray]:
        """Load onset or amp conditioning data."""
        cond_paths = entry.get('conditioning_paths', {})
        cond_path = fix_mount_path(cond_paths.get(cond_type, ''))

        if cond_path and os.path.exists(cond_path):
            try:
                return np.load(cond_path)
            except Exception as e:
                return None
        return None

    def _random_window(
        self,
        latent: torch.Tensor,
        onset_cond: Optional[np.ndarray] = None,
        amp_cond: Optional[np.ndarray] = None,
        attack_focused: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], bool, int]:
        """
        Extract a window from the latent and conditioning.

        Args:
            latent: [C, H, T] or [1, C, H, T] tensor
            onset_cond: [T_cond] onset conditioning array (may be different length)
            amp_cond: [T_cond] amplitude conditioning array
            attack_focused: If True, try to center window on an onset

        Returns:
            (latent_window, onset_window, amp_window, has_attack, start_frame)
        """
        # latent shape: [C, H, T] or [1, C, H, T]
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape

        if T <= self.window_frames:
            # Pad if too short
            pad_amount = self.window_frames - T
            latent = torch.nn.functional.pad(latent, (0, pad_amount))
            onset_window = torch.zeros(self.window_frames) if onset_cond is None else None
            amp_window = torch.zeros(self.window_frames) if amp_cond is None else None
            if onset_cond is not None:
                onset_window = torch.zeros(self.window_frames)
                onset_window[:len(onset_cond)] = torch.from_numpy(onset_cond[:self.window_frames]).float()
            if amp_cond is not None:
                amp_window = torch.zeros(self.window_frames)
                amp_window[:len(amp_cond)] = torch.from_numpy(amp_cond[:self.window_frames]).float()
            return latent, onset_window, amp_window, False, 0

        has_attack = False
        start = 0

        if attack_focused:
            # Use real onset data if available, otherwise detect from latent
            if onset_cond is not None and len(onset_cond) > 0:
                # Find frames with onsets in the conditioning data
                # Scale conditioning length to latent length
                scale = len(onset_cond) / T
                onset_frames = np.where(onset_cond > 0.5)[0]
                if len(onset_frames) > 0:
                    # Pick a random onset and scale to latent frames
                    onset_idx = random.choice(onset_frames)
                    onset_latent_frame = int(onset_idx / scale)

                    # Center window on onset (attack near beginning of window)
                    attack_offset = int(self.window_frames * 0.2)
                    start = max(0, onset_latent_frame - attack_offset)
                    start = min(start, T - self.window_frames)
                    has_attack = True
                else:
                    start = random.randint(0, T - self.window_frames)
            else:
                # Detect onsets from latent
                onsets = detect_onsets_in_latent(latent, threshold=0.3)
                if len(onsets) > 0:
                    onset = random.choice(onsets)
                    attack_offset = int(self.window_frames * 0.2)
                    start = max(0, onset - attack_offset)
                    start = min(start, T - self.window_frames)
                    has_attack = True
                else:
                    start = random.randint(0, T - self.window_frames)
        else:
            # Random start
            start = random.randint(0, T - self.window_frames)

        # Extract latent window
        latent_window = latent[:, :, start:start + self.window_frames]

        # Extract conditioning windows (need to scale to match latent)
        onset_window = None
        amp_window = None

        if onset_cond is not None and len(onset_cond) > 0:
            scale = len(onset_cond) / T
            cond_start = int(start * scale)
            cond_end = int((start + self.window_frames) * scale)
            onset_slice = onset_cond[cond_start:cond_end]
            # Resample to window_frames length
            onset_window = torch.from_numpy(
                np.interp(
                    np.linspace(0, len(onset_slice)-1, self.window_frames),
                    np.arange(len(onset_slice)),
                    onset_slice
                )
            ).float()

        if amp_cond is not None and len(amp_cond) > 0:
            scale = len(amp_cond) / T
            cond_start = int(start * scale)
            cond_end = int((start + self.window_frames) * scale)
            amp_slice = amp_cond[cond_start:cond_end]
            # Resample to window_frames length
            amp_window = torch.from_numpy(
                np.interp(
                    np.linspace(0, len(amp_slice)-1, self.window_frames),
                    np.arange(len(amp_slice)),
                    amp_slice
                )
            ).float()

        return latent_window, onset_window, amp_window, has_attack, start

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample random dry and muted entries
        dry_entry = random.choice(self.dry_entries)
        muted_entry = random.choice(self.muted_entries)

        # Load latents
        dry_latent = self._load_latent(dry_entry)
        muted_latent = self._load_latent(muted_entry)

        # Load conditioning for dry entry
        dry_onsets = self._load_conditioning(dry_entry, 'onsets')
        dry_amp = self._load_conditioning(dry_entry, 'amp')

        # Retry if loading failed
        retries = 0
        while dry_latent is None and retries < 10:
            dry_entry = random.choice(self.dry_entries)
            dry_latent = self._load_latent(dry_entry)
            dry_onsets = self._load_conditioning(dry_entry, 'onsets')
            dry_amp = self._load_conditioning(dry_entry, 'amp')
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
                'dry_onsets': torch.zeros(self.window_frames),
                'dry_amp': torch.zeros(self.window_frames),
                'valid': torch.tensor(False),
                'has_attack': torch.tensor(False),
            }

        # Decide whether to focus on attacks for this sample
        attack_focused = random.random() < self.attack_focus_ratio

        # Extract windows with conditioning
        dry_window, dry_onset_win, dry_amp_win, dry_has_attack, _ = self._random_window(
            dry_latent, dry_onsets, dry_amp, attack_focused=attack_focused
        )
        muted_window, _, _, muted_has_attack, _ = self._random_window(
            muted_latent, attack_focused=attack_focused
        )

        # Default conditioning if not loaded
        if dry_onset_win is None:
            dry_onset_win = torch.zeros(self.window_frames)
        if dry_amp_win is None:
            dry_amp_win = torch.zeros(self.window_frames)

        return {
            'dry_latent': dry_window,
            'muted_latent': muted_window,
            'dry_onsets': dry_onset_win,
            'dry_amp': dry_amp_win,
            'valid': torch.tensor(True),
            'has_attack': torch.tensor(dry_has_attack or muted_has_attack),
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
            latent = latent.get('latents', latent.get('latent', latent.get('z')))

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
