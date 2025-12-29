#!/usr/bin/env python3
"""
Register Translator Dataset

Loads high register and low register trumpet samples for distribution matching training.
Splits data by median F0 pitch into high and low register domains.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def fix_path(path: str) -> str:
    """Fix storage path mapping."""
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_manifest(manifest_path: str) -> List[Dict]:
    """Load and parse manifest JSON."""
    with open(manifest_path) as f:
        return json.load(f)


def get_median_pitch(f0_path: str) -> Optional[float]:
    """Get median MIDI pitch from F0 file."""
    try:
        f0 = np.load(f0_path)
        f0 = np.nan_to_num(f0, nan=0.0)
        f0_valid = f0[f0 > 20]  # Filter unvoiced
        if len(f0_valid) < 10:
            return None
        freq = np.median(f0_valid)
        midi = 12 * np.log2(freq / 440) + 69
        return midi
    except Exception:
        return None


class RegisterTranslatorDataset(Dataset):
    """
    Dataset for register translator training.

    Strategy:
    - Split trumpet samples into HIGH and LOW domains with a GAP in between
    - HIGH: samples with median pitch > high_threshold (e.g., > 70 = Bb4)
    - LOW: samples with median pitch < low_threshold (e.g., < 55 = G3)
    - GAP: samples in between are EXCLUDED (ensures genuine formant differences)
    - Training: input from HIGH domain, target from LOW domain
    - No pairing required - distribution matching only

    Key insight: Formant differences between MIDI 64 and 66 are negligible.
    You need to compare truly different registers (15+ semitone gap).

    This enables the mute translator approach: train on valid inputs,
    sox shift after inference.
    """

    def __init__(
        self,
        manifest_path: str,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        low_threshold: float = 55.0,   # Below this = LOW (G3)
        high_threshold: float = 70.0,  # Above this = HIGH (Bb4)
        instrument: str = 'trumpet',
        attack_focus_ratio: float = 0.0,  # Not used for register, but kept for API compat
    ):
        self.manifest_path = manifest_path
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.instrument = instrument

        # Load manifest
        manifest = load_manifest(manifest_path)

        # Separate into high and low register (with gap)
        self.high_entries = []
        self.low_entries = []

        print(f"Loading manifest with register gap...")
        print(f"  LOW: < {low_threshold} MIDI (G3)")
        print(f"  HIGH: > {high_threshold} MIDI (Bb4)")
        print(f"  GAP: {low_threshold}-{high_threshold} MIDI ({high_threshold - low_threshold:.0f} semitones) - EXCLUDED")
        n_skipped = 0
        n_high = 0
        n_low = 0
        n_gap = 0

        for entry in manifest:
            # Filter by instrument
            if entry.get('sub_group') != instrument:
                continue

            latent_path = fix_path(entry.get('latent_path', ''))
            cond_paths = entry.get('conditioning_paths') or {}
            f0_path = fix_path(cond_paths.get('f0', ''))

            if not latent_path or not f0_path:
                continue
            if not os.path.exists(latent_path) or not os.path.exists(f0_path):
                n_skipped += 1
                continue

            # Get median pitch
            median_midi = get_median_pitch(f0_path)
            if median_midi is None:
                n_skipped += 1
                continue

            entry_data = {
                'latent_path': latent_path,
                'f0_path': f0_path,
                'median_midi': median_midi,
            }

            # Three-way split with gap
            if median_midi > high_threshold:
                self.high_entries.append(entry_data)
                n_high += 1
            elif median_midi < low_threshold:
                self.low_entries.append(entry_data)
                n_low += 1
            else:
                # In the gap - skip
                n_gap += 1

        print(f"  HIGH register (>{high_threshold}): {len(self.high_entries)} samples")
        print(f"  LOW register (<{low_threshold}): {len(self.low_entries)} samples")
        print(f"  GAP (excluded): {n_gap} samples")
        print(f"  Skipped (missing files): {n_skipped}")

        if len(self.high_entries) == 0 or len(self.low_entries) == 0:
            raise ValueError(f"Need samples in both domains! HIGH: {len(self.high_entries)}, LOW: {len(self.low_entries)}")

        # Compute per-domain statistics
        high_pitches = [e['median_midi'] for e in self.high_entries]
        low_pitches = [e['median_midi'] for e in self.low_entries]
        print(f"  HIGH pitch range: {min(high_pitches):.1f} - {max(high_pitches):.1f} MIDI (mean: {np.mean(high_pitches):.1f})")
        print(f"  LOW pitch range: {min(low_pitches):.1f} - {max(low_pitches):.1f} MIDI (mean: {np.mean(low_pitches):.1f})")
        print(f"  Effective gap: {min(high_pitches):.1f} - {max(low_pitches):.1f} = {min(high_pitches) - max(low_pitches):.1f} semitones")

    def _load_latent(self, entry: Dict) -> Optional[torch.Tensor]:
        """Load latent from path."""
        try:
            data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent.dim() == 4:
                latent = latent.squeeze(0)
            return latent
        except Exception as e:
            print(f"Error loading latent {entry['latent_path']}: {e}")
            return None

    def _random_window(self, latent: torch.Tensor) -> torch.Tensor:
        """Extract random window from latent."""
        T = latent.shape[-1]
        if T <= self.window_frames:
            # Pad if too short
            pad = self.window_frames - T
            latent = torch.nn.functional.pad(latent, (0, pad))
            return latent
        else:
            start = random.randint(0, T - self.window_frames)
            return latent[:, :, start:start + self.window_frames]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns a training pair:
        - high_latent: window from HIGH register sample (input)
        - low_latent: window from LOW register sample (target distribution)
        """
        # Sample random entries
        high_entry = random.choice(self.high_entries)
        low_entry = random.choice(self.low_entries)

        # Load latents
        high_latent = self._load_latent(high_entry)
        low_latent = self._load_latent(low_entry)

        # Handle loading failures
        valid = True
        if high_latent is None:
            high_latent = torch.zeros(8, 16, self.window_frames)
            valid = False
        if low_latent is None:
            low_latent = torch.zeros(8, 16, self.window_frames)
            valid = False

        # Extract windows
        high_latent = self._random_window(high_latent)
        low_latent = self._random_window(low_latent)

        return {
            'high_latent': high_latent,
            'low_latent': low_latent,
            'valid': torch.tensor(valid, dtype=torch.bool),
            'high_midi': torch.tensor(high_entry['median_midi'], dtype=torch.float32),
            'low_midi': torch.tensor(low_entry['median_midi'], dtype=torch.float32),
        }


class RegisterTranslatorDatasetMultiBin(Dataset):
    """
    Multi-bin version: divides pitch range into multiple bins.

    Instead of binary high/low, divides into N bins.
    Training can then transfer between any pair of bins.

    This enables more granular register transfer:
    - Bin 0: very low (40-50 MIDI)
    - Bin 1: low (50-60 MIDI)
    - Bin 2: mid (60-70 MIDI)
    - Bin 3: high (70-80 MIDI)
    - Bin 4: very high (80-90 MIDI)
    """

    def __init__(
        self,
        manifest_path: str,
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        num_bins: int = 5,
        pitch_range: Tuple[float, float] = (40.0, 90.0),
        instrument: str = 'trumpet',
        source_bin: int = -1,  # -1 = random source bin each sample
        target_bin: int = 0,   # Target bin for distribution matching
    ):
        self.manifest_path = manifest_path
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.num_bins = num_bins
        self.pitch_range = pitch_range
        self.instrument = instrument
        self.source_bin = source_bin
        self.target_bin = target_bin

        # Compute bin boundaries
        self.bin_size = (pitch_range[1] - pitch_range[0]) / num_bins
        self.bin_edges = [pitch_range[0] + i * self.bin_size for i in range(num_bins + 1)]

        # Load manifest and bin entries
        manifest = load_manifest(manifest_path)
        self.bins = [[] for _ in range(num_bins)]

        print(f"Loading manifest and binning by pitch ({num_bins} bins)...")
        n_skipped = 0

        for entry in manifest:
            if entry.get('sub_group') != instrument:
                continue

            latent_path = fix_path(entry.get('latent_path', ''))
            f0_path = fix_path(entry.get('conditioning_paths', {}).get('f0', ''))

            if not latent_path or not f0_path:
                continue
            if not os.path.exists(latent_path) or not os.path.exists(f0_path):
                n_skipped += 1
                continue

            median_midi = get_median_pitch(f0_path)
            if median_midi is None:
                n_skipped += 1
                continue

            # Determine bin
            bin_idx = int((median_midi - pitch_range[0]) / self.bin_size)
            bin_idx = max(0, min(num_bins - 1, bin_idx))

            self.bins[bin_idx].append({
                'latent_path': latent_path,
                'f0_path': f0_path,
                'median_midi': median_midi,
            })

        print(f"  Skipped: {n_skipped}")
        for i, bin_entries in enumerate(self.bins):
            midi_low = self.bin_edges[i]
            midi_high = self.bin_edges[i + 1]
            print(f"  Bin {i} ({midi_low:.0f}-{midi_high:.0f} MIDI): {len(bin_entries)} samples")

    def _load_latent(self, entry: Dict) -> Optional[torch.Tensor]:
        try:
            data = torch.load(entry['latent_path'], map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent.dim() == 4:
                latent = latent.squeeze(0)
            return latent
        except Exception:
            return None

    def _random_window(self, latent: torch.Tensor) -> torch.Tensor:
        T = latent.shape[-1]
        if T <= self.window_frames:
            pad = self.window_frames - T
            latent = torch.nn.functional.pad(latent, (0, pad))
            return latent
        else:
            start = random.randint(0, T - self.window_frames)
            return latent[:, :, start:start + self.window_frames]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Select source bin
        if self.source_bin >= 0:
            src_bin = self.source_bin
        else:
            # Random source bin (excluding target)
            valid_bins = [i for i in range(self.num_bins) if i != self.target_bin and len(self.bins[i]) > 0]
            if not valid_bins:
                valid_bins = [i for i in range(self.num_bins) if len(self.bins[i]) > 0]
            src_bin = random.choice(valid_bins)

        # Sample from bins
        source_entry = random.choice(self.bins[src_bin])
        target_entry = random.choice(self.bins[self.target_bin])

        # Load latents
        source_latent = self._load_latent(source_entry)
        target_latent = self._load_latent(target_entry)

        valid = True
        if source_latent is None:
            source_latent = torch.zeros(8, 16, self.window_frames)
            valid = False
        if target_latent is None:
            target_latent = torch.zeros(8, 16, self.window_frames)
            valid = False

        source_latent = self._random_window(source_latent)
        target_latent = self._random_window(target_latent)

        return {
            'high_latent': source_latent,  # Keep name for trainer compatibility
            'low_latent': target_latent,
            'valid': torch.tensor(valid, dtype=torch.bool),
            'source_bin': torch.tensor(src_bin, dtype=torch.long),
            'target_bin': torch.tensor(self.target_bin, dtype=torch.long),
            'high_midi': torch.tensor(source_entry['median_midi'], dtype=torch.float32),
            'low_midi': torch.tensor(target_entry['median_midi'], dtype=torch.float32),
        }


if __name__ == "__main__":
    # Test dataset
    manifest = "/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json"

    print("Testing RegisterTranslatorDataset with 15-semitone gap...")
    dataset = RegisterTranslatorDataset(
        manifest_path=manifest,
        window_frames=128,
        samples_per_epoch=100,
        low_threshold=55.0,   # G3
        high_threshold=70.0,  # Bb4
    )

    sample = dataset[0]
    print(f"  high_latent: {sample['high_latent'].shape}")
    print(f"  low_latent: {sample['low_latent'].shape}")
    print(f"  high_midi: {sample['high_midi'].item():.1f}")
    print(f"  low_midi: {sample['low_midi'].item():.1f}")

    print("\nTesting RegisterTranslatorDatasetMultiBin...")
    dataset_mb = RegisterTranslatorDatasetMultiBin(
        manifest_path=manifest,
        window_frames=128,
        samples_per_epoch=100,
        num_bins=5,
        target_bin=0,  # Target = lowest bin
    )

    sample = dataset_mb[0]
    print(f"  source_bin: {sample['source_bin'].item()}")
    print(f"  target_bin: {sample['target_bin'].item()}")
    print(f"  high_midi: {sample['high_midi'].item():.1f}")
    print(f"  low_midi: {sample['low_midi'].item():.1f}")
