#!/usr/bin/env python3
"""
v7 Dataset: Formant Corrector with OVERLAPPED pitch ranges.

Uses precomputed sox-shifted latents where:
- Input: HIGH audio shifted to LOW range (wrong formants, low pitch)
- Target: Natural LOW audio (correct formants, low pitch)

Both at SAME pitch range - model only learns formant correction.
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_ensemble_blacklist(ensemble_results_path: str) -> set:
    """Load set of latent paths flagged as ensemble/polyphonic."""
    if not os.path.exists(ensemble_results_path):
        return set()

    with open(ensemble_results_path) as f:
        data = json.load(f)

    flagged = data.get('flagged_recordings', [])
    blacklist = set()
    for rec in flagged:
        # Get latent path from the recording
        latent_path = rec.get('latent_path', '')
        if latent_path:
            latent_path = fix_path(latent_path)
            blacklist.add(latent_path)
        # Also blacklist by f0 path pattern -> latent path
        f0_path = rec.get('f0_path', '')
        if f0_path:
            # Convert f0 path to latent path pattern
            # f0: .../conditioning/xxx.f0.npy -> latent: .../xxx.pt
            f0_path = fix_path(f0_path)
            base = f0_path.replace('.f0.npy', '.pt')
            # Try common latent locations
            for pattern in [
                base.replace('/conditioning/', '/'),
                base.replace('/conditioningnew/', '/dcae_latentsnew/'),
            ]:
                blacklist.add(pattern)

    return blacklist


class OverlappedFormantCorrectorDataset(Dataset):
    """
    Dataset for formant correction with OVERLAPPING pitch ranges.

    Uses manifest from precompute_overlapped.py which ensures:
    - Shifted inputs land in <55 MIDI range
    - Natural targets are <55 MIDI range
    - Full overlap, so model only learns formant correction

    Pairs samples by similar target pitch for best training.
    """

    def __init__(
        self,
        manifest_path: str = '/mnt/msdd2/pitchshift_v7_overlapped/manifest.json',
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        pitch_tolerance: float = 3.0,  # Match samples within N semitones
        ensemble_results_path: str = '/home/arlo/Data/pitchshift/v3/ensemble_detection_results.json',
        min_target_midi: float = 43.0,  # Filter targets below this
        max_target_midi: float = 52.0,  # Filter targets above this (tighter range)
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.pitch_tolerance = pitch_tolerance

        # Load ensemble blacklist
        ensemble_blacklist = load_ensemble_blacklist(ensemble_results_path)
        print(f"Loaded {len(ensemble_blacklist)} ensemble recordings to filter")

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Load shifted entries (inputs with wrong formants)
        self.shifted_entries = []
        skipped_ensemble = 0
        skipped_range = 0
        for entry in manifest['shifted_entries']:
            latent_path = entry['latent_path']
            target_midi = entry['target_midi']

            if not os.path.exists(latent_path):
                continue

            # Filter by target range
            if target_midi < min_target_midi or target_midi > max_target_midi:
                skipped_range += 1
                continue

            # Check if original source was ensemble (check original latent path pattern)
            original_path = entry.get('original_path', latent_path)
            if any(bl in original_path or bl in latent_path for bl in ensemble_blacklist):
                skipped_ensemble += 1
                continue

            self.shifted_entries.append({
                'latent_path': latent_path,
                'target_midi': target_midi,
                'original_midi': entry['original_midi'],
                'shift': entry['shift'],
            })

        print(f"Shifted entries: kept {len(self.shifted_entries)}, skipped {skipped_ensemble} ensemble, {skipped_range} out of range")

        # Load natural LOW entries (targets with correct formants)
        self.low_entries = []
        skipped_ensemble_low = 0
        skipped_range_low = 0
        for entry in manifest['low_entries']:
            latent_path = fix_path(entry['latent_path'])
            median_midi = entry['median_midi']

            if not os.path.exists(latent_path):
                continue

            # Filter by range
            if median_midi < min_target_midi or median_midi > max_target_midi:
                skipped_range_low += 1
                continue

            # Check ensemble blacklist
            if any(bl in latent_path for bl in ensemble_blacklist):
                skipped_ensemble_low += 1
                continue

            self.low_entries.append({
                'latent_path': latent_path,
                'median_midi': median_midi,
            })

        print(f"LOW entries: kept {len(self.low_entries)}, skipped {skipped_ensemble_low} ensemble, {skipped_range_low} out of range")

        # Build pitch index for efficient matching
        self._build_pitch_index()

        print(f"Loaded OverlappedFormantCorrectorDataset:")
        print(f"  Shifted inputs: {len(self.shifted_entries)}")
        print(f"  Natural targets: {len(self.low_entries)}")
        print(f"  Pitch tolerance: {pitch_tolerance} semitones")

        if self.shifted_entries and self.low_entries:
            shifted_range = [e['target_midi'] for e in self.shifted_entries]
            low_range = [e['median_midi'] for e in self.low_entries]
            print(f"  Shifted target range: {min(shifted_range):.1f} - {max(shifted_range):.1f}")
            print(f"  Natural LOW range: {min(low_range):.1f} - {max(low_range):.1f}")

    def _build_pitch_index(self):
        """Build index for finding LOW entries near a given pitch."""
        self.pitch_to_low = {}
        for entry in self.low_entries:
            pitch = int(round(entry['median_midi']))
            if pitch not in self.pitch_to_low:
                self.pitch_to_low[pitch] = []
            self.pitch_to_low[pitch].append(entry)

    def _find_matching_low(self, target_midi: float) -> Optional[Dict]:
        """Find a LOW entry with similar pitch. Returns None if no close match."""
        target_int = int(round(target_midi))

        # Search in expanding radius up to pitch_tolerance
        for offset in range(int(self.pitch_tolerance) + 1):
            for sign in [0, -1, 1]:
                pitch = target_int + sign * offset
                if pitch in self.pitch_to_low:
                    return random.choice(self.pitch_to_low[pitch])

        # Expand search a bit more (up to 2x tolerance) but no random fallback
        for offset in range(int(self.pitch_tolerance) + 1, int(self.pitch_tolerance * 2) + 1):
            for sign in [-1, 1]:
                pitch = target_int + sign * offset
                if pitch in self.pitch_to_low:
                    return random.choice(self.pitch_to_low[pitch])

        # No match found - return None, caller should skip this sample
        return None

    def _load_latent(self, path: str) -> Optional[torch.Tensor]:
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent is None:
                return None
            if latent.dim() == 4:
                latent = latent.squeeze(0)
            return latent
        except Exception:
            return None

    def _random_window(self, latent: torch.Tensor) -> torch.Tensor:
        T = latent.shape[-1]
        if T <= self.window_frames:
            pad = self.window_frames - T
            latent = F.pad(latent, (0, pad))
            return latent
        else:
            start = random.randint(0, T - self.window_frames)
            return latent[:, :, start:start + self.window_frames]

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
        - corrupted: Shifted HIGH (wrong formants, low pitch)
        - target: Natural LOW (correct formants, low pitch)
        - pitch_diff: Difference between shifted target and natural LOW pitch
        """
        if not self.shifted_entries or not self.low_entries:
            return self._fallback()

        # Get random shifted entry
        shifted_entry = random.choice(self.shifted_entries)

        # Find matching LOW entry (similar pitch)
        low_entry = self._find_matching_low(shifted_entry['target_midi'])

        if low_entry is None:
            return self._fallback()

        # Load latents
        corrupted = self._load_latent(shifted_entry['latent_path'])
        target = self._load_latent(low_entry['latent_path'])

        if corrupted is None or target is None:
            return self._fallback()

        # Window
        corrupted = self._random_window(corrupted)
        target = self._random_window(target)

        # Pitch difference (for potential conditioning)
        pitch_diff = shifted_entry['target_midi'] - low_entry['median_midi']

        return {
            'corrupted': corrupted,
            'target': target,
            'valid': torch.tensor(True),
            'pitch_diff': torch.tensor(pitch_diff, dtype=torch.float32),
            'shifted_midi': torch.tensor(shifted_entry['target_midi'], dtype=torch.float32),
            'target_midi': torch.tensor(low_entry['median_midi'], dtype=torch.float32),
        }

    def _fallback(self) -> Dict[str, torch.Tensor]:
        return {
            'corrupted': torch.zeros(8, 16, self.window_frames),
            'target': torch.zeros(8, 16, self.window_frames),
            'valid': torch.tensor(False),
            'pitch_diff': torch.tensor(0.0),
            'shifted_midi': torch.tensor(0.0),
            'target_midi': torch.tensor(0.0),
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/mnt/msdd2/pitchshift_v7_overlapped/manifest.json')
    args = parser.parse_args()

    if os.path.exists(args.manifest):
        dataset = OverlappedFormantCorrectorDataset(manifest_path=args.manifest)

        print(f"\nTesting dataset...")
        sample = dataset[0]
        print(f"  corrupted shape: {sample['corrupted'].shape}")
        print(f"  target shape: {sample['target'].shape}")
        print(f"  valid: {sample['valid'].item()}")
        print(f"  pitch_diff: {sample['pitch_diff'].item():.1f}")
        print(f"  shifted_midi: {sample['shifted_midi'].item():.1f}")
        print(f"  target_midi: {sample['target_midi'].item():.1f}")
    else:
        print(f"Manifest not found: {args.manifest}")
        print("Run precompute_overlapped.py first!")
