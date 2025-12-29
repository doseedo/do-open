#!/usr/bin/env python3
"""
v9 Dataset: Mixed Formant-Shifted Pairs + Sox-Shifted Distribution Matching

Two training modes mixed 50/50:
1. Formant-shifted pairs: Real audio with formant shift → L1 loss to real target
2. Sox-shifted: Sox pitch-shifted audio → distribution matching to target group

This teaches the model:
- Exact content alignment from formant pairs
- Sox artifact correction from sox-shifted data
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
import numpy as np


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


@dataclass
class TrainingSample:
    """A training sample with conditioning info."""
    input_latent: torch.Tensor   # Sox or formant-shifted input
    target_latent: torch.Tensor  # Natural target
    source_group: int            # 0, 1, 2 for groups 1, 2, 3
    direction: int               # 0=down, 1=up
    is_paired: bool              # True for formant pairs (use L1), False for sox (distribution)


class FormantCorrectorDatasetV9(Dataset):
    """
    Mixed dataset for v9 training.

    Data sources:
    1. Formant pairs: Group N audio → formant shift → matches Group N±1 timbre
       - These have EXACT content alignment, use L1 loss
    2. Sox shifted: HIGH audio → sox down → should match LOW distribution
       - No content alignment, use distribution matching loss

    Conditioning:
    - source_group: Which group the input came from (before shifting)
    - direction: 0=shifted down (needs lower formants), 1=shifted up (needs higher formants)
    """

    def __init__(
        self,
        manifest_path: str,
        sox_manifest_path: str = '/mnt/msdd2/pitchshift_v7_overlapped/manifest.json',
        window_frames: int = 128,
        samples_per_epoch: int = 10000,
        formant_pair_ratio: float = 0.5,
        pitch_tolerance: float = 3.0,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.formant_pair_ratio = formant_pair_ratio
        self.pitch_tolerance = pitch_tolerance

        # Load v9 group manifest
        print(f"Loading v9 manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.group_entries = manifest.get('group_entries', {})
        self.formant_pairs = manifest.get('formant_pairs', [])

        # Load sox manifest (from v7 overlapped preprocessing)
        if os.path.exists(sox_manifest_path):
            print(f"Loading sox manifest: {sox_manifest_path}")
            with open(sox_manifest_path) as f:
                sox_manifest = json.load(f)
            self.sox_shifted = sox_manifest.get('shifted_entries', [])
            self.sox_targets = sox_manifest.get('low_entries', [])
        else:
            self.sox_shifted = []
            self.sox_targets = []

        # Convert group_entries from str keys to int
        self.group_entries = {int(k): v for k, v in self.group_entries.items()}

        # Build lookup for target matching
        self._build_target_lookups()

        print(f"Loaded:")
        print(f"  Group 1 segments: {len(self.group_entries.get(1, []))}")
        print(f"  Group 2 segments: {len(self.group_entries.get(2, []))}")
        print(f"  Group 3 segments: {len(self.group_entries.get(3, []))}")
        print(f"  Formant pairs: {len(self.formant_pairs)}")
        print(f"  Sox shifted entries: {len(self.sox_shifted)}")
        print(f"  Sox target entries: {len(self.sox_targets)}")

    def _build_target_lookups(self):
        """Build pitch-indexed lookup for target matching."""
        # For formant pairs: targets are in the target group
        self.group_targets_by_pitch = {}
        for group_id in [1, 2, 3]:
            entries = self.group_entries.get(group_id, [])
            by_pitch = {}
            for entry in entries:
                midi = int(round(entry.get('median_midi', 0)))
                if midi not in by_pitch:
                    by_pitch[midi] = []
                by_pitch[midi].append(entry)
            self.group_targets_by_pitch[group_id] = by_pitch

        # For sox shifted: targets are sox_targets (LOW register)
        self.sox_targets_by_pitch = {}
        for entry in self.sox_targets:
            midi = int(round(entry.get('median_midi', 0)))
            if midi not in self.sox_targets_by_pitch:
                self.sox_targets_by_pitch[midi] = []
            self.sox_targets_by_pitch[midi].append(entry)

    def _find_matching_target(self, target_midi: float, target_group: int) -> Optional[Dict]:
        """Find a target in target_group with similar pitch."""
        by_pitch = self.group_targets_by_pitch.get(target_group, {})
        target_int = int(round(target_midi))

        # Search within tolerance
        for offset in range(int(self.pitch_tolerance) + 1):
            for sign in [0, 1, -1]:
                pitch = target_int + sign * offset
                if pitch in by_pitch and by_pitch[pitch]:
                    return random.choice(by_pitch[pitch])
        return None

    def _find_sox_target(self, target_midi: float) -> Optional[Dict]:
        """Find a sox target with similar pitch."""
        target_int = int(round(target_midi))

        for offset in range(int(self.pitch_tolerance) + 1):
            for sign in [0, 1, -1]:
                pitch = target_int + sign * offset
                if pitch in self.sox_targets_by_pitch and self.sox_targets_by_pitch[pitch]:
                    return random.choice(self.sox_targets_by_pitch[pitch])
        return None

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
                latent = latent.squeeze(0)  # [C, H, T]

            return latent.float()
        except Exception:
            return None

    def _crop_window(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        """Crop a random window from latent."""
        C, H, T = latent.shape
        if T < self.window_frames:
            return None

        start = random.randint(0, T - self.window_frames)
        return latent[:, :, start:start + self.window_frames]

    def _get_formant_pair_sample(self) -> Optional[TrainingSample]:
        """Get a formant-shifted pair sample."""
        if not self.formant_pairs:
            return None

        pair = random.choice(self.formant_pairs)

        source_group = pair['source_group']
        target_group = pair['target_group']
        shift = pair['formant_shift']

        # Direction: +12 means going UP (from lower to higher group)
        direction = 1 if shift > 0 else 0

        # Load source latent
        source_latent = self._load_latent(pair['source_latent_path'])
        if source_latent is None:
            return None

        # Find a target in the target group with matching pitch
        # The source pitch AFTER formant shift should match target pitch
        source_midi = pair.get('median_midi', 65)
        # Formant shift doesn't change pitch, so source_midi should match target
        target = self._find_matching_target(source_midi, target_group)
        if target is None:
            return None

        target_latent = self._load_latent(target['latent_path'])
        if target_latent is None:
            return None

        # Crop windows
        source_crop = self._crop_window(source_latent)
        target_crop = self._crop_window(target_latent)

        if source_crop is None or target_crop is None:
            return None

        # For formant pairs, input IS the source (formant-shifted)
        # The preprocessing should have already created formant-shifted latents
        # But currently we just have metadata - we need to apply formant shift at train time
        # OR use the source as-is and have the model learn the mapping

        # Actually, for the formant pair, the SOURCE is from source_group
        # and needs to be transformed to LOOK LIKE target_group
        # The "input" is the source latent, "target" is from target group
        return TrainingSample(
            input_latent=source_crop,
            target_latent=target_crop,
            source_group=source_group - 1,  # Convert 1,2,3 to 0,1,2
            direction=direction,
            is_paired=True,
        )

    def _get_sox_sample(self) -> Optional[TrainingSample]:
        """Get a sox-shifted sample."""
        if not self.sox_shifted or not self.sox_targets:
            return None

        entry = random.choice(self.sox_shifted)

        # Load shifted latent
        input_latent = self._load_latent(entry['latent_path'])
        if input_latent is None:
            return None

        # Find target with matching pitch
        target_midi = entry.get('target_midi', 50)
        target = self._find_sox_target(target_midi)
        if target is None:
            return None

        target_latent = self._load_latent(target['latent_path'])
        if target_latent is None:
            return None

        # Crop windows
        input_crop = self._crop_window(input_latent)
        target_crop = self._crop_window(target_latent)

        if input_crop is None or target_crop is None:
            return None

        # Sox shifted always goes DOWN (high → low)
        # Source group: based on original MIDI
        original_midi = entry.get('original_midi', 65)
        if original_midi >= 77:
            source_group = 2  # Group 3 → index 2
        elif original_midi >= 65:
            source_group = 1  # Group 2 → index 1
        else:
            source_group = 0  # Group 1 → index 0

        return TrainingSample(
            input_latent=input_crop,
            target_latent=target_crop,
            source_group=source_group,
            direction=0,  # Sox is always shifting DOWN
            is_paired=False,
        )

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        # Try to get a sample, with retries
        for _ in range(50):
            # Decide sample type based on ratio
            use_formant = random.random() < self.formant_pair_ratio

            if use_formant:
                sample = self._get_formant_pair_sample()
            else:
                sample = self._get_sox_sample()

            if sample is not None:
                return {
                    'input': sample.input_latent,
                    'target': sample.target_latent,
                    'source_group': sample.source_group,
                    'direction': sample.direction,
                    'is_paired': sample.is_paired,
                    'valid': True,
                }

        # Return invalid sample
        return {
            'input': torch.zeros(8, 16, self.window_frames),
            'target': torch.zeros(8, 16, self.window_frames),
            'source_group': 0,
            'direction': 0,
            'is_paired': False,
            'valid': False,
        }


class FormantCorrectorDatasetV9Simple(Dataset):
    """
    Simplified v9 dataset that only uses group segments.

    For each sample:
    - Input: segment from group N
    - Target: segment from adjacent group (N-1 or N+1)
    - Condition: source group + direction

    This teaches: given group N input, produce group M output.
    """

    def __init__(
        self,
        manifest_path: str,
        window_frames: int = 16,  # Short segments from preprocessing
        samples_per_epoch: int = 10000,
        pitch_tolerance: float = 3.0,
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.pitch_tolerance = pitch_tolerance

        print(f"Loading manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.group_entries = {int(k): v for k, v in manifest.get('group_entries', {}).items()}

        # Build pitch lookups
        self.by_pitch = {}
        for group_id in [1, 2, 3]:
            self.by_pitch[group_id] = {}
            for entry in self.group_entries.get(group_id, []):
                midi = int(round(entry.get('median_midi', 0)))
                if midi not in self.by_pitch[group_id]:
                    self.by_pitch[group_id][midi] = []
                self.by_pitch[group_id][midi].append(entry)

        # Define valid transitions (adjacent groups only)
        self.transitions = [
            (1, 2, 1),   # Group 1 → 2, direction=up
            (2, 1, 0),   # Group 2 → 1, direction=down
            (2, 3, 1),   # Group 2 → 3, direction=up
            (3, 2, 0),   # Group 3 → 2, direction=down
        ]

        print(f"Loaded groups: {[len(self.group_entries.get(g, [])) for g in [1,2,3]]}")

    def _load_latent(self, path: str) -> Optional[torch.Tensor]:
        try:
            path = fix_path(path)
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latent', data.get('latents'))
            else:
                latent = data
            if latent is not None and latent.dim() == 4:
                latent = latent.squeeze(0)
            return latent.float() if latent is not None else None
        except:
            return None

    def _crop_window(self, latent: torch.Tensor) -> Optional[torch.Tensor]:
        C, H, T = latent.shape
        if T < self.window_frames:
            return None
        start = random.randint(0, T - self.window_frames)
        return latent[:, :, start:start + self.window_frames]

    def _find_target(self, target_group: int, target_midi: float) -> Optional[Dict]:
        """Find entry in target_group with similar pitch (shifted by 12 for octave)."""
        by_pitch = self.by_pitch.get(target_group, {})

        # Target pitch is shifted by ~12 semitones from source
        target_int = int(round(target_midi))

        for offset in range(int(self.pitch_tolerance) + 1):
            for sign in [0, 1, -1]:
                pitch = target_int + sign * offset
                if pitch in by_pitch and by_pitch[pitch]:
                    return random.choice(by_pitch[pitch])
        return None

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        for _ in range(50):
            # Pick random transition
            src_group, tgt_group, direction = random.choice(self.transitions)

            # Get source entry
            src_entries = self.group_entries.get(src_group, [])
            if not src_entries:
                continue

            src_entry = random.choice(src_entries)
            src_midi = src_entry.get('median_midi', 65)

            # Target pitch depends on direction
            # Going up: target is ~12 semitones higher
            # Going down: target is ~12 semitones lower
            if direction == 1:  # up
                target_midi = src_midi + 12
            else:  # down
                target_midi = src_midi - 12

            tgt_entry = self._find_target(tgt_group, target_midi)
            if tgt_entry is None:
                continue

            src_latent = self._load_latent(src_entry['latent_path'])
            tgt_latent = self._load_latent(tgt_entry['latent_path'])

            if src_latent is None or tgt_latent is None:
                continue

            src_crop = self._crop_window(src_latent)
            tgt_crop = self._crop_window(tgt_latent)

            if src_crop is None or tgt_crop is None:
                continue

            return {
                'input': src_crop,
                'target': tgt_crop,
                'source_group': src_group - 1,  # 0, 1, 2
                'direction': direction,
                'is_paired': False,  # Not exact pairs
                'valid': True,
            }

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
    ds = FormantCorrectorDatasetV9Simple(
        manifest_path='/mnt/msdd2/pitchshift_v9_groups/manifest.json',
        samples_per_epoch=100,
    )

    sample = ds[0]
    print(f"Sample shape: input={sample['input'].shape}, target={sample['target'].shape}")
    print(f"Conditioning: group={sample['source_group']}, direction={sample['direction']}")
    print(f"Valid: {sample['valid']}")
