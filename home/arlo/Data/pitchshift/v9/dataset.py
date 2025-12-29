#!/usr/bin/env python3
"""
v7 Dataset: Formant Corrector

Key insight: Both input and target are at the SAME pitch range.
- Input: HIGH audio sox-shifted DOWN (wrong formants, low pitch)
- Target: natural LOW audio (correct formants, low pitch)

No pitch distribution mismatch = model only fixes formants.
"""

import os
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def sox_pitch_shift(waveform: torch.Tensor, sr: int, semitones: float) -> torch.Tensor:
    """Apply sox pitch shift to waveform tensor."""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_in:
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f_out:
        out_path = f_out.name

    try:
        # Save input
        torchaudio.save(in_path, waveform, sr)

        # Sox shift
        cents = int(semitones * 100)
        cmd = ['sox', in_path, out_path, 'pitch', str(cents)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sox failed: {result.stderr}")

        # Load output
        shifted, _ = torchaudio.load(out_path)
        return shifted
    finally:
        os.unlink(in_path)
        os.unlink(out_path)


def get_median_pitch(f0_path: str) -> Optional[float]:
    try:
        f0 = np.load(f0_path)
        f0 = np.nan_to_num(f0, nan=0.0)
        f0_valid = f0[f0 > 20]
        if len(f0_valid) < 10:
            return None
        freq = np.median(f0_valid)
        midi = 12 * np.log2(freq / 440) + 69
        return midi
    except Exception:
        return None


class FormantCorrectorDataset(Dataset):
    """
    Dataset for formant correction training.

    Creates pairs:
    - Input: HIGH latent → decode → sox DOWN → re-encode (corrupted)
    - Target: natural LOW latent (correct formants)

    Both at LOW pitch range - no distribution mismatch.
    """

    def __init__(
        self,
        manifest_path: str,
        dcae,  # Pre-loaded DCAE model
        window_frames: int = 128,
        samples_per_epoch: int = 2000,  # Smaller due to decode/encode overhead
        low_threshold: float = 55.0,
        high_threshold: float = 70.0,
        shift_semitones: float = -12.0,
        instrument: str = 'trumpet',
        device: str = 'cuda',
    ):
        self.dcae = dcae
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.shift_semitones = shift_semitones
        self.device = device

        # Load manifest
        with open(manifest_path) as f:
            manifest = json.load(f)

        self.high_entries = []
        self.low_entries = []

        print(f"Loading manifest for formant corrector...")
        print(f"  LOW: < {low_threshold} MIDI")
        print(f"  HIGH: > {high_threshold} MIDI")
        print(f"  Shift: {shift_semitones} semitones")

        n_skipped = 0

        for entry in manifest:
            if entry.get('sub_group') != instrument:
                continue

            latent_path = fix_path(entry.get('latent_path', ''))
            cond = entry.get('conditioning_paths') or {}
            f0_path = fix_path(cond.get('f0', ''))

            if not latent_path or not f0_path:
                continue
            if not os.path.exists(latent_path) or not os.path.exists(f0_path):
                n_skipped += 1
                continue

            median_midi = get_median_pitch(f0_path)
            if median_midi is None:
                n_skipped += 1
                continue

            entry_data = {
                'latent_path': latent_path,
                'f0_path': f0_path,
                'median_midi': median_midi,
            }

            if median_midi > high_threshold:
                self.high_entries.append(entry_data)
            elif median_midi < low_threshold:
                self.low_entries.append(entry_data)

        print(f"  HIGH register: {len(self.high_entries)} samples")
        print(f"  LOW register: {len(self.low_entries)} samples")
        print(f"  Skipped: {n_skipped}")

        if len(self.high_entries) == 0 or len(self.low_entries) == 0:
            raise ValueError(f"Need samples in both domains!")

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
        except Exception as e:
            print(f"Error loading {entry['latent_path']}: {e}")
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

    def _create_corrupted_input(self, high_latent: torch.Tensor) -> torch.Tensor:
        """
        Decode HIGH latent → sox shift DOWN → re-encode.
        Creates corrupted LOW-pitch latent with HIGH formants.
        """
        with torch.no_grad():
            # Decode to audio
            latent_4d = high_latent.unsqueeze(0).to(self.device)
            sr, wavs = self.dcae.decode(latent_4d)
            audio = wavs[0].cpu()  # [C, samples]

            # Sox shift down
            shifted_audio = sox_pitch_shift(audio, sr, self.shift_semitones)

            # Re-encode
            shifted_audio = shifted_audio.unsqueeze(0).to(self.device)
            corrupted_latent = self.dcae.encode(shifted_audio)

            # Remove batch dim if present
            if corrupted_latent.dim() == 4:
                corrupted_latent = corrupted_latent.squeeze(0)

            return corrupted_latent.cpu()

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
        - corrupted: HIGH sox-shifted to LOW (wrong formants)
        - target: natural LOW (correct formants)
        """
        # Get random entries
        high_entry = random.choice(self.high_entries)
        low_entry = random.choice(self.low_entries)

        # Load latents
        high_latent = self._load_latent(high_entry)
        low_latent = self._load_latent(low_entry)

        valid = True
        if high_latent is None or low_latent is None:
            # Return zeros for invalid
            return {
                'corrupted': torch.zeros(8, 16, self.window_frames),
                'target': torch.zeros(8, 16, self.window_frames),
                'valid': torch.tensor(False),
            }

        # Window first (before expensive decode/encode)
        high_latent = self._random_window(high_latent)
        low_latent = self._random_window(low_latent)

        # Create corrupted input (decode → sox → encode)
        try:
            corrupted = self._create_corrupted_input(high_latent)
            # Match window size
            if corrupted.shape[-1] != self.window_frames:
                corrupted = self._random_window(corrupted)
        except Exception as e:
            print(f"Error creating corrupted input: {e}")
            valid = False
            corrupted = torch.zeros(8, 16, self.window_frames)

        return {
            'corrupted': corrupted,
            'target': low_latent,
            'valid': torch.tensor(valid),
        }


class PrecomputedFormantCorrectorDataset(Dataset):
    """
    Uses pre-computed sox-shifted latents from v4.

    v4 manifest has entries with shifted_latents[-12] = path to shifted version.
    """

    def __init__(
        self,
        v4_manifest_path: str = '/mnt/msdd2/pitchshift_v4_precomputed/shifted_manifest.json',
        mute_manifest_path: str = '/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
        window_frames: int = 128,
        samples_per_epoch: int = 5000,
        low_threshold: float = 55.0,
        high_threshold: float = 70.0,
        shift: str = '-12',
        instrument: str = 'trumpet',
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch

        # Load v4 precomputed manifest
        with open(v4_manifest_path) as f:
            v4_data = json.load(f)

        # Get HIGH register entries with -12 shift
        self.shifted_entries = []
        for entry in v4_data['entries']:
            median_midi = entry.get('median_midi', 0)
            if median_midi > high_threshold:
                shifted_path = entry['shifted_latents'].get(shift)
                if shifted_path and os.path.exists(shifted_path):
                    self.shifted_entries.append({
                        'latent_path': shifted_path,
                        'original_midi': median_midi,
                        'target_midi': median_midi + int(shift),
                    })

        # Load natural LOW targets from mute manifest
        with open(mute_manifest_path) as f:
            mute_manifest = json.load(f)

        self.low_entries = []
        for entry in mute_manifest:
            if entry.get('sub_group') != instrument:
                continue

            latent_path = fix_path(entry.get('latent_path', ''))
            cond = entry.get('conditioning_paths') or {}
            f0_path = fix_path(cond.get('f0', ''))

            if not latent_path or not f0_path:
                continue
            if not os.path.exists(latent_path) or not os.path.exists(f0_path):
                continue

            median_midi = get_median_pitch(f0_path)
            if median_midi is None:
                continue

            if median_midi < low_threshold:
                self.low_entries.append({
                    'latent_path': latent_path,
                    'median_midi': median_midi,
                })

        print(f"  Shifted HIGH→LOW inputs: {len(self.shifted_entries)}")
        print(f"  Natural LOW targets: {len(self.low_entries)}")

    def _load_latent(self, path: str) -> Optional[torch.Tensor]:
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
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
        shifted_entry = random.choice(self.shifted_entries)
        low_entry = random.choice(self.low_entries)

        corrupted = self._load_latent(shifted_entry['latent_path'])
        target = self._load_latent(low_entry['latent_path'])

        valid = corrupted is not None and target is not None

        if not valid:
            return {
                'corrupted': torch.zeros(8, 16, self.window_frames),
                'target': torch.zeros(8, 16, self.window_frames),
                'valid': torch.tensor(False),
            }

        corrupted = self._random_window(corrupted)
        target = self._random_window(target)

        return {
            'corrupted': corrupted,
            'target': target,
            'valid': torch.tensor(True),
        }


if __name__ == "__main__":
    print("Testing FormantCorrectorDataset requires DCAE - use precompute script instead")
