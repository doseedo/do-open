#!/usr/bin/env python3
"""
v9 Dataset: Full Audio Files with Runtime Formant Shifting

Instead of pre-segmented clips, this loads full audio files and:
1. Samples random windows during training
2. Applies formant shift on-the-fly using precomputed shifted audio
3. Filters by RMS to skip silent sections
4. Gets pitch group from f0 data

This gives access to the full 2000+ trumpet recordings.
"""

import os
import json
import random
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset


# Pitch group ranges (MIDI)
GROUP_RANGES = {
    1: (53, 65),   # F3-F4 (LOW)
    2: (65, 77),   # F4-F5 (MID)
    3: (77, 89),   # F5-F6 (HIGH)
}

MIN_RMS = 0.01
HOP_SIZE = 512
SAMPLE_RATE = 44100


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def hz_to_midi(hz: float) -> float:
    if hz <= 20:
        return 0
    return 69 + 12 * np.log2(hz / 440.0)


def get_pitch_group(midi: float) -> Optional[int]:
    """Get group for a MIDI note."""
    for group, (low, high) in GROUP_RANGES.items():
        if low <= midi < high:
            return group
    if midi >= 77 and midi <= 89:
        return 3
    return None


class FullAudioDataset(Dataset):
    """
    Dataset using full audio files with runtime window sampling.

    Precomputes formant-shifted versions of audio files and samples
    aligned windows from natural/shifted pairs.
    """

    def __init__(
        self,
        manifest_path: str,
        cache_dir: str = '/mnt/msdd2/pitchshift_v9_full_cache',
        window_seconds: float = 1.0,
        samples_per_epoch: int = 10000,
        formant_shifts: List[int] = [-12, 12],
        instrument: str = 'trumpet',
        min_rms: float = MIN_RMS,
        device: str = 'cuda',
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.window_seconds = window_seconds
        self.window_samples = int(window_seconds * SAMPLE_RATE)
        self.samples_per_epoch = samples_per_epoch
        self.formant_shifts = formant_shifts
        self.min_rms = min_rms
        self.device = device

        # Load manifest
        print(f"Loading manifest: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Filter entries
        self.entries = []
        for e in manifest:
            if e.get('sub_group') != instrument:
                continue
            if e.get('is_muted', False):
                continue

            audio_path = fix_path(e.get('audio_path', ''))
            f0_path = fix_path(e.get('conditioning_paths', {}).get('f0', ''))
            latent_path = fix_path(e.get('latent_path', ''))

            if not all([audio_path, f0_path, latent_path]):
                continue
            if not all(os.path.exists(p) for p in [audio_path, f0_path, latent_path]):
                continue

            self.entries.append({
                'audio_path': audio_path,
                'f0_path': f0_path,
                'latent_path': latent_path,
            })

        print(f"Found {len(self.entries)} valid entries")

        # Load DCAE for encoding (lazy load)
        self._dcae = None

    @property
    def dcae(self):
        if self._dcae is None:
            import sys
            sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Data/dø')
            from do.pipeline_do import DoTrainComponents
            components = DoTrainComponents(
                checkpoint_dir="/home/arlo/Data/ACE-Step/checkpoints",
                device_id=0
            )
            components.load_dcae()
            self._dcae = components.music_dcae
        return self._dcae

    def _get_shifted_audio_path(self, audio_path: str, shift: int) -> str:
        """Get path to cached formant-shifted audio."""
        audio_hash = str(hash(audio_path))[-8:]
        shift_name = f"shift{shift:+d}"
        return str(self.cache_dir / f"{audio_hash}_{shift_name}.wav")

    def _create_shifted_audio(self, audio_path: str, shift: int) -> Optional[str]:
        """Create formant-shifted version of audio."""
        out_path = self._get_shifted_audio_path(audio_path, shift)

        if os.path.exists(out_path):
            return out_path

        try:
            # Formant shift: speed up/down then pitch shift back
            speed_factor = 2 ** (shift / 12)
            pitch_cents = -shift * 100

            cmd = [
                'sox', audio_path, out_path,
                'speed', str(speed_factor),
                'pitch', str(pitch_cents),
                'rate', '44100',
            ]
            subprocess.run(cmd, capture_output=True, check=True)
            return out_path if os.path.exists(out_path) else None
        except Exception as e:
            return None

    def _check_rms(self, audio: torch.Tensor) -> bool:
        """Check if audio window has sufficient RMS."""
        rms = audio.pow(2).mean().sqrt().item()
        return rms >= self.min_rms

    def _get_pitch_info(self, f0_path: str, start_sample: int, end_sample: int) -> Tuple[int, int]:
        """Get pitch group and direction for a window."""
        try:
            f0 = np.load(f0_path)
            f0 = np.nan_to_num(f0, nan=0.0)

            # Convert samples to f0 frames
            start_frame = start_sample // HOP_SIZE
            end_frame = end_sample // HOP_SIZE

            # Get f0 for this window
            f0_window = f0[start_frame:end_frame]
            valid = f0_window > 20

            if not valid.any():
                return None, None

            # Get median pitch
            midi_values = 69 + 12 * np.log2(f0_window[valid] / 440.0)
            median_midi = np.median(midi_values)

            group = get_pitch_group(median_midi)
            if group is None:
                return None, None

            # Direction based on group
            # Group 1 (low): can only go UP
            # Group 2 (mid): can go both
            # Group 3 (high): can only go DOWN
            if group == 1:
                direction = 1  # UP
                shift = 12
            elif group == 3:
                direction = 0  # DOWN
                shift = -12
            else:  # Group 2
                direction = random.choice([0, 1])
                shift = 12 if direction == 1 else -12

            return group, direction, shift

        except Exception:
            return None, None, None

    @torch.no_grad()
    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode audio to latent using DCAE."""
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)

        # Ensure stereo
        if audio.shape[1] == 1:
            audio = audio.repeat(1, 2, 1)

        audio = audio.to(self.device)
        result = self.dcae.encode(audio)
        latent = result[0] if isinstance(result, tuple) else result
        return latent.cpu()

    def _get_sample(self) -> Optional[Dict]:
        """Get a single training sample."""
        for _ in range(50):  # Max retries
            entry = random.choice(self.entries)

            try:
                # Load natural audio
                audio, sr = torchaudio.load(entry['audio_path'])
                if sr != SAMPLE_RATE:
                    audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)

                # Ensure mono for processing
                if audio.shape[0] > 1:
                    audio = audio.mean(0, keepdim=True)

                total_samples = audio.shape[-1]
                if total_samples < self.window_samples:
                    continue

                # Try random windows until we find one with audio
                for _ in range(10):
                    start = random.randint(0, total_samples - self.window_samples)
                    end = start + self.window_samples

                    window = audio[:, start:end]

                    if not self._check_rms(window):
                        continue

                    # Get pitch info
                    result = self._get_pitch_info(entry['f0_path'], start, end)
                    if result[0] is None:
                        continue

                    group, direction, shift = result

                    # Get or create shifted audio
                    shifted_path = self._create_shifted_audio(entry['audio_path'], shift)
                    if shifted_path is None:
                        continue

                    # Load shifted audio window
                    shifted_audio, _ = torchaudio.load(shifted_path)
                    if shifted_audio.shape[0] > 1:
                        shifted_audio = shifted_audio.mean(0, keepdim=True)

                    # Shifted audio might be slightly different length due to resampling
                    if shifted_audio.shape[-1] < end:
                        continue

                    shifted_window = shifted_audio[:, start:end]

                    # Make stereo for DCAE
                    natural_stereo = window.repeat(2, 1)
                    shifted_stereo = shifted_window.repeat(2, 1)

                    # Encode both
                    natural_latent = self._encode_audio(natural_stereo)
                    shifted_latent = self._encode_audio(shifted_stereo)

                    # Match lengths
                    min_t = min(natural_latent.shape[-1], shifted_latent.shape[-1])
                    natural_latent = natural_latent[..., :min_t]
                    shifted_latent = shifted_latent[..., :min_t]

                    return {
                        'input': shifted_latent.squeeze(0),  # [C, H, T]
                        'target': natural_latent.squeeze(0),
                        'source_group': group - 1,  # 0-indexed
                        'direction': direction,
                        'is_paired': True,
                        'valid': True,
                    }

            except Exception as e:
                continue

        # Return invalid sample
        return {
            'input': torch.zeros(8, 16, 6),
            'target': torch.zeros(8, 16, 6),
            'source_group': 0,
            'direction': 0,
            'is_paired': False,
            'valid': False,
        }

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        return self._get_sample()


if __name__ == "__main__":
    # Test dataset
    ds = FullAudioDataset(
        manifest_path='/home/arlo/do-repo/home/arlo/Data/mute_translator/mute_manifest_deduped.json',
        samples_per_epoch=10,
    )

    print(f"\nTesting {len(ds)} samples...")
    valid = 0
    for i in range(len(ds)):
        sample = ds[i]
        if sample['valid']:
            valid += 1
            print(f"  {i}: group={sample['source_group']}, dir={sample['direction']}, shape={sample['input'].shape}")

    print(f"\nValid samples: {valid}/{len(ds)}")
