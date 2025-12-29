#!/usr/bin/env python3
"""
V4 Dataset: Synthetic pitch-shifted pairs for artifact correction.

Key idea:
- Source: DSP pitch-shift a real clip → has artifacts
- Target: Real clip at the destination pitch → clean reference
- Conditioning: shift amount in semitones

This matches the inference task exactly:
- At inference: input is librosa/rubberband shifted audio
- Model learns to fix the artifacts and make it sound real
"""

import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset


def fix_path(path: str) -> str:
    """Fix mount paths."""
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


def hz_to_midi(hz: float) -> float:
    """Convert Hz to MIDI note number."""
    if hz <= 0:
        return 0
    return 69 + 12 * np.log2(hz / 440.0)


def get_pitch_group(midi: float, group_size: int = 6, base_pitch: int = 48) -> int:
    """Get pitch group for a MIDI note."""
    return int((midi - base_pitch) // group_size)


class PitchShiftCorrectionDataset(Dataset):
    """
    Dataset for V4 pitch-shift artifact correction.

    For each sample:
    1. Pick a source segment from any pitch group
    2. Choose a random shift amount s
    3. DSP pitch-shift the source audio by s semitones → input (has artifacts)
    4. Find a target segment at pitch (source_pitch + s) → target (clean)
    5. Return (input_latent, target_latent, s)

    The model learns to correct DSP pitch-shift artifacts.

    If shifted_manifest is provided, uses precomputed DSP-shifted latents
    (more accurate, recommended). Otherwise falls back to synthetic shifting.
    """

    def __init__(
        self,
        segments_json: str,
        window_frames: int = 64,
        samples_per_epoch: int = 10000,
        max_shift: int = 12,
        min_shift: int = -12,
        preload_latents: bool = True,
        preload_audio: bool = False,  # Audio is large, load on demand
        sample_rate: int = 44100,
        shifted_manifest: str = None,  # Path to precomputed shifted latents manifest
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.max_shift = max_shift
        self.min_shift = min_shift
        self.sample_rate = sample_rate

        # Load segments
        print(f"Loading segments from: {segments_json}")
        with open(segments_json, 'r') as f:
            data = json.load(f)

        self.config = data['config']
        self.group_size = self.config['group_size']
        self.base_pitch = self.config['base_pitch']
        self.num_groups = self.config['num_groups']

        # Organize segments by group
        self.segments_by_group: Dict[int, List[Dict]] = defaultdict(list)
        self.all_segments: List[Dict] = []

        for group_id, segments in data['segments_by_group'].items():
            group_int = int(group_id)
            for seg in segments:
                seg['group'] = group_int
                self.segments_by_group[group_int].append(seg)
                self.all_segments.append(seg)

        self.valid_groups = sorted(self.segments_by_group.keys())
        print(f"Loaded {len(self.all_segments)} segments across {len(self.valid_groups)} groups")

        # Load precomputed shifted latents manifest if provided
        self.shifted_entries: Dict[str, Dict] = {}  # source_hash -> entry
        self.available_shifts: List[int] = []
        self.use_precomputed = False

        if shifted_manifest and os.path.exists(shifted_manifest):
            print(f"Loading precomputed shifted latents from: {shifted_manifest}")
            with open(shifted_manifest, 'r') as f:
                shifted_data = json.load(f)

            self.available_shifts = shifted_data.get('shifts', [])
            for entry in shifted_data.get('entries', []):
                # Create lookup key from source latent path and frame range
                key = f"{entry['source_latent_path']}:{entry['start_frame']}:{entry['end_frame']}"
                self.shifted_entries[key] = entry

            self.use_precomputed = len(self.shifted_entries) > 0
            print(f"  Loaded {len(self.shifted_entries)} precomputed entries")
            print(f"  Available shifts: {self.available_shifts}")
        else:
            print("No precomputed shifted latents - using synthetic shifting")

        # Preload latents
        self.latent_cache: Dict[str, torch.Tensor] = {}
        if preload_latents:
            self._preload_latents()

    def _preload_latents(self):
        """Preload all latents into memory."""
        print("Preloading latents...")
        unique_paths = set(s['latent_path'] for s in self.all_segments)

        for path in unique_paths:
            path = fix_path(path)
            if os.path.exists(path):
                try:
                    latent = torch.load(path, map_location='cpu', weights_only=True)
                    if isinstance(latent, dict):
                        # Try multiple possible keys
                        latent = latent.get('latents', latent.get('latent', latent.get('z', None)))
                    if latent is not None:
                        self.latent_cache[path] = latent
                except Exception as e:
                    pass

        print(f"Cached {len(self.latent_cache)} latents")

    def _load_latent(self, path: str) -> Optional[torch.Tensor]:
        """Load a latent tensor."""
        path = fix_path(path)

        if path in self.latent_cache:
            return self.latent_cache[path]

        if not os.path.exists(path):
            return None

        try:
            latent = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(latent, dict):
                # Try multiple possible keys
                latent = latent.get('latents', latent.get('latent', latent.get('z', None)))
            return latent
        except:
            return None

    def _get_segment_latent(self, segment: Dict) -> Optional[torch.Tensor]:
        """Extract a window from a segment's latent."""
        latent = self._load_latent(segment['latent_path'])
        if latent is None:
            return None

        # latent shape: [C, H, T] or [1, C, H, T]
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        start = segment['start_frame']
        end = segment['end_frame']

        # Ensure we have enough frames
        if end - start < self.window_frames:
            return None

        # Random window within segment
        max_start = end - self.window_frames
        if max_start <= start:
            window_start = start
        else:
            window_start = random.randint(start, max_start)

        window_end = window_start + self.window_frames

        if window_end > latent.shape[-1]:
            return None

        return latent[:, :, window_start:window_end]

    def _find_target_segment(self, target_group: int) -> Optional[Dict]:
        """Find a segment in the target pitch group."""
        if target_group not in self.segments_by_group:
            # Find nearest valid group
            valid = self.valid_groups
            if not valid:
                return None
            target_group = min(valid, key=lambda g: abs(g - target_group))

        segments = self.segments_by_group[target_group]
        if not segments:
            return None

        return random.choice(segments)

    def _apply_pitch_shift_to_latent(
        self,
        latent: torch.Tensor,
        shift: int,
    ) -> torch.Tensor:
        """
        Apply approximate pitch shift effect to latent.

        In a full implementation, we'd:
        1. Decode latent to audio
        2. DSP pitch-shift the audio
        3. Re-encode to latent

        For efficiency during training, we approximate this with
        spectral shifting in the latent space. This isn't perfect
        but captures the main artifact characteristics.

        TODO: For best results, pre-compute pitch-shifted latents offline.
        """
        # Simple approximation: shift along frequency axis + add noise
        # This simulates formant shift artifacts

        C, H, T = latent.shape

        # Frequency axis shift (approximate formant shift)
        shift_amount = int(shift * H / 24)  # Scale shift to latent height

        if shift_amount != 0:
            if shift_amount > 0:
                # Shift up: pad bottom, crop top
                padding = torch.zeros(C, shift_amount, T, device=latent.device)
                shifted = torch.cat([padding, latent[:, :-shift_amount, :]], dim=1)
            else:
                # Shift down: pad top, crop bottom
                shift_amount = abs(shift_amount)
                padding = torch.zeros(C, shift_amount, T, device=latent.device)
                shifted = torch.cat([latent[:, shift_amount:, :], padding], dim=1)
        else:
            shifted = latent

        # Add slight noise to simulate DSP artifacts
        noise_scale = abs(shift) / 12.0 * 0.05  # More shift = more artifacts
        noise = torch.randn_like(shifted) * noise_scale
        shifted = shifted + noise

        return shifted

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _get_precomputed_shifted_latent(
        self,
        source_seg: Dict,
        shift: int,
    ) -> Optional[torch.Tensor]:
        """
        Get precomputed DSP-shifted latent for a source segment.

        Returns None if not available.
        """
        # Create lookup key
        key = f"{source_seg['latent_path']}:{source_seg['start_frame']}:{source_seg['end_frame']}"

        if key not in self.shifted_entries:
            return None

        entry = self.shifted_entries[key]
        shifted_latents = entry.get('shifted_latents', {})

        # Convert shift to string key (handles +/- formatting)
        shift_key = str(shift)
        if shift_key not in shifted_latents:
            # Try with explicit sign
            shift_key = f"+{shift}" if shift > 0 else str(shift)
            if shift_key not in shifted_latents:
                return None

        shifted_path = fix_path(shifted_latents[shift_key])
        if not os.path.exists(shifted_path):
            return None

        # Check cache first
        if shifted_path in self.latent_cache:
            latent = self.latent_cache[shifted_path]
        else:
            try:
                data = torch.load(shifted_path, map_location='cpu', weights_only=True)
                if isinstance(data, dict):
                    latent = data.get('latent', data.get('latents', data.get('z', None)))
                else:
                    latent = data

                if latent is None:
                    return None

                # Cache it
                self.latent_cache[shifted_path] = latent

            except Exception:
                return None

        # Handle shape
        if latent.dim() == 4:
            latent = latent.squeeze(0)

        # Extract window
        T = latent.shape[-1]
        if T < self.window_frames:
            return None

        # Random window
        max_start = T - self.window_frames
        if max_start <= 0:
            window_start = 0
        else:
            window_start = random.randint(0, max_start)

        return latent[:, :, window_start:window_start + self.window_frames]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            source_latent: [C, H, T] - DSP pitch-shifted (has artifacts)
            target_latent: [C, H, T] - Real audio at destination pitch (clean)
            shift: scalar - shift amount in semitones
            valid: bool
        """
        max_attempts = 50

        for _ in range(max_attempts):
            # Pick source segment
            source_seg = random.choice(self.all_segments)
            source_group = source_seg['group']

            # Choose random shift
            if self.use_precomputed and self.available_shifts:
                # Use only available precomputed shifts
                shift = random.choice(self.available_shifts)
            else:
                shift = random.randint(self.min_shift, self.max_shift)

            # Calculate target group
            target_group = source_group + (shift // self.group_size)
            target_group = max(0, min(target_group, self.num_groups - 1))

            # Get shifted latent (precomputed or synthetic)
            if self.use_precomputed:
                shifted_latent = self._get_precomputed_shifted_latent(source_seg, shift)
                if shifted_latent is None:
                    # Fall back to synthetic
                    source_latent = self._get_segment_latent(source_seg)
                    if source_latent is None:
                        continue
                    shifted_latent = self._apply_pitch_shift_to_latent(source_latent, shift)
            else:
                # Use synthetic shifting
                source_latent = self._get_segment_latent(source_seg)
                if source_latent is None:
                    continue
                shifted_latent = self._apply_pitch_shift_to_latent(source_latent, shift)

            # Find target segment at destination pitch
            target_seg = self._find_target_segment(target_group)
            if target_seg is None:
                continue

            target_latent = self._get_segment_latent(target_seg)
            if target_latent is None:
                continue

            # Validate shapes
            if shifted_latent.shape != target_latent.shape:
                continue

            return {
                'source_latent': shifted_latent,  # Pitch-shifted with artifacts
                'target_latent': target_latent,   # Real clean reference
                'shift': torch.tensor(shift, dtype=torch.float32),
                'source_group': torch.tensor(source_group),
                'target_group': torch.tensor(target_group),
                'valid': torch.tensor(True),
            }

        # Failed to find valid pair
        return {
            'source_latent': torch.zeros(8, 16, self.window_frames),
            'target_latent': torch.zeros(8, 16, self.window_frames),
            'shift': torch.tensor(0.0),
            'source_group': torch.tensor(0),
            'target_group': torch.tensor(0),
            'valid': torch.tensor(False),
        }


class PitchShiftCorrectionDatasetWithAudio(PitchShiftCorrectionDataset):
    """
    Extended dataset that actually applies DSP pitch shift to audio.

    This is more accurate but slower. Use for validation or if
    pre-computing shifted latents offline.
    """

    def __init__(self, *args, dcae=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dcae = dcae  # DCAE model for encode/decode
        self._pitch_shift_transform = None

    def _get_pitch_shift_transform(self, shift: int):
        """Get torchaudio pitch shift transform."""
        return torchaudio.transforms.PitchShift(self.sample_rate, shift)

    def _apply_real_pitch_shift(
        self,
        segment: Dict,
        shift: int,
    ) -> Optional[torch.Tensor]:
        """
        Apply real DSP pitch shift to audio and re-encode.

        This is the "correct" way but requires DCAE at training time.
        """
        if self.dcae is None:
            return None

        audio_path = fix_path(segment.get('audio_path', ''))
        if not os.path.exists(audio_path):
            return None

        try:
            # Load audio
            audio, sr = torchaudio.load(audio_path)
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)

            # Extract segment
            start_sample = int(segment['start_frame'] * 512)  # Approximate
            end_sample = int(segment['end_frame'] * 512)
            audio = audio[:, start_sample:end_sample]

            # Apply pitch shift
            pitch_shift = self._get_pitch_shift_transform(shift)
            shifted_audio = pitch_shift(audio)

            # Encode with DCAE
            # ... (would need DCAE integration)

            return None  # Placeholder

        except Exception as e:
            return None
