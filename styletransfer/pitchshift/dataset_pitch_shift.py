"""
Dataset for Register-Aware Pitch Shift Training

Key difference from mute translator: we have PAIRED data via double-shift,
and we condition on target pitch for register-specific timbre correction.

The double-shift strategy:
1. Take audio at pitch P
2. Shift up by N semitones (introduces artifacts)
3. Shift back down by N semitones (same pitch P, but with pitch-shift artifacts)
4. Train model to remove these artifacts

For actual pitch shifting:
1. Shift audio to target pitch P+N
2. Use reference recordings at P+N to learn correct timbre
3. Model learns register-specific formant characteristics
"""

import os
import json
import random
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import torchaudio


def fix_mount_path(path: str) -> str:
    """Fix mount path issues - try multiple possible locations."""
    if not path:
        return path

    # Try the original path first
    if os.path.exists(path):
        return path

    # Try various path fixes
    fixes = [
        # dcae_latentsnew -> dcae_latents
        (('/mnt/msdd/dcae_latentsnew/', '/mnt/msdd/dcae_latents/')),
        # msdd -> msdd2
        (('/mnt/msdd/', '/mnt/msdd2/')),
        # msdd2 -> msdd
        (('/mnt/msdd2/', '/mnt/msdd/')),
    ]

    for old, new in fixes:
        if old in path:
            fixed = path.replace(old, new)
            if os.path.exists(fixed):
                return fixed

    # Return original if no fix works
    return path


def audio_path_to_midi_path(audio_path: str) -> str:
    """
    Convert audio path to corresponding MIDI path.

    Audio: /home/arlo/gcs-bucket/protools/2025-04-02/New/.../Audio Files/file.wav
    MIDI:  /home/arlo/gcs-bucket/BasicPitch/2025-04-02/New/.../Audio Files/file.mid
    """
    if not audio_path:
        return ""
    # Replace protools with BasicPitch and .wav with .mid
    midi_path = audio_path.replace('/protools/', '/BasicPitch/')
    midi_path = midi_path.replace('.wav', '.mid')
    return midi_path


class PitchShiftDataset(Dataset):
    """
    Dataset for training the register-aware pitch shift corrector.

    For each sample, returns:
    - degraded_latent: Double-shifted latent (has artifacts, same pitch as original)
    - target_latent: Original clean latent
    - shift_amount: How much it was shifted (for conditioning)
    - target_pitch: The pitch of this segment (for timbre conditioning)

    Also generates "actual shift" examples:
    - shifted_latent: Actually pitch-shifted latent (different pitch)
    - target_pitch: Target pitch (for timbre reference)
    - reference_latent: A real latent at target_pitch (different content, correct timbre)
    """

    def __init__(
        self,
        manifest_path: str,
        instrument: str = 'trumpet',
        window_frames: int = 128,
        samples_per_epoch: int = 10000,
        shift_range: Tuple[int, int] = (-12, 12),
        artifact_removal_ratio: float = 0.5,
        use_cached_latents: bool = True,
        latent_dim: Tuple[int, int, int] = (8, 16, None),  # C, H, T
    ):
        self.window_frames = window_frames
        self.samples_per_epoch = samples_per_epoch
        self.shift_range = shift_range
        self.artifact_removal_ratio = artifact_removal_ratio
        self.use_cached_latents = use_cached_latents
        self.latent_dim = latent_dim

        # Load manifest
        self.entries = self._load_manifest(manifest_path, instrument)
        print(f"Loaded {len(self.entries)} {instrument} entries")

        # Build pitch index from MIDI data
        self.pitch_index = self._build_pitch_index()
        print(f"Pitch index covers {len(self.pitch_index)} unique pitches")
        if self.pitch_index:
            print(f"Pitch range: {min(self.pitch_index.keys())} - {max(self.pitch_index.keys())}")

        # Cache for loaded latents
        self._latent_cache = {}
        self._cache_max_size = 500

    def _load_manifest(self, manifest_path: str, instrument: str) -> List[Dict]:
        """Load and filter manifest."""
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        entries = []
        for entry in manifest:
            # Filter by instrument (sub_group)
            if instrument and entry.get('sub_group') != instrument:
                continue

            # Get and fix paths
            audio_path = entry.get('audio_path', '')
            latent_path = fix_mount_path(entry.get('latent_path', ''))

            # Derive MIDI path
            midi_path = audio_path_to_midi_path(audio_path)

            # Store paths
            entry['audio_path'] = audio_path
            entry['latent_path'] = latent_path
            entry['midi_path'] = midi_path

            # Check if audio exists (latents can be computed on-the-fly)
            if os.path.exists(audio_path):
                entries.append(entry)

        return entries

    def _build_pitch_index(self) -> Dict[int, List[Dict]]:
        """
        Build index: pitch -> list of entries containing that pitch.
        Uses MIDI alignment to determine pitch content of each file.
        """
        pitch_index = defaultdict(list)

        for entry in self.entries:
            midi_path = entry.get('midi_path', '')

            # Try to load MIDI for pitch info
            if midi_path and os.path.exists(midi_path):
                try:
                    import pretty_midi
                    midi = pretty_midi.PrettyMIDI(midi_path)
                    pitches = set()
                    for instrument in midi.instruments:
                        for note in instrument.notes:
                            pitches.add(note.pitch)
                    for pitch in pitches:
                        pitch_index[pitch].append(entry)
                    entry['_pitches'] = list(pitches)
                    entry['_pitch_range'] = (min(pitches), max(pitches)) if pitches else (60, 60)
                except Exception as e:
                    # Fallback to instrument's typical range
                    self._add_default_pitch_range(entry, pitch_index)
            else:
                # Fallback to instrument's typical range
                self._add_default_pitch_range(entry, pitch_index)

        return dict(pitch_index)

    def _add_default_pitch_range(self, entry: Dict, pitch_index: Dict):
        """Add default pitch range based on instrument."""
        sub_group = entry.get('sub_group', '')

        # Instrument-specific pitch ranges (MIDI notes)
        ranges = {
            'trumpet': (52, 82),      # E3 to Bb5
            'trombone': (40, 72),     # E2 to C5
            'french_horn': (34, 77),  # Bb1 to F5
            'tuba': (28, 58),         # E1 to Bb3
            'violin': (55, 103),      # G3 to G7
            'viola': (48, 91),        # C3 to G6
            'cello': (36, 76),        # C2 to E5
            'upright_bass': (28, 60), # E1 to C4
            'electric_bass': (28, 60),
            'acoustic_guitar': (40, 88),
            'electric_guitar': (40, 88),
            'flute': (60, 96),        # C4 to C7
            'clarinet': (50, 94),     # D3 to Bb6
            'oboe': (58, 91),         # Bb3 to G6
        }

        low, high = ranges.get(sub_group, (48, 84))  # Default C3 to C6

        for pitch in range(low, high + 1):
            pitch_index[pitch].append(entry)

        entry['_pitches'] = list(range(low, high + 1))
        entry['_pitch_range'] = (low, high)

    def _load_latent(self, entry: Dict) -> Optional[torch.Tensor]:
        """Load latent from disk or cache."""
        latent_path = entry.get('latent_path', '')

        # Check cache
        if latent_path in self._latent_cache:
            return self._latent_cache[latent_path]

        # Try to load from disk
        if os.path.exists(latent_path):
            try:
                latent = torch.load(latent_path, map_location='cpu', weights_only=True)
                if isinstance(latent, dict):
                    latent = latent.get('latents', latent.get('latent', latent.get('z')))

                # Ensure correct shape [C, H, T]
                if latent.dim() == 4:
                    latent = latent.squeeze(0)

                # Cache with size limit
                if len(self._latent_cache) < self._cache_max_size:
                    self._latent_cache[latent_path] = latent

                return latent
            except Exception as e:
                print(f"Error loading {latent_path}: {e}")

        return None

    def _random_window(
        self,
        latent: torch.Tensor,
        target_frames: int = None,
    ) -> Tuple[torch.Tensor, int]:
        """Extract random window from latent."""
        if target_frames is None:
            target_frames = self.window_frames

        if latent.dim() == 4:
            latent = latent.squeeze(0)

        C, H, T = latent.shape

        if T <= target_frames:
            # Pad with zeros
            pad_amount = target_frames - T
            latent = F.pad(latent, (0, pad_amount))
            return latent, 0

        start = random.randint(0, T - target_frames)
        return latent[:, :, start:start + target_frames], start

    def _get_reference_at_pitch(self, target_pitch: int) -> Optional[torch.Tensor]:
        """Get a random latent segment at the target pitch for timbre reference."""
        if target_pitch not in self.pitch_index:
            # Find nearest available pitch
            available = list(self.pitch_index.keys())
            if not available:
                return None
            target_pitch = min(available, key=lambda x: abs(x - target_pitch))

        entries_at_pitch = self.pitch_index.get(target_pitch, [])
        if not entries_at_pitch:
            return None

        # Try several times to find a valid entry
        for _ in range(5):
            entry = random.choice(entries_at_pitch)
            latent = self._load_latent(entry)
            if latent is not None:
                window, _ = self._random_window(latent)
                return window

        return None

    def _simulate_shift_degradation(
        self,
        latent: torch.Tensor,
        shift: int
    ) -> torch.Tensor:
        """
        Simulate pitch shift degradation in latent space.

        This is a PLACEHOLDER - for production, you should:
        1. Decode latent to audio
        2. Apply actual pitch shift (librosa, rubberband, etc.)
        3. Re-encode to latent

        For initial testing, we simulate degradation:
        - Add noise proportional to shift amount
        - Blur along frequency axis (simulates formant smearing)
        """
        degraded = latent.clone()

        # Scale degradation by shift amount (0-1 for octave)
        intensity = abs(shift) / 12.0

        # 1. Add noise (pitch shift introduces spectral noise)
        noise = torch.randn_like(degraded) * 0.1 * intensity
        degraded = degraded + noise

        # 2. Frequency smearing (blur along H dimension)
        if intensity > 0.3:
            C, H, T = degraded.shape
            # Reshape for 2D pooling
            degraded_2d = degraded.unsqueeze(0)  # [1, C, H, T]
            kernel_size = int(2 * intensity) * 2 + 1
            if kernel_size >= 3:
                padding = kernel_size // 2
                degraded_2d = F.avg_pool2d(
                    degraded_2d,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0)
                )
                degraded = degraded_2d.squeeze(0)[:, :H, :T]

        return degraded

    def _fallback_sample(self) -> Dict[str, torch.Tensor]:
        """Return zeros as fallback for failed samples."""
        C, H, _ = self.latent_dim
        zeros = torch.zeros(C, H, self.window_frames)
        return {
            'input_latent': zeros,
            'target_latent': zeros,
            'reference_latent': zeros,
            'shift_amount': torch.tensor(0.0),
            'target_pitch': torch.tensor(60),
            'source_pitch': torch.tensor(60),
            'loss_type': torch.tensor(0),
            'valid': torch.tensor(False),
        }

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns training example.

        Two types:
        1. Artifact removal (double-shift): Learn to remove pitch-shift artifacts
        2. Register transfer (actual shift): Learn correct timbre for target pitch
        """
        # Randomly choose example type
        is_artifact_removal = random.random() < self.artifact_removal_ratio

        # Sample random entry and shift amount
        entry = random.choice(self.entries)
        shift = random.randint(self.shift_range[0], self.shift_range[1])
        if shift == 0:
            shift = random.choice([-1, 1])  # Avoid zero shift

        # Load original latent
        original_latent = self._load_latent(entry)

        # Retry if failed
        retries = 0
        while original_latent is None and retries < 10:
            entry = random.choice(self.entries)
            original_latent = self._load_latent(entry)
            retries += 1

        if original_latent is None:
            return self._fallback_sample()

        # Get window from original
        original_window, start_frame = self._random_window(original_latent)

        # Estimate pitch of this segment from MIDI info
        pitches = entry.get('_pitches', [60])
        source_pitch = random.choice(pitches) if pitches else 60

        if is_artifact_removal:
            # TYPE 1: Double-shift for artifact removal
            # Target pitch = source pitch (same pitch, just artifacts)
            target_pitch = source_pitch

            # Simulate double-shift degradation
            degraded_window = self._simulate_shift_degradation(original_window, shift)

            return {
                'input_latent': degraded_window,
                'target_latent': original_window,
                'reference_latent': original_window,  # Same as target for artifact removal
                'shift_amount': torch.tensor(shift, dtype=torch.float32),
                'target_pitch': torch.tensor(target_pitch, dtype=torch.long),
                'source_pitch': torch.tensor(source_pitch, dtype=torch.long),
                'loss_type': torch.tensor(0),  # 0 = artifact removal (full reconstruction)
                'valid': torch.tensor(True),
            }

        else:
            # TYPE 2: Actual shift for register transfer
            target_pitch = source_pitch + shift

            # Clamp to valid MIDI range
            target_pitch = max(24, min(96, target_pitch))

            # Get reference at target pitch (for timbre loss)
            reference = self._get_reference_at_pitch(target_pitch)
            if reference is None:
                reference = torch.zeros_like(original_window)

            # Simulate shifted input
            shifted_window = self._simulate_shift_degradation(original_window, shift)

            return {
                'input_latent': shifted_window,
                'target_latent': original_window,  # Content reference
                'reference_latent': reference,  # Timbre reference at target pitch
                'shift_amount': torch.tensor(shift, dtype=torch.float32),
                'target_pitch': torch.tensor(target_pitch, dtype=torch.long),
                'source_pitch': torch.tensor(source_pitch, dtype=torch.long),
                'loss_type': torch.tensor(1),  # 1 = register transfer (content + timbre)
                'valid': torch.tensor(True),
            }


class PitchShiftDatasetWithRealDegradation(PitchShiftDataset):
    """
    Extended dataset that uses actual audio pitch shifting for degradation.

    Requires DCAE encode/decode functions to be passed in.
    More accurate but slower.
    """

    def __init__(
        self,
        manifest_path: str,
        dcae_encode_fn,
        dcae_decode_fn,
        sample_rate: int = 44100,
        pitch_shift_algorithm: str = 'librosa',
        **kwargs
    ):
        super().__init__(manifest_path, **kwargs)
        self.dcae_encode = dcae_encode_fn
        self.dcae_decode = dcae_decode_fn
        self.sample_rate = sample_rate
        self.pitch_shift_algorithm = pitch_shift_algorithm

        # Audio cache
        self._audio_cache = {}
        self._audio_cache_max = 100

    def _load_audio(self, entry: Dict) -> Optional[Tuple[torch.Tensor, int]]:
        """Load audio from disk."""
        audio_path = entry.get('audio_path', '')

        if audio_path in self._audio_cache:
            return self._audio_cache[audio_path]

        if os.path.exists(audio_path):
            try:
                audio, sr = torchaudio.load(audio_path)

                # Resample if needed
                if sr != self.sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                    audio = resampler(audio)
                    sr = self.sample_rate

                # Ensure stereo
                if audio.shape[0] == 1:
                    audio = audio.repeat(2, 1)

                # Cache
                if len(self._audio_cache) < self._audio_cache_max:
                    self._audio_cache[audio_path] = (audio, sr)

                return audio, sr
            except Exception as e:
                print(f"Error loading audio {audio_path}: {e}")

        return None

    def _pitch_shift_audio(
        self,
        audio: torch.Tensor,
        sr: int,
        shift_semitones: int
    ) -> torch.Tensor:
        """Apply pitch shift to audio using librosa."""
        if shift_semitones == 0:
            return audio

        import librosa

        # Convert to numpy
        audio_np = audio.numpy()

        # Process each channel
        shifted_channels = []
        for ch in range(audio_np.shape[0]):
            shifted = librosa.effects.pitch_shift(
                audio_np[ch],
                sr=sr,
                n_steps=shift_semitones
            )
            shifted_channels.append(shifted)

        return torch.from_numpy(np.stack(shifted_channels))

    def _create_real_degraded_pair(
        self,
        entry: Dict,
        shift: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Create real degraded pair via audio domain double-shift.

        Returns: (degraded_latent, original_latent)
        """
        audio_data = self._load_audio(entry)
        if audio_data is None:
            return None

        audio, sr = audio_data

        # Double-shift in audio domain
        shifted_up = self._pitch_shift_audio(audio, sr, shift)
        shifted_back = self._pitch_shift_audio(shifted_up, sr, -shift)

        # Encode both
        with torch.no_grad():
            audio_batch = audio.unsqueeze(0)
            degraded_batch = shifted_back.unsqueeze(0)

            original_latent = self.dcae_encode(audio_batch, sr)
            degraded_latent = self.dcae_encode(degraded_batch, sr)

        return degraded_latent.squeeze(0), original_latent.squeeze(0)


if __name__ == "__main__":
    # Test the dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str,
                        default='/home/arlo/Data/final_training_manifest_final.json')
    parser.add_argument('--instrument', type=str, default='trumpet')
    args = parser.parse_args()

    print("Testing PitchShiftDataset...")

    dataset = PitchShiftDataset(
        manifest_path=args.manifest,
        instrument=args.instrument,
        window_frames=128,
        samples_per_epoch=100,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Number of entries: {len(dataset.entries)}")
    print(f"Pitches covered: {len(dataset.pitch_index)}")

    # Test a few samples
    for i in range(5):
        sample = dataset[i]
        print(f"\nSample {i}:")
        print(f"  Input shape: {sample['input_latent'].shape}")
        print(f"  Target shape: {sample['target_latent'].shape}")
        print(f"  Shift: {sample['shift_amount'].item()}")
        print(f"  Target pitch: {sample['target_pitch'].item()}")
        print(f"  Valid: {sample['valid'].item()}")
        print(f"  Loss type: {sample['loss_type'].item()} ({'artifact' if sample['loss_type'].item() == 0 else 'register'})")
