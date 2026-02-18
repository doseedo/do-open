"""
Datasets and DataModules for Inverse Audio Effects System.
"""

import os
import torch
import torchaudio
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import random
import json

from .synthetic_chain_generator import EffectChainGenerator, ChainSpec
from .augmentations import AudioAugmentations


class InverseAFxDataset(Dataset):
    """
    Dataset for inverse audio effects training.

    Supports two modes:
    1. Online generation: Generate wet audio on-the-fly from dry audio files
    2. Precomputed: Load pre-generated dry/wet pairs
    """

    # Param ranges for normalizing manifest values to [0, 1]
    # Must match synthetic_chain_generator.py ranges
    PARAM_RANGES = {
        'compressor': {
            'threshold_db': (-60.0, 0.0),
            'ratio': (1.0, 20.0),
            'attack_ms': (0.1, 100.0),
            'release_ms': (10.0, 1000.0),
            'knee_db': (0.0, 12.0),
            'makeup_db': (0.0, 24.0),
        },
        'reverb': {
            'decay_time': (0.1, 10.0),
            'pre_delay_ms': (0.0, 100.0),
            'wet_mix': (0.0, 1.0),
            'damping': (0.0, 1.0),
        },
        'distortion': {
            'drive': (0.0, 1.0),
            'tone': (0.0, 1.0),
            'mix': (0.0, 1.0),
            'output_gain_db': (-12.0, 12.0),
        },
        'chorus': {
            'rate': (0.1, 10.0),
            'depth': (0.0, 1.0),
            'mix': (0.0, 1.0),
            'feedback': (0.0, 0.9),
        },
        'delay': {
            'delay_ms': (1.0, 2000.0),
            'feedback': (0.0, 0.95),
            'mix': (0.0, 1.0),
        },
        'eq': {
            # Ranges must match nablafx.processors.ddsp.ParametricEQ
            # Order: low_shelf (gain, freq, q), band0 (gain, freq, q), band1, band2, high_shelf
            'low_shelf_gain_db': (-12.0, 12.0),
            'low_shelf_cutoff_freq': (20.0, 2000.0),
            'low_shelf_q_factor': (0.1, 10.0),
            'band0_gain_db': (-12.0, 12.0),
            'band0_cutoff_freq': (20.0, 200.0),
            'band0_q_factor': (0.1, 10.0),
            'band1_gain_db': (-12.0, 12.0),
            'band1_cutoff_freq': (200.0, 2000.0),
            'band1_q_factor': (0.1, 10.0),
            'band2_gain_db': (-12.0, 12.0),
            'band2_cutoff_freq': (2000.0, 12000.0),
            'band2_q_factor': (0.1, 10.0),
            'high_shelf_gain_db': (-12.0, 12.0),
            'high_shelf_cutoff_freq': (4000.0, 16000.0),
            'high_shelf_q_factor': (0.1, 10.0),
        },
    }

    def __init__(
        self,
        audio_dir: Union[str, Path],
        sample_rate: int = 44100,
        segment_length: int = 131072,  # ~3 seconds at 44.1kHz
        max_chain_length: int = 4,
        effect_types: Optional[List[str]] = None,
        mode: str = 'online',  # 'online' or 'precomputed'
        precomputed_manifest: Optional[str] = None,
        augment: bool = True,
        device: str = 'cpu',
        load_dry: bool = True,  # Set False to skip dry audio (faster for encoder-only training)
    ):
        super().__init__()
        self.audio_dir = Path(audio_dir)
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.max_chain_length = max_chain_length
        self.effect_types = effect_types or EffectChainGenerator.EFFECT_TYPES
        self.mode = mode
        self.augment = augment
        self.device = device
        self.load_dry = load_dry

        # Initialize components
        if mode == 'online':
            self.chain_generator = EffectChainGenerator(
                sample_rate=sample_rate,
                max_chain_length=max_chain_length,
                effect_types=effect_types,
                device=device,
            )
            self.audio_files = self._find_audio_files()
        elif mode == 'precomputed':
            assert precomputed_manifest is not None
            self.manifest = self._load_manifest(precomputed_manifest)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if augment:
            self.augmentations = AudioAugmentations(sample_rate=sample_rate)
        else:
            self.augmentations = None

    def _find_audio_files(self) -> List[Path]:
        """Find all audio files in directory."""
        extensions = ['.wav', '.mp3', '.flac', '.ogg', '.aiff']
        files = []
        for ext in extensions:
            files.extend(self.audio_dir.rglob(f'*{ext}'))
        return sorted(files)

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load precomputed data manifest."""
        with open(manifest_path, 'r') as f:
            return json.load(f)

    def __len__(self) -> int:
        if self.mode == 'online':
            return len(self.audio_files)
        else:
            return len(self.manifest)

    def _load_audio(self, path: Path) -> torch.Tensor:
        """Load audio file and resample if needed."""
        waveform, sr = torchaudio.load(str(path))

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform

    def _get_segment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract random segment of specified length."""
        length = waveform.size(-1)

        if length < self.segment_length:
            # Pad if too short
            padding = self.segment_length - length
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        elif length > self.segment_length:
            # Random crop if too long
            start = random.randint(0, length - self.segment_length)
            waveform = waveform[..., start:start + self.segment_length]

        return waveform

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == 'online':
            return self._get_online_item(idx)
        else:
            return self._get_precomputed_item(idx)

    def _get_online_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Generate sample on-the-fly."""
        # Load dry audio
        audio_path = self.audio_files[idx]
        dry_audio = self._load_audio(audio_path)

        # Get random segment
        dry_audio = self._get_segment(dry_audio)

        # Apply augmentations to dry audio
        if self.augmentations is not None:
            dry_audio = self.augmentations(dry_audio)

        # Normalize
        dry_audio = dry_audio / (dry_audio.abs().max() + 1e-8)

        # Generate wet audio
        dry_audio = dry_audio.unsqueeze(0)  # [1, 1, T]
        wet_audio, chain_spec, intermediates = self.chain_generator.generate_sample(
            dry_audio
        )

        # Convert chain spec to tensors
        chain_data = self._chain_spec_to_tensors(chain_spec)

        return {
            'dry_audio': dry_audio.squeeze(0),  # [1, T]
            'wet_audio': wet_audio.squeeze(0),  # [1, T]
            'effect_types': chain_data['effect_types'],
            'effect_params': chain_data['effect_params'],
            'chain_length': chain_data['chain_length'],
        }

    def _get_precomputed_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load precomputed sample."""
        item = self.manifest[idx]

        # Load wet audio directly
        wet_audio = self._load_audio(Path(item['wet_path']))
        wet_audio = self._get_segment(wet_audio)

        # Force exact segment length (safety check for mismatched sample rates)
        if wet_audio.size(-1) != self.segment_length:
            if wet_audio.size(-1) < self.segment_length:
                wet_audio = torch.nn.functional.pad(wet_audio, (0, self.segment_length - wet_audio.size(-1)))
            else:
                wet_audio = wet_audio[..., :self.segment_length]

        # Load dry audio only if needed (skip for encoder-only training)
        if self.load_dry:
            if 'dry_path' in item:
                # Old format: dry audio saved separately
                dry_audio = self._load_audio(Path(item['dry_path']))
                dry_audio = self._get_segment(dry_audio)
            else:
                # New format: load segment from source file
                dry_audio = self._load_dry_segment(item)

            if dry_audio.size(-1) != self.segment_length:
                if dry_audio.size(-1) < self.segment_length:
                    dry_audio = torch.nn.functional.pad(dry_audio, (0, self.segment_length - dry_audio.size(-1)))
                else:
                    dry_audio = dry_audio[..., :self.segment_length]
        else:
            # Dummy tensor when dry audio not needed
            dry_audio = torch.zeros(1, self.segment_length)

        # Parse chain spec to tensors
        chain_data = self._parse_chain_spec(item['chain_spec'], item['chain_length'])

        return {
            'dry_audio': dry_audio,
            'wet_audio': wet_audio,
            'effect_types': chain_data['effect_types'],
            'effect_params': chain_data['effect_params'],
            'chain_length': chain_data['chain_length'],
        }

    def _load_dry_segment(self, item: Dict) -> torch.Tensor:
        """Load dry audio segment - prefer precomputed, fallback to source."""
        # Check for precomputed dry audio first (fast local storage)
        dry_path = item.get('dry_path')
        if dry_path and Path(dry_path).exists():
            waveform, sr = torchaudio.load(dry_path)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            # Ensure correct length
            if waveform.size(-1) < self.segment_length:
                padding = self.segment_length - waveform.size(-1)
                waveform = torch.nn.functional.pad(waveform, (0, padding))
            elif waveform.size(-1) > self.segment_length:
                waveform = waveform[..., :self.segment_length]
            return waveform

        # Fallback: load from source file (slow for gcsfuse)
        source_path = item['source_path']
        segment_start = item['segment_start']
        segment_length = item['segment_length']
        needs_padding = item.get('needs_padding', False)

        waveform, sr = torchaudio.load(source_path)

        # Convert to mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            # Adjust segment positions for new sample rate
            ratio = self.sample_rate / sr
            segment_start = int(segment_start * ratio)
            segment_length = int(segment_length * ratio)

        # Extract segment
        if needs_padding:
            segment = waveform
        else:
            end_pos = min(segment_start + segment_length, waveform.size(-1))
            segment = waveform[..., segment_start:end_pos]

        # Always ensure correct length by padding if needed
        if segment.size(-1) < self.segment_length:
            padding = self.segment_length - segment.size(-1)
            segment = torch.nn.functional.pad(segment, (0, padding))
        elif segment.size(-1) > self.segment_length:
            segment = segment[..., :self.segment_length]

        # Normalize (same as during generation)
        max_val = segment.abs().max()
        if max_val > 0:
            segment = segment / (max_val + 1e-8)
        segment = segment * 0.9

        return segment

    def _normalize_param(self, effect_type: str, param_name: str, value: float) -> float:
        """Normalize a parameter value to [0, 1] range based on effect type."""
        if effect_type not in self.PARAM_RANGES:
            return value  # Return as-is if effect type unknown

        ranges = self.PARAM_RANGES[effect_type]
        if param_name not in ranges:
            # If param name not found, assume it's already normalized
            return max(0.0, min(1.0, value))

        min_val, max_val = ranges[param_name]
        # Normalize: (value - min) / (max - min)
        normalized = (value - min_val) / (max_val - min_val + 1e-8)
        # Clamp to [0, 1] to handle out-of-range values
        return max(0.0, min(1.0, normalized))

    def _parse_chain_spec(self, chain_spec: List, chain_length: int) -> Dict[str, torch.Tensor]:
        """Parse chain_spec list format to tensors with normalized params."""
        effect_type_map = {t: i for i, t in enumerate(self.effect_types)}

        max_params = 20
        max_effects = self.max_chain_length

        effect_types = torch.zeros(max_effects, dtype=torch.long)
        effect_params = torch.zeros(max_effects, max_params)

        for i, (effect_type, params) in enumerate(chain_spec):
            if i >= max_effects:
                break
            effect_types[i] = effect_type_map.get(effect_type, 0)

            # Store NORMALIZED parameters
            for j, (param_name, v) in enumerate(params.items()):
                if j < max_params:
                    raw_value = float(v) if isinstance(v, (int, float)) else 0.0
                    # Normalize to [0, 1] based on effect type and param name
                    normalized_value = self._normalize_param(effect_type, param_name, raw_value)
                    effect_params[i, j] = normalized_value

        return {
            'effect_types': effect_types,
            'effect_params': effect_params,
            'chain_length': torch.tensor(chain_length),
        }

    def _chain_spec_to_tensors(self, chain_spec: ChainSpec) -> Dict[str, torch.Tensor]:
        """Convert ChainSpec to tensor format for batching."""
        # Map effect types to indices
        effect_type_map = {t: i for i, t in enumerate(self.effect_types)}

        max_params = 20  # Maximum parameters per effect
        max_effects = self.max_chain_length

        effect_types = torch.zeros(max_effects, dtype=torch.long)
        effect_params = torch.zeros(max_effects, max_params)

        for i, effect in enumerate(chain_spec.effects):
            if i >= max_effects:
                break

            effect_types[i] = effect_type_map.get(effect.effect_type, 0)

            # Store normalized parameters
            for j, (k, v) in enumerate(effect.normalized_params.items()):
                if j < max_params:
                    effect_params[i, j] = v if isinstance(v, float) else v.item()

        return {
            'effect_types': effect_types,
            'effect_params': effect_params,
            'chain_length': torch.tensor(len(chain_spec)),
        }


class InverseAFxDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for inverse audio effects.
    """

    def __init__(
        self,
        train_dir: Union[str, Path],
        val_dir: Optional[Union[str, Path]] = None,
        test_dir: Optional[Union[str, Path]] = None,
        sample_rate: int = 44100,
        segment_length: int = 131072,
        batch_size: int = 16,
        num_workers: int = 8,
        max_chain_length: int = 4,
        effect_types: Optional[List[str]] = None,
        mode: str = 'online',
        train_manifest: Optional[str] = None,
        val_manifest: Optional[str] = None,
        test_manifest: Optional[str] = None,
        pin_memory: bool = True,
        load_dry: bool = True,  # Set False for encoder-only training
        prefetch_factor: int = 4,  # Prefetch batches per worker
    ):
        super().__init__()
        self.save_hyperparameters()

        self.train_dir = Path(train_dir) if train_dir else None
        self.val_dir = Path(val_dir) if val_dir else self.train_dir
        self.test_dir = Path(test_dir) if test_dir else self.val_dir

    def setup(self, stage: Optional[str] = None):
        """Set up datasets."""
        if stage == 'fit' or stage is None:
            self.train_dataset = InverseAFxDataset(
                audio_dir=self.train_dir,
                sample_rate=self.hparams.sample_rate,
                segment_length=self.hparams.segment_length,
                max_chain_length=self.hparams.max_chain_length,
                effect_types=self.hparams.effect_types,
                mode=self.hparams.mode,
                precomputed_manifest=self.hparams.train_manifest,
                augment=False,  # Disabled: precomputed mode doesn't use augmentation anyway
                load_dry=self.hparams.load_dry,
            )

            self.val_dataset = InverseAFxDataset(
                audio_dir=self.val_dir,
                sample_rate=self.hparams.sample_rate,
                segment_length=self.hparams.segment_length,
                max_chain_length=self.hparams.max_chain_length,
                effect_types=self.hparams.effect_types,
                mode=self.hparams.mode,
                precomputed_manifest=self.hparams.val_manifest,
                augment=False,
                load_dry=self.hparams.load_dry,
            )

        if stage == 'test' or stage is None:
            self.test_dataset = InverseAFxDataset(
                audio_dir=self.test_dir,
                sample_rate=self.hparams.sample_rate,
                segment_length=self.hparams.segment_length,
                max_chain_length=self.hparams.max_chain_length,
                effect_types=self.hparams.effect_types,
                mode=self.hparams.mode,
                precomputed_manifest=self.hparams.test_manifest,
                augment=False,
                load_dry=self.hparams.load_dry,
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            drop_last=True,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            prefetch_factor=self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None,
            persistent_workers=self.hparams.num_workers > 0,
        )
