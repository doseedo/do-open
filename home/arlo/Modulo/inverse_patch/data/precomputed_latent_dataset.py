"""
Dataset for loading precomputed DCAE latents.

Skips audio encoding during training for much faster iteration.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from typing import Dict, List, Optional
from pathlib import Path


class PrecomputedLatentDataset(Dataset):
    """Dataset that loads precomputed latent pairs with effect conditioning."""

    EFFECT_TYPES = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    # Parameter ranges for normalization (must match datasets.py)
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
            'low_gain_db': (-12.0, 12.0),
            'mid_gain_db': (-12.0, 12.0),
            'high_gain_db': (-12.0, 12.0),
            'low_freq': (20.0, 500.0),
            'mid_freq': (200.0, 5000.0),
            'high_freq': (2000.0, 20000.0),
        },
    }

    def __init__(
        self,
        manifest_path: str,
        max_samples: Optional[int] = None,
        max_chain_length: int = 4,
        max_params: int = 20,
    ):
        """
        Args:
            manifest_path: Path to JSON manifest with wet_latent_path/dry_latent_path
            max_samples: Limit number of samples (for debugging)
            max_chain_length: Maximum number of effects in chain
            max_params: Maximum parameters per effect
        """
        with open(manifest_path) as f:
            self.manifest = json.load(f)

        if max_samples:
            self.manifest = self.manifest[:max_samples]

        self.max_chain_length = max_chain_length
        self.max_params = max_params
        self.effect_type_map = {t: i for i, t in enumerate(self.EFFECT_TYPES)}

    def __len__(self):
        return len(self.manifest)

    def _normalize_param(self, effect_type: str, param_name: str, value: float) -> float:
        """Normalize a parameter value to [0, 1] range."""
        if effect_type not in self.PARAM_RANGES:
            return max(0.0, min(1.0, value))

        ranges = self.PARAM_RANGES[effect_type]
        if param_name not in ranges:
            return max(0.0, min(1.0, value))

        min_val, max_val = ranges[param_name]
        normalized = (value - min_val) / (max_val - min_val + 1e-8)
        return max(0.0, min(1.0, normalized))

    def _parse_chain_spec(self, chain_spec: List, chain_length: int) -> Dict[str, torch.Tensor]:
        """Parse chain_spec to tensors."""
        effect_types = torch.full((self.max_chain_length,), -1, dtype=torch.long)  # -1 = padding
        effect_params = torch.zeros(self.max_chain_length, self.max_params)

        for i, (effect_type, params) in enumerate(chain_spec):
            if i >= self.max_chain_length:
                break
            effect_types[i] = self.effect_type_map.get(effect_type, 0)

            for j, (param_name, v) in enumerate(params.items()):
                if j < self.max_params:
                    raw_value = float(v) if isinstance(v, (int, float)) else 0.0
                    effect_params[i, j] = self._normalize_param(effect_type, param_name, raw_value)

        return {
            'effect_types': effect_types,
            'effect_params': effect_params,
            'chain_length': torch.tensor(chain_length, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.manifest[idx]

        # Load precomputed latents
        wet_data = torch.load(item['wet_latent_path'], weights_only=True)
        dry_data = torch.load(item['dry_latent_path'], weights_only=True)

        wet_latent = wet_data['latent']  # [C, H, T]
        dry_latent = dry_data['latent']  # [C, H, T]

        result = {
            'wet_latent': wet_latent,
            'dry_latent': dry_latent,
            'id': item['id'],
        }

        # Parse effect info if available
        if 'chain_spec' in item:
            chain_data = self._parse_chain_spec(
                item['chain_spec'],
                item.get('chain_length', len(item['chain_spec']))
            )
            result.update(chain_data)
        else:
            # Default: no effect info (for backwards compatibility)
            result['effect_types'] = torch.full((self.max_chain_length,), -1, dtype=torch.long)
            result['effect_params'] = torch.zeros(self.max_chain_length, self.max_params)
            result['chain_length'] = torch.tensor(0, dtype=torch.long)

        return result


class PrecomputedLatentDataModule(pl.LightningDataModule):
    """DataModule for precomputed latent training."""

    def __init__(
        self,
        manifest_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        val_split: float = 0.1,
        max_samples: Optional[int] = None,
        max_chain_length: int = 4,
        max_params: int = 20,
    ):
        super().__init__()
        self.manifest_path = manifest_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.max_samples = max_samples
        self.max_chain_length = max_chain_length
        self.max_params = max_params

    def setup(self, stage: Optional[str] = None):
        # Load full manifest
        with open(self.manifest_path) as f:
            full_manifest = json.load(f)

        if self.max_samples:
            full_manifest = full_manifest[:self.max_samples]

        # Split into train/val
        n_val = int(len(full_manifest) * self.val_split)
        n_train = len(full_manifest) - n_val

        # Create temp manifests
        import tempfile
        import os

        self.train_manifest_path = os.path.join(tempfile.gettempdir(), 'train_manifest.json')
        self.val_manifest_path = os.path.join(tempfile.gettempdir(), 'val_manifest.json')

        with open(self.train_manifest_path, 'w') as f:
            json.dump(full_manifest[:n_train], f)
        with open(self.val_manifest_path, 'w') as f:
            json.dump(full_manifest[n_train:], f)

        self.train_dataset = PrecomputedLatentDataset(
            self.train_manifest_path,
            max_chain_length=self.max_chain_length,
            max_params=self.max_params,
        )
        self.val_dataset = PrecomputedLatentDataset(
            self.val_manifest_path,
            max_chain_length=self.max_chain_length,
            max_params=self.max_params,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
