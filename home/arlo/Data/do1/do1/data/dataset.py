# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
DO1 Dataset for multi-task training.

Handles 7 training tasks with appropriate data construction for each:
- reconstruction (35%): Corrupt z_target, x_ref from same session
- separation (20%): Mix stems, x_ref is instrument reference
- cross_instrument (15%): Transfer to different instrument
- fx (10%): FX removal/application
- generation (10%): Generate from scratch with reference
- inpainting (5%): Fill temporal gaps
- synth_diversity (5%): Transfer between VST patches
"""

import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable

import torch
from torch.utils.data import Dataset

from .task_samplers import (
    TASK_DISTRIBUTION,
    sample_task,
    get_reconstruction_sample,
    get_separation_sample,
    get_cross_instrument_sample,
    get_generation_sample,
    get_inpainting_sample,
    get_synth_diversity_sample,
)
from .corruption import match_length
from .latent_synth import LatentSynthesizer
from .fx_pipeline import FXPipeline


class DO1Dataset(Dataset):
    """
    Multi-task dataset for DO1 training.

    Expected data layout:
        latents_dir/
            session_001/
                drums.pt        # [8, 16, T] latent
                bass.pt
                guitar.pt
                ...
            session_002/
                ...

        fx_pairs_dir/
            pair_001/
                dry.pt
                wet.pt
            ...

        vst_synths_dir/
            midi_001/
                patch_A.pt
                patch_B.pt
            ...

        instrument_labels.json  # {"session_001/drums.pt": "drums", ...}

    Args:
        latents_dir: Directory containing session latents
        fx_pairs_dir: Optional directory containing FX pairs
        vst_synths_dir: Optional directory containing VST synth pairs
        labels_path: Optional path to instrument labels JSON
        task_distribution: Task weight distribution (default: TASK_DISTRIBUTION)
        cfg_dropout_rate: Probability of dropping x_ref (default: 0.3)
        max_time_frames: Maximum temporal frames (for truncation)
        samples_per_epoch: Virtual epoch size (dataset repeats indefinitely)
    """

    def __init__(
        self,
        latents_dir: str,
        fx_pairs_dir: Optional[str] = None,
        vst_synths_dir: Optional[str] = None,
        labels_path: Optional[str] = None,
        task_distribution: Optional[Dict[str, float]] = None,
        cfg_dropout_rate: float = 0.3,
        max_time_frames: int = 4096,
        samples_per_epoch: int = 100000,
    ):
        self.latents_dir = Path(latents_dir)
        self.fx_pairs_dir = Path(fx_pairs_dir) if fx_pairs_dir else None
        self.vst_synths_dir = Path(vst_synths_dir) if vst_synths_dir else None
        self.task_distribution = task_distribution or TASK_DISTRIBUTION
        self.cfg_dropout_rate = cfg_dropout_rate
        self.max_time_frames = max_time_frames
        self.samples_per_epoch = samples_per_epoch

        # Load instrument labels
        self.instrument_labels: Dict[str, str] = {}
        if labels_path and Path(labels_path).exists():
            with open(labels_path, 'r') as f:
                self.instrument_labels = json.load(f)

        # Build indexes
        self._build_session_index()
        self._build_fx_index()
        self._build_vst_index()
        self._build_instrument_index()

        # Initialize helpers
        self.latent_synth = LatentSynthesizer()
        if self.fx_pairs_dir:
            self.fx_pipeline = FXPipeline(str(self.fx_pairs_dir))
        else:
            self.fx_pipeline = None

    def _build_session_index(self):
        """Build index of sessions and their stems."""
        self.sessions: Dict[str, List[Path]] = {}

        if not self.latents_dir.exists():
            raise ValueError(f"Latents directory does not exist: {self.latents_dir}")

        for session_dir in self.latents_dir.iterdir():
            if session_dir.is_dir():
                stems = list(session_dir.glob("*.pt")) + list(session_dir.glob("*.npy"))
                if stems:
                    self.sessions[session_dir.name] = stems

        self.all_stems: List[Path] = []
        for stems in self.sessions.values():
            self.all_stems.extend(stems)

        if not self.all_stems:
            raise ValueError(f"No latent files found in {self.latents_dir}")

    def _build_fx_index(self):
        """Build index of FX pairs."""
        self.fx_pairs: List[Path] = []

        if self.fx_pairs_dir and self.fx_pairs_dir.exists():
            for pair_dir in self.fx_pairs_dir.iterdir():
                if pair_dir.is_dir():
                    dry = pair_dir / "dry.pt"
                    wet = pair_dir / "wet.pt"
                    if dry.exists() and wet.exists():
                        self.fx_pairs.append(pair_dir)

    def _build_vst_index(self):
        """Build index of VST synth pairs."""
        self.vst_data: Dict[str, Dict[str, Path]] = {}

        if self.vst_synths_dir and self.vst_synths_dir.exists():
            for midi_dir in self.vst_synths_dir.iterdir():
                if midi_dir.is_dir():
                    patches = {}
                    for patch_file in midi_dir.glob("*.pt"):
                        patch_name = patch_file.stem
                        patches[patch_name] = patch_file
                    if patches:
                        self.vst_data[midi_dir.name] = patches

    def _build_instrument_index(self):
        """Build index mapping instruments to stems."""
        self.instrument_to_stems: Dict[str, List[Path]] = {}

        for stem_path in self.all_stems:
            rel_path = str(stem_path.relative_to(self.latents_dir))
            if rel_path in self.instrument_labels:
                instrument = self.instrument_labels[rel_path]
                if instrument not in self.instrument_to_stems:
                    self.instrument_to_stems[instrument] = []
                self.instrument_to_stems[instrument].append(stem_path)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_latent(self, path: Path) -> torch.Tensor:
        """Load a latent file (supports .pt and .npy)."""
        if path.suffix == '.npy':
            import numpy as np
            data = np.load(path)
            return torch.from_numpy(data).float()
        else:
            data = torch.load(path, map_location='cpu', weights_only=True)
            if isinstance(data, dict):
                # Try various key names
                for key in ['latent', 'latents', 'data', 'z']:
                    if key in data:
                        data = data[key]
                        break
            return data.float()

    def _load_random_stem(self) -> torch.Tensor:
        """Load a random stem from the dataset."""
        path = random.choice(self.all_stems)
        z = self._load_latent(path)
        return self._truncate(z)

    def _truncate(self, z: torch.Tensor) -> torch.Tensor:
        """Truncate to max time frames."""
        if z.shape[-1] > self.max_time_frames:
            start = random.randint(0, z.shape[-1] - self.max_time_frames)
            z = z[..., start:start + self.max_time_frames]
        return z

    def _get_session_stems(self, stem_path: Path) -> Dict[str, torch.Tensor]:
        """Get other stems from the same session."""
        session_name = stem_path.parent.name
        if session_name not in self.sessions:
            return {}

        other_stems = {}
        for other_path in self.sessions[session_name]:
            if other_path != stem_path:
                try:
                    z = self._load_latent(other_path)
                    other_stems[other_path.stem] = self._truncate(z)
                except Exception:
                    continue
        return other_stems

    def _get_instrument_ref(self, instrument: Optional[str]) -> torch.Tensor:
        """Get a reference stem for a given instrument."""
        if instrument and instrument in self.instrument_to_stems:
            path = random.choice(self.instrument_to_stems[instrument])
            z = self._load_latent(path)
            return self._truncate(z)
        else:
            # Fallback: random stem
            return self._load_random_stem()

    def _load_vst_dataset(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load VST dataset (lazy loading for the sample)."""
        result = {}
        for midi_id, patches in self.vst_data.items():
            result[midi_id] = {}
            for patch_id, path in patches.items():
                try:
                    z = self._load_latent(path)
                    result[midi_id][patch_id] = self._truncate(z)
                except Exception:
                    continue
        return result

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a training sample."""
        # Sample task
        task = sample_task()

        try:
            if task == 'reconstruction':
                return self._get_reconstruction_sample()
            elif task == 'separation':
                return self._get_separation_sample()
            elif task == 'cross_instrument':
                return self._get_cross_instrument_sample()
            elif task == 'fx':
                return self._get_fx_sample()
            elif task == 'generation':
                return self._get_generation_sample()
            elif task == 'inpainting':
                return self._get_inpainting_sample()
            elif task == 'synth_diversity':
                return self._get_synth_diversity_sample()
            else:
                # Fallback to reconstruction
                return self._get_reconstruction_sample()
        except Exception as e:
            # On error, return a simple reconstruction sample
            print(f"Error getting {task} sample: {e}, falling back to reconstruction")
            return self._get_reconstruction_sample()

    def _get_reconstruction_sample(self) -> Dict[str, Any]:
        """Get a reconstruction task sample."""
        # Load random target
        stem_path = random.choice(self.all_stems)
        z_target = self._load_latent(stem_path)
        z_target = self._truncate(z_target)

        # Get session stems
        session_stems = self._get_session_stems(stem_path)

        return get_reconstruction_sample(
            z_target=z_target,
            session_stems=session_stems,
            random_stem_fn=self._load_random_stem,
            cfg_dropout_rate=self.cfg_dropout_rate,
        )

    def _get_separation_sample(self) -> Dict[str, Any]:
        """Get a separation task sample."""
        # Find a session with multiple stems
        sessions_with_multiple = [
            name for name, stems in self.sessions.items() if len(stems) >= 2
        ]

        if not sessions_with_multiple:
            # Fallback to reconstruction
            return self._get_reconstruction_sample()

        session_name = random.choice(sessions_with_multiple)
        stems_paths = self.sessions[session_name]
        target_path = random.choice(stems_paths)

        z_target = self._load_latent(target_path)
        z_target = self._truncate(z_target)

        # Load other stems
        session_stems = {}
        for path in stems_paths:
            if path != target_path:
                try:
                    z = self._load_latent(path)
                    session_stems[path.stem] = self._truncate(z)
                except Exception:
                    continue

        # Get instrument label if available
        rel_path = str(target_path.relative_to(self.latents_dir))
        target_instrument = self.instrument_labels.get(rel_path)

        return get_separation_sample(
            z_target=z_target,
            session_stems=session_stems,
            instrument_ref_fn=self._get_instrument_ref,
            target_instrument=target_instrument,
            instrument_labels=self.instrument_labels,
            cfg_dropout_rate=self.cfg_dropout_rate,
        )

    def _get_cross_instrument_sample(self) -> Dict[str, Any]:
        """Get a cross-instrument task sample."""
        stem_path = random.choice(self.all_stems)
        z_target = self._load_latent(stem_path)
        z_target = self._truncate(z_target)

        session_stems = self._get_session_stems(stem_path)

        return get_cross_instrument_sample(
            z_target=z_target,
            session_stems=session_stems,
            latent_synth=self.latent_synth,
            midi_data=None,  # TODO: load MIDI if available
            cfg_dropout_rate=self.cfg_dropout_rate,
        )

    def _get_fx_sample(self) -> Dict[str, Any]:
        """Get an FX task sample."""
        if self.fx_pipeline is None or not self.fx_pairs:
            # Fallback to reconstruction with synthetic FX
            stem_path = random.choice(self.all_stems)
            z_dry = self._load_latent(stem_path)
            z_dry = self._truncate(z_dry)

            # Apply stub FX
            z_wet = self.latent_synth.normalize(
                self.latent_synth.unnormalize(z_dry) * random.uniform(0.8, 1.2) +
                0.05 * torch.randn_like(z_dry)
            )

            # Random direction
            if random.random() < 0.5:
                x_cond, z_target = z_wet, z_dry
            else:
                x_cond, z_target = z_dry, z_wet

            x_ref = z_target.clone()
            if random.random() < self.cfg_dropout_rate:
                x_ref = torch.zeros_like(x_ref)

            return {
                'x_cond': x_cond,
                'x_ref': x_ref,
                'z_target': z_target,
                'mask': torch.ones(1, z_target.shape[1], z_target.shape[2]),
                'task': 'fx',
                'loss_weight': 0.3,  # Synthetic weight
            }

        # Use FX pipeline
        sample = self.fx_pipeline.get_sample(
            random_stem_fn=self._load_random_stem
        )

        # Truncate all tensors
        T = min(self.max_time_frames, sample['z_target'].shape[-1])
        for key in ['x_cond', 'x_ref', 'z_target', 'mask']:
            sample[key] = match_length(sample[key], T)

        # CFG dropout
        if random.random() < self.cfg_dropout_rate:
            sample['x_ref'] = torch.zeros_like(sample['x_ref'])

        sample['task'] = 'fx'
        return sample

    def _get_generation_sample(self) -> Dict[str, Any]:
        """Get a generation task sample."""
        stem_path = random.choice(self.all_stems)
        z_target = self._load_latent(stem_path)
        z_target = self._truncate(z_target)

        rel_path = str(stem_path.relative_to(self.latents_dir))
        target_instrument = self.instrument_labels.get(rel_path)

        return get_generation_sample(
            z_target=z_target,
            instrument_ref_fn=self._get_instrument_ref,
            target_instrument=target_instrument,
            cfg_dropout_rate=self.cfg_dropout_rate,
        )

    def _get_inpainting_sample(self) -> Dict[str, Any]:
        """Get an inpainting task sample."""
        stem_path = random.choice(self.all_stems)
        z_target = self._load_latent(stem_path)
        z_target = self._truncate(z_target)

        session_stems = self._get_session_stems(stem_path)

        return get_inpainting_sample(
            z_target=z_target,
            session_stems=session_stems,
            cfg_dropout_rate=self.cfg_dropout_rate,
        )

    def _get_synth_diversity_sample(self) -> Dict[str, Any]:
        """Get a synth diversity task sample."""
        if not self.vst_data:
            # Fallback to cross-instrument
            return self._get_cross_instrument_sample()

        vst_dataset = self._load_vst_dataset()
        if not vst_dataset:
            return self._get_cross_instrument_sample()

        sample = get_synth_diversity_sample(
            vst_dataset=vst_dataset,
            cfg_dropout_rate=self.cfg_dropout_rate,
        )
        return sample
