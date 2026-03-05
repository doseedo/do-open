# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
FX Processing Pipeline for DO1.

This module handles:
1. Loading real dry/wet FX pairs
2. Applying synthetic FX to create augmented pairs
3. Temporal FX application (partial FX regions)

TODO: Implement audio-space FX via DCAE decode/encode
TODO: Implement FX chain (eq, compressor, reverb, etc.)
"""

import random
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path

import torch


class FXPipeline:
    """
    Pipeline for FX-related training tasks.

    Handles loading dry/wet pairs and (eventually) applying
    synthetic FX through DCAE decode -> FX -> encode pipeline.

    Args:
        fx_pairs_dir: Directory containing real dry/wet pairs
        dcae: Optional DCAE model for synthetic FX application
    """

    def __init__(
        self,
        fx_pairs_dir: Optional[str] = None,
        dcae: Optional[Any] = None,
    ):
        self.fx_pairs_dir = Path(fx_pairs_dir) if fx_pairs_dir else None
        self.dcae = dcae

        # Build index of available pairs
        self.pair_index: List[Path] = []
        if self.fx_pairs_dir and self.fx_pairs_dir.exists():
            self._build_pair_index()

        # Available FX types for synthetic processing
        self.available_fx = [
            'eq',
            'compressor',
            'distortion',
            'reverb',
            'chorus',
            'delay',
            'pitch_shift',
        ]

    def _build_pair_index(self):
        """Build index of available FX pairs."""
        for pair_dir in self.fx_pairs_dir.iterdir():
            if pair_dir.is_dir():
                dry_path = pair_dir / "dry.pt"
                wet_path = pair_dir / "wet.pt"
                if dry_path.exists() and wet_path.exists():
                    self.pair_index.append(pair_dir)

    def load_random_pair(self) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Load a random dry/wet pair.

        Returns:
            Tuple of (dry_latent, wet_latent, source_type)
            - source_type is "real" for loaded pairs
        """
        if not self.pair_index:
            raise ValueError("No FX pairs available. Check fx_pairs_dir.")

        pair_dir = random.choice(self.pair_index)
        dry = torch.load(pair_dir / "dry.pt", map_location="cpu")
        wet = torch.load(pair_dir / "wet.pt", map_location="cpu")

        # Handle different storage formats
        if isinstance(dry, dict):
            dry = dry.get('latent', dry.get('data', dry.get('latents')))
        if isinstance(wet, dict):
            wet = wet.get('latent', wet.get('data', wet.get('latents')))

        return dry, wet, "real"

    def apply_fx_chain(
        self,
        z_dry: torch.Tensor,
        fx_chain: Optional[List[str]] = None,
        num_fx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply FX chain to dry latent.

        TODO: Implement actual FX processing through DCAE.

        For now, this returns a stub that adds noise to simulate
        FX application (placeholder for real implementation).

        Args:
            z_dry: Dry latent [8, 16, T]
            fx_chain: List of FX to apply, or None for random selection
            num_fx: Number of FX to apply if fx_chain is None

        Returns:
            Wet latent [8, 16, T]
        """
        if self.dcae is None:
            # Stub: add noise to simulate FX
            return self._apply_stub_fx(z_dry)

        # Full implementation would:
        # 1. Decode to audio: audio = dcae.decode(z_dry)
        # 2. Apply FX chain in audio space
        # 3. Re-encode: z_wet = dcae.encode(audio)
        raise NotImplementedError(
            "Full FX pipeline not yet implemented. "
            "Provide DCAE model and implement audio-space FX."
        )

    def _apply_stub_fx(self, z_dry: torch.Tensor) -> torch.Tensor:
        """
        Stub FX application for testing.

        Applies simple transformations to simulate FX:
        - Add noise (simulates distortion/saturation)
        - Scale (simulates compression)
        - Temporal smearing (simulates reverb/delay)
        """
        z_wet = z_dry.clone()

        # Random scaling (compression-like)
        scale = random.uniform(0.8, 1.2)
        z_wet = z_wet * scale

        # Add small noise (saturation-like)
        noise_level = random.uniform(0.01, 0.1)
        z_wet = z_wet + torch.randn_like(z_wet) * noise_level

        # Temporal smearing (reverb-like) via averaging
        if random.random() < 0.5:
            kernel_size = random.choice([3, 5, 7])
            z_wet = torch.nn.functional.avg_pool1d(
                z_wet.view(-1, z_wet.shape[-1]).unsqueeze(0),
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            ).squeeze(0).view_as(z_dry)

        return z_wet

    def apply_temporal_fx_mask(
        self,
        z_dry: torch.Tensor,
        z_wet: torch.Tensor,
        coverage_range: Tuple[float, float] = (0.3, 0.7),
        num_regions_range: Tuple[int, int] = (2, 5),
    ) -> torch.Tensor:
        """
        Apply FX only to random temporal regions.

        Creates a mix where some regions have FX and others are dry,
        simulating partial FX application (e.g., reverb on verse only).

        Args:
            z_dry: Dry latent [8, 16, T]
            z_wet: Wet latent [8, 16, T]
            coverage_range: Range of total coverage (fraction of frames)
            num_regions_range: Range of number of wet regions

        Returns:
            Partially processed latent [8, 16, T]
        """
        T = z_dry.shape[-1]
        fx_mask = torch.zeros(1, 1, T)

        target_coverage = random.uniform(*coverage_range)
        frames_to_fill = int(T * target_coverage)
        num_regions = random.randint(*num_regions_range)

        filled = 0
        for _ in range(num_regions):
            if filled >= frames_to_fill:
                break
            span = random.randint(5, max(6, T // num_regions))
            span = min(span, frames_to_fill - filled)
            start = random.randint(0, max(0, T - span))
            fx_mask[:, :, start:start + span] = 1.0
            filled += span

        # Blend dry and wet based on mask
        z_partial = z_dry * (1 - fx_mask) + z_wet * fx_mask

        return z_partial

    def get_sample(
        self,
        random_stem_fn: Optional[Callable[[], torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Get a complete FX task sample.

        Handles the three source tiers:
        - Real (40%): Load real dry/wet pair, weight 1.0
        - Augmented (30%): Real dry + synthetic FX, weight 0.5
        - Synthetic (30%): Random stem + synthetic FX, weight 0.3

        Args:
            random_stem_fn: Function to get a random stem for synthetic pairs

        Returns:
            Dict with:
                - x_cond: Input latent (dry or wet depending on direction)
                - x_ref: Reference latent (target character)
                - z_target: Target latent
                - mask: Task mask (all ones for FX)
                - loss_weight: Per-sample loss weight
                - direction: "removal" or "application"
        """
        # Select source tier
        source_weights = {'real': 0.4, 'augmented': 0.3, 'synthetic': 0.3}
        source = random.choices(
            list(source_weights.keys()),
            weights=list(source_weights.values()),
            k=1
        )[0]

        if source == 'real':
            z_dry, z_wet, _ = self.load_random_pair()
            loss_weight = 1.0
        elif source == 'augmented':
            z_dry, _, _ = self.load_random_pair()
            z_wet = self.apply_fx_chain(z_dry)
            loss_weight = 0.5
        else:  # synthetic
            if random_stem_fn is None:
                raise ValueError("random_stem_fn required for synthetic FX")
            z_dry = random_stem_fn()
            z_wet = self.apply_fx_chain(z_dry)
            loss_weight = 0.3

        # Apply temporal mask 50% of the time
        if random.random() < 0.5:
            z_wet = self.apply_temporal_fx_mask(z_dry, z_wet)

        # Random direction: 50% removal, 50% application
        if random.random() < 0.5:
            # FX removal: wet -> dry
            x_cond = z_wet
            z_target = z_dry
            direction = "removal"
            # x_ref should indicate dry character
            x_ref = z_dry  # Could be a different dry reference
        else:
            # FX application: dry -> wet
            x_cond = z_dry
            z_target = z_wet
            direction = "application"
            # x_ref should indicate wet character
            x_ref = z_wet  # Could be a different wet reference

        mask = torch.ones(1, z_target.shape[1], z_target.shape[2])

        return {
            'x_cond': x_cond,
            'x_ref': x_ref,
            'z_target': z_target,
            'mask': mask,
            'loss_weight': loss_weight,
            'direction': direction,
        }
