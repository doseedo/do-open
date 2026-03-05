# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Inference pipeline for DO1.

Provides a high-level interface for running DO1 inference with:
- Classifier-free guidance (CFG)
- Multiple sampling methods
- Batch processing
"""

from typing import Optional, Union, Literal
from pathlib import Path

import torch
import torch.nn.functional as F

from ..models import DO1Transformer2DModel, get_do1_config_3b, get_do1_config_small
from .sampling import get_sampler, EulerSampler, HeunSampler


class DO1InferencePipeline:
    """
    Inference pipeline for DO1 model.

    Provides easy-to-use interface for running inference with:
    - Automatic model loading
    - Classifier-free guidance
    - Multiple sampler options
    - Batch processing

    Args:
        model: DO1Transformer2DModel or path to checkpoint
        device: Device to run on
        dtype: Data type (default: bfloat16 for efficiency)
    """

    def __init__(
        self,
        model: Union[DO1Transformer2DModel, str, Path],
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        # Load model
        if isinstance(model, (str, Path)):
            self.model = self._load_checkpoint(model)
        else:
            self.model = model

        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def _load_checkpoint(self, path: Union[str, Path]) -> DO1Transformer2DModel:
        """Load model from checkpoint."""
        path = Path(path)

        if path.is_dir():
            # Load from directory (assume config.json + model.pt)
            config_path = path / "config.json"
            model_path = path / "model.pt"

            if config_path.exists():
                import json
                with open(config_path) as f:
                    config = json.load(f)
            else:
                config = get_do1_config_3b()

            model = DO1Transformer2DModel(**config)

            if model_path.exists():
                state_dict = torch.load(model_path, map_location='cpu')
                model.load_state_dict(state_dict)
        else:
            # Load from single file
            state_dict = torch.load(path, map_location='cpu')

            # Extract config if present
            if 'config' in state_dict:
                config = state_dict['config']
                model = DO1Transformer2DModel(**config)
                model.load_state_dict(state_dict['model'])
            else:
                # Assume default config
                model = DO1Transformer2DModel(**get_do1_config_3b())
                model.load_state_dict(state_dict)

        return model

    @torch.no_grad()
    def __call__(
        self,
        x_cond: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
        num_steps: int = 50,
        cfg_scale: float = 2.0,
        sampler: Literal["euler", "heun", "midpoint"] = "heun",
        verbose: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Run inference.

        Args:
            x_cond: Conditioning latent [B, 8, 16, T]
            x_ref: Reference latent [B, 8, 16, T']
            mask: Task mask [B, 1, 16, T]
            num_steps: Number of sampling steps
            cfg_scale: Classifier-free guidance scale (1.0 = no guidance)
            sampler: Sampling method
            verbose: Whether to show progress bar
            generator: Optional random generator for reproducibility

        Returns:
            Generated latent [B, 8, 16, T]
        """
        # Move inputs to device
        x_cond = x_cond.to(device=self.device, dtype=self.dtype)
        x_ref = x_ref.to(device=self.device, dtype=self.dtype)
        mask = mask.to(device=self.device, dtype=self.dtype)

        B, C, H, T = x_cond.shape
        shape = (B, C, H, T)

        # Create model function with CFG
        if cfg_scale > 1.0:
            model_fn = self._create_cfg_model_fn(x_cond, x_ref, mask, cfg_scale)
        else:
            model_fn = self._create_model_fn(x_cond, x_ref, mask)

        # Get sampler
        sampler_obj = get_sampler(sampler, num_steps=num_steps, verbose=verbose)

        # Sample
        output = sampler_obj.sample(
            model_fn=model_fn,
            shape=shape,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )

        return output

    def _create_model_fn(
        self,
        x_cond: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
    ):
        """Create model function for sampling."""
        def model_fn(x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            output = self.model(
                x_noisy=x_noisy,
                x_cond=x_cond,
                x_ref=x_ref,
                mask=mask,
                timestep=t,
                return_dict=True,
            )
            return output.sample

        return model_fn

    def _create_cfg_model_fn(
        self,
        x_cond: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
        cfg_scale: float,
    ):
        """Create model function with classifier-free guidance."""
        # Pre-compute unconditional x_ref (zeros)
        x_ref_uncond = torch.zeros_like(x_ref)

        def model_fn(x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Conditional prediction
            v_cond = self.model(
                x_noisy=x_noisy,
                x_cond=x_cond,
                x_ref=x_ref,
                mask=mask,
                timestep=t,
                return_dict=True,
            ).sample

            # Unconditional prediction
            v_uncond = self.model(
                x_noisy=x_noisy,
                x_cond=x_cond,
                x_ref=x_ref_uncond,
                mask=mask,
                timestep=t,
                return_dict=True,
            ).sample

            # CFG combination
            v = v_uncond + cfg_scale * (v_cond - v_uncond)
            return v

        return model_fn

    @torch.no_grad()
    def reconstruct(
        self,
        x_corrupted: torch.Tensor,
        x_ref: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Reconstruct from corrupted input.

        Convenience wrapper for reconstruction task.

        Args:
            x_corrupted: Corrupted latent [B, 8, 16, T]
            x_ref: Reference latent [B, 8, 16, T']
            mask: Optional task mask (defaults to all ones)
            **kwargs: Additional arguments for __call__

        Returns:
            Reconstructed latent [B, 8, 16, T]
        """
        if mask is None:
            mask = torch.ones(
                x_corrupted.shape[0], 1, x_corrupted.shape[2], x_corrupted.shape[3],
                device=x_corrupted.device,
                dtype=x_corrupted.dtype,
            )

        return self(x_cond=x_corrupted, x_ref=x_ref, mask=mask, **kwargs)

    @torch.no_grad()
    def separate(
        self,
        x_mix: torch.Tensor,
        x_ref: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Separate target stem from mix.

        Convenience wrapper for separation task.

        Args:
            x_mix: Mixed latent [B, 8, 16, T]
            x_ref: Reference for target instrument [B, 8, 16, T']
            **kwargs: Additional arguments for __call__

        Returns:
            Separated stem latent [B, 8, 16, T]
        """
        # Mask is all zeros for separation (predict everything)
        mask = torch.zeros(
            x_mix.shape[0], 1, x_mix.shape[2], x_mix.shape[3],
            device=x_mix.device,
            dtype=x_mix.dtype,
        )

        return self(x_cond=x_mix, x_ref=x_ref, mask=mask, **kwargs)

    @torch.no_grad()
    def generate(
        self,
        x_ref: torch.Tensor,
        length: int,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate new audio from reference.

        Convenience wrapper for generation task.

        Args:
            x_ref: Reference latent (style/timbre) [B, 8, 16, T']
            length: Target length in frames
            **kwargs: Additional arguments for __call__

        Returns:
            Generated latent [B, 8, 16, length]
        """
        B = x_ref.shape[0]

        # x_cond is all zeros for generation
        x_cond = torch.zeros(
            B, 8, 16, length,
            device=x_ref.device,
            dtype=x_ref.dtype,
        )

        # Mask is all zeros (generate everything)
        mask = torch.zeros(
            B, 1, 16, length,
            device=x_ref.device,
            dtype=x_ref.dtype,
        )

        return self(x_cond=x_cond, x_ref=x_ref, mask=mask, **kwargs)

    @torch.no_grad()
    def inpaint(
        self,
        x_partial: torch.Tensor,
        x_ref: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Fill in masked regions.

        Convenience wrapper for inpainting task.

        Args:
            x_partial: Partial latent with gaps [B, 8, 16, T]
            x_ref: Reference latent [B, 8, 16, T']
            mask: Mask where 0=inpaint, 1=keep [B, 1, 16, T]
            **kwargs: Additional arguments for __call__

        Returns:
            Inpainted latent [B, 8, 16, T]
        """
        return self(x_cond=x_partial, x_ref=x_ref, mask=mask, **kwargs)

    @torch.no_grad()
    def transfer_timbre(
        self,
        x_source: torch.Tensor,
        x_ref: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Transfer timbre from reference to source.

        Convenience wrapper for cross-instrument/timbre transfer.

        Args:
            x_source: Source latent (content) [B, 8, 16, T]
            x_ref: Reference latent (timbre) [B, 8, 16, T']
            **kwargs: Additional arguments for __call__

        Returns:
            Timbre-transferred latent [B, 8, 16, T]
        """
        # Mask is all ones (transfer timbre while preserving content)
        mask = torch.ones(
            x_source.shape[0], 1, x_source.shape[2], x_source.shape[3],
            device=x_source.device,
            dtype=x_source.dtype,
        )

        return self(x_cond=x_source, x_ref=x_ref, mask=mask, **kwargs)
