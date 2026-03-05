# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
Rectified Flow / Conditional Flow Matching implementation for DO1.

Flow matching provides a simpler and more stable training objective
compared to DDPM-style diffusion:

- Forward process: x_t = (1-t)*noise + t*x_1 (linear interpolation)
- Target: v = x_1 - noise (velocity field)
- Loss: MSE(v_pred, v_target)

At inference, we solve the ODE dx/dt = v(x, t) from t=0 to t=1.
"""

from typing import Tuple, Optional, Literal
import torch
import torch.nn.functional as F


class RectifiedFlowMatching:
    """
    Conditional Flow Matching with Rectified Flow.

    This class handles:
    - Timestep sampling with various distributions
    - Noise addition according to flow matching interpolation
    - Velocity target computation

    Args:
        timestep_distribution: How to sample timesteps
            - "uniform": t ~ U[0, 1]
            - "logit_normal": t ~ sigmoid(N(mean, std)) for better training (SD3 style)
        logit_mean: Mean for logit-normal distribution
        logit_std: Std for logit-normal distribution
        shift: Timestep shift factor (similar to SD3)
    """

    def __init__(
        self,
        timestep_distribution: Literal["uniform", "logit_normal"] = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        shift: float = 1.0,
    ):
        self.timestep_distribution = timestep_distribution
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.shift = shift

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Sample timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Device to create tensor on

        Returns:
            Timesteps tensor [B] in range [0, 1]
        """
        if self.timestep_distribution == "uniform":
            t = torch.rand(batch_size, device=device)

        elif self.timestep_distribution == "logit_normal":
            # Sample from logit-normal distribution
            # This concentrates samples around the middle timesteps
            # which is where the model needs to learn the most
            u = torch.randn(batch_size, device=device) * self.logit_std + self.logit_mean
            t = torch.sigmoid(u)

        else:
            raise ValueError(f"Unknown timestep distribution: {self.timestep_distribution}")

        # Apply shift if needed
        if self.shift != 1.0:
            t = self.shift * t / (1 + (self.shift - 1) * t)

        # Clamp to avoid edge cases
        t = t.clamp(1e-5, 1.0 - 1e-5)

        return t

    def add_noise(
        self,
        z_target: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Add noise according to rectified flow interpolation.

        Forward process: x_t = (1-t)*noise + t*x_1

        Args:
            z_target: Target clean sample [B, C, H, W]
            t: Timesteps [B] in range [0, 1]
            noise: Optional pre-sampled noise (for reproducibility)

        Returns:
            Tuple of:
                - x_noisy: Noised sample at timestep t [B, C, H, W]
                - noise: The noise that was added [B, C, H, W]
                - v_target: Target velocity field [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(z_target)

        # Expand t for broadcasting: [B] -> [B, 1, 1, 1]
        t_expanded = t[:, None, None, None]

        # Linear interpolation: x_t = (1-t)*noise + t*x_1
        x_noisy = (1 - t_expanded) * noise + t_expanded * z_target

        # Velocity target: v = x_1 - noise
        v_target = z_target - noise

        return x_noisy, noise, v_target

    def compute_loss(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        loss_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute weighted MSE loss for velocity prediction.

        Args:
            v_pred: Predicted velocity [B, C, H, W]
            v_target: Target velocity [B, C, H, W]
            loss_weight: Optional per-sample weights [B]
            reduction: "mean", "sum", or "none"

        Returns:
            Loss value (scalar if reduction != "none")
        """
        # Compute MSE per sample
        mse = F.mse_loss(v_pred, v_target, reduction="none")  # [B, C, H, W]

        # Reduce spatial dimensions
        mse = mse.mean(dim=(1, 2, 3))  # [B]

        # Apply per-sample weights
        if loss_weight is not None:
            mse = mse * loss_weight

        # Final reduction
        if reduction == "mean":
            return mse.mean()
        elif reduction == "sum":
            return mse.sum()
        else:
            return mse

    def compute_loss_masked(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute MSE loss only on valid (non-padded) positions.

        Args:
            v_pred: Predicted velocity [B, C, H, W]
            v_target: Target velocity [B, C, H, W]
            attention_mask: Valid positions [B, W] where 1=valid, 0=padding
            loss_weight: Optional per-sample weights [B]

        Returns:
            Mean loss over valid positions
        """
        # Compute MSE
        mse = F.mse_loss(v_pred, v_target, reduction="none")  # [B, C, H, W]

        # Expand mask: [B, W] -> [B, 1, 1, W]
        mask = attention_mask[:, None, None, :]

        # Apply mask
        masked_mse = mse * mask

        # Mean over valid positions per sample
        valid_elements = mask.sum(dim=(1, 2, 3))  # [B]
        mse_per_sample = masked_mse.sum(dim=(1, 2, 3)) / (valid_elements + 1e-8)  # [B]

        # Apply per-sample weights
        if loss_weight is not None:
            mse_per_sample = mse_per_sample * loss_weight

        return mse_per_sample.mean()


class FlowMatchingScheduler:
    """
    Scheduler for flow matching inference.

    Handles timestep scheduling and step computation for ODE solving.

    Args:
        num_train_timesteps: Number of discretization steps
        shift: Timestep shift factor
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
    ):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift

        # Compute sigma schedule
        timesteps = torch.linspace(1.0, num_train_timesteps, num_train_timesteps)[::-1]
        sigmas = timesteps / num_train_timesteps

        if shift != 1.0:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """
        Set timesteps for inference.

        Args:
            num_inference_steps: Number of inference steps
            device: Device for tensors
        """
        self.num_inference_steps = num_inference_steps

        # Compute inference timesteps (evenly spaced in sigma space)
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = torch.arange(0, num_inference_steps) * step_ratio
        timesteps = timesteps.flip(0)  # Start from high noise

        sigmas = timesteps / self.num_train_timesteps
        if self.shift != 1.0:
            sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)

        self.inference_sigmas = sigmas.to(device)
        self.inference_timesteps = (sigmas * self.num_train_timesteps).to(device)

    def step(
        self,
        v_pred: torch.Tensor,
        t: torch.Tensor,
        x: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """
        Perform one Euler step of the ODE.

        ODE: dx/dt = v(x, t)
        Euler: x_{t+dt} = x_t + v(x_t, t) * dt

        Args:
            v_pred: Predicted velocity
            t: Current timestep
            x: Current sample
            dt: Step size (typically 1/num_steps)

        Returns:
            Updated sample
        """
        return x + v_pred * dt
