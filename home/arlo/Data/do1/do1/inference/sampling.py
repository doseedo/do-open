# Copyright 2024 DO1 Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""
ODE solvers for flow matching inference.

At inference, we solve the ODE: dx/dt = v(x, t) from t=0 (noise) to t=1 (data).
We provide several solver options:
- EulerSampler: First-order Euler method (fast, lower quality)
- HeunSampler: Second-order Heun method (better quality)
- MidpointSampler: Second-order midpoint method
"""

from typing import Callable, Optional, Tuple
import torch
from tqdm import tqdm


class EulerSampler:
    """
    First-order Euler ODE solver.

    Simple and fast, but may require more steps for good quality.
    x_{t+dt} = x_t + v(x_t, t) * dt

    Args:
        num_steps: Number of discretization steps
        verbose: Whether to show progress bar
    """

    def __init__(
        self,
        num_steps: int = 50,
        verbose: bool = True,
    ):
        self.num_steps = num_steps
        self.verbose = verbose

    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Euler method.

        Args:
            model_fn: Function (x_t, t) -> v(x_t, t) that returns velocity
            shape: Shape of samples to generate
            device: Device to generate on
            dtype: Data type
            generator: Optional random generator

        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        dt = 1.0 / self.num_steps
        timesteps = torch.linspace(0, 1 - dt, self.num_steps, device=device)

        iterator = tqdm(timesteps, desc="Euler sampling") if self.verbose else timesteps

        for t in iterator:
            t_batch = t.expand(shape[0])
            v = model_fn(x, t_batch)
            x = x + v * dt

        return x


class HeunSampler:
    """
    Second-order Heun ODE solver (improved Euler / explicit trapezoid).

    Higher quality than Euler with the same number of steps,
    but requires two model evaluations per step.

    Heun's method:
        k1 = v(x_t, t)
        x_tilde = x_t + k1 * dt
        k2 = v(x_tilde, t + dt)
        x_{t+dt} = x_t + (k1 + k2) * dt / 2

    Args:
        num_steps: Number of discretization steps
        verbose: Whether to show progress bar
    """

    def __init__(
        self,
        num_steps: int = 25,
        verbose: bool = True,
    ):
        self.num_steps = num_steps
        self.verbose = verbose

    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate samples using Heun method.

        Args:
            model_fn: Function (x_t, t) -> v(x_t, t) that returns velocity
            shape: Shape of samples to generate
            device: Device to generate on
            dtype: Data type
            generator: Optional random generator

        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        dt = 1.0 / self.num_steps
        timesteps = torch.linspace(0, 1 - dt, self.num_steps, device=device)

        iterator = tqdm(timesteps, desc="Heun sampling") if self.verbose else timesteps

        for t in iterator:
            t_batch = t.expand(shape[0])
            t_next_batch = (t + dt).expand(shape[0])

            # First evaluation
            k1 = model_fn(x, t_batch)

            # Euler prediction
            x_tilde = x + k1 * dt

            # Second evaluation
            k2 = model_fn(x_tilde, t_next_batch)

            # Heun update
            x = x + (k1 + k2) * dt / 2

        return x


class MidpointSampler:
    """
    Second-order midpoint ODE solver.

    Alternative to Heun with similar quality.

    Midpoint method:
        k1 = v(x_t, t)
        x_mid = x_t + k1 * dt / 2
        k2 = v(x_mid, t + dt/2)
        x_{t+dt} = x_t + k2 * dt

    Args:
        num_steps: Number of discretization steps
        verbose: Whether to show progress bar
    """

    def __init__(
        self,
        num_steps: int = 25,
        verbose: bool = True,
    ):
        self.num_steps = num_steps
        self.verbose = verbose

    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate samples using midpoint method.

        Args:
            model_fn: Function (x_t, t) -> v(x_t, t) that returns velocity
            shape: Shape of samples to generate
            device: Device to generate on
            dtype: Data type
            generator: Optional random generator

        Returns:
            Generated samples
        """
        # Start from pure noise
        x = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        dt = 1.0 / self.num_steps
        timesteps = torch.linspace(0, 1 - dt, self.num_steps, device=device)

        iterator = tqdm(timesteps, desc="Midpoint sampling") if self.verbose else timesteps

        for t in iterator:
            t_batch = t.expand(shape[0])
            t_mid_batch = (t + dt / 2).expand(shape[0])

            # First evaluation
            k1 = model_fn(x, t_batch)

            # Midpoint
            x_mid = x + k1 * dt / 2

            # Midpoint evaluation
            k2 = model_fn(x_mid, t_mid_batch)

            # Update
            x = x + k2 * dt

        return x


class DPMSolverPlusPlusSampler:
    """
    DPM-Solver++ for flow matching (experimental).

    Higher-order solver that can achieve good quality with fewer steps.
    This is a simplified implementation - for best results, use the
    full DPM-Solver++ implementation.

    Args:
        num_steps: Number of discretization steps
        verbose: Whether to show progress bar
    """

    def __init__(
        self,
        num_steps: int = 20,
        verbose: bool = True,
    ):
        self.num_steps = num_steps
        self.verbose = verbose

    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, ...],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Generate samples using DPM-Solver++.

        For now, this falls back to Heun method.
        TODO: Implement full DPM-Solver++ for flow matching.
        """
        # Fallback to Heun for now
        heun = HeunSampler(num_steps=self.num_steps, verbose=self.verbose)
        return heun.sample(model_fn, shape, device, dtype, generator)


def get_sampler(
    name: str,
    num_steps: int = 50,
    verbose: bool = True,
):
    """
    Get a sampler by name.

    Args:
        name: Sampler name ('euler', 'heun', 'midpoint', 'dpm')
        num_steps: Number of sampling steps
        verbose: Whether to show progress bar

    Returns:
        Sampler instance
    """
    samplers = {
        'euler': EulerSampler,
        'heun': HeunSampler,
        'midpoint': MidpointSampler,
        'dpm': DPMSolverPlusPlusSampler,
    }

    if name not in samplers:
        raise ValueError(f"Unknown sampler: {name}. Available: {list(samplers.keys())}")

    return samplers[name](num_steps=num_steps, verbose=verbose)
