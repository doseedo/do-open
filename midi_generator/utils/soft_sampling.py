"""
Differentiable Discrete Sampling - Agent 2
==========================================

GumbelSoftmaxSampler: Differentiable discrete sampling for neural networks.

Implements the Gumbel-Softmax distribution (also known as Concrete distribution)
which provides a continuous, differentiable approximation to sampling from
categorical distributions.

Key Features:
- Fully differentiable sampling from discrete distributions
- Temperature annealing for training schedules
- Straight-through estimator for hard samples with soft gradients
- Supports multi-dimensional inputs

References:
- "Categorical Reparameterization with Gumbel-Softmax" (Jang et al., 2017)
- "The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables" (Maddison et al., 2017)

Author: Agent 2 - Differentiable MIDI & Utilities Support
Date: November 22, 2025
License: MIT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union
import math


# ============================================================================
# Gumbel-Softmax Sampler
# ============================================================================

class GumbelSoftmaxSampler:
    """
    Implements Gumbel-Softmax (Concrete distribution) for differentiable discrete sampling.

    The Gumbel-Softmax distribution provides a continuous, differentiable
    approximation to sampling from categorical distributions. This enables
    training neural networks that need to make discrete decisions while
    maintaining gradient flow.

    Mathematical Background:
    -----------------------
    Given logits z = (z₁, ..., zₖ), the Gumbel-Softmax distribution samples:

        y_i = exp((log(π_i) + g_i) / τ) / Σⱼ exp((log(π_j) + g_j) / τ)

    where:
    - π_i = softmax(z)_i are the categorical probabilities
    - g_i ~ Gumbel(0, 1) are Gumbel noise samples
    - τ > 0 is the temperature parameter

    Properties:
    - As τ → 0: Approaches one-hot (deterministic, discrete)
    - As τ → ∞: Approaches uniform distribution
    - For all τ > 0: Fully differentiable

    Usage:
        # During decoder forward pass
        logits = decoder.predict_pitch(hidden)  # (batch, time, 128)

        # Soft sampling (training)
        temp = GumbelSoftmaxSampler.anneal_temperature(step, 1.0, 0.1)
        soft_samples = GumbelSoftmaxSampler.sample(logits, temp, hard=False)

        # Hard sampling (inference)
        hard_samples = GumbelSoftmaxSampler.sample(logits, 0.1, hard=True)
    """

    @staticmethod
    def sample(
        logits: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False,
        dim: int = -1,
        eps: float = 1e-10
    ) -> torch.Tensor:
        """
        Sample from Gumbel-Softmax distribution.

        Args:
            logits: Unnormalized log probabilities, shape (..., num_categories)
            temperature: Temperature parameter (lower = more discrete)
                - τ = 1.0: Standard softmax
                - τ → 0.0: Approaches argmax (one-hot)
                - τ → ∞: Uniform distribution
            hard: If True, return one-hot (forward) but soft (backward) - straight-through estimator
            dim: Dimension to apply softmax (default: -1)
            eps: Small constant for numerical stability

        Returns:
            Sampled tensor, same shape as logits
            - If hard=False: Soft probabilities (fully differentiable)
            - If hard=True: One-hot vectors (forward), soft gradients (backward)

        Example:
            >>> logits = torch.randn(32, 100, 128)  # Batch of pitch predictions
            >>> soft = GumbelSoftmaxSampler.sample(logits, temperature=1.0, hard=False)
            >>> hard = GumbelSoftmaxSampler.sample(logits, temperature=0.1, hard=True)
        """
        # Sample Gumbel noise
        gumbel_noise = GumbelSoftmaxSampler._sample_gumbel(logits.shape, eps=eps, device=logits.device)

        # Add Gumbel noise to logits
        gumbel_logits = logits + gumbel_noise

        # Apply temperature scaling
        scaled_logits = gumbel_logits / temperature

        # Softmax
        y_soft = F.softmax(scaled_logits, dim=dim)

        if hard:
            # Straight-through estimator
            # Forward pass: one-hot (discrete)
            # Backward pass: soft (continuous gradients)
            index = y_soft.max(dim, keepdim=True)[1]
            y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)

            # Trick: y_hard - y_soft.detach() + y_soft
            # Forward: y_hard (one-hot)
            # Backward: gradient of y_soft flows through
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft

        return ret

    @staticmethod
    def _sample_gumbel(
        shape: torch.Size,
        eps: float = 1e-10,
        device: Union[str, torch.device] = 'cpu'
    ) -> torch.Tensor:
        """
        Sample from Gumbel(0, 1) distribution.

        The Gumbel distribution is sampled using the inverse CDF method:
            G = -log(-log(U))
        where U ~ Uniform(0, 1)

        Args:
            shape: Shape of samples to generate
            eps: Small constant for numerical stability
            device: Device to create tensor on

        Returns:
            Gumbel samples of specified shape
        """
        uniform = torch.rand(shape, device=device)
        # Clamp to avoid log(0)
        uniform = torch.clamp(uniform, min=eps, max=1.0 - eps)
        gumbel = -torch.log(-torch.log(uniform))
        return gumbel

    @staticmethod
    def anneal_temperature(
        step: int,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        anneal_rate: float = 0.00003,
        anneal_type: str = 'exponential'
    ) -> float:
        """
        Compute annealed temperature for training schedule.

        Temperature annealing gradually reduces temperature during training,
        moving from soft, exploratory sampling (high temp) to hard, deterministic
        sampling (low temp).

        Args:
            step: Current training step
            initial_temp: Starting temperature (default: 1.0, soft/exploratory)
            final_temp: Final temperature (default: 0.1, hard/deterministic)
            anneal_rate: Decay rate (default: 0.00003)
            anneal_type: Type of annealing schedule
                - 'exponential': temp = max(final, initial * exp(-rate * step))
                - 'linear': temp = max(final, initial - rate * step)
                - 'cosine': Cosine annealing schedule

        Returns:
            Current temperature

        Example Training Schedule (exponential, rate=0.00003):
            Step 0:     temp = 1.0
            Step 10k:   temp = 0.74
            Step 30k:   temp = 0.41
            Step 60k:   temp = 0.17
            Step 80k:   temp = 0.1 (final)
        """
        if anneal_type == 'exponential':
            temp = initial_temp * math.exp(-anneal_rate * step)
            return max(final_temp, temp)

        elif anneal_type == 'linear':
            temp = initial_temp - anneal_rate * step
            return max(final_temp, temp)

        elif anneal_type == 'cosine':
            # Cosine annealing: smooth decay
            # Need to know total steps for proper cosine
            # Assume we want to reach final_temp at step = 100k
            total_steps = int(math.log(initial_temp / final_temp) / anneal_rate)
            progress = min(step / total_steps, 1.0)
            temp = final_temp + (initial_temp - final_temp) * 0.5 * (1 + math.cos(math.pi * progress))
            return temp

        else:
            raise ValueError(f"Unknown anneal_type: {anneal_type}")

    @staticmethod
    def sample_categorical(
        probs: torch.Tensor,
        temperature: float = 1.0,
        hard: bool = False
    ) -> torch.Tensor:
        """
        Sample from categorical distribution using Gumbel-Softmax.

        Convenience method that converts probabilities to logits.

        Args:
            probs: Probability distribution (must sum to 1 along last dim)
            temperature: Temperature parameter
            hard: Whether to use straight-through estimator

        Returns:
            Sampled distribution
        """
        # Convert probabilities to logits
        eps = 1e-10
        logits = torch.log(probs + eps)
        return GumbelSoftmaxSampler.sample(logits, temperature, hard)


# ============================================================================
# Temperature Scheduler (for training loops)
# ============================================================================

class TemperatureScheduler:
    """
    Temperature scheduler for managing Gumbel-Softmax temperature during training.

    Usage in training loop:
        scheduler = TemperatureScheduler(initial=1.0, final=0.1, decay=0.00003)

        for step in range(num_steps):
            temp = scheduler.get_temperature(step)
            samples = GumbelSoftmaxSampler.sample(logits, temperature=temp)

            # Or use step() method
            temp = scheduler.step()
    """

    def __init__(
        self,
        initial_temp: float = 1.0,
        final_temp: float = 0.1,
        anneal_rate: float = 0.00003,
        anneal_type: str = 'exponential'
    ):
        """
        Initialize temperature scheduler.

        Args:
            initial_temp: Starting temperature
            final_temp: Minimum temperature
            anneal_rate: Decay rate
            anneal_type: 'exponential', 'linear', or 'cosine'
        """
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_rate = anneal_rate
        self.anneal_type = anneal_type
        self.current_step = 0

    def get_temperature(self, step: Optional[int] = None) -> float:
        """
        Get temperature for a given step.

        Args:
            step: Training step (uses self.current_step if None)

        Returns:
            Temperature value
        """
        step = step if step is not None else self.current_step
        return GumbelSoftmaxSampler.anneal_temperature(
            step=step,
            initial_temp=self.initial_temp,
            final_temp=self.final_temp,
            anneal_rate=self.anneal_rate,
            anneal_type=self.anneal_type
        )

    def step(self) -> float:
        """
        Advance one step and return temperature.

        Returns:
            Temperature at current step
        """
        temp = self.get_temperature(self.current_step)
        self.current_step += 1
        return temp

    def reset(self):
        """Reset step counter"""
        self.current_step = 0

    def __repr__(self) -> str:
        return (
            f"TemperatureScheduler(initial={self.initial_temp}, "
            f"final={self.final_temp}, rate={self.anneal_rate}, "
            f"type={self.anneal_type}, step={self.current_step})"
        )


# ============================================================================
# Utilities
# ============================================================================

def test_gradient_flow():
    """
    Test that gradients flow through Gumbel-Softmax sampling.

    This test verifies that:
    1. Soft sampling has gradients
    2. Hard sampling has gradients (via straight-through)
    3. Temperature affects output distribution
    """
    print("Testing Gumbel-Softmax gradient flow...")

    # Create dummy logits (batch=2, time=10, categories=128)
    logits = torch.randn(2, 10, 128, requires_grad=True)

    # Soft sampling
    soft_samples = GumbelSoftmaxSampler.sample(logits, temperature=1.0, hard=False)
    soft_loss = soft_samples.sum()
    soft_loss.backward()

    assert logits.grad is not None, "Soft sampling: No gradient!"
    print("✓ Soft sampling: Gradients flow")

    # Hard sampling
    logits.grad = None  # Reset gradient
    hard_samples = GumbelSoftmaxSampler.sample(logits, temperature=0.1, hard=True)
    hard_loss = hard_samples.sum()
    hard_loss.backward()

    assert logits.grad is not None, "Hard sampling: No gradient!"
    print("✓ Hard sampling: Gradients flow (straight-through)")

    # Temperature effect
    high_temp = GumbelSoftmaxSampler.sample(logits, temperature=10.0, hard=False)
    low_temp = GumbelSoftmaxSampler.sample(logits, temperature=0.1, hard=False)

    high_entropy = -(high_temp * torch.log(high_temp + 1e-10)).sum(dim=-1).mean()
    low_entropy = -(low_temp * torch.log(low_temp + 1e-10)).sum(dim=-1).mean()

    assert high_entropy > low_entropy, "Temperature doesn't affect entropy!"
    print(f"✓ Temperature effect: High temp entropy={high_entropy:.3f}, Low temp entropy={low_entropy:.3f}")

    print("All gradient flow tests passed! ✅")


if __name__ == "__main__":
    # Run gradient flow test
    test_gradient_flow()

    # Demonstrate temperature annealing
    print("\nTemperature annealing schedule:")
    for step in [0, 10000, 30000, 60000, 100000]:
        temp = GumbelSoftmaxSampler.anneal_temperature(step, 1.0, 0.1, 0.00003)
        print(f"  Step {step:6d}: temp = {temp:.4f}")

    # Demonstrate temperature scheduler
    print("\nTemperature scheduler:")
    scheduler = TemperatureScheduler(initial=1.0, final=0.1, anneal_rate=0.0001)
    for i in range(5):
        temp = scheduler.step()
        print(f"  Step {scheduler.current_step - 1}: temp = {temp:.4f}")
