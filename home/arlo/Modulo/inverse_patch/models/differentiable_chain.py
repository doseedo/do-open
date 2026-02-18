"""
Differentiable FX Chain for verification and refinement.

Applies effect chains differentiably for:
1. Verification: apply estimated chain to dry -> should match wet
2. Refinement: gradient descent on params to minimize wet reconstruction error
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import sys
sys.path.insert(0, '../..')

from nablafx.processors import ParametricEQ, Gain
from ..data.synthetic_chain_generator import (
    DifferentiableCompressor,
    DifferentiableReverb,
    DifferentiableDistortion,
    DifferentiableChorus,
    DifferentiableDelay,
)


class DifferentiableFXChain(nn.Module):
    """
    Differentiable effect chain for verification and refinement.

    Given estimated parameters, applies effect chain differentiably.
    Used for:
    1. Verification: apply estimated chain to dry -> should match wet
    2. Refinement: gradient descent on params to minimize wet reconstruction error
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        effect_types: Optional[List[str]] = None,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.effect_types = effect_types or [
            'eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay'
        ]

        # Initialize all effect processors
        self.effects = nn.ModuleDict()

        if 'eq' in self.effect_types:
            self.effects['eq'] = ParametricEQ(
                sample_rate=sample_rate,
                control_type='static',
            )

        if 'compressor' in self.effect_types:
            self.effects['compressor'] = DifferentiableCompressor(
                sample_rate=sample_rate,
            )

        if 'reverb' in self.effect_types:
            self.effects['reverb'] = DifferentiableReverb(
                sample_rate=sample_rate,
            )

        if 'distortion' in self.effect_types:
            self.effects['distortion'] = DifferentiableDistortion(
                sample_rate=sample_rate,
            )

        if 'chorus' in self.effect_types:
            self.effects['chorus'] = DifferentiableChorus(
                sample_rate=sample_rate,
            )

        if 'delay' in self.effect_types:
            self.effects['delay'] = DifferentiableDelay(
                sample_rate=sample_rate,
            )

        if 'gain' in self.effect_types:
            self.effects['gain'] = Gain(
                sample_rate=sample_rate,
                control_type='static',
            )

    def forward(
        self,
        dry_audio: torch.Tensor,
        chain_spec: List[Tuple[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        Apply effect chain to dry audio.

        Args:
            dry_audio: Dry audio [B, 1, T]
            chain_spec: List of (effect_type, params) tuples
                params should be normalized [0, 1] tensors

        Returns:
            wet_audio: Processed audio [B, 1, T]
        """
        if dry_audio.dim() == 2:
            dry_audio = dry_audio.unsqueeze(1)

        signal = dry_audio

        for effect_type, params in chain_spec:
            if effect_type not in self.effects:
                continue

            effect = self.effects[effect_type]

            # Format params for effect
            if params.dim() == 1:
                params = params.unsqueeze(0)
            if params.dim() == 2:
                params = params.unsqueeze(-1)

            signal, _ = effect(signal, params, train=True)

        return signal

    def apply_single_effect(
        self,
        audio: torch.Tensor,
        effect_type: str,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply a single effect.

        Args:
            audio: Input audio [B, 1, T]
            effect_type: Type of effect
            params: Effect parameters (normalized)

        Returns:
            Processed audio
        """
        if effect_type not in self.effects:
            raise ValueError(f"Unknown effect type: {effect_type}")

        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        if params.dim() == 1:
            params = params.unsqueeze(0)
        if params.dim() == 2:
            params = params.unsqueeze(-1)

        output, _ = self.effects[effect_type](audio, params, train=True)
        return output

    def refine_params(
        self,
        wet_target: torch.Tensor,
        dry_estimate: torch.Tensor,
        chain_spec: List[Tuple[str, torch.Tensor]],
        n_steps: int = 100,
        lr: float = 0.01,
        loss_fn: Optional[nn.Module] = None,
    ) -> List[Tuple[str, torch.Tensor]]:
        """
        Gradient descent refinement of estimated parameters.

        Args:
            wet_target: Target wet audio [B, 1, T]
            dry_estimate: Estimated dry audio [B, 1, T]
            chain_spec: Initial chain specification
            n_steps: Number of optimization steps
            lr: Learning rate
            loss_fn: Loss function (default: MSE + STFT)

        Returns:
            Refined chain specification
        """
        # Create learnable parameter tensors
        param_list = []
        refined_spec = []

        for effect_type, params in chain_spec:
            # Clone and make learnable
            learnable_params = params.clone().detach().requires_grad_(True)
            param_list.append(learnable_params)
            refined_spec.append((effect_type, learnable_params))

        if not param_list:
            return chain_spec

        optimizer = torch.optim.Adam(param_list, lr=lr)

        # Default loss function
        if loss_fn is None:
            from ..training.losses import MultiResolutionSTFTLoss
            loss_fn = MultiResolutionSTFTLoss()

        for step in range(n_steps):
            optimizer.zero_grad()

            # Apply chain
            wet_reconstructed = self.forward(dry_estimate, refined_spec)

            # Compute loss
            loss = loss_fn(wet_reconstructed, wet_target)

            # Backward
            loss.backward()
            optimizer.step()

            # Clamp to valid ranges
            with torch.no_grad():
                for params in param_list:
                    params.clamp_(0.0, 1.0)

        # Detach parameters
        final_spec = [
            (effect_type, params.detach())
            for effect_type, params in refined_spec
        ]

        return final_spec

    def verify_chain(
        self,
        wet_target: torch.Tensor,
        dry_estimate: torch.Tensor,
        chain_spec: List[Tuple[str, torch.Tensor]],
    ) -> Dict[str, float]:
        """
        Verify chain estimation by comparing reconstruction to target.

        Args:
            wet_target: Target wet audio
            dry_estimate: Estimated dry audio
            chain_spec: Estimated chain specification

        Returns:
            Dictionary of verification metrics
        """
        with torch.no_grad():
            wet_reconstructed = self.forward(dry_estimate, chain_spec)

            # Compute metrics
            mse = F.mse_loss(wet_reconstructed, wet_target).item()

            # Correlation
            wet_flat = wet_target.flatten()
            recon_flat = wet_reconstructed.flatten()
            correlation = torch.corrcoef(
                torch.stack([wet_flat, recon_flat])
            )[0, 1].item()

            # SNR
            noise = wet_target - wet_reconstructed
            signal_power = (wet_target ** 2).mean()
            noise_power = (noise ** 2).mean()
            snr = 10 * torch.log10(signal_power / (noise_power + 1e-8)).item()

        return {
            'mse': mse,
            'correlation': correlation,
            'snr_db': snr,
        }


class ChainOrderOptimizer(nn.Module):
    """
    Optimizes the order of effects in a chain.
    Some orderings may produce better inversions than others.
    """

    def __init__(
        self,
        fx_chain: DifferentiableFXChain,
        effect_types: List[str],
    ):
        super().__init__()
        self.fx_chain = fx_chain
        self.effect_types = effect_types

    def find_best_order(
        self,
        wet_target: torch.Tensor,
        dry_estimate: torch.Tensor,
        effects_present: List[str],
        params_dict: Dict[str, torch.Tensor],
        max_permutations: int = 24,
    ) -> Tuple[List[str], float]:
        """
        Find the best ordering of effects.

        Args:
            wet_target: Target wet audio
            dry_estimate: Estimated dry audio
            effects_present: List of effects in the chain
            params_dict: Parameters for each effect
            max_permutations: Maximum number of orderings to try

        Returns:
            best_order: Best effect ordering
            best_score: Score for best ordering
        """
        from itertools import permutations

        all_orders = list(permutations(effects_present))

        # Limit to max_permutations
        if len(all_orders) > max_permutations:
            import random
            all_orders = random.sample(all_orders, max_permutations)

        best_order = None
        best_score = float('inf')

        with torch.no_grad():
            for order in all_orders:
                chain_spec = [
                    (fx, params_dict[fx])
                    for fx in order
                ]

                wet_recon = self.fx_chain(dry_estimate, chain_spec)

                score = F.mse_loss(wet_recon, wet_target).item()

                if score < best_score:
                    best_score = score
                    best_order = list(order)

        return best_order, best_score
