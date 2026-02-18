"""
Iterative Chain Estimator Module.

Estimates effect chain by iteratively identifying and removing the last effect.
Based on DAFx 2024 approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .effect_encoder import EffectEncoder, EffectEncoderConfig
from .inverters import (
    EQInverter,
    CompressorInverter,
    ReverbInverter,
    DistortionInverter,
    ChorusInverter,
    DelayInverter,
)


@dataclass
class ChainEstimatorConfig:
    """Configuration for IterativeChainEstimator."""
    encoder_config: EffectEncoderConfig = None
    max_iterations: int = 6
    hidden_dim: int = 256
    effect_types: List[str] = None
    sample_rate: int = 44100

    def __post_init__(self):
        if self.encoder_config is None:
            self.encoder_config = EffectEncoderConfig()
        if self.effect_types is None:
            self.effect_types = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']


class ParameterEstimator(nn.Module):
    """
    Estimates effect parameters from embedding.
    """

    def __init__(
        self,
        effect_type: str,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.effect_type = effect_type

        # Number of parameters per effect type
        param_counts = {
            'eq': 15,
            'compressor': 6,
            'reverb': 4,
            'distortion': 4,
            'chorus': 4,
            'delay': 3,
        }

        self.num_params = param_counts.get(effect_type, 10)

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_params),
            nn.Sigmoid(),  # Output in [0, 1]
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Estimate parameters.

        Args:
            embedding: Effect embedding [B, embedding_dim]

        Returns:
            Estimated parameters [B, num_params]
        """
        return self.network(embedding)


class IterativeChainEstimator(nn.Module):
    """
    Estimates effect chain by iteratively identifying and removing last effect.

    Process:
    1. Encode wet audio
    2. Predict last effect type
    3. Estimate parameters for that effect
    4. Invert the effect to get intermediate signal
    5. Repeat until "no more effects" is predicted
    """

    def __init__(self, config: Optional[ChainEstimatorConfig] = None):
        super().__init__()
        self.config = config or ChainEstimatorConfig()

        # Effect encoder
        self.encoder = EffectEncoder(self.config.encoder_config)

        # Last effect predictor
        # +1 for "no more effects" token
        num_classes = len(self.config.effect_types) + 1
        self.last_effect_predictor = nn.Sequential(
            nn.Linear(self.config.encoder_config.embedding_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_dim, num_classes),
        )

        # Confidence estimator (for uncertainty quantification)
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.config.encoder_config.embedding_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Per-effect parameter estimators
        self.param_estimators = nn.ModuleDict({
            fx: ParameterEstimator(
                fx,
                self.config.encoder_config.embedding_dim,
                self.config.hidden_dim,
            )
            for fx in self.config.effect_types
        })

        # Per-effect inverters
        self.inverters = nn.ModuleDict({
            'eq': EQInverter(self.config.sample_rate),
            'compressor': CompressorInverter(self.config.sample_rate),
            'reverb': ReverbInverter(self.config.sample_rate),
            'distortion': DistortionInverter(self.config.sample_rate),
            'chorus': ChorusInverter(self.config.sample_rate),
            'delay': DelayInverter(self.config.sample_rate),
        })

    def forward(
        self,
        wet_audio: torch.Tensor,
        max_iterations: Optional[int] = None,
        return_intermediates: bool = False,
        force_iterations: Optional[int] = None,  # Force N iterations during training
        gt_effect_types: Optional[List[int]] = None,  # GT effect type indices (reversed order) for training
        gt_params: Optional[List[torch.Tensor]] = None,  # GT params (reversed order) for training
    ) -> Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]]]:
        """
        Estimate effect chain and recover dry audio.

        Args:
            wet_audio: Wet audio [B, 1, T] or [B, T]
            max_iterations: Override max iterations
            return_intermediates: Return intermediate signals

        Returns:
            dry_estimate: Recovered dry signal [B, 1, T]
            chain: List of (effect_type, params) in application order
        """
        if wet_audio.dim() == 2:
            wet_audio = wet_audio.unsqueeze(1)

        # Use force_iterations if provided (training), otherwise use max_iterations
        if force_iterations is not None:
            max_iters = force_iterations
        else:
            max_iters = max_iterations or self.config.max_iterations
        current_signal = wet_audio
        chain = []
        intermediates = []
        confidences = []

        for iteration in range(max_iters):
            # Encode current signal
            embedding = self.encoder(current_signal)

            # Predict last effect
            effect_logits = self.last_effect_predictor(embedding)
            effect_probs = F.softmax(effect_logits, dim=-1)
            effect_idx = effect_logits.argmax(dim=-1)

            # Estimate confidence
            confidence = self.confidence_estimator(embedding)
            confidences.append(confidence)

            # Check if chain complete (last class = "no more effects")
            # Skip this check if force_iterations is set (training mode)
            if force_iterations is None:
                if (effect_idx == len(self.config.effect_types)).all():
                    break

            # Get effect type for each sample in batch
            # For simplicity, use the most common prediction in batch
            effect_idx_mode = effect_idx.mode().values.item()

            # During training with force_iterations, don't allow "no effect" prediction
            if force_iterations is None and effect_idx_mode == len(self.config.effect_types):
                break
            elif effect_idx_mode == len(self.config.effect_types):
                # Forced training: pick the second most likely effect instead
                effect_logits_masked = effect_logits.clone()
                effect_logits_masked[:, len(self.config.effect_types)] = float('-inf')
                effect_idx_mode = effect_logits_masked.argmax(dim=-1).mode().values.item()

            # Use GT effect type if provided (teacher forcing for inverters)
            if gt_effect_types is not None and iteration < len(gt_effect_types):
                gt_idx = gt_effect_types[iteration]
                if gt_idx < len(self.config.effect_types):
                    effect_type = self.config.effect_types[gt_idx]
                else:
                    effect_type = self.config.effect_types[effect_idx_mode]
            else:
                effect_type = self.config.effect_types[effect_idx_mode]

            # Estimate parameters (or use GT if provided)
            pred_params = self.param_estimators[effect_type](embedding)

            if gt_params is not None and iteration < len(gt_params):
                # Use GT params for inverter (teacher forcing)
                params = gt_params[iteration]
            else:
                params = pred_params

            # Invert effect
            current_signal = self.inverters[effect_type](current_signal, params)

            # Store in chain (in reverse order) - include logits for supervision
            chain.append((effect_type, params, effect_probs[:, effect_idx_mode], effect_logits))

            if return_intermediates:
                intermediates.append(current_signal.clone())

        # Reverse chain to get application order
        chain = chain[::-1]

        if return_intermediates:
            return current_signal, chain, intermediates, confidences

        return current_signal, chain

    def predict_last_effect(
        self,
        audio: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict the last effect applied to the audio.

        Args:
            audio: Audio signal [B, 1, T]

        Returns:
            effect_logits: Logits for each effect type [B, num_types + 1]
            embedding: Audio embedding [B, embedding_dim]
        """
        embedding = self.encoder(audio)
        effect_logits = self.last_effect_predictor(embedding)
        return effect_logits, embedding

    def estimate_single_effect(
        self,
        audio: torch.Tensor,
        effect_type: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate and invert a single known effect.

        Args:
            audio: Audio signal [B, 1, T]
            effect_type: Type of effect to invert

        Returns:
            inverted_audio: Audio with effect removed [B, 1, T]
            params: Estimated parameters
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)

        embedding = self.encoder(audio)
        params = self.param_estimators[effect_type](embedding)
        inverted = self.inverters[effect_type](audio, params)

        return inverted, params

    def get_chain_embedding(
        self,
        wet_audio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get a single embedding representing the entire effect chain.

        Args:
            wet_audio: Wet audio [B, 1, T]

        Returns:
            Chain embedding [B, embedding_dim]
        """
        if wet_audio.dim() == 2:
            wet_audio = wet_audio.unsqueeze(1)

        # Simply use the encoder embedding of wet audio
        # Could be enhanced with chain-aware aggregation
        return self.encoder(wet_audio)


class MultiPathChainEstimator(IterativeChainEstimator):
    """
    Chain estimator that considers multiple possible orderings.
    Returns top-k most likely chains.
    """

    def __init__(
        self,
        config: Optional[ChainEstimatorConfig] = None,
        top_k: int = 3,
    ):
        super().__init__(config)
        self.top_k = top_k

    def forward(
        self,
        wet_audio: torch.Tensor,
        max_iterations: Optional[int] = None,
    ) -> List[Tuple[torch.Tensor, List[Tuple[str, torch.Tensor]], float]]:
        """
        Estimate top-k most likely effect chains.

        Returns:
            List of (dry_estimate, chain, log_probability) tuples
        """
        if wet_audio.dim() == 2:
            wet_audio = wet_audio.unsqueeze(1)

        max_iters = max_iterations or self.config.max_iterations

        # Beam search over possible chains
        # Each beam: (current_signal, chain_so_far, log_prob)
        beams = [(wet_audio, [], 0.0)]
        completed = []

        for iteration in range(max_iters):
            new_beams = []

            for signal, chain, log_prob in beams:
                embedding = self.encoder(signal)
                effect_logits = self.last_effect_predictor(embedding)
                effect_log_probs = F.log_softmax(effect_logits, dim=-1)

                # Get top-k effects
                topk_log_probs, topk_indices = effect_log_probs.topk(
                    min(self.top_k, len(self.config.effect_types) + 1),
                    dim=-1
                )

                for k in range(topk_indices.size(-1)):
                    effect_idx = topk_indices[0, k].item()
                    new_log_prob = log_prob + topk_log_probs[0, k].item()

                    # Check if "no more effects"
                    if effect_idx == len(self.config.effect_types):
                        completed.append((signal, chain[::-1], new_log_prob))
                        continue

                    effect_type = self.config.effect_types[effect_idx]

                    # Estimate parameters and invert
                    params = self.param_estimators[effect_type](embedding)
                    new_signal = self.inverters[effect_type](signal, params)

                    new_chain = chain + [(effect_type, params)]
                    new_beams.append((new_signal, new_chain, new_log_prob))

            # Keep top beams
            new_beams.sort(key=lambda x: x[2], reverse=True)
            beams = new_beams[:self.top_k]

            if not beams:
                break

        # Add remaining beams to completed
        for signal, chain, log_prob in beams:
            completed.append((signal, chain[::-1], log_prob))

        # Sort by probability and return top-k
        completed.sort(key=lambda x: x[2], reverse=True)
        return completed[:self.top_k]
