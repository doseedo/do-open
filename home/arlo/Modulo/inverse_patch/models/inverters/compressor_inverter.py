"""
Compressor Inverter.

Compressor is partially invertible using envelope-based gain estimation.
Key insight: Compressor operates on amplitude ENVELOPE, not raw signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CompressorInverter(nn.Module):
    """
    Physics-based Compressor Inverter using envelope analysis.

    Compressors apply time-varying gain based on the signal envelope.
    Key insight: Extract envelope → Predict gain curve → Apply inverse gain.

    This encodes the fundamental physics:
    - Compressor sees envelope, not raw signal
    - Gain reduction is a function of envelope level vs threshold
    - Attack/release shape the gain curve over time
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        hidden_channels: int = 32,
        envelope_window: int = 1024,
        envelope_hop: int = 256,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.envelope_window = envelope_window
        self.envelope_hop = envelope_hop

        # Envelope extractor - long window captures amplitude dynamics
        self.envelope_net = nn.Sequential(
            nn.Conv1d(1, hidden_channels, envelope_window, stride=envelope_hop, padding=envelope_window//2),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 1),
            nn.Softplus(),  # Envelope is always positive
        )

        # Parameter encoder - learns how params affect gain curve
        self.param_encoder = nn.Sequential(
            nn.Linear(6, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_channels),
            nn.Tanh(),
        )

        # Gain curve predictor - predicts gain reduction from envelope + params
        self.gain_predictor = nn.Sequential(
            nn.Conv1d(1 + hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, 1, 1),
        )

    def forward(
        self,
        wet_audio: torch.Tensor,
        estimated_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invert compression using envelope-based gain estimation.

        Args:
            wet_audio: Compressed audio [B, 1, T]
            estimated_params: Normalized compressor parameters [B, 6]
                (threshold, ratio, attack, release, knee, makeup)

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Step 1: Extract envelope (what compressor "sees")
        envelope = self.envelope_net(wet_audio)  # [B, 1, T//hop]
        envelope_len = envelope.size(-1)

        # Step 2: Analytical gain estimation from physics
        # Denormalize parameters
        threshold_db = estimated_params[:, 0] * (-60.0)  # [-60, 0] dB
        ratio = 1.0 + estimated_params[:, 1] * 19.0  # [1, 20]
        makeup_db = estimated_params[:, 5] * 24.0  # [0, 24] dB

        # Convert envelope to dB
        envelope_db = 20 * torch.log10(envelope + 1e-8)

        # Compute analytical gain reduction
        # Above threshold: gain_reduction = (level - threshold) * (1 - 1/ratio)
        threshold_exp = threshold_db.view(bs, 1, 1)
        ratio_exp = ratio.view(bs, 1, 1)

        over_threshold = F.relu(envelope_db - threshold_exp)
        analytical_gain_reduction_db = over_threshold * (1.0 - 1.0 / ratio_exp)

        # Step 3: Neural refinement (learns attack/release dynamics)
        param_features = self.param_encoder(estimated_params)  # [B, hidden]
        param_features = param_features.unsqueeze(-1).expand(-1, -1, envelope_len)

        combined = torch.cat([envelope, param_features], dim=1)
        learned_correction = self.gain_predictor(combined)

        # Combine analytical + learned
        # Clamp learned correction to prevent untrained network from exploding
        learned_correction = torch.tanh(learned_correction) * 1.0  # Bound to [-1, 1] dB
        total_gain_reduction_db = analytical_gain_reduction_db + learned_correction * 0.2

        # Step 4: Interpolate gain curve back to audio rate
        gain_reduction_db = F.interpolate(
            total_gain_reduction_db, size=seq_len, mode='linear', align_corners=False
        )

        # Step 5: Apply inverse gain (add back what was removed)
        # Also remove makeup gain
        makeup_linear = 10 ** (makeup_db.view(bs, 1, 1) / 20)
        inverse_gain_linear = 10 ** (gain_reduction_db / 20)

        dry_estimate = wet_audio * inverse_gain_linear / (makeup_linear + 1e-8)

        # Handle NaN and clamp
        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_estimate, -1.0, 1.0)


class SGICompressorInverter(nn.Module):
    """
    Sample-wise Gain Inversion for compressor.
    Uses envelope following to estimate applied gain.
    """

    def __init__(self, sample_rate: float = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        wet_audio: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invert using SGI method.

        Args:
            wet_audio: Compressed audio [B, 1, T]
            params: Compressor parameters [B, 6]

        Returns:
            Dry estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Denormalize parameters
        threshold_db = params[:, 0] * (-60.0)  # [-60, 0]
        ratio = 1.0 + params[:, 1] * 19.0  # [1, 20]
        attack_ms = 0.1 + params[:, 2] * 99.9  # [0.1, 100]
        release_ms = 10.0 + params[:, 3] * 990.0  # [10, 1000]
        knee_db = params[:, 4] * 12.0  # [0, 12]
        makeup_db = params[:, 5] * 24.0  # [0, 24]

        # Compute envelope of wet signal
        wet_abs = torch.abs(wet_audio) + 1e-8
        wet_db = 20 * torch.log10(wet_abs)

        # Estimate what the gain reduction was
        threshold = threshold_db.view(bs, 1, 1)
        r = ratio.view(bs, 1, 1)
        knee = knee_db.view(bs, 1, 1)

        # Approximate the gain that was applied
        # This is a simplified model
        over_threshold = wet_db - threshold

        # Estimate original level
        # If compressed, original was higher by (1 - 1/ratio) * over_threshold
        gain_applied_db = torch.zeros_like(wet_db)

        above_knee = over_threshold > knee / 2
        gain_applied_db = torch.where(
            above_knee,
            (1 - 1/r) * over_threshold,
            gain_applied_db
        )

        # Convert to linear and invert
        gain_applied = 10 ** (gain_applied_db / 20)
        dry_estimate = wet_audio / (gain_applied + 1e-8)

        # Remove makeup gain
        makeup_linear = 10 ** (makeup_db.view(bs, 1, 1) / 20)
        dry_estimate = dry_estimate / (makeup_linear + 1e-8)

        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_estimate, -1.0, 1.0)
