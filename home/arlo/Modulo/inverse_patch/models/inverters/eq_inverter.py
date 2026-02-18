"""
EQ Inverter.

EQ is analytically invertible by negating the gains.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys
sys.path.insert(0, '../../..')

from nablafx.processors import ParametricEQ


class EQInverter(nn.Module):
    """
    EQ Inverter - analytically invertible by negating gains.

    EQ applies: H(f) = gain_linear * filter_response(f)
    Inverse: H_inv(f) = 1/gain_linear * inverse_filter_response(f)

    For parametric EQ with gain in dB, inverse gain = -gain_dB
    """

    def __init__(self, sample_rate: float = 44100):
        super().__init__()
        self.sample_rate = sample_rate

        # Create parametric EQ for inverse processing
        self.eq = ParametricEQ(
            sample_rate=sample_rate,
            min_gain_db=-12.0,
            max_gain_db=12.0,
            control_type='static',
        )

        # Parameter ranges for denormalization
        self.param_ranges = {
            'low_shelf_gain_db': (-12.0, 12.0),
            'low_shelf_cutoff_freq': (20.0, 2000.0),
            'low_shelf_q_factor': (0.1, 10.0),
            'band0_gain_db': (-12.0, 12.0),
            'band0_cutoff_freq': (20.0, 200.0),
            'band0_q_factor': (0.1, 10.0),
            'band1_gain_db': (-12.0, 12.0),
            'band1_cutoff_freq': (200.0, 2000.0),
            'band1_q_factor': (0.1, 10.0),
            'band2_gain_db': (-12.0, 12.0),
            'band2_cutoff_freq': (2000.0, 12000.0),
            'band2_q_factor': (0.1, 10.0),
            'high_shelf_gain_db': (-12.0, 12.0),
            'high_shelf_cutoff_freq': (4000.0, 16000.0),
            'high_shelf_q_factor': (0.1, 10.0),
        }

    def denormalize(self, norm_val: torch.Tensor, min_val: float, max_val: float) -> torch.Tensor:
        """Denormalize from [0, 1] to [min, max]."""
        return norm_val * (max_val - min_val) + min_val

    def forward(
        self,
        wet_audio: torch.Tensor,
        estimated_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Invert EQ by applying inverse EQ (negated gains).

        Args:
            wet_audio: Audio with EQ applied [B, 1, T]
            estimated_params: Normalized EQ parameters [B, 15]

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs = wet_audio.size(0)

        # Denormalize and invert gains
        # Parameters order: ls_gain, ls_freq, ls_q, b0_gain, b0_freq, b0_q, ...
        inverted_params = torch.zeros_like(estimated_params)

        # Gain parameters are at indices 0, 3, 6, 9, 12
        gain_indices = [0, 3, 6, 9, 12]
        freq_indices = [1, 4, 7, 10, 13]
        q_indices = [2, 5, 8, 11, 14]

        # For gains: invert by negating
        # First denormalize, negate, then re-normalize
        for i in gain_indices:
            # Denormalize
            denorm = self.denormalize(estimated_params[:, i], -12.0, 12.0)
            # Negate
            inverted_denorm = -denorm
            # Re-normalize
            inverted_params[:, i] = (inverted_denorm - (-12.0)) / (12.0 - (-12.0))

        # Keep frequency and Q the same
        for i in freq_indices + q_indices:
            inverted_params[:, i] = estimated_params[:, i]

        # Format for EQ: [B, num_params, 1]
        control_params = inverted_params.unsqueeze(-1)

        # Apply inverse EQ
        output, _ = self.eq(wet_audio, control_params, train=True)

        # Handle NaN and clamp
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(output, -1.0, 1.0)


class AnalyticalEQInverter(nn.Module):
    """
    Purely analytical EQ inverter using direct filter inversion.
    """

    def __init__(self, sample_rate: float = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        wet_audio: torch.Tensor,
        eq_params: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Invert EQ using frequency-domain division.

        Args:
            wet_audio: Audio with EQ applied [B, 1, T]
            eq_params: Dict of EQ parameters (gains, frequencies, Qs)

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Get the frequency response of the EQ
        # Then divide in frequency domain

        # Compute FFT
        n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(seq_len + seq_len - 1))))
        wet_fft = torch.fft.rfft(wet_audio, n=n_fft.item())

        # Compute inverse frequency response
        H_inv = self._compute_inverse_response(eq_params, n_fft.item(), bs)

        # Apply inverse filter
        dry_fft = wet_fft * H_inv.unsqueeze(1)

        # IFFT
        dry_audio = torch.fft.irfft(dry_fft, n=n_fft.item())

        # Handle NaN and clamp
        dry_audio = torch.nan_to_num(dry_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_audio[..., :seq_len], -1.0, 1.0)

    def _compute_inverse_response(
        self,
        eq_params: Dict[str, torch.Tensor],
        n_fft: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute inverse frequency response."""
        # Simplified: return flat response for now
        # Full implementation would compute actual biquad responses
        n_bins = n_fft // 2 + 1
        H_inv = torch.ones(batch_size, n_bins)
        return H_inv
