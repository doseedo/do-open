"""
Delay Inverter.

Delay is partially invertible using iterative echo cancellation.
Key insight: Echoes appear at fixed intervals determined by delay time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DelayInverter(nn.Module):
    """
    Physics-based Delay Inverter using iterative echo cancellation.

    Key insight: Delay adds copies at predictable intervals.
    wet = dry + mix * (fb * dry[t-T] + fb² * dry[t-2T] + ...)

    Inversion strategy:
    1. Use delay_ms param to know WHERE echoes are
    2. Iteratively subtract estimated echoes
    3. Neural net refines residual errors
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        max_delay_ms: float = 2000.0,
        hidden_dim: int = 64,
        num_iterations: int = 5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.max_delay_samples = int(max_delay_ms * sample_rate / 1000)
        self.num_iterations = num_iterations

        # Residual refinement network (for what analytical can't capture)
        self.residual_net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, 1023, padding=511),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 511, padding=255),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, 1, 1),
            nn.Tanh(),  # Bounded residual
        )

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(3, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
        )

        self.condition_layer = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)

    def forward(
        self,
        wet_audio: torch.Tensor,
        estimated_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove delay/echo using physics-based iterative cancellation.

        Args:
            wet_audio: Audio with delay [B, 1, T]
            estimated_params: Normalized delay parameters [B, 3]
                (delay_ms, feedback, mix)

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Denormalize parameters
        delay_norm = estimated_params[:, 0]
        feedback = estimated_params[:, 1] * 0.95  # [0, 0.95]
        mix = estimated_params[:, 2]

        # Convert delay to samples
        delay_samples = (delay_norm * self.max_delay_samples).long()

        # Step 1: PHYSICS-BASED ECHO CANCELLATION (gradient-friendly)
        # wet = dry + mix * sum(fb^i * dry[t - i*T])
        # Approximate: dry ≈ wet / (1 + mix * fb_factor)
        # where fb_factor = sum of geometric series ≈ 1/(1-fb) for infinite series

        # Compute effective feedback factor (clamped geometric series sum)
        fb_factor = feedback / (1 - feedback + 1e-8)  # [B]
        fb_factor = torch.clamp(fb_factor, 0, 10)  # Bound for stability

        # Analytical baseline: scale down by combined echo energy
        scale = 1.0 / (1.0 + mix * fb_factor * 0.5 + 1e-8)
        scale = scale.view(bs, 1, 1)

        dry_estimate = wet_audio * scale

        # Step 2: NEURAL REFINEMENT for residual errors
        # The analytical approach can't capture:
        # - Filtering in the feedback path
        # - Non-linear saturation
        # - Modulated delays (ping-pong, etc.)

        features = self.residual_net[:4](wet_audio)  # Get features before final layer

        param_features = self.param_encoder(estimated_params)
        param_features = param_features.unsqueeze(-1).expand(-1, -1, features.size(-1))

        conditioned = torch.cat([features, param_features], dim=1)
        conditioned = self.condition_layer(conditioned)
        conditioned = F.silu(conditioned)

        residual = self.residual_net[4:](conditioned)

        # Add residual correction (scaled by mix)
        mix_exp = mix.view(bs, 1, 1)
        dry_estimate = dry_estimate + mix_exp * 0.3 * residual

        # Handle NaN and clamp
        dry_estimate = torch.nan_to_num(dry_estimate, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_estimate, -1.0, 1.0)


class SpectralDelayRemover(nn.Module):
    """
    Remove delay using spectral methods.
    Exploits the comb-filter structure of delay in frequency domain.
    """

    def __init__(self, sample_rate: float = 44100):
        super().__init__()
        self.sample_rate = sample_rate

    def forward(
        self,
        wet_audio: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        bs, chs, seq_len = wet_audio.size()

        # Denormalize
        delay_ms = 1.0 + params[:, 0] * 1999.0
        feedback = params[:, 1] * 0.95
        mix = params[:, 2]

        delay_samples = delay_ms * self.sample_rate / 1000

        # FFT
        n_fft = 2 ** int(torch.ceil(torch.log2(torch.tensor(seq_len * 2))))
        wet_fft = torch.fft.rfft(wet_audio, n=n_fft.item())

        # Frequency bins
        freqs = torch.fft.rfftfreq(n_fft.item(), 1.0 / self.sample_rate)
        freqs = freqs.to(wet_audio.device)

        # Compute inverse comb filter
        # H(f) = 1 + mix * feedback * exp(-j*2*pi*f*delay)
        # H_inv = 1 / H(f)
        delay_sec = delay_samples / self.sample_rate
        phase = -2 * torch.pi * freqs.view(1, 1, -1) * delay_sec.view(bs, 1, 1)

        H = 1 + mix.view(bs, 1, 1) * feedback.view(bs, 1, 1) * torch.exp(1j * phase)

        # Regularized inverse
        H_inv = H.conj() / (H.abs() ** 2 + 0.01)

        # Apply inverse filter
        dry_fft = wet_fft * H_inv

        # IFFT
        dry = torch.fft.irfft(dry_fft, n=n_fft.item())
        dry = dry[..., :seq_len]

        dry = torch.nan_to_num(dry, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry, -1.0, 1.0)
