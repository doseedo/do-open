"""
Reverb Inverter (Dereverberation).

Reverb is challenging to invert as it's an additive effect.
Key insight: Reverb energy decays exponentially in time-frequency domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConvBlock(nn.Module):
    """Convolutional block for UNet."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        return x


class ReverbInverter(nn.Module):
    """
    Physics-based Reverb Inverter using spectral decay estimation.

    Key insight: Reverb adds exponentially decaying energy.
    - Energy decays as e^(-t/decay_time)
    - Higher frequencies decay faster (damping)
    - Pre-delay shifts the onset

    Strategy: Spectral subtraction with physics-informed decay model.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_bins = n_fft // 2 + 1

        # Spectral mask predictor - learns residual from analytical decay
        self.mask_net = nn.Sequential(
            nn.Conv2d(1, hidden_dim // 2, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, 1, (1, 1)),
            nn.Sigmoid(),
        )

        # Parameter encoder
        self.param_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_dim),
            nn.Tanh(),
        )

    def forward(
        self,
        wet_audio: torch.Tensor,
        estimated_params: torch.Tensor,
    ) -> torch.Tensor:
        """
        Remove reverb using spectral decay estimation.

        Args:
            wet_audio: Reverberant audio [B, 1, T]
            estimated_params: Normalized reverb parameters [B, 4]
                (decay_time, pre_delay, wet_mix, damping)

        Returns:
            Dry audio estimate [B, 1, T]
        """
        bs, chs, seq_len = wet_audio.size()

        # Denormalize parameters
        decay_time = 0.1 + estimated_params[:, 0] * 9.9  # [0.1, 10] seconds
        pre_delay_ms = estimated_params[:, 1] * 100.0  # [0, 100] ms
        wet_mix = estimated_params[:, 2]  # [0, 1]
        damping = estimated_params[:, 3]  # [0, 1]

        # Step 1: STFT
        window = torch.hann_window(self.n_fft, device=wet_audio.device)
        stft = torch.stft(
            wet_audio.squeeze(1),
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        )  # [B, F, T]

        magnitude = stft.abs()
        phase = stft.angle()
        n_frames = magnitude.size(-1)

        # Step 2: PHYSICS-BASED DECAY ESTIMATION
        # Reverb energy decays as: E(t) = E0 * exp(-t / decay_time)
        # Higher frequencies decay faster due to damping

        # Time axis (frames to seconds)
        t = torch.arange(n_frames, device=wet_audio.device, dtype=wet_audio.dtype)
        t = t * self.hop_length / self.sample_rate  # Convert to seconds

        # Frequency axis (for damping)
        freqs = torch.fft.rfftfreq(self.n_fft, 1.0 / self.sample_rate)
        freqs = freqs.to(wet_audio.device)  # [F]

        # Compute decay envelope for each batch
        decay_envelope = torch.zeros(bs, self.n_bins, n_frames, device=wet_audio.device)

        for b in range(bs):
            # Time decay: exp(-t / decay_time)
            dt = decay_time[b].item()
            time_decay = torch.exp(-t / (dt + 0.01))  # [T]

            # Frequency-dependent damping: high freqs decay faster
            # damping parameter controls how much faster
            damp = damping[b].item()
            freq_factor = 1.0 + damp * (freqs / (self.sample_rate / 2))  # [F]

            # Combined decay: outer product
            decay_envelope[b] = torch.outer(freq_factor, time_decay)  # [F, T]

        # Normalize decay envelope to [0, 1]
        decay_envelope = decay_envelope / (decay_envelope.max(dim=-1, keepdim=True)[0] + 1e-8)

        # Reverb mask = wet_mix * decay_envelope
        # This estimates what fraction of energy at each TF bin is reverb
        reverb_mask = wet_mix.view(bs, 1, 1) * decay_envelope * 0.5  # Scale down

        # Step 3: NEURAL REFINEMENT
        # The analytical decay can't capture:
        # - Room modes
        # - Early reflections
        # - Complex impulse responses

        mag_input = magnitude.unsqueeze(1)  # [B, 1, F, T]
        learned_mask = self.mask_net(mag_input).squeeze(1)  # [B, F, T]

        # Combine analytical + learned
        # Analytical provides structure, neural learns details
        # Scale learned mask so network can meaningfully contribute
        final_mask = 0.1 * reverb_mask + 0.2 * learned_mask * wet_mix.view(bs, 1, 1)

        # Step 4: SPECTRAL SUBTRACTION
        # Wiener-like filtering: dry_mag = wet_mag * (1 - mask)
        # Don't subtract too much to avoid artifacts
        dry_magnitude = magnitude * (1 - final_mask.clamp(0, 0.9))

        # Step 5: RECONSTRUCT
        dry_stft = torch.polar(dry_magnitude, phase)
        dry_audio = torch.istft(
            dry_stft,
            self.n_fft,
            self.hop_length,
            window=window,
            length=seq_len,
        )

        dry_audio = torch.nan_to_num(dry_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_audio.unsqueeze(1), -1.0, 1.0)


class SpectralDereverb(nn.Module):
    """
    Spectral-domain dereverberation.
    Operates on STFT representation.
    """

    def __init__(
        self,
        sample_rate: float = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

        n_bins = n_fft // 2 + 1

        # Magnitude mask predictor
        self.mask_net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 1, (1, 1)),
            nn.Sigmoid(),
        )

    def forward(
        self,
        wet_audio: torch.Tensor,
        params: torch.Tensor,
    ) -> torch.Tensor:
        bs, chs, seq_len = wet_audio.size()

        # STFT
        window = torch.hann_window(self.n_fft, device=wet_audio.device)
        stft = torch.stft(
            wet_audio.squeeze(1),
            self.n_fft,
            self.hop_length,
            window=window,
            return_complex=True,
        )

        magnitude = stft.abs()
        phase = stft.angle()

        # Predict mask
        mag_input = magnitude.unsqueeze(1)  # [B, 1, F, T]
        mask = self.mask_net(mag_input).squeeze(1)  # [B, F, T]

        # Apply mask
        clean_magnitude = magnitude * mask

        # Reconstruct
        clean_stft = clean_magnitude * torch.exp(1j * phase)
        dry_audio = torch.istft(
            clean_stft,
            self.n_fft,
            self.hop_length,
            window=window,
            length=seq_len,
        )

        dry_audio = torch.nan_to_num(dry_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        return torch.clamp(dry_audio.unsqueeze(1), -1.0, 1.0)
