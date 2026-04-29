"""DDSP-from-latent: z → polyphonic DDSP params → audio.

Polyphonic sinusoidal synthesis supervised by BasicPitch posteriograms.

Model outputs per frame:
  - pitch_activation [264]      BCE-supervised by BasicPitch (polyphonic)
  - partial_amps [264, N_partials]  learned timbre per pitch
  - noise_filter [N_noise_taps]  time-varying FIR filter coefficients
  - transient_gate, transient_spec  (optional, v2)

Synthesis:
  audio = Σ_pitch activation × Σ_partial amp × cos(2π × freq × partial × t + phase)
        + IFFT(noise_filter × white_noise_spec)

BasicPitch freq grid: 264 pitches covering piano range at ~3 bins/semitone.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# BasicPitch standard output: 88 bins, 1 per semitone, MIDI 21 (A0) to 108 (C8)
BP_N_PITCHES = 88
BP_F0_HZ = 27.5  # A0


def bp_pitch_freqs(n_pitches=BP_N_PITCHES):
    """Compute center frequencies for all 88 BasicPitch bins (1 per semitone)."""
    # 1 bin per semitone → log2 step of 1/12
    bins = np.arange(n_pitches)
    freqs = BP_F0_HZ * 2 ** (bins / 12.0)
    return freqs  # Hz


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DDSPFromLatent(nn.Module):
    """z [B, 64, T] → DDSP params (then synthesized to audio).

    Predicts (at z's temporal rate, 25Hz):
      - pitch_logits [B, 264, T]      BCE against BasicPitch
      - partial_amps [B, 264*N_part, T]  (reshape later)
      - noise_filter [B, N_noise, T]  FIR freq-response magnitudes
    """
    def __init__(self, latent_dim=64, n_pitches=264, n_partials=10,
                 n_noise_taps=64, hidden=128):
        super().__init__()
        self.n_pitches = n_pitches
        self.n_partials = n_partials
        self.n_noise_taps = n_noise_taps

        # Shared backbone
        self.encoder = nn.Sequential(
            ConvBlock(latent_dim, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden * 2),
            ConvBlock(hidden * 2, hidden * 2),
        )

        # Heads
        self.pitch_head = nn.Conv1d(hidden * 2, n_pitches, 1)
        self.amp_head = nn.Conv1d(hidden * 2, n_pitches * n_partials, 1)
        self.noise_head = nn.Conv1d(hidden * 2, n_noise_taps, 1)

    def forward(self, z):
        """z: [B, 64, T@25Hz] → dict of params."""
        h = self.encoder(z)

        pitch_logits = self.pitch_head(h)           # [B, 264, T]
        amp_logits = self.amp_head(h)                # [B, 264*P, T]
        B, _, T = amp_logits.shape
        amp_logits = amp_logits.reshape(B, self.n_pitches, self.n_partials, T)
        noise_logits = self.noise_head(h)            # [B, noise_taps, T]

        return {
            "pitch_logits": pitch_logits,
            "pitch_activation": torch.sigmoid(pitch_logits),
            "partial_amps": F.softplus(amp_logits),   # non-negative
            "noise_filter": F.softplus(noise_logits),  # non-negative
        }


def render_harmonic(pitch_activation, partial_amps, pitch_freqs_hz,
                    n_audio_samples, sr=48000, fps=25):
    """Render sum of sinusoids.

    pitch_activation: [B, P, T]  (soft mask, 0-1)
    partial_amps:     [B, P, N_partials, T]
    pitch_freqs_hz:   [P] tensor of Hz values
    Returns: [B, n_audio_samples]
    """
    B, P, T = pitch_activation.shape
    N = pitch_activation.shape[2]  # T frames
    n_partials = partial_amps.shape[2]
    device = pitch_activation.device

    # Upsample params to audio rate
    # From T frames → n_audio_samples
    activation_hr = F.interpolate(pitch_activation, size=n_audio_samples,
                                   mode='linear', align_corners=False)  # [B, P, N_samples]
    # partial_amps [B, P, n_partials, T] → [B, P*n_partials, T]
    amps_flat = partial_amps.reshape(B, P * n_partials, T)
    amps_hr = F.interpolate(amps_flat, size=n_audio_samples,
                            mode='linear', align_corners=False)
    amps_hr = amps_hr.reshape(B, P, n_partials, n_audio_samples)

    # Time axis
    t = torch.arange(n_audio_samples, device=device).float() / sr  # [N_samples]

    # Compute phase for each pitch × partial
    # freq_kh = f0_p * h (h = 1, 2, ..., n_partials)
    partial_indices = torch.arange(1, n_partials + 1, device=device).float()  # [n_partials]
    # freq[p, h] = pitch_freqs_hz[p] * partial_indices[h]
    freqs = pitch_freqs_hz.unsqueeze(-1) * partial_indices.unsqueeze(0)  # [P, n_partials]
    # Phase: 2π × freq × t
    # Output needed: [P, n_partials, N_samples]
    # But P × n_partials × N_samples is HUGE for 264 × 10 × 2.6M
    # For training, we MUST sparsify or use smaller sizes

    # TRAINING: use top-K active pitches only (sparse rendering)
    # Pick top K pitches per batch based on activation
    K = 32  # sparse voices
    # Average activation over time
    avg_act = pitch_activation.mean(dim=-1)  # [B, P]
    topk_vals, topk_idx = avg_act.topk(K, dim=-1)  # [B, K]

    # Gather for the top-K pitches per sample
    # activation_hr[b, topk_idx[b], :], amps_hr[b, topk_idx[b], :, :]
    bi = torch.arange(B, device=device).unsqueeze(-1).expand(-1, K)  # [B, K]
    act_k = activation_hr[bi, topk_idx]  # [B, K, N_samples]
    amps_k = amps_hr[bi, topk_idx]  # [B, K, n_partials, N_samples]
    freqs_k = freqs[topk_idx]  # [B, K, n_partials]

    # Phase: [B, K, n_partials, N_samples]
    phase = 2 * math.pi * freqs_k.unsqueeze(-1) * t  # broadcast
    # clip phase freq to Nyquist
    nyquist_mask = (freqs_k < sr / 2).float().unsqueeze(-1)  # [B, K, n_partials, 1]

    # sin components (use sin for phase=0 at t=0 → no DC pop)
    sines = torch.sin(phase) * nyquist_mask  # [B, K, n_partials, N_samples]
    # Weight by amps and activation
    weighted = sines * amps_k  # [B, K, n_partials, N_samples]
    # Sum over partials
    per_pitch = weighted.sum(dim=2)  # [B, K, N_samples]
    # Weight by activation and sum over pitches
    out = (per_pitch * act_k).sum(dim=1)  # [B, N_samples]

    return out


def render_filtered_noise(noise_filter, n_audio_samples, n_fft=1024, hop=512):
    """Filtered noise synthesis via frame-wise IFFT.

    noise_filter: [B, n_taps, T]  magnitude response per frame
    Returns: [B, n_audio_samples]
    """
    B, n_taps, T = noise_filter.shape
    device = noise_filter.device
    # Number of STFT frames needed
    n_frames = n_audio_samples // hop + 1

    # Upsample filter to n_frames
    filter_hr = F.interpolate(noise_filter, size=n_frames,
                              mode='linear', align_corners=False)  # [B, n_taps, n_frames]

    # Interpolate n_taps → n_fft//2+1 bins
    target_bins = n_fft // 2 + 1
    filter_full = F.interpolate(filter_hr.unsqueeze(1),
                                size=(target_bins, n_frames),
                                mode='bilinear',
                                align_corners=False).squeeze(1)  # [B, target_bins, n_frames]

    # Random phase × filter magnitude
    random_phase = torch.rand(B, target_bins, n_frames, device=device) * 2 * math.pi
    complex_spec = filter_full * torch.exp(1j * random_phase)

    window = torch.hann_window(n_fft, device=device)
    audio = torch.istft(complex_spec, n_fft, hop, window=window,
                        length=n_audio_samples)
    return audio


class DDSPSynth(nn.Module):
    """Full DDSP-from-latent synth: z → audio."""
    def __init__(self, latent_dim=64, sr=48000, fps=25,
                 n_pitches=BP_N_PITCHES, n_partials=10,
                 n_noise_taps=64, hidden=128):
        super().__init__()
        self.net = DDSPFromLatent(latent_dim=latent_dim, n_pitches=n_pitches,
                                  n_partials=n_partials,
                                  n_noise_taps=n_noise_taps, hidden=hidden)
        self.sr = sr
        self.fps = fps
        # Fixed pitch frequencies (BasicPitch grid)
        freqs = torch.tensor(bp_pitch_freqs(n_pitches), dtype=torch.float32)
        self.register_buffer("pitch_freqs_hz", freqs)

    def forward(self, z, n_audio_samples=None):
        """z: [B, 64, T] → audio [B, N] (mono)."""
        B, _, T = z.shape
        if n_audio_samples is None:
            samples_per_frame = self.sr // self.fps
            n_audio_samples = T * samples_per_frame

        params = self.net(z)
        harmonic = render_harmonic(
            params["pitch_activation"], params["partial_amps"],
            self.pitch_freqs_hz, n_audio_samples, sr=self.sr, fps=self.fps)
        noise = render_filtered_noise(
            params["noise_filter"], n_audio_samples)
        audio = harmonic + noise
        return audio, params


if __name__ == "__main__":
    m = DDSPSynth().cuda()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"DDSPSynth: {n:.2f}M params")

    z = torch.randn(2, 64, 50).cuda()  # 2 sec at 25Hz
    audio, params = m(z)
    print(f"z {tuple(z.shape)} → audio {tuple(audio.shape)}")
    for k, v in params.items():
        print(f"  {k}: {tuple(v.shape)}")
