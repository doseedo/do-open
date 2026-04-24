"""DDSP-from-latent v2: bigger, more partials, with transient events.

Improvements over v1:
  - hidden 128 → 256 (4x params)
  - n_partials 10 → 20 (richer timbre)
  - Added transient detection + spectrum heads (drum/percussion support)
  - Optional HPSS-based supervision (harmonic vs percussive target separation)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


BP_N_PITCHES = 88
BP_F0_HZ = 27.5  # A0


def bp_pitch_freqs(n_pitches=BP_N_PITCHES):
    bins = np.arange(n_pitches)
    return BP_F0_HZ * 2 ** (bins / 12.0)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class DDSPFromLatentV2(nn.Module):
    """Bigger DDSP encoder with transient heads."""
    def __init__(self, latent_dim=64, n_pitches=88, n_partials=20,
                 n_noise_taps=64, n_transient_dims=32, hidden=256):
        super().__init__()
        self.n_pitches = n_pitches
        self.n_partials = n_partials
        self.n_noise_taps = n_noise_taps
        self.n_transient_dims = n_transient_dims

        # Deeper backbone
        self.encoder = nn.Sequential(
            ConvBlock(latent_dim, hidden),
            ConvBlock(hidden, hidden),
            ConvBlock(hidden, hidden * 2),
            ConvBlock(hidden * 2, hidden * 2),
            ConvBlock(hidden * 2, hidden * 2),
        )

        out_ch = hidden * 2

        # Standard heads
        self.pitch_head = nn.Conv1d(out_ch, n_pitches, 1)
        self.amp_head = nn.Conv1d(out_ch, n_pitches * n_partials, 1)
        self.noise_head = nn.Conv1d(out_ch, n_noise_taps, 1)

        # NEW: transient detection + spectrum
        self.transient_gate_head = nn.Conv1d(out_ch, 1, 1)
        self.transient_spec_head = nn.Conv1d(out_ch, n_transient_dims, 1)

    def forward(self, z):
        h = self.encoder(z)
        B, _, T = h.shape

        pitch_logits = self.pitch_head(h)            # [B, 88, T]
        amp_logits = self.amp_head(h)                 # [B, 88*P, T]
        amp_logits = amp_logits.reshape(B, self.n_pitches, self.n_partials, T)
        noise_logits = self.noise_head(h)             # [B, noise_taps, T]
        transient_gate_logits = self.transient_gate_head(h).squeeze(1)  # [B, T]
        transient_spec_logits = self.transient_spec_head(h)             # [B, td, T]

        # Strictly bounded outputs to prevent divergence
        # Per-partial amp max 0.05 (so summed over 20 partials × 48 active pitches max ~48)
        # Per-noise-tap and per-transient-bin max 0.5
        partial_amps = torch.sigmoid(amp_logits) * 0.05
        noise_filter = torch.sigmoid(noise_logits) * 0.5
        transient_spec = torch.sigmoid(transient_spec_logits) * 0.5

        return {
            "pitch_logits": pitch_logits,
            "pitch_activation": torch.sigmoid(pitch_logits),
            "partial_amps": partial_amps,
            "noise_filter": noise_filter,
            "transient_gate": torch.sigmoid(transient_gate_logits),
            "transient_gate_logits": transient_gate_logits,
            "transient_spec": transient_spec,
        }


def render_harmonic_v2(pitch_activation, partial_amps, pitch_freqs_hz,
                       n_audio_samples, sr=48000, top_k=48):
    """Sparse harmonic synthesis with top-K active pitches (more than v1)."""
    B, P, T = pitch_activation.shape
    n_partials = partial_amps.shape[2]
    device = pitch_activation.device

    activation_hr = F.interpolate(pitch_activation, size=n_audio_samples,
                                   mode='linear', align_corners=False)
    amps_flat = partial_amps.reshape(B, P * n_partials, T)
    amps_hr = F.interpolate(amps_flat, size=n_audio_samples,
                            mode='linear', align_corners=False)
    amps_hr = amps_hr.reshape(B, P, n_partials, n_audio_samples)

    t = torch.arange(n_audio_samples, device=device).float() / sr

    partial_indices = torch.arange(1, n_partials + 1, device=device).float()
    freqs = pitch_freqs_hz.unsqueeze(-1) * partial_indices.unsqueeze(0)  # [P, n_partials]

    K = min(top_k, P)
    avg_act = pitch_activation.mean(dim=-1)  # [B, P]
    topk_vals, topk_idx = avg_act.topk(K, dim=-1)

    bi = torch.arange(B, device=device).unsqueeze(-1).expand(-1, K)
    act_k = activation_hr[bi, topk_idx]      # [B, K, N_samples]
    amps_k = amps_hr[bi, topk_idx]            # [B, K, n_partials, N_samples]
    freqs_k = freqs[topk_idx]                 # [B, K, n_partials]

    phase = 2 * math.pi * freqs_k.unsqueeze(-1) * t
    nyquist_mask = (freqs_k < sr / 2).float().unsqueeze(-1)

    sines = torch.sin(phase) * nyquist_mask
    weighted = sines * amps_k
    per_pitch = weighted.sum(dim=2)
    out = (per_pitch * act_k).sum(dim=1)
    return out


def render_filtered_noise(noise_filter, n_audio_samples, n_fft=1024, hop=512):
    B, n_taps, T = noise_filter.shape
    device = noise_filter.device
    n_frames = n_audio_samples // hop + 1

    filter_hr = F.interpolate(noise_filter, size=n_frames,
                              mode='linear', align_corners=False)
    target_bins = n_fft // 2 + 1
    filter_full = F.interpolate(filter_hr.unsqueeze(1),
                                size=(target_bins, n_frames),
                                mode='bilinear',
                                align_corners=False).squeeze(1)

    random_phase = torch.rand(B, target_bins, n_frames, device=device) * 2 * math.pi
    complex_spec = filter_full * torch.exp(1j * random_phase)

    window = torch.hann_window(n_fft, device=device)
    audio = torch.istft(complex_spec, n_fft, hop, window=window,
                        length=n_audio_samples)
    return audio


def render_transients(gate, spec, n_audio_samples, sr=48000,
                       fps=25, n_fft=1024, hop=512, gate_thresh=0.3):
    """Render transient events as short noise bursts shaped by spec.

    gate: [B, T_z]  probability of transient at each z-frame
    spec: [B, td=32, T_z]  spectrum template per frame
    Returns: [B, n_audio_samples]
    """
    B, T_z = gate.shape
    device = gate.device
    samples_per_frame = n_audio_samples // T_z if T_z > 0 else 1

    # For each frame, if gate > thresh, render a short impulse with spec at that time
    # Use soft gating (multiply gate prob into the impulse amplitude)
    # Render: for each frame, place a 256-sample exponentially-decaying noise burst
    burst_len = 1024  # ~21ms at 48kHz
    # Generate one per-frame impulse weighted by gate × spec_amplitude
    # Simpler: convolve a template with gate signal
    #
    # Fast version: upsample gate to audio rate, multiply by white_noise * env
    gate_hr = F.interpolate(gate.unsqueeze(1), size=n_audio_samples,
                            mode='linear', align_corners=False).squeeze(1)
    # Apply threshold-like weighting — keep above-thresh contributions
    gate_hr = torch.relu(gate_hr - gate_thresh) / (1 - gate_thresh + 1e-6)

    # Spec → audio-rate via IFFT (similar to filtered noise)
    n_frames = n_audio_samples // hop + 1
    spec_hr = F.interpolate(spec, size=n_frames, mode='linear', align_corners=False)
    target_bins = n_fft // 2 + 1
    spec_full = F.interpolate(spec_hr.unsqueeze(1),
                              size=(target_bins, n_frames),
                              mode='bilinear', align_corners=False).squeeze(1)
    random_phase = torch.rand(B, target_bins, n_frames, device=device) * 2 * math.pi
    complex_spec = spec_full * torch.exp(1j * random_phase)
    window = torch.hann_window(n_fft, device=device)
    transient_audio = torch.istft(complex_spec, n_fft, hop, window=window,
                                  length=n_audio_samples)

    # Multiply by gate envelope (only loud where transients are)
    return transient_audio * gate_hr


class DDSPSynthV2(nn.Module):
    def __init__(self, latent_dim=64, sr=48000, fps=25,
                 n_pitches=BP_N_PITCHES, n_partials=20,
                 n_noise_taps=64, hidden=256):
        super().__init__()
        self.net = DDSPFromLatentV2(
            latent_dim=latent_dim, n_pitches=n_pitches,
            n_partials=n_partials, n_noise_taps=n_noise_taps, hidden=hidden)
        self.sr = sr
        self.fps = fps
        freqs = torch.tensor(bp_pitch_freqs(n_pitches), dtype=torch.float32)
        self.register_buffer("pitch_freqs_hz", freqs)

    def forward(self, z, n_audio_samples=None):
        B, _, T = z.shape
        if n_audio_samples is None:
            n_audio_samples = T * (self.sr // self.fps)

        params = self.net(z)
        harmonic = render_harmonic_v2(
            params["pitch_activation"], params["partial_amps"],
            self.pitch_freqs_hz, n_audio_samples, sr=self.sr, top_k=48)
        noise = render_filtered_noise(params["noise_filter"], n_audio_samples)
        transient = render_transients(params["transient_gate"], params["transient_spec"],
                                      n_audio_samples, sr=self.sr, fps=self.fps)
        audio = harmonic + noise + transient
        # Backstop: clamp to a reasonable audio range
        audio = torch.clamp(audio, -10.0, 10.0)
        return audio, params


def detect_transient_targets(audio, sr=48000, fps=25):
    """Compute target transient gate from audio via spectral flux.

    Returns: [B, T_z]  values in [0, 1]
    """
    B, N = audio.shape
    # STFT
    n_fft, hop = 1024, 512
    window = torch.hann_window(n_fft, device=audio.device)
    spec = torch.stft(audio, n_fft, hop, window=window, return_complex=True).abs()  # [B, F, T]
    # Spectral flux: positive change in magnitude
    diff = (spec[:, :, 1:] - spec[:, :, :-1]).clamp(min=0).sum(dim=1)  # [B, T-1]
    flux = F.pad(diff, (1, 0))  # [B, T]
    # Normalize per-clip
    flux_max = flux.amax(dim=-1, keepdim=True).clamp(min=1e-6)
    flux = flux / flux_max  # [0, 1]
    # Downsample to z's fps
    T_z = N * fps // sr
    flux = F.interpolate(flux.unsqueeze(1), size=T_z,
                          mode='linear', align_corners=False).squeeze(1)
    return flux  # [B, T_z]


if __name__ == "__main__":
    m = DDSPSynthV2().cuda()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    print(f"DDSPSynthV2: {n:.2f}M params")

    z = torch.randn(2, 64, 50).cuda()
    audio, params = m(z)
    print(f"z {tuple(z.shape)} → audio {tuple(audio.shape)}")
    for k, v in params.items():
        print(f"  {k}: {tuple(v.shape)}")

    # Test transient detection
    fake_audio = torch.randn(2, 96000).cuda()
    targets = detect_transient_targets(fake_audio)
    print(f"transient targets: {tuple(targets.shape)}")
