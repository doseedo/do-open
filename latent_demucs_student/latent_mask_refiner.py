"""LatentMaskRefiner: per-stem latent + noisy STFT mask → refined STFT mask.

Replaces the expensive Oobleck decoder with a fast mask refinement step.
At inference, the refined mask is applied to the mix STFT and iSTFT'd
for high-quality per-stem audio without running a full audio decoder.

Inputs:
  latent:     [B, 64, T]         (per-stem latent from v4cond, clean)
  noisy_mask: [B, F, T_stft]     (STFT mask from v4-small, imperfect hint)

Output:
  refined_mask: [B, F, T_stft]   (closer to ground-truth htdemucs teacher mask)

Target: the actual STFT magnitude ratio mask for this stem in the mix.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, kernel // 2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class LatentMaskRefiner(nn.Module):
    """Small model: (latent, noisy_mask) → refined_mask.

    ~1.5M params. Pure 1D convs + a small 2D head for freq reshaping.
    """

    def __init__(self, latent_dim=64, n_freqs=1025, hidden=128,
                 spec_hidden=48, n_latent_layers=4):
        super().__init__()
        self.n_freqs = n_freqs
        self.hidden = hidden
        self.spec_hidden = spec_hidden

        # Latent encoder: [B, 64, T] → [B, hidden, T] (1D convs, no downsample)
        layers = [ConvBlock(latent_dim, hidden, 5)]
        for _ in range(n_latent_layers - 1):
            layers.append(ConvBlock(hidden, hidden, 5))
        self.latent_enc = nn.Sequential(*layers)

        # Mask input compressor: [B, F, T_stft] → [B, spec_hidden, T_stft]
        # Compress freq via pointwise 1x1 conv along freq dim (treat F as channel)
        self.mask_compress = nn.Sequential(
            nn.Conv1d(n_freqs, spec_hidden * 2, 1),
            nn.GroupNorm(8, spec_hidden * 2),
            nn.GELU(),
            nn.Conv1d(spec_hidden * 2, spec_hidden, 1),
            nn.GroupNorm(8, spec_hidden),
            nn.GELU(),
        )

        # Refinement: combine latent features + mask features along time
        # Latent is at 25Hz, STFT at ~94Hz (nfft=2048, hop=512, 48kHz)
        # We upsample latent to match STFT time resolution
        self.combine_refine = nn.Sequential(
            nn.Conv1d(hidden + spec_hidden, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )

        # Output: project [B, hidden, T_stft] → [B, n_freqs, T_stft]
        self.mask_out = nn.Conv1d(hidden, n_freqs, 1)

        # Residual gate: learned per-freq scalar so model can refine rather
        # than replace the input mask. Init near 1.0 so output ≈ noisy_mask
        # at start, then model learns to shift off of it.
        self.residual_gate = nn.Parameter(torch.zeros(1, n_freqs, 1))

    def forward(self, latent, noisy_mask):
        """
        latent:     [B, 64, T]
        noisy_mask: [B, F, T_stft]   (softmax mask, values in [0,1])
        Returns: refined_mask [B, F, T_stft]
        """
        B, _, T = latent.shape
        T_stft = noisy_mask.shape[-1]

        # Encode latent
        lat_h = self.latent_enc(latent)  # [B, hidden, T]

        # Upsample latent features to STFT time resolution
        lat_h = F.interpolate(lat_h, size=T_stft, mode='linear', align_corners=False)
        # [B, hidden, T_stft]

        # Compress noisy mask input
        # Apply log scaling to map [0, 1] → something better
        noisy_log = (noisy_mask + 1e-6).log()
        mask_h = self.mask_compress(noisy_log)  # [B, spec_hidden, T_stft]

        # Combine
        combined = torch.cat([lat_h, mask_h], dim=1)  # [B, hidden+spec_hidden, T_stft]
        h = self.combine_refine(combined)  # [B, hidden, T_stft]

        # Project to mask delta
        delta = self.mask_out(h)  # [B, F, T_stft]

        # Residual around noisy mask (in logit space)
        # noisy_logit ≈ log(noisy_mask / (1 - noisy_mask + eps))
        noisy_logit = (noisy_mask + 1e-6).log() - (1 - noisy_mask + 1e-6).log()
        refined_logit = noisy_logit + self.residual_gate.sigmoid() * delta

        # For per-stem masks that sum to 1 across stems, we return sigmoid
        # (will be re-normalized at inference if needed)
        return torch.sigmoid(refined_logit)


if __name__ == "__main__":
    m = LatentMaskRefiner()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    lat = torch.randn(2, 64, 100)  # 4s at 25Hz
    noisy = torch.rand(2, 1025, 376)  # STFT time
    out = m(lat, noisy)
    print(f"Params: {n:.2f}M")
    print(f"lat {tuple(lat.shape)} + noisy_mask {tuple(noisy.shape)}")
    print(f"→ refined_mask {tuple(out.shape)}")
    print(f"min={out.min():.3f} max={out.max():.3f} mean={out.mean():.3f}")
