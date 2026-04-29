"""Additive-conditioned waveform → stem latents model.

Same Oobleck encoder backbone as FiLM version, but conditioning is
additive instead of multiplicative:
  h_stem = h + bias(conditioning)

instead of FiLM's:
  h_stem = gamma(conditioning) * h + beta(conditioning)

Half the conditioning params, faster forward, and better inductive bias
for frame-level conditioning from SemDemucs v2 (per-frame pitch + RMS
already carry the routing signal — no need for learned gating).

Supports two conditioning modes:
  - global: 128-dim embedding per stem → broadcast over time
  - framelevel: [T', C_cond] per stem → interpolated to match backbone T
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class WaveformToStemLatentsAdditive(nn.Module):
    def __init__(self, vae_path="/scratch/ACE-Step-1.5/checkpoints/vae",
                 n_stems=4, cond_dim=128, freeze_backbone=False):
        super().__init__()
        from diffusers.models import AutoencoderOobleck
        vae = AutoencoderOobleck.from_pretrained(vae_path)
        self.encoder = vae.encoder
        self.n_stems = n_stems
        self.latent_dim = 64
        self.cond_dim = cond_dim

        # Materialize weight_norm on old conv2
        old_conv2 = self.encoder.conv2
        with torch.no_grad():
            dummy = torch.zeros(1, old_conv2.in_channels, 4)
            _ = old_conv2(dummy)
        in_ch = old_conv2.in_channels

        # Per-stem additive bias: cond → bias vector added to backbone features
        self.stem_bias = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cond_dim, in_ch),
                nn.GELU(),
                nn.Linear(in_ch, in_ch),
            ) for _ in range(n_stems)
        ])
        # Init to zero so step-0 is identity (all stems = backbone output)
        for bias in self.stem_bias:
            nn.init.zeros_(bias[-1].weight)
            nn.init.zeros_(bias[-1].bias)

        # Per-stem projection: backbone_channels → 64 latent dims
        k = old_conv2.kernel_size[0] if hasattr(old_conv2.kernel_size, '__len__') else old_conv2.kernel_size
        pad = old_conv2.padding[0] if hasattr(old_conv2.padding, '__len__') else old_conv2.padding
        self.stem_projs = nn.ModuleList([
            nn.Conv1d(in_ch, self.latent_dim, kernel_size=k,
                      stride=old_conv2.stride, padding=pad)
            for _ in range(n_stems)
        ])
        # Init from original conv2 mean weights
        with torch.no_grad():
            w_full = old_conv2.weight.detach()
            b_full = old_conv2.bias.detach()
            w_mean = w_full[:self.latent_dim]
            b_mean = b_full[:self.latent_dim]
            for proj in self.stem_projs:
                proj.weight.copy_(w_mean)
                proj.bias.copy_(b_mean)
                proj.weight.add_(torch.randn_like(proj.weight) * 1e-4)

        # Replace conv2 with identity
        self.encoder.conv2 = nn.Identity()

        if freeze_backbone:
            for n, p in self.encoder.named_parameters():
                p.requires_grad = False

    def forward(self, waveform, cond):
        """
        waveform: [B, 2, N] @ 48kHz
        cond: [B, n_stems, cond_dim] global embeddings
              OR [B, n_stems, T_cond, cond_dim] frame-level conditioning
        returns: [B, n_stems, 64, T]
        """
        h = self.encoder(waveform)  # [B, in_ch, T]
        B, C, T = h.shape

        stems = []
        for i in range(self.n_stems):
            if cond.dim() == 3:
                # Global: [B, n_stems, cond_dim] → [B, cond_dim]
                c = cond[:, i]  # [B, cond_dim]
                bias = self.stem_bias[i](c)  # [B, in_ch]
                h_mod = h + bias.unsqueeze(-1)  # [B, C, T] + [B, C, 1]
            elif cond.dim() == 4:
                # Frame-level: [B, n_stems, T_cond, cond_dim]
                c = cond[:, i]  # [B, T_cond, cond_dim]
                # Interpolate to match backbone T
                # c is [B, T_cond, cond_dim] → need [B, cond_dim, T_cond] for interp
                c_t = c.transpose(1, 2)  # [B, cond_dim, T_cond]
                if c_t.shape[-1] != T:
                    c_t = F.interpolate(c_t, size=T, mode='linear', align_corners=False)
                # Project to backbone channels
                # c_t: [B, cond_dim, T] → bias: [B, in_ch, T]
                # Use the bias MLP on each frame
                c_frames = c_t.transpose(1, 2)  # [B, T, cond_dim]
                bias = self.stem_bias[i](c_frames)  # [B, T, in_ch]
                h_mod = h + bias.transpose(1, 2)  # [B, C, T]
            else:
                h_mod = h

            stem_lat = self.stem_projs[i](h_mod)  # [B, 64, T]
            stems.append(stem_lat)

        return torch.stack(stems, dim=1)  # [B, n_stems, 64, T]


if __name__ == "__main__":
    m = WaveformToStemLatentsAdditive(n_stems=4)
    n = sum(p.numel() for p in m.parameters()) / 1e6
    nt = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"additive 4-stem: {n:.1f}M total, {nt:.1f}M trainable")

    x = torch.randn(1, 2, 48000 * 4)
    # Global conditioning
    c_global = torch.randn(1, 4, 128)
    y = m(x, c_global)
    print(f"global:     in {tuple(x.shape)} + cond {tuple(c_global.shape)} → {tuple(y.shape)}")

    # Frame-level conditioning
    c_frame = torch.randn(1, 4, 100, 128)  # 100 frames of 128-dim per stem
    y2 = m(x, c_frame)
    print(f"frame-level: in {tuple(x.shape)} + cond {tuple(c_frame.shape)} → {tuple(y2.shape)}")

    # Compare param count vs FiLM
    from distill_model import WaveformToStemLatentsSemCond
    m_film = WaveformToStemLatentsSemCond(n_stems=4)
    nf = sum(p.numel() for p in m_film.parameters()) / 1e6
    print(f"\nFiLM:     {nf:.1f}M")
    print(f"Additive: {n:.1f}M  ({(1-n/nf)*100:.0f}% fewer params)")
