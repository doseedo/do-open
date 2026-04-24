"""Distilled `waveform → 4 stem latents` model.

Architecture:
  - Backbone: Oobleck VAE encoder (84M params), pretrained.
  - Head: replaces encoder.conv2 (2048→128) with Conv1d(2048, 4*64, 1)
    that outputs 4 stem latent means at 25 Hz.
  - Forward: stereo waveform [B, 2, N_samples] @ 48 kHz
            → [B, 4, 64, T] where T = N_samples / 1920
  - Init: encoder backbone copies Oobleck weights; new head is initialized
    from the original conv2 mean weights tiled 4× so step-0 output = 4
    copies of the unmixed VAE latent (a sane starting point).
"""
import torch
import torch.nn as nn


class WaveformToFourStemLatents(nn.Module):
    def __init__(self, vae_path="/scratch/ACE-Step-1.5/checkpoints/vae",
                 n_stems=4, freeze_backbone=False):
        super().__init__()
        from diffusers.models import AutoencoderOobleck
        vae = AutoencoderOobleck.from_pretrained(vae_path)
        self.encoder = vae.encoder
        self.n_stems = n_stems
        self.latent_dim = 64

        # The original conv2 outputs 128 channels (mean[:64] + std[64:]).
        # weight_norm makes .weight lazy — run a dummy forward to materialize.
        old_conv2 = self.encoder.conv2
        with torch.no_grad():
            dummy = torch.zeros(1, old_conv2.in_channels, 4)
            _ = old_conv2(dummy)
        in_ch = old_conv2.in_channels
        out_ch_per_stem = self.latent_dim
        new_conv2 = nn.Conv1d(in_ch, n_stems * out_ch_per_stem,
                              kernel_size=old_conv2.kernel_size,
                              stride=old_conv2.stride,
                              padding=old_conv2.padding)
        # Init from original mean weights (the first 64 output channels of
        # the original 128-channel conv2). Tile them n_stems times so each
        # stem head starts as the unmixed VAE encoder output.
        with torch.no_grad():
            # weight may be a parametrized weight_norm output — materialize it
            w_full = old_conv2.weight.detach().to(new_conv2.weight.device,
                                                  dtype=new_conv2.weight.dtype)
            b_full = old_conv2.bias.detach().to(new_conv2.bias.device,
                                                dtype=new_conv2.bias.dtype)
            w_mean = w_full[:out_ch_per_stem]                       # [64, in, k]
            b_mean = b_full[:out_ch_per_stem]                       # [64]
            new_conv2.weight.copy_(w_mean.repeat(n_stems, 1, 1))
            new_conv2.bias.copy_(b_mean.repeat(n_stems))
        # Add tiny noise to break symmetry between stem heads
        with torch.no_grad():
            new_conv2.weight.add_(torch.randn_like(new_conv2.weight) * 1e-4)

        # Wire it back into encoder so we can keep using encoder.forward()
        self.encoder.conv2 = new_conv2

        if freeze_backbone:
            for n, p in self.encoder.named_parameters():
                if "conv2" not in n:
                    p.requires_grad = False

    def forward(self, waveform):
        """waveform: [B, 2, N_samples] @ 48 kHz
        returns:    [B, n_stems, 64, T]
        """
        h = self.encoder(waveform)        # [B, n_stems*64, T]
        B, C, T = h.shape
        return h.view(B, self.n_stems, self.latent_dim, T)


class WaveformToStemLatentsSemCond(nn.Module):
    """Semantic-conditioned variant: waveform + sem_emb → stem latents.

    Same Oobleck encoder backbone, but the stem head is FiLM-conditioned
    on a 128-dim semantic embedding (from frozen v1 or v2g encoder).

    The embedding is computed externally from the mix latent:
      waveform → VAE encoder → mix latent → sem encoder → 128-dim emb

    FiLM is applied per-stem on the backbone output before the final
    projection, so each stem gets independently modulated.
    """
    def __init__(self, vae_path="/scratch/ACE-Step-1.5/checkpoints/vae",
                 n_stems=4, sem_dim=128, freeze_backbone=False):
        super().__init__()
        from diffusers.models import AutoencoderOobleck
        vae = AutoencoderOobleck.from_pretrained(vae_path)
        self.encoder = vae.encoder
        self.n_stems = n_stems
        self.latent_dim = 64
        self.sem_dim = sem_dim

        # Materialize weight_norm on old conv2
        old_conv2 = self.encoder.conv2
        with torch.no_grad():
            dummy = torch.zeros(1, old_conv2.in_channels, 4)
            _ = old_conv2(dummy)
        in_ch = old_conv2.in_channels

        # Per-stem FiLM: sem_emb → (gamma, beta) for each stem's channels
        self.stem_film = nn.ModuleList([
            nn.Sequential(
                nn.Linear(sem_dim, in_ch),
                nn.GELU(),
                nn.Linear(in_ch, in_ch * 2),  # gamma + beta
            ) for _ in range(n_stems)
        ])
        # Init FiLM to identity (gamma=1, beta=0)
        for film in self.stem_film:
            nn.init.zeros_(film[-1].weight)
            nn.init.zeros_(film[-1].bias)
            film[-1].bias.data[:in_ch] = 1.0  # gamma=1

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
            for i, proj in enumerate(self.stem_projs):
                proj.weight.copy_(w_mean)
                proj.bias.copy_(b_mean)
                proj.weight.add_(torch.randn_like(proj.weight) * 1e-4)

        # Replace conv2 with identity so encoder.forward() gives us
        # the pre-conv2 hidden state
        self.encoder.conv2 = nn.Identity()

        if freeze_backbone:
            for n, p in self.encoder.named_parameters():
                p.requires_grad = False

    def forward(self, waveform, sem_emb):
        """waveform: [B, 2, N] @ 48kHz, sem_emb: [B, 128]
        returns: [B, n_stems, 64, T]
        """
        h = self.encoder(waveform)  # [B, in_ch, T] (pre-conv2 features)
        B, C, T = h.shape

        stems = []
        for i in range(self.n_stems):
            # FiLM: modulate backbone features per stem
            gb = self.stem_film[i](sem_emb)  # [B, C*2]
            gamma = gb[:, :C].unsqueeze(-1)   # [B, C, 1]
            beta = gb[:, C:].unsqueeze(-1)    # [B, C, 1]
            h_mod = gamma * h + beta          # [B, C, T]
            # Project to latent dim
            stem_lat = self.stem_projs[i](h_mod)  # [B, 64, T]
            stems.append(stem_lat)

        return torch.stack(stems, dim=1)  # [B, n_stems, 64, T]


if __name__ == "__main__":
    m = WaveformToFourStemLatents()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    nt = sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6
    print(f"uncond: {n:.1f}M total, {nt:.1f}M trainable")
    x = torch.randn(1, 2, 48000 * 4)
    y = m(x)
    print(f"in {tuple(x.shape)} → out {tuple(y.shape)}")

    m2 = WaveformToStemLatentsSemCond()
    n2 = sum(p.numel() for p in m2.parameters()) / 1e6
    emb = torch.randn(1, 128)
    y2 = m2(x, emb)
    print(f"\nsem_cond: {n2:.1f}M total")
    print(f"in {tuple(x.shape)} + emb {tuple(emb.shape)} → out {tuple(y2.shape)}")
