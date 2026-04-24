"""HTDemucs-Enc: encoder-only, no decoder. Predicts masks + embeddings directly.

Takes pretrained htdemucs encoder weights (2.8M params, all convs).
Replaces CrossTransformer with ConvFusion (1.2M).
Adds mask head + embed head on bottleneck features.
No decoder, no stem waveforms, no iSTFT.

Outputs:
  - stft_masks  [B, S, 1025, T]  (softmax across stems)
  - embedding   [B, S, 128]       (per-stem conditioning for v4cond)

Total: ~5M params, all convs, no attention, no decoder. Fast on WebGPU.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from htdemucs_lite import ConvFusion


class MaskHead(nn.Module):
    """Predict STFT masks from freq bottleneck [B, 384, 8, T] → [B, S, n_freqs, T].

    Flatten 384*8=3072 → compress → project to S*n_freqs per frame.
    Targets n_fft=2048 (1025 bins) matching v4cond, not htdemucs's internal 4096.
    """
    def __init__(self, channels=384, n_stems=4, n_freqs=1025, n_freq_bottleneck=8):
        super().__init__()
        self.n_stems = n_stems
        self.n_freqs = n_freqs
        in_ch = channels * n_freq_bottleneck  # 3072
        hid = 128

        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hid, 1),  # pointwise compress 3072→128
            nn.GroupNorm(8, hid),
            nn.GELU(),
            nn.Conv1d(hid, hid, 5, padding=2),  # temporal refine
            nn.GroupNorm(8, hid),
            nn.GELU(),
        )
        self.proj = nn.Conv1d(hid, n_stems * n_freqs, 1)

    def forward(self, x_freq):
        """x_freq: [B, 384, 8, T] → mask_logits [B, S, n_freqs, T]."""
        B, C, F, T = x_freq.shape
        h = x_freq.reshape(B, C * F, T)  # [B, 3072, T]
        h = self.net(h)  # [B, 256, T]
        h = self.proj(h)  # [B, S*n_freqs, T]
        return h.reshape(B, self.n_stems, self.n_freqs, T)


class HTDemucsEnc(nn.Module):
    """Encoder-only htdemucs: predicts STFT masks + embeddings from bottleneck.

    No decoder, no stem waveforms. Just encoder → fusion → heads.
    """

    def __init__(self, n_stems=4, embed_dim=128, model_name='htdemucs'):
        super().__init__()
        from demucs.pretrained import get_model
        teacher = get_model(model_name).models[0]

        # Take only the encoders
        self.encoder = teacher.encoder
        self.tencoder = teacher.tencoder
        self.freq_emb = teacher.freq_emb
        self.freq_emb_scale = teacher.freq_emb_scale

        self.samplerate = teacher.samplerate
        self.nfft = teacher.nfft
        self.n_stems = n_stems

        # Keep spec methods for input STFT
        self._spec = teacher._spec
        self._magnitude = teacher._magnitude

        # Conv fusion at bottleneck
        self.conv_fusion = ConvFusion(channels=384, n_freq=8)

        del teacher

        # Mask head: freq bottleneck → STFT masks (1025 bins = nfft 2048, matching v4cond)
        self.mask_head = MaskHead(channels=384, n_stems=n_stems, n_freqs=1025)

        # Embed head: time bottleneck → per-stem embeddings
        self.embed_dim = embed_dim
        self.stem_queries = nn.Parameter(torch.randn(n_stems, 1, 384) * 0.02)
        self.embed_proj = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, mix):
        """
        mix: [B, 2, N] @ 44100Hz
        Returns dict with:
          stft_masks: [B, S, F, T]  (softmax across stems)
          embedding:  [B, S, 128]
        """
        # STFT
        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape

        # Normalize
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        # Encoder (both freq and time paths)
        for idx, encode in enumerate(self.encoder):
            inject = None
            if idx < len(self.tencoder):
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    pass  # skip connections not needed (no decoder)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

        # Conv fusion
        x, xt = self.conv_fusion(x, xt)
        # x: [B, 384, 8, T], xt: [B, 384, T_time]

        # Mask head on freq features
        mask_logits = self.mask_head(x)  # [B, S, 1025, T]
        stft_masks = torch.softmax(mask_logits, dim=1)

        # Embed head on time features
        S = self.n_stems
        embeddings = []
        kv = xt.transpose(1, 2)  # [B, T_time, 384]
        for i in range(S):
            q = self.stem_queries[i].expand(B, -1, -1)  # [B, 1, 384]
            attn = torch.bmm(q, kv.transpose(1, 2)).softmax(dim=-1)
            pooled = torch.bmm(attn, kv).squeeze(1)  # [B, 384]
            embeddings.append(self.embed_proj(pooled))
        embeddings = torch.stack(embeddings, dim=1)  # [B, S, 128]

        return {
            "stft_masks": stft_masks,
            "mask_logits": mask_logits,
            "embedding": embeddings,
        }


if __name__ == "__main__":
    m = HTDemucsEnc(n_stems=4)
    total = sum(p.numel() for p in m.parameters()) / 1e6
    enc = sum(p.numel() for n, p in m.named_parameters()
              if 'encoder' in n or 'tencoder' in n or 'freq_emb' in n) / 1e6
    fusion = sum(p.numel() for p in m.conv_fusion.parameters()) / 1e6
    mask = sum(p.numel() for p in m.mask_head.parameters()) / 1e6
    embed = (sum(p.numel() for p in m.embed_proj.parameters()) +
             m.stem_queries.numel()) / 1e6
    print(f"Total: {total:.1f}M params")
    print(f"  Encoder (htdemucs pretrained): {enc:.1f}M")
    print(f"  Conv fusion:                   {fusion:.1f}M")
    print(f"  Mask head:                     {mask:.1f}M")
    print(f"  Embed head:                    {embed:.1f}M")

    x = torch.randn(1, 2, 44100 * 4)
    m.eval()
    with torch.no_grad():
        out = m(x)
    for k, v in out.items():
        print(f"  {k:15s} {tuple(v.shape)}")
