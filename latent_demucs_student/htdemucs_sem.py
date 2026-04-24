"""HTDemucs encoder+transformer → masks + sem embeddings. No decoder.

Runs only the htdemucs encoder + cross-transformer (the separation work),
skips the decoder entirely (saves ~7M params + iSTFT). Trainable heads
on the bottleneck predict:
  - STFT ratio masks [B, 4, F, T_stft] for instant stem audio
  - Per-stem semantic embeddings [B, 4, 128] for latent demucs conditioning
  - Mix-level embedding [B, 128] for deployed decoder compatibility

At inference, the browser applies: stem_audio = iSTFT(mix_STFT × mask_i)

Compared to full htdemucs (42M):
  - Encoder + cross-transformer: ~35M (frozen)
  - Trainable heads: ~3M
  - Skips: decoder (6.8M) + iSTFT + full waveform reconstruction
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.pretrained import get_model


class HTDemucsHeads(nn.Module):
    """Frozen htdemucs encoder+transformer with trainable mask + embedding heads.

    Outputs:
        stft_masks:    [B, 4, F, T_stft] — magnitude ratio masks
        embedding:     [B, 4, 128]       — per-stem semantic embedding
        mix_embedding: [B, 128]          — mix-level semantic embedding
    """

    def __init__(self, embed_dim=128, n_fft=4096):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1  # 2049

        # Load htdemucs — we only need encoder + cross-transformer
        bag = get_model('htdemucs')
        demucs = bag.models[0]
        self.n_stems = len(demucs.sources)  # 4

        # Keep only the parts we need (frozen)
        self.encoder = demucs.encoder
        self.tencoder = demucs.tencoder
        self.crosstransformer = demucs.crosstransformer
        self.freq_emb = demucs.freq_emb
        self.freq_emb_scale = demucs.freq_emb_scale
        self.channel_upsampler = demucs.channel_upsampler
        self.channel_downsampler = demucs.channel_downsampler
        self.channel_upsampler_t = demucs.channel_upsampler_t
        self.channel_downsampler_t = demucs.channel_downsampler_t
        self.bottom_channels = demucs.bottom_channels

        # STFT params from htdemucs
        self._spec = demucs._spec
        self._magnitude = demucs._magnitude
        self.hop_length = demucs.hop_length
        self.samplerate = demucs.samplerate
        self.use_train_segment = demucs.use_train_segment
        self.segment = demucs.segment

        # Freeze all htdemucs components
        for p in self.encoder.parameters(): p.requires_grad = False
        for p in self.tencoder.parameters(): p.requires_grad = False
        for p in self.crosstransformer.parameters(): p.requires_grad = False
        if self.freq_emb is not None:
            for p in self.freq_emb.parameters(): p.requires_grad = False
        for m in [self.channel_upsampler, self.channel_downsampler,
                  self.channel_upsampler_t, self.channel_downsampler_t]:
            for p in m.parameters(): p.requires_grad = False

        # Bottleneck dims after channel_downsampler
        # freq: [B, 384, F=8, T=336] → flattened to [B, 384, F*T]
        # time: [B, 384, T_time=1344]
        bt_dim = 384

        # ── Mask head: bottleneck → [B, 4, F, T_stft] ──
        # Predict masks from freq branch (it's already in freq domain)
        # freq bottleneck is [B, 384, F=8, T_freq=336]
        # We need to upsample F from 8 to 2049 (full STFT bins)
        # Use a small conv stack that upsamples in frequency
        self.mask_head = nn.Sequential(
            # [B, 384, 8, T] → [B, 128, 8, T]
            nn.Conv2d(bt_dim, 128, 3, padding=1),
            nn.GELU(),
            # Upsample freq dim: 8 → 64 → 512 → 2049
            nn.ConvTranspose2d(128, 64, (8, 1), stride=(8, 1)),  # → [B, 64, 64, T]
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, (8, 1), stride=(8, 1)),   # → [B, 32, 512, T]
            nn.GELU(),
            nn.ConvTranspose2d(32, self.n_stems, (4, 1), stride=(4, 1), output_padding=(0, 0)),  # → [B, 4, 2048, T]
        )
        # Final freq adjustment to get exactly 2049 bins will be done with padding

        # ── Embedding heads: bottleneck → [B, 4, 128] and [B, 128] ──
        # Pool time branch for embeddings (richer temporal info)
        self.stem_queries = nn.Parameter(torch.randn(self.n_stems, 1, bt_dim) * 0.02)
        self.stem_embed_head = nn.Sequential(
            nn.Linear(bt_dim, bt_dim),
            nn.GELU(),
            nn.Linear(bt_dim, embed_dim),
        )

        self.mix_query = nn.Parameter(torch.randn(1, 1, bt_dim) * 0.02)
        self.mix_embed_head = nn.Sequential(
            nn.Linear(bt_dim, bt_dim),
            nn.GELU(),
            nn.Linear(bt_dim, embed_dim),
        )

        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[htdemucs-heads] frozen: {frozen/1e6:.1f}M, trainable: {trainable/1e6:.1f}M")

    def _run_encoder(self, mix):
        """Run htdemucs encoder + cross-transformer. Returns bottleneck features."""
        from einops import rearrange

        length = mix.shape[-1]
        training_length = int(self.segment * self.samplerate)
        length_pre_pad = None
        if self.use_train_segment and not self.training:
            if mix.shape[-1] < training_length:
                length_pre_pad = mix.shape[-1]
                mix = F.pad(mix, (0, training_length - length_pre_pad))

        z = self._spec(mix)
        mag = self._magnitude(z).to(mix.device)
        x = mag

        B, C, Fq, T = x.shape
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        xt = mix
        meant = xt.mean(dim=(1, 2), keepdim=True)
        stdt = xt.std(dim=(1, 2), keepdim=True)
        xt = (xt - meant) / (1e-5 + stdt)

        for idx, encode in enumerate(self.encoder):
            inject = None
            if idx < len(self.tencoder):
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if tenc.empty:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

        if self.crosstransformer:
            if self.bottom_channels:
                b, c, f, t = x.shape
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_upsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_upsampler_t(xt)

            x, xt = self.crosstransformer(x, xt)

            if self.bottom_channels:
                x = rearrange(x, "b c f t-> b c (f t)")
                x = self.channel_downsampler(x)
                x = rearrange(x, "b c (f t)-> b c f t", f=f)
                xt = self.channel_downsampler_t(xt)

        # x: [B, 384, F=8, T_freq], xt: [B, 384, T_time]
        return x, xt

    def forward(self, mix):
        B = mix.shape[0]

        # Frozen encoder + cross-transformer
        with torch.no_grad():
            x_freq, x_time = self._run_encoder(mix)
            # x_freq: [B, 384, 8, T_freq]
            # x_time: [B, 384, T_time]

        # ── Mask prediction from freq branch ──
        mask_raw = self.mask_head(x_freq)  # [B, 4, ~2048, T_freq]
        # Adjust to exactly n_freqs bins
        F_raw = mask_raw.shape[2]
        if F_raw < self.n_freqs:
            mask_raw = F.pad(mask_raw, (0, 0, 0, self.n_freqs - F_raw))
        elif F_raw > self.n_freqs:
            mask_raw = mask_raw[:, :, :self.n_freqs]

        # Upsample T_freq to match actual STFT T
        # STFT T = N // hop + 1. For now, leave as-is — will align at inference
        # Softmax across stems → masks sum to 1
        stft_masks = torch.softmax(mask_raw, dim=1)  # [B, 4, F, T_freq]

        # ── Embeddings from time branch ──
        xt_seq = x_time.transpose(1, 2)  # [B, T, 384]

        embeddings = []
        for i in range(self.n_stems):
            q = self.stem_queries[i].expand(B, -1, -1)
            attn = torch.bmm(q, xt_seq.transpose(1, 2)).softmax(dim=-1)
            pooled = torch.bmm(attn, xt_seq).squeeze(1)
            embeddings.append(self.stem_embed_head(pooled))
        embedding = torch.stack(embeddings, dim=1)  # [B, 4, 128]

        mix_q = self.mix_query.expand(B, -1, -1)
        mix_attn = torch.bmm(mix_q, xt_seq.transpose(1, 2)).softmax(dim=-1)
        mix_pooled = torch.bmm(mix_attn, xt_seq).squeeze(1)
        mix_embedding = self.mix_embed_head(mix_pooled)

        return {
            "stft_masks": stft_masks,
            "embedding": embedding,
            "mix_embedding": mix_embedding,
        }


if __name__ == "__main__":
    m = HTDemucsHeads()
    x = torch.randn(1, 2, 48000 * 4)
    with torch.no_grad():
        out = m(x)
    for k, v in out.items():
        print(f"  {k:15s} {tuple(v.shape)}")
