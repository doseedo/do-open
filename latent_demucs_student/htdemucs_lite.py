"""HTDemucs-Lite: htdemucs with CrossTransformer replaced by conv fusion.

Takes pretrained htdemucs encoder/decoder weights (9.6M params, all convs).
Replaces the 31.6M CrossTransformer (attention, WebGPU-killer) with a ~2M
conv fusion module. Adds embedding head for latent demucs conditioning.

Outputs:
  - Separated stems [B, 4, 2, N] (same as htdemucs)
  - Per-stem embeddings [B, 4, 128] (for v4cond conditioning)
  - STFT masks [B, 4, F, T] (derived from separated stems)

Total: ~12M params, all convs, no attention. WebGPU-safe.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ConvFusion(nn.Module):
    """Lightweight conv replacement for CrossTransformer.

    At the bottleneck:
      freq: [B, 384, 8, T]   (8 freq bins remaining after 4x encoder downsampling)
      time: [B, 384, T_time]  (waveform path)

    Cross-fusion via:
      1. Each branch gets self-refinement (depthwise-sep conv)
      2. Freq info injected into time via FiLM (pool freq → modulate time)
      3. Time info injected into freq via FiLM (pool time → modulate freq)
    """
    def __init__(self, channels=384, n_freq=8):
        super().__init__()
        self.n_freq = n_freq

        # Freq self-refinement (operate on flattened freq×time)
        self.freq_refine = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 7), padding=(1, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, (3, 7), padding=(1, 3), groups=channels),
            nn.Conv2d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        # Time self-refinement
        self.time_refine = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, 7, padding=3, groups=channels),
            nn.Conv1d(channels, channels, 1),
            nn.GroupNorm(8, channels),
            nn.GELU(),
        )

        # Cross-modulation: freq → time
        self.freq_to_time_pool = nn.AdaptiveAvgPool2d((1, None))  # pool freq dim
        self.freq_to_time_film = nn.Conv1d(channels, channels * 2, 1)

        # Cross-modulation: time → freq
        self.time_to_freq_film = nn.Conv1d(channels, channels * 2, 1)

        # Zero-init for identity at start
        self._zero_init()

    def _zero_init(self):
        """Zero-init so fusion starts as identity (passthrough).

        - Last conv in each refinement block → zero weights/bias
          so residual contribution starts at zero: x + 0 = x
        - FiLM layers → zero weights/bias
          so gamma=0, beta=0: x * (1+0) + 0 = x
        """
        # Zero the last pointwise conv in freq refinement (index 5)
        nn.init.zeros_(self.freq_refine[5].weight)
        nn.init.zeros_(self.freq_refine[5].bias)
        # Zero the last pointwise conv in time refinement (index 5)
        nn.init.zeros_(self.time_refine[5].weight)
        nn.init.zeros_(self.time_refine[5].bias)
        # Zero FiLM layers
        nn.init.zeros_(self.freq_to_time_film.weight)
        nn.init.zeros_(self.freq_to_time_film.bias)
        nn.init.zeros_(self.time_to_freq_film.weight)
        nn.init.zeros_(self.time_to_freq_film.bias)

    def forward(self, x, xt):
        """
        x:  [B, 384, 8, T]     (freq branch)
        xt: [B, 384, T_time]   (time branch)
        Returns: (x', xt') same shapes
        """
        # Self-refinement (starts as identity due to zero-init)
        x = x + self.freq_refine(x)
        xt = xt + self.time_refine(xt)

        # Cross-modulation: freq → time (starts as identity due to zero-init)
        freq_pooled = self.freq_to_time_pool(x).squeeze(2)  # [B, 384, T]
        freq_pooled = F.interpolate(freq_pooled, size=xt.shape[-1], mode='linear',
                                    align_corners=False)
        ft_film = self.freq_to_time_film(freq_pooled)  # [B, 768, T_time]
        gamma_ft, beta_ft = ft_film.chunk(2, dim=1)
        xt = xt * (1 + gamma_ft) + beta_ft

        # Cross-modulation: time → freq (starts as identity due to zero-init)
        time_pooled = F.adaptive_avg_pool1d(xt, x.shape[-1])  # [B, 384, T]
        tf_film = self.time_to_freq_film(time_pooled)  # [B, 768, T]
        gamma_tf, beta_tf = tf_film.chunk(2, dim=1)
        x = x * (1 + gamma_tf.unsqueeze(2)) + beta_tf.unsqueeze(2)

        return x, xt


class HTDemucsLite(nn.Module):
    """HTDemucs with attention replaced by conv fusion.

    Loads pretrained htdemucs weights for encoder/decoder.
    Replaces CrossTransformer (31.6M, attention) with ConvFusion (~2M, all convs).
    Adds embedding head for per-stem conditioning.
    """

    def __init__(self, embed_dim=128):
        super().__init__()
        from demucs.pretrained import get_model
        teacher = get_model('htdemucs').models[0]

        # Steal encoder/decoder weights
        self.encoder = teacher.encoder
        self.decoder = teacher.decoder
        self.tencoder = teacher.tencoder
        self.tdecoder = teacher.tdecoder
        self.freq_emb = teacher.freq_emb
        self.freq_emb_scale = teacher.freq_emb_scale

        # Copy normalization params and other attrs
        self.samplerate = teacher.samplerate
        self.audio_channels = teacher.audio_channels
        self.sources = teacher.sources
        self.nfft = teacher.nfft
        self.depth = len(self.encoder)
        self.use_train_segment = False

        # Copy the spec methods
        self._spec = teacher._spec
        self._magnitude = teacher._magnitude
        self._ispec = teacher._ispec
        self._mask = teacher._mask

        # Replace crosstransformer with conv fusion
        self.conv_fusion = ConvFusion(channels=384, n_freq=8)
        # No channel up/downsampler needed — we operate at 384 directly

        # Delete the original transformer (don't carry the 31.6M)
        del teacher

        # Per-stem embedding head from bottleneck features
        # xt at bottleneck is [B, 384, T_time] — shared, need per-stem
        self.embed_dim = embed_dim
        self.n_stems = len(self.sources)
        self.stem_queries = nn.Parameter(torch.randn(self.n_stems, 1, 384) * 0.02)
        self.embed_proj = nn.Sequential(
            nn.Linear(384, 256),
            nn.GELU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, mix):
        """
        mix: [B, 2, N] @ 44100Hz
        Returns dict with:
          stems: [B, 4, 2, N]
          embedding: [B, 4, 128]
          stft_masks: [B, 4, F, T]
        """
        length = mix.shape[-1]

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

        saved = []
        saved_t = []
        lengths = []
        lengths_t = []

        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            if idx < len(self.tencoder):
                lengths_t.append(xt.shape[-1])
                tenc = self.tencoder[idx]
                xt = tenc(xt)
                if not tenc.empty:
                    saved_t.append(xt)
                else:
                    inject = xt
            x = encode(x, inject)
            if idx == 0 and self.freq_emb is not None:
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb
            saved.append(x)

        # Conv fusion instead of crosstransformer
        x, xt = self.conv_fusion(x, xt)

        # Extract per-stem embeddings from bottleneck time features
        # xt is [B, 384, T_time] — shared representation
        # We need per-stem: split into 4 using learned projections
        emb_features = xt  # [B, 384, T_time]

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x, pre = decode(x, skip, lengths.pop(-1))

            offset = self.depth - len(self.tdecoder)
            if idx >= offset:
                tdec = self.tdecoder[idx - offset]
                length_t = lengths_t.pop(-1)
                if tdec.empty:
                    assert pre.shape[2] == 1, pre.shape
                    pre = pre[:, :, 0]
                    xt, _ = tdec(pre, None, length_t)
                else:
                    skip = saved_t.pop(-1)
                    xt, _ = tdec(xt, skip, length_t)

        assert len(saved) == 0
        assert len(lengths_t) == 0
        assert len(saved_t) == 0

        S = len(self.sources)
        x = x.view(B, S, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        zout = self._mask(z, x)
        x = self._ispec(zout, length)

        xt = xt.view(B, S, -1, length)
        xt = xt * stdt[:, None] + meant[:, None]
        stems = xt + x  # [B, 4, 2, N]

        # Per-stem embeddings via query-based attention pooling on bottleneck
        embeddings = []
        for i in range(S):
            q = self.stem_queries[i].expand(B, -1, -1)  # [B, 1, 384]
            # emb_features is [B, 384, T_time] → [B, T_time, 384]
            kv = emb_features.transpose(1, 2)
            attn = torch.bmm(q, kv.transpose(1, 2)).softmax(dim=-1)  # [B, 1, T]
            pooled = torch.bmm(attn, kv).squeeze(1)  # [B, 384]
            embeddings.append(self.embed_proj(pooled))
        embeddings = torch.stack(embeddings, dim=1)  # [B, 4, 128]

        # STFT masks derived from separated stems
        with torch.no_grad():
            stem_mono = stems.mean(dim=2)  # [B, 4, N]
            n_fft = 2048
            hop = 512
            window = torch.hann_window(n_fft, device=mix.device)
            mags = []
            for i in range(S):
                spec = torch.stft(stem_mono[:, i], n_fft, hop, window=window,
                                  return_complex=True).abs()
                mags.append(spec)
            mags = torch.stack(mags, dim=1)
            stft_masks = mags / (mags.sum(dim=1, keepdim=True) + 1e-8)

        return {
            "stems": stems,             # [B, 4, 2, N]
            "embedding": embeddings,    # [B, 4, 128]
            "stft_masks": stft_masks,   # [B, 4, F, T]
        }


if __name__ == "__main__":
    m = HTDemucsLite()
    # Count params
    total = sum(p.numel() for p in m.parameters()) / 1e6
    fusion = sum(p.numel() for p in m.conv_fusion.parameters()) / 1e6
    embed = sum(p.numel() for p in m.embed_proj.parameters()) / 1e6
    enc_dec = total - fusion - embed
    print(f"Total: {total:.1f}M params")
    print(f"  Encoder/decoder (from htdemucs): {enc_dec:.1f}M")
    print(f"  Conv fusion (new):               {fusion:.1f}M")
    print(f"  Embed head (new):                {embed:.1f}M")

    # Test forward
    x = torch.randn(1, 2, 44100 * 4)
    m.eval()
    with torch.no_grad():
        out = m(x)
    for k, v in out.items():
        print(f"  {k:15s} {tuple(v.shape)}")
