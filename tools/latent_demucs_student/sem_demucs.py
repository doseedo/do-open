"""SemDemucs v4: waveform → STFT masks + per-stem sem embeddings. Pure convs.

One model, all signals, per-stem. Outputs everything needed for Option C:
  - STFT masks [B, 4, F, T'] for instant stem audio (mix_STFT × mask → iSTFT)
  - Per-stem semantic embeddings [B, 4, 128] for latent demucs conditioning
  - Pitch/RMS/vocal for downstream use

All pure convolutions — no attention, no transformer. WebGPU-friendly.

Architecture:
  - Shared waveform encoder (conv stack, 48kHz → ~31Hz)
  - Per-stem separation via learned biases + refinement MLPs
  - STFT mask head: per-stem features → [B, 4, F, T'] via conv upsampling in freq
  - Embedding head: per-stem attention pooling → [B, 4, 128]
  - Pitch/RMS/vocal heads (same as v2/v3)

Training targets:
  - Masks: STFT ratio masks from htdemucs teacher (frequency-domain)
  - Embedding: SemanticEncoder v1 on individual stem latents (distilled)
  - Pitch/RMS: frozen teacher models on GT stem latents
  - Vocal: manifest voice label
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=7, stride=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride, kernel // 2)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class StemSeparator(nn.Module):
    """Separate shared encoder features into per-stem frame sequences."""
    def __init__(self, n_stems, hidden):
        super().__init__()
        self.n_stems = n_stems
        self.stem_bias = nn.Parameter(torch.randn(n_stems, 1, hidden) * 0.02)
        self.stem_refine = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Linear(hidden, hidden),
            ) for _ in range(n_stems)
        ])

    def forward(self, h_seq):
        """h_seq: [B, T, C] → [B, n_stems, T, C]."""
        B, T, C = h_seq.shape
        stems = []
        for i in range(self.n_stems):
            h_stem = h_seq + self.stem_bias[i]
            h_stem = h_stem + self.stem_refine[i](h_stem)
            stems.append(h_stem)
        return torch.stack(stems, dim=1)


class STFTMaskHead(nn.Module):
    """Per-stem features + mix spectrogram → STFT masks.

    The mix spectrogram gives the mask head direct access to frequency content.
    The 1D encoder features tell it *which* stem each frequency belongs to.
    Combined: much better mask prediction than 1D features alone.
    """
    def __init__(self, hidden, n_freqs=1025, n_stems=4, spec_bins=64):
        super().__init__()
        self.n_freqs = n_freqs
        self.n_stems = n_stems
        self.spec_bins = spec_bins  # compressed freq resolution

        # Compress spectrogram: [B, 1, F, T_stft] → [B, 16, spec_bins, T_stft]
        # Downsample freq from 1025 → spec_bins via strided conv
        self.spec_compress = nn.Sequential(
            nn.Conv2d(1, 16, (15, 1), stride=(n_freqs // spec_bins, 1),
                      padding=(7, 0)),
            nn.GroupNorm(4, 16),
            nn.GELU(),
        )

        # Per-stem 1D refinement on encoder features
        self.conv_stack = nn.Sequential(
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )

        # Stem features → compressed freq bins
        self.freq_proj = nn.Linear(hidden, spec_bins)

        # Combine stem logits + compressed spec → mask at compressed resolution
        # Then upsample to full freq
        self.combine = nn.Sequential(
            nn.Conv2d(16 + 1, 16, (3, 3), padding=(1, 1)),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, h_stems, mix_spec=None):
        """
        h_stems:  [B, S, T, C]
        mix_spec: [B, 1, F, T_stft] (magnitude spectrogram, log-scaled)
        Returns:  mask_logits [B, S, n_freqs, T_stft]
        """
        B, S, T, C = h_stems.shape

        # Per-stem 1D refinement → compressed freq projection
        h = h_stems.reshape(B * S, T, C).transpose(1, 2)  # [B*S, C, T]
        h = self.conv_stack(h)  # [B*S, C, T]
        h = h.transpose(1, 2)  # [B*S, T, C]
        stem_logits = self.freq_proj(h)  # [B*S, T, spec_bins]
        stem_logits = stem_logits.transpose(1, 2)  # [B*S, spec_bins, T]

        if mix_spec is None:
            # Fallback: upsample to full freq and return
            sl = stem_logits.reshape(B, S, self.spec_bins, T)
            return F.interpolate(sl.reshape(B * S, 1, self.spec_bins, T),
                                 size=(self.n_freqs, T), mode='bilinear',
                                 align_corners=False).squeeze(1).reshape(B, S, self.n_freqs, T)

        # Compress spectrogram to spec_bins freq resolution
        spec_feat = self.spec_compress(mix_spec)  # [B, 16, spec_bins, T_stft]
        T_stft = spec_feat.shape[-1]
        Fb = spec_feat.shape[2]  # actual compressed bins

        # Align stem temporal dim to spectrogram T_stft
        stem_2d = stem_logits.reshape(B * S, self.spec_bins, T)
        if T != T_stft:
            stem_2d = F.interpolate(stem_2d.unsqueeze(1),
                                    size=(Fb, T_stft),
                                    mode='bilinear', align_corners=False).squeeze(1)
        else:
            if stem_2d.shape[1] != Fb:
                stem_2d = F.interpolate(stem_2d.unsqueeze(1),
                                        size=(Fb, T),
                                        mode='bilinear', align_corners=False).squeeze(1)
        stem_2d = stem_2d.reshape(B, S, Fb, T_stft)

        # Combine stem logits + compressed spec → masks at compressed resolution
        # Process all stems in one batch
        sl = stem_2d.reshape(B * S, 1, Fb, T_stft)
        sf = spec_feat.unsqueeze(1).expand(-1, S, -1, -1, -1).reshape(B * S, 16, Fb, T_stft)
        combined = torch.cat([sl, sf], dim=1)  # [B*S, 17, Fb, T_stft]
        out = self.combine(combined)  # [B*S, 1, Fb, T_stft]

        # Upsample to full freq resolution
        out = F.interpolate(out, size=(self.n_freqs, T_stft),
                            mode='bilinear', align_corners=False)
        return out.squeeze(1).reshape(B, S, self.n_freqs, T_stft)


class SemDemucs(nn.Module):
    """Waveform [B, 2, N] @ 48kHz → STFT masks + per-stem sem embeddings.

    Returns dict with:
      stft_masks:    [B, 4, F, T']     (softmax STFT masks for instant audio)
      mask_logits:   [B, 4, F, T']     (raw logits before softmax)
      embedding:     [B, 4, 128]       (per-stem sem embedding)
      pitch_logits:  [B, 4, T', 128]   (frame-level MIDI logits)
      rms:           [B, 4, T', 2]     (frame-level min/max envelope)
      vocal:         [B, 4]            (vocal probability per stem)
    """

    def __init__(self, n_stems=4, embed_dim=128, channels=64, n_pitch=128,
                 n_fft=2048):
        super().__init__()
        self.n_stems = n_stems
        self.embed_dim = embed_dim
        self.n_fft = n_fft
        self.n_freqs = n_fft // 2 + 1  # 1025
        self.stride = 2 * 4 * 4 * 4 * 4 * 3  # 1536

        # Shared waveform encoder: 48kHz → ~31Hz
        self.encoder = nn.Sequential(
            ConvBlock(2, channels, 7, 2),
            ConvBlock(channels, channels, 7, 4),
            ConvBlock(channels, channels * 2, 7, 4),
            ConvBlock(channels * 2, channels * 2, 5, 4),
            ConvBlock(channels * 2, channels * 4, 5, 4),
            ConvBlock(channels * 4, channels * 4, 5, 3),
        )

        hidden = channels * 4  # 256

        # Per-stem separation
        self.separator = StemSeparator(n_stems, hidden)

        # ── STFT mask head: [B, S, T, hidden] → [B, S, F, T] ──
        self.mask_head = STFTMaskHead(hidden, n_freqs=self.n_freqs, n_stems=n_stems)

        # ── Frame-level heads ──
        self.pitch_head = nn.Linear(hidden, n_pitch)
        self.rms_head = nn.Linear(hidden, 2)

        # ── Global per-stem embedding ──
        self.pool_query = nn.Parameter(torch.randn(n_stems, 1, hidden) * 0.02)
        self.embed_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

        # Vocal
        self.vocal_head = nn.Linear(embed_dim, 1)

    def forward(self, waveform):
        """waveform: [B, 2, N] → dict of per-stem signals."""
        h = self.encoder(waveform)  # [B, C, T']
        B, C, T = h.shape
        h_seq = h.transpose(1, 2)  # [B, T', C]

        # Compute mix spectrogram (free — just FFT, no learned params)
        with torch.no_grad():
            mono = waveform.mean(dim=1)  # [B, N]
            window = torch.hann_window(self.n_fft, device=waveform.device)
            spec = torch.stft(mono, self.n_fft, self.n_fft // 4, window=window,
                              return_complex=True).abs()  # [B, F, T_stft]
            mix_spec = (spec + 1e-8).log().unsqueeze(1)  # [B, 1, F, T_stft]

        # Separate into per-stem frame sequences
        h_stems = self.separator(h_seq)  # [B, S, T, C]

        # STFT masks (frequency-domain, with spectrogram input)
        mask_logits = self.mask_head(h_stems, mix_spec)  # [B, S, F, T_stft]
        stft_masks = torch.softmax(mask_logits, dim=1)  # sum to 1 across stems

        # Frame-level outputs
        pitch_logits = self.pitch_head(h_stems)  # [B, S, T, 128]
        rms = self.rms_head(h_stems)  # [B, S, T, 2]

        # Per-stem global embeddings
        embeddings = []
        for i in range(self.n_stems):
            q = self.pool_query[i].expand(B, -1, -1)
            h_stem = h_stems[:, i]
            attn = torch.bmm(q, h_stem.transpose(1, 2)).softmax(dim=-1)
            pooled = torch.bmm(attn, h_stem).squeeze(1)
            embeddings.append(self.embed_head(pooled))
        embeddings = torch.stack(embeddings, dim=1)
        vocal = self.vocal_head(embeddings).squeeze(-1)

        return {
            "stft_masks": stft_masks,      # [B, 4, F, T'] → instant stem audio
            "mask_logits": mask_logits,     # [B, 4, F, T'] → raw logits
            "embedding": embeddings,        # [B, 4, 128]   → latent demucs cond
            "pitch_logits": pitch_logits,   # [B, 4, T', 128]
            "rms": rms,                     # [B, 4, T', 2]
            "vocal": vocal,                 # [B, 4]
        }


if __name__ == "__main__":
    m = SemDemucs()
    n = sum(p.numel() for p in m.parameters()) / 1e6
    x = torch.randn(2, 2, 48000 * 4)
    out = m(x)
    print(f"{n:.1f}M params")
    print(f"in: {tuple(x.shape)}")
    for k, v in out.items():
        print(f"  {k:15s} {tuple(v.shape)}")
    # Size comparison
    masks_mb = out["stft_masks"].numel() * 4 / 1e6
    print(f"\n  STFT masks: {masks_mb:.1f} MB (4s clip)")
