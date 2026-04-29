"""Latent Demucs student: 1D U-Net mapping mix latent → 4 stem latents.

Input:  [B, 64, T]
Output: [B, 4, 64, T]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def gn(c):
    return nn.GroupNorm(min(32, c), c)


class ResBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.n1 = gn(c_in);  self.c1 = nn.Conv1d(c_in,  c_out, 3, padding=1)
        self.n2 = gn(c_out); self.c2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class LatentDemucsStudent(nn.Module):
    def __init__(self, in_ch=64, n_stems=4, channels=(128, 256, 384, 512)):
        super().__init__()
        self.n_stems = n_stems
        self.in_ch = in_ch
        self.channels = channels
        L = len(channels)

        self.in_proj = nn.Conv1d(in_ch, channels[0], 3, padding=1)

        # Encoder: at each level, two ResBlocks then (if not last) downsample
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        for i in range(L):
            c_in = channels[i-1] if i > 0 else channels[0]
            c    = channels[i]
            self.enc_blocks.append(nn.ModuleList([
                ResBlock(c_in, c),
                ResBlock(c, c),
            ]))
            if i < L - 1:
                self.downs.append(nn.Conv1d(c, c, 4, 2, 1))

        self.mid = nn.Sequential(ResBlock(channels[-1], channels[-1]),
                                 ResBlock(channels[-1], channels[-1]))

        # Decoder: upsample then two ResBlocks (with skip cat on first)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for i in reversed(range(L - 1)):  # from L-2 down to 0
            c_deep = channels[i+1]
            c      = channels[i]
            self.ups.append(nn.ConvTranspose1d(c_deep, c, 4, 2, 1))
            self.dec_blocks.append(nn.ModuleList([
                ResBlock(c * 2, c),  # cat with skip
                ResBlock(c, c),
            ]))

        self.out_norm = gn(channels[0])
        self.out_proj = nn.Conv1d(channels[0], n_stems * in_ch, 3, padding=1)

    def forward(self, x):
        B, C, T = x.shape
        L = len(self.channels)
        pad = (-T) % (2 ** (L - 1))
        if pad:
            x = F.pad(x, (0, pad))

        h = self.in_proj(x)
        skips = []
        for i in range(L):
            for blk in self.enc_blocks[i]:
                h = blk(h)
            if i < L - 1:
                skips.append(h)
                h = self.downs[i](h)

        h = self.mid(h)

        for j, i in enumerate(reversed(range(L - 1))):
            h = self.ups[j](h)
            h = torch.cat([h, skips[i]], dim=1)
            for blk in self.dec_blocks[j]:
                h = blk(h)

        h = self.out_proj(F.silu(self.out_norm(h)))   # [B, 4*64, T+pad]
        if pad:
            h = h[..., :T]
        return h.view(B, self.n_stems, self.in_ch, T)


if __name__ == "__main__":
    m = LatentDemucsStudent()
    n = sum(p.numel() for p in m.parameters())
    print(f"params: {n/1e6:.1f}M")
    for T in (250, 500, 137):
        x = torch.randn(2, 64, T)
        y = m(x)
        print(f"in {tuple(x.shape)} → out {tuple(y.shape)}")
