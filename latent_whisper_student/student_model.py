"""Latent Lyric Student: Oobleck latents → ACE-Step lyric tokens.

A small encoder-decoder transformer:
  Encoder : [B, 64, T_lat] Oobleck VAE latents @ 25 Hz → [B, T_lat, d_model]
  Decoder : autoregressive, cross-attends to encoder, outputs ACE-Step lyric
            tokens (VoiceBpeTokenizer, vocab ≈ 6693 + PAD/SOS/EOS).

Trained with teacher-forced cross-entropy against pre-computed lyric
token IDs from VocalConditioning (they were originally Whisper-decoded,
then re-tokenized with ACE-Step's tokenizer).

The architecture is intentionally small — see configure() for a "base"
preset (d=512, 6 enc, 6 dec, 8 heads, ~35 M params).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Special token ids. We keep the ACE-Step token ids unchanged (0..6692) and
# add three new ids on top.
ACE_VOCAB = 6693                 # VoiceBpeTokenizer.get_number_tokens()
PAD_ID = ACE_VOCAB               # 6693
SOS_ID = ACE_VOCAB + 1           # 6694
EOS_ID = ACE_VOCAB + 2           # 6695
VOCAB  = ACE_VOCAB + 3           # 6696


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))      # [1, L, D]

    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads,
                                          batch_first=True, dropout=dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model),
        )

    def forward(self, x, src_key_padding_mask=None):
        h = self.n1(x)
        a, _ = self.attn(h, h, h,
                         key_padding_mask=src_key_padding_mask,
                         need_weights=False)
        x = x + a
        x = x + self.mlp(self.n2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_mult: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads,
                                               batch_first=True,
                                               dropout=dropout)
        self.n2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads,
                                                batch_first=True,
                                                dropout=dropout)
        self.n3 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_mult),
            nn.GELU(),
            nn.Linear(d_model * mlp_mult, d_model),
        )

    def forward(self, y, mem, causal_mask,
                tgt_key_padding_mask=None,
                mem_key_padding_mask=None):
        h = self.n1(y)
        a, _ = self.self_attn(h, h, h,
                              attn_mask=causal_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)
        y = y + a
        h = self.n2(y)
        c, _ = self.cross_attn(h, mem, mem,
                               key_padding_mask=mem_key_padding_mask,
                               need_weights=False)
        y = y + c
        y = y + self.mlp(self.n3(y))
        return y


class LatentLyricStudent(nn.Module):
    def __init__(self,
                 in_ch: int = 64,
                 d_model: int = 512,
                 n_enc_layers: int = 6,
                 n_dec_layers: int = 6,
                 n_heads: int = 8,
                 vocab: int = VOCAB,
                 max_src_len: int = 1024,
                 max_tgt_len: int = 1024,
                 dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # --- encoder: Oobleck latents -> hidden states -----------------
        self.in_proj = nn.Sequential(
            nn.Conv1d(in_ch,     d_model // 2, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(d_model // 2, d_model,    kernel_size=7, padding=3),
            nn.GELU(),
        )
        self.enc_pos = SinusoidalPE(d_model, max_len=max_src_len)
        self.enc_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_ln = nn.LayerNorm(d_model)

        # --- decoder: lyric token stream --------------------------------
        self.tok_embed = nn.Embedding(vocab, d_model, padding_idx=PAD_ID)
        self.dec_pos = SinusoidalPE(d_model, max_len=max_tgt_len)
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_ln = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab, bias=False)
        # weight-tie output projection with token embedding
        self.out_proj.weight = self.tok_embed.weight

        self.apply(self._init_weights)
        # re-initialise the tied embedding with a small std so that
        # the weight-tied output logits start out near uniform (CE ≈ log V)
        nn.init.normal_(self.tok_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.tok_embed.weight[PAD_ID].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # -----------------------------------------------------------------
    def encode(self, lat: torch.Tensor,
               src_key_padding_mask: torch.Tensor | None = None):
        """lat: [B, in_ch, T_lat]  →  [B, T_lat, d_model]."""
        h = self.in_proj(lat)                            # [B, D, T]
        h = h.transpose(1, 2).contiguous()               # [B, T, D]
        h = self.enc_pos(h)
        for blk in self.enc_blocks:
            h = blk(h, src_key_padding_mask=src_key_padding_mask)
        return self.enc_ln(h)

    def _causal_mask(self, T: int, device):
        return torch.triu(
            torch.full((T, T), float("-inf"), device=device), diagonal=1)

    def decode(self, tgt_tokens: torch.Tensor, mem: torch.Tensor,
               tgt_key_padding_mask=None, mem_key_padding_mask=None):
        """tgt_tokens: [B, T_tok]  → logits [B, T_tok, vocab]."""
        y = self.tok_embed(tgt_tokens)                   # [B, T_tok, D]
        y = self.dec_pos(y)
        cmask = self._causal_mask(tgt_tokens.shape[1], y.device)
        for blk in self.dec_blocks:
            y = blk(y, mem, causal_mask=cmask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    mem_key_padding_mask=mem_key_padding_mask)
        y = self.dec_ln(y)
        return self.out_proj(y)

    def forward(self, lat: torch.Tensor, tgt_tokens: torch.Tensor,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        mem = self.encode(lat, src_key_padding_mask=src_key_padding_mask)
        return self.decode(tgt_tokens, mem,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           mem_key_padding_mask=src_key_padding_mask)

    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, lat: torch.Tensor, max_len: int = 512,
                 sos_id: int = SOS_ID, eos_id: int = EOS_ID,
                 temperature: float = 1.0):
        """Greedy autoregressive generation. lat: [B, C, T_lat]."""
        self.eval()
        mem = self.encode(lat)
        B = lat.shape[0]
        ys = torch.full((B, 1), sos_id, dtype=torch.long, device=lat.device)
        done = torch.zeros(B, dtype=torch.bool, device=lat.device)
        for _ in range(max_len - 1):
            logits = self.decode(ys, mem)                # [B, T, V]
            next_logits = logits[:, -1] / max(temperature, 1e-6)
            next_tok = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            # lock finished sequences to EOS
            next_tok = torch.where(done.unsqueeze(-1),
                                   torch.full_like(next_tok, eos_id),
                                   next_tok)
            ys = torch.cat([ys, next_tok], dim=1)
            done = done | (next_tok.squeeze(-1) == eos_id)
            if done.all():
                break
        return ys


def configure(size: str = "base") -> dict:
    presets = {
        "tiny":  dict(d_model=256, n_enc_layers=4, n_dec_layers=4, n_heads=4),
        "base":  dict(d_model=512, n_enc_layers=6, n_dec_layers=6, n_heads=8),
        "small": dict(d_model=768, n_enc_layers=8, n_dec_layers=8, n_heads=12),
    }
    return presets[size]


if __name__ == "__main__":
    for size in ("tiny", "base"):
        m = LatentLyricStudent(**configure(size))
        n = sum(p.numel() for p in m.parameters()) / 1e6
        lat = torch.randn(2, 64, 750)
        tok = torch.randint(0, ACE_VOCAB, (2, 40))
        logits = m(lat, tok)
        gen = m.generate(lat, max_len=16)
        print(f"{size}: {n:.1f}M  in lat {tuple(lat.shape)} tok {tuple(tok.shape)} "
              f"→ logits {tuple(logits.shape)}  gen {tuple(gen.shape)}")
