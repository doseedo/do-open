# conditioning_encoder_no_encodec.py
# Version without encodec timbre conditioning - uses only PR, scalars, and instrument embeddings
import torch
from torch import nn
import torch.nn.functional as F

class PerformanceConditionEncoderNoEncodec(nn.Module):
    def __init__(
        self,
        d_text: int = 768,
        pr_dim: int = 128,
        fast_per_slow: float = 6.96,
        group_vocab: int = 6,
        subgroup_vocab: int = 13,
        inst_emb_dim: int = 384,
        inst_strength: float = 3.0,
        pr_strength: float = 2.0,
        enable_beat_posenc: bool = True,
    ):
        super().__init__()
        self.d_text = d_text
        self.fast_per_slow = fast_per_slow
        self.inst_strength = float(inst_strength)
        self.pr_strength = float(pr_strength)
        self.enable_beat_posenc = bool(enable_beat_posenc)

        # per-frame projections
        self.pr_ln   = nn.LayerNorm(pr_dim)
        self.pr_proj = nn.Linear(pr_dim, d_text)
        # +2 scalar channels (voiced, onset) → 5 inputs
        self.sclr_proj = nn.Sequential(
            nn.Linear(5, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )
        self.sclr_ln = nn.LayerNorm(5)

        # globals (instrument embeddings only, no encodec timbre)
        self.group_emb    = nn.Embedding(group_vocab,    inst_emb_dim // 2)
        self.subgroup_emb = nn.Embedding(subgroup_vocab, inst_emb_dim // 2)
        self.inst_fuse    = nn.Sequential(
            nn.Linear(inst_emb_dim, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )

        self.fuse_ln  = nn.LayerNorm(d_text)
        self.cond_pos = nn.Embedding(8192, d_text)

        self.beat_pos = nn.Linear(2, d_text) if self.enable_beat_posenc else None
        self.register_buffer("phase_pr_boost", torch.tensor(1.0))
        self.post_token_ln = nn.LayerNorm(d_text)


    @staticmethod
    def _safe_ids(ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        ids = ids.to(dtype=torch.long)
        bad = (ids < 0) | (ids >= num_embeddings)
        if bad.any():
            ids = ids.clone()
            ids[bad] = 0
        return ids

    def forward(
        self,
        piano_roll: torch.Tensor,     # [B, 128, T_slow]
        amp: torch.Tensor,            # [B, T_slow]
        rframe: torch.Tensor,         # [B, T_slow]
        rbend: torch.Tensor,          # [B, T_slow]
        rbend_mask: torch.Tensor,     # [B, T_slow]
        group_id: torch.Tensor,       # [B]
        subgroup_id: torch.Tensor,    # [B]
    ):
        B, _, T_slow = piano_roll.shape
        D = self.d_text

        # ----- voiced + onset from PR -----
        voiced = (piano_roll > 0).any(dim=1, keepdim=False).float()
        onset = F.pad((piano_roll[:, :, 1:] > piano_roll[:, :, :-1]).any(dim=1).float(),
                      (1, 0))

        # ----- PR path (louder, normalized) -----
        pr_in  = self.pr_ln(piano_roll.transpose(1, 2))
        pr_tok = self.pr_proj(pr_in)
        pr_tok = pr_tok * (self.pr_strength * self.phase_pr_boost)

        # ----- Scalars (+ voiced & onset) -----
        sclr_in = torch.stack([amp, rframe, rbend * rbend_mask, voiced, onset], dim=-1)
        sclr_in = self.sclr_ln(sclr_in)
        sclr_tok = self.sclr_proj(sclr_in)

        # Instrument embedding
        g_safe  = self._safe_ids(group_id,    self.group_emb.num_embeddings)
        sg_safe = self._safe_ids(subgroup_id, self.subgroup_emb.num_embeddings)
        g_emb   = self.group_emb(g_safe)
        sg_emb  = self.subgroup_emb(sg_safe)
        inst_cat = torch.cat([g_emb, sg_emb], dim=-1)

        # ----- Optional beat-phase positional bias from rframe -----
        if self.enable_beat_posenc and self.beat_pos is not None:
            phi = (rframe - rframe.floor()).unsqueeze(-1)
            beat_feats = torch.cat([torch.sin(2*torch.pi*phi),
                                    torch.cos(2*torch.pi*phi)], dim=-1)
            pr_tok = pr_tok + 0.25 * self.beat_pos(beat_feats)

        # Fuse only PR + scalars (no timbre from encodec)
        frame_tok = self.fuse_ln(pr_tok + sclr_tok)

        # Global instrument token (strong, configurable)
        inst = self.inst_fuse(inst_cat).unsqueeze(1) * self.inst_strength

        # Compose token stream: [INST, FRAME_0..T-1] (no global timbre token)
        tokens = torch.cat([inst, frame_tok], dim=1)

        # positional embeddings
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tokens = tokens + self.cond_pos(pos)[None, :, :]

        # FINAL SAFETY: sanitize + LN
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e3, neginf=-1e3)
        tokens = self.post_token_ln(tokens)

        mask = torch.ones((B, L), device=tokens.device, dtype=torch.bool)
        return tokens, mask
