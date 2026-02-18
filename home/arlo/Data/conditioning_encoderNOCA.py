# conditioning_encoder.py
import torch
from torch import nn
import torch.nn.functional as F

class PerformanceConditionEncoder(nn.Module):
    def __init__(
        self,
        d_text: int = 768,
        pr_dim: int = 128,
        enc_channels: int = 8,
        fast_per_slow: float = 6.96,
        group_vocab: int = 6,
        subgroup_vocab: int = 13,
        inst_emb_dim: int = 384,
        inst_strength: float = 3.0,          # NEW: scales the global instrument token
        film_strength: float = 1.0,          # NEW: scales FiLM modulation strength
        channel_mod_strength: float = 1.0, 
        pr_strength: float = 2.0,                 # ← make roll louder before fusion
        timbre_voiced_suppress: float = 0.8,      # ← 0..1 how much to mute timbre when voiced
        enable_beat_posenc: bool = True,   # NEW: scales channel-wise (pre-proj) modulation
    ):
        super().__init__()
        self.d_text = d_text
        self.fast_per_slow = fast_per_slow
        self.inst_strength = float(inst_strength)
        self.film_strength = float(film_strength)
        self.channel_mod_strength = float(channel_mod_strength)

        # per-frame projections
        self.pr_proj   = nn.Linear(pr_dim, d_text)
        self.sclr_proj = nn.Sequential(
            nn.Linear(3, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )

        # encodec (fast -> slow) timbre
        k = int(round(self.fast_per_slow))
        self.timbre_pool = nn.AvgPool1d(kernel_size=k, stride=k, ceil_mode=True)
        self.timbre_proj = nn.Conv1d(enc_channels, d_text, kernel_size=1)

        # globals
        self.enc_global_pool = nn.AdaptiveAvgPool1d(1)
        self.timbre_global   = nn.Sequential(
            nn.Linear(enc_channels, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )
        self.group_emb    = nn.Embedding(group_vocab,    inst_emb_dim // 2)
        self.subgroup_emb = nn.Embedding(subgroup_vocab, inst_emb_dim // 2)
        self.inst_fuse    = nn.Sequential(
            nn.Linear(inst_emb_dim, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )

        # NEW: channel-wise timbre modulation (acts on EnCodec channels before 1x1 proj)
        self.timbre_mod_scale = nn.Sequential(
            nn.Linear(inst_emb_dim, enc_channels), nn.SiLU(),
            nn.Linear(enc_channels, enc_channels), nn.Tanh()
        )
        self.timbre_mod_bias = nn.Sequential(
            nn.Linear(inst_emb_dim, enc_channels), nn.SiLU(),
            nn.Linear(enc_channels, enc_channels)
        )

        # NEW: FiLM on post-projection timbre features (acts in D_text space)
        self.film_scale = nn.Sequential(
            nn.Linear(inst_emb_dim, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )
        self.film_bias = nn.Sequential(
            nn.Linear(inst_emb_dim, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )

        self.fuse_ln  = nn.LayerNorm(d_text)
        self.cond_pos = nn.Embedding(8192, d_text)  # large enough for 2+T_slow

    @staticmethod
    def _safe_ids(ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        ids = ids.to(dtype=torch.long)
        bad = (ids < 0) | (ids >= num_embeddings)
        if bad.any():
            ids = ids.clone()
            ids[bad] = 0
        return ids

    def _downsample_encodec_to_slow(
        self,
        enc_tokens: torch.Tensor,  # [B, C_fast, T_fast]
        T_slow: int,
        group_id: torch.Tensor,    # [B]
        subgroup_id: torch.Tensor  # [B]
    ) -> torch.Tensor:
        """
        Instrument-aware channel-wise modulation applied BEFORE 1x1 projection to d_text.
        Returns [B, T_slow, d_text]
        """
        x = enc_tokens.float()                    # [B, C_fast, T_fast]
        x = self.timbre_pool(x)                   # [B, C_fast, ~T_slow]

        # Build instrument embedding
        g_safe  = self._safe_ids(group_id,    self.group_emb.num_embeddings)
        sg_safe = self._safe_ids(subgroup_id, self.subgroup_emb.num_embeddings)
        g_emb   = self.group_emb(g_safe)          # [B, inst_emb_dim/2]
        sg_emb  = self.subgroup_emb(sg_safe)      # [B, inst_emb_dim/2]
        inst_emb = torch.cat([g_emb, sg_emb], dim=-1)  # [B, inst_emb_dim]

        # Channel-wise (FiLM-like) modulation in EnCodec-channel space
        if self.channel_mod_strength != 0.0:
            scale_c = torch.tanh(self.timbre_mod_scale(inst_emb))   # [B, C_fast], properly bounded [-1,1]
            bias_c  = torch.tanh(self.timbre_mod_bias(inst_emb))    # [B, C_fast], properly bounded [-1,1]
            x = x * (1.0 + self.channel_mod_strength * scale_c).unsqueeze(-1) \
                  + (self.channel_mod_strength * bias_c).unsqueeze(-1)

        # Project to d_text per frame
        x = self.timbre_proj(x)                 # [B, d_text, ~T_slow]

        # Trim/pad to exactly T_slow and return [B, T_slow, d_text]
        cur = x.shape[-1]
        if cur < T_slow:
            x = F.pad(x, (0, T_slow - cur))
        elif cur > T_slow:
            x = x[..., :T_slow]
        return x.transpose(1, 2)

    def forward(
        self,
        piano_roll: torch.Tensor,     # [B, 128, T_slow]
        amp: torch.Tensor,            # [B, T_slow]
        rframe: torch.Tensor,         # [B, T_slow]
        rbend: torch.Tensor,          # [B, T_slow]
        rbend_mask: torch.Tensor,     # [B, T_slow]
        encodec_tokens: torch.Tensor, # [B, C_fast, T_fast]
        group_id: torch.Tensor,       # [B]
        subgroup_id: torch.Tensor,    # [B]
    ):
        B, _, T_slow = piano_roll.shape
        D = self.d_text

        # per-frame tokens (score-roll + scalars)
        pr_tok   = self.pr_proj(piano_roll.transpose(1, 2))                # [B, T, D]
        sclr_in  = torch.stack([amp, rframe, rbend * rbend_mask], dim=-1)  # [B, T, 3]
        sclr_tok = self.sclr_proj(sclr_in)                                  # [B, T, D]

        # Instrument embedding (for FiLM + global token)
        g_safe  = self._safe_ids(group_id,    self.group_emb.num_embeddings)
        sg_safe = self._safe_ids(subgroup_id, self.subgroup_emb.num_embeddings)
        g_emb   = self.group_emb(g_safe)          # [B, inst_emb_dim/2]
        sg_emb  = self.subgroup_emb(sg_safe)      # [B, inst_emb_dim/2]
        inst_cat = torch.cat([g_emb, sg_emb], dim=-1)  # [B, inst_emb_dim]

        # Instrument-aware timbre downsample (channel-space modulation)
        timb_T = self._downsample_encodec_to_slow(
            encodec_tokens, T_slow, group_id, subgroup_id
        )  # [B, T, D]

        # FiLM in D-text space (frame-wise)
        if self.film_strength != 0.0:
            gamma = (1.0 + torch.tanh(self.film_scale(inst_cat)) * self.film_strength).unsqueeze(1)  # [B,1,D]
            beta  = (torch.tanh(self.film_bias(inst_cat)) * self.film_strength).unsqueeze(1)         # [B,1,D] - now bounded
            timb_T = timb_T * gamma + beta

        frame_tok = self.fuse_ln(pr_tok + sclr_tok + timb_T)  # [B, T, D]

        # Global instrument token (strong, configurable)
        inst = self.inst_fuse(inst_cat).unsqueeze(1) * self.inst_strength  # [B,1,D]

        # Global timbre token from encodec
        enc_glb    = self.enc_global_pool(encodec_tokens.float()).squeeze(-1)  # [B, C_fast]
        timbre_glb = self.timbre_global(enc_glb).unsqueeze(1)                  # [B,1,D]

        # Compose token stream: [INST, TIMBRE_GLB, FRAME_0..T-1]
        tokens = torch.cat([inst, timbre_glb, frame_tok], dim=1)  # [B, 2+T, D]

        # positional embeddings
        L   = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tokens = tokens + self.cond_pos(pos)[None, :, :]

        # mask kept for API compatibility (unused when x-attn is off)
        mask = torch.ones((B, L), device=tokens.device, dtype=torch.bool)
        return tokens, mask
