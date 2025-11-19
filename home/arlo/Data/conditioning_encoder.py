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
        self.pr_strength = float(pr_strength)
        self.timbre_voiced_suppress = float(timbre_voiced_suppress)
        self.enable_beat_posenc = bool(enable_beat_posenc)


        # per-frame projections
        self.pr_ln   = nn.LayerNorm(pr_dim)                 # NEW
        self.pr_proj = nn.Linear(pr_dim, d_text)
        # +2 scalar channels (voiced, onset) → change in_features from 3 → 5
        self.sclr_proj = nn.Sequential(
            nn.Linear(5, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )
        self.sclr_ln = nn.LayerNorm(5) 

        # encodec (fast -> slow) timbre
        k = int(round(self.fast_per_slow))
        self.timbre_pool = nn.AvgPool1d(kernel_size=k, stride=k, ceil_mode=True)
        self.timbre_proj = nn.Conv1d(enc_channels, d_text, kernel_size=1)
        self.timbre_ln   = nn.LayerNorm(d_text)
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

        self.beat_pos = nn.Linear(2, d_text) if self.enable_beat_posenc else None
        self.register_buffer("phase_voiced_suppress", torch.tensor(1.0))  # 1.0 = fully active
        self.register_buffer("phase_pr_boost", torch.tensor(1.0))  
        self.register_buffer("dbg_roll_calls",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("dbg_roll_ok",      torch.zeros(1, dtype=torch.long))
        self.register_buffer("dbg_roll_empty",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("dbg_roll_errors",  torch.zeros(1, dtype=torch.long))
        self.post_token_ln = nn.LayerNorm(d_text)

        
    @staticmethod
    def _safe_ids(ids: torch.Tensor, num_embeddings: int) -> torch.Tensor:
        ids = ids.to(dtype=torch.long)
        bad = (ids < 0) | (ids >= num_embeddings)
        if bad.any():
            ids = ids.clone()
            ids[bad] = 0
        return ids

    def _downsample_encodec_to_slow(self, enc_tokens, T_slow, group_id, subgroup_id):
        # Detect index-like inputs and scale to [-1,1] safely before any conv
        is_index = not torch.is_floating_point(enc_tokens)
        x = enc_tokens.to(torch.float32)

        # Heuristic: if these look like raw indices (max >> 64), map to [-1,1]
        with torch.no_grad():
            mx = torch.amax(x).item()
        if is_index or mx > 64:
            # assume codebook indices, usually up to ~1023
            denom = max(1.0, mx)   # per-batch safe
            x = (x / denom - 0.5) * 2.0

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # fast→slow
        x = self.timbre_pool(x)                    # [B, C_fast, ~T_slow]
        x = self.timbre_proj(x)                    # [B, D, ~T_slow]

        # LN in D space to control scale
        x = x.transpose(1, 2)                      # [B, ~T_slow, D]
        x = self.timbre_ln(x)                      # ← NEW
        x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)
        x = x.transpose(1, 2)                      # [B, D, ~T_slow]

        # Trim/pad to exactly T_slow
        cur = x.shape[-1]
        if cur < T_slow: x = F.pad(x, (0, T_slow - cur))
        elif cur > T_slow: x = x[..., :T_slow]
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

        # ----- voiced + onset from PR -----
        # voiced if any pitch active at frame
        voiced = (piano_roll > 0).any(dim=1, keepdim=False).float()            # [B, T]
        # crude onset: rising edge of any note
        onset = F.pad((piano_roll[:, :, 1:] > piano_roll[:, :, :-1]).any(dim=1).float(),
                      (1, 0))                                                  # [B, T]
        
        # ----- PR path (louder, normalized) -----
        pr_in  = self.pr_ln(piano_roll.transpose(1, 2))                        # [B, T, 128]
        pr_tok = self.pr_proj(pr_in)                                           # [B, T, D]
        pr_tok = pr_tok * (self.pr_strength * self.phase_pr_boost)            # ← make PR dominant

        # ----- Scalars (+ voiced & onset) -----
        # NOTE: rbend is already masked, keep as before but add v/onset
        sclr_in = torch.stack([amp, rframe, rbend * rbend_mask, voiced, onset], dim=-1) # [B, T, 5]
        sclr_in = self.sclr_ln(sclr_in)
        sclr_tok = self.sclr_proj(sclr_in)                                     # [B, T, D]

        # Instrument embedding (for FiLM + global token)
        g_safe  = self._safe_ids(group_id,    self.group_emb.num_embeddings)
        sg_safe = self._safe_ids(subgroup_id, self.subgroup_emb.num_embeddings)
        g_emb   = self.group_emb(g_safe)          # [B, inst_emb_dim/2]
        sg_emb  = self.subgroup_emb(sg_safe)      # [B, inst_emb_dim/2]
        inst_cat = torch.cat([g_emb, sg_emb], dim=-1)  # [B, inst_emb_dim]

        # ----- Timbre path, then voiced gating -----
        timb_T = self._downsample_encodec_to_slow(encodec_tokens, T_slow, group_id, subgroup_id)  # [B,T,D]

        # Frame-wise FiLM as you had it
        if self.film_strength != 0.0:
            gamma = (1.0 + torch.tanh(self.film_scale(inst_cat)) * self.film_strength).unsqueeze(1)  # [B,1,D]
            beta  = (torch.tanh(self.film_bias(inst_cat)) * self.film_strength).unsqueeze(1)         # [B,1,D] - now bounded
            timb_T = timb_T * gamma + beta

        # Timbre suppression disabled for cross-attention - let model learn to use timbre selectively
        # if self.timbre_voiced_suppress > 0.0:
        #     vgate = voiced.unsqueeze(-1)                                       # [B,T,1]
        #     suppress = self.timbre_voiced_suppress * self.phase_voiced_suppress
        #     timb_T = timb_T * (1.0 - suppress * vgate)                         # mute timbre when voiced

        # ----- Optional beat-phase positional bias from rframe -----
        if self.enable_beat_posenc and self.beat_pos is not None:
            # crude 0..1 phase from rframe; sin/cos features
            phi = (rframe - rframe.floor()).unsqueeze(-1)                      # [B,T,1]
            beat_feats = torch.cat([torch.sin(2*torch.pi*phi),
                                    torch.cos(2*torch.pi*phi)], dim=-1)        # [B,T,2]
            pr_tok = pr_tok + 0.25 * self.beat_pos(beat_feats)                 # small bias; PR still dominates

        frame_tok = self.fuse_ln(pr_tok + sclr_tok + timb_T)                   # [B, T, D]

        # Global instrument token (strong, configurable)
        inst = self.inst_fuse(inst_cat).unsqueeze(1) * self.inst_strength  # [B,1,D]

        # Global timbre token from encodec
        enc_glb    = self.enc_global_pool(encodec_tokens.float()).squeeze(-1)  # [B, C_fast]
        timbre_glb = self.timbre_global(enc_glb).unsqueeze(1)                  # [B,1,D]

        # Compose token stream: [INST, TIMBRE_GLB, FRAME_0..T-1]
        tokens = torch.cat([inst, timbre_glb, frame_tok], dim=1)         # [B,2+T,D]

        # positional embeddings
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tokens = tokens + self.cond_pos(pos)[None, :, :]

        # FINAL SAFETY: sanitize + LN the whole token stream before returning
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e3, neginf=-1e3)
        tokens = self.post_token_ln(tokens)                         # ← NEW

        mask = torch.ones((B, L), device=tokens.device, dtype=torch.bool)
        return tokens, mask
