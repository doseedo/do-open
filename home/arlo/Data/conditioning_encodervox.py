# conditioning_encodervox.py - Vocal-aware conditioning encoder with lyric integration
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List


class VocalLyricEncoder(nn.Module):
    """
    Processes lyric data (lyrics_tensors, syllable_boundaries) into conditioning tokens.
    """
    def __init__(self, d_text: int = 768, lyric_emb_dim: int = 256, max_syllables: int = 512):
        super().__init__()
        self.d_text = d_text
        self.lyric_emb_dim = lyric_emb_dim

        # Project lyric embeddings to d_text
        self.lyric_proj = nn.Sequential(
            nn.Linear(lyric_emb_dim, d_text),
            nn.SiLU(),
            nn.Linear(d_text, d_text)
        )
        self.lyric_ln = nn.LayerNorm(d_text)

        # Syllable boundary encoding (1D conv to detect syllable edges)
        self.syllable_detector = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, d_text, kernel_size=3, padding=1)
        )

        # Cross-modal fusion: lyric content + boundary timing
        self.lyric_boundary_fusion = nn.MultiheadAttention(d_text, num_heads=8, batch_first=True)
        self.fusion_ln = nn.LayerNorm(d_text)

        # Positional encoding for syllable positions
        self.syllable_pos = nn.Embedding(max_syllables, d_text)

    def forward(self,
                lyrics_tensors: Dict[str, torch.Tensor],
                syllable_boundaries: torch.Tensor,  # [B, T_slow]
                T_slow: int) -> torch.Tensor:
        """
        Args:
            lyrics_tensors: dict with 'lyrics_embeddings' [B, N_syllables, lyric_emb_dim]
            syllable_boundaries: [B, T_slow] boundary markers
            T_slow: target sequence length

        Returns:
            lyric_tokens: [B, T_slow, D] frame-aligned lyric conditioning
        """
        B = syllable_boundaries.shape[0]
        device = syllable_boundaries.device

        # Extract lyric embeddings (could be pre-computed from ACE-Step)
        if "lyrics_embeddings" in lyrics_tensors:
            lyric_emb = lyrics_tensors["lyrics_embeddings"]  # [B, N_syllables, lyric_emb_dim]
        elif "phoneme_embeddings" in lyrics_tensors:
            lyric_emb = lyrics_tensors["phoneme_embeddings"]
        else:
            # Fallback: create dummy embeddings
            lyric_emb = torch.zeros(B, 1, self.lyric_emb_dim, device=device)

        # Project to d_text
        lyric_content = self.lyric_proj(lyric_emb)  # [B, N_syllables, D]
        lyric_content = self.lyric_ln(lyric_content)

        # Add positional encoding to syllables
        N_syllables = lyric_content.shape[1]
        syll_pos = torch.arange(min(N_syllables, self.syllable_pos.num_embeddings), device=device)
        lyric_content = lyric_content + self.syllable_pos(syll_pos)[None, :lyric_content.shape[1], :]

        # Process syllable boundaries to get frame-level timing
        boundary_signal = syllable_boundaries.unsqueeze(1)  # [B, 1, T_slow]
        boundary_features = self.syllable_detector(boundary_signal)  # [B, D, T_slow]
        boundary_features = boundary_features.transpose(1, 2)  # [B, T_slow, D]

        # Cross-attention: frame timing attends to syllable content
        # Query: boundary timing, Key/Value: lyric content
        lyric_tokens, _ = self.lyric_boundary_fusion(
            query=boundary_features,      # [B, T_slow, D]
            key=lyric_content,            # [B, N_syllables, D]
            value=lyric_content,          # [B, N_syllables, D]
        )

        lyric_tokens = self.fusion_ln(lyric_tokens + boundary_features)  # [B, T_slow, D]

        return lyric_tokens


class PerformanceConditionEncoderVocal(nn.Module):
    """
    Extended conditioning encoder with vocal/lyric awareness.
    """
    def __init__(
        self,
        d_text: int = 768,
        pr_dim: int = 128,
        enc_channels: int = 8,
        fast_per_slow: float = 6.96,
        group_vocab: int = 7,  # Extended for 'vocal' group
        subgroup_vocab: int = 17,  # Extended for vocal subgroups
        inst_emb_dim: int = 384,
        inst_strength: float = 3.0,
        film_strength: float = 1.0,
        channel_mod_strength: float = 1.0,
        pr_strength: float = 2.0,
        timbre_voiced_suppress: float = 0.8,
        enable_beat_posenc: bool = True,
        # NEW: Vocal-specific parameters
        lyric_strength: float = 1.5,
        lyric_emb_dim: int = 256,
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
        self.lyric_strength = float(lyric_strength)

        # per-frame projections
        self.pr_ln   = nn.LayerNorm(pr_dim)
        self.pr_proj = nn.Linear(pr_dim, d_text)
        # +2 scalar channels (voiced, onset)
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

        # channel-wise timbre modulation
        self.timbre_mod_scale = nn.Sequential(
            nn.Linear(inst_emb_dim, enc_channels), nn.SiLU(),
            nn.Linear(enc_channels, enc_channels), nn.Tanh()
        )
        self.timbre_mod_bias = nn.Sequential(
            nn.Linear(inst_emb_dim, enc_channels), nn.SiLU(),
            nn.Linear(enc_channels, enc_channels)
        )

        # FiLM on post-projection timbre features
        self.film_scale = nn.Sequential(
            nn.Linear(inst_emb_dim, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )
        self.film_bias = nn.Sequential(
            nn.Linear(inst_emb_dim, d_text), nn.SiLU(), nn.Linear(d_text, d_text)
        )

        self.fuse_ln  = nn.LayerNorm(d_text)
        self.cond_pos = nn.Embedding(8192, d_text)

        self.beat_pos = nn.Linear(2, d_text) if self.enable_beat_posenc else None
        self.register_buffer("phase_voiced_suppress", torch.tensor(1.0))
        self.register_buffer("phase_pr_boost", torch.tensor(1.0))
        self.register_buffer("dbg_roll_calls",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("dbg_roll_ok",      torch.zeros(1, dtype=torch.long))
        self.register_buffer("dbg_roll_empty",   torch.zeros(1, dtype=torch.long))
        self.register_buffer("dbg_roll_errors",  torch.zeros(1, dtype=torch.long))
        self.post_token_ln = nn.LayerNorm(d_text)

        # NEW: Vocal lyric encoder
        self.vocal_lyric_encoder = VocalLyricEncoder(
            d_text=d_text,
            lyric_emb_dim=lyric_emb_dim
        )

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
            denom = max(1.0, mx)
            x = (x / denom - 0.5) * 2.0

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        # fast→slow
        x = self.timbre_pool(x)                    # [B, C_fast, ~T_slow]
        x = self.timbre_proj(x)                    # [B, D, ~T_slow]

        # LN in D space to control scale
        x = x.transpose(1, 2)                      # [B, ~T_slow, D]
        x = self.timbre_ln(x)
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
        # NEW: Vocal conditioning (optional)
        vocal_conditioning: Optional[List[Optional[Dict[str, Any]]]] = None,
    ):
        B, _, T_slow = piano_roll.shape
        D = self.d_text

        # ----- voiced + onset from PR -----
        voiced = (piano_roll > 0).any(dim=1, keepdim=False).float()  # [B, T]
        onset = F.pad((piano_roll[:, :, 1:] > piano_roll[:, :, :-1]).any(dim=1).float(),
                      (1, 0))  # [B, T]

        # ----- PR path (louder, normalized) -----
        pr_in  = self.pr_ln(piano_roll.transpose(1, 2))  # [B, T, 128]
        pr_tok = self.pr_proj(pr_in)  # [B, T, D]
        pr_tok = pr_tok * (self.pr_strength * self.phase_pr_boost)

        # ----- Scalars (+ voiced & onset) -----
        sclr_in = torch.stack([amp, rframe, rbend * rbend_mask, voiced, onset], dim=-1)  # [B, T, 5]
        sclr_in = self.sclr_ln(sclr_in)
        sclr_tok = self.sclr_proj(sclr_in)  # [B, T, D]

        # Instrument embedding
        g_safe  = self._safe_ids(group_id,    self.group_emb.num_embeddings)
        sg_safe = self._safe_ids(subgroup_id, self.subgroup_emb.num_embeddings)
        g_emb   = self.group_emb(g_safe)  # [B, inst_emb_dim/2]
        sg_emb  = self.subgroup_emb(sg_safe)  # [B, inst_emb_dim/2]
        inst_cat = torch.cat([g_emb, sg_emb], dim=-1)  # [B, inst_emb_dim]

        # ----- Timbre path -----
        timb_T = self._downsample_encodec_to_slow(encodec_tokens, T_slow, group_id, subgroup_id)  # [B,T,D]

        # Frame-wise FiLM
        if self.film_strength != 0.0:
            gamma = (1.0 + torch.tanh(self.film_scale(inst_cat)) * self.film_strength).unsqueeze(1)  # [B,1,D]
            beta  = (torch.tanh(self.film_bias(inst_cat)) * self.film_strength).unsqueeze(1)  # [B,1,D]
            timb_T = timb_T * gamma + beta

        # ----- Optional beat-phase positional bias -----
        if self.enable_beat_posenc and self.beat_pos is not None:
            phi = (rframe - rframe.floor()).unsqueeze(-1)  # [B,T,1]
            beat_feats = torch.cat([torch.sin(2*torch.pi*phi),
                                    torch.cos(2*torch.pi*phi)], dim=-1)  # [B,T,2]
            pr_tok = pr_tok + 0.25 * self.beat_pos(beat_feats)

        # ----- NEW: Vocal lyric conditioning -----
        lyric_tok = None
        if vocal_conditioning is not None:
            # Process each item in batch
            lyric_tok_list = []
            for b_idx in range(B):
                vc = vocal_conditioning[b_idx]
                if vc is not None:
                    # Extract data for this batch item
                    lyrics_tensors_b = vc["lyrics_tensors"]
                    syllable_boundaries_b = vc["syllable_boundaries"].unsqueeze(0)  # [1, T_slow]

                    # Process through vocal encoder
                    lyric_tokens_b = self.vocal_lyric_encoder(
                        lyrics_tensors=lyrics_tensors_b,
                        syllable_boundaries=syllable_boundaries_b,
                        T_slow=T_slow
                    )  # [1, T_slow, D]
                    lyric_tok_list.append(lyric_tokens_b.squeeze(0))
                else:
                    # No vocal conditioning for this item - use zeros
                    lyric_tok_list.append(torch.zeros(T_slow, D, device=pr_tok.device))

            lyric_tok = torch.stack(lyric_tok_list, dim=0)  # [B, T_slow, D]
            lyric_tok = lyric_tok * self.lyric_strength

        # ----- Fusion -----
        if lyric_tok is not None:
            frame_tok = self.fuse_ln(pr_tok + sclr_tok + timb_T + lyric_tok)  # [B, T, D]
        else:
            frame_tok = self.fuse_ln(pr_tok + sclr_tok + timb_T)  # [B, T, D]

        # Global instrument token
        inst = self.inst_fuse(inst_cat).unsqueeze(1) * self.inst_strength  # [B,1,D]

        # Global timbre token
        enc_glb    = self.enc_global_pool(encodec_tokens.float()).squeeze(-1)  # [B, C_fast]
        timbre_glb = self.timbre_global(enc_glb).unsqueeze(1)  # [B,1,D]

        # Compose token stream: [INST, TIMBRE_GLB, FRAME_0..T-1]
        tokens = torch.cat([inst, timbre_glb, frame_tok], dim=1)  # [B,2+T,D]

        # positional embeddings
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tokens = tokens + self.cond_pos(pos)[None, :, :]

        # FINAL SAFETY: sanitize + LN
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e3, neginf=-1e3)
        tokens = self.post_token_ln(tokens)

        mask = torch.ones((B, L), device=tokens.device, dtype=torch.bool)
        return tokens, mask
