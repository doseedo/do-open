# conditioning_encodervox_simple.py - Simple vocal conditioning matching CN2 architecture
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List

# Import base encoder
from conditioning_encoder import PerformanceConditionEncoder


class VocalLyricProcessor(nn.Module):
    """
    Simple lyric processing - no cross-attention, just 1D convolution like your working CN2.
    Processes syllable boundaries and lyric embeddings into frame-aligned features.
    """
    def __init__(self, lyric_emb_dim: int = 256, hidden_dim: int = 768):
        super().__init__()
        self.lyric_emb_dim = lyric_emb_dim
        self.hidden_dim = hidden_dim

        # Process syllable boundaries with 1D conv (like piano roll processing)
        self.syllable_conv = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, padding=3),
            nn.SiLU(),
            nn.BatchNorm1d(128),  # Use BatchNorm1d for Conv1d (channels dimension)
            nn.Conv1d(128, hidden_dim, kernel_size=5, padding=2),
        )

        # Process lyric embeddings - simple average pooling to frame level
        self.lyric_proj = nn.Sequential(
            nn.Linear(lyric_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Combine syllable timing + lyric content
        self.fuse = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

    def forward(self,
                lyrics_tensors: Dict[str, torch.Tensor],
                syllable_boundaries: torch.Tensor,  # [B, T_slow]
                T_slow: int) -> torch.Tensor:
        """
        Simple processing - no attention, just convolution + fusion.

        Args:
            lyrics_tensors: dict with 'lyrics_embeddings' [B, N_syllables, lyric_emb_dim]
            syllable_boundaries: [B, T_slow] boundary markers
            T_slow: target sequence length

        Returns:
            lyric_features: [B, hidden_dim, T_slow] frame-aligned lyric features
        """
        B = syllable_boundaries.shape[0]
        device = syllable_boundaries.device

        # 1. Process syllable boundaries with conv (like piano roll)
        syll_signal = syllable_boundaries.unsqueeze(1)  # [B, 1, T_slow]
        syll_features = self.syllable_conv(syll_signal)  # [B, hidden_dim, T_slow]

        # 2. Process lyric embeddings
        if "lyrics_embeddings" in lyrics_tensors:
            lyric_emb = lyrics_tensors["lyrics_embeddings"]  # [B, N_syllables, lyric_emb_dim]
        elif "phoneme_embeddings" in lyrics_tensors:
            lyric_emb = lyrics_tensors["phoneme_embeddings"]
        else:
            # Fallback
            lyric_emb = torch.zeros(B, 1, self.lyric_emb_dim, device=device)

        # Project lyrics
        lyric_content = self.lyric_proj(lyric_emb)  # [B, N_syllables, hidden_dim]

        # Average pool lyric content to get global representation
        lyric_global = lyric_content.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        lyric_global = lyric_global.transpose(1, 2)  # [B, hidden_dim, 1]

        # Broadcast to T_slow
        lyric_broadcast = lyric_global.expand(-1, -1, T_slow)  # [B, hidden_dim, T_slow]

        # 3. Combine syllable timing + lyric content (additive like CN2)
        combined = syll_features + lyric_broadcast  # [B, hidden_dim, T_slow]

        # 4. Final fusion (transpose for LayerNorm, then back)
        combined_t = combined.transpose(1, 2)  # [B, T_slow, hidden_dim]
        fused = self.fuse(combined_t)  # [B, T_slow, hidden_dim]
        fused = fused.transpose(1, 2)  # [B, hidden_dim, T_slow]

        return fused


class PerformanceConditionEncoderVocalSimple(PerformanceConditionEncoder):
    """
    Extended conditioning encoder with simple vocal support - NO cross-attention!
    Matches your proven CN2 architecture.
    """
    def __init__(
        self,
        d_text: int = 768,
        pr_dim: int = 128,
        enc_channels: int = 8,
        fast_per_slow: float = 6.96,
        group_vocab: int = 7,
        subgroup_vocab: int = 17,
        inst_emb_dim: int = 384,
        inst_strength: float = 3.0,
        film_strength: float = 1.0,
        channel_mod_strength: float = 1.0,
        pr_strength: float = 2.0,
        timbre_voiced_suppress: float = 0.8,
        enable_beat_posenc: bool = True,
        # NEW: Vocal-specific parameters
        lyric_strength: float = 1.0,
        lyric_emb_dim: int = 256,
        voice_reference_strength: float = 2.0,
    ):
        # Initialize base encoder
        super().__init__(
            d_text=d_text,
            pr_dim=pr_dim,
            enc_channels=enc_channels,
            fast_per_slow=fast_per_slow,
            group_vocab=group_vocab,
            subgroup_vocab=subgroup_vocab,
            inst_emb_dim=inst_emb_dim,
            inst_strength=inst_strength,
            film_strength=film_strength,
            channel_mod_strength=channel_mod_strength,
            pr_strength=pr_strength,
            timbre_voiced_suppress=timbre_voiced_suppress,
            enable_beat_posenc=enable_beat_posenc,
        )

        self.lyric_strength = float(lyric_strength)
        self.voice_reference_strength = float(voice_reference_strength)

        # NEW: Simple lyric processor (no attention!)
        self.lyric_processor = VocalLyricProcessor(
            lyric_emb_dim=lyric_emb_dim,
            hidden_dim=d_text
        )

        # Project lyric features to d_text for final fusion
        self.lyric_to_tokens = nn.Sequential(
            nn.LayerNorm(d_text),
            nn.Linear(d_text, d_text)
        )

        # NEW: Voice reference projection (replaces EnCodec timbre)
        # Projects [256] Resemblyzer speaker embedding to inst_emb_dim
        self.voice_reference_proj = nn.Sequential(
            nn.Linear(256, inst_emb_dim),
            nn.SiLU(),
            nn.Linear(inst_emb_dim, inst_emb_dim)
        )

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
        reference_latent: Optional[torch.Tensor] = None,  # [B, 8, 16] voice reference
    ):
        B, _, T_slow = piano_roll.shape
        D = self.d_text

        # ----- Standard conditioning (same as base encoder) -----
        # voiced + onset from PR
        voiced = (piano_roll > 0).any(dim=1, keepdim=False).float()
        onset = F.pad((piano_roll[:, :, 1:] > piano_roll[:, :, :-1]).any(dim=1).float(), (1, 0))

        # PR path
        pr_in  = self.pr_ln(piano_roll.transpose(1, 2))
        pr_tok = self.pr_proj(pr_in)
        pr_tok = pr_tok * (self.pr_strength * self.phase_pr_boost)

        # Scalars
        sclr_in = torch.stack([amp, rframe, rbend * rbend_mask, voiced, onset], dim=-1)
        sclr_in = self.sclr_ln(sclr_in)
        sclr_tok = self.sclr_proj(sclr_in)

        # Instrument embedding
        g_safe  = self._safe_ids(group_id,    self.group_emb.num_embeddings)
        sg_safe = self._safe_ids(subgroup_id, self.subgroup_emb.num_embeddings)
        g_emb   = self.group_emb(g_safe)
        sg_emb  = self.subgroup_emb(sg_safe)
        inst_cat = torch.cat([g_emb, sg_emb], dim=-1)

        # Timbre path (still use EnCodec for non-vocal instruments)
        timb_T = self._downsample_encodec_to_slow(encodec_tokens, T_slow, group_id, subgroup_id)

        # NEW: Voice reference pathway (replaces EnCodec timbre for vocals)
        if reference_latent is not None:
            # reference_latent is [B, 256] speaker embedding from Resemblyzer
            # Project to d_text
            voice_emb = self.voice_reference_proj(reference_latent)  # [B, d_text]
            # Add to instrument token (same position as EnCodec timbre was added)
            inst_cat = inst_cat + voice_emb * self.voice_reference_strength

        # FiLM
        if self.film_strength != 0.0:
            gamma = (1.0 + torch.tanh(self.film_scale(inst_cat)) * self.film_strength).unsqueeze(1)
            beta  = (torch.tanh(self.film_bias(inst_cat)) * self.film_strength).unsqueeze(1)
            timb_T = timb_T * gamma + beta

        # Beat positional
        if self.enable_beat_posenc and self.beat_pos is not None:
            phi = (rframe - rframe.floor()).unsqueeze(-1)
            beat_feats = torch.cat([torch.sin(2*torch.pi*phi), torch.cos(2*torch.pi*phi)], dim=-1)
            pr_tok = pr_tok + 0.25 * self.beat_pos(beat_feats)

        # ----- NEW: Vocal lyric processing (SIMPLE - no attention!) -----
        lyric_tok = torch.zeros(B, T_slow, D, device=pr_tok.device)  # Default: zeros

        if vocal_conditioning is not None:
            # Process each batch item
            for b_idx in range(B):
                vc = vocal_conditioning[b_idx]
                if vc is not None:
                    # Simple processing - just conv + fusion
                    lyrics_tensors = vc["lyrics_tensors"]
                    syllable_boundaries = vc["syllable_boundaries"].unsqueeze(0)  # [1, T_slow]

                    # Process with simple conv-based processor
                    lyric_features = self.lyric_processor(
                        lyrics_tensors={k: v.unsqueeze(0) if isinstance(v, torch.Tensor) and v.dim() == 2 else v
                                       for k, v in lyrics_tensors.items()},
                        syllable_boundaries=syllable_boundaries,
                        T_slow=T_slow
                    )  # [1, D, T_slow]

                    # Convert to token format
                    lyric_tokens_b = lyric_features.transpose(1, 2).squeeze(0)  # [T_slow, D]
                    lyric_tok[b_idx] = lyric_tokens_b

        # Apply strength scaling
        lyric_tok = lyric_tok * self.lyric_strength

        # Project lyric features
        lyric_tok = self.lyric_to_tokens(lyric_tok)  # [B, T_slow, D]

        # ----- Fusion (additive like CN2, no attention!) -----
        frame_tok = self.fuse_ln(pr_tok + sclr_tok + timb_T + lyric_tok)  # [B, T, D]

        # Global tokens
        inst = self.inst_fuse(inst_cat).unsqueeze(1) * self.inst_strength
        enc_glb = self.enc_global_pool(encodec_tokens.float()).squeeze(-1)
        timbre_glb = self.timbre_global(enc_glb).unsqueeze(1)

        # Compose token stream
        tokens = torch.cat([inst, timbre_glb, frame_tok], dim=1)  # [B,2+T,D]

        # Positional embeddings
        L = tokens.shape[1]
        pos = torch.arange(L, device=tokens.device)
        tokens = tokens + self.cond_pos(pos)[None, :, :]

        # Final safety
        tokens = torch.nan_to_num(tokens, nan=0.0, posinf=1e3, neginf=-1e3)
        tokens = self.post_token_ln(tokens)

        mask = torch.ones((B, L), device=tokens.device, dtype=torch.bool)
        return tokens, mask
