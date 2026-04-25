"""
Stemphonic Training Module — Modified training step for ACE-Step 1.5.

Implements the two core Stemphonic training interventions:
1. Shared noise: stems in the same group get the same noise latent
2. Independent CFG dropout per conditioning signal (1/3 drop rate)

The model architecture is NOT modified. We only change:
- How noise is sampled (shared within groups)
- How batches are constructed (done in dataset.py)
- How conditioning is assembled and dropped out

This module handles the full fine-tune (all DiT parameters), not LoRA.
For the initial LoRA experiment, set --lora flag.
"""

from __future__ import annotations

import logging
import os
import random
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def sample_timesteps_logit_normal(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    mu: float = -0.4,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Sample timesteps from logit-normal distribution, matching ACE-Step."""
    t = torch.sigmoid(
        torch.randn((batch_size,), device=device, dtype=dtype) * sigma + mu
    )
    return t


def shared_noise_for_groups(
    target_shape: tuple,
    group_ids: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample noise with sharing within groups (Stemphonic Sec 3.2).

    All stems with the same group_id get the same noise tensor.
    Different groups get independent noise.

    Args:
        target_shape: (B, T, D) shape of target latents
        group_ids: [B] integer group assignment per stem
        device: target device
        dtype: target dtype

    Returns:
        noise: [B, T, D] with shared noise within groups
    """
    B, T, D = target_shape
    noise = torch.zeros(B, T, D, device=device, dtype=dtype)

    unique_groups = group_ids.unique()
    for gid in unique_groups:
        mask = group_ids == gid
        # Sample one noise for the entire group
        group_noise = torch.randn(1, T, D, device=device, dtype=dtype)
        noise[mask] = group_noise.expand(mask.sum(), -1, -1)

    return noise


class StemphonicTrainingModule(nn.Module):
    """Training module implementing Stemphonic fine-tuning on ACE-Step 1.5.

    Takes a full ACE-Step model and wraps the training step with:
    - Shared noise sampling across stem groups
    - Sub-mix conditioning via context_latents channel concatenation
    - Activity mask conditioning via channel concatenation
    - Per-signal CFG dropout (1/3 rate, independent per signal)
    - Flow matching loss

    The model's forward pass is:
        decoder(hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=..., encoder_hidden_states=...,
                encoder_attention_mask=..., context_latents=...)

    Where:
        - hidden_states: [B, T, 64] noised target
        - context_latents: [B, T, 128] = concat(src_latents[64], masks[64])
            We repurpose src_latents to carry sub-mix conditioning
            and inject activity into the mask channel
    """

    def __init__(
        self,
        model: nn.Module,
        timestep_mu: float = -0.4,
        timestep_sigma: float = 1.0,
        cfg_dropout_rate: float = 1 / 3,
        use_lora: bool = False,  # False = frozen cross-attn (proven), True = LoRA cross-attn
        lora_rank: int = 128,
        lora_alpha: int = 128,
        midi_embed_dim: int = 32,
        freeze_lower_layers: int = 12,
        enable_midi: bool = True,  # Stage 1: False (no MIDI adapter/pr_loss)
        enable_pr_loss: bool = True,  # Stage 1: False
        cross_attn_lora: bool = True,  # Stage 1: False (self-attn LoRA only)
    ):
        super().__init__()
        self.model = model
        self.timestep_mu = timestep_mu
        self.timestep_sigma = timestep_sigma
        self.cfg_dropout_rate = cfg_dropout_rate
        self.enable_midi = enable_midi
        self.enable_pr_loss = enable_pr_loss
        self.cross_attn_lora = cross_attn_lora

        # Get null condition embedding for CFG dropout
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None
            logger.warning("model.null_condition_emb not found — CFG dropout on encoder disabled")

        # Get silence latent for building context_latents
        self._silence_latent = None

        # MIDI conditioning — CN2-style per-layer injection
        # Pipeline: [146, T] MIDI → Conv1d encoder → [B, T, D_midi] shared features
        # Then per-layer: Linear(D_midi, 2048) zero-init'd → added to hidden_states
        # This matches ControlNet2: control signal injected at every decoder layer
        from .midi_utils import TOTAL_MIDI_BINS, PITCHED_BINS  # 146 total, 128 pitched
        D_midi = 256
        NUM_DECODER_LAYERS = 24
        DECODER_HIDDEN = 2048

        if self.enable_midi:
            # Shared MIDI encoder — [B, 146, T] → [B, D_midi, T]
            self.midi_adapter = nn.Sequential(
                nn.Conv1d(TOTAL_MIDI_BINS, D_midi, 3, padding=1, bias=False),
                nn.LayerNorm([D_midi]),  # applied after permute in forward
                nn.SiLU(),
                nn.Conv1d(D_midi, D_midi, 3, padding=1, bias=False),
                nn.SiLU(),
            )

            # Shared temporal refinement — [B, T, D_midi] → [B, T, D_midi]
            self.midi_adapter_temporal = nn.Sequential(
                nn.Linear(D_midi, D_midi, bias=False),
                nn.SiLU(),
                nn.Linear(D_midi, D_midi),
            )
            nn.init.xavier_uniform_(self.midi_adapter_temporal[-1].weight, gain=0.01)
            nn.init.zeros_(self.midi_adapter_temporal[-1].bias)

            # Per-layer projections: D_midi → 2048 (zero-init for stable resume)
            self.midi_layer_projs = nn.ModuleList([
                nn.Linear(D_midi, DECODER_HIDDEN, bias=False)
                for _ in range(NUM_DECODER_LAYERS)
            ])
            for proj in self.midi_layer_projs:
                nn.init.zeros_(proj.weight)

            # Learnable pitch-to-height bank — spatial mask in MIDI feature space
            self.pitch2h_bank = nn.Parameter(torch.zeros(D_midi, PITCHED_BINS))
            with torch.no_grad():
                for h in range(D_midi):
                    center = int(h * 128 / D_midi)
                    sigma = 128 / D_midi
                    for p in range(128):
                        self.pitch2h_bank[h, p] = -((p - center) ** 2) / (2 * sigma ** 2)

            # Fixed scale for adapter output
            self.register_buffer("midi_adapter_scale", torch.tensor(1.0))

        # Activity embedding: per-frame binary activity → 16-dim learned embedding
        # Stemphonic Sec 3.3: "small (16-dim.) embedding" — project to 64 for additive patch
        self.activity_embed = nn.Embedding(2, 16)   # 0=silent, 1=active (paper uses 16-dim)
        self.activity_proj = nn.Linear(16, 64)       # project to latent dim for additive patch
        nn.init.zeros_(self.activity_embed.weight)   # zero-init: no-op at start
        nn.init.zeros_(self.activity_proj.weight)
        nn.init.zeros_(self.activity_proj.bias)

        # Resonance scalar: cosine sim between timbre ref and target → [B, 64] added to xt
        self.resonance_proj = nn.Sequential(
            nn.Linear(1, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        nn.init.zeros_(self.resonance_proj[-1].weight)
        nn.init.zeros_(self.resonance_proj[-1].bias)

        if self.enable_pr_loss:
            # PR-BCE auxiliary loss head (CN2-style roundtrip loss)
            self.pr_head = nn.Sequential(
                nn.Conv1d(64, 256, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv1d(256, 128, kernel_size=1),
            )
            self.pr_loss_weight = 0.5
            self.bce_logits = nn.BCEWithLogitsLoss(reduction='none')
        else:
            self.pr_loss_weight = 0.0

        self._setup_hybrid_finetune(
            freeze_lower_layers=freeze_lower_layers,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
        )

    def _setup_hybrid_finetune(self, freeze_lower_layers: int = 12,
                                use_lora: bool = False,
                                lora_rank: int = 128, lora_alpha: int = 128):
        """Hybrid fine-tuning: freeze lower layers, unfreeze upper layers.

        When use_lora=True:
          - Self-attention: LoRA on all 24 layers, base weights frozen
          - FFN: unfrozen in upper layers only
        When use_lora=False (full fine-tune):
          - Upper layers: self-attn + FFN fully unfrozen
          - Cross-attention: always FROZEN (preserves text/timbre/lyric conditioning)
          - Lower layers: fully frozen
        """
        # Step 1: Freeze everything
        for param in self.model.parameters():
            param.requires_grad = False

        lora_params = 0
        n_self = 0
        n_cross = 0

        # Step 2: Optionally inject LoRA on attention layers
        if use_lora:
            from peft import LoraConfig, get_peft_model

            attn_types_for_lora = ["self_attn"]
            if self.cross_attn_lora:
                attn_types_for_lora.append("cross_attn")

            attn_targets = []
            for name, _ in self.model.named_modules():
                if any(k in name for k in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    if "decoder" in name and any(at in name for at in attn_types_for_lora):
                        attn_targets.append(name)

            if not attn_targets:
                attn_targets = [
                    f"decoder.layers.{i}.{attn_type}.{proj}"
                    for i in range(24)
                    for attn_type in attn_types_for_lora
                    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
                ]

            lora_rank = 64
            lora_alpha = 128
            lora_config = LoraConfig(
                r=lora_rank, lora_alpha=lora_alpha,
                target_modules=attn_targets,
                lora_dropout=0.0, bias="none",
            )
            self.model = get_peft_model(self.model, lora_config)
            lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            n_self = sum(1 for t in attn_targets if "self_attn" in t)
            n_cross = sum(1 for t in attn_targets if "cross_attn" in t)
            logger.info("LoRA on %d modules (rank=%d, alpha=%d): %dM params (%d self-attn, %d cross-attn)",
                         len(attn_targets), lora_rank, lora_alpha, lora_params // 1_000_000, n_self, n_cross)

        # Step 3: Unfreeze upper layer components + global decoder components
        upper_unfrozen = 0
        for name, param in self.model.named_parameters():
            # Global decoder components (proj_in, proj_out, time_embed, norm_out, etc.)
            if "decoder." in name and "decoder.layers." not in name:
                param.requires_grad = True
                upper_unfrozen += param.numel()
                continue

            if "decoder.layers." not in name:
                continue
            try:
                layer_num = int(name.split("decoder.layers.")[1].split(".")[0])
            except (ValueError, IndexError):
                continue

            if layer_num < freeze_lower_layers:
                continue

            # Cross-attention base weights always frozen (preserves conditioning)
            is_cross_attn = "cross_attn" in name and "lora" not in name
            if is_cross_attn:
                continue

            if use_lora:
                # With LoRA: unfreeze FFN only, self-attn handled by LoRA
                is_self_attn_base = "self_attn" in name and "lora" not in name
                if not is_self_attn_base:
                    param.requires_grad = True
                    upper_unfrozen += param.numel()
            else:
                # No LoRA: fully unfreeze self-attn + FFN in upper layers
                param.requires_grad = True
                upper_unfrozen += param.numel()

        # Step 4: Count totals
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)

        if use_lora:
            lora_label = "self-attn + cross-attn LoRA" if self.cross_attn_lora else "self-attn LoRA only"
            logger.info("Hybrid finetune setup (%s):", lora_label)
            logger.info("  LoRA (24 layers, rank %d, alpha %d): %dM (%d self + %d cross)",
                         lora_rank, lora_alpha, lora_params // 1_000_000, n_self, n_cross)
        else:
            logger.info("Full fine-tune setup (no LoRA):")
        logger.info("  Upper %d layers unfrozen: %dM",
                     24 - freeze_lower_layers, upper_unfrozen // 1_000_000)
        logger.info("  Total trainable: %dM (%.1f%% of model)",
                     total_trainable // 1_000_000,
                     100 * total_trainable / (total_trainable + total_frozen))
        logger.info("  Total frozen: %dM", total_frozen // 1_000_000)

    def install_midi_layer_hooks(self):
        """Monkey-patch decoder layers to inject MIDI features after each layer.

        Each layer's output hidden_states gets:
            hidden_states = hidden_states + midi_layer_projs[i](midi_features)

        The per-layer projections are zero-initialized, so this is a no-op at start
        and gradually learns to inject MIDI control at each layer (CN2-style).
        """
        if not self.enable_midi or not hasattr(self, 'midi_layer_projs'):
            return

        decoder = self.model.decoder
        self._midi_hook_active = True
        self._midi_features_for_hook = None
        training_module = self  # capture for closure

        for layer_idx, layer in enumerate(decoder.layers):
            orig_forward = layer.forward
            proj = self.midi_layer_projs[layer_idx]

            def _make_hooked_forward(orig_fwd, layer_proj, idx):
                def hooked_forward(*args, **kwargs):
                    outputs = orig_fwd(*args, **kwargs)
                    # Inject MIDI features if available
                    mf = training_module._midi_features_for_hook
                    if mf is not None:
                        try:
                            midi_inject = layer_proj(mf)  # [1, T_dec, 2048]
                            if isinstance(outputs, tuple):
                                hs = outputs[0]
                            else:
                                hs = outputs
                            # Defensive: hs may be None/wrong shape during a
                            # mid-generation encoder swap (e.g. when
                            # generate_audio switches from cover→non-cover
                            # context). Skip injection on shape mismatch.
                            if hs is None:
                                return outputs
                            B_h = hs.shape[0]
                            B_m = midi_inject.shape[0]
                            if B_h == 2 * B_m:
                                # Batched CFG: inject into conditional half only
                                half = B_h // 2
                                hs = hs.clone()
                                hs[:half] = hs[:half] + midi_inject
                            elif B_h == B_m:
                                hs = hs + midi_inject
                            else:
                                # Unexpected batch geometry — bail rather than crash
                                return outputs
                            if isinstance(outputs, tuple):
                                outputs = (hs,) + outputs[1:]
                            else:
                                outputs = hs
                        except Exception as _midi_hook_err:
                            # Never bring down generation — log once and continue
                            if not getattr(training_module, '_midi_hook_warned', False):
                                logger.warning("MIDI hook %d skipped: %s", idx, _midi_hook_err)
                                training_module._midi_hook_warned = True
                            return outputs
                    return outputs
                return hooked_forward

            layer.forward = _make_hooked_forward(orig_forward, proj, layer_idx)

        logger.info("MIDI per-layer injection hooks installed on %d decoder layers",
                     len(decoder.layers))

    def set_silence_latent(self, silence_latent: torch.Tensor):
        """Set the silence latent used for building context_latents."""
        self._silence_latent = silence_latent

    def build_context_latents(
        self,
        sub_mix_latents: torch.Tensor,
        is_conditional: torch.Tensor,
        T: int,
        device: torch.device,
        dtype: torch.dtype,
        repaint_mask: torch.Tensor = None,
        repaint_src: torch.Tensor = None,
        is_repaint: torch.Tensor = None,
        cover_fsq_raw: torch.Tensor = None,
        is_cover: torch.Tensor = None,
    ) -> torch.Tensor:
        """Build context_latents [B, T, 128] for the DiT.

        context_latents = cat(src_latents[64], chunk_masks[64])

        Four modes:
        - From-scratch: src = zeros, chunk_masks = ones
        - Conditional: src = sub_mix_latent, chunk_masks = ones
        - Repainting: src = original stem in unmasked region, chunk_masks = repaint_mask
        - Cover: src = detokenized FSQ LM hints [T, 64], chunk_masks = ones
        """
        B = sub_mix_latents.shape[0]

        # Source latents: sub_mix for conditional, zeros for from-scratch
        src = torch.zeros(B, T, 64, device=device, dtype=dtype)
        latent_cond = is_conditional
        if latent_cond.any():
            src[latent_cond, :T] = sub_mix_latents[latent_cond, :T].to(device, dtype=dtype)

        # Chunk masks: default all ones (full generation)
        chunk_masks = torch.ones(B, T, 64, device=device, dtype=dtype)

        # Repainting: override src and chunk_masks for repaint stems
        if is_repaint is not None and is_repaint.any():
            rp_cpu = is_repaint.bool()
            rp = rp_cpu.to(device)
            rp_mask = repaint_mask[rp_cpu, :T].to(device, dtype=dtype)
            rp_src = repaint_src[rp_cpu, :T].to(device, dtype=dtype)
            src[rp, :T] = rp_src * rp_mask.unsqueeze(-1)
            chunk_masks[rp, :T] = rp_mask.unsqueeze(-1).expand_as(chunk_masks[rp, :T])

        # Cover mode: run raw FSQ through ACE-Step's detokenizer to get LM hints
        if is_cover is not None and is_cover.any() and cover_fsq_raw is not None:
            cv_cpu = is_cover.bool()
            cv = cv_cpu.to(device)
            fsq_in = cover_fsq_raw[cv_cpu].to(device, dtype=dtype)  # [N_cover, T_5Hz, 2048]
            with torch.no_grad():
                lm_hints = self.model.detokenize(fsq_in)  # [N_cover, T_25Hz, 64]
            # Crop/pad to match T
            T_hint = lm_hints.shape[1]
            if T_hint > T:
                lm_hints = lm_hints[:, :T]
            elif T_hint < T:
                lm_hints = F.pad(lm_hints, (0, 0, 0, T - T_hint))
            src[cv, :T] = lm_hints.to(dtype=dtype)

        return torch.cat([src, chunk_masks], dim=-1)  # [B, T, 128]

    def build_midi_features(
        self,
        stem_midi: torch.Tensor,
        has_stem_midi: torch.Tensor,
        B: int, T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Encode MIDI into shared features [B, T, D_midi] for per-layer injection.

        Pipeline:
          1. Conv1d encoder: [146, T] → [D_midi, T]
          2. Temporal refinement: [D_midi] → [D_midi] per frame
          3. Pitch-height spatial mask in D_midi space
          4. Result is injected per-layer via midi_layer_projs in the decoder hook

        Returns [B, T, D_midi] features (D_midi=256). Zeros for stems without MIDI.
        """
        D_midi = self.midi_adapter_temporal[-1].out_features  # 256
        features = torch.zeros(B, T, D_midi, device=device, dtype=dtype)

        if not has_stem_midi.any():
            # Dummy forward to keep gradient graph connected
            dummy = torch.zeros(1, stem_midi.shape[1], T, device=device, dtype=dtype)
            h = self.midi_adapter[0](dummy)
            h = h.permute(0, 2, 1); h = self.midi_adapter[1](h); h = self.midi_adapter[2](h)
            h = h.permute(0, 2, 1); h = self.midi_adapter[3](h); h = self.midi_adapter[4](h)
            h = h.permute(0, 2, 1)
            dummy_out = self.midi_adapter_temporal(h)
            features = features + dummy_out.sum() * 0
            return features

        midi_input = stem_midi[has_stem_midi].to(device, dtype=dtype)
        if midi_input.shape[-1] > T:
            midi_input = midi_input[:, :, :T]
        elif midi_input.shape[-1] < T:
            midi_input = F.pad(midi_input, (0, T - midi_input.shape[-1]))

        N = midi_input.shape[0]

        # Stage 1: Encode MIDI → [N, D_midi, T]
        h = self.midi_adapter[0](midi_input)
        h = h.permute(0, 2, 1)
        h = self.midi_adapter[1](h)
        h = self.midi_adapter[2](h)
        h = h.permute(0, 2, 1)
        h = self.midi_adapter[3](h)
        h = self.midi_adapter[4](h)

        if h.shape[-1] != T:
            h = F.interpolate(h, size=T, mode="linear", align_corners=False)

        # Stage 2: Temporal refinement [N, T, D_midi] → [N, T, D_midi]
        h = h.permute(0, 2, 1)
        midi_feat = self.midi_adapter_temporal(h)

        # Stage 3: Pitch-height spatial mask in D_midi space
        pr = midi_input[:, :128, :]  # [N, 128, T]
        W_hp = F.softplus(self.pitch2h_bank)  # [D_midi, 128]
        height_map = torch.einsum('npt,hp->nht', pr, W_hp).permute(0, 2, 1)  # [N, T, D_midi]
        height_map = height_map / (height_map.amax(dim=-1, keepdim=True) + 1e-6)
        has_pitched = (pr.sum(dim=1) > 0).any(dim=-1).float()
        gate = has_pitched.unsqueeze(1).unsqueeze(2)
        spatial_mask = (0.5 + 0.5 * height_map) * gate + (1.0 - gate)
        midi_feat = midi_feat * spatial_mask

        midi_feat = midi_feat * self.midi_adapter_scale
        features[has_stem_midi, :T] = midi_feat.to(dtype=dtype)
        return features

    # Keep old method name as alias for compatibility with probes
    def build_midi_additive_patch(self, stem_midi, has_stem_midi, B, T, device, dtype):
        """Legacy wrapper — probes still call this. Returns [B, T, 64] zeros."""
        return torch.zeros(B, T, 64, device=device, dtype=dtype)

    def apply_independent_cfg_dropout(
        self,
        encoder_hidden_states: torch.Tensor,
        sub_mix_latents: torch.Tensor,
        is_conditional: torch.Tensor,
        B: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply independent CFG dropout to each conditioning signal.

        Per Stemphonic: "All conditions are independently dropped out 1/3
        of the time to enable classifier-free guidance."

        We drop:
        1. encoder_hidden_states (text + lyric + timbre) → replace with null_cond_emb
        2. sub_mix_latents → replace with zeros
        """
        # Whole-encoder dropout at 15% (matching ACE-Step).
        # Per-signal dropout (timbre, lyrics) stays at 33% inside encode_conditioning().
        enc_rate = 0.15

        # Drop encoder conditioning (text/lyric/timbre)
        n_enc_dropped = 0
        if self._null_cond_emb is not None and enc_rate > 0:
            drop_mask = torch.rand(B, device=device) < enc_rate
            n_enc_dropped = drop_mask.sum().item()
            if drop_mask.any():
                null_expanded = self._null_cond_emb.expand_as(encoder_hidden_states)
                encoder_hidden_states = torch.where(
                    drop_mask.view(-1, 1, 1),
                    null_expanded,
                    encoder_hidden_states,
                )

        # Drop sub-mix conditioning (replace with zeros) — also at 15%
        n_sub_dropped = 0
        if enc_rate > 0:
            drop_sub_mix = torch.rand(B, device=device) < enc_rate
            n_sub_dropped = drop_sub_mix.sum().item()
            if drop_sub_mix.any():
                sub_mix_latents = torch.where(
                    drop_sub_mix.view(-1, 1, 1),
                    torch.zeros_like(sub_mix_latents),
                    sub_mix_latents,
                )

        self._last_cfg_drops = (n_enc_dropped, n_sub_dropped)
        return encoder_hidden_states, sub_mix_latents

    def encode_conditioning(
        self,
        batch: Dict[str, Any],
        text_embeddings_cache: Dict,
        B: int, T: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run condition encoder with text + timbre reference per stem.

        Text: per-stem-type cached embeddings (from Qwen3)
        Timbre: each stem's own VAE latent OR a Gemini-NN neighbor's latent
                (with 1/3 dropout for CFG)

        The timbre encoder takes VAE latents directly — no audio decode needed.
        """
        stem_types = batch["stem_types"]

        # Build per-stem timbre references
        REF_FRAMES = 30  # 1.2s at 25Hz
        timbre_refs = []
        timbre_order = []
        timbre_drop_flags = []

        for i in range(B):
            ref = None

            # Try loading a Gemini-NN neighbor's latent as timbre reference
            # (teaches the model to generalize, not just reconstruct)
            if hasattr(self, '_timbre_neighbors') and self._timbre_neighbors:
                latent_path = batch.get("latent_paths", [""])[i] if "latent_paths" in batch else ""
                # Map latent path back to GCS URI for neighbor lookup
                gcs_uri = latent_path.replace("/scratch/Latents2/", "gs://ptxsessiondata/")
                gcs_uri = gcs_uri.replace("/home/arlo/gcs-bucket/Latents2/", "gs://ptxsessiondata/")
                gcs_uri = gcs_uri.replace(".vae.pt", ".wav")

                nn_uris = self._timbre_neighbors.get(gcs_uri, [])
                if nn_uris and random.random() < 0.5:  # 50% chance use neighbor
                    nn_uri = random.choice(nn_uris)
                    # Convert neighbor URI back to latent path
                    nn_lat = nn_uri.replace("gs://ptxsessiondata/", "/scratch/Latents2/")
                    nn_lat = nn_lat.replace(".wav", ".vae.pt")
                    try:
                        raw = torch.load(nn_lat, map_location="cpu", weights_only=True)
                        z = raw["latents"] if isinstance(raw, dict) else raw
                        if z.dim() == 2 and z.shape[0] == 64:
                            z = z.T  # [T, 64]
                        t_start = random.randint(0, max(0, z.shape[0] - REF_FRAMES))
                        ref = z[t_start:t_start + REF_FRAMES].unsqueeze(0)
                    except Exception:
                        pass

            # Fallback: use stem's own latent as reference
            if ref is None:
                lat = batch["target_latents"][i]  # [T, 64]
                t_start = random.randint(0, max(0, lat.shape[0] - REF_FRAMES))
                ref = lat[t_start:t_start + REF_FRAMES].unsqueeze(0)  # [1, 30, 64]

            # CFG dropout: 1/3 chance replace with zeros (no timbre)
            timbre_dropped = random.random() < self.cfg_dropout_rate
            if timbre_dropped:
                ref = torch.zeros(1, REF_FRAMES, 64)

            timbre_refs.append(ref.to(device, dtype=dtype))
            timbre_order.append(i)
            timbre_drop_flags.append(timbre_dropped)

        refer_packed = torch.cat(timbre_refs, dim=0)  # [B, 30, 64]
        refer_order = torch.tensor(timbre_order, device=device, dtype=torch.long)

        # Compute resonance scalar: cosine similarity between timbre ref and target stem
        # Resonance is coupled to timbre — when timbre is dropped, resonance must also be 0
        resonance_values = torch.zeros(B, 1, device=device, dtype=dtype)
        for i in range(B):
            if timbre_drop_flags[i]:
                continue  # timbre dropped → resonance = 0
            ref_mean = refer_packed[i].mean(dim=0)  # [64]
            tgt_mean = batch["target_latents"][i].to(device, dtype=dtype).mean(dim=0)  # [64]
            ref_norm = ref_mean.norm()
            tgt_norm = tgt_mean.norm()
            if ref_norm > 1e-6 and tgt_norm > 1e-6:
                resonance_values[i, 0] = F.cosine_similarity(
                    ref_mean.unsqueeze(0), tgt_mean.unsqueeze(0)
                )
        self._last_resonance = resonance_values

        # Get text+lyric embeddings
        # Text: from cache (per stem type)
        # Lyrics: load from vocal conditioning JSON if available, else use cached [Instrumental]
        #
        # For vocal stems with lyrics, we load the pre-tokenized lyric_token_ids
        # and embed them through text_encoder.embed_tokens() (same as ACE-Step training).
        # The text_encoder is loaded during precompute and stored on self._text_encoder.
        all_text_hs, all_text_mask = [], []
        all_lyric_hs, all_lyric_mask = [], []
        n_cluster_hits = 0

        for i, stype in enumerate(stem_types):
            # Look up by v4 cache key (group|subgroup|bpm_bucket|timesig)
            cached = None

            # Try v4 key with full metadata
            if "stem_bpms" in batch and "stem_timesigs" in batch and "stem_subgroups" in batch:
                from stemphonic_trainer.preprocess_v4 import make_cache_key, bpm_bucket
                sg = batch["stem_subgroups"][i] if i < len(batch["stem_subgroups"]) else ""
                bpm = batch["stem_bpms"][i] if i < len(batch["stem_bpms"]) else ""
                ts = batch["stem_timesigs"][i] if i < len(batch["stem_timesigs"]) else ""
                key = make_cache_key(stype, sg, bpm_bucket(bpm), ts)
                cached = text_embeddings_cache.get(key)

            # Fallback: generic group key (group|||)
            if cached is None:
                from stemphonic_trainer.preprocess_v4 import make_cache_key
                cached = text_embeddings_cache.get(make_cache_key(stype))

            # Legacy fallback: bare stem type (for v3 index compatibility)
            if cached is None:
                cached = text_embeddings_cache.get(stype)

            if cached is None:
                cached = next(iter(text_embeddings_cache.values()))

            all_text_hs.append(cached["text_hs"])
            all_text_mask.append(cached["text_mask"])

            # Try loading real lyrics for vocal stems
            vc_path = ""
            if "vocal_cond_paths" in batch and i < len(batch["vocal_cond_paths"]):
                vc_path = batch["vocal_cond_paths"][i]

            used_real = False
            if vc_path and hasattr(self, '_text_encoder') and self._text_encoder is not None:
                try:
                    import json as _json
                    # CFG dropout: 1/3 chance skip lyrics
                    if random.random() >= self.cfg_dropout_rate:
                        with open(vc_path) as f:
                            vc_data = _json.load(f)
                        original_text = vc_data.get("original_text", "")
                        language = vc_data.get("primary_language", "en")
                        if original_text and len(original_text) > 5:
                            # Format and tokenize with Qwen3 (same as handler)
                            from stemphonic_trainer.preprocess import format_lyrics
                            lyrics_text = format_lyrics(original_text, language)
                            tok_out = self._tokenizer(
                                lyrics_text, return_tensors="pt",
                                padding=True, truncation=True, max_length=512,
                            )
                            tok_ids = tok_out["input_ids"].to(device)
                            msk = tok_out["attention_mask"].to(device, dtype=dtype)
                            with torch.no_grad():
                                lyric_emb = self._text_encoder.embed_tokens(tok_ids).to(dtype=dtype)
                            all_lyric_hs.append(lyric_emb.cpu())
                            all_lyric_mask.append(msk.cpu())
                            used_real = True
                except Exception:
                    pass

            if not used_real:
                all_lyric_hs.append(cached["lyric_hs"])
                all_lyric_mask.append(cached["lyric_mask"])

        # Pad and stack
        max_text_L = max(h.shape[-2] for h in all_text_hs)
        max_lyric_L = max(h.shape[-2] for h in all_lyric_hs)
        D_text = all_text_hs[0].shape[-1]
        D_lyric = all_lyric_hs[0].shape[-1]

        text_hs = torch.zeros(B, max_text_L, D_text, device=device, dtype=dtype)
        text_mask = torch.zeros(B, max_text_L, device=device, dtype=dtype)
        lyric_hs = torch.zeros(B, max_lyric_L, D_lyric, device=device, dtype=dtype)
        lyric_mask = torch.zeros(B, max_lyric_L, device=device, dtype=dtype)

        for i in range(B):
            L = all_text_hs[i].shape[-2]
            text_hs[i, :L] = all_text_hs[i].squeeze(0).to(dtype=dtype)
            text_mask[i, :L] = all_text_mask[i].squeeze(0).to(dtype=dtype)
            L = all_lyric_hs[i].shape[-2]
            lyric_hs[i, :L] = all_lyric_hs[i].squeeze(0).to(dtype=dtype)
            lyric_mask[i, :L] = all_lyric_mask[i].squeeze(0).to(dtype=dtype)

        # Run encoder with text + lyric + timbre (encoder is frozen)
        with torch.no_grad():
            encoder_hs, encoder_mask = self.model.encoder(
                text_hidden_states=text_hs,
                text_attention_mask=text_mask,
                lyric_hidden_states=lyric_hs,
                lyric_attention_mask=lyric_mask,
                refer_audio_acoustic_hidden_states_packed=refer_packed,
                refer_audio_order_mask=refer_order,
            )

        # Scale only the timbre token by resonance (cosine sim between timbre ref and target)
        # Token order from encoder: [lyrics (L_lyric), timbre (1), text (L_text)]
        # Timbre position = number of lyric tokens
        if hasattr(self, '_last_resonance') and self._last_resonance is not None:
            res = self._last_resonance.to(device, dtype=dtype)  # [B, 1]
            scale = (0.5 + 0.5 * res)  # [B, 1] — range 0.5 to 1.0
            # Timbre token is at position max_lyric_L (right after lyrics)
            timbre_idx = max_lyric_L
            if timbre_idx < encoder_hs.shape[1]:
                encoder_hs[:, timbre_idx, :] = encoder_hs[:, timbre_idx, :] * scale

        self._last_cluster_hits = n_cluster_hits
        return encoder_hs, encoder_mask

    def training_step(
        self,
        batch: Dict[str, Any],
        text_embeddings_cache: Optional[Dict] = None,
    ) -> Dict[str, torch.Tensor]:
        """Single Stemphonic training step.

        Args:
            batch: Dict from stemphonic_collate
            text_embeddings_cache: Pre-computed text/lyric embeddings per stem type

        Returns:
            Dict with 'loss' and optional diagnostics.
        """
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype

        autocast_ctx = torch.autocast(device_type="cuda", dtype=dtype) if device.type == "cuda" else nullcontext()

        with autocast_ctx:
            target_latents = batch["target_latents"].to(device, dtype=dtype)
            attention_mask = batch["attention_mask"].to(device, dtype=dtype)
            group_ids = batch["group_ids"].to(device)
            is_conditional = batch["is_conditional"].to(device)
            sub_mix_latents = batch["sub_mix_latents"].to(device, dtype=dtype)

            B, T, D = target_latents.shape

            # --- Encode conditioning (text + timbre per stem) ---
            if text_embeddings_cache is not None:
                encoder_hidden_states, encoder_attention_mask = self.encode_conditioning(
                    batch, text_embeddings_cache, B, T, device, dtype,
                )
            else:
                # Fallback: use pre-injected embeddings
                encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=dtype)
                encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=dtype)

            # Per-signal CFG dropout on timbre/lyrics already applied inside encode_conditioning().
            # Additionally drop whole encoder output → null_cond_emb, and sub_mix → zeros.
            # This is essential for inference-time CFG to work.
            encoder_hidden_states, sub_mix_latents = self.apply_independent_cfg_dropout(
                encoder_hidden_states, sub_mix_latents, is_conditional,
                B, device, dtype,
            )

            # --- Build context_latents [B, T, 128] ---
            context_latents = self.build_context_latents(
                sub_mix_latents, is_conditional,
                T, device, dtype,
                repaint_mask=batch.get("repaint_mask"),
                repaint_src=batch.get("repaint_src"),
                is_repaint=batch.get("is_repaint"),
                cover_fsq_raw=batch.get("cover_fsq_raw"),
                is_cover=batch.get("is_cover"),
            )

            # --- MIDI conditioning: build shared features for per-layer injection ---
            midi_features = None
            has_midi = torch.zeros(B, dtype=torch.bool)
            if self.enable_midi:
                stem_midi = batch.get("stem_midi", torch.zeros(B, 144, T))
                has_stem_midi = batch.get("has_stem_midi", torch.zeros(B, dtype=torch.bool))
                has_midi = has_stem_midi
                midi_features = self.build_midi_features(
                    stem_midi, has_stem_midi,
                    B, T, device, dtype,
                )

            # --- Activity → modify mask channels (dims 64-128) ---
            # Silent frames get mask=0 (decoder interprets as "don't generate here")
            # Active frames keep mask=1. Same semantics as repainting masks.
            activity = batch.get("activity_mask", torch.zeros(B, T, dtype=torch.long))
            activity = activity[:, :T].to(device).float()
            context_latents[:, :, 64:128] = context_latents[:, :, 64:128] * activity.unsqueeze(-1)

            # --- Shared noise (Stemphonic Sec 3.2) ---
            noise = shared_noise_for_groups(
                target_shape=(B, T, D),
                group_ids=group_ids,
                device=device,
                dtype=dtype,
            )

            # --- Timestep sampling (logit-normal) ---
            t = sample_timesteps_logit_normal(
                B, device, dtype,
                mu=self.timestep_mu,
                sigma=self.timestep_sigma,
            )
            t_ = t.view(-1, 1, 1)

            # --- Flow matching interpolation ---
            x0 = target_latents  # data
            x1 = noise           # noise
            xt = t_ * x1 + (1.0 - t_) * x0

            # --- Decoder forward with per-layer MIDI injection ---
            # Install MIDI features for the decoder hook to consume
            if midi_features is not None and hasattr(self, '_midi_hook_active'):
                # Downsample MIDI features to match decoder's patched sequence length
                # Decoder uses patch_size=2: T_decoder = T // 2
                patch_size = getattr(self.model.decoder, 'patch_size', 2)
                T_dec = T // patch_size
                # Average-pool adjacent frames: [B, T, D_midi] → [B, T//2, D_midi]
                mf = midi_features
                if mf.shape[1] > T_dec:
                    mf = mf[:, :T_dec * patch_size].reshape(B, T_dec, patch_size, -1).mean(dim=2)
                elif mf.shape[1] < T_dec:
                    mf = F.pad(mf, (0, 0, 0, T_dec - mf.shape[1]))
                self._midi_features_for_hook = mf
            elif midi_features is not None:
                self._midi_features_for_hook = None

            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # --- Flow matching loss ---
            flow = x1 - x0
            pred = decoder_outputs[0]

            # Per-element squared error [B, T, D]
            per_elem_loss = (pred - flow).pow(2)

            # Masked loss: only compute on valid frames
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            flow_loss = (per_elem_loss * mask).sum() / mask.sum() / D

            # --- PR-BCE auxiliary loss (disabled in Stage 1) ---
            pr_loss = torch.tensor(0.0, device=device)
            if self.enable_pr_loss and self.pr_loss_weight > 0:
                x0_hat = xt - t_ * pred
                has_pr = batch.get("has_pr_target", batch.get("has_stem_midi", torch.zeros(B, dtype=torch.bool)))
                if has_pr.any():
                    stem_midi_for_pr = batch.get("stem_midi", torch.zeros(B, 144, T))
                    pr_logits = self.pr_head(x0_hat.permute(0, 2, 1))
                    pr_tgt = batch.get("pr_target", stem_midi_for_pr[:, :128]).to(device, dtype=dtype)
                    if pr_tgt.shape[-1] < T:
                        pr_tgt = F.pad(pr_tgt, (0, T - pr_tgt.shape[-1]))
                    elif pr_tgt.shape[-1] > T:
                        pr_tgt = pr_tgt[:, :, :T]
                    pr_mask = has_pr.to(device).float()
                    bce = self.bce_logits(pr_logits, pr_tgt)
                    d = (pr_tgt[:, :, 1:] - pr_tgt[:, :, :-1]).clamp(min=0.0)
                    onset = torch.cat([torch.zeros_like(pr_tgt[:, :, :1]), d], dim=2).amax(dim=1)
                    bce = bce * (1.0 + onset).unsqueeze(1)
                    per_stem_bce = bce.mean(dim=(1, 2))
                    pr_loss = (per_stem_bce * pr_mask).sum() / pr_mask.sum().clamp(min=1)

                if not hasattr(self, '_train_step_count'):
                    self._train_step_count = 0
                self._train_step_count += 1
                pr_ramp = min(1.0, self._train_step_count / 5000)
                effective_pr_w = 0.1 + (self.pr_loss_weight - 0.1) * pr_ramp
                loss = flow_loss + effective_pr_w * pr_loss
            else:
                loss = flow_loss

            # Per-task loss breakdown
            per_stem_loss = (per_elem_loss * mask).sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1) / D  # [B]
            is_rp = batch.get("is_repaint", torch.zeros(B, dtype=torch.bool)).to(device)
            is_cv = batch.get("is_cover", torch.zeros(B, dtype=torch.bool)).to(device)
            is_extract = is_conditional & is_cv  # extraction = conditional + cover FSQ
            scratch_mask = (~is_conditional) & (~is_rp) & (~is_cv)
            loss_scratch = per_stem_loss[scratch_mask].mean().item() if scratch_mask.any() else 0.0
            loss_cond = per_stem_loss[is_conditional & ~is_extract].mean().item() if (is_conditional & ~is_extract).any() else 0.0
            loss_repaint = per_stem_loss[is_rp].mean().item() if is_rp.any() else 0.0
            loss_cover = per_stem_loss[is_cv & ~is_extract].mean().item() if (is_cv & ~is_extract).any() else 0.0
            loss_extract = per_stem_loss[is_extract].mean().item() if is_extract.any() else 0.0

        loss = loss.float()

        # Diagnostics for TensorBoard
        has_midi = batch.get("has_stem_midi", torch.zeros(B, dtype=torch.bool))
        resonance_val = self._last_resonance.mean().item() if hasattr(self, '_last_resonance') and self._last_resonance is not None else 0.0
        resonance_std = self._last_resonance.std().item() if hasattr(self, '_last_resonance') and self._last_resonance is not None and self._last_resonance.numel() > 1 else 0.0
        midi_scale_val = self.midi_adapter_scale.item() if self.enable_midi else 0.0
        activity_frac = activity.mean().item()

        # --- Cross-attention health monitoring ---
        # Deferred: only computed when caller sets _do_health_probe=True (every 50 steps)
        # Avoids wasting a full decoder forward pass on every step
        cross_attn_output_norm = 0.0

        # --- Prediction magnitude (health check) ---
        pred_magnitude = pred.detach().float().abs().mean().item()

        return {
            "loss": loss,
            "flow_loss": flow_loss.item(),
            "pr_loss": pr_loss.item() if isinstance(pr_loss, torch.Tensor) else pr_loss,
            "loss_from_scratch": loss_scratch,
            "loss_conditional": loss_cond,
            "loss_repaint": loss_repaint,
            "loss_cover": loss_cover,
            "loss_extract": loss_extract,
            "batch_size": B,
            "num_groups": batch["num_groups"],
            "num_conditional": is_conditional.sum().item(),
            "num_midi": has_midi.sum().item(),
            "num_repaint": is_rp.sum().item(),
            "num_cover": is_cv.sum().item(),
            "resonance_mean": resonance_val,
            "resonance_std": resonance_std,
            "midi_scale": midi_scale_val,
            "activity_frac": activity_frac,
            "cross_attn_output_norm": cross_attn_output_norm,
            "pred_magnitude": pred_magnitude,
            "cfg_enc_dropped": self._last_cfg_drops[0] if hasattr(self, '_last_cfg_drops') else 0,
            "cfg_sub_dropped": self._last_cfg_drops[1] if hasattr(self, '_last_cfg_drops') else 0,
            "cluster_hits": self._last_cluster_hits if hasattr(self, '_last_cluster_hits') else 0,
        }
