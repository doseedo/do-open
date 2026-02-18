#!/usr/bin/env python3
"""
Spacetime Keyframe Tree: trajectory ops with temporal intensity keyframes.

Fixes the two failure modes of the pure spacetime tree:
  1. All ops collapsed to slow envelopes → keyframes let ops specialize to
     different temporal regions (attack vs sustain vs release)
  2. No gating differentiation → ops that serve different temporal roles
     naturally get different gate activations

Architecture per operation:
    type_params: [16 values]     — WHAT pattern (global, FiLM conditioning)
    keyframes:   [K intensities] — WHEN active (interpolated to per-frame envelope)
    trajectory = op_temporal_net(type_params, t) * interp(keyframes, t)

Compression:
    Per-frame tree:  ~7748 values/sample
    Pure spacetime:  ~68 values/sample   (too compressed → 0.905 cos_sim)
    Keyframe hybrid: ~104-200 values/sample (target: closer to 0.987 per-frame)
    Compression: ~40-75x

Two encoder paths:
    1. Attention pool → global features → type params + gates (what ops, which sample)
    2. Stride pool → keyframe features → per-op intensities (when, from frame-level detail)

Symmetry-breaking: each op's Fourier basis is initialized with different frequency
biases so they start exploring different temporal scales.
"""

import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import gc
from tqdm import tqdm

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "spacetime_keyframe_tree"

Z_DIM = 128
MAX_FRAMES = 256
SAMPLE_RATE = 44100
DCAE_HOP = 4083
FPS = SAMPLE_RATE / DCAE_HOP  # ~10.8


# ============================================================
# Keyframe Interpolation
# ============================================================

def interpolate_keyframes(keyframe_values, T, positions=None):
    """
    Linearly interpolate K keyframe values to T frames.

    Args:
        keyframe_values: [B, K] — values at keyframe positions
        T: int — number of output frames
        positions: [K] — normalized positions in [0, 1]. If None, evenly spaced.

    Returns:
        [B, T] — interpolated values at each frame
    """
    B, K = keyframe_values.shape
    device = keyframe_values.device

    if positions is None:
        positions = torch.linspace(0, 1, K, device=device)

    # Frame positions in [0, 1]
    frame_pos = torch.linspace(0, 1, T, device=device)  # [T]

    # For each frame, find surrounding keyframes and interpolate
    # positions: [K], frame_pos: [T]
    # Expand for broadcasting: positions [1, K], frame_pos [T, 1]
    pos_expanded = positions.unsqueeze(0)    # [1, K]
    frame_expanded = frame_pos.unsqueeze(1)  # [T, 1]

    # Distance from each frame to each keyframe
    dist = frame_expanded - pos_expanded     # [T, K]

    # Find right neighbor for each frame (first keyframe with position >= frame)
    # and left neighbor (last keyframe with position <= frame)
    right_idx = torch.zeros(T, dtype=torch.long, device=device)
    for t in range(T):
        candidates = (positions >= frame_pos[t]).nonzero(as_tuple=True)[0]
        if len(candidates) > 0:
            right_idx[t] = candidates[0]
        else:
            right_idx[t] = K - 1

    left_idx = (right_idx - 1).clamp(min=0)

    # Handle edge cases: before first keyframe or after last
    left_pos = positions[left_idx]   # [T]
    right_pos = positions[right_idx] # [T]

    # Interpolation weight (how far between left and right)
    span = (right_pos - left_pos).clamp(min=1e-6)
    alpha = ((frame_pos - left_pos) / span).clamp(0, 1)  # [T]

    # Gather keyframe values
    # keyframe_values: [B, K]
    left_vals = keyframe_values[:, left_idx]    # [B, T]
    right_vals = keyframe_values[:, right_idx]  # [B, T]

    result = (1 - alpha).unsqueeze(0) * left_vals + alpha.unsqueeze(0) * right_vals
    return result  # [B, T]


# ============================================================
# Keyframe Spacetime Operation
# ============================================================

class KeyframeSpacetimeOperation(nn.Module):
    """
    Trajectory generator with keyframe intensity envelope.

    Two components:
      1. Base trajectory: FiLM(type_params) × temporal_net → [B, T, z_dim]
         (what kind of pattern — vibrato, decay, noise, etc.)
      2. Intensity envelope: interpolate(keyframes) → [B, T, 1]
         (when and how strongly this pattern contributes)

    Final output: base_trajectory * intensity_envelope

    Each op is initialized with a different frequency bias to break symmetry:
      - Op 0-1: biased toward low frequencies (slow envelopes, ~0.1-1 Hz)
      - Op 2-3: biased toward mid frequencies (modulation, ~1-3 Hz)
      - Op 4-5: biased toward high frequencies (vibrato-rate, ~3-8 Hz)
      - Op 6-7: broadband (no bias)
    """

    def __init__(self, param_dim=16, z_dim=128, hidden=64, n_basis=32,
                 freq_bias='broadband'):
        super().__init__()
        self.z_dim = z_dim
        self.hidden = hidden

        # Fixed Fourier basis frequencies (Hz)
        n_freq = n_basis // 2
        base_freqs = torch.logspace(math.log10(0.1), math.log10(8.0), n_freq)

        # Apply frequency bias to break symmetry between ops
        if freq_bias == 'low':
            # Emphasize low frequencies: compress range to 0.05-2 Hz
            base_freqs = torch.logspace(math.log10(0.05), math.log10(2.0), n_freq)
        elif freq_bias == 'mid':
            # Emphasize mid frequencies: 0.5-4 Hz
            base_freqs = torch.logspace(math.log10(0.5), math.log10(4.0), n_freq)
        elif freq_bias == 'high':
            # Emphasize high frequencies: 2-8 Hz
            base_freqs = torch.logspace(math.log10(2.0), math.log10(8.0), n_freq)
        # 'broadband' keeps default 0.1-8 Hz

        self.register_buffer('freqs', base_freqs)

        time_feat_dim = n_basis + 2  # sin/cos pairs + scaled_time + bias

        # Two-stage FiLM conditioning from type params
        self.film1 = nn.Linear(param_dim, hidden * 2)
        self.film2 = nn.Linear(param_dim, hidden * 2)

        # Temporal network
        self.layer1 = nn.Linear(time_feat_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, z_dim)

        # Small init
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, type_params, keyframe_intensities, T):
        """
        type_params: [B, param_dim] — what pattern (global)
        keyframe_intensities: [B, K] — when/how strong (sparse temporal)
        T: int — number of time frames
        Returns: [B, T, z_dim]
        """
        B = type_params.shape[0]
        device = type_params.device

        # ---- Base trajectory from type params ----
        t = torch.arange(T, device=device, dtype=torch.float32) / FPS
        t_col = t.unsqueeze(-1)

        phases = 2 * math.pi * self.freqs.unsqueeze(0) * t_col
        time_feats = torch.cat([
            torch.sin(phases),
            torch.cos(phases),
            t_col / 10.0,
            torch.ones_like(t_col),
        ], dim=-1).unsqueeze(0).expand(B, -1, -1)

        gamma1, beta1 = self.film1(type_params).chunk(2, dim=-1)
        h = self.layer1(time_feats)
        h = gamma1.unsqueeze(1) * h + beta1.unsqueeze(1)
        h = TF.gelu(h)

        gamma2, beta2 = self.film2(type_params).chunk(2, dim=-1)
        h = self.layer2(h)
        h = gamma2.unsqueeze(1) * h + beta2.unsqueeze(1)
        h = TF.gelu(h)

        base_trajectory = self.output(h)  # [B, T, z_dim]

        # ---- Intensity envelope from keyframes ----
        envelope = interpolate_keyframes(keyframe_intensities, T)  # [B, T]
        # Allow negative envelope (for subtractive contributions)
        # but scale so default (all 1s) = no change
        envelope = envelope.unsqueeze(-1)  # [B, T, 1]

        return base_trajectory * envelope


# ============================================================
# Keyframe Spacetime Codec
# ============================================================

class KeyframeSpacetimeCodec(nn.Module):
    """
    Spacetime tree with keyframe temporal control.

    Two encoder paths:
      1. Attention pool → global_feat → type_params + gates
         (what ops and which sample — same as pure spacetime)
      2. Stride pool → keyframe_feat → per-op keyframe intensities
         (when each op is active — preserves temporal detail)

    This dual-path design ensures keyframes are informed by frame-level
    temporal structure, not just the compressed global representation.
    """

    def __init__(self, z_dim=128, n_ops=8, param_dim=16, n_keyframes=8,
                 encoder_hidden=256, top_k=4, n_basis=32):
        super().__init__()
        self.z_dim = z_dim
        self.n_ops = n_ops
        self.param_dim = param_dim
        self.n_keyframes = n_keyframes
        self.top_k = top_k
        self.encoder_hidden = encoder_hidden

        # ---- Shared temporal encoder ----
        self.input_proj = nn.Sequential(
            nn.Linear(z_dim, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
        )

        self.temporal_conv = nn.Sequential(
            nn.Conv1d(encoder_hidden, encoder_hidden, kernel_size=7, padding=3),
            nn.GroupNorm(1, encoder_hidden),
            nn.GELU(),
            nn.Conv1d(encoder_hidden, encoder_hidden, kernel_size=5, padding=2),
            nn.GroupNorm(1, encoder_hidden),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            encoder_hidden, encoder_hidden // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )

        self.gru_out = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
        )

        # ---- Path 1: Attention pool → global features ----
        self.pool_key = nn.Linear(encoder_hidden, encoder_hidden)
        self.pool_query = nn.Parameter(torch.randn(1, 1, encoder_hidden) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
        )

        # ---- Path 2: Stride pool → keyframe features ----
        kf_hidden = encoder_hidden // 2
        self.keyframe_proj = nn.Sequential(
            nn.Linear(encoder_hidden, kf_hidden),
            nn.LayerNorm(kf_hidden),
            nn.GELU(),
        )

        # Per-op keyframe intensity heads (from keyframe-level features)
        self.keyframe_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(kf_hidden, kf_hidden // 2),
                nn.GELU(),
                nn.Linear(kf_hidden // 2, 1),
            )
            for _ in range(n_ops)
        ])

        # ---- Global heads: type params + gates ----
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_hidden, param_dim * 2),
                nn.GELU(),
                nn.Linear(param_dim * 2, param_dim),
            )
            for _ in range(n_ops)
        ])

        self.gate_head = nn.Sequential(
            nn.Linear(encoder_hidden, n_ops * 2),
            nn.GELU(),
            nn.Linear(n_ops * 2, n_ops),
        )

        # ---- Operations with frequency-biased initialization ----
        freq_biases = self._assign_freq_biases(n_ops)
        self.operations = nn.ModuleList([
            KeyframeSpacetimeOperation(
                param_dim=param_dim, z_dim=z_dim,
                hidden=64, n_basis=n_basis,
                freq_bias=freq_biases[k],
            )
            for k in range(n_ops)
        ])

    @staticmethod
    def _assign_freq_biases(n_ops):
        """Assign frequency biases to break op symmetry."""
        biases = []
        patterns = ['low', 'low', 'mid', 'mid', 'high', 'high',
                     'broadband', 'broadband']
        for k in range(n_ops):
            biases.append(patterns[k % len(patterns)])
        return biases

    def _encode_shared(self, residual, mask=None):
        """Shared encoder: residual → frame-level features."""
        h = self.input_proj(residual)
        h = h.permute(0, 2, 1)
        h = self.temporal_conv(h)
        h = h.permute(0, 2, 1)
        h, _ = self.gru(h)
        h = self.gru_out(h)  # [B, T, H]
        return h

    def _pool_global(self, h, mask=None):
        """Path 1: attention pool → global feature vector."""
        keys = self.pool_key(h)
        H = self.encoder_hidden
        attn_logits = (self.pool_query * keys).sum(-1) / math.sqrt(H)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn = attn_logits.softmax(dim=1)
        pooled = (attn.unsqueeze(-1) * h).sum(dim=1)
        return self.global_proj(pooled)  # [B, H]

    def _pool_keyframes(self, h, mask=None):
        """
        Path 2: stride pool → keyframe-level features.

        Uses adaptive_avg_pool1d to downsample T frames to K keyframes,
        preserving temporal structure that the global pool discards.
        """
        B, T, H = h.shape

        # Mask out padded frames before pooling
        if mask is not None:
            h = h * mask.unsqueeze(-1)

        h_t = h.permute(0, 2, 1)  # [B, H, T]
        kf_feat = TF.adaptive_avg_pool1d(h_t, self.n_keyframes)  # [B, H, K]
        kf_feat = kf_feat.permute(0, 2, 1)  # [B, K, H]

        return self.keyframe_proj(kf_feat)  # [B, K, kf_hidden]

    def encode(self, residual, mask=None):
        """
        Returns:
            global_feat: [B, H] — for type params + gates
            kf_feat: [B, K, kf_hidden] — for keyframe intensities
        """
        h = self._encode_shared(residual, mask)
        global_feat = self._pool_global(h, mask)
        kf_feat = self._pool_keyframes(h, mask)
        return global_feat, kf_feat

    def decode(self, global_feat, kf_feat, T):
        """
        Returns:
            z_recon: [B, T, z_dim]
            gates_sparse: [B, n_ops]
            all_type_params: [B, n_ops, param_dim]
            all_keyframes: [B, n_ops, K]
        """
        B = global_feat.shape[0]

        # Global gates (sparse)
        raw_gates = self.gate_head(global_feat)
        gates = TF.softplus(raw_gates)

        topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        gate_mask = torch.zeros_like(gates)
        gate_mask.scatter_(-1, topk_idx, 1.0)
        gates_sparse = gates * gate_mask

        # Per-op: type params (from global) + keyframe intensities (from kf_feat)
        all_type_params = []
        all_keyframes = []
        z_recon = torch.zeros(B, T, self.z_dim, device=global_feat.device)

        for k in range(self.n_ops):
            type_params_k = self.param_heads[k](global_feat)   # [B, param_dim]
            all_type_params.append(type_params_k)

            # Keyframe intensities from temporal features
            kf_intensity_k = self.keyframe_heads[k](kf_feat).squeeze(-1)  # [B, K]
            # Softplus to keep intensities non-negative (default ~1.0)
            kf_intensity_k = TF.softplus(kf_intensity_k)
            all_keyframes.append(kf_intensity_k)

            gate_k = gates_sparse[:, k]
            if gate_k.sum() > 1e-8:
                traj_k = self.operations[k](type_params_k, kf_intensity_k, T)
                z_recon = z_recon + gate_k.unsqueeze(1).unsqueeze(2) * traj_k

        all_type_params = torch.stack(all_type_params, dim=1)  # [B, n_ops, param_dim]
        all_keyframes = torch.stack(all_keyframes, dim=1)      # [B, n_ops, K]

        return z_recon, gates_sparse, all_type_params, all_keyframes

    def forward(self, residual, mask=None):
        T = residual.shape[1]
        global_feat, kf_feat = self.encode(residual, mask)
        return self.decode(global_feat, kf_feat, T)


# ============================================================
# Losses
# ============================================================

def diversity_loss(tree, global_feat, T, n_probe=64):
    """
    Encourage operations to produce different temporal patterns.
    Penalize high cosine similarity between pairs of op trajectories.

    Only computed on a short probe trajectory (n_probe frames) for efficiency.
    """
    B = global_feat.shape[0]
    device = global_feat.device

    # Generate probe trajectories for each op with their current params
    trajs = []
    for k in range(tree.n_ops):
        type_params_k = tree.param_heads[k](global_feat)  # [B, param_dim]
        # Use uniform keyframes for probing (no temporal variation)
        uniform_kf = torch.ones(B, tree.n_keyframes, device=device)
        traj_k = tree.operations[k](type_params_k, uniform_kf, n_probe)  # [B, n_probe, z_dim]
        # Flatten temporal + z dims for comparison
        trajs.append(traj_k.reshape(B, -1))  # [B, n_probe * z_dim]

    # Pairwise cosine similarity between ops (averaged over batch)
    total_sim = 0
    n_pairs = 0
    for i in range(tree.n_ops):
        for j in range(i + 1, tree.n_ops):
            cos = TF.cosine_similarity(trajs[i], trajs[j], dim=-1).mean()
            total_sim += cos.abs()
            n_pairs += 1

    return total_sim / max(n_pairs, 1)


# ============================================================
# Data Loading (same as spacetime_operation_tree.py)
# ============================================================

def load_phase1_codec(device):
    print(f"Loading Phase 1 codec from {PHASE1_PATH}...")
    model = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
    state = torch.load(PHASE1_PATH, weights_only=True, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device).eval()
    return model


def gather_real_latents(max_samples=2000):
    print(f"\nGathering real audio latents from {LATENT_BASE}...")
    data = []
    skipped = 0

    for pt_file in LATENT_BASE.rglob("*.pt"):
        if len(data) >= max_samples:
            break
        try:
            loaded = torch.load(pt_file, weights_only=False, map_location='cpu')
            if isinstance(loaded, dict) and 'latents' in loaded:
                z = loaded['latents']
            elif isinstance(loaded, torch.Tensor):
                z = loaded
            else:
                skipped += 1
                continue

            if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16 or z.shape[2] < 10:
                skipped += 1
                continue

            if z.shape[2] > MAX_FRAMES:
                z = z[:, :, :MAX_FRAMES]

            data.append({'z_real': z, 'path': str(pt_file)})

            if len(data) % 200 == 0:
                print(f"    Loaded {len(data)} samples...")
        except Exception:
            skipped += 1
            continue

    print(f"  Loaded {len(data)} latents (skipped {skipped})")
    return data


def compute_sms_and_residuals(codec, data, device):
    print("\nComputing SMS approximations and residuals...")
    results = []
    cos_sims = []

    codec.eval()
    with torch.no_grad():
        for i, sample in enumerate(data):
            z_real = sample['z_real'].unsqueeze(0).to(device)
            z_flat = codec.z_to_flat(z_real)

            sms_pred = codec.forward_F(z_flat)
            z_sms = codec.forward_G(sms_pred)
            residual = z_flat - z_sms

            cos = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
            cos_sims.append(cos)

            results.append({
                'z_flat': z_flat.squeeze(0).cpu(),
                'z_sms': z_sms.squeeze(0).cpu(),
                'residual': residual.squeeze(0).cpu(),
            })

            if i % 200 == 0:
                print(f"    {i}/{len(data)}  cos_sim={cos:.4f}")

    print(f"  Mean SMS cos_sim: {np.mean(cos_sims):.4f}")
    return results


def pad_and_batch(samples, key, device):
    tensors = [s[key] for s in samples]
    T_max = max(t.shape[0] for t in tensors)
    D = tensors[0].shape[1]
    B = len(tensors)

    padded = torch.zeros(B, T_max, D)
    mask = torch.zeros(B, T_max)
    for i, t in enumerate(tensors):
        T = t.shape[0]
        padded[i, :T] = t
        mask[i, :T] = 1.0

    return padded.to(device), mask.to(device)


# ============================================================
# Training
# ============================================================

def train_keyframe_tree(train_data, device, n_ops=8, param_dim=16, top_k=4,
                         n_keyframes=8, n_basis=32, epochs=300, batch_size=32,
                         lr=3e-4, diversity_weight=0.01):
    """
    Train keyframe spacetime tree.

    Key additions over pure spacetime:
      - Dual encoder paths (global + keyframe)
      - Diversity loss to prevent op collapse
      - Frequency-biased op initialization for symmetry breaking
      - No top-k warmup needed (diversity loss + freq bias handle specialization)
    """
    print("\n" + "=" * 60)
    print("TRAINING KEYFRAME SPACETIME TREE")
    print("=" * 60)

    all_res_frames = torch.cat([d['residual'] for d in train_data], dim=0)
    res_mean = all_res_frames.mean(dim=0, keepdim=True).to(device)
    res_std = all_res_frames.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)
    del all_res_frames

    frame_counts = [d['residual'].shape[0] for d in train_data]
    mean_T = np.mean(frame_counts)

    # Compute expected compression
    params_per_op = param_dim + n_keyframes  # type_params + keyframe_intensities
    spacetime_vals = top_k * (params_per_op + 1)  # +1 for gate
    perframe_vals = top_k * (param_dim + 1) * mean_T

    print(f"  Samples: {len(train_data)}")
    print(f"  Mean length: {mean_T:.0f} frames ({mean_T/FPS:.1f}s)")
    print(f"  Operations: {n_ops} (param_dim={param_dim}, keyframes={n_keyframes})")
    print(f"  Top-k: {top_k}")
    print(f"  Expected params/sample: {spacetime_vals:.0f} vs {perframe_vals:.0f} (per-frame)")
    print(f"  Expected compression: {perframe_vals/spacetime_vals:.0f}x")
    print(f"  Diversity loss weight: {diversity_weight}")

    model = KeyframeSpacetimeCodec(
        z_dim=Z_DIM, n_ops=n_ops, param_dim=param_dim,
        n_keyframes=n_keyframes, encoder_hidden=256,
        top_k=top_k, n_basis=n_basis,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    # Log op frequency biases
    for k in range(n_ops):
        freqs = model.operations[k].freqs
        print(f"    Op {k}: freq range [{freqs.min():.2f}, {freqs.max():.2f}] Hz "
              f"({model._assign_freq_biases(n_ops)[k]})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_data))

        total_recon = 0
        total_div = 0
        total_samples = 0

        pbar = tqdm(range(0, len(train_data), batch_size),
                    desc=f"Epoch {epoch:3d}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for b_start in pbar:
            batch_idx = perm[b_start:b_start + batch_size]
            batch_samples = [train_data[i] for i in batch_idx]

            residual_batch, mask = pad_and_batch(batch_samples, 'residual', device)
            residual_norm = (residual_batch - res_mean) / res_std

            optimizer.zero_grad(set_to_none=True)

            recon_norm, gates, all_type_params, all_keyframes = model(residual_norm, mask)

            # Reconstruction loss (masked)
            sq_err = (recon_norm - residual_norm).pow(2).mean(dim=-1)
            loss_recon = (sq_err * mask).sum() / mask.sum()

            # Diversity loss (encourage different ops)
            global_feat, _ = model.encode(residual_norm, mask)
            loss_div = diversity_loss(model, global_feat.detach(), T=32)

            loss = loss_recon + diversity_weight * loss_div

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += loss_recon.item() * len(batch_idx)
            total_div += loss_div.item() * len(batch_idx)
            total_samples += len(batch_idx)

            pbar.set_postfix_str(
                f"recon={total_recon/total_samples:.4f} "
                f"div={total_div/total_samples:.3f}"
            )

        pbar.close()
        scheduler.step()

        if epoch % 10 == 0 or epoch == epochs - 1:
            avg_r = total_recon / total_samples
            avg_d = total_div / total_samples
            print(f"  Epoch {epoch:3d}: recon={avg_r:.6f}  div={avg_d:.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return model, res_mean, res_std


# ============================================================
# Testing & Analysis
# ============================================================

def test_keyframe_tree(codec, tree, test_data, res_mean, res_std, device, n_test=50):
    print("\n" + "=" * 60)
    print("TESTING KEYFRAME SPACETIME TREE")
    print("=" * 60)

    codec.eval()
    tree.eval()

    sms_only_cos = []
    tree_cos = []
    tree_mse = []
    param_counts = []

    with torch.no_grad():
        for i, sample in enumerate(test_data[:n_test]):
            z_flat = sample['z_flat'].unsqueeze(0).to(device)
            z_sms = sample['z_sms'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)
            T = z_flat.shape[1]

            residual_norm = (residual - res_mean) / res_std
            mask = torch.ones(1, T, device=device)

            recon_norm, gates, all_type_params, all_keyframes = tree(residual_norm, mask)
            recon = recon_norm * res_std + res_mean
            z_full = z_sms + recon

            cos_sms = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
            cos_full = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
            mse_full = TF.mse_loss(z_full, z_flat).item()
            n_active = (gates > 0.01).float().sum(dim=-1).mean().item()

            sms_only_cos.append(cos_sms)
            tree_cos.append(cos_full)
            tree_mse.append(mse_full)

            # Param count: type_params + keyframes + gate per active op
            st_vals = int(n_active) * (tree.param_dim + tree.n_keyframes + 1)
            pf_vals = int(n_active) * (tree.param_dim + 1) * T
            param_counts.append((st_vals, pf_vals, T))

            if i < 20:
                # Show keyframe intensity range for most active op
                most_active_op = gates.squeeze(0).argmax().item()
                kf = all_keyframes[0, most_active_op]
                kf_str = f"kf=[{kf.min():.1f}-{kf.max():.1f}]"

                print(f"  Sample {i:2d}: SMS={cos_sms:.4f}  "
                      f"SMS+tree={cos_full:.4f}  "
                      f"gain={cos_full-cos_sms:+.4f}  "
                      f"active={n_active:.0f}  T={T}  "
                      f"params={st_vals} vs {pf_vals}  {kf_str}")

    print(f"\n  Average SMS-only cos_sim:     {np.mean(sms_only_cos):.4f}")
    print(f"  Average SMS+tree cos_sim:     {np.mean(tree_cos):.4f}")
    print(f"  Average improvement:          {np.mean(tree_cos)-np.mean(sms_only_cos):+.4f}")
    print(f"  Average SMS+tree MSE:         {np.mean(tree_mse):.6f}")

    st_vals = [p[0] for p in param_counts]
    pf_vals = [p[1] for p in param_counts]
    print(f"\n  Compression:")
    print(f"    Keyframe params/sample:   {np.mean(st_vals):.0f}")
    print(f"    Per-frame params/sample:  {np.mean(pf_vals):.0f}")
    print(f"    Compression ratio:        {np.mean(pf_vals)/np.mean(st_vals):.1f}x")

    return {
        'sms_cos': np.mean(sms_only_cos),
        'tree_cos': np.mean(tree_cos),
        'tree_mse': np.mean(tree_mse),
    }


def analyze_operations(tree, test_data, res_mean, res_std, device):
    """Analyze temporal patterns and keyframe usage per operation."""
    print("\n" + "=" * 60)
    print("KEYFRAME OPERATION ANALYSIS")
    print("=" * 60)

    tree.eval()

    op_gates_all = [[] for _ in range(tree.n_ops)]
    op_keyframes_all = [[] for _ in range(tree.n_ops)]
    op_type_params_all = [[] for _ in range(tree.n_ops)]

    with torch.no_grad():
        for sample in test_data[:200]:
            residual = sample['residual'].unsqueeze(0).to(device)
            T = residual.shape[1]
            residual_norm = (residual - res_mean) / res_std
            mask = torch.ones(1, T, device=device)

            global_feat, kf_feat = tree.encode(residual_norm, mask)

            raw_gates = tree.gate_head(global_feat)
            gates = TF.softplus(raw_gates)

            for k in range(tree.n_ops):
                op_gates_all[k].append(gates[0, k].item())
                type_params_k = tree.param_heads[k](global_feat)
                op_type_params_all[k].append(type_params_k.squeeze(0).cpu())

                kf_intensity_k = tree.keyframe_heads[k](kf_feat).squeeze(-1)
                kf_intensity_k = TF.softplus(kf_intensity_k)
                op_keyframes_all[k].append(kf_intensity_k.squeeze(0).cpu())

    T_analysis = 128
    freq_biases = tree._assign_freq_biases(tree.n_ops)

    print(f"\n  {'Op':>3s}  {'Bias':>10s}  {'MeanGate':>8s}  {'%Active':>8s}  "
          f"{'DomFreq':>8s}  {'KF Range':>12s}  {'KF Std':>7s}  Classification")

    for k in range(tree.n_ops):
        all_gates = np.array(op_gates_all[k])
        mean_gate = all_gates.mean()
        pct_active = (all_gates > 0.01).mean() * 100

        # Keyframe statistics
        kf_stack = torch.stack(op_keyframes_all[k])  # [N, K]
        kf_mean = kf_stack.mean(dim=0).numpy()  # [K] — average temporal profile
        kf_std = kf_stack.std(dim=0).mean().item()
        kf_range = f"[{kf_mean.min():.1f}-{kf_mean.max():.1f}]"

        # Temporal profile: is this op biased toward onset, sustain, or end?
        K = len(kf_mean)
        onset_energy = kf_mean[:K//3].mean()
        sustain_energy = kf_mean[K//3:2*K//3].mean()
        release_energy = kf_mean[2*K//3:].mean()

        # Generate trajectory for frequency analysis
        mean_params = torch.stack(op_type_params_all[k]).mean(dim=0).unsqueeze(0).to(device)
        uniform_kf = torch.ones(1, tree.n_keyframes, device=device)

        with torch.no_grad():
            traj = tree.operations[k](mean_params, uniform_kf, T_analysis)
            traj = traj.squeeze(0).cpu().numpy()

        traj_c = traj - traj.mean(axis=0, keepdims=True)
        fft_mag = np.abs(np.fft.rfft(traj_c, axis=0))
        freq_axis = np.fft.rfftfreq(T_analysis, d=1.0/FPS)
        total_fft = fft_mag.sum(axis=1)

        if total_fft.sum() > 1e-8:
            dom_freq = (freq_axis * total_fft).sum() / total_fft.sum()
        else:
            dom_freq = 0.0

        # Classify
        dc_energy = total_fft[0]
        ac_energy = total_fft[1:].sum()
        total_e = dc_energy + ac_energy + 1e-8

        # Temporal role from keyframe profile
        if onset_energy > sustain_energy * 1.5 and onset_energy > release_energy * 1.5:
            temporal_role = "ONSET"
        elif release_energy > sustain_energy * 1.5:
            temporal_role = "RELEASE"
        elif sustain_energy > onset_energy * 1.3:
            temporal_role = "SUSTAIN"
        else:
            temporal_role = "UNIFORM"

        # Frequency classification
        if dc_energy / total_e > 0.8:
            freq_class = "static"
        elif ac_energy > 0:
            peak_freq_idx = total_fft[1:].argmax() + 1
            peak_freq = freq_axis[peak_freq_idx]
            if peak_freq < 0.5:
                freq_class = f"slow-env({peak_freq:.1f}Hz)"
            elif peak_freq < 2.0:
                freq_class = f"modulation({peak_freq:.1f}Hz)"
            elif peak_freq < 7.0:
                freq_class = f"vibrato({peak_freq:.1f}Hz)"
            else:
                freq_class = f"fast({peak_freq:.1f}Hz)"
        else:
            freq_class = "near-zero"

        classification = f"{temporal_role} {freq_class}"

        print(f"  {k:3d}  {freq_biases[k]:>10s}  {mean_gate:8.3f}  {pct_active:7.1f}%  "
              f"{dom_freq:7.2f}Hz  {kf_range:>12s}  {kf_std:6.3f}  {classification}")

    # Diversity check: pairwise trajectory similarity
    print(f"\n  Op trajectory similarity (lower = more diverse):")
    trajs_flat = []
    with torch.no_grad():
        for k in range(tree.n_ops):
            mean_params = torch.stack(op_type_params_all[k]).mean(dim=0).unsqueeze(0).to(device)
            uniform_kf = torch.ones(1, tree.n_keyframes, device=device)
            traj = tree.operations[k](mean_params, uniform_kf, 64).squeeze(0).reshape(-1)
            trajs_flat.append(traj)

    for i in range(tree.n_ops):
        sims = []
        for j in range(tree.n_ops):
            if i != j:
                cos = TF.cosine_similarity(
                    trajs_flat[i].unsqueeze(0),
                    trajs_flat[j].unsqueeze(0)
                ).item()
                sims.append(cos)
        mean_sim = np.mean(sims)
        print(f"    Op {i}: mean cos_sim to others = {mean_sim:.3f}")


def compare_all(tree, test_data, res_mean, res_std, device):
    """Compare with per-frame tree and pure spacetime tree."""
    print("\n" + "=" * 60)
    print("COMPARISON: KEYFRAME vs PER-FRAME vs PURE SPACETIME")
    print("=" * 60)

    tree.eval()

    # Load per-frame tree
    pf_path = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree" / "operation_tree.pt"
    pf_cos = None
    if pf_path.exists():
        from phase2_operation_tree import OperationTreeCodec
        pf_ckpt = torch.load(pf_path, weights_only=False, map_location='cpu')
        pf_tree = OperationTreeCodec(
            z_dim=Z_DIM, n_ops=pf_ckpt['n_ops'], param_dim=pf_ckpt['param_dim'],
            encoder_hidden=256, top_k=pf_ckpt['top_k'],
        )
        pf_tree.load_state_dict(pf_ckpt['model'])
        pf_tree = pf_tree.to(device).eval()
        pf_res_mean = pf_ckpt['res_mean'].to(device)
        pf_res_std = pf_ckpt['res_std'].to(device)

    # Load pure spacetime tree
    st_path = SCRIPT_DIR.parent / "test_outputs" / "spacetime_tree" / "spacetime_tree.pt"
    st_cos = None
    if st_path.exists():
        from spacetime_operation_tree import SpacetimeCodec
        st_ckpt = torch.load(st_path, weights_only=False, map_location='cpu')
        st_tree = SpacetimeCodec(
            z_dim=Z_DIM, n_ops=st_ckpt['n_ops'], param_dim=st_ckpt['param_dim'],
            encoder_hidden=256, top_k=st_ckpt['top_k'], n_basis=st_ckpt.get('n_basis', 32),
        )
        st_tree.load_state_dict(st_ckpt['model'])
        st_tree = st_tree.to(device).eval()
        st_res_mean = st_ckpt['res_mean'].to(device)
        st_res_std = st_ckpt['res_std'].to(device)

    kf_cos_list = []
    pf_cos_list = []
    st_cos_list = []

    with torch.no_grad():
        for sample in test_data[:50]:
            z_flat = sample['z_flat'].unsqueeze(0).to(device)
            z_sms = sample['z_sms'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)
            T = z_flat.shape[1]
            mask = torch.ones(1, T, device=device)

            # Keyframe tree
            kf_res_norm = (residual - res_mean) / res_std
            kf_recon_norm, _, _, _ = tree(kf_res_norm, mask)
            kf_recon = kf_recon_norm * res_std + res_mean
            kf_full = z_sms + kf_recon
            kf_cos_list.append(TF.cosine_similarity(z_flat, kf_full, dim=-1).mean().item())

            # Per-frame tree
            if pf_path.exists():
                pf_res_norm = (residual - pf_res_mean) / pf_res_std
                pf_recon_norm, _, _ = pf_tree(pf_res_norm)
                pf_recon = pf_recon_norm * pf_res_std + pf_res_mean
                pf_full = z_sms + pf_recon
                pf_cos_list.append(TF.cosine_similarity(z_flat, pf_full, dim=-1).mean().item())

            # Pure spacetime tree
            if st_path.exists():
                st_res_norm = (residual - st_res_mean) / st_res_std
                st_recon_norm, _, _ = st_tree(st_res_norm, mask)
                st_recon = st_recon_norm * st_res_std + st_res_mean
                st_full = z_sms + st_recon
                st_cos_list.append(TF.cosine_similarity(z_flat, st_full, dim=-1).mean().item())

    mean_T = np.mean([s['residual'].shape[0] for s in test_data[:50]])
    kf_vals = tree.top_k * (tree.param_dim + tree.n_keyframes + 1)

    print(f"\n  {'Method':<25s}  {'cos_sim':>8s}  {'params/sample':>14s}  {'compression':>12s}")
    print(f"  {'-'*25}  {'-'*8}  {'-'*14}  {'-'*12}")

    if pf_cos_list:
        pf_vals = 4 * (16 + 1) * mean_T
        print(f"  {'Per-frame tree':<25s}  {np.mean(pf_cos_list):8.4f}  "
              f"{pf_vals:14.0f}  {'1x (baseline)':>12s}")

    print(f"  {'Keyframe spacetime':<25s}  {np.mean(kf_cos_list):8.4f}  "
          f"{kf_vals:14.0f}  {mean_T*(4*17)/kf_vals:11.0f}x")

    if st_cos_list:
        st_vals = 4 * (16 + 1)
        print(f"  {'Pure spacetime':<25s}  {np.mean(st_cos_list):8.4f}  "
              f"{st_vals:14.0f}  {mean_T*(4*17)/st_vals:11.0f}x")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("KEYFRAME SPACETIME TREE")
    print("=" * 60)
    print()
    print("Architecture: z_{0:T} = SMS + sum_k gate_k * op_k(type_k, keyframes_k)")
    print("  type_k:      global params (what pattern)")
    print("  keyframes_k: temporal intensity (when/how strong)")
    print("  Dual encoder: attention pool → type+gates, stride pool → keyframes")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Config
    N_OPS = 8
    PARAM_DIM = 16
    N_KEYFRAMES = 8     # ~1 keypoint per 1.5s for typical sample
    TOP_K = 4
    N_BASIS = 32
    DIVERSITY_WEIGHT = 0.01

    # Load Phase 1
    codec = load_phase1_codec(device)
    data = gather_real_latents(max_samples=2000)
    all_computed = compute_sms_and_residuals(codec, data, device)

    codec = codec.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    n_test = min(50, len(all_computed) // 10)
    train_data = all_computed[:-n_test]
    test_data = all_computed[-n_test:]
    print(f"\nTrain: {len(train_data)}, Test: {n_test}")

    # Train
    tree, res_mean, res_std = train_keyframe_tree(
        train_data, device,
        n_ops=N_OPS,
        param_dim=PARAM_DIM,
        top_k=TOP_K,
        n_keyframes=N_KEYFRAMES,
        n_basis=N_BASIS,
        epochs=300,
        batch_size=32,
        lr=3e-4,
        diversity_weight=DIVERSITY_WEIGHT,
    )

    # Save
    save_path = OUTPUT_DIR / "keyframe_tree.pt"
    torch.save({
        'model': tree.state_dict(),
        'res_mean': res_mean.cpu(),
        'res_std': res_std.cpu(),
        'n_ops': N_OPS,
        'param_dim': PARAM_DIM,
        'top_k': TOP_K,
        'n_keyframes': N_KEYFRAMES,
        'n_basis': N_BASIS,
    }, save_path)
    print(f"\nSaved to {save_path}")

    # Test
    codec = codec.to(device).eval()
    results = test_keyframe_tree(codec, tree, test_data, res_mean, res_std, device, n_test)

    # Analyze
    analyze_operations(tree, test_data, res_mean, res_std, device)

    # Compare with other trees
    compare_all(tree, test_data, res_mean, res_std, device)

    # Summary
    print("\n" + "=" * 60)
    print("KEYFRAME SPACETIME TREE RESULTS")
    print("=" * 60)
    print(f"\n  SMS-only cos_sim:     {results['sms_cos']:.4f}")
    print(f"  SMS+tree cos_sim:     {results['tree_cos']:.4f}")
    print(f"  Improvement:          {results['tree_cos']-results['sms_cos']:+.4f}")

    kf_vals = TOP_K * (PARAM_DIM + N_KEYFRAMES + 1)
    mean_T = np.mean([d['residual'].shape[0] for d in test_data])
    pf_vals = TOP_K * (PARAM_DIM + 1) * mean_T

    print(f"\n  Params/sample: {kf_vals} (keyframe) vs {pf_vals:.0f} (per-frame)")
    print(f"  Compression:   {pf_vals/kf_vals:.0f}x")
    print(f"\n  Ops: {N_OPS} (type_dim={PARAM_DIM}, keyframes={N_KEYFRAMES}, top_k={TOP_K})")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
