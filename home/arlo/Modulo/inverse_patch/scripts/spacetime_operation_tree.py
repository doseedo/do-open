#!/usr/bin/env python3
"""
Spacetime Operation Tree: Trajectory-level audio decomposition

Key insight: timbral qualities (vibrato, growl, breathiness, attack shape) are
temporal patterns, not per-frame properties. This codec decomposes z-sequences
into a sparse sum of trajectory-generating operations, each parameterized by
a few global (time-constant) values.

Architecture:
    z_{0:T} = SMS_trajectory + sum_k gate_k * op_k(params_k)

Where:
  - SMS_trajectory = Phase 1 codec G(F(z)), frozen (pitch/harmonic base)
  - op_k = learned trajectory generator (few global params -> full [T, 128])
  - gate_k = global activation weight (hard top-k sparse, constant across time)
  - params_k = global operation parameters (constant across time)

Compression vs per-frame tree:
  Per-frame:  ~68T values per sample (16 params x T frames x top_k active ops + gates)
  Spacetime:  ~68 values per sample  (16 params x top_k active ops + gates)
  Ratio: T x  (e.g. 32x for a 3-second sample at 10.8fps)

Each operation uses a FiLM-conditioned temporal network with fixed Fourier bases,
enabling it to learn patterns like:
  - Vibrato: sinusoidal modulation (~5Hz), params control rate/depth/phase
  - Attack envelope: exponential decay, params control rate/amplitude
  - Breathiness: noise-like modulation, params control bandwidth/level
  - Pitch drift: slow trajectory, params control slope/curvature

The encoder sees the full residual sequence, attention-pools to a global
representation, and predicts per-op params + gates. The decoder generates
full trajectories from those few params. Sparsity (top-k on global gates)
means only a few operations explain each sample.

For generation: a diffusion model generates op params (small, constant across
time) rather than z-frames. Much lower dimensional and more structured.
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
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "spacetime_tree"

Z_DIM = 128       # 8 * 16 flattened DCAE latent
MAX_FRAMES = 256
SAMPLE_RATE = 44100
DCAE_HOP = 4083   # int(SAMPLE_RATE / 10.8)
FPS = SAMPLE_RATE / DCAE_HOP  # ~10.8 frames per second


# ============================================================
# Spacetime Operation: trajectory generator
# ============================================================

class SpacetimeOperation(nn.Module):
    """
    Generates a full z-space trajectory from time-global parameters.

    Uses fixed Fourier temporal basis + FiLM conditioning:
      1. Time coords -> Fourier features [T, n_basis+2]
      2. FiLM(params) modulates the temporal network at two stages
      3. Output: [B, T, z_dim] trajectory

    The Fourier basis spans 0.1-8 Hz (slow envelopes to near-Nyquist for
    10.8fps DCAE). The operation learns WHAT pattern to produce; the params
    control HOW to instantiate it.
    """

    def __init__(self, param_dim=16, z_dim=128, hidden=64, n_basis=32):
        super().__init__()
        self.z_dim = z_dim
        self.hidden = hidden

        # Fixed Fourier basis frequencies (Hz)
        # 0.1 Hz (10s period, phrase dynamics) to 8 Hz (near Nyquist)
        n_freq = n_basis // 2
        freqs = torch.logspace(math.log10(0.1), math.log10(8.0), n_freq)
        self.register_buffer('freqs', freqs)  # [n_freq]

        time_feat_dim = n_basis + 2  # sin/cos pairs + scaled_time + bias

        # Two-stage FiLM conditioning from params
        self.film1 = nn.Linear(param_dim, hidden * 2)
        self.film2 = nn.Linear(param_dim, hidden * 2)

        # Temporal network
        self.layer1 = nn.Linear(time_feat_dim, hidden)
        self.layer2 = nn.Linear(hidden, hidden)
        self.output = nn.Linear(hidden, z_dim)

        # Small init so operations start near zero
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    def forward(self, params, T):
        """
        params: [B, param_dim] — global params (constant across time)
        T: int — number of time frames to generate
        Returns: [B, T, z_dim] — full trajectory contribution
        """
        B = params.shape[0]
        device = params.device

        # Absolute time coordinates in seconds
        t = torch.arange(T, device=device, dtype=torch.float32) / FPS  # [T]
        t_col = t.unsqueeze(-1)  # [T, 1]

        # Fourier features at each basis frequency
        phases = 2 * math.pi * self.freqs.unsqueeze(0) * t_col  # [T, n_freq]

        time_feats = torch.cat([
            torch.sin(phases),              # [T, n_freq]
            torch.cos(phases),              # [T, n_freq]
            t_col / 10.0,                   # [T, 1] — time in units of 10s
            torch.ones_like(t_col),         # [T, 1] — bias
        ], dim=-1)  # [T, time_feat_dim]

        time_feats = time_feats.unsqueeze(0).expand(B, -1, -1)  # [B, T, feat]

        # FiLM stage 1
        gamma1, beta1 = self.film1(params).chunk(2, dim=-1)  # each [B, hidden]
        h = self.layer1(time_feats)                            # [B, T, hidden]
        h = gamma1.unsqueeze(1) * h + beta1.unsqueeze(1)
        h = TF.gelu(h)

        # FiLM stage 2
        gamma2, beta2 = self.film2(params).chunk(2, dim=-1)
        h = self.layer2(h)
        h = gamma2.unsqueeze(1) * h + beta2.unsqueeze(1)
        h = TF.gelu(h)

        return self.output(h)  # [B, T, z_dim]


# ============================================================
# Spacetime Codec: encoder + operations + gates
# ============================================================

class SpacetimeCodec(nn.Module):
    """
    Spacetime operation tree for z-space residual decomposition.

    Encoder: residual z-sequence [B, T, 128]
             -> temporal conv + biGRU -> attention pool -> global_feat [B, H]
    Decoder: global_feat -> per-op params [B, n_ops, param_dim]
                         -> sparse gates [B, n_ops]
                         -> Sigma gate_k * op_k(params_k) -> [B, T, 128]

    vs per-frame tree:
      Per-frame: encoder outputs [B, T, H], ops are per-frame MLPs
      Spacetime: encoder outputs [B, H] (global), ops generate full trajectories
    """

    def __init__(self, z_dim=128, n_ops=8, param_dim=16,
                 encoder_hidden=256, top_k=4, n_basis=32):
        super().__init__()
        self.z_dim = z_dim
        self.n_ops = n_ops
        self.param_dim = param_dim
        self.top_k = top_k
        self.encoder_hidden = encoder_hidden

        # ---- Encoder: sequence -> frame-level features ----
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

        # ---- Attention pooling: T frames -> 1 global vector ----
        self.pool_key = nn.Linear(encoder_hidden, encoder_hidden)
        self.pool_query = nn.Parameter(torch.randn(1, 1, encoder_hidden) * 0.02)

        self.global_proj = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
        )

        # ---- Per-op parameter heads ----
        self.param_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_hidden, param_dim * 2),
                nn.GELU(),
                nn.Linear(param_dim * 2, param_dim),
            )
            for _ in range(n_ops)
        ])

        # ---- Gate head: global sparse activation ----
        self.gate_head = nn.Sequential(
            nn.Linear(encoder_hidden, n_ops * 2),
            nn.GELU(),
            nn.Linear(n_ops * 2, n_ops),
        )

        # ---- Spacetime operations ----
        self.operations = nn.ModuleList([
            SpacetimeOperation(
                param_dim=param_dim, z_dim=z_dim,
                hidden=64, n_basis=n_basis,
            )
            for _ in range(n_ops)
        ])

    def encode(self, residual, mask=None):
        """
        residual: [B, T, z_dim]
        mask: [B, T] (1=real, 0=padded)
        Returns: global_feat [B, encoder_hidden]
        """
        h = self.input_proj(residual)         # [B, T, H]
        h = h.permute(0, 2, 1)               # [B, H, T]
        h = self.temporal_conv(h)
        h = h.permute(0, 2, 1)               # [B, T, H]
        h, _ = self.gru(h)                   # [B, T, H]
        h = self.gru_out(h)                  # [B, T, H]

        # Attention pooling over time
        keys = self.pool_key(h)               # [B, T, H]
        H = self.encoder_hidden
        attn_logits = (self.pool_query * keys).sum(-1) / math.sqrt(H)  # [B, T]

        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -1e9)

        attn = attn_logits.softmax(dim=1)     # [B, T]
        pooled = (attn.unsqueeze(-1) * h).sum(dim=1)  # [B, H]

        return self.global_proj(pooled)       # [B, H]

    def decode(self, global_feat, T):
        """
        global_feat: [B, encoder_hidden]
        T: int — number of time frames to generate
        Returns:
          z_recon: [B, T, z_dim]
          gates_sparse: [B, n_ops]
          all_params: [B, n_ops, param_dim]
        """
        B = global_feat.shape[0]

        # Global gates (sparse)
        raw_gates = self.gate_head(global_feat)   # [B, n_ops]
        gates = TF.softplus(raw_gates)            # non-negative

        # Hard top-k
        topk_vals, topk_idx = torch.topk(gates, self.top_k, dim=-1)
        gate_mask = torch.zeros_like(gates)
        gate_mask.scatter_(-1, topk_idx, 1.0)
        gates_sparse = gates * gate_mask          # [B, n_ops]

        # Generate trajectories for each active operation
        all_params = []
        z_recon = torch.zeros(B, T, self.z_dim, device=global_feat.device)

        for k in range(self.n_ops):
            params_k = self.param_heads[k](global_feat)   # [B, param_dim]
            all_params.append(params_k)

            gate_k = gates_sparse[:, k]                    # [B]
            if gate_k.sum() > 1e-8:
                traj_k = self.operations[k](params_k, T)   # [B, T, z_dim]
                z_recon = z_recon + gate_k.unsqueeze(1).unsqueeze(2) * traj_k

        all_params = torch.stack(all_params, dim=1)  # [B, n_ops, param_dim]
        return z_recon, gates_sparse, all_params

    def forward(self, residual, mask=None):
        """
        residual: [B, T, z_dim]
        mask: [B, T]
        Returns: (z_recon, gates, all_params)
        """
        T = residual.shape[1]
        global_feat = self.encode(residual, mask)
        return self.decode(global_feat, T)


# ============================================================
# Data Loading (same pipeline as phase2_operation_tree.py)
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
    """Compute z_sms (frozen Phase 1) and residuals for all samples."""
    print("\nComputing SMS approximations and residuals...")
    results = []
    cos_sims = []

    codec.eval()
    with torch.no_grad():
        for i, sample in enumerate(data):
            z_real = sample['z_real'].unsqueeze(0).to(device)
            z_flat = codec.z_to_flat(z_real)  # [1, T, 128]

            sms_pred = codec.forward_F(z_flat)
            z_sms = codec.forward_G(sms_pred)
            residual = z_flat - z_sms

            cos = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
            cos_sims.append(cos)

            results.append({
                'z_flat': z_flat.squeeze(0).cpu(),     # [T, 128]
                'z_sms': z_sms.squeeze(0).cpu(),       # [T, 128]
                'residual': residual.squeeze(0).cpu(),  # [T, 128]
            })

            if i % 200 == 0:
                print(f"    {i}/{len(data)}  cos_sim={cos:.4f}")

    print(f"  Mean SMS cos_sim: {np.mean(cos_sims):.4f}")
    return results


def pad_and_batch(samples, key, device):
    """Pad variable-length [T, D] tensors into a batch [B, T_max, D] + mask."""
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

def train_spacetime_tree(train_data, device, n_ops=8, param_dim=16, top_k=4,
                          n_basis=32, epochs=300, batch_size=32, lr=3e-4):
    """
    Train the spacetime operation tree to reconstruct residual trajectories.

    Key differences from per-frame tree:
      - Encoder attention-pools to global representation
      - Operations generate full trajectories from global params
      - Gates are global (per-sample, not per-frame)
      - top_k warmup: start with all ops active, anneal to target top_k
    """
    print("\n" + "=" * 60)
    print("TRAINING SPACETIME OPERATION TREE")
    print("=" * 60)

    # Normalization stats
    all_res_frames = torch.cat([d['residual'] for d in train_data], dim=0)
    res_mean = all_res_frames.mean(dim=0, keepdim=True).to(device)
    res_std = all_res_frames.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)
    del all_res_frames

    print(f"  Samples: {len(train_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Operations: {n_ops}")
    print(f"  Param dim: {param_dim}")
    print(f"  Top-k: {top_k}")
    print(f"  Fourier basis: {n_basis} ({n_basis//2} freq pairs, 0.1-8 Hz)")

    # Compute frame count stats for compression ratio reporting
    frame_counts = [d['residual'].shape[0] for d in train_data]
    mean_T = np.mean(frame_counts)
    print(f"  Mean sample length: {mean_T:.0f} frames ({mean_T/FPS:.1f}s)")
    print(f"  Expected compression vs per-frame: ~{mean_T:.0f}x")

    model = SpacetimeCodec(
        z_dim=Z_DIM, n_ops=n_ops, param_dim=param_dim,
        encoder_hidden=256, top_k=top_k, n_basis=n_basis,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    # Top-k warmup: start with all ops active, anneal to target
    # This prevents dead operations that never get trained
    warmup_epochs = min(50, epochs // 4)

    for epoch in range(epochs):
        model.train()

        # Scheduled top-k: n_ops -> top_k over warmup_epochs
        if epoch < warmup_epochs:
            progress = epoch / warmup_epochs
            current_top_k = n_ops - int(progress * (n_ops - top_k))
            current_top_k = max(top_k, current_top_k)
        else:
            current_top_k = top_k
        model.top_k = current_top_k

        perm = np.random.permutation(len(train_data))
        total_recon = 0
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

            recon_norm, gates, all_params = model(residual_norm, mask)

            # Masked MSE loss
            sq_err = (recon_norm - residual_norm).pow(2).mean(dim=-1)  # [B, T]
            loss = (sq_err * mask).sum() / mask.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

            pbar.set_postfix_str(f"recon={total_recon/total_samples:.4f} k={current_top_k}")

        pbar.close()
        scheduler.step()

        avg_recon = total_recon / total_samples

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: recon={avg_recon:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}  top_k={current_top_k}")

    # Restore target top_k
    model.top_k = top_k
    return model, res_mean, res_std


# ============================================================
# Testing & Analysis
# ============================================================

def test_spacetime_tree(codec, tree, test_data, res_mean, res_std, device, n_test=50):
    """Test full pipeline on held-out data."""
    print("\n" + "=" * 60)
    print("TESTING SPACETIME TREE")
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

            # Normalize and reconstruct
            residual_norm = (residual - res_mean) / res_std
            mask = torch.ones(1, T, device=device)

            recon_norm, gates, all_params = tree(residual_norm, mask)
            recon = recon_norm * res_std + res_mean

            z_full = z_sms + recon

            # Metrics
            cos_sms = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
            cos_full = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
            mse_full = TF.mse_loss(z_full, z_flat).item()
            n_active = (gates > 0.01).float().sum(dim=-1).mean().item()

            sms_only_cos.append(cos_sms)
            tree_cos.append(cos_full)
            tree_mse.append(mse_full)

            # Count values needed to represent this sample
            # Per-frame tree: top_k * (param_dim + 1) * T
            # Spacetime tree: top_k * (param_dim + 1)
            spacetime_vals = int(n_active) * (tree.param_dim + 1)
            perframe_vals = int(n_active) * (tree.param_dim + 1) * T
            param_counts.append((spacetime_vals, perframe_vals, T))

            if i < 20:
                print(f"  Sample {i:2d}: SMS={cos_sms:.4f}  "
                      f"SMS+tree={cos_full:.4f}  "
                      f"gain={cos_full-cos_sms:+.4f}  "
                      f"active={n_active:.0f}  "
                      f"T={T}  "
                      f"params={spacetime_vals} vs {perframe_vals} (per-frame)")

    print(f"\n  Average SMS-only cos_sim:     {np.mean(sms_only_cos):.4f}")
    print(f"  Average SMS+tree cos_sim:     {np.mean(tree_cos):.4f}")
    print(f"  Average improvement:          {np.mean(tree_cos)-np.mean(sms_only_cos):+.4f}")
    print(f"  Average SMS+tree MSE:         {np.mean(tree_mse):.6f}")

    # Compression stats
    st_vals = [p[0] for p in param_counts]
    pf_vals = [p[1] for p in param_counts]
    print(f"\n  Compression:")
    print(f"    Spacetime params/sample:  {np.mean(st_vals):.0f}")
    print(f"    Per-frame params/sample:  {np.mean(pf_vals):.0f}")
    print(f"    Compression ratio:        {np.mean(pf_vals)/np.mean(st_vals):.1f}x")

    return {
        'sms_cos': np.mean(sms_only_cos),
        'tree_cos': np.mean(tree_cos),
        'tree_mse': np.mean(tree_mse),
    }


def analyze_operations(tree, test_data, res_mean, res_std, device):
    """
    Analyze what temporal patterns each operation learned.

    For each op:
      - Generate trajectory from mean params across test set
      - FFT to show modulation spectrum (what frequencies the op produces)
      - Channel energy distribution (which z-dims the op affects)
      - Temporal autocorrelation (smooth envelope vs fast modulation)
    """
    print("\n" + "=" * 60)
    print("SPACETIME OPERATION ANALYSIS")
    print("=" * 60)

    tree.eval()

    # Collect per-op global params and gates across test samples
    op_params_all = [[] for _ in range(tree.n_ops)]
    op_gates_all = [[] for _ in range(tree.n_ops)]

    with torch.no_grad():
        for sample in test_data[:200]:
            residual = sample['residual'].unsqueeze(0).to(device)
            T = residual.shape[1]
            residual_norm = (residual - res_mean) / res_std
            mask = torch.ones(1, T, device=device)

            global_feat = tree.encode(residual_norm, mask)
            raw_gates = tree.gate_head(global_feat)
            gates = TF.softplus(raw_gates)

            for k in range(tree.n_ops):
                params_k = tree.param_heads[k](global_feat)
                op_params_all[k].append(params_k.squeeze(0).cpu())
                op_gates_all[k].append(gates[0, k].item())

    # Per-operation analysis
    print(f"\n  {'Op':>3s}  {'Mean Gate':>9s}  {'% Active':>9s}  "
          f"{'Dom. Freq':>10s}  {'Autocorr':>9s}  {'Top Channels':>20s}")

    op_importance = []
    T_analysis = 128  # frames for trajectory analysis (~12s at 10.8fps)

    for k in range(tree.n_ops):
        all_gates = np.array(op_gates_all[k])
        mean_gate = all_gates.mean()
        pct_active = (all_gates > 0.01).mean() * 100

        # Generate trajectory from mean params
        mean_params = torch.stack(op_params_all[k]).mean(dim=0).unsqueeze(0).to(device)

        with torch.no_grad():
            traj = tree.operations[k](mean_params, T_analysis)  # [1, T, 128]
            traj = traj.squeeze(0).cpu().numpy()  # [T, 128]

        # Temporal analysis of the trajectory
        # 1. Dominant frequency per z-dim (FFT)
        traj_centered = traj - traj.mean(axis=0, keepdims=True)
        fft_mag = np.abs(np.fft.rfft(traj_centered, axis=0))  # [T//2+1, 128]
        freq_axis = np.fft.rfftfreq(T_analysis, d=1.0/FPS)     # Hz

        # Dominant frequency: weighted average across z-dims
        total_energy_per_freq = fft_mag.sum(axis=1)  # [T//2+1]
        if total_energy_per_freq.sum() > 1e-8:
            dom_freq = (freq_axis * total_energy_per_freq).sum() / total_energy_per_freq.sum()
        else:
            dom_freq = 0.0

        # 2. Temporal autocorrelation (smoothness)
        # Average lag-1 autocorrelation across z-dims
        if traj_centered.std() > 1e-8:
            autocorr = np.mean([
                np.corrcoef(traj_centered[:-1, d], traj_centered[1:, d])[0, 1]
                for d in range(128)
                if traj_centered[:, d].std() > 1e-8
            ]) if any(traj_centered[:, d].std() > 1e-8 for d in range(128)) else 0.0
        else:
            autocorr = 0.0

        # 3. Channel energy distribution
        traj_reshaped = traj.reshape(T_analysis, 8, 16)
        ch_energy = np.linalg.norm(traj_reshaped.reshape(T_analysis, 8, 16), axis=(0, 2))
        ch_energy_norm = ch_energy / (ch_energy.sum() + 1e-8)
        top_ch = np.argsort(ch_energy_norm)[::-1][:3]
        ch_str = ", ".join([f"ch{c}={ch_energy_norm[c]:.2f}" for c in top_ch])

        importance = mean_gate * np.linalg.norm(traj.mean(axis=0))
        op_importance.append(importance)

        print(f"  {k:3d}  {mean_gate:9.4f}  {pct_active:8.1f}%  "
              f"{dom_freq:9.2f}Hz  {autocorr:9.4f}  {ch_str}")

        # Detailed frequency spectrum for top ops
        if pct_active > 20:
            # Find peak frequencies
            peaks = []
            for f_idx in range(1, len(freq_axis) - 1):
                if (total_energy_per_freq[f_idx] > total_energy_per_freq[f_idx-1] and
                    total_energy_per_freq[f_idx] > total_energy_per_freq[f_idx+1] and
                    total_energy_per_freq[f_idx] > total_energy_per_freq.max() * 0.1):
                    peaks.append((freq_axis[f_idx], total_energy_per_freq[f_idx]))
            if peaks:
                peaks.sort(key=lambda x: -x[1])
                peak_str = ", ".join([f"{f:.2f}Hz" for f, _ in peaks[:3]])
                print(f"        Spectral peaks: {peak_str}")

    # Classification of operation types
    print(f"\n  Operation type classification:")
    ranked = np.argsort(op_importance)[::-1]
    for rank, k in enumerate(ranked[:6]):
        all_gates = np.array(op_gates_all[k])
        mean_gate = all_gates.mean()

        # Regenerate for classification
        mean_params = torch.stack(op_params_all[k]).mean(dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            traj = tree.operations[k](mean_params, T_analysis).squeeze(0).cpu().numpy()

        traj_c = traj - traj.mean(axis=0, keepdims=True)
        fft_mag = np.abs(np.fft.rfft(traj_c, axis=0))
        total_fft = fft_mag.sum(axis=1)
        freq_axis_local = np.fft.rfftfreq(T_analysis, d=1.0/FPS)

        # Classify based on temporal profile
        dc_energy = total_fft[0] if len(total_fft) > 0 else 0
        ac_energy = total_fft[1:].sum() if len(total_fft) > 1 else 0
        total_e = dc_energy + ac_energy + 1e-8

        if dc_energy / total_e > 0.8:
            op_type = "STATIC (offset/bias)"
        elif ac_energy > 0:
            peak_freq_idx = total_fft[1:].argmax() + 1
            peak_freq = freq_axis_local[peak_freq_idx]
            if peak_freq < 0.5:
                op_type = f"SLOW ENVELOPE ({peak_freq:.2f}Hz)"
            elif peak_freq < 2.0:
                op_type = f"MODULATION ({peak_freq:.2f}Hz)"
            elif peak_freq < 7.0:
                op_type = f"VIBRATO-LIKE ({peak_freq:.2f}Hz)"
            else:
                op_type = f"FAST MOD ({peak_freq:.2f}Hz)"
        else:
            op_type = "NEAR-ZERO"

        print(f"    Op {k} (rank {rank}): {op_type}  "
              f"[DC={dc_energy/total_e:.1%}, AC={ac_energy/total_e:.1%}]")


def compare_with_perframe(tree, test_data, res_mean, res_std, device):
    """
    Load per-frame tree results (if available) and compare metrics.
    """
    perframe_path = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree" / "operation_tree.pt"
    if not perframe_path.exists():
        print("\n  Per-frame tree not found — skipping comparison")
        return

    from phase2_operation_tree import OperationTreeCodec

    print("\n" + "=" * 60)
    print("COMPARISON: SPACETIME vs PER-FRAME TREE")
    print("=" * 60)

    # Load per-frame tree
    pf_ckpt = torch.load(perframe_path, weights_only=False, map_location='cpu')
    pf_tree = OperationTreeCodec(
        z_dim=Z_DIM, n_ops=pf_ckpt['n_ops'], param_dim=pf_ckpt['param_dim'],
        encoder_hidden=256, top_k=pf_ckpt['top_k'],
    )
    pf_tree.load_state_dict(pf_ckpt['model'])
    pf_tree = pf_tree.to(device).eval()
    pf_res_mean = pf_ckpt['res_mean'].to(device)
    pf_res_std = pf_ckpt['res_std'].to(device)

    tree.eval()

    pf_cos = []
    st_cos = []

    with torch.no_grad():
        for sample in test_data[:50]:
            z_flat = sample['z_flat'].unsqueeze(0).to(device)
            z_sms = sample['z_sms'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)
            T = z_flat.shape[1]

            # Per-frame tree
            pf_res_norm = (residual - pf_res_mean) / pf_res_std
            pf_recon_norm, _, _ = pf_tree(pf_res_norm)
            pf_recon = pf_recon_norm * pf_res_std + pf_res_mean
            pf_full = z_sms + pf_recon
            pf_cos.append(TF.cosine_similarity(z_flat, pf_full, dim=-1).mean().item())

            # Spacetime tree
            st_res_norm = (residual - res_mean) / res_std
            mask = torch.ones(1, T, device=device)
            st_recon_norm, gates, _ = tree(st_res_norm, mask)
            st_recon = st_recon_norm * res_std + res_mean
            st_full = z_sms + st_recon
            st_cos.append(TF.cosine_similarity(z_flat, st_full, dim=-1).mean().item())

    print(f"\n  Per-frame tree cos_sim:  {np.mean(pf_cos):.4f}")
    print(f"  Spacetime tree cos_sim: {np.mean(st_cos):.4f}")
    print(f"  Delta:                  {np.mean(st_cos)-np.mean(pf_cos):+.4f}")

    # Compression comparison
    mean_T = np.mean([s['residual'].shape[0] for s in test_data[:50]])
    pf_params_per_sample = pf_ckpt['top_k'] * (pf_ckpt['param_dim'] + 1) * mean_T
    st_params_per_sample = tree.top_k * (tree.param_dim + 1)
    print(f"\n  Compression:")
    print(f"    Per-frame: {pf_params_per_sample:.0f} values/sample")
    print(f"    Spacetime: {st_params_per_sample:.0f} values/sample")
    print(f"    Ratio:     {pf_params_per_sample/st_params_per_sample:.0f}x")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("SPACETIME OPERATION TREE")
    print("=" * 60)
    print()
    print("Architecture: z_{0:T} = SMS_traj + sum_k gate_k * op_k(params_k)")
    print("  - SMS_traj: frozen Phase 1 codec (pitch/harmonic base)")
    print("  - op_k: trajectory generator (global params -> [T, 128] trajectory)")
    print("  - gate_k: global sparse activation (top-k, constant across time)")
    print("  - params_k: global operation parameters (constant across time)")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Config
    N_OPS = 8          # number of trajectory operations
    PARAM_DIM = 16     # per-operation global parameter dimensionality
    TOP_K = 4          # operations active per sample
    N_BASIS = 32       # Fourier basis size (16 sin/cos pairs)

    # Load Phase 1 codec (frozen)
    codec = load_phase1_codec(device)

    # Load real audio latents
    data = gather_real_latents(max_samples=2000)

    # Compute SMS + residuals using frozen Phase 1
    all_computed = compute_sms_and_residuals(codec, data, device)

    # Free Phase 1 from GPU
    codec = codec.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Split
    n_test = min(50, len(all_computed) // 10)
    train_data = all_computed[:-n_test]
    test_data = all_computed[-n_test:]
    print(f"\nTrain: {len(train_data)}, Test: {n_test}")

    # Train
    tree, res_mean, res_std = train_spacetime_tree(
        train_data, device,
        n_ops=N_OPS,
        param_dim=PARAM_DIM,
        top_k=TOP_K,
        n_basis=N_BASIS,
        epochs=300,
        batch_size=32,
        lr=3e-4,
    )

    # Save
    save_path = OUTPUT_DIR / "spacetime_tree.pt"
    torch.save({
        'model': tree.state_dict(),
        'res_mean': res_mean.cpu(),
        'res_std': res_std.cpu(),
        'n_ops': N_OPS,
        'param_dim': PARAM_DIM,
        'top_k': TOP_K,
        'n_basis': N_BASIS,
    }, save_path)
    print(f"\nSaved to {save_path}")

    # Test
    codec = codec.to(device).eval()
    results = test_spacetime_tree(codec, tree, test_data, res_mean, res_std, device, n_test)

    # Analyze temporal patterns
    analyze_operations(tree, test_data, res_mean, res_std, device)

    # Compare with per-frame tree (if available)
    compare_with_perframe(tree, test_data, res_mean, res_std, device)

    # Summary
    print("\n" + "=" * 60)
    print("SPACETIME TREE RESULTS")
    print("=" * 60)
    print(f"\n  SMS-only cos_sim:     {results['sms_cos']:.4f}")
    print(f"  SMS+tree cos_sim:     {results['tree_cos']:.4f}")
    print(f"  Improvement:          {results['tree_cos']-results['sms_cos']:+.4f}")
    print(f"\n  Operations: {N_OPS} (param_dim={PARAM_DIM}, top_k={TOP_K})")
    print(f"  Fourier basis: {N_BASIS} ({N_BASIS//2} frequency pairs, 0.1-8 Hz)")

    mean_T = np.mean([d['residual'].shape[0] for d in test_data])
    print(f"\n  vs per-frame tree:")
    print(f"    Per-frame tree: {TOP_K}ops x ({PARAM_DIM}params + 1gate) x {mean_T:.0f}frames"
          f" = {TOP_K*(PARAM_DIM+1)*mean_T:.0f} values/sample")
    print(f"    Spacetime tree: {TOP_K}ops x ({PARAM_DIM}params + 1gate)"
          f" = {TOP_K*(PARAM_DIM+1)} values/sample")
    print(f"    Compression:    {mean_T:.0f}x")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
