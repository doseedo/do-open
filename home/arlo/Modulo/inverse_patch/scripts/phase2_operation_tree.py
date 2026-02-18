#!/usr/bin/env python3
"""
Phase 2: Unified Sparse Operation Tree

Replaces the old "SMS + residual atoms" approach with a unified model:

    z_real ≈ op_sms(z) + Σ_k α_k(t) * op_k(params_k(t))

Where:
  - op_sms = Phase 1 codec G(F(z)), frozen — what SMS captures
  - op_k   = learned parameterized operation (small MLP: params → z-contribution)
  - α_k(t) = per-frame activation weight (hard top-k sparse)
  - params_k(t) = per-frame operation parameters predicted by shared encoder

Key insight from residual analysis:
  - Residuals are low-dimensional (~14 dims for 90% variance)
  - Continuous, not discrete (silhouette=0.25, weak clusters)
  - Temporally smooth (autocorr=0.56) — needs temporal modeling
  - Channel 3 dominates (65% of energy)
  - SMS captures direction (cos_sim=0.89) but misses 60% of magnitude

Each operation is a parameterized FUNCTION, not a static vector atom.
A 14-dim parameter space per operation can represent curved manifolds in z-space,
unlike atoms which are 1D lines.
"""

import sys
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
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree"

Z_DIM = 128       # 8 * 16 flattened DCAE latent
MAX_FRAMES = 256


# ============================================================
# Operation Tree Model
# ============================================================

class ParameterizedOperation(nn.Module):
    """
    One discovered operation: maps parameters → z-space contribution.

    Unlike a static atom (1D line in z-space), this MLP maps a param_dim
    parameter vector to a z_dim output, representing a curved manifold.
    """

    def __init__(self, param_dim=16, z_dim=128, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(param_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, z_dim),
        )
        # Small init so operations start near zero
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, params):
        """params: [..., param_dim] → z_contribution: [..., z_dim]"""
        return self.net(params)


class OperationTreeCodec(nn.Module):
    """
    Unified sparse operation tree for z-space.

    Encoder:  z_real → per-operation params + activations
    Decoder:  Σ_k α_k * op_k(params_k) → residual_reconstructed

    The SMS contribution (from frozen Phase 1) is computed externally.
    This model learns what SMS can't capture.
    """

    def __init__(self, z_dim=128, n_ops=8, param_dim=16,
                 encoder_hidden=256, top_k=4):
        super().__init__()
        self.z_dim = z_dim
        self.n_ops = n_ops
        self.param_dim = param_dim
        self.top_k = top_k

        # Shared temporal encoder: residual → hidden features
        self.input_proj = nn.Sequential(
            nn.Linear(z_dim, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
        )

        # Temporal convolutions for local context
        self.temporal = nn.Sequential(
            nn.Conv1d(encoder_hidden, encoder_hidden, kernel_size=7, padding=3),
            nn.GroupNorm(1, encoder_hidden),
            nn.GELU(),
            nn.Conv1d(encoder_hidden, encoder_hidden, kernel_size=5, padding=2),
            nn.GroupNorm(1, encoder_hidden),
            nn.GELU(),
        )

        # GRU for longer-range temporal dependencies
        self.gru = nn.GRU(
            encoder_hidden, encoder_hidden // 2,
            num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )

        self.encoder_out = nn.Sequential(
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
        )

        # Per-operation parameter heads: hidden → params_k
        self.param_heads = nn.ModuleList([
            nn.Linear(encoder_hidden, param_dim)
            for _ in range(n_ops)
        ])

        # Activation head: hidden → alpha [n_ops] (before top-k)
        self.activation_head = nn.Sequential(
            nn.Linear(encoder_hidden, n_ops * 2),
            nn.GELU(),
            nn.Linear(n_ops * 2, n_ops),
        )

        # Parameterized operations: params → z_contribution
        self.operations = nn.ModuleList([
            ParameterizedOperation(param_dim=param_dim, z_dim=z_dim, hidden_dim=64)
            for _ in range(n_ops)
        ])

    def encode(self, residual):
        """
        residual: [B, T, z_dim]
        Returns: hidden features [B, T, encoder_hidden]
        """
        h = self.input_proj(residual)      # [B, T, H]
        h = h.permute(0, 2, 1)             # [B, H, T]
        h = self.temporal(h)
        h = h.permute(0, 2, 1)             # [B, T, H]
        h, _ = self.gru(h)                 # [B, T, H]
        h = self.encoder_out(h)            # [B, T, H]
        return h

    def decode(self, hidden):
        """
        hidden: [B, T, encoder_hidden]
        Returns:
          z_recon: [B, T, z_dim] — reconstructed residual
          activations: [B, T, n_ops] — sparse operation weights
          all_params: [B, T, n_ops, param_dim] — operation parameters
        """
        B, T, _ = hidden.shape

        # Get activations (sparse)
        raw_alpha = self.activation_head(hidden)  # [B, T, n_ops]
        alpha = TF.softplus(raw_alpha)            # non-negative

        # Hard top-k sparsity
        topk_vals, topk_idx = torch.topk(alpha, self.top_k, dim=-1)
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, topk_idx, 1.0)
        alpha_sparse = alpha * mask  # [B, T, n_ops]

        # Get per-operation params and contributions
        all_params = []
        contributions = torch.zeros(B, T, self.z_dim, device=hidden.device)

        for k in range(self.n_ops):
            params_k = self.param_heads[k](hidden)       # [B, T, param_dim]
            contrib_k = self.operations[k](params_k)      # [B, T, z_dim]
            alpha_k = alpha_sparse[:, :, k:k+1]           # [B, T, 1]
            contributions = contributions + alpha_k * contrib_k
            all_params.append(params_k)

        all_params = torch.stack(all_params, dim=2)  # [B, T, n_ops, param_dim]

        return contributions, alpha_sparse, all_params

    def forward(self, residual):
        """
        residual: [B, T, z_dim]
        Returns: (z_recon, activations, all_params)
        """
        hidden = self.encode(residual)
        return self.decode(hidden)


# ============================================================
# Data Loading
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
    """
    Compute z_sms (frozen Phase 1) and residuals for all samples.
    Returns list of dicts with z_flat, z_sms, residual per sample.
    """
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


# ============================================================
# Training
# ============================================================

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


def train_operation_tree(train_data, device, n_ops=8, param_dim=16, top_k=4,
                         epochs=300, batch_size=32, lr=3e-4):
    """
    Train the operation tree to reconstruct residuals.
    """
    print("\n" + "=" * 60)
    print("TRAINING OPERATION TREE")
    print("=" * 60)

    # Compute normalization stats from residuals
    all_res_frames = torch.cat([d['residual'] for d in train_data], dim=0)
    res_mean = all_res_frames.mean(dim=0, keepdim=True).to(device)
    res_std = all_res_frames.std(dim=0, keepdim=True).clamp(min=1e-6).to(device)
    del all_res_frames

    print(f"  Samples: {len(train_data)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Operations: {n_ops}")
    print(f"  Param dim: {param_dim}")
    print(f"  Top-k: {top_k}")

    model = OperationTreeCodec(
        z_dim=Z_DIM, n_ops=n_ops, param_dim=param_dim,
        encoder_hidden=256, top_k=top_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_steps = (len(train_data) + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_data))

        total_recon = 0
        total_samples = 0

        pbar = tqdm(range(0, len(train_data), batch_size),
                    desc=f"Epoch {epoch:3d}", leave=False,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

        for b_start in pbar:
            batch_idx = perm[b_start:b_start + batch_size]
            batch_samples = [train_data[i] for i in batch_idx]

            # Pad into real batch
            residual_batch, mask = pad_and_batch(batch_samples, 'residual', device)
            # [B, T_max, 128], [B, T_max]

            residual_norm = (residual_batch - res_mean) / res_std

            optimizer.zero_grad(set_to_none=True)

            recon_norm, activations, _ = model(residual_norm)

            # Masked MSE: only count real (non-padded) frames
            sq_err = (recon_norm - residual_norm).pow(2).mean(dim=-1)  # [B, T]
            loss = (sq_err * mask).sum() / mask.sum()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_recon += loss.item() * len(batch_idx)
            total_samples += len(batch_idx)

            pbar.set_postfix_str(f"recon={total_recon/total_samples:.4f}")

        pbar.close()
        scheduler.step()

        avg_recon = total_recon / total_samples

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: recon={avg_recon:.6f}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return model, res_mean, res_std


# ============================================================
# Testing & Analysis
# ============================================================

def test_operation_tree(codec, tree, test_data, res_mean, res_std, device, n_test=50):
    """Test full pipeline on held-out data."""
    print("\n" + "=" * 60)
    print("TESTING OPERATION TREE")
    print("=" * 60)

    codec.eval()
    tree.eval()

    sms_only_cos = []
    tree_cos = []
    tree_mse = []

    with torch.no_grad():
        for i, sample in enumerate(test_data[:n_test]):
            z_flat = sample['z_flat'].unsqueeze(0).to(device)
            z_sms = sample['z_sms'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)

            # Normalize and reconstruct
            residual_norm = (residual - res_mean) / res_std
            recon_norm, activations, all_params = tree(residual_norm)
            recon = recon_norm * res_std + res_mean

            # Full reconstruction
            z_full = z_sms + recon

            # Metrics
            cos_sms = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
            cos_full = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
            mse_full = TF.mse_loss(z_full, z_flat).item()
            n_active = (activations > 0.01).float().sum(dim=-1).mean().item()

            sms_only_cos.append(cos_sms)
            tree_cos.append(cos_full)
            tree_mse.append(mse_full)

            if i < 20:
                print(f"  Sample {i:2d}: SMS={cos_sms:.4f}  "
                      f"SMS+tree={cos_full:.4f}  "
                      f"gain={cos_full-cos_sms:+.4f}  "
                      f"active={n_active:.1f}")

    print(f"\n  Average SMS-only cos_sim:     {np.mean(sms_only_cos):.4f}")
    print(f"  Average SMS+tree cos_sim:     {np.mean(tree_cos):.4f}")
    print(f"  Average improvement:          {np.mean(tree_cos)-np.mean(sms_only_cos):+.4f}")
    print(f"  Average SMS+tree MSE:         {np.mean(tree_mse):.6f}")

    return {
        'sms_cos': np.mean(sms_only_cos),
        'tree_cos': np.mean(tree_cos),
        'tree_mse': np.mean(tree_mse),
    }


def analyze_operations(tree, test_data, res_mean, res_std, device):
    """Analyze what each operation learned."""
    print("\n" + "=" * 60)
    print("OPERATION ANALYSIS")
    print("=" * 60)

    tree.eval()

    # Collect per-operation statistics
    op_activations = [[] for _ in range(tree.n_ops)]
    op_contributions_norm = [[] for _ in range(tree.n_ops)]

    with torch.no_grad():
        for sample in test_data[:200]:
            residual = sample['residual'].unsqueeze(0).to(device)
            residual_norm = (residual - res_mean) / res_std

            hidden = tree.encode(residual_norm)
            raw_alpha = tree.activation_head(hidden)
            alpha = TF.softplus(raw_alpha)

            for k in range(tree.n_ops):
                params_k = tree.param_heads[k](hidden)
                contrib_k = tree.operations[k](params_k)
                alpha_k = alpha[:, :, k]

                op_activations[k].append(alpha_k.squeeze(0).cpu())
                contrib_norm_k = contrib_k.norm(dim=-1).squeeze(0).cpu()
                op_contributions_norm[k].append(contrib_norm_k)

    print(f"\n  Per-operation statistics:")
    print(f"  {'Op':>3s}  {'Mean α':>8s}  {'% Active':>9s}  {'Mean ||c||':>10s}  "
          f"{'Contribution':>12s}")

    op_importance = []
    for k in range(tree.n_ops):
        all_alpha = torch.cat(op_activations[k])
        all_cnorm = torch.cat(op_contributions_norm[k])

        mean_alpha = all_alpha.mean().item()
        pct_active = (all_alpha > 0.01).float().mean().item() * 100
        mean_cnorm = all_cnorm.mean().item()
        importance = mean_alpha * mean_cnorm

        op_importance.append(importance)
        print(f"  {k:3d}  {mean_alpha:8.4f}  {pct_active:8.1f}%  "
              f"{mean_cnorm:10.4f}  {importance:12.4f}")

    # Channel analysis for top operations
    print(f"\n  Top operations — DCAE channel energy distribution:")
    ranked = np.argsort(op_importance)[::-1]

    for rank, k in enumerate(ranked[:6]):
        with torch.no_grad():
            # Sample some params and check what z-space direction the operation outputs
            sample = test_data[0]
            residual = sample['residual'].unsqueeze(0).to(device)
            residual_norm = (residual - res_mean) / res_std
            hidden = tree.encode(residual_norm)
            params_k = tree.param_heads[k](hidden)
            contrib_k = tree.operations[k](params_k)  # [1, T, 128]

            # Average contribution direction
            avg_contrib = contrib_k.squeeze(0).mean(dim=0).cpu().numpy()  # [128]
            reshaped = avg_contrib.reshape(8, 16)
            ch_energy = np.linalg.norm(reshaped, axis=1)
            ch_energy = ch_energy / (ch_energy.sum() + 1e-8)
            top_ch = np.argsort(ch_energy)[::-1][:3]
            ch_str = ", ".join([f"ch{c}={ch_energy[c]:.2f}" for c in top_ch])
            print(f"    Op {k} (rank {rank}): {ch_str}")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PHASE 2: UNIFIED OPERATION TREE")
    print("=" * 60)
    print()
    print("Architecture: z_real ≈ op_sms(z) + Σ_k α_k * op_k(params_k)")
    print("  - op_sms: frozen Phase 1 codec (what SMS captures)")
    print("  - op_k:   parameterized MLP (params → z-contribution)")
    print("  - α_k:    sparse activation weights (hard top-k)")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Config (informed by residual analysis)
    N_OPS = 8        # number of discovered operations
    PARAM_DIM = 16   # per-operation parameter dimensionality (~14 for 90% PCA)
    TOP_K = 4        # operations active per frame (best silhouette at k=4)

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
    tree, res_mean, res_std = train_operation_tree(
        train_data, device,
        n_ops=N_OPS,
        param_dim=PARAM_DIM,
        top_k=TOP_K,
        epochs=300,
        batch_size=32,
        lr=3e-4,
    )

    # Save
    save_path = OUTPUT_DIR / "operation_tree.pt"
    torch.save({
        'model': tree.state_dict(),
        'res_mean': res_mean.cpu(),
        'res_std': res_std.cpu(),
        'n_ops': N_OPS,
        'param_dim': PARAM_DIM,
        'top_k': TOP_K,
    }, save_path)
    print(f"\nSaved to {save_path}")

    # Test
    codec = codec.to(device).eval()
    results = test_operation_tree(codec, tree, test_data, res_mean, res_std, device, n_test)

    # Analyze operations
    analyze_operations(tree, test_data, res_mean, res_std, device)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS")
    print("=" * 60)
    print(f"\n  SMS-only cos_sim:     {results['sms_cos']:.4f}")
    print(f"  SMS+tree cos_sim:     {results['tree_cos']:.4f}")
    print(f"  Improvement:          {results['tree_cos']-results['sms_cos']:+.4f}")
    print(f"\n  Operations: {N_OPS} (param_dim={PARAM_DIM}, top_k={TOP_K})")
    print(f"\n  vs old static atoms:")
    print(f"    Static atom = 1D line in z-space (linear)")
    print(f"    Parameterized op = {PARAM_DIM}D manifold in z-space (nonlinear)")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
