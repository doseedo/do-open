#!/usr/bin/env python3
"""
Phase 2: Residual Dictionary Learning

Uses trained Phase 1 SMS↔z codec to:
1. Compute z_sms_approx = G(F(z_real)) for real audio
2. Extract residual = z_real - z_sms_approx
3. Learn a sparse dictionary of atoms to reconstruct the residual
4. Discover what SMS can't capture

The learned atoms ARE the missing operations — discovered, not prescribed.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import orjson
import os
import gc

sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# Phase 1 codec (import the model class)
# ============================================================

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM

PHASE1_MODEL_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
SMS_MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "phase2_residual_dictionary"

Z_FLAT_DIM = 128  # 8 * 16
MAX_FRAMES = 256  # cap frame length for memory


# ============================================================
# Sparse Dictionary
# ============================================================

class SparseDictionary(nn.Module):
    """
    Sparse dictionary for residual reconstruction with hard top-k sparsity.

    atoms: [K, z_dim] — learned basis vectors in z-space
    encoder: residual_frame [z_dim] → activations [K] (hard top-k sparse)

    Reconstruction: activations @ atoms → residual_approx
    """

    def __init__(self, z_dim=128, n_atoms=128, top_k=8, encoder_hidden=256):
        super().__init__()
        self.z_dim = z_dim
        self.n_atoms = n_atoms
        self.top_k = top_k

        # Dictionary atoms — each is a direction in z-space
        self.atoms = nn.Parameter(torch.randn(n_atoms, z_dim) * 0.01)

        # Encoder: maps residual frame → atom scores
        self.encoder = nn.Sequential(
            nn.Linear(z_dim, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, encoder_hidden),
            nn.LayerNorm(encoder_hidden),
            nn.GELU(),
            nn.Linear(encoder_hidden, n_atoms),
        )

    def encode(self, residual):
        """
        residual: [B, z_dim] or [B, T, z_dim]
        Returns: activations [same shape prefix, K] (non-negative, exactly top_k active)
        """
        raw = self.encoder(residual)
        acts = TF.relu(raw)

        # Hard top-k: zero out everything except top_k activations
        topk_vals, topk_idx = torch.topk(acts, self.top_k, dim=-1)
        mask = torch.zeros_like(acts)
        mask.scatter_(-1, topk_idx, 1.0)

        # Straight-through: mask in forward, gradients flow through acts
        sparse_acts = acts * mask
        return sparse_acts

    def decode(self, activations):
        """
        activations: [B, K] or [B, T, K]
        Returns: residual_reconstructed [B, z_dim] or [B, T, z_dim]
        """
        return activations @ self.atoms

    def forward(self, residual):
        """Full forward: encode → decode."""
        activations = self.encode(residual)
        reconstructed = self.decode(activations)
        return reconstructed, activations


# ============================================================
# Data Loading
# ============================================================

def load_phase1_codec(device):
    """Load trained Phase 1 bidirectional codec."""
    print(f"Loading Phase 1 codec from {PHASE1_MODEL_PATH}...")

    model = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_FLAT_DIM, g_hidden=384, f_hidden=256)
    state = torch.load(PHASE1_MODEL_PATH, weights_only=True, map_location='cpu')
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    g_params = sum(p.numel() for p in model.G.parameters())
    f_params = sum(p.numel() for p in model.F.parameters())
    print(f"  G params: {g_params:,}  F params: {f_params:,}")
    return model


def gather_real_latents(max_samples=2000):
    """
    Gather z_real from pre-encoded DCAE latent .pt files.
    These are real audio encoded by DCAE — the ground truth z vectors.
    """
    print(f"\nGathering real audio latents from {LATENT_BASE}...")

    latent_paths = []
    for pt_file in LATENT_BASE.rglob("*.pt"):
        latent_paths.append(pt_file)
        if len(latent_paths) >= max_samples * 2:  # gather extras, filter later
            break

    print(f"  Found {len(latent_paths)} .pt files")

    data = []
    skipped = 0

    for i, path in enumerate(latent_paths):
        if len(data) >= max_samples:
            break

        try:
            loaded = torch.load(path, weights_only=False, map_location='cpu')

            if isinstance(loaded, dict) and 'latents' in loaded:
                z = loaded['latents']  # [8, 16, T]
            elif isinstance(loaded, torch.Tensor):
                z = loaded
            else:
                skipped += 1
                continue

            if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16:
                skipped += 1
                continue

            T = z.shape[2]
            if T < 10:
                skipped += 1
                continue

            # Cap frame length
            if T > MAX_FRAMES:
                z = z[:, :, :MAX_FRAMES]

            data.append({
                'z_real': z,  # [8, 16, T]
                'path': str(path),
            })

            if len(data) % 200 == 0:
                print(f"    Loaded {len(data)} samples...")

        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"    Skip {path.name}: {e}")
            continue

    print(f"  Loaded {len(data)} real latents (skipped {skipped})")
    return data


# ============================================================
# Residual Computation
# ============================================================

def compute_residuals(codec, real_data, device):
    """
    For each real audio z:
      1. Flatten: z_flat = flatten(z_real)
      2. Reverse: sms_params = F(z_flat)
      3. Forward: z_sms_approx = G(sms_params)
      4. residual = z_flat - z_sms_approx

    Returns list of residual tensors and stats.
    """
    print("\nComputing residuals: z_real - G(F(z_real))...")

    residuals = []
    cos_sims = []
    residual_norms = []

    codec.eval()
    with torch.no_grad():
        for i, sample in enumerate(real_data):
            z_real = sample['z_real'].unsqueeze(0).to(device)  # [1, 8, 16, T]
            z_flat = codec.z_to_flat(z_real)  # [1, T, 128]

            # Phase 1 roundtrip: z → SMS → z
            sms_params = codec.forward_F(z_flat)        # [1, T, 102]
            z_sms_approx = codec.forward_G(sms_params)  # [1, T, 128]

            # Residual
            residual = z_flat - z_sms_approx  # [1, T, 128]

            # Stats
            cos_sim = TF.cosine_similarity(z_flat, z_sms_approx, dim=-1).mean().item()
            res_norm = residual.norm(dim=-1).mean().item()
            z_norm = z_flat.norm(dim=-1).mean().item()

            cos_sims.append(cos_sim)
            residual_norms.append(res_norm)

            residuals.append(residual.squeeze(0).cpu())  # [T, 128]

            if i % 200 == 0:
                print(f"    {i}/{len(real_data)}  cos_sim={cos_sim:.4f}  "
                      f"residual_norm={res_norm:.4f}  z_norm={z_norm:.4f}  "
                      f"ratio={res_norm/max(z_norm,1e-8):.4f}")

    avg_cos = np.mean(cos_sims)
    avg_res_norm = np.mean(residual_norms)
    print(f"\n  Average cos_sim(z_real, z_sms_approx): {avg_cos:.4f}")
    print(f"  Average residual norm: {avg_res_norm:.4f}")
    print(f"  This means SMS captures {avg_cos*100:.1f}% of z-space direction,")
    print(f"  and the residual is what we need the dictionary to learn.")

    return residuals, {
        'avg_cos_sim': avg_cos,
        'avg_residual_norm': avg_res_norm,
        'cos_sims': cos_sims,
    }


# ============================================================
# Training
# ============================================================

def train_dictionary(residuals, device, n_atoms=128, top_k=8, epochs=300,
                     batch_size=256, lr=3e-4):
    """
    Train sparse dictionary on residual frames with hard top-k sparsity.

    Each frame is independent (per-frame sparse coding).
    Exactly top_k atoms active per frame — no soft L1 penalty needed.
    """
    print("\n" + "=" * 60)
    print("TRAINING SPARSE DICTIONARY (top-k)")
    print("=" * 60)

    # Stack all frames into one big matrix [N_total_frames, 128]
    all_frames = torch.cat(residuals, dim=0)  # [N, 128]
    n_frames = all_frames.shape[0]
    print(f"  Total residual frames: {n_frames:,}")
    print(f"  Residual dim: {all_frames.shape[1]}")
    print(f"  Atoms: {n_atoms}")
    print(f"  Top-k: {top_k} (exactly {top_k} atoms active per frame)")
    print(f"  Batch size: {batch_size}")

    # Normalize residuals (helps training stability)
    res_mean = all_frames.mean(dim=0, keepdim=True)
    res_std = all_frames.std(dim=0, keepdim=True).clamp(min=1e-6)
    all_frames_norm = (all_frames - res_mean) / res_std

    all_frames_norm = all_frames_norm.to(device)
    res_mean = res_mean.to(device)
    res_std = res_std.to(device)

    # Model
    model = SparseDictionary(z_dim=Z_FLAT_DIM, n_atoms=n_atoms, top_k=top_k).to(device)
    print(f"  Dictionary params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    n_batches = (n_frames + batch_size - 1) // batch_size

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_frames, device=device)

        total_recon = 0
        total_active = 0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, n_frames)
            idx = perm[start:end]

            batch = all_frames_norm[idx]  # [bs, 128]

            optimizer.zero_grad(set_to_none=True)

            recon, activations = model(batch)

            # Pure reconstruction loss — sparsity is structural (top-k), not penalized
            loss = TF.mse_loss(recon, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Normalize atoms to unit length (prevents scale drift)
            with torch.no_grad():
                model.atoms.data = TF.normalize(model.atoms.data, dim=-1)

            total_recon += loss.item() * (end - start)
            total_active += (activations > 0.01).float().sum(dim=-1).mean().item() * (end - start)

        scheduler.step()

        avg_recon = total_recon / n_frames
        avg_active = total_active / n_frames

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}: recon={avg_recon:.6f}  "
                  f"active_atoms={avg_active:.1f}/{n_atoms}  "
                  f"lr={scheduler.get_last_lr()[0]:.6f}")

    return model, res_mean, res_std


# ============================================================
# Analysis
# ============================================================

def analyze_dictionary(model, residuals, codec, res_mean, res_std, device):
    """
    Analyze what the learned atoms capture.
    """
    print("\n" + "=" * 60)
    print("ANALYZING LEARNED ATOMS")
    print("=" * 60)

    model.eval()

    all_frames = torch.cat(residuals, dim=0).to(device)
    all_frames_norm = (all_frames - res_mean) / res_std

    with torch.no_grad():
        recon, activations = model(all_frames_norm)

    # 1. Reconstruction quality
    recon_denorm = recon * res_std + res_mean
    recon_mse = TF.mse_loss(recon_denorm, all_frames).item()
    residual_var = all_frames.var().item()
    variance_explained = 1.0 - recon_mse / max(residual_var, 1e-8)

    print(f"\n  Reconstruction MSE: {recon_mse:.6f}")
    print(f"  Residual variance: {residual_var:.6f}")
    print(f"  Variance explained: {variance_explained*100:.1f}%")

    # 2. Atom usage statistics
    activations_np = activations.cpu().numpy()  # [N, K]
    atom_usage = (activations_np > 0.01).mean(axis=0)  # fraction of frames each atom is active
    atom_mean_activation = activations_np.mean(axis=0)

    print(f"\n  Atom usage (fraction of frames active):")
    sorted_usage = np.argsort(atom_usage)[::-1]
    for rank, idx in enumerate(sorted_usage[:20]):
        print(f"    Atom {idx:3d}: used in {atom_usage[idx]*100:.1f}% of frames, "
              f"mean activation={atom_mean_activation[idx]:.4f}")

    dead_atoms = (atom_usage < 0.01).sum()
    print(f"\n  Dead atoms (used in <1% frames): {dead_atoms}/{model.n_atoms}")
    print(f"  Active atoms: {model.n_atoms - dead_atoms}")

    # 3. Sparsity: how many atoms active per frame?
    active_per_frame = (activations_np > 0.01).sum(axis=1)
    print(f"\n  Atoms active per frame:")
    print(f"    Mean: {active_per_frame.mean():.1f}")
    print(f"    Median: {np.median(active_per_frame):.1f}")
    print(f"    Min: {active_per_frame.min()}")
    print(f"    Max: {active_per_frame.max()}")

    # 4. Atom similarity (are they diverse or redundant?)
    atoms_np = model.atoms.data.cpu().numpy()  # [K, 128]
    atom_cosines = atoms_np @ atoms_np.T  # [K, K]
    np.fill_diagonal(atom_cosines, 0)
    max_similarity = atom_cosines.max()
    mean_similarity = np.abs(atom_cosines).mean()
    print(f"\n  Atom diversity:")
    print(f"    Max pairwise cos_sim: {max_similarity:.4f}")
    print(f"    Mean |cos_sim|: {mean_similarity:.4f}")
    print(f"    (Lower = more diverse, independent atoms)")

    # 5. What does each top atom DO in z-space?
    # Each atom is a direction in z-space [128].
    # Reshape to [8, 16] to see which DCAE channels it affects.
    print(f"\n  Top 10 atoms — DCAE channel energy distribution:")
    for rank, idx in enumerate(sorted_usage[:10]):
        atom = atoms_np[idx]  # [128]
        atom_reshaped = atom.reshape(8, 16)  # [channels, latent_dim]
        channel_energy = np.linalg.norm(atom_reshaped, axis=1)  # [8]
        channel_energy = channel_energy / (channel_energy.sum() + 1e-8)
        top_channels = np.argsort(channel_energy)[::-1][:3]
        ch_str = ", ".join([f"ch{c}={channel_energy[c]:.2f}" for c in top_channels])
        print(f"    Atom {idx:3d} (rank {rank}): {ch_str}")

    return {
        'variance_explained': variance_explained,
        'atom_usage': atom_usage,
        'active_per_frame': active_per_frame,
        'dead_atoms': int(dead_atoms),
    }


def test_full_reconstruction(codec, model, real_data, res_mean, res_std, device, n_test=20):
    """
    Test full pipeline: z_real → F → SMS → G → z_sms + dictionary(residual) → z_full
    Compare z_full vs z_real.
    """
    print("\n" + "=" * 60)
    print("FULL RECONSTRUCTION TEST")
    print("=" * 60)

    codec.eval()
    model.eval()

    sms_only_cos = []
    full_recon_cos = []

    with torch.no_grad():
        for i, sample in enumerate(real_data[:n_test]):
            z_real = sample['z_real'].unsqueeze(0).to(device)  # [1, 8, 16, T]
            z_flat = codec.z_to_flat(z_real)  # [1, T, 128]

            # SMS approximation
            sms_params = codec.forward_F(z_flat)
            z_sms_approx = codec.forward_G(sms_params)

            # Residual
            residual = z_flat - z_sms_approx  # [1, T, 128]

            # Dictionary reconstruction of residual
            residual_norm = (residual - res_mean) / res_std
            residual_recon_norm, activations = model(residual_norm)
            residual_recon = residual_recon_norm * res_std + res_mean

            # Full reconstruction
            z_full = z_sms_approx + residual_recon  # [1, T, 128]

            # Metrics
            cos_sms = TF.cosine_similarity(z_flat, z_sms_approx, dim=-1).mean().item()
            cos_full = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
            n_active = (activations > 0.01).float().sum(dim=-1).mean().item()

            sms_only_cos.append(cos_sms)
            full_recon_cos.append(cos_full)

            print(f"  Sample {i:2d}: SMS_only={cos_sms:.4f}  "
                  f"SMS+dict={cos_full:.4f}  "
                  f"improvement={cos_full-cos_sms:+.4f}  "
                  f"active_atoms={n_active:.1f}")

    print(f"\n  Average SMS-only cos_sim:   {np.mean(sms_only_cos):.4f}")
    print(f"  Average SMS+dict cos_sim:   {np.mean(full_recon_cos):.4f}")
    print(f"  Average improvement:        {np.mean(full_recon_cos)-np.mean(sms_only_cos):+.4f}")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PHASE 2: RESIDUAL DICTIONARY LEARNING")
    print("=" * 60)
    print()
    print("Strategy:")
    print("  1. Apply Phase 1 codec to real audio z vectors")
    print("  2. Compute residual = z_real - G(F(z_real))")
    print("  3. Learn sparse dictionary to reconstruct residual")
    print("  4. Discover what SMS can't capture")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Phase 1 codec
    codec = load_phase1_codec(device)

    # Load real audio latents
    real_data = gather_real_latents(max_samples=2000)

    # Split
    n_test = min(50, len(real_data) // 10)
    train_data = real_data[:-n_test]
    test_data = real_data[-n_test:]
    print(f"\nTrain: {len(train_data)}, Test: {n_test}")

    # Compute residuals
    train_residuals, residual_stats = compute_residuals(codec, train_data, device)

    # Free some memory
    gc.collect()
    torch.cuda.empty_cache()

    # Train dictionary
    TOP_K = 8
    N_ATOMS = 128
    dict_model, res_mean, res_std = train_dictionary(
        train_residuals, device,
        n_atoms=N_ATOMS,
        top_k=TOP_K,
        epochs=300,
        batch_size=256,
        lr=3e-4,
    )

    # Save
    save_path = OUTPUT_DIR / "residual_dictionary.pt"
    torch.save({
        'model': dict_model.state_dict(),
        'res_mean': res_mean.cpu(),
        'res_std': res_std.cpu(),
        'n_atoms': N_ATOMS,
        'top_k': TOP_K,
        'residual_stats': residual_stats,
    }, save_path)
    print(f"\nSaved dictionary to {save_path}")

    # Analyze
    analysis = analyze_dictionary(dict_model, train_residuals, codec, res_mean, res_std, device)

    # Test full reconstruction
    test_residuals, _ = compute_residuals(codec, test_data, device)
    test_full_reconstruction(codec, dict_model, test_data, res_mean, res_std, device, n_test=n_test)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 RESULTS")
    print("=" * 60)
    print(f"\n  Variance explained by dictionary: {analysis['variance_explained']*100:.1f}%")
    print(f"  Active atoms: {N_ATOMS - analysis['dead_atoms']}/{N_ATOMS}")
    print(f"  Atoms per frame: {analysis['active_per_frame'].mean():.1f} (hard top-k={TOP_K})")
    print(f"\n  Next: listen to what individual atoms sound like")
    print(f"  (decode z_sms + alpha*atom_k for each atom k)")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
