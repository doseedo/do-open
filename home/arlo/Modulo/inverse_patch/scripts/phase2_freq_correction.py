#!/usr/bin/env python3
"""
Phase 2.5: Frequency Correction Layer + Timbral Operation Tree

Architecture:
    z_real
      → SMS (captures pitch/harmonics, ~0.87 cos_sim)
      → Frequency correction layer (fixes SMS frequency imprecision)
      → Timbral operation tree (discovers breathiness, vibrato, growl)

The GT tree analysis showed all ops were doing frequency corrections (f0s and
independent sine freqs dominated the SMS delta). This script adds a simple
frequency correction layer to absorb that signal, leaving a timbral residual
for the tree to decompose into meaningful perceptual operations.
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import gc
import os
import orjson

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from phase2_operation_tree import (
    OperationTreeCodec,
    pad_and_batch,
    train_operation_tree,
    analyze_operations,
    Z_DIM,
    MAX_FRAMES,
)

# Paths
DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_DATA_DIR = SCRIPT_DIR.parent / "data" / "sms_v4"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "phase2_freq_correction"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")

HOP_SIZE = 4096


# ============================================================
# Frequency Correction Model
# ============================================================

class FrequencyCorrector(nn.Module):
    """
    Simple per-frame MLP that corrects SMS frequency imprecision.
    Maps z_gt_sms → z_freq_corrected with residual connection.

    This is "boring plumbing" — no sparsity, no interpretability needed.
    Just absorb the frequency correction signal so the tree gets timbre.
    """
    def __init__(self, z_dim=128, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, z_dim),
        )

    def forward(self, z_sms):
        """z_sms: [B, T, 128] → z_corrected: [B, T, 128]"""
        return z_sms + self.net(z_sms)  # residual connection


# ============================================================
# Data Loading (reused from phase2_gt_tree.py)
# ============================================================

def find_start_frame(sms_path):
    try:
        sms_data = torch.load(sms_path, weights_only=True, map_location='cpu')
        amps = sms_data['amps']
        frame_energy = amps.sum(dim=1)
        for t in range(len(frame_energy)):
            if frame_energy[t] > 0.001:
                return t
        return 0
    except Exception:
        return 0


def build_paired_data():
    """Build paired (z_real, z_gt_sms) dataset."""
    print("=" * 60)
    print("BUILDING PAIRED DATASET")
    print("=" * 60)

    print(f"\nLoading Phase 1 data cache from {DATA_CACHE_PATH}...")
    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')
    print(f"  {len(cache)} cached entries")

    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']

    data = []
    skipped = 0

    for i, sample in enumerate(cache):
        sms_path = sample['path']
        z_gt_sms = sample['z_sms']
        T_cache = z_gt_sms.shape[2]

        latent_path = sms_to_latent.get(sms_path)
        if not latent_path or not os.path.exists(latent_path):
            skipped += 1
            continue

        try:
            loaded = torch.load(latent_path, weights_only=False, map_location='cpu')
            if isinstance(loaded, dict) and 'latents' in loaded:
                z_real_full = loaded['latents']
            elif isinstance(loaded, torch.Tensor):
                z_real_full = loaded
            else:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        if z_real_full.dim() != 3 or z_real_full.shape[0] != 8 or z_real_full.shape[1] != 16:
            skipped += 1
            continue

        sms_file = sms_path
        if not os.path.exists(sms_file):
            sms_file = str(SCRIPT_DIR.parent / sms_path)
        if not os.path.exists(sms_file):
            sms_file = str(SMS_DATA_DIR / Path(sms_path).name)

        start_frame = find_start_frame(sms_file) if os.path.exists(sms_file) else 0

        end_frame = start_frame + T_cache
        if end_frame > z_real_full.shape[2]:
            end_frame = z_real_full.shape[2]
            T_actual = end_frame - start_frame
            if T_actual < 10:
                skipped += 1
                continue
            z_gt_sms = z_gt_sms[:, :, :T_actual]
        else:
            T_actual = T_cache

        z_real_crop = z_real_full[:, :, start_frame:end_frame]

        z_real_flat = z_real_crop.permute(2, 0, 1).reshape(T_actual, Z_DIM)
        z_sms_flat = z_gt_sms.permute(2, 0, 1).reshape(T_actual, Z_DIM)

        data.append({
            'z_flat': z_real_flat,
            'z_sms': z_sms_flat,
            'residual': z_real_flat - z_sms_flat,
        })

        if (i + 1) % 500 == 0:
            print(f"    Paired {len(data)} samples ({skipped} skipped)...")

    print(f"\n  Total: {len(data)} paired samples ({skipped} skipped)")
    return data


# ============================================================
# Step 1: Train Frequency Corrector
# ============================================================

def train_freq_corrector(data, device, epochs=200, batch_size=64, lr=1e-3):
    """Train the frequency correction MLP."""
    print("\n" + "=" * 60)
    print("STEP 1: TRAINING FREQUENCY CORRECTOR")
    print("=" * 60)

    model = FrequencyCorrector(z_dim=Z_DIM, hidden=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")
    print(f"  Samples: {len(data)}, Epochs: {epochs}, Batch: {batch_size}")

    # Pre-stack all frames for efficient batching
    all_z_sms = torch.cat([d['z_sms'] for d in data], dim=0)     # [N_frames, 128]
    all_z_real = torch.cat([d['z_flat'] for d in data], dim=0)    # [N_frames, 128]
    N = all_z_sms.shape[0]
    print(f"  Total frames: {N:,}")

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(N)
        epoch_loss = 0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            z_sms_batch = all_z_sms[idx].unsqueeze(1).to(device)   # [B, 1, 128]
            z_real_batch = all_z_real[idx].unsqueeze(1).to(device)  # [B, 1, 128]

            z_corrected = model(z_sms_batch)
            loss = TF.mse_loss(z_corrected, z_real_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / n_batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0 or epoch == 0:
            # Compute cos_sim improvement
            model.eval()
            with torch.no_grad():
                sample_idx = torch.arange(min(5000, N))
                z_s = all_z_sms[sample_idx].unsqueeze(1).to(device)
                z_r = all_z_real[sample_idx].unsqueeze(1).to(device)
                z_c = model(z_s)

                cos_before = TF.cosine_similarity(z_r, z_s, dim=-1).mean().item()
                cos_after = TF.cosine_similarity(z_r, z_c, dim=-1).mean().item()

            print(f"  Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.6f}  "
                  f"cos_sim: {cos_before:.4f} → {cos_after:.4f}  "
                  f"(+{cos_after - cos_before:.4f})")

    model.load_state_dict(best_state)
    model = model.to(device).eval()
    print(f"\n  Best loss: {best_loss:.6f}")
    return model


# ============================================================
# Step 2: Compute Timbral Residuals
# ============================================================

def compute_timbral_residuals(data, freq_corrector, device):
    """Apply freq correction, compute timbral residual."""
    print("\n" + "=" * 60)
    print("STEP 2: COMPUTING TIMBRAL RESIDUALS")
    print("=" * 60)

    timbral_data = []
    cos_sms = []
    cos_corrected = []
    timbral_norms = []
    freq_norms = []

    freq_corrector.eval()
    with torch.no_grad():
        for i, sample in enumerate(data):
            z_real = sample['z_flat'].unsqueeze(0).to(device)   # [1, T, 128]
            z_sms = sample['z_sms'].unsqueeze(0).to(device)     # [1, T, 128]

            z_corrected = freq_corrector(z_sms)

            # Residuals
            freq_residual = z_corrected - z_sms       # what freq corrector added
            timbral_residual = z_real - z_corrected    # what's left for the tree

            cs_sms = TF.cosine_similarity(z_real, z_sms, dim=-1).mean().item()
            cs_corr = TF.cosine_similarity(z_real, z_corrected, dim=-1).mean().item()
            cos_sms.append(cs_sms)
            cos_corrected.append(cs_corr)

            fn = freq_residual.squeeze(0).norm(dim=-1).mean().item()
            tn = timbral_residual.squeeze(0).norm(dim=-1).mean().item()
            freq_norms.append(fn)
            timbral_norms.append(tn)

            timbral_data.append({
                'z_flat': sample['z_flat'],
                'z_sms': sample['z_sms'],
                'z_corrected': z_corrected.squeeze(0).cpu(),
                'residual': timbral_residual.squeeze(0).cpu(),  # for tree training
            })

    print(f"  Samples: {len(timbral_data)}")
    print(f"\n  Cos_sim progression:")
    print(f"    SMS only:       {np.mean(cos_sms):.4f}")
    print(f"    + freq corr:    {np.mean(cos_corrected):.4f}  "
          f"(+{np.mean(cos_corrected) - np.mean(cos_sms):.4f})")
    print(f"\n  Residual norms:")
    print(f"    Freq correction: {np.mean(freq_norms):.4f} (what the corrector absorbed)")
    print(f"    Timbral residual: {np.mean(timbral_norms):.4f} (what's left for the tree)")
    print(f"    Ratio: {np.mean(timbral_norms) / np.mean(freq_norms):.2f}x")

    return timbral_data


# ============================================================
# Step 3: Train Timbral Tree + Analyze
# ============================================================

def analyze_timbral_ops(tree, test_data, res_mean, res_std, device, codec=None):
    """Analyze what timbral ops affect in SMS space."""
    print("\n" + "=" * 60)
    print("STEP 4: TIMBRAL OP SMS BRIDGE ANALYSIS")
    print("=" * 60)
    print("  What do timbral ops change in SMS space?")
    print("  (Frequencies should be LOW now, amplitudes/noise should be HIGH)")

    if codec is None:
        from bidirectional_sms_z import BidirectionalCodec, SMS_DIM
        codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
        codec_path = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
        codec.load_state_dict(torch.load(codec_path, weights_only=True, map_location='cpu'))
        codec = codec.to(device).eval()

    tree.eval()
    n_ops = tree.n_ops

    # SMS dim layout
    SMS_NAMES = {}
    for d in range(6):
        SMS_NAMES[d] = f"f0_{d}"
    for g in range(6):
        for p in range(8):
            SMS_NAMES[6 + g * 8 + p] = f"g{g}_amp{p}"
    for d in range(20):
        SMS_NAMES[54 + d] = f"ifreq_{d}"
    for d in range(20):
        SMS_NAMES[74 + d] = f"iamp_{d}"
    for d in range(8):
        SMS_NAMES[94 + d] = f"noise_{d}"

    SMS_GROUPS = {
        'group_f0s': list(range(0, 6)),
        'group_amps': list(range(6, 54)),
        'indep_freqs': list(range(54, 74)),
        'indep_amps': list(range(74, 94)),
        'noise_bands': list(range(94, 102)),
    }

    # Compute per-op SMS deltas across test samples
    n_test = min(40, len(test_data))
    op_deltas = {k: [] for k in range(n_ops)}

    with torch.no_grad():
        for sample in test_data[:n_test]:
            z_corrected = sample['z_corrected'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)

            residual_norm = (residual - res_mean) / res_std
            hidden = tree.encode(residual_norm)

            # Get sparse alpha
            raw_alpha = tree.activation_head(hidden)
            alpha = TF.softplus(raw_alpha)
            topk_vals, topk_idx = torch.topk(alpha, tree.top_k, dim=-1)
            mask = torch.zeros_like(alpha)
            mask.scatter_(-1, topk_idx, 1.0)
            alpha_sparse = alpha * mask

            for k in range(n_ops):
                params_k = tree.param_heads[k](hidden)
                contrib_k = tree.operations[k](params_k)
                alpha_k = alpha_sparse[:, :, k:k+1]
                op_contrib = (alpha_k * contrib_k) * res_std  # denormalized

                z_base = z_corrected + res_mean
                z_with_op = z_base + op_contrib

                sms_base = codec.forward_F(z_base)
                sms_with = codec.forward_F(z_with_op)
                delta = (sms_with - sms_base).abs().squeeze(0).mean(dim=0).cpu().numpy()
                op_deltas[k].append(delta)

    # Analyze
    print(f"\n  {'Op':>4s}  {'group_f0s':>10s}  {'group_amps':>10s}  "
          f"{'indep_freqs':>12s}  {'indep_amps':>10s}  {'noise_bands':>11s}  "
          f"{'Primary':>16s}")

    for k in range(n_ops):
        if not op_deltas[k]:
            continue
        mean_delta = np.mean(op_deltas[k], axis=0)
        total = mean_delta.sum()
        if total < 1e-8:
            print(f"  {k:4d}  {'(inactive)':>10s}")
            continue

        group_sums = {}
        for gname, dims in SMS_GROUPS.items():
            group_sums[gname] = mean_delta[dims].sum()

        primary = max(group_sums, key=group_sums.get)
        pct = group_sums[primary] / total * 100

        print(f"  {k:4d}  {group_sums['group_f0s']:10.5f}  "
              f"{group_sums['group_amps']:10.5f}  "
              f"{group_sums['indep_freqs']:12.5f}  "
              f"{group_sums['indep_amps']:10.5f}  "
              f"{group_sums['noise_bands']:11.5f}  "
              f"{primary} ({pct:.0f}%)")

    # Compare to GT tree (which was all freq-dominated)
    print("\n  Compare with GT tree (no freq correction):")
    print("    GT tree: ALL ops → frequency dims (~45-48%)")
    print("    This tree: see above")


def test_timbral_tree(tree, test_data, res_mean, res_std, device, n_test=50):
    """Test timbral tree on held-out data."""
    print("\n" + "=" * 60)
    print("STEP 3: TESTING TIMBRAL OPERATION TREE")
    print("=" * 60)

    tree.eval()
    corr_cos = []
    tree_cos = []

    with torch.no_grad():
        for i, sample in enumerate(test_data[:n_test]):
            z_flat = sample['z_flat'].unsqueeze(0).to(device)
            z_corrected = sample['z_corrected'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)

            residual_norm = (residual - res_mean) / res_std
            recon_norm, _, _ = tree.decode(tree.encode(residual_norm))
            recon = recon_norm * res_std + res_mean
            z_full = z_corrected + recon

            cc = TF.cosine_similarity(z_flat, z_corrected, dim=-1).mean().item()
            cf = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
            corr_cos.append(cc)
            tree_cos.append(cf)

            if i < 15:
                print(f"  Sample {i:2d}: freq_corr={cc:.4f}  +tree={cf:.4f}  "
                      f"gain={cf - cc:+.4f}")

    print(f"\n  Average freq_corr cos_sim:      {np.mean(corr_cos):.4f}")
    print(f"  Average freq_corr+tree cos_sim: {np.mean(tree_cos):.4f}")
    print(f"  Average tree improvement:       {np.mean(tree_cos) - np.mean(corr_cos):+.4f}")

    return {
        'corr_cos': float(np.mean(corr_cos)),
        'tree_cos': float(np.mean(tree_cos)),
        'improvement': float(np.mean(tree_cos) - np.mean(corr_cos)),
    }


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PHASE 2.5: FREQUENCY CORRECTION + TIMBRAL TREE")
    print("=" * 60)
    print()
    print("Architecture:")
    print("  z_real → SMS → freq_correction → timbral_tree")
    print("  Each layer peels off dominant variation:")
    print("    SMS:        pitch, harmonics")
    print("    Freq corr:  frequency imprecision from SMS compression")
    print("    Tree:       timbral qualities (breathiness, vibrato, growl)")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build paired data
    data = build_paired_data()
    if len(data) < 100:
        print(f"\nERROR: Only {len(data)} samples. Need more data.")
        return

    n_test = min(50, len(data) // 10)
    train_data = data[:-n_test]
    test_data = data[-n_test:]
    print(f"\nTrain: {len(train_data)}, Test: {n_test}")

    # Step 1: Train frequency corrector
    freq_corrector = train_freq_corrector(train_data, device,
                                          epochs=200, batch_size=128, lr=1e-3)

    # Save freq corrector
    fc_path = OUTPUT_DIR / "freq_corrector.pt"
    torch.save({
        'model': freq_corrector.state_dict(),
        'hidden': 256,
    }, fc_path)
    print(f"\n  Saved freq corrector to {fc_path}")

    # Step 2: Compute timbral residuals
    all_data_with_correction = compute_timbral_residuals(data, freq_corrector, device)
    timbral_train = all_data_with_correction[:-n_test]
    timbral_test = all_data_with_correction[-n_test:]

    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Train timbral operation tree
    N_OPS = 8
    PARAM_DIM = 16
    TOP_K = 4

    tree, res_mean, res_std = train_operation_tree(
        timbral_train, device,
        n_ops=N_OPS, param_dim=PARAM_DIM, top_k=TOP_K,
        epochs=300, batch_size=32, lr=3e-4,
    )

    # Save timbral tree
    tree_path = OUTPUT_DIR / "timbral_tree.pt"
    torch.save({
        'model': tree.state_dict(),
        'res_mean': res_mean.cpu(),
        'res_std': res_std.cpu(),
        'n_ops': N_OPS,
        'param_dim': PARAM_DIM,
        'top_k': TOP_K,
        'variant': 'timbral',
    }, tree_path)
    print(f"\n  Saved timbral tree to {tree_path}")

    # Step 3b: Test
    results = test_timbral_tree(tree, timbral_test, res_mean, res_std, device, n_test)

    # Step 4: Analyze what timbral ops actually capture in SMS space
    analyze_timbral_ops(tree, timbral_test, res_mean, res_std, device)

    # Step 4b: Also analyze ops in regular tree for comparison
    analyze_operations(tree, timbral_test, res_mean, res_std, device)

    # Summary
    print("\n" + "=" * 60)
    print("FULL PIPELINE SUMMARY")
    print("=" * 60)
    print(f"\n  Layer 1 — SMS:          cos_sim ≈ 0.873")
    print(f"  Layer 2 — Freq corr:    cos_sim = {results['corr_cos']:.4f}")
    print(f"  Layer 3 — Timbral tree: cos_sim = {results['tree_cos']:.4f}")
    print(f"\n  Freq corr absorbed: +{results['corr_cos'] - 0.873:.4f}")
    print(f"  Tree improvement:   {results['improvement']:+.4f}")
    print(f"\n  If timbral ops now affect amplitudes/noise instead of frequencies,")
    print(f"  the progressive decomposition is working!")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
