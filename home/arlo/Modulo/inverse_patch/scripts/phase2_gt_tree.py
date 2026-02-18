#!/usr/bin/env python3
"""
Phase 2 GT: Operation Tree trained on Ground-Truth SMS residuals.

The original Phase 2 tree was trained on:
    residual_pred = z_real - G(F(z_real))      ← includes codec prediction error

This script trains on:
    residual_gt = z_real - z_gt_sms            ← only what SMS truly can't capture

Where z_gt_sms = DCAE_encode(additive_synth(SMS_params)), already cached
from Phase 1 data preparation.

This isolates the operation tree to model ONLY the information gap between
SMS representation and reality, without codec imperfections.
"""

import sys
import torch
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
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "phase2_gt_tree"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")

HOP_SIZE = 4096  # same as Phase 1


def find_start_frame(sms_path):
    """Reload SMS file to find the first active frame (same logic as Phase 1)."""
    try:
        sms_data = torch.load(sms_path, weights_only=True, map_location='cpu')
        amps = sms_data['amps']
        frame_energy = amps.sum(dim=1)
        for t in range(len(frame_energy)):
            if frame_energy[t] > 0.001:
                return t
        return 0  # all silent, shouldn't happen for cached samples
    except Exception:
        return 0


def build_gt_residuals():
    """
    Build GT residual dataset by pairing:
        z_real (from latent files) with z_gt_sms (from Phase 1 data cache)
    """
    print("=" * 60)
    print("BUILDING GT RESIDUAL DATASET")
    print("=" * 60)

    # 1. Load Phase 1 data cache
    print(f"\nLoading Phase 1 data cache from {DATA_CACHE_PATH}...")
    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')
    print(f"  {len(cache)} cached entries")

    # 2. Load manifest and build sms_path → latent_path map
    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']
    print(f"  {len(sms_to_latent)} manifest entries")

    # 3. For each cache entry, find latent, crop z_real, compute GT residual
    print("\nPairing z_real with z_gt_sms...")
    data = []
    skipped = 0
    cos_sims = []

    for i, sample in enumerate(cache):
        sms_path = sample['path']
        z_gt_sms = sample['z_sms']       # [8, 16, T_cache]
        T_cache = z_gt_sms.shape[2]

        # Find latent path
        latent_path = sms_to_latent.get(sms_path)
        if not latent_path or not os.path.exists(latent_path):
            skipped += 1
            continue

        # Load z_real
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

        # Find the same crop window that Phase 1 used
        # Resolve SMS file path
        sms_file = sms_path
        if not os.path.exists(sms_file):
            sms_file = str(SCRIPT_DIR.parent / sms_path)
        if not os.path.exists(sms_file):
            sms_file = str(SMS_DATA_DIR / Path(sms_path).name)

        start_frame = find_start_frame(sms_file) if os.path.exists(sms_file) else 0

        # Crop z_real to same window
        end_frame = start_frame + T_cache
        if end_frame > z_real_full.shape[2]:
            # If z_real is shorter, take what we can
            end_frame = z_real_full.shape[2]
            T_actual = end_frame - start_frame
            if T_actual < 10:
                skipped += 1
                continue
            z_gt_sms = z_gt_sms[:, :, :T_actual]
        else:
            T_actual = T_cache

        z_real_crop = z_real_full[:, :, start_frame:end_frame]

        # Flatten both to [T, 128]
        z_real_flat = z_real_crop.permute(2, 0, 1).reshape(T_actual, Z_DIM)   # [T, 128]
        z_sms_flat = z_gt_sms.permute(2, 0, 1).reshape(T_actual, Z_DIM)       # [T, 128]

        # GT residual
        residual = z_real_flat - z_sms_flat

        # Quick sanity check
        cos = TF.cosine_similarity(
            z_real_flat.unsqueeze(0), z_sms_flat.unsqueeze(0), dim=-1
        ).mean().item()
        cos_sims.append(cos)

        data.append({
            'z_flat': z_real_flat,      # [T, 128]  ground truth
            'z_sms': z_sms_flat,        # [T, 128]  GT SMS (DCAE-encoded SMS render)
            'residual': residual,        # [T, 128]  what SMS truly can't capture
        })

        if (i + 1) % 200 == 0:
            print(f"    Paired {len(data)} samples ({skipped} skipped)...")

    print(f"\n  Total: {len(data)} paired samples ({skipped} skipped)")
    if cos_sims:
        print(f"  Mean GT SMS cos_sim: {np.mean(cos_sims):.4f}")
        print(f"  (This is how well SMS represents reality — the ceiling)")
    return data


def test_gt_tree(tree, test_data, res_mean, res_std, device, n_test=50):
    """Test GT tree on held-out data."""
    print("\n" + "=" * 60)
    print("TESTING GT OPERATION TREE")
    print("=" * 60)

    tree.eval()
    sms_cos = []
    tree_cos = []
    tree_mse = []

    with torch.no_grad():
        for i, sample in enumerate(test_data[:n_test]):
            z_flat = sample['z_flat'].unsqueeze(0).to(device)
            z_sms = sample['z_sms'].unsqueeze(0).to(device)
            residual = sample['residual'].unsqueeze(0).to(device)

            residual_norm = (residual - res_mean) / res_std
            recon_norm, activations, all_params = tree(residual_norm)
            recon = recon_norm * res_std + res_mean

            z_full = z_sms + recon

            cs = TF.cosine_similarity(z_flat, z_sms, dim=-1).mean().item()
            cf = TF.cosine_similarity(z_flat, z_full, dim=-1).mean().item()
            mse = TF.mse_loss(z_full, z_flat).item()
            n_active = (activations > 0.01).float().sum(dim=-1).mean().item()

            sms_cos.append(cs)
            tree_cos.append(cf)
            tree_mse.append(mse)

            if i < 20:
                print(f"  Sample {i:2d}: GT_SMS={cs:.4f}  "
                      f"GT_SMS+tree={cf:.4f}  "
                      f"gain={cf-cs:+.4f}  "
                      f"active={n_active:.1f}")

    print(f"\n  Average GT SMS cos_sim:       {np.mean(sms_cos):.4f}")
    print(f"  Average GT SMS+tree cos_sim:  {np.mean(tree_cos):.4f}")
    print(f"  Average improvement:          {np.mean(tree_cos)-np.mean(sms_cos):+.4f}")
    print(f"  Average MSE:                  {np.mean(tree_mse):.6f}")

    return {
        'sms_cos': np.mean(sms_cos),
        'tree_cos': np.mean(tree_cos),
        'tree_mse': np.mean(tree_mse),
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("PHASE 2 GT: OPERATION TREE ON GROUND-TRUTH SMS RESIDUALS")
    print("=" * 60)
    print()
    print("Original tree:  residual = z_real - G(F(z_real))     ← codec error included")
    print("GT tree:        residual = z_real - z_gt_sms         ← pure SMS gap")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Config (same as original)
    N_OPS = 8
    PARAM_DIM = 16
    TOP_K = 4

    # Build GT residual dataset
    data = build_gt_residuals()

    if len(data) < 100:
        print(f"\nERROR: Only {len(data)} samples paired. Need more data.")
        return

    # Split
    n_test = min(50, len(data) // 10)
    train_data = data[:-n_test]
    test_data = data[-n_test:]
    print(f"\nTrain: {len(train_data)}, Test: {n_test}")

    # Compare GT residuals to pred residuals
    all_res = torch.cat([d['residual'] for d in data], dim=0)
    gt_res_norm = all_res.norm(dim=-1).mean().item()
    gt_res_std = all_res.std().item()
    print(f"\n  GT residual stats:")
    print(f"    Mean norm: {gt_res_norm:.4f}")
    print(f"    Std: {gt_res_std:.4f}")

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
    save_path = OUTPUT_DIR / "operation_tree_gt.pt"
    torch.save({
        'model': tree.state_dict(),
        'res_mean': res_mean.cpu(),
        'res_std': res_std.cpu(),
        'n_ops': N_OPS,
        'param_dim': PARAM_DIM,
        'top_k': TOP_K,
        'variant': 'gt_sms',  # flag to distinguish from pred variant
    }, save_path)
    print(f"\nSaved GT tree to {save_path}")

    # Test
    results = test_gt_tree(tree, test_data, res_mean, res_std, device, n_test)

    # Analyze operations
    analyze_operations(tree, test_data, res_mean, res_std, device)

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 GT RESULTS")
    print("=" * 60)
    print(f"\n  GT SMS cos_sim:         {results['sms_cos']:.4f}  (what SMS captures)")
    print(f"  GT SMS+tree cos_sim:    {results['tree_cos']:.4f}  (with GT tree)")
    print(f"  Improvement:            {results['tree_cos']-results['sms_cos']:+.4f}")
    print(f"\n  Operations: {N_OPS} (param_dim={PARAM_DIM}, top_k={TOP_K})")
    print(f"\n  Compare with original tree (pred SMS residuals):")
    print(f"    If GT tree improvement >> pred tree improvement:")
    print(f"      → operations are meaningful, codec error was drowning the signal")
    print(f"    If GT tree improvement ≈ pred tree improvement:")
    print(f"      → operations capture similar structure either way")
    print(f"    If GT tree improvement << pred tree improvement:")
    print(f"      → original tree was mostly fixing codec errors, not SMS gaps")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
