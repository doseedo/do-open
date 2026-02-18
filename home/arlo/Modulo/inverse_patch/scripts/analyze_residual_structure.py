#!/usr/bin/env python3
"""
Analyze Residual Structure: What does SMS miss?

Computes residuals (z_real - G(F(z_real))) from the Phase 1 codec,
then analyzes their structure to determine if they fall into distinct
types that could become parameterized operations.

Analysis:
  1. PCA — how many dimensions matter?
  2. K-means clustering — do residuals form distinct types?
  3. Temporal patterns — are residuals frame-independent or structured over time?
  4. DCAE channel analysis — which channels carry the residual?
  5. Correlation with SMS params — does residual type depend on what SMS captured?
"""

import sys
import torch
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
import gc

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "residual_analysis"

Z_FLAT_DIM = 128
MAX_FRAMES = 256


def load_phase1_codec(device):
    print(f"Loading Phase 1 codec from {PHASE1_PATH}...")
    model = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_FLAT_DIM, g_hidden=384, f_hidden=256)
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

            data.append({
                'z_real': z,
                'path': str(pt_file),
                'name': pt_file.stem,
            })

            if len(data) % 200 == 0:
                print(f"    Loaded {len(data)} samples...")
        except Exception:
            skipped += 1
            continue

    print(f"  Loaded {len(data)} latents (skipped {skipped})")
    return data


def compute_residuals(codec, data, device):
    """Compute residuals and SMS predictions for all samples."""
    print("\nComputing residuals: z_real - G(F(z_real))...")

    all_z_flat = []      # [T, 128] per sample
    all_z_sms = []       # [T, 128] per sample
    all_residuals = []   # [T, 128] per sample
    all_sms_pred = []    # [T, 102] per sample
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

            all_z_flat.append(z_flat.squeeze(0).cpu())
            all_z_sms.append(z_sms.squeeze(0).cpu())
            all_residuals.append(residual.squeeze(0).cpu())
            all_sms_pred.append(sms_pred.squeeze(0).cpu())

            if i % 200 == 0:
                print(f"    {i}/{len(data)}  cos_sim={cos:.4f}")

    print(f"  Mean cos_sim(z_real, z_sms): {np.mean(cos_sims):.4f}")
    return all_z_flat, all_z_sms, all_residuals, all_sms_pred, cos_sims


def analyze_pca(all_frames, label="residual"):
    """PCA: how many dimensions explain the variance?"""
    print(f"\n{'='*60}")
    print(f"PCA ANALYSIS ({label})")
    print(f"{'='*60}")

    n = min(100000, all_frames.shape[0])
    subset = all_frames[np.random.choice(all_frames.shape[0], n, replace=False)]

    pca = PCA(n_components=min(128, n))
    pca.fit(subset)

    cumvar = np.cumsum(pca.explained_variance_ratio_)

    thresholds = [0.50, 0.80, 0.90, 0.95, 0.99]
    for t in thresholds:
        n_dims = np.searchsorted(cumvar, t) + 1
        print(f"  {t*100:.0f}% variance: {n_dims} dimensions")

    print(f"\n  Top 10 components explain: {cumvar[9]*100:.1f}%")
    print(f"  Top 20 components explain: {cumvar[19]*100:.1f}%")
    print(f"  Top 50 components explain: {cumvar[min(49, len(cumvar)-1)]*100:.1f}%")

    return pca, cumvar


def analyze_clustering(all_frames, label="residual"):
    """K-means: do residuals form distinct clusters?"""
    print(f"\n{'='*60}")
    print(f"CLUSTERING ANALYSIS ({label})")
    print(f"{'='*60}")

    n = min(50000, all_frames.shape[0])
    subset = all_frames[np.random.choice(all_frames.shape[0], n, replace=False)]

    # First do PCA to reduce dimensionality for clustering
    pca = PCA(n_components=32)
    subset_pca = pca.fit_transform(subset)
    print(f"  Reduced to {subset_pca.shape[1]} PCA dims for clustering")

    # Try different k values
    results = {}
    for k in [4, 8, 16, 32, 64]:
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=42)
        labels = km.fit_predict(subset_pca)

        # Silhouette (subsample for speed)
        sil_n = min(5000, n)
        sil_idx = np.random.choice(n, sil_n, replace=False)
        sil = silhouette_score(subset_pca[sil_idx], labels[sil_idx])

        # Cluster sizes
        sizes = np.bincount(labels)
        min_pct = sizes.min() / n * 100
        max_pct = sizes.max() / n * 100

        inertia_per_sample = km.inertia_ / n

        results[k] = {'sil': sil, 'inertia': inertia_per_sample, 'labels': labels, 'km': km}
        print(f"  k={k:3d}: silhouette={sil:.4f}  inertia/n={inertia_per_sample:.4f}  "
              f"cluster sizes: {min_pct:.1f}%-{max_pct:.1f}%")

    # Find best k by silhouette
    best_k = max(results, key=lambda k: results[k]['sil'])
    print(f"\n  Best k by silhouette: {best_k}")

    return results, pca


def analyze_channels(all_residuals_frames):
    """Which DCAE channels carry the most residual energy?"""
    print(f"\n{'='*60}")
    print(f"DCAE CHANNEL ANALYSIS")
    print(f"{'='*60}")

    # Reshape [N, 128] → [N, 8, 16]
    reshaped = all_residuals_frames.reshape(-1, 8, 16)

    # Energy per channel
    channel_energy = np.mean(reshaped ** 2, axis=(0, 2))  # [8]
    total = channel_energy.sum()
    print(f"\n  Per-channel residual energy (fraction of total):")
    for ch in range(8):
        pct = channel_energy[ch] / total * 100
        bar = '#' * int(pct)
        print(f"    Channel {ch}: {pct:5.1f}%  {bar}")

    # Per-dimension variance within each channel
    print(f"\n  Per-channel residual std (avg across latent dims):")
    channel_std = np.std(reshaped, axis=0).mean(axis=1)  # [8]
    for ch in range(8):
        print(f"    Channel {ch}: std={channel_std[ch]:.4f}")

    # Correlation between channels
    channel_means = reshaped.mean(axis=2)  # [N, 8]
    corr = np.corrcoef(channel_means.T)  # [8, 8]
    print(f"\n  Cross-channel correlation (top pairs):")
    pairs = []
    for i in range(8):
        for j in range(i+1, 8):
            pairs.append((abs(corr[i, j]), i, j, corr[i, j]))
    pairs.sort(reverse=True)
    for r, i, j, raw in pairs[:5]:
        print(f"    ch{i}-ch{j}: corr={raw:+.3f}")

    return channel_energy


def analyze_temporal(all_residuals, sample_limit=200):
    """Are residuals temporally structured (autocorrelation, smoothness)?"""
    print(f"\n{'='*60}")
    print(f"TEMPORAL STRUCTURE ANALYSIS")
    print(f"{'='*60}")

    autocorrs_lag1 = []
    autocorrs_lag2 = []
    autocorrs_lag5 = []
    frame_diffs = []

    for res in all_residuals[:sample_limit]:
        # res: [T, 128]
        T = res.shape[0]
        if T < 10:
            continue

        res_np = res.numpy()

        # Frame-to-frame difference (smoothness)
        diff = np.sqrt(np.mean((res_np[1:] - res_np[:-1]) ** 2, axis=1))
        frame_diffs.extend(diff.tolist())

        # Autocorrelation of residual norm at different lags
        norms = np.sqrt(np.sum(res_np ** 2, axis=1))  # [T]
        norms_centered = norms - norms.mean()
        var = np.var(norms)
        if var < 1e-10:
            continue

        for lag, store in [(1, autocorrs_lag1), (2, autocorrs_lag2), (5, autocorrs_lag5)]:
            if T > lag:
                ac = np.mean(norms_centered[:-lag] * norms_centered[lag:]) / var
                store.append(ac)

    print(f"  Temporal autocorrelation of residual magnitude:")
    print(f"    Lag 1: {np.mean(autocorrs_lag1):.4f} (high = smooth, low = random)")
    print(f"    Lag 2: {np.mean(autocorrs_lag2):.4f}")
    print(f"    Lag 5: {np.mean(autocorrs_lag5):.4f}")

    print(f"\n  Frame-to-frame residual change (RMS):")
    print(f"    Mean: {np.mean(frame_diffs):.4f}")
    print(f"    Std:  {np.std(frame_diffs):.4f}")

    if np.mean(autocorrs_lag1) > 0.5:
        print(f"\n  >> Residuals are TEMPORALLY SMOOTH — operations need temporal modeling")
    elif np.mean(autocorrs_lag1) > 0.2:
        print(f"\n  >> Residuals have MODERATE temporal structure")
    else:
        print(f"\n  >> Residuals are FRAME-INDEPENDENT — per-frame operations may suffice")


def analyze_residual_vs_sms(all_residuals, all_sms_pred, cluster_results, cluster_pca):
    """Does residual type correlate with what SMS captured?"""
    print(f"\n{'='*60}")
    print(f"RESIDUAL-SMS CORRELATION")
    print(f"{'='*60}")

    # Stack all frames
    res_frames = torch.cat(all_residuals, dim=0).numpy()
    sms_frames = torch.cat(all_sms_pred, dim=0).numpy()

    n = min(50000, res_frames.shape[0])
    idx = np.random.choice(res_frames.shape[0], n, replace=False)
    res_sub = res_frames[idx]
    sms_sub = sms_frames[idx]

    # Use best clustering from earlier
    best_k = max(cluster_results, key=lambda k: cluster_results[k]['sil'])
    km = cluster_results[best_k]['km']

    # Assign clusters to all frames
    res_pca = cluster_pca.transform(res_sub)
    labels = km.predict(res_pca)

    # For each cluster, compute mean SMS params
    print(f"\n  Per-cluster SMS characteristics (k={best_k}):")

    # SMS dims: 6 f0s, 48 partial amps, 20 indep freqs, 20 indep amps, 8 noise
    sms_groups = {
        'f0s (0-5)': slice(0, 6),
        'partial_amps (6-53)': slice(6, 54),
        'indep_freqs (54-73)': slice(54, 74),
        'indep_amps (74-93)': slice(74, 94),
        'noise (94-101)': slice(94, 102),
    }

    for c in range(min(best_k, 16)):
        mask = labels == c
        n_in_cluster = mask.sum()
        if n_in_cluster < 10:
            continue

        sms_cluster = sms_sub[mask]
        res_cluster = res_sub[mask]

        # Residual magnitude
        res_mag = np.sqrt(np.mean(res_cluster ** 2))

        # SMS characteristics
        parts = {}
        for name, sl in sms_groups.items():
            parts[name] = np.mean(np.abs(sms_cluster[:, sl]))

        top_sms = sorted(parts.items(), key=lambda x: x[1], reverse=True)
        top_str = ", ".join([f"{n}={v:.3f}" for n, v in top_sms[:2]])
        print(f"    Cluster {c:2d} ({n_in_cluster:5d} frames): res_mag={res_mag:.4f}  SMS: {top_str}")


def analyze_residual_magnitude_distribution(all_residuals, all_z_flat):
    """How big are residuals relative to z? Distribution analysis."""
    print(f"\n{'='*60}")
    print(f"RESIDUAL MAGNITUDE DISTRIBUTION")
    print(f"{'='*60}")

    res_frames = torch.cat(all_residuals, dim=0).numpy()
    z_frames = torch.cat(all_z_flat, dim=0).numpy()

    res_norms = np.sqrt(np.sum(res_frames ** 2, axis=1))
    z_norms = np.sqrt(np.sum(z_frames ** 2, axis=1))
    ratios = res_norms / (z_norms + 1e-8)

    print(f"  Residual ||r|| / ||z|| ratio:")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(ratios, pct)
        print(f"    P{pct:2d}: {val:.4f}")

    print(f"\n  Residual norm distribution:")
    print(f"    Mean: {res_norms.mean():.4f}")
    print(f"    Std:  {res_norms.std():.4f}")
    print(f"    Min:  {res_norms.min():.4f}")
    print(f"    Max:  {res_norms.max():.4f}")

    # Are there bimodal residuals? (some near zero, some large)
    small = (ratios < 0.1).sum()
    medium = ((ratios >= 0.1) & (ratios < 0.3)).sum()
    large = (ratios >= 0.3).sum()
    n = len(ratios)
    print(f"\n  Residual size breakdown:")
    print(f"    Small  (r/z < 0.1): {small/n*100:.1f}%")
    print(f"    Medium (0.1-0.3):   {medium/n*100:.1f}%")
    print(f"    Large  (> 0.3):     {large/n*100:.1f}%")


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("RESIDUAL STRUCTURE ANALYSIS")
    print("What does SMS miss? What should operations look like?")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    codec = load_phase1_codec(device)
    data = gather_real_latents(max_samples=2000)

    all_z_flat, all_z_sms, all_residuals, all_sms_pred, cos_sims = \
        compute_residuals(codec, data, device)

    # Free GPU
    codec = codec.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    # Stack all residual frames for analysis
    all_res_frames = torch.cat(all_residuals, dim=0).numpy()
    all_z_frames = torch.cat(all_z_flat, dim=0).numpy()
    print(f"\nTotal residual frames for analysis: {all_res_frames.shape[0]:,}")

    # 1. Magnitude distribution
    analyze_residual_magnitude_distribution(all_residuals, all_z_flat)

    # 2. PCA
    pca_res, cumvar_res = analyze_pca(all_res_frames, "residual")
    pca_z, cumvar_z = analyze_pca(all_z_frames, "z_real (for comparison)")

    # 3. Clustering
    cluster_results, cluster_pca = analyze_clustering(all_res_frames, "residual")

    # 4. Channel analysis
    analyze_channels(all_res_frames)

    # 5. Temporal structure
    analyze_temporal(all_residuals)

    # 6. Residual vs SMS correlation
    analyze_residual_vs_sms(all_residuals, all_sms_pred, cluster_results, cluster_pca)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: WHAT SHOULD OPERATIONS LOOK LIKE?")
    print("=" * 60)

    # PCA dimensionality
    n90 = np.searchsorted(cumvar_res, 0.90) + 1
    n95 = np.searchsorted(cumvar_res, 0.95) + 1
    print(f"\n  Intrinsic dimensionality: ~{n90} dims for 90%, ~{n95} dims for 95%")

    # Clustering
    best_k = max(cluster_results, key=lambda k: cluster_results[k]['sil'])
    best_sil = cluster_results[best_k]['sil']
    print(f"  Best cluster count: {best_k} (silhouette={best_sil:.4f})")
    if best_sil > 0.3:
        print(f"    >> Strong clusters — residuals have {best_k} distinct types!")
        print(f"    >> Each type could become a parameterized operation")
    elif best_sil > 0.15:
        print(f"    >> Moderate clusters — some structure, operations should help")
    else:
        print(f"    >> Weak clusters — residuals are continuous, not discrete types")
        print(f"    >> Operations should be basis functions, not discrete categories")

    print(f"\n  Recommendation for unified operation tree architecture:")
    if best_sil > 0.2 and n90 < 30:
        print(f"    Low-dimensional + clustered:")
        print(f"    → {best_k} parameterized operations with ~{n90}-dim parameter spaces")
        print(f"    → Each operation maps params → z-contribution via learned function")
    elif n90 < 30:
        print(f"    Low-dimensional but continuous:")
        print(f"    → Continuous basis functions (like current atoms) but parameterized")
        print(f"    → k={n90} operations with learnable activation functions")
    else:
        print(f"    High-dimensional residual space:")
        print(f"    → Need more operations or richer parameterization")
        print(f"    → Consider hierarchical operation tree")

    # Save analysis results
    save_path = OUTPUT_DIR / "analysis_results.pt"
    torch.save({
        'cos_sims': cos_sims,
        'pca_explained_variance': cumvar_res.tolist(),
        'cluster_silhouettes': {k: v['sil'] for k, v in cluster_results.items()},
        'best_k': best_k,
        'n_frames': all_res_frames.shape[0],
    }, save_path)
    print(f"\n  Results saved to {save_path}")


if __name__ == "__main__":
    main()
