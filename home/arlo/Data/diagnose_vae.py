#!/usr/bin/env python3
"""
Diagnose VAE reconstruction quality and identify issues.
"""

import sys
sys.path.insert(0, '/home/arlo/do-repo/home/arlo/soundspace')

import torch
import numpy as np
from pathlib import Path

from timbre_vae import TimbreVAE

print("=" * 60)
print("VAE Reconstruction Diagnosis")
print("=" * 60)

# Load VAE
VAE_PATH = "/home/arlo/soundspace_checkpoints/timbre_vae_final.pt"
vae_ckpt = torch.load(VAE_PATH, map_location='cpu', weights_only=False)
vae_config = vae_ckpt['config']

vae = TimbreVAE(**{k: v for k, v in vae_config.items()
                   if k in ['input_dim', 'latent_dim', 'hidden_dim',
                           'n_residual', 'dropout', 'student_t_df']})
vae.load_state_dict(vae_ckpt['model'])
vae.eval()
vae_mean = vae_ckpt['mean']
vae_std = vae_ckpt['std']

print(f"VAE loaded: {vae_config['latent_dim']}D latent")

# Load validation frames
VAL_FRAMES_PATH = "/home/arlo/soundspace_data/val_frames.pt"
val_frames = torch.load(VAL_FRAMES_PATH, map_location='cpu')
print(f"Loaded {len(val_frames)} validation frames")

# ============== 1. Check frame energy distribution ==============
print("\n[1] Frame Energy Distribution")
energy = val_frames.pow(2).sum(dim=-1)
print(f"  Energy range: [{energy.min():.2f}, {energy.max():.2f}]")
print(f"  Energy mean: {energy.mean():.2f}")
print(f"  Energy median: {energy.median():.2f}")

# What fraction is "silent" (low energy)?
thresholds = [0.01, 0.1, 0.5, 1.0, 5.0]
for t in thresholds:
    frac = (energy < t).float().mean() * 100
    print(f"  Frames with energy < {t}: {frac:.1f}%")

# ============== 2. Reconstruction error analysis ==============
print("\n[2] Reconstruction Error Analysis")

# Sample random frames
n_samples = 1000
sample_idx = torch.randperm(len(val_frames))[:n_samples]
sample_frames = val_frames[sample_idx].float()

# Normalize
frames_norm = (sample_frames - vae_mean) / (vae_std + 1e-8)

# Encode/decode
with torch.no_grad():
    mu, logvar = vae.encode(frames_norm)
    z = vae.reparameterize(mu, logvar)
    recon_norm = vae.decode(z)

# Denormalize
recon = recon_norm * (vae_std + 1e-8) + vae_mean

# MSE
mse_per_frame = (sample_frames - recon).pow(2).mean(dim=-1)
print(f"  MSE per frame: mean={mse_per_frame.mean():.4f}, std={mse_per_frame.std():.4f}")
print(f"  MSE range: [{mse_per_frame.min():.4f}, {mse_per_frame.max():.4f}]")

# Correlation per frame
correlations = []
for i in range(min(100, n_samples)):
    orig = sample_frames[i]
    rec = recon[i]
    if orig.std() > 1e-6 and rec.std() > 1e-6:  # Skip near-constant frames
        corr = torch.corrcoef(torch.stack([orig, rec]))[0, 1].item()
        correlations.append(corr)

correlations = np.array(correlations)
print(f"  Correlation: mean={correlations.mean():.4f}, std={correlations.std():.4f}")
print(f"  Correlation range: [{correlations.min():.4f}, {correlations.max():.4f}]")

# ============== 3. Energy-stratified analysis ==============
print("\n[3] Energy-Stratified Reconstruction")

sample_energy = sample_frames.pow(2).sum(dim=-1)
quartiles = torch.quantile(sample_energy, torch.tensor([0.25, 0.5, 0.75]))

labels = ["Q1 (low energy)", "Q2", "Q3", "Q4 (high energy)"]
bounds = [0, quartiles[0], quartiles[1], quartiles[2], float('inf')]

for i in range(4):
    mask = (sample_energy >= bounds[i]) & (sample_energy < bounds[i+1])
    if mask.sum() > 0:
        mse_q = mse_per_frame[mask].mean()
        print(f"  {labels[i]}: MSE={mse_q:.4f}, n={mask.sum()}")

# ============== 4. Check what low-energy frames look like ==============
print("\n[4] Low-Energy Frame Analysis")

low_energy_mask = energy < 1.0
low_frames = val_frames[low_energy_mask][:10]
print(f"  Low-energy frames sample (first 10):")
for i, f in enumerate(low_frames):
    print(f"    Frame {i}: mean={f.mean():.4f}, std={f.std():.4f}, max={f.abs().max():.4f}")

# ============== 5. Check cluster centers ==============
print("\n[5] Cluster Center Analysis")

from sklearn.cluster import KMeans

# Use subset for clustering
cluster_frames = val_frames[torch.randperm(len(val_frames))[:10000]].float()
frames_norm_cluster = (cluster_frames - vae_mean) / (vae_std + 1e-8)

with torch.no_grad():
    mu_cluster, _ = vae.encode(frames_norm_cluster)

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
kmeans.fit(mu_cluster.numpy())
centers = torch.from_numpy(kmeans.cluster_centers_)

print("  Cluster centers (8D):")
for i, c in enumerate(centers):
    # Decode center
    with torch.no_grad():
        decoded = vae.decode(c.unsqueeze(0))
    decoded_denorm = decoded * (vae_std + 1e-8) + vae_mean
    energy_c = decoded_denorm.pow(2).sum().item()
    print(f"    Cluster {i}: z_norm={c.norm():.2f}, decoded_energy={energy_c:.2f}")

# ============== 6. Recommendations ==============
print("\n" + "=" * 60)
print("DIAGNOSIS SUMMARY")
print("=" * 60)

high_corr = correlations.mean() > 0.9
low_mse = mse_per_frame.mean() < 1.0
low_energy_issue = (energy < 1.0).float().mean() > 0.1

print(f"""
Reconstruction Quality:
  - Correlation: {correlations.mean():.4f} {'✓ Good' if high_corr else '⚠ Low'}
  - MSE: {mse_per_frame.mean():.4f} {'✓ Good' if low_mse else '⚠ High'}

Data Issues:
  - Low-energy frames: {(energy < 1.0).float().mean()*100:.1f}% {'⚠ Many silent frames' if low_energy_issue else '✓ OK'}

Recommendations:
""")

if low_energy_issue:
    print("  1. FILTER LOW-ENERGY FRAMES before generating audio demos")
    print("     The buzzing is DCAE trying to decode near-zero latents")

if not high_corr:
    print("  2. VAE reconstruction quality may be insufficient")
    print("     Consider retraining with higher capacity or lower beta")
else:
    print("  2. VAE reconstruction is good - issues are likely in demo generation")

print("""
  3. For roundtrip demos, use CONSECUTIVE frames from same recording
     Random frames from different sources will sound unrelated

  4. Polyphonic bias is expected if 15% of data is ensemble
     The cluster centers reflect data distribution
""")
