#!/usr/bin/env python3
"""
Analyze z structure to find interpretable sine operations.

Even if z is entangled, we can find transforms that make it interpretable.

Approach order:
2. Analyze DCAE decoder's first layer weights
4. Probe what each z direction does (perturbation analysis)
1. Dictionary learning / ICA on z
3. Learn a disentangling transform
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

import orjson
from sklearn.decomposition import FastICA, DictionaryLearning
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("=" * 70)
print("ANALYZING Z STRUCTURE FOR INTERPRETABLE SINE OPERATIONS")
print("=" * 70)

# Load DCAE
print("\nLoading DCAE...")
DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

dcae = MusicDCAE(
    dcae_checkpoint_path=DCAE_PATH,
    vocoder_checkpoint_path=VOCODER_PATH,
)
decoder = dcae.dcae.decoder
decoder.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
decoder.to(device)

# Load some z samples for analysis
print("\nLoading z samples...")
with open('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json', 'rb') as f:
    manifest = orjson.loads(f.read())

z_samples = []
for entry in manifest['entries'][:200]:
    try:
        lat = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
        latent = lat.get('latents', lat.get('latent'))
        if latent.shape[-1] >= 22:
            z_samples.append(latent[:, :, :22])  # [8, 16, 22]
    except:
        continue

z_stack = torch.stack(z_samples)  # [N, 8, 16, 22]
print(f"Loaded {len(z_samples)} z samples, shape: {z_stack.shape}")

# ============================================================
# ANALYSIS 2: DCAE Decoder First Layer Weights
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 2: DCAE Decoder First Layer")
print("=" * 70)

# The decoder's conv_in transforms z
conv_in = decoder.conv_in
print(f"\nconv_in type: {type(conv_in)}")
print(f"conv_in: {conv_in}")

# Get the weight
if hasattr(conv_in, 'weight'):
    W = conv_in.weight.detach().cpu()  # [out_channels, in_channels, kH, kW]
    print(f"conv_in weight shape: {W.shape}")

    # Analyze what input channels (z dimensions) are most important
    # Importance = sum of absolute weights across outputs
    importance = W.abs().sum(dim=(0, 2, 3))  # [in_channels]
    print(f"\nInput channel importance (which z dims matter most):")
    top_dims = importance.argsort(descending=True)[:20]
    print(f"  Top 20 most important z dims: {top_dims.tolist()}")

    # Check coarse vs fine
    # z is [8, 16, T] -> conv_in input might be [8, T, 16] or similar
    # In_channels = 8 (the channel dim of z)
    if importance.shape[0] == 8:
        print(f"\n  Channel importance:")
        for i in range(8):
            print(f"    Channel {i}: {importance[i]:.4f}")
        print(f"  Coarse (0-3): {importance[:4].sum():.4f}")
        print(f"  Fine (4-7): {importance[4:].sum():.4f}")

# ============================================================
# ANALYSIS 4: Perturbation Analysis
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 4: Perturbation Analysis (what each z dim does)")
print("=" * 70)

def get_mel_from_decoder(decoder, z, device):
    """Get mel spectrogram from decoder (skip vocoder)."""
    # z: [B, 8, 16, T] -> decoder expects [B, 8, T, 16]
    z_dec = z.permute(0, 1, 3, 2).to(device)
    with torch.no_grad():
        mel = decoder(z_dec)  # [B, C, H, W] mel spectrogram
    return mel

# Take one sample
z_base = z_stack[0:1].to(device)  # [1, 8, 16, 22]
print(f"\nBase z shape: {z_base.shape}")

# Get base mel (skip vocoder - just use decoder output)
with torch.no_grad():
    base_mel = get_mel_from_decoder(decoder, z_base, device)

print(f"Base mel shape: {base_mel.shape}")

# Use mel frequency bins as our "spectrum"
# Mel is [B, channels, freq_bins, time] typically
base_spectrum = base_mel[0].mean(dim=(0, -1))  # Average over channels and time
n_freq_bins = base_spectrum.shape[0]
print(f"Base spectrum shape: {base_spectrum.shape}")

# Perturb each z dimension and measure spectral change
influence_matrix = torch.zeros(128, n_freq_bins)  # [z_dims, freq_bins]

delta = 2.0  # Perturbation magnitude

print("\nComputing influence matrix (perturbing each z dim)...")

for dim in range(128):
    # Perturb dimension in [8, 16] space
    # dim 0-15 = channel 0, dim 16-31 = channel 1, etc.
    ch = dim // 16
    pos = dim % 16

    z_perturbed = z_base.clone()
    z_perturbed[0, ch, pos, :] += delta

    with torch.no_grad():
        perturbed_mel = get_mel_from_decoder(decoder, z_perturbed, device)
        perturbed_spectrum = perturbed_mel[0].mean(dim=(0, -1))

    # Measure spectral change
    spectral_change = (perturbed_spectrum - base_spectrum).abs().cpu()
    influence_matrix[dim] = spectral_change

    if (dim + 1) % 32 == 0:
        print(f"  Processed {dim + 1}/128 dimensions")

# Analyze influence matrix
print("\nInfluence matrix analysis:")

# Which z dims have most overall influence?
total_influence = influence_matrix.sum(dim=1)
top_influential = total_influence.argsort(descending=True)[:20]
print(f"\nTop 20 most influential z dims: {top_influential.tolist()}")

# Coarse vs fine influence
coarse_influence = total_influence[:64].sum()
fine_influence = total_influence[64:].sum()
print(f"Coarse (dims 0-63) total influence: {coarse_influence:.2f}")
print(f"Fine (dims 64-127) total influence: {fine_influence:.2f}")

# For each z dim, what frequency range does it affect most?
# Mel bins are log-spaced, roughly split into thirds
bass_end = n_freq_bins // 3
mid_end = 2 * n_freq_bins // 3
treble_end = n_freq_bins

print(f"\nFrequency band analysis per z dim:")
print(f"  (Bass: 0-500Hz, Mid: 500-2000Hz, Treble: 2000-8000Hz)")

# Find dims that specialize in each band
bass_specialists = []
mid_specialists = []
treble_specialists = []

for dim in range(128):
    bass_inf = influence_matrix[dim, :bass_end].sum().item()
    mid_inf = influence_matrix[dim, bass_end:mid_end].sum().item()
    treble_inf = influence_matrix[dim, mid_end:treble_end].sum().item()
    total = bass_inf + mid_inf + treble_inf + 1e-6

    if bass_inf / total > 0.5:
        bass_specialists.append((dim, bass_inf / total))
    if mid_inf / total > 0.5:
        mid_specialists.append((dim, mid_inf / total))
    if treble_inf / total > 0.5:
        treble_specialists.append((dim, treble_inf / total))

print(f"\n  Bass specialists (>50% influence in bass): {len(bass_specialists)}")
print(f"    Top 5: {sorted(bass_specialists, key=lambda x: -x[1])[:5]}")
print(f"\n  Mid specialists (>50% influence in mid): {len(mid_specialists)}")
print(f"    Top 5: {sorted(mid_specialists, key=lambda x: -x[1])[:5]}")
print(f"\n  Treble specialists (>50% influence in treble): {len(treble_specialists)}")
print(f"    Top 5: {sorted(treble_specialists, key=lambda x: -x[1])[:5]}")

# Save influence matrix for visualization
out_dir = Path('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/outputs/z_analysis')
out_dir.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(16, 8))
plt.imshow(influence_matrix.numpy(), aspect='auto', cmap='hot')
plt.colorbar(label='Spectral change')
plt.xlabel('Mel frequency bin')
plt.ylabel('Z dimension')
plt.title('Z Dimension → Frequency Influence Matrix')
plt.savefig(out_dir / 'influence_matrix.png', dpi=150)
print(f"\nSaved influence matrix plot to {out_dir / 'influence_matrix.png'}")

# ============================================================
# ANALYSIS 1: ICA / Dictionary Learning on z
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 1: ICA / Dictionary Learning on z")
print("=" * 70)

# Flatten z samples: [N, 8, 16, T] -> [N*T, 128]
z_for_ica = z_stack.permute(0, 3, 1, 2).reshape(-1, 128).numpy()
print(f"\nZ data for ICA: {z_for_ica.shape}")

# ICA: Find independent components
print("\nRunning FastICA...")
ica = FastICA(n_components=32, random_state=42, max_iter=500)
try:
    z_ica = ica.fit_transform(z_for_ica[:5000])  # Use subset for speed
    ica_mixing = ica.mixing_  # [128, 32] - how ICs combine to form z

    print(f"ICA mixing matrix shape: {ica_mixing.shape}")

    # Analyze: are ICA components sparse in z-space?
    sparsity_per_ic = (np.abs(ica_mixing) > 0.1).sum(axis=0) / 128
    print(f"ICA component sparsity (fraction of z dims with |weight| > 0.1):")
    print(f"  Mean: {sparsity_per_ic.mean():.3f}")
    print(f"  Min: {sparsity_per_ic.min():.3f}")
    print(f"  Max: {sparsity_per_ic.max():.3f}")

    # If sparsity is low, ICA found components that only depend on few z dims
    if sparsity_per_ic.mean() < 0.3:
        print("  → ICA found relatively sparse components!")
    else:
        print("  → ICA components are dense (z is entangled)")

except Exception as e:
    print(f"ICA failed: {e}")

# Dictionary learning: find sparse code
print("\nRunning Dictionary Learning...")
try:
    dict_learner = DictionaryLearning(
        n_components=64,
        alpha=1.0,  # Sparsity
        max_iter=100,
        random_state=42,
    )
    z_sparse = dict_learner.fit_transform(z_for_ica[:2000])  # Smaller subset
    dictionary = dict_learner.components_  # [64, 128]

    print(f"Dictionary shape: {dictionary.shape}")
    print(f"Sparse code shape: {z_sparse.shape}")

    # How sparse is the code?
    sparsity = (np.abs(z_sparse) > 0.01).mean()
    print(f"Sparse code density: {sparsity:.3f} (lower = sparser)")

    # Reconstruction error
    z_reconstructed = z_sparse @ dictionary
    recon_error = np.mean((z_for_ica[:2000] - z_reconstructed) ** 2)
    print(f"Reconstruction MSE: {recon_error:.4f}")

except Exception as e:
    print(f"Dictionary learning failed: {e}")

# ============================================================
# ANALYSIS 3: Learn Disentangling Transform
# ============================================================
print("\n" + "=" * 70)
print("ANALYSIS 3: Learning Disentangling Transform")
print("=" * 70)

print("\nTraining a simple disentangling network...")
print("Goal: z -> W @ z -> interpretable_z where each dim affects specific freqs")

class DisentangleNet(nn.Module):
    """Learn a transform that makes z more interpretable."""
    def __init__(self):
        super().__init__()
        # Learnable linear transform (basis change)
        self.transform = nn.Linear(128, 128, bias=False)

        # Initialize as identity
        nn.init.eye_(self.transform.weight)

    def forward(self, z):
        # z: [B, 128]
        return self.transform(z)

    def get_sparsity_loss(self):
        """Encourage sparse weights (each output uses few inputs)."""
        W = self.transform.weight  # [128, 128]
        # L1 on rows (each output dim should use few inputs)
        return W.abs().mean()

# We need a target to train against
# Use influence_matrix: we want transformed z where each dim maps to specific freq band

# Actually, let's train to predict frequency band activations
# This will learn a transform that separates z into frequency-meaningful components

class FreqPredictor(nn.Module):
    def __init__(self, n_freq_bands=16):
        super().__init__()
        self.disentangle = nn.Linear(128, 128)
        self.predict = nn.Linear(128, n_freq_bands)

    def forward(self, z):
        z_dis = F.relu(self.disentangle(z))
        return self.predict(z_dis)

# Quick training to see if we can predict frequency content from z
print("\nTraining frequency predictor from z...")

# Prepare data: z -> frequency band energies
# Use the influence_matrix knowledge: we know z affects frequencies
# Let's predict coarse frequency bands from z

from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader

# Get frequency content from decoder mel output
n_train_samples = 50
freq_targets = []
z_inputs = []

print("Collecting frequency targets from decoder mel...")
for i in range(min(n_train_samples, len(z_samples))):
    z_sample = z_stack[i:i+1].to(device)
    with torch.no_grad():
        mel = get_mel_from_decoder(decoder, z_sample, device)
        spec = mel[0].mean(dim=(0, -1)).cpu()  # [freq_bins]

        # Bin into 16 frequency bands
        n_bands = 16
        band_size = len(spec) // n_bands
        bands = torch.tensor([spec[j*band_size:(j+1)*band_size].mean() for j in range(n_bands)])

        freq_targets.append(bands)
        z_inputs.append(z_sample.cpu().reshape(128, -1).mean(dim=-1))  # Average over time

    if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{n_train_samples}")

freq_targets = torch.stack(freq_targets)
z_inputs = torch.stack(z_inputs)

print(f"Training data: z={z_inputs.shape}, freq={freq_targets.shape}")

# Train
model = FreqPredictor(n_freq_bands=16)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(100):
    optimizer.zero_grad()
    pred = model(z_inputs)
    loss = F.mse_loss(pred, freq_targets)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"  Epoch {epoch}: loss={loss.item():.6f}")

# Analyze learned transform
W_dis = model.disentangle.weight.detach()
W_pred = model.predict.weight.detach()

print(f"\nLearned transform analysis:")
print(f"  Disentangle weight shape: {W_dis.shape}")
print(f"  Prediction weight shape: {W_pred.shape}")

# How sparse is the prediction layer?
# Each freq band should ideally depend on few disentangled dims
pred_sparsity = (W_pred.abs() > 0.1).float().mean(dim=1)
print(f"  Prediction sparsity per band (fraction of dims with |w|>0.1):")
print(f"    Mean: {pred_sparsity.mean():.3f}")
print(f"    Range: [{pred_sparsity.min():.3f}, {pred_sparsity.max():.3f}]")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Key findings:

1. DECODER FIRST LAYER: Shows which z channels (0-7) are most important
   for the decoder's initial transform.

2. PERTURBATION ANALYSIS: Created influence_matrix showing which z dims
   affect which frequencies. Some dims specialize in bass/mid/treble.

3. ICA/DICTIONARY LEARNING: Tests if z has a sparse basis.
   - If ICA components are sparse: z = W @ sparse_sources
   - If dictionary works: z can be represented sparsely

4. DISENTANGLING TRANSFORM: Can we learn W such that W @ z has
   interpretable structure (each dim → specific frequencies)?

IMPLICATIONS FOR SINE EXTRACTION:
- If z has frequency-band specialists → route them to appropriate sines
- If ICA finds sparse components → use those as intermediate representation
- If dictionary learning works → use sparse code for sine prediction
""")

print(f"\nOutputs saved to: {out_dir}")
