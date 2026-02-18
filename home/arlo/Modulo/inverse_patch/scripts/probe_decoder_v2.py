#!/usr/bin/env python3
"""
Probe decoder features more carefully.

Previous probe got negative R² for raw frequencies.
Try:
1. Pitch class (chroma) - maybe decoder encodes pitch cyclically
2. Relative pitch (intervals) - maybe decoder encodes changes, not absolutes
3. Different layers - maybe pitch is resolved later
4. Small MLP instead of linear - maybe relationship is slightly nonlinear
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

print("=" * 70)
print("Decoder Feature Probing V2")
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

# Load SMS data
print("Loading SMS samples...")
with open('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json', 'rb') as f:
    manifest = orjson.loads(f.read())

samples = []
for entry in manifest['entries'][:300]:
    try:
        sms = torch.load(entry['path'], weights_only=True, map_location='cpu')
        lat = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
        latent = lat.get('latents', lat.get('latent'))

        if any(kw in sms.get('audio_path', '').lower()
               for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            continue

        T = min(latent.shape[-1], 22)
        if T < 22:
            continue

        latent = latent[:, :, :T]
        freqs = sms['freqs'][:T]
        amps = sms['amps'][:T] * 10.0

        samples.append({
            'latent': latent,
            'freqs': freqs,
            'amps': amps,
        })
    except:
        continue

print(f"Loaded {len(samples)} samples")

# Collect activations at multiple layers
print("\nCollecting decoder activations at multiple layers...")

activations = {
    'z_raw': [],
    'conv_in': [],
    'up_block_3': [],  # After attention
    'up_block_2': [],  # After first upsample
}

hooks = []

def make_hook(name):
    def hook(module, input, output):
        activations[name].append(output.detach().cpu())
    return hook

hooks.append(decoder.conv_in.register_forward_hook(make_hook('conv_in')))
hooks.append(decoder.up_blocks[3].register_forward_hook(make_hook('up_block_3')))
hooks.append(decoder.up_blocks[2].register_forward_hook(make_hook('up_block_2')))

all_freqs = []
all_amps = []

with torch.no_grad():
    for i, sample in enumerate(samples[:150]):
        z = sample['latent'].unsqueeze(0)
        activations['z_raw'].append(z.cpu())

        z_dec = z.permute(0, 1, 3, 2)  # [1, 8, T, 16] for decoder
        _ = decoder(z_dec)

        all_freqs.append(sample['freqs'])
        all_amps.append(sample['amps'])

        if (i + 1) % 30 == 0:
            print(f"  Processed {i + 1} samples")

for h in hooks:
    h.remove()

# Stack
N = len(all_freqs)
all_freqs = torch.stack(all_freqs)  # [N, T, n_sines]
all_amps = torch.stack(all_amps)

print(f"\nActivation shapes:")
for name, acts in activations.items():
    stacked = torch.cat(acts, dim=0)
    activations[name] = stacked
    print(f"  {name}: {stacked.shape}")

# Get dominant f0 per frame
def get_f0(freqs, amps, threshold=0.1):
    """Get lowest strong frequency as f0 estimate."""
    f0s = []
    for i in range(freqs.shape[0]):
        for t in range(freqs.shape[1]):
            mask = amps[i, t] > threshold
            if mask.sum() > 0:
                active_freqs = freqs[i, t][mask]
                f0 = active_freqs.min().item()
            else:
                f0 = 200.0  # Default
            f0s.append(f0)
    return torch.tensor(f0s)

f0_flat = get_f0(all_freqs, all_amps)
log_f0 = torch.log(f0_flat.clamp(min=20))

# Pitch class (chroma): log2(f0) mod 1
chroma = (torch.log2(f0_flat.clamp(min=20)) % 1.0)

# Convert to sin/cos for circular
chroma_sin = torch.sin(2 * np.pi * chroma)
chroma_cos = torch.cos(2 * np.pi * chroma)
chroma_2d = torch.stack([chroma_sin, chroma_cos], dim=-1)

print(f"\nTarget shapes:")
print(f"  log_f0: {log_f0.shape}")
print(f"  chroma_2d: {chroma_2d.shape}")

# ============================================================
# Probes
# ============================================================

def flatten_for_probe(act, n_samples, n_frames=22):
    """Flatten activation to [N*T, features]."""
    if act.dim() == 4:
        # [N, C, H, W] -> [N, H, C*W] -> [N*H, C*W] for temporal dim H
        # Assume dim 2 is time
        N, C, T, H = act.shape
        return act.permute(0, 2, 1, 3).reshape(N * T, C * H)
    elif act.dim() == 3:
        return act.reshape(-1, act.shape[-1])
    else:
        return act.reshape(n_samples * n_frames, -1)

def train_mlp_probe(X, y, hidden=128, n_epochs=200):
    """Train small MLP probe."""
    n_train = int(0.8 * X.shape[0])

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Normalize
    X_mean, X_std = X_train.mean(0), X_train.std(0).clamp(min=1e-6)
    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # MLP
    out_dim = y.shape[-1] if y.dim() > 1 else 1
    if y.dim() == 1:
        y_train = y_train.unsqueeze(-1)
        y_test = y_test.unsqueeze(-1)

    mlp = nn.Sequential(
        nn.Linear(X.shape[1], hidden),
        nn.GELU(),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Linear(hidden, out_dim),
    )

    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3)

    for epoch in range(n_epochs):
        opt.zero_grad()
        pred = mlp(X_train)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        test_pred = mlp(X_test)
        test_loss = F.mse_loss(test_pred, y_test).item()

        ss_res = ((test_pred - y_test) ** 2).sum()
        ss_tot = ((y_test - y_test.mean(0)) ** 2).sum()
        r2 = (1 - ss_res / ss_tot.clamp(min=1e-6)).item()

    return {'test_loss': test_loss, 'r2': r2}

# Run probes
print("\n" + "=" * 70)
print("MLP Probes (features → target)")
print("=" * 70)

N_samples = len(samples[:150])
N_frames = 22

# Flatten activations
z_flat = flatten_for_probe(activations['z_raw'], N_samples)
conv_in_flat = flatten_for_probe(activations['conv_in'], N_samples)
attn_flat = flatten_for_probe(activations['up_block_3'], N_samples)
up2_flat = flatten_for_probe(activations['up_block_2'], N_samples)

print(f"\nFlattened shapes:")
print(f"  z_raw: {z_flat.shape}")
print(f"  conv_in: {conv_in_flat.shape}")
print(f"  up_block_3 (attn): {attn_flat.shape}")
print(f"  up_block_2: {up2_flat.shape}")

# Probe for log_f0
print("\n--- Probing for log(f0) ---")
for name, X in [('z_raw', z_flat), ('conv_in', conv_in_flat),
                ('attn', attn_flat), ('up_block_2', up2_flat)]:
    if X.shape[0] != log_f0.shape[0]:
        print(f"  {name}: SKIP (shape mismatch {X.shape[0]} vs {log_f0.shape[0]})")
        continue
    result = train_mlp_probe(X, log_f0)
    print(f"  {name:12s}: R² = {result['r2']:.4f}, loss = {result['test_loss']:.4f}")

# Probe for chroma (pitch class)
print("\n--- Probing for chroma (pitch class) ---")
for name, X in [('z_raw', z_flat), ('conv_in', conv_in_flat),
                ('attn', attn_flat), ('up_block_2', up2_flat)]:
    if X.shape[0] != chroma_2d.shape[0]:
        continue
    result = train_mlp_probe(X, chroma_2d)
    print(f"  {name:12s}: R² = {result['r2']:.4f}, loss = {result['test_loss']:.4f}")

# Probe for amplitude (average amplitude of top sines)
print("\n--- Probing for amplitude ---")
top_k = 8
amp_sorted = all_amps.sort(dim=-1, descending=True)[0][:, :, :top_k]
amp_flat = amp_sorted.reshape(-1, top_k)

for name, X in [('z_raw', z_flat), ('conv_in', conv_in_flat),
                ('attn', attn_flat)]:
    if X.shape[0] != amp_flat.shape[0]:
        continue
    result = train_mlp_probe(X, amp_flat)
    print(f"  {name:12s}: R² = {result['r2']:.4f}, loss = {result['test_loss']:.4f}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
R² > 0.5: Feature strongly encodes target (good for tapping)
R² 0.2-0.5: Moderate encoding (might work with more capacity)
R² < 0.2: Weak encoding (probably not useful to tap)

If attn > z_raw: Decoder's temporal processing helps extract pitch
If attn ≈ z_raw: Decoder doesn't add much for pitch (stick with z directly)
If up_block_2 > attn: Information gets clearer after upsampling
""")
