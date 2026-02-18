#!/usr/bin/env python3
"""
Probe DCAE decoder to discover where sine parameters are encoded.

Question: Can we linearly decode freq/amp/phase from decoder activations?
If yes → decoder explicitly represents sine params at that layer
If no → representation is distributed/nonlinear

This tells us WHERE the decoder encodes pitch/amplitude/phase,
and whether we can extract sparse sine operations from it.
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
print("Probing DCAE Decoder for Sine Representations")
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

# Collect samples
samples = []
for entry in manifest['entries'][:200]:  # First 200 samples
    try:
        sms = torch.load(entry['path'], weights_only=True, map_location='cpu')
        lat = torch.load(entry['latent_path'], weights_only=True, map_location='cpu')
        latent = lat.get('latents', lat.get('latent'))

        # Skip drums
        if any(kw in sms.get('audio_path', '').lower()
               for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            continue

        # Crop to 22 frames
        T = min(latent.shape[-1], 22)
        latent = latent[:, :, :T]
        freqs = sms['freqs'][:T]
        amps = sms['amps'][:T] * 10.0  # Scale

        if T < 22:
            continue

        samples.append({
            'latent': latent,  # [8, 16, T]
            'freqs': freqs,    # [T, n_sines]
            'amps': amps,
        })
    except:
        continue

print(f"Loaded {len(samples)} samples")

# Collect activations
print("\nCollecting decoder activations...")

activations = {
    'conv_in': [],
    'attention': [],
}

def hook_conv_in(module, input, output):
    activations['conv_in'].append(output.detach().cpu())

def hook_attention(module, input, output):
    activations['attention'].append(output.detach().cpu())

h1 = decoder.conv_in.register_forward_hook(hook_conv_in)
h2 = decoder.up_blocks[3].register_forward_hook(hook_attention)

# Collect target sine params
all_freqs = []
all_amps = []

with torch.no_grad():
    for i, sample in enumerate(samples[:100]):  # First 100 for probing
        z = sample['latent'].unsqueeze(0)  # [1, 8, 16, T]
        z = z.permute(0, 1, 3, 2)  # [1, 8, T, 16] for decoder

        _ = decoder(z)

        all_freqs.append(sample['freqs'])
        all_amps.append(sample['amps'])

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1} samples")

h1.remove()
h2.remove()

# Stack activations
print("\nProcessing activations...")
conv_in_acts = torch.cat(activations['conv_in'], dim=0)  # [N, 1024, T, 16]
attention_acts = torch.cat(activations['attention'], dim=0)  # [N, 1024, T, 16]

all_freqs = torch.stack(all_freqs)  # [N, T, n_sines]
all_amps = torch.stack(all_amps)

N, T, n_sines = all_freqs.shape
print(f"conv_in activations: {conv_in_acts.shape}")
print(f"attention activations: {attention_acts.shape}")
print(f"Target freqs: {all_freqs.shape}")

# Flatten activations per frame
# [N, 1024, T, 16] -> [N*T, 1024*16]
conv_in_flat = conv_in_acts.permute(0, 2, 1, 3).reshape(N * T, -1)
attention_flat = attention_acts.permute(0, 2, 1, 3).reshape(N * T, -1)

# Flatten targets per frame
# [N, T, n_sines] -> [N*T, n_sines]
freqs_flat = all_freqs.reshape(N * T, -1)
amps_flat = all_amps.reshape(N * T, -1)

# Use log frequency for better scale
log_freqs_flat = torch.log(freqs_flat.clamp(min=20))

print(f"\nFlatted shapes:")
print(f"  conv_in: {conv_in_flat.shape}")
print(f"  attention: {attention_flat.shape}")
print(f"  log_freqs: {log_freqs_flat.shape}")

# ============================================================
# Train linear probes
# ============================================================

def train_probe(activations, targets, name, n_epochs=100):
    """Train linear probe: activations -> targets"""
    # Use subset for training, rest for testing
    n_train = int(0.8 * activations.shape[0])

    X_train = activations[:n_train]
    X_test = activations[n_train:]
    y_train = targets[:n_train]
    y_test = targets[n_train:]

    # Normalize
    X_mean = X_train.mean(dim=0)
    X_std = X_train.std(dim=0).clamp(min=1e-6)
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # Linear probe
    input_dim = activations.shape[1]
    output_dim = targets.shape[1]

    probe = nn.Linear(input_dim, output_dim)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    # Train
    probe.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        pred = probe(X_train_norm)
        loss = F.mse_loss(pred, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    probe.eval()
    with torch.no_grad():
        train_pred = probe(X_train_norm)
        train_loss = F.mse_loss(train_pred, y_train).item()

        test_pred = probe(X_test_norm)
        test_loss = F.mse_loss(test_pred, y_test).item()

        # R² score
        ss_res = ((test_pred - y_test) ** 2).sum()
        ss_tot = ((y_test - y_test.mean(dim=0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot.clamp(min=1e-6)

    return {
        'train_loss': train_loss,
        'test_loss': test_loss,
        'r2': r2.item(),
    }

print("\n" + "=" * 70)
print("Linear Probe Results")
print("=" * 70)

# Probe for log-frequency (top 16 sines by amplitude)
top_k = 16
amp_sorted_idx = amps_flat.argsort(dim=-1, descending=True)[:, :top_k]
log_freqs_topk = torch.gather(log_freqs_flat, -1, amp_sorted_idx)

print(f"\nProbing for log-frequency (top {top_k} sines by amplitude):")
print("-" * 50)

result = train_probe(conv_in_flat, log_freqs_topk, "conv_in -> log_freq")
print(f"conv_in -> log_freq:    R² = {result['r2']:.4f}, test_loss = {result['test_loss']:.4f}")

result = train_probe(attention_flat, log_freqs_topk, "attention -> log_freq")
print(f"attention -> log_freq:  R² = {result['r2']:.4f}, test_loss = {result['test_loss']:.4f}")

# Probe for amplitude
amps_topk = torch.gather(amps_flat, -1, amp_sorted_idx)

print(f"\nProbing for amplitude (top {top_k} sines):")
print("-" * 50)

result = train_probe(conv_in_flat, amps_topk, "conv_in -> amp")
print(f"conv_in -> amp:         R² = {result['r2']:.4f}, test_loss = {result['test_loss']:.4f}")

result = train_probe(attention_flat, amps_topk, "attention -> amp")
print(f"attention -> amp:       R² = {result['r2']:.4f}, test_loss = {result['test_loss']:.4f}")

# Probe for just f0 (fundamental)
# Get the lowest strong frequency per frame
f0_per_frame = []
for i in range(freqs_flat.shape[0]):
    mask = amps_flat[i] > 0.1
    if mask.sum() > 0:
        active_freqs = freqs_flat[i][mask]
        f0 = active_freqs.min()
    else:
        f0 = torch.tensor(100.0)  # Default
    f0_per_frame.append(f0)

f0_flat = torch.stack(f0_per_frame).unsqueeze(-1)
log_f0_flat = torch.log(f0_flat.clamp(min=20))

print(f"\nProbing for fundamental frequency (f0):")
print("-" * 50)

result = train_probe(conv_in_flat, log_f0_flat, "conv_in -> log_f0")
print(f"conv_in -> log_f0:      R² = {result['r2']:.4f}, test_loss = {result['test_loss']:.4f}")

result = train_probe(attention_flat, log_f0_flat, "attention -> log_f0")
print(f"attention -> log_f0:    R² = {result['r2']:.4f}, test_loss = {result['test_loss']:.4f}")

# ============================================================
# Interpretation
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
R² interpretation:
  R² > 0.8  = Strong linear encoding (can extract directly)
  R² 0.5-0.8 = Moderate encoding (partially linear)
  R² < 0.5  = Weak/nonlinear encoding (distributed representation)

If attention layer has higher R² than conv_in for frequency:
  → Attention is where frequency information gets "resolved"
  → Temporal mixing is important for pitch

If conv_in already has high R² for frequency:
  → Frequency is encoded directly in z, attention just refines it
""")
