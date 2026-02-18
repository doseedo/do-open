#!/usr/bin/env python3
"""
Diagnose what the DCAE latent z actually encodes.
Does perturbing z change predicted frequencies systematically?
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'training'))
from train_sms_hybrid import HybridSAMIMapper, HybridSMSDataset

# Load trained mapper
print("Loading trained mapper...")
ckpt = torch.load('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/checkpoints/sms_hybrid/best_model.pt',
                  weights_only=True, map_location='cpu')
n_sines = ckpt['n_sines']
n_noise_bands = ckpt['n_noise_bands']

mapper = HybridSAMIMapper(n_sines=n_sines, n_noise_bands=n_noise_bands)
mapper.load_state_dict(ckpt['model_state_dict'])
mapper.eval()
print(f"  Loaded mapper with {n_sines} sines")

# Load a few samples
print("\nLoading test samples...")
dataset = HybridSMSDataset(
    '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
    max_samples=100,
    skip_drums=True,
    n_sines=n_sines,
)

# Test 1: Channel perturbation sensitivity
print("\n" + "="*70)
print("TEST 1: Channel Perturbation Sensitivity")
print("="*70)
print("Which channels affect frequency predictions most?")

perturbation_strength = 1.0
results = {i: [] for i in range(8)}

for sample_idx in range(min(50, len(dataset))):
    sample = dataset[sample_idx]
    z = sample['latent'].unsqueeze(0)  # [1, 8, 16, T]

    with torch.no_grad():
        base_pred = mapper(z)
        base_freqs = base_pred['freqs']  # [1, T, n_sines]
        base_amps = base_pred['amps']

        # Get dominant frequencies (top 5 by amplitude)
        top_k = 5
        top_idx = base_amps[0].mean(dim=0).argsort(descending=True)[:top_k]
        base_top_freqs = base_freqs[0, :, top_idx].mean(dim=0)  # [top_k]

        for channel in range(8):
            z_perturbed = z.clone()
            z_perturbed[:, channel, :, :] += perturbation_strength

            perturbed_pred = mapper(z_perturbed)
            perturbed_freqs = perturbed_pred['freqs']
            perturbed_top_freqs = perturbed_freqs[0, :, top_idx].mean(dim=0)

            # Frequency ratio (log scale)
            freq_ratio = (perturbed_top_freqs / base_top_freqs).log2().mean().item()
            results[channel].append(freq_ratio)

print(f"\nFrequency shift (octaves) when perturbing each channel by +{perturbation_strength}:")
print("  Channel  |  Shift  |  Interpretation")
print("-" * 50)
for ch in range(8):
    shift = np.mean(results[ch])
    std = np.std(results[ch])
    label = "coarse" if ch < 4 else "fine"
    interpretation = ""
    if abs(shift) > 0.1:
        interpretation = "<<< Affects pitch!"
    elif abs(shift) > 0.05:
        interpretation = "< Mild effect"
    print(f"    {ch} ({label})  |  {shift:+.3f}  |  {interpretation}")

# Test 2: Does z encode pitch consistently?
print("\n" + "="*70)
print("TEST 2: Pitch Encoding Consistency")
print("="*70)
print("Do samples with similar target f0 have similar z structure?")

# Group samples by estimated f0
f0_groups = {'low': [], 'mid': [], 'high': []}
for i in range(min(100, len(dataset))):
    sample = dataset[i]
    freqs = sample['freqs']
    amps = sample['amps']

    # Estimate f0 from lowest high-amplitude frequency
    active_mask = amps > 0.05
    if active_mask.sum() < 1:
        continue

    # Get average f0 across frames
    active_freqs = []
    for t in range(freqs.shape[0]):
        if active_mask[t].sum() > 0:
            frame_freqs = freqs[t][active_mask[t]]
            if len(frame_freqs) > 0:
                active_freqs.append(frame_freqs.min().item())

    if len(active_freqs) == 0:
        continue

    f0 = np.median(active_freqs)

    if f0 < 150:
        f0_groups['low'].append((i, f0, sample['latent']))
    elif f0 < 400:
        f0_groups['mid'].append((i, f0, sample['latent']))
    else:
        f0_groups['high'].append((i, f0, sample['latent']))

print(f"\nSamples by f0 range: low={len(f0_groups['low'])}, mid={len(f0_groups['mid'])}, high={len(f0_groups['high'])}")

# Compare z structure within and across groups
def compute_z_similarity(z1, z2):
    """Cosine similarity of flattened z"""
    z1_flat = z1.flatten()
    z2_flat = z2.flatten()
    return F.cosine_similarity(z1_flat.unsqueeze(0), z2_flat.unsqueeze(0)).item()

within_sims = []
across_sims = []

groups = ['low', 'mid', 'high']
for group in groups:
    samples = f0_groups[group]
    if len(samples) < 2:
        continue
    # Within-group similarity
    for i in range(min(5, len(samples))):
        for j in range(i+1, min(5, len(samples))):
            sim = compute_z_similarity(samples[i][2], samples[j][2])
            within_sims.append((group, sim))

# Across-group similarity
for g1, g2 in [('low', 'high'), ('low', 'mid'), ('mid', 'high')]:
    s1 = f0_groups[g1]
    s2 = f0_groups[g2]
    if len(s1) < 1 or len(s2) < 1:
        continue
    for i in range(min(3, len(s1))):
        for j in range(min(3, len(s2))):
            sim = compute_z_similarity(s1[i][2], s2[j][2])
            across_sims.append((f"{g1}-{g2}", sim))

print(f"\nWithin-group z similarity (same f0 range):")
for group in groups:
    group_sims = [s for g, s in within_sims if g == group]
    if group_sims:
        print(f"  {group}: {np.mean(group_sims):.3f} +/- {np.std(group_sims):.3f}")

print(f"\nAcross-group z similarity (different f0 ranges):")
for pair in ['low-mid', 'mid-high', 'low-high']:
    pair_sims = [s for p, s in across_sims if p == pair]
    if pair_sims:
        print(f"  {pair}: {np.mean(pair_sims):.3f} +/- {np.std(pair_sims):.3f}")

if within_sims and across_sims:
    within_avg = np.mean([s for _, s in within_sims])
    across_avg = np.mean([s for _, s in across_sims])
    if within_avg > across_avg + 0.05:
        print("\n>>> z structure IS correlated with pitch (within > across)")
    else:
        print("\n>>> z structure NOT clearly correlated with pitch")

# Test 3: SMS target quality check
print("\n" + "="*70)
print("TEST 3: SMS Target Quality")
print("="*70)
print("How clean are the extraction targets?")

freq_stds = []
amp_ranges = []

for i in range(min(50, len(dataset))):
    sample = dataset[i]
    freqs = sample['freqs']  # [T, n_sines]
    amps = sample['amps']

    # For each sine, check temporal stability
    for s in range(n_sines):
        amp_trace = amps[:, s]
        freq_trace = freqs[:, s]

        # Only look at active sines
        active = amp_trace > 0.05
        if active.sum() < 5:
            continue

        active_freqs = freq_trace[active]
        # Coefficient of variation (relative stability)
        freq_std = active_freqs.std().item() / (active_freqs.mean().item() + 1e-6)
        freq_stds.append(freq_std)

print(f"\nFrequency temporal stability (coefficient of variation):")
print(f"  Mean CV: {np.mean(freq_stds):.3f}")
print(f"  Median CV: {np.median(freq_stds):.3f}")
print(f"  90th percentile CV: {np.percentile(freq_stds, 90):.3f}")

if np.median(freq_stds) > 0.1:
    print("  >>> High frame-to-frame frequency variation! Targets may be noisy.")
else:
    print("  >>> Targets are reasonably stable across frames.")

# Test 4: What does the mapper actually predict vs target?
print("\n" + "="*70)
print("TEST 4: Prediction vs Target Analysis")
print("="*70)

errors_by_amp = {'high': [], 'mid': [], 'low': []}

for i in range(min(50, len(dataset))):
    sample = dataset[i]
    z = sample['latent'].unsqueeze(0)
    target_freqs = sample['freqs']
    target_amps = sample['amps']

    with torch.no_grad():
        pred = mapper(z)
        pred_freqs = pred['freqs'][0]  # [T, n_sines]
        pred_amps = pred['amps'][0]

    # Sort by amplitude for matching
    for t in range(target_freqs.shape[0]):
        pred_order = pred_amps[t].argsort(descending=True)
        target_order = target_amps[t].argsort(descending=True)

        pred_f = pred_freqs[t][pred_order]
        target_f = target_freqs[t][target_order]
        target_a = target_amps[t][target_order]

        for s in range(min(10, n_sines)):
            if target_a[s] < 0.01:
                continue

            log_error = abs(np.log2(pred_f[s].item() / max(target_f[s].item(), 20)))

            if target_a[s] > 0.3:
                errors_by_amp['high'].append(log_error)
            elif target_a[s] > 0.1:
                errors_by_amp['mid'].append(log_error)
            else:
                errors_by_amp['low'].append(log_error)

print("\nFrequency error (octaves) by target amplitude:")
for level in ['high', 'mid', 'low']:
    errs = errors_by_amp[level]
    if errs:
        print(f"  {level:>4} amp: {np.mean(errs):.3f} octaves (n={len(errs)})")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
