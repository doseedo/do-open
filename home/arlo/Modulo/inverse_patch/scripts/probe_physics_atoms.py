#!/usr/bin/env python3
"""
Probe for PHYSICAL atoms in DCAE latent space.

NOT looking for: "brightness", "envelope" (our mental model)
LOOKING FOR: actual physics the model learned

Physical atoms might be:
- Coupled oscillator modes (dims that ring together)
- Energy channels (dims that trade off - conservation)
- Phase relationships (dims with fixed ratios)
- Nonlinear resonances (dims that interact multiplicatively)
- Damping modes (dims that decay together)

Probe strategies:
1. Correlation structure: which dims move together across sounds?
2. Energy conservation: which dims trade off (sum conserved)?
3. Phase coupling: which dims maintain fixed phase relationships?
4. Dynamical systems: perturb and watch evolution over time
5. Nonlinear interactions: effect(A+B) vs effect(A) + effect(B)
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson


def load_dcae(device='cuda'):
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    dcae = MusicDCAE(dcae_checkpoint_path=DCAE_PATH, vocoder_checkpoint_path=VOCODER_PATH)
    dcae.dcae.to(device).eval()
    return dcae


def get_mel_from_z(dcae, z):
    with torch.no_grad():
        z_denorm = z / dcae.scale_factor + dcae.shift_factor
        mel = dcae.dcae.decoder(z_denorm)
        return mel.mean(dim=1)  # [B, 128, T]


def load_real_latents(manifest_path, n_samples=100, device='cuda'):
    """Load real latents from dataset."""
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    latents = []
    for entry in manifest['entries'][:n_samples * 2]:
        try:
            data = torch.load(entry['path'], weights_only=True, map_location='cpu')
            lat_path = data.get('latent_path')
            if not lat_path or not os.path.exists(lat_path):
                continue

            lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
            if isinstance(lat_data, dict):
                z = lat_data.get('latents', lat_data)
            else:
                z = lat_data

            if z.dim() == 4:
                z = z.squeeze(0)

            # Take middle 16 frames
            T = z.shape[-1]
            if T > 16:
                start = (T - 16) // 2
                z = z[..., start:start + 16]

            latents.append(z.to(device))

            if len(latents) >= n_samples:
                break
        except:
            continue

    return latents


# ============================================================================
# PROBE 1: Correlation Structure (what dims move together?)
# ============================================================================

def probe_correlation_structure(latents):
    """
    Find dims that correlate across different sounds.
    These might be coupled physical modes.
    """
    print("\n" + "=" * 70)
    print("PROBE 1: Correlation Structure")
    print("Which dims move together across different sounds?")
    print("=" * 70)

    # Stack all latents and flatten to [n_samples, 128, T]
    z_all = torch.stack([z.reshape(128, -1).mean(dim=-1) for z in latents])  # [N, 128]

    # Compute correlation matrix
    z_centered = z_all - z_all.mean(dim=0, keepdim=True)
    z_std = z_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_norm = z_centered / z_std

    corr = (z_norm.T @ z_norm) / len(latents)  # [128, 128]
    corr = corr.cpu().numpy()

    # Find highly correlated dim pairs (excluding diagonal)
    np.fill_diagonal(corr, 0)
    high_corr_pairs = []

    for i in range(128):
        for j in range(i + 1, 128):
            if abs(corr[i, j]) > 0.7:
                high_corr_pairs.append((i, j, corr[i, j]))

    high_corr_pairs.sort(key=lambda x: -abs(x[2]))

    print(f"\nHighly correlated dim pairs (|r| > 0.7):")
    for i, j, r in high_corr_pairs[:20]:
        sign = "+" if r > 0 else "-"
        print(f"  dims ({i:3d}, {j:3d}): r = {sign}{abs(r):.3f}")

    # Find clusters of correlated dims
    print("\n\nFinding correlated clusters...")
    from sklearn.cluster import AgglomerativeClustering

    # Use 1 - |corr| as distance
    dist = 1 - np.abs(corr)
    clustering = AgglomerativeClustering(n_clusters=8, metric='precomputed', linkage='average')
    labels = clustering.fit_predict(dist)

    clusters = defaultdict(list)
    for dim, label in enumerate(labels):
        clusters[label].append(dim)

    print(f"\nCorrelation clusters:")
    for label in sorted(clusters.keys()):
        dims = clusters[label]
        if len(dims) > 3:
            print(f"  Cluster {label}: {dims[:15]}{'...' if len(dims) > 15 else ''} ({len(dims)} dims)")

    return corr, clusters


# ============================================================================
# PROBE 2: Energy Conservation (which dims trade off?)
# ============================================================================

def probe_energy_conservation(latents):
    """
    Find dims that trade off against each other.
    When one goes up, another goes down - conserved quantity.
    """
    print("\n" + "=" * 70)
    print("PROBE 2: Energy Conservation")
    print("Which dims trade off (negative correlation = conservation)?")
    print("=" * 70)

    z_all = torch.stack([z.reshape(128, -1).mean(dim=-1) for z in latents])

    # Find anti-correlated pairs
    z_centered = z_all - z_all.mean(dim=0, keepdim=True)
    z_std = z_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_norm = z_centered / z_std
    corr = (z_norm.T @ z_norm) / len(latents)
    corr = corr.cpu().numpy()

    # Strong negative correlations suggest energy trading
    neg_corr_pairs = []
    for i in range(128):
        for j in range(i + 1, 128):
            if corr[i, j] < -0.5:
                neg_corr_pairs.append((i, j, corr[i, j]))

    neg_corr_pairs.sort(key=lambda x: x[2])

    print(f"\nAnti-correlated pairs (r < -0.5) - possible energy conservation:")
    for i, j, r in neg_corr_pairs[:15]:
        print(f"  dims ({i:3d}, {j:3d}): r = {r:.3f}")

    # Check if sums are conserved
    print("\n\nChecking for conserved sums...")
    for i, j, r in neg_corr_pairs[:5]:
        z_i = z_all[:, i].cpu().numpy()
        z_j = z_all[:, j].cpu().numpy()
        sum_ij = z_i + z_j
        print(f"  dims ({i}, {j}): sum variance = {np.var(sum_ij):.4f} (lower = more conserved)")

    return neg_corr_pairs


# ============================================================================
# PROBE 3: Dynamical Systems (how do perturbations evolve?)
# ============================================================================

def probe_dynamical_systems(dcae, base_z, device='cuda'):
    """
    Perturb a dim and watch how the effect evolves over time.
    Looking for: oscillations, damping, resonance.
    """
    print("\n" + "=" * 70)
    print("PROBE 3: Dynamical Systems")
    print("How do perturbations evolve over time?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, 128, T)

    # Get baseline
    base_mel = get_mel_from_z(dcae, base_z)
    base_energy = torch.exp(base_mel).sum(dim=1)  # [B, T_mel]

    # Test a few dims
    test_dims = [48, 56, 62, 64, 80, 96]

    print("\nTemporal response to impulse perturbation:")

    for dim in test_dims:
        # Perturb only at t=T//2 (impulse)
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, T // 2] += 1.0
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d)
        perturbed_energy = torch.exp(perturbed_mel).sum(dim=1)

        diff = (perturbed_energy - base_energy).squeeze().cpu().numpy()
        T_mel = len(diff)
        t_impulse = (T // 2) * 8  # Mel is 8x upsampled

        # Analyze response
        pre_impulse = diff[:t_impulse]
        post_impulse = diff[t_impulse:]

        pre_energy = np.sum(np.abs(pre_impulse))
        post_energy = np.sum(np.abs(post_impulse))

        # Check for oscillation (zero crossings)
        zero_crossings = np.sum(np.diff(np.sign(post_impulse)) != 0)

        # Check for damping (decay rate)
        if len(post_impulse) > 10:
            early = np.mean(np.abs(post_impulse[:10]))
            late = np.mean(np.abs(post_impulse[-10:]))
            damping = late / (early + 1e-8)
        else:
            damping = 1.0

        print(f"\n  Dim {dim}:")
        print(f"    Pre-impulse energy:  {pre_energy:.4f}")
        print(f"    Post-impulse energy: {post_energy:.4f}")
        print(f"    Zero crossings:      {zero_crossings} (oscillation indicator)")
        print(f"    Damping ratio:       {damping:.3f} (1=sustained, 0=decayed)")

        # Classify behavior
        if zero_crossings > 5 and damping > 0.3:
            print(f"    → OSCILLATOR (rings, sustains)")
        elif zero_crossings > 5 and damping < 0.3:
            print(f"    → DAMPED OSCILLATOR")
        elif damping > 0.8:
            print(f"    → SUSTAINED RESPONSE")
        else:
            print(f"    → IMPULSE RESPONSE (decays)")


# ============================================================================
# PROBE 4: Nonlinear Interactions
# ============================================================================

def probe_nonlinear_interactions(dcae, base_z, device='cuda'):
    """
    Test: effect(A+B) vs effect(A) + effect(B)
    Deviation = nonlinear interaction = coupled physics
    """
    print("\n" + "=" * 70)
    print("PROBE 4: Nonlinear Interactions")
    print("Does effect(A+B) = effect(A) + effect(B)?")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, 128, T)

    base_mel = get_mel_from_z(dcae, base_z)
    base_energy = torch.exp(base_mel).sum(dim=1).mean().item()

    # Test pairs of dims
    dim_pairs = [(48, 56), (56, 64), (60, 62), (48, 80), (64, 96)]
    delta = 0.5

    print(f"\nTesting dim pairs for interaction (δ = {delta}):")
    print(f"{'Pair':<15} {'E(A)':<10} {'E(B)':<10} {'E(A+B)':<10} {'E(A)+E(B)':<12} {'Interaction':<12}")
    print("-" * 70)

    interactions = []

    for dim_a, dim_b in dim_pairs:
        # Effect of A alone
        z_a = z_flat.clone()
        z_a[:, dim_a, :] += delta
        mel_a = get_mel_from_z(dcae, z_a.reshape(B, C, H, T))
        effect_a = torch.exp(mel_a).sum(dim=1).mean().item() - base_energy

        # Effect of B alone
        z_b = z_flat.clone()
        z_b[:, dim_b, :] += delta
        mel_b = get_mel_from_z(dcae, z_b.reshape(B, C, H, T))
        effect_b = torch.exp(mel_b).sum(dim=1).mean().item() - base_energy

        # Effect of A+B together
        z_ab = z_flat.clone()
        z_ab[:, dim_a, :] += delta
        z_ab[:, dim_b, :] += delta
        mel_ab = get_mel_from_z(dcae, z_ab.reshape(B, C, H, T))
        effect_ab = torch.exp(mel_ab).sum(dim=1).mean().item() - base_energy

        # Interaction = deviation from additivity
        expected = effect_a + effect_b
        interaction = effect_ab - expected
        rel_interaction = abs(interaction) / (abs(expected) + 1e-6)

        interactions.append((dim_a, dim_b, rel_interaction))

        print(f"({dim_a:3d}, {dim_b:3d})     {effect_a:+.4f}    {effect_b:+.4f}    {effect_ab:+.4f}    {expected:+.4f}      {interaction:+.4f} ({rel_interaction:.0%})")

    # Find strongest interactions
    interactions.sort(key=lambda x: -x[2])
    print(f"\n→ Strongest interactions indicate coupled physical modes")


# ============================================================================
# PROBE 5: Phase Relationships
# ============================================================================

def probe_phase_relationships(latents):
    """
    Look for dims that maintain fixed ratios (phase locking).
    """
    print("\n" + "=" * 70)
    print("PROBE 5: Phase Relationships")
    print("Which dims maintain fixed ratios?")
    print("=" * 70)

    z_all = torch.stack([z.reshape(128, -1).mean(dim=-1) for z in latents])
    z_np = z_all.cpu().numpy()

    # For each pair, compute ratio stability
    ratio_stability = []

    for i in range(128):
        for j in range(i + 1, 128):
            z_i = z_np[:, i]
            z_j = z_np[:, j]

            # Avoid division by zero
            valid = np.abs(z_j) > 0.01
            if np.sum(valid) < 10:
                continue

            ratios = z_i[valid] / z_j[valid]
            ratio_std = np.std(ratios)
            ratio_mean = np.mean(ratios)

            # Low std = stable ratio = phase locked
            if ratio_std < 0.5 and abs(ratio_mean) > 0.1:
                ratio_stability.append((i, j, ratio_mean, ratio_std))

    ratio_stability.sort(key=lambda x: x[3])

    print(f"\nDim pairs with stable ratios (potential phase locking):")
    for i, j, mean, std in ratio_stability[:15]:
        print(f"  dims ({i:3d}, {j:3d}): ratio = {mean:.3f} ± {std:.3f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading DCAE...")
    dcae = load_dcae(device)

    print("\nLoading real latents from dataset...")
    latents = load_real_latents(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        n_samples=100,
        device=device
    )
    print(f"  Loaded {len(latents)} latents")

    # Create base z for perturbation experiments
    if latents:
        base_z = latents[0].unsqueeze(0)  # Use real sample
    else:
        base_z = torch.randn(1, 8, 16, 16, device=device) * 0.1

    # Run probes
    corr, clusters = probe_correlation_structure(latents)
    neg_pairs = probe_energy_conservation(latents)
    probe_dynamical_systems(dcae, base_z, device)
    probe_nonlinear_interactions(dcae, base_z, device)
    probe_phase_relationships(latents)

    print("\n" + "=" * 70)
    print("SUMMARY: Physical Atoms Discovered")
    print("=" * 70)
    print("""
These probes reveal the PHYSICAL structure, not our mental model:

1. CORRELATED CLUSTERS = Coupled oscillator modes
   Dims that move together are physically coupled

2. ANTI-CORRELATED PAIRS = Energy conservation
   When one dim up, another down = conserved quantity

3. DYNAMICAL RESPONSE = Resonator behavior
   Oscillating response = resonant mode
   Damped response = dissipative mode

4. NONLINEAR INTERACTIONS = Coupled physics
   Deviation from additivity = physical coupling

5. STABLE RATIOS = Phase locking
   Fixed ratios across sounds = locked modes

The atoms aren't "brightness" and "envelope".
They're coupled oscillators, resonant modes, and energy channels.
The entanglement IS the physics.
    """)


if __name__ == "__main__":
    main()
