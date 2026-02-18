#!/usr/bin/env python3
"""
Extract physical mode structure from DCAE latent space.

Based on probe results:
1. Group dims by coupling → identify modes
2. Find energy conservation channels
3. Find phase lock relationships
4. Output mode definitions for UI
"""

import torch
import numpy as np
from collections import defaultdict
import json
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson


def load_latents(manifest_path, n_samples=200, device='cuda'):
    """Load latents from dataset."""
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
            z = lat_data.get('latents', lat_data) if isinstance(lat_data, dict) else lat_data
            if z.dim() == 4:
                z = z.squeeze(0)

            # Take middle frames
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


def compute_correlation_matrix(latents):
    """Compute correlation matrix across all latents."""
    z_all = torch.stack([z.reshape(128, -1).mean(dim=-1) for z in latents])
    z_centered = z_all - z_all.mean(dim=0, keepdim=True)
    z_std = z_all.std(dim=0, keepdim=True).clamp(min=1e-6)
    z_norm = z_centered / z_std
    corr = (z_norm.T @ z_norm) / len(latents)
    return corr.cpu().numpy()


def extract_coupled_modes(corr, threshold=0.85):
    """
    Group dims into modes based on coupling strength.
    Dims with correlation > threshold are in the same mode.
    """
    n_dims = corr.shape[0]
    visited = set()
    modes = []

    for i in range(n_dims):
        if i in visited:
            continue

        # BFS to find all dims coupled to i
        mode = {i}
        queue = [i]
        visited.add(i)

        while queue:
            current = queue.pop(0)
            for j in range(n_dims):
                if j not in visited and corr[current, j] > threshold:
                    mode.add(j)
                    visited.add(j)
                    queue.append(j)

        if len(mode) >= 2:  # Only keep non-trivial modes
            modes.append(sorted(mode))

    # Sort modes by first dim
    modes.sort(key=lambda m: m[0])
    return modes


def extract_energy_channels(corr, threshold=-0.7):
    """
    Find pairs of dim groups that anti-correlate (energy conservation).
    """
    n_dims = corr.shape[0]
    channels = []

    # Find strongly anti-correlated pairs
    anti_pairs = []
    for i in range(n_dims):
        for j in range(i + 1, n_dims):
            if corr[i, j] < threshold:
                anti_pairs.append((i, j, corr[i, j]))

    # Group into channels
    # A channel is: source_dims <-> sink_dims where all cross-correlations are negative
    used = set()
    for i, j, r in sorted(anti_pairs, key=lambda x: x[2]):
        if i in used or j in used:
            continue

        # Find all dims that anti-correlate with j
        source_group = {i}
        sink_group = {j}

        for k in range(n_dims):
            if k == i or k == j:
                continue
            # k correlates with i and anti-correlates with j?
            if corr[i, k] > 0.7 and corr[j, k] < threshold:
                source_group.add(k)
            # k correlates with j and anti-correlates with i?
            elif corr[j, k] > 0.7 and corr[i, k] < threshold:
                sink_group.add(k)

        if len(source_group) >= 2 or len(sink_group) >= 2:
            channels.append({
                'source': sorted(source_group),
                'sink': sorted(sink_group),
                'strength': abs(r)
            })
            used.update(source_group)
            used.update(sink_group)

    return channels


def extract_phase_locks(latents, ratio_threshold=0.1):
    """
    Find dim pairs with stable ratios (phase locking).
    """
    z_all = torch.stack([z.reshape(128, -1).mean(dim=-1) for z in latents])
    z_np = z_all.cpu().numpy()

    phase_locks = []

    for i in range(128):
        for j in range(i + 1, 128):
            z_i = z_np[:, i]
            z_j = z_np[:, j]

            valid = np.abs(z_j) > 0.01
            if np.sum(valid) < 20:
                continue

            ratios = z_i[valid] / z_j[valid]
            ratio_std = np.std(ratios)
            ratio_mean = np.mean(ratios)

            if ratio_std < ratio_threshold and abs(ratio_mean) > 0.05:
                phase_locks.append({
                    'dims': (i, j),
                    'ratio': float(ratio_mean),
                    'stability': float(1.0 / (ratio_std + 0.01))
                })

    # Sort by stability
    phase_locks.sort(key=lambda x: -x['stability'])
    return phase_locks[:30]  # Top 30


def analyze_mode_function(dcae, modes, base_z, device='cuda'):
    """
    For each mode, analyze what it controls in the output.
    """
    from collections import defaultdict

    def get_spectral_features(mel):
        mel_power = torch.exp(mel.mean(dim=1))  # [B, 128, T]
        freq_bins = torch.arange(128, device=mel.device).float()

        features = {}
        total = mel_power.sum(dim=1, keepdim=True).clamp(min=1e-8)

        # Centroid
        features['centroid'] = (mel_power * freq_bins.view(1, -1, 1) / total).sum(dim=1).mean().item()

        # Band energies
        features['low'] = mel_power[:, :32, :].sum().item()
        features['mid'] = mel_power[:, 32:96, :].sum().item()
        features['high'] = mel_power[:, 96:, :].sum().item()
        features['total'] = mel_power.sum().item()

        return features

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, 128, T)

    # Baseline
    with torch.no_grad():
        z_denorm = base_z / dcae.scale_factor + dcae.shift_factor
        base_mel = dcae.dcae.decoder(z_denorm)
    base_features = get_spectral_features(base_mel)

    mode_functions = []

    for mode_idx, mode_dims in enumerate(modes[:10]):  # Analyze top 10 modes
        # Perturb all dims in the mode together
        z_perturbed = z_flat.clone()
        for dim in mode_dims:
            z_perturbed[:, dim, :] += 0.5

        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)
        with torch.no_grad():
            z_denorm = z_perturbed_4d / dcae.scale_factor + dcae.shift_factor
            perturbed_mel = dcae.dcae.decoder(z_denorm)
        perturbed_features = get_spectral_features(perturbed_mel)

        # Compute effects
        effects = {}
        for key in base_features:
            effects[key] = perturbed_features[key] - base_features[key]

        # Determine primary function
        max_effect = max(effects.items(), key=lambda x: abs(x[1]))

        mode_functions.append({
            'mode_id': mode_idx,
            'dims': mode_dims,
            'primary_effect': max_effect[0],
            'effect_magnitude': max_effect[1],
            'all_effects': effects
        })

    return mode_functions


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    print("\nLoading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8",
        vocoder_checkpoint_path="/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"
    )
    dcae.dcae.to(device).eval()

    print("\nLoading latents...")
    latents = load_latents(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        n_samples=200,
        device=device
    )
    print(f"  Loaded {len(latents)} latents")

    print("\nComputing correlation matrix...")
    corr = compute_correlation_matrix(latents)

    # Extract structures
    print("\n" + "=" * 70)
    print("EXTRACTING PHYSICAL MODE STRUCTURE")
    print("=" * 70)

    # 1. Coupled modes
    print("\n1. COUPLED MODES (dims that move together = one physical mode)")
    modes = extract_coupled_modes(corr, threshold=0.95)  # Higher threshold for finer modes
    print(f"   Found {len(modes)} coupled modes:")
    for i, mode in enumerate(modes[:15]):
        print(f"   Mode {i}: dims {mode[:10]}{'...' if len(mode) > 10 else ''} ({len(mode)} dims)")

    # 2. Energy channels
    print("\n2. ENERGY CONSERVATION CHANNELS")
    channels = extract_energy_channels(corr, threshold=-0.7)
    print(f"   Found {len(channels)} energy channels:")
    for i, ch in enumerate(channels[:10]):
        print(f"   Channel {i}: {ch['source'][:5]} <-> {ch['sink'][:5]} (r={-ch['strength']:.2f})")

    # 3. Phase locks
    print("\n3. PHASE-LOCKED PAIRS")
    phase_locks = extract_phase_locks(latents, ratio_threshold=0.08)
    print(f"   Found {len(phase_locks)} phase-locked pairs:")
    for pl in phase_locks[:10]:
        print(f"   dims {pl['dims']}: ratio={pl['ratio']:.3f}, stability={pl['stability']:.1f}")

    # 4. Analyze mode functions
    print("\n4. MODE FUNCTIONS (what each mode controls)")
    base_z = latents[0].unsqueeze(0)
    mode_functions = analyze_mode_function(dcae, modes, base_z, device)
    for mf in mode_functions:
        print(f"   Mode {mf['mode_id']} ({len(mf['dims'])} dims): {mf['primary_effect']} ({mf['effect_magnitude']:+.2f})")

    # Save mode definitions (convert numpy types to python native)
    def to_native(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_native(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_native(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(to_native(v) for v in obj)
        return obj

    mode_definitions = to_native({
        'modes': [{'id': i, 'dims': m} for i, m in enumerate(modes)],
        'energy_channels': channels,
        'phase_locks': phase_locks,
        'mode_functions': mode_functions
    })

    output_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/physical_modes.json'
    with open(output_path, 'w') as f:
        json.dump(mode_definitions, f, indent=2)
    print(f"\n   Saved mode definitions to {output_path}")

    # Print summary for UI
    print("\n" + "=" * 70)
    print("UI CONTROL MAPPING")
    print("=" * 70)
    print("""
Based on physical mode structure, the UI should have:

1. MODE SLIDERS (not dim sliders):
   - Each slider controls an entire coupled mode
   - Moving one mode affects all its dims together
   - Preserves physical coupling

2. ENERGY BALANCE CONTROLS:
   - Paired sliders for energy channels
   - When source increases, sink decreases (linked)
   - Maintains energy conservation

3. PHASE LOCK GROUPS:
   - Dims with locked ratios move together
   - User controls the group, ratios are maintained

4. RESONATOR CONTROLS:
   - Excitation (how much energy into the mode)
   - Damping (how quickly it decays)
   - These map to physical oscillator behavior
""")


if __name__ == "__main__":
    main()
