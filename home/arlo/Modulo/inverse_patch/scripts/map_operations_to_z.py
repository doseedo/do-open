#!/usr/bin/env python3
"""
Map discovered SMS operations to z dimensions.

For each operation type, find which z dims control its parameters:
- HarmonicSeries: which z dims control f0, partial weights?
- DampedOscillator: which z dims control decay rate?
- EnergyConservation: which z dims are involved in conserved quantities?

This creates the z → operations → sines mapping.
"""

import torch
import numpy as np
from collections import defaultdict
import sys
import os
import json

import orjson

sys.path.insert(0, '/home/arlo/Data/ACE-Step')


def load_paired_data(manifest_path, n_samples=100):
    """Load both z latents and SMS data for each sample."""
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    paired = []
    for entry in manifest['entries'][:n_samples * 2]:
        path = entry['path']
        if any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            continue
        if not os.path.exists(path):
            continue

        try:
            data = torch.load(path, weights_only=True, map_location='cpu')
            freqs = data['freqs']
            amps = data['amps']

            # Get latent
            lat_path = data.get('latent_path')
            if not lat_path or not os.path.exists(lat_path):
                continue

            lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
            z = lat_data.get('latents', lat_data) if isinstance(lat_data, dict) else lat_data
            if z.dim() == 4:
                z = z.squeeze(0)

            # Align lengths
            T = min(freqs.shape[0], 64)
            T_z = z.shape[-1]

            paired.append({
                'freqs': freqs[:T],
                'amps': amps[:T],
                'z': z,  # [C, H, T_z] or [8, 16, T_z]
                'path': path
            })

            if len(paired) >= n_samples:
                break
        except:
            continue

    return paired


# ============================================================================
# OPERATION EXTRACTION (from discover_sms_operations.py)
# ============================================================================

def find_harmonic_groups(freqs, amps, f0_threshold=20, ratio_tolerance=0.05):
    """Find groups of sines that form harmonic series."""
    T, n_sines = freqs.shape
    avg_freqs = freqs.mean(dim=0).numpy()
    avg_amps = amps.mean(dim=0).numpy()
    amp_order = np.argsort(avg_amps)[::-1]

    groups = []
    used = set()

    for i in amp_order:
        if i in used or avg_amps[i] < 0.001:
            continue

        f0 = avg_freqs[i]
        if f0 < f0_threshold:
            continue

        group = {'f0': f0, 'partials': [(1, i, avg_amps[i])], 'sine_indices': [i]}
        used.add(i)

        for j in range(n_sines):
            if j in used or avg_amps[j] < 0.0001:
                continue

            fj = avg_freqs[j]
            ratio = fj / f0

            nearest_int = round(ratio)
            if nearest_int >= 2 and nearest_int <= 16:
                if abs(ratio - nearest_int) / nearest_int < ratio_tolerance:
                    group['partials'].append((nearest_int, j, avg_amps[j]))
                    group['sine_indices'].append(j)
                    used.add(j)

        if len(group['partials']) >= 2:
            # Compute harmonic centroid (weighted by amplitude)
            total_amp = sum(a for _, _, a in group['partials'])
            centroid = sum(r * a for r, _, a in group['partials']) / (total_amp + 1e-8)
            group['centroid'] = centroid
            groups.append(group)

    return groups


def find_damped_sines(amps, min_decay=0.01):
    """Find sines that decay over time."""
    T, n_sines = amps.shape
    amps_np = amps.numpy()

    damped = []
    for i in range(n_sines):
        amp_i = amps_np[:, i]
        if amp_i.max() < 0.01:
            continue

        first_half = amp_i[:T//2].mean()
        second_half = amp_i[T//2:].mean()

        if first_half > second_half * 1.5 and first_half > 0.01:
            ratio = second_half / (first_half + 1e-8)
            decay_rate = -np.log(ratio + 1e-8) / (T // 2)

            if decay_rate > min_decay:
                damped.append({
                    'sine_idx': i,
                    'initial_amp': float(first_half),
                    'decay_rate': float(decay_rate)
                })

    return damped


def extract_program(sample):
    """Extract operation program from a sample."""
    freqs, amps = sample['freqs'], sample['amps']

    program = {
        'harmonic_groups': find_harmonic_groups(freqs, amps),
        'damped_sines': find_damped_sines(amps)
    }

    # Extract scalar features
    features = {}

    # Harmonic features
    if program['harmonic_groups']:
        features['n_harmonics'] = len(program['harmonic_groups'])
        features['primary_f0'] = program['harmonic_groups'][0]['f0']
        features['harmonic_centroid'] = np.mean([g['centroid'] for g in program['harmonic_groups']])

        # Total harmonic amplitude
        total_harm_amp = 0
        for g in program['harmonic_groups']:
            total_harm_amp += sum(a for _, _, a in g['partials'])
        features['total_harmonic_amp'] = total_harm_amp
    else:
        features['n_harmonics'] = 0
        features['primary_f0'] = 0
        features['harmonic_centroid'] = 0
        features['total_harmonic_amp'] = 0

    # Damping features
    if program['damped_sines']:
        features['n_damped'] = len(program['damped_sines'])
        features['avg_decay_rate'] = np.mean([d['decay_rate'] for d in program['damped_sines']])
        features['max_decay_rate'] = max(d['decay_rate'] for d in program['damped_sines'])
    else:
        features['n_damped'] = 0
        features['avg_decay_rate'] = 0
        features['max_decay_rate'] = 0

    # Overall energy
    features['total_energy'] = amps.sum().item()
    features['energy_variance'] = amps.sum(dim=1).std().item()

    return program, features


# ============================================================================
# Z DIMENSION ANALYSIS
# ============================================================================

def extract_z_features(z):
    """Extract features from z latent."""
    # z shape: [C, H, T] = [8, 16, T] → flatten to [128, T]
    z_flat = z.reshape(128, -1)

    features = {}

    # Mean activation per dim
    z_mean = z_flat.mean(dim=1).numpy()  # [128]
    features['z_mean'] = z_mean

    # Std per dim (activity level)
    z_std = z_flat.std(dim=1).numpy()
    features['z_std'] = z_std

    # Energy per dim
    z_energy = (z_flat ** 2).sum(dim=1).numpy()
    features['z_energy'] = z_energy

    return features


def correlate_features(paired_data):
    """
    Correlate program features with z dimensions.

    For each program feature (f0, decay_rate, etc),
    find which z dims are most correlated.
    """
    # Extract all features
    all_program_features = []
    all_z_features = []

    for sample in paired_data:
        _, prog_features = extract_program(sample)
        z_features = extract_z_features(sample['z'])

        all_program_features.append(prog_features)
        all_z_features.append(z_features)

    # Build feature matrices
    n_samples = len(paired_data)
    prog_keys = list(all_program_features[0].keys())

    # Program features: [n_samples, n_prog_features]
    prog_matrix = np.array([[pf[k] for k in prog_keys] for pf in all_program_features])

    # Z features: [n_samples, 128] for mean
    z_mean_matrix = np.array([zf['z_mean'] for zf in all_z_features])
    z_energy_matrix = np.array([zf['z_energy'] for zf in all_z_features])

    print("\n" + "=" * 70)
    print("Z → OPERATION MAPPING")
    print("=" * 70)

    correlations = {}

    for i, prog_key in enumerate(prog_keys):
        prog_values = prog_matrix[:, i]

        # Skip if no variance
        if np.std(prog_values) < 1e-6:
            continue

        # Correlate with z_mean
        corrs = []
        for dim in range(128):
            z_values = z_mean_matrix[:, dim]
            if np.std(z_values) < 1e-6:
                corrs.append(0)
                continue

            r = np.corrcoef(prog_values, z_values)[0, 1]
            corrs.append(r if not np.isnan(r) else 0)

        corrs = np.array(corrs)

        # Find top correlated dims
        top_positive = np.argsort(corrs)[-5:][::-1]
        top_negative = np.argsort(corrs)[:5]

        correlations[prog_key] = {
            'positive_dims': [(int(d), float(corrs[d])) for d in top_positive],
            'negative_dims': [(int(d), float(corrs[d])) for d in top_negative],
            'max_correlation': float(np.max(np.abs(corrs))),
            'best_dim': int(np.argmax(np.abs(corrs)))
        }

        print(f"\n{prog_key}:")
        print(f"  Best dim: {correlations[prog_key]['best_dim']} (r={corrs[correlations[prog_key]['best_dim']]:.3f})")
        print(f"  Top positive: {[(d, f'{r:.2f}') for d, r in correlations[prog_key]['positive_dims'][:3]]}")
        print(f"  Top negative: {[(d, f'{r:.2f}') for d, r in correlations[prog_key]['negative_dims'][:3]]}")

    return correlations


def build_z_to_operation_map(correlations, threshold=0.3):
    """
    Build the actual z → operation mapping.

    Group z dims by which operation parameters they control.
    """
    print("\n" + "=" * 70)
    print("Z DIMENSION → OPERATION PARAMETER MAPPING")
    print("=" * 70)

    # Invert: for each z dim, which operations does it control?
    dim_to_ops = defaultdict(list)

    for prog_key, corr_data in correlations.items():
        for dim, r in corr_data['positive_dims'] + corr_data['negative_dims']:
            if abs(r) > threshold:
                dim_to_ops[dim].append((prog_key, r))

    # Find z dims that control multiple operation parameters
    multi_control = {d: ops for d, ops in dim_to_ops.items() if len(ops) >= 2}

    print("\nZ dims controlling multiple operation parameters:")
    for dim, ops in sorted(multi_control.items(), key=lambda x: len(x[1]), reverse=True)[:20]:
        ops_str = ", ".join([f"{op}({r:+.2f})" for op, r in ops])
        print(f"  dim {dim}: {ops_str}")

    # Group operation parameters by controlling z dims
    print("\nOperation parameter → Z dim groups:")

    op_to_dims = {}
    for prog_key, corr_data in correlations.items():
        strong_dims = [d for d, r in corr_data['positive_dims'] + corr_data['negative_dims']
                       if abs(r) > threshold]
        op_to_dims[prog_key] = strong_dims
        if strong_dims:
            print(f"  {prog_key}: dims {strong_dims}")

    return dim_to_ops, op_to_dims


def analyze_operation_z_structure(paired_data):
    """
    Deep analysis: for samples with specific operations,
    what z patterns do we see?
    """
    print("\n" + "=" * 70)
    print("Z PATTERNS FOR SPECIFIC OPERATIONS")
    print("=" * 70)

    # Separate samples by operation presence
    has_harmonics = []
    no_harmonics = []
    has_damping = []
    no_damping = []

    for sample in paired_data:
        prog, _ = extract_program(sample)
        z_flat = sample['z'].reshape(128, -1).mean(dim=1).numpy()

        if prog['harmonic_groups']:
            has_harmonics.append(z_flat)
        else:
            no_harmonics.append(z_flat)

        if prog['damped_sines']:
            has_damping.append(z_flat)
        else:
            no_damping.append(z_flat)

    # Compare z patterns
    if has_harmonics and no_harmonics:
        h_mean = np.mean(has_harmonics, axis=0)
        nh_mean = np.mean(no_harmonics, axis=0)
        diff = h_mean - nh_mean

        top_diff = np.argsort(np.abs(diff))[-10:][::-1]
        print("\nHarmonic presence → Z dim differences:")
        for d in top_diff:
            print(f"  dim {d}: +harmonics {h_mean[d]:+.3f}, -harmonics {nh_mean[d]:+.3f}, diff {diff[d]:+.3f}")

    if has_damping and no_damping:
        d_mean = np.mean(has_damping, axis=0)
        nd_mean = np.mean(no_damping, axis=0)
        diff = d_mean - nd_mean

        top_diff = np.argsort(np.abs(diff))[-10:][::-1]
        print("\nDamping presence → Z dim differences:")
        for d in top_diff:
            print(f"  dim {d}: +damping {d_mean[d]:+.3f}, -damping {nd_mean[d]:+.3f}, diff {diff[d]:+.3f}")


def main():
    print("Loading paired data (z + SMS)...")
    paired_data = load_paired_data(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        n_samples=100
    )
    print(f"  Loaded {len(paired_data)} samples")

    # Correlate features
    correlations = correlate_features(paired_data)

    # Build mapping
    dim_to_ops, op_to_dims = build_z_to_operation_map(correlations, threshold=0.25)

    # Analyze z patterns
    analyze_operation_z_structure(paired_data)

    # Save mapping
    output = {
        'correlations': correlations,
        'dim_to_operations': {int(k): v for k, v in dim_to_ops.items()},
        'operation_to_dims': op_to_dims
    }

    output_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/z_operation_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved mapping to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Z → OPERATIONS MAPPING")
    print("=" * 70)
    print("""
The mapping shows which z dimensions control which operation parameters:

1. HARMONIC CONTROL:
   - primary_f0: controlled by dims that set fundamental frequency
   - harmonic_centroid: spectral brightness of harmonic content
   - total_harmonic_amp: overall harmonic energy

2. DAMPING CONTROL:
   - avg_decay_rate: how fast resonances die out
   - max_decay_rate: sharpest transient decay

3. ENERGY CONTROL:
   - total_energy: overall loudness
   - energy_variance: temporal dynamics

NEXT: Use this mapping to build the synthesizer:
z dims → operation parameters → sines → audio
""")


if __name__ == "__main__":
    main()
