#!/usr/bin/env python3
"""
Discover operations from SMS sine data (bottom-up).

SMS = 64 sines × T frames (freqs, amps, phases)

We discover:
1. HARMONIC GROUPS: Sines with integer frequency ratios (phase-locked)
2. COUPLED SINES: Sines that exchange energy (amplitude trade-off)
3. DAMPED RESONATORS: Sines that decay together
4. CONSERVED QUANTITIES: Sums that stay constant

These operations compose to create the DCAE entanglement.
"""

import torch
import numpy as np
from collections import defaultdict
import sys
import os
import json

import orjson


def load_sms_samples(manifest_path, n_samples=50):
    """Load SMS data from manifest."""
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    samples = []
    for entry in manifest['entries'][:n_samples * 2]:
        path = entry['path']
        if any(kw in path.lower() for kw in ['drum', 'kick', 'snare', 'hat', 'perc']):
            continue
        if not os.path.exists(path):
            continue

        try:
            data = torch.load(path, weights_only=True, map_location='cpu')
            freqs = data['freqs']  # [T, n_sines]
            amps = data['amps']
            phases = data.get('phases', torch.zeros_like(freqs))

            # Limit length
            T = min(freqs.shape[0], 64)
            samples.append({
                'freqs': freqs[:T],
                'amps': amps[:T],
                'phases': phases[:T],
                'path': path
            })

            if len(samples) >= n_samples:
                break
        except:
            continue

    return samples


# ============================================================================
# 1. HARMONIC GROUP DETECTION
# ============================================================================

def find_harmonic_groups(freqs, amps, f0_threshold=20, ratio_tolerance=0.05):
    """
    Find groups of sines that form harmonic series.

    A harmonic group has:
    - A fundamental f0
    - Partials at integer multiples: f0, 2*f0, 3*f0, ...

    Returns list of harmonic groups with their partials.
    """
    T, n_sines = freqs.shape

    # Average frequencies and amps across time (for stable estimate)
    avg_freqs = freqs.mean(dim=0).numpy()
    avg_amps = amps.mean(dim=0).numpy()

    # Sort by amplitude (strongest first)
    amp_order = np.argsort(avg_amps)[::-1]

    groups = []
    used = set()

    for i in amp_order:
        if i in used or avg_amps[i] < 0.001:
            continue

        f0 = avg_freqs[i]
        if f0 < f0_threshold:
            continue

        # Look for harmonics
        group = {'f0': f0, 'partials': [(1, i, avg_amps[i])]}  # (ratio, sine_idx, amp)
        used.add(i)

        for j in range(n_sines):
            if j in used or avg_amps[j] < 0.0001:
                continue

            fj = avg_freqs[j]
            ratio = fj / f0

            # Check if ratio is close to integer
            nearest_int = round(ratio)
            if nearest_int >= 2 and nearest_int <= 16:
                if abs(ratio - nearest_int) / nearest_int < ratio_tolerance:
                    group['partials'].append((nearest_int, j, avg_amps[j]))
                    used.add(j)

        if len(group['partials']) >= 2:
            groups.append(group)

    return groups


def analyze_harmonic_groups(samples):
    """Analyze harmonic structure across all samples."""
    print("\n" + "=" * 70)
    print("1. HARMONIC GROUPS (phase-locked sines at integer ratios)")
    print("=" * 70)

    all_groups = []

    for i, sample in enumerate(samples[:20]):
        groups = find_harmonic_groups(sample['freqs'], sample['amps'])
        all_groups.append(groups)

        if i < 5:  # Show first 5
            print(f"\nSample {i}: {os.path.basename(sample['path'])}")
            for g in groups[:3]:
                partials_str = ", ".join([f"{r}x({a:.3f})" for r, idx, a in sorted(g['partials'])])
                print(f"  f0={g['f0']:.1f}Hz: {partials_str}")

    # Statistics
    n_groups = [len(g) for g in all_groups]
    print(f"\n  Average harmonic groups per sample: {np.mean(n_groups):.1f}")
    print(f"  This suggests: HarmonicSeries(f0, partials, weights) is a valid operation")

    return all_groups


# ============================================================================
# 2. COUPLED SINES (energy exchange)
# ============================================================================

def find_coupled_sines(amps, threshold=-0.5):
    """
    Find sine pairs that exchange energy.
    When one gets louder, the other gets quieter.

    This is amplitude anti-correlation across time.
    """
    T, n_sines = amps.shape
    amps_np = amps.numpy()

    # Only analyze sines with significant amplitude
    significant = np.where(amps_np.mean(axis=0) > 0.001)[0]

    if len(significant) < 2:
        return []

    # Compute correlation matrix for significant sines
    amps_sig = amps_np[:, significant]

    # Normalize
    amps_centered = amps_sig - amps_sig.mean(axis=0, keepdims=True)
    amps_std = amps_sig.std(axis=0, keepdims=True)
    amps_std[amps_std < 1e-6] = 1
    amps_norm = amps_centered / amps_std

    corr = (amps_norm.T @ amps_norm) / T

    # Find anti-correlated pairs
    coupled = []
    for i in range(len(significant)):
        for j in range(i + 1, len(significant)):
            if corr[i, j] < threshold:
                coupled.append({
                    'sine_a': int(significant[i]),
                    'sine_b': int(significant[j]),
                    'correlation': float(corr[i, j]),
                    'type': 'energy_exchange'
                })

    return coupled


def analyze_coupling(samples):
    """Analyze energy coupling across samples."""
    print("\n" + "=" * 70)
    print("2. COUPLED SINES (energy exchange)")
    print("=" * 70)

    all_couplings = []

    for i, sample in enumerate(samples[:20]):
        coupled = find_coupled_sines(sample['amps'])
        all_couplings.extend(coupled)

        if i < 5 and coupled:
            print(f"\nSample {i}:")
            for c in coupled[:3]:
                print(f"  sines ({c['sine_a']}, {c['sine_b']}): r={c['correlation']:.2f} (energy trade-off)")

    print(f"\n  Total couplings found: {len(all_couplings)}")
    if all_couplings:
        avg_corr = np.mean([c['correlation'] for c in all_couplings])
        print(f"  Average coupling strength: {avg_corr:.2f}")
        print(f"  This suggests: EnergyCoupling(sine_a, sine_b, strength) operation")

    return all_couplings


# ============================================================================
# 3. DAMPED RESONATORS
# ============================================================================

def find_damped_sines(amps, min_decay=0.01):
    """
    Find sines that decay over time (damped resonators).

    Fit exponential decay: amp(t) = A * exp(-decay * t)
    """
    T, n_sines = amps.shape
    amps_np = amps.numpy()

    damped = []
    t = np.arange(T)

    for i in range(n_sines):
        amp_i = amps_np[:, i]

        # Check if amplitude is significant and decaying
        if amp_i.max() < 0.01:
            continue

        # Simple decay detection: compare first half to second half
        first_half = amp_i[:T//2].mean()
        second_half = amp_i[T//2:].mean()

        if first_half > second_half * 1.5 and first_half > 0.01:
            # Estimate decay rate
            ratio = second_half / (first_half + 1e-8)
            decay_rate = -np.log(ratio + 1e-8) / (T // 2)

            if decay_rate > min_decay:
                damped.append({
                    'sine_idx': i,
                    'initial_amp': float(first_half),
                    'decay_rate': float(decay_rate),
                    'type': 'damped_resonator'
                })

    return damped


def analyze_damping(samples):
    """Analyze damping patterns."""
    print("\n" + "=" * 70)
    print("3. DAMPED RESONATORS (decaying sines)")
    print("=" * 70)

    all_damped = []

    for i, sample in enumerate(samples[:20]):
        damped = find_damped_sines(sample['amps'])
        all_damped.extend(damped)

        if i < 5 and damped:
            print(f"\nSample {i}:")
            for d in damped[:3]:
                print(f"  sine {d['sine_idx']}: decay={d['decay_rate']:.3f}, A0={d['initial_amp']:.3f}")

    if all_damped:
        decay_rates = [d['decay_rate'] for d in all_damped]
        print(f"\n  Total damped sines: {len(all_damped)}")
        print(f"  Decay rate range: [{min(decay_rates):.3f}, {max(decay_rates):.3f}]")
        print(f"  This suggests: DampedOscillator(sine, decay_rate) operation")

    return all_damped


# ============================================================================
# 4. CONSERVED QUANTITIES
# ============================================================================

def find_conserved_quantities(amps, threshold=0.1):
    """
    Find groups of sines whose total amplitude is conserved.

    If sum(amps[group]) is constant across time, energy is conserved in that group.
    """
    T, n_sines = amps.shape
    amps_np = amps.numpy()

    # Total amplitude across all sines per frame
    total_amp = amps_np.sum(axis=1)
    total_std = total_amp.std() / (total_amp.mean() + 1e-8)

    conserved = []

    # Check if total is conserved
    if total_std < threshold:
        conserved.append({
            'type': 'total_energy',
            'sines': list(range(n_sines)),
            'variance': float(total_std)
        })

    # Look for pairs with conserved sum
    significant = np.where(amps_np.mean(axis=0) > 0.001)[0]

    for i in range(len(significant)):
        for j in range(i + 1, len(significant)):
            si, sj = significant[i], significant[j]
            pair_sum = amps_np[:, si] + amps_np[:, sj]
            pair_std = pair_sum.std() / (pair_sum.mean() + 1e-8)

            if pair_std < threshold:
                conserved.append({
                    'type': 'pair_conserved',
                    'sines': [int(si), int(sj)],
                    'variance': float(pair_std)
                })

    return conserved


def analyze_conservation(samples):
    """Analyze energy conservation."""
    print("\n" + "=" * 70)
    print("4. CONSERVED QUANTITIES (constant sums)")
    print("=" * 70)

    all_conserved = []

    for i, sample in enumerate(samples[:20]):
        conserved = find_conserved_quantities(sample['amps'])
        all_conserved.extend(conserved)

        if i < 5 and conserved:
            print(f"\nSample {i}:")
            for c in conserved[:3]:
                if c['type'] == 'total_energy':
                    print(f"  Total energy conserved (var={c['variance']:.3f})")
                else:
                    print(f"  sines {c['sines']}: sum conserved (var={c['variance']:.3f})")

    total_conserved = sum(1 for c in all_conserved if c['type'] == 'total_energy')
    pair_conserved = len(all_conserved) - total_conserved

    print(f"\n  Samples with total energy conservation: {total_conserved}")
    print(f"  Conserved pairs found: {pair_conserved}")
    print(f"  This suggests: EnergyConservation(sine_group) constraint")

    return all_conserved


# ============================================================================
# 5. BUILD OPERATION TREE
# ============================================================================

def build_operation_tree(sample):
    """
    Build a tree of operations that generates this sample's sines.

    Returns a program-like structure.
    """
    freqs, amps = sample['freqs'], sample['amps']

    program = {
        'harmonic_groups': [],
        'couplings': [],
        'damped': [],
        'conserved': []
    }

    # Find harmonic groups
    groups = find_harmonic_groups(freqs, amps)
    for g in groups:
        program['harmonic_groups'].append({
            'f0': float(g['f0']),
            'partials': [(int(r), float(a)) for r, idx, a in g['partials']]
        })

    # Find couplings
    couplings = find_coupled_sines(amps)
    program['couplings'] = couplings

    # Find damped sines
    damped = find_damped_sines(amps)
    program['damped'] = damped

    # Find conserved quantities
    conserved = find_conserved_quantities(amps)
    program['conserved'] = conserved

    return program


def analyze_compression(samples):
    """Analyze how much we can compress with discovered operations."""
    print("\n" + "=" * 70)
    print("5. COMPRESSION ANALYSIS (MDL)")
    print("=" * 70)

    for i, sample in enumerate(samples[:5]):
        freqs, amps = sample['freqs'], sample['amps']
        T, n_sines = freqs.shape

        raw_params = T * n_sines * 2  # freqs + amps

        program = build_operation_tree(sample)

        # Count program params
        program_params = 0

        # Harmonic groups: f0 + n_partials weights
        for g in program['harmonic_groups']:
            program_params += 1 + len(g['partials'])

        # Couplings: 2 indices + strength
        program_params += len(program['couplings']) * 3

        # Damped: index + decay
        program_params += len(program['damped']) * 2

        compression = raw_params / (program_params + 1)

        print(f"\nSample {i}:")
        print(f"  Raw: {raw_params} params ({T} frames × {n_sines} sines × 2)")
        print(f"  Program: {program_params} params")
        print(f"  Compression: {compression:.1f}x")
        print(f"    Harmonic groups: {len(program['harmonic_groups'])}")
        print(f"    Couplings: {len(program['couplings'])}")
        print(f"    Damped: {len(program['damped'])}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Loading SMS samples...")
    samples = load_sms_samples(
        '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_v4/sms_manifest.json',
        n_samples=50
    )
    print(f"  Loaded {len(samples)} samples")

    # Analyze each pattern type
    harmonic_groups = analyze_harmonic_groups(samples)
    couplings = analyze_coupling(samples)
    damped = analyze_damping(samples)
    conserved = analyze_conservation(samples)

    # Compression analysis
    analyze_compression(samples)

    # Summary
    print("\n" + "=" * 70)
    print("DISCOVERED OPERATIONS (bottom-up from sines)")
    print("=" * 70)
    print("""
From SMS sine data, we discovered these operations:

1. HarmonicSeries(f0, partials, weights)
   - Groups sines at integer frequency ratios
   - Captures tonal structure

2. EnergyCoupling(sine_a, sine_b, strength)
   - Sines that trade amplitude (anti-correlated)
   - Captures energy flow between partials

3. DampedOscillator(sine, decay_rate)
   - Sines that decay exponentially
   - Captures resonator behavior

4. EnergyConservation(sine_group)
   - Groups where sum of amps is constant
   - Captures physical conservation laws

NEXT: Map these operations to z dimensions.
Find which z dims control which operations.
""")

    # Save operations
    output = {
        'operations': {
            'harmonic_series': 'HarmonicSeries(f0, partials, weights)',
            'energy_coupling': 'EnergyCoupling(sine_a, sine_b, strength)',
            'damped_oscillator': 'DampedOscillator(sine, decay_rate)',
            'energy_conservation': 'EnergyConservation(sine_group)'
        },
        'sample_programs': [build_operation_tree(s) for s in samples[:10]]
    }

    output_path = '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/data/sms_operations.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
