#!/usr/bin/env python3
"""
Probe: Is the mute effect pitch-invariant?

Compare muted vs dry trumpet latents at different pitch ranges.
If mute effect is pitch-invariant:
  - (muted - dry) should look similar at low vs high pitches
  - The "mute signature" should be consistent

If mute effect is pitch-dependent:
  - (muted - dry) should look different at different pitches
  - Then why did mute translator work with non-aligned data?
"""

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


def load_f0_and_get_median_pitch(f0_path: str) -> float:
    """Load f0 and compute median MIDI pitch."""
    f0_path = fix_path(f0_path)
    if not Path(f0_path).exists():
        return None

    f0 = np.load(f0_path)
    # Filter out zeros/silence
    voiced = f0[f0 > 50]  # Hz threshold
    if len(voiced) == 0:
        return None

    # Convert Hz to MIDI
    midi = 69 + 12 * np.log2(voiced / 440.0)
    return float(np.median(midi))


def load_latent(latent_path: str) -> torch.Tensor:
    """Load latent tensor."""
    latent_path = fix_path(latent_path)
    if not Path(latent_path).exists():
        return None

    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(data, dict):
        latent = data['latents']
    else:
        latent = data

    if latent.dim() == 3:
        latent = latent.unsqueeze(0)

    return latent.squeeze(0)  # [C, H, T]


def compute_latent_profile(latent: torch.Tensor) -> dict:
    """Compute per-channel and per-H statistics."""
    C, H, T = latent.shape
    return {
        'energy_per_h': (latent ** 2).mean(dim=(0, 2)).numpy(),
        'energy_per_c': (latent ** 2).mean(dim=(1, 2)).numpy(),
        'mean_per_h': latent.mean(dim=(0, 2)).numpy(),
        'mean_per_c': latent.mean(dim=(1, 2)).numpy(),
        'mean_per_ch': latent.mean(dim=2).numpy(),  # [C, H]
    }


def main():
    manifest_path = '/home/arlo/Data/mute_translator/mute_manifest_deduped.json'

    print("Loading manifest...")
    with open(manifest_path) as f:
        data = json.load(f)

    muted_entries = [e for e in data if e.get('is_muted', False)]
    dry_entries = [e for e in data if not e.get('is_muted', False)]

    print(f"Muted: {len(muted_entries)}, Dry: {len(dry_entries)}")

    # Group by pitch range
    def get_pitch_group(midi):
        if midi < 60:
            return 'low'
        elif midi < 72:
            return 'mid'
        else:
            return 'high'

    # Collect latents by muted/dry and pitch group
    profiles = {
        'muted_low': [], 'muted_mid': [], 'muted_high': [],
        'dry_low': [], 'dry_mid': [], 'dry_high': [],
    }

    print("\nProcessing muted entries...")
    for entry in tqdm(muted_entries):
        f0_path = entry.get('conditioning_paths', {}).get('f0')
        if not f0_path:
            continue

        pitch = load_f0_and_get_median_pitch(f0_path)
        if pitch is None:
            continue

        latent = load_latent(entry['latent_path'])
        if latent is None:
            continue

        group = get_pitch_group(pitch)
        profile = compute_latent_profile(latent)
        profile['pitch'] = pitch
        profiles[f'muted_{group}'].append(profile)

    print("\nProcessing dry entries (sampling)...")
    # Sample dry entries (too many to process all)
    dry_sample = np.random.choice(len(dry_entries), min(200, len(dry_entries)), replace=False)
    for idx in tqdm(dry_sample):
        entry = dry_entries[idx]
        f0_path = entry.get('conditioning_paths', {}).get('f0')
        if not f0_path:
            continue

        pitch = load_f0_and_get_median_pitch(f0_path)
        if pitch is None:
            continue

        latent = load_latent(entry['latent_path'])
        if latent is None:
            continue

        group = get_pitch_group(pitch)
        profile = compute_latent_profile(latent)
        profile['pitch'] = pitch
        profiles[f'dry_{group}'].append(profile)

    # Print counts
    print("\nCounts:")
    for k, v in profiles.items():
        print(f"  {k}: {len(v)}")

    # Aggregate profiles
    def aggregate(profile_list):
        if not profile_list:
            return None
        return {
            'energy_per_h': np.mean([p['energy_per_h'] for p in profile_list], axis=0),
            'energy_per_c': np.mean([p['energy_per_c'] for p in profile_list], axis=0),
            'mean_per_h': np.mean([p['mean_per_h'] for p in profile_list], axis=0),
            'mean_per_ch': np.mean([p['mean_per_ch'] for p in profile_list], axis=0),
            'avg_pitch': np.mean([p['pitch'] for p in profile_list]),
        }

    agg = {k: aggregate(v) for k, v in profiles.items()}

    # Compute mute effect = muted - dry at each pitch range
    print("\n" + "="*70)
    print("MUTE EFFECT ANALYSIS: Is it pitch-invariant?")
    print("="*70)

    mute_effects = {}
    for group in ['low', 'mid', 'high']:
        muted_key = f'muted_{group}'
        dry_key = f'dry_{group}'

        if agg[muted_key] is None or agg[dry_key] is None:
            print(f"\n{group.upper()}: Insufficient data")
            continue

        effect = {
            'energy_diff_h': agg[muted_key]['energy_per_h'] - agg[dry_key]['energy_per_h'],
            'energy_diff_c': agg[muted_key]['energy_per_c'] - agg[dry_key]['energy_per_c'],
            'mean_diff_h': agg[muted_key]['mean_per_h'] - agg[dry_key]['mean_per_h'],
        }
        mute_effects[group] = effect

        print(f"\n{group.upper()} (muted pitch ~{agg[muted_key]['avg_pitch']:.1f}, dry pitch ~{agg[dry_key]['avg_pitch']:.1f}):")
        print(f"  Energy diff per H: {effect['energy_diff_h']}")
        print(f"  Energy diff per C: {effect['energy_diff_c']}")

    # Compare mute effects across pitch ranges
    print("\n" + "-"*70)
    print("CROSS-PITCH COMPARISON: Do mute effects look similar?")
    print("-"*70)

    groups_with_data = [g for g in ['low', 'mid', 'high'] if g in mute_effects]

    if len(groups_with_data) >= 2:
        for i, g1 in enumerate(groups_with_data):
            for g2 in groups_with_data[i+1:]:
                corr_h = np.corrcoef(
                    mute_effects[g1]['energy_diff_h'],
                    mute_effects[g2]['energy_diff_h']
                )[0, 1]
                corr_c = np.corrcoef(
                    mute_effects[g1]['energy_diff_c'],
                    mute_effects[g2]['energy_diff_c']
                )[0, 1]

                print(f"\n{g1.upper()} vs {g2.upper()} mute effect:")
                print(f"  H-dimension correlation: {corr_h:.4f}")
                print(f"  C-dimension correlation: {corr_c:.4f}")

                if corr_h > 0.8 and corr_c > 0.8:
                    print(f"  → Mute effect is SIMILAR across these pitches")
                else:
                    print(f"  → Mute effect DIFFERS across these pitches")

    # Plot
    output_dir = Path('/mnt/msdd2/latent_probe_mute')
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Energy per H for muted vs dry
    ax = axes[0, 0]
    for group in groups_with_data:
        if agg[f'muted_{group}']:
            ax.plot(agg[f'muted_{group}']['energy_per_h'],
                   label=f'muted {group}', linestyle='-')
        if agg[f'dry_{group}']:
            ax.plot(agg[f'dry_{group}']['energy_per_h'],
                   label=f'dry {group}', linestyle='--')
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Energy')
    ax.set_title('Energy per H: Muted vs Dry')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mute effect per H at different pitches
    ax = axes[0, 1]
    for group in groups_with_data:
        ax.plot(mute_effects[group]['energy_diff_h'], label=f'{group} register')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Energy difference (muted - dry)')
    ax.set_title('Mute Effect on H: Is it pitch-invariant?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Energy per C for muted vs dry
    ax = axes[1, 0]
    x = np.arange(8)
    width = 0.15
    for i, group in enumerate(groups_with_data):
        if agg[f'muted_{group}']:
            ax.bar(x + i*width*2, agg[f'muted_{group}']['energy_per_c'],
                  width, label=f'muted {group}', alpha=0.7)
        if agg[f'dry_{group}']:
            ax.bar(x + i*width*2 + width, agg[f'dry_{group}']['energy_per_c'],
                  width, label=f'dry {group}', alpha=0.5)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Energy')
    ax.set_title('Energy per Channel')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mute effect per C at different pitches
    ax = axes[1, 1]
    x = np.arange(8)
    width = 0.25
    for i, group in enumerate(groups_with_data):
        ax.bar(x + i*width, mute_effects[group]['energy_diff_c'],
              width, label=f'{group} register')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Energy difference')
    ax.set_title('Mute Effect on Channels: Is it pitch-invariant?')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'mute_pitch_variance.png', dpi=150)
    plt.close()

    print(f"\nPlot saved to {output_dir}/mute_pitch_variance.png")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
If mute effects correlate highly across pitches:
  → Mute transformation is pitch-invariant
  → Distribution matching works because ONE rule applies everywhere

If mute effects differ across pitches:
  → Mute transformation is pitch-dependent
  → Need to investigate WHY mute translator worked anyway
""")


if __name__ == "__main__":
    main()
