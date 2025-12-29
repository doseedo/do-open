#!/usr/bin/env python3
"""
Probe DCAE Latent Space for Register Differences

Goal: Understand if formant/register information is separable in DCAE latent space.

Compares natural trumpet performances at different registers:
- Low register (groups 0-1): ~C3-D4
- Mid register (groups 2-3): ~D4-G4
- High register (groups 5-7): ~G4-C6

If H dimension captures frequency/formant info, we should see:
- Clear patterns along H that correlate with register
- Different energy distributions in different H bins

If NOT separable, we're building losses on a foundation that can't work.
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/dø')


def load_segments_by_register(segments_json: str, num_per_group: int = 20):
    """Load segment latent paths grouped by register."""
    with open(segments_json) as f:
        data = json.load(f)

    segments_by_group = data.get('segments_by_group', {})

    # Group into low/mid/high
    register_groups = {
        'low': [],    # groups 0-1
        'mid': [],    # groups 2-3
        'high': [],   # groups 5-7
    }

    for group_id, segments in segments_by_group.items():
        gid = int(group_id)
        if gid in [0, 1]:
            register_groups['low'].extend(segments[:num_per_group])
        elif gid in [2, 3]:
            register_groups['mid'].extend(segments[:num_per_group])
        elif gid in [5, 6, 7]:
            register_groups['high'].extend(segments[:num_per_group])

    return register_groups


def load_latent_segment(segment: dict) -> torch.Tensor:
    """Load a latent segment from disk."""
    latent_path = segment['latent_path']
    start_frame = segment['start_frame']
    end_frame = segment['end_frame']

    # Fix paths
    latent_path = latent_path.replace('/mnt/msdd/', '/mnt/msdd2/')

    if not Path(latent_path).exists():
        return None

    data = torch.load(latent_path, map_location='cpu', weights_only=False)

    # Handle dict format
    if isinstance(data, dict):
        latent = data['latents']
    else:
        latent = data

    if latent.dim() == 3:
        latent = latent.unsqueeze(0)  # [1, C, H, T]

    # Extract segment
    latent = latent[:, :, :, start_frame:end_frame]

    if latent.shape[-1] < 16:  # Too short
        return None

    return latent


def compute_latent_statistics(latents: list) -> dict:
    """Compute statistics across a list of latents."""
    if not latents:
        return None

    # Stack all latents [N, C, H, T] -> analyze
    all_latents = []
    for lat in latents:
        if lat is not None:
            all_latents.append(lat)

    if not all_latents:
        return None

    # Concatenate along time dimension for statistics
    concat = torch.cat(all_latents, dim=-1)  # [1, C, H, total_T]
    concat = concat.squeeze(0)  # [C, H, T]

    C, H, T = concat.shape

    stats = {
        'mean_per_h': concat.mean(dim=(0, 2)).numpy(),  # [H]
        'std_per_h': concat.std(dim=(0, 2)).numpy(),    # [H]
        'energy_per_h': (concat ** 2).mean(dim=(0, 2)).numpy(),  # [H]
        'mean_per_c': concat.mean(dim=(1, 2)).numpy(),  # [C]
        'std_per_c': concat.std(dim=(1, 2)).numpy(),    # [C]
        'mean_per_ch': concat.mean(dim=2).numpy(),      # [C, H]
        'std_per_ch': concat.std(dim=2).numpy(),        # [C, H]
        'global_mean': concat.mean().item(),
        'global_std': concat.std().item(),
        'num_samples': len(all_latents),
        'total_frames': T,
    }

    return stats


def plot_register_comparison(stats_by_register: dict, output_dir: Path):
    """Create visualization comparing registers."""
    output_dir.mkdir(parents=True, exist_ok=True)

    registers = ['low', 'mid', 'high']
    colors = {'low': 'blue', 'mid': 'green', 'high': 'red'}

    # 1. Energy per H bin comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Energy distribution along H
    ax = axes[0, 0]
    for reg in registers:
        if stats_by_register[reg]:
            energy = stats_by_register[reg]['energy_per_h']
            ax.plot(range(len(energy)), energy, label=f'{reg} register',
                   color=colors[reg], linewidth=2)
    ax.set_xlabel('H dimension (latent "frequency" bins)')
    ax.set_ylabel('Mean Energy')
    ax.set_title('Energy Distribution Along H Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Mean per H bin
    ax = axes[0, 1]
    for reg in registers:
        if stats_by_register[reg]:
            mean = stats_by_register[reg]['mean_per_h']
            ax.plot(range(len(mean)), mean, label=f'{reg} register',
                   color=colors[reg], linewidth=2)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Mean Value')
    ax.set_title('Mean Value Along H Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Std per H bin
    ax = axes[1, 0]
    for reg in registers:
        if stats_by_register[reg]:
            std = stats_by_register[reg]['std_per_h']
            ax.plot(range(len(std)), std, label=f'{reg} register',
                   color=colors[reg], linewidth=2)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Std Dev')
    ax.set_title('Variability Along H Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Energy per Channel
    ax = axes[1, 1]
    x = np.arange(8)
    width = 0.25
    for i, reg in enumerate(registers):
        if stats_by_register[reg]:
            energy = (stats_by_register[reg]['mean_per_c'] ** 2 +
                     stats_by_register[reg]['std_per_c'] ** 2)
            ax.bar(x + i*width, energy, width, label=f'{reg} register',
                  color=colors[reg], alpha=0.7)
    ax.set_xlabel('Channel (C dimension)')
    ax.set_ylabel('Energy')
    ax.set_title('Energy per Channel')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'C{i}' for i in range(8)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'register_comparison_summary.png', dpi=150)
    plt.close()

    # 2. Heatmap: Mean per [C, H] for each register
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, reg in enumerate(registers):
        ax = axes[i]
        if stats_by_register[reg]:
            data = stats_by_register[reg]['mean_per_ch']  # [C, H]
            im = ax.imshow(data, aspect='auto', cmap='RdBu_r',
                          vmin=-2, vmax=2)
            ax.set_xlabel('H dimension')
            ax.set_ylabel('Channel')
            ax.set_title(f'{reg.capitalize()} Register\n(n={stats_by_register[reg]["num_samples"]})')
            plt.colorbar(im, ax=ax, label='Mean')

    plt.tight_layout()
    plt.savefig(output_dir / 'register_heatmaps.png', dpi=150)
    plt.close()

    # 3. Difference heatmaps (high - low, high - mid)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if stats_by_register['high'] and stats_by_register['low']:
        diff = stats_by_register['high']['mean_per_ch'] - stats_by_register['low']['mean_per_ch']
        ax = axes[0]
        im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('H dimension')
        ax.set_ylabel('Channel')
        ax.set_title('High - Low Register Difference')
        plt.colorbar(im, ax=ax, label='Difference')

    if stats_by_register['high'] and stats_by_register['mid']:
        diff = stats_by_register['high']['mean_per_ch'] - stats_by_register['mid']['mean_per_ch']
        ax = axes[1]
        im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('H dimension')
        ax.set_ylabel('Channel')
        ax.set_title('High - Mid Register Difference')
        plt.colorbar(im, ax=ax, label='Difference')

    plt.tight_layout()
    plt.savefig(output_dir / 'register_differences.png', dpi=150)
    plt.close()

    print(f"\nPlots saved to {output_dir}/")


def print_summary(stats_by_register: dict):
    """Print numerical summary."""
    print("\n" + "="*60)
    print("REGISTER COMPARISON SUMMARY")
    print("="*60)

    for reg in ['low', 'mid', 'high']:
        stats = stats_by_register[reg]
        if stats:
            print(f"\n{reg.upper()} REGISTER:")
            print(f"  Samples: {stats['num_samples']}, Frames: {stats['total_frames']}")
            print(f"  Global mean: {stats['global_mean']:.4f}, std: {stats['global_std']:.4f}")
            print(f"  Energy per H: {stats['energy_per_h']}")

    # Compute separability metrics
    print("\n" + "-"*60)
    print("SEPARABILITY ANALYSIS")
    print("-"*60)

    if stats_by_register['high'] and stats_by_register['low']:
        high_h = stats_by_register['high']['energy_per_h']
        low_h = stats_by_register['low']['energy_per_h']

        # Which H bins differ most?
        diff = high_h - low_h
        max_diff_idx = np.argmax(np.abs(diff))

        print(f"\nH dimension analysis (high vs low):")
        print(f"  Max difference at H={max_diff_idx}: {diff[max_diff_idx]:.4f}")
        print(f"  Energy shift: {diff}")

        # Correlation
        corr = np.corrcoef(high_h, low_h)[0, 1]
        print(f"  Energy correlation (high vs low): {corr:.4f}")
        print(f"  (Low correlation = more separable)")

        # Simple separability score
        total_diff = np.abs(diff).sum()
        avg_energy = (high_h.sum() + low_h.sum()) / 2
        separability = total_diff / avg_energy
        print(f"  Separability score: {separability:.4f}")
        print(f"  (Higher = more different, more separable)")


def main():
    parser = argparse.ArgumentParser(description="Probe DCAE latent space for register differences")
    parser.add_argument('--segments', type=str,
                       default='/home/arlo/Data/pitchshift/v3/trumpet_segments_filtered.json',
                       help='Path to segments JSON')
    parser.add_argument('--output_dir', type=str,
                       default='/tmp/latent_probe',
                       help='Output directory for plots')
    parser.add_argument('--num_per_group', type=int, default=30,
                       help='Number of segments per register group')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("Loading segments by register...")
    register_groups = load_segments_by_register(args.segments, args.num_per_group)

    for reg, segs in register_groups.items():
        print(f"  {reg}: {len(segs)} segments")

    print("\nLoading latents and computing statistics...")
    stats_by_register = {}

    for reg, segments in register_groups.items():
        print(f"\nProcessing {reg} register...")
        latents = []
        for seg in segments:
            lat = load_latent_segment(seg)
            if lat is not None:
                latents.append(lat)

        print(f"  Loaded {len(latents)} valid latents")
        stats_by_register[reg] = compute_latent_statistics(latents)

    print_summary(stats_by_register)
    plot_register_comparison(stats_by_register, output_dir)

    print("\n" + "="*60)
    print("INTERPRETATION GUIDE")
    print("="*60)
    print("""
If H dimension captures formant/register info:
  - Different registers should have different energy profiles along H
  - High register should show more energy in different H bins than low
  - Difference heatmaps should show clear patterns

If NOT separable:
  - All registers have similar H profiles
  - Differences are random/noisy
  - Building H-based losses won't work

Look at the plots in the output directory!
""")


if __name__ == "__main__":
    main()
