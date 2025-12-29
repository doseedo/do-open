#!/usr/bin/env python3
"""
Probe: DSP-Shifted vs Natural at Same Pitch

The key question: Does DSP shifting create artifacts that look like
"wrong register" in latent space, or something completely different?

Compare:
1. Natural low register trumpet
2. Natural high register trumpet
3. Low register shifted UP to high register pitch (DSP artifacts)

If (3) looks like (1) in H-distribution but should look like (2),
then our training signal makes sense.

If (3) looks completely different from both, we're solving the wrong problem.
"""

import sys
import json
import argparse
import subprocess
import tempfile
from pathlib import Path
from collections import defaultdict

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/dø')


def load_dcae(device='cuda'):
    """Load DCAE model."""
    from do.pipeline_do import DoTrainComponents
    components = DoTrainComponents(
        checkpoint_dir='/home/arlo/Data/ACE-Step/checkpoints',
        device_id=0 if device == 'cuda' else -1,
    )
    dcae = components.load_dcae()
    return dcae


def apply_pitch_shift_sox(audio: torch.Tensor, sr: int, shift_semitones: int) -> torch.Tensor:
    """Apply pitch shift using sox."""
    if shift_semitones == 0:
        return audio

    cents = shift_semitones * 100

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as in_f:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as out_f:
            in_path = in_f.name
            out_path = out_f.name

    try:
        torchaudio.save(in_path, audio, sr)
        cmd = ['sox', in_path, out_path, 'pitch', str(cents), 'rate', '-v', str(sr)]
        subprocess.run(cmd, capture_output=True, check=True)
        shifted, _ = torchaudio.load(out_path)

        # Match length
        if shifted.shape[1] > audio.shape[1]:
            shifted = shifted[:, :audio.shape[1]]
        elif shifted.shape[1] < audio.shape[1]:
            pad = torch.zeros(shifted.shape[0], audio.shape[1] - shifted.shape[1])
            shifted = torch.cat([shifted, pad], dim=1)

        return shifted
    finally:
        import os
        if os.path.exists(in_path):
            os.remove(in_path)
        if os.path.exists(out_path):
            os.remove(out_path)


def fix_path(path: str) -> str:
    """Fix mount paths."""
    return path.replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/').replace('/mnt/msdd/', '/mnt/msdd2/')


def load_segments_by_register(segments_json: str, num_per_group: int = 10):
    """Load segments grouped by register."""
    with open(segments_json) as f:
        data = json.load(f)

    segments_by_group = data.get('segments_by_group', {})

    register_groups = {
        'low': [],    # groups 0-1
        'high': [],   # groups 5-7
    }

    for group_id, segments in segments_by_group.items():
        gid = int(group_id)
        if gid in [0, 1]:
            register_groups['low'].extend(segments[:num_per_group])
        elif gid in [5, 6, 7]:
            register_groups['high'].extend(segments[:num_per_group])

    return register_groups


@torch.no_grad()
def encode_audio_segment(dcae, audio_path: str, start_sec: float, end_sec: float, device='cuda'):
    """Load and encode an audio segment."""
    audio, sr = torchaudio.load(audio_path)

    # Convert to stereo if mono
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]

    # Extract segment
    start_sample = int(start_sec * sr)
    end_sample = int(end_sec * sr)
    audio = audio[:, start_sample:end_sample]

    # Resample to 48kHz if needed
    if sr != 48000:
        resampler = torchaudio.transforms.Resample(sr, 48000)
        audio = resampler(audio)
        sr = 48000

    # Encode
    audio_gpu = audio.unsqueeze(0).to(device)
    latent, _ = dcae.encode(audio_gpu)

    return latent.cpu(), audio, sr


@torch.no_grad()
def encode_shifted_audio(dcae, audio: torch.Tensor, sr: int, shift_semitones: int, device='cuda'):
    """Apply DSP shift and encode."""
    shifted = apply_pitch_shift_sox(audio, sr, shift_semitones)

    # Ensure stereo
    if shifted.shape[0] == 1:
        shifted = shifted.repeat(2, 1)

    shifted_gpu = shifted.unsqueeze(0).to(device)
    latent, _ = dcae.encode(shifted_gpu)

    return latent.cpu(), shifted


def compute_latent_stats(latent: torch.Tensor) -> dict:
    """Compute per-H and per-C statistics."""
    lat = latent.squeeze(0)  # [C, H, T]
    C, H, T = lat.shape

    return {
        'energy_per_h': (lat ** 2).mean(dim=(0, 2)).numpy(),
        'mean_per_h': lat.mean(dim=(0, 2)).numpy(),
        'std_per_h': lat.std(dim=(0, 2)).numpy(),
        'energy_per_c': (lat ** 2).mean(dim=(1, 2)).numpy(),
        'mean_per_c': lat.mean(dim=(1, 2)).numpy(),
        'mean_per_ch': lat.mean(dim=2).numpy(),  # [C, H]
        'global_mean': lat.mean().item(),
        'global_std': lat.std().item(),
    }


def aggregate_stats(stats_list: list) -> dict:
    """Average statistics across multiple samples."""
    if not stats_list:
        return None

    agg = {}
    for key in stats_list[0].keys():
        vals = [s[key] for s in stats_list]
        if isinstance(vals[0], np.ndarray):
            agg[key] = np.mean(vals, axis=0)
        else:
            agg[key] = np.mean(vals)

    return agg


def plot_comparison(stats_dict: dict, output_dir: Path):
    """Plot comparison of natural low, natural high, and DSP-shifted."""
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = ['natural_low', 'natural_high', 'dsp_shifted_up']
    colors = {'natural_low': 'blue', 'natural_high': 'red', 'dsp_shifted_up': 'orange'}

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Energy per H
    ax = axes[0, 0]
    for label in labels:
        if stats_dict.get(label):
            ax.plot(stats_dict[label]['energy_per_h'], label=label,
                   color=colors[label], linewidth=2)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Energy')
    ax.set_title('Energy Distribution Along H')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Energy per Channel
    ax = axes[0, 1]
    x = np.arange(8)
    width = 0.25
    for i, label in enumerate(labels):
        if stats_dict.get(label):
            ax.bar(x + i*width, stats_dict[label]['energy_per_c'], width,
                  label=label, color=colors[label], alpha=0.7)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Energy')
    ax.set_title('Energy per Channel')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'C{i}' for i in range(8)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Mean per H
    ax = axes[0, 2]
    for label in labels:
        if stats_dict.get(label):
            ax.plot(stats_dict[label]['mean_per_h'], label=label,
                   color=colors[label], linewidth=2)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Mean')
    ax.set_title('Mean Value Along H')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Difference: DSP vs Natural High (what model needs to learn)
    ax = axes[1, 0]
    if stats_dict.get('dsp_shifted_up') and stats_dict.get('natural_high'):
        diff = stats_dict['dsp_shifted_up']['energy_per_h'] - stats_dict['natural_high']['energy_per_h']
        ax.bar(range(16), diff, color='purple', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Energy Difference')
    ax.set_title('DSP Shifted - Natural High\n(What model must correct)')
    ax.grid(True, alpha=0.3)

    # 5. Difference: DSP vs Natural Low (how much DSP changed things)
    ax = axes[1, 1]
    if stats_dict.get('dsp_shifted_up') and stats_dict.get('natural_low'):
        diff = stats_dict['dsp_shifted_up']['energy_per_h'] - stats_dict['natural_low']['energy_per_h']
        ax.bar(range(16), diff, color='green', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('H dimension')
    ax.set_ylabel('Energy Difference')
    ax.set_title('DSP Shifted - Natural Low\n(What DSP changed)')
    ax.grid(True, alpha=0.3)

    # 6. Channel-wise analysis
    ax = axes[1, 2]
    if stats_dict.get('dsp_shifted_up') and stats_dict.get('natural_high'):
        diff_c = stats_dict['dsp_shifted_up']['energy_per_c'] - stats_dict['natural_high']['energy_per_c']
        ax.bar(range(8), diff_c, color='purple', alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Channel')
    ax.set_ylabel('Energy Difference')
    ax.set_title('DSP - Natural High (per Channel)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'dsp_vs_natural_comparison.png', dpi=150)
    plt.close()

    # Heatmaps
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for i, label in enumerate(labels):
        ax = axes[i]
        if stats_dict.get(label):
            im = ax.imshow(stats_dict[label]['mean_per_ch'], aspect='auto',
                          cmap='RdBu_r', vmin=-2, vmax=2)
            ax.set_xlabel('H')
            ax.set_ylabel('Channel')
            ax.set_title(label.replace('_', ' ').title())
            plt.colorbar(im, ax=ax)

    # Difference heatmap
    ax = axes[3]
    if stats_dict.get('dsp_shifted_up') and stats_dict.get('natural_high'):
        diff = stats_dict['dsp_shifted_up']['mean_per_ch'] - stats_dict['natural_high']['mean_per_ch']
        im = ax.imshow(diff, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xlabel('H')
        ax.set_ylabel('Channel')
        ax.set_title('DSP - Natural High\n(Correction needed)')
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(output_dir / 'dsp_vs_natural_heatmaps.png', dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir}/")


def print_analysis(stats_dict: dict):
    """Print quantitative analysis."""
    print("\n" + "="*70)
    print("DSP-SHIFTED vs NATURAL COMPARISON")
    print("="*70)

    for label in ['natural_low', 'natural_high', 'dsp_shifted_up']:
        if stats_dict.get(label):
            s = stats_dict[label]
            print(f"\n{label.upper()}:")
            print(f"  Global: mean={s['global_mean']:.4f}, std={s['global_std']:.4f}")

    print("\n" + "-"*70)
    print("KEY QUESTION: Does DSP-shifted look like source or target?")
    print("-"*70)

    if all(k in stats_dict for k in ['natural_low', 'natural_high', 'dsp_shifted_up']):
        low_h = stats_dict['natural_low']['energy_per_h']
        high_h = stats_dict['natural_high']['energy_per_h']
        dsp_h = stats_dict['dsp_shifted_up']['energy_per_h']

        # Correlations
        corr_dsp_low = np.corrcoef(dsp_h, low_h)[0, 1]
        corr_dsp_high = np.corrcoef(dsp_h, high_h)[0, 1]
        corr_low_high = np.corrcoef(low_h, high_h)[0, 1]

        print(f"\nH-dimension energy correlations:")
        print(f"  DSP-shifted vs Natural Low:  {corr_dsp_low:.4f}")
        print(f"  DSP-shifted vs Natural High: {corr_dsp_high:.4f}")
        print(f"  Natural Low vs Natural High: {corr_low_high:.4f}")

        # Distances
        dist_dsp_low = np.sqrt(((dsp_h - low_h) ** 2).sum())
        dist_dsp_high = np.sqrt(((dsp_h - high_h) ** 2).sum())
        dist_low_high = np.sqrt(((low_h - high_h) ** 2).sum())

        print(f"\nH-dimension energy distances:")
        print(f"  DSP-shifted to Natural Low:  {dist_dsp_low:.4f}")
        print(f"  DSP-shifted to Natural High: {dist_dsp_high:.4f}")
        print(f"  Natural Low to Natural High: {dist_low_high:.4f}")

        # Interpretation
        print("\n" + "-"*70)
        print("INTERPRETATION:")
        print("-"*70)

        if corr_dsp_low > corr_dsp_high:
            print("  DSP-shifted is MORE similar to Natural Low (source)")
            print("  → DSP shifting preserves source register characteristics")
            print("  → Training signal makes sense: need to push toward target register")
        else:
            print("  DSP-shifted is MORE similar to Natural High (target)")
            print("  → DSP shifting already moves toward target register?")
            print("  → Check if this is just pitch, not formants")

        # Channel analysis
        print("\n" + "-"*70)
        print("CHANNEL ANALYSIS:")
        print("-"*70)

        low_c = stats_dict['natural_low']['energy_per_c']
        high_c = stats_dict['natural_high']['energy_per_c']
        dsp_c = stats_dict['dsp_shifted_up']['energy_per_c']

        for c in range(8):
            diff_from_target = dsp_c[c] - high_c[c]
            diff_natural = high_c[c] - low_c[c]
            print(f"  C{c}: DSP-target diff={diff_from_target:+.4f}, "
                  f"natural register diff={diff_natural:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--segments', type=str,
                       default='/home/arlo/Data/pitchshift/v3/trumpet_segments_filtered.json')
    parser.add_argument('--output_dir', type=str, default='/mnt/msdd2/latent_probe_dsp')
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--shift', type=int, default=12,
                       help='Semitones to shift (default: 12 = one octave)')

    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print("Loading DCAE...")
    dcae = load_dcae()

    print("Loading segments...")
    register_groups = load_segments_by_register(args.segments, args.num_samples)
    print(f"  Low register: {len(register_groups['low'])} segments")
    print(f"  High register: {len(register_groups['high'])} segments")

    # Collect statistics
    stats_natural_low = []
    stats_natural_high = []
    stats_dsp_shifted = []

    print(f"\nProcessing samples (shift={args.shift} semitones)...")

    # Process low register (natural + DSP shifted up)
    for seg in register_groups['low'][:args.num_samples]:
        audio_path = fix_path(seg['audio_path'])
        if not Path(audio_path).exists():
            continue

        # Convert frame to seconds (assuming ~86 frames/sec for DCAE)
        start_sec = seg['start_frame'] / 86.0
        end_sec = seg['end_frame'] / 86.0

        try:
            # Natural low
            latent, audio, sr = encode_audio_segment(dcae, audio_path, start_sec, end_sec)
            stats_natural_low.append(compute_latent_stats(latent))

            # DSP shifted up
            latent_shifted, _ = encode_shifted_audio(dcae, audio, sr, args.shift)
            stats_dsp_shifted.append(compute_latent_stats(latent_shifted))

            print(f"  Processed: {Path(audio_path).name}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Process high register (natural only)
    for seg in register_groups['high'][:args.num_samples]:
        audio_path = fix_path(seg['audio_path'])
        if not Path(audio_path).exists():
            continue

        start_sec = seg['start_frame'] / 86.0
        end_sec = seg['end_frame'] / 86.0

        try:
            latent, _, _ = encode_audio_segment(dcae, audio_path, start_sec, end_sec)
            stats_natural_high.append(compute_latent_stats(latent))
            print(f"  Processed high: {Path(audio_path).name}")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"\nCollected: {len(stats_natural_low)} low, {len(stats_natural_high)} high, "
          f"{len(stats_dsp_shifted)} shifted")

    # Aggregate
    stats_dict = {
        'natural_low': aggregate_stats(stats_natural_low),
        'natural_high': aggregate_stats(stats_natural_high),
        'dsp_shifted_up': aggregate_stats(stats_dsp_shifted),
    }

    print_analysis(stats_dict)
    plot_comparison(stats_dict, output_dir)


if __name__ == "__main__":
    main()
