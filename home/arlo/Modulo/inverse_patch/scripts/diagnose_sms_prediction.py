#!/usr/bin/env python3
"""
Diagnose why mel → sines prediction sounds wrong despite low loss.

Check:
1. Are predicted frequencies in the right ballpark?
2. Is amplitude distribution correct?
3. What does the loss actually measure vs what we hear?
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
import orjson

from full_whitebox_chain import MelMapperV2, SMSMapperV2, MEL_FREQS


def load_ground_truth_sms(sms_path):
    """Load ground truth SMS extraction."""
    data = torch.load(sms_path, weights_only=True, map_location='cpu')
    return data['freqs'], data['amps']


def analyze_prediction(pred_freqs, pred_amps, gt_freqs, gt_amps):
    """Detailed analysis of prediction vs ground truth."""

    # Take single frame for analysis
    T = min(pred_freqs.shape[0], gt_freqs.shape[0])

    results = {
        'per_frame': [],
        'overall': {}
    }

    for t in range(0, T, T//5):  # Sample 5 frames
        pf = pred_freqs[t].numpy()
        pa = pred_amps[t].numpy()
        gf = gt_freqs[t].numpy()
        ga = gt_amps[t].numpy()

        # Sort by amplitude (most important sines first)
        pred_order = np.argsort(pa)[::-1]
        gt_order = np.argsort(ga)[::-1]

        frame_result = {
            'frame': t,
            'pred_top5_freqs': pf[pred_order[:5]].tolist(),
            'pred_top5_amps': pa[pred_order[:5]].tolist(),
            'gt_top5_freqs': gf[gt_order[:5]].tolist(),
            'gt_top5_amps': ga[gt_order[:5]].tolist(),
        }

        # For top 5 GT sines, find closest predicted
        matches = []
        for i in range(5):
            gt_f = gf[gt_order[i]]
            gt_a = ga[gt_order[i]]

            # Find closest predicted frequency
            freq_diffs = np.abs(pf - gt_f)
            closest_idx = np.argmin(freq_diffs)
            pred_f = pf[closest_idx]
            pred_a = pa[closest_idx]

            semitones = 12 * np.log2((pred_f + 1) / (gt_f + 1))

            matches.append({
                'gt_freq': gt_f,
                'gt_amp': gt_a,
                'pred_freq': pred_f,
                'pred_amp': pred_a,
                'semitone_error': semitones,
                'amp_ratio': pred_a / (gt_a + 1e-6)
            })

        frame_result['matches'] = matches
        results['per_frame'].append(frame_result)

    # Overall statistics
    all_gt_freqs = gt_freqs.numpy().flatten()
    all_gt_amps = gt_amps.numpy().flatten()
    all_pred_freqs = pred_freqs.numpy().flatten()
    all_pred_amps = pred_amps.numpy().flatten()

    # Weighted by amplitude
    gt_weighted_centroid = np.sum(all_gt_freqs * all_gt_amps) / (np.sum(all_gt_amps) + 1e-6)
    pred_weighted_centroid = np.sum(all_pred_freqs * all_pred_amps) / (np.sum(all_pred_amps) + 1e-6)

    results['overall'] = {
        'gt_freq_range': [all_gt_freqs.min(), all_gt_freqs.max()],
        'pred_freq_range': [all_pred_freqs.min(), all_pred_freqs.max()],
        'gt_weighted_centroid': gt_weighted_centroid,
        'pred_weighted_centroid': pred_weighted_centroid,
        'gt_total_energy': all_gt_amps.sum(),
        'pred_total_energy': all_pred_amps.sum(),
        'gt_max_amp': all_gt_amps.max(),
        'pred_max_amp': all_pred_amps.max(),
    }

    return results


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load mappers
    print("Loading mappers...")
    mel_mapper = MelMapperV2().to(device)
    sms_mapper = SMSMapperV2().to(device)

    mel_ckpt = torch.load('checkpoints/mel_mapper/best_model.pt', weights_only=True)
    mel_mapper.load_state_dict(mel_ckpt['model_state_dict'])
    mel_mapper.eval()

    sms_ckpt = torch.load('checkpoints/mel_to_sms/best_model_v2.pt', weights_only=True)
    sms_mapper.load_state_dict(sms_ckpt['model_state_dict'])
    sms_mapper.eval()

    # Load test sample
    manifest_path = 'data/sms_v4/sms_manifest.json'
    with open(manifest_path, 'rb') as f:
        manifest = orjson.loads(f.read())

    for entry in manifest['entries'][:50]:
        path = entry['path']
        if 'drum' in path.lower():
            continue
        try:
            data = torch.load(path, weights_only=True, map_location='cpu')
            lat_path = data.get('latent_path')
            if lat_path and os.path.exists(lat_path):
                # Load latent
                lat_data = torch.load(lat_path, weights_only=True, map_location='cpu')
                z = lat_data.get('latents', lat_data)
                if z.dim() == 3:
                    z = z.unsqueeze(0)
                z = z[:, :, :, :32].to(device)

                # Load ground truth SMS
                gt_freqs = data['freqs'][:256]  # Limit frames
                gt_amps = data['amps'][:256]

                print(f"\nSample: {path}")
                print(f"z shape: {z.shape}")
                print(f"GT SMS shape: {gt_freqs.shape}")
                break
        except Exception as e:
            continue

    # Predict
    with torch.no_grad():
        mel_pred = mel_mapper(z)  # [B, T, 128]
        pred_freqs, pred_amps = sms_mapper(mel_pred)

    pred_freqs = pred_freqs[0].cpu()  # [T, 64]
    pred_amps = pred_amps[0].cpu()

    print(f"Pred shape: {pred_freqs.shape}")

    # Match lengths
    T = min(pred_freqs.shape[0], gt_freqs.shape[0])
    pred_freqs = pred_freqs[:T]
    pred_amps = pred_amps[:T]
    gt_freqs = gt_freqs[:T]
    gt_amps = gt_amps[:T]

    # Analyze
    print("\n" + "="*70)
    print("DIAGNOSIS: Why does 2 semitone loss sound bad?")
    print("="*70)

    results = analyze_prediction(pred_freqs, pred_amps, gt_freqs, gt_amps)

    print("\n1. OVERALL STATISTICS")
    print("-"*50)
    o = results['overall']
    print(f"  GT freq range: {o['gt_freq_range'][0]:.0f} - {o['gt_freq_range'][1]:.0f} Hz")
    print(f"  Pred freq range: {o['pred_freq_range'][0]:.0f} - {o['pred_freq_range'][1]:.0f} Hz")
    print(f"  GT weighted centroid: {o['gt_weighted_centroid']:.0f} Hz")
    print(f"  Pred weighted centroid: {o['pred_weighted_centroid']:.0f} Hz")
    print(f"  GT total energy: {o['gt_total_energy']:.3f}")
    print(f"  Pred total energy: {o['pred_total_energy']:.3f}")
    print(f"  GT max amp: {o['gt_max_amp']:.4f}")
    print(f"  Pred max amp: {o['pred_max_amp']:.4f}")

    print("\n2. PER-FRAME TOP-5 SINE MATCHING")
    print("-"*50)

    all_errors = []
    all_amp_ratios = []

    for frame in results['per_frame']:
        print(f"\n  Frame {frame['frame']}:")
        print(f"    GT top 5 freqs:   {[f'{f:.0f}' for f in frame['gt_top5_freqs']]}")
        print(f"    Pred top 5 freqs: {[f'{f:.0f}' for f in frame['pred_top5_freqs']]}")
        print(f"    GT top 5 amps:    {[f'{a:.4f}' for a in frame['gt_top5_amps']]}")
        print(f"    Pred top 5 amps:  {[f'{a:.4f}' for a in frame['pred_top5_amps']]}")

        print(f"    Matches (GT → closest Pred):")
        for m in frame['matches']:
            print(f"      {m['gt_freq']:.0f}Hz ({m['gt_amp']:.4f}) → "
                  f"{m['pred_freq']:.0f}Hz ({m['pred_amp']:.4f}): "
                  f"{m['semitone_error']:+.1f} st, amp×{m['amp_ratio']:.2f}")
            all_errors.append(abs(m['semitone_error']))
            all_amp_ratios.append(m['amp_ratio'])

    print("\n3. ERROR STATISTICS")
    print("-"*50)
    print(f"  Mean |semitone error|: {np.mean(all_errors):.1f}")
    print(f"  Median |semitone error|: {np.median(all_errors):.1f}")
    print(f"  Max |semitone error|: {np.max(all_errors):.1f}")
    print(f"  Errors > 6 semitones: {sum(e > 6 for e in all_errors)}/{len(all_errors)}")
    print(f"  Errors > 12 semitones: {sum(e > 12 for e in all_errors)}/{len(all_errors)}")

    print(f"\n  Mean amp ratio: {np.mean(all_amp_ratios):.2f}")
    print(f"  Amp ratios < 0.5 or > 2: {sum(r < 0.5 or r > 2 for r in all_amp_ratios)}/{len(all_amp_ratios)}")

    print("\n4. THE PROBLEM")
    print("-"*50)
    print("""
    The training loss computes average error over all 64 sines.
    But audio perception weights by amplitude - wrong loud sines hurt more!

    If the TOP sines (by amplitude) are wrong, audio sounds bad even if
    average error is low.

    Solutions:
    1. Amplitude-weighted loss (penalize errors on loud sines more)
    2. Audio-domain loss (compare synthesized spectrograms)
    3. Skip mel entirely: z → sines directly
    """)

    # Plot frequency distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Freq histogram
    ax = axes[0, 0]
    ax.hist(gt_freqs.numpy().flatten(), bins=50, alpha=0.5, label='GT', density=True)
    ax.hist(pred_freqs.numpy().flatten(), bins=50, alpha=0.5, label='Pred', density=True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Density')
    ax.set_title('Frequency Distribution')
    ax.legend()

    # Amp histogram
    ax = axes[0, 1]
    ax.hist(gt_amps.numpy().flatten(), bins=50, alpha=0.5, label='GT', density=True)
    ax.hist(pred_amps.numpy().flatten(), bins=50, alpha=0.5, label='Pred', density=True)
    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Density')
    ax.set_title('Amplitude Distribution')
    ax.legend()

    # Scatter: GT freq vs Pred freq (weighted by GT amp)
    ax = axes[1, 0]
    gt_f = gt_freqs.numpy().flatten()
    gt_a = gt_amps.numpy().flatten()
    pred_f = pred_freqs.numpy().flatten()
    # For each GT sine, find closest pred
    for i in range(0, len(gt_f), 100):  # Subsample
        closest = np.argmin(np.abs(pred_f - gt_f[i]))
        ax.scatter(gt_f[i], pred_f[closest], s=gt_a[i]*1000, alpha=0.3)
    ax.plot([0, 8000], [0, 8000], 'r--', label='Perfect')
    ax.set_xlabel('GT Frequency (Hz)')
    ax.set_ylabel('Pred Frequency (Hz)')
    ax.set_title('Frequency Matching (size=amplitude)')
    ax.set_xlim(0, 4000)
    ax.set_ylim(0, 4000)

    # Mel bin visualization
    ax = axes[1, 1]
    mel_freqs = MEL_FREQS.numpy()
    ax.bar(range(len(mel_freqs)), mel_freqs, alpha=0.5)
    ax.set_xlabel('Mel Bin')
    ax.set_ylabel('Center Frequency (Hz)')
    ax.set_title('Mel Bin Frequencies (what SMS mapper selects from)')

    plt.tight_layout()
    plt.savefig('outputs/sms_diagnosis.png', dpi=150)
    print(f"\nSaved diagnosis plot to outputs/sms_diagnosis.png")


if __name__ == "__main__":
    main()
