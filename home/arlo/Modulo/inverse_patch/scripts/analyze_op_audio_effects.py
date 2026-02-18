#!/usr/bin/env python3
"""
Paired Difference Analysis: Do operation tree operations have
consistent audio-domain effects across samples?

For each top operation (by weight norm), across 30+ samples:
  1. Decode z_gt_sms + res_mean (no ops) -> audio_base
  2. Decode z_gt_sms + res_mean + op_k_contrib -> audio_with_op
  3. audio_diff = audio_with_op - audio_base
  4. STFT(audio_diff) / (|STFT(audio_base)| + eps)  ->  spectral ratio

Then analyze consistency:
  - Mean/std spectral ratio across samples
  - Pairwise correlation of spectral profiles
  - Time-domain analysis (attack/sustain/release)
  - RMS of diff relative to original
  - Spectral centroid of the diff

If spectral ratios are highly correlated across samples, the operation
has a consistent audio-domain effect that could be parameterized as DSP.
"""

import sys
import os
import gc
import torch
import torch.nn.functional as TF
import numpy as np
import orjson
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM
from phase2_operation_tree import OperationTreeCodec
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
GT_TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_gt_tree" / "operation_tree_gt.pt"
DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_DATA_DIR = SCRIPT_DIR.parent / "data" / "sms_v4"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "op_audio_analysis"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

Z_DIM = 128
SAMPLE_RATE = 44100
N_FFT = 2048
HOP_LENGTH = 512
MIN_SAMPLES = 30


# ============================================================
# Model loading
# ============================================================

def load_models(device):
    """Load Phase 1 codec, GT tree, and DCAE."""
    print("Loading Phase 1 codec...")
    codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()

    print("Loading GT operation tree...")
    gt_ckpt = torch.load(GT_TREE_PATH, weights_only=False, map_location='cpu')
    gt_tree = OperationTreeCodec(
        z_dim=Z_DIM, n_ops=gt_ckpt['n_ops'], param_dim=gt_ckpt['param_dim'],
        encoder_hidden=256, top_k=gt_ckpt['top_k'],
    )
    gt_tree.load_state_dict(gt_ckpt['model'])
    gt_tree = gt_tree.to(device).eval()
    gt_res_mean = gt_ckpt['res_mean'].to(device)
    gt_res_std = gt_ckpt['res_std'].to(device)

    # Operation ranking by output weight norm
    state = gt_ckpt['model']
    importance = []
    for k in range(gt_ckpt['n_ops']):
        key = f'operations.{k}.net.4.weight'
        importance.append(state[key].norm().item() if key in state else 0.0)
    op_ranking = sorted(range(len(importance)), key=lambda i: -importance[i])

    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()

    print(f"  GT tree: {gt_ckpt['n_ops']} ops, param_dim={gt_ckpt['param_dim']}, "
          f"top_k={gt_ckpt['top_k']}")
    print(f"  Op ranking by weight norm: {op_ranking}")

    return {
        'codec': codec,
        'gt_tree': gt_tree,
        'dcae': dcae,
        'gt_res_mean': gt_res_mean,
        'gt_res_std': gt_res_std,
        'n_ops': gt_ckpt['n_ops'],
        'op_ranking': op_ranking,
        'device': device,
    }


# ============================================================
# Data loading: build sample pairs from Phase 1 cache + manifest
# ============================================================

def load_sample_pairs(max_samples=60):
    """
    Load sample pairs: z_real (from latent files) + z_gt_sms (from data cache).
    Returns list of dicts with keys: z_gt_sms_flat, z_real_flat, latent_path, sms_path.
    """
    print(f"\nLoading Phase 1 data cache from {DATA_CACHE_PATH}...")
    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')
    print(f"  {len(cache)} cached entries")

    print(f"Loading manifest from {MANIFEST_PATH}...")
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']
    print(f"  {len(sms_to_latent)} manifest entries")

    pairs = []
    skipped = 0

    for sample in cache:
        if len(pairs) >= max_samples:
            break

        sms_path = sample['path']
        z_gt_sms_4d = sample['z_sms']  # [8, 16, T_cache]
        T_cache = z_gt_sms_4d.shape[2]

        latent_path = sms_to_latent.get(sms_path)
        if not latent_path or not os.path.exists(latent_path):
            skipped += 1
            continue

        # Load z_real
        try:
            loaded = torch.load(latent_path, weights_only=False, map_location='cpu')
            if isinstance(loaded, dict) and 'latents' in loaded:
                z_real_full = loaded['latents']
            elif isinstance(loaded, torch.Tensor):
                z_real_full = loaded
            else:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        if z_real_full.dim() != 3 or z_real_full.shape[0] != 8 or z_real_full.shape[1] != 16:
            skipped += 1
            continue

        # Find start_frame from SMS file
        sms_file = sms_path
        if not os.path.exists(sms_file):
            sms_file = str(SCRIPT_DIR.parent / sms_path)
        if not os.path.exists(sms_file):
            sms_file = str(SMS_DATA_DIR / Path(sms_path).name)

        start_frame = 0
        if os.path.exists(sms_file):
            try:
                sms_data = torch.load(sms_file, weights_only=True, map_location='cpu')
                frame_energy = sms_data['amps'].sum(dim=1)
                for t in range(len(frame_energy)):
                    if frame_energy[t] > 0.001:
                        start_frame = t
                        break
            except Exception:
                pass

        # Crop z_real to match z_gt_sms temporal window
        end_frame = min(start_frame + T_cache, z_real_full.shape[2])
        T_aligned = end_frame - start_frame
        if T_aligned < 10:
            skipped += 1
            continue

        z_gt_sms_crop = z_gt_sms_4d[:, :, :T_aligned]
        z_real_crop = z_real_full[:, :, start_frame:end_frame]

        # Flatten both to [1, T, 128]
        z_gt_sms_flat = z_gt_sms_crop.permute(2, 0, 1).reshape(1, T_aligned, Z_DIM)
        z_real_flat = z_real_crop.permute(2, 0, 1).reshape(1, T_aligned, Z_DIM)

        pairs.append({
            'z_gt_sms_flat': z_gt_sms_flat,
            'z_real_flat': z_real_flat,
            'latent_path': latent_path,
            'sms_path': sms_path,
            'T': T_aligned,
        })

        if len(pairs) % 10 == 0:
            print(f"    Loaded {len(pairs)} pairs...")

    print(f"  Total: {len(pairs)} pairs ({skipped} skipped)")
    return pairs


# ============================================================
# DCAE decode helper
# ============================================================

def decode_z_to_audio(dcae, z_flat_cpu, device):
    """
    z_flat [1, T, 128] CPU tensor -> numpy audio array.
    """
    z_flat = z_flat_cpu.to(device)
    B, T, D = z_flat.shape
    z_4d = z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)  # [1, 8, 16, T]
    audio_len = int(T * SAMPLE_RATE / 10.8)
    audio_lengths = torch.tensor([audio_len], device=device)

    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)

    audio = wavs[0].mean(dim=0).cpu().numpy()
    return audio


# ============================================================
# GT tree decomposition (per-op contributions)
# ============================================================

def decompose_gt_tree(models, z_gt_sms_flat, z_real_flat):
    """
    Decompose using GT tree.

    Args:
        models: dict from load_models()
        z_gt_sms_flat: [1, T, 128] CPU tensor
        z_real_flat: [1, T, 128] CPU tensor

    Returns:
        gt_res_mean: [1, 1, 128] on device
        op_contribs: list of [1, T, 128] CPU tensors (per-op contribution, denormalized)
        alpha_sparse: [1, T, n_ops] CPU tensor
    """
    device = models['device']
    gt_tree = models['gt_tree']
    gt_res_mean = models['gt_res_mean']
    gt_res_std = models['gt_res_std']

    z_gt_sms = z_gt_sms_flat.to(device)
    z_real = z_real_flat.to(device)

    with torch.no_grad():
        gt_residual = z_real - z_gt_sms
        gt_residual_norm = (gt_residual - gt_res_mean) / gt_res_std

        gt_hidden = gt_tree.encode(gt_residual_norm)

        # Sparse alpha (same as gradio_ui.py)
        gt_raw_alpha = gt_tree.activation_head(gt_hidden)
        gt_alpha_full = TF.softplus(gt_raw_alpha)
        gt_topk_vals, gt_topk_idx = torch.topk(gt_alpha_full, gt_tree.top_k, dim=-1)
        gt_mask = torch.zeros_like(gt_alpha_full)
        gt_mask.scatter_(-1, gt_topk_idx, 1.0)
        gt_alpha_sparse = gt_alpha_full * gt_mask

        # Per-op contributions: (alpha_sparse[:,:,k:k+1] * contrib_k) * gt_res_std
        op_contribs = []
        for k in range(gt_tree.n_ops):
            gp = gt_tree.param_heads[k](gt_hidden)
            gc = gt_tree.operations[k](gp)
            ga = gt_alpha_sparse[:, :, k:k+1]
            gt_scaled = (ga * gc) * gt_res_std
            op_contribs.append(gt_scaled.cpu())

    return gt_res_mean.cpu(), op_contribs, gt_alpha_sparse.cpu()


# ============================================================
# Spectral analysis utilities
# ============================================================

def compute_stft_magnitude(audio, n_fft=N_FFT, hop_length=HOP_LENGTH):
    """
    Compute STFT magnitude.
    Returns: mag [n_freq_bins, n_time_frames], freqs [n_freq_bins]
    """
    # Pad audio to be a multiple of hop_length
    pad_len = n_fft - (len(audio) % hop_length)
    if pad_len > 0 and pad_len < n_fft:
        audio = np.pad(audio, (0, pad_len))

    window = np.hanning(n_fft)
    n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
    n_freq = n_fft // 2 + 1

    stft_mag = np.zeros((n_freq, n_frames), dtype=np.float32)
    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))
        spectrum = np.fft.rfft(frame * window)
        stft_mag[:, i] = np.abs(spectrum)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / SAMPLE_RATE)
    return stft_mag, freqs


def spectral_ratio_profile(stft_with, stft_base, eps=1e-8):
    """
    Compute |STFT(audio_with)| / (|STFT(audio_base)| + eps), averaged over time.
    Returns: [n_freq_bins] mean ratio profile.
    """
    # Match time dimensions
    T_min = min(stft_with.shape[1], stft_base.shape[1])
    ratio = stft_with[:, :T_min] / (stft_base[:, :T_min] + eps)
    # Average over time
    return ratio.mean(axis=1)


def spectral_centroid_from_stft(stft_mag, freqs):
    """Compute spectral centroid from STFT magnitude, averaged over time."""
    # Per-frame centroid
    total_energy = stft_mag.sum(axis=0) + 1e-10  # [n_frames]
    weighted_freq = (freqs[:, None] * stft_mag).sum(axis=0)  # [n_frames]
    centroids = weighted_freq / total_energy
    return float(np.mean(centroids))


def time_domain_envelope(audio, hop=HOP_LENGTH):
    """
    Compute RMS envelope of audio in frames.
    Returns: rms_envelope [n_frames]
    """
    n_frames = max(1, len(audio) // hop)
    envelope = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = min(start + hop, len(audio))
        chunk = audio[start:end]
        envelope[i] = np.sqrt(np.mean(chunk ** 2)) if len(chunk) > 0 else 0.0
    return envelope


def analyze_temporal_regions(diff_envelope, base_envelope, n_segments=3):
    """
    Split the envelope into n_segments equal parts (attack/sustain/release proxy)
    and compute the ratio of diff energy to base energy in each segment.

    Returns: list of (segment_name, energy_ratio) tuples.
    """
    n = len(diff_envelope)
    if n < n_segments:
        return [("all", 0.0)]

    seg_len = n // n_segments
    names = ["attack", "sustain", "release"] if n_segments == 3 else [f"seg_{i}" for i in range(n_segments)]
    results = []

    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else n
        d_rms = np.sqrt(np.mean(diff_envelope[start:end] ** 2)) if end > start else 0.0
        b_rms = np.sqrt(np.mean(base_envelope[start:end] ** 2)) if end > start else 0.0
        ratio = d_rms / (b_rms + 1e-10)
        results.append((names[i], ratio))

    return results


# ============================================================
# Main analysis
# ============================================================

def analyze_operations(models, pairs, top_k_ops=4):
    """
    For each top operation, across all sample pairs:
      - Decode base (z_gt_sms + res_mean, no ops) -> audio_base
      - Decode base + op_k_contrib -> audio_with_op
      - Compute spectral ratio, temporal analysis, etc.
      - Analyze consistency across samples.
    """
    device = models['device']
    dcae = models['dcae']
    op_ranking = models['op_ranking']
    n_ops = models['n_ops']

    top_ops = op_ranking[:top_k_ops]
    n_samples = len(pairs)

    print(f"\n{'='*60}")
    print(f"PAIRED DIFFERENCE ANALYSIS")
    print(f"{'='*60}")
    print(f"  Top {top_k_ops} ops (by weight norm): {top_ops}")
    print(f"  Samples: {n_samples}")
    print()

    # Storage for per-op, per-sample results
    # spectral_ratios[op_idx] = list of [n_freq] arrays
    spectral_ratios = {k: [] for k in top_ops}
    rms_ratios = {k: [] for k in top_ops}
    diff_centroids = {k: [] for k in top_ops}
    temporal_regions = {k: [] for k in top_ops}  # list of [(name, ratio), ...]
    diff_stft_profiles = {k: [] for k in top_ops}  # |STFT(diff)| averaged over time

    freqs = None  # will be set on first STFT computation

    for si, pair in enumerate(pairs):
        z_gt_sms_flat = pair['z_gt_sms_flat']
        z_real_flat = pair['z_real_flat']
        T = pair['T']

        print(f"  Sample {si+1}/{n_samples} (T={T})...", end="", flush=True)

        # Decompose GT tree
        gt_res_mean, op_contribs, alpha_sparse = decompose_gt_tree(
            models, z_gt_sms_flat, z_real_flat
        )

        # Build base z: z_gt_sms + res_mean (no ops)
        z_base = z_gt_sms_flat + gt_res_mean

        # Decode base audio
        audio_base = decode_z_to_audio(dcae, z_base, device)
        base_rms = np.sqrt(np.mean(audio_base ** 2))

        if base_rms < 1e-6:
            print(" [skip: silent base]")
            continue

        # STFT of base
        stft_base, freqs_local = compute_stft_magnitude(audio_base)
        if freqs is None:
            freqs = freqs_local

        # Base envelope
        base_envelope = time_domain_envelope(audio_base)

        for k in top_ops:
            # Build z with op k: z_gt_sms + res_mean + op_k_contrib
            z_with_op = z_base + op_contribs[k][:, :T]

            # Decode
            audio_with_op = decode_z_to_audio(dcae, z_with_op, device)

            # Audio diff
            min_len = min(len(audio_with_op), len(audio_base))
            audio_diff = audio_with_op[:min_len] - audio_base[:min_len]

            # STFT of audio_with_op and audio_diff
            stft_with, _ = compute_stft_magnitude(audio_with_op)
            stft_diff, _ = compute_stft_magnitude(audio_diff)

            # Spectral ratio: |STFT(with)| / (|STFT(base)| + eps)
            ratio_profile = spectral_ratio_profile(stft_with, stft_base)
            spectral_ratios[k].append(ratio_profile)

            # |STFT(diff)| averaged over time (for seeing where the diff energy lives)
            diff_profile = stft_diff.mean(axis=1)
            diff_stft_profiles[k].append(diff_profile)

            # RMS of diff relative to base
            diff_rms = np.sqrt(np.mean(audio_diff ** 2))
            rms_ratios[k].append(diff_rms / (base_rms + 1e-10))

            # Spectral centroid of the diff
            diff_centroid = spectral_centroid_from_stft(stft_diff, freqs)
            diff_centroids[k].append(diff_centroid)

            # Temporal region analysis
            diff_envelope = time_domain_envelope(audio_diff)
            regions = analyze_temporal_regions(diff_envelope, base_envelope)
            temporal_regions[k].append(regions)

        print(" done")

        # Free GPU memory periodically
        if (si + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    return {
        'spectral_ratios': spectral_ratios,
        'diff_stft_profiles': diff_stft_profiles,
        'rms_ratios': rms_ratios,
        'diff_centroids': diff_centroids,
        'temporal_regions': temporal_regions,
        'freqs': freqs,
        'top_ops': top_ops,
    }


def analyze_consistency(results):
    """
    For each operation, analyze consistency of spectral effects across samples.
    """
    spectral_ratios = results['spectral_ratios']
    diff_stft_profiles = results['diff_stft_profiles']
    rms_ratios = results['rms_ratios']
    diff_centroids = results['diff_centroids']
    temporal_regions = results['temporal_regions']
    freqs = results['freqs']
    top_ops = results['top_ops']

    print(f"\n{'='*60}")
    print(f"CONSISTENCY ANALYSIS")
    print(f"{'='*60}")

    summary = {}

    for k in top_ops:
        ratios = spectral_ratios[k]
        if len(ratios) < 2:
            print(f"\n  Op {k}: too few samples ({len(ratios)}), skipping")
            continue

        # Stack into matrix [n_samples, n_freq]
        # Truncate to minimum length across samples
        min_freq_len = min(len(r) for r in ratios)
        ratio_matrix = np.array([r[:min_freq_len] for r in ratios])
        diff_matrix = np.array([d[:min_freq_len] for d in diff_stft_profiles[k]])
        n_samples = ratio_matrix.shape[0]

        # Mean and std spectral ratio
        mean_ratio = ratio_matrix.mean(axis=0)
        std_ratio = ratio_matrix.std(axis=0)

        # Pairwise correlation of spectral ratio profiles
        correlations = []
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                r_i = ratio_matrix[i]
                r_j = ratio_matrix[j]
                if np.std(r_i) < 1e-8 or np.std(r_j) < 1e-8:
                    continue
                corr = np.corrcoef(r_i, r_j)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        mean_corr = float(np.mean(correlations)) if correlations else 0.0
        std_corr = float(np.std(correlations)) if correlations else 0.0
        median_corr = float(np.median(correlations)) if correlations else 0.0

        # Mean RMS ratio
        mean_rms_ratio = float(np.mean(rms_ratios[k]))
        std_rms_ratio = float(np.std(rms_ratios[k]))

        # Mean spectral centroid of diff
        mean_centroid = float(np.mean(diff_centroids[k]))
        std_centroid = float(np.std(diff_centroids[k]))

        # Temporal region analysis: average across samples
        region_names = [r[0] for r in temporal_regions[k][0]]
        n_regions = len(region_names)
        region_means = []
        region_stds = []
        for ri in range(n_regions):
            vals = [temporal_regions[k][si][ri][1] for si in range(n_samples)]
            region_means.append(float(np.mean(vals)))
            region_stds.append(float(np.std(vals)))

        # Find where the spectral ratio is max/min (boost/cut frequencies)
        freq_bins = freqs[:min_freq_len] if freqs is not None else np.arange(min_freq_len)

        # Smooth mean ratio for peak/trough finding
        kernel_size = max(3, min_freq_len // 50)
        if kernel_size % 2 == 0:
            kernel_size += 1
        smoothed = np.convolve(mean_ratio, np.ones(kernel_size) / kernel_size, mode='same')

        # Find frequency of max boost and max cut
        max_boost_idx = np.argmax(smoothed)
        # For cut, look for minimum below 1.0
        below_one = smoothed.copy()
        below_one[below_one >= 1.0] = 1.0
        max_cut_idx = np.argmin(below_one)

        max_boost_freq = freq_bins[max_boost_idx] if max_boost_idx < len(freq_bins) else 0
        max_cut_freq = freq_bins[max_cut_idx] if max_cut_idx < len(freq_bins) else 0
        max_boost_val = float(smoothed[max_boost_idx])
        max_cut_val = float(smoothed[max_cut_idx])

        # Compute frequency band summary (low/mid/high)
        low_mask = freq_bins < 500
        mid_mask = (freq_bins >= 500) & (freq_bins < 4000)
        high_mask = freq_bins >= 4000

        low_ratio = float(mean_ratio[low_mask].mean()) if low_mask.any() else 1.0
        mid_ratio = float(mean_ratio[mid_mask].mean()) if mid_mask.any() else 1.0
        high_ratio = float(mean_ratio[high_mask].mean()) if high_mask.any() else 1.0

        print(f"\n  OP {k} (rank {top_ops.index(k)}):")
        print(f"  {'='*50}")

        print(f"  Spectral ratio (|with| / |base|):")
        print(f"    Low (<500 Hz):     {low_ratio:.4f}")
        print(f"    Mid (500-4k Hz):   {mid_ratio:.4f}")
        print(f"    High (>4k Hz):     {high_ratio:.4f}")
        print(f"    Max boost: {max_boost_val:.4f} at {max_boost_freq:.0f} Hz")
        print(f"    Max cut:   {max_cut_val:.4f} at {max_cut_freq:.0f} Hz")

        print(f"  Pairwise consistency:")
        print(f"    Mean correlation:   {mean_corr:.4f}")
        print(f"    Median correlation: {median_corr:.4f}")
        print(f"    Std correlation:    {std_corr:.4f}")
        print(f"    n_pairs:            {len(correlations)}")

        if mean_corr > 0.7:
            verdict = "HIGHLY CONSISTENT - strong candidate for DSP parameterization"
        elif mean_corr > 0.4:
            verdict = "MODERATELY CONSISTENT - partially generalizable effect"
        elif mean_corr > 0.2:
            verdict = "WEAKLY CONSISTENT - sample-dependent with some shared structure"
        else:
            verdict = "NOT CONSISTENT - effect is sample-specific"
        print(f"    Verdict:            {verdict}")

        print(f"  Energy analysis:")
        print(f"    RMS(diff)/RMS(base):  {mean_rms_ratio:.4f} +/- {std_rms_ratio:.4f}")
        print(f"    Diff centroid:        {mean_centroid:.0f} Hz +/- {std_centroid:.0f} Hz")

        print(f"  Temporal distribution (diff energy by region):")
        for ri, name in enumerate(region_names):
            print(f"    {name:>10}: {region_means[ri]:.4f} +/- {region_stds[ri]:.4f}")

        # Determine if effect is concentrated in a region
        if n_regions >= 3 and max(region_means) > 0:
            dominant_region = region_names[np.argmax(region_means)]
            ratio_dom = max(region_means) / (np.mean(region_means) + 1e-10)
            if ratio_dom > 1.5:
                print(f"    -> Effect concentrated in {dominant_region} "
                      f"(ratio={ratio_dom:.2f}x vs mean)")
            else:
                print(f"    -> Effect spread across all regions (ratio={ratio_dom:.2f}x)")

        summary[k] = {
            'mean_corr': mean_corr,
            'median_corr': median_corr,
            'std_corr': std_corr,
            'mean_rms_ratio': mean_rms_ratio,
            'std_rms_ratio': std_rms_ratio,
            'mean_centroid': mean_centroid,
            'std_centroid': std_centroid,
            'low_ratio': low_ratio,
            'mid_ratio': mid_ratio,
            'high_ratio': high_ratio,
            'max_boost_freq': float(max_boost_freq),
            'max_boost_val': max_boost_val,
            'max_cut_freq': float(max_cut_freq),
            'max_cut_val': max_cut_val,
            'temporal_regions': list(zip(region_names, region_means, region_stds)),
            'n_samples': n_samples,
            'verdict': verdict,
        }

    return summary


def print_summary(summary, top_ops):
    """Print a final summary table."""
    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    print()
    print(f"  {'Op':>4}  {'Rank':>4}  {'Corr':>6}  {'RMS%':>6}  {'Centroid':>8}  "
          f"{'Low':>6}  {'Mid':>6}  {'High':>6}  Verdict")
    print(f"  {'----':>4}  {'----':>4}  {'------':>6}  {'------':>6}  {'--------':>8}  "
          f"{'------':>6}  {'------':>6}  {'------':>6}  -------")

    for rank, k in enumerate(top_ops):
        if k not in summary:
            continue
        s = summary[k]
        consistent = "YES" if s['mean_corr'] > 0.4 else "no"
        print(f"  {k:4d}  {rank:4d}  {s['mean_corr']:6.3f}  "
              f"{s['mean_rms_ratio']*100:5.1f}%  "
              f"{s['mean_centroid']:7.0f}Hz  "
              f"{s['low_ratio']:6.3f}  {s['mid_ratio']:6.3f}  {s['high_ratio']:6.3f}  "
              f"{consistent}")

    print()
    print("  Key:")
    print("    Corr   = mean pairwise correlation of spectral ratio profiles")
    print("    RMS%   = diff RMS as % of base RMS (how much the op changes)")
    print("    Centroid = spectral centroid of the diff (where the change lives)")
    print("    Low/Mid/High = mean spectral ratio in each band (1.0 = no change)")
    print("    Verdict: YES = corr > 0.4, consistent enough for DSP parameterization")
    print()

    # Overall assessment
    consistent_ops = [k for k in top_ops if k in summary and summary[k]['mean_corr'] > 0.4]
    total_ops = len([k for k in top_ops if k in summary])
    print(f"  {len(consistent_ops)}/{total_ops} operations show consistent audio-domain effects.")

    if len(consistent_ops) > total_ops / 2:
        print("  >> Operations have CONSISTENT spectral signatures across samples.")
        print("     They capture transferable audio effects, not sample-specific artifacts.")
        print("     Strong candidates for DSP parameterization (EQ curves, filter shapes, etc.)")
    elif consistent_ops:
        print("  >> Some operations are consistent, others are sample-specific.")
        print("     Partial DSP parameterization is feasible for the consistent ops.")
    else:
        print("  >> Operations are mostly sample-specific.")
        print("     The effects don't generalize to fixed DSP curves.")
        print("     They may still capture meaningful per-sample structure.")


def save_results(results, summary, output_dir):
    """Save spectral profiles and summary to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    top_ops = results['top_ops']
    freqs = results['freqs']

    # Save per-op spectral ratio matrices
    for k in top_ops:
        ratios = results['spectral_ratios'][k]
        if not ratios:
            continue
        min_len = min(len(r) for r in ratios)
        ratio_matrix = np.array([r[:min_len] for r in ratios])
        np.save(str(output_dir / f"op{k}_spectral_ratios.npy"), ratio_matrix)

        diff_profiles = results['diff_stft_profiles'][k]
        diff_matrix = np.array([d[:min_len] for d in diff_profiles])
        np.save(str(output_dir / f"op{k}_diff_stft_profiles.npy"), diff_matrix)

    # Save frequency axis
    if freqs is not None:
        np.save(str(output_dir / "freqs.npy"), freqs)

    # Save summary as JSON-compatible dict
    summary_serializable = {}
    for k, v in summary.items():
        summary_serializable[str(k)] = {
            key: (val if not isinstance(val, np.floating) else float(val))
            for key, val in v.items()
        }

    summary_path = output_dir / "summary.json"
    with open(summary_path, 'wb') as f:
        f.write(orjson.dumps(summary_serializable, option=orjson.OPT_INDENT_2))

    # Save RMS ratios and centroids
    for k in top_ops:
        np.save(str(output_dir / f"op{k}_rms_ratios.npy"), np.array(results['rms_ratios'][k]))
        np.save(str(output_dir / f"op{k}_diff_centroids.npy"), np.array(results['diff_centroids'][k]))

    print(f"\n  Results saved to {output_dir}")
    print(f"  Files:")
    print(f"    summary.json                  - Overall summary and verdicts")
    print(f"    freqs.npy                     - Frequency axis for spectral profiles")
    print(f"    op<k>_spectral_ratios.npy     - [n_samples, n_freq] ratio profiles")
    print(f"    op<k>_diff_stft_profiles.npy  - [n_samples, n_freq] diff magnitude profiles")
    print(f"    op<k>_rms_ratios.npy          - [n_samples] RMS ratio per sample")
    print(f"    op<k>_diff_centroids.npy      - [n_samples] centroid per sample")


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("OPERATION AUDIO EFFECT ANALYSIS")
    print("Paired Difference: do ops have consistent spectral effects?")
    print("=" * 60)
    print()
    print(f"  Device: {device}")
    print(f"  STFT: n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    # Load models
    models = load_models(device)

    # Load sample pairs
    pairs = load_sample_pairs(max_samples=60)

    if len(pairs) < MIN_SAMPLES:
        print(f"\nERROR: Only {len(pairs)} pairs loaded, need at least {MIN_SAMPLES}.")
        print("Check that data cache and latent files are accessible.")
        return

    # Run paired difference analysis for top 4 ops
    results = analyze_operations(models, pairs, top_k_ops=4)

    # Analyze consistency
    summary = analyze_consistency(results)

    # Print final summary
    print_summary(summary, results['top_ops'])

    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_results(results, summary, OUTPUT_DIR)

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
