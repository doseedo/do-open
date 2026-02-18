#!/usr/bin/env python3
"""
Verify whether operation tree decomposition is MEANINGFUL or just an arbitrary split.

Tests:
  1. Operation swap (interpolation) — swap one op's params between samples
  2. Param sweep — scale one op's contribution within a sample
  3. Consistency — do ops activate differently for different instruments?
  4. Audio feature correlation — do op activations correlate with spectral features?

Uses instrument-specific samples: piano, strings, brass, winds
"""

import sys
import torch
import torch.nn.functional as TF
import numpy as np
from pathlib import Path
import torchaudio
import gc
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, "/home/arlo/Data/ACE-Step")

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM
from phase2_operation_tree import OperationTreeCodec

from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree" / "operation_tree.pt"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "verify_operation_meaning"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
Z_DIM = 128
MAX_FRAMES = 256

# Instrument sample paths (verified working)
INSTRUMENT_SAMPLES = {
    'piano': [
        '/home/arlo/gcs-bucket/Latents/protools/2025-03-28/New/29Sep_Jocelyn_Vox_Sess_DONE/Audio Files/人生,起起落落落落落_PianoOnly.pt',
        '/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/01-I Can\'t Have You by-Austin Armstrong/Audio Files/PIANO FX 1.pt',
        '/home/arlo/gcs-bucket/Latents/protools/2025-03-29/New/2025.02.21_Sueñero_VoxRec/Audio Files/Piano_bip_1.pt',
    ],
    'strings': [
        '/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/01-I Can\'t Have You by-Austin Armstrong/Audio Files/Strings L.L.pt',
        '/home/arlo/gcs-bucket/Latents/protools/2025-03-30/New/01-I Can\'t Have You by-Austin Armstrong/Audio Files/Strings L.R.pt',
    ],
    'brass': [
        '/home/arlo/gcs-bucket/Latents/protools/2025-04-01/New/LuciaSageMaggie_MP-212 Lab 3 - Salvation/Audio Files/New Horns_01.L.pt',
        '/home/arlo/gcs-bucket/Latents/protools/2025-04-02/New/Joav .04/Audio Files/414 trumpet.02_05.pt',
    ],
    'winds': [
        '/home/arlo/gcs-bucket/Latents/protools/2025-03-31/New/Pro Tools Hands On Music Project_JMartin/Audio Files/Driving-TronFlutes-51.pt',
        '/home/arlo/gcs-bucket/Latents/protools/2025-04-01/New/Pro Tools Sessions/THE SUN MIX/Audio Files/SAX.cm.pt',
        '/home/arlo/gcs-bucket/Latents/protools/2025-04-01/New/Pro Tools Sessions/Too Good (Madyson Session)/Audio Files/Sax (Ronnie).pt',
    ],
}


def decode_z_flat(dcae, z_flat, device):
    """z_flat [1, T, 128] → audio numpy."""
    B, T, D = z_flat.shape
    z_4d = z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)
    audio_len = int(T * SAMPLE_RATE / 10.8)
    audio_lengths = torch.tensor([audio_len], device=device)
    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)
    return wavs[0].mean(dim=0).cpu().numpy()


def save_wav(audio, path):
    t = torch.from_numpy(audio).float().unsqueeze(0)
    peak = t.abs().max()
    if peak > 0.95:
        t = t * (0.95 / peak)
    torchaudio.save(str(path), t, SAMPLE_RATE)


def load_models(device):
    print("Loading Phase 1 codec...")
    codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()

    print("Loading operation tree...")
    ckpt = torch.load(TREE_PATH, weights_only=False, map_location='cpu')
    tree = OperationTreeCodec(
        z_dim=Z_DIM, n_ops=ckpt['n_ops'], param_dim=ckpt['param_dim'],
        encoder_hidden=256, top_k=ckpt['top_k'],
    )
    tree.load_state_dict(ckpt['model'])
    tree = tree.to(device).eval()
    res_mean = ckpt['res_mean'].to(device)
    res_std = ckpt['res_std'].to(device)

    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()

    return codec, tree, dcae, res_mean, res_std


def load_latent(path):
    """Load a latent file → z [8, 16, T], truncated to MAX_FRAMES."""
    d = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(d, dict) and 'latents' in d:
        z = d['latents']
    elif isinstance(d, torch.Tensor):
        z = d
    else:
        raise ValueError(f"Unknown format in {path}")

    if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16:
        raise ValueError(f"Bad shape {z.shape} in {path}")

    if z.shape[2] > MAX_FRAMES:
        z = z[:, :, :MAX_FRAMES]
    return z


def load_instrument_samples():
    """Load all instrument samples, return dict of category → list of loaded samples."""
    samples = {}
    for cat, paths in INSTRUMENT_SAMPLES.items():
        samples[cat] = []
        for p in paths:
            try:
                z = load_latent(p)
                name = Path(p).stem
                samples[cat].append({'z_real': z, 'path': p, 'name': name, 'category': cat})
                print(f"  [{cat}] {name}: T={z.shape[2]}")
            except Exception as e:
                print(f"  [{cat}] SKIP {Path(p).name}: {e}")
        if not samples[cat]:
            print(f"  WARNING: no samples for {cat}!")
    return samples


def decompose_sample(codec, tree, res_mean, res_std, z_real_4d, device):
    """
    Run full decomposition pipeline.
    Returns dict with z_flat, z_sms, residual, hidden, all_params, activations, per-op contributions.
    """
    z_real = z_real_4d.unsqueeze(0).to(device)
    z_flat = codec.z_to_flat(z_real)

    with torch.no_grad():
        z_sms = codec.forward_G(codec.forward_F(z_flat))
        residual = z_flat - z_sms
        residual_norm = (residual - res_mean) / res_std

        hidden = tree.encode(residual_norm)
        recon_norm, activations, all_params = tree.decode(hidden)
        recon = recon_norm * res_std + res_mean
        z_full = z_sms + recon

        # Per-op contributions (unnormalized)
        raw_alpha = tree.activation_head(hidden)
        alpha = TF.softplus(raw_alpha)

        op_contribs = []
        for k in range(tree.n_ops):
            params_k = tree.param_heads[k](hidden)
            contrib_k = tree.operations[k](params_k)
            alpha_k = alpha[:, :, k:k+1]
            scaled = (alpha_k * contrib_k) * res_std + res_mean
            op_contribs.append(scaled)

    return {
        'z_flat': z_flat,
        'z_sms': z_sms,
        'residual_norm': residual_norm,
        'hidden': hidden,
        'activations': alpha,          # [1, T, n_ops] — full (not sparse)
        'all_params': all_params,      # [1, T, n_ops, param_dim]
        'z_full': z_full,
        'op_contribs': op_contribs,    # list of [1, T, 128]
    }


def compute_spectral_features(audio, sr=44100):
    """
    Compute basic spectral features from audio numpy array.
    Returns dict of scalar features.
    """
    # Trim silence
    threshold = np.abs(audio).max() * 0.01
    nonzero = np.where(np.abs(audio) > threshold)[0]
    if len(nonzero) < sr // 10:
        return {'centroid': 0, 'bandwidth': 0, 'rms': 0, 'zcr': 0, 'rolloff': 0}

    audio = audio[nonzero[0]:nonzero[-1]+1]

    # RMS energy
    rms = np.sqrt(np.mean(audio ** 2))

    # Zero crossing rate
    zcr = np.mean(np.abs(np.diff(np.sign(audio)))) / 2

    # Spectral features via FFT
    n_fft = 2048
    hop = 512
    n_frames = max(1, (len(audio) - n_fft) // hop + 1)

    centroids = []
    bandwidths = []
    rolloffs = []

    for i in range(n_frames):
        start = i * hop
        frame = audio[start:start+n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        spec = np.abs(np.fft.rfft(frame * np.hanning(n_fft)))
        freqs = np.fft.rfftfreq(n_fft, 1.0/sr)

        total = spec.sum() + 1e-10
        centroid = np.sum(freqs * spec) / total
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spec) / total)

        cumsum = np.cumsum(spec)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs)-1)]

        centroids.append(centroid)
        bandwidths.append(bandwidth)
        rolloffs.append(rolloff)

    return {
        'centroid': float(np.mean(centroids)),
        'bandwidth': float(np.mean(bandwidths)),
        'rms': float(rms),
        'zcr': float(zcr),
        'rolloff': float(np.mean(rolloffs)),
    }


# ============================================================
# TEST 1: Operation Swap (Cross-sample interpolation)
# ============================================================

def test_operation_swap(codec, tree, dcae, res_mean, res_std, samples_by_cat, device):
    """
    Swap one operation's params between two different-category samples.
    If operations are meaningful, swapping should transfer one quality.
    If entangled, result will sound broken.
    """
    print("\n" + "=" * 60)
    print("TEST 1: OPERATION SWAP (Cross-Instrument)")
    print("  Swap one op's contribution between two samples")
    print("=" * 60)

    swap_dir = OUTPUT_DIR / "01_operation_swap"
    swap_dir.mkdir(parents=True, exist_ok=True)

    # Pick one sample from each category
    cat_samples = {}
    for cat, slist in samples_by_cat.items():
        if slist:
            cat_samples[cat] = slist[0]

    if len(cat_samples) < 2:
        print("  Need at least 2 categories with samples, skipping")
        return

    # Get operation ranking
    op_ranking = get_op_ranking(tree)
    top_ops = op_ranking[:4]

    # Decompose all samples
    decomposed = {}
    for cat, sample in cat_samples.items():
        decomposed[cat] = decompose_sample(
            codec, tree, res_mean, res_std, sample['z_real'], device
        )

    # Test swaps between each pair of categories
    pairs = []
    cats = list(cat_samples.keys())
    for i in range(len(cats)):
        for j in range(i+1, len(cats)):
            pairs.append((cats[i], cats[j]))

    for cat_A, cat_B in pairs:
        pair_dir = swap_dir / f"{cat_A}_x_{cat_B}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        d_A = decomposed[cat_A]
        d_B = decomposed[cat_B]

        # Match temporal lengths
        T = min(d_A['z_flat'].shape[1], d_B['z_flat'].shape[1])
        z_A_full = d_A['z_full'][:, :T]
        z_B_full = d_B['z_full'][:, :T]
        z_A_sms = d_A['z_sms'][:, :T]
        z_B_sms = d_B['z_sms'][:, :T]

        # Save ground truth (raw DCAE encoding) and reconstruction
        z_A_gt = d_A['z_flat'][:, :T]
        z_B_gt = d_B['z_flat'][:, :T]
        audio_A_gt = decode_z_flat(dcae, z_A_gt, device)
        audio_B_gt = decode_z_flat(dcae, z_B_gt, device)
        save_wav(audio_A_gt, pair_dir / f"00_{cat_A}_ground_truth.wav")
        save_wav(audio_B_gt, pair_dir / f"01_{cat_B}_ground_truth.wav")

        audio_A_recon = decode_z_flat(dcae, z_A_full, device)
        audio_B_recon = decode_z_flat(dcae, z_B_full, device)
        save_wav(audio_A_recon, pair_dir / f"00_{cat_A}_recon.wav")
        save_wav(audio_B_recon, pair_dir / f"01_{cat_B}_recon.wav")

        cos_AB = TF.cosine_similarity(z_A_full, z_B_full, dim=-1).mean().item()
        print(f"\n  {cat_A} ({cat_samples[cat_A]['name']}) x "
              f"{cat_B} ({cat_samples[cat_B]['name']})")
        print(f"    cos_sim(A, B) = {cos_AB:.4f}")

        # For each top operation: swap B's contribution into A
        for rank, k in enumerate(top_ops):
            contrib_A_k = d_A['op_contribs'][k][:, :T]
            contrib_B_k = d_B['op_contribs'][k][:, :T]

            # A with B's op k: remove A's op k, add B's op k
            z_hybrid = z_A_full - contrib_A_k + contrib_B_k

            cos_hybrid_A = TF.cosine_similarity(z_hybrid, z_A_full, dim=-1).mean().item()
            cos_hybrid_B = TF.cosine_similarity(z_hybrid, z_B_full, dim=-1).mean().item()

            audio_hybrid = decode_z_flat(dcae, z_hybrid, device)
            save_wav(audio_hybrid, pair_dir / f"02_{cat_A}_with_{cat_B}_op{rank}_id{k}.wav")

            print(f"    Swap op {k} (rank {rank}): "
                  f"cos(hybrid,A)={cos_hybrid_A:.4f}  cos(hybrid,B)={cos_hybrid_B:.4f}")

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# TEST 2: Parameter Sweep
# ============================================================

def test_param_sweep(codec, tree, dcae, res_mean, res_std, samples_by_cat, device):
    """
    Scale one operation's contribution by 0x, 0.5x, 1x, 1.5x, 2x.
    If meaningful, should smoothly change one perceptual quality.
    If not, sound degrades immediately.
    """
    print("\n" + "=" * 60)
    print("TEST 2: PARAMETER SWEEP")
    print("  Scale each op's contribution: 0x, 0.5x, 1x, 1.5x, 2x")
    print("=" * 60)

    sweep_dir = OUTPUT_DIR / "02_param_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    op_ranking = get_op_ranking(tree)
    top_ops = op_ranking[:4]
    scales = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Pick one sample from each category
    for cat, slist in samples_by_cat.items():
        if not slist:
            continue
        sample = slist[0]
        d = decompose_sample(codec, tree, res_mean, res_std, sample['z_real'], device)

        cat_dir = sweep_dir / f"{cat}_{sample['name'][:20]}"
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Save ground truth (raw DCAE) and reconstruction baseline
        audio_gt = decode_z_flat(dcae, d['z_flat'], device)
        save_wav(audio_gt, cat_dir / "ground_truth.wav")

        audio_full = decode_z_flat(dcae, d['z_full'], device)
        save_wav(audio_full, cat_dir / "baseline_recon.wav")

        print(f"\n  {cat}: {sample['name']}")

        for rank, k in enumerate(top_ops):
            contrib_k = d['op_contribs'][k]

            print(f"    Op {k} (rank {rank}):", end="")

            for scale in scales:
                # Replace op k at original scale with new scale
                z_swept = d['z_full'] - contrib_k + scale * contrib_k

                cos_vs_full = TF.cosine_similarity(z_swept, d['z_full'], dim=-1).mean().item()
                cos_vs_gt = TF.cosine_similarity(z_swept, d['z_flat'], dim=-1).mean().item()

                audio_swept = decode_z_flat(dcae, z_swept, device)
                fname = f"op{rank}_id{k}_scale{scale:.1f}.wav"
                save_wav(audio_swept, cat_dir / fname)

                print(f"  {scale:.1f}x→cos={cos_vs_full:.3f}", end="")

            print()

        gc.collect()
        torch.cuda.empty_cache()


# ============================================================
# TEST 3: Cross-Instrument Consistency
# ============================================================

def test_consistency(codec, tree, res_mean, res_std, samples_by_cat, device):
    """
    Do operations activate differently across instrument categories?
    If consistent, same ops should fire for same instrument types.
    """
    print("\n" + "=" * 60)
    print("TEST 3: CROSS-INSTRUMENT CONSISTENCY")
    print("  Do ops activate differently for different instruments?")
    print("=" * 60)

    # Also load some extra samples for statistical power
    extra_samples = load_extra_samples_by_category(samples_by_cat, max_per_cat=8)

    op_ranking = get_op_ranking(tree)
    n_ops = tree.n_ops

    # Collect per-category activation profiles
    cat_profiles = defaultdict(list)  # cat → list of [n_ops] mean activations

    for cat, slist in extra_samples.items():
        for sample in slist:
            d = decompose_sample(codec, tree, res_mean, res_std, sample['z_real'], device)
            # Mean activation per op across time
            mean_alpha = d['activations'].squeeze(0).mean(dim=0).cpu().numpy()  # [n_ops]
            cat_profiles[cat].append(mean_alpha)

    print(f"\n  Samples per category: {', '.join(f'{k}={len(v)}' for k,v in cat_profiles.items())}")

    # Print per-category mean activation profile
    print(f"\n  Mean activation profile by instrument (ops ranked by importance):")
    print(f"  {'Category':<12}", end="")
    for rank, k in enumerate(op_ranking[:6]):
        print(f"  Op{k}(r{rank})", end="")
    print()

    cat_means = {}
    for cat, profiles in sorted(cat_profiles.items()):
        arr = np.array(profiles)
        mean = arr.mean(axis=0)
        cat_means[cat] = mean
        print(f"  {cat:<12}", end="")
        for rank, k in enumerate(op_ranking[:6]):
            print(f"  {mean[k]:7.2f}", end="")
        print()

    # Compute inter-category distance vs intra-category variance
    print(f"\n  Activation variance analysis:")
    if len(cat_means) >= 2:
        cats_arr = list(cat_means.keys())
        inter_dists = []
        for i in range(len(cats_arr)):
            for j in range(i+1, len(cats_arr)):
                dist = np.linalg.norm(cat_means[cats_arr[i]] - cat_means[cats_arr[j]])
                inter_dists.append(dist)
                print(f"    {cats_arr[i]} vs {cats_arr[j]}: L2 distance = {dist:.4f}")

        # Intra-category variance
        intra_vars = []
        for cat, profiles in cat_profiles.items():
            if len(profiles) >= 2:
                arr = np.array(profiles)
                var = np.mean(np.var(arr, axis=0))
                intra_vars.append(var)
                print(f"    {cat} intra-variance: {var:.4f}")

        mean_inter = np.mean(inter_dists) if inter_dists else 0
        mean_intra = np.mean(intra_vars) if intra_vars else 0
        ratio = mean_inter / (np.sqrt(mean_intra) + 1e-8)

        print(f"\n    Mean inter-category distance: {mean_inter:.4f}")
        print(f"    Mean intra-category std:      {np.sqrt(mean_intra):.4f}")
        print(f"    Ratio (higher = more discriminative): {ratio:.4f}")

        if ratio > 2.0:
            print(f"    >> Operations are DISCRIMINATIVE across instruments")
        elif ratio > 1.0:
            print(f"    >> Operations show MODERATE instrument sensitivity")
        else:
            print(f"    >> Operations are NOT discriminative — activations are similar")

    # Per-operation: which op varies MOST across categories?
    print(f"\n  Per-operation cross-category variance:")
    if len(cat_means) >= 2:
        means_matrix = np.array(list(cat_means.values()))  # [n_cats, n_ops]
        op_variance = np.var(means_matrix, axis=0)  # [n_ops]
        for rank, k in enumerate(op_ranking[:6]):
            print(f"    Op {k} (rank {rank}): cross-cat variance = {op_variance[k]:.4f}")


# ============================================================
# TEST 4: Audio Feature Correlation
# ============================================================

def test_feature_correlation(codec, tree, dcae, res_mean, res_std, samples_by_cat, device):
    """
    Correlate operation activations with spectral audio features.
    If Op 3 correlates with spectral centroid → it captures "brightness".
    """
    print("\n" + "=" * 60)
    print("TEST 4: AUDIO FEATURE CORRELATION")
    print("  Do op activations correlate with spectral features?")
    print("=" * 60)

    extra_samples = load_extra_samples_by_category(samples_by_cat, max_per_cat=8)
    op_ranking = get_op_ranking(tree)

    features_list = []
    activations_list = []
    op_norms_list = []
    categories = []

    all_samples = []
    for cat, slist in extra_samples.items():
        for s in slist:
            s['category'] = cat
            all_samples.append(s)

    print(f"  Processing {len(all_samples)} samples...")

    for si, sample in enumerate(all_samples):
        d = decompose_sample(codec, tree, res_mean, res_std, sample['z_real'], device)

        # Decode ground truth to audio for feature extraction
        audio = decode_z_flat(dcae, d['z_flat'], device)
        feats = compute_spectral_features(audio)

        # Mean activation per op
        mean_alpha = d['activations'].squeeze(0).mean(dim=0).cpu().numpy()

        # Mean contribution norm per op
        op_norms = np.array([
            d['op_contribs'][k].squeeze(0).norm(dim=-1).mean().item()
            for k in range(tree.n_ops)
        ])

        features_list.append(feats)
        activations_list.append(mean_alpha)
        op_norms_list.append(op_norms)
        categories.append(sample['category'])

        if (si + 1) % 5 == 0:
            print(f"    Processed {si+1}/{len(all_samples)}")

        gc.collect()
        torch.cuda.empty_cache()

    if len(features_list) < 5:
        print("  Too few samples for meaningful correlation, skipping")
        return

    # Build feature matrix
    feat_names = ['centroid', 'bandwidth', 'rms', 'zcr', 'rolloff']
    feat_matrix = np.array([[f[k] for k in feat_names] for f in features_list])
    alpha_matrix = np.array(activations_list)
    norm_matrix = np.array(op_norms_list)

    # Compute correlations: each op activation vs each audio feature
    print(f"\n  Correlation: op activation × audio feature")
    print(f"  {'Op':<12}", end="")
    for fn in feat_names:
        print(f"  {fn:>10}", end="")
    print()

    for rank, k in enumerate(op_ranking[:6]):
        alpha_k = alpha_matrix[:, k]
        print(f"  Op{k} (r{rank})", end="")
        for fi, fn in enumerate(feat_names):
            feat_vals = feat_matrix[:, fi]
            if np.std(alpha_k) < 1e-8 or np.std(feat_vals) < 1e-8:
                corr = 0.0
            else:
                corr = np.corrcoef(alpha_k, feat_vals)[0, 1]
            marker = " *" if abs(corr) > 0.5 else ""
            print(f"  {corr:+10.3f}{marker}", end="")
        print()

    print(f"\n  Correlation: op contribution norm × audio feature")
    print(f"  {'Op':<12}", end="")
    for fn in feat_names:
        print(f"  {fn:>10}", end="")
    print()

    for rank, k in enumerate(op_ranking[:6]):
        norm_k = norm_matrix[:, k]
        print(f"  Op{k} (r{rank})", end="")
        for fi, fn in enumerate(feat_names):
            feat_vals = feat_matrix[:, fi]
            if np.std(norm_k) < 1e-8 or np.std(feat_vals) < 1e-8:
                corr = 0.0
            else:
                corr = np.corrcoef(norm_k, feat_vals)[0, 1]
            marker = " *" if abs(corr) > 0.5 else ""
            print(f"  {corr:+10.3f}{marker}", end="")
        print()

    # Per-category feature means
    print(f"\n  Audio feature means by instrument:")
    print(f"  {'Category':<12}", end="")
    for fn in feat_names:
        print(f"  {fn:>10}", end="")
    print()

    for cat in sorted(set(categories)):
        idx = [i for i, c in enumerate(categories) if c == cat]
        cat_feats = feat_matrix[idx].mean(axis=0)
        print(f"  {cat:<12}", end="")
        for v in cat_feats:
            print(f"  {v:10.1f}", end="")
        print()


# ============================================================
# Helpers
# ============================================================

def get_op_ranking(tree):
    """Get operation ranking from checkpoint or just return range."""
    ckpt = torch.load(TREE_PATH, weights_only=False, map_location='cpu')
    # Reconstruct: load model and estimate importance from param norms
    # Simple heuristic: operations with larger output weight norms are more important
    state = ckpt['model']
    importance = []
    for k in range(ckpt['n_ops']):
        # Output layer weight of each operation MLP
        key = f'operations.{k}.net.4.weight'
        if key in state:
            importance.append(state[key].norm().item())
        else:
            importance.append(0.0)
    ranking = sorted(range(len(importance)), key=lambda i: -importance[i])
    return ranking


def load_extra_samples_by_category(samples_by_cat, max_per_cat=8):
    """Load additional samples from GCS for each category."""
    extra = {}
    for cat, existing in samples_by_cat.items():
        extra[cat] = list(existing)  # start with what we have

        if len(extra[cat]) >= max_per_cat:
            extra[cat] = extra[cat][:max_per_cat]
            continue

        # Find more from GCS
        keywords = {
            'piano': ['piano'],
            'strings': ['violin', 'cello', 'string', 'viola'],
            'brass': ['trumpet', 'horn', 'trombone'],
            'winds': ['sax', 'flute', 'clarinet'],
        }
        kws = keywords.get(cat, [cat])
        seen_paths = {s['path'] for s in extra[cat]}

        for pt in LATENT_BASE.rglob("*.pt"):
            if len(extra[cat]) >= max_per_cat:
                break
            name_lower = pt.name.lower()
            if not any(kw in name_lower for kw in kws):
                continue
            p = str(pt)
            if p in seen_paths:
                continue
            try:
                z = load_latent(p)
                rms = z.float().pow(2).mean().sqrt().item()
                if rms > 0.3:
                    extra[cat].append({'z_real': z, 'path': p, 'name': pt.stem, 'category': cat})
                    seen_paths.add(p)
            except Exception:
                continue

        print(f"  {cat}: {len(extra[cat])} samples loaded")

    return extra


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("VERIFY OPERATION MEANING")
    print("Are operations meaningful or just arbitrary splits?")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    codec, tree, dcae, res_mean, res_std = load_models(device)

    print("\nLoading instrument samples...")
    samples_by_cat = load_instrument_samples()

    total = sum(len(v) for v in samples_by_cat.values())
    print(f"  Total: {total} samples across {len(samples_by_cat)} categories")

    # Run all tests
    test_operation_swap(codec, tree, dcae, res_mean, res_std, samples_by_cat, device)
    test_param_sweep(codec, tree, dcae, res_mean, res_std, samples_by_cat, device)
    test_consistency(codec, tree, res_mean, res_std, samples_by_cat, device)
    test_feature_correlation(codec, tree, dcae, res_mean, res_std, samples_by_cat, device)

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print(f"\nOutputs: {OUTPUT_DIR}")
    print(f"""
How to interpret results:

  01_operation_swap/
    Listen: Does swapping one op from brass→piano change one quality?
    Or does the whole sound break? (meaningful vs entangled)

  02_param_sweep/
    Listen: Does scaling an op 0x→2x smoothly change one quality?
    Or does the sound degrade non-monotonically? (continuous vs binary)

  Test 3 (console): Inter vs intra category distance
    High ratio → ops are discriminative (meaningful)
    Low ratio  → ops fire uniformly (arbitrary)

  Test 4 (console): Correlation with spectral features
    |corr| > 0.5 → op captures that feature (e.g., brightness, energy)
    All ~0 → ops don't correspond to known audio qualities
""")


if __name__ == "__main__":
    main()
