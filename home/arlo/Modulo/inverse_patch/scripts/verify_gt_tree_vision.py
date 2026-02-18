#!/usr/bin/env python3
"""
Verify the Full Hierarchical Sparse Codec Vision using the GT SMS Pipeline.

Three main analyses:

  Part 1: Cross-Instrument Op Activation Analysis
    For each sample in the data cache (~2000), compute GT tree decomposition,
    record which ops are most active (by alpha * contrib norm), group by
    instrument category, and compute specialization scores (F-statistic analog).

  Part 2: Generate Isolated Op Audio Files
    For a diverse set of samples (~3 per instrument category), render audio:
      - GT original, GT SMS only, GT SMS + full tree
      - Each op solo, each top-4 op removed

  Part 3: Op-to-SMS Bridge Analysis
    For the top 4 ops, across 30+ samples:
      - Run codec.forward_F() on z with/without each op
      - Compute delta_sms: what each op changes in SMS parameter space
      - Analyze which SMS components (f0, harmonics, noise) each op modifies
      - Check cross-sample consistency
"""

import sys
import os
import gc
import torch
import torch.nn.functional as TF
import numpy as np
import torchaudio
from pathlib import Path
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM
from phase2_operation_tree import OperationTreeCodec
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
import orjson

# ============================================================
# Paths and Constants
# ============================================================
SCRIPT_DIR = Path('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
GT_TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_gt_tree" / "operation_tree_gt.pt"
DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_DATA_DIR = SCRIPT_DIR.parent / "data" / "sms_v4"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "gt_tree_vision"

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

Z_DIM = 128
SAMPLE_RATE = 44100

# Instrument categorization (match from latent path, case-insensitive)
CATEGORY_KEYWORDS = {
    'Piano': ['piano'],
    'Guitar': ['guitar', 'gtr'],
    'Bass': ['bass'],
    'Strings': ['string', 'violin', 'cello', 'viola'],
    'Brass': ['trumpet', 'horn', 'trombone'],
    'Winds': ['sax', 'flute', 'clarinet', 'oboe'],
    'Vocals': ['vocal', 'vox'],
    'Drums': ['drum', 'kick', 'snare', 'hat', 'hh', 'tom', 'perc'],
    'Synth': ['synth', 'pad', 'keys'],
}

# SMS 102-dim layout for Part 3 analysis
SMS_GROUPS = {
    'group_f0s': (0, 6),         # dims 0-5: group f0s (log-normalized)
    'group_amps': (6, 54),       # dims 6-53: group amps (6 groups x 8 partials)
    'indep_freqs': (54, 74),     # dims 54-73: independent freqs (log-normalized)
    'indep_amps': (74, 94),      # dims 74-93: independent amps (raw)
    'noise_bands': (94, 102),    # dims 94-101: noise bands (raw)
}


# ============================================================
# Utility Functions
# ============================================================

def classify_instrument(path_str):
    """Classify a file path into an instrument category."""
    path_lower = path_str.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in path_lower:
                return category
    return 'Other'


def decode_z_to_audio(dcae, z_flat_tensor, device):
    """
    Decode z_flat [1, T, 128] tensor (on any device) to numpy audio.
    """
    z_flat = z_flat_tensor.to(device)
    B, T, D = z_flat.shape
    z_4d = z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)  # [1, 8, 16, T]
    audio_len = int(T * SAMPLE_RATE / 10.8)
    audio_lengths = torch.tensor([audio_len], device=device)

    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)

    audio = wavs[0].mean(dim=0).cpu().numpy()
    return audio


def save_wav(audio, path):
    """Save numpy audio to wav file with peak normalization."""
    t = torch.from_numpy(audio).float().unsqueeze(0)
    peak = t.abs().max()
    if peak > 0.95:
        t = t * (0.95 / peak)
    torchaudio.save(str(path), t, SAMPLE_RATE)


def load_models(device):
    """Load Phase 1 codec, GT operation tree, and DCAE."""
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

    print("Loading DCAE...")
    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_CKPT,
        vocoder_checkpoint_path=VOCODER_CKPT,
    ).to(device).eval()

    return {
        'codec': codec,
        'gt_tree': gt_tree,
        'dcae': dcae,
        'gt_res_mean': gt_res_mean,
        'gt_res_std': gt_res_std,
        'n_ops': gt_ckpt['n_ops'],
        'top_k': gt_ckpt['top_k'],
        'device': device,
    }


def load_paired_data(max_samples=None):
    """
    Load Phase 1 data cache + manifest, pair z_real with z_gt_sms.
    Returns list of dicts: z_gt_sms_flat [1,T,128], z_real_flat [1,T,128],
    latent_path, sms_path, T, category.
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
        if max_samples is not None and len(pairs) >= max_samples:
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

        category = classify_instrument(latent_path)

        pairs.append({
            'z_gt_sms_flat': z_gt_sms_flat,
            'z_real_flat': z_real_flat,
            'latent_path': latent_path,
            'sms_path': sms_path,
            'T': T_aligned,
            'category': category,
        })

        if len(pairs) % 200 == 0:
            print(f"    Loaded {len(pairs)} pairs...")

    print(f"  Total: {len(pairs)} pairs ({skipped} skipped)")

    # Report category distribution
    cat_counts = defaultdict(int)
    for p in pairs:
        cat_counts[p['category']] += 1
    print(f"  Category distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f"    {cat}: {count}")

    return pairs


def decompose_gt_tree(models, z_gt_sms_flat, z_real_flat):
    """
    Decompose using GT tree, returning per-op contributions and sparse activations.

    Returns:
        gt_res_mean_val: [1, 1, 128] on CPU
        op_contribs: list of [1, T, 128] CPU tensors (denormalized per-op contribution)
        alpha_sparse: [1, T, n_ops] CPU tensor
        contrib_norms: [1, T, n_ops] CPU tensor (norm of each op's contribution)
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

        hidden = gt_tree.encode(gt_residual_norm)

        # Sparse alpha
        raw_alpha = gt_tree.activation_head(hidden)
        alpha = TF.softplus(raw_alpha)
        topk_vals, topk_idx = torch.topk(alpha, gt_tree.top_k, dim=-1)
        mask = torch.zeros_like(alpha)
        mask.scatter_(-1, topk_idx, 1.0)
        alpha_sparse = alpha * mask  # [1, T, n_ops]

        # Per-op contributions
        op_contribs = []
        contrib_norms = []
        for k in range(gt_tree.n_ops):
            params_k = gt_tree.param_heads[k](hidden)
            contrib_k = gt_tree.operations[k](params_k)
            alpha_k = alpha_sparse[:, :, k:k + 1]
            gt_scaled = (alpha_k * contrib_k) * gt_res_std
            op_contribs.append(gt_scaled.cpu())
            contrib_norms.append(contrib_k.norm(dim=-1).cpu())

    contrib_norms_stacked = torch.stack(contrib_norms, dim=-1)  # [1, T, n_ops]

    return gt_res_mean.cpu(), op_contribs, alpha_sparse.cpu(), contrib_norms_stacked


# ============================================================
# Part 1: Cross-Instrument Op Activation Analysis
# ============================================================

def part1_cross_instrument(models, pairs):
    """
    For each sample, compute GT tree decomposition and record which ops are
    most active (by alpha * contrib_norm). Group by instrument category.
    Compute specialization score (between/within variance ratio, F-statistic).
    """
    print("\n" + "=" * 60)
    print("PART 1: CROSS-INSTRUMENT OP ACTIVATION ANALYSIS")
    print("=" * 60)

    device = models['device']
    n_ops = models['n_ops']

    # Collect per-sample, per-op activation strength = alpha * contrib_norm
    # Grouped by category
    cat_op_activations = defaultdict(list)  # cat -> list of [n_ops] arrays

    n_total = len(pairs)
    for si, pair in enumerate(pairs):
        z_gt_sms_flat = pair['z_gt_sms_flat']
        z_real_flat = pair['z_real_flat']
        category = pair['category']

        gt_res_mean_val, op_contribs, alpha_sparse, contrib_norms = decompose_gt_tree(
            models, z_gt_sms_flat, z_real_flat
        )

        # Compute activation strength: alpha * contrib_norm, averaged over time
        # alpha_sparse: [1, T, n_ops], contrib_norms: [1, T, n_ops]
        activation_strength = (alpha_sparse * contrib_norms).squeeze(0).mean(dim=0).numpy()  # [n_ops]
        cat_op_activations[category].append(activation_strength)

        if (si + 1) % 100 == 0:
            print(f"    Processed {si + 1}/{n_total}...")

    # Build table: for each op, mean activation per instrument category
    all_categories = sorted(cat_op_activations.keys())

    print(f"\n  Mean activation (alpha * ||contrib||) per op per instrument:")
    print(f"\n  {'Category':<12}", end="")
    for k in range(n_ops):
        print(f"   Op{k:d}", end="")
    print(f"  {'N':>5}")

    cat_means = {}
    for cat in all_categories:
        arr = np.array(cat_op_activations[cat])  # [n_samples, n_ops]
        mean = arr.mean(axis=0)
        cat_means[cat] = mean
        print(f"  {cat:<12}", end="")
        for k in range(n_ops):
            print(f"  {mean[k]:5.3f}", end="")
        print(f"  {len(arr):5d}")

    # Overall mean
    all_arr = np.concatenate([np.array(v) for v in cat_op_activations.values()], axis=0)
    overall_mean = all_arr.mean(axis=0)
    print(f"  {'OVERALL':<12}", end="")
    for k in range(n_ops):
        print(f"  {overall_mean[k]:5.3f}", end="")
    print(f"  {len(all_arr):5d}")

    # Check if ops specialize by instrument or are universal
    print(f"\n  Op specialization analysis:")
    print(f"  {'Op':>4}  {'Between Var':>12}  {'Within Var':>12}  {'F-stat':>8}  {'Verdict'}")

    specialization_scores = {}
    for k in range(n_ops):
        # Between-group variance: variance of group means
        group_means = np.array([cat_means[cat][k] for cat in all_categories])
        group_sizes = np.array([len(cat_op_activations[cat]) for cat in all_categories])
        grand_mean = overall_mean[k]

        # Sum of squares between groups
        ss_between = np.sum(group_sizes * (group_means - grand_mean) ** 2)
        df_between = len(all_categories) - 1

        # Sum of squares within groups
        ss_within = 0.0
        df_within = 0
        for cat in all_categories:
            arr = np.array(cat_op_activations[cat])[:, k]
            ss_within += np.sum((arr - cat_means[cat][k]) ** 2)
            df_within += len(arr) - 1

        # F-statistic
        ms_between = ss_between / max(df_between, 1)
        ms_within = ss_within / max(df_within, 1)
        f_stat = ms_between / max(ms_within, 1e-10)

        if f_stat > 5.0:
            verdict = "SPECIALIZED"
        elif f_stat > 2.0:
            verdict = "moderate"
        else:
            verdict = "universal"

        specialization_scores[k] = {
            'f_stat': float(f_stat),
            'between_var': float(ms_between),
            'within_var': float(ms_within),
            'verdict': verdict,
        }

        print(f"  {k:4d}  {ms_between:12.6f}  {ms_within:12.6f}  {f_stat:8.2f}  {verdict}")

    # Find which category each op is most active for
    print(f"\n  Peak category per op:")
    for k in range(n_ops):
        peak_cat = max(all_categories, key=lambda c: cat_means[c][k])
        peak_val = cat_means[peak_cat][k]
        second_val = sorted([cat_means[c][k] for c in all_categories])[-2] if len(all_categories) > 1 else 0
        ratio = peak_val / max(second_val, 1e-10)
        print(f"    Op {k}: peak={peak_cat} ({peak_val:.4f}), ratio to 2nd={ratio:.2f}")

    results = {
        'cat_means': {cat: cat_means[cat].tolist() for cat in all_categories},
        'overall_mean': overall_mean.tolist(),
        'specialization_scores': specialization_scores,
        'n_samples_per_cat': {cat: len(v) for cat, v in cat_op_activations.items()},
    }

    print(f"\n  Part 1 complete: analyzed {len(all_arr)} samples across {len(all_categories)} categories.")
    return results


# ============================================================
# Part 2: Generate Isolated Op Audio Files
# ============================================================

def part2_audio_renders(models, pairs):
    """
    For a diverse set of samples (~3 per instrument category), render audio files:
      - {name}_gt_original.wav
      - {name}_gt_sms.wav
      - {name}_gt_sms_plus_tree.wav
      - {name}_op{k}_solo.wav (for each of 8 ops)
      - {name}_op{k}_removed.wav (for top 4 ops)
    """
    print("\n" + "=" * 60)
    print("PART 2: GENERATE ISOLATED OP AUDIO FILES")
    print("=" * 60)

    device = models['device']
    dcae = models['dcae']
    gt_tree = models['gt_tree']
    n_ops = models['n_ops']
    top_k = models['top_k']

    audio_dir = OUTPUT_DIR / "audio_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # Group pairs by category and pick up to 3 per category
    cat_pairs = defaultdict(list)
    for pair in pairs:
        cat_pairs[pair['category']].append(pair)

    selected = []
    for cat in sorted(cat_pairs.keys()):
        available = cat_pairs[cat]
        chosen = available[:3]
        selected.extend(chosen)
        print(f"  {cat}: selected {len(chosen)}/{len(available)} samples")

    print(f"  Total selected: {len(selected)} samples")

    rendered_info = []

    for si, pair in enumerate(selected):
        category = pair['category']
        latent_path = pair['latent_path']
        T = pair['T']
        sample_name = Path(latent_path).stem[:40]  # truncate long names

        # Create output directory
        sample_dir = audio_dir / category / sample_name
        sample_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  [{si + 1}/{len(selected)}] {category}/{sample_name} (T={T})")

        z_gt_sms_flat = pair['z_gt_sms_flat']
        z_real_flat = pair['z_real_flat']

        # GT tree decomposition
        gt_res_mean_val, op_contribs, alpha_sparse, contrib_norms = decompose_gt_tree(
            models, z_gt_sms_flat, z_real_flat
        )

        # 1. GT original: decode z_real_flat
        print(f"    Rendering gt_original...", end="", flush=True)
        audio_original = decode_z_to_audio(dcae, z_real_flat, device)
        save_wav(audio_original, sample_dir / f"{sample_name}_gt_original.wav")
        print(" done")

        # 2. GT SMS only: decode z_gt_sms_flat
        print(f"    Rendering gt_sms...", end="", flush=True)
        audio_sms = decode_z_to_audio(dcae, z_gt_sms_flat, device)
        save_wav(audio_sms, sample_dir / f"{sample_name}_gt_sms.wav")
        print(" done")

        # 3. GT SMS + full tree: z_gt_sms + res_mean + sum of all op_contribs
        print(f"    Rendering gt_sms_plus_tree...", end="", flush=True)
        z_full = z_gt_sms_flat + gt_res_mean_val
        for k in range(n_ops):
            z_full = z_full + op_contribs[k]
        audio_full = decode_z_to_audio(dcae, z_full, device)
        save_wav(audio_full, sample_dir / f"{sample_name}_gt_sms_plus_tree.wav")
        print(" done")

        # 4. Op k solo: z_gt_sms + res_mean + only op k
        print(f"    Rendering op solos...", end="", flush=True)
        for k in range(n_ops):
            z_op_solo = z_gt_sms_flat + gt_res_mean_val + op_contribs[k]
            audio_solo = decode_z_to_audio(dcae, z_op_solo, device)
            save_wav(audio_solo, sample_dir / f"{sample_name}_op{k}_solo.wav")
        print(" done")

        # 5. Op k removed: z_gt_sms + res_mean + all ops EXCEPT k (top 4 only)
        # Determine which ops are the top 4 by mean activation * contrib norm
        mean_strengths = (alpha_sparse * contrib_norms).squeeze(0).mean(dim=0).numpy()
        top_4_ops = sorted(range(n_ops), key=lambda i: -mean_strengths[i])[:4]

        print(f"    Rendering op removals (top 4: {top_4_ops})...", end="", flush=True)
        for k in top_4_ops:
            z_removed = z_gt_sms_flat + gt_res_mean_val
            for j in range(n_ops):
                if j != k:
                    z_removed = z_removed + op_contribs[j]
            audio_removed = decode_z_to_audio(dcae, z_removed, device)
            save_wav(audio_removed, sample_dir / f"{sample_name}_op{k}_removed.wav")
        print(" done")

        rendered_info.append({
            'category': category,
            'sample_name': sample_name,
            'latent_path': latent_path,
            'T': T,
            'top_4_ops': top_4_ops,
            'mean_strengths': mean_strengths.tolist(),
        })

        # Free GPU periodically
        if (si + 1) % 5 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"\n  Part 2 complete: rendered {len(rendered_info)} samples to {audio_dir}")
    return rendered_info


# ============================================================
# Part 3: Op-to-SMS Bridge Analysis
# ============================================================

def part3_sms_bridge(models, pairs, n_samples=40):
    """
    For each top 4 op, across 30+ samples:
      - Compute z_with_op = z_gt_sms + res_mean + op_k_contrib
      - Compute z_without_op = z_gt_sms + res_mean
      - Run both through codec.forward_F() -> SMS params
      - Compute delta_sms = F(z_with_op) - F(z_without_op)
      - Analyze which SMS components each op modifies
      - Check consistency across samples
    """
    print("\n" + "=" * 60)
    print("PART 3: OP-TO-SMS BRIDGE ANALYSIS")
    print("=" * 60)

    device = models['device']
    codec = models['codec']
    n_ops = models['n_ops']

    # Use up to n_samples pairs
    analysis_pairs = pairs[:n_samples]
    n_actual = len(analysis_pairs)
    print(f"  Analyzing {n_actual} samples")

    # First pass: determine top 4 ops across all samples
    print(f"\n  Determining top ops across dataset...")
    all_strengths = []
    for pair in analysis_pairs:
        _, op_contribs, alpha_sparse, contrib_norms = decompose_gt_tree(
            models, pair['z_gt_sms_flat'], pair['z_real_flat']
        )
        mean_strengths = (alpha_sparse * contrib_norms).squeeze(0).mean(dim=0).numpy()
        all_strengths.append(mean_strengths)

    global_mean_strengths = np.mean(all_strengths, axis=0)
    top_4_ops = sorted(range(n_ops), key=lambda i: -global_mean_strengths[i])[:4]
    print(f"  Top 4 ops (by global mean activation * norm): {top_4_ops}")
    for k in top_4_ops:
        print(f"    Op {k}: mean strength = {global_mean_strengths[k]:.4f}")

    # Second pass: compute delta_sms for each top op across samples
    # delta_sms_per_op[k] = list of [T, SMS_DIM] arrays
    delta_sms_per_op = {k: [] for k in top_4_ops}
    # Also store absolute delta per SMS group
    delta_per_group = {k: {g: [] for g in SMS_GROUPS} for k in top_4_ops}

    print(f"\n  Computing SMS deltas...")
    codec.eval()

    for si, pair in enumerate(analysis_pairs):
        z_gt_sms_flat = pair['z_gt_sms_flat']
        z_real_flat = pair['z_real_flat']

        gt_res_mean_val, op_contribs, alpha_sparse, contrib_norms = decompose_gt_tree(
            models, z_gt_sms_flat, z_real_flat
        )

        # Base z: z_gt_sms + res_mean (no ops)
        z_base = (z_gt_sms_flat + gt_res_mean_val).to(device)

        with torch.no_grad():
            sms_base = codec.forward_F(z_base)  # [1, T, 102]

        for k in top_4_ops:
            z_with_op = z_base + op_contribs[k].to(device)

            with torch.no_grad():
                sms_with_op = codec.forward_F(z_with_op)  # [1, T, 102]

            delta_sms = (sms_with_op - sms_base).squeeze(0).cpu().numpy()  # [T, 102]
            delta_sms_per_op[k].append(delta_sms)

            # Compute per-group absolute delta
            for group_name, (start, end) in SMS_GROUPS.items():
                group_delta = np.abs(delta_sms[:, start:end]).mean()
                delta_per_group[k][group_name].append(float(group_delta))

        if (si + 1) % 10 == 0:
            print(f"    {si + 1}/{n_actual}...")
            gc.collect()
            torch.cuda.empty_cache()

    # Analyze results
    print(f"\n  SMS delta analysis per op:")
    print(f"\n  {'Op':>4}  {'group_f0s':>10}  {'group_amps':>11}  {'indep_freqs':>12}  "
          f"{'indep_amps':>11}  {'noise_bands':>12}")

    op_sms_profiles = {}

    for k in top_4_ops:
        profile = {}
        print(f"  {k:4d}", end="")
        for group_name in SMS_GROUPS:
            vals = delta_per_group[k][group_name]
            mean_delta = float(np.mean(vals))
            std_delta = float(np.std(vals))
            profile[group_name] = {'mean': mean_delta, 'std': std_delta}
            print(f"  {mean_delta:10.5f}", end="")
        print()
        op_sms_profiles[k] = profile

    # Detailed per-dimension analysis for each top op
    print(f"\n  Detailed per-dimension analysis:")
    dim_names = []
    for i in range(6):
        dim_names.append(f"f0_{i}")
    for g in range(6):
        for p in range(8):
            dim_names.append(f"g{g}_amp{p}")
    for i in range(20):
        dim_names.append(f"ifreq_{i}")
    for i in range(20):
        dim_names.append(f"iamp_{i}")
    for i in range(8):
        dim_names.append(f"noise_{i}")

    for k in top_4_ops:
        # Stack all deltas: [n_samples, T, 102] -> average over T -> [n_samples, 102]
        all_deltas = np.array([d.mean(axis=0) for d in delta_sms_per_op[k]])  # [n_samples, 102]
        mean_abs_delta = np.abs(all_deltas).mean(axis=0)  # [102]

        # Top 10 most affected dimensions
        top_dims = np.argsort(mean_abs_delta)[::-1][:10]
        print(f"\n    Op {k} top 10 most affected SMS dims:")
        for rank, d in enumerate(top_dims):
            name = dim_names[d] if d < len(dim_names) else f"dim_{d}"
            print(f"      {rank + 1:2d}. {name:>12}: mean|delta|={mean_abs_delta[d]:.5f}")

    # Consistency analysis: does each op always modify the same SMS components?
    print(f"\n  Cross-sample consistency per op:")
    consistency_results = {}

    for k in top_4_ops:
        # [n_samples, 102] mean-over-time abs deltas
        all_deltas = np.array([np.abs(d).mean(axis=0) for d in delta_sms_per_op[k]])
        n_samp = all_deltas.shape[0]

        if n_samp < 2:
            print(f"    Op {k}: too few samples for consistency check")
            continue

        # Pairwise correlation of delta profiles
        correlations = []
        for i in range(n_samp):
            for j in range(i + 1, n_samp):
                r_i = all_deltas[i]
                r_j = all_deltas[j]
                if np.std(r_i) < 1e-8 or np.std(r_j) < 1e-8:
                    continue
                corr = np.corrcoef(r_i, r_j)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        if correlations:
            mean_corr = float(np.mean(correlations))
            median_corr = float(np.median(correlations))
            std_corr = float(np.std(correlations))
        else:
            mean_corr = median_corr = std_corr = 0.0

        if mean_corr > 0.7:
            verdict = "HIGHLY CONSISTENT"
        elif mean_corr > 0.4:
            verdict = "moderately consistent"
        elif mean_corr > 0.2:
            verdict = "weakly consistent"
        else:
            verdict = "NOT consistent"

        print(f"    Op {k}: mean_corr={mean_corr:.4f}  median={median_corr:.4f}  "
              f"std={std_corr:.4f}  -> {verdict}")

        consistency_results[k] = {
            'mean_corr': mean_corr,
            'median_corr': median_corr,
            'std_corr': std_corr,
            'verdict': verdict,
        }

    # Per-group summary: which group does each op primarily affect?
    print(f"\n  Primary SMS effect per op:")
    primary_effects = {}
    for k in top_4_ops:
        profile = op_sms_profiles[k]
        primary_group = max(profile.keys(), key=lambda g: profile[g]['mean'])
        primary_val = profile[primary_group]['mean']
        total_val = sum(profile[g]['mean'] for g in profile)
        pct = primary_val / max(total_val, 1e-10) * 100

        effect_desc = ""
        if primary_group == 'group_f0s':
            effect_desc = "pitch shift"
        elif primary_group == 'group_amps':
            effect_desc = "harmonic amplitude modification"
        elif primary_group == 'indep_freqs':
            effect_desc = "independent sine frequency shift"
        elif primary_group == 'indep_amps':
            effect_desc = "independent sine amplitude modification"
        elif primary_group == 'noise_bands':
            effect_desc = "noise band modification"

        print(f"    Op {k}: primary={primary_group} ({pct:.1f}% of total delta) -> {effect_desc}")
        primary_effects[k] = {
            'primary_group': primary_group,
            'primary_pct': float(pct),
            'effect_desc': effect_desc,
        }

    results = {
        'top_4_ops': top_4_ops,
        'global_mean_strengths': global_mean_strengths.tolist(),
        'op_sms_profiles': {
            str(k): {g: {'mean': v['mean'], 'std': v['std']}
                     for g, v in profile.items()}
            for k, profile in op_sms_profiles.items()
        },
        'consistency': {str(k): v for k, v in consistency_results.items()},
        'primary_effects': {str(k): v for k, v in primary_effects.items()},
        'n_samples': n_actual,
    }

    print(f"\n  Part 3 complete: analyzed {n_actual} samples for {len(top_4_ops)} ops.")
    return results


# ============================================================
# Main
# ============================================================

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("VERIFY GT TREE VISION")
    print("Full hierarchical sparse codec verification")
    print("=" * 60)
    print(f"  Device: {device}")
    print(f"  Output: {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load all models
    models = load_models(device)

    # Load all paired data
    pairs = load_paired_data()

    if len(pairs) < 30:
        print(f"\nERROR: Only {len(pairs)} pairs loaded. Need at least 30.")
        print("Check that data cache, manifest, and latent files are accessible.")
        return

    # ---- Part 1: Cross-Instrument Op Activation Analysis ----
    part1_results = part1_cross_instrument(models, pairs)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- Part 2: Generate Isolated Op Audio Files ----
    part2_results = part2_audio_renders(models, pairs)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- Part 3: Op-to-SMS Bridge Analysis ----
    part3_results = part3_sms_bridge(models, pairs, n_samples=40)

    gc.collect()
    torch.cuda.empty_cache()

    # ---- Save combined summary ----
    summary = {
        'part1_cross_instrument': part1_results,
        'part2_rendered_samples': part2_results,
        'part3_sms_bridge': part3_results,
    }

    summary_path = OUTPUT_DIR / "vision_verification.json"
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print(f"\n  Part 1 — Cross-Instrument Activation:")
    n_specialized = sum(1 for v in part1_results['specialization_scores'].values()
                        if v['verdict'] == 'SPECIALIZED')
    n_moderate = sum(1 for v in part1_results['specialization_scores'].values()
                     if v['verdict'] == 'moderate')
    n_universal = sum(1 for v in part1_results['specialization_scores'].values()
                      if v['verdict'] == 'universal')
    print(f"    {n_specialized} specialized, {n_moderate} moderate, {n_universal} universal ops")

    print(f"\n  Part 2 — Audio Renders:")
    print(f"    {len(part2_results)} samples rendered to {OUTPUT_DIR / 'audio_samples'}")
    cats_rendered = set(r['category'] for r in part2_results)
    print(f"    Categories: {', '.join(sorted(cats_rendered))}")

    print(f"\n  Part 3 — SMS Bridge:")
    top_4 = part3_results['top_4_ops']
    for k in top_4:
        k_str = str(k)
        if k_str in part3_results['consistency']:
            cons = part3_results['consistency'][k_str]
            eff = part3_results['primary_effects'].get(k_str, {})
            print(f"    Op {k}: {cons.get('verdict', 'N/A')} "
                  f"(corr={cons.get('mean_corr', 0):.3f}), "
                  f"effect={eff.get('effect_desc', 'unknown')}")

    print(f"\n  All outputs: {OUTPUT_DIR}")
    print(f"  Audio samples: {OUTPUT_DIR / 'audio_samples'}")
    print(f"  Summary JSON: {summary_path}")


if __name__ == "__main__":
    main()
