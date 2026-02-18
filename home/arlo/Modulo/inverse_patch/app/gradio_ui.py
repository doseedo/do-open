#!/usr/bin/env python3
"""
Operation Tree Explorer — Gradio UI served at doseedo.com/do

Interactive interface for exploring the Phase 1 SMS codec + Phase 2 operation tree
decomposition. Users can:
  - Browse samples by instrument group
  - See the SMS → residual → operation tree decomposition
  - Adjust strength of each operation with sliders
  - Swap operations between up to 4 samples (1 base + 3 donors)
  - Hear results in real time as edits are made
"""

import sys
import os
import gc
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as TF
import torchaudio
import orjson
from pathlib import Path
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from bidirectional_sms_z import (
    BidirectionalCodec, SMS_DIM,
    additive_synth, get_noise_band_edges,
    MAX_GROUPS, MAX_PARTIALS, MAX_INDEPENDENT, N_NOISE_BANDS,
    LOG10_20, LOG10_20K, HOP_SIZE,
)
from phase2_operation_tree import OperationTreeCodec
from audio_domain_editor import AudioDomainEditor, AudioDomainEditorInference
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

# ============================================================
# Paths
# ============================================================
SCRIPT_DIR = Path('/home/arlo/do-repo/home/arlo/Modulo/inverse_patch/scripts')
PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_operation_tree" / "operation_tree.pt"
GT_TREE_PATH = SCRIPT_DIR.parent / "test_outputs" / "phase2_gt_tree" / "operation_tree_gt.pt"
DISCOVERED_AXES_PATH = SCRIPT_DIR.parent / "test_outputs" / "contrastive_discovery" / "discovered_axes.pt"
PITCHBIN_AXES_PATH = SCRIPT_DIR.parent / "test_outputs" / "pitchbin_discovery" / "pitchbin_axes.pt"
AUDIO_EDITOR_PATH = SCRIPT_DIR.parent / "test_outputs" / "audio_domain_editor" / "audio_domain_editor.pt"
DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_DATA_DIR = SCRIPT_DIR.parent / "data" / "sms_v4"
LATENT_BASE = Path("/home/arlo/gcs-bucket/Latents/protools")

DCAE_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
VOCODER_CKPT = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

SAMPLE_RATE = 44100
Z_DIM = 128
MAX_FRAMES = 256
DCAE_HOP = int(SAMPLE_RATE / 10.8)  # ~4083 samples per DCAE frame

# SMS 102-dim layout (from bidirectional_sms_z.py normalize_sms)
# [6 group f0s | 48 group amps (6×8) | 20 indep freqs | 20 indep amps | 8 noise bands]
SMS_GROUP_F0 = slice(0, 6)
SMS_GROUP_AMPS = [(6 + g*8, 6 + (g+1)*8) for g in range(6)]  # per-group slices
SMS_INDEP_FREQS = slice(54, 74)
SMS_INDEP_AMPS = slice(74, 94)
SMS_NOISE = slice(94, 102)

# ============================================================
# Instrument categories
# ============================================================
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

# ============================================================
# Global state
# ============================================================
models = {}  # codec, tree, dcae, res_mean, res_std, gt_tree, gt_res_mean, gt_res_std
sample_catalog = {}  # category → list of {name, latent_path}
gt_sms_index = {}  # latent_path → {z_sms: [8,16,T], start_frame: int}
decomposition_cache = {}  # latent_path → decomposition dict


def load_models():
    """Load all models into GPU once."""
    if models:
        return
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    models['codec'] = codec
    models['tree'] = tree
    models['dcae'] = dcae
    models['res_mean'] = res_mean
    models['res_std'] = res_std
    models['device'] = device
    models['n_ops'] = ckpt['n_ops']

    # Compute operation ranking from weight norms
    state = ckpt['model']
    importance = []
    for k in range(ckpt['n_ops']):
        key = f'operations.{k}.net.4.weight'
        importance.append(state[key].norm().item() if key in state else 0.0)
    models['op_ranking'] = sorted(range(len(importance)), key=lambda i: -importance[i])

    # Load GT tree if available
    if GT_TREE_PATH.exists():
        print("Loading GT operation tree...")
        gt_ckpt = torch.load(GT_TREE_PATH, weights_only=False, map_location='cpu')
        gt_tree = OperationTreeCodec(
            z_dim=Z_DIM, n_ops=gt_ckpt['n_ops'], param_dim=gt_ckpt['param_dim'],
            encoder_hidden=256, top_k=gt_ckpt['top_k'],
        )
        gt_tree.load_state_dict(gt_ckpt['model'])
        gt_tree = gt_tree.to(device).eval()
        models['gt_tree'] = gt_tree
        models['gt_res_mean'] = gt_ckpt['res_mean'].to(device)
        models['gt_res_std'] = gt_ckpt['res_std'].to(device)
        print("  GT tree loaded!")
    else:
        print(f"  GT tree not found at {GT_TREE_PATH} — will show pred tree only")

    # Load Audio Domain Editor (STFT-domain editing, bypasses DCAE)
    if AUDIO_EDITOR_PATH.exists():
        print("Loading Audio Domain Editor...")
        models['audio_editor'] = AudioDomainEditorInference(
            model_path=str(AUDIO_EDITOR_PATH), device=device
        )
        print("  Audio Domain Editor loaded!")
    else:
        print(f"  Audio Domain Editor not found at {AUDIO_EDITOR_PATH}")

    # Build GT SMS index: latent_path → z_gt_sms
    _build_gt_sms_index()

    # Load discovered perceptual axes (from contrastive discovery)
    _load_discovered_axes()

    print(f"Models loaded! Device={device}, ops={ckpt['n_ops']}, "
          f"ranking={models['op_ranking']}, "
          f"gt_tree={'yes' if 'gt_tree' in models else 'no'}, "
          f"gt_sms_entries={len(gt_sms_index)}, "
          f"audio_editor={'yes' if 'audio_editor' in models else 'no'}, "
          f"perceptual_axes={len(models.get('perceptual_axes', []))}")


def _build_gt_sms_index():
    """Build index mapping latent_path → GT SMS latent from Phase 1 data cache."""
    if gt_sms_index:
        return

    if not DATA_CACHE_PATH.exists():
        print(f"  No data cache at {DATA_CACHE_PATH} — GT SMS unavailable")
        return

    print("Building GT SMS index from Phase 1 data cache...")
    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')

    # Build sms_path → latent_path from manifest
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())
    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']

    matched = 0
    for sample in cache:
        sms_path = sample['path']
        latent_path = sms_to_latent.get(sms_path)
        if latent_path:
            # Find start_frame by reloading SMS file
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

            gt_sms_index[latent_path] = {
                'z_sms': sample['z_sms'],  # [8, 16, T_cache]
                'sms_norm': sample['sms_norm'],  # [T_cache, 102]
                'start_frame': start_frame,
            }
            matched += 1

    print(f"  GT SMS index: {matched} entries")


def _load_discovered_axes():
    """Load contrastive-discovered perceptual axes for z-space editing."""
    if not DISCOVERED_AXES_PATH.exists():
        print(f"  No discovered axes at {DISCOVERED_AXES_PATH}")
        return

    print("Loading discovered perceptual axes...")
    data = torch.load(DISCOVERED_AXES_PATH, weights_only=False, map_location='cpu')

    # within_pca: list of {direction: np.array[128], variance_explained: float}
    pca_axes = []
    for ax in data.get('within_pca', [])[:8]:
        direction = ax['direction']
        if isinstance(direction, np.ndarray):
            direction = torch.from_numpy(direction).float()
        elif not isinstance(direction, torch.Tensor):
            direction = torch.tensor(direction, dtype=torch.float32)
        # Ensure unit norm
        direction = direction / (direction.norm() + 1e-10)
        pca_axes.append({
            'direction': direction,
            'variance_explained': ax['variance_explained'],
        })

    # within_ica: list of lists (each is 128-dim direction vector)
    ica_axes = []
    for comp in data.get('within_ica', [])[:6]:
        if isinstance(comp, np.ndarray):
            direction = torch.from_numpy(comp).float()
        elif isinstance(comp, (list, tuple)):
            direction = torch.tensor(comp, dtype=torch.float32)
        else:
            continue
        direction = direction / (direction.norm() + 1e-10)
        ica_axes.append({'direction': direction})

    models['perceptual_axes'] = pca_axes
    models['ica_axes'] = ica_axes
    print(f"  Loaded {len(pca_axes)} contrastive PCA axes, {len(ica_axes)} ICA axes")

    # Load pitch-binned timbral axes
    pitchbin_axes = []
    if PITCHBIN_AXES_PATH.exists():
        pb_data = torch.load(PITCHBIN_AXES_PATH, weights_only=False, map_location='cpu')
        for ax in pb_data.get('pitchbin_pca', [])[:8]:
            direction = ax['direction']
            if isinstance(direction, np.ndarray):
                direction = torch.from_numpy(direction).float()
            elif not isinstance(direction, torch.Tensor):
                direction = torch.tensor(direction, dtype=torch.float32)
            direction = direction / (direction.norm() + 1e-10)
            pitchbin_axes.append({
                'direction': direction,
                'variance_explained': ax['variance_explained'],
            })
        print(f"  Loaded {len(pitchbin_axes)} pitch-binned timbral axes")
    else:
        print(f"  No pitch-binned axes at {PITCHBIN_AXES_PATH}")
    models['pitchbin_axes'] = pitchbin_axes

    # Compute safe edit scales from data cache
    # Project all z values onto each axis → std = 1 slider unit
    _compute_axis_safe_scales()


def _compute_axis_safe_scales():
    """Compute per-axis safe scales so slider=1 means 1σ of observed variation."""
    if not DATA_CACHE_PATH.exists():
        print("  No data cache — using default axis scales")
        return

    print("  Computing safe axis scales from data cache...")
    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')

    # Collect all z_sms values [N, 128] from cache
    z_samples = []
    for sample in cache:
        z_sms = sample.get('z_sms')  # [8, 16, T]
        if z_sms is not None:
            T = z_sms.shape[2]
            z_flat = z_sms.permute(2, 0, 1).reshape(T, Z_DIM).float()
            z_samples.append(z_flat)

    if not z_samples:
        print("  No z_sms in cache — using default scales")
        return

    z_all = torch.cat(z_samples, dim=0)  # [total_frames, 128]
    print(f"  z_all: {z_all.shape[0]} frames for scale computation")

    for axis_group_name in ['perceptual_axes', 'ica_axes', 'pitchbin_axes']:
        axes = models.get(axis_group_name, [])
        for ax in axes:
            direction = ax['direction']  # [128]
            projections = z_all @ direction  # [N]
            ax['safe_scale'] = float(projections.std())

    # Report
    for name in ['pitchbin_axes', 'perceptual_axes', 'ica_axes']:
        axes = models.get(name, [])
        if axes:
            scales = [f"{ax.get('safe_scale', 1.0):.3f}" for ax in axes[:4]]
            print(f"    {name} scales: [{', '.join(scales)}, ...]")


def build_sample_catalog():
    """Build catalog of available samples from manifest."""
    if sample_catalog:
        return

    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    for entry in manifest['entries']:
        lp = entry.get('latent_path', '')
        name = Path(lp).stem
        lp_lower = lp.lower()

        cat = 'Other'
        for category, keywords in CATEGORY_KEYWORDS.items():
            if any(kw in lp_lower for kw in keywords):
                cat = category
                break

        sample_catalog.setdefault(cat, []).append({
            'name': name,
            'latent_path': lp,
            'display': f"{name}",
        })

    # Sort each category
    for cat in sample_catalog:
        sample_catalog[cat].sort(key=lambda x: x['name'])

    total = sum(len(v) for v in sample_catalog.values())
    print(f"Catalog: {total} samples in {len(sample_catalog)} categories")


def get_categories():
    build_sample_catalog()
    return sorted(sample_catalog.keys())


def get_samples_for_category(category):
    build_sample_catalog()
    items = sample_catalog.get(category, [])
    return [s['display'] for s in items]


def find_latent_path(category, display_name):
    items = sample_catalog.get(category, [])
    for s in items:
        if s['display'] == display_name:
            return s['latent_path']
    return None


# ============================================================
# Audio / decomposition functions
# ============================================================

def load_latent(path):
    d = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(d, dict) and 'latents' in d:
        z = d['latents']
    elif isinstance(d, torch.Tensor):
        z = d
    else:
        raise ValueError(f"Unknown format: {path}")
    if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16:
        raise ValueError(f"Bad shape {z.shape}")
    if z.shape[2] > MAX_FRAMES:
        z = z[:, :, :MAX_FRAMES]
    return z


def decompose(latent_path):
    """Full decomposition pipeline, cached."""
    if latent_path in decomposition_cache:
        return decomposition_cache[latent_path]

    load_models()
    device = models['device']
    codec = models['codec']
    tree = models['tree']
    res_mean = models['res_mean']
    res_std = models['res_std']

    z_4d = load_latent(latent_path)
    z_real = z_4d.unsqueeze(0).to(device)
    z_flat = codec.z_to_flat(z_real)

    with torch.no_grad():
        sms_params = codec.forward_F(z_flat)  # [1, T, 102]
        z_sms = codec.forward_G(sms_params)
        residual = z_flat - z_sms
        residual_norm = (residual - res_mean) / res_std

        hidden = tree.encode(residual_norm)
        recon_norm, activations_sparse, all_params = tree.decode(hidden)
        recon = recon_norm * res_std + res_mean
        z_full = z_sms + recon

        # Per-op contributions using SPARSE alpha (matches tree.decode behavior)
        # Each contrib is just the denormalized op output WITHOUT res_mean
        # (res_mean is added once in the render function, not per-op)
        raw_alpha = tree.activation_head(hidden)
        alpha_full = TF.softplus(raw_alpha)  # [1, T, n_ops]

        # Apply same top-k mask as tree.decode
        topk_vals, topk_idx = torch.topk(alpha_full, tree.top_k, dim=-1)
        mask = torch.zeros_like(alpha_full)
        mask.scatter_(-1, topk_idx, 1.0)
        alpha_sparse = alpha_full * mask

        op_contribs = []
        for k in range(tree.n_ops):
            params_k = tree.param_heads[k](hidden)
            contrib_k = tree.operations[k](params_k)
            alpha_k = alpha_sparse[:, :, k:k+1]
            # NO res_mean here — just scaled contribution
            scaled = (alpha_k * contrib_k) * res_std
            op_contribs.append(scaled.cpu())

    # GT SMS decomposition (if available)
    gt_entry = gt_sms_index.get(latent_path)
    z_gt_sms = None
    z_gt_full = None
    z_gt_flat_crop = None
    gt_op_contribs = None

    if gt_entry is not None:
        z_gt_sms_4d = gt_entry['z_sms']  # [8, 16, T_cache]
        start_frame = gt_entry['start_frame']
        T_cache = z_gt_sms_4d.shape[2]

        T_full = z_flat.shape[1]
        end_frame = min(start_frame + T_cache, T_full)
        T_aligned = end_frame - start_frame

        if T_aligned >= 10:
            z_gt_sms_flat = z_gt_sms_4d[:, :, :T_aligned].permute(2, 0, 1).reshape(1, T_aligned, Z_DIM).to(device)
            z_flat_crop = z_flat[:, start_frame:end_frame, :]

            z_gt_sms = z_gt_sms_flat
            z_gt_flat_crop = z_flat_crop  # GT audio from same temporal window

            # If GT tree is available, compute GT tree reconstruction
            gt_tree = models.get('gt_tree')
            if gt_tree is not None:
                gt_res_mean = models['gt_res_mean']
                gt_res_std = models['gt_res_std']

                with torch.no_grad():
                    gt_residual = z_flat_crop - z_gt_sms_flat
                    gt_residual_norm = (gt_residual - gt_res_mean) / gt_res_std

                    gt_hidden = gt_tree.encode(gt_residual_norm)
                    gt_recon_norm, _, _ = gt_tree.decode(gt_hidden)
                    gt_recon = gt_recon_norm * gt_res_std + gt_res_mean
                    z_gt_full = z_gt_sms_flat + gt_recon

                    # Per-op with sparse alpha, NO res_mean per-op
                    gt_raw_alpha = gt_tree.activation_head(gt_hidden)
                    gt_alpha_full = TF.softplus(gt_raw_alpha)
                    gt_topk_vals, gt_topk_idx = torch.topk(gt_alpha_full, gt_tree.top_k, dim=-1)
                    gt_mask = torch.zeros_like(gt_alpha_full)
                    gt_mask.scatter_(-1, gt_topk_idx, 1.0)
                    gt_alpha_sparse = gt_alpha_full * gt_mask

                    gt_op_contribs = []
                    for k in range(gt_tree.n_ops):
                        gp = gt_tree.param_heads[k](gt_hidden)
                        gc = gt_tree.operations[k](gp)
                        ga = gt_alpha_sparse[:, :, k:k+1]
                        gt_scaled = (ga * gc) * gt_res_std
                        gt_op_contribs.append(gt_scaled.cpu())

    # GT SMS params (from data cache sms_norm, or from codec F on cropped z)
    gt_sms_params = None
    if gt_entry is not None and 'sms_norm' in gt_entry:
        gt_sms_params = gt_entry['sms_norm']  # [T_cache, 102]
        # Align length
        if z_gt_sms is not None:
            T_gt = z_gt_sms.shape[1] if z_gt_sms.dim() == 3 else z_gt_sms.shape[0]
            gt_sms_params = gt_sms_params[:T_gt]

    result = {
        'z_flat': z_flat.cpu(),
        'z_sms': z_sms.cpu(),
        'z_full': z_full.cpu(),
        'sms_params': sms_params.cpu(),  # [1, T, 102] — pred SMS params from F(z)
        'res_mean': res_mean.cpu(),
        'alpha': alpha_sparse.cpu(),
        'op_contribs': [c for c in op_contribs],
        # GT SMS fields (may be None)
        'z_gt_sms': z_gt_sms.cpu() if z_gt_sms is not None else None,
        'z_gt_flat_crop': z_gt_flat_crop.cpu() if z_gt_flat_crop is not None else None,
        'z_gt_full': z_gt_full.cpu() if z_gt_full is not None else None,
        'gt_sms_params': gt_sms_params.cpu() if gt_sms_params is not None else None,  # [T, 102]
        'gt_res_mean': models.get('gt_res_mean', torch.zeros(1)).cpu() if gt_op_contribs else None,
        'gt_op_contribs': gt_op_contribs,
    }

    decomposition_cache[latent_path] = result

    # Keep cache bounded
    if len(decomposition_cache) > 50:
        oldest = next(iter(decomposition_cache))
        del decomposition_cache[oldest]

    return result


def render_sms_audio(sms_norm_2d, sms_group_scales, sms_indep_scale, sms_noise_scale):
    """
    Render SMS params directly via additive synthesis (NOT through G()/DCAE).
    This gives true isolation of sines vs noise.

    Args:
        sms_norm_2d: [T, 102] normalized SMS params (from F(z) or data cache)
        sms_group_scales: list of 6 floats for harmonic group amplitudes
        sms_indep_scale: float for independent sine amplitudes
        sms_noise_scale: float for noise band amplitudes

    Returns:
        (sr, audio_sines), (sr, audio_noise), (sr, audio_mixed)
    """
    T = sms_norm_2d.shape[0]
    sms = sms_norm_2d.clone()

    # Denormalize group f0s: normalized = (log10(f) - LOG10_20) / (LOG10_20K - LOG10_20)
    # → f = 10^(normalized * (LOG10_20K - LOG10_20) + LOG10_20)
    log_range = LOG10_20K - LOG10_20

    group_f0s_norm = sms[:, SMS_GROUP_F0]  # [T, 6]
    group_f0s_hz = torch.pow(10, group_f0s_norm * log_range + LOG10_20)
    group_f0s_hz[group_f0s_norm == 0] = 0  # silent groups stay silent

    # Group amps: already raw, apply scales
    # Each group has 8 partials, we need to expand to sine tracks
    # Build freqs and amps for all group partials: 6 groups × 8 partials = 48 sine tracks
    n_group_sines = MAX_GROUPS * MAX_PARTIALS  # 48
    group_freqs = torch.zeros(T, n_group_sines)
    group_amps = torch.zeros(T, n_group_sines)

    for g in range(MAX_GROUPS):
        f0 = group_f0s_hz[:, g]  # [T]
        start, end = SMS_GROUP_AMPS[g]
        partial_amps = sms[:, start:end].clone()  # [T, 8]
        partial_amps *= sms_group_scales[g]

        for p in range(MAX_PARTIALS):
            idx = g * MAX_PARTIALS + p
            group_freqs[:, idx] = f0 * (p + 1)  # harmonic series
            group_amps[:, idx] = partial_amps[:, p]

    # Independent sines: denormalize freqs, apply amp scale
    indep_freqs_norm = sms[:, SMS_INDEP_FREQS]  # [T, 20]
    indep_freqs_hz = torch.pow(10, indep_freqs_norm * log_range + LOG10_20)
    indep_freqs_hz[indep_freqs_norm == 0] = 0

    indep_amps_raw = sms[:, SMS_INDEP_AMPS].clone()  # [T, 20]
    indep_amps_raw *= sms_indep_scale

    # Combine all sine tracks: 48 group + 20 independent = 68
    all_freqs = torch.cat([group_freqs, indep_freqs_hz], dim=1)  # [T, 68]
    all_amps = torch.cat([group_amps, indep_amps_raw], dim=1)    # [T, 68]

    # Noise bands: already raw, apply scale
    noise_raw = sms[:, SMS_NOISE].clone()  # [T, 8]
    noise_scaled = noise_raw * sms_noise_scale

    hop = DCAE_HOP

    # Render sines only
    audio_sines = additive_synth(all_freqs, all_amps, noise_amps=None, hop=hop)
    peak = np.abs(audio_sines).max()
    if peak > 0.95:
        audio_sines = audio_sines * (0.95 / peak)

    # Render noise only — use additive_synth with real amps so noise clamping
    # (which limits noise to 2x sine energy) works correctly, then subtract sines
    audio_with_noise = additive_synth(all_freqs, all_amps, noise_amps=noise_scaled, hop=hop)
    audio_sines_ref = additive_synth(all_freqs, all_amps, noise_amps=None, hop=hop)
    audio_noise = (audio_with_noise - audio_sines_ref).astype(np.float32)
    peak = np.abs(audio_noise).max()
    if peak > 0.95:
        audio_noise = audio_noise * (0.95 / peak)

    # Render mixed
    audio_mixed = audio_with_noise.copy()
    peak = np.abs(audio_mixed).max()
    if peak > 0.95:
        audio_mixed = audio_mixed * (0.95 / peak)

    return (
        (SAMPLE_RATE, audio_sines.astype(np.float32)),
        (SAMPLE_RATE, audio_noise.astype(np.float32)),
        (SAMPLE_RATE, audio_mixed.astype(np.float32)),
    )


def decode_z_to_audio(z_flat_cpu):
    """z_flat [1, T, 128] CPU tensor → (sr, audio_np)."""
    load_models()
    device = models['device']
    dcae = models['dcae']

    z_flat = z_flat_cpu.to(device)
    B, T, D = z_flat.shape
    z_4d = z_flat.reshape(B, T, 8, 16).permute(0, 2, 3, 1)
    audio_len = int(T * SAMPLE_RATE / 10.8)
    audio_lengths = torch.tensor([audio_len], device=device)

    with torch.no_grad():
        sr, wavs = dcae.decode(z_4d, audio_lengths=audio_lengths, sr=SAMPLE_RATE)

    audio = wavs[0].mean(dim=0).cpu().numpy()
    peak = np.abs(audio).max()
    if peak > 0.95:
        audio = audio * (0.95 / peak)
    return (SAMPLE_RATE, audio.astype(np.float32))


def apply_stft_edits(audio_np, pb_deltas):
    """
    Apply timbral axis edits in STFT domain using the Audio Domain Editor.
    Bypasses DCAE decode — preserves original audio fidelity.

    Args:
        audio_np: (sr, audio_array) tuple from decode_z_to_audio
        pb_deltas: list of floats for pitchbin axis 0-5

    Returns:
        (sr, edited_audio_array) or None if no editor loaded
    """
    editor = models.get('audio_editor')
    if editor is None:
        print("  STFT edit: no audio_editor model loaded")
        return None

    sr, audio = audio_np
    audio_tensor = torch.from_numpy(audio).float()

    # Apply each axis edit sequentially
    for axis_idx, delta in enumerate(pb_deltas):
        if delta != 0.0 and axis_idx < 6:
            audio_tensor = editor.edit(audio_tensor, axis_idx, float(delta))

    result = audio_tensor.numpy().astype(np.float32)
    peak = np.abs(result).max()
    if peak > 0.95:
        result = result * (0.95 / peak)
    return (sr, result)


def render_stft_comparison(original_audio_np, edited_audio_np):
    """Generate side-by-side STFT spectrograms: original vs edited.

    Args:
        original_audio_np: (sr, audio_array) tuple
        edited_audio_np: (sr, audio_array) tuple or None
    Returns:
        matplotlib Figure
    """
    plt.close('all')

    sr, orig = original_audio_np
    has_edit = edited_audio_np is not None

    fig, axes = plt.subplots(1, 2 if has_edit else 1, figsize=(14, 4), dpi=120)
    if not has_edit:
        axes = [axes]

    window = np.hanning(2048)
    hop = 512

    def compute_spectrogram(audio):
        # Compute log-magnitude STFT
        n_frames = (len(audio) - 2048) // hop + 1
        spec = np.zeros((1025, max(n_frames, 1)))
        for t in range(n_frames):
            frame = audio[t * hop:t * hop + 2048] * window
            fft_result = np.fft.rfft(frame)
            spec[:, t] = np.log1p(np.abs(fft_result))
        return spec

    orig_spec = compute_spectrogram(orig)

    # Frequency axis in kHz
    freqs_khz = np.linspace(0, sr / 2 / 1000, orig_spec.shape[0])
    time_s = np.arange(orig_spec.shape[1]) * hop / sr

    vmin = orig_spec.min()
    vmax = orig_spec.max()

    im1 = axes[0].pcolormesh(time_s, freqs_khz, orig_spec, cmap='magma',
                              vmin=vmin, vmax=vmax, shading='auto')
    axes[0].set_title('Original STFT', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Freq (kHz)', fontsize=9)
    axes[0].set_xlabel('Time (s)', fontsize=9)
    axes[0].set_ylim(0, 16)

    if has_edit:
        _, edit = edited_audio_np
        edit_spec = compute_spectrogram(edit)
        # Match dimensions
        min_t = min(orig_spec.shape[1], edit_spec.shape[1])
        edit_spec = edit_spec[:, :min_t]

        im2 = axes[1].pcolormesh(time_s[:min_t], freqs_khz, edit_spec, cmap='magma',
                                  vmin=vmin, vmax=vmax, shading='auto')
        axes[1].set_title('Edited STFT', fontsize=11, fontweight='bold')
        axes[1].set_xlabel('Time (s)', fontsize=9)
        axes[1].set_ylim(0, 16)
        fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def render_z_map(z_original_flat, z_edited_flat):
    """Generate z-space heatmap showing current values and edit delta.

    Args:
        z_original_flat: [1, T, 128] original z (default reconstruction)
        z_edited_flat: [1, T, 128] z after user edits
    Returns:
        matplotlib Figure
    """
    plt.close('all')

    orig = z_original_flat.squeeze(0).mean(0).numpy()  # [128]
    edit = z_edited_flat.squeeze(0).mean(0).numpy()    # [128]
    delta = edit - orig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # Left: current z values reshaped to 8x16 (DCAE channel structure)
    grid = edit.reshape(8, 16)
    im1 = ax1.imshow(grid, cmap='viridis', aspect='equal', interpolation='nearest')
    ax1.set_title('Current z', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Channel', fontsize=8)
    ax1.set_xlabel('Feature', fontsize=8)
    ax1.set_yticks(range(8))
    ax1.set_xticks(range(0, 16, 4))
    ax1.tick_params(labelsize=7)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Right: edit delta (diverging colormap: blue=decrease, red=increase)
    delta_grid = delta.reshape(8, 16)
    dmax = max(np.abs(delta).max(), 0.001)
    im2 = ax2.imshow(delta_grid, cmap='RdBu_r', aspect='equal',
                     vmin=-dmax, vmax=dmax, interpolation='nearest')
    ax2.set_title('Edit \u0394', fontsize=10, fontweight='bold')
    ax2.set_xlabel('Feature', fontsize=8)
    ax2.set_yticks(range(8))
    ax2.set_xticks(range(0, 16, 4))
    ax2.tick_params(labelsize=7)
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Add dim-index annotations on delta for cells with large changes
    threshold = dmax * 0.4
    for r in range(8):
        for c in range(16):
            dim_idx = r * 16 + c
            val = delta_grid[r, c]
            if abs(val) > threshold:
                color = 'white' if abs(val) > dmax * 0.6 else 'black'
                ax2.text(c, r, str(dim_idx), ha='center', va='center',
                         fontsize=5, color=color, fontweight='bold')

    fig.tight_layout()
    return fig


# ============================================================
# Core UI callbacks
# ============================================================

def on_category_change(category):
    """When instrument category changes, update sample dropdown."""
    samples = get_samples_for_category(category)
    return gr.update(choices=samples, value=samples[0] if samples else None)


def on_sample_select(category, sample_name):
    """When a sample is selected, decompose and return info + audio."""
    n_ops = models.get('n_ops', 8)
    n_pca = len(models.get('perceptual_axes', []))
    n_ica = len(models.get('ica_axes', []))
    n_pb = len(models.get('pitchbin_axes', []))
    # Returns: gt, pred_sms, pred_sms+tree, raw_pred_sms,
    #          gt_crop, gt_sms, gt_sms+gt_tree, raw_gt_sms,
    #          stft_audio,
    #          info, z_map, *op_scales, *pca_zeros, *ica_zeros, *pb_zeros
    n_out = 9  # audio outputs before info (8 decomp + 1 stft)
    empty = (*([None] * n_out), "", None, *([1.0] * n_ops), *([0.0] * n_pca), *([0.0] * n_ica), *([0.0] * n_pb))

    if not sample_name or not category:
        return empty

    lp = find_latent_path(category, sample_name)
    if not lp:
        return (*([None] * n_out), "Sample not found", None, *([1.0] * n_ops), *([0.0] * n_pca), *([0.0] * n_ica), *([0.0] * n_pb))

    try:
        d = decompose(lp)
    except Exception as e:
        return (*([None] * n_out), f"Error: {e}", None, *([1.0] * n_ops), *([0.0] * n_pca), *([0.0] * n_ica), *([0.0] * n_pb))

    # Audio outputs
    audio_gt = decode_z_to_audio(d['z_flat'])
    audio_pred_sms = decode_z_to_audio(d['z_sms'])
    audio_pred_full = decode_z_to_audio(d['z_full'])

    # Raw pred SMS via additive synth (not DCAE-encoded)
    ones6 = [1.0] * 6
    _, _, audio_raw_pred_sms = render_sms_audio(d['sms_params'].squeeze(0), ones6, 1.0, 1.0)

    # GT SMS audio (if available) — use z_gt_flat_crop so GT audio matches temporal window
    audio_gt_sms = None
    audio_gt_full = None
    audio_gt_crop = None
    audio_raw_gt_sms = None
    if d['z_gt_sms'] is not None:
        audio_gt_sms = decode_z_to_audio(d['z_gt_sms'])
    if d['z_gt_flat_crop'] is not None:
        audio_gt_crop = decode_z_to_audio(d['z_gt_flat_crop'])
    if d['z_gt_full'] is not None:
        audio_gt_full = decode_z_to_audio(d['z_gt_full'])
    if d.get('gt_sms_params') is not None:
        _, _, audio_raw_gt_sms = render_sms_audio(d['gt_sms_params'], ones6, 1.0, 1.0)

    # Info text
    cos_pred_sms = TF.cosine_similarity(d['z_flat'], d['z_sms'], dim=-1).mean().item()
    cos_pred_full = TF.cosine_similarity(d['z_flat'], d['z_full'], dim=-1).mean().item()

    info_lines = [
        "**Pred SMS (codec roundtrip)**:",
        f"  cos_sim: {cos_pred_sms:.4f} → +tree: {cos_pred_full:.4f}  "
        f"(gain: +{cos_pred_full - cos_pred_sms:.4f})",
    ]

    if d['z_gt_sms'] is not None:
        # GT SMS comparison uses cropped z_flat to match temporal window
        gt_entry = gt_sms_index.get(lp)
        if gt_entry:
            sf = gt_entry['start_frame']
            T_gt = d['z_gt_sms'].shape[1]
            z_flat_crop = d['z_flat'][:, sf:sf+T_gt, :]
            cos_gt_sms = TF.cosine_similarity(z_flat_crop, d['z_gt_sms'], dim=-1).mean().item()
            info_lines.append(f"\n**GT SMS (actual SMS→DCAE)**:")
            info_lines.append(f"  cos_sim: {cos_gt_sms:.4f}")
            if d['z_gt_full'] is not None:
                cos_gt_full = TF.cosine_similarity(z_flat_crop, d['z_gt_full'], dim=-1).mean().item()
                info_lines.append(f"  +GT tree: {cos_gt_full:.4f}  "
                                  f"(gain: +{cos_gt_full - cos_gt_sms:.4f})")
    else:
        info_lines.append("\n*GT SMS not available for this sample*")

    ranking = models.get('op_ranking', list(range(8)))
    mean_alpha = d['alpha'].squeeze(0).mean(dim=0)

    info_lines.append(f"\n**Frames**: {d['z_flat'].shape[1]}")
    info_lines.append("\n**Op activations** (pred tree):")
    for rank, k in enumerate(ranking[:6]):
        contrib_norm = d['op_contribs'][k].squeeze(0).norm(dim=-1).mean().item()
        info_lines.append(
            f"  Op {k} (rank {rank}): alpha={mean_alpha[k]:.2f}  "
            f"contrib={contrib_norm:.3f}"
        )

    info_text = "\n".join(info_lines)
    default_scales = [1.0] * n_ops
    axis_zeros = [0.0] * (n_pca + n_ica + n_pb)

    # Initial z-map (no edits, so delta = 0)
    initial_z_map = render_z_map(d['z_full'], d['z_full'])

    # STFT audio: show original (unedited) as baseline if editor loaded
    stft_baseline = audio_gt if models.get('audio_editor') is not None else None

    return (audio_gt, audio_pred_sms, audio_pred_full, audio_raw_pred_sms,
            audio_gt_crop, audio_gt_sms, audio_gt_full, audio_raw_gt_sms,
            stft_baseline,
            info_text, initial_z_map, *default_scales, *axis_zeros)


def render_edited_audio(mode, category, sample_name,
                        donor1_cat, donor1_sample, donor1_op, donor1_mix,
                        donor2_cat, donor2_sample, donor2_op, donor2_mix,
                        donor3_cat, donor3_sample, donor3_op, donor3_mix,
                        sms_g1, sms_g2, sms_g3, sms_g4, sms_g5, sms_g6,
                        sms_indep, sms_noise, res_mix,
                        *op_and_axis_scales):
    """
    Resynthesize audio with edited SMS params + operation strengths + donor swaps
    + discovered perceptual axis deltas.
    SMS group sliders edit the sinusoidal model, then re-encode through G().
    res_mix controls SMS/residual balance: 0=SMS only, 1=default, 2=exaggerated residual.
    Returns: (sines_audio, noise_audio, mixed_audio, edited_audio, z_map_figure)
    """
    if not sample_name or not category:
        return None, None, None, None, None, None, None

    lp = find_latent_path(category, sample_name)
    if not lp:
        return None, None, None, None, None, None, None

    try:
        d = decompose(lp)
    except Exception:
        return None, None, None, None, None, None, None

    n_ops = models.get('n_ops', 8)
    n_pca_axes = len(models.get('perceptual_axes', []))
    n_ica_axes = len(models.get('ica_axes', []))
    n_pb_axes = len(models.get('pitchbin_axes', []))

    all_scales = list(op_and_axis_scales)
    scales = all_scales[:n_ops]
    while len(scales) < n_ops:
        scales.append(1.0)

    pca_deltas = all_scales[n_ops:n_ops + n_pca_axes]
    while len(pca_deltas) < n_pca_axes:
        pca_deltas.append(0.0)

    ica_deltas = all_scales[n_ops + n_pca_axes:n_ops + n_pca_axes + n_ica_axes]
    while len(ica_deltas) < n_ica_axes:
        ica_deltas.append(0.0)

    pb_deltas = all_scales[n_ops + n_pca_axes + n_ica_axes:n_ops + n_pca_axes + n_ica_axes + n_pb_axes]
    while len(pb_deltas) < n_pb_axes:
        pb_deltas.append(0.0)

    device = models['device']
    codec = models['codec']
    sms_group_scales = [sms_g1, sms_g2, sms_g3, sms_g4, sms_g5, sms_g6]

    use_gt = (mode == "GT SMS + GT Tree") and d.get('gt_op_contribs') is not None

    # Pick the right SMS params and tree contribs for this mode
    if use_gt:
        contribs = d['gt_op_contribs']
        r_mean = d['gt_res_mean']
        # Use GT sms_params if available, else fall back to pred
        if d.get('gt_sms_params') is not None:
            sms_raw = d['gt_sms_params'].unsqueeze(0).clone()  # [1, T, 102]
        else:
            sms_raw = d['sms_params'].clone()
    else:
        contribs = d['op_contribs']
        r_mean = d['res_mean']
        sms_raw = d['sms_params'].clone()  # [1, T, 102]

    # Edit SMS params: scale each group's partial amplitudes
    sms_edited = sms_raw.clone()
    for g, scale in enumerate(sms_group_scales):
        start, end = SMS_GROUP_AMPS[g]
        sms_edited[:, :, start:end] *= scale

    # Scale independent sines (amps only, keep freqs)
    sms_edited[:, :, SMS_INDEP_AMPS] *= sms_indep

    # Scale noise bands
    sms_edited[:, :, SMS_NOISE] *= sms_noise

    sms_is_edited = (any(s != 1.0 for s in sms_group_scales)
                     or sms_indep != 1.0 or sms_noise != 1.0)

    if use_gt and d.get('z_gt_sms') is not None:
        # GT mode: use actual z_gt_sms (DCAE-encoded SMS) as base
        # This matches the preview exactly at default settings
        z_sms_base = d['z_gt_sms'].clone()  # [1, T_gt, 128]
        T = z_sms_base.shape[1]

        if sms_is_edited:
            # Apply SMS edits as a delta: z_base + (G(edited) - G(original))
            with torch.no_grad():
                z_g_original = codec.forward_G(sms_raw[:, :T].to(device)).cpu()
                z_g_edited = codec.forward_G(sms_edited[:, :T].to(device)).cpu()
            z_sms_base = z_sms_base + (z_g_edited - z_g_original)
    else:
        # Pred mode: use G(sms) as base
        with torch.no_grad():
            z_sms_base = codec.forward_G(sms_edited.to(device)).cpu()
        T = z_sms_base.shape[1]

    # SMS direct render via additive synthesis (true sine/noise isolation)
    if use_gt and d.get('gt_sms_params') is not None:
        sms_for_render = d['gt_sms_params']  # [T, 102]
    else:
        sms_for_render = d['sms_params'].squeeze(0)  # [T, 102]
    sms_for_render = sms_for_render[:T]

    audio_sines, audio_noise, audio_sms_mixed = render_sms_audio(
        sms_for_render, sms_group_scales, sms_indep, sms_noise
    )

    # Build residual separately from SMS base
    residual = r_mean.unsqueeze(0).expand_as(z_sms_base).clone()

    for k in range(n_ops):
        residual = residual + scales[k] * contribs[k][:, :T]

    # Apply donor swaps (modify residual)
    donors = [
        (donor1_cat, donor1_sample, donor1_op, donor1_mix),
        (donor2_cat, donor2_sample, donor2_op, donor2_mix),
        (donor3_cat, donor3_sample, donor3_op, donor3_mix),
    ]

    for d_cat, d_sample, d_op_str, d_mix in donors:
        if not d_cat or not d_sample or d_mix == 0 or d_op_str == "None":
            continue

        d_lp = find_latent_path(d_cat, d_sample)
        if not d_lp:
            continue

        try:
            d_donor = decompose(d_lp)
        except Exception:
            continue

        try:
            op_idx = int(d_op_str.split("Op ")[1].split(" ")[0])
        except (IndexError, ValueError):
            continue

        if use_gt and d_donor.get('gt_op_contribs') is not None:
            donor_contribs = d_donor['gt_op_contribs']
            donor_T = d_donor['z_gt_sms'].shape[1] if d_donor.get('z_gt_sms') is not None else d_donor['z_flat'].shape[1]
        else:
            donor_contribs = d_donor['op_contribs']
            donor_T = d_donor['z_flat'].shape[1]

        T_min = min(T, donor_T)

        base_contrib = contribs[op_idx][:, :T_min] * scales[op_idx]
        donor_contrib = donor_contribs[op_idx][:, :T_min]
        blended = (1 - d_mix) * base_contrib + d_mix * donor_contrib
        residual[:, :T_min] = residual[:, :T_min] - base_contrib + blended

    # Apply axis deltas scaled by safe_scale (slider=1 → 1σ of observed data)
    for axes_list, deltas in [
        (models.get('perceptual_axes', []), pca_deltas),
        (models.get('ica_axes', []), ica_deltas),
        (models.get('pitchbin_axes', []), pb_deltas),
    ]:
        for i, ax in enumerate(axes_list):
            if i < len(deltas) and deltas[i] != 0.0:
                scale = ax.get('safe_scale', 1.0)
                direction = ax['direction']  # [128]
                residual = residual + (deltas[i] * scale) * direction.unsqueeze(0).unsqueeze(0)

    # Mix: res_mix=0 → SMS only, res_mix=1 → default, res_mix=2 → exaggerated residual
    z_edited = z_sms_base + res_mix * residual

    # Z-space map: compare edited vs default reconstruction
    if use_gt and d.get('z_gt_full') is not None:
        z_original = d['z_gt_full'][:, :T]
    else:
        z_original = d['z_full'][:, :T]
    z_map = render_z_map(z_original, z_edited)

    z_audio = decode_z_to_audio(z_edited)

    # STFT-domain high-fidelity edit (always render when editor is loaded)
    # Use matching temporal region: GT mode uses cropped window, pred uses full
    stft_audio = None
    if use_gt and d.get('z_gt_flat_crop') is not None:
        original_audio = decode_z_to_audio(d['z_gt_flat_crop'][:, :T])
    else:
        original_audio = decode_z_to_audio(d['z_flat'][:, :T])
    has_editor = models.get('audio_editor') is not None
    if has_editor:
        try:
            any_pb_active = any(v != 0.0 for v in pb_deltas[:6])
            if any_pb_active:
                stft_audio = apply_stft_edits(original_audio, pb_deltas[:6])
            else:
                stft_audio = original_audio
        except Exception as e:
            print(f"STFT edit error: {e}")
            import traceback; traceback.print_exc()

    # STFT spectrogram comparison (original vs edited)
    stft_fig = render_stft_comparison(original_audio, stft_audio)

    return audio_sines, audio_noise, audio_sms_mixed, z_audio, stft_audio, z_map, stft_fig


def make_op_choices():
    """Build operation dropdown choices."""
    load_models()
    ranking = models.get('op_ranking', list(range(8)))
    choices = ["None"]
    for rank, k in enumerate(ranking[:6]):
        choices.append(f"Op {k} (rank {rank})")
    return choices


# ============================================================
# Build Gradio Interface
# ============================================================

CUSTOM_CSS = """
.decomposition-info { font-family: monospace; font-size: 13px; line-height: 1.5; }
.op-slider-group { padding: 4px 8px; margin: 1px 0; border-radius: 4px;
                    background: rgba(255,255,255,0.03); }
.donor-box { border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
             padding: 8px; margin: 3px 0; }

/* Both columns scroll independently within viewport height */
#controls-sidebar {
    max-height: 100vh;
    overflow-y: auto;
}
#map-panel {
    max-height: 100vh;
    overflow-y: auto;
}

/* Z-Space map stays pinned at top of right panel while rest scrolls */
#z-map {
    position: sticky;
    top: 0;
    z-index: 10;
    background: var(--background-fill-primary);
}

/* STFT audio highlight */
.stft-audio { border: 2px solid #4CAF50; border-radius: 8px; padding: 4px; }
"""

def build_ui():
    load_models()
    build_sample_catalog()

    categories = get_categories()
    default_cat = 'Piano' if 'Piano' in categories else categories[0]
    default_samples = get_samples_for_category(default_cat)
    n_ops = models['n_ops']
    ranking = models['op_ranking']
    op_choices = make_op_choices()
    n_pca_axes = len(models.get('perceptual_axes', []))
    n_ica_axes = len(models.get('ica_axes', []))
    n_pb_axes = len(models.get('pitchbin_axes', []))
    has_gt = 'gt_tree' in models

    with gr.Blocks(
        title="Operation Tree Explorer",
        theme=gr.themes.Soft(),
        css=CUSTOM_CSS,
    ) as demo:
        gr.Markdown("# Operation Tree Explorer")

        with gr.Row(equal_height=False, elem_id="main-row"):
            # ==========================================================
            # LEFT SIDEBAR — all controls (scrolls naturally)
            # ==========================================================
            with gr.Column(scale=2, min_width=320, elem_id="controls-sidebar"):

                # --- Sample Selection ---
                with gr.Group():
                    category_dd = gr.Dropdown(
                        choices=categories, value=default_cat,
                        label="Instrument Group",
                    )
                    sample_dd = gr.Dropdown(
                        choices=default_samples,
                        value=default_samples[0] if default_samples else None,
                        label="Sample",
                    )
                    load_btn = gr.Button("Load & Decompose", variant="primary")

                # --- Mode ---
                mode_choices = ["Pred SMS + Pred Tree"]
                if has_gt:
                    mode_choices.append("GT SMS + GT Tree")
                mode_radio = gr.Radio(
                    choices=mode_choices, value=mode_choices[0],
                    label="Mode",
                )

                # --- Residual Mix ---
                res_mix_slider = gr.Slider(
                    minimum=0, maximum=2, value=1.0, step=0.05,
                    label="Residual Mix (0=SMS only, 1=default, 2=exaggerated)",
                )

                # --- SMS Controls ---
                with gr.Accordion("SMS Controls", open=True):
                    sms_sliders = []
                    for g in range(6):
                        s = gr.Slider(
                            minimum=0, maximum=3, value=1.0, step=0.05,
                            label=f"Harm Group {g+1}",
                            elem_classes=["op-slider-group"],
                        )
                        sms_sliders.append(s)
                    sms_indep = gr.Slider(
                        minimum=0, maximum=3, value=1.0, step=0.05,
                        label="Indep Sines",
                        elem_classes=["op-slider-group"],
                    )
                    sms_noise = gr.Slider(
                        minimum=0, maximum=3, value=1.0, step=0.05,
                        label="Noise Bands",
                        elem_classes=["op-slider-group"],
                    )
                    sms_reset_btn = gr.Button("Reset SMS", variant="secondary", size="sm")

                # --- Residual Operations ---
                with gr.Accordion("Residual Operations", open=True):
                    gr.Markdown("1.0 = original, 0 = remove, 2 = double")
                    op_sliders = []
                    for rank in range(min(8, n_ops)):
                        k = ranking[rank] if rank < len(ranking) else rank
                        s = gr.Slider(
                            minimum=0, maximum=3, value=1.0, step=0.05,
                            label=f"Op {k} (rank {rank})",
                            elem_classes=["op-slider-group"],
                        )
                        op_sliders.append(s)
                    # Pad to n_ops if needed
                    while len(op_sliders) < n_ops:
                        op_sliders.append(gr.Slider(visible=False, value=1.0))

                    with gr.Row():
                        reset_btn = gr.Button("Reset Ops", variant="secondary", size="sm")
                    solo_btns = []
                    with gr.Row():
                        for rank in range(min(4, n_ops)):
                            k = ranking[rank] if rank < len(ranking) else rank
                            solo_btns.append(
                                gr.Button(f"Solo {k}", variant="secondary", size="sm")
                            )

                # --- Perceptual Axes ---
                pca_axis_sliders = []
                ica_axis_sliders = []
                pb_axis_sliders = []

                if n_pca_axes > 0 or n_ica_axes > 0 or n_pb_axes > 0:
                    with gr.Accordion("Perceptual Axes", open=True):

                        # Pitch-binned timbral axes (primary — cleanest method)
                        if n_pb_axes > 0:
                            gr.Markdown("**Timbral axes** (pitch-controlled)")
                            for i in range(min(8, n_pb_axes)):
                                ax = models['pitchbin_axes'][i]
                                var_pct = ax['variance_explained'] * 100
                                s = gr.Slider(
                                    minimum=-3, maximum=3, value=0.0, step=0.1,
                                    label=f"Timbre {i} ({var_pct:.1f}%)",
                                    elem_classes=["op-slider-group"],
                                )
                                pb_axis_sliders.append(s)

                        # Contrastive PCA axes
                        if n_pca_axes > 0:
                            with gr.Accordion("Contrastive Axes", open=False):
                                gr.Markdown("0 = original.")
                                for i in range(min(8, n_pca_axes)):
                                    ax = models['perceptual_axes'][i]
                                    var_pct = ax['variance_explained'] * 100
                                    s = gr.Slider(
                                        minimum=-3, maximum=3, value=0.0, step=0.1,
                                        label=f"Contr {i} ({var_pct:.1f}%)",
                                        elem_classes=["op-slider-group"],
                                    )
                                    pca_axis_sliders.append(s)

                        if n_ica_axes > 0:
                            with gr.Accordion("ICA Axes", open=False):
                                for i in range(min(6, n_ica_axes)):
                                    s = gr.Slider(
                                        minimum=-3, maximum=3, value=0.0, step=0.1,
                                        label=f"ICA {i}",
                                        elem_classes=["op-slider-group"],
                                    )
                                    ica_axis_sliders.append(s)

                        axis_reset_btn = gr.Button("Reset Axes", variant="secondary", size="sm")

                # --- Donors ---
                with gr.Accordion("Donor Samples", open=False):
                    gr.Markdown("Swap ops between samples")
                    donor_cats = []
                    donor_samples = []
                    donor_ops = []
                    donor_mixes = []
                    for i in range(3):
                        with gr.Group(elem_classes=["donor-box"]):
                            dc = gr.Dropdown(
                                choices=categories, value=None,
                                label=f"Donor {i+1} Cat",
                            )
                            ds = gr.Dropdown(
                                choices=[], value=None,
                                label=f"Donor {i+1}",
                            )
                            with gr.Row():
                                do = gr.Dropdown(
                                    choices=op_choices, value="None",
                                    label="Op", scale=1,
                                )
                                dm = gr.Slider(
                                    minimum=0, maximum=1, value=0, step=0.05,
                                    label="Mix", scale=1,
                                )
                            donor_cats.append(dc)
                            donor_samples.append(ds)
                            donor_ops.append(do)
                            donor_mixes.append(dm)

            # ==========================================================
            # RIGHT PANEL — maps on top, audio below
            # ==========================================================
            with gr.Column(scale=3, elem_id="map-panel"):

                # --- Z-Space Map (large) ---
                z_map_plot = gr.Plot(label="Z-Space Map (8x16 = 128 dims)", elem_id="z-map")

                # --- STFT Spectrogram Comparison ---
                stft_plot = gr.Plot(label="STFT Comparison (Original vs Edited)")

                # --- Render Button ---
                render_btn = gr.Button("Render", variant="primary")

                # --- Audio Outputs ---
                with gr.Row():
                    audio_edited = gr.Audio(label="Z-Space Edit", type="numpy")
                    audio_stft = gr.Audio(
                        label="STFT Edit (high-fidelity)",
                        type="numpy",
                        elem_classes=["stft-audio"],
                    )

                # --- SMS Direct Render ---
                with gr.Accordion("SMS Render", open=False):
                    with gr.Row():
                        audio_sms_sines = gr.Audio(label="Sines", type="numpy")
                        audio_sms_noise = gr.Audio(label="Noise", type="numpy")
                        audio_sms_mixed = gr.Audio(label="Mixed", type="numpy")

                # --- Decomposition Preview ---
                with gr.Accordion("Decomposition Preview", open=False):
                    gr.Markdown("**Pred SMS**")
                    with gr.Row():
                        audio_gt = gr.Audio(label="GT", type="numpy")
                        audio_pred_sms = gr.Audio(label="Pred SMS", type="numpy")
                        audio_pred_full = gr.Audio(label="SMS+Tree", type="numpy")
                        audio_raw_pred_sms = gr.Audio(label="Raw SMS", type="numpy")
                    gr.Markdown("**GT SMS**")
                    with gr.Row():
                        audio_gt_crop = gr.Audio(label="GT Crop", type="numpy")
                        audio_gt_sms = gr.Audio(label="GT SMS", type="numpy")
                        audio_gt_full = gr.Audio(label="GT+Tree", type="numpy")
                        audio_raw_gt_sms = gr.Audio(label="Raw GT", type="numpy")

                # --- Info ---
                decomp_info = gr.Markdown("*Select a sample and click Load*",
                                          elem_classes=["decomposition-info"])

        # ============================================================
        # Event wiring
        # ============================================================

        # Category change → update sample list
        category_dd.change(
            fn=on_category_change,
            inputs=[category_dd],
            outputs=[sample_dd],
        )

        # Wire donor category → sample dropdowns
        for dc, ds in zip(donor_cats, donor_samples):
            dc.change(
                fn=lambda cat: gr.update(
                    choices=get_samples_for_category(cat) if cat else [],
                    value=None
                ),
                inputs=[dc],
                outputs=[ds],
            )

        # Load button → decompose
        load_outputs = [audio_gt, audio_pred_sms, audio_pred_full, audio_raw_pred_sms,
                        audio_gt_crop, audio_gt_sms, audio_gt_full, audio_raw_gt_sms,
                        audio_stft,
                        decomp_info, z_map_plot] + op_sliders + pca_axis_sliders + ica_axis_sliders + pb_axis_sliders
        load_btn.click(
            fn=on_sample_select,
            inputs=[category_dd, sample_dd],
            outputs=load_outputs,
        )

        # Render inputs/outputs
        render_inputs = ([mode_radio, category_dd, sample_dd]
                         + [donor_cats[0], donor_samples[0], donor_ops[0], donor_mixes[0]]
                         + [donor_cats[1], donor_samples[1], donor_ops[1], donor_mixes[1]]
                         + [donor_cats[2], donor_samples[2], donor_ops[2], donor_mixes[2]]
                         + sms_sliders + [sms_indep, sms_noise, res_mix_slider]
                         + op_sliders + pca_axis_sliders + ica_axis_sliders + pb_axis_sliders)

        render_outputs = [audio_sms_sines, audio_sms_noise, audio_sms_mixed,
                          audio_edited, audio_stft, z_map_plot, stft_plot]

        render_btn.click(
            fn=render_edited_audio,
            inputs=render_inputs,
            outputs=render_outputs,
        )

        # Auto-render on slider changes
        for s in op_sliders + sms_sliders + [sms_indep, sms_noise, res_mix_slider] + pca_axis_sliders + ica_axis_sliders + pb_axis_sliders:
            s.release(
                fn=render_edited_audio,
                inputs=render_inputs,
                outputs=render_outputs,
            )

        # Auto-render on mode change
        mode_radio.change(
            fn=render_edited_audio,
            inputs=render_inputs,
            outputs=render_outputs,
        )

        # Auto-render on donor changes
        for dm in donor_mixes:
            dm.release(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)
        for do in donor_ops:
            do.change(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)
        for ds in donor_samples:
            ds.change(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)

        # Reset buttons
        def reset_op_scales():
            return [1.0] * n_ops
        reset_btn.click(
            fn=reset_op_scales, outputs=op_sliders
        ).then(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)

        def reset_sms_scales():
            return [1.0] * 6 + [1.0, 1.0]
        sms_reset_btn.click(
            fn=reset_sms_scales, outputs=sms_sliders + [sms_indep, sms_noise]
        ).then(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)

        # Solo buttons
        for btn_idx, btn in enumerate(solo_btns):
            def make_solo(rank):
                def solo_fn():
                    scales = [0.0] * n_ops
                    k = ranking[rank] if rank < len(ranking) else rank
                    scales[k] = 1.0
                    return scales
                return solo_fn
            btn.click(
                fn=make_solo(btn_idx), outputs=op_sliders
            ).then(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)

        # Reset axes
        if n_pca_axes > 0 or n_ica_axes > 0 or n_pb_axes > 0:
            def reset_axes():
                return ([0.0] * len(pca_axis_sliders)
                        + [0.0] * len(ica_axis_sliders)
                        + [0.0] * len(pb_axis_sliders))
            axis_reset_btn.click(
                fn=reset_axes, outputs=pca_axis_sliders + ica_axis_sliders + pb_axis_sliders
            ).then(fn=render_edited_audio, inputs=render_inputs, outputs=render_outputs)

    return demo


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=8098,
        root_path="/do",
        share=False,
    )
