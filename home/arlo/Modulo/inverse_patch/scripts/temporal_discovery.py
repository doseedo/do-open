#!/usr/bin/env python3
"""
Temporal Discovery: find perceptual axes from within-track z-variation.

Key insight: within a single recording, instrument identity is CONSTANT.
The only things that vary are pitch, dynamics, and timbral qualities
(breathiness, vibrato depth, articulation, growl, etc.).

After centering (removes instrument identity) and regressing out pitch,
the remaining frame-to-frame variation = perceptual timbral qualities.

This requires zero labels and naturally controls for instrument identity.
"""

import sys
import torch
import numpy as np
import orjson
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM

MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
CONTRASTIVE_AXES_PATH = SCRIPT_DIR.parent / "test_outputs" / "contrastive_discovery" / "discovered_axes.pt"
PROBE_PATH = SCRIPT_DIR.parent / "test_outputs" / "z_probes" / "probe_results.pt"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "temporal_discovery"

Z_DIM = 128
MIN_FRAMES = 80       # ~7.4 seconds minimum
MAX_FRAMES = 1000     # cap per track to avoid dominating PCA
MAX_TRACKS = 150

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


def load_codec():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    codec = BidirectionalCodec(sms_dim=SMS_DIM, z_dim=Z_DIM, g_hidden=384, f_hidden=256)
    codec.load_state_dict(torch.load(PHASE1_PATH, weights_only=True, map_location='cpu'))
    codec = codec.to(device).eval()
    return codec, device


def categorize(path_str):
    lp = path_str.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in lp for kw in keywords):
            return cat
    return 'Other'


def get_unique_latent_paths():
    """Get unique latent file paths from manifest."""
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    seen = set()
    paths = []
    for entry in manifest['entries']:
        lp = entry.get('latent_path', '')
        if lp and lp not in seen:
            seen.add(lp)
            paths.append({
                'path': lp,
                'category': categorize(lp),
                'name': Path(lp).stem,
            })
    return paths


def load_track(path):
    """Load latent file → [T, 128] or None."""
    try:
        d = torch.load(path, map_location='cpu', weights_only=False)
        if isinstance(d, dict) and 'latents' in d:
            z = d['latents']
        elif isinstance(d, torch.Tensor):
            z = d
        else:
            return None
        if z.dim() != 3 or z.shape[0] != 8 or z.shape[1] != 16:
            return None
        T = z.shape[2]
        if T < MIN_FRAMES:
            return None
        # Flatten: [8, 16, T] → [T, 128]
        z_flat = z.permute(2, 0, 1).reshape(T, Z_DIM).float()
        if T > MAX_FRAMES:
            z_flat = z_flat[:MAX_FRAMES]
        return z_flat
    except Exception:
        return None


def extract_pitch_features(z_flat, codec, device):
    """Run F(z) → SMS params → group f0s as pitch features. Returns [T, 6]."""
    with torch.no_grad():
        sms = codec.forward_F(z_flat.unsqueeze(0).to(device))  # [1, T, 102]
    sms = sms.squeeze(0).cpu().numpy()  # [T, 102]
    # Group f0s are SMS dims 0-5 (normalized log frequency)
    return sms[:, :6]


def main():
    print("=" * 60)
    print("TEMPORAL DISCOVERY OF PERCEPTUAL AXES")
    print("Within-track variation after pitch removal")
    print("=" * 60)
    print()
    print("Method:")
    print("  1. Load full-length tracks (not 3s chunks)")
    print("  2. Center per track (removes instrument identity)")
    print("  3. Regress out pitch via F(z) → f0")
    print("  4. PCA on remaining = timbral variation")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load codec
    print("Loading Phase 1 codec...")
    codec, device = load_codec()

    # Get latent paths
    print("Scanning manifest...")
    all_paths = get_unique_latent_paths()
    print(f"  {len(all_paths)} unique latent paths")

    # Load tracks with sufficient length
    print(f"\nLoading tracks (min {MIN_FRAMES} frames = {MIN_FRAMES/10.8:.1f}s, "
          f"max {MAX_TRACKS} tracks)...")
    tracks = []
    skipped_short = 0
    skipped_err = 0

    for info in all_paths:
        if len(tracks) >= MAX_TRACKS:
            break
        z_flat = load_track(info['path'])
        if z_flat is None:
            skipped_short += 1
            continue
        tracks.append({
            'z_flat': z_flat,
            'name': info['name'],
            'category': info['category'],
            'n_frames': len(z_flat),
        })
        if len(tracks) % 25 == 0:
            dur = z_flat.shape[0] / 10.8
            print(f"  Loaded {len(tracks)} tracks (last: {info['name'][:40]}, "
                  f"{z_flat.shape[0]} frames = {dur:.1f}s)")

    print(f"\n  Loaded: {len(tracks)} tracks, skipped: {skipped_short} (too short/error)")

    # Per-category stats
    cat_counts = defaultdict(int)
    cat_frames = defaultdict(int)
    for t in tracks:
        cat_counts[t['category']] += 1
        cat_frames[t['category']] += t['n_frames']

    total_frames = sum(t['n_frames'] for t in tracks)
    total_dur = total_frames / 10.8
    print(f"  Total frames: {total_frames} ({total_dur:.0f}s = {total_dur/60:.1f} min)")
    for cat in sorted(cat_counts.keys(), key=lambda c: -cat_frames[c]):
        print(f"    {cat:>10s}: {cat_counts[cat]:3d} tracks, "
              f"{cat_frames[cat]:6d} frames ({cat_frames[cat]/10.8:.0f}s)")

    # ============================================================
    # Process each track: center → extract pitch → regress out
    # ============================================================
    print("\n" + "=" * 60)
    print("PROCESSING: center, extract pitch, regress out")
    print("=" * 60)

    all_residuals = []
    track_stats = []

    for i, track in enumerate(tracks):
        z = track['z_flat']  # [T, 128]
        T = len(z)

        # 1. Center (remove mean = instrument/mic/room identity)
        z_mean = z.mean(0, keepdim=True)
        z_centered = (z - z_mean).numpy()

        # 2. Extract pitch features via codec F()
        f0_features = extract_pitch_features(z, codec, device)  # [T, 6]

        # 3. Regress out pitch (only f0 channels that actually vary)
        f0_std = f0_features.std(axis=0)
        active = f0_std > 0.001

        if active.any():
            f0_active = f0_features[:, active]
            ridge = Ridge(alpha=1.0)
            ridge.fit(f0_active, z_centered)
            pitch_r2 = ridge.score(f0_active, z_centered)
            residual = z_centered - ridge.predict(f0_active)
        else:
            # No pitch variation (sustained note, drums, etc.)
            pitch_r2 = 0.0
            residual = z_centered

        all_residuals.append(residual.astype(np.float32))
        track_stats.append({
            'name': track['name'],
            'category': track['category'],
            'n_frames': T,
            'pitch_r2': float(pitch_r2),
            'z_var': float(np.var(z_centered)),
            'residual_var': float(np.var(residual)),
            'n_active_f0': int(active.sum()),
        })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tracks)} — last pitch R²={pitch_r2:.4f}")

    # Pitch regression summary
    r2_vals = [s['pitch_r2'] for s in track_stats]
    print(f"\n  Pitch regression R²:")
    print(f"    mean={np.mean(r2_vals):.4f}, median={np.median(r2_vals):.4f}, "
          f"max={max(r2_vals):.4f}")
    print(f"    Tracks where pitch explains >10% variance: "
          f"{sum(1 for r in r2_vals if r > 0.1)}/{len(tracks)}")
    print(f"    Tracks with no pitch variation: "
          f"{sum(1 for s in track_stats if s['n_active_f0'] == 0)}/{len(tracks)}")

    var_ratio = np.mean([s['residual_var'] / (s['z_var'] + 1e-10) for s in track_stats])
    print(f"    Mean residual/total variance ratio: {var_ratio:.4f}")

    # ============================================================
    # PCA on pooled timbral residuals
    # ============================================================
    print("\n" + "=" * 60)
    print("PCA ON POOLED TIMBRAL RESIDUALS")
    print("=" * 60)

    pooled = np.concatenate(all_residuals)  # [total_frames, 128]
    print(f"\n  Pooled: {pooled.shape[0]} frames × {pooled.shape[1]} dims")

    scaler = StandardScaler()
    pooled_scaled = scaler.fit_transform(pooled)

    pca = PCA(n_components=min(16, pooled_scaled.shape[1]))
    pca.fit(pooled_scaled)

    print(f"\n  Temporal timbral axes:")
    print(f"  {'Axis':>6s}  {'Var':>8s}  {'Cum':>8s}  {'Top z-dims'}")
    print("  " + "-" * 80)

    axes = []
    for k in range(min(8, len(pca.components_))):
        var = pca.explained_variance_ratio_[k]
        cum = pca.explained_variance_ratio_[:k + 1].sum()

        # Back to raw z-space
        direction = pca.components_[k] / (scaler.scale_ + 1e-10)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        top_idx = np.argsort(np.abs(direction))[::-1][:5]
        top_str = ", ".join([f"z[{i}]={direction[i]:+.3f}" for i in top_idx])

        print(f"  Axis {k:2d}  {var:8.4f}  {cum:8.4f}  {top_str}")

        axes.append({
            'direction': direction,
            'variance_explained': float(var),
        })

    # ============================================================
    # Validation: axes should NOT correlate with pitch
    # ============================================================
    print("\n" + "=" * 60)
    print("VALIDATION: AXIS-PITCH CORRELATION (should be ~0)")
    print("=" * 60)

    pitch_corrs = defaultdict(list)
    for i, track in enumerate(tracks):
        z = track['z_flat'].numpy()
        z_centered = z - z.mean(0, keepdims=True)

        f0_raw = extract_pitch_features(track['z_flat'], codec, device)
        f0_mean = f0_raw.mean(axis=1)  # [T] average f0 across groups

        if np.std(f0_mean) < 0.001:
            continue

        for k in range(min(6, len(axes))):
            proj = z_centered @ axes[k]['direction']
            corr = np.corrcoef(proj, f0_mean)[0, 1]
            if not np.isnan(corr):
                pitch_corrs[k].append(abs(corr))

    print("\n  Mean |correlation| with pitch (lower = better):")
    for k in range(min(6, len(axes))):
        if pitch_corrs[k]:
            mc = np.mean(pitch_corrs[k])
            print(f"    Axis {k}: {mc:.4f}  {'OK' if mc < 0.15 else 'WARNING: pitch leak'}")

    # ============================================================
    # Compare with contrastive discovery axes
    # ============================================================
    if CONTRASTIVE_AXES_PATH.exists():
        print("\n" + "=" * 60)
        print("TEMPORAL vs CONTRASTIVE AXES")
        print("=" * 60)

        contrastive = torch.load(CONTRASTIVE_AXES_PATH, weights_only=False, map_location='cpu')

        n_c = min(6, len(contrastive['within_pca']))
        n_t = min(6, len(axes))

        print(f"\n  Cosine similarity (temporal rows × contrastive cols):")
        header = "            " + "".join([f"  Contr {j}" for j in range(n_c)])
        print(header)

        for i in range(n_t):
            td = axes[i]['direction']
            row = f"  Temp {i:2d}   "
            for j in range(n_c):
                cd = contrastive['within_pca'][j]['direction']
                cos = abs(np.dot(td, cd))
                marker = "*" if cos > 0.3 else " "
                row += f"  {cos:.3f}{marker}"
            print(row)

        # Vs instrument identity axes
        n_b = min(4, len(contrastive.get('between_pca', [])))
        if n_b > 0:
            print(f"\n  Temporal axes vs instrument identity axes:")
            header = "            " + "".join([f"   Inst {j}" for j in range(n_b)])
            print(header)

            for i in range(n_t):
                td = axes[i]['direction']
                row = f"  Temp {i:2d}   "
                for j in range(n_b):
                    cd = contrastive['between_pca'][j]['direction']
                    cos = abs(np.dot(td, cd))
                    marker = "*" if cos > 0.3 else " "
                    row += f"  {cos:.3f}{marker}"
                print(row)

    # ============================================================
    # Per-instrument consistency
    # ============================================================
    print("\n" + "=" * 60)
    print("PER-INSTRUMENT CONSISTENCY")
    print("(do axes capture similar variation across instruments?)")
    print("=" * 60)

    axis_dirs = np.stack([a['direction'] for a in axes[:6]])  # [6, 128]

    for cat in sorted(cat_counts.keys(), key=lambda c: -cat_counts[c]):
        if cat_counts[cat] < 3:
            continue
        cat_residuals = [all_residuals[i] for i, t in enumerate(tracks) if t['category'] == cat]
        cat_pooled = np.concatenate(cat_residuals)

        projections = cat_pooled @ axis_dirs.T  # [frames, 6]
        variances = np.var(projections, axis=0)
        var_str = "  ".join([f"{v:.3f}" for v in variances])
        print(f"  {cat:>10s} ({cat_counts[cat]:2d} tracks, {len(cat_pooled):5d} fr): [{var_str}]")

    # ============================================================
    # Validate with supervised probes (if available)
    # ============================================================
    if PROBE_PATH.exists():
        print("\n" + "=" * 60)
        print("TEMPORAL AXES vs SUPERVISED PROBES")
        print("=" * 60)

        probes = torch.load(PROBE_PATH, weights_only=False, map_location='cpu')
        probe_names = list(probes.keys()) if isinstance(probes, dict) else []

        if probe_names:
            header = "            " + "".join([f"  {n[:8]:>8s}" for n in probe_names])
            print(header)

            for i in range(min(6, len(axes))):
                td = axes[i]['direction']
                row = f"  Temp {i:2d}   "
                for name in probe_names:
                    pw = probes[name]
                    if isinstance(pw, dict) and 'weight' in pw:
                        pw = pw['weight']
                    if isinstance(pw, np.ndarray) and pw.shape[-1] == 128:
                        pw_flat = pw.flatten()[:128]
                        pw_norm = pw_flat / (np.linalg.norm(pw_flat) + 1e-10)
                        cos = abs(np.dot(td, pw_norm))
                        row += f"  {cos:8.3f}"
                    else:
                        row += f"       -"
                print(row)

    # ============================================================
    # Save
    # ============================================================
    save_data = {
        'temporal_pca': [{
            'direction': ax['direction'],
            'variance_explained': ax['variance_explained'],
        } for ax in axes],
        'n_tracks': len(tracks),
        'n_total_frames': len(pooled),
        'track_stats': track_stats,
        'pitch_regression_mean_r2': float(np.mean(r2_vals)),
    }

    save_path = OUTPUT_DIR / "temporal_axes.pt"
    torch.save(save_data, save_path)
    print(f"\n  Saved to {save_path}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_var = sum(ax['variance_explained'] for ax in axes[:6])
    print(f"\n  {len(tracks)} tracks, {total_frames} frames ({total_dur/60:.1f} min audio)")
    print(f"  Top 6 temporal axes explain {total_var:.1%} of timbral variation")
    print(f"  Pitch regression removed {np.mean(r2_vals):.1%} of within-track variance (mean)")
    print()
    print("  Key advantages over contrastive approach:")
    print("    - Instrument identity naturally controlled (same track)")
    print("    - Pitch explicitly regressed out")
    print("    - Uses full-length recordings (10-200s), not 3s chunks")
    print("    - Captures dynamic timbral qualities that fluctuate over time")
    print()
    print("  Next: add temporal axes to Gradio UI alongside contrastive axes")


if __name__ == '__main__':
    main()
