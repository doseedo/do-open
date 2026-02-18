#!/usr/bin/env python3
"""
Pitch-Binned Discovery: find timbral axes from same-pitch z-variation.

Key insight: within a single track, group frames by pitch (0.5 semitone bins).
Frames at the SAME pitch on the SAME instrument have ZERO pitch/identity variation.
Any remaining z-variation is purely timbral: dynamics, breathiness, vibrato,
articulation, growl, etc.

No pitch subtraction needed — pitch is controlled by binning.
No labels needed — just observing what varies.
No cross-instrument contamination — within one track.

Then check:
  1. Are axes consistent across pitch bins within a track? (should be)
  2. Are axes consistent across tracks of same instrument type? (partially)
  3. Are any axes universal across all instruments? (dynamics probably)
"""

import sys
import torch
import numpy as np
import orjson
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from bidirectional_sms_z import BidirectionalCodec, SMS_DIM

MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
PHASE1_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "bidirectional_codec.pt"
CONTRASTIVE_AXES_PATH = SCRIPT_DIR.parent / "test_outputs" / "contrastive_discovery" / "discovered_axes.pt"
TEMPORAL_AXES_PATH = SCRIPT_DIR.parent / "test_outputs" / "temporal_discovery" / "temporal_axes.pt"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "pitchbin_discovery"

Z_DIM = 128
MIN_FRAMES = 80        # ~7.4 seconds minimum track length
MAX_FRAMES = 1000      # cap per track
MAX_TRACKS = 200
PITCH_BIN_SEMITONES = 0.5   # bin resolution
MIN_FRAMES_PER_BIN = 10     # need enough frames per bin for meaningful variance
FPS = 10.8                   # DCAE frame rate

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
        z_flat = z.permute(2, 0, 1).reshape(T, Z_DIM).float()
        if T > MAX_FRAMES:
            z_flat = z_flat[:MAX_FRAMES]
        return z_flat
    except Exception:
        return None


def extract_pitch_per_frame(z_flat, codec, device):
    """Run F(z) → SMS → mean group f0 per frame. Returns [T] in normalized log-freq."""
    with torch.no_grad():
        sms = codec.forward_F(z_flat.unsqueeze(0).to(device))  # [1, T, 102]
    sms = sms.squeeze(0).cpu().numpy()  # [T, 102]
    # Group f0s are SMS dims 0-5, take mean across groups
    f0_groups = sms[:, :6]  # [T, 6]
    # Use the dominant f0 (highest amplitude group) for binning
    group_amps = sms[:, 6:54].reshape(-1, 6, 8)  # [T, 6groups, 8harmonics]
    group_energy = (group_amps ** 2).sum(axis=2)  # [T, 6]
    dominant_group = group_energy.argmax(axis=1)  # [T]
    f0_dominant = f0_groups[np.arange(len(f0_groups)), dominant_group]  # [T]
    return f0_dominant, f0_groups


def f0_to_semitone_bin(f0_values, bin_size=PITCH_BIN_SEMITONES):
    """Convert normalized log-freq values to semitone bins.

    SMS f0 is normalized log frequency. We quantize to semitone bins.
    The exact scale doesn't matter for binning — we just need same-pitch frames
    to land in the same bin.
    """
    # Scale to semitones (the codec normalizes 0-1 over its frequency range)
    # Approximate: SMS f0 range covers ~6 octaves = 72 semitones
    semitones = f0_values * 72.0  # rough mapping
    bins = np.round(semitones / bin_size).astype(int)
    return bins


def compute_within_bin_residuals(z_flat_np, pitch_bins, min_per_bin=MIN_FRAMES_PER_BIN):
    """For each pitch bin with enough frames, center z and collect residuals.

    Returns:
        residuals: [N, 128] array of pitch-controlled timbral residuals
        bin_info: list of (bin_id, n_frames, variance) for diagnostics
    """
    residuals = []
    bin_info = []

    unique_bins = np.unique(pitch_bins)
    for b in unique_bins:
        mask = pitch_bins == b
        n = mask.sum()
        if n < min_per_bin:
            continue

        z_bin = z_flat_np[mask]  # [n, 128]
        z_bin_centered = z_bin - z_bin.mean(axis=0, keepdims=True)

        residuals.append(z_bin_centered)
        bin_info.append({
            'bin': int(b),
            'n_frames': int(n),
            'variance': float(np.var(z_bin_centered)),
        })

    if residuals:
        return np.concatenate(residuals), bin_info
    return np.zeros((0, Z_DIM), dtype=np.float32), []


def main():
    print("=" * 60)
    print("PITCH-BINNED DISCOVERY OF TIMBRAL AXES")
    print("Same-pitch frames → z-variation = pure timbre")
    print("=" * 60)
    print()
    print("Method:")
    print(f"  1. Load full-length tracks (≥{MIN_FRAMES} frames)")
    print(f"  2. Extract pitch per frame via F(z) → SMS f0")
    print(f"  3. Bin frames by pitch ({PITCH_BIN_SEMITONES} semitone resolution)")
    print(f"  4. Within each bin (≥{MIN_FRAMES_PER_BIN} frames): center z")
    print("  5. PCA on pooled bin-centered residuals = timbral axes")
    print("  6. Check consistency: within-track, per-instrument, universal")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load codec
    print("Loading Phase 1 codec...")
    codec, device = load_codec()

    # Get latent paths
    print("Scanning manifest...")
    all_paths = get_unique_latent_paths()
    print(f"  {len(all_paths)} unique latent paths")

    # Load tracks
    print(f"\nLoading tracks (min {MIN_FRAMES} frames = {MIN_FRAMES/FPS:.1f}s, "
          f"max {MAX_TRACKS} tracks)...")
    tracks = []
    skipped = 0

    for info in all_paths:
        if len(tracks) >= MAX_TRACKS:
            break
        z_flat = load_track(info['path'])
        if z_flat is None:
            skipped += 1
            continue
        tracks.append({
            'z_flat': z_flat,
            'name': info['name'],
            'category': info['category'],
            'n_frames': len(z_flat),
        })
        if len(tracks) % 25 == 0:
            dur = z_flat.shape[0] / FPS
            print(f"  Loaded {len(tracks)} tracks (last: {info['name'][:40]}, "
                  f"{z_flat.shape[0]} frames = {dur:.1f}s)")

    print(f"\n  Loaded: {len(tracks)} tracks, skipped: {skipped}")

    cat_counts = defaultdict(int)
    cat_frames = defaultdict(int)
    for t in tracks:
        cat_counts[t['category']] += 1
        cat_frames[t['category']] += t['n_frames']

    total_frames = sum(t['n_frames'] for t in tracks)
    total_dur = total_frames / FPS
    print(f"  Total frames: {total_frames} ({total_dur:.0f}s = {total_dur/60:.1f} min)")
    for cat in sorted(cat_counts.keys(), key=lambda c: -cat_frames[c]):
        print(f"    {cat:>10s}: {cat_counts[cat]:3d} tracks, "
              f"{cat_frames[cat]:6d} frames ({cat_frames[cat]/FPS:.0f}s)")

    # ============================================================
    # Process each track: extract pitch → bin → collect residuals
    # ============================================================
    print("\n" + "=" * 60)
    print("PITCH BINNING & RESIDUAL COLLECTION")
    print("=" * 60)

    all_residuals = []         # global pool
    per_track_residuals = []   # per-track for consistency analysis
    per_track_axes = []        # per-track PCA axes (if enough data)
    track_stats = []

    total_bins_used = 0
    total_frames_in_bins = 0

    for i, track in enumerate(tracks):
        z = track['z_flat']  # [T, 128]
        z_np = z.numpy()

        # Extract pitch
        f0_dominant, f0_groups = extract_pitch_per_frame(z, codec, device)
        pitch_bins = f0_to_semitone_bin(f0_dominant)

        # Collect within-bin residuals
        residuals, bin_info = compute_within_bin_residuals(z_np, pitch_bins)

        n_bins = len(bin_info)
        n_res_frames = len(residuals)

        track_stats.append({
            'name': track['name'],
            'category': track['category'],
            'n_frames': track['n_frames'],
            'n_pitch_bins': n_bins,
            'n_usable_frames': n_res_frames,
            'pct_usable': n_res_frames / track['n_frames'] * 100 if track['n_frames'] > 0 else 0,
            'mean_bin_size': np.mean([b['n_frames'] for b in bin_info]) if bin_info else 0,
            'residual_var': float(np.var(residuals)) if n_res_frames > 0 else 0,
        })

        total_bins_used += n_bins
        total_frames_in_bins += n_res_frames

        if n_res_frames > 0:
            all_residuals.append(residuals)
            per_track_residuals.append({
                'residuals': residuals,
                'name': track['name'],
                'category': track['category'],
            })

            # Per-track PCA (if enough frames)
            if n_res_frames >= 50:
                track_pca = PCA(n_components=min(8, n_res_frames - 1, Z_DIM))
                track_pca.fit(residuals)
                per_track_axes.append({
                    'axes': track_pca.components_[:6].copy(),
                    'var_explained': track_pca.explained_variance_ratio_[:6].copy(),
                    'name': track['name'],
                    'category': track['category'],
                    'n_frames': n_res_frames,
                })

        if (i + 1) % 25 == 0:
            pct = track_stats[-1]['pct_usable']
            print(f"  {i+1}/{len(tracks)} — bins: {n_bins}, "
                  f"usable: {n_res_frames}/{track['n_frames']} ({pct:.0f}%)")

    # Stats
    usable_pcts = [s['pct_usable'] for s in track_stats]
    print(f"\n  Frame utilization:")
    print(f"    Total bins used: {total_bins_used} across {len(tracks)} tracks")
    print(f"    Total frames in bins: {total_frames_in_bins}/{total_frames} "
          f"({total_frames_in_bins/total_frames*100:.1f}%)")
    print(f"    Mean per-track usable: {np.mean(usable_pcts):.1f}%")
    print(f"    Tracks with 0 usable bins: {sum(1 for p in usable_pcts if p == 0)}")
    print(f"    Mean bin size: {total_frames_in_bins/max(total_bins_used,1):.1f} frames")

    # ============================================================
    # GLOBAL PCA on pooled pitch-controlled residuals
    # ============================================================
    print("\n" + "=" * 60)
    print("GLOBAL PCA ON PITCH-CONTROLLED TIMBRAL RESIDUALS")
    print("=" * 60)

    pooled = np.concatenate(all_residuals).astype(np.float32)
    print(f"\n  Pooled: {pooled.shape[0]} frames × {pooled.shape[1]} dims")

    scaler = StandardScaler()
    pooled_scaled = scaler.fit_transform(pooled)

    pca = PCA(n_components=min(16, pooled_scaled.shape[1]))
    pca.fit(pooled_scaled)

    print(f"\n  Pitch-controlled timbral axes:")
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
    # VALIDATION 1: pitch correlation (should be ~0 by construction)
    # ============================================================
    print("\n" + "=" * 60)
    print("VALIDATION: AXIS-PITCH CORRELATION")
    print("(should be ~0 — pitch is controlled by binning)")
    print("=" * 60)

    pitch_corrs = defaultdict(list)
    for i, track in enumerate(tracks):
        z = track['z_flat'].numpy()
        f0_dominant, _ = extract_pitch_per_frame(track['z_flat'], codec, device)

        if np.std(f0_dominant) < 0.001:
            continue

        for k in range(min(6, len(axes))):
            proj = z @ axes[k]['direction']
            # Correlation within this track (not across tracks)
            corr = np.corrcoef(proj, f0_dominant)[0, 1]
            if not np.isnan(corr):
                pitch_corrs[k].append(abs(corr))

    print("\n  Mean |correlation| with pitch (lower = better):")
    for k in range(min(6, len(axes))):
        if pitch_corrs[k]:
            mc = np.mean(pitch_corrs[k])
            status = 'CLEAN' if mc < 0.15 else ('OK' if mc < 0.25 else 'pitch leak')
            print(f"    Axis {k}: {mc:.4f}  {status}")

    # ============================================================
    # VALIDATION 2: per-track axis consistency
    # (do different tracks of same instrument find similar axes?)
    # ============================================================
    print("\n" + "=" * 60)
    print("PER-TRACK AXIS CONSISTENCY")
    print("(within same instrument type, do local PCA axes align?)")
    print("=" * 60)

    # Group per-track axes by category
    cat_axes = defaultdict(list)
    for ta in per_track_axes:
        cat_axes[ta['category']].append(ta)

    global_dirs = np.stack([a['direction'] for a in axes[:6]])  # [6, 128]

    for cat in sorted(cat_axes.keys(), key=lambda c: -len(cat_axes[c])):
        entries = cat_axes[cat]
        if len(entries) < 3:
            continue

        print(f"\n  {cat} ({len(entries)} tracks with ≥50 usable frames):")

        # How well does each track's top axis align with global axes?
        alignments = []
        for ta in entries:
            track_ax0 = ta['axes'][0]  # top local axis
            track_ax0 = track_ax0 / (np.linalg.norm(track_ax0) + 1e-10)
            cos_with_global = np.abs(global_dirs @ track_ax0)  # [6]
            alignments.append(cos_with_global)

        alignments = np.stack(alignments)  # [n_tracks, 6]
        mean_align = alignments.mean(axis=0)
        print(f"    Track axis 0 alignment with global axes: "
              f"[{', '.join(f'{v:.3f}' for v in mean_align)}]")

        # Cross-track axis alignment: do tracks agree with each other?
        if len(entries) >= 4:
            cross_sims = []
            for j in range(len(entries)):
                for k in range(j + 1, len(entries)):
                    ax_j = entries[j]['axes'][0]
                    ax_k = entries[k]['axes'][0]
                    ax_j = ax_j / (np.linalg.norm(ax_j) + 1e-10)
                    ax_k = ax_k / (np.linalg.norm(ax_k) + 1e-10)
                    cross_sims.append(abs(np.dot(ax_j, ax_k)))
            print(f"    Cross-track axis 0 similarity: "
                  f"mean={np.mean(cross_sims):.3f}, "
                  f"std={np.std(cross_sims):.3f}, "
                  f"max={np.max(cross_sims):.3f}")

    # ============================================================
    # VALIDATION 3: per-instrument variance on global axes
    # ============================================================
    print("\n" + "=" * 60)
    print("PER-INSTRUMENT VARIANCE ON GLOBAL AXES")
    print("(how much each instrument varies along each axis)")
    print("=" * 60)

    for cat in sorted(cat_counts.keys(), key=lambda c: -cat_counts[c]):
        if cat_counts[cat] < 3:
            continue
        cat_res = [pr['residuals'] for pr in per_track_residuals if pr['category'] == cat]
        if not cat_res:
            continue
        cat_pooled = np.concatenate(cat_res)
        projections = cat_pooled @ global_dirs.T  # [frames, 6]
        variances = np.var(projections, axis=0)
        var_str = "  ".join([f"{v:.3f}" for v in variances])
        n_fr = len(cat_pooled)
        print(f"  {cat:>10s} ({cat_counts[cat]:2d} trks, {n_fr:5d} fr): [{var_str}]")

    # ============================================================
    # Compare with previous discovery methods
    # ============================================================
    print("\n" + "=" * 60)
    print("COMPARISON WITH PREVIOUS METHODS")
    print("=" * 60)

    if CONTRASTIVE_AXES_PATH.exists():
        contrastive = torch.load(CONTRASTIVE_AXES_PATH, weights_only=False, map_location='cpu')
        n_c = min(6, len(contrastive['within_pca']))
        n_pb = min(6, len(axes))

        print(f"\n  vs Contrastive axes (cosine similarity):")
        header = "              " + "".join([f" Contr {j}" for j in range(n_c)])
        print(header)
        for i in range(n_pb):
            td = axes[i]['direction']
            row = f"  PitchBin {i}: "
            for j in range(n_c):
                cd = contrastive['within_pca'][j]['direction']
                cos = abs(np.dot(td, cd))
                marker = "*" if cos > 0.3 else " "
                row += f"  {cos:.3f}{marker}"
            print(row)

        # vs instrument identity
        n_b = min(4, len(contrastive.get('between_pca', [])))
        if n_b > 0:
            print(f"\n  vs Instrument identity axes:")
            header = "              " + "".join([f"  Inst {j}" for j in range(n_b)])
            print(header)
            for i in range(n_pb):
                td = axes[i]['direction']
                row = f"  PitchBin {i}: "
                for j in range(n_b):
                    cd = contrastive['between_pca'][j]['direction']
                    cos = abs(np.dot(td, cd))
                    marker = "*" if cos > 0.3 else " "
                    row += f"  {cos:.3f}{marker}"
                print(row)

    if TEMPORAL_AXES_PATH.exists():
        temporal = torch.load(TEMPORAL_AXES_PATH, weights_only=False, map_location='cpu')
        n_t = min(6, len(temporal['temporal_pca']))
        n_pb = min(6, len(axes))

        print(f"\n  vs Temporal axes (cosine similarity):")
        header = "              " + "".join([f" Temp {j} " for j in range(n_t)])
        print(header)
        for i in range(n_pb):
            td = axes[i]['direction']
            row = f"  PitchBin {i}: "
            for j in range(n_t):
                cd = temporal['temporal_pca'][j]['direction']
                cos = abs(np.dot(td, cd))
                marker = "*" if cos > 0.3 else " "
                row += f"  {cos:.3f}{marker}"
            print(row)

    # ============================================================
    # Save
    # ============================================================
    save_data = {
        'pitchbin_pca': [{
            'direction': ax['direction'],
            'variance_explained': ax['variance_explained'],
        } for ax in axes],
        'n_tracks': len(tracks),
        'n_total_frames': len(pooled),
        'n_bins_used': total_bins_used,
        'pitch_bin_semitones': PITCH_BIN_SEMITONES,
        'min_frames_per_bin': MIN_FRAMES_PER_BIN,
        'track_stats': track_stats,
        'per_track_axes': [{
            'axes': ta['axes'],
            'var_explained': ta['var_explained'],
            'name': ta['name'],
            'category': ta['category'],
        } for ta in per_track_axes],
        'scaler_mean': scaler.mean_,
        'scaler_scale': scaler.scale_,
    }

    save_path = OUTPUT_DIR / "pitchbin_axes.pt"
    torch.save(save_data, save_path)
    print(f"\n  Saved to {save_path}")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_var = sum(ax['variance_explained'] for ax in axes[:6])
    print(f"\n  {len(tracks)} tracks, {total_frames_in_bins} pitch-controlled frames")
    print(f"  {total_bins_used} pitch bins ({PITCH_BIN_SEMITONES} semitone resolution)")
    print(f"  Top 6 axes explain {total_var:.1%} of within-bin timbral variation")
    print()
    print("  Advantages:")
    print("    - Pitch controlled by construction, not regression")
    print("    - No pitch leakage possible (same pitch in same bin)")
    print("    - Instrument identity constant (same track)")
    print("    - Captures per-instrument timbral invariances")


if __name__ == '__main__':
    main()
