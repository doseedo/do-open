#!/usr/bin/env python3
"""
Contrastive Discovery: find perceptual axes in z-space WITHOUT prescription.

Key insight: group samples by (instrument, pitch). Within each group, pitch and
instrument are held constant — the ONLY thing that varies is perceptual quality
(breathiness, vibrato, attack, articulation, etc.).

PCA on within-group z-differences discovers the axes of perceptual variation.
No labels needed beyond instrument category (from file path) and f0 (from SMS).

Also discovers between-instrument axes (what makes trumpet ≠ sax at same pitch).
"""

import sys
import torch
import numpy as np
from pathlib import Path
import os
import orjson
from collections import defaultdict
from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import StandardScaler

sys.stdout.reconfigure(line_buffering=True)

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

DATA_CACHE_PATH = SCRIPT_DIR.parent / "test_outputs" / "bidirectional_sms_z" / "data_cache.pt"
MANIFEST_PATH = SCRIPT_DIR.parent / "data" / "sms_v4" / "sms_manifest.json"
SMS_DATA_DIR = SCRIPT_DIR.parent / "data" / "sms_v4"
OUTPUT_DIR = SCRIPT_DIR.parent / "test_outputs" / "contrastive_discovery"

Z_DIM = 128
SEMITONE_BUCKET_SIZE = 3  # group pitches within 3 semitones

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


def classify_instrument(path):
    path_lower = path.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in path_lower for kw in keywords):
            return cat
    return 'Other'


def hz_to_midi(hz):
    if hz <= 0:
        return 0
    return 69 + 12 * np.log2(hz / 440.0)


def midi_to_bucket(midi_note):
    return int(midi_note / SEMITONE_BUCKET_SIZE) * SEMITONE_BUCKET_SIZE


def extract_mean_f0(sms_file, start_frame, T):
    """Extract mean f0 from raw SMS file."""
    try:
        sms_data = torch.load(sms_file, weights_only=True, map_location='cpu')
        freqs = sms_data['freqs']  # [T_full, n_sines]
        amps = sms_data['amps']    # [T_full, n_sines]

        end = min(start_frame + T, freqs.shape[0])
        freqs_crop = freqs[start_frame:end]
        amps_crop = amps[start_frame:end]

        # Find dominant frequency (highest amplitude) per frame
        dominant_idx = amps_crop.argmax(dim=1)
        f0s = freqs_crop[torch.arange(len(dominant_idx)), dominant_idx]

        # Filter active frames
        active = f0s > 20
        if active.sum() < 3:
            return 0.0
        return f0s[active].median().item()
    except Exception:
        return 0.0


def build_grouped_dataset():
    """Build dataset grouped by (instrument, pitch_bucket)."""
    print("=" * 60)
    print("BUILDING GROUPED DATASET")
    print("=" * 60)

    cache = torch.load(DATA_CACHE_PATH, weights_only=False, map_location='cpu')
    with open(MANIFEST_PATH, 'rb') as f:
        manifest = orjson.loads(f.read())

    sms_to_latent = {}
    for entry in manifest['entries']:
        sms_to_latent[entry['path']] = entry['latent_path']

    samples = []
    skipped = 0

    for i, sample in enumerate(cache):
        sms_path = sample['path']
        z_sms_4d = sample['z_sms']  # [8, 16, T]
        T_cache = z_sms_4d.shape[2]

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

        # Find start_frame
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

        # Crop z_real
        end_frame = min(start_frame + T_cache, z_real_full.shape[2])
        T_actual = end_frame - start_frame
        if T_actual < 10:
            skipped += 1
            continue

        z_real_crop = z_real_full[:, :, start_frame:end_frame]
        z_flat = z_real_crop.permute(2, 0, 1).reshape(T_actual, Z_DIM)
        z_mean = z_flat.mean(dim=0).numpy()  # [128] mean z for this sample

        # Get f0
        mean_f0 = 0.0
        if os.path.exists(sms_file):
            mean_f0 = extract_mean_f0(sms_file, start_frame, T_actual)

        # Classify instrument
        instrument = classify_instrument(latent_path)

        midi = hz_to_midi(mean_f0) if mean_f0 > 20 else 0
        bucket = midi_to_bucket(midi) if midi > 0 else -1

        samples.append({
            'z_mean': z_mean,
            'z_flat': z_flat.numpy(),  # [T, 128] full temporal z
            'instrument': instrument,
            'f0_hz': mean_f0,
            'midi': midi,
            'pitch_bucket': bucket,
            'path': latent_path,
            'name': Path(latent_path).stem,
        })

        if (i + 1) % 500 == 0:
            print(f"    Processed {len(samples)} samples ({skipped} skipped)...")

    print(f"\n  Total: {len(samples)} samples ({skipped} skipped)")

    # Group by (instrument, pitch_bucket)
    groups = defaultdict(list)
    for s in samples:
        if s['pitch_bucket'] < 0:
            continue  # skip samples with no detected pitch
        key = (s['instrument'], s['pitch_bucket'])
        groups[key].append(s)

    # Filter groups with >= 2 samples
    valid_groups = {k: v for k, v in groups.items() if len(v) >= 2}
    print(f"  Groups (instrument, pitch): {len(groups)} total, {len(valid_groups)} with >= 2 samples")

    # Summary
    inst_counts = defaultdict(int)
    for (inst, _), samples_in_group in valid_groups.items():
        inst_counts[inst] += len(samples_in_group)
    print(f"  Samples in valid groups: {sum(inst_counts.values())}")
    for inst, count in sorted(inst_counts.items(), key=lambda x: -x[1]):
        n_groups = sum(1 for (i, _) in valid_groups if i == inst)
        print(f"    {inst:>10s}: {count} samples in {n_groups} pitch groups")

    return samples, valid_groups


def discover_within_instrument_axes(valid_groups, n_components=8):
    """
    Discover perceptual axes from within-group variation.
    Same instrument + same pitch → differences are perceptual qualities.
    """
    print("\n" + "=" * 60)
    print("WITHIN-INSTRUMENT CONTRASTIVE DISCOVERY")
    print("Same instrument + same pitch = perceptual differences only")
    print("=" * 60)

    # Collect all within-group difference vectors
    diff_vectors = []
    diff_meta = []  # track which instrument/pitch each diff came from

    for (inst, pitch_bucket), group_samples in valid_groups.items():
        n = len(group_samples)
        z_means = np.stack([s['z_mean'] for s in group_samples])  # [n, 128]

        # All pairwise differences within group
        for i in range(n):
            for j in range(i + 1, n):
                diff = z_means[i] - z_means[j]
                diff_vectors.append(diff)
                diff_meta.append({
                    'instrument': inst,
                    'pitch_bucket': pitch_bucket,
                    'sample_a': group_samples[i]['name'],
                    'sample_b': group_samples[j]['name'],
                })

    diff_matrix = np.stack(diff_vectors)  # [N_pairs, 128]
    print(f"\n  Total within-group pairs: {len(diff_vectors)}")
    print(f"  Diff vector norm: mean={np.linalg.norm(diff_matrix, axis=1).mean():.4f}, "
          f"std={np.linalg.norm(diff_matrix, axis=1).std():.4f}")

    # PCA on difference vectors → principal axes of perceptual variation
    print(f"\n  Running PCA on within-group differences...")
    scaler = StandardScaler()
    diff_scaled = scaler.fit_transform(diff_matrix)

    pca = PCA(n_components=min(n_components, diff_scaled.shape[1]))
    pca.fit(diff_scaled)

    print(f"\n  Principal axes of within-instrument perceptual variation:")
    print(f"  {'Axis':>6s}  {'Var explained':>13s}  {'Cumulative':>10s}  {'Top z-dims':>40s}")
    print("  " + "-" * 75)

    axes = []
    for k in range(min(n_components, len(pca.components_))):
        var = pca.explained_variance_ratio_[k]
        cum_var = pca.explained_variance_ratio_[:k+1].sum()

        # Convert PCA component back to raw z-space
        direction = pca.components_[k] / (scaler.scale_ + 1e-10)
        direction = direction / (np.linalg.norm(direction) + 1e-10)

        top_idx = np.argsort(np.abs(direction))[::-1][:5]
        top_str = ", ".join([f"z[{i}]={direction[i]:+.2f}" for i in top_idx])

        print(f"  Axis {k:2d}  {var:13.4f}  {cum_var:10.4f}  {top_str}")

        axes.append({
            'direction': direction,
            'variance_explained': float(var),
            'cumulative_variance': float(cum_var),
        })

    # Also try ICA for more independent components
    print(f"\n  Running ICA for maximally independent axes...")
    n_ica = min(6, diff_scaled.shape[1], len(diff_vectors) - 1)
    try:
        ica = FastICA(n_components=n_ica, random_state=42, max_iter=1000)
        ica.fit(diff_scaled)

        ica_axes = []
        for k in range(n_ica):
            direction = ica.components_[k] / (scaler.scale_ + 1e-10)
            direction = direction / (np.linalg.norm(direction) + 1e-10)
            ica_axes.append(direction)

        # Check ICA independence via kurtosis of projections
        print(f"\n  ICA axes (maximally independent):")
        projections = diff_scaled @ ica.components_.T
        for k in range(n_ica):
            proj = projections[:, k]
            kurtosis = np.mean(proj**4) / (np.mean(proj**2)**2 + 1e-10) - 3
            top_idx = np.argsort(np.abs(ica_axes[k]))[::-1][:5]
            top_str = ", ".join([f"z[{i}]={ica_axes[k][i]:+.2f}" for i in top_idx])
            print(f"  ICA {k:2d}  kurtosis={kurtosis:+6.2f}  {top_str}")
    except Exception as e:
        print(f"  ICA failed: {e}")
        ica_axes = []

    return axes, ica_axes if ica_axes else [], diff_matrix, diff_meta


def discover_between_instrument_axes(valid_groups, n_components=6):
    """
    Discover instrument-identity axes.
    Same pitch + different instrument → the difference is timbre identity.
    """
    print("\n" + "=" * 60)
    print("BETWEEN-INSTRUMENT CONTRASTIVE DISCOVERY")
    print("Same pitch + different instrument = timbre identity")
    print("=" * 60)

    # Group by pitch bucket only, collect cross-instrument pairs
    pitch_groups = defaultdict(list)
    for (inst, pitch), group_samples in valid_groups.items():
        mean_z = np.mean([s['z_mean'] for s in group_samples], axis=0)
        pitch_groups[pitch].append({
            'instrument': inst,
            'z_mean': mean_z,
            'n_samples': len(group_samples),
        })

    # Cross-instrument differences at same pitch
    diff_vectors = []
    diff_labels = []

    for pitch, entries in pitch_groups.items():
        if len(entries) < 2:
            continue
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                diff = entries[i]['z_mean'] - entries[j]['z_mean']
                diff_vectors.append(diff)
                diff_labels.append(f"{entries[i]['instrument']}-{entries[j]['instrument']}")

    if len(diff_vectors) < 10:
        print("  Insufficient cross-instrument pairs at same pitch")
        return [], []

    diff_matrix = np.stack(diff_vectors)
    print(f"\n  Cross-instrument pairs at same pitch: {len(diff_vectors)}")

    # PCA
    scaler = StandardScaler()
    diff_scaled = scaler.fit_transform(diff_matrix)
    pca = PCA(n_components=min(n_components, diff_scaled.shape[1]))
    pca.fit(diff_scaled)

    print(f"\n  Principal axes of instrument identity:")
    print(f"  {'Axis':>6s}  {'Var explained':>13s}  {'Cumulative':>10s}")
    print("  " + "-" * 35)

    axes = []
    for k in range(min(n_components, len(pca.components_))):
        var = pca.explained_variance_ratio_[k]
        cum_var = pca.explained_variance_ratio_[:k+1].sum()
        direction = pca.components_[k] / (scaler.scale_ + 1e-10)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        print(f"  Axis {k:2d}  {var:13.4f}  {cum_var:10.4f}")
        axes.append({'direction': direction, 'variance_explained': float(var)})

    return axes, diff_labels


def check_orthogonality(within_axes, between_axes):
    """Check if within-instrument and between-instrument axes are orthogonal."""
    print("\n" + "=" * 60)
    print("WITHIN vs BETWEEN AXIS ORTHOGONALITY")
    print("=" * 60)
    print("  Low overlap = perceptual axes are independent of instrument identity")
    print("  High overlap = some 'perceptual' axes are actually instrument axes")

    n_within = min(6, len(within_axes))
    n_between = min(4, len(between_axes))

    if n_between == 0:
        print("  (no between-instrument axes to compare)")
        return

    print(f"\n  {'':>12s}", end="")
    for j in range(n_between):
        print(f"  Inst {j:1d}", end="")
    print()

    for i in range(n_within):
        d_w = within_axes[i]['direction']
        print(f"  Perc {i:2d}    ", end="")
        for j in range(n_between):
            d_b = between_axes[j]['direction']
            cos = abs(np.dot(d_w, d_b))
            marker = " *" if cos > 0.3 else ""
            print(f"  {cos:.3f}{marker}", end="")
        print()

    print("\n  * = overlap > 0.3 (this axis mixes perceptual and instrument identity)")


def validate_with_probes(within_axes, probe_path=None):
    """
    Cross-reference discovered axes with supervised probes (if available).
    Shows what audio features each discovered axis correlates with.
    """
    probe_file = SCRIPT_DIR.parent / "test_outputs" / "z_probes" / "z_probes.pt"
    if not probe_file.exists():
        print("\n  No supervised probes found — skip validation")
        return

    print("\n" + "=" * 60)
    print("VALIDATION: DISCOVERED AXES vs SUPERVISED PROBES")
    print("=" * 60)
    print("  Do discovered axes align with known audio features?")
    print("  High correlation = the axis IS that feature")
    print("  Low correlation = the axis is something new/unexpected")

    probes = torch.load(probe_file, weights_only=False, map_location='cpu')

    probe_names = list(probes.keys())
    probe_dirs = [probes[name]['direction'] for name in probe_names]

    n_axes = min(8, len(within_axes))
    print(f"\n  {'':>10s}", end="")
    for name in probe_names:
        print(f"  {name[:8]:>8s}", end="")
    print(f"  {'Best match':>20s}")

    for i in range(n_axes):
        d = within_axes[i]['direction']
        print(f"  Axis {i:2d}   ", end="")

        best_cos = 0
        best_name = "UNKNOWN"
        for j, name in enumerate(probe_names):
            cos = abs(np.dot(d, probe_dirs[j]))
            if cos > best_cos:
                best_cos = cos
                best_name = name
            print(f"  {cos:8.3f}", end="")

        label = f"{best_name} ({best_cos:.2f})" if best_cos > 0.2 else "NOVEL"
        print(f"  {label:>20s}")


def main():
    print("=" * 60)
    print("CONTRASTIVE DISCOVERY OF PERCEPTUAL AXES")
    print("Discovery, not prescription")
    print("=" * 60)
    print()
    print("Method: group by (instrument, pitch)")
    print("  Within-group variation = perceptual quality (discovery)")
    print("  Between-group variation = instrument identity")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build grouped dataset
    samples, valid_groups = build_grouped_dataset()

    # Discover within-instrument perceptual axes
    within_pca, within_ica, diff_matrix, diff_meta = discover_within_instrument_axes(valid_groups)

    # Discover between-instrument identity axes
    between_axes, between_labels = discover_between_instrument_axes(valid_groups)

    # Check orthogonality
    check_orthogonality(within_pca, between_axes)

    # Validate against supervised probes
    validate_with_probes(within_pca)

    # Save all discovered axes
    save_data = {
        'within_pca': [{
            'direction': ax['direction'],
            'variance_explained': ax['variance_explained'],
        } for ax in within_pca],
        'within_ica': [d.tolist() for d in within_ica] if within_ica else [],
        'between_pca': [{
            'direction': ax['direction'],
            'variance_explained': ax['variance_explained'],
        } for ax in between_axes],
        'n_within_pairs': len(diff_matrix),
        'n_groups': len(valid_groups),
    }

    save_path = OUTPUT_DIR / "discovered_axes.pt"
    torch.save(save_data, save_path)
    print(f"\n  Saved discovered axes to {save_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_var = sum(ax['variance_explained'] for ax in within_pca[:6])
    print(f"\n  Top 6 within-instrument axes explain {total_var:.1%} of perceptual variation")

    if between_axes:
        between_var = sum(ax['variance_explained'] for ax in between_axes[:4])
        print(f"  Top 4 between-instrument axes explain {between_var:.1%} of instrument identity")

    print(f"\n  These axes are DISCOVERED, not prescribed.")
    print(f"  To identify what they mean: listen to audio edited along each axis.")
    print(f"  Some may align with known features (breathiness, brightness).")
    print(f"  Others may be novel — perceptual qualities without standard names.")

    print(f"\n  Next step: add these axes to the Gradio UI as editing sliders.")
    print(f"  z_edited = z_real + delta * discovered_axis[k]")
    print(f"  Listen to what each axis does across different instruments.")
    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
