#!/usr/bin/env python3
"""
DISCOVER atomic operations from DCAE decoder behavior.

Philosophy: Discovery not prescription.
- Don't assume what the atoms are
- Probe the decoder systematically
- Let patterns emerge from the data
- Cluster/factorize to find natural atomic units

Probing strategy:
1. Single-dim perturbations: how does each z[i] affect output?
2. Response curves: is the mapping linear? sigmoid? threshold?
3. Pairwise interactions: do dims interact nonlinearly?
4. Temporal patterns: does z[t] affect output at t+1, t+2?
5. Clustering: group dims by similar behavior → candidate atoms
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from collections import defaultdict
import argparse

sys.path.insert(0, '/home/arlo/Data/ACE-Step')
from acestep.music_dcae.music_dcae_pipeline import MusicDCAE


def load_dcae(device='cuda'):
    """Load DCAE model."""
    DCAE_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_dcae_f8c8"
    VOCODER_PATH = "/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c/music_vocoder"

    dcae = MusicDCAE(
        dcae_checkpoint_path=DCAE_PATH,
        vocoder_checkpoint_path=VOCODER_PATH,
    )
    dcae.dcae.to(device)
    dcae.dcae.eval()
    return dcae


def get_mel_from_z(dcae, z, device='cuda'):
    """Decode z to mel spectrogram (skip vocoder for speed)."""
    with torch.no_grad():
        # Denormalize
        z_denorm = z / 0.1786 + (-1.9091)
        # Decode to mel
        mel = dcae.dcae.decode(z_denorm).sample
        # mel shape: [B, 2, 128, T*8]
        return mel


def mel_to_spectral_features(mel):
    """Extract interpretable features from mel spectrogram."""
    # mel: [B, 2, 128, T]
    B, C, F, T = mel.shape

    # Average over stereo channels
    mel_mono = mel.mean(dim=1)  # [B, 128, T]

    features = {}

    # 1. Spectral centroid (brightness) per frame
    freq_bins = torch.arange(F, device=mel.device).float()
    mel_power = torch.exp(mel_mono)  # Convert from log
    total_power = mel_power.sum(dim=1, keepdim=True).clamp(min=1e-8)
    features['centroid'] = (mel_power * freq_bins.view(1, -1, 1) / total_power).sum(dim=1)  # [B, T]

    # 2. Total energy per frame
    features['energy'] = mel_power.sum(dim=1)  # [B, T]

    # 3. Spectral flatness (tonality vs noise)
    log_mel = mel_mono.clamp(min=-10)
    geo_mean = torch.exp(log_mel.mean(dim=1))
    arith_mean = mel_power.mean(dim=1).clamp(min=1e-8)
    features['flatness'] = geo_mean / arith_mean  # [B, T]

    # 4. Peak frequency bin
    features['peak_bin'] = mel_mono.argmax(dim=1).float()  # [B, T]

    # 5. Low/mid/high energy ratio
    features['low_energy'] = mel_power[:, :32, :].sum(dim=1)
    features['mid_energy'] = mel_power[:, 32:96, :].sum(dim=1)
    features['high_energy'] = mel_power[:, 96:, :].sum(dim=1)

    return features


def probe_single_dimensions(dcae, base_z, n_dims=128, delta=0.5, device='cuda'):
    """
    Probe each z dimension independently.

    Returns: influence matrix [128, n_features] showing how each dim affects output
    """
    print("\n" + "=" * 70)
    print("PROBE 1: Single Dimension Effects")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)  # [B, 128, T]

    # Get baseline features
    base_mel = get_mel_from_z(dcae, base_z, device)
    base_features = mel_to_spectral_features(base_mel)

    influences = defaultdict(list)

    for dim in range(n_dims):
        if dim % 32 == 0:
            print(f"  Probing dims {dim}-{min(dim+31, n_dims-1)}...")

        # Perturb this dimension
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, :] += delta

        # Reshape back and decode
        z_perturbed_4d = z_perturbed.reshape(B, C, H, T)
        perturbed_mel = get_mel_from_z(dcae, z_perturbed_4d, device)
        perturbed_features = mel_to_spectral_features(perturbed_mel)

        # Measure change for each feature
        for feat_name in base_features:
            diff = (perturbed_features[feat_name] - base_features[feat_name]).mean().item()
            influences[feat_name].append(diff)

    # Convert to arrays
    influence_matrix = {}
    for feat_name in influences:
        influence_matrix[feat_name] = np.array(influences[feat_name])

    return influence_matrix


def probe_response_curves(dcae, base_z, top_dims, n_points=11, device='cuda'):
    """
    For influential dimensions, measure response curve.

    Is the response linear, sigmoid, threshold, saturating?
    """
    print("\n" + "=" * 70)
    print("PROBE 2: Response Curves (linearity test)")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    deltas = np.linspace(-2, 2, n_points)
    response_curves = {}

    for dim in top_dims:
        print(f"  Probing dim {dim} response curve...")
        responses = []

        for delta in deltas:
            z_perturbed = z_flat.clone()
            z_perturbed[:, dim, :] += delta
            z_perturbed_4d = z_perturbed.reshape(B, C, H, T)

            mel = get_mel_from_z(dcae, z_perturbed_4d, device)
            features = mel_to_spectral_features(mel)

            # Track multiple features
            responses.append({
                'delta': delta,
                'centroid': features['centroid'].mean().item(),
                'energy': features['energy'].mean().item(),
                'peak_bin': features['peak_bin'].mean().item(),
            })

        response_curves[dim] = responses

    return response_curves


def probe_pairwise_interactions(dcae, base_z, top_dims, delta=0.5, device='cuda'):
    """
    Test if dimensions interact nonlinearly.

    If effect(A+B) ≠ effect(A) + effect(B), there's interaction.
    """
    print("\n" + "=" * 70)
    print("PROBE 3: Pairwise Interactions")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    base_mel = get_mel_from_z(dcae, base_z, device)
    base_centroid = mel_to_spectral_features(base_mel)['centroid'].mean().item()

    interactions = {}

    n_pairs = 0
    for i, dim_i in enumerate(top_dims):
        for dim_j in top_dims[i+1:]:
            # Effect of dim_i alone
            z_i = z_flat.clone()
            z_i[:, dim_i, :] += delta
            mel_i = get_mel_from_z(dcae, z_i.reshape(B, C, H, T), device)
            effect_i = mel_to_spectral_features(mel_i)['centroid'].mean().item() - base_centroid

            # Effect of dim_j alone
            z_j = z_flat.clone()
            z_j[:, dim_j, :] += delta
            mel_j = get_mel_from_z(dcae, z_j.reshape(B, C, H, T), device)
            effect_j = mel_to_spectral_features(mel_j)['centroid'].mean().item() - base_centroid

            # Effect of both together
            z_ij = z_flat.clone()
            z_ij[:, dim_i, :] += delta
            z_ij[:, dim_j, :] += delta
            mel_ij = get_mel_from_z(dcae, z_ij.reshape(B, C, H, T), device)
            effect_ij = mel_to_spectral_features(mel_ij)['centroid'].mean().item() - base_centroid

            # Interaction = deviation from additivity
            expected_additive = effect_i + effect_j
            interaction = effect_ij - expected_additive

            interactions[(dim_i, dim_j)] = {
                'effect_i': effect_i,
                'effect_j': effect_j,
                'effect_ij': effect_ij,
                'interaction': interaction,
                'relative_interaction': abs(interaction) / (abs(expected_additive) + 1e-6),
            }

            n_pairs += 1

    print(f"  Tested {n_pairs} pairs")

    return interactions


def probe_temporal_influence(dcae, base_z, top_dims, device='cuda'):
    """
    Does z[t] affect output at t+1, t+2, etc?

    This reveals temporal dependencies in the decoder.
    """
    print("\n" + "=" * 70)
    print("PROBE 4: Temporal Influence")
    print("=" * 70)

    B, C, H, T = base_z.shape
    z_flat = base_z.reshape(B, C * H, T)

    if T < 5:
        print("  Not enough time steps for temporal analysis")
        return {}

    temporal_influence = {}

    for dim in top_dims[:5]:  # Just top 5 for speed
        print(f"  Probing dim {dim} temporal influence...")

        influences_by_offset = []

        # Perturb at middle time step
        t_perturb = T // 2
        z_perturbed = z_flat.clone()
        z_perturbed[:, dim, t_perturb] += 1.0

        mel_base = get_mel_from_z(dcae, base_z, device)
        mel_perturbed = get_mel_from_z(dcae, z_perturbed.reshape(B, C, H, T), device)

        # Measure effect at different output time steps
        base_energy = mel_to_spectral_features(mel_base)['energy']  # [B, T_mel]
        perturbed_energy = mel_to_spectral_features(mel_perturbed)['energy']

        # Output T is 8x input T
        T_mel = base_energy.shape[1]
        t_perturb_mel = t_perturb * 8

        for offset in range(-16, 17, 4):
            t_check = t_perturb_mel + offset
            if 0 <= t_check < T_mel:
                diff = (perturbed_energy[:, t_check] - base_energy[:, t_check]).mean().item()
                influences_by_offset.append((offset, diff))

        temporal_influence[dim] = influences_by_offset

    return temporal_influence


def cluster_dimensions(influence_matrix, n_clusters=8):
    """
    Cluster dimensions by similar influence patterns.

    Dimensions in the same cluster might form an atomic operation.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS: Clustering Dimensions")
    print("=" * 70)

    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    # Stack all features into feature vectors per dimension
    feature_names = list(influence_matrix.keys())
    n_dims = len(influence_matrix[feature_names[0]])

    X = np.column_stack([influence_matrix[f] for f in feature_names])
    X = StandardScaler().fit_transform(X)

    # Cluster
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    clusters = defaultdict(list)
    for dim, label in enumerate(labels):
        clusters[label].append(dim)

    print(f"\n  Found {n_clusters} clusters:")
    for label in sorted(clusters.keys()):
        dims = clusters[label]
        # Characterize cluster by average influence
        avg_influences = {}
        for feat in feature_names:
            avg_influences[feat] = np.mean([influence_matrix[feat][d] for d in dims])

        dominant_feat = max(avg_influences.items(), key=lambda x: abs(x[1]))

        print(f"\n  Cluster {label} ({len(dims)} dims): {dims[:8]}{'...' if len(dims) > 8 else ''}")
        print(f"    Dominant effect: {dominant_feat[0]} ({dominant_feat[1]:+.4f})")

    return clusters, labels


def analyze_response_linearity(response_curves):
    """Analyze if responses are linear, sigmoid, threshold, etc."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Response Linearity")
    print("=" * 70)

    for dim, responses in response_curves.items():
        deltas = [r['delta'] for r in responses]
        centroids = [r['centroid'] for r in responses]

        # Fit linear
        coeffs = np.polyfit(deltas, centroids, 1)
        linear_pred = np.polyval(coeffs, deltas)
        linear_residual = np.mean((np.array(centroids) - linear_pred) ** 2)

        # Fit quadratic
        coeffs_q = np.polyfit(deltas, centroids, 2)
        quad_pred = np.polyval(coeffs_q, deltas)
        quad_residual = np.mean((np.array(centroids) - quad_pred) ** 2)

        # Check for threshold/saturation
        mid_range = centroids[len(centroids)//2 - 1 : len(centroids)//2 + 2]
        edge_range = centroids[:2] + centroids[-2:]
        mid_var = np.var(mid_range)
        edge_var = np.var(edge_range)

        print(f"\n  Dim {dim}:")
        print(f"    Linear residual:    {linear_residual:.6f}")
        print(f"    Quadratic residual: {quad_residual:.6f}")
        print(f"    Linear slope:       {coeffs[0]:.4f}")

        if quad_residual < linear_residual * 0.5:
            print(f"    → NONLINEAR (quadratic fits 2x better)")
        elif mid_var < edge_var * 0.3:
            print(f"    → SATURATING (flat in middle)")
        else:
            print(f"    → APPROXIMATELY LINEAR")


def analyze_interactions(interactions):
    """Analyze which dimension pairs interact."""
    print("\n" + "=" * 70)
    print("ANALYSIS: Dimension Interactions")
    print("=" * 70)

    # Sort by interaction strength
    sorted_pairs = sorted(
        interactions.items(),
        key=lambda x: abs(x[1]['interaction']),
        reverse=True
    )

    print("\n  Top interacting pairs:")
    for (dim_i, dim_j), data in sorted_pairs[:10]:
        print(f"    dims ({dim_i}, {dim_j}): interaction={data['interaction']:+.4f} "
              f"(relative: {data['relative_interaction']:.1%})")

    print("\n  Most independent pairs:")
    for (dim_i, dim_j), data in sorted_pairs[-5:]:
        print(f"    dims ({dim_i}, {dim_j}): interaction={data['interaction']:+.4f}")

    # Count strong interactions per dim
    interaction_count = defaultdict(int)
    for (dim_i, dim_j), data in interactions.items():
        if data['relative_interaction'] > 0.2:  # >20% nonlinear
            interaction_count[dim_i] += 1
            interaction_count[dim_j] += 1

    if interaction_count:
        print("\n  Dims with most interactions:")
        for dim, count in sorted(interaction_count.items(), key=lambda x: -x[1])[:10]:
            print(f"    dim {dim}: {count} strong interactions")


def discover_candidate_atoms(clusters, interactions, influence_matrix):
    """
    Synthesize findings into candidate atomic operations.
    """
    print("\n" + "=" * 70)
    print("DISCOVERY: Candidate Atomic Operations")
    print("=" * 70)

    print("""
  Based on probing, here are candidate atoms:

  An 'atom' is a group of z dimensions that:
  1. Have similar influence patterns (same cluster)
  2. Interact nonlinearly with each other (form a unit)
  3. Are relatively independent from other groups

  These are DISCOVERED from data, not assumed.
    """)

    feature_names = list(influence_matrix.keys())

    for cluster_id, dims in clusters.items():
        if len(dims) < 3:
            continue

        # Characterize this cluster
        avg_influences = {}
        for feat in feature_names:
            avg_influences[feat] = np.mean([influence_matrix[feat][d] for d in dims])

        # Count internal vs external interactions
        internal_interactions = 0
        external_interactions = 0
        dims_set = set(dims)

        for (dim_i, dim_j), data in interactions.items():
            if data['relative_interaction'] > 0.1:
                if dim_i in dims_set and dim_j in dims_set:
                    internal_interactions += 1
                elif dim_i in dims_set or dim_j in dims_set:
                    external_interactions += 1

        print(f"\n  CANDIDATE ATOM {cluster_id}:")
        print(f"    Dimensions: {dims[:10]}{'...' if len(dims) > 10 else ''}")
        print(f"    Size: {len(dims)} dims")
        print(f"    Internal interactions: {internal_interactions}")
        print(f"    External interactions: {external_interactions}")
        print(f"    Dominant effects:")
        for feat, val in sorted(avg_influences.items(), key=lambda x: -abs(x[1]))[:3]:
            if abs(val) > 0.001:
                print(f"      {feat}: {val:+.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_top_dims', type=int, default=20,
                        help='Number of top dimensions to analyze in detail')
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of clusters for dimension grouping')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load DCAE
    print("\nLoading DCAE...")
    dcae = load_dcae(device)

    # Create a base z (could also load from dataset)
    print("\nCreating base latent...")
    # Shape: [B, 8, 16, T] where T ~ audio_length / 512 / 8
    T = 16  # ~1.5 seconds of audio
    base_z = torch.randn(1, 8, 16, T, device=device) * 0.1  # Small random z

    # Probe 1: Single dimension effects
    influence_matrix = probe_single_dimensions(dcae, base_z, device=device)

    # Find most influential dimensions
    total_influence = np.zeros(128)
    for feat, influences in influence_matrix.items():
        total_influence += np.abs(influences)

    top_dims = np.argsort(total_influence)[-args.n_top_dims:][::-1].tolist()
    print(f"\n  Most influential dims: {top_dims[:10]}")

    # Probe 2: Response curves for top dims
    response_curves = probe_response_curves(dcae, base_z, top_dims[:10], device=device)
    analyze_response_linearity(response_curves)

    # Probe 3: Pairwise interactions
    interactions = probe_pairwise_interactions(dcae, base_z, top_dims[:15], device=device)
    analyze_interactions(interactions)

    # Probe 4: Temporal influence
    temporal = probe_temporal_influence(dcae, base_z, top_dims[:5], device=device)
    if temporal:
        print("\n" + "=" * 70)
        print("ANALYSIS: Temporal Spread")
        print("=" * 70)
        for dim, influences in temporal.items():
            print(f"\n  Dim {dim} temporal influence:")
            for offset, effect in influences:
                bar = "█" * int(abs(effect) * 100)
                print(f"    t{offset:+3d}: {effect:+.4f} {bar}")

    # Cluster dimensions
    clusters, labels = cluster_dimensions(influence_matrix, n_clusters=args.n_clusters)

    # Discover candidate atoms
    discover_candidate_atoms(clusters, interactions, influence_matrix)

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("""
  1. Each cluster is a CANDIDATE atomic operation
  2. The influence patterns show what each atom DOES
  3. Interactions show which atoms work TOGETHER
  4. Response curves show LINEAR vs NONLINEAR behavior

  To confirm atoms:
  - Test if cluster dims are substitutable (same effect)
  - Test if removing one cluster breaks specific audio features
  - Build tree from confirmed atoms bottom-up
    """)


if __name__ == "__main__":
    main()
