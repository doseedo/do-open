#!/usr/bin/env python3
"""
Timbre Profiler - Discover technique clusters within subgroups

For each subgroup, this tool:
1. Extracts pooled latent features from all samples
2. Reduces to 2D with UMAP for visualization
3. Clusters samples to discover natural groupings
4. Outputs cluster assignments for human review
5. Generates interactive HTML visualization

Usage:
  # Profile a specific subgroup
  python3 timbre_profiler.py --group strings --subgroup violin

  # Profile all subgroups in a group
  python3 timbre_profiler.py --group brass

  # Profile everything (slow)
  python3 timbre_profiler.py --all

After review, you can add confirmed technique labels to corrections.json
"""

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch

# Optional imports - install if needed
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: umap-learn not installed. Install with: pip install umap-learn")

try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
    print("Warning: hdbscan not installed. Install with: pip install hdbscan")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Paths
MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")
LATENTS_ROOT = Path("/home/arlo/gcs-bucket/Latents")
OUTPUT_DIR = Path("/home/arlo/Data/timbre_profiles")

# Feature extraction settings
POOL_METHODS = ['mean', 'std', 'max']


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )


def load_latent(latent_path: Path) -> Optional[torch.Tensor]:
    """Load latent tensor from file."""
    try:
        data = torch.load(latent_path, map_location='cpu', weights_only=False)
        if isinstance(data, dict):
            return data.get('latents', data.get('z', None))
        return data
    except Exception:
        return None


def pool_latent(latent: torch.Tensor) -> np.ndarray:
    """Pool latent [8, 16, T] to fixed-size feature vector."""
    if latent is None:
        return None

    # Handle different shapes
    if latent.dim() == 2:
        if latent.shape[0] == 128:
            latent = latent.view(8, 16, -1)
        else:
            features = []
            if 'mean' in POOL_METHODS:
                features.append(latent.mean(dim=-1))
            if 'std' in POOL_METHODS:
                features.append(latent.std(dim=-1))
            if 'max' in POOL_METHODS:
                features.append(latent.max(dim=-1)[0])
            return torch.cat(features, dim=-1).flatten().numpy()

    # [8, 16, T] format - mask silent regions
    if latent.shape[-1] > 1:
        energy = torch.sqrt((latent ** 2).mean(dim=(0, 1)))
        non_silent = energy > 0.01
        if non_silent.sum() > 0:
            latent = latent[:, :, non_silent]

    features = []
    if 'mean' in POOL_METHODS:
        features.append(latent.mean(dim=-1))
    if 'std' in POOL_METHODS:
        features.append(latent.std(dim=-1))
    if 'max' in POOL_METHODS:
        features.append(latent.max(dim=-1)[0])

    stacked = torch.stack(features, dim=-1)
    return stacked.flatten().numpy()


def audio_path_to_latent_path(audio_path: str) -> Optional[Path]:
    """Convert audio path to latent path."""
    audio_path = Path(audio_path)

    parts = audio_path.parts
    if 'gcs-bucket' in parts:
        idx = parts.index('gcs-bucket')
        rel_path = Path(*parts[idx+1:])
    elif 'protools' in parts:
        idx = parts.index('protools')
        rel_path = Path(*parts[idx:])
    else:
        rel_path = audio_path

    stem = rel_path.with_suffix('')
    stem_no_ext = Path(str(stem).replace('.wav', '').replace('.mp3', '').replace('.flac', ''))

    for s in [stem, stem_no_ext]:
        for ext in ['.dcae.pt', '.pt']:
            latent_path = LATENTS_ROOT / f"{s}{ext}"
            if latent_path.exists():
                return latent_path

    return None


def load_manifest() -> Dict:
    """Load unified manifest."""
    with open(MANIFEST_PATH, 'rb') as f:
        return orjson.loads(f.read())


def get_samples_for_subgroup(
    manifest: Dict,
    group: str,
    subgroup: Optional[str] = None,
    max_samples: int = 5000
) -> List[Dict]:
    """Get samples for a specific group/subgroup."""
    entries = manifest.get('entries', [])

    samples = []
    for e in entries:
        if e.get('group') != group:
            continue
        if subgroup and e.get('subgroup') != subgroup:
            continue
        if not e.get('has_latent', False):
            continue

        samples.append(e)

    # Sample if too many
    if len(samples) > max_samples:
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    return samples


def extract_features(samples: List[Dict], show_progress: bool = True) -> Tuple[np.ndarray, List[Dict]]:
    """Extract pooled features from samples."""
    features = []
    valid_samples = []

    total = len(samples)
    for i, sample in enumerate(samples):
        if show_progress and i % 500 == 0:
            logging.info(f"  Extracting features: {i}/{total}")

        # Try to load latent
        latent_path = None
        if sample.get('latent_path'):
            latent_path = Path(sample['latent_path'])
            if not latent_path.exists():
                latent_path = None

        if latent_path is None:
            latent_path = audio_path_to_latent_path(sample.get('audio_path', ''))

        if latent_path is None:
            continue

        latent = load_latent(latent_path)
        if latent is None:
            continue

        feat = pool_latent(latent)
        if feat is None:
            continue

        features.append(feat)
        valid_samples.append(sample)

    if not features:
        return np.array([]), []

    return np.stack(features), valid_samples


def cluster_samples(
    features: np.ndarray,
    n_clusters: int = None,
    method: str = 'auto'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cluster samples and reduce to 2D.

    Returns:
        coords_2d: (N, 2) array for visualization
        labels: cluster assignments
    """
    # Normalize features
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)

    # Dimensionality reduction with UMAP
    if HAS_UMAP:
        logging.info("  Running UMAP...")
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.1,
            metric='cosine',
            random_state=42
        )
        coords_2d = reducer.fit_transform(features_norm)
    else:
        # Fallback to PCA
        from sklearn.decomposition import PCA
        logging.info("  Running PCA (UMAP not available)...")
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(features_norm)

    # Clustering
    if method == 'auto':
        method = 'hdbscan' if HAS_HDBSCAN else 'kmeans'

    if method == 'hdbscan' and HAS_HDBSCAN:
        logging.info("  Running HDBSCAN clustering...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(10, len(features) // 50),
            min_samples=5,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        labels = clusterer.fit_predict(coords_2d)
    else:
        # KMeans fallback
        if n_clusters is None:
            # Estimate number of clusters
            n_clusters = min(10, max(2, len(features) // 100))
        logging.info(f"  Running KMeans with {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coords_2d)

    return coords_2d, labels


def generate_html_visualization(
    coords: np.ndarray,
    labels: np.ndarray,
    samples: List[Dict],
    group: str,
    subgroup: str,
    output_path: Path
):
    """Generate interactive HTML visualization."""

    # Prepare data for JavaScript
    points = []
    for i, (coord, label, sample) in enumerate(zip(coords, labels, samples)):
        filename = Path(sample.get('audio_path', '')).name
        technique = sample.get('technique', 'unknown')
        points.append({
            'x': float(coord[0]),
            'y': float(coord[1]),
            'cluster': int(label),
            'filename': filename,
            'technique': technique,
            'path': sample.get('audio_path', ''),
        })

    # Count per cluster
    cluster_counts = Counter(labels)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Timbre Profile: {group}/{subgroup}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #333; }}
        .stats {{ background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }}
        .cluster-list {{ columns: 3; }}
        #plot {{ width: 100%; height: 700px; }}
        .instructions {{ background: #e8f4f8; padding: 15px; margin: 10px 0; border-radius: 5px; }}
    </style>
</head>
<body>
    <h1>Timbre Profile: {group}/{subgroup}</h1>

    <div class="instructions">
        <strong>Instructions:</strong>
        <ul>
            <li>Each point is a sample, colored by cluster</li>
            <li>Hover to see filename and any existing technique label</li>
            <li>Click a point to copy its path (for adding to corrections)</li>
            <li>Review clusters - do they represent distinct techniques?</li>
            <li>If a cluster clearly represents a technique, add samples to corrections.json</li>
        </ul>
    </div>

    <div class="stats">
        <strong>Statistics:</strong>
        <ul>
            <li>Total samples: {len(samples)}</li>
            <li>Clusters found: {len(set(labels)) - (1 if -1 in labels else 0)}</li>
            <li>Noise points (cluster -1): {cluster_counts.get(-1, 0)}</li>
        </ul>
        <strong>Cluster sizes:</strong>
        <div class="cluster-list">
        {"".join(f"<div>Cluster {c}: {n} samples</div>" for c, n in sorted(cluster_counts.items()) if c != -1)}
        </div>
    </div>

    <div id="plot"></div>

    <script>
        const points = {orjson.dumps(points).decode()};

        // Group by cluster
        const clusters = {{}};
        points.forEach(p => {{
            if (!clusters[p.cluster]) clusters[p.cluster] = {{x: [], y: [], text: [], paths: []}};
            clusters[p.cluster].x.push(p.x);
            clusters[p.cluster].y.push(p.y);
            clusters[p.cluster].text.push(`${{p.filename}}<br>Technique: ${{p.technique}}`);
            clusters[p.cluster].paths.push(p.path);
        }});

        // Create traces
        const traces = Object.entries(clusters).map(([cluster, data]) => ({{
            x: data.x,
            y: data.y,
            text: data.text,
            customdata: data.paths,
            mode: 'markers',
            type: 'scatter',
            name: cluster == -1 ? 'Noise' : `Cluster ${{cluster}}`,
            marker: {{
                size: 6,
                opacity: cluster == -1 ? 0.3 : 0.7
            }},
            hovertemplate: '%{{text}}<extra></extra>'
        }}));

        const layout = {{
            title: 'Timbre Space (UMAP projection)',
            xaxis: {{ title: 'UMAP 1' }},
            yaxis: {{ title: 'UMAP 2' }},
            hovermode: 'closest',
            showlegend: true
        }};

        Plotly.newPlot('plot', traces, layout);

        // Click handler to copy path
        document.getElementById('plot').on('plotly_click', function(data) {{
            const path = data.points[0].customdata;
            navigator.clipboard.writeText(path).then(() => {{
                alert('Copied to clipboard:\\n' + path);
            }});
        }});
    </script>
</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    logging.info(f"  Saved visualization to {output_path}")


def save_cluster_data(
    coords: np.ndarray,
    labels: np.ndarray,
    samples: List[Dict],
    group: str,
    subgroup: str,
    output_dir: Path
):
    """Save cluster assignments as JSON for programmatic access."""

    cluster_data = {
        'group': group,
        'subgroup': subgroup,
        'total_samples': len(samples),
        'num_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'clusters': defaultdict(list)
    }

    for coord, label, sample in zip(coords, labels, samples):
        cluster_data['clusters'][str(label)].append({
            'audio_path': sample.get('audio_path', ''),
            'filename': Path(sample.get('audio_path', '')).name,
            'existing_technique': sample.get('technique'),
            'coord': [float(coord[0]), float(coord[1])],
        })

    # Convert defaultdict to dict
    cluster_data['clusters'] = dict(cluster_data['clusters'])

    output_path = output_dir / f"{group}_{subgroup}_clusters.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(cluster_data, option=orjson.OPT_INDENT_2))

    logging.info(f"  Saved cluster data to {output_path}")


def profile_subgroup(
    manifest: Dict,
    group: str,
    subgroup: str,
    max_samples: int = 5000,
    output_dir: Path = OUTPUT_DIR
):
    """Profile a single subgroup."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Profiling: {group}/{subgroup}")
    logging.info('='*60)

    # Get samples
    samples = get_samples_for_subgroup(manifest, group, subgroup, max_samples)
    logging.info(f"Found {len(samples)} samples")

    if len(samples) < 20:
        logging.warning(f"Too few samples ({len(samples)}), skipping")
        return

    # Extract features
    features, valid_samples = extract_features(samples)
    logging.info(f"Extracted features for {len(valid_samples)} samples")

    if len(valid_samples) < 20:
        logging.warning(f"Too few valid samples ({len(valid_samples)}), skipping")
        return

    # Cluster
    coords, labels = cluster_samples(features)

    # Report
    cluster_counts = Counter(labels)
    logging.info(f"Found {len(cluster_counts)} clusters:")
    for c, n in sorted(cluster_counts.items()):
        if c == -1:
            logging.info(f"  Noise: {n} samples")
        else:
            logging.info(f"  Cluster {c}: {n} samples")

    # Check if clusters correlate with existing techniques
    if any(s.get('technique') for s in valid_samples):
        logging.info("\nExisting technique distribution per cluster:")
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                continue
            samples_in_cluster = [s for s, l in zip(valid_samples, labels) if l == cluster_id]
            tech_dist = Counter(s.get('technique', 'unlabeled') for s in samples_in_cluster)
            logging.info(f"  Cluster {cluster_id}: {dict(tech_dist)}")

    # Generate outputs
    subgroup_name = subgroup or 'all'
    generate_html_visualization(
        coords, labels, valid_samples, group, subgroup_name,
        output_dir / f"{group}_{subgroup_name}_profile.html"
    )
    save_cluster_data(coords, labels, valid_samples, group, subgroup_name, output_dir)


def main():
    parser = argparse.ArgumentParser(description='Timbre Profiler')
    parser.add_argument('--group', type=str, help='Group to profile')
    parser.add_argument('--subgroup', type=str, help='Subgroup to profile')
    parser.add_argument('--all', action='store_true', help='Profile all groups')
    parser.add_argument('--max-samples', type=int, default=5000, help='Max samples per subgroup')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))

    args = parser.parse_args()

    setup_logging()
    output_dir = Path(args.output_dir)

    logging.info("Loading manifest...")
    manifest = load_manifest()
    entries = manifest.get('entries', [])
    logging.info(f"Loaded {len(entries)} entries")

    if args.all:
        # Profile all subgroups
        subgroup_counts = Counter((e.get('group'), e.get('subgroup')) for e in entries)
        for (group, subgroup), count in subgroup_counts.most_common():
            if group in ('undefined', 'unknown') or count < 50:
                continue
            profile_subgroup(manifest, group, subgroup, args.max_samples, output_dir)

    elif args.group:
        if args.subgroup:
            # Profile specific subgroup
            profile_subgroup(manifest, args.group, args.subgroup, args.max_samples, output_dir)
        else:
            # Profile all subgroups in group
            subgroup_counts = Counter(
                e.get('subgroup') for e in entries if e.get('group') == args.group
            )
            for subgroup, count in subgroup_counts.most_common():
                if count < 50:
                    continue
                profile_subgroup(manifest, args.group, subgroup, args.max_samples, output_dir)
    else:
        parser.print_help()
        print("\nExample usage:")
        print("  python3 timbre_profiler.py --group strings --subgroup violin")
        print("  python3 timbre_profiler.py --group brass")


if __name__ == '__main__':
    main()
