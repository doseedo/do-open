#!/usr/bin/env python3
"""
Latent Silence Detector (Optimized)

Detects silent/near-silent audio files by analyzing their latent representations.
Uses high-parallelism threading for fast GCS access.

Usage:
    python3 latent_silence_detector.py --mode detect
    python3 latent_silence_detector.py --mode stats
"""

import argparse
import orjson
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os

# Paths
OUTPUT_FILE = Path("/home/arlo/Data/silence_detector/silent_files.json")
MANIFEST_FILE = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")

# Silence detection threshold (based on latent energy)
SILENCE_THRESHOLD = 0.001
NEAR_SILENCE_THRESHOLD = 0.01

# Lazy torch import to speed up startup
_torch = None
def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def process_single_file(args):
    """Worker function - optimized for speed."""
    latent_path_str, audio_path, threshold, near_threshold = args

    # Fast path existence check
    latent_path = latent_path_str
    if not os.path.exists(latent_path):
        # Try .pt instead of .dcae.pt
        alt_path = latent_path.replace('.dcae.pt', '.pt')
        if os.path.exists(alt_path):
            latent_path = alt_path
        else:
            return ("missing", audio_path, None, None)

    try:
        torch = get_torch()
        # Load with minimal overhead - weights_only=False since files are dicts
        data = torch.load(latent_path, map_location='cpu', weights_only=False)

        # Fast energy calculation - tensor is stored in 'latents' key
        if data is None:
            return ("error", audio_path, None, None)

        latent = data['latents'] if isinstance(data, dict) else data
        energy = float(latent.abs().mean())

        if energy < threshold:
            return ("silent", audio_path, latent_path, energy)
        elif energy < near_threshold:
            return ("near_silent", audio_path, latent_path, energy)
        else:
            return ("normal", audio_path, None, energy)
    except Exception:
        return ("error", audio_path, None, None)


def get_latent_paths_from_manifest() -> list:
    """Get latent paths by converting audio paths from manifest."""
    print(f"Loading manifest from {MANIFEST_FILE}...")

    if not MANIFEST_FILE.exists():
        print(f"Manifest not found: {MANIFEST_FILE}")
        return []

    with open(MANIFEST_FILE, 'rb') as f:
        manifest = orjson.loads(f.read())

    entries = manifest.get("entries", [])
    print(f"Found {len(entries)} entries in manifest")

    # Vectorized path conversion
    latent_paths = []
    for entry in entries:
        audio_path = entry.get("audio_path", "")
        if not audio_path:
            continue

        # Convert: /gcs-bucket/X -> /gcs-bucket/Latents/X
        latent_path = audio_path.replace("/gcs-bucket/", "/gcs-bucket/Latents/")
        latent_path = latent_path.rsplit('.', 1)[0] + '.dcae.pt'
        latent_paths.append((latent_path, audio_path))

    print(f"Generated {len(latent_paths)} latent paths")
    return latent_paths


def detect_silent_files(latent_audio_pairs: list, threshold: float = SILENCE_THRESHOLD, num_workers: int = None) -> dict:
    """Analyze latent files using high-parallelism threading."""
    results = {
        "silent": [],
        "near_silent": [],
        "normal": 0,
        "errors": 0,
        "missing": 0,
        "total": len(latent_audio_pairs),
    }

    energy_values = []

    # Use many threads - this is I/O bound on GCS
    if num_workers is None:
        num_workers = 64  # High parallelism for GCS

    print(f"Analyzing {len(latent_audio_pairs)} latent files with {num_workers} threads...")

    # Prepare work items
    work_items = [(lp, ap, threshold, NEAR_SILENCE_THRESHOLD) for lp, ap in latent_audio_pairs]

    # Use ThreadPoolExecutor with map for better performance
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Process with progress bar
        for status, audio_path, latent_path_str, energy in tqdm(
            executor.map(process_single_file, work_items),
            total=len(work_items),
            desc="Analyzing latents",
            smoothing=0.1
        ):
            if status == "missing":
                results["missing"] += 1
            elif status == "error":
                results["errors"] += 1
            elif status == "silent":
                energy_values.append(energy)
                results["silent"].append({
                    "path": audio_path,
                    "latent_path": latent_path_str,
                    "energy": energy,
                    "is_silent": True,
                })
            elif status == "near_silent":
                energy_values.append(energy)
                results["near_silent"].append({
                    "path": audio_path,
                    "latent_path": latent_path_str,
                    "energy": energy,
                    "is_near_silent": True,
                })
            else:  # normal
                energy_values.append(energy)
                results["normal"] += 1

    # Add statistics
    if energy_values:
        ev = np.array(energy_values)
        results["stats"] = {
            "min_energy": float(ev.min()),
            "max_energy": float(ev.max()),
            "mean_energy": float(ev.mean()),
            "median_energy": float(np.median(ev)),
            "std_energy": float(ev.std()),
            "silence_threshold": threshold,
            "near_silence_threshold": NEAR_SILENCE_THRESHOLD,
        }

    return results


def update_manifest_with_silence(results: dict):
    """Update the manifest to mark silent files."""
    if not MANIFEST_FILE.exists():
        print(f"Manifest not found: {MANIFEST_FILE}")
        return

    # Build lookup of silent paths
    silent_paths = set()
    for entry in results.get("silent", []):
        silent_paths.add(entry["path"])
    for entry in results.get("near_silent", []):
        silent_paths.add(entry["path"])

    print(f"Marking {len(silent_paths)} files as silent in manifest...")

    with open(MANIFEST_FILE, 'rb') as f:
        manifest = orjson.loads(f.read())

    updated = 0
    for entry in manifest.get("entries", []):
        audio_path = entry.get("audio_path", "")
        if audio_path in silent_paths:
            entry["is_silent"] = True
            updated += 1

    with open(MANIFEST_FILE, 'wb') as f:
        f.write(orjson.dumps(manifest, option=orjson.OPT_INDENT_2))

    print(f"Updated {updated} entries in manifest")


def main():
    parser = argparse.ArgumentParser(description="Detect silent audio files from latents")
    parser.add_argument("--mode", choices=["detect", "stats", "update_manifest"],
                        default="detect", help="Mode of operation")
    parser.add_argument("--threshold", type=float, default=SILENCE_THRESHOLD,
                        help=f"Silence threshold (default: {SILENCE_THRESHOLD})")
    parser.add_argument("--workers", type=int, default=None,
                        help="Number of threads (default: 64)")
    args = parser.parse_args()

    threshold = args.threshold
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "detect":
        latent_audio_pairs = get_latent_paths_from_manifest()
        results = detect_silent_files(latent_audio_pairs, threshold=threshold, num_workers=args.workers)

        with open(OUTPUT_FILE, 'wb') as f:
            f.write(orjson.dumps(results, option=orjson.OPT_INDENT_2))

        print(f"\n{'='*50}")
        print("SILENCE DETECTION RESULTS")
        print(f"{'='*50}")
        print(f"Total in manifest: {results['total']}")
        print(f"Missing latents: {results.get('missing', 0)}")
        print(f"Silent files: {len(results['silent'])}")
        print(f"Near-silent files: {len(results['near_silent'])}")
        print(f"Normal files: {results['normal']}")
        print(f"Errors: {results['errors']}")
        if results.get("stats"):
            print(f"\nEnergy Statistics:")
            print(f"  Min: {results['stats']['min_energy']:.6f}")
            print(f"  Max: {results['stats']['max_energy']:.6f}")
            print(f"  Mean: {results['stats']['mean_energy']:.6f}")
            print(f"  Median: {results['stats']['median_energy']:.6f}")
        print(f"\nResults saved to: {OUTPUT_FILE}")

        update_manifest_with_silence(results)

    elif args.mode == "stats":
        if not OUTPUT_FILE.exists():
            print(f"No results file found. Run with --mode detect first.")
            return

        with open(OUTPUT_FILE, 'rb') as f:
            results = orjson.loads(f.read())

        print(f"Silent files: {len(results.get('silent', []))}")
        print(f"Near-silent files: {len(results.get('near_silent', []))}")
        print(f"Normal files: {results.get('normal', 0)}")
        if results.get("stats"):
            print(f"\nEnergy Statistics:")
            for k, v in results["stats"].items():
                print(f"  {k}: {v}")

    elif args.mode == "update_manifest":
        if not OUTPUT_FILE.exists():
            print(f"No results file found. Run with --mode detect first.")
            return

        with open(OUTPUT_FILE, 'rb') as f:
            results = orjson.loads(f.read())

        update_manifest_with_silence(results)


if __name__ == "__main__":
    main()
