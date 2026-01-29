#!/usr/bin/env python3
"""
Format Manifest Builder - Maps audio files to their derived format pairs.

Scans audio files in protools/ and protoolsA/, checks for corresponding:
- Latents (.pt)
- Conditioning (.amp.npy, .f0.npy, .f0_masked.npy, .onsets.npy, .rbend.npy, .rframe.npy)
- MIDI/BasicPitch (.mid)

Uses gsutil for fast GCS bucket listing.
Outputs encrypted cache for monitor service.
"""

import os
import sys
import json
import time
import hashlib
import base64
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configuration
GCS_BUCKET = "gs://ptxsessiondata"
LOCAL_MOUNT = Path("/home/arlo/gcs-bucket")
AUDIO_SOURCES = ["protools", "protoolsA"]
AUDIO_EXTENSIONS = {".wav", ".mp3", ".aiff", ".flac", ".aif"}

# Derived format prefixes in bucket
FORMAT_PREFIXES = {
    "latent": "Latents",
    "conditioning": "Conditioning",
    "midi": "BasicPitch"
}

# Conditioning suffixes
CONDITIONING_SUFFIXES = ["amp", "f0", "f0_masked", "onsets", "rbend", "rframe"]

# Output
CACHE_FILE = Path("/tmp/.format_cache.enc")
MANIFEST_FILE = LOCAL_MOUNT / "Manifests" / "format_manifest.json"


def get_encryption_key():
    """Generate machine-specific encryption key."""
    machine_id = os.popen("cat /etc/machine-id 2>/dev/null || hostname").read().strip()
    return hashlib.sha256(f"format_cache_{machine_id}".encode()).digest()


def encrypt(data: str) -> bytes:
    """Simple XOR encryption with key."""
    key = get_encryption_key()
    data_bytes = data.encode('utf-8')
    encrypted = bytes([data_bytes[i] ^ key[i % len(key)] for i in range(len(data_bytes))])
    return base64.b64encode(encrypted)


def gsutil_list_audio(source: str) -> list:
    """List all audio files in a source directory using local mount (faster & no wildcard issues)."""
    files = []
    local_path = LOCAL_MOUNT / source

    print(f"  Listing {source}/ via local mount...")
    start = time.time()

    try:
        # Build find command with all audio extensions
        ext_args = []
        for ext in AUDIO_EXTENSIONS:
            if ext_args:
                ext_args.append("-o")
            ext_args.extend(["-name", f"*{ext}"])

        # Use find command on local mount - handles special chars in filenames
        cmd = ["find", str(local_path), "-type", "f", "("] + ext_args + [")"]
        result = subprocess.run(
            cmd,
            capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0 and result.stderr:
            print(f"    Warning: find returned {result.returncode}")
            print(f"    stderr: {result.stderr[:200]}")

        mount_prefix = str(LOCAL_MOUNT) + "/"
        for line in result.stdout.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            # Convert local path to relative path
            if line.startswith(mount_prefix):
                rel_path = line[len(mount_prefix):]
                files.append(rel_path)

        elapsed = time.time() - start
        print(f"    Found {len(files):,} audio files in {elapsed:.1f}s")

    except subprocess.TimeoutExpired:
        print(f"    Warning: find timed out for {source}")
    except Exception as e:
        print(f"    Error listing {source}: {e}")

    return files


def build_format_sets():
    """Build sets of existing format files for fast lookup."""
    format_sets = {
        "latent": set(),
        "conditioning": set(),  # Base paths (without suffix)
        "midi": set()
    }

    print("Building format lookup sets...")

    # Define patterns for each format type
    patterns = {
        "latent": ("Latents", "**/*.pt"),
        "conditioning": ("Conditioning", "**/*.npy"),
        "midi": ("BasicPitch", "**/*.mid")
    }

    for format_type, (prefix, pattern) in patterns.items():
        bucket_pattern = f"{GCS_BUCKET}/{prefix}/{pattern}"
        print(f"  Scanning {prefix}/ for {pattern.split('/')[-1]}...")
        start = time.time()

        try:
            result = subprocess.run(
                ["gsutil", "ls", bucket_pattern],
                capture_output=True, text=True, timeout=600
            )

            count = 0
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue

                rel_path = line.replace(f"{GCS_BUCKET}/", "")

                if format_type == "latent" and rel_path.endswith(".pt"):
                    # Store without extension: Latents/protools/.../file
                    base = rel_path[:-3]  # Remove .pt
                    format_sets["latent"].add(base)
                    count += 1

                elif format_type == "conditioning" and rel_path.endswith(".npy"):
                    # Store base path without suffix: Conditioning/protools/.../file
                    # file.amp.npy -> file
                    parts = Path(rel_path).name.rsplit('.', 2)
                    if len(parts) >= 3 and parts[1] in CONDITIONING_SUFFIXES:
                        base = str(Path(rel_path).parent / parts[0])
                        format_sets["conditioning"].add(base)
                        count += 1

                elif format_type == "midi" and rel_path.endswith(".mid"):
                    # Store without extension
                    base = rel_path[:-4]  # Remove .mid
                    format_sets["midi"].add(base)
                    count += 1

            elapsed = time.time() - start
            unique = len(format_sets[format_type])
            print(f"    Found {count:,} files ({unique:,} unique bases) in {elapsed:.1f}s")

        except subprocess.TimeoutExpired:
            print(f"    Warning: gsutil timed out for {prefix}")
        except Exception as e:
            print(f"    Error: {e}")

    return format_sets


def check_formats(audio_path: str, format_sets: dict) -> dict:
    """Check which format files exist for an audio file."""
    # audio_path: protools/2025-03-28/New/Session/Audio Files/track.wav
    parts = audio_path.split('/', 1)
    if len(parts) < 2:
        return None

    source = parts[0]  # protools or protoolsA
    rel_after_source = parts[1]  # 2025-03-28/New/Session/Audio Files/track.wav

    # Get stem (filename without extension)
    stem = Path(audio_path).stem
    parent = str(Path(audio_path).parent)

    result = {
        "has_latent": False,
        "has_conditioning": False,
        "has_midi": False,
    }

    # Latent: Latents/protools/2025-03-28/.../track (no .pt)
    latent_base = f"Latents/{audio_path}".rsplit('.', 1)[0]
    if latent_base in format_sets["latent"]:
        result["has_latent"] = True

    # Conditioning: Conditioning/protools/2025-03-28/.../track
    cond_base = f"Conditioning/{parent}/{stem}"
    if cond_base in format_sets["conditioning"]:
        result["has_conditioning"] = True

    # MIDI: BasicPitch/protools/... OR BasicPitch/2025-... (legacy for protools)
    midi_base = f"BasicPitch/{audio_path}".rsplit('.', 1)[0]
    if midi_base in format_sets["midi"]:
        result["has_midi"] = True
    elif source == "protools":
        # Try legacy path: BasicPitch/2025-03-28/...
        legacy_base = f"BasicPitch/{rel_after_source}".rsplit('.', 1)[0]
        if legacy_base in format_sets["midi"]:
            result["has_midi"] = True

    return result


def build_manifest():
    """Build the complete format manifest."""
    print(f"Format Manifest Builder (gsutil edition)")
    print(f"=" * 50)
    print(f"Started at: {datetime.now().isoformat()}")
    print()

    # First, build lookup sets for all format files
    format_sets = build_format_sets()
    print()

    # Then scan audio files
    print("Scanning audio files...")
    all_audio = []
    for source in AUDIO_SOURCES:
        files = gsutil_list_audio(source)
        all_audio.extend(files)

    total_audio = len(all_audio)
    print(f"Total audio files: {total_audio:,}")
    print()

    if total_audio == 0:
        print("No audio files found!")
        return

    # Check formats for each audio file
    print("Checking format pairs...")
    start = time.time()

    stats = {
        "total_audio": total_audio,
        "with_latent": 0,
        "with_conditioning": 0,
        "with_midi": 0,
        "completeness": defaultdict(int),
        "by_source": defaultdict(int)
    }

    entries = []
    for i, audio_path in enumerate(all_audio):
        result = check_formats(audio_path, format_sets)
        if result is None:
            continue

        # Track stats
        source = audio_path.split('/')[0]
        stats["by_source"][source] += 1

        if result["has_latent"]:
            stats["with_latent"] += 1
        if result["has_conditioning"]:
            stats["with_conditioning"] += 1
        if result["has_midi"]:
            stats["with_midi"] += 1

        # Completeness category
        flags = []
        if result["has_latent"]:
            flags.append("L")
        if result["has_conditioning"]:
            flags.append("C")
        if result["has_midi"]:
            flags.append("M")

        if len(flags) == 3:
            key = "all_formats"
        elif len(flags) == 2:
            key = f"has_{'_'.join(flags)}"
        elif len(flags) == 1:
            key = f"only_{flags[0]}"
        else:
            key = "audio_only"

        stats["completeness"][key] += 1

        # Store entry
        entries.append({
            "path": audio_path,
            "source": source,
            **result
        })

        # Progress
        if (i + 1) % 50000 == 0:
            pct = ((i + 1) / total_audio) * 100
            print(f"  Progress: {i+1:,}/{total_audio:,} ({pct:.0f}%)")

    elapsed = time.time() - start
    print(f"  Completed in {elapsed:.1f}s")
    print()

    # Calculate percentages
    stats["pct_latent"] = (stats["with_latent"] / total_audio) * 100
    stats["pct_conditioning"] = (stats["with_conditioning"] / total_audio) * 100
    stats["pct_midi"] = (stats["with_midi"] / total_audio) * 100
    stats["needs_latent"] = total_audio - stats["with_latent"]
    stats["needs_conditioning"] = total_audio - stats["with_conditioning"]
    stats["needs_midi"] = total_audio - stats["with_midi"]
    stats["completeness"] = dict(stats["completeness"])
    stats["by_source"] = dict(stats["by_source"])

    # Save manifest
    print(f"Saving manifest to {MANIFEST_FILE}...")
    manifest = {
        "generated_at": time.time(),
        "generated_at_iso": datetime.now().isoformat(),
        "total_audio": total_audio,
        "stats": stats,
        "entries": entries
    }

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f)

    size_mb = MANIFEST_FILE.stat().st_size / (1024 * 1024)
    print(f"  Manifest size: {size_mb:.1f} MB")

    # Save encrypted cache (stats only)
    cache_data = {
        "generated_at": manifest["generated_at"],
        "total_audio": total_audio,
        "stats": stats
    }
    CACHE_FILE.write_bytes(encrypt(json.dumps(cache_data)))
    print(f"  Cache saved to {CACHE_FILE}")

    # Print summary
    print()
    print("=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total audio files: {stats['total_audio']:,}")
    print()
    print("Format coverage:")
    print(f"  With latent:       {stats['with_latent']:>7,} ({stats['pct_latent']:.1f}%)")
    print(f"  With conditioning: {stats['with_conditioning']:>7,} ({stats['pct_conditioning']:.1f}%)")
    print(f"  With MIDI:         {stats['with_midi']:>7,} ({stats['pct_midi']:.1f}%)")
    print()
    print("Completeness breakdown:")
    for key, count in sorted(stats['completeness'].items(), key=lambda x: -x[1]):
        pct = (count / stats['total_audio']) * 100
        print(f"  {key:20s}: {count:>7,} ({pct:.1f}%)")
    print()
    print("By source:")
    for source, count in stats['by_source'].items():
        print(f"  {source}: {count:,}")
    print()
    print(f"Completed at: {datetime.now().isoformat()}")


if __name__ == "__main__":
    build_manifest()
