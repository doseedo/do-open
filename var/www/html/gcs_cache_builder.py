#!/usr/bin/env python3
"""
GCS Bucket Cache Builder - Fast Version
Uses GCS API directly instead of slow FUSE mount.

Run this script to build/refresh the cache:
    python3 gcs_cache_builder.py

The cache is encrypted and saved to /tmp/.gcs_cache.enc
"""

import os
import sys
import json
import time
import hashlib
import base64
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# Try to import GCS library
try:
    from google.cloud import storage
    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    print("Warning: google-cloud-storage not installed. Install with:")
    print("  pip install google-cloud-storage")

# Configuration
GCS_BUCKET_NAME = "ptxsessiondata"
GCS_BUCKET_PATH = "/home/arlo/gcs-bucket"  # For fallback
CACHE_FILE = Path("/tmp/.gcs_cache.enc")

# Folder mapping (prefix -> display info)
FOLDERS = {
    "protools": {"name": "ProTools Sessions", "icon": "🎛️"},
    "protoolsA": {"name": "ProTools A", "icon": "🎚️"},
    "drum_bus": {"name": "Drum Bus", "icon": "🥁"},
    "drum_midi": {"name": "Drum MIDI", "icon": "🎹"},
    "BasicPitch": {"name": "Basic Pitch", "icon": "🎵"},
    "mel_spectrograms": {"name": "Mel Spectrograms", "icon": "📊"},
}

# Extensions to track
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aiff', '.aif', '.flac', '.ogg', '.m4a'}
MIDI_EXTENSIONS = {'.mid', '.midi'}
ALL_EXTENSIONS = AUDIO_EXTENSIONS | MIDI_EXTENSIONS | {'.npz', '.npy', '.json', '.txt', '.ptx', '.ptf'}

# Manifest file for instrument group/subgroup mapping (combined manifest)
MANIFEST_FILE = Path("/home/arlo/gcs-bucket/Manifests/combined_manifest.json")
_manifest_lookup = None  # Cached lookup from audio path to group/subgroup


def load_manifest_lookup():
    """Load manifest and build lookup from audio path to group/subgroup."""
    global _manifest_lookup
    if _manifest_lookup is not None:
        return _manifest_lookup

    _manifest_lookup = {}
    if not MANIFEST_FILE.exists():
        print(f"Warning: Manifest file not found: {MANIFEST_FILE}")
        return _manifest_lookup

    try:
        print(f"Loading instrument manifest from {MANIFEST_FILE}...")
        with open(MANIFEST_FILE) as f:
            manifest = json.load(f)

        # New format: {path: {group, subgroup, filename}}
        for audio_path, info in manifest.items():
            group = info.get("group", "undefined") or "undefined"
            subgroup = info.get("subgroup", "undefined") or "undefined"

            # Store with multiple key formats for matching
            # Full path
            _manifest_lookup[audio_path] = {"group": group, "subgroup": subgroup}

            # Relative path (after gcs-bucket/)
            if "/gcs-bucket/" in audio_path:
                rel_path = audio_path.split("/gcs-bucket/", 1)[1]
                _manifest_lookup[rel_path] = {"group": group, "subgroup": subgroup}

        print(f"  Loaded {len(manifest)} entries from manifest")
    except Exception as e:
        print(f"Error loading manifest: {e}")

    return _manifest_lookup


def get_instrument_info(blob_name):
    """Get group and subgroup for a blob from manifest lookup."""
    lookup = load_manifest_lookup()

    # Try direct match first
    info = lookup.get(blob_name)
    if info:
        return info["group"], info["subgroup"]

    # Try with full path prefix
    full_path = f"/home/arlo/gcs-bucket/{blob_name}"
    info = lookup.get(full_path)
    if info:
        return info["group"], info["subgroup"]

    return "undefined", "undefined"


def get_encryption_key():
    """Generate machine-specific encryption key."""
    machine_id = os.popen("cat /etc/machine-id 2>/dev/null || hostname").read().strip()
    return hashlib.sha256(f"gcs_cache_{machine_id}".encode()).digest()


def encrypt(data: str) -> bytes:
    """Simple XOR encryption with key."""
    key = get_encryption_key()
    data_bytes = data.encode('utf-8')
    encrypted = bytes([data_bytes[i] ^ key[i % len(key)] for i in range(len(data_bytes))])
    return base64.b64encode(encrypted)


def decrypt(data: bytes) -> str:
    """Simple XOR decryption."""
    key = get_encryption_key()
    encrypted = base64.b64decode(data)
    decrypted = bytes([encrypted[i] ^ key[i % len(key)] for i in range(len(encrypted))])
    return decrypted.decode('utf-8')


def format_size(size_bytes):
    """Format bytes to human readable string."""
    if size_bytes == 0:
        return "0 B"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != 'B' else f"{int(size_bytes)} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def format_audio_duration(seconds):
    """Format seconds to human readable duration."""
    if seconds < 60:
        return f"{int(seconds)} sec"
    elif seconds < 3600:
        return f"{seconds/60:.1f} min"
    else:
        hours = seconds / 3600
        if hours >= 1000:
            return f"{hours/1000:.1f}K hrs"
        return f"{hours:.1f} hrs"


def get_folder_from_path(blob_name):
    """Extract top-level folder from blob path."""
    parts = blob_name.split('/')
    return parts[0] if parts else None


def build_cache_gcs_api():
    """Build cache using GCS API - FAST method."""
    print(f"Connecting to GCS bucket: {GCS_BUCKET_NAME}")
    start_time = time.time()

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    # Initialize cache structure
    cache_data = {
        "generated_at": time.time(),
        "bucket_name": GCS_BUCKET_NAME,
        "bucket_path": GCS_BUCKET_PATH,
        "total_files": 0,
        "total_size": 0,
        "total_size_formatted": "0 B",
        "folder_count": 0,
        "modified_today": 0,
        "audio_seconds": 0,
        "audio_hours_formatted": "0 hrs",
        "instrument_hours": {},  # { group: { seconds, subgroups: { subgroup: seconds } } }
        "file_type_breakdown": {  # Breakdown by processing type
            "audio": {"count": 0, "size": 0, "label": "Audio Files"},
            "latent": {"count": 0, "size": 0, "label": "Latent Files"},
            "midi": {"count": 0, "size": 0, "label": "MIDI Files"},
            "conditioning": {"count": 0, "size": 0, "label": "Conditioning Files", "raw_count": 0}
        },
        "total_sessions": 0,  # Count of session folders (subfolders in date folders)
        "folders": {},
        "scan_method": "gcs_api"
    }

    # Track instrument hours
    instrument_stats = defaultdict(lambda: {
        "seconds": 0,
        "subgroups": defaultdict(float)
    })

    # Track unique sessions (folder/date/session paths)
    unique_sessions = set()

    # Initialize folder stats
    folder_stats = defaultdict(lambda: {
        "total_files": 0,
        "total_size": 0,
        "modified_today": 0,
        "audio_seconds": 0,
        "subfolders": set(),
        "subfolder_stats": defaultdict(lambda: {"files": 0, "size": 0}),
        "by_date": defaultdict(lambda: {"count": 0, "size": 0}),
        "by_ext": defaultdict(lambda: {"count": 0, "size": 0}),
        "files": []
    })

    # Estimate audio duration from file size (bytes per second for different formats)
    # WAV 44.1kHz 16-bit stereo = 176400 bytes/sec
    # WAV 48kHz 24-bit stereo = 288000 bytes/sec
    # MP3 ~16000 bytes/sec (128kbps)
    # FLAC ~80000 bytes/sec (varies)
    # AIFF similar to WAV
    BYTES_PER_SECOND = {
        '.wav': 176400,   # Assume 44.1kHz 16-bit stereo
        '.aiff': 176400,
        '.aif': 176400,
        '.mp3': 16000,    # ~128kbps
        '.flac': 80000,   # Variable, estimate
        '.ogg': 16000,
        '.m4a': 16000,
    }

    today = datetime.now(timezone.utc).date()
    file_count = 0

    print("Listing all objects (this is fast via API)...")

    # List ALL blobs in bucket - single API call with pagination
    blobs = bucket.list_blobs()

    for blob in blobs:
        file_count += 1

        # Progress indicator every 1000 files
        if file_count % 1000 == 0:
            print(f"  Processed {file_count} objects...", flush=True)

        # Get folder
        folder = get_folder_from_path(blob.name)
        if not folder:
            continue

        # Get file extension
        ext = os.path.splitext(blob.name)[1].lower()

        # Update folder stats
        stats = folder_stats[folder]
        stats["total_files"] += 1
        stats["total_size"] += blob.size or 0

        # Check modified today
        if blob.updated:
            blob_date = blob.updated.date()
            if blob_date == today:
                stats["modified_today"] += 1

        # Track subfolders and aggregate by subfolder
        parts = blob.name.split('/')
        if len(parts) > 1:
            subfolder = parts[1]
            stats["subfolders"].add(subfolder)
            stats["subfolder_stats"][subfolder]["files"] += 1
            stats["subfolder_stats"][subfolder]["size"] += blob.size or 0

        # Aggregate by date (extract date from path like "protools/2025-03-28/...")
        if len(parts) > 1:
            for part in parts[1:4]:  # Check first few path components for date
                if len(part) == 10 and part[4:5] == '-' and part[7:8] == '-':
                    try:
                        int(part[:4])
                        int(part[5:7])
                        int(part[8:10])
                        stats["by_date"][part]["count"] += 1
                        stats["by_date"][part]["size"] += blob.size or 0
                        break
                    except ValueError:
                        pass

        # Aggregate by extension
        if ext:
            stats["by_ext"][ext]["count"] += 1
            stats["by_ext"][ext]["size"] += blob.size or 0

        # Store file metadata (limit per folder - just for sample)
        if len(stats["files"]) < 100 and ext in ALL_EXTENSIONS:
            stats["files"].append({
                "name": os.path.basename(blob.name),
                "path": blob.name,
                "size": blob.size or 0,
                "ext": ext
            })

        # Estimate audio duration and track by instrument
        if ext in BYTES_PER_SECOND and blob.size:
            estimated_seconds = blob.size / BYTES_PER_SECOND[ext]
            stats["audio_seconds"] += estimated_seconds
            cache_data["audio_seconds"] += estimated_seconds

            # Track by instrument group/subgroup
            group, subgroup = get_instrument_info(blob.name)
            instrument_stats[group]["seconds"] += estimated_seconds
            instrument_stats[group]["subgroups"][subgroup] += estimated_seconds

        # Track file type breakdown by folder
        ftb = cache_data["file_type_breakdown"]
        if folder in ("protools", "protoolsA") and ext in AUDIO_EXTENSIONS:
            ftb["audio"]["count"] += 1
            ftb["audio"]["size"] += blob.size or 0
            # Track sessions: protools/date/newOrPrev/session_name/...
            if len(parts) >= 4:
                session_path = f"{parts[0]}/{parts[1]}/{parts[2]}/{parts[3]}"
                unique_sessions.add(session_path)
        elif folder == "Latents":
            ftb["latent"]["count"] += 1
            ftb["latent"]["size"] += blob.size or 0
        elif folder == "BasicPitch":
            ftb["midi"]["count"] += 1
            ftb["midi"]["size"] += blob.size or 0
        elif folder == "Conditioning":
            ftb["conditioning"]["raw_count"] += 1
            ftb["conditioning"]["size"] += blob.size or 0
            # Display count is raw_count / 6 (6 conditioning files per audio file)
            ftb["conditioning"]["count"] = ftb["conditioning"]["raw_count"] // 6

        # Update totals
        cache_data["total_files"] += 1
        cache_data["total_size"] += blob.size or 0
        if blob.updated and blob.updated.date() == today:
            cache_data["modified_today"] += 1

    print(f"  Total objects processed: {file_count}")

    # Build folder entries
    for folder_key, stats in folder_stats.items():
        folder_info = FOLDERS.get(folder_key, {"name": folder_key, "icon": "📂"})

        # Convert subfolder stats to list sorted by file count
        subfolders_list = [
            {"name": name, "files": data["files"], "size": data["size"], "size_formatted": format_size(data["size"])}
            for name, data in sorted(stats["subfolder_stats"].items(), key=lambda x: -x[1]["files"])
        ]

        cache_data["folders"][folder_key] = {
            "name": folder_info["name"],
            "icon": folder_info["icon"],
            "total_files": stats["total_files"],
            "total_size": stats["total_size"],
            "total_size_formatted": format_size(stats["total_size"]),
            "folder_count": len(stats["subfolders"]),
            "modified_today": stats["modified_today"],
            "audio_seconds": stats["audio_seconds"],
            "audio_hours_formatted": format_audio_duration(stats["audio_seconds"]),
            "subfolders": subfolders_list,
            "by_date": dict(stats["by_date"]),
            "by_ext": dict(stats["by_ext"]),
            "files": stats["files"]
        }

        cache_data["folder_count"] += len(stats["subfolders"]) + 1

        print(f"  {folder_info['name']}: {stats['total_files']} files, {format_size(stats['total_size'])}, {format_audio_duration(stats['audio_seconds'])}")

    cache_data["total_size_formatted"] = format_size(cache_data["total_size"])
    cache_data["audio_hours_formatted"] = format_audio_duration(cache_data["audio_seconds"])

    # Format file type breakdown sizes
    for key, data in cache_data["file_type_breakdown"].items():
        data["size_formatted"] = format_size(data["size"])

    # Save session count
    cache_data["total_sessions"] = len(unique_sessions)

    print(f"\nFile type breakdown:")
    for key, data in cache_data["file_type_breakdown"].items():
        count_display = data["count"]
        if key == "conditioning":
            print(f"  {data['label']}: {count_display:,} ({data['raw_count']:,} raw / 6), {data['size_formatted']}")
        else:
            print(f"  {data['label']}: {count_display:,}, {data['size_formatted']}")
    print(f"\nTotal sessions: {cache_data['total_sessions']:,}")

    # Build instrument hours data
    for group, data in sorted(instrument_stats.items(), key=lambda x: -x[1]["seconds"]):
        cache_data["instrument_hours"][group] = {
            "seconds": data["seconds"],
            "hours_formatted": format_audio_duration(data["seconds"]),
            "subgroups": {
                sg: {
                    "seconds": secs,
                    "hours_formatted": format_audio_duration(secs)
                }
                for sg, secs in sorted(data["subgroups"].items(), key=lambda x: -x[1])
            }
        }

    print(f"\nInstrument breakdown:")
    for group, data in cache_data["instrument_hours"].items():
        print(f"  {group}: {data['hours_formatted']}")
        for sg, sg_data in data["subgroups"].items():
            print(f"    - {sg}: {sg_data['hours_formatted']}")

    elapsed = time.time() - start_time
    print(f"\n✅ Scan complete in {elapsed:.1f} seconds")
    print(f"Total: {cache_data['total_files']:,} files, {cache_data['total_size_formatted']}, {cache_data['audio_hours_formatted']} audio")
    print(f"Folders: {cache_data['folder_count']}, Modified today: {cache_data['modified_today']}")

    return cache_data


def build_cache_gsutil():
    """Build cache using gsutil - medium speed fallback."""
    import subprocess

    print(f"Using gsutil to list bucket: gs://{GCS_BUCKET_NAME}")
    start_time = time.time()

    cache_data = {
        "generated_at": time.time(),
        "bucket_name": GCS_BUCKET_NAME,
        "bucket_path": GCS_BUCKET_PATH,
        "total_files": 0,
        "total_size": 0,
        "total_size_formatted": "0 B",
        "folder_count": 0,
        "modified_today": 0,
        "folders": {},
        "scan_method": "gsutil"
    }

    folder_stats = defaultdict(lambda: {
        "total_files": 0,
        "total_size": 0,
        "files": []
    })

    try:
        # gsutil ls -lR gives size and path
        result = subprocess.run(
            ["gsutil", "ls", "-lR", f"gs://{GCS_BUCKET_NAME}/"],
            capture_output=True, text=True, timeout=300
        )

        for line in result.stdout.split('\n'):
            line = line.strip()
            if not line or line.startswith('TOTAL:') or line.endswith(':'):
                continue

            parts = line.split()
            if len(parts) >= 3:
                try:
                    size = int(parts[0])
                    path = parts[-1].replace(f"gs://{GCS_BUCKET_NAME}/", "")
                    folder = get_folder_from_path(path)

                    if folder:
                        stats = folder_stats[folder]
                        stats["total_files"] += 1
                        stats["total_size"] += size

                        cache_data["total_files"] += 1
                        cache_data["total_size"] += size
                except (ValueError, IndexError):
                    continue

        # Build folder entries
        for folder_key, stats in folder_stats.items():
            folder_info = FOLDERS.get(folder_key, {"name": folder_key, "icon": "📂"})
            cache_data["folders"][folder_key] = {
                "name": folder_info["name"],
                "icon": folder_info["icon"],
                "total_files": stats["total_files"],
                "total_size": stats["total_size"],
                "total_size_formatted": format_size(stats["total_size"]),
                "folder_count": 0,
                "modified_today": 0,
                "files": []
            }
            cache_data["folder_count"] += 1

        cache_data["total_size_formatted"] = format_size(cache_data["total_size"])

    except Exception as e:
        print(f"gsutil error: {e}")
        return None

    elapsed = time.time() - start_time
    print(f"\n✅ Scan complete in {elapsed:.1f} seconds")
    print(f"Total: {cache_data['total_files']:,} files, {cache_data['total_size_formatted']}")

    return cache_data


def incremental_update(existing_cache):
    """Update cache with only new/modified files since last scan."""
    if not HAS_GCS:
        print("GCS API not available - cannot do incremental update")
        return None

    last_scan_time = existing_cache.get("generated_at", 0)
    last_scan_dt = datetime.fromtimestamp(last_scan_time, tz=timezone.utc)

    print(f"Incremental update since: {last_scan_dt.isoformat()}")
    start_time = time.time()

    client = storage.Client()
    bucket = client.bucket(GCS_BUCKET_NAME)

    # Bytes per second estimates for audio duration
    BYTES_PER_SECOND = {
        '.wav': 176400, '.aiff': 176400, '.aif': 176400,
        '.mp3': 16000, '.flac': 80000, '.ogg': 16000, '.m4a': 16000,
    }

    today = datetime.now(timezone.utc).date()
    new_files = 0
    new_size = 0
    new_audio_seconds = 0
    new_modified_today = 0
    folder_updates = defaultdict(lambda: {
        "new_files": 0, "new_size": 0, "new_audio_seconds": 0, "new_modified_today": 0
    })

    print("Scanning for new/modified files...")

    # List all blobs and filter by updated time
    blobs = bucket.list_blobs()

    for blob in blobs:
        # Skip if not modified since last scan
        if blob.updated and blob.updated <= last_scan_dt:
            continue

        new_files += 1
        new_size += blob.size or 0

        if new_files % 100 == 0:
            print(f"  Found {new_files} new files...", flush=True)

        folder = get_folder_from_path(blob.name)
        if not folder:
            continue

        ext = os.path.splitext(blob.name)[1].lower()

        # Track folder updates
        folder_updates[folder]["new_files"] += 1
        folder_updates[folder]["new_size"] += blob.size or 0

        # Check modified today
        if blob.updated and blob.updated.date() == today:
            new_modified_today += 1
            folder_updates[folder]["new_modified_today"] += 1

        # Estimate audio duration
        if ext in BYTES_PER_SECOND and blob.size:
            estimated_seconds = blob.size / BYTES_PER_SECOND[ext]
            new_audio_seconds += estimated_seconds
            folder_updates[folder]["new_audio_seconds"] += estimated_seconds

    if new_files == 0:
        print("No new files found since last scan")
        # Still update the timestamp and modified_today
        existing_cache["generated_at"] = time.time()
        existing_cache["modified_today"] = 0  # Reset for new day

        # Recalculate modified_today for existing folders
        for folder_key, folder_data in existing_cache.get("folders", {}).items():
            folder_data["modified_today"] = 0

        return existing_cache

    print(f"\nFound {new_files} new files ({format_size(new_size)})")
    print(f"New audio duration: {format_audio_duration(new_audio_seconds)}")

    # Update cache with new stats
    existing_cache["total_files"] += new_files
    existing_cache["total_size"] += new_size
    existing_cache["total_size_formatted"] = format_size(existing_cache["total_size"])
    existing_cache["audio_seconds"] = existing_cache.get("audio_seconds", 0) + new_audio_seconds
    existing_cache["audio_hours_formatted"] = format_audio_duration(existing_cache["audio_seconds"])
    existing_cache["modified_today"] = new_modified_today
    existing_cache["generated_at"] = time.time()

    # Update folder stats
    for folder_key, updates in folder_updates.items():
        if folder_key in existing_cache.get("folders", {}):
            folder = existing_cache["folders"][folder_key]
            folder["total_files"] += updates["new_files"]
            folder["total_size"] += updates["new_size"]
            folder["total_size_formatted"] = format_size(folder["total_size"])
            folder["audio_seconds"] = folder.get("audio_seconds", 0) + updates["new_audio_seconds"]
            folder["audio_hours_formatted"] = format_audio_duration(folder["audio_seconds"])
            folder["modified_today"] = updates["new_modified_today"]
            print(f"  Updated {folder_key}: +{updates['new_files']} files")
        else:
            # New folder - create entry
            folder_info = FOLDERS.get(folder_key, {"name": folder_key, "icon": "📂"})
            existing_cache["folders"][folder_key] = {
                "name": folder_info["name"],
                "icon": folder_info["icon"],
                "total_files": updates["new_files"],
                "total_size": updates["new_size"],
                "total_size_formatted": format_size(updates["new_size"]),
                "folder_count": 0,
                "modified_today": updates["new_modified_today"],
                "audio_seconds": updates["new_audio_seconds"],
                "audio_hours_formatted": format_audio_duration(updates["new_audio_seconds"]),
                "subfolders": [],
                "by_date": {},
                "by_ext": {},
                "files": []
            }
            existing_cache["folder_count"] += 1
            print(f"  New folder {folder_key}: {updates['new_files']} files")

    elapsed = time.time() - start_time
    print(f"\n✅ Incremental update complete in {elapsed:.1f} seconds")
    print(f"Total now: {existing_cache['total_files']:,} files, {existing_cache['total_size_formatted']}")
    print(f"Audio: {existing_cache['audio_hours_formatted']}")

    return existing_cache


def save_cache(cache_data):
    """Save cache to encrypted file."""
    print(f"\nSaving encrypted cache to {CACHE_FILE}...")
    try:
        json_data = json.dumps(cache_data)
        encrypted = encrypt(json_data)
        CACHE_FILE.write_bytes(encrypted)
        print(f"Cache saved ({len(encrypted):,} bytes encrypted)")
        return True
    except Exception as e:
        print(f"ERROR saving cache: {e}")
        return False


def load_cache():
    """Load cache from encrypted file."""
    try:
        if CACHE_FILE.exists():
            decrypted = decrypt(CACHE_FILE.read_bytes())
            return json.loads(decrypted)
    except Exception as e:
        print(f"Error loading cache: {e}")
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(description="GCS Bucket Cache Builder")
    parser.add_argument("--full", action="store_true", help="Force full rebuild (no incremental)")
    parser.add_argument("--incremental", "-i", action="store_true", help="Incremental update only")
    args = parser.parse_args()

    print("=" * 60)
    print("GCS Bucket Cache Builder (Fast API Version)")
    print("=" * 60)

    # Check for existing cache
    existing = load_cache()
    if existing:
        age = time.time() - existing.get("generated_at", 0)
        age_str = f"{age/3600:.1f} hours" if age > 3600 else f"{age/60:.1f} minutes"
        method = existing.get("scan_method", "unknown")
        print(f"Existing cache found (age: {age_str}, method: {method})")
        print(f"  Files: {existing.get('total_files', 0):,}")
        print(f"  Size: {existing.get('total_size_formatted', 'unknown')}")
        print(f"  Audio: {existing.get('audio_hours_formatted', 'unknown')}")

    cache_data = None

    # Determine update mode
    if args.full:
        print("\n--full specified: Forcing full rebuild...")
    elif existing and (args.incremental or (age < 86400)):  # 24 hours
        # Use incremental if cache is less than 24 hours old or --incremental specified
        print("\nPerforming incremental update...")
        try:
            cache_data = incremental_update(existing)
        except Exception as e:
            print(f"Incremental update failed: {e}")
            print("Falling back to full rebuild...")
            cache_data = None

    # Full rebuild if no cache or incremental failed
    if cache_data is None:
        print("\nBuilding full cache...")

        if HAS_GCS:
            try:
                cache_data = build_cache_gcs_api()
            except Exception as e:
                print(f"GCS API failed: {e}")
                print("Falling back to gsutil...")

        if cache_data is None:
            try:
                cache_data = build_cache_gsutil()
            except Exception as e:
                print(f"gsutil failed: {e}")
                print("ERROR: No scan method available. Install google-cloud-storage:")
                print("  pip install google-cloud-storage")
                sys.exit(1)

    if cache_data and save_cache(cache_data):
        print("\n" + "=" * 60)
        print("Cache build complete!")
        print(f"  Total files: {cache_data['total_files']:,}")
        print(f"  Total size: {cache_data['total_size_formatted']}")
        print(f"  Audio duration: {cache_data['audio_hours_formatted']}")
        print("=" * 60)
    else:
        print("\nCache build failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
