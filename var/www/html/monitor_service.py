#!/usr/bin/env python3
"""
Monitor Service - File browser and audio player backend
Runs on port 8096

Provides:
- /stats - Basic system stats
- /audio - Stream audio files from filesystem
- /browse - List files in a directory
"""

import os
import sys
import json
import orjson
import time
import threading
import mimetypes
import subprocess
import hashlib
import base64
import secrets
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Header, Request, Depends
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import shared auth utilities
sys.path.insert(0, '/home/arlo')
try:
    from shared_auth import get_current_user, get_current_user_optional
except ImportError:
    # Fallback if shared_auth not available
    async def get_current_user(request: Request):
        return {"user_id": 1, "username": "admin", "is_pro": True}
    async def get_current_user_optional(request: Request):
        return None
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

app = FastAPI(title="Monitor Service", version="1.0.0")

# Persistent encrypted cache config
CACHE_FILE = Path("/tmp/.monitor_cache.enc")
GCS_CACHE_FILE = Path("/tmp/.gcs_cache.enc")
FORMAT_CACHE_FILE = Path("/tmp/.format_cache.enc")
CACHE_TTL = 3600  # 1 hour (persisted cache can be longer)
_audio_cache = {"count": 0, "last_updated": 0, "updating": False}
_gcs_cache = None  # Loaded from gcs_cache_builder.py output
_format_cache = None  # Loaded from format_manifest_builder.py output
_master_manifest_cache = {"data": None, "mtime": 0}  # Cached master manifest
_manifest_groups_cache = {"data": None, "mtime": 0}  # Cached manifest groups lookup
_session_instruments_cache = {"data": None, "mtime": 0}  # Cached session instruments map
_predictions_cache = {}  # Keyed by file path: {"data": ..., "mtime": ...}
_silent_files_cache = {"data": None, "mtime": 0}  # Cached silent file paths

# Simple encryption using machine-specific key
def _get_encryption_key():
    """Generate machine-specific encryption key."""
    machine_id = os.popen("cat /etc/machine-id 2>/dev/null || hostname").read().strip()
    return hashlib.sha256(f"monitor_cache_{machine_id}".encode()).digest()

def _encrypt(data: str) -> bytes:
    """Simple XOR encryption with key."""
    key = _get_encryption_key()
    data_bytes = data.encode('utf-8')
    encrypted = bytes([data_bytes[i] ^ key[i % len(key)] for i in range(len(data_bytes))])
    return base64.b64encode(encrypted)

def _decrypt(data: bytes) -> str:
    """Simple XOR decryption."""
    key = _get_encryption_key()
    encrypted = base64.b64decode(data)
    decrypted = bytes([encrypted[i] ^ key[i % len(key)] for i in range(len(encrypted))])
    return decrypted.decode('utf-8')

def _save_cache():
    """Save cache to encrypted file."""
    try:
        cache_data = json.dumps({
            "count": _audio_cache["count"],
            "last_updated": _audio_cache["last_updated"]
        })
        CACHE_FILE.write_bytes(_encrypt(cache_data))
    except Exception:
        pass

def _load_cache():
    """Load cache from encrypted file."""
    global _audio_cache
    try:
        if CACHE_FILE.exists():
            decrypted = _decrypt(CACHE_FILE.read_bytes())
            data = json.loads(decrypted)
            _audio_cache["count"] = data.get("count", 0)
            _audio_cache["last_updated"] = data.get("last_updated", 0)
    except Exception:
        pass

# Load cache on startup
_load_cache()

# API Response Encryption (AES-GCM for browser compatibility)
_RESPONSE_KEY = secrets.token_bytes(32)  # 256-bit key, regenerated on restart

def encrypt_response(data: dict) -> dict:
    """Encrypt response data using AES-GCM (Web Crypto compatible)."""
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    plaintext = json.dumps(data).encode()
    nonce = secrets.token_bytes(12)  # 96-bit nonce for GCM

    aesgcm = AESGCM(_RESPONSE_KEY)
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    return {
        "encrypted": True,
        "nonce": base64.b64encode(nonce).decode(),
        "data": base64.b64encode(ciphertext).decode()
    }


def _load_gcs_cache():
    """Load GCS cache from encrypted file (built by gcs_cache_builder.py)."""
    global _gcs_cache
    try:
        if GCS_CACHE_FILE.exists():
            # Use same encryption as gcs_cache_builder.py
            machine_id = os.popen("cat /etc/machine-id 2>/dev/null || hostname").read().strip()
            key = hashlib.sha256(f"gcs_cache_{machine_id}".encode()).digest()
            encrypted = base64.b64decode(GCS_CACHE_FILE.read_bytes())
            decrypted = bytes([encrypted[i] ^ key[i % len(key)] for i in range(len(encrypted))])
            _gcs_cache = json.loads(decrypted.decode('utf-8'))
            return _gcs_cache
    except Exception as e:
        print(f"Error loading GCS cache: {e}")
    return None


def _load_format_cache():
    """Load format cache from encrypted file (built by format_manifest_builder.py)."""
    global _format_cache
    try:
        if FORMAT_CACHE_FILE.exists():
            machine_id = os.popen("cat /etc/machine-id 2>/dev/null || hostname").read().strip()
            key = hashlib.sha256(f"format_cache_{machine_id}".encode()).digest()
            encrypted = base64.b64decode(FORMAT_CACHE_FILE.read_bytes())
            decrypted = bytes([encrypted[i] ^ key[i % len(key)] for i in range(len(encrypted))])
            _format_cache = json.loads(decrypted.decode('utf-8'))
            return _format_cache
    except Exception as e:
        print(f"Error loading format cache: {e}")
    return None


def get_format_stats():
    """Get format pairing stats from cache."""
    global _format_cache
    if _format_cache is None:
        _load_format_cache()
    return _format_cache


def get_gcs_stats():
    """Get GCS bucket stats from cache."""
    global _gcs_cache
    if _gcs_cache is None:
        _load_gcs_cache()
    return _gcs_cache


def get_manifest_groups():
    """Get cached manifest groups lookup (path -> group). Loads once on first call."""
    global _manifest_groups_cache
    manifest_path = MANIFESTS_DIR / "unified_manifest.json"

    if not manifest_path.exists():
        return {}

    # Check if file changed
    try:
        current_mtime = manifest_path.stat().st_mtime
    except Exception:
        return _manifest_groups_cache.get("data") or {}

    if _manifest_groups_cache["data"] is not None and _manifest_groups_cache["mtime"] == current_mtime:
        return _manifest_groups_cache["data"]

    # Load with orjson (much faster for large files)
    try:
        with open(manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())
        groups = {}
        for entry in manifest.get('entries', []):
            groups[entry.get('audio_path', '')] = entry.get('group', 'undefined')
        _manifest_groups_cache["data"] = groups
        _manifest_groups_cache["mtime"] = current_mtime
        print(f"Loaded manifest groups: {len(groups)} entries")
        return groups
    except Exception as e:
        print(f"Error loading manifest groups: {e}")
        return _manifest_groups_cache.get("data") or {}


def get_session_instruments():
    """Get cached session -> instruments map. Loads once on first call."""
    global _session_instruments_cache
    combined_path = MANIFESTS_DIR / "combined_manifest.json"

    if not combined_path.exists():
        return {}

    # Check if file changed
    try:
        current_mtime = combined_path.stat().st_mtime
    except Exception:
        return _session_instruments_cache.get("data") or {}

    if _session_instruments_cache["data"] is not None and _session_instruments_cache["mtime"] == current_mtime:
        return _session_instruments_cache["data"]

    # Load with orjson
    try:
        from collections import defaultdict
        with open(combined_path, 'rb') as f:
            combined_manifest = orjson.loads(f.read())

        sessions = defaultdict(set)
        exclude_groups = {'undefined', 'room', 'fx', 'click', 'silent', 'junk', 'ensemble', 'full-track'}

        for path, entry in combined_manifest.items():
            if isinstance(entry, dict):
                parts = Path(path).parts
                if 'Audio Files' in parts:
                    idx = parts.index('Audio Files')
                    if idx > 0:
                        session = parts[idx - 1]
                        group = entry.get('group', 'undefined')
                        if group not in exclude_groups:
                            sessions[session].add(group)

        result = {s: sorted(list(insts)) for s, insts in sessions.items()}
        _session_instruments_cache["data"] = result
        _session_instruments_cache["mtime"] = current_mtime
        print(f"Loaded session instruments: {len(result)} sessions")
        return result
    except Exception as e:
        print(f"Error loading session instruments: {e}")
        return _session_instruments_cache.get("data") or {}


def get_cached_predictions(predictions_file: Path) -> dict:
    """Load predictions with caching. Only reloads if file changed."""
    global _predictions_cache

    if not predictions_file.exists():
        return {}

    file_key = str(predictions_file)
    try:
        current_mtime = predictions_file.stat().st_mtime
    except Exception:
        return _predictions_cache.get(file_key, {}).get("data") or {}

    # Return cached if unchanged
    if file_key in _predictions_cache and _predictions_cache[file_key].get("mtime") == current_mtime:
        return _predictions_cache[file_key]["data"]

    # Load with orjson
    try:
        with open(predictions_file, 'rb') as f:
            data = orjson.loads(f.read())
        _predictions_cache[file_key] = {"data": data, "mtime": current_mtime}
        print(f"Loaded predictions: {predictions_file.name} ({len(data.get('predictions', data.get('results', [])))} entries)")
        return data
    except Exception as e:
        print(f"Error loading predictions {predictions_file}: {e}")
        return _predictions_cache.get(file_key, {}).get("data") or {}


SILENT_FILES_PATH = Path("/home/arlo/Data/silence_detector/silent_files.json")


def get_silent_files() -> set:
    """Get cached set of silent file paths. Skipped by classifiers and labeler."""
    global _silent_files_cache

    if not SILENT_FILES_PATH.exists():
        return set()

    try:
        current_mtime = SILENT_FILES_PATH.stat().st_mtime
    except Exception:
        return _silent_files_cache.get("data") or set()

    if _silent_files_cache["data"] is not None and _silent_files_cache["mtime"] == current_mtime:
        return _silent_files_cache["data"]

    # Load silent files
    try:
        with open(SILENT_FILES_PATH, 'rb') as f:
            data = orjson.loads(f.read())

        silent_paths = set()
        for entry in data.get("silent", []):
            silent_paths.add(entry.get("path", ""))
        for entry in data.get("near_silent", []):
            silent_paths.add(entry.get("path", ""))

        _silent_files_cache["data"] = silent_paths
        _silent_files_cache["mtime"] = current_mtime
        print(f"Loaded silent files: {len(silent_paths)} paths")
        return silent_paths
    except Exception as e:
        print(f"Error loading silent files: {e}")
        return _silent_files_cache.get("data") or set()


def _count_audio_files():
    """Count audio files efficiently using find command."""
    # Use find command - much faster than Python glob
    try:
        result = subprocess.run(
            ["find", "/home/arlo/do-repo", "-type", "f",
             "(", "-name", "*.wav", "-o", "-name", "*.mp3",
             "-o", "-name", "*.aiff", "-o", "-name", "*.flac", ")"],
            capture_output=True, text=True, timeout=60
        )
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except Exception:
        return 0


def _update_audio_cache_background():
    """Update cache in background thread."""
    if _audio_cache["updating"]:
        return
    _audio_cache["updating"] = True
    try:
        count = _count_audio_files()
        _audio_cache["count"] = count
        _audio_cache["last_updated"] = time.time()
        _save_cache()  # Persist to disk
    finally:
        _audio_cache["updating"] = False


def get_audio_count():
    """Get cached audio count, trigger background refresh if stale."""
    now = time.time()
    if now - _audio_cache["last_updated"] > CACHE_TTL:
        threading.Thread(target=_update_audio_cache_background, daemon=True).start()
    return _audio_cache["count"]

# Add CORS middleware - restricted to known origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://doseedo.com",
        "https://www.doseedo.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed base paths for security
ALLOWED_PATHS = [
    "/home/arlo/gcs-bucket",
    "/home/arlo/do-repo",
    "/tmp",
]

def is_path_allowed(path: str) -> bool:
    """Check if path is under an allowed directory."""
    abs_path = os.path.abspath(path)
    return any(abs_path.startswith(allowed) for allowed in ALLOWED_PATHS)


@app.get("/stats")
async def get_stats(encrypted: bool = Query(True, description="Return encrypted response")):
    """Get GCS bucket stats from preprocessed cache."""
    gcs = get_gcs_stats()

    if gcs:
        # Format folders for frontend
        folders = []
        for key, folder in gcs.get("folders", {}).items():
            folders.append({
                "id": key,
                "name": folder.get("name", key),
                "icon": folder.get("icon", "📂"),
                "files": folder.get("total_files", 0),
                "size": folder.get("total_size", 0),
                "size_formatted": folder.get("total_size_formatted", "0 B"),
                "subdirs": folder.get("folder_count", 0),
                "modified_today": folder.get("modified_today", 0),
                "audio_hours": folder.get("audio_hours_formatted", "0 hrs")
            })

        # Get format stats for the Formats box
        fmt = get_format_stats()
        format_summary = None
        if fmt:
            fmt_stats = fmt.get("stats", {})
            total_audio = fmt_stats.get("total_audio", 0)
            all_formats = fmt_stats.get("completeness", {}).get("all_formats", 0)
            format_summary = {
                "total_audio": total_audio,
                "all_formats": all_formats,
                "pct_complete": round((all_formats / total_audio * 100) if total_audio > 0 else 0, 1),
                "with_latent": fmt_stats.get("with_latent", 0),
                "with_conditioning": fmt_stats.get("with_conditioning", 0),
                "with_midi": fmt_stats.get("with_midi", 0),
                "pct_latent": round(fmt_stats.get("pct_latent", 0), 1),
                "pct_conditioning": round(fmt_stats.get("pct_conditioning", 0), 1),
                "pct_midi": round(fmt_stats.get("pct_midi", 0), 1),
            }

        response_data = {
            "status": "ok",
            "total_files": gcs.get("total_files", 0),
            "total_size": gcs.get("total_size", 0),
            "total_size_formatted": gcs.get("total_size_formatted", "0 B"),
            "folder_count": gcs.get("folder_count", 0),
            "modified_today": gcs.get("modified_today", 0),
            "audio_seconds": gcs.get("audio_seconds", 0),
            "audio_hours_formatted": gcs.get("audio_hours_formatted", "0 hrs"),
            "instrument_hours": gcs.get("instrument_hours", {}),
            "file_type_breakdown": gcs.get("file_type_breakdown", {}),
            "total_sessions": gcs.get("total_sessions", 0),
            "folders": folders,
            "format_summary": format_summary,
            "cache_age": time.time() - gcs.get("generated_at", 0),
            "generated_at": gcs.get("generated_at", 0),
            "service": "monitor",
            "version": "2.5.0"
        }
        return encrypt_response(response_data) if encrypted else response_data
    else:
        # No cache - return zeros and prompt to run cache builder
        response_data = {
            "status": "no_cache",
            "message": "Run gcs_cache_builder.py to build cache",
            "total_files": 0,
            "total_size": 0,
            "total_size_formatted": "0 B",
            "folder_count": 0,
            "modified_today": 0,
            "audio_seconds": 0,
            "audio_hours_formatted": "0 hrs",
            "file_type_breakdown": {},
            "total_sessions": 0,
            "folders": [],
            "format_summary": None,
            "service": "monitor",
            "version": "2.5.0"
        }
        return encrypt_response(response_data) if encrypted else response_data


@app.get("/stats/system")
async def get_system_stats():
    """Get system resource stats (CPU, memory, disk)."""
    try:
        import psutil

        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "status": "ok",
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_used_gb": round(disk.used / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2)
        }
    except ImportError:
        return {"status": "error", "message": "psutil not installed"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/stats/refresh")
async def refresh_cache():
    """Trigger a cache refresh by reloading from disk."""
    global _gcs_cache
    _gcs_cache = None
    result = _load_gcs_cache()
    if result:
        return {"status": "ok", "message": "Cache reloaded", "total_files": result.get("total_files", 0)}
    else:
        return {"status": "error", "message": "No cache found. Run gcs_cache_builder.py first."}


@app.post("/stats/incremental")
async def incremental_refresh():
    """Trigger an incremental cache update (only scans new files)."""
    import subprocess
    try:
        # Run gcs_cache_builder.py with incremental flag
        script_path = Path(__file__).parent / "gcs_cache_builder.py"
        result = subprocess.run(
            ["python3", str(script_path), "--incremental"],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        # Reload the updated cache
        global _gcs_cache
        _gcs_cache = None
        cache = _load_gcs_cache()

        if cache:
            return {
                "status": "ok",
                "message": "Incremental update complete",
                "total_files": cache.get("total_files", 0),
                "audio_hours_formatted": cache.get("audio_hours_formatted", "0 hrs"),
                "output": result.stdout[-500:] if result.stdout else ""
            }
        else:
            return {
                "status": "error",
                "message": "Incremental update failed",
                "stderr": result.stderr[-500:] if result.stderr else ""
            }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Update timed out (>5 min)"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/stats/timeline")
async def get_timeline(
    granularity: str = Query("day", description="Granularity: day or month"),
    metric: str = Query("hours", description="Metric: files, hours, or gb"),
    encrypted: bool = Query(False, description="Return encrypted response")
):
    """Get timeline data for charting files/hours/GB over time using existing by_date folder data."""
    gcs = get_gcs_stats()

    if not gcs:
        response_data = {
            "status": "no_data",
            "message": "Cache not available",
            "data": [],
            "sources": []
        }
        return encrypt_response(response_data) if encrypted else response_data

    folders = gcs.get("folders", {})

    # Audio source folders with their colors
    audio_sources = {
        "protools": {"name": "ProTools", "color": "#3b82f6"},      # blue
        "protoolsA": {"name": "ProTools A", "color": "#22c55e"},   # green
    }

    # Bytes per second for estimating audio duration (WAV 44.1kHz 16-bit stereo)
    BYTES_PER_SECOND = 176400

    # Aggregate by_date data from audio folders
    combined = {}  # {date: {sources: {source: {count, size}}}}

    for folder_id, source_info in audio_sources.items():
        folder_data = folders.get(folder_id, {})
        by_date = folder_data.get("by_date", {})

        for date_str, stats in by_date.items():
            if date_str not in combined:
                combined[date_str] = {"sources": {}}
            combined[date_str]["sources"][folder_id] = {
                "count": stats.get("count", 0),
                "size": stats.get("size", 0)
            }

    # Build response data based on granularity
    if granularity == "month":
        # Aggregate by month (YYYY-MM)
        monthly = {}
        for date_str, data in combined.items():
            month_key = date_str[:7]  # YYYY-MM
            if month_key not in monthly:
                monthly[month_key] = {"sources": {}}
            for source, sdata in data["sources"].items():
                if source not in monthly[month_key]["sources"]:
                    monthly[month_key]["sources"][source] = {"count": 0, "size": 0}
                monthly[month_key]["sources"][source]["count"] += sdata["count"]
                monthly[month_key]["sources"][source]["size"] += sdata["size"]
        source_data = monthly
    else:
        source_data = combined

    # Convert to array format for charting
    chart_data = []
    for date_key in sorted(source_data.keys()):
        entry = source_data[date_key]
        point = {"date": date_key, "total": 0, "sources": {}}

        for source_id in audio_sources.keys():
            sdata = entry.get("sources", {}).get(source_id, {})
            count = sdata.get("count", 0)
            size = sdata.get("size", 0)

            if metric == "files":
                value = count
            elif metric == "gb":
                value = round(size / (1024 ** 3), 2)
            else:  # hours
                value = round((size / BYTES_PER_SECOND) / 3600, 1)

            point["sources"][source_id] = value
            point["total"] += value

        if point["total"] > 0:
            chart_data.append(point)

    # Build sources metadata
    sources_meta = [
        {"id": sid, "name": sinfo["name"], "color": sinfo["color"]}
        for sid, sinfo in audio_sources.items()
    ]

    response_data = {
        "status": "ok",
        "granularity": granularity,
        "metric": metric,
        "data": chart_data,
        "sources": sources_meta,
        "total_dates": len(chart_data)
    }
    return encrypt_response(response_data) if encrypted else response_data


@app.get("/formats")
async def get_formats(encrypted: bool = Query(False, description="Return encrypted response")):
    """Get format pairing statistics (latent, conditioning, MIDI coverage)."""
    fmt = get_format_stats()

    if fmt:
        stats = fmt.get("stats", {})
        total = stats.get("total_audio", 0)

        # Build completeness breakdown for UI
        completeness = stats.get("completeness", {})
        breakdown = []
        labels = {
            "all_formats": ("All Formats", "L+C+M"),
            "has_L_C": ("Latent + Conditioning", "L+C"),
            "has_L_M": ("Latent + MIDI", "L+M"),
            "has_C_M": ("Conditioning + MIDI", "C+M"),
            "only_L": ("Latent Only", "L"),
            "only_C": ("Conditioning Only", "C"),
            "only_M": ("MIDI Only", "M"),
            "audio_only": ("Audio Only", "None"),
        }
        for key, (label, short) in labels.items():
            count = completeness.get(key, 0)
            if count > 0 or key in ["all_formats", "audio_only"]:
                pct = (count / total * 100) if total > 0 else 0
                breakdown.append({
                    "key": key,
                    "label": label,
                    "short": short,
                    "count": count,
                    "pct": round(pct, 1)
                })

        response_data = {
            "status": "ok",
            "total_audio": total,
            "with_latent": stats.get("with_latent", 0),
            "with_conditioning": stats.get("with_conditioning", 0),
            "with_midi": stats.get("with_midi", 0),
            "pct_latent": round(stats.get("pct_latent", 0), 1),
            "pct_conditioning": round(stats.get("pct_conditioning", 0), 1),
            "pct_midi": round(stats.get("pct_midi", 0), 1),
            "needs_latent": stats.get("needs_latent", 0),
            "needs_conditioning": stats.get("needs_conditioning", 0),
            "needs_midi": stats.get("needs_midi", 0),
            "breakdown": breakdown,
            "by_source": stats.get("by_source", {}),
            "cache_age": time.time() - fmt.get("generated_at", 0),
            "generated_at": fmt.get("generated_at", 0)
        }
        return encrypt_response(response_data) if encrypted else response_data
    else:
        response_data = {
            "status": "no_cache",
            "message": "Run format_manifest_builder.py to build cache",
            "total_audio": 0,
            "with_latent": 0,
            "with_conditioning": 0,
            "with_midi": 0,
            "breakdown": [],
            "by_source": {}
        }
        return encrypt_response(response_data) if encrypted else response_data


@app.post("/formats/refresh")
async def refresh_format_cache():
    """Trigger a format cache refresh by reloading from disk."""
    global _format_cache
    _format_cache = None
    result = _load_format_cache()
    if result:
        return {
            "status": "ok",
            "message": "Format cache reloaded",
            "total_audio": result.get("total_audio", 0)
        }
    else:
        return {
            "status": "error",
            "message": "No format cache found. Run format_manifest_builder.py first."
        }


@app.get("/audio")
async def stream_audio(path: str = Query(..., description="Absolute path to audio file")):
    """Stream an audio file from the filesystem."""

    # Security check
    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail=f"Access denied: path not in allowed directories")

    # Check file exists
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not os.path.isfile(path):
        raise HTTPException(status_code=400, detail=f"Not a file: {path}")

    # Get mime type
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type is None:
        # Default to audio/wav for unknown types
        ext = os.path.splitext(path)[1].lower()
        mime_map = {
            '.wav': 'audio/wav',
            '.mp3': 'audio/mpeg',
            '.aiff': 'audio/aiff',
            '.aif': 'audio/aiff',
            '.flac': 'audio/flac',
            '.ogg': 'audio/ogg',
            '.m4a': 'audio/mp4',
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')

    # Get file size for Content-Length header
    file_size = os.path.getsize(path)

    # Stream the file
    def iterfile():
        with open(path, 'rb') as f:
            while chunk := f.read(65536):  # 64KB chunks
                yield chunk

    return StreamingResponse(
        iterfile(),
        media_type=mime_type,
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges": "bytes",
            "Content-Disposition": f'inline; filename="{os.path.basename(path)}"'
        }
    )


@app.get("/browse")
async def browse_directory(path: str = Query(..., description="Absolute path to directory")):
    """List files and directories in a path."""

    # Security check
    if not is_path_allowed(path):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not os.path.isdir(path):
        # If it's a file, return file info
        stat = os.stat(path)
        return {
            "type": "file",
            "path": path,
            "name": os.path.basename(path),
            "size": stat.st_size,
            "modified": stat.st_mtime
        }

    # List directory contents
    items = []
    try:
        for name in sorted(os.listdir(path)):
            full_path = os.path.join(path, name)
            try:
                stat = os.stat(full_path)
                is_dir = os.path.isdir(full_path)
                items.append({
                    "name": name,
                    "path": full_path,
                    "type": "directory" if is_dir else "file",
                    "size": stat.st_size if not is_dir else None,
                    "modified": stat.st_mtime
                })
            except (PermissionError, OSError):
                continue
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")

    return {
        "type": "directory",
        "path": path,
        "items": items
    }


@app.get("/folder/{folder_id}")
async def get_folder_details(folder_id: str):
    """Get detailed stats for a specific folder."""
    gcs = get_gcs_stats()

    if not gcs:
        raise HTTPException(status_code=503, detail="Cache not available")

    folders = gcs.get("folders", {})
    if folder_id not in folders:
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder_id}")

    folder = folders[folder_id]

    # Use pre-aggregated data from cache
    by_date = folder.get("by_date", {})
    by_ext = folder.get("by_ext", {})
    subfolders = folder.get("subfolders", [])

    return {
        "id": folder_id,
        "name": folder.get("name", folder_id),
        "icon": folder.get("icon", "📁"),
        "total_files": folder.get("total_files", 0),
        "total_size": folder.get("total_size", 0),
        "total_size_formatted": folder.get("total_size_formatted", "0 B"),
        "folder_count": folder.get("folder_count", 0),
        "modified_today": folder.get("modified_today", 0),
        "audio_seconds": folder.get("audio_seconds", 0),
        "audio_hours_formatted": folder.get("audio_hours_formatted", "0 hrs"),
        "byType": by_ext,
        "byDate": by_date if by_date else None,
        "subfolders": subfolders
    }


@app.get("/decrypt-key")
async def get_decrypt_key():
    """Return the encryption key for authenticated clients.
    Only accessible behind nginx basic auth, so only authenticated users can decrypt.
    """
    return {
        "key": base64.b64encode(_RESPONSE_KEY).decode()
    }


# ===================== LABELER ENDPOINTS =====================

MANIFESTS_DIR = Path("/home/arlo/gcs-bucket/Manifests")
CORRECTIONS_FILE = MANIFESTS_DIR / "corrections.json"
CORRECTED_MANIFEST_FILE = MANIFESTS_DIR / "corrected_manifest.json"

# Classifier output directories - base dirs for each classifier type
CLASSIFIER_BASE_DIRS = {
    "instrument": Path("/home/arlo/Data/latent_classifier"),
    "subgroup": Path("/home/arlo/Data/latent_subgroup_classifier"),
    "multilabel": Path("/home/arlo/Data/multilabel_classifier"),
    "other": Path("/home/arlo/Data/other_classifier"),
}

# Legacy alias for backwards compatibility
CLASSIFIER_DIRS = CLASSIFIER_BASE_DIRS

# Groups to exclude from predictions display (meta-groups, not instruments)
EXCLUDED_PREDICTION_GROUPS = {'silent', 'junk', 'click', 'undefined', 'room', 'fx', 'review_vocals'}

# Legacy path for compatibility
PREDICTIONS_FILE = Path("/home/arlo/Data/latent_classifier/predictions.json")

# Mix/Isolated classification cache (from ensemble detector - latent-based)
_mix_isolated_cache = {"data": None, "mtime": 0}
ENSEMBLE_FILE = Path("/home/arlo/Data/ensemble_detector/ensemble_detections.json")


def get_mix_isolated_lookup() -> dict:
    """Load mix/isolated classification from ensemble detector (latent-based).

    The ensemble detector is trained to distinguish solo vs ensemble recordings.
    Returns dict mapping audio_path -> {"is_isolated": bool, "ensemble_probability": float}
    """
    global _mix_isolated_cache

    if not ENSEMBLE_FILE.exists():
        return {}

    # Check if we need to reload
    current_mtime = ENSEMBLE_FILE.stat().st_mtime
    if _mix_isolated_cache["data"] is not None and _mix_isolated_cache["mtime"] == current_mtime:
        return _mix_isolated_cache["data"]

    try:
        with open(ENSEMBLE_FILE, 'rb') as f:
            data = orjson.loads(f.read())

        lookup = {}
        threshold = data.get("threshold", 0.5)

        # Files detected as ensemble (mix)
        ensemble_paths = set()
        for entry in data.get("detected", []):
            path = entry.get("path", "")
            if path:
                ensemble_paths.add(path)
                lookup[path] = {
                    "is_isolated": False,
                    "ensemble_probability": entry.get("ensemble_probability", 1.0),
                }

        # Note: Files not in 'detected' list are isolated (below threshold)
        # But we don't have the full list of checked files in this JSON
        # The filter will treat missing entries as "unknown" for now

        _mix_isolated_cache["data"] = lookup
        _mix_isolated_cache["mtime"] = current_mtime
        _mix_isolated_cache["total_checked"] = data.get("total_checked", 0)
        _mix_isolated_cache["detected_count"] = len(ensemble_paths)
        return lookup
    except Exception as e:
        print(f"Error loading mix/isolated data: {e}")
        return {}


def get_classifier_versions(classifier_type: str) -> list:
    """Get all available versions for a classifier type.

    Looks for:
    - predictions.json (treated as 'v1' or 'current')
    - predictions_v1.json, predictions_v2.json, etc.
    - v1/, v2/ subdirectories with predictions.json
    """
    if classifier_type not in CLASSIFIER_BASE_DIRS:
        return []

    base_dir = CLASSIFIER_BASE_DIRS[classifier_type]
    if not base_dir.exists():
        return []

    versions = []

    # Check for default predictions.json (current/v1)
    if (base_dir / "predictions.json").exists():
        versions.append({
            "version": "current",
            "predictions_file": base_dir / "predictions.json",
            "validation_file": base_dir / "validation_results.json",
            "model_file": base_dir / "model.pt",
        })

    # Check for versioned files (predictions_v1.json, predictions_v2.json, etc.)
    for pred_file in sorted(base_dir.glob("predictions_v*.json")):
        version = pred_file.stem.replace("predictions_", "")  # e.g., "v1", "v2"
        val_file = base_dir / f"validation_results_{version}.json"
        model_file = base_dir / f"model_{version}.pt"
        versions.append({
            "version": version,
            "predictions_file": pred_file,
            "validation_file": val_file if val_file.exists() else None,
            "model_file": model_file if model_file.exists() else None,
        })

    # Check for versioned subdirectories (v1/, v2/, etc.)
    for subdir in sorted(base_dir.glob("v*")):
        if subdir.is_dir() and (subdir / "predictions.json").exists():
            version = subdir.name  # e.g., "v1", "v2"
            # Skip if we already have this version from files
            if any(v["version"] == version for v in versions):
                continue
            versions.append({
                "version": version,
                "predictions_file": subdir / "predictions.json",
                "validation_file": subdir / "validation_results.json" if (subdir / "validation_results.json").exists() else None,
                "model_file": subdir / "model.pt" if (subdir / "model.pt").exists() else None,
            })

    return versions


def get_version_files(classifier_type: str, version: str) -> dict:
    """Get the file paths for a specific classifier version."""
    versions = get_classifier_versions(classifier_type)
    for v in versions:
        if v["version"] == version:
            return v
    # Fallback to current
    if versions:
        return versions[0]
    return None


@app.get("/classifiers")
async def list_classifiers():
    """List available classifier predictions for review, including all versions."""
    classifiers = []

    for classifier_type, classifier_dir in CLASSIFIER_BASE_DIRS.items():
        versions = get_classifier_versions(classifier_type)

        # If no versions found, still report the classifier (might have model but no predictions yet)
        if not versions:
            model_file = classifier_dir / "model.pt"
            classifiers.append({
                "type": classifier_type,
                "version": "none",
                "id": classifier_type,  # Unique identifier for frontend
                "name": f"{classifier_type.title()} Classifier",
                "dir": str(classifier_dir),
                "has_model": model_file.exists() if classifier_dir.exists() else False,
                "has_predictions": False,
                "has_validation": False,
                "predictions_count": 0,
                "validation_count": 0,
                "flagged_count": 0,
            })
            continue

        # Add each version as a separate entry
        for ver_info in versions:
            version = ver_info["version"]
            predictions_file = ver_info["predictions_file"]
            validation_file = ver_info["validation_file"]
            model_file = ver_info["model_file"]

            # Create unique ID combining type and version
            classifier_id = f"{classifier_type}:{version}" if version != "current" else classifier_type

            # Build display name
            if version == "current":
                display_name = f"{classifier_type.title()} Classifier"
            else:
                display_name = f"{classifier_type.title()} Classifier ({version})"

            classifier_info = {
                "type": classifier_type,
                "version": version,
                "id": classifier_id,
                "name": display_name,
                "dir": str(classifier_dir),
                "has_model": model_file.exists() if model_file else False,
                "has_predictions": predictions_file.exists() if predictions_file else False,
                "has_validation": validation_file.exists() if validation_file else False,
                "predictions_count": 0,
                "validation_count": 0,
                "flagged_count": 0,
            }

            # Get prediction stats - read only header to avoid loading huge files
            if predictions_file and predictions_file.exists():
                try:
                    # Read only first 4KB to extract counts from header
                    with open(predictions_file) as f:
                        header = f.read(4096)

                    # Try to extract counts from header using regex (fast)
                    import re
                    total_match = re.search(r'"total"\s*:\s*(\d+)', header)
                    multilabel_match = re.search(r'"multilabel_count"\s*:\s*(\d+)', header)

                    if total_match:
                        classifier_info["predictions_count"] = int(total_match.group(1))
                    if multilabel_match:
                        classifier_info["multilabel_count"] = int(multilabel_match.group(1))

                    # Extract label_distribution if present in header
                    label_dist_match = re.search(r'"label_distribution"\s*:\s*\{([^}]+)\}', header)
                    if label_dist_match:
                        # Parse the label distribution
                        dist_str = label_dist_match.group(1)
                        label_dist = {}
                        for pair in re.findall(r'"(\w+)"\s*:\s*(\d+)', dist_str):
                            label_dist[pair[0]] = int(pair[1])
                        classifier_info["label_distribution"] = label_dist

                    # Only load full file if we couldn't get total from header (small files only)
                    if not total_match:
                        # File doesn't have total in header - check size first
                        file_size = predictions_file.stat().st_size
                        if file_size < 10_000_000:  # Only load files < 10MB
                            with open(predictions_file) as f:
                                data = json.load(f)
                            predictions = data.get("predictions", []) or data.get("results", [])
                            classifier_info["predictions_count"] = len(predictions)
                            classifier_info["predictions_summary"] = data.get("summary", {})
                        else:
                            classifier_info["predictions_count"] = -1  # Unknown
                            classifier_info["predictions_summary"] = {}

                    classifier_info["predictions_modified"] = predictions_file.stat().st_mtime
                except Exception as e:
                    print(f"Error loading predictions for {classifier_type}/{version}: {e}")

            # Get validation stats - read only header
            if validation_file and validation_file.exists():
                try:
                    with open(validation_file) as f:
                        header = f.read(2048)

                    # Extract counts from header using regex
                    import re
                    validated_match = re.search(r'"total_validated"\s*:\s*(\d+)', header)
                    flagged_match = re.search(r'"total_flagged"\s*:\s*(\d+)', header)

                    if validated_match:
                        classifier_info["validation_count"] = int(validated_match.group(1))
                    if flagged_match:
                        classifier_info["flagged_count"] = int(flagged_match.group(1))

                    classifier_info["validation_modified"] = validation_file.stat().st_mtime
                except Exception:
                    pass

            classifiers.append(classifier_info)

    # Add special classifiers not in CLASSIFIER_BASE_DIRS
    special_classifiers = [
        {
            "type": "subgroups",
            "version": "current",
            "id": "subgroups",
            "name": "Subgroup Classifiers",
            "endpoint": "/classifier/subgroups",
            "dir": "/home/arlo/Data/subgroup_classifiers",
            "has_model": True,
            "has_predictions": Path("/home/arlo/Data/subgroup_classifiers/all_predictions.json").exists(),
            "predictions_count": 25444,
            "description": "Classify within groups: brass→trumpet/trombone, strings→violin/cello, etc.",
        },
        {
            "type": "separated-stems",
            "version": "current",
            "id": "separated-stems",
            "name": "Separated Stems (Demucs + V3)",
            "endpoint": "/classifier/separated-stems",
            "dir": "/home/arlo/Data/mix_classifier",
            "has_model": True,
            "has_predictions": Path("/home/arlo/Data/mix_classifier/separated_stems_temporal.json").exists(),
            "predictions_count": 2962,
            "description": "8-category temporal: vocals, drums, bass, guitar, piano, brass, strings, winds",
        },
        {
            "type": "mix-classifier-v3",
            "version": "current",
            "id": "mix-classifier-v3",
            "name": "Mix Classifier V3",
            "endpoint": "/classifier/mix-classifier-v3",
            "dir": "/home/arlo/Data/mix_classifier",
            "has_model": True,
            "has_predictions": Path("/home/arlo/Data/mix_classifier/mix_classifier_v3_temporal.json").exists(),
            "description": "Temporal brass/strings/winds on mix files",
        },
    ]
    classifiers.extend(special_classifiers)

    return {"classifiers": classifiers}


@app.get("/classifier/{classifier_type}/predictions")
async def get_classifier_predictions(
    classifier_type: str,
    version: str = Query("current", description="Classifier version: current, v1, v2, etc."),
    confidence: str = Query("all", description="Filter: all, high, medium, low"),
    group: str = Query(None, description="Filter by predicted group"),
    subgroup: str = Query(None, description="Filter by predicted subgroup"),
    manifest_group: str = Query(None, description="Filter by original manifest group"),
    match_filter: str = Query(None, description="Filter: matches, mismatches"),
    multi_filter: str = Query(None, description="Filter: multi, single"),
    mix_filter: str = Query(None, description="Filter by filename: mix, room, mix_or_room"),
    isolated_filter: str = Query(None, description="Filter by mix/isolated: isolated, mix, unknown"),
    session_instrument: str = Query(None, description="Filter by session instrument"),
    sort_by: str = Query(None, description="Sort by: confidence_asc, confidence_desc"),
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get predictions from a specific classifier version for review."""
    if classifier_type not in CLASSIFIER_BASE_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown classifier: {classifier_type}")

    # Get versioned file paths
    ver_info = get_version_files(classifier_type, version)
    if not ver_info:
        return {
            "classifier": classifier_type,
            "version": version,
            "status": "no_predictions",
            "message": f"No predictions found for {classifier_type} version {version}",
            "entries": []
        }

    predictions_file = ver_info["predictions_file"]
    if not predictions_file or not predictions_file.exists():
        return {
            "classifier": classifier_type,
            "version": version,
            "status": "no_predictions",
            "message": f"Run {classifier_type} classifier first to generate predictions",
            "entries": []
        }

    # Load predictions with caching (only reloads if file changed)
    data = get_cached_predictions(predictions_file)
    if not data:
        raise HTTPException(status_code=500, detail=f"Error loading predictions")

    # Use cached manifest lookups (loaded once, much faster)
    manifest_groups = get_manifest_groups()
    session_instruments_map = get_session_instruments()
    mix_isolated_lookup = get_mix_isolated_lookup()
    silent_files = get_silent_files()

    entries = []

    # Handle multilabel classifier format (uses 'results' with 'predicted_labels' array)
    if classifier_type == "multilabel":
        for pred in data.get("results", []):
            # Skip silent files
            path = pred.get("path", "")
            audio_path = path.replace('.dcae.pt', '.wav').replace('/Latents/', '/')
            if audio_path in silent_files:
                continue
            predicted_labels = pred.get("predicted_labels", [])
            top_probs = pred.get("top_probabilities", {})
            # Get confidence as max probability of predicted labels
            confidence = max([top_probs.get(l, 0) for l in predicted_labels]) if predicted_labels else 0
            path = pred.get("path", "")
            manifest_group = manifest_groups.get(path, "unknown")
            predicted_group = predicted_labels[0] if predicted_labels else "undefined"
            # Get mix/isolated status from ensemble detector (latent-based)
            audio_path = path.replace('.dcae.pt', '.wav').replace('/Latents/', '/')
            mix_info = mix_isolated_lookup.get(audio_path)
            # If in lookup -> ensemble/mix, if not in lookup -> isolated (assuming detector ran)
            if mix_info is not None:
                is_isolated = mix_info.get("is_isolated", False)
                ensemble_prob = mix_info.get("ensemble_probability", 0)
            else:
                # Not in ensemble list = isolated (or not yet checked)
                is_isolated = True if mix_isolated_lookup else None  # None if no data
                ensemble_prob = 0
            entry = {
                "path": path,
                "predicted_group": predicted_group,
                "predicted_labels": predicted_labels,
                "manifest_group": manifest_group,  # Original label from manifest
                "matches_manifest": predicted_group == manifest_group,  # Comparison flag
                "is_multilabel": pred.get("is_multilabel", False),
                "confidence": confidence,
                "all_probabilities": pred.get("all_probabilities", {}),
                "filename": pred.get("filename", os.path.basename(path)),
                "is_isolated": is_isolated,
                "ensemble_probability": ensemble_prob,
            }
            entries.append(entry)
    else:
        # Standard single-label classifier format
        for pred in data.get("predictions", []):
            path = pred.get("path", "")
            # Skip silent files
            audio_path = path.replace('.dcae.pt', '.wav').replace('/Latents/', '/')
            if audio_path in silent_files:
                continue

            manifest_group = manifest_groups.get(path, "unknown")
            predicted_group = pred.get("predicted_group", "undefined")

            # Get session instruments
            session_instruments = []
            session_name = ""
            parts = Path(path).parts
            if 'Audio Files' in parts:
                idx = parts.index('Audio Files')
                if idx > 0:
                    session_name = parts[idx - 1]
                    session_instruments = session_instruments_map.get(session_name, [])

            # Get mix/isolated status from ensemble detector (latent-based)
            audio_path = path.replace('.dcae.pt', '.wav').replace('/Latents/', '/')
            mix_info = mix_isolated_lookup.get(audio_path)
            # If in lookup -> ensemble/mix, if not in lookup -> isolated (assuming detector ran)
            if mix_info is not None:
                is_isolated = mix_info.get("is_isolated", False)
                ensemble_prob = mix_info.get("ensemble_probability", 0)
            else:
                # Not in ensemble list = isolated (or not yet checked)
                is_isolated = True if mix_isolated_lookup else None  # None if no data
                ensemble_prob = 0

            entry = {
                "path": path,
                "predicted_group": predicted_group,
                "manifest_group": manifest_group,  # Original label from manifest
                "matches_manifest": predicted_group == manifest_group,  # Comparison flag
                "confidence": pred.get("confidence", 0),
                "all_probabilities": pred.get("all_probabilities", {}),
                "filename": os.path.basename(path),
                "is_multi": pred.get("is_multi", False),  # From binary multi classifier
                "multi_probability": pred.get("multi_probability", 0),
                "session_name": session_name,
                "session_instruments": session_instruments,
                "is_isolated": is_isolated,
                "ensemble_probability": ensemble_prob,
            }
            # For subgroup classifier
            if "predicted_subgroup" in pred:
                entry["predicted_subgroup"] = pred["predicted_subgroup"]
                entry["parent_group"] = pred.get("parent_group", "")
            entries.append(entry)

    # Get unique groups for filters
    all_groups = set()
    for e in entries:
        if "predicted_labels" in e:
            all_groups.update(e["predicted_labels"])
        else:
            all_groups.add(e["predicted_group"])
    all_groups = sorted(all_groups)

    # Get unique manifest groups for filter dropdown
    all_manifest_groups = sorted(set(e.get("manifest_group", "unknown") for e in entries))

    # Count matches vs mismatches
    match_count = sum(1 for e in entries if e.get("matches_manifest"))
    mismatch_count = len(entries) - match_count

    # Count multi vs single
    multi_count = sum(1 for e in entries if e.get("is_multi"))
    single_count = len(entries) - multi_count

    # Count isolated vs mix vs unknown
    isolated_count = sum(1 for e in entries if e.get("is_isolated") is True)
    mix_count = sum(1 for e in entries if e.get("is_isolated") is False)
    unknown_count = sum(1 for e in entries if e.get("is_isolated") is None)

    # Apply filters
    filtered = entries

    # PIPELINE ENFORCEMENT: Group/Subgroup classifiers are for ISOLATED files only
    # Mix files should use Demucs pipeline (stem energy → other classifier)
    if classifier_type in ("instrument", "subgroup") and isolated_filter != "mix":
        filtered = [e for e in filtered if e.get("is_isolated") is not False]

    if group:
        if classifier_type == "multilabel":
            filtered = [e for e in filtered if group in e.get("predicted_labels", [])]
        else:
            filtered = [e for e in filtered if e["predicted_group"] == group]

    if subgroup:
        filtered = [e for e in filtered if e.get("predicted_subgroup") == subgroup]

    if manifest_group:
        filtered = [e for e in filtered if e.get("manifest_group") == manifest_group]

    if match_filter == "matches":
        filtered = [e for e in filtered if e.get("matches_manifest")]
    elif match_filter == "mismatches":
        filtered = [e for e in filtered if not e.get("matches_manifest")]

    if confidence == "high":
        filtered = [e for e in filtered if e["confidence"] >= 0.85]
    elif confidence == "medium":
        filtered = [e for e in filtered if 0.65 <= e["confidence"] < 0.85]
    elif confidence == "low":
        filtered = [e for e in filtered if e["confidence"] < 0.65]

    if multi_filter == "multi":
        filtered = [e for e in filtered if e.get("is_multi")]
    elif multi_filter == "single":
        filtered = [e for e in filtered if not e.get("is_multi")]

    # Filter by filename containing mix/room
    if mix_filter == "mix":
        filtered = [e for e in filtered if "mix" in e.get("filename", "").lower()]
    elif mix_filter == "room":
        filtered = [e for e in filtered if "room" in e.get("filename", "").lower()]
    elif mix_filter == "mix_or_room":
        filtered = [e for e in filtered if "mix" in e.get("filename", "").lower() or "room" in e.get("filename", "").lower()]

    # Filter by mix/isolated classification (based on stem separation analysis)
    if isolated_filter == "isolated":
        filtered = [e for e in filtered if e.get("is_isolated") is True]
    elif isolated_filter == "mix":
        filtered = [e for e in filtered if e.get("is_isolated") is False]
    elif isolated_filter == "unknown":
        filtered = [e for e in filtered if e.get("is_isolated") is None]

    # Filter by session instrument
    if session_instrument:
        filtered = [e for e in filtered if session_instrument in e.get("session_instruments", [])]

    # Collect all session instruments for filter dropdown
    all_session_instruments = set()
    for e in entries:
        all_session_instruments.update(e.get("session_instruments", []))
    all_session_instruments = sorted(all_session_instruments)

    # Collect available subgroups
    all_subgroups = sorted(set(e.get("predicted_subgroup", "") for e in entries if e.get("predicted_subgroup")))

    # Sort by confidence or other criteria
    if sort_by == "confidence_desc":
        filtered.sort(key=lambda x: x["confidence"], reverse=True)
    elif sort_by == "confidence_asc":
        filtered.sort(key=lambda x: x["confidence"])
    elif sort_by == "group":
        filtered.sort(key=lambda x: (x.get("predicted_group", ""), -x["confidence"]))
    elif sort_by == "subgroup":
        filtered.sort(key=lambda x: (x.get("predicted_subgroup") or "", -x["confidence"]))
    elif sort_by == "isolated_first":
        # Isolated files first (is_isolated=True), then by confidence
        filtered.sort(key=lambda x: (0 if x.get("is_isolated") is True else 1, x["confidence"]))
    elif sort_by == "mix_first":
        # Mix files first (is_isolated=False), then by ensemble probability desc
        filtered.sort(key=lambda x: (0 if x.get("is_isolated") is False else 1, -x.get("ensemble_probability", 0)))
    elif sort_by == "ensemble_prob_desc":
        # Highest ensemble probability first (most likely to be mix)
        filtered.sort(key=lambda x: -x.get("ensemble_probability", 0))
    elif sort_by == "ensemble_prob_asc":
        # Lowest ensemble probability first (most likely to be isolated)
        filtered.sort(key=lambda x: x.get("ensemble_probability", 0))
    else:
        # Default: low confidence first for review (these need most attention)
        filtered.sort(key=lambda x: x["confidence"])

    total = len(filtered)
    filtered = filtered[offset:offset + limit]

    return {
        "classifier": classifier_type,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": filtered,
        "available_groups": all_groups,
        "available_manifest_groups": all_manifest_groups,
        "match_stats": {
            "matches": match_count,
            "mismatches": mismatch_count,
            "match_rate": round(match_count / len(entries) * 100, 1) if entries else 0
        },
        "multi_stats": {
            "multi": multi_count,
            "single": single_count,
            "multi_rate": round(multi_count / len(entries) * 100, 1) if entries else 0
        },
        "isolated_stats": {
            "isolated": isolated_count,
            "mix": mix_count,
            "unknown": unknown_count,
            "analyzed_rate": round((isolated_count + mix_count) / len(entries) * 100, 1) if entries else 0
        },
        "available_session_instruments": all_session_instruments,
        "available_subgroups": all_subgroups,
        "summary": data.get("summary", {})
    }


@app.get("/classifier/{classifier_type}/flagged")
async def get_flagged_entries(
    classifier_type: str,
    version: str = Query("current", description="Classifier version: current, v1, v2, etc."),
    flag_type: str = Query("all", description="Filter: all, disagreement, uncertain, outlier"),
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get flagged entries from validation results for review."""
    if classifier_type not in CLASSIFIER_BASE_DIRS:
        raise HTTPException(status_code=404, detail=f"Unknown classifier: {classifier_type}")

    # Get versioned file paths
    ver_info = get_version_files(classifier_type, version)
    validation_file = ver_info["validation_file"] if ver_info else None
    if not validation_file or not validation_file.exists():
        return {
            "classifier": classifier_type,
            "version": version,
            "status": "no_validation",
            "message": f"Run validation on {classifier_type} classifier (version {version}) first",
            "entries": []
        }

    try:
        with open(validation_file) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading validation: {e}")

    # Get flagged entries based on type
    entries = []

    if flag_type in ["all", "disagreement"]:
        for item in data.get("flagged_disagreement", []):
            entries.append({
                "path": item["path"],
                "true_label": item["true_label"],
                "predicted_label": item["predicted_label"],
                "confidence": item["confidence"],
                "flag_type": "disagreement",
                "filename": os.path.basename(item["path"])
            })

    if flag_type in ["all", "uncertain"]:
        for item in data.get("flagged_uncertain", []):
            entries.append({
                "path": item["path"],
                "true_label": item["true_label"],
                "entropy": item["entropy"],
                "top_predictions": item.get("top_predictions", {}),
                "flag_type": "uncertain",
                "filename": os.path.basename(item["path"])
            })

    if flag_type in ["all", "outlier"]:
        for item in data.get("flagged_outlier", []):
            entries.append({
                "path": item["path"],
                "true_label": item["true_label"],
                "distance_to_centroid": item.get("distance_to_centroid"),
                "flag_type": "outlier",
                "filename": os.path.basename(item["path"])
            })

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": classifier_type,
        "flag_type": flag_type,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "summary": data.get("summary", {})
    }


@app.get("/classifier/separated-stems")
async def get_separated_stems_predictions(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    stem_filter: str = Query(None, description="Filter by stem: drums, bass, vocals, other, guitar, piano"),
    mode: str = Query("auto", description="Mode: auto, normal, temporal")
):
    """Get predictions for separated stems from mix files.

    Each entry represents an original mix file with classification results
    for each of its separated stems (drums, bass, vocals, other, guitar, piano).
    """
    classifier_dir = CLASSIFIER_DIRS.get("latent", Path("/home/arlo/Data/latent_classifier"))
    normal_file = classifier_dir / "separated_stems_predictions.json"
    temporal_file = classifier_dir / "separated_stems_temporal.json"

    # Determine which file to use
    is_temporal = False
    if mode == "temporal" and temporal_file.exists():
        predictions_file = temporal_file
        is_temporal = True
    elif mode == "normal" and normal_file.exists():
        predictions_file = normal_file
    elif mode == "auto":
        # Prefer temporal if it exists, otherwise normal
        if temporal_file.exists():
            predictions_file = temporal_file
            is_temporal = True
        elif normal_file.exists():
            predictions_file = normal_file
        else:
            return {
                "status": "not_available",
                "message": "Run separate_and_classify_mixes.py first to generate predictions",
                "entries": [],
                "total": 0
            }
    else:
        return {
            "status": "not_available",
            "message": "No predictions file found for requested mode",
            "entries": [],
            "total": 0
        }

    try:
        with open(predictions_file) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading predictions: {e}")

    results = data.get("results", [])

    # Transform to UI-friendly format
    entries = []
    for r in results:
        stems_info = r.get("stems", {})

        # Build a summary of all stems with their classifications
        stems_summary = []
        for stem_name in ["drums", "bass", "vocals", "other", "guitar", "piano"]:
            stem_data = stems_info.get(stem_name, {})
            if is_temporal:
                # Temporal mode: include segments
                stems_summary.append({
                    "stem": stem_name,
                    "segments": stem_data.get("segments", []),
                    "total_duration": stem_data.get("total_duration", 0),
                    "active_duration": stem_data.get("active_duration", 0),
                    "is_silent": stem_data.get("is_silent", True)
                })
            else:
                stems_summary.append({
                    "stem": stem_name,
                    "predicted_group": stem_data.get("predicted_group", "unknown"),
                    "confidence": stem_data.get("confidence", 0),
                    "is_silent": stem_data.get("is_silent", False)
                })

        entry = {
            "path": r.get("original_path", ""),
            "filename": r.get("original_filename", ""),
            "original_group": r.get("original_group", "undefined"),
            "original_duration": r.get("original_duration", 0),
            "detected_instruments": r.get("detected_instruments", []),
            "stems": stems_summary,
            "stems_detail": stems_info,
            "is_temporal": is_temporal,
            "timeline": r.get("timeline", []) if is_temporal else [],
            "silent_stems": r.get("silent_stems", 0),
            "active_stems": r.get("active_stems", 0),
            "processed_at": r.get("processed_at", "")
        }
        entries.append(entry)

    # Apply stem filter if provided (for non-temporal mode)
    if stem_filter and not is_temporal:
        filtered = []
        for e in entries:
            stem_data = e.get("stems_detail", {}).get(stem_filter, {})
            if stem_data.get("confidence", 0) > 0.5:
                e["predicted_group"] = stem_data.get("predicted_group", "unknown")
                e["confidence"] = stem_data.get("confidence", 0)
                filtered.append(e)
        entries = filtered

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "separated_stems",
        "is_temporal": is_temporal,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "instrument_distribution": data.get("instrument_distribution", {}),
        "available_modes": {
            "normal": normal_file.exists(),
            "temporal": temporal_file.exists()
        },
        "summary": {
            "total_processed": data.get("total", 0),
            "failed": data.get("failed", 0),
            "generated_at": data.get("generated_at", "")
        }
    }


@app.get("/classifier/multilabel-comparison")
async def get_multilabel_comparison(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get temporal stem analysis compared with GT multi-label corrections.

    Returns entries with both estimated timeline regions and GT regions for comparison.
    """
    classifier_dir = CLASSIFIER_DIRS.get("latent", Path("/home/arlo/Data/latent_classifier"))
    temporal_file = classifier_dir / "multilabel_temporal.json"
    corrections_file = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")

    if not temporal_file.exists():
        return {
            "status": "not_available",
            "message": "Run separate_and_classify_mixes.py --temporal --paths-file first",
            "entries": [],
            "total": 0
        }

    try:
        with open(temporal_file) as f:
            temporal_data = json.load(f)
        with open(corrections_file) as f:
            corrections = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

    results = temporal_data.get("results", [])
    entries = []

    for r in results:
        path = r.get("original_path", "")
        gt_correction = corrections.get(path, {})
        gt_regions = gt_correction.get("regions", [])

        # Convert timeline to regions format for consistency
        estimated_regions = []
        for t in r.get("timeline", []):
            instruments = [inst["instrument"] for inst in t.get("instruments", [])]
            estimated_regions.append({
                "start": t["start"],
                "end": t["end"],
                "labels": instruments,
                "confidences": {inst["instrument"]: inst["confidence"] for inst in t.get("instruments", [])}
            })

        # Also extract per-stem segments
        stem_segments = {}
        for stem_name, stem_data in r.get("stems", {}).items():
            if stem_data.get("segments"):
                stem_segments[stem_name] = stem_data["segments"]

        entry = {
            "path": path,
            "filename": r.get("original_filename", ""),
            "original_group": r.get("original_group", ""),
            "duration": r.get("original_duration", gt_correction.get("duration", 0)),
            # Estimated from temporal analysis
            "estimated_regions": estimated_regions,
            "estimated_instruments": r.get("detected_instruments", []),
            "stem_segments": stem_segments,
            # Ground truth from corrections
            "gt_regions": gt_regions,
            "gt_instruments": list(set(label for region in gt_regions for label in region.get("labels", []))),
            "has_gt": len(gt_regions) > 0,
            # Metadata
            "silent_stems": r.get("silent_stems", 0),
            "active_stems": r.get("active_stems", 0),
            "processed_at": r.get("processed_at", "")
        }
        entries.append(entry)

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "multilabel_comparison",
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "summary": {
            "total_processed": temporal_data.get("total", 0),
            "with_gt": sum(1 for e in entries if e["has_gt"]),
            "instrument_distribution": temporal_data.get("instrument_distribution", {}),
            "generated_at": temporal_data.get("generated_at", "")
        }
    }


@app.get("/classifier/demucs-other")
async def get_demucs_other_predictions(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    compare_gt: bool = Query(True, description="Include GT comparison if available")
):
    """Get temporal classification results for Demucs 'other' stems.

    Uses the other_classifier (brass/strings/winds/synth) on separated 'other' stems.
    Can compare against GT multi-label corrections.
    """
    other_classifier_dir = Path("/home/arlo/Data/other_classifier")
    # Use V2 results (has audio stems available)
    temporal_file = other_classifier_dir / "demucs_other_v2_temporal.json"
    corrections_file = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")

    if not temporal_file.exists():
        return {
            "status": "not_available",
            "message": "Run: python3 other_stem_classifier.py --mode batch --manifest ... first",
            "entries": [],
            "total": 0
        }

    try:
        with open(temporal_file) as f:
            data = json.load(f)
        corrections = {}
        if compare_gt and corrections_file.exists():
            with open(corrections_file) as f:
                corrections = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

    results = data.get("results", [])
    other_classes = ['brass', 'strings', 'winds', 'synth']
    entries = []

    for r in results:
        path = r.get("path", "")
        detected = r.get("detected", [])
        temporal = r.get("temporal", [])
        merged = r.get("merged", [])  # Pre-merged regions if available

        # Use merged regions if available, otherwise fall back to raw temporal
        estimated_regions = []
        if merged:
            for m in merged:
                estimated_regions.append({
                    "start": m["start_sec"],
                    "end": m["end_sec"],
                    "labels": m["classes"],
                    "confidences": m.get("avg_confidence", {})
                })
        else:
            for t in temporal:
                active_classes = t.get("active_classes", [])
                probs = t.get("probabilities", t.get("class_probs", {}))
                if active_classes:
                    estimated_regions.append({
                        "start": t["start_sec"],
                        "end": t["end_sec"],
                        "labels": active_classes,
                        "confidences": {c: probs.get(c, 0) for c in active_classes}
                    })

        # Find corresponding GT correction (need to map from demucs stem path to original)
        # Demucs path: .../LatentDemucsV2/.../other.pt -> original: .../protools/.../file.wav
        gt_regions = []
        gt_instruments = []
        original_path = ""
        # Always build original_path for audio playback
        # /home/arlo/gcs-bucket/LatentDemucsV2/protools/date/session/Audio Files/file/other.pt
        # -> /home/arlo/gcs-bucket/protools/date/session/Audio Files/file.wav
        if "LatentDemucsV2" in path:
            original_path = path.replace("/LatentDemucsV2/", "/").replace("/other.pt", ".wav")
        elif "LatentDemucs" in path:
            original_path = path.replace("/LatentDemucs/", "/").replace("/other.pt", ".wav")

        if compare_gt and original_path:
            if original_path in corrections:
                    gt_correction = corrections[original_path]
                    gt_regions = gt_correction.get("regions", [])
                    gt_instruments = list(set(
                        label for region in gt_regions
                        for label in region.get("labels", [])
                        if label in other_classes
                    ))

        # Build timeline for display - use merged if available
        timeline = []
        if merged:
            for m in merged:
                timeline.append({
                    "start": m["start_sec"],
                    "end": m["end_sec"],
                    "instruments": [
                        {"instrument": c, "confidence": m.get("avg_confidence", {}).get(c, 0)}
                        for c in m["classes"]
                    ]
                })
        else:
            for t in temporal:
                if t.get("active_classes"):
                    probs = t.get("probabilities", t.get("class_probs", {}))
                    timeline.append({
                        "start": t["start_sec"],
                        "end": t["end_sec"],
                        "instruments": [
                            {"instrument": c, "confidence": probs.get(c, 0)}
                            for c in t["active_classes"]
                        ]
                    })

        # Build stem audio paths from AudioDemucsV2
        # Latent path: /home/arlo/gcs-bucket/LatentDemucsV2/protools/.../file/other.pt
        # Audio path: /home/arlo/gcs-bucket/AudioDemucsV2/protools/.../file/other.wav
        # Note: Skip existence check to avoid slow GCS lookups - frontend handles missing files
        stem_audio_paths = {}
        stem_names = ["drums", "bass", "vocals", "other", "guitar", "piano"]
        if "LatentDemucs" in path:
            # Get the relative path after LatentDemucs
            audio_base = path.replace("/LatentDemucs", "/AudioDemucsV2").replace("/LatentDemucsV2", "/AudioDemucsV2")
            stem_dir = Path(audio_base).parent  # Directory containing stems
            for stem in stem_names:
                stem_audio_paths[stem] = str(stem_dir / f"{stem}.wav")

        entry = {
            "path": path,
            "filename": Path(path).parent.name if path else "",  # Parent dir is original file name
            "detected_instruments": detected,
            "estimated_regions": estimated_regions,
            "estimated_instruments": detected,
            "gt_regions": [r for r in gt_regions if any(l in other_classes for l in r.get("labels", []))],
            "gt_instruments": gt_instruments,
            "has_gt": len(gt_instruments) > 0,
            "original_path": original_path,
            "stem_audio_paths": stem_audio_paths,  # {stem_name: audio_path}
            "timeline": timeline,
            "is_temporal": True,
            "is_comparison": compare_gt and len(gt_instruments) > 0,
        }
        entries.append(entry)

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "demucs_other",
        "is_temporal": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "detection_counts": data.get("detection_counts", {}),
        "other_classes": other_classes,
        "summary": {
            "total_processed": data.get("total", 0),
            "stem_filter": data.get("stem_filter", "other"),
            "with_gt": sum(1 for e in entries if e["has_gt"]),
        }
    }


@app.get("/classifier/stem-energy")
async def get_stem_energy_annotations(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    stem_filter: str = Query(None, description="Filter by stem: vocals, drums, bass, guitar, piano, other")
):
    """Get Demucs stem energy annotations over time.

    This is the ground truth data used to train multi-group classifiers.
    Shows energy levels and detected instruments per stem segment.
    """
    stems_file = Path("/home/arlo/Data/mix_classifier/stems_classified.json")

    if not stems_file.exists():
        return {
            "status": "not_available",
            "message": "Run stem classification pipeline first",
            "entries": [],
            "total": 0
        }

    try:
        with open(stems_file, 'rb') as f:
            data = orjson.loads(f.read())
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "entries": [],
            "total": 0
        }

    results = data.get("results", [])
    stem_names = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']

    entries = []
    for r in results:
        original_path = r.get("original_path", "")
        stems = r.get("stems", {})
        timeline = r.get("timeline", [])
        detected = r.get("detected_instruments", [])

        # Build combined timeline from all stems
        combined_timeline = []
        for stem_name in stem_names:
            stem_data = stems.get(stem_name, {})
            temporal = stem_data.get("temporal", [])

            for seg in temporal:
                if seg.get("silent", True):
                    continue
                combined_timeline.append({
                    "stem": stem_name,
                    "start_sec": seg.get("start_sec", 0),
                    "end_sec": seg.get("end_sec", 0),
                    "energy": seg.get("energy", 0),
                    "top_class": seg.get("top_class", ""),
                    "top_confidence": seg.get("top_confidence", 0),
                    "predictions": seg.get("predictions", {}),
                })

        # Sort by start time
        combined_timeline.sort(key=lambda x: (x["start_sec"], x["stem"]))

        # Filter by stem if requested
        if stem_filter:
            combined_timeline = [t for t in combined_timeline if t["stem"] == stem_filter]

        # Count active stems
        active_stems = sum(1 for s in stems.values() if not s.get("silent", True) and s.get("temporal"))

        entry = {
            "path": original_path,
            "filename": os.path.basename(original_path),
            "detected_instruments": detected,
            "active_stems": active_stems,
            "timeline": combined_timeline,
            "stems_summary": {
                stem: {
                    "top_class": stems.get(stem, {}).get("top_class"),
                    "confidence": stems.get(stem, {}).get("top_confidence", 0),
                    "silent": stems.get(stem, {}).get("final_class") is None,
                }
                for stem in stem_names
            },
            "is_temporal": True,
            "duration": max([t["end_sec"] for t in combined_timeline]) if combined_timeline else 0,
        }
        entries.append(entry)

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "stem_energy",
        "is_temporal": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "stem_names": stem_names,
        "classifier_classes": data.get("classifier_classes", []),
        "summary": {
            "total_files": data.get("total", 0),
            "errors": data.get("errors", 0),
        }
    }


@app.get("/classifier/mix-classifier-v2")
async def get_mix_classifier_v2_predictions(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    compare_gt: bool = Query(True, description="Include GT comparison if available")
):
    """Get temporal classification results from mix classifier v2.

    V2 model classifies brass/strings/winds directly from mix audio latents
    using a fresh classifier head (bypasses solo classifier issues).
    """
    mix_classifier_dir = Path("/home/arlo/Data/mix_classifier")
    temporal_file = mix_classifier_dir / "mix_classifier_v2_temporal.json"
    corrections_file = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")

    if not temporal_file.exists():
        return {
            "status": "not_available",
            "message": "Run: python3 run_mix_classifier_v2.py --input-dir /path/to/latents first",
            "entries": [],
            "total": 0
        }

    try:
        with open(temporal_file) as f:
            data = json.load(f)
        corrections = {}
        if compare_gt and corrections_file.exists():
            with open(corrections_file) as f:
                corrections = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

    results = data.get("results", [])
    target_classes = data.get("target_classes", ['brass', 'strings', 'winds'])
    entries = []

    for r in results:
        path = r.get("path", "")
        audio_path = r.get("audio_path", "")
        detected = r.get("detected", [])
        temporal = r.get("temporal", [])
        merged = r.get("merged", [])
        duration = r.get("duration", 0)

        # Build estimated regions from merged data
        estimated_regions = []
        if merged:
            for m in merged:
                estimated_regions.append({
                    "start": m["start_sec"],
                    "end": m["end_sec"],
                    "labels": m["classes"],
                    "confidences": m.get("avg_confidence", {})
                })
        else:
            for t in temporal:
                active_classes = t.get("active_classes", [])
                probs = t.get("probabilities", {})
                if active_classes:
                    estimated_regions.append({
                        "start": t["start_sec"],
                        "end": t["end_sec"],
                        "labels": active_classes,
                        "confidences": {c: probs.get(c, 0) for c in active_classes}
                    })

        # Find GT correction if available
        gt_regions = []
        gt_instruments = []
        if compare_gt and audio_path and audio_path in corrections:
            gt_correction = corrections[audio_path]
            gt_regions = gt_correction.get("regions", [])
            gt_instruments = list(set(
                label for region in gt_regions
                for label in region.get("labels", [])
                if label in target_classes
            ))

        # Build timeline for display
        timeline = []
        if merged:
            for m in merged:
                timeline.append({
                    "start": m["start_sec"],
                    "end": m["end_sec"],
                    "instruments": [
                        {"instrument": c, "confidence": m.get("avg_confidence", {}).get(c, 0)}
                        for c in m["classes"]
                    ]
                })
        else:
            for t in temporal:
                if t.get("active_classes"):
                    probs = t.get("probabilities", {})
                    timeline.append({
                        "start": t["start_sec"],
                        "end": t["end_sec"],
                        "instruments": [
                            {"instrument": c, "confidence": probs.get(c, 0)}
                            for c in t["active_classes"]
                        ]
                    })

        entry = {
            "path": path,
            "audio_path": audio_path,
            "filename": r.get("filename", Path(path).stem),
            "detected_instruments": detected,
            "estimated_regions": estimated_regions,
            "estimated_instruments": detected,
            "gt_regions": [r for r in gt_regions if any(l in target_classes for l in r.get("labels", []))],
            "gt_instruments": gt_instruments,
            "has_gt": len(gt_instruments) > 0,
            "original_path": audio_path,
            "timeline": timeline,
            "duration": duration,
            "is_temporal": True,
            "is_comparison": compare_gt and len(gt_instruments) > 0,
        }
        entries.append(entry)

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "mix_classifier_v2",
        "is_temporal": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "detection_counts": data.get("detection_counts", {}),
        "target_classes": target_classes,
        "summary": {
            "total_processed": data.get("total", 0),
            "with_gt": sum(1 for e in entries if e["has_gt"]),
            "generated_at": data.get("generated_at", ""),
        }
    }


@app.get("/classifier/mix-classifier-v3-compare")
async def get_mix_classifier_v3_comparison(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get side-by-side comparison of V3 model predictions vs GT annotations.

    Returns files that have both model predictions and GT labels for
    visual comparison in the dual-waveform view.
    """
    comparison_file = Path("/home/arlo/Data/mix_classifier/mix_classifier_v3_comparison.json")

    if not comparison_file.exists():
        return {
            "status": "not_available",
            "message": "Run comparison script first",
            "entries": [],
            "total": 0
        }

    try:
        with open(comparison_file) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

    results = data.get("results", [])
    target_classes = data.get("target_classes", ['brass', 'strings', 'winds'])

    entries = []
    for r in results:
        entry = {
            "path": r.get("path", ""),
            "audio_path": r.get("audio_path", ""),
            "original_path": r.get("audio_path", ""),
            "filename": r.get("filename", ""),
            "duration": r.get("duration", 0),
            "predicted_regions": r.get("predicted_regions", []),
            "predicted_instruments": r.get("predicted_instruments", []),
            "gt_regions": r.get("gt_regions", []),
            "gt_instruments": r.get("gt_instruments", []),
            "estimated_regions": r.get("predicted_regions", []),  # Alias for UI
            "estimated_instruments": r.get("predicted_instruments", []),
            "has_gt": True,
            "is_temporal": True,
            "is_comparison": True,
            "is_dual_waveform": True,  # Signal for dual waveform mode
        }
        entries.append(entry)

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "mix_classifier_v3_compare",
        "model_version": data.get("model_version", "v3"),
        "is_temporal": True,
        "is_dual_waveform": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "target_classes": target_classes,
        "generated_at": data.get("generated_at", ""),
    }


@app.get("/classifier/merged-manifest-v2")
async def get_merged_manifest_v2(
    group: str = Query(None, description="Filter by group"),
    subgroup: str = Query(None, description="Filter by subgroup"),
    source_filter: str = Query(None, description="Filter by source: original, classifier, flagged, corrected, mix"),
    confidence_min: float = Query(None, description="Min confidence for classifier-filled entries"),
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get entries from merged manifest v2 for review/correction.

    This manifest combines:
    - Original labels
    - Classifier predictions for undefined subgroups
    - 972 manual corrections
    - 49k flagged disagreements
    - Mix detection (multi-classifier and filename)
    """
    manifest_path = Path("/home/arlo/gcs-bucket/Manifests/merged_manifest_v2.json")

    if not manifest_path.exists():
        return {
            "status": "not_available",
            "message": "Run create_merged_manifest.py first",
            "entries": [],
            "total": 0
        }

    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading manifest: {e}")

    manifest_entries = data.get("entries", {})
    manifest_stats = data.get("stats", {})

    # Load corrections for enrichment
    corrections = {}
    if CORRECTIONS_FILE.exists():
        try:
            with open(CORRECTIONS_FILE) as f:
                corrections = json.load(f)
        except:
            pass

    entries = []
    all_groups = set()
    all_subgroups = set()
    source_counts = {
        "original": 0,
        "classifier": 0,
        "flagged": 0,
        "corrected": 0,
        "mix": 0
    }

    for path, entry in manifest_entries.items():
        if not isinstance(entry, dict):
            continue

        entry_group = entry.get("group", "undefined")
        entry_subgroup = entry.get("subgroup", "undefined")
        all_groups.add(entry_group)
        if entry_subgroup and entry_subgroup != "undefined":
            all_subgroups.add(entry_subgroup)

        # Determine source
        sources = entry.get("sources", ["original"])
        if isinstance(sources, str):
            sources = [sources]

        is_mix = entry.get("mix", False)
        is_corrected = path in corrections
        is_classifier = "classifier_prediction" in sources or entry.get("subgroup_source") == "classifier"
        is_flagged = "flagged_disagreement" in sources or entry.get("group_source") == "classifier_correction"

        # Count sources
        if is_mix:
            source_counts["mix"] += 1
        if is_corrected:
            source_counts["corrected"] += 1
        elif is_flagged:
            source_counts["flagged"] += 1
        elif is_classifier:
            source_counts["classifier"] += 1
        else:
            source_counts["original"] += 1

        # Apply filters
        if group and entry_group != group:
            continue
        if subgroup and entry_subgroup != subgroup:
            continue

        if source_filter == "original" and (is_classifier or is_flagged or is_corrected):
            continue
        if source_filter == "classifier" and not is_classifier:
            continue
        if source_filter == "flagged" and not is_flagged:
            continue
        if source_filter == "corrected" and not is_corrected:
            continue
        if source_filter == "mix" and not is_mix:
            continue

        if confidence_min is not None:
            conf = entry.get("subgroup_confidence") or entry.get("flagged_confidence") or 1.0
            if conf < confidence_min:
                continue

        # Get correction data if exists
        correction = corrections.get(path, {})
        has_correction = bool(correction)

        entries.append({
            "path": path,
            "current_label": correction.get("group", entry_group) if has_correction else entry_group,
            "original_group": entry.get("original_group", entry_group),
            "subgroup": correction.get("subgroup", entry_subgroup) if has_correction else entry_subgroup,
            "original_subgroup": entry.get("original_subgroup", "undefined"),
            "subgroup_source": entry.get("subgroup_source", ""),
            "subgroup_confidence": entry.get("subgroup_confidence", 0),
            "group_source": entry.get("group_source", ""),
            "flagged_original": entry.get("flagged_original", ""),
            "flagged_confidence": entry.get("flagged_confidence", 0),
            "mix": is_mix,
            "mix_source": entry.get("mix_source", ""),
            "multi_probability": entry.get("multi_probability", 0),
            "filename": os.path.basename(path),
            "sources": sources,
            # Correction details
            "has_correction": has_correction,
            "roomy": correction.get("roomy", False),
            "has_bleed": correction.get("has_bleed", False),
            "bleed_instruments": correction.get("bleed_instruments", []),
            "multi_label": correction.get("multi_label", False),
            "regions": correction.get("regions", []),
        })

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "merged_manifest_v2",
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "available_groups": sorted(all_groups),
        "available_subgroups": sorted(all_subgroups),
        "manifest_stats": manifest_stats,
        "source_counts": source_counts,
        "created_at": data.get("created_at", ""),
    }


@app.get("/classifier/mix-classifier-v3")
async def get_mix_classifier_v3_predictions(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    compare_gt: bool = Query(True, description="Include GT comparison if available")
):
    """Get temporal classification results from mix classifier v3.

    V3 model is trained on segment-level data with timestamps for better
    temporal accuracy. Uses overlapping windows for finer resolution.
    """
    mix_classifier_dir = Path("/home/arlo/Data/mix_classifier")
    temporal_file = mix_classifier_dir / "mix_classifier_v3_temporal.json"
    corrections_file = Path("/home/arlo/gcs-bucket/Manifests/corrections.json")

    if not temporal_file.exists():
        return {
            "status": "not_available",
            "message": "Run: python3 run_mix_classifier_v3.py --input-dir /path/to/latents first",
            "entries": [],
            "total": 0
        }

    try:
        with open(temporal_file) as f:
            data = json.load(f)
        corrections = {}
        if compare_gt and corrections_file.exists():
            with open(corrections_file) as f:
                corrections = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

    results = data.get("results", [])
    target_classes = data.get("target_classes", ['brass', 'strings', 'winds'])
    entries = []

    for r in results:
        path = r.get("path", "")
        audio_path = r.get("audio_path", "")
        detected = r.get("detected", [])
        temporal = r.get("temporal", [])
        merged = r.get("merged", [])
        duration = r.get("duration", 0)

        # Build estimated regions from merged data
        estimated_regions = []
        if merged:
            for m in merged:
                estimated_regions.append({
                    "start": m["start_sec"],
                    "end": m["end_sec"],
                    "labels": m["classes"],
                    "confidences": m.get("avg_confidence", {})
                })
        else:
            for t in temporal:
                active_classes = t.get("active_classes", [])
                probs = t.get("probabilities", {})
                if active_classes:
                    estimated_regions.append({
                        "start": t["start_sec"],
                        "end": t["end_sec"],
                        "labels": active_classes,
                        "confidences": {c: probs.get(c, 0) for c in active_classes}
                    })

        # Find GT correction if available
        gt_regions = []
        gt_instruments = []
        if compare_gt and audio_path and audio_path in corrections:
            gt_correction = corrections[audio_path]
            gt_regions = gt_correction.get("regions", [])
            gt_instruments = list(set(
                label for region in gt_regions
                for label in region.get("labels", [])
                if label in target_classes
            ))

        # Build timeline for display
        timeline = []
        if merged:
            for m in merged:
                timeline.append({
                    "start": m["start_sec"],
                    "end": m["end_sec"],
                    "instruments": [
                        {"instrument": c, "confidence": m.get("avg_confidence", {}).get(c, 0)}
                        for c in m["classes"]
                    ]
                })
        else:
            for t in temporal:
                if t.get("active_classes"):
                    probs = t.get("probabilities", {})
                    timeline.append({
                        "start": t["start_sec"],
                        "end": t["end_sec"],
                        "instruments": [
                            {"instrument": c, "confidence": probs.get(c, 0)}
                            for c in t["active_classes"]
                        ]
                    })

        entry = {
            "path": path,
            "audio_path": audio_path,
            "filename": r.get("filename", Path(path).stem),
            "detected_instruments": detected,
            "estimated_regions": estimated_regions,
            "estimated_instruments": detected,
            "gt_regions": [r for r in gt_regions if any(l in target_classes for l in r.get("labels", []))],
            "gt_instruments": gt_instruments,
            "has_gt": len(gt_instruments) > 0,
            "original_path": audio_path,
            "timeline": timeline,
            "duration": duration,
            "is_temporal": True,
            "is_comparison": compare_gt and len(gt_instruments) > 0,
        }
        entries.append(entry)

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "mix_classifier_v3",
        "model_version": data.get("model_version", "v3"),
        "training_type": data.get("training_type", "segment-level"),
        "is_temporal": True,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "detection_counts": data.get("detection_counts", {}),
        "target_classes": target_classes,
        "summary": {
            "total_processed": data.get("total", 0),
            "with_gt": sum(1 for e in entries if e["has_gt"]),
            "generated_at": data.get("generated_at", ""),
            "threshold": data.get("threshold", 0.5),
            "segment_sec": data.get("segment_sec", 2.0),
            "hop_sec": data.get("hop_sec", 0.5),
        }
    }


# Ensemble detector cache
_ensemble_cache = {"data": None, "mtime": 0}
ENSEMBLE_DETECTIONS_FILE = Path("/home/arlo/Data/ensemble_detector/ensemble_detections.json")


def get_ensemble_lookup() -> dict:
    """Load ensemble detection results (mix vs isolated from latent-based detector).

    Returns dict mapping audio_path -> {"is_ensemble": bool, "probability": float}
    """
    global _ensemble_cache

    if not ENSEMBLE_DETECTIONS_FILE.exists():
        return {}

    current_mtime = ENSEMBLE_DETECTIONS_FILE.stat().st_mtime
    if _ensemble_cache["data"] is not None and _ensemble_cache["mtime"] == current_mtime:
        return _ensemble_cache["data"]

    try:
        with open(ENSEMBLE_DETECTIONS_FILE, 'rb') as f:
            data = orjson.loads(f.read())

        lookup = {}
        threshold = data.get("threshold", 0.5)

        # All detected entries are ensembles (probability >= threshold)
        for entry in data.get("detected", []):
            path = entry.get("path", "")
            if path:
                lookup[path] = {
                    "is_ensemble": True,
                    "probability": entry.get("ensemble_probability", 1.0),
                }

        _ensemble_cache["data"] = lookup
        _ensemble_cache["mtime"] = current_mtime
        _ensemble_cache["threshold"] = threshold
        _ensemble_cache["total_checked"] = data.get("total_checked", 0)
        _ensemble_cache["detected_count"] = data.get("detected_count", 0)
        return lookup
    except Exception as e:
        print(f"Error loading ensemble detections: {e}")
        return {}


@app.get("/classifier/ensemble")
async def get_ensemble_predictions(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    filter_type: str = Query(None, description="Filter: ensemble, isolated"),
    sort_by: str = Query(None, description="Sort: ensemble_prob_desc, ensemble_prob_asc, mix_first, isolated_first"),
    group: str = Query(None, description="Filter by group (from manifest)"),
    isolated_filter: str = Query(None, description="Filter: isolated, mix")
):
    """Get ensemble detector results (mix vs isolated from latent-based detector).

    The ensemble detector is trained to distinguish:
    - Solo/isolated recordings (single instrument)
    - Ensemble/mix recordings (multiple instruments)
    """
    if not ENSEMBLE_DETECTIONS_FILE.exists():
        return {
            "status": "not_available",
            "message": "Run: python3 latent_ensemble_detector.py --mode detect first",
            "entries": [],
            "total": 0,
            "ensemble_count": 0,
            "isolated_count": 0,
        }

    try:
        with open(ENSEMBLE_DETECTIONS_FILE, 'rb') as f:
            data = orjson.loads(f.read())
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "entries": [],
            "total": 0,
        }

    detected = data.get("detected", [])
    total_checked = data.get("total_checked", 0)
    threshold = data.get("threshold", 0.5)

    # Get manifest groups for enrichment
    manifest_groups = get_manifest_groups()

    # Build entries list
    entries = []
    for d in detected:
        path = d.get("path", "")
        manifest_group = manifest_groups.get(path, "unknown")
        entries.append({
            "path": path,
            "filename": d.get("filename", ""),
            "ensemble_probability": d.get("ensemble_probability", 0),
            "confidence": d.get("ensemble_probability", 0),  # For consistent UI
            "is_ensemble": True,
            "is_isolated": False,
            "predicted_class": "mix",
            "manifest_group": manifest_group,
        })

    # Filter by group
    if group:
        entries = [e for e in entries if e.get("manifest_group") == group]

    # Filter by isolated/mix
    if isolated_filter == "isolated":
        entries = []  # All entries in this file are mix
    elif isolated_filter == "mix":
        pass  # All entries are already mix

    # Sort
    if sort_by == "ensemble_prob_asc":
        entries.sort(key=lambda x: x["ensemble_probability"])
    elif sort_by == "ensemble_prob_desc" or sort_by == "mix_first":
        entries.sort(key=lambda x: -x["ensemble_probability"])
    elif sort_by == "confidence_asc":
        entries.sort(key=lambda x: x["ensemble_probability"])
    elif sort_by == "confidence_desc":
        entries.sort(key=lambda x: -x["ensemble_probability"])
    else:
        # Default: highest probability first
        entries.sort(key=lambda x: -x["ensemble_probability"])

    # Legacy filter
    if filter_type == "isolated":
        entries = []

    total = len(entries)
    entries = entries[offset:offset + limit]

    return {
        "classifier": "ensemble_detector",
        "status": "ok",
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "stats": {
            "total_checked": total_checked,
            "ensemble_count": data.get("detected_count", len(detected)),
            "isolated_count": total_checked - data.get("detected_count", len(detected)),
            "threshold": threshold,
        },
        "detected_at": data.get("detected_at", ""),
    }


@app.get("/classifier/subgroups")
async def get_subgroup_predictions(
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    group: str = Query(None, description="Filter by group: brass, strings, winds, bass, guitar, piano")
):
    """Get subgroup classifier predictions.

    Shows predictions for instrument subgroups within each main group:
    - brass: french_horn, trombone, trumpet, tuba
    - strings: cello, viola, violin
    - winds: clarinet, flute, oboe, sax
    - bass: electric_bass, upright_bass
    - guitar: acoustic_guitar, electric_guitar
    - piano: acoustic_piano, keys
    """
    predictions_file = Path("/home/arlo/Data/subgroup_classifiers/all_predictions.json")

    if not predictions_file.exists():
        return {
            "status": "not_available",
            "message": "Run: python3 run_subgroup_classifiers.py first",
            "entries": [],
            "total": 0
        }

    try:
        with open(predictions_file) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {e}")

    groups = data.get("groups", [])
    results = data.get("results", {})
    entries = []

    # Filter by group if specified
    groups_to_process = [group] if group and group in results else list(results.keys())

    for grp in groups_to_process:
        grp_results = results.get(grp, [])
        for r in grp_results:
            if "error" in r:
                continue

            audio_path = r.get("path", "")
            current = r.get("current_subgroup")
            predicted = r.get("predicted")
            confidence = r.get("confidence", 0)
            all_probs = r.get("all_probs", {})

            # Check if prediction matches current label
            matches = current == predicted if current and predicted else None

            entries.append({
                "path": audio_path,
                "filename": os.path.basename(audio_path),
                "group": grp,
                "current_subgroup": current,
                "predicted_subgroup": predicted,
                "confidence": confidence,
                "all_probs": all_probs,
                "matches": matches,
                "is_mismatch": matches is False and confidence > 0.5,
            })

    # Sort by: undefined subgroup first, then mismatches, then confidence
    entries.sort(key=lambda x: (
        x.get("current_subgroup") not in (None, "", "undefined"),  # undefined first
        not x.get("is_mismatch", False),  # then mismatches
        -x.get("confidence", 0)  # then by confidence desc
    ))

    total = len(entries)
    entries = entries[offset:offset + limit]

    # Compute summary stats
    mismatches = sum(1 for e in entries if e.get("is_mismatch"))
    matches = sum(1 for e in entries if e.get("matches") is True)

    return {
        "classifier": "subgroups",
        "groups": groups,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": entries,
        "summary": {
            "total_predictions": total,
            "matches": matches,
            "mismatches": mismatches,
            "groups_loaded": data.get("models_loaded", []),
        }
    }


@app.get("/manifests")
async def list_manifests():
    """List all available manifests for the labeler."""
    manifests = []

    # Add classifier predictions as virtual manifests
    for classifier_type, classifier_dir in CLASSIFIER_DIRS.items():
        predictions_file = classifier_dir / "predictions.json"
        if predictions_file.exists():
            try:
                stat = predictions_file.stat()
                with open(predictions_file) as f:
                    data = json.load(f)
                entry_count = len(data.get("predictions", []))
                manifests.append({
                    "filename": f"[{classifier_type.upper()}] predictions.json",
                    "path": str(predictions_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "entries": entry_count,
                    "type": "classifier_predictions",
                    "classifier_type": classifier_type,
                    "is_classifier_result": True,
                    "modified": stat.st_mtime,
                    "summary": data.get("summary", {})
                })
            except Exception:
                pass

        # Add validation results
        validation_file = classifier_dir / "validation_results.json"
        if validation_file.exists():
            try:
                stat = validation_file.stat()
                with open(validation_file) as f:
                    data = json.load(f)
                flagged_count = data.get("summary", {}).get("total_flagged", 0)
                manifests.append({
                    "filename": f"[{classifier_type.upper()}] flagged_entries.json",
                    "path": str(validation_file),
                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                    "entries": flagged_count,
                    "type": "classifier_flagged",
                    "classifier_type": classifier_type,
                    "is_classifier_result": True,
                    "modified": stat.st_mtime,
                    "summary": data.get("summary", {})
                })
            except Exception:
                pass

    if MANIFESTS_DIR.exists():
        for f in sorted(MANIFESTS_DIR.glob("*.json")):
            # Skip corrections and corrected manifests from listing
            if f.name in ["corrections.json"]:
                continue

            try:
                # Get basic info without loading entire file
                stat = f.stat()
                size_mb = stat.st_size / (1024 * 1024)

                # Quick peek to get entry count
                with open(f) as fp:
                    # Read first few bytes to check format
                    content_start = fp.read(1000)
                    fp.seek(0)
                    data = json.load(fp)

                # Determine type
                if isinstance(data, dict):
                    if "predictions" in data:
                        # Prediction format
                        entry_count = len(data.get("predictions", []))
                        manifest_type = "predictions"
                    elif "entries" in data:
                        # Unified manifest format
                        entry_count = len(data.get("entries", []))
                        manifest_type = "unified_manifest"
                    else:
                        # Standard manifest format
                        entry_count = len(data)
                        manifest_type = "manifest"
                else:
                    entry_count = 0
                    manifest_type = "unknown"

                # Detect if this is a classifier result
                is_classifier_result = "predictions_manifest" in f.name or "subgroup" in f.name

                manifests.append({
                    "filename": f.name,
                    "path": str(f),
                    "size_mb": round(size_mb, 2),
                    "entries": entry_count,
                    "type": manifest_type,
                    "is_classifier_result": is_classifier_result,
                    "modified": stat.st_mtime
                })
            except Exception as e:
                manifests.append({
                    "filename": f.name,
                    "path": str(f),
                    "error": str(e)
                })

    # Sort: classifier results first, then by modified time
    manifests.sort(key=lambda x: (not x.get("is_classifier_result", False), -x.get("modified", 0)))

    return {"manifests": manifests}


@app.get("/correction")
async def get_correction(path: str = Query(..., description="Audio path to get correction for")):
    """Get correction details for a specific audio path."""
    if not CORRECTIONS_FILE.exists():
        return {"found": False}

    try:
        with open(CORRECTIONS_FILE) as f:
            corrections = json.load(f)

        if path in corrections:
            return {"found": True, "correction": corrections[path]}
        else:
            return {"found": False}
    except Exception as e:
        return {"found": False, "error": str(e)}


@app.get("/corrections/list")
async def list_corrections(
    filter_type: str = Query(None, description="Filter: multilabel, roomy, bleed, or all"),
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """List corrections with optional filtering."""
    if not CORRECTIONS_FILE.exists():
        return {"total": 0, "entries": []}

    try:
        with open(CORRECTIONS_FILE) as f:
            corrections = json.load(f)

        entries = []
        for path, data in corrections.items():
            # Apply filters
            if filter_type == 'multilabel' and not data.get('multi_label'):
                continue
            if filter_type == 'roomy' and not data.get('roomy'):
                continue
            if filter_type == 'bleed' and not data.get('has_bleed'):
                continue

            entries.append({
                "path": path,
                "filename": os.path.basename(path),
                "current_label": data.get("group", "unknown"),
                "subgroup": data.get("subgroup", ""),
                "roomy": data.get("roomy", False),
                "has_bleed": data.get("has_bleed", False),
                "bleed_instruments": data.get("bleed_instruments", []),
                "multi_label": data.get("multi_label", False),
                "regions": data.get("regions", []),
                "corrected_at": data.get("corrected_at"),
            })

        # Sort by corrected_at descending (most recent first)
        entries.sort(key=lambda x: x.get("corrected_at", ""), reverse=True)

        total = len(entries)
        entries = entries[offset:offset + limit]

        # Count stats
        stats = {
            "total": len(corrections),
            "multilabel": sum(1 for d in corrections.values() if d.get("multi_label")),
            "roomy": sum(1 for d in corrections.values() if d.get("roomy")),
            "bleed": sum(1 for d in corrections.values() if d.get("has_bleed")),
        }

        return {
            "total": total,
            "entries": entries,
            "stats": stats,
            "filter_type": filter_type,
        }
    except Exception as e:
        return {"total": 0, "entries": [], "error": str(e)}


@app.get("/manifest/entries")
async def get_manifest_entries(
    group: str = Query(None, description="Filter by group label"),
    correction_filter: str = Query(None, description="Filter: corrected, multilabel, roomy, bleed"),
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Get entries from the ground truth manifest for review/editing."""
    manifest_path = Path("/home/arlo/gcs-bucket/Manifests/unified_manifest.json")

    if not manifest_path.exists():
        return {"status": "error", "message": "Manifest not found", "entries": []}

    try:
        with open(manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        # Load corrections for enrichment and filtering
        corrections = {}
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE, 'rb') as f:
                corrections = orjson.loads(f.read())

        # Get all groups for filter dropdown
        all_groups = set()
        entries = []

        # Handle both formats: {entries: [...]} or direct {path: data, ...}
        manifest_entries = manifest.get("entries", manifest)

        # If entries is a list, iterate directly
        if isinstance(manifest_entries, list):
            for entry in manifest_entries:
                if not isinstance(entry, dict):
                    continue
                entry_group = entry.get("group", "undefined")
                all_groups.add(entry_group)

                if group and entry_group != group:
                    continue

                # Get path - could be 'path' or 'audio_path'
                entry_path = entry.get("path") or entry.get("audio_path", "")

                # Get correction data if exists
                correction = corrections.get(entry_path, {})
                has_correction = bool(correction)

                # Apply correction filters
                if correction_filter == 'corrected' and not has_correction:
                    continue
                if correction_filter == 'multilabel' and not correction.get('multi_label'):
                    continue
                if correction_filter == 'roomy' and not correction.get('roomy'):
                    continue
                if correction_filter == 'bleed' and not correction.get('has_bleed'):
                    continue

                entries.append({
                    "path": entry_path,
                    "current_label": correction.get("group", entry_group) if has_correction else entry_group,
                    "subgroup": correction.get("subgroup", entry.get("subgroup", "")),
                    "filename": os.path.basename(entry_path),
                    "labeling_method": entry.get("labeling_method", entry.get("source", "unknown")),
                    # Include correction details
                    "has_correction": has_correction,
                    "roomy": correction.get("roomy", False),
                    "has_bleed": correction.get("has_bleed", False),
                    "bleed_instruments": correction.get("bleed_instruments", []),
                    "multi_label": correction.get("multi_label", False),
                    "regions": correction.get("regions", []),
                    "corrected_at": correction.get("corrected_at"),
                })
        else:
            # Dict format: {path: data, ...}
            for path, data in manifest_entries.items():
                # Skip non-dict entries (metadata keys)
                if not isinstance(data, dict):
                    continue

                entry_group = data.get("group", "undefined")
                all_groups.add(entry_group)

                if group and entry_group != group:
                    continue

                # Get correction data if exists
                correction = corrections.get(path, {})
                has_correction = bool(correction)

                # Apply correction filters
                if correction_filter == 'corrected' and not has_correction:
                    continue
                if correction_filter == 'multilabel' and not correction.get('multi_label'):
                    continue
                if correction_filter == 'roomy' and not correction.get('roomy'):
                    continue
                if correction_filter == 'bleed' and not correction.get('has_bleed'):
                    continue

                entries.append({
                    "path": path,
                    "current_label": correction.get("group", entry_group) if has_correction else entry_group,
                    "subgroup": correction.get("subgroup", data.get("subgroup", "")),
                    "filename": os.path.basename(path),
                    "labeling_method": data.get("labeling_method", "unknown"),
                    # Include correction details
                    "has_correction": has_correction,
                    "roomy": correction.get("roomy", False),
                    "has_bleed": correction.get("has_bleed", False),
                    "bleed_instruments": correction.get("bleed_instruments", []),
                    "multi_label": correction.get("multi_label", False),
                    "regions": correction.get("regions", []),
                    "corrected_at": correction.get("corrected_at"),
                })

        total = len(entries)
        entries = entries[offset:offset + limit]

        # Count correction stats
        correction_stats = {
            "total_corrections": len(corrections),
            "multilabel": sum(1 for d in corrections.values() if d.get("multi_label")),
            "roomy": sum(1 for d in corrections.values() if d.get("roomy")),
            "bleed": sum(1 for d in corrections.values() if d.get("has_bleed")),
        }

        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "entries": entries,
            "available_groups": sorted(all_groups),
            "group_filter": group,
            "correction_filter": correction_filter,
            "correction_stats": correction_stats,
        }
    except Exception as e:
        return {"status": "error", "message": str(e), "entries": []}


@app.get("/manifest/{manifest_name:path}")
async def get_manifest(
    manifest_name: str,
    group: str = Query(None, description="Filter by group"),
    subgroup: str = Query(None, description="Filter by subgroup"),
    confidence_min: float = Query(None, description="Minimum confidence"),
    confidence_max: float = Query(None, description="Maximum confidence"),
    limit: int = Query(1000, description="Max entries to return"),
    offset: int = Query(0, description="Offset for pagination")
):
    """Load a specific manifest with optional filters."""
    manifest_path = MANIFESTS_DIR / manifest_name

    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"Manifest not found: {manifest_name}")

    try:
        with open(manifest_path) as f:
            data = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading manifest: {e}")

    # Convert to list of entries for filtering
    entries = []

    if isinstance(data, dict):
        if "predictions" in data:
            # Prediction format - already list
            for pred in data.get("predictions", []):
                entries.append({
                    "path": pred.get("path", ""),
                    "group": pred.get("group", pred.get("predicted_group", "undefined")),
                    "subgroup": pred.get("predicted_subgroup", pred.get("subgroup", "undefined")),
                    "confidence": pred.get("confidence", 0),
                    "all_probabilities": pred.get("all_probabilities", {}),
                    "filename": os.path.basename(pred.get("path", ""))
                })
        else:
            # Standard manifest format - dict keyed by path
            for path, meta in data.items():
                if not isinstance(meta, dict):
                    continue
                entries.append({
                    "path": path,
                    "group": meta.get("group", "undefined"),
                    "subgroup": meta.get("subgroup", "undefined"),
                    "confidence": meta.get("confidence", 1.0),
                    "all_probabilities": meta.get("all_probabilities", {}),
                    "filename": meta.get("filename", os.path.basename(path))
                })

    # Get unique groups and subgroups for filter options
    all_groups = sorted(set(e["group"] for e in entries if e.get("group")))
    all_subgroups = sorted(set(e["subgroup"] for e in entries if e.get("subgroup") and e["subgroup"] != "undefined"))

    # Apply filters
    filtered = entries

    if group:
        filtered = [e for e in filtered if e["group"] == group]

    if subgroup:
        filtered = [e for e in filtered if e["subgroup"] == subgroup]

    if confidence_min is not None:
        filtered = [e for e in filtered if e.get("confidence", 1.0) >= confidence_min]

    if confidence_max is not None:
        filtered = [e for e in filtered if e.get("confidence", 1.0) <= confidence_max]

    # Sort by confidence descending
    filtered.sort(key=lambda x: x.get("confidence", 0), reverse=True)

    total = len(filtered)
    filtered = filtered[offset:offset + limit]

    return {
        "manifest": manifest_name,
        "total": total,
        "offset": offset,
        "limit": limit,
        "entries": filtered,
        "available_groups": all_groups,
        "available_subgroups": all_subgroups
    }


@app.get("/predictions")
async def get_predictions():
    """Get classifier predictions for labeling UI."""
    if not PREDICTIONS_FILE.exists():
        return {"predictions": [], "error": "No predictions file found. Run classifier first."}

    try:
        with open(PREDICTIONS_FILE) as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"predictions": [], "error": str(e)}


@app.get("/corrections")
async def get_corrections():
    """Get saved corrections."""
    if not CORRECTIONS_FILE.exists():
        return {"corrections": {}}

    try:
        with open(CORRECTIONS_FILE) as f:
            data = json.load(f)
        return {"corrections": data}
    except Exception as e:
        return {"corrections": {}, "error": str(e)}


@app.post("/corrections")
async def save_correction(request: dict):
    """Save a single correction."""
    path = request.get("path")
    group = request.get("group")
    subgroup = request.get("subgroup", "undefined")

    if not path or not group:
        return {"status": "error", "message": "path and group required"}

    # Load existing corrections
    corrections = {}
    if CORRECTIONS_FILE.exists():
        try:
            with open(CORRECTIONS_FILE) as f:
                corrections = json.load(f)
        except Exception:
            pass

    # Add/update correction
    corrections[path] = {
        "group": group,
        "subgroup": subgroup,
        "corrected_at": datetime.now().isoformat()
    }

    # Save
    try:
        CORRECTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CORRECTIONS_FILE, 'w') as f:
            json.dump(corrections, f, indent=2)
        return {"status": "ok", "count": len(corrections)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/correction")
async def delete_correction(path: str = Query(..., description="Path to remove correction for")):
    """Delete a correction (undo a label)."""
    if not CORRECTIONS_FILE.exists():
        return {"status": "error", "message": "No corrections file"}

    try:
        with open(CORRECTIONS_FILE) as f:
            corrections = json.load(f)

        if path not in corrections:
            return {"status": "error", "message": "Correction not found"}

        # Get the deleted correction data before removing
        deleted = corrections.pop(path)

        with open(CORRECTIONS_FILE, 'w') as f:
            json.dump(corrections, f, indent=2)

        return {
            "status": "ok",
            "deleted_path": path,
            "deleted_correction": deleted,
            "remaining_count": len(corrections)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/corrections/export")
async def export_corrections():
    """Export corrections to a corrected manifest file."""
    if not CORRECTIONS_FILE.exists():
        return {"status": "error", "message": "No corrections to export"}

    try:
        with open(CORRECTIONS_FILE) as f:
            corrections = json.load(f)

        # Build manifest format
        manifest = {}
        for path, correction in corrections.items():
            manifest[path] = {
                "group": correction["group"],
                "subgroup": correction.get("subgroup", "undefined"),
                "filename": os.path.basename(path),
                "corrected_at": correction.get("corrected_at"),
                "labeling_method": "manual_correction"
            }

        # Save corrected manifest
        with open(CORRECTED_MANIFEST_FILE, 'w') as f:
            json.dump(manifest, f, indent=2)

        return {
            "status": "ok",
            "count": len(manifest),
            "path": str(CORRECTED_MANIFEST_FILE)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/labeled-paths")
async def get_labeled_paths():
    """Get list of paths that have already been labeled."""
    try:
        labeled = set()

        # Check corrections file
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE) as f:
                corrections = json.load(f)
                labeled.update(corrections.keys())

        return {"count": len(labeled), "paths": list(labeled)}
    except Exception as e:
        return {"count": 0, "paths": [], "error": str(e)}


@app.post("/label")
async def save_label(request: Request):
    """Save a label from the AudioLabeler UI."""
    try:
        data = await request.json()
        path = data.get("path")
        group = data.get("group")
        multi_label = data.get("multi_label", False)

        # For multi-label, we don't require a single group
        if not path or (not group and not multi_label):
            return {"status": "error", "message": "Missing path or group"}

        # Load existing corrections
        corrections = {}
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE) as f:
                corrections = json.load(f)

        # Build correction entry
        correction = {
            "previous_group": data.get("previous_group"),
            "previous_subgroup": data.get("previous_subgroup"),
            "source": data.get("source", "manual"),
            "corrected_at": datetime.now().isoformat(),
            "filename": os.path.basename(path),
            "roomy": data.get("roomy", False),
            "has_bleed": data.get("has_bleed", False),
            "bleed_instruments": data.get("bleed_instruments", []),
            "is_mix": data.get("is_mix", False),
        }

        # Handle subgroup correction
        if data.get("subgroup"):
            correction["subgroup"] = data.get("subgroup")
        if data.get("subgroups"):
            correction["subgroups"] = data.get("subgroups")  # Multi-subgroup support

        if multi_label:
            # Multi-label mode: regions with time-based labels
            correction["multi_label"] = True
            correction["regions"] = data.get("regions", [])
            correction["duration"] = data.get("duration", 0)
            # Set group to first region's first label for backwards compat
            if correction["regions"]:
                first_labels = correction["regions"][0].get("labels", [])
                correction["group"] = first_labels[0] if first_labels else "undefined"
            else:
                correction["group"] = "undefined"
        else:
            correction["group"] = group

        corrections[path] = correction

        # Save corrections
        with open(CORRECTIONS_FILE, 'w') as f:
            json.dump(corrections, f, indent=2)

        return {"status": "ok", "path": path, "group": correction.get("group")}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post("/push-corrections")
async def push_corrections_to_manifest(user: dict = Depends(get_current_user)):
    """Apply all corrections to the unified manifest. Requires authentication."""
    try:
        from collections import defaultdict

        manifest_path = MANIFESTS_DIR / "unified_manifest.json"

        if not manifest_path.exists():
            return {"status": "error", "message": "Manifest not found"}

        if not CORRECTIONS_FILE.exists():
            return {"status": "error", "message": "No corrections to apply"}

        # Load manifest and corrections
        with open(manifest_path, 'rb') as f:
            manifest = orjson.loads(f.read())

        with open(CORRECTIONS_FILE, 'rb') as f:
            corrections = orjson.loads(f.read())

        # Index manifest by path
        path_to_idx = {e['audio_path']: i for i, e in enumerate(manifest['entries'])}

        # Apply corrections
        applied = 0
        not_found = 0

        for path, correction in corrections.items():
            if path in path_to_idx:
                idx = path_to_idx[path]
                entry = manifest['entries'][idx]

                new_group = correction.get('group', entry['group'])
                roomy = correction.get('roomy', False)

                # Handle _roomy suffix
                if new_group.endswith('_roomy'):
                    new_group = new_group.replace('_roomy', '')
                    roomy = True

                # Update entry
                entry['group'] = new_group
                if correction.get('subgroup'):
                    entry['subgroup'] = correction['subgroup']
                entry['roomy'] = roomy
                entry['has_bleed'] = correction.get('has_bleed', False)
                entry['bleed_instruments'] = correction.get('bleed_instruments', [])
                entry['manually_corrected'] = True

                # Multi-label support
                if correction.get('multi_label'):
                    entry['multi_label'] = True
                    entry['regions'] = correction.get('regions', [])
                    entry['duration'] = correction.get('duration', 0)

                applied += 1
            else:
                not_found += 1

        # Recalculate group stats
        group_counts = defaultdict(int)
        for e in manifest['entries']:
            group_counts[e['group']] += 1

        manifest['groups'] = dict(group_counts)
        manifest['corrections_applied_at'] = datetime.now().isoformat()

        # Save manifest
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)

        # Regenerate group files
        groups = defaultdict(list)
        for e in manifest['entries']:
            groups[e['group']].append(e['audio_path'])

        for group, paths in groups.items():
            group_file = MANIFESTS_DIR / f"{group}_files.txt"
            with open(group_file, 'w') as f:
                f.write('\n'.join(sorted(paths)))

        return {
            "status": "ok",
            "applied": applied,
            "not_found": not_found,
            "total_corrections": len(corrections),
            "groups": dict(group_counts)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ============================================================================
# MASTER MANIFEST ENDPOINTS - Single Source of Truth
# ============================================================================

MASTER_MANIFEST_PATH = Path("/home/arlo/gcs-bucket/Manifests/master_manifest.json")

def _load_master_manifest():
    """Load master manifest with caching and orjson for speed."""
    global _master_manifest_cache
    if not MASTER_MANIFEST_PATH.exists():
        return None
    try:
        current_mtime = MASTER_MANIFEST_PATH.stat().st_mtime
        if _master_manifest_cache["data"] is not None and _master_manifest_cache["mtime"] == current_mtime:
            return _master_manifest_cache["data"]
        with open(MASTER_MANIFEST_PATH, 'rb') as f:
            data = orjson.loads(f.read())
        _master_manifest_cache["data"] = data
        _master_manifest_cache["mtime"] = current_mtime
        return data
    except Exception as e:
        print(f"Error loading master manifest: {e}")
        return None

@app.get("/master-manifest")
async def get_master_manifest(
    group: str = Query(None, description="Filter by group"),
    subgroup: str = Query(None, description="Filter by subgroup"),
    view: str = Query("all", description="View: all, needs_review, flagged, corrected, mix, temporal, valid_only"),
    source: str = Query(None, description="Filter by source: original, manual, classifier"),
    search: str = Query(None, description="Search in filename"),
    limit: int = Query(500, description="Max entries"),
    offset: int = Query(0, description="Offset for pagination"),
    validate_files: bool = Query(False, description="Check if files exist (slower)")
):
    """Get entries from master manifest with flexible filtering."""
    data = _load_master_manifest()
    if data is None:
        return {"status": "error", "message": "Master manifest not found. Run create_master_manifest.py first.", "entries": []}

    try:
        manifest_entries = data.get('entries', {})
        all_groups = set()
        all_subgroups = set()
        entries = []

        for path, entry in manifest_entries.items():
            if not isinstance(entry, dict):
                continue

            # Skip files that don't exist if validate_files is true or view is valid_only
            if (validate_files or view == 'valid_only') and not Path(path).exists():
                continue

            entry_group = entry.get('group', 'undefined')
            entry_subgroup = entry.get('subgroup', 'undefined')
            entry_flags = entry.get('flags', [])
            entry_source = entry.get('source', 'original')

            all_groups.add(entry_group)
            if entry_subgroup and entry_subgroup != 'undefined':
                all_subgroups.add(entry_subgroup)

            # Apply filters
            if group and entry_group != group:
                continue
            if subgroup and entry_subgroup != subgroup:
                continue
            if source and entry_source != source:
                continue
            if search and search.lower() not in entry.get('filename', '').lower():
                continue

            # Apply view filters
            if view == 'needs_review' and 'needs_review' not in entry_flags and 'low_confidence' not in entry_flags:
                continue
            if view == 'flagged' and 'auto_corrected' not in entry_flags and 'needs_review' not in entry_flags:
                continue
            if view == 'corrected' and entry_source != 'manual':
                continue
            if view == 'mix' and not entry.get('is_mix'):
                continue
            if view == 'temporal' and 'has_temporal' not in entry_flags:
                continue

            entries.append({
                'path': path,
                'group': entry_group,
                'subgroup': entry_subgroup,
                'original_group': entry.get('original_group', entry_group),
                'original_subgroup': entry.get('original_subgroup', 'undefined'),
                'is_mix': entry.get('is_mix', False),
                'roomy': entry.get('roomy', False),
                'bleed_instruments': entry.get('bleed_instruments', []),
                'regions': entry.get('regions', []),
                'source': entry_source,
                'confidence': entry.get('confidence', 1.0),
                'flags': entry_flags,
                'filename': entry.get('filename', os.path.basename(path)),
                'last_modified': entry.get('last_modified', ''),
            })

        total = len(entries)

        # Sort by last_modified descending
        entries.sort(key=lambda x: x.get('last_modified', ''), reverse=True)
        entries = entries[offset:offset + limit]

        return {
            'status': 'ok',
            'total': total,
            'offset': offset,
            'limit': limit,
            'entries': entries,
            'available_groups': sorted(all_groups),
            'available_subgroups': sorted(all_subgroups),
            'manifest_stats': data.get('stats', {}),
            'group_distribution': data.get('group_distribution', {}),
            'created_at': data.get('created_at', ''),
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e), 'entries': []}


@app.post("/master-manifest/edit")
async def edit_master_manifest_entry(request: Request, user: dict = Depends(get_current_user)):
    """Edit a single entry in the master manifest. Requires authentication."""
    try:
        data = await request.json()
        path = data.get('path')
        if not path:
            return {'status': 'error', 'message': 'Missing path'}

        if not MASTER_MANIFEST_PATH.exists():
            return {'status': 'error', 'message': 'Master manifest not found'}

        with open(MASTER_MANIFEST_PATH, 'rb') as f:
            manifest = orjson.loads(f.read())

        entries = manifest.get('entries', {})
        if path not in entries:
            return {'status': 'error', 'message': f'Path not found in manifest: {path}'}

        entry = entries[path]
        now = datetime.now().isoformat()

        # Update fields
        if 'group' in data:
            entry['group'] = data['group']
        if 'subgroup' in data:
            entry['subgroup'] = data['subgroup']
        if 'is_mix' in data:
            entry['is_mix'] = data['is_mix']
        if 'roomy' in data:
            entry['roomy'] = data['roomy']
        if 'bleed_instruments' in data:
            entry['bleed_instruments'] = data['bleed_instruments']
        if 'regions' in data:
            entry['regions'] = data['regions']
            if data['regions']:
                if 'has_temporal' not in entry.get('flags', []):
                    entry.setdefault('flags', []).append('has_temporal')

        entry['source'] = 'manual'
        entry['last_modified'] = now

        # Remove needs_review flag if manually edited
        if 'needs_review' in entry.get('flags', []):
            entry['flags'].remove('needs_review')
        if 'low_confidence' in entry.get('flags', []):
            entry['flags'].remove('low_confidence')

        # Save back
        with open(MASTER_MANIFEST_PATH, 'wb') as f:
            f.write(orjson.dumps(manifest))
        # Invalidate cache
        _master_manifest_cache["data"] = None
        _master_manifest_cache["mtime"] = 0

        # Also save to corrections.json for backwards compatibility
        corrections = {}
        if CORRECTIONS_FILE.exists():
            with open(CORRECTIONS_FILE) as f:
                corrections = json.load(f)

        corrections[path] = {
            'group': entry['group'],
            'subgroup': entry.get('subgroup'),
            'roomy': entry.get('roomy', False),
            'has_bleed': bool(entry.get('bleed_instruments')),
            'bleed_instruments': entry.get('bleed_instruments', []),
            'multi_label': bool(entry.get('regions')),
            'regions': entry.get('regions', []),
            'corrected_at': now,
            'source': 'master_manifest_edit',
            'filename': entry.get('filename', os.path.basename(path)),
        }

        with open(CORRECTIONS_FILE, 'w') as f:
            json.dump(corrections, f, indent=2)

        return {
            'status': 'ok',
            'path': path,
            'entry': entry,
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.get("/master-manifest/stats")
async def get_master_manifest_stats():
    """Get statistics from master manifest."""
    data = _load_master_manifest()
    if data is None:
        return {'status': 'error', 'message': 'Master manifest not found'}

    try:
        entries = data.get('entries', {})

        # Calculate current stats
        stats = {
            'total': len(entries),
            'by_source': {'original': 0, 'manual': 0, 'classifier': 0, 'flagged_correction': 0},
            'by_flag': {},
            'mix_files': 0,
            'has_temporal': 0,
            'has_bleed': 0,
            'roomy': 0,
            'needs_review': 0,
        }

        for entry in entries.values():
            source = entry.get('source', 'original')
            if source in stats['by_source']:
                stats['by_source'][source] += 1

            for flag in entry.get('flags', []):
                stats['by_flag'][flag] = stats['by_flag'].get(flag, 0) + 1

            if entry.get('is_mix'):
                stats['mix_files'] += 1
            if entry.get('regions'):
                stats['has_temporal'] += 1
            if entry.get('bleed_instruments'):
                stats['has_bleed'] += 1
            if entry.get('roomy'):
                stats['roomy'] += 1
            if 'needs_review' in entry.get('flags', []) or 'low_confidence' in entry.get('flags', []):
                stats['needs_review'] += 1

        return {
            'status': 'ok',
            'stats': stats,
            'group_distribution': data.get('group_distribution', {}),
            'subgroup_distribution': data.get('subgroup_distribution', {}),
            'created_at': data.get('created_at', ''),
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.post("/master-manifest/refresh")
async def refresh_master_manifest(user: dict = Depends(get_current_user)):
    """Regenerate master manifest from all sources. Requires authentication."""
    try:
        import subprocess
        script_path = Path("/home/arlo/Data/create_master_manifest.py")
        if not script_path.exists():
            return {'status': 'error', 'message': 'create_master_manifest.py not found'}

        # Run synchronously since it's fast enough
        result = subprocess.run(
            ['python3', str(script_path)],
            capture_output=True,
            text=True,
            timeout=120
        )

        if result.returncode != 0:
            return {'status': 'error', 'message': f'Script failed: {result.stderr}'}

        # Invalidate cache after refresh
        _master_manifest_cache["data"] = None
        _master_manifest_cache["mtime"] = 0

        # Return updated stats
        data = _load_master_manifest()
        if data:
            return {
                'status': 'ok',
                'message': 'Master manifest refreshed',
                'stats': data.get('stats', {}),
                'total': len(data.get('entries', {})),
            }
        return {'status': 'ok', 'message': 'Refresh completed'}
    except subprocess.TimeoutExpired:
        return {'status': 'error', 'message': 'Refresh timed out'}
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


# ============================================================================
# TRAINING ENDPOINTS
# ============================================================================

TRAINING_STATUS_FILE = Path("/home/arlo/Data/training_status.json")
_classifier_info_cache = {}  # Cache classifier info by type and model mtime

def get_classifier_info(classifier_type):
    """Get info about a classifier's current state."""
    global _classifier_info_cache
    base_dirs = {
        'instrument': Path('/home/arlo/Data/latent_classifier'),
        'subgroup': Path('/home/arlo/Data/subgroup_classifiers'),
        'multilabel': Path('/home/arlo/Data/multilabel_classifier'),
        'mix_v3': Path('/home/arlo/Data/mix_classifier'),
        'other': Path('/home/arlo/Data/other_classifier'),
    }

    base_dir = base_dirs.get(classifier_type)
    if not base_dir or not base_dir.exists():
        return None

    # Check cache - use predictions file mtime as cache key
    pred_file = base_dir / 'predictions.json'
    cache_key = classifier_type
    pred_mtime = pred_file.stat().st_mtime if pred_file.exists() else 0

    cached = _classifier_info_cache.get(cache_key)
    if cached and cached.get('_pred_mtime') == pred_mtime:
        return {k: v for k, v in cached.items() if not k.startswith('_')}

    info = {
        'type': classifier_type,
        'has_model': False,
        'model_date': None,
        'predictions_count': 0,
        'validation_accuracy': None,
        '_pred_mtime': pred_mtime,  # For cache validation
    }

    # Check for model - use direct path check first for speed
    model_path = base_dir / 'model.pt'
    if model_path.exists():
        info['has_model'] = True
        info['model_date'] = datetime.fromtimestamp(model_path.stat().st_mtime).isoformat()
        info['model_path'] = str(model_path)
    else:
        # Fallback to glob for other .pt files
        model_files = list(base_dir.glob('*.pt'))
        if model_files:
            info['has_model'] = True
            newest = max(model_files, key=lambda x: x.stat().st_mtime)
            info['model_date'] = datetime.fromtimestamp(newest.stat().st_mtime).isoformat()
            info['model_path'] = str(newest)

    # Check predictions - use orjson for speed
    if pred_file.exists():
        try:
            with open(pred_file, 'rb') as f:
                pred_data = orjson.loads(f.read())
            preds = pred_data.get('predictions', pred_data.get('results', []))
            info['predictions_count'] = len(preds) if isinstance(preds, list) else len(preds.keys()) if isinstance(preds, dict) else 0
        except:
            pass

    # Check validation results - use orjson for speed
    val_file = base_dir / 'validation_results.json'
    if val_file.exists():
        try:
            with open(val_file, 'rb') as f:
                val_data = orjson.loads(f.read())
            summary = val_data.get('summary', {})
            info['validation_accuracy'] = summary.get('accuracy', summary.get('overall_accuracy'))
            info['validation_date'] = datetime.fromtimestamp(val_file.stat().st_mtime).isoformat()
        except:
            pass

    # Cache the result
    _classifier_info_cache[cache_key] = info
    # Return copy without internal fields
    return {k: v for k, v in info.items() if not k.startswith('_')}


@app.get("/training/status")
async def get_training_status():
    """Get training status for all classifiers."""
    classifiers = ['instrument', 'subgroup', 'multilabel', 'mix_v3', 'other']

    status = {
        'classifiers': {},
        'corrections_pending': 0,
        'last_training': None,
    }

    for clf_type in classifiers:
        info = get_classifier_info(clf_type)
        if info:
            status['classifiers'][clf_type] = info

    # Count pending corrections (edits since last training)
    if CORRECTIONS_FILE.exists():
        try:
            with open(CORRECTIONS_FILE) as f:
                corrections = json.load(f)
            status['corrections_pending'] = len(corrections)
        except:
            pass

    # Check training status file
    if TRAINING_STATUS_FILE.exists():
        try:
            with open(TRAINING_STATUS_FILE) as f:
                train_status = json.load(f)
            status['last_training'] = train_status.get('last_training')
            status['training_history'] = train_status.get('history', [])[-10:]  # Last 10
        except:
            pass

    return status


@app.post("/training/retrain")
async def trigger_retraining(request: Request, user: dict = Depends(get_current_user)):
    """Trigger retraining of a classifier. Requires authentication."""
    try:
        data = await request.json()
        classifier_type = data.get('classifier')

        if not classifier_type:
            return {'status': 'error', 'message': 'Missing classifier type'}

        # Training configurations with full commands
        manifest_path = '/home/arlo/gcs-bucket/Manifests/combined_manifest.json'
        corrections_path = str(CORRECTIONS_FILE)

        training_configs = {
            'instrument': {
                'script': '/home/arlo/Data/latent_instrument_classifier.py',
                'args': f'--mode train --manifest {manifest_path} --corrections {corrections_path} --output-dir /home/arlo/Data/latent_classifier'
            },
            'subgroup': {
                'script': '/home/arlo/Data/latent_instrument_classifier.py',
                'args': f'--mode train --manifest {manifest_path} --corrections {corrections_path} --output-dir /home/arlo/Data/subgroup_classifiers --subgroup-mode'
            },
            'multilabel': {
                'script': '/home/arlo/Data/latent_multilabel_classifier.py',
                'args': f'--mode train --manifest {manifest_path} --output-dir /home/arlo/Data/multilabel_classifier'
            },
            'mix_v3': {
                'script': '/home/arlo/Data/learn_mix_transform_v3.py',
                'args': ''
            },
            'other': {
                'script': '/home/arlo/Data/other_stem_classifier.py',
                'args': '--mode train --output-dir /home/arlo/Data/other_classifier'
            },
        }

        config = training_configs.get(classifier_type)
        if not config or not Path(config['script']).exists():
            return {'status': 'error', 'message': f'Training script not found for {classifier_type}'}

        # Start training in background
        import subprocess
        log_file = f'/tmp/training_{classifier_type}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'

        # Use nohup to run in background
        cmd = f'nohup python3 {config["script"]} {config["args"]} > {log_file} 2>&1 &'
        subprocess.Popen(cmd, shell=True)

        # Update training status
        train_status = {}
        if TRAINING_STATUS_FILE.exists():
            try:
                with open(TRAINING_STATUS_FILE) as f:
                    train_status = json.load(f)
            except:
                pass

        train_status.setdefault('history', []).append({
            'classifier': classifier_type,
            'started_at': datetime.now().isoformat(),
            'log_file': log_file,
            'status': 'running',
        })
        train_status['last_training'] = datetime.now().isoformat()

        with open(TRAINING_STATUS_FILE, 'w') as f:
            json.dump(train_status, f, indent=2)

        return {
            'status': 'ok',
            'message': f'Training started for {classifier_type}',
            'log_file': log_file,
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.get("/training/job/{job_index}")
async def get_training_job_status(job_index: int):
    """Get status of a specific training job and update if completed."""
    if not TRAINING_STATUS_FILE.exists():
        return {'status': 'error', 'message': 'No training history'}

    try:
        with open(TRAINING_STATUS_FILE) as f:
            train_status = json.load(f)

        history = train_status.get('history', [])
        if job_index < 0 or job_index >= len(history):
            return {'status': 'error', 'message': 'Job index out of range'}

        job = history[job_index]
        log_file = job.get('log_file')
        classifier_type = job.get('classifier')

        # If still marked as running, check if completed
        if job.get('status') == 'running' and log_file:
            log_path = Path(log_file)
            if log_path.exists():
                # Read last lines of log to check completion
                with open(log_path, 'r') as f:
                    content = f.read()

                # Check for completion markers
                if 'Model saved to' in content or 'Training complete' in content or 'saved to' in content.lower():
                    job['status'] = 'completed'
                    job['completed_at'] = datetime.now().isoformat()

                    # Try to extract accuracy from model
                    info = get_classifier_info(classifier_type)
                    if info:
                        job['validation_accuracy'] = info.get('validation_accuracy')
                        job['model_path'] = info.get('model_path')

                    # Save updated status
                    with open(TRAINING_STATUS_FILE, 'w') as f:
                        json.dump(train_status, f, indent=2)

                elif 'error' in content.lower() or 'traceback' in content.lower():
                    job['status'] = 'failed'
                    job['error'] = content[-500:]  # Last 500 chars for error context
                    with open(TRAINING_STATUS_FILE, 'w') as f:
                        json.dump(train_status, f, indent=2)

        # Read log tail for progress
        log_tail = ''
        if log_file and Path(log_file).exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                log_tail = ''.join(lines[-20:])  # Last 20 lines

        return {
            'status': 'ok',
            'job': job,
            'log_tail': log_tail,
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.get("/training/check-all")
async def check_all_training_jobs():
    """Check and update status of all running training jobs."""
    if not TRAINING_STATUS_FILE.exists():
        return {'status': 'ok', 'updated': 0}

    try:
        with open(TRAINING_STATUS_FILE) as f:
            train_status = json.load(f)

        updated = 0
        history = train_status.get('history', [])

        for job in history:
            if job.get('status') != 'running':
                continue

            log_file = job.get('log_file')
            classifier_type = job.get('classifier')

            if not log_file or not Path(log_file).exists():
                continue

            with open(log_file, 'r') as f:
                content = f.read()

            if 'Model saved to' in content or 'Training complete' in content:
                job['status'] = 'completed'
                job['completed_at'] = datetime.now().isoformat()
                info = get_classifier_info(classifier_type)
                if info:
                    job['validation_accuracy'] = info.get('validation_accuracy')
                    job['model_path'] = info.get('model_path')
                updated += 1
            elif 'Error' in content or 'Traceback' in content:
                job['status'] = 'failed'
                job['error'] = content[-500:]
                updated += 1

        if updated > 0:
            with open(TRAINING_STATUS_FILE, 'w') as f:
                json.dump(train_status, f, indent=2)

        return {
            'status': 'ok',
            'updated': updated,
            'history': history[-10:]  # Return last 10 jobs
        }
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# ============== SoundSpace Integration ==============
# Mount the soundspace visualization app under /soundspace
def mount_soundspace():
    """Mount the soundspace visualization app."""
    try:
        soundspace_path = Path("/home/arlo/do-repo/home/arlo/soundspace")
        if soundspace_path.exists():
            sys.path.insert(0, str(soundspace_path))
            sys.path.insert(0, str(soundspace_path / "viz"))
            from server import app as soundspace_app
            # Mount at /soundspace - nginx will route /space/ here
            app.mount("/soundspace", soundspace_app)
            print("SoundSpace visualization mounted at /soundspace")
            return True
    except Exception as e:
        print(f"Warning: Could not mount SoundSpace: {e}")
    return False

# Try to mount soundspace on import
_soundspace_mounted = mount_soundspace()


if __name__ == "__main__":
    print("Starting Monitor Service on port 8096...")
    if _soundspace_mounted:
        print("  - SoundSpace visualization: /soundspace")
    uvicorn.run(app, host="127.0.0.1", port=8096, log_level="info")
