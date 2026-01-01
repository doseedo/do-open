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
import time
import threading
import mimetypes
import subprocess
import hashlib
import base64
import secrets
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

app = FastAPI(title="Monitor Service", version="1.0.0")

# Persistent encrypted cache config
CACHE_FILE = Path("/tmp/.monitor_cache.enc")
GCS_CACHE_FILE = Path("/tmp/.gcs_cache.enc")
CACHE_TTL = 3600  # 1 hour (persisted cache can be longer)
_audio_cache = {"count": 0, "last_updated": 0, "updating": False}
_gcs_cache = None  # Loaded from gcs_cache_builder.py output

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


def get_gcs_stats():
    """Get GCS bucket stats from cache."""
    global _gcs_cache
    if _gcs_cache is None:
        _load_gcs_cache()
    return _gcs_cache

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

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
            "folders": folders,
            "cache_age": time.time() - gcs.get("generated_at", 0),
            "generated_at": gcs.get("generated_at", 0),
            "service": "monitor",
            "version": "2.3.0"
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
            "folders": [],
            "service": "monitor",
            "version": "2.3.0"
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

# Legacy path for compatibility
PREDICTIONS_FILE = Path("/home/arlo/Data/latent_classifier/predictions.json")


@app.get("/manifests")
async def list_manifests():
    """List all available manifests for the labeler."""
    manifests = []

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

    return {"manifests": manifests}


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


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    print("Starting Monitor Service on port 8096...")
    uvicorn.run(app, host="127.0.0.1", port=8096, log_level="info")
