#!/usr/bin/env python3
"""
Data Monitor API
Provides statistics and folder details from the GCS bucket mount
Uses subprocess for efficient file counting on GCS FUSE
"""

import os
import re
import subprocess
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from collections import defaultdict
import time

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="Data Monitor API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# GCS bucket mount path
GCS_MOUNT_PATH = Path("/home/arlo/gcs-bucket")

# Known folders to monitor
MONITORED_FOLDERS = [
    "protools",
    "protoolsA",
    "drum_bus",
    "drum_midi",
    "BasicPitch",
    "mel_spectrograms"
]

# Cache for stats (avoid repeated slow filesystem scans)
STATS_CACHE = {
    "data": None,
    "timestamp": 0,
    "ttl": 300  # 5 minutes cache
}

FOLDER_CACHE = {}


def is_date_folder(name: str) -> bool:
    """Check if folder name matches date pattern YYYY-MM-DD"""
    return bool(re.match(r'^\d{4}-\d{2}-\d{2}$', name))


def run_command(cmd: List[str], timeout: int = 60) -> str:
    """Run a shell command and return output"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, Exception):
        return ""


def get_folder_file_count(folder_path: str) -> int:
    """Get recursive file count using find command"""
    output = run_command(["find", folder_path, "-type", "f"], timeout=120)
    if output:
        return len(output.split('\n'))
    return 0


def get_folder_size_bytes(folder_path: str) -> int:
    """Get folder size in bytes using du command"""
    output = run_command(["du", "-sb", folder_path], timeout=120)
    if output:
        parts = output.split()
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                pass
    return 0


def get_subdir_count(folder_path: str) -> int:
    """Count immediate subdirectories"""
    output = run_command(["find", folder_path, "-maxdepth", "1", "-type", "d"], timeout=30)
    if output:
        # Subtract 1 for the folder itself
        return max(0, len(output.split('\n')) - 1)
    return 0


def list_subdirs(folder_path: Path) -> List[Path]:
    """List subdirectories"""
    try:
        return [d for d in folder_path.iterdir() if d.is_dir()]
    except (OSError, IOError):
        return []


def count_files_in_dir(folder_path: str) -> int:
    """Count files recursively in a directory"""
    output = run_command(["find", folder_path, "-type", "f"], timeout=60)
    if output:
        return len(output.split('\n'))
    return 0


def get_dir_size(folder_path: str) -> int:
    """Get directory size in bytes"""
    output = run_command(["du", "-sb", folder_path], timeout=60)
    if output:
        parts = output.split()
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                pass
    return 0


@app.get("/api/monitor/stats")
async def get_stats():
    """Get overall statistics for all monitored folders"""
    try:
        # Check cache
        now = time.time()
        if STATS_CACHE["data"] and (now - STATS_CACHE["timestamp"]) < STATS_CACHE["ttl"]:
            cached_data = STATS_CACHE["data"].copy()
            cached_data["cached"] = True
            cached_data["cacheAge"] = int(now - STATS_CACHE["timestamp"])
            return cached_data

        stats = {
            "totalFiles": 0,
            "totalSize": 0,
            "totalFolders": 0,
            "recentlyModified": 0,
            "folders": {},
            "recentFiles": [],
            "cached": False,
            "cacheAge": 0
        }

        for folder_name in MONITORED_FOLDERS:
            folder_path = GCS_MOUNT_PATH / folder_name
            folder_str = str(folder_path)

            if not folder_path.exists():
                stats["folders"][folder_name] = {"count": 0, "size": 0, "subdirs": 0}
                continue

            try:
                # Get actual recursive file count
                file_count = get_folder_file_count(folder_str)

                # Get actual folder size
                folder_size = get_folder_size_bytes(folder_str)

                # Count subdirectories
                subdir_count = get_subdir_count(folder_str)

                stats["folders"][folder_name] = {
                    "count": file_count,
                    "size": folder_size,
                    "subdirs": subdir_count
                }
                stats["totalFiles"] += file_count
                stats["totalSize"] += folder_size
                stats["totalFolders"] += subdir_count

            except Exception as e:
                print(f"Error reading folder {folder_name}: {e}")
                stats["folders"][folder_name] = {"count": 0, "size": 0, "subdirs": 0}

        # Cache the results
        STATS_CACHE["data"] = stats
        STATS_CACHE["timestamp"] = now

        return stats

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


@app.get("/api/monitor/folder/{folder_id}")
async def get_folder_details(folder_id: str):
    """Get detailed breakdown for a specific folder"""
    try:
        if folder_id not in MONITORED_FOLDERS:
            raise HTTPException(status_code=404, detail="Folder not found")

        folder_path = GCS_MOUNT_PATH / folder_id

        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Folder does not exist")

        # Check cache
        cache_key = folder_id
        now = time.time()
        if cache_key in FOLDER_CACHE:
            cached = FOLDER_CACHE[cache_key]
            if (now - cached["timestamp"]) < 300:  # 5 min cache
                return cached["data"]

        result: Dict[str, Any] = {}

        # Get list of subdirectories
        subdirs = list_subdirs(folder_path)

        date_folders = [d for d in subdirs if is_date_folder(d.name)]
        session_folders = [d for d in subdirs if not is_date_folder(d.name)]

        # Date-based breakdown (for protools, BasicPitch, mel_spectrograms)
        if date_folders:
            by_date = {}
            # Process last 30 date folders
            for date_dir in sorted(date_folders, key=lambda x: x.name, reverse=True)[:30]:
                try:
                    file_count = count_files_in_dir(str(date_dir))
                    dir_size = get_dir_size(str(date_dir))
                    by_date[date_dir.name] = {
                        "count": file_count,
                        "size": dir_size
                    }
                except Exception:
                    pass
            result["byDate"] = by_date

        # Session-based breakdown (for drum_bus, drum_midi)
        if session_folders:
            sessions = []
            # Process all sessions (up to 100)
            for session_dir in sorted(session_folders, key=lambda x: x.name)[:100]:
                try:
                    file_count = count_files_in_dir(str(session_dir))
                    dir_size = get_dir_size(str(session_dir))
                    sessions.append({
                        "name": session_dir.name,
                        "fileCount": file_count,
                        "size": dir_size
                    })
                except Exception:
                    pass
            result["sessions"] = sessions
            result["totalSessions"] = len(subdirs)

        # File type breakdown
        by_type: Dict[str, Dict[str, int]] = defaultdict(lambda: {"count": 0, "size": 0})

        # Use find to get file list with extensions
        output = run_command(["find", str(folder_path), "-type", "f", "-printf", "%s %f\n"], timeout=120)
        if output:
            for line in output.split('\n'):
                if ' ' in line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        try:
                            size = int(parts[0])
                            filename = parts[1]
                            ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else 'no_ext'
                            by_type[ext]["count"] += 1
                            by_type[ext]["size"] += size
                        except ValueError:
                            pass

        result["byType"] = dict(by_type)

        # Cache results
        FOLDER_CACHE[cache_key] = {
            "data": result,
            "timestamp": now
        }

        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get folder details: {str(e)}")


@app.get("/api/monitor/refresh")
async def refresh_cache():
    """Force refresh the cache"""
    global STATS_CACHE, FOLDER_CACHE
    STATS_CACHE = {"data": None, "timestamp": 0, "ttl": 300}
    FOLDER_CACHE = {}
    return {"message": "Cache cleared"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "monitor_api",
        "gcs_mount_exists": GCS_MOUNT_PATH.exists()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8096)
