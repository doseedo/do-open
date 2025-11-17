#!/usr/bin/env python3
"""
Google Cloud Storage utility for handling audio generation outputs.

This module provides functions to:
- Upload latents (.pt files) and audio (.wav files) to GCS
- Download files from GCS when needed
- Manage file lifecycle (audio deleted after 7 days, latents after 90 days)

Bucket structure:
- gs://score-ai-generations/audio/{user_id}/{timestamp}.wav
- gs://score-ai-generations/latents/{user_id}/{timestamp}.pt
- gs://score-ai-generations/uploads/{filename}
- gs://score-ai-generations/audiofiles/{filename}
- gs://score-ai-generations/voice_debug/{session_id}/{filename}
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Optional, Union
import tempfile

# GCS bucket name
BUCKET_NAME = "score-ai-generations"
BUCKET_URI = f"gs://{BUCKET_NAME}"

# Default user ID for non-authenticated requests
DEFAULT_USER_ID = "anonymous"


def get_gcs_path(local_path: str, prefix: str, user_id: Optional[str] = None) -> str:
    """
    Generate a GCS path from a local path.

    Args:
        local_path: Local file path
        prefix: GCS prefix (e.g., 'audio', 'latents', 'uploads')
        user_id: User identifier (optional, defaults to anonymous)

    Returns:
        Full GCS URI (gs://bucket/prefix/user_id/filename)
    """
    filename = Path(local_path).name
    user_id = user_id or DEFAULT_USER_ID

    return f"{BUCKET_URI}/{prefix}/{user_id}/{filename}"


def upload_to_gcs(local_path: str, gcs_path: Optional[str] = None, prefix: str = "audio",
                  user_id: Optional[str] = None, make_public: bool = True) -> str:
    """
    Upload a file to Google Cloud Storage using gsutil.

    Args:
        local_path: Path to local file
        gcs_path: Full GCS URI (if None, will be generated from prefix and user_id)
        prefix: GCS prefix folder ('audio', 'latents', 'uploads', etc.)
        user_id: User identifier for organizing files
        make_public: Whether to make the file publicly readable

    Returns:
        GCS URI of uploaded file
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file not found: {local_path}")

    # Generate GCS path if not provided
    if gcs_path is None:
        gcs_path = get_gcs_path(local_path, prefix, user_id)

    # Upload using gsutil with cache control headers
    try:
        cmd = ["gsutil", "-m", "-h", "Cache-Control:public, max-age=3600", "cp", local_path, gcs_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Uploaded to GCS: {gcs_path}")

        # Make public if requested
        if make_public:
            subprocess.run(["gsutil", "acl", "ch", "-u", "AllUsers:R", gcs_path],
                          capture_output=True, check=False)

        return gcs_path
    except subprocess.CalledProcessError as e:
        print(f"❌ GCS upload failed: {e.stderr}")
        raise


def download_from_gcs(gcs_path: str, local_path: Optional[str] = None) -> str:
    """
    Download a file from Google Cloud Storage using gsutil.

    Args:
        gcs_path: Full GCS URI
        local_path: Local destination path (if None, uses temp file)

    Returns:
        Path to downloaded local file
    """
    # Create temp file if no local path specified
    if local_path is None:
        ext = Path(gcs_path).suffix
        fd, local_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)

    # Ensure directory exists
    os.makedirs(Path(local_path).parent, exist_ok=True)

    # Download using gsutil
    try:
        cmd = ["gsutil", "cp", gcs_path, local_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ Downloaded from GCS: {gcs_path} -> {local_path}")
        return local_path
    except subprocess.CalledProcessError as e:
        print(f"❌ GCS download failed: {e.stderr}")
        raise


def gcs_file_exists(gcs_path: str) -> bool:
    """
    Check if a file exists in GCS.

    Args:
        gcs_path: Full GCS URI

    Returns:
        True if file exists, False otherwise
    """
    try:
        result = subprocess.run(["gsutil", "ls", gcs_path],
                              capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError:
        return False


def get_gcs_url(gcs_path: str) -> str:
    """
    Convert GCS URI to public HTTP URL.

    Args:
        gcs_path: GCS URI (gs://bucket/path)

    Returns:
        Public HTTP URL
    """
    # gs://bucket/path -> https://storage.googleapis.com/bucket/path
    if gcs_path.startswith("gs://"):
        path = gcs_path[5:]  # Remove 'gs://'
        return f"https://storage.googleapis.com/{path}"
    return gcs_path


def save_audio_with_latents(audio_path: str, latents_tensor, user_id: Optional[str] = None,
                            metadata: Optional[dict] = None) -> dict:
    """
    Save both audio and latents to GCS with proper organization.

    Args:
        audio_path: Path to audio file (.wav)
        latents_tensor: PyTorch tensor containing latents
        user_id: User identifier
        metadata: Optional metadata dict to save alongside

    Returns:
        dict with 'audio_url', 'latents_url', 'audio_gcs', 'latents_gcs'
    """
    import torch

    # Upload audio to GCS
    audio_gcs = upload_to_gcs(audio_path, prefix="audio", user_id=user_id, make_public=True)
    audio_url = get_gcs_url(audio_gcs)

    # Save latents to temp file, then upload
    latents_filename = Path(audio_path).stem + ".pt"
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        latents_path = tmp.name
        torch.save({
            'latents': latents_tensor,
            'metadata': metadata or {}
        }, latents_path)

    try:
        latents_gcs = upload_to_gcs(latents_path, prefix="latents", user_id=user_id, make_public=False)
        latents_url = get_gcs_url(latents_gcs)
    finally:
        os.unlink(latents_path)

    return {
        'audio_url': audio_url,
        'latents_url': latents_url,
        'audio_gcs': audio_gcs,
        'latents_gcs': latents_gcs
    }


def migrate_directory_to_gcs(local_dir: str, prefix: str, user_id: Optional[str] = None,
                             delete_local: bool = False) -> list:
    """
    Migrate an entire directory to GCS.

    Args:
        local_dir: Local directory path
        prefix: GCS prefix folder
        user_id: User identifier
        delete_local: Whether to delete local files after successful upload

    Returns:
        List of GCS URIs for uploaded files
    """
    local_path = Path(local_dir)
    if not local_path.exists():
        print(f"⚠️  Directory not found: {local_dir}")
        return []

    uploaded = []
    for file_path in local_path.rglob("*"):
        if file_path.is_file():
            try:
                # Preserve subdirectory structure
                rel_path = file_path.relative_to(local_path)
                gcs_path = f"{BUCKET_URI}/{prefix}/{user_id or DEFAULT_USER_ID}/{rel_path}"

                upload_to_gcs(str(file_path), gcs_path=gcs_path, make_public=True)
                uploaded.append(gcs_path)

                if delete_local:
                    file_path.unlink()
                    print(f"🗑️  Deleted local file: {file_path}")
            except Exception as e:
                print(f"❌ Failed to upload {file_path}: {e}")

    return uploaded


def cleanup_old_local_files(directory: str, age_days: int = 1):
    """
    Clean up local files older than specified days.

    Args:
        directory: Directory to clean
        age_days: Delete files older than this many days
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return

    cutoff_time = time.time() - (age_days * 86400)
    deleted_count = 0

    for file_path in dir_path.rglob("*"):
        if file_path.is_file():
            if file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Failed to delete {file_path}: {e}")

    if deleted_count > 0:
        print(f"🗑️  Cleaned up {deleted_count} files from {directory}")


if __name__ == "__main__":
    # Test the module
    print(f"GCS Bucket: {BUCKET_URI}")
    print(f"Testing GCS connectivity...")

    # List bucket contents
    try:
        result = subprocess.run(["gsutil", "ls", BUCKET_URI],
                              capture_output=True, text=True, check=True)
        print(f"✅ GCS bucket accessible")
        print(result.stdout[:500])
    except Exception as e:
        print(f"❌ GCS bucket not accessible: {e}")
