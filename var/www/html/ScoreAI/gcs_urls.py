#!/usr/bin/env python3
"""
GCS Signed URL generation for scalable file access.

This module generates signed URLs that allow clients to download directly
from GCS without going through the backend, enabling:
- Better performance (no backend bottleneck)
- Horizontal scaling (stateless backends)
- Lower bandwidth costs (direct from GCS/CDN)
- Automatic CDN caching
"""

import datetime
from pathlib import Path
from typing import Optional
from google.cloud import storage

# GCS bucket name
BUCKET_NAME = "score-ai-generations"

# Default URL expiration (1 hour)
DEFAULT_EXPIRATION = datetime.timedelta(hours=1)


def get_storage_client():
    """Get GCS storage client (cached)."""
    if not hasattr(get_storage_client, '_client'):
        get_storage_client._client = storage.Client()
    return get_storage_client._client


def generate_signed_url(
    gcs_path: str,
    expiration: datetime.timedelta = DEFAULT_EXPIRATION,
    method: str = "GET"
) -> str:
    """
    Generate a signed URL for direct GCS access.

    Args:
        gcs_path: GCS path (gs://bucket/path or just path)
        expiration: How long the URL is valid
        method: HTTP method (GET, PUT, etc.)

    Returns:
        Signed URL that expires after specified time

    Example:
        >>> url = generate_signed_url("gs://bucket/audio/file.wav")
        >>> # Client downloads directly from GCS using this URL
    """
    # Parse GCS path
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]  # Remove gs://

    parts = gcs_path.split("/", 1)
    bucket_name = parts[0] if len(parts) > 1 else BUCKET_NAME
    blob_name = parts[1] if len(parts) > 1 else parts[0]

    # Get storage client and bucket
    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Generate signed URL
    url = blob.generate_signed_url(
        version="v4",
        expiration=expiration,
        method=method,
    )

    return url


def generate_upload_signed_url(
    gcs_path: str,
    content_type: str = "audio/wav",
    expiration: datetime.timedelta = DEFAULT_EXPIRATION
) -> str:
    """
    Generate a signed URL for direct upload to GCS (client-side upload).

    Args:
        gcs_path: Where to upload in GCS
        content_type: MIME type of file
        expiration: How long the URL is valid

    Returns:
        Signed URL for PUT request

    Example:
        >>> url = generate_upload_signed_url("audio/user123/file.wav")
        >>> # Client uploads directly: fetch(url, {method: 'PUT', body: audioBlob})
    """
    return generate_signed_url(gcs_path, expiration, method="PUT")


def get_public_url(gcs_path: str) -> str:
    """
    Get public URL for a GCS object (if bucket is public).

    Args:
        gcs_path: GCS path

    Returns:
        Public HTTP URL
    """
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    return f"https://storage.googleapis.com/{gcs_path}"


def get_cdn_url(gcs_path: str) -> str:
    """
    Get CDN URL for a GCS object (after Cloud CDN is enabled).

    Args:
        gcs_path: GCS path

    Returns:
        CDN URL (uses Cloud CDN if configured)
    """
    # For now, return storage URL
    # After Cloud CDN setup, this would return custom domain
    return get_public_url(gcs_path)


def file_exists_in_gcs(gcs_path: str) -> bool:
    """
    Check if file exists in GCS.

    Args:
        gcs_path: GCS path to check

    Returns:
        True if file exists, False otherwise
    """
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    parts = gcs_path.split("/", 1)
    bucket_name = parts[0] if len(parts) > 1 else BUCKET_NAME
    blob_name = parts[1] if len(parts) > 1 else parts[0]

    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    return blob.exists()


def get_file_metadata(gcs_path: str) -> Optional[dict]:
    """
    Get metadata for a GCS file.

    Args:
        gcs_path: GCS path

    Returns:
        Dict with metadata (size, content_type, created, etc.) or None
    """
    if gcs_path.startswith("gs://"):
        gcs_path = gcs_path[5:]

    parts = gcs_path.split("/", 1)
    bucket_name = parts[0] if len(parts) > 1 else BUCKET_NAME
    blob_name = parts[1] if len(parts) > 1 else parts[0]

    client = get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists():
        return None

    blob.reload()  # Fetch metadata

    return {
        "name": blob.name,
        "size": blob.size,
        "content_type": blob.content_type,
        "created": blob.time_created.isoformat() if blob.time_created else None,
        "updated": blob.updated.isoformat() if blob.updated else None,
        "public_url": get_public_url(gcs_path),
        "cdn_url": get_cdn_url(gcs_path),
    }


def list_gcs_files(prefix: str, bucket_name: str = BUCKET_NAME) -> list[dict]:
    """
    List all files in GCS under a prefix.

    Args:
        prefix: GCS prefix to search (e.g., 'audiofiles/ace_step_output_123')
        bucket_name: Bucket name, default: BUCKET_NAME

    Returns:
        List of dicts with file metadata

    Example:
        >>> files = list_gcs_files('audiofiles/ace_step_output_123')
        >>> # Returns: [
        >>> #   {
        >>> #     'filename': 'output.wav',
        >>> #     'gcs_path': 'gs://bucket/audiofiles/ace_step_output_123/output.wav',
        >>> #     'size': 1048576,
        >>> #     'created': '2025-11-14T...'
        >>> #   },
        >>> #   ...
        >>> # ]
    """
    client = get_storage_client()
    bucket = client.bucket(bucket_name)

    # List all blobs with the prefix
    blobs = bucket.list_blobs(prefix=prefix)

    files = []
    for blob in blobs:
        # Skip directories (blobs ending with /)
        if blob.name.endswith('/'):
            continue

        files.append({
            "filename": blob.name.split('/')[-1],
            "gcs_path": f"gs://{bucket_name}/{blob.name}",
            "full_path": blob.name,
            "size": blob.size,
            "content_type": blob.content_type or "application/octet-stream",
            "created": blob.time_created.isoformat() if blob.time_created else None,
            "updated": blob.updated.isoformat() if blob.updated else None,
        })

    return files


if __name__ == "__main__":
    # Test signed URL generation
    print("Testing GCS Signed URL generation...")

    # Test with a known file (if any exists)
    test_path = "audio/anonymous/test.wav"

    try:
        url = generate_signed_url(f"gs://{BUCKET_NAME}/{test_path}")
        print(f"✅ Generated signed URL:")
        print(f"   {url[:100]}...")
        print(f"   Expires in: {DEFAULT_EXPIRATION}")
    except Exception as e:
        print(f"⚠️  Error: {e}")
        print("   (This is normal if no files exist yet)")
