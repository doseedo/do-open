#!/usr/bin/env python3
"""
Custom StaticFiles middleware with GCS fallback.

This extends FastAPI's StaticFiles to automatically download from GCS
if a file is not found locally.
"""

import sys
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.exceptions import HTTPException

# Add Data directory for GCS modules
sys.path.append('/home/arlo/Data')
from gcs_fallback import get_file_with_gcs_fallback


class GCSStaticFiles(StaticFiles):
    """
    StaticFiles with automatic GCS fallback.

    If a file is not found locally, it attempts to download from GCS.
    """

    def __init__(self, directory: str, gcs_prefix: str = "audiofiles", **kwargs):
        """
        Initialize GCS-aware static files.

        Args:
            directory: Local directory to serve files from
            gcs_prefix: GCS prefix to check for missing files
            **kwargs: Additional arguments for StaticFiles
        """
        super().__init__(directory=directory, **kwargs)
        self.gcs_prefix = gcs_prefix

    async def get_response(self, path: str, scope):
        """
        Override to add GCS fallback.
        """
        try:
            # Try normal static file serving first
            return await super().get_response(path, scope)
        except HTTPException as e:
            if e.status_code == 404:
                # File not found locally, try GCS
                local_path = Path(self.directory) / path
                print(f"⚠️  File not found locally: {local_path}")
                print(f"   Attempting GCS fallback...")

                restored_path = get_file_with_gcs_fallback(
                    str(local_path),
                    gcs_prefix=self.gcs_prefix
                )

                if restored_path:
                    print(f"✅ Restored from GCS: {restored_path}")
                    # Retry with the downloaded file
                    return await super().get_response(path, scope)
                else:
                    print(f"❌ File not found in GCS either")
                    raise

            # Re-raise other errors
            raise
