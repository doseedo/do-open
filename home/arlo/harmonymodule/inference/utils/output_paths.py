"""
Master Output Path Configuration
Centralized path management for all file operations in genfrominterface.py

All temporary files are stored in /mnt/models/ subdirectories and then uploaded to GCS.
This provides a scalable, production-ready storage solution.
"""

from pathlib import Path

# Base directory for all temporary storage (on mounted disk)
BASE_OUTPUT_DIR = Path("/mnt/models")

# Subdirectories for different types of operations
PATHS = {
    # Audio generation outputs
    "generated_audio": BASE_OUTPUT_DIR / "generated_ui",  # Main UI generations (ACE-Step, etc.)
    "ace_step_output": BASE_OUTPUT_DIR / "generated_ui",  # ACE-Step specific outputs

    # Uploads and inputs
    "uploads": BASE_OUTPUT_DIR / "uploads",  # User uploaded files (MIDI, audio, etc.)

    # Stem separation outputs
    "stems": BASE_OUTPUT_DIR / "stems",  # Separated audio stems

    # Temporary processing directories
    "temp_recordings": BASE_OUTPUT_DIR / "temp_exports",  # Audio recordings
    "temp_fx_processing": BASE_OUTPUT_DIR / "temp_exports",  # FX processing temp files
    "temp_omnisphere": BASE_OUTPUT_DIR / "temp_exports",  # Omnisphere processing
    "temp_videos": BASE_OUTPUT_DIR / "temp_videos",  # Video processing

    # Generated content
    "images": BASE_OUTPUT_DIR / "generated_ui",  # Generated images
    "drums": BASE_OUTPUT_DIR / "generated_ui",  # Generated drum patterns
}

def get_output_path(path_type, process_id=None, subdir=None):
    """
    Get a standardized output path for a specific operation.

    Args:
        path_type (str): Type of path from PATHS dict (e.g., 'generated_audio', 'uploads')
        process_id (str, optional): Unique process ID for this operation
        subdir (str, optional): Additional subdirectory name

    Returns:
        Path: Complete path for the operation

    Examples:
        >>> get_output_path('uploads')
        Path('/mnt/models/uploads')

        >>> get_output_path('ace_step_output', process_id='abc123')
        Path('/mnt/models/generated_ui/ace_step_output_abc123')

        >>> get_output_path('stems', process_id='xyz789')
        Path('/mnt/models/stems/stems_xyz789')
    """
    if path_type not in PATHS:
        raise ValueError(f"Unknown path type: {path_type}. Valid types: {list(PATHS.keys())}")

    base_path = PATHS[path_type]

    # Handle special cases with process_id
    if process_id:
        if path_type == 'ace_step_output':
            return base_path / f"ace_step_output_{process_id}"
        elif path_type == 'stems':
            return base_path / f"stems_{process_id}"
        elif path_type == 'images':
            return base_path / f"images_{process_id}"
        elif path_type == 'drums':
            return base_path / f"drums_{process_id}"
        elif subdir:
            return base_path / subdir / process_id
        else:
            return base_path / process_id

    if subdir:
        return base_path / subdir

    return base_path

def ensure_path_exists(path):
    """
    Ensure a path exists, creating it if necessary.

    Args:
        path (Path): Path to check/create

    Returns:
        Path: The same path (for chaining)
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

# Ensure all base directories exist on import
for path in PATHS.values():
    ensure_path_exists(path)
