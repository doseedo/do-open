#!/usr/bin/env python3
"""
Create Vocal Training Manifest

Creates a vocal-only manifest using the same format as final_training_manifest_final.json
but focusing on vocal files identified from audio paths and processing pipeline.

Based on matchall.py pattern matching approach.
"""

import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Union, Dict, List
import re

# --- CONFIGURATION ---

# Master list of all source audio files
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths5.txt")

# Base directories for all feature types (matching original structure)
BASIC_PITCH_DIR = Path("/home/arlo/gcs-bucket/BasicPitch")
PIANO_ROLL_ROOT = Path("/mnt/msdd/piano_rolls")
LATENT_ROOT = Path("/mnt/msdd/dcae_latentsnew")
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens")
CONDITIONING_ROOT = Path("/mnt/msdd/evenmoreconditioning")
ALT_CONDITIONING_ROOT = Path("/mnt/msdd/moreconditioning")
NEWCONDITIONING_ROOT = Path("/mnt/msdd/newconditioning")

# --- OUTPUT FILES ---
OUTPUT_JSON = Path("/home/arlo/Data/vocal_training_manifest.json")
LOG_NO_PIANO_ROLL = Path("/home/arlo/Data/log_vocal_no_piano_roll.txt")
LOG_NO_LATENT = Path("/home/arlo/Data/log_vocal_no_latent.txt")
LOG_NO_ENCODEC = Path("/home/arlo/Data/log_vocal_no_encodec.txt")
LOG_NO_CONDITIONING = Path("/home/arlo/Data/log_vocal_no_conditioning.txt")
VOCAL_PATHS_LOG = Path("/home/arlo/Data/vocal_audio_paths.txt")

def is_vocal_file(audio_path_str: str) -> bool:
    """
    Determine if an audio file is vocal-related based on path patterns.
    """
    audio_path = Path(audio_path_str)
    path_str = str(audio_path).upper()
    filename = audio_path.name.upper()

    # Session-level patterns (in directory names)
    vocal_session_patterns = [
        'VOX_SESS',
        'VOX_SESSION',
        'VOCAL_SESS',
        'VOCAL_SESSION',
        'VOCALS',
        'VOX',
        '_VOX_',
        'LEAD_VOX',
        'BACKING_VOX'
    ]

    # File-level patterns (in filenames)
    vocal_file_patterns = [
        r'\bVOX\b',           # VOX as standalone word
        r'\bVOCAL\b',         # VOCAL as standalone word
        r'\bVOICE\b',         # VOICE as standalone word
        r'\bLEAD\s*VOX\b',    # LEAD VOX
        r'\bLEAD\s*VOCAL\b',  # LEAD VOCAL
        r'\bBG\s*VOX\b',      # BG VOX
        r'\bBACKING\b',       # BACKING
        r'\bCHOIR\b',         # CHOIR
        r'\bLVOX\b',          # LVOX
        r'\bRVOX\b',          # RVOX
        r'^VOX\.',            # Files starting with VOX.
        r'^VOCAL\.',          # Files starting with VOCAL.
        r'^VOICE\.',          # Files starting with VOICE.
    ]

    # Check session-level patterns first
    for pattern in vocal_session_patterns:
        if pattern in path_str:
            return True

    # Check file-level patterns
    for pattern in vocal_file_patterns:
        if re.search(pattern, filename):
            return True

    return False

def extract_session_info(audio_path_str: str) -> Dict[str, str]:
    """Extract session information for grouping and metadata."""
    path_parts = Path(audio_path_str).parts

    group = "vocal"
    sub_group = "vocal_track"

    # Try to determine more specific grouping
    filename = Path(audio_path_str).name.upper()

    if 'LEAD' in filename or 'LEAD' in audio_path_str.upper():
        sub_group = "lead_vocal"
    elif 'BACKING' in filename or 'BG' in filename:
        sub_group = "backing_vocal"
    elif 'CHOIR' in filename:
        sub_group = "choir"
    elif 'HARMONY' in filename:
        sub_group = "harmony_vocal"

    return {"group": group, "sub_group": sub_group}

def find_piano_roll_match(audio_path: Path, piano_roll_root: Path) -> Union[str, None]:
    """Find the corresponding piano roll .npy file."""
    npy_filename = audio_path.with_suffix(".pianoroll.npy").name
    path_parts = audio_path.parts

    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path