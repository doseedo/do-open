"""
MIDI Cache - Agent 8
====================

Efficient caching system for parsed MIDI files to avoid redundant I/O and parsing.

This module provides:
- MIDICache: SHA256-keyed caching with LRU eviction
- Parsed MIDI data + metadata storage
- Configurable cache size (default 5GB)

Pattern: Follows GapCache architecture (Agent 4)

Author: Agent 8 - Data Pipeline & Preprocessing
"""

import json
import pickle
import hashlib
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. Install with: pip install mido")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CachedMIDI:
    """
    Cached MIDI data with metadata.

    Stores both raw MIDI object and extracted metadata to avoid re-parsing.
    """
    file_id: str
    file_path: str

    # MIDI data
    midi_object: Any  # mido.MidiFile object

    # Metadata
    n_tracks: int
    n_notes: int
    duration_seconds: float
    tempo_bpm: float
    time_signature: str

    # Parsed note data (for quick access)
    notes: List[Dict[str, Any]]  # List of {pitch, velocity, start_time, duration, track}

    # Cache metadata
    timestamp: float = 0.0
    file_size_bytes: int = 0

    def to_dict(self) -> Dict:
        """Convert to dictionary (without midi_object for serialization)"""
        return {
            'file_id': self.file_id,
            'file_path': self.file_path,
            'n_tracks': self.n_tracks,
            'n_notes': self.n_notes,
            'duration_seconds': self.duration_seconds,
            'tempo_bpm': self.tempo_bpm,
            'time_signature': self.time_signature,
            'timestamp': self.timestamp,
            'file_size_bytes': self.file_size_bytes
        }


# ============================================================================
# MIDICache
# ============================================================================

class MIDICache:
    """
    Efficient caching system for parsed MIDI files.

    Speeds up data loading by caching:
    - Parsed mido.MidiFile objects
    - Extracted metadata (tempo, time signature, etc.)
    - Parsed note events

    Cache size: 5 GB (configurable)
    Cache key: SHA256 hash of MIDI file
    Eviction: LRU (Least Recently Used)

    Pattern: Follows GapCache architecture from Agent 4

    Example:
        cache = MIDICache(cache_dir=Path('cache/midi'), max_size_gb=5.0)

        # Try to get from cache
        cached_midi = cache.get(midi_file_path)
        if cached_midi is None:
            # Parse MIDI file
            midi_obj = mido.MidiFile(str(midi_file_path))
            notes = extract_notes(midi_obj)
            # ... extract metadata ...

            # Cache for future use
            cached_midi = CachedMIDI(...)
            cache.put(midi_file_path, cached_midi)
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 5.0,
        verbose: bool = False
    ):
        """
        Initialize MIDI cache.

        Args:
            cache_dir: Directory to store cache files
            max_size_gb: Maximum cache size in gigabytes
            verbose: Print cache operations
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.verbose = verbose

        # Cache metadata
        self.metadata_file = self.cache_dir / 'midi_cache_metadata.json'
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'entries': {},
            'total_size_bytes': 0,
            'hit_count': 0,
            'miss_count': 0,
            'created_at': time.time()
        }

    def _save_metadata(self):
        """Save cache metadata"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of MIDI file.

        Args:
            file_path: Path to MIDI file

        Returns:
            SHA256 hash (hex string)
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _get_cache_path(self, file_hash: str) -> Path:
        """
        Get cache file path for hash.

        Uses first 2 chars as subdirectory to avoid too many files in one dir.

        Args:
            file_hash: SHA256 hash

        Returns:
            Path to cache file
        """
        subdir = file_hash[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{file_hash}.pkl"

    def get(self, midi_file: Path) -> Optional[CachedMIDI]:
        """
        Get cached MIDI data for file.

        Args:
            midi_file: Path to MIDI file

        Returns:
            CachedMIDI if cached, None otherwise
        """
        file_hash = self._compute_file_hash(midi_file)
        cache_path = self._get_cache_path(file_hash)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_midi = pickle.load(f)

                # Update timestamp for LRU
                if file_hash in self.metadata['entries']:
                    self.metadata['entries'][file_hash]['timestamp'] = time.time()
                    self._save_metadata()

                self.metadata['hit_count'] += 1

                if self.verbose:
                    print(f"✅ Cache hit for {midi_file.name}")

                return cached_midi
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Cache read error: {e}")
                return None

        self.metadata['miss_count'] += 1
        return None

    def put(self, midi_file: Path, cached_midi: CachedMIDI):
        """
        Cache MIDI data.

        Args:
            midi_file: Path to MIDI file
            cached_midi: CachedMIDI object to cache
        """
        file_hash = self._compute_file_hash(midi_file)
        cache_path = self._get_cache_path(file_hash)

        # Check if we need to evict old entries
        self._evict_if_needed()

        # Save cached MIDI
        try:
            # Update timestamp
            cached_midi.timestamp = time.time()

            with open(cache_path, 'wb') as f:
                pickle.dump(cached_midi, f)

            # Update metadata
            file_size = cache_path.stat().st_size
            self.metadata['entries'][file_hash] = {
                'file_path': str(midi_file),
                'cache_path': str(cache_path),
                'size_bytes': file_size,
                'timestamp': cached_midi.timestamp
            }
            self.metadata['total_size_bytes'] += file_size
            self._save_metadata()

            if self.verbose:
                print(f"✅ Cached MIDI for {midi_file.name} ({file_size / 1024:.1f} KB)")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ Cache write error: {e}")

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds max size (LRU)"""
        while self.metadata['total_size_bytes'] > self.max_size_bytes:
            # Find oldest entry (LRU)
            oldest_hash = None
            oldest_time = float('inf')

            for file_hash, entry in self.metadata['entries'].items():
                if entry['timestamp'] < oldest_time:
                    oldest_time = entry['timestamp']
                    oldest_hash = file_hash

            if oldest_hash is None:
                break

            # Remove oldest entry
            entry = self.metadata['entries'][oldest_hash]
            cache_path = Path(entry['cache_path'])

            if cache_path.exists():
                cache_path.unlink()

            self.metadata['total_size_bytes'] -= entry['size_bytes']
            del self.metadata['entries'][oldest_hash]

            if self.verbose:
                print(f"⚠️ Evicted cache entry {oldest_hash[:8]}...")

        self._save_metadata()

    def clear(self):
        """Clear entire cache"""
        for entry in self.metadata['entries'].values():
            cache_path = Path(entry['cache_path'])
            if cache_path.exists():
                cache_path.unlink()

        self.metadata = {
            'entries': {},
            'total_size_bytes': 0,
            'hit_count': 0,
            'miss_count': 0,
            'created_at': time.time()
        }
        self._save_metadata()

        if self.verbose:
            print("✅ Cache cleared")

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats:
            - total_size_gb: Current cache size
            - entries: Number of cached files
            - hit_count: Cache hits
            - miss_count: Cache misses
            - hit_rate: Cache hit rate (0-1)
        """
        total_requests = self.metadata['hit_count'] + self.metadata['miss_count']
        hit_rate = (
            self.metadata['hit_count'] / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            'total_size_gb': self.metadata['total_size_bytes'] / (1024**3),
            'total_size_bytes': self.metadata['total_size_bytes'],
            'max_size_gb': self.max_size_bytes / (1024**3),
            'entries': len(self.metadata['entries']),
            'hit_count': self.metadata['hit_count'],
            'miss_count': self.metadata['miss_count'],
            'hit_rate': hit_rate,
            'created_at': self.metadata.get('created_at', 0)
        }

    def print_stats(self):
        """Print cache statistics"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"MIDI Cache Statistics")
        print(f"{'='*60}")
        print(f"  Cache size: {stats['total_size_gb']:.2f} / {stats['max_size_gb']:.2f} GB")
        print(f"  Entries: {stats['entries']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%} ({stats['hit_count']} hits, {stats['miss_count']} misses)")
        print(f"{'='*60}\n")


# ============================================================================
# Utility Functions
# ============================================================================

def extract_midi_metadata(midi_file: Path) -> Dict[str, Any]:
    """
    Extract metadata from MIDI file.

    Args:
        midi_file: Path to MIDI file

    Returns:
        Dictionary with metadata:
        - n_tracks: Number of tracks
        - n_notes: Total note count
        - duration_seconds: Duration in seconds
        - tempo_bpm: Tempo in BPM
        - time_signature: Time signature string (e.g., "4/4")
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido is required. Install with: pip install mido")

    midi = mido.MidiFile(str(midi_file))

    # Extract notes
    notes = []
    current_time = [0.0] * len(midi.tracks)
    tempo = 500000  # Default tempo (120 BPM)
    time_sig_num = 4
    time_sig_denom = 4

    for track_idx, track in enumerate(midi.tracks):
        for msg in track:
            current_time[track_idx] += mido.tick2second(
                msg.time, midi.ticks_per_beat, tempo
            )

            if msg.type == 'set_tempo':
                tempo = msg.tempo
            elif msg.type == 'time_signature':
                time_sig_num = msg.numerator
                time_sig_denom = msg.denominator
            elif msg.type == 'note_on' and msg.velocity > 0:
                notes.append({
                    'pitch': msg.pitch,
                    'velocity': msg.velocity,
                    'start_time': current_time[track_idx],
                    'track': track_idx
                })

    duration = max(current_time) if current_time else 0.0
    tempo_bpm = 60_000_000 / tempo
    time_signature = f"{time_sig_num}/{time_sig_denom}"

    return {
        'n_tracks': len(midi.tracks),
        'n_notes': len(notes),
        'duration_seconds': duration,
        'tempo_bpm': tempo_bpm,
        'time_signature': time_signature,
        'notes': notes
    }


def create_cached_midi(midi_file: Path) -> CachedMIDI:
    """
    Create CachedMIDI object from MIDI file.

    Args:
        midi_file: Path to MIDI file

    Returns:
        CachedMIDI object
    """
    if not MIDO_AVAILABLE:
        raise ImportError("mido is required. Install with: pip install mido")

    # Load MIDI
    midi_obj = mido.MidiFile(str(midi_file))

    # Extract metadata
    metadata = extract_midi_metadata(midi_file)

    # Create CachedMIDI
    return CachedMIDI(
        file_id=midi_file.stem,
        file_path=str(midi_file),
        midi_object=midi_obj,
        n_tracks=metadata['n_tracks'],
        n_notes=metadata['n_notes'],
        duration_seconds=metadata['duration_seconds'],
        tempo_bpm=metadata['tempo_bpm'],
        time_signature=metadata['time_signature'],
        notes=metadata['notes'],
        timestamp=time.time(),
        file_size_bytes=midi_file.stat().st_size
    )
