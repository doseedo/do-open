"""
DNA Cache - Agent 8
===================

Efficient caching system for encoded DNA representations to avoid redundant encoding.

This module provides:
- DNACache: SHA256-keyed caching for DNA vectors (300D)
- Encoded DNA + metadata storage
- Configurable cache size (default 10GB)

Pattern: Follows GapCache architecture (Agent 4)

Author: Agent 8 - Data Pipeline & Preprocessing
"""

import json
import pickle
import hashlib
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class CachedDNA:
    """
    Cached DNA representation with metadata.

    Stores encoded DNA vector (300D) and metadata to avoid re-encoding.
    """
    file_id: str
    file_path: str

    # DNA representation
    dna_vector: np.ndarray  # 300D DNA vector
    dna_dim: int = 300

    # Encoder metadata
    encoder_version: str = "v1.0"
    encoding_params: Dict[str, Any] = None

    # Module breakdown (optional)
    module_vectors: Dict[str, np.ndarray] = None  # Individual module encodings
    # {
    #     'harmony': 50D,
    #     'rhythm': 50D,
    #     'form': 50D,
    #     'orchestration': 50D,
    #     'texture': 50D,
    #     'expression': 50D
    # }

    # Cache metadata
    timestamp: float = 0.0
    computation_time: float = 0.0

    def __post_init__(self):
        """Validate DNA vector"""
        if self.dna_vector is not None:
            assert self.dna_vector.shape == (self.dna_dim,), \
                f"DNA vector must be {self.dna_dim}D, got {self.dna_vector.shape}"
        if self.encoding_params is None:
            self.encoding_params = {}
        if self.module_vectors is None:
            self.module_vectors = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary (without numpy arrays for JSON serialization)"""
        return {
            'file_id': self.file_id,
            'file_path': self.file_path,
            'dna_dim': self.dna_dim,
            'encoder_version': self.encoder_version,
            'encoding_params': self.encoding_params,
            'timestamp': self.timestamp,
            'computation_time': self.computation_time,
            'has_module_vectors': bool(self.module_vectors)
        }

    def to_torch(self) -> 'torch.Tensor':
        """Convert DNA vector to PyTorch tensor"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        return torch.from_numpy(self.dna_vector).float()


# ============================================================================
# DNACache
# ============================================================================

class DNACache:
    """
    Efficient caching system for encoded DNA representations.

    Speeds up training by caching:
    - Encoded DNA vectors (300D)
    - Module-specific encodings
    - Encoding metadata

    Cache size: 10 GB (configurable)
    Cache key: SHA256 hash of (MIDI file + encoder version + encoding params)
    Eviction: LRU (Least Recently Used)

    Pattern: Follows GapCache architecture from Agent 4

    Example:
        cache = DNACache(cache_dir=Path('cache/dna'), max_size_gb=10.0)

        # Create cache key from MIDI file + encoder config
        cache_key = cache.create_cache_key(
            midi_file_path,
            encoder_version='v1.0',
            encoding_params={'normalize': True}
        )

        # Try to get from cache
        cached_dna = cache.get(cache_key)
        if cached_dna is None:
            # Encode MIDI → DNA
            dna_vector = encoder.encode(midi_data)

            # Cache for future use
            cached_dna = CachedDNA(
                file_id=midi_file.stem,
                file_path=str(midi_file),
                dna_vector=dna_vector,
                encoder_version='v1.0'
            )
            cache.put(cache_key, cached_dna)
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        verbose: bool = False
    ):
        """
        Initialize DNA cache.

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
        self.metadata_file = self.cache_dir / 'dna_cache_metadata.json'
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

    def create_cache_key(
        self,
        midi_file: Path,
        encoder_version: str = "v1.0",
        encoding_params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create cache key from MIDI file + encoder configuration.

        This ensures that the same MIDI file encoded with different
        encoder versions or parameters gets different cache entries.

        Args:
            midi_file: Path to MIDI file
            encoder_version: Encoder version string
            encoding_params: Encoder parameters dict

        Returns:
            Cache key (SHA256 hash)
        """
        # Compute file hash
        sha256_hash = hashlib.sha256()

        # Add file content
        with open(midi_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        # Add encoder version
        sha256_hash.update(encoder_version.encode('utf-8'))

        # Add encoding params (sorted for consistency)
        if encoding_params:
            params_str = json.dumps(encoding_params, sort_keys=True)
            sha256_hash.update(params_str.encode('utf-8'))

        return sha256_hash.hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """
        Get cache file path for key.

        Uses first 2 chars as subdirectory to avoid too many files in one dir.

        Args:
            cache_key: Cache key (SHA256 hash)

        Returns:
            Path to cache file
        """
        subdir = cache_key[:2]
        cache_subdir = self.cache_dir / subdir
        cache_subdir.mkdir(exist_ok=True)
        return cache_subdir / f"{cache_key}.pkl"

    def get(self, cache_key: str) -> Optional[CachedDNA]:
        """
        Get cached DNA data for key.

        Args:
            cache_key: Cache key (from create_cache_key)

        Returns:
            CachedDNA if cached, None otherwise
        """
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached_dna = pickle.load(f)

                # Update timestamp for LRU
                if cache_key in self.metadata['entries']:
                    self.metadata['entries'][cache_key]['timestamp'] = time.time()
                    self._save_metadata()

                self.metadata['hit_count'] += 1

                if self.verbose:
                    print(f"✅ DNA cache hit for {cache_key[:8]}...")

                return cached_dna
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ DNA cache read error: {e}")
                return None

        self.metadata['miss_count'] += 1
        return None

    def put(self, cache_key: str, cached_dna: CachedDNA):
        """
        Cache DNA data.

        Args:
            cache_key: Cache key (from create_cache_key)
            cached_dna: CachedDNA object to cache
        """
        cache_path = self._get_cache_path(cache_key)

        # Check if we need to evict old entries
        self._evict_if_needed()

        # Save cached DNA
        try:
            # Update timestamp
            cached_dna.timestamp = time.time()

            with open(cache_path, 'wb') as f:
                pickle.dump(cached_dna, f)

            # Update metadata
            file_size = cache_path.stat().st_size
            self.metadata['entries'][cache_key] = {
                'file_path': cached_dna.file_path,
                'cache_path': str(cache_path),
                'size_bytes': file_size,
                'timestamp': cached_dna.timestamp,
                'encoder_version': cached_dna.encoder_version
            }
            self.metadata['total_size_bytes'] += file_size
            self._save_metadata()

            if self.verbose:
                print(f"✅ Cached DNA for {cache_key[:8]}... ({file_size / 1024:.1f} KB)")

        except Exception as e:
            if self.verbose:
                print(f"⚠️ DNA cache write error: {e}")

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds max size (LRU)"""
        while self.metadata['total_size_bytes'] > self.max_size_bytes:
            # Find oldest entry (LRU)
            oldest_key = None
            oldest_time = float('inf')

            for cache_key, entry in self.metadata['entries'].items():
                if entry['timestamp'] < oldest_time:
                    oldest_time = entry['timestamp']
                    oldest_key = cache_key

            if oldest_key is None:
                break

            # Remove oldest entry
            entry = self.metadata['entries'][oldest_key]
            cache_path = Path(entry['cache_path'])

            if cache_path.exists():
                cache_path.unlink()

            self.metadata['total_size_bytes'] -= entry['size_bytes']
            del self.metadata['entries'][oldest_key]

            if self.verbose:
                print(f"⚠️ Evicted DNA cache entry {oldest_key[:8]}...")

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
            print("✅ DNA cache cleared")

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats:
            - total_size_gb: Current cache size
            - entries: Number of cached DNAs
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
        print(f"DNA Cache Statistics")
        print(f"{'='*60}")
        print(f"  Cache size: {stats['total_size_gb']:.2f} / {stats['max_size_gb']:.2f} GB")
        print(f"  Entries: {stats['entries']}")
        print(f"  Hit rate: {stats['hit_rate']:.1%} ({stats['hit_count']} hits, {stats['miss_count']} misses)")
        print(f"{'='*60}\n")

    def invalidate_encoder_version(self, encoder_version: str):
        """
        Invalidate all cache entries for a specific encoder version.

        Useful when encoder is updated and old cached DNAs are no longer valid.

        Args:
            encoder_version: Encoder version to invalidate
        """
        keys_to_remove = []
        for cache_key, entry in self.metadata['entries'].items():
            if entry.get('encoder_version') == encoder_version:
                keys_to_remove.append(cache_key)

        for cache_key in keys_to_remove:
            entry = self.metadata['entries'][cache_key]
            cache_path = Path(entry['cache_path'])
            if cache_path.exists():
                cache_path.unlink()

            self.metadata['total_size_bytes'] -= entry['size_bytes']
            del self.metadata['entries'][cache_key]

        self._save_metadata()

        if self.verbose:
            print(f"✅ Invalidated {len(keys_to_remove)} entries for encoder {encoder_version}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_dna_from_modules(
    harmony: np.ndarray,
    rhythm: np.ndarray,
    form: np.ndarray,
    orchestration: np.ndarray,
    texture: np.ndarray,
    expression: np.ndarray
) -> np.ndarray:
    """
    Concatenate module vectors into full 300D DNA.

    Args:
        harmony: 50D harmony encoding
        rhythm: 50D rhythm encoding
        form: 50D form encoding
        orchestration: 50D orchestration encoding
        texture: 50D texture encoding
        expression: 50D expression encoding

    Returns:
        300D DNA vector
    """
    return np.concatenate([
        harmony,
        rhythm,
        form,
        orchestration,
        texture,
        expression
    ])


def split_dna_to_modules(dna_vector: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Split 300D DNA into module vectors.

    Args:
        dna_vector: 300D DNA vector

    Returns:
        Dictionary with module vectors:
        - harmony: 50D
        - rhythm: 50D
        - form: 50D
        - orchestration: 50D
        - texture: 50D
        - expression: 50D
    """
    assert dna_vector.shape == (300,), f"Expected 300D DNA, got {dna_vector.shape}"

    return {
        'harmony': dna_vector[0:50],
        'rhythm': dna_vector[50:100],
        'form': dna_vector[100:150],
        'orchestration': dna_vector[150:200],
        'texture': dna_vector[200:250],
        'expression': dna_vector[250:300]
    }
