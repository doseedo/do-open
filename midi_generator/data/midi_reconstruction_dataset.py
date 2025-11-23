"""
MIDI Reconstruction Dataset - Agent 8
======================================

PyTorch Dataset for end-to-end MIDI → DNA → MIDI training.

This module provides:
- MIDIReconstructionDataset: Core dataset for reconstruction training
- Pianoroll conversion utilities
- Integration with caching and augmentation

Pattern: Follows existing dataset patterns (GapDataset, HierarchicalMIDIDataset)

Author: Agent 8 - Data Pipeline & Preprocessing
"""

import warnings
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np
import time

from .midi_cache import MIDICache, create_cached_midi
from .dna_cache import DNACache

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available. Install with: pip install mido")

# Optional imports
try:
    from midi_generator.multi_genre.augmentation import GenreAugmentationPipeline
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False
    GenreAugmentationPipeline = None


# ============================================================================
# MIDI to Pianoroll Conversion
# ============================================================================

def midi_to_pianoroll(
    notes: List[Dict[str, Any]],
    duration_seconds: float,
    time_resolution: float = 0.0625,  # 16th note at 120 BPM
    max_tracks: int = 20,
    quantize: bool = True
) -> np.ndarray:
    """
    Convert MIDI note data to pianoroll representation.

    Args:
        notes: List of notes with {pitch, velocity, start_time, track}
        duration_seconds: Total duration in seconds
        time_resolution: Time resolution in seconds (default: 16th note at 120 BPM)
        max_tracks: Maximum number of tracks
        quantize: Whether to quantize note timings

    Returns:
        Pianoroll tensor: (time_steps, 128_pitches, tracks)
        - Values are velocities (0-127)
    """
    # Calculate number of time steps
    n_timesteps = int(np.ceil(duration_seconds / time_resolution))

    # Initialize pianoroll (time, pitch, track)
    pianoroll = np.zeros((n_timesteps, 128, max_tracks), dtype=np.float32)

    # Process each note
    for note in notes:
        pitch = note['pitch']
        velocity = note['velocity']
        start_time = note['start_time']
        track = note.get('track', 0)

        # Clamp track to max_tracks
        if track >= max_tracks:
            track = max_tracks - 1

        # Convert time to timestep
        if quantize:
            start_step = int(np.round(start_time / time_resolution))
        else:
            start_step = int(start_time / time_resolution)

        # Handle duration if available
        if 'duration' in note:
            duration = note['duration']
            if quantize:
                end_step = int(np.round((start_time + duration) / time_resolution))
            else:
                end_step = int((start_time + duration) / time_resolution)
        else:
            # Default: single timestep
            end_step = start_step + 1

        # Clamp to valid range
        start_step = max(0, min(start_step, n_timesteps - 1))
        end_step = max(start_step + 1, min(end_step, n_timesteps))

        # Fill pianoroll
        pianoroll[start_step:end_step, pitch, track] = velocity

    return pianoroll


def pianoroll_to_midi_data(
    pianoroll: np.ndarray,
    time_resolution: float = 0.0625,
    velocity_threshold: float = 10.0
) -> List[Dict[str, Any]]:
    """
    Convert pianoroll back to MIDI note data.

    Args:
        pianoroll: (time_steps, 128_pitches, tracks) array
        time_resolution: Time resolution in seconds
        velocity_threshold: Minimum velocity to consider as note

    Returns:
        List of notes: [{pitch, velocity, start_time, duration, track}, ...]
    """
    n_timesteps, n_pitches, n_tracks = pianoroll.shape
    notes = []

    for track_idx in range(n_tracks):
        for pitch in range(n_pitches):
            # Find note onsets (where velocity > threshold)
            velocity_track = pianoroll[:, pitch, track_idx]

            i = 0
            while i < n_timesteps:
                if velocity_track[i] > velocity_threshold:
                    # Note onset
                    start_step = i
                    velocity = velocity_track[i]

                    # Find note offset (where velocity drops below threshold)
                    end_step = start_step + 1
                    while end_step < n_timesteps and velocity_track[end_step] > velocity_threshold:
                        end_step += 1

                    # Convert to time
                    start_time = start_step * time_resolution
                    end_time = end_step * time_resolution
                    duration = end_time - start_time

                    notes.append({
                        'pitch': pitch,
                        'velocity': int(velocity),
                        'start_time': start_time,
                        'duration': duration,
                        'track': track_idx
                    })

                    # Skip to end of note
                    i = end_step
                else:
                    i += 1

    # Sort by start time
    notes.sort(key=lambda n: n['start_time'])

    return notes


# ============================================================================
# Dataset
# ============================================================================

class MIDIReconstructionDataset(Dataset):
    """
    PyTorch Dataset for end-to-end MIDI → DNA → MIDI reconstruction training.

    Data flow:
    1. Load MIDI file (with MIDICache)
    2. Apply augmentation (optional, if mode='train')
    3. Convert to pianoroll representation
    4. Return: {pianoroll, features, file_id, length, ...}

    The encoder/decoder models will use this data:
    - Encoder: pianoroll → DNA (300D)
    - Decoder: DNA → pianoroll

    Example:
        dataset = MIDIReconstructionDataset(
            midi_files=list(Path('data/midi').glob('*.mid')),
            midi_cache=MIDICache(Path('cache/midi')),
            augmentation_pipeline=GenreAugmentationPipeline('jazz'),
            mode='train'
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            collate_fn=variable_length_collate_fn
        )

        for batch in dataloader:
            # batch['pianorolls']: (batch, time, 128, tracks)
            # batch['features']: (batch, 200)
            # batch['attention_mask']: (batch, time)
            ...
    """

    def __init__(
        self,
        midi_files: List[Path],
        midi_cache: Optional[MIDICache] = None,
        dna_cache: Optional[DNACache] = None,
        augmentation_pipeline: Optional[Any] = None,
        max_length_seconds: float = 300.0,
        max_tracks: int = 20,
        time_resolution: float = 0.0625,  # 16th note at 120 BPM
        quantize: bool = True,
        mode: str = 'train',
        verbose: bool = False
    ):
        """
        Initialize MIDI reconstruction dataset.

        Args:
            midi_files: List of MIDI file paths
            midi_cache: MIDICache for caching parsed MIDI (optional)
            dna_cache: DNACache for caching encoded DNA (optional)
            augmentation_pipeline: GenreAugmentationPipeline for data augmentation
            max_length_seconds: Maximum MIDI length (longer files are truncated)
            max_tracks: Maximum number of tracks
            time_resolution: Time resolution in seconds (default: 16th note at 120 BPM)
            quantize: Whether to quantize note timings
            mode: 'train', 'val', or 'test' (augmentation only applied in 'train')
            verbose: Print progress
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install with: pip install torch")

        if not MIDO_AVAILABLE:
            raise ImportError("mido required. Install with: pip install mido")

        self.midi_files = [Path(f) for f in midi_files]
        self.midi_cache = midi_cache
        self.dna_cache = dna_cache
        self.augmentation_pipeline = augmentation_pipeline
        self.max_length_seconds = max_length_seconds
        self.max_tracks = max_tracks
        self.time_resolution = time_resolution
        self.quantize = quantize
        self.mode = mode
        self.verbose = verbose

        # Validate augmentation availability
        if augmentation_pipeline is not None and not AUGMENTATION_AVAILABLE:
            warnings.warn(
                "Augmentation pipeline provided but augmentation module not available. "
                "Augmentation will be skipped."
            )
            self.augmentation_pipeline = None

        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'augmentations_applied': 0,
            'total_loads': 0
        }

        if self.verbose:
            print(f"MIDIReconstructionDataset initialized:")
            print(f"  Files: {len(self.midi_files)}")
            print(f"  Mode: {self.mode}")
            print(f"  Augmentation: {self.augmentation_pipeline is not None}")
            print(f"  MIDI cache: {self.midi_cache is not None}")
            print(f"  DNA cache: {self.dna_cache is not None}")

    def __len__(self) -> int:
        """Dataset size"""
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get dataset item.

        Args:
            idx: Index

        Returns:
            Dictionary with:
            - pianoroll: (time, 128, tracks) tensor
            - length: int (number of time steps)
            - file_id: str
            - tempo_bpm: float
            - time_signature: str
            - n_tracks: int
            - n_notes: int
        """
        self.stats['total_loads'] += 1

        midi_file = self.midi_files[idx]

        # Try to load from cache
        cached_midi = None
        if self.midi_cache is not None:
            cached_midi = self.midi_cache.get(midi_file)
            if cached_midi is not None:
                self.stats['cache_hits'] += 1
            else:
                self.stats['cache_misses'] += 1

        # Load and cache if not cached
        if cached_midi is None:
            cached_midi = create_cached_midi(midi_file)
            if self.midi_cache is not None:
                self.midi_cache.put(midi_file, cached_midi)

        # Extract MIDI data
        notes = cached_midi.notes
        duration = min(cached_midi.duration_seconds, self.max_length_seconds)

        # Convert to dictionary format for augmentation
        midi_data = {
            'notes': notes,
            'tempo_bpm': cached_midi.tempo_bpm,
            'time_signature': cached_midi.time_signature,
            'duration_seconds': duration,
            'file_id': cached_midi.file_id
        }

        # Apply augmentation (only in train mode)
        if self.mode == 'train' and self.augmentation_pipeline is not None:
            variations = self.augmentation_pipeline.augment(
                midi_data,
                num_variations=1
            )
            midi_data = variations[0]
            self.stats['augmentations_applied'] += 1

        # Convert to pianoroll
        pianoroll = midi_to_pianoroll(
            notes=midi_data['notes'],
            duration_seconds=duration,
            time_resolution=self.time_resolution,
            max_tracks=self.max_tracks,
            quantize=self.quantize
        )

        # Convert to PyTorch tensor
        pianoroll_tensor = torch.from_numpy(pianoroll).float()

        # Prepare output
        item = {
            'pianoroll': pianoroll_tensor,
            'length': pianoroll_tensor.shape[0],
            'file_id': midi_data['file_id'],
            'tempo_bpm': midi_data['tempo_bpm'],
            'time_signature': midi_data['time_signature'],
            'n_tracks': cached_midi.n_tracks,
            'n_notes': len(midi_data['notes']),
            'augmented': self.mode == 'train' and self.augmentation_pipeline is not None
        }

        return item

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = (
            self.stats['cache_hits'] / total_requests
            if total_requests > 0 else 0.0
        )

        return {
            **self.stats,
            'cache_hit_rate': cache_hit_rate,
            'n_files': len(self.midi_files)
        }

    def print_stats(self):
        """Print dataset statistics"""
        stats = self.get_stats()
        print(f"\n{'='*60}")
        print(f"MIDIReconstructionDataset Statistics")
        print(f"{'='*60}")
        print(f"  Files: {stats['n_files']}")
        print(f"  Total loads: {stats['total_loads']}")
        print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
        print(f"  Augmentations applied: {stats['augmentations_applied']}")
        print(f"{'='*60}\n")


# ============================================================================
# Dataset Factory
# ============================================================================

def create_midi_datasets(
    data_dir: Path,
    cache_dir: Optional[Path] = None,
    genre: Optional[str] = None,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    max_length_seconds: float = 300.0,
    max_tracks: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> Tuple[MIDIReconstructionDataset, MIDIReconstructionDataset, MIDIReconstructionDataset]:
    """
    Create train/val/test datasets from MIDI directory.

    Args:
        data_dir: Directory containing MIDI files
        cache_dir: Cache directory (optional)
        genre: Genre for augmentation (optional)
        train_split: Fraction for training (default 0.7)
        val_split: Fraction for validation (default 0.15)
        test_split: Fraction for testing (default 0.15)
        max_length_seconds: Maximum MIDI length
        max_tracks: Maximum number of tracks
        seed: Random seed for splitting
        verbose: Print progress

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Install with: pip install torch")

    # Find all MIDI files
    midi_files = sorted(list(Path(data_dir).glob('**/*.mid')) +
                       list(Path(data_dir).glob('**/*.midi')))

    if len(midi_files) == 0:
        raise ValueError(f"No MIDI files found in {data_dir}")

    if verbose:
        print(f"Found {len(midi_files)} MIDI files in {data_dir}")

    # Split dataset
    np.random.seed(seed)
    indices = np.random.permutation(len(midi_files))

    n_train = int(len(midi_files) * train_split)
    n_val = int(len(midi_files) * val_split)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_files = [midi_files[i] for i in train_indices]
    val_files = [midi_files[i] for i in val_indices]
    test_files = [midi_files[i] for i in test_indices]

    if verbose:
        print(f"Split: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

    # Create caches
    midi_cache = None
    if cache_dir is not None:
        midi_cache = MIDICache(
            cache_dir=cache_dir / 'midi',
            max_size_gb=5.0,
            verbose=verbose
        )

    # Create augmentation pipeline
    augmentation_pipeline = None
    if genre is not None and AUGMENTATION_AVAILABLE:
        augmentation_pipeline = GenreAugmentationPipeline(genre)
        if verbose:
            print(f"Using {genre} augmentation pipeline")

    # Create datasets
    train_dataset = MIDIReconstructionDataset(
        midi_files=train_files,
        midi_cache=midi_cache,
        augmentation_pipeline=augmentation_pipeline,
        max_length_seconds=max_length_seconds,
        max_tracks=max_tracks,
        mode='train',
        verbose=verbose
    )

    val_dataset = MIDIReconstructionDataset(
        midi_files=val_files,
        midi_cache=midi_cache,
        augmentation_pipeline=None,  # No augmentation in validation
        max_length_seconds=max_length_seconds,
        max_tracks=max_tracks,
        mode='val',
        verbose=verbose
    )

    test_dataset = MIDIReconstructionDataset(
        midi_files=test_files,
        midi_cache=midi_cache,
        augmentation_pipeline=None,  # No augmentation in testing
        max_length_seconds=max_length_seconds,
        max_tracks=max_tracks,
        mode='test',
        verbose=verbose
    )

    return train_dataset, val_dataset, test_dataset
