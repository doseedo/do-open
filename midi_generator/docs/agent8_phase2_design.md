# Agent 8: Data Pipeline Design - Phase 2

**Author:** Agent 8 - Data Pipeline & Preprocessing Optimization
**Date:** 2025-11-22
**Status:** Phase 2 - Design

## Design Overview

Building on the existing infrastructure analyzed in Phase 1, this document designs **four new components** for end-to-end MIDI → DNA → MIDI training:

1. **MIDIReconstructionDataset** - PyTorch Dataset for end-to-end training
2. **DNACache** - Caching system for DNA representations
3. **Variable-Length Collate Functions** - Efficient batching
4. **Reconstruction Metrics** - MIDI similarity measurement

## 1. MIDIReconstructionDataset Design

### 1.1 Purpose
Enable end-to-end training: `MIDI (input) → Encoder → DNA (300D) → Decoder → MIDI (output)`

### 1.2 Architecture

```python
class MIDIReconstructionDataset(Dataset):
    """
    PyTorch Dataset for end-to-end MIDI reconstruction training.

    Workflow:
        1. Load MIDI file from disk (with caching)
        2. Apply optional augmentation (reuse GenreAugmentationPipeline)
        3. Convert to tensor representation
        4. Return: (midi_tensor, metadata)

    Integrates with:
        - Agent 1 (Decoder): Provides MIDI input/target
        - Agent 2 (Differentiable MIDI): Uses their tensor format
        - Agent 3 (DNA): Supports 300D DNA
        - Agent 5 (Training): Feeds into training loop
    """

    def __init__(
        self,
        midi_files: List[Path],
        feature_extractor: Optional[Any] = None,
        augmentation_pipeline: Optional[GenreAugmentationPipeline] = None,
        midi_cache: Optional['MIDICache'] = None,
        dna_cache: Optional['DNACache'] = None,
        max_length_seconds: float = 300.0,  # 5 minutes
        max_tracks: int = 20,
        quantize_resolution: int = 16,  # 16th notes
        normalize_features: bool = True,
        augment_on_the_fly: bool = True,
        mode: str = 'train'  # 'train', 'val', 'test'
    ):
        """
        Initialize dataset.

        Args:
            midi_files: List of MIDI file paths
            feature_extractor: Optional feature extractor (for loss computation)
            augmentation_pipeline: Optional augmentation (applied in 'train' mode)
            midi_cache: Cache for loaded MIDI files
            dna_cache: Cache for encoded DNA representations
            max_length_seconds: Maximum MIDI duration (longer files truncated)
            max_tracks: Maximum number of tracks
            quantize_resolution: Quantization grid (16 = 16th notes)
            normalize_features: Whether to normalize extracted features
            augment_on_the_fly: Apply augmentation during __getitem__ (True)
                               or precompute augmented versions (False)
            mode: 'train', 'val', 'test' (augmentation only in 'train')
        """
        self.midi_files = midi_files
        self.feature_extractor = feature_extractor
        self.augmentation = augmentation_pipeline if mode == 'train' else None
        self.midi_cache = midi_cache
        self.dna_cache = dna_cache
        self.max_length = max_length_seconds
        self.max_tracks = max_tracks
        self.quantize_res = quantize_resolution
        self.normalize = normalize_features
        self.augment_on_the_fly = augment_on_the_fly
        self.mode = mode

        # Feature normalizer (fitted on training set)
        self.normalizer = None

        # Validation
        self._validate_files()

    def __len__(self) -> int:
        return len(self.midi_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process one MIDI file.

        Returns:
            {
                'midi_tensor': torch.Tensor,  # Shape: (time, pitch, tracks)
                'features': torch.Tensor,      # Optional: extracted features
                'metadata': {
                    'file_path': str,
                    'duration': float,
                    'num_tracks': int,
                    'num_notes': int,
                    'genre': Optional[str],
                    'augmented': bool
                }
            }
        """
        midi_path = self.midi_files[idx]

        # 1. Load MIDI (with caching)
        midi_data = self._load_midi(midi_path)

        # 2. Apply augmentation (only in train mode)
        if self.augmentation is not None and self.augment_on_the_fly:
            midi_data = self.augmentation(midi_data)
            augmented = True
        else:
            augmented = False

        # 3. Convert to tensor representation
        midi_tensor = self._midi_to_tensor(midi_data)

        # 4. Extract features (optional, for loss computation)
        features = None
        if self.feature_extractor is not None:
            features = self.feature_extractor.extract(midi_data)
            if self.normalize and self.normalizer is not None:
                features = self.normalizer.transform(features)
            features = torch.FloatTensor(features)

        # 5. Create metadata
        metadata = {
            'file_path': str(midi_path),
            'duration': self._get_duration(midi_data),
            'num_tracks': len(midi_data.get('tracks', [])),
            'num_notes': self._count_notes(midi_data),
            'genre': midi_data.get('genre'),
            'augmented': augmented
        }

        return {
            'midi_tensor': midi_tensor,
            'features': features,
            'metadata': metadata
        }

    def _load_midi(self, path: Path) -> Dict[str, Any]:
        """
        Load MIDI file with caching.

        Returns MIDI data dict with keys:
            - 'notes': List[Dict] with 'pitch', 'velocity', 'start', 'end', 'track'
            - 'tempo_bpm': float
            - 'time_signature': Tuple[int, int]
            - 'key': Optional[str]
            - 'tracks': List of track info
        """
        # Check cache first
        if self.midi_cache is not None:
            cached = self.midi_cache.get(path)
            if cached is not None:
                return cached

        # Load from disk
        import mido
        midi = mido.MidiFile(str(path))

        # Parse to dict format (compatible with augmentation)
        midi_data = self._parse_mido_to_dict(midi)

        # Validate and enforce constraints
        midi_data = self._enforce_constraints(midi_data)

        # Cache for future use
        if self.midi_cache is not None:
            self.midi_cache.put(path, midi_data)

        return midi_data

    def _parse_mido_to_dict(self, midi: mido.MidiFile) -> Dict[str, Any]:
        """
        Parse mido.MidiFile to dict format.

        Reuses existing MIDI parsing patterns from the codebase.
        """
        notes = []
        tempo_bpm = 120.0  # Default
        time_sig = (4, 4)

        for track_idx, track in enumerate(midi.tracks):
            time = 0
            active_notes = {}  # pitch -> start_time

            for msg in track:
                time += msg.time

                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = time

                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        start = active_notes.pop(msg.note)
                        notes.append({
                            'pitch': msg.note,
                            'velocity': msg.velocity if msg.type != 'note_off' else 64,
                            'start': start,
                            'end': time,
                            'track': track_idx
                        })

                elif msg.type == 'set_tempo':
                    tempo_bpm = mido.tempo2bpm(msg.tempo)

                elif msg.type == 'time_signature':
                    time_sig = (msg.numerator, msg.denominator)

        return {
            'notes': notes,
            'tempo_bpm': tempo_bpm,
            'time_signature': time_sig,
            'key': None,  # TODO: Detect key
            'tracks': list(range(len(midi.tracks)))
        }

    def _enforce_constraints(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce constraints:
            - Max duration: 5 minutes
            - Max tracks: 20
            - Quantize to 16th notes
        """
        # Truncate to max length
        max_time = self.max_length
        midi_data['notes'] = [
            n for n in midi_data['notes']
            if n['start'] < max_time
        ]

        # Truncate end times
        for note in midi_data['notes']:
            note['end'] = min(note['end'], max_time)

        # Limit tracks
        if len(midi_data['tracks']) > self.max_tracks:
            # Keep first N tracks with most notes
            track_counts = {}
            for note in midi_data['notes']:
                track = note['track']
                track_counts[track] = track_counts.get(track, 0) + 1

            top_tracks = sorted(track_counts.items(), key=lambda x: -x[1])[:self.max_tracks]
            keep_tracks = {t for t, _ in top_tracks}

            midi_data['notes'] = [n for n in midi_data['notes'] if n['track'] in keep_tracks]
            midi_data['tracks'] = list(keep_tracks)

        # Quantize (optional - could be done in tensor conversion)
        # TODO: Implement quantization if needed

        return midi_data

    def _midi_to_tensor(self, midi_data: Dict[str, Any]) -> torch.Tensor:
        """
        Convert MIDI dict to tensor representation.

        This will coordinate with Agent 2 (Differentiable MIDI) for the
        exact tensor format. Options:

        Option 1: Pianoroll representation
            Shape: (time_steps, 128_pitches, num_tracks)
            Values: 0-1 (velocity normalized)

        Option 2: Event-based representation
            List of (onset, pitch, duration, velocity, track) tuples

        Option 3: Sparse representation
            COO format for memory efficiency

        For now, use pianoroll (most compatible with existing code).
        """
        # Determine time grid
        time_steps = int(self.max_length * 4)  # 4 steps per second (16th at 120 BPM)
        num_pitches = 128
        num_tracks = len(midi_data['tracks'])

        # Initialize pianoroll
        pianoroll = np.zeros((time_steps, num_pitches, num_tracks), dtype=np.float32)

        # Fill pianoroll
        for note in midi_data['notes']:
            start_step = int(note['start'] * 4)
            end_step = int(note['end'] * 4)
            pitch = note['pitch']
            track = note['track']
            velocity_norm = note['velocity'] / 127.0

            # Clip to valid range
            start_step = max(0, min(start_step, time_steps - 1))
            end_step = max(0, min(end_step, time_steps))
            pitch = max(0, min(pitch, 127))

            if track < num_tracks:
                pianoroll[start_step:end_step, pitch, track] = velocity_norm

        return torch.FloatTensor(pianoroll)

    def _get_duration(self, midi_data: Dict[str, Any]) -> float:
        """Get MIDI duration in seconds."""
        if not midi_data['notes']:
            return 0.0
        return max(n['end'] for n in midi_data['notes'])

    def _count_notes(self, midi_data: Dict[str, Any]) -> int:
        """Count total notes."""
        return len(midi_data['notes'])

    def _validate_files(self):
        """Validate MIDI files exist and are readable."""
        invalid = []
        for path in self.midi_files:
            if not path.exists():
                invalid.append(path)

        if invalid:
            raise FileNotFoundError(f"Missing MIDI files: {invalid[:5]}")

    def fit_normalizer(self):
        """
        Fit feature normalizer on this dataset (training set only).

        Call this on the training dataset before creating val/test datasets.
        """
        if self.feature_extractor is None:
            return

        print("Fitting feature normalizer...")
        all_features = []

        for i in range(len(self)):
            midi_path = self.midi_files[i]
            midi_data = self._load_midi(midi_path)
            features = self.feature_extractor.extract(midi_data)
            all_features.append(features)

        all_features = np.vstack(all_features)

        # Fit normalizer
        from sklearn.preprocessing import StandardScaler
        self.normalizer = StandardScaler()
        self.normalizer.fit(all_features)

        print(f"Normalizer fitted on {len(all_features)} samples")
```

### 1.3 Usage Example

```python
# Create datasets
train_files = list(Path('data/midi/train').glob('*.mid'))
val_files = list(Path('data/midi/val').glob('*.mid'))

# Initialize components
from midi_generator.multi_genre.augmentation import GenreAugmentationPipeline
augmentation = GenreAugmentationPipeline('jazz')

# Create training dataset
train_dataset = MIDIReconstructionDataset(
    midi_files=train_files,
    augmentation_pipeline=augmentation,
    midi_cache=MIDICache(cache_dir=Path('cache/midi'), max_size_gb=5.0),
    mode='train'
)

# Fit normalizer on training data
train_dataset.fit_normalizer()

# Create validation dataset (no augmentation, same normalizer)
val_dataset = MIDIReconstructionDataset(
    midi_files=val_files,
    mode='val'
)
val_dataset.normalizer = train_dataset.normalizer  # Share normalizer

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=variable_length_collate_fn,  # Custom collate
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    collate_fn=variable_length_collate_fn,
    num_workers=4,
    pin_memory=True
)
```

### 1.4 Integration Points

- **Agent 1 (Decoder):**
  - Input: `midi_tensor` from dataset
  - Output: Reconstructed `midi_tensor`

- **Agent 2 (Differentiable MIDI):**
  - Coordinate on `_midi_to_tensor()` format
  - Use their soft representations during training

- **Agent 3 (DNA):**
  - Encoder: `midi_tensor → DNA (300D)`
  - Decoder: `DNA (300D) → midi_tensor`

- **Agent 5 (Training):**
  - Receives batches from DataLoader
  - Computes reconstruction loss

## 2. DNACache Design

### 2.1 Purpose
Cache encoded DNA representations to avoid re-encoding during training.

### 2.2 Architecture

```python
class DNACache:
    """
    Cache for encoded DNA representations.

    Extends the proven GapCache architecture with DNA-specific features.

    Features:
        - SHA256-based cache keys (file path + augmentation params)
        - LRU eviction when cache exceeds max size
        - Persistence to disk (pickle format)
        - Thread-safe operations
        - Automatic invalidation on encoder changes
    """

    def __init__(
        self,
        cache_dir: Path,
        max_size_gb: float = 10.0,
        encoder_version: str = "v1.0",
        dna_dim: int = 300
    ):
        """
        Initialize DNA cache.

        Args:
            cache_dir: Directory for cache storage
            max_size_gb: Maximum cache size in GB
            encoder_version: Encoder version (for invalidation)
            dna_dim: DNA dimensionality (300D)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.encoder_version = encoder_version
        self.dna_dim = dna_dim

        # Metadata
        self.metadata_file = self.cache_dir / 'metadata.json'
        self.metadata = self._load_metadata()

        # Statistics
        self.hits = 0
        self.misses = 0

        # Thread safety
        import threading
        self.lock = threading.Lock()

    def get(
        self,
        midi_file: Path,
        augmentation_params: Optional[Dict] = None
    ) -> Optional[np.ndarray]:
        """
        Get cached DNA representation.

        Args:
            midi_file: MIDI file path
            augmentation_params: Augmentation parameters (for cache key)

        Returns:
            DNA array (300D) or None if not cached
        """
        cache_key = self._compute_key(midi_file, augmentation_params)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        with self.lock:
            if cache_path.exists():
                # Check version
                if self._is_valid(cache_key):
                    dna = np.load(str(cache_path))
                    self.hits += 1
                    self._update_access_time(cache_key)
                    return dna
                else:
                    # Stale cache, remove
                    cache_path.unlink()
                    del self.metadata['entries'][cache_key]

        self.misses += 1
        return None

    def put(
        self,
        midi_file: Path,
        dna: np.ndarray,
        augmentation_params: Optional[Dict] = None
    ):
        """
        Cache DNA representation.

        Args:
            midi_file: MIDI file path
            dna: DNA array (300D)
            augmentation_params: Augmentation parameters
        """
        cache_key = self._compute_key(midi_file, augmentation_params)
        cache_path = self.cache_dir / f"{cache_key}.npy"

        with self.lock:
            # Save DNA
            np.save(str(cache_path), dna)

            # Update metadata
            self.metadata['entries'][cache_key] = {
                'midi_file': str(midi_file),
                'augmentation': augmentation_params,
                'encoder_version': self.encoder_version,
                'dna_dim': self.dna_dim,
                'size_bytes': cache_path.stat().st_size,
                'access_time': time.time(),
                'create_time': time.time()
            }

            # Evict if needed
            self._evict_if_needed()

            # Save metadata
            self._save_metadata()

    def _compute_key(
        self,
        midi_file: Path,
        augmentation_params: Optional[Dict]
    ) -> str:
        """
        Compute SHA256 cache key.

        Key includes:
            - MIDI file path
            - File modification time
            - Augmentation parameters
            - Encoder version
        """
        import hashlib

        key_data = {
            'file_path': str(midi_file),
            'file_mtime': midi_file.stat().st_mtime if midi_file.exists() else 0,
            'augmentation': augmentation_params or {},
            'encoder_version': self.encoder_version,
            'dna_dim': self.dna_dim
        }

        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()

    def _is_valid(self, cache_key: str) -> bool:
        """Check if cached entry is valid (not stale)."""
        if cache_key not in self.metadata['entries']:
            return False

        entry = self.metadata['entries'][cache_key]

        # Check encoder version
        if entry['encoder_version'] != self.encoder_version:
            return False

        # Check DNA dimension
        if entry['dna_dim'] != self.dna_dim:
            return False

        return True

    def _evict_if_needed(self):
        """Evict oldest entries if cache exceeds max size."""
        total_size = sum(
            entry['size_bytes']
            for entry in self.metadata['entries'].values()
        )

        if total_size > self.max_size_bytes:
            # Sort by access time (LRU)
            entries_by_time = sorted(
                self.metadata['entries'].items(),
                key=lambda x: x[1]['access_time']
            )

            # Evict until under limit
            for cache_key, entry in entries_by_time:
                if total_size <= self.max_size_bytes * 0.9:  # 90% target
                    break

                # Remove file
                cache_path = self.cache_dir / f"{cache_key}.npy"
                if cache_path.exists():
                    cache_path.unlink()
                    total_size -= entry['size_bytes']

                # Remove from metadata
                del self.metadata['entries'][cache_key]

    def _update_access_time(self, cache_key: str):
        """Update access time for LRU."""
        self.metadata['entries'][cache_key]['access_time'] = time.time()
        self._save_metadata()

    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {
            'entries': {},
            'created': time.time(),
            'encoder_version': self.encoder_version,
            'dna_dim': self.dna_dim
        }

    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_size = sum(
            entry['size_bytes']
            for entry in self.metadata['entries'].values()
        )

        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0

        return {
            'hit_rate': hit_rate,
            'hits': self.hits,
            'misses': self.misses,
            'total_entries': len(self.metadata['entries']),
            'total_size_gb': total_size / (1024**3),
            'max_size_gb': self.max_size_bytes / (1024**3)
        }

    def clear(self):
        """Clear entire cache."""
        with self.lock:
            for cache_file in self.cache_dir.glob('*.npy'):
                cache_file.unlink()

            self.metadata = {
                'entries': {},
                'created': time.time(),
                'encoder_version': self.encoder_version,
                'dna_dim': self.dna_dim
            }
            self._save_metadata()

            self.hits = 0
            self.misses = 0
```

### 2.3 Usage Example

```python
# Initialize cache
dna_cache = DNACache(
    cache_dir=Path('cache/dna'),
    max_size_gb=10.0,
    encoder_version="v1.0",
    dna_dim=300
)

# Use in training loop
for midi_file in midi_files:
    # Try to get cached DNA
    dna = dna_cache.get(midi_file, augmentation_params={'transpose': 2})

    if dna is None:
        # Cache miss - encode and cache
        midi_tensor = load_midi(midi_file)
        dna = encoder(midi_tensor)  # Encoder from Agent 1
        dna_cache.put(midi_file, dna, augmentation_params={'transpose': 2})

    # Use DNA for training
    reconstructed = decoder(dna)
    loss = criterion(reconstructed, target)

# Print statistics
print(dna_cache.get_stats())
# {'hit_rate': 0.73, 'hits': 730, 'misses': 270, ...}
```

## 3. Variable-Length Collate Functions

### 3.1 Purpose
Handle variable-length MIDI sequences in batches efficiently.

### 3.2 Design

```python
def variable_length_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length MIDI sequences.

    Handles:
        - Variable time lengths (pad to max in batch)
        - Variable number of tracks (pad to max)
        - Metadata aggregation

    Args:
        batch: List of samples from MIDIReconstructionDataset

    Returns:
        Collated batch with padded tensors and masks
    """
    # Extract components
    midi_tensors = [sample['midi_tensor'] for sample in batch]
    features = [sample['features'] for sample in batch] if batch[0]['features'] is not None else None
    metadata = [sample['metadata'] for sample in batch]

    # Find max dimensions in this batch
    max_time = max(tensor.shape[0] for tensor in midi_tensors)
    max_tracks = max(tensor.shape[2] for tensor in midi_tensors)
    num_pitches = 128  # Fixed

    # Pad MIDI tensors
    padded_midi = []
    masks = []

    for tensor in midi_tensors:
        T, P, K = tensor.shape  # (time, pitch, tracks)

        # Create padded tensor
        padded = torch.zeros(max_time, num_pitches, max_tracks)
        padded[:T, :P, :K] = tensor

        # Create mask (True = valid data, False = padding)
        mask = torch.zeros(max_time, num_pitches, max_tracks, dtype=torch.bool)
        mask[:T, :P, :K] = True

        padded_midi.append(padded)
        masks.append(mask)

    # Stack into batch
    batch_midi = torch.stack(padded_midi, dim=0)  # (B, T, P, K)
    batch_masks = torch.stack(masks, dim=0)       # (B, T, P, K)

    # Stack features if present
    batch_features = None
    if features is not None:
        batch_features = torch.stack(features, dim=0)  # (B, F)

    return {
        'midi_tensor': batch_midi,
        'mask': batch_masks,
        'features': batch_features,
        'metadata': metadata,
        'batch_size': len(batch),
        'max_time': max_time,
        'max_tracks': max_tracks
    }


def packed_sequence_collate_fn(
    batch: List[Dict[str, torch.Tensor]]
) -> Dict[str, Any]:
    """
    Alternative collate using PackedSequence for memory efficiency.

    More memory-efficient than padding for highly variable lengths.
    """
    from torch.nn.utils.rnn import pack_sequence

    # Sort batch by sequence length (required for pack_sequence)
    batch = sorted(batch, key=lambda x: x['midi_tensor'].shape[0], reverse=True)

    midi_tensors = [sample['midi_tensor'] for sample in batch]

    # Pack sequences
    # Note: pack_sequence requires sequences as list
    packed = pack_sequence(midi_tensors, enforce_sorted=True)

    return {
        'packed_midi': packed,
        'metadata': [sample['metadata'] for sample in batch]
    }
```

### 3.3 Usage Example

```python
# Create DataLoader with custom collate
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=variable_length_collate_fn,
    num_workers=4,
    pin_memory=True
)

# In training loop
for batch in train_loader:
    midi = batch['midi_tensor']      # (B, T, 128, K)
    mask = batch['mask']             # (B, T, 128, K)

    # Encode (Agent 1)
    dna = encoder(midi, mask=mask)   # (B, 300)

    # Decode (Agent 1)
    reconstructed = decoder(dna, target_length=midi.shape[1])  # (B, T, 128, K)

    # Compute loss (only on valid data, not padding)
    loss = criterion(reconstructed, midi, mask=mask)
```

## 4. Reconstruction Metrics

### 4.1 Purpose
Measure MIDI reconstruction quality (Agent 5 and Agent 9 will use these).

### 4.2 Design

```python
class MIDIReconstructionMetrics:
    """
    Metrics for evaluating MIDI reconstruction quality.

    Metrics:
        1. Pitch Accuracy - Percentage of correct pitches
        2. Timing Similarity - Correlation of note onsets
        3. Velocity Similarity - MAE of velocities
        4. Harmonic Similarity - Chord progression similarity
        5. Overall Reconstruction Score - Weighted average
    """

    def __init__(
        self,
        pitch_threshold: float = 0.5,
        timing_tolerance_ms: float = 50.0,
        velocity_tolerance: int = 10
    ):
        """
        Initialize metrics.

        Args:
            pitch_threshold: Threshold for considering pitch present
            timing_tolerance_ms: Tolerance for timing errors (ms)
            velocity_tolerance: Tolerance for velocity errors
        """
        self.pitch_threshold = pitch_threshold
        self.timing_tolerance = timing_tolerance_ms / 1000.0  # Convert to seconds
        self.velocity_tolerance = velocity_tolerance

    def compute_all_metrics(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute all reconstruction metrics.

        Args:
            original: Original MIDI tensor (B, T, 128, K)
            reconstructed: Reconstructed MIDI tensor (B, T, 128, K)
            mask: Valid data mask (B, T, 128, K)

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Apply mask if provided
        if mask is not None:
            original = original * mask
            reconstructed = reconstructed * mask

        # 1. Pitch Accuracy
        metrics['pitch_accuracy'] = self.pitch_accuracy(original, reconstructed)

        # 2. Timing Similarity
        metrics['timing_similarity'] = self.timing_similarity(original, reconstructed)

        # 3. Velocity Similarity
        metrics['velocity_mae'] = self.velocity_mae(original, reconstructed)

        # 4. Harmonic Similarity
        metrics['harmonic_similarity'] = self.harmonic_similarity(original, reconstructed)

        # 5. Overall Score (weighted average)
        metrics['overall_score'] = (
            0.3 * metrics['pitch_accuracy'] +
            0.25 * metrics['timing_similarity'] +
            0.20 * (1.0 - metrics['velocity_mae'] / 127.0) +  # Normalize
            0.25 * metrics['harmonic_similarity']
        )

        return metrics

    def pitch_accuracy(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """
        Compute pitch accuracy (percentage of correct pitches).

        Binarize pianorolls and compute overlap.
        """
        # Binarize
        orig_binary = (original > self.pitch_threshold).float()
        recon_binary = (reconstructed > self.pitch_threshold).float()

        # Compute precision and recall
        true_positives = (orig_binary * recon_binary).sum()
        predicted_positives = recon_binary.sum()
        actual_positives = orig_binary.sum()

        precision = true_positives / (predicted_positives + 1e-8)
        recall = true_positives / (actual_positives + 1e-8)

        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

        return f1.item()

    def timing_similarity(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """
        Compute timing similarity (onset correlation).

        Extract onset times and compute correlation.
        """
        # Compute onset strength (sum across pitches and tracks)
        orig_onset = original.sum(dim=(2, 3))    # (B, T)
        recon_onset = reconstructed.sum(dim=(2, 3))

        # Compute correlation
        correlation = self._correlation(orig_onset, recon_onset)

        return correlation.item()

    def velocity_mae(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """
        Compute Mean Absolute Error of velocities.

        Only compute on active notes (not silence).
        """
        # Mask for active notes
        active = (original > self.pitch_threshold) | (reconstructed > self.pitch_threshold)

        if active.sum() == 0:
            return 0.0

        # Scale to 0-127
        orig_vel = original * 127
        recon_vel = reconstructed * 127

        # Compute MAE on active notes
        mae = (torch.abs(orig_vel - recon_vel) * active).sum() / active.sum()

        return mae.item()

    def harmonic_similarity(
        self,
        original: torch.Tensor,
        reconstructed: torch.Tensor
    ) -> float:
        """
        Compute harmonic similarity (chord progression similarity).

        Simplified: compute pitch class histograms and compare.
        """
        # Sum across time and tracks to get pitch class distribution
        orig_pc = original.sum(dim=(1, 3))    # (B, 128)
        recon_pc = reconstructed.sum(dim=(1, 3))

        # Normalize to distributions
        orig_pc = orig_pc / (orig_pc.sum(dim=1, keepdim=True) + 1e-8)
        recon_pc = recon_pc / (recon_pc.sum(dim=1, keepdim=True) + 1e-8)

        # Compute cosine similarity
        cosine_sim = (orig_pc * recon_pc).sum(dim=1) / (
            torch.norm(orig_pc, dim=1) * torch.norm(recon_pc, dim=1) + 1e-8
        )

        return cosine_sim.mean().item()

    def _correlation(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Pearson correlation."""
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_centered = x - x_mean
        y_centered = y - y_mean

        cov = (x_centered * y_centered).sum(dim=1)
        std_x = torch.sqrt((x_centered ** 2).sum(dim=1))
        std_y = torch.sqrt((y_centered ** 2).sum(dim=1))

        corr = cov / (std_x * std_y + 1e-8)

        return corr.mean()
```

### 4.3 Usage Example

```python
# Initialize metrics
metrics = MIDIReconstructionMetrics(
    pitch_threshold=0.5,
    timing_tolerance_ms=50.0,
    velocity_tolerance=10
)

# Compute metrics on batch
for batch in val_loader:
    original = batch['midi_tensor']
    mask = batch['mask']

    # Forward pass
    dna = encoder(original, mask=mask)
    reconstructed = decoder(dna, target_length=original.shape[1])

    # Compute metrics
    batch_metrics = metrics.compute_all_metrics(original, reconstructed, mask)

    print(f"Pitch Accuracy: {batch_metrics['pitch_accuracy']:.3f}")
    print(f"Timing Similarity: {batch_metrics['timing_similarity']:.3f}")
    print(f"Velocity MAE: {batch_metrics['velocity_mae']:.2f}")
    print(f"Harmonic Similarity: {batch_metrics['harmonic_similarity']:.3f}")
    print(f"Overall Score: {batch_metrics['overall_score']:.3f}")
```

## 5. File Organization

### 5.1 New Files to Create

```
midi_generator/
├── data/                                    # NEW: Agent 8's module
│   ├── __init__.py
│   ├── midi_reconstruction_dataset.py       # MIDIReconstructionDataset
│   ├── dna_cache.py                         # DNACache
│   ├── midi_cache.py                        # MIDICache (simple cache)
│   ├── variable_length_collate.py           # Collate functions
│   └── reconstruction_metrics.py            # MIDIReconstructionMetrics
├── learning/
│   └── gap_dataset.py                       # EXISTING - reuse GapCache patterns
├── multi_genre/
│   └── augmentation.py                      # EXISTING - reuse augmentation
└── training/
    └── hierarchical_mtl/data/
        └── dataset.py                       # EXISTING - reuse patterns
```

## 6. Performance Considerations

### 6.1 Memory Efficiency
- **Pianoroll size:** `(T, 128, K)` = ~5-10 MB per song (5 min, 20 tracks)
- **Batch size 32:** ~160-320 MB per batch
- **DNA cache:** 300 floats × 4 bytes = 1.2 KB per song → 10 GB = ~8M songs

### 6.2 Speed Optimizations
- **MIDI loading:** Use cache, hit rate target 70-80%
- **Augmentation:** On-the-fly (no disk storage)
- **Batching:** Use `num_workers=4` for parallel loading
- **DNA cache:** In-memory for hot paths, disk for cold storage

### 6.3 Bottleneck Analysis
1. **MIDI parsing:** ~10-50 ms per file (cached after first load)
2. **Feature extraction:** ~50-100 ms per file (optional)
3. **Augmentation:** ~5-10 ms per file (on-the-fly)
4. **Tensor conversion:** ~5-10 ms per file

**Total:** ~70-160 ms per file → **6-14 files/second**

**With caching:** ~5-10 ms per file → **100-200 files/second** ✅ Meets target

## 7. Integration Timeline

### Week 2 (Implementation)
- Day 1-2: `MIDIReconstructionDataset` + `MIDICache`
- Day 3-4: `DNACache` + integration
- Day 5: Variable-length collate functions
- Day 6-7: `MIDIReconstructionMetrics`

### Week 3 (Testing & Optimization)
- Day 1-2: Unit tests for all components
- Day 3-4: Integration testing with Agents 1, 2, 3
- Day 5: Performance benchmarking
- Day 6-7: Optimization and profiling

### Week 4 (Documentation & Finalization)
- Day 1-2: Write comprehensive documentation
- Day 3-4: Create usage examples and tutorials
- Day 5: Final integration testing
- Day 6-7: Code review and cleanup

## 8. Success Criteria

✅ **MIDIReconstructionDataset:**
- Loads MIDI files efficiently (>100 files/sec with cache)
- Integrates with existing augmentation pipeline
- Returns proper tensor format for Agent 1/2

✅ **DNACache:**
- Reduces encoding time >5x (cache hit rate >70%)
- Handles 10 GB cache without slowdown
- Thread-safe for multi-worker DataLoader

✅ **Variable-Length Collate:**
- Handles batches with different lengths
- Produces efficient padded tensors
- No crashes or memory leaks

✅ **Reconstruction Metrics:**
- Computes meaningful quality scores
- Correlates with human perception
- Fast enough for validation loop (<10 ms/batch)

---

**Phase 2 Status:** ✅ **DESIGN COMPLETE**
**Next:** Phase 3 Implementation (Weeks 2-4)
