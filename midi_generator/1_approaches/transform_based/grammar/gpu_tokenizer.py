"""
GPU-Accelerated Tokenization for SEQUITUR

This module provides GPU-accelerated preprocessing for musical data
before SEQUITUR grammar induction.

Key optimizations:
- Batch pitch_class extraction: pitch_class = pitch % 12 (single GPU kernel)
- Batch octave extraction: octave = pitch // 12 (single GPU kernel)
- Parallel rhythm hashing
- Batch token ID lookup

Author: GPU Tokenization for Musical Pattern Discovery
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import torch


@dataclass
class TokenizationConfig:
    """Configuration for tokenization vocabulary sizes."""
    rhythm_vocab_size: int = 1000  # Max distinct rhythm patterns
    pitch_class_size: int = 12  # Always 12 (chromatic)
    octave_size: int = 10  # MIDI range ~10 octaves
    duration_vocab_size: int = 64  # Quantized durations
    velocity_levels: int = 8  # ppp to fff

    @property
    def pitch_token_offset(self) -> int:
        """Start of pitch tokens in vocabulary."""
        return self.rhythm_vocab_size

    @property
    def duration_token_offset(self) -> int:
        """Start of duration tokens."""
        return self.pitch_token_offset + self.pitch_class_size * self.octave_size

    @property
    def velocity_token_offset(self) -> int:
        """Start of velocity tokens."""
        return self.duration_token_offset + self.duration_vocab_size

    @property
    def total_vocab_size(self) -> int:
        """Total vocabulary size."""
        return self.velocity_token_offset + self.velocity_levels


class GPUTokenizer:
    """
    GPU-accelerated tokenizer for musical objects.

    Processes batches of FactoredObjects in parallel on GPU.
    """

    def __init__(self,
                 config: Optional[TokenizationConfig] = None,
                 device: str = 'cuda'):
        """
        Initialize tokenizer.

        Args:
            config: Tokenization configuration
            device: PyTorch device ('cuda' or 'cpu')
        """
        self.config = config or TokenizationConfig()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Pre-allocate tensors for velocity quantization
        self._velocity_boundaries = torch.linspace(
            0, 1, self.config.velocity_levels + 1,
            device=self.device
        )

    def tokenize_batch(self,
                       objects: List,
                       verbose: bool = False) -> Tuple[List[List[int]], Dict]:
        """
        Tokenize a batch of FactoredObjects using GPU acceleration.

        Args:
            objects: List of FactoredObjects
            verbose: Print progress

        Returns:
            (token_sequences, stats) where each sequence is list of int tokens
        """
        if not objects:
            return [], {'total_objects': 0}

        if verbose:
            print(f"  GPU tokenizing {len(objects)} objects on {self.device}")

        # Group by number of notes for efficient batching
        objects_by_notes = {}
        object_indices = {}  # Map back to original order

        for i, obj in enumerate(objects):
            n = obj.num_notes
            if n not in objects_by_notes:
                objects_by_notes[n] = []
            objects_by_notes[n].append((i, obj))
            object_indices[i] = len(objects_by_notes[n]) - 1

        # Process each group
        all_tokens = [None] * len(objects)
        total_tokens = 0

        for n_notes, group in objects_by_notes.items():
            if n_notes == 0:
                # Empty objects get empty token list
                for orig_idx, obj in group:
                    rhythm_token = self._hash_rhythm(obj.rhythm)
                    all_tokens[orig_idx] = [rhythm_token]
                    total_tokens += 1
                continue

            # Extract data for this group
            batch_size = len(group)
            indices = [g[0] for g in group]
            objs = [g[1] for g in group]

            # Batch extract pitches, durations, velocities
            pitches = np.stack([obj.pitches for obj in objs])  # (B, N)
            durations = np.stack([obj.durations for obj in objs])  # (B, N)
            velocities = np.stack([obj.velocities for obj in objs])  # (B, N)

            # Move to GPU
            pitch_tensor = torch.tensor(pitches, device=self.device, dtype=torch.int32)
            duration_tensor = torch.tensor(durations, device=self.device, dtype=torch.int32)
            velocity_tensor = torch.tensor(velocities, device=self.device, dtype=torch.float32)

            # GPU operations: pitch_class and octave extraction
            pitch_class = (pitch_tensor % 12)  # (B, N)
            octave = torch.clamp(pitch_tensor // 12, 0, self.config.octave_size - 1)  # (B, N)

            # Quantize durations
            duration_quantized = torch.clamp(
                duration_tensor, 0, self.config.duration_vocab_size - 1
            )

            # Quantize velocities to levels
            velocity_quantized = torch.bucketize(
                velocity_tensor,
                self._velocity_boundaries[1:-1]  # Internal boundaries
            ).clamp(0, self.config.velocity_levels - 1)

            # Compute token IDs
            pitch_tokens = (
                self.config.pitch_token_offset +
                octave * 12 + pitch_class
            )  # (B, N)

            duration_tokens = (
                self.config.duration_token_offset +
                duration_quantized
            )  # (B, N)

            velocity_tokens = (
                self.config.velocity_token_offset +
                velocity_quantized
            )  # (B, N)

            # Move back to CPU
            pitch_tokens_cpu = pitch_tokens.cpu().numpy()
            duration_tokens_cpu = duration_tokens.cpu().numpy()
            velocity_tokens_cpu = velocity_tokens.cpu().numpy()

            # Build token sequences
            for batch_idx, orig_idx in enumerate(indices):
                obj = objs[batch_idx]
                tokens = []

                # Rhythm token (CPU - hashing not easily parallelizable)
                rhythm_token = self._hash_rhythm(obj.rhythm)
                tokens.append(rhythm_token)

                # Interleave pitch, duration, velocity for each note
                for note_idx in range(n_notes):
                    tokens.append(int(pitch_tokens_cpu[batch_idx, note_idx]))
                    tokens.append(int(duration_tokens_cpu[batch_idx, note_idx]))
                    tokens.append(int(velocity_tokens_cpu[batch_idx, note_idx]))

                all_tokens[orig_idx] = tokens
                total_tokens += len(tokens)

        stats = {
            'total_objects': len(objects),
            'total_tokens': total_tokens,
            'avg_tokens_per_object': total_tokens / len(objects) if objects else 0,
            'device': str(self.device)
        }

        if verbose:
            print(f"    Total tokens: {total_tokens:,}")
            print(f"    Avg per object: {stats['avg_tokens_per_object']:.1f}")

        return all_tokens, stats

    def _hash_rhythm(self, rhythm: np.ndarray) -> int:
        """Hash a rhythm pattern to a token ID."""
        return hash(rhythm.tobytes()) % self.config.rhythm_vocab_size

    def tokenize_corpus(self,
                        tracks: List[List],
                        object_separator: int = -1,
                        track_separator: int = -2,
                        verbose: bool = True) -> Tuple[List[int], Dict]:
        """
        Tokenize entire corpus (multiple tracks of objects).

        Args:
            tracks: List of lists of FactoredObjects
            object_separator: Token between objects
            track_separator: Token between tracks
            verbose: Print progress

        Returns:
            (flat_tokens, stats) single sequence for SEQUITUR
        """
        if verbose:
            print(f"\n{'='*70}")
            print("GPU CORPUS TOKENIZATION")
            print(f"{'='*70}")
            total_objects = sum(len(t) for t in tracks)
            print(f"  Tracks: {len(tracks)}")
            print(f"  Total objects: {total_objects}")

        flat_tokens = []
        all_objects = []

        # Collect all objects
        for track in tracks:
            all_objects.extend(track)

        # Batch tokenize
        token_sequences, stats = self.tokenize_batch(all_objects, verbose=verbose)

        # Flatten with separators
        seq_idx = 0
        for track_idx, track in enumerate(tracks):
            for obj_idx, obj in enumerate(track):
                flat_tokens.extend(token_sequences[seq_idx])
                seq_idx += 1

                if obj_idx < len(track) - 1:
                    flat_tokens.append(object_separator)

            if track_idx < len(tracks) - 1:
                flat_tokens.append(track_separator)

        stats['flat_tokens'] = len(flat_tokens)

        if verbose:
            print(f"  Flat sequence length: {len(flat_tokens):,}")

        return flat_tokens, stats


class BatchPitchExtractor:
    """
    Specialized GPU extractor for pitch-class and octave.

    This is the critical path for GPU optimization.
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def extract(self, pitch_arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch-class and octave from batch of pitch arrays.

        Args:
            pitch_arrays: List of pitch arrays (variable length)

        Returns:
            (pitch_class_padded, octave_padded) with shape (B, max_len)
        """
        if not pitch_arrays:
            return np.array([]), np.array([])

        # Find max length
        max_len = max(len(p) for p in pitch_arrays)
        batch_size = len(pitch_arrays)

        # Pad and stack
        padded = np.zeros((batch_size, max_len), dtype=np.int32)
        mask = np.zeros((batch_size, max_len), dtype=bool)

        for i, p in enumerate(pitch_arrays):
            n = len(p)
            if n > 0:
                padded[i, :n] = p
                mask[i, :n] = True

        # Move to GPU
        pitch_tensor = torch.tensor(padded, device=self.device, dtype=torch.int32)

        # Single kernel operations
        pitch_class = (pitch_tensor % 12).cpu().numpy()
        octave = (pitch_tensor // 12).cpu().numpy()

        return pitch_class, octave, mask


def create_dense_token_lookup_table(config: TokenizationConfig,
                                     device: str = 'cuda') -> torch.Tensor:
    """
    Create a dense lookup table for pitch -> token conversion.

    This enables O(1) token lookup via table indexing.

    Args:
        config: Tokenization config
        device: PyTorch device

    Returns:
        Lookup table tensor of shape (128,) mapping MIDI pitch to token
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # MIDI pitches 0-127
    lookup = torch.zeros(128, dtype=torch.int32, device=device)

    for pitch in range(128):
        pitch_class = pitch % 12
        octave = min(pitch // 12, config.octave_size - 1)
        token = config.pitch_token_offset + octave * 12 + pitch_class
        lookup[pitch] = token

    return lookup


# =============================================================================
# INTEGRATION FUNCTIONS
# =============================================================================

def tokenize_factored_objects_gpu(objects: List,
                                   config: Optional[TokenizationConfig] = None,
                                   verbose: bool = True) -> Tuple[List[int], Dict]:
    """
    High-level function to tokenize FactoredObjects with GPU.

    Args:
        objects: List of FactoredObjects
        config: Optional tokenization config
        verbose: Print progress

    Returns:
        (token_sequence, stats)
    """
    tokenizer = GPUTokenizer(config=config)
    return tokenizer.tokenize_corpus([[obj] for obj in objects], verbose=verbose)


def run_gpu_tokenization_benchmark(n_objects: int = 10000,
                                    n_notes_per_object: int = 16):
    """
    Benchmark GPU vs CPU tokenization.
    """
    import time

    print(f"\nBenchmarking tokenization: {n_objects} objects, {n_notes_per_object} notes each")

    # Create mock objects
    class MockObject:
        def __init__(self, n_notes):
            self.num_notes = n_notes
            self.rhythm = np.random.randint(0, 2, 32).astype(np.float32)
            self.pitches = np.random.randint(36, 84, n_notes).astype(np.int32)
            self.durations = np.random.randint(1, 16, n_notes).astype(np.int32)
            self.velocities = np.random.uniform(0.3, 1.0, n_notes).astype(np.float32)

    objects = [MockObject(n_notes_per_object) for _ in range(n_objects)]

    # GPU timing
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = time.time()

        tokenizer = GPUTokenizer(device='cuda')
        tokens, stats = tokenizer.tokenize_batch(objects, verbose=False)

        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"  GPU time: {gpu_time:.3f}s ({n_objects / gpu_time:.0f} obj/sec)")
    else:
        print("  GPU not available")

    # CPU timing
    start = time.time()
    tokenizer_cpu = GPUTokenizer(device='cpu')
    tokens_cpu, stats_cpu = tokenizer_cpu.tokenize_batch(objects, verbose=False)
    cpu_time = time.time() - start

    print(f"  CPU time: {cpu_time:.3f}s ({n_objects / cpu_time:.0f} obj/sec)")

    if torch.cuda.is_available():
        print(f"  Speedup: {cpu_time / gpu_time:.1f}x")


if __name__ == '__main__':
    run_gpu_tokenization_benchmark()
