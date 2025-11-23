"""
Tensorized MIDI representation for GPU processing.

Key insight: Represent MIDI as fixed-size tensors for batch processing.

Expected speedup: 10-50x for discovery pipeline
- Batch sparse coding: 2000 pieces simultaneously
- Parallel transform application
- GPU memory optimized for A100 (80GB)

Author: Agent 8 - GPU Tensorization
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import mido


class TensorMIDICorpus:
    """
    Convert MIDI corpus to GPU-friendly tensor representation.

    Representation:
        pieces_tensor: (B, T, F) where:
            B = batch size (number of pieces)
            T = time steps (max length, padded)
            F = features (pitch + velocity + track + channel + is_drum)

    Feature Encoding (F=132):
        [0:128]   - Pitch (one-hot encoding for 128 MIDI pitches)
        [128]     - Velocity (normalized 0-1)
        [129]     - Track ID (normalized 0-1, representing 0-20 tracks)
        [130]     - Channel (normalized 0-1, representing 0-15 channels)
        [131]     - is_drum flag (0 or 1)

    Benefits:
        - Fixed-size tensors enable batch processing
        - GPU-friendly format (contiguous memory)
        - Parallel operations across all pieces
        - Memory-efficient for A100 (2000 pieces × 2000 steps × 132 features × 4 bytes ≈ 2 GB)
    """

    def __init__(self, max_time_steps: int = 2000, num_features: int = 133):
        """
        Args:
            max_time_steps: Maximum time steps (bars × 16 for 16th note resolution)
                           Default 2000 = 125 bars at 16th note resolution
            num_features: 133 total:
                [0:128]   - Pitch (one-hot encoding for 128 MIDI pitches)
                [128]     - Velocity (normalized 0-1)
                [129]     - Program/Instrument (General MIDI 0-127, normalized 0-1) ← CRITICAL!
                [130]     - Channel (0-15, normalized 0-1)
                [131]     - is_drum flag (0 or 1)
                [132]     - Track ID (0-19, normalized 0-1) - auxiliary for debugging
        """
        self.max_time_steps = max_time_steps
        self.num_features = num_features

    def midi_to_tensor(self, midi_file: mido.MidiFile) -> torch.Tensor:
        """
        Convert single MIDI file to tensor.

        Args:
            midi_file: mido.MidiFile object

        Returns:
            tensor: (T, F) - time steps × features
        """
        from core.space_level_transforms import extract_notes_from_midi

        # Extract notes with metadata
        notes = extract_notes_from_midi(midi_file)

        if not notes:
            # Empty MIDI file
            return torch.zeros(self.max_time_steps, self.num_features)

        # Create time grid (16th note resolution)
        ticks_per_16th = midi_file.ticks_per_beat // 4
        max_ticks = max(n['start_time'] + n['duration'] for n in notes)
        num_steps = min(int(max_ticks / ticks_per_16th) + 1, self.max_time_steps)

        # Initialize tensor (T, F)
        tensor = torch.zeros(self.max_time_steps, self.num_features)

        # Fill in notes
        for note in notes:
            start_step = int(note['start_time'] / ticks_per_16th)
            duration_steps = max(1, int(note['duration'] / ticks_per_16th))
            end_step = min(start_step + duration_steps, self.max_time_steps)

            if start_step >= self.max_time_steps:
                continue

            for step in range(start_step, end_step):
                # Pitch (one-hot encoding for first 128 dims)
                pitch = np.clip(note['pitch'], 0, 127)
                tensor[step, pitch] = 1.0

                # Velocity (normalized 0-1)
                tensor[step, 128] = note['velocity'] / 127.0

                # Program/Instrument (General MIDI 0-127, normalized 0-1)
                # CRITICAL: This is instrument IDENTITY, not track position!
                # program=0 is always "Acoustic Grand Piano" across ALL files
                # program=32 is always "Acoustic Bass" across ALL files
                # program=56 is always "Trumpet" across ALL files, etc.
                tensor[step, 129] = note.get('program', 0) / 127.0

                # Channel (normalized 0-1)
                tensor[step, 130] = note.get('channel', 0) / 15.0

                # Is drum flag (channel 9 = drums)
                tensor[step, 131] = 1.0 if note.get('is_drum', False) else 0.0

                # Track ID (normalized 0-1, auxiliary for debugging)
                # Note: Track position varies across files, so this is NOT used for filtering
                tensor[step, 132] = np.clip(note.get('track', 0), 0, 19) / 20.0

        return tensor

    def batch_midi_to_tensor(
        self,
        midi_files: List[mido.MidiFile],
        device: str = 'cuda',
        show_progress: bool = True
    ) -> torch.Tensor:
        """
        Convert batch of MIDI files to tensor.

        Args:
            midi_files: List of mido.MidiFile objects
            device: 'cuda' or 'cpu'
            show_progress: Show progress bar

        Returns:
            batch_tensor: (B, T, F) on specified device
        """
        tensors = []

        iterator = midi_files
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(midi_files, desc="Converting MIDI to tensors")
            except ImportError:
                pass

        for midi in iterator:
            tensor = self.midi_to_tensor(midi)
            tensors.append(tensor)

        # Stack into batch
        batch = torch.stack(tensors, dim=0)

        # Move to device
        return batch.to(device)

    def tensor_to_midi(
        self,
        tensor: torch.Tensor,
        ticks_per_beat: int = 480,
        velocity_threshold: float = 0.3
    ) -> mido.MidiFile:
        """
        Convert tensor back to MIDI file.

        Args:
            tensor: (T, F) or (B, T, F)
            ticks_per_beat: MIDI ticks per beat
            velocity_threshold: Minimum velocity to consider a note (0-1)

        Returns:
            midi_file: mido.MidiFile object
        """
        if tensor.dim() == 3:
            # Take first in batch if batched
            tensor = tensor[0]

        tensor = tensor.cpu()

        # Extract notes from tensor
        notes = []
        active_notes = {}  # (pitch, track) -> start_step

        ticks_per_16th = ticks_per_beat // 4

        for step in range(tensor.shape[0]):
            # Find active pitches at this step
            pitch_vector = tensor[step, :128]
            velocity = tensor[step, 128].item()

            # Skip if velocity too low
            if velocity < velocity_threshold:
                continue

            active_pitches = torch.where(pitch_vector > 0.5)[0]

            # Extract metadata
            program = int(tensor[step, 129].item() * 127)  # Instrument identity
            channel = int(tensor[step, 130].item() * 15)
            is_drum = tensor[step, 131].item() > 0.5
            track = int(tensor[step, 132].item() * 20)  # Auxiliary (for track assignment)

            for pitch_tensor in active_pitches:
                pitch = int(pitch_tensor.item())
                key = (pitch, track)

                if key not in active_notes:
                    # Note onset
                    active_notes[key] = {
                        'start_step': step,
                        'velocity': velocity,
                        'channel': channel,
                        'program': program,  # Instrument identity
                        'is_drum': is_drum
                    }

            # Check for note offsets (pitch was active, now inactive)
            to_remove = []
            for key, note_info in active_notes.items():
                pitch, track = key
                start_step = note_info['start_step']

                # If note was active but now inactive, or we're at a new onset
                if step > start_step and (tensor[step, pitch] < 0.5 or
                                         (tensor[step, pitch] > 0.5 and step > start_step + 1)):
                    # Note offset - create note
                    notes.append({
                        'pitch': pitch,
                        'velocity': int(note_info['velocity'] * 127),
                        'start_time': start_step * ticks_per_16th,
                        'duration': max(ticks_per_16th, (step - start_step) * ticks_per_16th),
                        'track': track,
                        'channel': note_info['channel'],
                        'program': note_info.get('program', 0),  # Instrument identity
                        'is_drum': note_info['is_drum']
                    })

                    to_remove.append(key)

            for key in to_remove:
                del active_notes[key]

        # Close any remaining active notes
        final_step = tensor.shape[0]
        for key, note_info in active_notes.items():
            pitch, track = key
            start_step = note_info['start_step']

            notes.append({
                'pitch': pitch,
                'velocity': int(note_info['velocity'] * 127),
                'start_time': start_step * ticks_per_16th,
                'duration': max(ticks_per_16th, (final_step - start_step) * ticks_per_16th),
                'track': track,
                'channel': note_info['channel'],
                'program': note_info.get('program', 0),  # Instrument identity
                'is_drum': note_info['is_drum']
            })

        # Convert notes to MIDI
        from core.space_level_transforms import notes_to_midi
        return notes_to_midi(notes, ticks_per_beat)

    def estimate_memory_usage(self, num_pieces: int, device: str = 'cuda') -> Dict[str, float]:
        """
        Estimate GPU memory usage for a corpus.

        Args:
            num_pieces: Number of MIDI files in corpus
            device: 'cuda' or 'cpu'

        Returns:
            memory_stats: Dict with memory estimates in GB
        """
        bytes_per_element = 4  # float32

        # Corpus tensor: (B, T, F) where F=133
        corpus_elements = num_pieces * self.max_time_steps * self.num_features
        corpus_gb = (corpus_elements * bytes_per_element) / 1e9

        # Transform dictionary (estimate 500 transforms): (M, T, F)
        num_transforms = 500
        dict_elements = num_transforms * self.max_time_steps * self.num_features
        dict_gb = (dict_elements * bytes_per_element) / 1e9

        # Sparse encodings: (B, M)
        encodings_elements = num_pieces * num_transforms
        encodings_gb = (encodings_elements * bytes_per_element) / 1e9

        # Working memory (intermediate computations, estimate 3x corpus)
        working_gb = corpus_gb * 3

        total_gb = corpus_gb + dict_gb + encodings_gb + working_gb

        return {
            'corpus_tensor_gb': corpus_gb,
            'transform_dict_gb': dict_gb,
            'sparse_encodings_gb': encodings_gb,
            'working_memory_gb': working_gb,
            'total_estimated_gb': total_gb,
            'num_pieces': num_pieces,
            'num_transforms': num_transforms,
            'fits_in_a100_80gb': total_gb < 70  # Reserve 10GB for system
        }


# ============================================================================
# Utility Functions
# ============================================================================

def load_corpus_to_gpu(
    midi_files: List[mido.MidiFile],
    max_time_steps: int = 2000,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, TensorMIDICorpus]:
    """
    Convenience function to load MIDI corpus to GPU.

    Args:
        midi_files: List of mido.MidiFile objects
        max_time_steps: Maximum time steps
        device: 'cuda' or 'cpu'

    Returns:
        corpus_tensor: (B, T, F) on GPU
        converter: TensorMIDICorpus instance for later conversion back
    """
    converter = TensorMIDICorpus(max_time_steps=max_time_steps)

    print(f"Loading {len(midi_files)} MIDI files to {device}...")

    # Estimate memory
    mem_stats = converter.estimate_memory_usage(len(midi_files), device)
    print(f"Estimated memory usage: {mem_stats['total_estimated_gb']:.2f} GB")

    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, cannot use GPU device")

        if not mem_stats['fits_in_a100_80gb']:
            print(f"WARNING: Estimated memory ({mem_stats['total_estimated_gb']:.1f} GB) " +
                  "may exceed A100 capacity (80 GB). Consider processing in chunks.")

    # Convert to tensors
    corpus_tensor = converter.batch_midi_to_tensor(midi_files, device=device, show_progress=True)

    print(f"Corpus tensor shape: {corpus_tensor.shape}")

    if device == 'cuda':
        actual_gb = corpus_tensor.element_size() * corpus_tensor.nelement() / 1e9
        print(f"Actual GPU memory (corpus only): {actual_gb:.2f} GB")

    return corpus_tensor, converter


def save_tensor_corpus(corpus_tensor: torch.Tensor, path: str):
    """Save tensorized corpus to disk."""
    torch.save(corpus_tensor.cpu(), path)
    print(f"Saved corpus tensor to {path}")


def load_tensor_corpus(path: str, device: str = 'cuda') -> torch.Tensor:
    """Load tensorized corpus from disk."""
    corpus_tensor = torch.load(path)
    corpus_tensor = corpus_tensor.to(device)
    print(f"Loaded corpus tensor from {path}")
    print(f"Shape: {corpus_tensor.shape}")
    return corpus_tensor
