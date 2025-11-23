"""
GPU-accelerated transform operations.

Key insight: Represent transforms as tensor operations for batch processing.
Each transform operates on (B, T, F) tensors where B=batch size.

Expected speedup: 50-100x per transform application
- Parallel across all pieces in batch
- GPU SIMD operations on note vectors
- Memory-coalesced access patterns

Author: Agent 8 - GPU Tensorization
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import math


class TensorTransformLibrary:
    """
    Library of transforms as batched tensor operations.

    Philosophy: Each transform is a pure function (B,T,F) → (B,T,F)
    This enables:
    - Batch processing (2000 pieces simultaneously)
    - GPU parallelism
    - Transform composition as function composition
    """

    @staticmethod
    def transpose_semitone(batch: torch.Tensor, amount: int) -> torch.Tensor:
        """
        Transpose all pitches by semitones (T₁ generator).

        Args:
            batch: (B, T, F) where F[0:128] is pitch one-hot
            amount: semitones to transpose (-12 to 12)

        Returns:
            transposed: (B, T, F)

        Implementation:
            - Circular shift of pitch vector
            - Skip drums (F[131] is is_drum flag)
            - Preserve all other features
        """
        B, T, F = batch.shape
        result = batch.clone()

        # Extract pitch vector
        pitch = batch[:, :, :128]  # (B, T, 128)

        # Don't transpose drums (F[131] is is_drum flag)
        is_drum = batch[:, :, 131:132]  # (B, T, 1)
        drum_mask = (is_drum > 0.5).expand(-1, -1, 128)

        # Circular shift (transpose)
        # torch.roll shifts along dimension, wrapping around
        pitch_transposed = torch.roll(pitch, shifts=amount, dims=2)

        # Apply only to non-drums
        pitch_final = torch.where(drum_mask, pitch, pitch_transposed)

        result[:, :, :128] = pitch_final
        return result

    @staticmethod
    def inversion(batch: torch.Tensor, center: Optional[int] = None) -> torch.Tensor:
        """
        Invert pitches around center (I₀ reflection).

        Args:
            batch: (B, T, F)
            center: Inversion center pitch (default: compute from data)

        Returns:
            inverted: (B, T, F)
        """
        result = batch.clone()

        # Extract pitch vector
        pitch = batch[:, :, :128]  # (B, T, 128)

        # Don't invert drums
        is_drum = batch[:, :, 131:132]  # (B, T, 1)
        drum_mask = (is_drum > 0.5).expand(-1, -1, 128)

        if center is None:
            # Compute center from non-drum pitches
            # Weighted average of pitch classes
            pitch_indices = torch.arange(128, device=batch.device)
            non_drum_pitch = pitch * (~drum_mask).float()
            total_weight = non_drum_pitch.sum()

            if total_weight > 0:
                center = int((non_drum_pitch * pitch_indices).sum() / total_weight)
            else:
                center = 60  # Middle C default

        # Create inverted pitch vector
        # For each pitch p, inverted pitch is (2*center - p)
        pitch_indices = torch.arange(128, device=batch.device)
        inverted_indices = 2 * center - pitch_indices
        inverted_indices = torch.clamp(inverted_indices, 0, 127)

        # Build mapping matrix (128 × 128)
        # mapping[i, j] = 1 if inverted_indices[i] == j
        mapping = torch.zeros(128, 128, device=batch.device)
        for i in range(128):
            j = inverted_indices[i]
            mapping[i, j] = 1.0

        # Apply inversion: (B, T, 128) @ (128, 128) = (B, T, 128)
        pitch_inverted = torch.matmul(pitch, mapping)

        # Apply only to non-drums
        pitch_final = torch.where(drum_mask, pitch, pitch_inverted)

        result[:, :, :128] = pitch_final
        return result

    @staticmethod
    def velocity_scale(batch: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale velocity by factor (V_s dynamics).

        Args:
            batch: (B, T, F) where F[128] is velocity
            scale: velocity multiplier (0.5 = softer, 1.5 = louder)

        Returns:
            scaled: (B, T, F)
        """
        result = batch.clone()
        result[:, :, 128] = torch.clamp(batch[:, :, 128] * scale, 0.0, 1.0)
        return result

    @staticmethod
    def instrument_filter(batch: torch.Tensor, target_program: int) -> torch.Tensor:
        """
        Keep only notes from target instrument (General MIDI program).

        CRITICAL: Uses program number (instrument identity), NOT track position!
        This ensures patterns generalize across files with different arrangements.

        Args:
            batch: (B, T, F) where F[129] is program/instrument number
            target_program: which instrument to keep (0-127, General MIDI)
                Examples:
                - 0 = Acoustic Grand Piano
                - 32 = Acoustic Bass
                - 56 = Trumpet
                - 64 = Soprano Sax
                - Channel 9 (any program) = Drums

        Returns:
            filtered: (B, T, F) with only target instrument
        """
        result = batch.clone()

        # Program number is normalized 0-1 (representing 0-127 instruments)
        program = batch[:, :, 129]  # (B, T)
        target_normalized = target_program / 127.0

        # Mask: keep only target instrument
        # Tolerance of ±0.004 ≈ ±0.5 program number for float precision
        mask = torch.abs(program - target_normalized) < 0.004
        mask = mask.unsqueeze(-1).expand(-1, -1, 128)  # Expand to pitch dimensions

        # Zero out non-matching instruments
        result[:, :, :128] = batch[:, :, :128] * mask.float()

        return result

    @staticmethod
    def track_filter(batch: torch.Tensor, target_track: int) -> torch.Tensor:
        """
        DEPRECATED: Use instrument_filter() instead!

        Track position varies across files - this breaks cross-file learning.
        Kept for backward compatibility only.

        Use instrument_filter(batch, program) where program is General MIDI number.
        """
        result = batch.clone()

        # Track ID is in feature 132 (auxiliary)
        track_id = batch[:, :, 132]  # (B, T)
        target_normalized = target_track / 20.0

        mask = torch.abs(track_id - target_normalized) < 0.025
        mask = mask.unsqueeze(-1).expand(-1, -1, 128)

        result[:, :, :128] = batch[:, :, :128] * mask.float()

        return result

    @staticmethod
    def time_scale(batch: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Scale time (augmentation/diminution S_r).

        Args:
            batch: (B, T, F)
            scale: time multiplier (0.5 = faster, 2.0 = slower)

        Returns:
            scaled: (B, T, F)
        """
        B, T, F = batch.shape

        # Resample along time dimension
        # Use interpolation to change temporal resolution
        batch_permuted = batch.permute(0, 2, 1)  # (B, F, T) for interpolate

        new_length = int(T * scale)
        resampled = F.interpolate(
            batch_permuted,
            size=new_length,
            mode='linear',
            align_corners=False
        )

        # Pad or truncate to original length
        if new_length < T:
            # Pad with zeros
            padding = torch.zeros(B, F, T - new_length, device=batch.device)
            resampled = torch.cat([resampled, padding], dim=2)
        elif new_length > T:
            # Truncate
            resampled = resampled[:, :, :T]

        return resampled.permute(0, 2, 1)  # Back to (B, T, F)

    @staticmethod
    def retrograde(batch: torch.Tensor) -> torch.Tensor:
        """
        Reverse time (R time reversal).

        Args:
            batch: (B, T, F)

        Returns:
            reversed: (B, T, F)
        """
        # Flip along time dimension
        return torch.flip(batch, dims=[1])

    @staticmethod
    def time_shift(batch: torch.Tensor, shift: int) -> torch.Tensor:
        """
        Shift in time (O_t temporal translation).

        Args:
            batch: (B, T, F)
            shift: number of time steps to shift (positive = later, negative = earlier)

        Returns:
            shifted: (B, T, F)
        """
        # Roll along time dimension
        return torch.roll(batch, shifts=shift, dims=1)

    @staticmethod
    def segment_slice(batch: torch.Tensor, start: float, end: float) -> torch.Tensor:
        """
        Extract temporal segment (score-level segmentation).

        Args:
            batch: (B, T, F)
            start: start position (0.0 - 1.0)
            end: end position (0.0 - 1.0)

        Returns:
            segment: (B, T, F) with only [start, end] region active
        """
        T = batch.shape[1]
        start_idx = int(start * T)
        end_idx = int(end * T)

        result = torch.zeros_like(batch)
        segment_length = end_idx - start_idx

        if segment_length > 0:
            result[:, :segment_length, :] = batch[:, start_idx:end_idx, :]

        return result

    @staticmethod
    def instrument_derive(
        batch: torch.Tensor,
        source_program: int,
        target_program: int
    ) -> torch.Tensor:
        """
        Derive notes from source instrument and assign to target instrument.

        CRITICAL: Uses program numbers (instrument identity), NOT track positions!

        Examples:
            - derive(piano=0, bass=32) ∘ T₁⁻¹² = "Bass plays octave below piano"
            - derive(sax1=64, sax2=65) ∘ T₁⁻³ = "Alto sax harmonizes 3 semitones below soprano"

        Args:
            batch: (B, T, F)
            source_program: instrument to copy from (0-127, General MIDI)
            target_program: instrument to assign to (0-127, General MIDI)

        Returns:
            derived: (B, T, F) with derived notes added
        """
        result = batch.clone()

        # Extract source instrument notes
        source_norm = source_program / 127.0
        program = batch[:, :, 129]  # Program is in feature 129
        source_mask = (torch.abs(program - source_norm) < 0.004).unsqueeze(-1).expand(-1, -1, 133)

        # Copy source notes
        source_notes = batch * source_mask.float()

        # Update program number for derived notes
        source_notes[:, :, 129] = target_program / 127.0

        # Add to result (combine with existing)
        result = result + source_notes

        # Clamp pitch (one-hot, so max 1.0)
        result[:, :, :128] = torch.clamp(result[:, :, :128], 0.0, 1.0)

        return result

    @staticmethod
    def track_derive(
        batch: torch.Tensor,
        source_track: int,
        target_track: int
    ) -> torch.Tensor:
        """
        DEPRECATED: Use instrument_derive() instead!

        Track position varies across files - this breaks cross-file learning.
        Use instrument_derive(batch, source_program, target_program) instead.
        """
        result = batch.clone()

        # Track ID is in feature 132 (auxiliary)
        source_norm = source_track / 20.0
        track_id = batch[:, :, 132]
        source_mask = (torch.abs(track_id - source_norm) < 0.025).unsqueeze(-1).expand(-1, -1, 133)

        source_notes = batch * source_mask.float()
        source_notes[:, :, 132] = target_track / 20.0

        result = result + source_notes
        result[:, :, :128] = torch.clamp(result[:, :, :128], 0.0, 1.0)

        return result

    @staticmethod
    def voice_select(batch: torch.Tensor, voice_index: int) -> torch.Tensor:
        """
        Extract specific voice from polyphonic texture.

        Args:
            batch: (B, T, F)
            voice_index: 0=bass, 1=tenor, 2=alto, 3=soprano

        Returns:
            voice: (B, T, F) with only selected voice
        """
        B, T, F = batch.shape
        result = torch.zeros_like(batch)

        # For each time step, find the voice_index-th active pitch
        pitch = batch[:, :, :128]  # (B, T, 128)

        for b in range(B):
            for t in range(T):
                active_pitches = torch.where(pitch[b, t] > 0.5)[0]

                if len(active_pitches) > voice_index:
                    # Sort pitches (0=lowest, 3=highest)
                    sorted_pitches = torch.sort(active_pitches)[0]

                    selected_pitch = sorted_pitches[voice_index]

                    # Copy selected pitch to result
                    result[b, t, :] = batch[b, t, :]
                    result[b, t, :128] = 0.0  # Clear all pitches
                    result[b, t, selected_pitch] = 1.0  # Set selected pitch

        return result

    @staticmethod
    def quantize_16th(batch: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Quantize to 16th note grid (Q quantization).

        Args:
            batch: (B, T, F)
            strength: quantization strength (1.0 = full, 0.5 = partial)

        Returns:
            quantized: (B, T, F)

        Note: Since tensor representation is already on 16th grid,
        this applies temporal smoothing to enforce grid alignment.
        """
        if strength < 0.01:
            return batch

        # Apply temporal convolution to smooth note onsets
        kernel_size = max(1, int(4 * (1.0 - strength)))  # Smaller kernel = more quantized

        # Convolve along time dimension
        # Create smoothing kernel
        kernel = torch.ones(1, 1, kernel_size, device=batch.device) / kernel_size

        # Apply to pitch features
        B, T, F = batch.shape
        pitch = batch[:, :, :128].permute(0, 2, 1).unsqueeze(1)  # (B, 1, 128, T)

        # Pad for same-size output
        padding = kernel_size // 2
        pitch_smoothed = F.conv2d(pitch, kernel, padding=(0, padding))

        # Threshold to enforce binary
        pitch_smoothed = (pitch_smoothed > 0.5).float()

        result = batch.clone()
        result[:, :, :128] = pitch_smoothed.squeeze(1).permute(0, 2, 1)

        return result

    @staticmethod
    def compose_transforms(
        batch: torch.Tensor,
        transforms: List[Tuple[str, float]]
    ) -> torch.Tensor:
        """
        Apply sequence of transforms (composition).

        Args:
            batch: (B, T, F)
            transforms: [(name, amount), ...] in order of application

        Returns:
            result: (B, T, F) after all transforms

        Example:
            transforms = [
                ('transpose_semitone', 7),     # T₁⁷ (transpose up 5th)
                ('track_filter', 0.1),         # Filter to track 1
                ('velocity_scale', 1.5)        # Louder
            ]
            # Result: "transpose track 1 up a 5th and make louder"
        """
        result = batch
        lib = TensorTransformLibrary()

        for transform_name, amount in transforms:
            if transform_name == 'transpose_semitone':
                result = lib.transpose_semitone(result, int(amount))

            elif transform_name == 'inversion':
                result = lib.inversion(result, center=int(amount) if amount != 0 else None)

            elif transform_name == 'velocity_scale':
                result = lib.velocity_scale(result, amount)

            elif transform_name == 'instrument_filter':
                result = lib.instrument_filter(result, int(amount * 127))

            elif transform_name == 'track_filter':
                # Deprecated: kept for backward compatibility
                result = lib.track_filter(result, int(amount * 20))

            elif transform_name == 'instrument_derive':
                # amount encodes [source, target] as 0.XXYY where XX and YY are program numbers
                source = int(amount * 100) // 100 * 127 // 100
                target = int(amount * 10000) % 100 * 127 // 100
                result = lib.instrument_derive(result, source, target)

            elif transform_name == 'time_scale':
                result = lib.time_scale(result, amount)

            elif transform_name == 'retrograde':
                result = lib.retrograde(result)

            elif transform_name == 'time_shift':
                result = lib.time_shift(result, int(amount))

            elif transform_name == 'segment_slice':
                # amount encodes [start, end] as 0.XXYY
                start = int(amount * 100) // 100
                end = int(amount * 100) % 100 / 100.0
                result = lib.segment_slice(result, start, end)

            elif transform_name == 'track_derive':
                # amount encodes [source, target] as 0.XY
                source = int(amount * 10)
                target = int(amount * 100) % 10
                result = lib.track_derive(result, source, target)

            elif transform_name == 'voice_select':
                result = lib.voice_select(result, int(amount * 4))

            elif transform_name == 'quantize_16th':
                result = lib.quantize_16th(result, amount)

            else:
                print(f"Warning: Unknown transform '{transform_name}', skipping")

        return result


# ============================================================================
# Transform Dictionary Builder
# ============================================================================

def create_transform_dictionary_tensor(
    transforms: List[Dict],
    max_time_steps: int = 2000,
    num_features: int = 133,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Create dictionary tensor from list of transforms.

    Args:
        transforms: List of {name: str, amount: float} dicts
        max_time_steps: T dimension
        num_features: F dimension (default 133)
        device: 'cuda' or 'cpu'

    Returns:
        dict_tensor: (M, T, F) where M = number of transforms

    Implementation:
        Apply each transform to identity tensor to get its matrix representation.
    """
    M = len(transforms)
    T = max_time_steps
    F = num_features

    # Create identity tensor (single note at middle C for reference)
    identity = torch.zeros(1, T, F, device=device)
    identity[0, T//2, 60] = 1.0  # Middle C
    identity[0, T//2, 128] = 0.8  # Velocity
    identity[0, T//2, 129] = 0.0  # Program 0 (Acoustic Grand Piano)
    identity[0, T//2, 130] = 0.0  # Channel 0
    identity[0, T//2, 131] = 0.0  # Not drum
    identity[0, T//2, 132] = 0.0  # Track 0 (auxiliary)

    lib = TensorTransformLibrary()
    dict_tensors = []

    for transform in transforms:
        # Apply transform to identity
        transformed = lib.compose_transforms(
            identity,
            [(transform['name'], transform['amount'])]
        )
        dict_tensors.append(transformed[0])  # Remove batch dim

    return torch.stack(dict_tensors, dim=0)  # (M, T, F)


def batch_apply_transform(
    corpus_tensor: torch.Tensor,
    transform: Dict,
    chunk_size: int = 500
) -> torch.Tensor:
    """
    Apply single transform to corpus in chunks (memory-efficient).

    Args:
        corpus_tensor: (B, T, F) on GPU
        transform: {name: str, amount: float}
        chunk_size: Process this many pieces at once

    Returns:
        result_tensor: (B, T, F) after transform
    """
    B = corpus_tensor.shape[0]
    num_chunks = (B + chunk_size - 1) // chunk_size

    lib = TensorTransformLibrary()
    results = []

    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, B)

        chunk = corpus_tensor[start:end]
        chunk_result = lib.compose_transforms(
            chunk,
            [(transform['name'], transform['amount'])]
        )

        results.append(chunk_result)

    return torch.cat(results, dim=0)
