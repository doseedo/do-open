"""
Numpy-based transform library for CPU multiprocessing.

This is a lightweight version of tensor_transforms.py using numpy
instead of PyTorch for better multiprocessing compatibility.
"""

import numpy as np
from typing import Tuple


class NumpyTransformLibrary:
    """Numpy implementation of MIDI transforms for CPU multiprocessing."""

    def apply_transform(self, batch: np.ndarray, transform_name: str, amount: float) -> np.ndarray:
        """
        Apply a single transform to batch.

        Args:
            batch: (B, T, F) numpy array
            transform_name: Name of transform
            amount: Transform amount/parameter

        Returns:
            transformed: (B, T, F) numpy array
        """
        if transform_name == 'transpose_semitone':
            return self.transpose_semitone(batch, int(amount))
        elif transform_name == 'inversion':
            return self.inversion(batch, center=int(amount) if amount != 0 else None)
        elif transform_name == 'velocity_scale':
            return self.velocity_scale(batch, amount)
        elif transform_name == 'time_scale':
            return self.time_scale(batch, amount)
        elif transform_name == 'time_shift':
            return self.time_shift(batch, int(amount))
        elif transform_name == 'retrograde':
            return self.retrograde(batch)
        elif transform_name == 'instrument_filter':
            return self.instrument_filter(batch, int(amount * 127))
        elif transform_name == 'quantize_16th':
            return self.quantize_rhythm(batch, resolution=16)
        elif transform_name == 'quantize_8th':
            return self.quantize_rhythm(batch, resolution=8)
        elif transform_name == 'track_derive':
            # Extract parameters from amount if needed
            return batch  # Placeholder
        else:
            return batch

    @staticmethod
    def transpose_semitone(batch: np.ndarray, semitones: int) -> np.ndarray:
        """Transpose pitch by semitones."""
        result = batch.copy()
        if semitones == 0:
            return result

        # Pitch is in features 0-127
        pitch = result[:, :, :128]

        # Roll pitch representation
        if semitones > 0:
            result[:, :, semitones:128] = pitch[:, :, :128-semitones]
            result[:, :, :semitones] = 0
        else:
            result[:, :, :128+semitones] = pitch[:, :, -semitones:]
            result[:, :, 128+semitones:128] = 0

        return result

    @staticmethod
    def inversion(batch: np.ndarray, center: int = None) -> np.ndarray:
        """Invert pitches around a center note."""
        result = batch.copy()

        # Find active pitches
        pitch = result[:, :, :128]

        if center is None:
            # Auto-detect center from mean pitch
            pitch_indices = np.arange(128)
            mean_pitch = np.sum(pitch * pitch_indices[None, None, :], axis=2) / (np.sum(pitch, axis=2) + 1e-8)
            center = int(np.mean(mean_pitch))

        # Invert: new_pitch = 2*center - old_pitch
        inverted = np.zeros_like(pitch)
        for p in range(128):
            if np.any(pitch[:, :, p] > 0):
                new_p = 2 * center - p
                if 0 <= new_p < 128:
                    inverted[:, :, new_p] = pitch[:, :, p]

        result[:, :, :128] = inverted
        return result

    @staticmethod
    def velocity_scale(batch: np.ndarray, scale: float) -> np.ndarray:
        """Scale velocity (feature 128)."""
        result = batch.copy()
        result[:, :, 128] = np.clip(result[:, :, 128] * scale, 0.0, 1.0)
        return result

    @staticmethod
    def time_scale(batch: np.ndarray, factor: float) -> np.ndarray:
        """Scale time dimension."""
        B, T, F = batch.shape
        new_T = int(T * factor)
        new_T = min(max(new_T, 1), T)

        if new_T == T:
            return batch

        # Simple resampling
        result = np.zeros((B, T, F))
        indices = np.linspace(0, new_T - 1, T).astype(int)
        result[:, :, :] = batch[:, indices, :]

        return result

    @staticmethod
    def time_shift(batch: np.ndarray, steps: int) -> np.ndarray:
        """Shift time by steps."""
        result = batch.copy()
        if steps == 0:
            return result

        if steps > 0:
            result[:, steps:, :] = batch[:, :-steps, :]
            result[:, :steps, :] = 0
        else:
            result[:, :steps, :] = batch[:, -steps:, :]
            result[:, steps:, :] = 0

        return result

    @staticmethod
    def retrograde(batch: np.ndarray) -> np.ndarray:
        """Reverse time dimension."""
        return batch[:, ::-1, :].copy()

    @staticmethod
    def instrument_filter(batch: np.ndarray, program: int) -> np.ndarray:
        """Filter to specific instrument (program number 0-127)."""
        result = batch.copy()

        # Program is in feature 129
        program_norm = program / 127.0
        mask = np.abs(result[:, :, 129] - program_norm) < 0.004

        # Zero out notes from other instruments
        result[:, :, :128] *= mask[:, :, None]

        return result

    @staticmethod
    def quantize_rhythm(batch: np.ndarray, resolution: int = 16) -> np.ndarray:
        """Quantize rhythm to 16th or 8th notes."""
        result = batch.copy()
        T = batch.shape[1]

        # Quantize time steps to grid
        step_size = T // resolution
        if step_size < 1:
            return result

        for i in range(resolution):
            start = i * step_size
            end = (i + 1) * step_size if i < resolution - 1 else T

            # Collapse this window to single timestep
            if end > start:
                collapsed = np.max(result[:, start:end, :], axis=1, keepdims=True)
                result[:, start:end, :] = 0
                result[:, start:start+1, :] = collapsed

        return result
