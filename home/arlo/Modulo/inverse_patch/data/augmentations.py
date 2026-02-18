"""
Audio augmentations for training data.
"""

import torch
import torch.nn as nn
import torchaudio
import random
from typing import Optional


class AudioAugmentations(nn.Module):
    """
    Audio augmentation pipeline for training.
    Includes various transformations to improve model robustness.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        # Gain augmentation
        gain_augment: bool = True,
        gain_range_db: tuple = (-6.0, 6.0),
        # Noise augmentation
        noise_augment: bool = True,
        noise_snr_range: tuple = (30.0, 60.0),  # SNR in dB
        # Time stretching (pitch-preserving)
        time_stretch: bool = False,
        stretch_range: tuple = (0.9, 1.1),
        # Pitch shifting
        pitch_shift: bool = False,
        pitch_range_semitones: tuple = (-2, 2),
        # High/low pass filtering
        filter_augment: bool = True,
        highpass_range: tuple = (20, 100),  # Hz
        lowpass_range: tuple = (16000, 20000),  # Hz
        # Probability of applying each augmentation
        prob: float = 0.5,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.gain_augment = gain_augment
        self.gain_range_db = gain_range_db
        self.noise_augment = noise_augment
        self.noise_snr_range = noise_snr_range
        self.time_stretch = time_stretch
        self.stretch_range = stretch_range
        self.pitch_shift = pitch_shift
        self.pitch_range_semitones = pitch_range_semitones
        self.filter_augment = filter_augment
        self.highpass_range = highpass_range
        self.lowpass_range = lowpass_range
        self.prob = prob

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to audio.

        Args:
            audio: Input audio tensor [C, T] or [B, C, T]

        Returns:
            Augmented audio tensor
        """
        # Handle batch dimension
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        # Apply augmentations
        if self.gain_augment and random.random() < self.prob:
            audio = self._apply_gain(audio)

        if self.noise_augment and random.random() < self.prob:
            audio = self._add_noise(audio)

        if self.filter_augment and random.random() < self.prob:
            audio = self._apply_filter(audio)

        if squeeze:
            audio = audio.squeeze(0)

        return audio

    def _apply_gain(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random gain change."""
        gain_db = random.uniform(*self.gain_range_db)
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear

    def _add_noise(self, audio: torch.Tensor) -> torch.Tensor:
        """Add random noise at specified SNR."""
        snr_db = random.uniform(*self.noise_snr_range)

        # Calculate signal power
        signal_power = audio.pow(2).mean()

        # Calculate noise power for desired SNR
        noise_power = signal_power / (10 ** (snr_db / 10))

        # Generate noise
        noise = torch.randn_like(audio) * torch.sqrt(noise_power)

        return audio + noise

    def _apply_filter(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply random high/low pass filtering."""
        bs, chs, length = audio.size()

        # Random highpass
        if random.random() < 0.5:
            cutoff = random.uniform(*self.highpass_range)
            # Simple first-order highpass using differentiation approximation
            # More sophisticated filtering would use torchaudio.functional
            alpha = 2 * 3.14159 * cutoff / self.sample_rate
            alpha = min(alpha, 0.5)
            audio = audio - alpha * torch.nn.functional.avg_pool1d(
                audio, kernel_size=int(self.sample_rate / cutoff) or 1,
                stride=1,
                padding=int(self.sample_rate / cutoff) // 2 or 0,
            )[..., :length]

        # Random lowpass
        if random.random() < 0.5:
            cutoff = random.uniform(*self.lowpass_range)
            kernel_size = max(3, int(self.sample_rate / cutoff))
            if kernel_size % 2 == 0:
                kernel_size += 1
            audio = torch.nn.functional.avg_pool1d(
                audio,
                kernel_size=kernel_size,
                stride=1,
                padding=kernel_size // 2,
            )[..., :length]

        return audio


class SpecAugment(nn.Module):
    """
    SpecAugment-style augmentation for spectrograms.
    Useful for encoder training.
    """

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to spectrogram.

        Args:
            spec: Spectrogram [B, F, T] or [B, C, F, T]

        Returns:
            Augmented spectrogram
        """
        if spec.dim() == 3:
            spec = spec.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False

        bs, chs, n_freq, n_time = spec.size()

        # Frequency masking
        for _ in range(self.n_freq_masks):
            f = random.randint(0, self.freq_mask_param)
            f0 = random.randint(0, n_freq - f)
            spec[:, :, f0:f0+f, :] = 0

        # Time masking
        for _ in range(self.n_time_masks):
            t = random.randint(0, min(self.time_mask_param, n_time // 4))
            t0 = random.randint(0, n_time - t)
            spec[:, :, :, t0:t0+t] = 0

        if squeeze:
            spec = spec.squeeze(1)

        return spec


class MixUp(nn.Module):
    """
    MixUp augmentation for audio.
    Linearly interpolates between two samples.
    """

    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        audio1: torch.Tensor,
        audio2: torch.Tensor,
        labels1: Optional[torch.Tensor] = None,
        labels2: Optional[torch.Tensor] = None,
    ):
        """
        Apply MixUp between two samples.

        Args:
            audio1, audio2: Audio tensors [B, C, T]
            labels1, labels2: Optional label tensors

        Returns:
            Mixed audio and optionally mixed labels
        """
        # Sample mixing coefficient from Beta distribution
        if self.alpha > 0:
            lam = torch.distributions.Beta(self.alpha, self.alpha).sample()
        else:
            lam = 0.5

        # Mix audio
        mixed_audio = lam * audio1 + (1 - lam) * audio2

        if labels1 is not None and labels2 is not None:
            mixed_labels = lam * labels1 + (1 - lam) * labels2
            return mixed_audio, mixed_labels, lam

        return mixed_audio, lam
