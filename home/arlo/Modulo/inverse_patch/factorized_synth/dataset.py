"""
Dataset generation for factorized synth prototype.
16 combinations: 4 cutoffs × 4 envelope attacks
"""

import numpy as np
import torch
from scipy import signal
from typing import List, Dict, Tuple

SAMPLE_RATE = 44100
DURATION = 2.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# Parameter grids
CUTOFFS = [500, 1000, 2000, 4000]  # Hz
ATTACKS = [0.01, 0.05, 0.1, 0.2]  # seconds


def generate_saw_with_params(cutoff_hz: float, attack_sec: float, pitch_hz: float = 220.0) -> np.ndarray:
    """Generate saw wave with specific filter cutoff and envelope attack."""
    t = np.linspace(0, DURATION, N_SAMPLES, endpoint=False)

    # Sawtooth oscillator
    saw = signal.sawtooth(2 * np.pi * pitch_hz * t).astype(np.float32)

    # Lowpass filter
    nyq = SAMPLE_RATE / 2
    norm_cutoff = min(cutoff_hz / nyq, 0.99)
    b, a = signal.butter(4, norm_cutoff, btype='low')
    filtered = signal.filtfilt(b, a, saw).astype(np.float32)

    # ADSR envelope with variable attack
    attack_samples = int(attack_sec * SAMPLE_RATE)
    decay_samples = int(0.1 * SAMPLE_RATE)
    release_samples = int(0.2 * SAMPLE_RATE)
    sustain_level = 0.7

    envelope = np.ones(N_SAMPLES, dtype=np.float32)

    # Attack
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)

    # Decay
    decay_end = attack_samples + decay_samples
    if decay_samples > 0 and decay_end < N_SAMPLES:
        envelope[attack_samples:decay_end] = np.linspace(1, sustain_level, decay_samples)

    # Sustain
    sustain_end = N_SAMPLES - release_samples
    if sustain_end > decay_end:
        envelope[decay_end:sustain_end] = sustain_level

    # Release
    if release_samples > 0:
        envelope[sustain_end:] = np.linspace(sustain_level, 0, N_SAMPLES - sustain_end)

    audio = filtered * envelope * 0.8
    return audio.astype(np.float32)


def compute_mel_spectrogram(audio: np.ndarray, n_mels: int = 64, n_fft: int = 1024, hop_length: int = 256) -> np.ndarray:
    """Compute mel spectrogram for the audio."""
    import torch
    import torchaudio.transforms as T

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
    )

    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    mel = mel_transform(audio_tensor)

    # Log scale
    mel = torch.log(mel + 1e-8)

    return mel.squeeze(0).numpy()


def generate_dataset() -> List[Dict]:
    """Generate full dataset of 16 combinations."""
    dataset = []

    for cutoff in CUTOFFS:
        for attack in ATTACKS:
            audio = generate_saw_with_params(cutoff, attack)
            mel = compute_mel_spectrogram(audio)

            # Normalize parameters to [0, 1]
            cutoff_norm = (CUTOFFS.index(cutoff)) / (len(CUTOFFS) - 1)
            attack_norm = (ATTACKS.index(attack)) / (len(ATTACKS) - 1)

            # Class labels for disentanglement loss
            cutoff_label = CUTOFFS.index(cutoff)
            attack_label = ATTACKS.index(attack)

            dataset.append({
                'audio': audio,
                'mel': mel,
                'cutoff_hz': cutoff,
                'attack_sec': attack,
                'cutoff_norm': cutoff_norm,
                'attack_norm': attack_norm,
                'cutoff_label': cutoff_label,
                'attack_label': attack_label,
            })

    return dataset


class SynthDataset(torch.utils.data.Dataset):
    """PyTorch dataset wrapper."""

    def __init__(self, data: List[Dict] = None):
        if data is None:
            data = generate_dataset()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'mel': torch.from_numpy(item['mel']).float(),
            'audio': torch.from_numpy(item['audio']).float(),
            'cutoff_norm': torch.tensor([item['cutoff_norm']]).float(),
            'attack_norm': torch.tensor([item['attack_norm']]).float(),
            'cutoff_label': torch.tensor(item['cutoff_label']).long(),
            'attack_label': torch.tensor(item['attack_label']).long(),
        }


if __name__ == "__main__":
    print("Generating dataset...")
    dataset = generate_dataset()
    print(f"Generated {len(dataset)} samples")

    for i, d in enumerate(dataset):
        print(f"  {i}: cutoff={d['cutoff_hz']}Hz, attack={d['attack_sec']}s, mel shape={d['mel'].shape}")
