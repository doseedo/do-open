#!/usr/bin/env python3
"""Neural parameter predictor for inverse synthesis.

MelParamPredictor: mel spectrogram → synth params + waveform ID.
~500k params, ~100ms inference on GPU.
"""

import torch
import torch.nn as nn
import numpy as np


# Parameter bounds matching fast_dsp.py
PARAM_NAMES = [
    'filter_base_hz', 'filter_peak_hz', 'resonance',
    'filter_attack', 'filter_decay', 'filter_sustain', 'filter_release', 'filter_noteoff',
    'amp_attack', 'amp_decay', 'amp_sustain', 'amp_release', 'amp_noteoff',
]

# Bounds for each parameter (min, max) — matching fast_dsp bounds
PARAM_BOUNDS = [
    (20, 20000),     # filter_base_hz
    (20, 20000),     # filter_peak_hz
    (0, 0.95),       # resonance
    (0.001, 2),      # filter_attack
    (0.001, 2),      # filter_decay
    (0, 1),          # filter_sustain
    (0.001, 2),      # filter_release
    (0.1, 3),        # filter_noteoff
    (0.001, 2),      # amp_attack
    (0.001, 2),      # amp_decay
    (0, 1),          # amp_sustain
    (0.001, 2),      # amp_release
    (0.1, 3),        # amp_noteoff
]

WAVEFORM_NAMES = ['saw', 'square', 'triangle', 'sine', 'pulse', 'noise', 'supersaw']
N_PARAMS = 13
N_WAVEFORMS = 7
N_MELS = 128


class MelParamPredictor(nn.Module):
    """Predict synth params from mel spectrogram.

    Architecture:
    - 3-layer CNN on mel spectrogram: [B, 1, 128, T] → [B, 256]
    - MLP param head: 256 + 1(pitch) → 128 → 13 (synth params)
    - MLP waveform head: 256 + 1(pitch) → 7 (waveform logits)
    - Sigmoid/softplus output activation per parameter
    """

    def __init__(self, n_mels=N_MELS, n_params=N_PARAMS, n_waveforms=N_WAVEFORMS):
        super().__init__()

        # CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 256, 1, 1]
        )

        # Param prediction head (256 + 1 pitch → 13 params)
        self.param_head = nn.Sequential(
            nn.Linear(257, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_params),
        )

        # Waveform classification head (256 + 1 pitch → 7 classes)
        self.wf_head = nn.Sequential(
            nn.Linear(257, 64),
            nn.ReLU(),
            nn.Linear(64, n_waveforms),
        )

        # Parameter bounds as buffers
        bounds_min = torch.tensor([b[0] for b in PARAM_BOUNDS], dtype=torch.float32)
        bounds_max = torch.tensor([b[1] for b in PARAM_BOUNDS], dtype=torch.float32)
        self.register_buffer('bounds_min', bounds_min)
        self.register_buffer('bounds_max', bounds_max)

        # Log-scale flags: filter_base_hz and filter_peak_hz use log scale
        self.log_params = [0, 1]

    def forward(self, mel, pitch_hz):
        """
        Args:
            mel: [B, 1, N_MELS, T] mel spectrogram (log-scale)
            pitch_hz: [B, 1] pitch in Hz

        Returns:
            params: [B, 13] predicted synth params (in original scale)
            wf_logits: [B, 7] waveform class logits
        """
        # CNN features
        features = self.cnn(mel)  # [B, 256, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 256]

        # Normalize pitch (log scale, centered)
        pitch_norm = torch.log2(pitch_hz / 440.0 + 1e-8)  # [B, 1]
        x = torch.cat([features, pitch_norm], dim=1)  # [B, 257]

        # Param prediction
        raw_params = self.param_head(x)  # [B, 13]

        # Apply bounds via sigmoid scaling
        params = torch.sigmoid(raw_params) * (self.bounds_max - self.bounds_min) + self.bounds_min

        # Log-scale parameters: use exp-sigmoid for better dynamic range
        for i in self.log_params:
            log_min = np.log10(float(self.bounds_min[i]))
            log_max = np.log10(float(self.bounds_max[i]))
            params[:, i] = 10 ** (torch.sigmoid(raw_params[:, i]) * (log_max - log_min) + log_min)

        # Waveform classification
        wf_logits = self.wf_head(x)  # [B, 7]

        return params, wf_logits


class ResBlock(nn.Module):
    """Residual block for CNN."""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class MelParamPredictorLarge(nn.Module):
    """Larger model (~2.5M params) with residual CNN + deeper MLP.

    Architecture:
    - 5-layer CNN with residual blocks: [B, 1, 128, T] → [B, 512]
    - MLP param head: 513 → 256 → 128 → 13 (synth params)
    - MLP waveform head: 513 → 128 → 7 (waveform logits)
    - Filter type head: 513 → 64 → 3 (filter type logits)
    """

    def __init__(self, n_mels=N_MELS, n_params=N_PARAMS, n_waveforms=N_WAVEFORMS):
        super().__init__()

        # CNN backbone with residual blocks
        self.cnn = nn.Sequential(
            # Block 1: 1 → 64
            nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            ResBlock(64),
            # Block 2: 64 → 128
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            ResBlock(128),
            # Block 3: 128 → 256
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            ResBlock(256),
            # Block 4: 256 → 512
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # [B, 512, 1, 1]
        )

        # Param prediction head (512 + 1 pitch → 256 → 128 → 13)
        self.param_head = nn.Sequential(
            nn.Linear(513, 256),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_params),
        )

        # Waveform classification head (512 + 1 pitch → 128 → 7)
        self.wf_head = nn.Sequential(
            nn.Linear(513, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, n_waveforms),
        )

        # Filter type classification head (512 + 1 pitch → 64 → 3)
        self.ft_head = nn.Sequential(
            nn.Linear(513, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # lowpass, highpass, bandpass
        )

        # Parameter bounds as buffers
        bounds_min = torch.tensor([b[0] for b in PARAM_BOUNDS], dtype=torch.float32)
        bounds_max = torch.tensor([b[1] for b in PARAM_BOUNDS], dtype=torch.float32)
        self.register_buffer('bounds_min', bounds_min)
        self.register_buffer('bounds_max', bounds_max)

        self.log_params = [0, 1]

    def forward(self, mel, pitch_hz):
        """
        Args:
            mel: [B, 1, N_MELS, T] mel spectrogram
            pitch_hz: [B, 1] pitch in Hz

        Returns:
            params: [B, 13] predicted synth params
            wf_logits: [B, 7] waveform class logits
        """
        features = self.cnn(mel)
        features = features.view(features.size(0), -1)  # [B, 512]

        pitch_norm = torch.log2(pitch_hz / 440.0 + 1e-8)  # [B, 1]
        x = torch.cat([features, pitch_norm], dim=1)  # [B, 513]

        raw_params = self.param_head(x)

        # Apply bounds via sigmoid scaling
        params = torch.sigmoid(raw_params) * (self.bounds_max - self.bounds_min) + self.bounds_min

        # Log-scale parameters
        for i in self.log_params:
            log_min = np.log10(float(self.bounds_min[i]))
            log_max = np.log10(float(self.bounds_max[i]))
            params[:, i] = 10 ** (torch.sigmoid(raw_params[:, i]) * (log_max - log_min) + log_min)

        wf_logits = self.wf_head(x)

        return params, wf_logits

    def predict_filter_type(self, mel, pitch_hz):
        """Predict filter type (lowpass/highpass/bandpass)."""
        features = self.cnn(mel)
        features = features.view(features.size(0), -1)
        pitch_norm = torch.log2(pitch_hz / 440.0 + 1e-8)
        x = torch.cat([features, pitch_norm], dim=1)
        return self.ft_head(x)


FILTER_TYPE_NAMES = ['lowpass', 'highpass', 'bandpass']


def compute_mel_spectrogram(audio, sr=44100, n_fft=2048, n_mels=128, hop_length=512):
    """Compute log-mel spectrogram from audio (numpy or torch).

    Returns [1, N_MELS, T] tensor.
    """
    import torchaudio

    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio).float()
    if audio.ndim == 1:
        audio = audio.unsqueeze(0)  # [1, N]

    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    mel = mel_transform(audio)  # [1, N_MELS, T]
    log_mel = torch.log1p(mel)  # log scale

    return log_mel


def predict_params(model, audio, pitch_hz, sr=44100, device='cuda'):
    """Run inference: audio → synth params + waveform.

    Args:
        model: trained MelParamPredictor
        audio: numpy array, mono audio
        pitch_hz: float, detected pitch
        sr: sample rate
        device: 'cuda' or 'cpu'

    Returns:
        dict with 'params' (list), 'waveform' (str), 'waveform_probs' (dict)
    """
    model.eval()
    model = model.to(device)

    # Compute mel spectrogram
    mel = compute_mel_spectrogram(audio, sr)
    mel = mel.unsqueeze(0).to(device)  # [1, 1, N_MELS, T]

    pitch = torch.tensor([[pitch_hz]], dtype=torch.float32, device=device)

    with torch.inference_mode():
        params, wf_logits = model(mel, pitch)

    params = params[0].cpu().numpy()
    wf_probs = torch.softmax(wf_logits[0], dim=0).cpu().numpy()

    # Convert to parameter dict
    param_dict = {}
    for i, name in enumerate(PARAM_NAMES):
        param_dict[name] = float(params[i])

    # Best waveform
    best_wf_idx = int(np.argmax(wf_probs))
    best_wf = WAVEFORM_NAMES[best_wf_idx]

    # All waveform probabilities
    wf_prob_dict = {WAVEFORM_NAMES[i]: float(wf_probs[i]) for i in range(N_WAVEFORMS)}

    return {
        'params': params.tolist(),
        'param_dict': param_dict,
        'waveform': best_wf,
        'waveform_idx': best_wf_idx,
        'waveform_probs': wf_prob_dict,
    }
