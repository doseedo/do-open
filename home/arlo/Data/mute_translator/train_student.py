#!/usr/bin/env python3
"""
Train Student Model

Step 4: Train a lightweight student model on paired (dry, synthetic_muted) data.

The student learns direct audio-to-audio transformation without needing
the heavy DCAE at inference time.

Architecture options:
1. Mel-spectrogram domain (lighter, good for VST)
2. Waveform domain (heavier, higher quality)
3. Latent domain (requires DCAE, but fast inference)

Usage:
    python train_student.py \
        --synthetic_manifest /path/to/synthetic_manifest.json \
        --output_dir ./student_checkpoints
"""

import os
import sys
import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

# Suppress torchaudio deprecation warnings
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder has been deprecated.*")
warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data')


class MelSpectrogramTransform:
    """Convert audio to/from mel spectrogram."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

        # For inversion (Griffin-Lim or learned vocoder)
        self.inverse_mel = torchaudio.transforms.InverseMelScale(
            n_stft=n_fft // 2 + 1,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )

        self.griffin_lim = torchaudio.transforms.GriffinLim(
            n_fft=n_fft,
            hop_length=hop_length,
        )

    def audio_to_mel(self, audio: torch.Tensor) -> torch.Tensor:
        """Convert audio to log mel spectrogram."""
        mel = self.mel_spec(audio)
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def mel_to_audio(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Convert log mel back to audio (approximate)."""
        mel = torch.exp(log_mel)
        spec = self.inverse_mel(mel)
        audio = self.griffin_lim(spec)
        return audio


class StudentResBlock(nn.Module):
    """Residual block for student model."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return self.act(x + residual)


class MelStudentModel(nn.Module):
    """
    Lightweight student model operating on mel spectrograms.

    Input: Log mel spectrogram of dry trumpet [B, 1, n_mels, T]
    Output: Log mel spectrogram of muted trumpet [B, 1, n_mels, T]
    """

    def __init__(
        self,
        n_mels: int = 128,
        hidden_channels: int = 64,
        num_blocks: int = 8,
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Conv2d(1, hidden_channels, 3, padding=1)

        # Residual blocks
        self.blocks = nn.ModuleList([
            StudentResBlock(hidden_channels, kernel_size=3, dilation=2**(i % 4))
            for i in range(num_blocks)
        ])

        # Output projection (residual)
        self.output_proj = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(hidden_channels, 1, 3, padding=1),
        )

        # Zero init for residual learning
        nn.init.zeros_(self.output_proj[-1].weight)
        nn.init.zeros_(self.output_proj[-1].bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dry_mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dry_mel: [B, 1, n_mels, T] dry trumpet log mel

        Returns:
            muted_mel: [B, 1, n_mels, T] predicted muted log mel
        """
        x = self.input_proj(dry_mel)

        for block in self.blocks:
            x = block(x)

        residual = self.output_proj(x)
        return dry_mel + self.residual_scale * residual


class WaveformStudentModel(nn.Module):
    """
    Student model operating directly on waveforms.

    Uses 1D convolutions with large receptive field.
    Higher quality but more expensive.
    """

    def __init__(
        self,
        hidden_channels: int = 64,
        num_blocks: int = 12,
        kernel_size: int = 7,
    ):
        super().__init__()

        self.input_proj = nn.Conv1d(2, hidden_channels, kernel_size, padding=kernel_size//2)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 5)
            padding = (kernel_size - 1) * dilation // 2
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                          padding=padding, dilation=dilation),
                nn.GroupNorm(8, hidden_channels),
                nn.SiLU(),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                          padding=padding, dilation=dilation),
                nn.GroupNorm(8, hidden_channels),
            ))

        self.output_proj = nn.Conv1d(hidden_channels, 2, kernel_size, padding=kernel_size//2)

        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, dry_audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dry_audio: [B, 2, T] dry trumpet audio

        Returns:
            muted_audio: [B, 2, T] predicted muted audio
        """
        x = self.input_proj(dry_audio)

        for block in self.blocks:
            x = x + block(x)

        residual = self.output_proj(x)
        return dry_audio + self.residual_scale * residual


class PairedAudioDataset(Dataset):
    """Dataset of paired (dry, muted) audio files."""

    def __init__(
        self,
        manifest_path: str,
        sample_rate: int = 44100,
        segment_length: int = 44100 * 3,  # 3 seconds
        use_mel: bool = True,
        n_mels: int = 128,
    ):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.use_mel = use_mel

        if use_mel:
            self.mel_transform = MelSpectrogramTransform(
                sample_rate=sample_rate, n_mels=n_mels
            )

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        # Support both formats:
        # 1. mixed_results.json format: {"samples": [{"dry_decoded": ..., "mixed_path": ...}]}
        # 2. Original format: {"files": [{"dry_audio": ..., "muted_audio": ...}]}
        if 'samples' in manifest:
            raw_pairs = manifest['samples']
            # Convert to standard format
            self.pairs = []
            for p in raw_pairs:
                if p.get('status') == 'success' and p.get('mixed_path'):
                    self.pairs.append({
                        'dry_audio': p['dry_decoded'],
                        'muted_audio': p['mixed_path'],
                        'basename': p.get('basename', '')
                    })
            print(f"Loaded {len(self.pairs)} paired samples from mixed_results format")
        else:
            self.pairs = manifest['files']
            print(f"Loaded {len(self.pairs)} paired samples")

        # Filter to existing
        self.pairs = [
            p for p in self.pairs
            if os.path.exists(p['dry_audio']) and os.path.exists(p['muted_audio'])
        ]
        print(f"Valid pairs: {len(self.pairs)}")

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_audio(self, path: str) -> torch.Tensor:
        """Load and preprocess audio."""
        audio, sr = torchaudio.load(path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Ensure stereo
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] > 2:
            audio = audio[:2]

        return audio

    def _random_segment(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract random segment."""
        if audio.shape[-1] <= self.segment_length:
            # Pad if too short
            pad = self.segment_length - audio.shape[-1]
            audio = F.pad(audio, (0, pad))
            return audio

        start = torch.randint(0, audio.shape[-1] - self.segment_length, (1,)).item()
        return audio[:, start:start + self.segment_length]

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]

        dry_audio = self._load_audio(pair['dry_audio'])
        muted_audio = self._load_audio(pair['muted_audio'])

        # Random segment (same position for both)
        min_len = min(dry_audio.shape[-1], muted_audio.shape[-1])
        if min_len > self.segment_length:
            start = torch.randint(0, min_len - self.segment_length, (1,)).item()
            dry_audio = dry_audio[:, start:start + self.segment_length]
            muted_audio = muted_audio[:, start:start + self.segment_length]
        else:
            dry_audio = self._random_segment(dry_audio)
            muted_audio = self._random_segment(muted_audio)

        if self.use_mel:
            # Convert to mel
            dry_mel = self.mel_transform.audio_to_mel(dry_audio.mean(dim=0, keepdim=True))
            muted_mel = self.mel_transform.audio_to_mel(muted_audio.mean(dim=0, keepdim=True))
            return {
                'dry': dry_mel,
                'muted': muted_mel,
            }
        else:
            return {
                'dry': dry_audio,
                'muted': muted_audio,
            }


class StudentTrainer:
    """Trainer for student model."""

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        model_type: str = "mel",  # "mel" or "waveform"
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = "cuda",
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.num_epochs = num_epochs

        print(f"Device: {self.device}")

        # Create model
        use_mel = model_type == "mel"
        if use_mel:
            self.model = MelStudentModel().to(self.device)
        else:
            self.model = WaveformStudentModel().to(self.device)

        print(f"Model type: {model_type}")
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Dataset
        self.dataset = PairedAudioDataset(
            manifest_path=manifest_path,
            use_mel=use_mel,
        )

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        # Loss
        self.criterion = nn.L1Loss()

        self.best_loss = float('inf')
        self.log_file = self.output_dir / "training.log"

    def train_step(self, batch: dict) -> float:
        """Single training step."""
        dry = batch['dry'].to(self.device)
        muted = batch['muted'].to(self.device)

        self.model.train()
        pred = self.model(dry)

        loss = self.criterion(pred, muted)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for batch in pbar:
            loss = self.train_step(batch)
            total_loss += loss
            n_batches += 1
            pbar.set_postfix({'loss': f"{loss:.4f}"})

        self.scheduler.step()
        return total_loss / max(n_batches, 1)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        torch.save(checkpoint, self.output_dir / "latest.pt")
        if is_best:
            torch.save(checkpoint, self.output_dir / "best.pt")
        if (epoch + 1) % 20 == 0:
            torch.save(checkpoint, self.output_dir / f"epoch_{epoch+1}.pt")

    def log(self, message: str):
        """Log message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}"
        print(log_msg)
        with open(self.log_file, 'a') as f:
            f.write(log_msg + "\n")

    def train(self):
        """Full training loop."""
        self.log("=" * 60)
        self.log("STUDENT MODEL TRAINING")
        self.log("=" * 60)
        self.log(f"Output: {self.output_dir}")
        self.log(f"Training pairs: {len(self.dataset)}")
        self.log("")

        for epoch in range(self.num_epochs):
            loss = self.train_epoch(epoch)

            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss

            self.save_checkpoint(epoch, loss, is_best)

            self.log(
                f"Epoch {epoch+1:3d} | "
                f"Loss: {loss:.4f} | "
                f"LR: {self.scheduler.get_last_lr()[0]:.2e}"
                + (" [BEST]" if is_best else "")
            )

        self.log("\nTraining complete!")
        self.log(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Student Model")
    parser.add_argument('--manifest', '--synthetic_manifest', type=str, required=True,
                        dest='manifest',
                        help='Path to manifest (mixed_results.json or synthetic_manifest.json)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--model_type', type=str, default='mel',
                        choices=['mel', 'waveform'],
                        help='Model type')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    trainer = StudentTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
