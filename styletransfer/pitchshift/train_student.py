#!/usr/bin/env python3
"""
Train Student Model for Pitch Shift

Step 4 of the pipeline: Train a lightweight student model on the synthetic
pitch-shifted data for VST deployment.

The student learns from:
- (original_mel, shift_amount) -> corrected_mel

This creates a fast model that operates directly on mel spectrograms,
suitable for real-time use in a VST plugin.

Usage:
    python train_student.py \
        --manifest /path/to/synthetic_manifest.json \
        --output_dir /path/to/student_checkpoints
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data')

from models_pitch_shift import PitchShiftStudentModel


class MelSpectrogramTransform:
    """Convert between audio and mel spectrogram."""

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

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )

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
        mel = self.mel_transform(audio)
        log_mel = torch.log(mel + 1e-8)
        return log_mel

    def mel_to_audio(self, log_mel: torch.Tensor) -> torch.Tensor:
        """Convert log mel spectrogram back to audio."""
        mel = torch.exp(log_mel)
        spec = self.inverse_mel(mel)
        audio = self.griffin_lim(spec)
        return audio


class StudentDataset(Dataset):
    """
    Dataset for student model training.

    Loads pairs of (original_audio, shifted_audio) and converts to mel spectrograms.
    """

    def __init__(
        self,
        manifest_path: str,
        mel_frames: int = 256,
        sample_rate: int = 44100,
    ):
        self.mel_frames = mel_frames
        self.sample_rate = sample_rate
        self.mel_transform = MelSpectrogramTransform(sample_rate=sample_rate)

        # Load manifest
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        self.entries = []
        for entry in manifest.get('files', []):
            original_path = entry.get('original_audio', '')
            shifted_path = entry.get('shifted_audio', '')

            if os.path.exists(original_path) and os.path.exists(shifted_path):
                self.entries.append({
                    'original_path': original_path,
                    'shifted_path': shifted_path,
                    'shift': entry.get('shift_semitones', 0),
                    'target_pitch': entry.get('target_pitch', 60),
                })

        print(f"Loaded {len(self.entries)} entries")

    def __len__(self) -> int:
        return len(self.entries)

    def _load_and_process(self, audio_path: str) -> torch.Tensor:
        """Load audio and convert to mel spectrogram."""
        audio, sr = torchaudio.load(audio_path)

        # Resample if needed
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Convert to mono
        mono = audio.mean(dim=0, keepdim=True)

        # Convert to mel
        mel = self.mel_transform.audio_to_mel(mono)  # [1, n_mels, T]

        return mel.squeeze(0)  # [n_mels, T]

    def _random_window(self, mel: torch.Tensor) -> torch.Tensor:
        """Extract random window from mel spectrogram."""
        n_mels, T = mel.shape

        if T <= self.mel_frames:
            # Pad
            pad_amount = self.mel_frames - T
            mel = F.pad(mel, (0, pad_amount))
            return mel

        start = random.randint(0, T - self.mel_frames)
        return mel[:, start:start + self.mel_frames]

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        try:
            # Load both audio files
            original_mel = self._load_and_process(entry['original_path'])
            shifted_mel = self._load_and_process(entry['shifted_path'])

            # Get aligned windows (same start position)
            n_mels, T = original_mel.shape
            T_shifted = shifted_mel.shape[1]
            T_min = min(T, T_shifted)

            if T_min <= self.mel_frames:
                # Pad both
                original_mel = F.pad(original_mel, (0, self.mel_frames - T_min))
                shifted_mel = F.pad(shifted_mel, (0, self.mel_frames - T_min))
            else:
                start = random.randint(0, T_min - self.mel_frames)
                original_mel = original_mel[:, start:start + self.mel_frames]
                shifted_mel = shifted_mel[:, start:start + self.mel_frames]

            return {
                'input_mel': original_mel.unsqueeze(0),  # [1, n_mels, frames]
                'target_mel': shifted_mel.unsqueeze(0),
                'shift_amount': torch.tensor(float(entry['shift'])),
                'target_pitch': torch.tensor(entry['target_pitch']),
                'valid': torch.tensor(True),
            }

        except Exception as e:
            print(f"Error loading {entry['original_path']}: {e}")
            # Return zeros
            zeros = torch.zeros(1, 128, self.mel_frames)
            return {
                'input_mel': zeros,
                'target_mel': zeros,
                'shift_amount': torch.tensor(0.0),
                'target_pitch': torch.tensor(60),
                'valid': torch.tensor(False),
            }


class StudentTrainer:
    """Trainer for the lightweight student model."""

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        num_epochs: int = 50,
        device: str = 'cuda',
        num_workers: int = 4,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs

        # Dataset
        print("Creating dataset...")
        self.dataset = StudentDataset(manifest_path)

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        # Model
        print("Creating student model...")
        self.model = PitchShiftStudentModel()
        self.model = self.model.to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.dataloader),
        )

        # Loss
        self.criterion = nn.L1Loss()

        # Tracking
        self.best_loss = float('inf')

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            valid = batch['valid']
            if not valid.any():
                continue

            input_mel = batch['input_mel'].to(self.device)
            target_mel = batch['target_mel'].to(self.device)
            shift_amount = batch['shift_amount'].to(self.device)
            target_pitch = batch['target_pitch'].to(self.device)

            # Forward
            output_mel = self.model(input_mel, target_pitch, shift_amount)

            # Loss
            loss = self.criterion(output_mel, target_mel)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }

        if epoch % 10 == 0:
            path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(checkpoint, path)

        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def train(self):
        """Run training."""
        print(f"\nTraining on {self.device}")
        print(f"Output: {self.output_dir}")

        for epoch in range(1, self.num_epochs + 1):
            loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: loss={loss:.4f}")

            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss

            self.save_checkpoint(epoch, loss, is_best)

        print("\nTraining complete!")
        print(f"Best loss: {self.best_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Student Model for Pitch Shift")

    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to synthetic data manifest')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    trainer = StudentTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        num_workers=args.num_workers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
