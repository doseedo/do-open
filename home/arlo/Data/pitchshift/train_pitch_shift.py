#!/usr/bin/env python3
"""
Train Register-Aware Pitch Shift Translator

Similar structure to mute translator training but with pitch conditioning.
The model learns to:
1. Remove pitch-shift artifacts (via double-shift training pairs)
2. Apply correct register-specific timbre (via pitch embeddings)

Usage:
    python train_pitch_shift.py \
        --manifest /home/arlo/Data.backup/final_training_manifest_final.json \
        --output_dir /path/to/checkpoints \
        --instrument trumpet \
        --batch_size 16 \
        --num_epochs 100
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm

# Add paths
sys.path.insert(0, '/home/arlo/Data')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')
sys.path.insert(0, '/home/arlo/Data/do')

from dataset_pitch_shift import PitchShiftDataset, PitchShiftDatasetWithRealDegradation
from models_pitch_shift import (
    RegisterAwareTranslator,
    RegisterAwareTranslatorDirect,
    RegisterAwareTranslatorLarge,
    CombinedLoss,
)


class PitchShiftTrainer:
    """
    Trainer for Register-Aware Pitch Shift Translator.

    Handles:
    - Dataset loading with pitch indexing
    - Model training with combined loss
    - Checkpoint saving and resumption
    - Logging and progress tracking
    """

    def __init__(
        self,
        manifest_path: str,
        output_dir: str,
        instrument: str = 'trumpet',
        model_type: str = 'residual',
        batch_size: int = 16,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        device: str = 'cuda',
        window_frames: int = 128,
        samples_per_epoch: int = 10000,
        shift_range: tuple = (-12, 12),
        artifact_removal_ratio: float = 0.5,
        reconstruction_weight: float = 1.0,
        timbre_weight: float = 0.5,
        content_weight: float = 0.3,
        num_workers: int = 4,
        checkpoint_every: int = 10,
        resume_from: str = None,
        use_real_degradation: bool = False,
        dcae_checkpoint: str = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_epochs = num_epochs
        self.checkpoint_every = checkpoint_every
        self.model_type = model_type

        # Save config
        self.config = {
            'manifest_path': manifest_path,
            'instrument': instrument,
            'model_type': model_type,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'num_epochs': num_epochs,
            'window_frames': window_frames,
            'samples_per_epoch': samples_per_epoch,
            'shift_range': shift_range,
            'artifact_removal_ratio': artifact_removal_ratio,
            'reconstruction_weight': reconstruction_weight,
            'timbre_weight': timbre_weight,
            'content_weight': content_weight,
        }

        config_path = self.output_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        # Create dataset
        print("Creating dataset...")
        if use_real_degradation and dcae_checkpoint:
            # Load DCAE for real degradation
            print(f"Loading DCAE from {dcae_checkpoint}...")
            self.dcae = self._load_dcae(dcae_checkpoint)
            if self.dcae is not None:
                self.dataset = PitchShiftDatasetWithRealDegradation(
                    manifest_path=manifest_path,
                    dcae_model=self.dcae,
                    device=device,
                    instrument=instrument,
                    window_frames=window_frames,
                    samples_per_epoch=samples_per_epoch,
                    shift_range=shift_range,
                    artifact_removal_ratio=artifact_removal_ratio,
                )
            else:
                print("Warning: DCAE load failed, falling back to simulated degradation")
                self.dataset = PitchShiftDataset(
                    manifest_path=manifest_path,
                    instrument=instrument,
                    window_frames=window_frames,
                    samples_per_epoch=samples_per_epoch,
                    shift_range=shift_range,
                    artifact_removal_ratio=artifact_removal_ratio,
                )
        else:
            self.dcae = None
            self.dataset = PitchShiftDataset(
                manifest_path=manifest_path,
                instrument=instrument,
                window_frames=window_frames,
                samples_per_epoch=samples_per_epoch,
                shift_range=shift_range,
                artifact_removal_ratio=artifact_removal_ratio,
            )

        # Use 0 workers for real degradation (DCAE uses GPU in main process)
        actual_workers = 0 if use_real_degradation and self.dcae is not None else num_workers

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=actual_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=actual_workers > 0,
        )

        # Create model
        print(f"Creating model (type={model_type})...")
        if model_type == 'direct':
            self.model = RegisterAwareTranslatorDirect()
        elif model_type == 'large':
            self.model = RegisterAwareTranslatorLarge()
        else:
            self.model = RegisterAwareTranslator()

        self.model = self.model.to(self.device)
        print(f"Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # Create loss function
        self.criterion = CombinedLoss(
            reconstruction_weight=reconstruction_weight,
            timbre_weight=timbre_weight,
            content_weight=content_weight,
        )

        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )

        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.dataloader),
            pct_start=0.1,
        )

        # Training state
        self.start_epoch = 1
        self.best_loss = float('inf')
        self.train_losses = []

        # Resume if checkpoint provided
        if resume_from:
            self._load_checkpoint(resume_from)

    def _load_dcae(self, checkpoint_dir: str):
        """Load DCAE for real degradation."""
        try:
            import sys
            import os
            import glob

            # Add ACE-Step to path for imports
            ace_step_path = '/home/arlo/Data/ACE-Step'
            if ace_step_path not in sys.path:
                sys.path.insert(0, ace_step_path)

            from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

            # Find the actual checkpoint paths - need both dcae and vocoder
            def find_subdir(base, name):
                if os.path.exists(os.path.join(base, name)):
                    return os.path.join(base, name)
                matches = glob.glob(os.path.join(base, f'**/{name}'), recursive=True)
                return matches[0] if matches else None

            dcae_path = find_subdir(checkpoint_dir, 'music_dcae_f8c8')
            vocoder_path = find_subdir(checkpoint_dir, 'music_vocoder')

            if not dcae_path:
                raise FileNotFoundError(f"Could not find music_dcae_f8c8 in {checkpoint_dir}")
            if not vocoder_path:
                raise FileNotFoundError(f"Could not find music_vocoder in {checkpoint_dir}")

            print(f"DCAE path: {dcae_path}")
            print(f"Vocoder path: {vocoder_path}")

            dcae = MusicDCAE(
                dcae_checkpoint_path=dcae_path,
                vocoder_checkpoint_path=vocoder_path
            )
            dcae = dcae.to(self.device).eval()
            print(f"DCAE loaded successfully")
            return dcae

        except Exception as e:
            print(f"Warning: Could not load DCAE: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _encode_fn(self, audio: torch.Tensor, sr: int = 44100) -> torch.Tensor:
        """Encode audio to latent using DCAE."""
        if self.dcae is None:
            raise RuntimeError("DCAE not loaded")
        with torch.no_grad():
            audio = audio.to(self.device)
            return self.dcae.encode(audio, sr=sr)

    def _decode_fn(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to audio using DCAE."""
        if self.dcae is None:
            raise RuntimeError("DCAE not loaded")
        with torch.no_grad():
            latent = latent.to(self.device)
            return self.dcae.decode(latent)

    def _load_checkpoint(self, checkpoint_path: str):
        """Load training state from checkpoint."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.train_losses = checkpoint.get('train_losses', [])

        print(f"Resumed from epoch {self.start_epoch - 1}, best_loss={self.best_loss:.4f}")

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'config': self.config,
        }

        # Regular checkpoint
        if epoch % self.checkpoint_every == 0:
            path = self.output_dir / f"checkpoint_epoch{epoch}.pt"
            torch.save(checkpoint, path)
            print(f"Saved checkpoint: {path}")

        # Best model
        if is_best:
            best_path = self.output_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")

        # Latest (always save)
        latest_path = self.output_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch."""
        self.model.train()

        total_loss = 0.0
        loss_counts = {'reconstruction': 0.0, 'content': 0.0, 'timbre': 0.0}
        num_batches = 0
        num_valid = 0

        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Skip if no valid samples
            valid_mask = batch['valid']
            if not valid_mask.any():
                continue

            # Move to device
            input_latent = batch['input_latent'].to(self.device)
            target_pitch = batch['target_pitch'].to(self.device)
            shift_amount = batch['shift_amount'].to(self.device)

            batch_device = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Forward pass
            output = self.model(input_latent, target_pitch, shift_amount)

            # Compute loss
            loss, loss_dict = self.criterion(output, batch_device)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            num_valid += valid_mask.sum().item()

            for k in loss_counts:
                if k in loss_dict:
                    loss_counts[k] += loss_dict[k]

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })

        # Average metrics
        avg_loss = total_loss / max(num_batches, 1)
        avg_losses = {k: v / max(num_batches, 1) for k, v in loss_counts.items()}

        return {
            'loss': avg_loss,
            'num_batches': num_batches,
            'num_valid': num_valid,
            **avg_losses,
        }

    def train(self):
        """Run full training loop."""
        print("\n" + "=" * 60)
        print("REGISTER-AWARE PITCH SHIFT TRAINING")
        print("=" * 60)
        print(f"Output: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Model type: {self.model_type}")
        print(f"Epochs: {self.num_epochs}")
        print(f"Samples per epoch: {len(self.dataset)}")
        print("=" * 60 + "\n")

        start_time = datetime.now()

        for epoch in range(self.start_epoch, self.num_epochs + 1):
            metrics = self.train_epoch(epoch)
            self.train_losses.append(metrics['loss'])

            # Log progress
            log_str = f"Epoch {epoch}: loss={metrics['loss']:.4f}"
            for k in ['reconstruction', 'content', 'timbre']:
                if k in metrics and metrics[k] > 0:
                    log_str += f", {k}={metrics[k]:.4f}"
            print(log_str)

            # Check for best model
            is_best = metrics['loss'] < self.best_loss
            if is_best:
                self.best_loss = metrics['loss']

            # Save checkpoint
            self.save_checkpoint(epoch, metrics['loss'], is_best)

        # Training complete
        elapsed = datetime.now() - start_time
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE")
        print("=" * 60)
        print(f"Total time: {elapsed}")
        print(f"Best loss: {self.best_loss:.4f}")
        print(f"Final checkpoint: {self.output_dir / 'latest.pt'}")
        print(f"Best checkpoint: {self.output_dir / 'best.pt'}")
        print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train Register-Aware Pitch Shift Translator"
    )

    # Data arguments
    parser.add_argument('--manifest', type=str, required=True,
                        help='Path to manifest JSON')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for checkpoints')
    parser.add_argument('--instrument', type=str, default='trumpet',
                        help='Instrument to train on (sub_group in manifest)')

    # Model arguments
    parser.add_argument('--model_type', type=str, default='residual',
                        choices=['residual', 'direct', 'large'],
                        help='Model architecture type')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')

    # Dataset arguments
    parser.add_argument('--window_frames', type=int, default=128,
                        help='Latent window size in frames')
    parser.add_argument('--samples_per_epoch', type=int, default=10000,
                        help='Number of samples per epoch')
    parser.add_argument('--shift_range', type=int, nargs=2, default=[-12, 12],
                        help='Min and max shift in semitones')
    parser.add_argument('--artifact_ratio', type=float, default=0.5,
                        help='Ratio of artifact removal vs register transfer samples')

    # Loss weights
    parser.add_argument('--reconstruction_weight', type=float, default=1.0)
    parser.add_argument('--timbre_weight', type=float, default=0.5)
    parser.add_argument('--content_weight', type=float, default=0.3)

    # Checkpoint arguments
    parser.add_argument('--checkpoint_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    # Real degradation (optional, requires DCAE)
    parser.add_argument('--use_real_degradation', action='store_true',
                        help='Use actual audio pitch shift for degradation')
    parser.add_argument('--dcae_checkpoint', type=str,
                        default='/home/arlo/Data/ACE-Step/checkpoints',
                        help='Path to DCAE checkpoint (for real degradation)')

    args = parser.parse_args()

    trainer = PitchShiftTrainer(
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        instrument=args.instrument,
        model_type=args.model_type,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        window_frames=args.window_frames,
        samples_per_epoch=args.samples_per_epoch,
        shift_range=tuple(args.shift_range),
        artifact_removal_ratio=args.artifact_ratio,
        reconstruction_weight=args.reconstruction_weight,
        timbre_weight=args.timbre_weight,
        content_weight=args.content_weight,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        resume_from=args.resume,
        use_real_degradation=args.use_real_degradation,
        dcae_checkpoint=args.dcae_checkpoint,
    )

    trainer.train()


if __name__ == "__main__":
    main()
