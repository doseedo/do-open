#!/usr/bin/env python3
"""
Stage 1 Training: Effect Encoder Only

Trains the effect encoder to classify which effects are present in wet audio.
This is a simpler task than full chain estimation - validates the architecture.

NOTE: Imports are deferred to after argparse to avoid CLAP/FAD argparse conflicts.
"""

import argparse
from pathlib import Path

# Parse args FIRST before any imports that trigger CLAP
def parse_args():
    parser = argparse.ArgumentParser(description="Train Effect Encoder (Stage 1)")

    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default="/mnt/models/inverse_afx_data",
        help="Directory containing generated data",
    )
    parser.add_argument(
        "--manifest", "-m",
        type=str,
        default=None,
        help="Path to manifest.json (default: data_dir/manifest.json)",
    )
    parser.add_argument(
        "--batch_size", "-b",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--max_epochs", "-e",
        type=int,
        default=50,
        help="Maximum epochs",
    )
    parser.add_argument(
        "--learning_rate", "-lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=512,
        help="Encoder embedding dimension",
    )
    parser.add_argument(
        "--num_workers", "-w",
        type=int,
        default=8,
        help="DataLoader workers",
    )
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of GPUs",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="16-mixed",
        help="Training precision",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="encoder_only",
        help="Experiment name",
    )

    return parser.parse_args()

# Parse args before any heavy imports
args = parse_args()

# Now import everything else
import sys
sys.argv = [sys.argv[0]]  # Clear args to prevent CLAP hijacking

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from inverse_afx.models.effect_encoder import EffectEncoder, EffectEncoderConfig
from inverse_afx.data.datasets import InverseAFxDataset, InverseAFxDataModule


class EncoderOnlySystem(pl.LightningModule):
    """
    Stage 1: Train encoder for multi-label effect classification.

    Given wet audio, predict which effects are present (multi-label).
    """

    def __init__(
        self,
        embedding_dim: int = 512,
        num_effects: int = 6,
        effect_types: list = None,
        learning_rate: float = 1e-4,
        sample_rate: int = 48000,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.effect_types = effect_types or ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']
        self.num_effects = len(self.effect_types)

        # Effect encoder
        encoder_config = EffectEncoderConfig(
            embedding_dim=embedding_dim,
            num_effect_types=self.num_effects,
            sample_rate=sample_rate,
        )
        self.encoder = EffectEncoder(encoder_config)

        # Classification head for multi-label prediction
        self.effect_classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, self.num_effects),
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, wet_audio):
        """Forward pass: wet audio -> effect predictions."""
        embedding = self.encoder(wet_audio)
        logits = self.effect_classifier(embedding)
        return logits, embedding

    def _batch_to_labels(self, batch):
        """Convert batch chain specs to multi-hot labels."""
        effect_types = batch['effect_types']  # [B, max_chain_length]
        chain_length = batch['chain_length']  # [B]

        batch_size = effect_types.size(0)
        labels = torch.zeros(batch_size, self.num_effects, device=self.device)

        for b in range(batch_size):
            for i in range(chain_length[b].item()):
                effect_idx = effect_types[b, i].item()
                if effect_idx < self.num_effects:
                    labels[b, effect_idx] = 1.0

        return labels

    def training_step(self, batch, batch_idx):
        wet_audio = batch['wet_audio']
        labels = self._batch_to_labels(batch)

        # Debug: check for NaN in input
        if torch.isnan(wet_audio).any():
            print(f"NaN in wet_audio at batch {batch_idx}")
            wet_audio = torch.nan_to_num(wet_audio, nan=0.0)

        logits, _ = self(wet_audio)

        # Debug: check for NaN in logits
        if torch.isnan(logits).any():
            print(f"NaN in logits at batch {batch_idx}")
            logits = torch.nan_to_num(logits, nan=0.0)

        loss = self.criterion(logits, labels)

        # Skip NaN losses
        if torch.isnan(loss):
            print(f"NaN loss at batch {batch_idx}, skipping")
            return None

        # Compute accuracy
        preds = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (preds == labels).float().mean()

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/accuracy', accuracy, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        wet_audio = batch['wet_audio']
        labels = self._batch_to_labels(batch)

        logits, _ = self(wet_audio)
        loss = self.criterion(logits, labels)

        # Compute metrics
        preds = (torch.sigmoid(logits) > 0.5).float()
        accuracy = (preds == labels).float().mean()

        # Per-effect accuracy
        for i, effect in enumerate(self.effect_types):
            effect_acc = (preds[:, i] == labels[:, i]).float().mean()
            self.log(f'val/acc_{effect}', effect_acc)

        self.log('val/loss', loss, prog_bar=True)
        self.log('val/accuracy', accuracy, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=1e-6,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


def main():
    # Use global args parsed at module load time
    global args

    # Setup paths
    data_dir = Path(args.data_dir)
    manifest_path = args.manifest or str(data_dir / "manifest.json")

    if not Path(manifest_path).exists():
        print(f"Error: Manifest not found at {manifest_path}")
        print("Make sure data generation is complete.")
        return

    # Create checkpoint dir
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Effect types
    effect_types = ['eq', 'compressor', 'reverb', 'distortion', 'chorus', 'delay']

    # Create data module
    # load_dry=False skips loading dry audio (not needed for encoder-only training)
    # This dramatically speeds up data loading
    data_module = InverseAFxDataModule(
        train_dir=data_dir,  # Not used in precomputed mode
        sample_rate=48000,
        segment_length=144000,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_chain_length=4,
        effect_types=effect_types,
        mode='precomputed',
        train_manifest=manifest_path,
        val_manifest=manifest_path,  # Use same for now, will split later
        load_dry=False,  # Skip dry audio - encoder only uses wet audio
        prefetch_factor=4,  # Prefetch batches for faster loading
    )

    # Create model
    model = EncoderOnlySystem(
        embedding_dim=args.embedding_dim,
        num_effects=len(effect_types),
        effect_types=effect_types,
        learning_rate=args.learning_rate,
        sample_rate=48000,
    )

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=args.checkpoint_dir,
            filename=f"{args.exp_name}-{{epoch:02d}}-{{val/accuracy:.3f}}",
            monitor="val/accuracy",
            mode="max",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.exp_name,
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.25,  # Validate 4x per epoch
    )

    # Train
    print("=" * 60)
    print("Stage 1: Effect Encoder Training")
    print("=" * 60)
    print(f"Data: {manifest_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Embedding dim: {args.embedding_dim}")
    print(f"Max epochs: {args.max_epochs}")
    print("=" * 60)

    trainer.fit(model, data_module)

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")


if __name__ == "__main__":
    main()
