#!/usr/bin/env python3
# /home/arlo/Data/clarifier/train.py
# Apache 2.0
# Training loop for InstrumentClarifier

"""
Usage:
    python train.py \
        --pairs_dir /path/to/clarifier_pairs_aligned \
        --output_dir /path/to/clarifier_checkpoints \
        --epochs 100 \
        --batch_size 8 \
        --lr 1e-4

With instrument classifier aux loss:
    python train.py \
        --pairs_dir /path/to/clarifier_pairs_aligned \
        --output_dir /path/to/clarifier_checkpoints \
        --use_classifier_aux \
        --classifier_weight 0.1
"""

import sys
sys.path.insert(0, '/home/arlo/Data/clarifier')

import argparse
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR

try:
    from torch.utils.tensorboard import SummaryWriter
    HAVE_TENSORBOARD = True
except ImportError:
    HAVE_TENSORBOARD = False

from tqdm import tqdm

from models import InstrumentClarifier, InstrumentClarifierLarge, SimpleInstrumentClassifier
from dataset import ClarifierPairDataset, collate_clarifier


def hf_weighted_loss(pred: torch.Tensor, target: torch.Tensor, hf_weight: float = 2.0) -> torch.Tensor:
    """
    L1 loss with higher weight on upper frequency bins (formants/brightness).

    Args:
        pred: [B, 8, 16, T]
        target: [B, 8, 16, T]
        hf_weight: Weight multiplier for upper half of H dimension

    Returns:
        weighted L1 loss
    """
    weight = torch.ones_like(pred)
    H = pred.shape[2]
    weight[:, :, H // 2:, :] = hf_weight  # Weight upper half more

    loss = (weight * (pred - target).abs()).mean()
    return loss


def spectral_convergence_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Spectral convergence loss (Frobenius norm ratio).

    Args:
        pred: [B, 8, 16, T]
        target: [B, 8, 16, T]
    """
    return torch.norm(pred - target, p='fro') / (torch.norm(target, p='fro') + 1e-8)


class ClarifierTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        classifier: Optional[nn.Module] = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        hf_weight: float = 2.0,
        classifier_weight: float = 0.1,
        use_spectral_loss: bool = True,
        spectral_weight: float = 0.5,
        output_dir: str = "./checkpoints",
        device: str = "cuda",
        grad_clip: float = 1.0,
        warmup_steps: int = 500,
    ):
        self.model = model.to(device)
        self.classifier = classifier.to(device) if classifier else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hf_weight = hf_weight
        self.classifier_weight = classifier_weight
        self.use_spectral_loss = use_spectral_loss
        self.spectral_weight = spectral_weight
        self.grad_clip = grad_clip
        self.warmup_steps = warmup_steps

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )

        # TensorBoard
        if HAVE_TENSORBOARD:
            log_dir = self.output_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        synthetic = batch["synthetic"].to(self.device)
        real = batch["real"].to(self.device)
        group_id = batch["group_id"].to(self.device)
        subgroup_id = batch["subgroup_id"].to(self.device)

        # Forward
        clarified = self.model(synthetic, group_id, subgroup_id)

        # Losses
        losses = {}

        # L1 reconstruction loss
        recon_loss = F.l1_loss(clarified, real)
        losses["recon"] = recon_loss.item()

        # HF-weighted loss
        hf_loss = hf_weighted_loss(clarified, real, self.hf_weight)
        losses["hf"] = hf_loss.item()

        # Spectral convergence
        if self.use_spectral_loss:
            spec_loss = spectral_convergence_loss(clarified, real)
            losses["spectral"] = spec_loss.item()
        else:
            spec_loss = torch.tensor(0.0)

        # Classifier aux loss
        if self.classifier is not None:
            with torch.no_grad():
                self.classifier.eval()
            g_logits, sg_logits = self.classifier(clarified)
            cls_loss = F.cross_entropy(g_logits, group_id) + F.cross_entropy(sg_logits, subgroup_id)
            losses["classifier"] = cls_loss.item()
        else:
            cls_loss = torch.tensor(0.0)

        # Total loss
        total_loss = (
            recon_loss +
            0.5 * hf_loss +
            self.spectral_weight * spec_loss +
            self.classifier_weight * cls_loss
        )
        losses["total"] = total_loss.item()

        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.global_step += 1

        return losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_losses = {
            "recon": 0.0,
            "hf": 0.0,
            "spectral": 0.0,
            "total": 0.0,
        }
        num_batches = 0

        for batch in self.val_loader:
            synthetic = batch["synthetic"].to(self.device)
            real = batch["real"].to(self.device)
            group_id = batch["group_id"].to(self.device)
            subgroup_id = batch["subgroup_id"].to(self.device)

            clarified = self.model(synthetic, group_id, subgroup_id)

            total_losses["recon"] += F.l1_loss(clarified, real).item()
            total_losses["hf"] += hf_weighted_loss(clarified, real, self.hf_weight).item()
            if self.use_spectral_loss:
                total_losses["spectral"] += spectral_convergence_loss(clarified, real).item()

            num_batches += 1

        # Average
        for k in total_losses:
            total_losses[k] /= max(1, num_batches)

        total_losses["total"] = (
            total_losses["recon"] +
            0.5 * total_losses["hf"] +
            self.spectral_weight * total_losses["spectral"]
        )

        return total_losses

    def train(
        self,
        epochs: int,
        save_every: int = 5,
        val_every: int = 1,
        log_every: int = 50,
    ):
        """Full training loop."""
        print(f"Starting training for {epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")
        print(f"Output dir: {self.output_dir}")

        # Learning rate scheduler
        total_steps = epochs * len(self.train_loader)
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos',
        )

        for epoch in range(epochs):
            epoch_losses = {
                "recon": 0.0,
                "hf": 0.0,
                "spectral": 0.0,
                "classifier": 0.0,
                "total": 0.0,
            }
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                losses = self.train_step(batch)
                scheduler.step()

                for k, v in losses.items():
                    epoch_losses[k] += v
                num_batches += 1

                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{losses['total']:.4f}",
                    "recon": f"{losses['recon']:.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                })

                # Log to tensorboard
                if self.writer and self.global_step % log_every == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                    self.writer.add_scalar("train/lr", scheduler.get_last_lr()[0], self.global_step)

                    # Log model scales
                    self.writer.add_scalar(
                        "model/input_scale",
                        self.model.input_scale.item(),
                        self.global_step
                    )
                    self.writer.add_scalar(
                        "model/output_scale",
                        self.model.output_scale.item(),
                        self.global_step
                    )

            # Epoch summary
            for k in epoch_losses:
                epoch_losses[k] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1} - Train Loss: {epoch_losses['total']:.4f}")

            # Validation
            if self.val_loader and (epoch + 1) % val_every == 0:
                val_losses = self.validate()
                print(f"  Val Loss: {val_losses['total']:.4f}")

                if self.writer:
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"val/{k}", v, self.global_step)

                # Save best model
                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.save_checkpoint("best.pt")
                    print(f"  New best model! Val loss: {self.best_val_loss:.4f}")

            # Save checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}.pt")

        # Final save
        self.save_checkpoint("final.pt")
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "model_config": {
                "input_scale": self.model.input_scale.item(),
                "output_scale": self.model.output_scale.item(),
            }
        }, path)
        print(f"Saved checkpoint: {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Loaded checkpoint from {path}")


def pretrain_classifier(
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    output_dir: str,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
):
    """
    Pretrain the instrument classifier on real latents.
    This is used as an auxiliary loss for clarifier training.
    """
    print("Pretraining instrument classifier...")

    classifier = SimpleInstrumentClassifier().to(device)
    optimizer = AdamW(classifier.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        correct_g = 0
        correct_sg = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}"):
            real = batch["real"].to(device)
            group_id = batch["group_id"].to(device)
            subgroup_id = batch["subgroup_id"].to(device)

            g_logits, sg_logits = classifier(real)
            loss = F.cross_entropy(g_logits, group_id) + F.cross_entropy(sg_logits, subgroup_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct_g += (g_logits.argmax(dim=1) == group_id).sum().item()
            correct_sg += (sg_logits.argmax(dim=1) == subgroup_id).sum().item()
            total += group_id.size(0)

        acc_g = 100.0 * correct_g / total
        acc_sg = 100.0 * correct_sg / total
        print(f"  Loss: {total_loss / len(train_loader):.4f}, "
              f"Group Acc: {acc_g:.1f}%, Subgroup Acc: {acc_sg:.1f}%")

        if acc_g > best_acc:
            best_acc = acc_g
            torch.save(classifier.state_dict(), f"{output_dir}/classifier_best.pt")

    print(f"Classifier training done. Best group accuracy: {best_acc:.1f}%")
    return classifier


def main():
    parser = argparse.ArgumentParser(description="Train InstrumentClarifier")
    parser.add_argument("--pairs_dir", type=str, required=True,
                        help="Directory containing aligned pair .pt files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints and logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=256,
                        help="Window size in slow frames")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split fraction")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Loss weights
    parser.add_argument("--hf_weight", type=float, default=2.0,
                        help="Weight for high-frequency bins")
    parser.add_argument("--spectral_weight", type=float, default=0.5)
    parser.add_argument("--classifier_weight", type=float, default=0.1)
    parser.add_argument("--use_classifier_aux", action="store_true",
                        help="Use instrument classifier auxiliary loss")
    parser.add_argument("--pretrain_classifier_epochs", type=int, default=20)

    # Model
    parser.add_argument("--model_size", type=str, default="base",
                        choices=["base", "large"],
                        help="Model size")
    parser.add_argument("--group_vocab", type=int, default=6,
                        help="Number of instrument groups")
    parser.add_argument("--subgroup_vocab", type=int, default=20,
                        help="Number of subgroups (should cover max subgroup_id + 1)")

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Create dataset
    full_dataset = ClarifierPairDataset(
        pairs_dir=args.pairs_dir,
        window_size=args.window_size,
        random_crop=True,
        seed=args.seed,
    )

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_clarifier,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_clarifier,
        pin_memory=True,
    ) if val_size > 0 else None

    # Create model
    if args.model_size == "large":
        model = InstrumentClarifierLarge(
            group_vocab=args.group_vocab,
            subgroup_vocab=args.subgroup_vocab,
        )
    else:
        model = InstrumentClarifier(
            group_vocab=args.group_vocab,
            subgroup_vocab=args.subgroup_vocab,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optionally pretrain classifier
    classifier = None
    if args.use_classifier_aux:
        classifier = pretrain_classifier(
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=args.output_dir,
            epochs=args.pretrain_classifier_epochs,
            device=args.device,
        )

    # Create trainer
    trainer = ClarifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classifier=classifier,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hf_weight=args.hf_weight,
        classifier_weight=args.classifier_weight if classifier else 0.0,
        spectral_weight=args.spectral_weight,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train(
        epochs=args.epochs,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
