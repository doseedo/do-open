#!/usr/bin/env python3
# /home/arlo/Data/clarifier/train_group.py
# Apache 2.0
# Group-specific clarifier training (e.g., brass only)

"""
Train a clarifier for a SINGLE instrument group.

Examples:
    # Train brass clarifier (trumpet, trombone, french_horn, tuba)
    python train_group.py \
        --group brass \
        --pairs_dir /path/to/clarifier_pairs_aligned \
        --output_dir /path/to/brass_clarifier \
        --epochs 100 --batch_size 8

    # Train strings clarifier
    python train_group.py \
        --group strings \
        --pairs_dir /path/to/clarifier_pairs_aligned \
        --output_dir /path/to/strings_clarifier \
        --epochs 100

    # Available groups: piano, guitar, bass, strings, brass, winds
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
from torch.optim.lr_scheduler import OneCycleLR

try:
    from torch.utils.tensorboard import SummaryWriter
    HAVE_TENSORBOARD = True
except ImportError:
    HAVE_TENSORBOARD = False

from tqdm import tqdm

from config import get_group_config, save_config, GROUP_CONFIGS
from models import GroupClarifier, GroupSubgroupClassifier, create_group_clarifier, create_group_classifier
from dataset import ClarifierPairDataset, collate_clarifier


def hf_weighted_loss(pred: torch.Tensor, target: torch.Tensor, hf_weight: float = 2.0) -> torch.Tensor:
    """L1 loss with higher weight on upper frequency bins."""
    weight = torch.ones_like(pred)
    H = pred.shape[2]
    weight[:, :, H // 2:, :] = hf_weight
    return (weight * (pred - target).abs()).mean()


def spectral_convergence_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Spectral convergence loss (Frobenius norm ratio)."""
    return torch.norm(pred - target, p='fro') / (torch.norm(target, p='fro') + 1e-8)


class GroupClarifierTrainer:
    """Trainer for group-specific clarifiers."""

    def __init__(
        self,
        model: GroupClarifier,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        classifier: Optional[GroupSubgroupClassifier] = None,
        group_config: Any = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        hf_weight: float = 2.0,
        classifier_weight: float = 0.1,
        spectral_weight: float = 0.5,
        output_dir: str = "./checkpoints",
        device: str = "cuda",
        grad_clip: float = 1.0,
    ):
        self.model = model.to(device)
        self.classifier = classifier.to(device) if classifier else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.group_config = group_config
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.hf_weight = hf_weight
        self.classifier_weight = classifier_weight
        self.spectral_weight = spectral_weight
        self.grad_clip = grad_clip

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        if HAVE_TENSORBOARD:
            log_dir = self.output_dir / "logs" / datetime.now().strftime("%Y%m%d_%H%M%S")
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None

        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step using local_subgroup_id."""
        self.model.train()

        synthetic = batch["synthetic"].to(self.device)
        real = batch["real"].to(self.device)
        # Use local subgroup ID for group-specific model
        subgroup_id = batch["local_subgroup_id"].to(self.device)

        # Forward
        clarified = self.model(synthetic, subgroup_id)

        # Losses
        losses = {}

        recon_loss = F.l1_loss(clarified, real)
        losses["recon"] = recon_loss.item()

        hf_loss = hf_weighted_loss(clarified, real, self.hf_weight)
        losses["hf"] = hf_loss.item()

        spec_loss = spectral_convergence_loss(clarified, real)
        losses["spectral"] = spec_loss.item()

        # Classifier aux loss
        if self.classifier is not None:
            with torch.no_grad():
                self.classifier.eval()
            logits = self.classifier(clarified)
            cls_loss = F.cross_entropy(logits, subgroup_id)
            losses["classifier"] = cls_loss.item()
        else:
            cls_loss = torch.tensor(0.0)

        total_loss = (
            recon_loss +
            0.5 * hf_loss +
            self.spectral_weight * spec_loss +
            self.classifier_weight * cls_loss
        )
        losses["total"] = total_loss.item()

        self.optimizer.zero_grad()
        total_loss.backward()

        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

        self.optimizer.step()
        self.global_step += 1

        return losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        if self.val_loader is None:
            return {}

        self.model.eval()

        total_losses = {"recon": 0.0, "hf": 0.0, "spectral": 0.0, "total": 0.0}
        num_batches = 0

        for batch in self.val_loader:
            synthetic = batch["synthetic"].to(self.device)
            real = batch["real"].to(self.device)
            subgroup_id = batch["local_subgroup_id"].to(self.device)

            clarified = self.model(synthetic, subgroup_id)

            total_losses["recon"] += F.l1_loss(clarified, real).item()
            total_losses["hf"] += hf_weighted_loss(clarified, real, self.hf_weight).item()
            total_losses["spectral"] += spectral_convergence_loss(clarified, real).item()
            num_batches += 1

        for k in total_losses:
            total_losses[k] /= max(1, num_batches)

        total_losses["total"] = (
            total_losses["recon"] +
            0.5 * total_losses["hf"] +
            self.spectral_weight * total_losses["spectral"]
        )

        return total_losses

    def train(self, epochs: int, save_every: int = 5, val_every: int = 1, log_every: int = 50):
        group_name = self.model.group_name if hasattr(self.model, 'group_name') else "unknown"
        print(f"Training {group_name.upper()} clarifier for {epochs} epochs")
        print(f"Subgroups: {self.group_config.subgroups if self.group_config else 'unknown'}")
        print(f"Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"Val batches: {len(self.val_loader)}")

        total_steps = epochs * len(self.train_loader)
        if total_steps < 100:
            # Use simple constant LR for tiny datasets
            scheduler = None
        else:
            scheduler = OneCycleLR(
                self.optimizer,
                max_lr=self.optimizer.param_groups[0]['lr'],
                total_steps=total_steps,
                pct_start=0.1,
                anneal_strategy='cos',
            )

        for epoch in range(epochs):
            epoch_losses = {"recon": 0.0, "hf": 0.0, "spectral": 0.0, "classifier": 0.0, "total": 0.0}
            num_batches = 0

            pbar = tqdm(self.train_loader, desc=f"[{group_name}] Epoch {epoch + 1}/{epochs}")
            for batch in pbar:
                losses = self.train_step(batch)
                if scheduler:
                    scheduler.step()

                for k, v in losses.items():
                    epoch_losses[k] += v
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{losses['total']:.4f}",
                    "recon": f"{losses['recon']:.4f}",
                })

                if self.writer and self.global_step % log_every == 0:
                    for k, v in losses.items():
                        self.writer.add_scalar(f"train/{k}", v, self.global_step)
                    lr = scheduler.get_last_lr()[0] if scheduler else self.optimizer.param_groups[0]['lr']
                    self.writer.add_scalar("train/lr", lr, self.global_step)
                    self.writer.add_scalar("model/input_scale", self.model.input_scale.item(), self.global_step)
                    self.writer.add_scalar("model/output_scale", self.model.output_scale.item(), self.global_step)

            for k in epoch_losses:
                epoch_losses[k] /= max(1, num_batches)

            print(f"\nEpoch {epoch + 1} - Train Loss: {epoch_losses['total']:.4f}")

            if self.val_loader and (epoch + 1) % val_every == 0:
                val_losses = self.validate()
                print(f"  Val Loss: {val_losses['total']:.4f}")

                if self.writer:
                    for k, v in val_losses.items():
                        self.writer.add_scalar(f"val/{k}", v, self.global_step)

                if val_losses["total"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total"]
                    self.save_checkpoint("best.pt")
                    print(f"  New best model! Val loss: {self.best_val_loss:.4f}")

            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"epoch_{epoch + 1:04d}.pt")

        self.save_checkpoint("final.pt")
        print(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    def save_checkpoint(self, filename: str):
        path = self.output_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "group_name": self.model.group_name if hasattr(self.model, 'group_name') else None,
            "subgroup_vocab": self.model.subgroup_vocab if hasattr(self.model, 'subgroup_vocab') else None,
            "model_config": {
                "input_scale": self.model.input_scale.item(),
                "output_scale": self.model.output_scale.item(),
            }
        }, path)
        print(f"Saved checkpoint: {path}")


def pretrain_group_classifier(
    train_loader: DataLoader,
    group_config: Any,
    output_dir: str,
    epochs: int = 20,
    lr: float = 1e-3,
    device: str = "cuda",
) -> GroupSubgroupClassifier:
    """Pretrain a subgroup classifier for the target group."""
    print(f"Pretraining {group_config.name} subgroup classifier...")
    print(f"Subgroups: {group_config.subgroups}")

    classifier = create_group_classifier(group_config).to(device)
    optimizer = AdamW(classifier.parameters(), lr=lr)

    best_acc = 0.0

    for epoch in range(epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc=f"Classifier Epoch {epoch + 1}/{epochs}"):
            real = batch["real"].to(device)
            subgroup_id = batch["local_subgroup_id"].to(device)

            logits = classifier(real)
            loss = F.cross_entropy(logits, subgroup_id)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == subgroup_id).sum().item()
            total += subgroup_id.size(0)

        acc = 100.0 * correct / total
        print(f"  Loss: {total_loss / len(train_loader):.4f}, Acc: {acc:.1f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(classifier.state_dict(), f"{output_dir}/classifier_best.pt")

    print(f"Classifier training done. Best accuracy: {best_acc:.1f}%")
    return classifier


def main():
    parser = argparse.ArgumentParser(description="Train group-specific clarifier")

    # Required args
    parser.add_argument("--group", type=str, required=True,
                        choices=list(GROUP_CONFIGS.keys()),
                        help="Instrument group to train (e.g., brass, strings)")
    parser.add_argument("--pairs_dir", type=str, required=True,
                        help="Directory containing aligned pair .pt files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")

    # Training args
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--window_size", type=int, default=256)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    # Loss weights
    parser.add_argument("--hf_weight", type=float, default=2.0)
    parser.add_argument("--spectral_weight", type=float, default=0.5)
    parser.add_argument("--classifier_weight", type=float, default=0.1)
    parser.add_argument("--use_classifier_aux", action="store_true")
    parser.add_argument("--pretrain_classifier_epochs", type=int, default=20)

    # Model size
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_blocks", type=int, default=6)

    # Checkpointing
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=5)

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Get group configuration
    group_config = get_group_config(args.group)
    print(f"\n=== Training {args.group.upper()} Clarifier ===")
    print(f"Group ID: {group_config.group_id}")
    print(f"Subgroups ({group_config.num_subgroups}):")
    for sg, sg_id in group_config.subgroup_ids.items():
        print(f"  {sg_id}: {sg}")
    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # Save configs
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    save_config(group_config, os.path.join(args.output_dir, "group_config.json"))

    # Create dataset with group filtering
    full_dataset = ClarifierPairDataset(
        pairs_dir=args.pairs_dir,
        window_size=args.window_size,
        random_crop=True,
        seed=args.seed,
        filter_group=args.group,
        group_config=group_config,
    )

    if len(full_dataset) == 0:
        print(f"ERROR: No pairs found for group '{args.group}'")
        print("Make sure pair files have correct group_id values.")
        return

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

    # Create group-specific model
    model = create_group_clarifier(
        group_config,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optionally pretrain classifier
    classifier = None
    if args.use_classifier_aux:
        classifier = pretrain_group_classifier(
            train_loader=train_loader,
            group_config=group_config,
            output_dir=args.output_dir,
            epochs=args.pretrain_classifier_epochs,
            device=args.device,
        )

    # Create trainer
    trainer = GroupClarifierTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        classifier=classifier,
        group_config=group_config,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hf_weight=args.hf_weight,
        classifier_weight=args.classifier_weight if classifier else 0.0,
        spectral_weight=args.spectral_weight,
        output_dir=args.output_dir,
        device=args.device,
    )

    if args.resume:
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            trainer.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        trainer.global_step = ckpt.get("global_step", 0)
        trainer.best_val_loss = ckpt.get("best_val_loss", float('inf'))
        print(f"Resumed from {args.resume}")

    trainer.train(epochs=args.epochs, save_every=args.save_every)


if __name__ == "__main__":
    main()
