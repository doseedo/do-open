#!/usr/bin/env python3
"""
Train Audio Enhancer - Post-DCAE super-resolution model.

Creates pairs of (decoded_dcae_audio, original_audio) on the fly
and trains the enhancer to restore high-fidelity audio.

Usage:
    python train_enhancer.py \
        --manifest /home/arlo/Data.backup/final_training_manifest_final.json \
        --output_dir /mnt/msdd2/audio_enhancer_checkpoints/brass_v1 \
        --groups brass \
        --epochs 50
"""

import os
import sys
import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import torchaudio
from tqdm import tqdm

sys.path.insert(0, '/home/arlo/Data/clarifier')
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from audio_enhancer import AudioEnhancer, AudioEnhancerLarge


# Group/subgroup mappings (same as clarifier)
GROUP_TO_ID = {
    'strings': 0, 'woodwinds': 1, 'keys': 2, 'percussion': 3,
    'brass': 4, 'vocals': 5,
}

SUBGROUP_TO_ID = {
    'violin': 0, 'viola': 1, 'cello': 2, 'double_bass': 3,
    'flute': 4, 'oboe': 5, 'clarinet': 6, 'bassoon': 7,
    'french_horn': 8, 'piano': 9, 'organ': 10, 'harpsichord': 11,
    'drums': 12, 'trumpet': 13, 'trombone': 14, 'tuba': 15,
    'sax': 16, 'soprano': 17, 'alto': 18, 'tenor': 19, 'bass': 20,
}


def fix_path(path: str) -> str:
    """Fix mount paths."""
    if not path:
        return path
    replacements = [
        ('/mnt/msdd/', '/mnt/msdd2/'),
    ]
    for old, new in replacements:
        if old in path:
            path = path.replace(old, new)
    return path


class PreprocessedEnhancerDataset(Dataset):
    """Dataset that loads preprocessed (original, decoded) pairs from disk."""

    def __init__(self, pairs_dir: str):
        self.pairs_dir = Path(pairs_dir)
        self.pair_files = sorted(self.pairs_dir.glob('pair_*.pt'))
        print(f"[PreprocessedEnhancerDataset] Found {len(self.pair_files)} pairs in {pairs_dir}")

        # Count by subgroup
        by_subgroup = {}
        for pf in self.pair_files[:100]:  # Sample first 100
            data = torch.load(pf, map_location='cpu', weights_only=False)
            sg = data.get('subgroup', 'unknown')
            by_subgroup[sg] = by_subgroup.get(sg, 0) + 1
        if by_subgroup:
            print("  Sample distribution:")
            for sg, count in sorted(by_subgroup.items()):
                print(f"    {sg}: {count}")

    def __len__(self):
        return len(self.pair_files)

    def __getitem__(self, idx: int) -> dict:
        data = torch.load(self.pair_files[idx], map_location='cpu', weights_only=False)
        return {
            'original': data['original'],
            'decoded': data['decoded'],
            'group_id': data['group_id'],
            'subgroup_id': data['subgroup_id'],
            'valid': True,
        }


class EnhancerDataset(Dataset):
    """
    Dataset for audio enhancer training (on-the-fly DCAE encode/decode).
    Use PreprocessedEnhancerDataset for faster training with preprocessed pairs.
    """

    def __init__(
        self,
        manifest_path: str,
        dcae,
        groups: list = None,
        segment_samples: int = 48000 * 3,
        sample_rate: int = 48000,
        device: str = 'cuda',
    ):
        self.dcae = dcae
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.device = device

        with open(manifest_path) as f:
            manifest = json.load(f)

        self.entries = []
        for entry in manifest:
            if entry.get('ensemble_detected') or entry.get('session_ensemble_flagged'):
                continue

            group = entry.get('group', '')
            subgroup = entry.get('sub_group', '')

            if groups and group not in groups:
                continue

            audio_path = fix_path(entry.get('audio_path', ''))
            if not audio_path or not os.path.exists(audio_path):
                continue

            if group not in GROUP_TO_ID:
                continue

            self.entries.append({
                'audio_path': audio_path,
                'group_id': GROUP_TO_ID[group],
                'subgroup_id': SUBGROUP_TO_ID.get(subgroup, 0),
                'group': group,
                'subgroup': subgroup,
            })

        print(f"[EnhancerDataset] Loaded {len(self.entries)} entries")
        by_subgroup = {}
        for e in self.entries:
            sg = e['subgroup']
            by_subgroup[sg] = by_subgroup.get(sg, 0) + 1
        for sg, count in sorted(by_subgroup.items()):
            print(f"  {sg}: {count}")

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int) -> dict:
        entry = self.entries[idx]

        try:
            audio, sr = torchaudio.load(entry['audio_path'])
            if sr != self.sample_rate:
                audio = torchaudio.transforms.Resample(sr, self.sample_rate)(audio)
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]
        except Exception:
            return {
                'original': torch.zeros(2, self.segment_samples),
                'decoded': torch.zeros(2, self.segment_samples),
                'group_id': entry['group_id'],
                'subgroup_id': entry['subgroup_id'],
                'valid': False,
            }

        T = audio.shape[-1]
        if T <= self.segment_samples:
            audio = F.pad(audio, (0, self.segment_samples - T))
        else:
            start = random.randint(0, T - self.segment_samples)
            audio = audio[:, start:start + self.segment_samples]

        with torch.no_grad():
            audio_gpu = audio.unsqueeze(0).to(self.device)
            latent, _ = self.dcae.encode(audio_gpu)
            audio_len = torch.tensor([audio_gpu.shape[-1]], device=self.device)
            _, decoded = self.dcae.decode(latent, audio_lengths=audio_len, sr=self.sample_rate)
            if isinstance(decoded, list):
                decoded = decoded[0]
            decoded = decoded.squeeze(0).cpu()

        min_len = min(audio.shape[-1], decoded.shape[-1])
        return {
            'original': audio[:, :min_len],
            'decoded': decoded[:, :min_len],
            'group_id': entry['group_id'],
            'subgroup_id': entry['subgroup_id'],
            'valid': True,
        }


def collate_fn(batch):
    """Filter invalid samples and stack."""
    valid = [b for b in batch if b['valid']]
    if not valid:
        return None

    return {
        'original': torch.stack([b['original'] for b in valid]),
        'decoded': torch.stack([b['decoded'] for b in valid]),
        'group_id': torch.tensor([b['group_id'] for b in valid]),
        'subgroup_id': torch.tensor([b['subgroup_id'] for b in valid]),
    }


class EnhancerTrainer:
    """Trainer for AudioEnhancer."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 1e-4,
        output_dir: str = './checkpoints',
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        self.l1_loss = nn.L1Loss()
        self.global_step = 0
        self.best_val_loss = float('inf')

    def multi_scale_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Multi-scale L1 loss."""
        loss = self.l1_loss(pred, target)

        # Add loss at lower resolutions
        for scale in [2, 4]:
            pred_down = F.avg_pool1d(pred, scale)
            target_down = F.avg_pool1d(target, scale)
            loss = loss + 0.5 * self.l1_loss(pred_down, target_down)

        return loss

    def train_step(self, batch: dict) -> dict:
        """Single training step."""
        self.model.train()

        decoded = batch['decoded'].to(self.device)
        original = batch['original'].to(self.device)
        group_id = batch['group_id'].to(self.device)
        subgroup_id = batch['subgroup_id'].to(self.device)

        # Forward
        enhanced = self.model(decoded, group_id, subgroup_id)

        # Loss
        loss = self.multi_scale_loss(enhanced, original)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        self.global_step += 1

        return {'loss': loss.item()}

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            if batch is None:
                continue

            decoded = batch['decoded'].to(self.device)
            original = batch['original'].to(self.device)
            group_id = batch['group_id'].to(self.device)
            subgroup_id = batch['subgroup_id'].to(self.device)

            enhanced = self.model(decoded, group_id, subgroup_id)
            loss = self.multi_scale_loss(enhanced, original)

            total_loss += loss.item()
            n_batches += 1

        return {'val_loss': total_loss / max(n_batches, 1)}

    def save_checkpoint(self, filename: str):
        """Save checkpoint."""
        path = self.output_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
        }, path)

    def train(self, epochs: int, save_every: int = 5):
        """Full training loop."""
        print(f"Starting training for {epochs} epochs")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"Output: {self.output_dir}")

        total_steps = epochs * len(self.train_loader)
        scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.optimizer.param_groups[0]['lr'],
            total_steps=total_steps,
            pct_start=0.1,
        )

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                if batch is None:
                    continue

                losses = self.train_step(batch)
                scheduler.step()

                epoch_loss += losses['loss']
                n_batches += 1

                pbar.set_postfix({
                    'loss': f"{losses['loss']:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })

            avg_loss = epoch_loss / max(n_batches, 1)
            print(f"\nEpoch {epoch+1} - Train Loss: {avg_loss:.4f}")

            # Validation
            val_metrics = self.validate()
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}")

            # Save best
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_checkpoint('best.pt')
                print(f"  New best! Val loss: {self.best_val_loss:.4f}")

            # Periodic save
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'epoch_{epoch+1:04d}.pt')

        self.save_checkpoint('final.pt')
        print(f"\nTraining complete. Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Audio Enhancer")
    parser.add_argument('--manifest', type=str, default=None,
                        help='Path to training manifest (for on-the-fly mode)')
    parser.add_argument('--pairs_dir', type=str, default=None,
                        help='Path to preprocessed pairs directory (faster)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--groups', type=str, nargs='+', default=['brass'],
                        help='Instrument groups to train on (on-the-fly mode only)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--segment_seconds', type=float, default=3.0)
    parser.add_argument('--val_split', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_size', type=str, default='base',
                        choices=['base', 'large'])
    parser.add_argument('--group_vocab', type=int, default=6)
    parser.add_argument('--subgroup_vocab', type=int, default=21)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    if not args.pairs_dir and not args.manifest:
        parser.error("Must specify either --pairs_dir or --manifest")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Create dataset
    if args.pairs_dir:
        # Use preprocessed pairs (fast)
        print(f"Using preprocessed pairs from {args.pairs_dir}")
        full_dataset = PreprocessedEnhancerDataset(args.pairs_dir)
        use_workers = args.num_workers
    else:
        # On-the-fly DCAE encode/decode (slow)
        print("Loading DCAE for on-the-fly processing...")
        from acestep.music_dcae.music_dcae_pipeline import MusicDCAE
        base = '/home/arlo/Data/ACE-Step/checkpoints/models--ACE-Step--ACE-Step-v1-3.5B/snapshots/82cd0d7b6322bd28cd4e830fe675ddb6180ce36c'
        dcae = MusicDCAE(
            dcae_checkpoint_path=f'{base}/music_dcae_f8c8',
            vocoder_checkpoint_path=f'{base}/music_vocoder'
        )
        dcae.to(args.device).eval()
        print("DCAE loaded")

        segment_samples = int(args.segment_seconds * 48000)
        full_dataset = EnhancerDataset(
            manifest_path=args.manifest,
            dcae=dcae,
            groups=args.groups,
            segment_samples=segment_samples,
            device=args.device,
        )
        use_workers = 0  # Can't use workers with CUDA in __getitem__

    # Split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=use_workers,
        collate_fn=collate_fn,
        pin_memory=use_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=use_workers,
        collate_fn=collate_fn,
        pin_memory=use_workers > 0,
    )

    # Create model
    if args.model_size == 'large':
        model = AudioEnhancerLarge(
            group_vocab=args.group_vocab,
            subgroup_vocab=args.subgroup_vocab,
        )
    else:
        model = AudioEnhancer(
            group_vocab=args.group_vocab,
            subgroup_vocab=args.subgroup_vocab,
        )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = EnhancerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        output_dir=args.output_dir,
        device=args.device,
    )

    trainer.train(epochs=args.epochs)


if __name__ == '__main__':
    main()
