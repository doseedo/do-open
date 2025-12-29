#!/usr/bin/env python3
"""
Train F0-Conditioned Register Disentanglement Model

KEY TRAINING STRATEGY:
- Content from segment A (tight bottleneck: timing/dynamics)
- F0 from segment A (explicit pitch)
- Register from segment B (DIFFERENT segment, SAME pitch bin)
- Target: reconstruct segment A

With explicit F0, the bottleneck can be MUCH tighter.

Usage:
    python train_f0cond.py --manifest /path/to/manifest.json --output_dir /path/to/output
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models_f0cond import F0ConditionedModel


def fix_path(path: str) -> str:
    if not path:
        return path
    return path.replace('/mnt/msdd/', '/mnt/msdd2/').replace('/mnt/gcs-bucket/', '/home/arlo/gcs-bucket/')


class F0ConditionedDataset(Dataset):
    """
    Dataset that provides content/F0/register triplets.

    Groups segments by pitch bin, samples:
    - Content segment + its F0
    - Register segment (different segment, same pitch bin)
    """

    def __init__(self, manifest_path: str, instrument: str = 'trumpet',
                 min_length: int = 64, max_length: int = 256,
                 pitch_bin_size: float = 2.0):
        self.min_length = min_length
        self.max_length = max_length
        self.pitch_bin_size = pitch_bin_size

        print(f"Loading manifest from: {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)

        # Filter for instrument
        entries = [e for e in manifest if e.get('sub_group') == instrument]
        print(f"Found {len(entries)} {instrument} entries")

        # Validate entries have required fields
        valid_entries = []
        for e in entries:
            latent_path = fix_path(e.get('latent_path', ''))
            f0_path = fix_path(e.get('conditioning_paths', {}).get('f0', ''))

            if latent_path and f0_path and os.path.exists(latent_path) and os.path.exists(f0_path):
                valid_entries.append({
                    'latent_path': latent_path,
                    'f0_path': f0_path,
                })

        print(f"Valid entries with latent + F0: {len(valid_entries)}")

        # Load and segment by pitch
        self.pitch_bins = defaultdict(list)
        self.latent_cache = {}
        self.f0_cache = {}

        print("Analyzing pitch distribution...")
        for entry in tqdm(valid_entries, desc="Loading"):
            # Load F0 to get pitch info
            f0 = self._load_f0(entry['f0_path'])
            if f0 is None:
                continue

            # Get median pitch (non-zero frames)
            f0_valid = f0[f0 > 20]
            if len(f0_valid) < min_length:
                continue

            # Convert to MIDI
            midi = 12 * np.log2(f0_valid / 440) + 69
            median_midi = np.median(midi)
            pitch_bin = int(median_midi / pitch_bin_size)

            # Check latent length
            latent = self._load_latent(entry['latent_path'])
            if latent is None or latent.shape[-1] < min_length:
                continue

            self.pitch_bins[pitch_bin].append({
                'latent_path': entry['latent_path'],
                'f0_path': entry['f0_path'],
                'median_midi': median_midi,
                'length': latent.shape[-1],
            })

        # Filter bins with at least 2 segments
        valid_bins = {k: v for k, v in self.pitch_bins.items() if len(v) >= 2}
        self.pitch_bins = valid_bins

        # Create sample list
        self.samples = []
        for pitch_bin, segments in self.pitch_bins.items():
            for idx in range(len(segments)):
                self.samples.append((pitch_bin, idx))

        print(f"Final: {len(self.samples)} segments in {len(self.pitch_bins)} pitch bins")

    def __len__(self):
        return len(self.samples)

    def _load_latent(self, path):
        if path in self.latent_cache:
            return self.latent_cache[path]

        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                latent = data.get('latents', data.get('latent'))
            else:
                latent = data
            if latent is None:
                return None
            if latent.dim() == 4:
                latent = latent.squeeze(0)
            self.latent_cache[path] = latent
            return latent
        except:
            return None

    def _load_f0(self, path):
        if path in self.f0_cache:
            return self.f0_cache[path]

        try:
            f0 = np.load(path)
            f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
            self.f0_cache[path] = f0
            return f0
        except:
            return None

    def _extract_segment(self, seg_info, latent, f0):
        """Extract aligned segment from latent and F0."""
        length = min(latent.shape[-1], len(f0))
        seg_len = min(length, self.max_length)

        # Random offset if longer than max_length
        if length > self.max_length:
            offset = random.randint(0, length - self.max_length)
        else:
            offset = 0

        end = offset + seg_len

        latent_seg = latent[:, :, offset:end].clone()
        f0_seg = torch.from_numpy(f0[offset:end].copy())

        return latent_seg, f0_seg

    def __getitem__(self, idx):
        pitch_bin, seg_idx = self.samples[idx]
        segments_in_bin = self.pitch_bins[pitch_bin]

        # Content segment
        content_seg = segments_in_bin[seg_idx]
        content_latent = self._load_latent(content_seg['latent_path'])
        content_f0 = self._load_f0(content_seg['f0_path'])

        if content_latent is None or content_f0 is None:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        content, f0 = self._extract_segment(content_seg, content_latent, content_f0)

        # Register segment - DIFFERENT segment, same pitch bin
        other_indices = [i for i in range(len(segments_in_bin)) if i != seg_idx]
        register_idx = random.choice(other_indices)
        register_seg = segments_in_bin[register_idx]
        register_latent = self._load_latent(register_seg['latent_path'])

        if register_latent is None:
            register_latent = content_latent

        # Extract register (just latent, no F0 needed)
        register_f0 = self._load_f0(register_seg['f0_path'])
        if register_f0 is None:
            register_f0 = content_f0
        register, _ = self._extract_segment(register_seg, register_latent, register_f0)

        return {
            'content': content,
            'f0': f0,
            'register': register,
            'content_length': content.shape[-1],
            'register_length': register.shape[-1],
        }


def collate_fn(batch):
    """Collate with padding."""
    max_content_len = max(item['content'].shape[-1] for item in batch)
    max_register_len = max(item['register'].shape[-1] for item in batch)

    contents = []
    f0s = []
    registers = []
    content_lengths = []

    for item in batch:
        # Pad content
        c = item['content']
        if c.shape[-1] < max_content_len:
            c = F.pad(c, (0, max_content_len - c.shape[-1]))
        contents.append(c)
        content_lengths.append(item['content_length'])

        # Pad F0
        f = item['f0']
        if len(f) < max_content_len:
            f = F.pad(f, (0, max_content_len - len(f)))
        f0s.append(f)

        # Pad register
        r = item['register']
        if r.shape[-1] < max_register_len:
            r = F.pad(r, (0, max_register_len - r.shape[-1]))
        registers.append(r)

    return {
        'content': torch.stack(contents),
        'f0': torch.stack(f0s),
        'register': torch.stack(registers),
        'content_lengths': torch.tensor(content_lengths),
    }


def train_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        content = batch['content'].to(device)
        f0 = batch['f0'].to(device)
        register = batch['register'].to(device)
        content_lengths = batch['content_lengths']

        optimizer.zero_grad()

        # Forward
        reconstructed, _, _ = model(content, f0, register)

        # Masked reconstruction loss
        B, C, H, T = content.shape
        mask = torch.zeros(B, 1, 1, T, device=device)
        for i, length in enumerate(content_lengths):
            mask[i, :, :, :length] = 1.0

        loss = F.mse_loss(reconstructed * mask, content * mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / num_batches


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    for batch in dataloader:
        content = batch['content'].to(device)
        f0 = batch['f0'].to(device)
        register = batch['register'].to(device)
        content_lengths = batch['content_lengths']

        reconstructed, _, _ = model(content, f0, register)

        B, C, H, T = content.shape
        mask = torch.zeros(B, 1, 1, T, device=device)
        for i, length in enumerate(content_lengths):
            mask[i, :, :, :length] = 1.0

        loss = F.mse_loss(reconstructed * mask, content * mask)
        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


@torch.no_grad()
def compute_register_embeddings(model, dataset, device, num_samples_per_bin=50):
    """Compute average register embeddings per pitch bin."""
    model.eval()
    embeddings_by_bin = {}

    for pitch_bin, segments in dataset.pitch_bins.items():
        samples = random.sample(segments, min(num_samples_per_bin, len(segments)))
        embs = []

        for seg in samples:
            latent = dataset._load_latent(seg['latent_path'])
            if latent is None:
                continue

            f0 = dataset._load_f0(seg['f0_path'])
            if f0 is None:
                continue

            segment, _ = dataset._extract_segment(seg, latent, f0)
            segment = segment.unsqueeze(0).to(device)

            register_emb = model.encode_register(segment)
            embs.append(register_emb.cpu())

        if embs:
            avg_emb = torch.cat(embs).mean(dim=0, keepdim=True)
            embeddings_by_bin[pitch_bin] = avg_emb
            midi_center = pitch_bin * dataset.pitch_bin_size + dataset.pitch_bin_size / 2
            print(f"  Bin {pitch_bin} (MIDI ~{midi_center:.0f}): {len(embs)} samples")

    return embeddings_by_bin


def main():
    parser = argparse.ArgumentParser(description="Train F0-Conditioned Model")
    parser.add_argument('--manifest', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--instrument', type=str, default='trumpet')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--bottleneck_dim', type=int, default=16)
    parser.add_argument('--f0_dim', type=int, default=64)
    parser.add_argument('--register_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--downsample', type=int, default=8)
    parser.add_argument('--pitch_bin_size', type=float, default=2.0)
    parser.add_argument('--min_length', type=int, default=64)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_every', type=int, default=10)
    parser.add_argument('--val_split', type=float, default=0.1)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    log_path = os.path.join(args.output_dir, 'training.log')

    def log(msg):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = f"[{timestamp}] {msg}"
        print(line)
        with open(log_path, 'a') as f:
            f.write(line + '\n')

    log("=" * 60)
    log("F0-Conditioned Register Disentanglement Training")
    log("=" * 60)
    log(f"Bottleneck: {args.bottleneck_dim} × T/{args.downsample}")
    log(f"F0 dim: {args.f0_dim}")
    log(f"Register dim: {args.register_dim}")
    log(f"Instrument: {args.instrument}")
    log(f"Training: Content+F0 from A, Register from B (same pitch), reconstruct A")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    log(f"Device: {device}")

    # Dataset
    dataset = F0ConditionedDataset(
        args.manifest,
        instrument=args.instrument,
        min_length=args.min_length,
        max_length=args.max_length,
        pitch_bin_size=args.pitch_bin_size,
    )

    # Split
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    log(f"Train: {train_size}, Val: {val_size}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = F0ConditionedModel(
        hidden_dim=args.hidden_dim,
        bottleneck_dim=args.bottleneck_dim,
        f0_dim=args.f0_dim,
        register_dim=args.register_dim,
        downsample=args.downsample,
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_loss = float('inf')
    log("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        scheduler.step()

        is_best = val_loss < best_val_loss
        best_val_loss = min(best_val_loss, val_loss)

        log(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"LR: {scheduler.get_last_lr()[0]:.2e}" + (" [BEST]" if is_best else ""))

        if epoch % args.save_every == 0 or is_best:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': vars(args),
            }

            if is_best:
                torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))

            if epoch % args.save_every == 0:
                torch.save(checkpoint, os.path.join(args.output_dir, f'epoch_{epoch:03d}.pt'))

    # Final save
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'config': vars(args),
    }, os.path.join(args.output_dir, 'final_model.pt'))

    log("=" * 60)
    log(f"Training complete! Best val loss: {best_val_loss:.4f}")

    # Compute register embeddings
    log("\nComputing register embeddings by pitch...")
    register_embeddings = compute_register_embeddings(model, dataset, device)

    torch.save({
        'embeddings': register_embeddings,
        'pitch_bin_size': args.pitch_bin_size,
    }, os.path.join(args.output_dir, 'register_embeddings.pt'))

    log(f"Saved {len(register_embeddings)} register embeddings")


if __name__ == "__main__":
    main()
